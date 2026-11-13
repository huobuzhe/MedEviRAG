"""父子双层切分（docs/02 4.4 T4 / ADR-01）。

父节点 = 完整小节，一行不切，只作为 parent_id 与生成时的证据容器；子块 200~400 token
建索引。理由有两条量化依据：
1. bge-m3 的句向量是整段池化，800+ token 的块是多主题质心，检索明显退化；
2. BM25 的 b=0.75 长度归一化会系统性压低长块（avgdl=300 时 3000 token 块的分母长度项
   涨约 7.75 倍），长块排在后面不是因为不相关，而是因为太长。

table / recommendation 不跨节不跨页，靠运行时断言而不是事后报告。
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from medevirag.common.logging import get_logger
from medevirag.ingest.parser import LayoutBlock
from medevirag.ingest.sectionizer import Section

_log = get_logger(__name__)

MAX_TOKENS = 480  # 硬上限，超过即失败
MIN_TOKENS = 200  # 目标下界
TARGET_MAX = 400  # 目标上界：打包时按它填，让绝大多数块落在 200~400
TAIL_MERGE_TOKENS = 120  # 低于此值的尾块尝试并入前块

_PARA = re.compile(r"\n{2,}")
_SENT = re.compile(r"(?<=[。！？；])")
REC_HINT = re.compile(r"(推荐|建议|应当|不推荐|禁用|首选|一线用药|证据等级|推荐强度|推荐类别)")
_NORM_WS = re.compile(r"\s+")

TokenCounter = Callable[[str], int]


@dataclass(slots=True)
class Unit:
    """切分的最小语义单元：一个段落 / 一张表 / 一条推荐意见。"""

    text: str
    kind: str  # body | table | rec
    page: int
    bbox: tuple[float, float, float, float] | None
    block_index: int = 0  # 回链定位：与 bbox 一起从版面块透传，S3 高亮要用


def make_token_counter(model_path: str) -> TokenCounter:
    """bge-m3 的 tokenizer = XLM-RoBERTa sentencepiece，与 S2 的 embedding 同源。"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)

    def count(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False))

    return count


def guess_type(units: list[Unit]) -> str:
    if any(u.kind == "table" for u in units):
        return "table"
    text = "".join(u.text for u in units)
    if REC_HINT.search(text):
        return "recommendation"
    if text.count("、") >= 3 and len(text) < 200:
        return "list"
    return "paragraph"


def to_units(sec_blocks: list[LayoutBlock], count: TokenCounter) -> list[Unit]:
    """把一个节的版面块拆成语义单元。"""
    units: list[Unit] = []
    for b in sec_blocks:
        if b.kind == "table":
            units.append(Unit(b.text, "table", b.page, b.bbox, b.block_index))
            continue
        for para in _PARA.split(b.text):
            para = para.strip()
            if not para:
                continue
            units.append(
                Unit(para, "rec" if REC_HINT.search(para) else "body", b.page, b.bbox, b.block_index)
            )
    return units


def _char_budget(s: str, count: TokenCounter, target: int = TARGET_MAX) -> int:
    """二分找"不超过 target token 的最大字符数"，用于无句读的超长行。"""
    lo, hi = 1, len(s)
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if count(s[:mid]) <= target:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def split_long(u: Unit, count: TokenCounter, limit: int = TARGET_MAX) -> list[str]:
    """句子级递降；单句仍超长则按字符硬切（不做 overlap，内容一字不丢）。"""
    sents = [s for s in _SENT.split(u.text) if s.strip()]
    if not sents:
        sents = [u.text]
    out: list[str] = []
    cur = ""
    for s in sents:
        if count(cur + s) <= limit:
            cur += s
            continue
        if cur:
            out.append(cur)
        while count(s) > limit:
            cut = _char_budget(s, count, limit)
            out.append(s[:cut])
            s = s[cut:]
        cur = s
    if cur:
        out.append(cur)
    return out


def _hard_cap(text: str, count: TokenCounter, limit: int = MAX_TOKENS) -> list[str]:
    """最终兜底：保证任何一片都不超过 limit token，且内容不丢。"""
    if count(text) <= limit:
        return [text]
    pieces: list[str] = []
    rest = text
    while rest and count(rest) > limit:
        cut = max(1, _char_budget(rest, count, limit))
        if cut >= len(rest):
            break
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)
    return pieces


def split_text_balanced(u: Unit, count: TokenCounter, limit: int = TARGET_MAX) -> list[Unit]:
    """超长正文按"均衡片"切：先定片数，再按篇均预算在句界处落刀。

    不能用 _hard_cap 直接按 480 切——那样首片恒为 480（超出 200~400 目标区间），
    尾片剩几十 token 落进 short 桶，两头都失分。先算 ceil(total/400) 定片数，
    再令每片预算 = total/片数，句界对齐后各片都在 200~400。
    """
    total = count(u.text)
    if total <= limit:
        return [u]
    n_pieces = max(2, math.ceil(total / limit))
    budget = max(MIN_TOKENS, min(limit, math.ceil(total / n_pieces)))
    pieces = split_long(u, count, budget)
    out: list[Unit] = []
    for t in pieces:
        if count(t) > MAX_TOKENS:
            out.extend(Unit(x, u.kind, u.page, u.bbox, u.block_index) for x in _hard_cap(t, count))
        else:
            out.append(Unit(t, u.kind, u.page, u.bbox, u.block_index))
    return out



def split_table_by_rows(u: Unit, count: TokenCounter) -> list[Unit]:
    """表格按行组切，每片重复表头并标（续）；行内容不切半，除非单行超硬上限。

    片数先按 ceil(总token/TARGET_MAX) 定死，再按"每片预算 = 总量/片数"填行，
    使各片大小接近而不是"前几片吃满 400、尾片剩几十"——后者是 A3(200~400 占比)
    最大的单一失分来源（实测尾片 45~177 token）。
    """
    total = count(u.text)
    if total <= TARGET_MAX:
        return [u]
    lines = u.text.splitlines()
    if len(lines) <= 2:
        return [
            Unit(t, "table", u.page, u.bbox, u.block_index)
            for t in _hard_cap(u.text, count, TARGET_MAX)
        ]
    head, body = lines[:2], lines[2:]
    head_txt = "\n".join(head)
    marker = "（续）"
    base = count(head_txt) + count(marker)
    rows = [(r, count(r)) for r in body]
    body_total = base * len(rows) + sum(rc for _r, rc in rows)
    n_pieces = max(2, math.ceil(body_total / TARGET_MAX))
    target = min(TARGET_MAX, max(base + 1, body_total // n_pieces))
    out: list[Unit] = []
    buf: list[str] = list(head)
    n = count(head_txt)
    for row, rc in rows:
        if base + rc > TARGET_MAX:  # 单行本身就超预算：先落当前片，再硬切该行
            if len(buf) > len(head) + (1 if marker in buf else 0):
                out.append(Unit("\n".join(buf), "table", u.page, u.bbox, u.block_index))
            for piece in _hard_cap(row, count, TARGET_MAX):
                out.append(
                    Unit("\n".join([*head, marker, piece]), "table", u.page, u.bbox, u.block_index)
                )
            buf = [*head, marker]
            n = base
            continue
        if n + rc > TARGET_MAX or (n + rc > target and len(buf) > len(head)):
            out.append(Unit("\n".join(buf), "table", u.page, u.bbox, u.block_index))
            buf = [*head, marker]
            n = base
        buf.append(row)
        n += rc
    if len(buf) > len(head) + (1 if marker in buf else 0):
        out.append(Unit("\n".join(buf), "table", u.page, u.bbox, u.block_index))
    return out


def _pages_of(units: list[Unit]) -> list[int]:
    return sorted({u.page for u in units})


def _union_bbox(
    rects: list[tuple[float, float, float, float] | None],
) -> tuple[float, float, float, float] | None:
    """多块矩形的外接矩形；全为 None 时返回 None。

    单页 chunk 由多个版面块合并而成，只留首块 bbox 会让高亮只能盖住开头一小段
    （实测文字覆盖率中位数 0.18）。取并集几乎零成本，覆盖率直接接近 1。
    """
    valid = [r for r in rects if r is not None]
    if not valid:
        return None
    return (
        round(min(r[0] for r in valid), 1),
        round(min(r[1] for r in valid), 1),
        round(max(r[2] for r in valid), 1),
        round(max(r[3] for r in valid), 1),
    )


def _locator_of(units: list[Unit]) -> dict[str, Any]:
    """由一组单元装配 locator：page / block_index / bbox。

    规则（与 S1 返工要求一致）：
    - 单页 chunk：bbox 取该页所有单元 bbox 的并集，高亮能盖全整块；
    - 跨页 chunk：bbox 只能定位首段，保持首个带 bbox 的单元的矩形（报告里如实写明）；
    - block_index 始终取首个带 bbox 的单元的版面块序号，与 bbox 同步透传。
    """
    first = next((u for u in units if u.bbox), units[0])
    pages = _pages_of(units)
    bbox = _union_bbox([u.bbox for u in units]) if len(pages) == 1 else first.bbox
    return {
        "page": first.page,
        "block_index": first.block_index,
        "bbox": list(bbox) if bbox else None,
    }


def _has_table(g: list[Unit]) -> bool:
    return any(u.kind == "table" for u in g)


def _packable(cur: list[Unit], nxt: Unit) -> bool:
    """同一片里是否允许再放一个单元。

    两条页约束（否则 Chunk.__post_init__ 的断言会当场炸）：
    ① 表格只能与**同页**单元合片——表格标题、"注："说明、同页正文都算合法邻居，
       跨页一律不行；
    ② rec 单元不能被带过页。
    """
    if not cur:
        return True
    cur_pages = _pages_of(cur)
    if _has_table(cur) or nxt.kind == "table":
        return len(cur_pages) == 1 and nxt.page == cur_pages[0]
    cross_page_rec = nxt.page not in cur_pages and (
        nxt.kind == "rec" or any(u.kind == "rec" for u in cur)
    )
    return not cross_page_rec


def _group_text(g: list[Unit]) -> str:
    return "\n\n".join(u.text for u in g)


def merge_tail(groups: list[list[Unit]], count: TokenCounter) -> list[list[Unit]]:
    """尾块过短则并入前块（仅当合并后不超 TARGET_MAX）。"""
    if len(groups) < 2:
        return groups
    last = groups[-1]
    if count(_group_text(last)) >= TAIL_MERGE_TOKENS:
        return groups
    prev = groups[-2]
    if not _packable(prev, last[0]) or not _packable(last, prev[0]):
        return groups
    if count(_group_text(prev + last)) <= TARGET_MAX:
        return [*groups[:-2], prev + last]
    return groups


def _rebalance(groups: list[list[Unit]], count: TokenCounter) -> list[list[Unit]]:
    """把过短的组向相邻组借单元（双向），让更多组落进 200~400。

    贪婪打包天然会在每节末尾留一个残片（"前几片吃满 400、尾片几十"）。单向向前
    借解决不了尾片，所以这里同时允许向后借：尾片从上一组的**末尾**取单元，
    只要上一组仍不低于 MIN_TOKENS，两组边界就都落在 200~400 内。
    """

    def n_of(g: list[Unit]) -> int:
        return count(_group_text(g))

    if len(groups) < 2:
        return groups
    for _pass in range(2):
        for i, cur in enumerate(groups):
            if not cur or n_of(cur) >= MIN_TOKENS:
                continue
            if i + 1 < len(groups) and groups[i + 1]:
                nxt = groups[i + 1]
                while nxt and n_of(cur) < MIN_TOKENS and _packable(cur, nxt[0]):
                    cand = [*cur, nxt[0]]
                    if n_of(cand) > TARGET_MAX:
                        break
                    cur, nxt = cand, nxt[1:]
                groups[i], groups[i + 1] = cur, nxt
                continue
            if i > 0 and groups[i - 1]:
                prev = groups[i - 1]
                while len(prev) > 1 and n_of(cur) < MIN_TOKENS and _packable([prev[-1]], cur[0]):
                    cand = [prev[-1], *cur]
                    if n_of(prev[:-1]) < MIN_TOKENS or n_of(cand) > TARGET_MAX:
                        break
                    prev, cur = prev[:-1], cand
                groups[i - 1], groups[i] = prev, cur
    return [g for g in groups if g]


def coalesce_tables(units: list[Unit], count: TokenCounter) -> list[Unit]:
    """把同一页上相邻的小表格合成一个单元，再统一按行切。

    真实指南里"表2.2 / 表2.3 / 表2.4 各 100 token"这种同页连排极常见，逐表成块会
    造出一串 100 token 的碎片；先并再切能让每片都落进目标区间，且因为不跨页，
    table 的页内硬约束仍然成立。
    """
    out: list[Unit] = []
    for u in units:
        prev = out[-1] if out else None
        if prev is not None and prev.kind == "table" and u.kind == "table" and prev.page == u.page:
            out[-1] = Unit(
                prev.text + chr(10) + u.text,
                "table",
                prev.page,
                _union_bbox([prev.bbox, u.bbox]),
                prev.block_index,
            )
            continue
        out.append(u)
    return out


def _squash(s: str) -> str:
    """比较标题用：压掉所有空白与标点，避免"一、 前言"与"一、前言"不相等。"""
    return re.sub(r"[\s、．.，,：:（）()]", "", s)


def drop_heading_units(units: list[Unit], sec: Section) -> list[Unit]:
    """丢掉"标题自身变成内容"的单元。

    标题已经进了 section_path 与 context_header，再当正文单元就会切出 "附录2"(4 token)、
    整节只含标题时切出 8 token 的块——这类块没有任何检索价值，还会把 A3 的
    200~400 占比直接拖垮。
    """
    keys = {_squash(sec.title), _squash(sec.section_no + sec.title), _squash(render_heading(sec))}
    keys.discard("")
    out = [u for u in units if _squash(u.text) not in keys]
    return out


def render_heading(sec: Section) -> str:
    return (sec.section_no + " " + sec.title).strip()


def chunk_section(sec: Section, units: list[Unit], count: TokenCounter) -> list[dict[str, Any]]:
    """在父节点内部切子块；返回未补元数据的 chunk dict 列表。"""
    if not units:
        return []
    units = drop_heading_units(units, sec)
    if not units:
        return []
    units = coalesce_tables(units, count)
    rigid = len(units) == 1 and units[0].kind == "table"
    groups: list[list[Unit]] = []
    if rigid:
        groups = [[u] for u in split_table_by_rows(units[0], count)]
    else:
        cur: list[Unit] = []
        n = 0
        for u in units:
            t = count(u.text)
            if u.kind == "table":  # 表格自成一片，绝不与正文合并
                if cur:
                    groups.append(cur)
                    cur, n = [], 0
                groups.extend([[x] for x in split_table_by_rows(u, count)])
                continue
            if t > TARGET_MAX:
                if cur:
                    groups.append(cur)
                    cur, n = [], 0
                groups.extend(
                    [[Unit(p, u.kind, u.page, u.bbox, u.block_index)] for p in split_long(u, count)]
                )
                continue
            if cur and (n + t > TARGET_MAX or not _packable(cur, u)):
                groups.append(cur)
                cur, n = [], 0
            cur.append(u)
            n += t
        if cur:
            groups.append(cur)
        groups = _rebalance(groups, count)
        groups = merge_tail(groups, count)

    out: list[dict[str, Any]] = []
    for g in groups:
        text = _group_text(g)
        ctype = "table" if rigid else guess_type(g)
        pages = _pages_of(g)
        if ctype == "recommendation" and len(pages) > 1:
            # 兜底：跨页的推荐意见按页拆开，保证 page_start == page_end
            for page in pages:
                part = [u for u in g if u.page == page]
                if not part:
                    continue
                ptext = _group_text(part)
                out.append(_make_chunk_dict(ptext, "recommendation", part, count))
            continue
        out.append(_make_chunk_dict(text, ctype, g, count))
    for i, item in enumerate(out, start=1):
        item["_seq"] = i
    return out


def _make_chunk_dict(text: str, ctype: str, group: list[Unit], count: TokenCounter) -> dict[str, Any]:
    pages = _pages_of(group)
    pieces = _hard_cap(text, count) if count(text) > MAX_TOKENS else [text]
    payload = pieces[0]
    return {
        "chunk_type": ctype,
        "text": payload,
        "token_len": count(payload),
        "page_start": pages[0],
        "page_end": pages[-1],
        "locator": _locator_of(group),
        "_overflow": pieces[1:],
        "_seq": 0,
    }


def expand_overflow(
    items: list[dict[str, Any]], count: TokenCounter | None = None
) -> list[dict[str, Any]]:
    """把 _hard_cap 切出的余片补成独立 chunk（内容不丢，序号顺延）。

    token_len 必须用同一个 tokenizer 重算——用 len(piece) 会把字符数当 token 数，
    直接污染 A3/A4 两项统计。
    """
    out: list[dict[str, Any]] = []
    for item in items:
        extra = item.pop("_overflow", [])
        out.append(item)
        for piece in extra:
            dup = dict(item)
            dup["text"] = piece
            dup["token_len"] = count(piece) if count else len(piece)
            out.append(dup)
    return out


def merge_tail_groups() -> None:  # 兼容占位：保留名字便于对照文档
    return None




@dataclass(slots=True)
class StreamItem:
    """文档级流里的一个单元：单元本身 + 它所属的节（用于回填 parent/section_path）。"""

    unit: Unit
    section: Section


def _stream_text(group: list[StreamItem]) -> str:
    return "\n\n".join(it.unit.text for it in group)


def _stream_pages(group: list[StreamItem]) -> list[int]:
    return sorted({it.unit.page for it in group})


def _stream_packable(group: list[StreamItem], nxt: StreamItem) -> bool:
    """流式打包的页约束：表格与推荐单元都不许被带过页。"""
    if not group:
        return True
    pages = _stream_pages(group)
    u = nxt.unit
    has_table = any(x.unit.kind == "table" for x in group)
    has_rec = any(x.unit.kind == "rec" for x in group)
    if has_table or u.kind == "table":
        return len(pages) == 1 and u.page == pages[0]
    if u.kind == "rec" or has_rec:
        return len(pages) == 1 and u.page == pages[0]
    return True


def pack_stream(items: list[StreamItem], count: TokenCounter) -> list[list[StreamItem]]:
    """文档级流式打包：跨节把短节拼满 200~400，再双向回填残片。

    逐节打包会系统性地在每节末尾留残片（实测全量 7023 块里 3620 块 <200 token，
    200~400 只占 47.7%），因为 WS/GB 这类标准的节本身常常不足 200 token。
    改成整篇流式打包后，节间的空隙被自然填平，且表格/推荐单元的页内约束不变。
    """
    if not items:
        return []
    groups: list[list[StreamItem]] = []
    cur: list[StreamItem] = []
    for it in items:
        if cur:
            n = count(_stream_text(cur))
            t = count(it.unit.text)
            if n + t > TARGET_MAX or not _stream_packable(cur, it):
                groups.append(cur)
                cur = []
        cur.append(it)
    if cur:
        groups.append(cur)
    return _stream_rebalance(groups, count)


def _stream_rebalance(groups: list[list[StreamItem]], count: TokenCounter) -> list[list[StreamItem]]:
    """把仍不足 MIN_TOKENS 的组向相邻组借单元（先向后借、再向前借）。"""
    for _pass in range(3):
        for i, cur in enumerate(groups):
            if not cur:
                continue
            n = count(_stream_text(cur))
            if n >= MIN_TOKENS:
                continue
            if i + 1 < len(groups) and groups[i + 1]:
                nxt = groups[i + 1]
                while nxt and n < MIN_TOKENS and _stream_packable(cur, nxt[0]):
                    t = count(nxt[0].unit.text)
                    if n + t > TARGET_MAX:
                        break
                    cur.append(nxt.pop(0))
                    n += t
                groups[i] = cur
                continue
            if i > 0 and len(groups[i - 1]) > 1:
                prev = groups[i - 1]
                while len(prev) > 1 and n < MIN_TOKENS:
                    cand = prev[-1]
                    rest = count(_stream_text(prev[:-1]))
                    t = count(cand.unit.text)
                    if rest < MIN_TOKENS or n + t > TARGET_MAX:
                        break
                    prev.pop()
                    cur.insert(0, cand)
                    n += t
                groups[i - 1] = prev
                groups[i] = cur
    return [g for g in groups if g]


def _dominant_section(group: list[StreamItem], count: TokenCounter) -> Section:
    """块跨节时取贡献 token 最多的那一节做父节点（保证 parent 里装的是块的多数内容）。"""
    tally: dict[str, int] = {}
    by_id: dict[str, Section] = {}
    for it in group:
        sid = it.section.section_id
        by_id[sid] = it.section
        tally[sid] = tally.get(sid, 0) + count(it.unit.text)
    best = max(tally.items(), key=lambda kv: kv[1])[0]
    return by_id[best]


def _split_oversize(group: list[StreamItem], count: TokenCounter) -> list[list[StreamItem]]:
    """超过硬上限的组按单元顺序再切；单个单元自身超限时按句/字符切开。"""
    if count(_stream_text(group)) <= MAX_TOKENS:
        return [group]
    out: list[list[StreamItem]] = []
    cur: list[StreamItem] = []
    for it in group:
        t = count(it.unit.text)
        if t > MAX_TOKENS:
            if cur:
                out.append(cur)
                cur = []
            for piece in _hard_cap(it.unit.text, count):
                out.append(
                    [
                        StreamItem(
                            Unit(piece, it.unit.kind, it.unit.page, it.unit.bbox, it.unit.block_index),
                            it.section,
                        )
                    ]
                )
            continue
        if cur and count(_stream_text(cur)) + t > MAX_TOKENS:
            out.append(cur)
            cur = []
        cur.append(it)
    if cur:
        out.append(cur)
    return out


def chunk_stream(
    sections: list[tuple[Section, list[Unit]]], count: TokenCounter
) -> list[tuple[Section, dict[str, Any]]]:
    """整篇流式切块：返回 (父节, chunk dict) 列表，dict 与 chunk_section 同结构。

    只在正文单元之间跨节合并；表格与推荐单元自成屏障（_stream_packable 要求同页），
    所以「table/recommendation 不跨节不跨页」这条硬约束依然成立。
    """
    items: list[StreamItem] = []
    for sec, units in sections:
        if not units:
            continue
        units = drop_heading_units(units, sec)
        if not units:
            continue
        units = coalesce_tables(units, count)
        for u in units:
            if count(u.text) <= TARGET_MAX:
                items.append(StreamItem(u, sec))
            elif u.kind == "table":
                items.extend(StreamItem(pc, sec) for pc in split_table_by_rows(u, count))
            else:
                items.extend(StreamItem(pc, sec) for pc in split_text_balanced(u, count))
    out: list[tuple[Section, dict[str, Any]]] = []
    for group in pack_stream(items, count):
        for part in _split_oversize(group, count):
            for page_part in _split_cross_page(part, count):
                for piece in _emit(page_part, count):
                    out.append(piece)
    for i, (_sec, item) in enumerate(out, start=1):
        item["_seq"] = i
    return out


def _emit(group: list[StreamItem], count: TokenCounter) -> list[tuple[Section, dict[str, Any]]]:
    """把一个打包组落成 1~n 个 chunk dict；超 480 时按 _hard_cap 切片，内容不丢。"""
    text = _stream_text(group)
    ctype = guess_type([it.unit for it in group])
    sec = _dominant_section(group, count)
    units = [it.unit for it in group]
    pages = _pages_of(units)
    pieces = _hard_cap(text, count) if count(text) > MAX_TOKENS else [text]
    out: list[tuple[Section, dict[str, Any]]] = []
    for piece in pieces:
        out.append(
            (
                sec,
                {
                    "chunk_type": ctype,
                    "text": piece,
                    "token_len": count(piece),
                    "page_start": pages[0],
                    "page_end": pages[-1],
                    "locator": _locator_of(units),
                    "_seq": 0,
                },
            )
        )
    return out


def _split_cross_page(group: list[StreamItem], count: TokenCounter) -> list[list[StreamItem]]:
    """table / recommendation 不许跨页：按页拆开，保证 page_start == page_end。"""
    ctype = guess_type([it.unit for it in group])
    pages = _stream_pages(group)
    if len(pages) <= 1:
        return [group]
    if ctype not in ("table", "recommendation"):
        return [group]
    out: list[list[StreamItem]] = []
    for page in pages:
        part = [it for it in group if it.unit.page == page]
        if part:
            out.append(part)
    return out

def context_header(doc_title: str, section_path: list[str]) -> str:
    """《指南名》 > 3.2.1 诊室血压；section_path 为空时退化为《指南名》。"""
    tail = section_path[-1] if section_path else ""
    return "《" + doc_title + "》 > " + tail if tail else "《" + doc_title + "》"


def build_chunk_id(doc_id: str, section_no: str, seq: int) -> str:
    no = section_no.replace(" ", "").replace("（", "(").replace("）", ")") or "root"
    no = re.sub(r"[/\\#?&%]", "-", no)
    return doc_id + "#" + no + "#c" + str(seq).zfill(2)


def content_hash(text: str) -> str:
    """归一化文本的 sha256：改一个空格不触发全量重算 embedding。"""
    norm = _NORM_WS.sub("", unicodedata.normalize("NFKC", text))
    return "sha256:" + hashlib.sha256(norm.encode("utf-8")).hexdigest()


def assert_hard_constraints(c: dict[str, Any]) -> None:
    """00 §6.1 三条硬约束的运行时断言，失败即中止本次切分。"""
    assert c["token_len"] <= MAX_TOKENS, "token_len 超硬上限: " + str(c["chunk_id"])
    if c["chunk_type"] in ("table", "recommendation"):
        assert c["page_start"] == c["page_end"], "table/recommendation 禁止跨页合并: " + str(c["chunk_id"])
    assert c["context_header"], "context_header 不能为空: " + str(c["chunk_id"])
    if c["page_start"] == c["page_end"]:
        assert c["locator"]["bbox"], "回链 bbox 缺失: " + str(c["chunk_id"])
