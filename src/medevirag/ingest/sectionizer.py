"""章节树重建（docs/02 4.3 T3）。

难点不在"识别一个标题"，而在四类噪声同时存在：多套编号体系混用、字号不可靠、
跨页标题被切断、页眉页脚插在正文里。所以这里是三路证据合成而不是单条正则：
编号深度优先、字号兜底、目录页交叉校验。

section_path 元素格式固定为 "<编号> <标题>"，chunker 直接取最后一段做
context_header，避免两处各自拼编号。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from medevirag.common.logging import get_logger
from medevirag.ingest.parser import LayoutBlock

_log = get_logger(__name__)

# 每项：(体系名, 正则)；编号统一放在 group(1)，标题放 group(2)
NUM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "cn_chapter",
        re.compile(r"^第\s*([一二三四五六七八九十百零〇\d]{1,4})\s*([章节部篇])\s*[、\.．:：]?\s*(\S.{0,60})$"),
    ),
    (
        "multi_arabic",
        re.compile(r"^(\d{1,2}(?:\s*[.．]\s*\d{1,2}){1,4})\s*[、.．]?\s*(\S.{0,60})$"),
    ),
    ("cn_paren", re.compile(r"^[（(]\s*([一二三四五六七八九十]{1,3}|\d{1,2})\s*[）)]\s*(\S.{0,60})$")),
    ("cn_dunhao", re.compile(r"^([一二三四五六七八九十]{1,3})\s*[、．\.]\s*(\S.{0,60})$")),
]
"""四级编号；multi_arabic 必须带点（3.2.1），单数字由 heading_level 用字号再筛一次——
否则 "5mg 每日一次" 这类正文会被当成标题。"""

_CN_DIGIT = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNIT_LEVEL = {"章": 1, "篇": 1, "部": 1, "节": 2}
_SYSTEM_LEVEL = {"cn_chapter": 0, "multi_arabic": 0, "cn_dunhao": 2, "cn_paren": 3}

TOC_LINE = re.compile(
    r"^(?P<title>[\u4e00-\u9fa5A-Za-z0-9（）()、．\.\s]{2,60}?)[\s.·…]{3,}(?P<page>\d{1,4})\s*$"
)
_HEAD_TAIL_PUNCT = re.compile(r"[。；，,、：:]\s*$")
_DOT_RUN = re.compile(r"[\s.·…]+")

SENT_END = re.compile(r"[。！？；：\)】」』!?;:]\s*$")
PURE_PAGE_NO = re.compile(r"^(?:第\s*\d{1,4}\s*页|[-—–]?\s*\d{1,4}\s*[-—–]?)$")


def render_heading_text(text: str) -> str:
    """无编号标题的规范文本：压掉内部多余空白。"""
    return re.sub(r"\s+", " ", text).strip()


def is_pure_page_number(text: str) -> bool:
    """纯页码行（含 — 27 — 这类带装饰的）。"""
    return bool(PURE_PAGE_NO.match(text.strip()))


def cn2int(s: str) -> int:
    """仅覆盖 1~99（指南章节编号不会更大）：'十一'->11，'二十'->20，'十'->10。"""
    if s.isdigit():
        return int(s)
    if "十" in s:
        a, _, b = s.partition("十")
        return (_CN_DIGIT[a] if a else 1) * 10 + (_CN_DIGIT[b] if b else 0)
    return _CN_DIGIT.get(s, 0)


def _norm_number(no: str) -> str:
    """编号归一：去空白、全角点转半角、全角括号转半角。"""
    no = re.sub(r"\s*[.．]\s*", ".", no)
    return no.replace("（", "(").replace("）", ")").strip()


_UNIT_TAIL = re.compile(r"^[%％]|^(?:mmhg|mg|ml|kg|cm|mm|kcal|kj)\b", re.I)
"""编号后面紧跟单位/百分号的是正文数字（27.5%、5mg、140/90 mmHg），不是章节号。"""


def _unit_tail(text: str, end: int) -> bool:
    rest = text[end:].lstrip(" 、.．")
    return bool(_UNIT_TAIL.match(rest))


def match_number(text: str) -> tuple[str, str, int] | None:
    """返回 (体系, 归一化编号串, 隐含层级)；不命中返回 None。"""
    t = unicodedata.normalize("NFKC", text.strip())
    for name, pat in NUM_PATTERNS:
        m = pat.match(t)
        if not m:
            continue
        if _unit_tail(t, m.end(1)):
            continue
        if name == "cn_chapter":
            no, unit = m.group(1), m.group(2)
            return name, "第" + no + unit, _CN_UNIT_LEVEL[unit]
        no = m.group(1)
        if name == "multi_arabic":
            norm = _norm_number(no)
            return name, norm, norm.count(".") + 1
        if name == "cn_paren":
            return name, "(" + no + ")", _SYSTEM_LEVEL[name]
        return name, no + "、", _SYSTEM_LEVEL[name]
    return None


def split_heading(text: str) -> tuple[str, str, str] | None:
    """返回 (编号, 标题, 体系)；标题是去掉编号前缀后的原文。"""
    hit = match_number(text)
    if hit is None:
        return None
    system, no, _level = hit
    t = unicodedata.normalize("NFKC", text.strip())
    if system == "cn_chapter":
        m = NUM_PATTERNS[0][1].match(t)
        title = m.group(3) if m else ""
    elif system == "multi_arabic":
        m = NUM_PATTERNS[1][1].match(t)
        title = m.group(2) if m else ""
    elif system == "cn_paren":
        m = NUM_PATTERNS[2][1].match(t)
        title = m.group(2) if m else ""
    else:
        m = NUM_PATTERNS[3][1].match(t)
        title = m.group(2) if m else ""
    return no, title.strip(), system


def _looks_like_heading(block: LayoutBlock) -> bool:
    """长度、结尾标点、表图前缀三重过滤；正文段落不满足这三条。"""
    text = block.text.strip()
    if not 2 <= len(text) <= 40:
        return False
    if _HEAD_TAIL_PUNCT.search(text):
        return False
    return not text.startswith(("表", "图", "注", "资料来源"))


def _heading_title_ok(title: str) -> bool:
    """标题正文的成色检查：去掉编号后必须短、且不含句读——这是区分
    "一、前言" 与 "27.5%，其中18～44 岁" 的唯一可靠信号（两者字号相同）。"""
    t = title.strip()
    if not 2 <= len(t) <= 30:
        return False
    return not re.search(r"[。；，、！？：%]", t)


def heading_level(block: LayoutBlock, body_size: float) -> int | None:
    """返回层级或 None（不是标题）。编号深度优先，字号兜底。

    卫健委类指南的章节标题常与正文同字号（实测 16pt vs 16pt），只靠字号会漏掉
    全部 "一、前言" 这类标题；因此"编号 + 短标题 + 标题内无句读"三条件同时成立
    时直接采信编号深度，字号只在无编号时兜底。
    """
    if block.kind == "table":
        return None
    if not _looks_like_heading(block):
        return None
    ratio = block.size / body_size if body_size else 1.0
    size_level = 1 if ratio >= 1.6 else 2 if ratio >= 1.2 else (3 if (block.bold or ratio >= 1.05) else None)
    hit = match_number(block.text)
    if hit is not None:
        _system, _no, level = hit
        split = split_heading(block.text)
        if split is not None and _heading_title_ok(split[1]):
            return level
        return size_level
    return size_level


def detect_body_size(blocks: list[LayoutBlock]) -> float:
    """全文字符数加权众数字号 -> body_size。"""
    counter: Counter[float] = Counter()
    for b in blocks:
        if b.kind == "body" and b.size > 0:
            counter[b.size] += b.n_chars
    return counter.most_common(1)[0][0] if counter else 0.0


def _norm_title(t: str) -> str:
    return _DOT_RUN.sub("", unicodedata.normalize("NFKC", t))


def parse_toc(pages_text: dict[int, str]) -> list[tuple[str, int]]:
    """pages_text: {PDF 页码: 该页纯文本}。返回 [(归一化标题, 印刷页码)]。"""
    out: list[tuple[str, int]] = []
    for pno in sorted(pages_text):
        head = pages_text[pno][:40].replace(" ", "")
        if not ("目录" in head or "目次" in head or "CONTENTS" in head.upper()):
            continue
        hit = 0
        for ln in pages_text[pno].splitlines():
            m = TOC_LINE.match(ln.strip())
            if m:
                out.append((_norm_title(m.group("title")), int(m.group("page"))))
                hit += 1
        if hit >= 5:  # 抽到 5 条以上才认这一页是目录
            break
    return out


def page_offset(toc: list[tuple[str, int]], headings: list[tuple[str, int]]) -> int:
    """headings: [(归一化标题, PDF 页序号)]。返回 印刷页码- PDF 序号 的众数偏移。"""
    index = {t: p for t, p in headings}
    deltas = Counter(index[t] - pp for t, pp in toc if t in index)
    return deltas.most_common(1)[0][0] if deltas else 0


def toc_trusted(toc: list[tuple[str, int]], headings: list[tuple[str, int]], min_match: float = 0.80) -> bool:
    """目录与正文标题匹配率 >=80% 才认为目录可信。"""
    if not toc:
        return False
    index = {t for t, _ in headings}
    hit = sum(1 for t, _ in toc if t in index)
    return hit / len(toc) >= min_match


def _margin_key(t: str) -> str:
    """页眉归一化键：数字统一成 #、去空白、NFKC。"""
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\d+", "#", t)
    return re.sub(r"\s+", "", t)


def detect_repeated_margins(
    lines_by_page: dict[int, list[tuple[str, float, float, float]]],
    *,
    top: float = 0.08,
    bottom: float = 0.92,
    min_pages: int = 3,
    min_ratio: float = 0.30,
) -> set[str]:
    """lines_by_page: {页码: [(文本, y0, y1, 页高)]} -> 应剔除的归一化键集合。"""
    seen: Counter[str] = Counter()
    for _pno, lines in lines_by_page.items():
        for text, y0, y1, h in lines:
            if not text.strip():
                continue
            if y0 < h * top or y1 > h * bottom:
                seen[_margin_key(text)] += 1
    n = max(1, len(lines_by_page))
    return {k for k, c in seen.items() if c >= min_pages and c / n >= min_ratio}


def is_margin(text: str, reject_keys: set[str]) -> bool:
    return is_pure_page_number(text) or _margin_key(text) in reject_keys


def merge_cross_page(blocks: list[LayoutBlock]) -> list[LayoutBlock]:
    """就地合并跨页的标题碎片与段落链（只为句子不断，不改变 section 归属）。"""
    merged: list[LayoutBlock] = []
    for b in blocks:
        if merged:
            prev = merged[-1]
            same_style = abs(prev.size - b.size) < 0.2 and prev.font == b.font
            if (
                b.page == prev.page + 1
                and same_style
                and len(prev.text) <= 8
                and match_number(b.text) is None
                and b.kind == prev.kind
            ):
                prev.text += b.text
                prev.n_chars = len(prev.text)
                # bbox 不跨页拼接：两块在不同页上，拼出的矩形 y0 > y1 是退化的，
                # 留着首段矩形才能高亮回链（跨页块只能定位首段，报告已如实声明）
                continue
            if (
                b.page == prev.page + 1
                and prev.kind == "body"
                and b.kind == "body"
                and not SENT_END.search(prev.text)
                and match_number(b.text) is None
            ):
                prev.text += b.text
                prev.n_chars = len(prev.text)
                continue
        merged.append(b)
    return merged


@dataclass(slots=True)
class Section:
    """章节树节点；section_path 是给 chunker 直接用的完整路径。"""

    section_id: str
    doc_id: str
    section_no: str
    title: str
    level: int
    page_start: int
    page_end: int
    order_index: int
    parent_section_id: str | None
    section_path: list[str]
    block_ids: list[str] = field(default_factory=list)


def block_id_of(block: LayoutBlock) -> str:
    return str(block.page) + ":" + str(block.block_index)


def detect_headings(blocks: list[LayoutBlock]) -> tuple[list[tuple[LayoutBlock, int, str, str]], float]:
    """返回 ([(block, level, section_no, title)], body_size)，按阅读顺序。"""
    body_size = detect_body_size(blocks)
    out: list[tuple[LayoutBlock, int, str, str]] = []
    for b in blocks:
        level = heading_level(b, body_size)
        if level is None:
            continue
        split = split_heading(b.text)
        if split is None:
            title = render_heading_text(b.text)
            out.append((b, level, "", title))
            continue
        no, title, _system = split
        out.append((b, level, no, title))
    return out, body_size


def _assign_blocks(sec: Section, blocks: list[LayoutBlock]) -> None:
    """把给定 blocks 的 id 追加到 sec.block_ids。"""
    sec.block_ids.extend(block_id_of(b) for b in blocks)


def build_sections(
    doc_id: str, headings: list[tuple[LayoutBlock, int, str, str]], blocks: list[LayoutBlock]
) -> list[Section]:
    """栈装配章节树；弹出时回填 page_end，最后按阅读顺序把 block 归入最深节。"""
    first_page = blocks[0].page if blocks else 1
    last_page = blocks[-1].page if blocks else first_page
    root = Section(doc_id + "#root", doc_id, "", "ROOT", 0, first_page, last_page, 0, None, [])
    flat: list[Section] = [root]
    stack: list[Section] = [root]
    used_ids: set[str] = {root.section_id}
    for order, (blk, level, no, title) in enumerate(headings, start=1):
        while len(stack) > 1 and stack[-1].level >= level:
            closed = stack.pop()
            closed.page_end = max(closed.page_start, blk.page - 1) if blk.page > closed.page_start else closed.page_start
        parent = stack[-1]
        base = doc_id + "#" + (no.replace(" ", "") or "h" + str(order))
        section_id = base
        dup = 2
        while section_id in used_ids:  # 同一编号可能在多个父节下重复出现
            section_id = base + "~" + str(dup)
            dup += 1
        used_ids.add(section_id)
        sec = Section(
            section_id=section_id,
            doc_id=doc_id,
            section_no=no,
            title=title,
            level=level,
            page_start=blk.page,
            page_end=blk.page,
            order_index=order,
            parent_section_id=None if parent is root else parent.section_id,
            section_path=(
                [] if parent is root else list(parent.section_path)
            )
            + [((no + " " + title).strip() or title)],
        )
        flat.append(sec)
        stack.append(sec)
    while len(stack) > 1:
        closed = stack.pop()
        closed.page_end = last_page
    # 按标题位置把 block 归入"最后一个已开启的节"，标题之前的 block 归 root
    ordered: list[tuple[tuple[int, int], Section]] = [
        ((blk.page, blk.block_index), flat[i + 1]) for i, (blk, _l, _n, _t) in enumerate(headings)
    ]
    ordered.sort(key=lambda x: x[0])
    ptr = 0
    cursor: Section = root
    for b in sorted(blocks, key=lambda x: (x.page, x.block_index)):
        pos = (b.page, b.block_index)
        while ptr < len(ordered) and ordered[ptr][0] <= pos:
            cursor = ordered[ptr][1]
            ptr += 1
        _assign_blocks(cursor, [b])
    return flat
