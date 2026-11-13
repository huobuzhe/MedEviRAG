"""PDF 解析与版面块抽取（docs/02 4.2 T2）。

为什么自己拼版面块：PDF 里没有"段落"和"标题"，只有"在某坐标画某字形"的指令。
下游的章节树重建唯一输入就是"块 + 字号 + 位置 + 是否加粗"，块切错了后面全靠拍脑袋。

三条与验收直接相关的实现约束：
1. 块级字号取"字符数加权众数"而非最大值——上标（mmHg 的角标）会把最大值拉偏；
2. 表格 bbox 内的文本块必须从正文流剔除，否则同一段内容进两次，chunk 数虚高、
   BM25 的 avgdl 被拉偏；
3. 扫描件与乱码件必须在这里就判出来，不让它们污染后续统计。
"""

from __future__ import annotations

import contextlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pymupdf >= 1.24 同时提供 pymupdf 与 fitz 两个 import 名
    import pymupdf as fitz
except ImportError:  # pragma: no cover - 只有老版本才会走这里
    import fitz

SCAN_CHARS_PER_PAGE = 150  # 整份文档平均字符数/页 低于此值 -> 扫描件
SCAN_PAGE_CHARS = 80  # 单页正文块字符数 低于此值 -> 该页无文本层
GARBAGE_LEGIBLE_RATIO = 0.85  # 可用字符占比低于此值 -> 乱码件（实测合法件最低 0.92）
GARBAGE_CJK_RATIO = 0.05  # 汉字几乎为零 ...
GARBAGE_STOP_RATIO = 0.02  # ... 且英文功能词也几乎没有 -> 拉丁乱码
FFFD_RATIO = 0.02  # 替换字符占比高于此值 -> 缺 ToUnicode CMap

_CJK = re.compile(r"[\u4e00-\u9fff]")
# 可用字符：ASCII 可见 + 汉字基本区 + 中日韩标点 + 全角形式 + 常用排版符号。
# 乱码件的特征是字形被映射到私用区/"犐犆犛"这类生僻汉区，落不进这个集合。
_LEGIBLE = re.compile(
    r"[\x20-\x7e\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"
    r"\u2010-\u206f\u00b7\u2026\u00d7\u00b1\u2264\u2265\u2103\u00b0\u00b2\u00b3]"
)
_STOPWORD = re.compile(
    r"(?<![A-Za-z])(?:the|of|and|to|in|for|with|is|are|that|this|by|from|on|as|at|be|or|an|it|"
    r"which|these|respectively|should)(?![A-Za-z])",
    re.I,
)
_SPACES = re.compile(r"\s+")


@dataclass(slots=True)
class LayoutBlock:
    """一个版面块；page + block_index 唯一定位它，bbox 用于回链高亮。"""

    page: int  # 1-based
    block_index: int
    kind: str  # body | title | table | figure_caption
    text: str
    bbox: tuple[float, float, float, float]
    size: float  # 字符数加权众数字号
    font: str
    bold: bool
    n_chars: int


def _weighted_size(spans: list[dict[str, Any]]) -> tuple[float, str, bool]:
    """按字符数加权取众数字号/字体；bold 以加粗字符占比 >0.6 判定。"""
    sizes: Counter[float] = Counter()
    fonts: Counter[str] = Counter()
    bold_chars = 0
    total = 0
    for sp in spans:
        n = len(str(sp.get("text", "")).strip())
        if not n:
            continue
        sizes[round(float(sp.get("size", 0.0)), 1)] += n
        fonts[str(sp.get("font", ""))] += n
        total += n
        if int(sp.get("flags", 0)) & 2**4:  # bit4 = bold
            bold_chars += n
    if not total:
        return 0.0, "", False
    return sizes.most_common(1)[0][0], fonts.most_common(1)[0][0], bold_chars / total > 0.6


def render_table(rows: list[list[str | None]]) -> str:
    """表格渲染成管道分隔文本，首行作表头，单元格内换行压成空格。"""
    clean = [[" ".join((c or "").split()) for c in row] for row in rows]
    if not clean:
        return ""
    width = max(len(r) for r in clean)
    padded = [r + [""] * (width - len(r)) for r in clean]
    head, body = padded[0], padded[1:]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _empty_table(rows: list[list[str | None]]) -> bool:
    return not any((c or "").strip() for r in rows for c in r)


def _rect_tuple(rect: Any) -> tuple[float, float, float, float] | None:
    """把矩形对象归一成四舍五入 1 位小数的四元组；退化矩形视为无效。"""
    if rect is None:
        return None
    try:
        x0, y0, x1, y1 = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))


def _plumber_tables(
    doc: Any, page_index: int
) -> list[tuple[list[list[str | None]], tuple[float, float, float, float] | None]]:
    """find_tables 空时的降级通道；pdfplumber 缺失则静默返回空。

    返回 (行数据, 表格矩形)：pdfplumber 的 Table 自带 .bbox，必须一并带出，
    否则降级通道产出的表格块只能退化成整页矩形、失去高亮能力。
    """
    try:
        page = doc.pages[page_index]
    except Exception:  # 降级通道失败不应让整份文档解析失败
        return []
    out: list[tuple[list[list[str | None]], tuple[float, float, float, float] | None]] = []
    try:
        tables = page.find_tables()
    except Exception:
        tables = []
    try:
        for t in tables:
            rows = t.extract()
            if rows and not _empty_table(rows):
                out.append((rows, _rect_tuple(getattr(t, "bbox", None))))
    except Exception:
        return []
    if out:
        return out
    try:  # 老版本 pdfplumber 没有 find_tables，退回 extract_tables（无矩形可用）
        return [
            (rows, None) for rows in page.extract_tables() if rows and not _empty_table(rows)
        ]
    except Exception:
        return []


def extract_blocks(pdf_path: Path, doc_id: str) -> tuple[list[LayoutBlock], dict[str, Any]]:
    """抽取版面块序列与解析统计。"""
    blocks: list[LayoutBlock] = []
    chars_per_page: list[int] = []
    images = 0
    plumber_doc: Any = None
    with fitz.open(str(pdf_path)) as doc:
        for pno in range(1, doc.page_count + 1):
            page = doc[pno - 1]
            tabs = page.find_tables().tables
            table_rects = [fitz.Rect(t.bbox) for t in tabs]
            page_chars = 0
            for bi, b in enumerate(page.get_text("dict")["blocks"]):
                if b["type"] != 0:  # 图片块留给图注逻辑
                    images += 1
                    continue
                spans = [sp for ln in b["lines"] for sp in ln["spans"]]
                text = "".join(str(sp.get("text", "")) for sp in spans).strip()
                if not text:
                    continue
                rect = fitz.Rect(b["bbox"])
                if any(rect in tr for tr in table_rects):  # 表格内部文本，另行抽取
                    continue
                size, font, bold = _weighted_size(spans)
                page_chars += len(text)
                blocks.append(
                    LayoutBlock(
                        page=pno,
                        block_index=bi,
                        kind="body",
                        text=text,
                        bbox=(
                            round(rect.x0, 1),
                            round(rect.y0, 1),
                            round(rect.x1, 1),
                            round(rect.y1, 1),
                        ),
                        size=size,
                        font=font,
                        bold=bold,
                        n_chars=len(text),
                    )
                )
            table_items: list[
                tuple[list[list[str | None]], tuple[float, float, float, float] | None]
            ] = [(t.extract(), _rect_tuple(tr)) for t, tr in zip(tabs, table_rects, strict=False)]
            if not table_items and page_chars >= SCAN_PAGE_CHARS:
                if plumber_doc is None:
                    try:
                        import pdfplumber

                        plumber_doc = pdfplumber.open(str(pdf_path))
                    except Exception:
                        plumber_doc = False
                if plumber_doc:
                    table_items = _plumber_tables(plumber_doc, pno - 1)
            # 兜底矩形永远是整页——绝不再写 (0,0,0,0)，否则该表格块无法高亮回链
            page_rect = (
                0.0,
                0.0,
                round(float(page.rect.width), 1),
                round(float(page.rect.height), 1),
            )
            for ti, (rows, tbbox) in enumerate(table_items):
                text = render_table(rows)
                if not text:
                    continue
                blocks.append(
                    LayoutBlock(
                        page=pno,
                        block_index=10_000 + ti,
                        kind="table",
                        text=text,
                        bbox=tbbox or page_rect,
                        size=0.0,
                        font="",
                        bold=False,
                        n_chars=sum(len(c or "") for r in rows for c in r),
                    )
                )
            chars_per_page.append(page_chars)
        page_count = doc.page_count
    if plumber_doc:
        with contextlib.suppress(Exception):
            plumber_doc.close()
    mean = sum(chars_per_page) / max(1, page_count)
    all_text = "\n".join(b.text for b in blocks)
    quality = text_quality(all_text)
    stats: dict[str, Any] = {
        "doc_id": doc_id,
        "page_count": page_count,
        "chars_total": sum(chars_per_page),
        "chars_per_page_mean": round(mean, 1),
        "empty_pages": [i + 1 for i, c in enumerate(chars_per_page) if c < SCAN_PAGE_CHARS],
        "is_scanned": mean < SCAN_CHARS_PER_PAGE,
        "n_blocks": len(blocks),
        "n_tables": sum(1 for b in blocks if b.kind == "table"),
        "images": images,
        **quality,
    }
    return blocks, stats


def text_quality(text: str) -> dict[str, Any]:
    """可读性判定：汉字占比 / 可用字符占比 / 英文功能词占比 / 替换字符占比。

    三条独立判据（全 81 份真实语料实测口径）：
    ① 可用字符占比 < 0.85 —— 字形被映射到私用区或"犐犆犛"这类生僻汉区（GB7793 扫描件）；
    ② 替换字符占比 > 0.02 —— 缺 ToUnicode CMap；
    ③ 汉字 < 5% 且英文功能词 < 2% —— 拉丁 mojibake（"*rfP1ttm11:~m!ituANftt1"）。
    合法英文文档（WHO 身体活动指南）汉字为 0 但功能词占比 4.3%，不会被③误杀。
    """
    body = _SPACES.sub("", text)
    n = max(1, len(body))
    cjk_ratio = len(_CJK.findall(body)) / n
    legible_ratio = len(_LEGIBLE.findall(body)) / n
    stop_ratio = len(_STOPWORD.findall(text)) / n
    fffd_ratio = body.count("\ufffd") / n
    is_garbage = (
        legible_ratio < GARBAGE_LEGIBLE_RATIO
        or fffd_ratio > FFFD_RATIO
        or (cjk_ratio < GARBAGE_CJK_RATIO and stop_ratio < GARBAGE_STOP_RATIO)
    )
    return {
        "cjk_ratio": round(cjk_ratio, 4),
        "legible_ratio": round(legible_ratio, 4),
        "stop_ratio": round(stop_ratio, 4),
        "fffd_ratio": round(fffd_ratio, 4),
        "is_garbage": is_garbage,
    }


def page_texts(pdf_path: Path) -> dict[int, str]:
    """{PDF 页码: 该页纯文本}；目录识别与质量评估共用。"""
    out: dict[int, str] = {}
    with fitz.open(str(pdf_path)) as doc:
        for pno in range(1, doc.page_count + 1):
            out[pno] = doc[pno - 1].get_text("text")
    return out


def collect_margin_lines(pdf_path: Path) -> dict[int, list[tuple[str, float, float, float]]]:
    """{页码: [(行文本, y0, y1, 页高)]}；页眉页脚检测的输入。"""
    out: dict[int, list[tuple[str, float, float, float]]] = {}
    with fitz.open(str(pdf_path)) as doc:
        for pno in range(1, doc.page_count + 1):
            page = doc[pno - 1]
            height = float(page.rect.height) or 1.0
            rows: list[tuple[str, float, float, float]] = []
            for b in page.get_text("dict")["blocks"]:
                if b["type"] != 0:
                    continue
                for ln in b["lines"]:
                    text = "".join(str(sp.get("text", "")) for sp in ln["spans"]).strip()
                    if not text:
                        continue
                    y0, y1 = float(ln["bbox"][1]), float(ln["bbox"][3])
                    rows.append((text, y0, y1, height))
            out[pno] = rows
    return out


def _norm_for_cer(s: str) -> str:
    """NFKC 归一 + 去全部空白。空白必须去，否则段间距会被算成错字。"""
    return _SPACES.sub("", unicodedata.normalize("NFKC", s))


def _edit_distance(a: str, b: str) -> int:
    """纯 Python DP 编辑距离：不为一个函数引入 Levenshtein 依赖。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(ref: str, hyp: str) -> float:
    """字符准确率 = 1 - 编辑距离 / 参考长度。"""
    r, h = _norm_for_cer(ref), _norm_for_cer(hyp)
    if not r:  # 空参考页：缺陷页（有输出）算 0 分，双空才算 1 分
        return 0.0 if h else 1.0
    return max(0.0, 1.0 - _edit_distance(r, h) / len(r))


def evaluate_parse_quality(
    doc_id: str, pages: dict[int, str], refs: dict[int, str], threshold: float = 0.95
) -> dict[str, Any]:
    """refs: {页码: 参考文本}。返回逐页准确率与是否达标。"""
    per_page = {p: round(char_accuracy(refs[p], pages.get(p, "")), 4) for p in refs}
    worst = min(per_page.values()) if per_page else 0.0
    return {
        "doc_id": doc_id,
        "per_page": per_page,
        "min": worst,
        "mean": round(sum(per_page.values()) / len(per_page), 4) if per_page else 0.0,
        "passed": worst >= threshold,
        "threshold": threshold,
    }
