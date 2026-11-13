"""文档级元数据抽取（docs/02 4.5 T5）。

全部基于规则，零 LLM：标题/年份/发布机构在指南类 PDF 的首页版式里高度稳定，
规则命中率高于小模型，而且结果可解释、可复查、可复现。
"""

from __future__ import annotations

import re
from pathlib import Path

from medevirag.ingest.parser import LayoutBlock

DOC_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("guideline", ("指南", "guideline", "防治指南", "临床指南")),
    ("consensus", ("共识", "consensus", "专家共识")),
    ("manual", ("手册", "规范", "manual", "标准", "操作规程")),
    ("paper", ("journal", "article", "abstract", "研究", "期刊")),
)

PUBLISHER_WHITELIST: tuple[str, ...] = (
    "中华医学会",
    "中国高血压联盟",
    "国家心血管病中心",
    "国家卫生健康委员会",
    "国家卫生健康委",
    "中国医师协会",
    "中国疾病预防控制中心",
    "WHO",
    "ESC",
    "AHA",
    "ACC",
    "NCCN",
    "NICE",
    "KDIGO",
)

_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_WS = re.compile(r"\s+")
_TITLE_DECOR = re.compile(r"[《》〈〉]|\(|\)|（|）.*?版?$")


def _cjk_count(text: str) -> int:
    return len(_CJK.findall(text))


def extract_doc_type(title: str, first_page_text: str) -> str:
    """四类 doc_type：guideline / consensus / manual / paper，兜底 paper。"""
    hay = (title + "\n" + first_page_text[:1500]).lower()
    for doc_type, keys in DOC_TYPE_RULES:
        if any(k.lower() in hay for k in keys):
            return doc_type
    return "paper"


def extract_publisher(title: str, first_page_text: str, file_name: str) -> str:
    """发布机构白名单匹配；命中多个取最长（"国家卫生健康委员会"优先于"国家卫生健康委"）。"""
    hay = title + file_name + first_page_text[:1500]
    hits = [p for p in PUBLISHER_WHITELIST if p in hay]
    if not hits:
        return "unknown"
    return max(hits, key=len)


def extract_publish_year(*candidates: str | Path) -> int | None:
    """从若干候选文本里找 1990~2026 之间的年份，取最大值（封面年份通常最新）。"""
    years: list[int] = []
    for c in candidates:
        text = str(c)
        for m in _YEAR.finditer(text):
            y = int(m.group(1))
            if 1990 <= y <= 2026:
                years.append(y)
    return max(years) if years else None


def extract_lang(text: str) -> str:
    """汉字占比 >=15% 判中文；空文本返回 unknown（调用方需按 Lang 字面量兜底）。"""
    if not text.strip():
        return "unknown"
    ratio = _cjk_count(text) / len(text)
    return "zh" if ratio >= 0.15 else "en"


def _norm_title_text(s: str) -> str:
    s = _WS.sub(" ", s).strip()
    s = s.replace("《", "").replace("》", "")
    # 去掉汉字之间被 PDF 抽取插入的空格："成人 高血压 食养指南" -> "成人高血压食养指南"
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)


def extract_title(blocks: list[LayoutBlock]) -> str:
    """首页最大字号块拼标题；抽不到返回空串，由调用方退化到文件名。"""
    page1 = [b for b in blocks if b.page == 1 and b.kind == "body" and b.n_chars >= 2]
    if not page1:
        return ""
    top = max(b.size for b in page1)
    cands = [b for b in page1 if b.size >= top - 0.5]
    cands.sort(key=lambda b: b.block_index)
    parts: list[str] = []
    for b in cands[:3]:
        t = _norm_title_text(b.text.splitlines()[0] if b.text else "")
        if t:
            parts.append(t)
    title = " ".join(parts).strip()
    return _TITLE_DECOR.sub("", title) if len(title) > 60 else title
