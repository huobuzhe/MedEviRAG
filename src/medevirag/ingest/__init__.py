"""S1 语料摄取：抓取 → 解析 → 章节识别 → 父子切分 → 元数据/证据抽取 → 落库。

本包在无 Docker / 无 PG / 无网络的环境下必须能 import 并跑单测：
网络只出现在 fetcher（且 client 可注入），PG 只出现在 store（且驱动缺失即降级）。
"""

from __future__ import annotations

from medevirag.ingest.chunker import (
    MAX_TOKENS,
    MIN_TOKENS,
    Unit,
    build_chunk_id,
    chunk_section,
    context_header,
    make_token_counter,
)
from medevirag.ingest.evidence import coverage_report, extract_evidence
from medevirag.ingest.parser import LayoutBlock, extract_blocks
from medevirag.ingest.pipeline import DocResult, run_doc, stats_of
from medevirag.ingest.sectionizer import Section, build_sections, detect_headings

__all__ = [
    "MAX_TOKENS",
    "MIN_TOKENS",
    "DocResult",
    "LayoutBlock",
    "Section",
    "Unit",
    "build_chunk_id",
    "build_sections",
    "chunk_section",
    "context_header",
    "coverage_report",
    "detect_headings",
    "extract_blocks",
    "extract_evidence",
    "make_token_counter",
    "run_doc",
    "stats_of",
]
