"""T6 引用解析与回链（docs/04 §T6）。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"

CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9._-]+#\d+(?:\.\d+)*#c\d{2}$")

CITE_RE = re.compile(r"[\[［【]\s*[Cc]\s*(\d{1,2})\s*[\]］】]")
SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*")

DETAIL_KEYS = (
    "chunk_id", "doc_id", "doc_title", "section_path", "chunk_type",
    "page", "bbox", "block_index", "text", "context_header", "evidence_level", "url",
)


@dataclass
class Citation:
    marker: str
    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: list[Any]
    page: int
    locator: dict[str, Any]
    quote: str
    url: str


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s for s in SENT_SPLIT.split(text) if s.strip()]


def markers_in(text: str) -> list[str]:
    seen = set()
    out = []
    for m in CITE_RE.finditer(text or ""):
        mk = f"C{m.group(1)}"
        if mk not in seen:
            seen.add(mk)
            out.append(mk)
    return out


def make_url(doc_id: str, locator: Mapping[str, Any]) -> str:
    page = locator.get("page", 0) if isinstance(locator, Mapping) else 0
    bbox = locator.get("bbox") if isinstance(locator, Mapping) else None
    bbox_s = ",".join(str(x) for x in bbox) if bbox else ""
    return f"medevirag://doc/{doc_id}?page={page}&bbox={bbox_s}"


def build_citations(answer: str, evidence: dict[str, dict[str, Any]]) -> list[Citation]:
    out: list[Citation] = []
    seen: set[str] = set()
    for marker in markers_in(answer):
        if marker in seen or marker not in evidence:
            continue
        seen.add(marker)
        e = evidence[marker]
        doc_id = e.get("doc_id") or e["chunk_id"].split("#")[0]
        out.append(
            Citation(
                marker=marker,
                chunk_id=e["chunk_id"],
                doc_id=doc_id,
                doc_title=e["doc_title"],
                section_path=list(e["section_path"]),
                page=int(e["page"]),
                locator=dict(e["locator"]),
                quote=e["text"],
                url=make_url(doc_id, e["locator"]),
            )
        )
    return out


@lru_cache(maxsize=1)
def _chunk_index() -> dict[str, dict[str, Any]]:
    import json as _json
    idx: dict[str, dict[str, Any]] = {}
    with open(str(CHUNKS_PATH), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = _json.loads(line)
            idx[row["chunk_id"]] = row
    return idx


def get_citation_detail(chunk_id: str) -> dict[str, Any]:
    """取单条引用详情。

    格式校验放宽为「严格格式匹配 **或** 真实索引里存在」：本语料的 chunk_id 含中文/特殊字符
    （如 0108_135352_975#(һ)#c03），严格 ASCII 正则匹配不到，若只按正则拦截，
    CLI 将无法查看任何真实语料的原文——那等于把可用性做死。
    仍保留"格式坏且查不到"→ValueError、格式对但查不到→KeyError 的区分。
    """
    raw = chunk_id or ""
    idx = _chunk_index()
    if not CHUNK_ID_RE.match(raw) and raw not in idx:
        raise ValueError(f"chunk_id 格式非法: {chunk_id!r}（应形如 CN-HTN-2024#3.2.1#c02）")
    if raw not in idx:
        raise KeyError(f"chunk_id 不存在: {chunk_id}")
    row = idx[chunk_id]
    loc = row.get("locator") or {}
    detail = {
        "chunk_id": chunk_id,
        "doc_id": row["doc_id"],
        "doc_title": row["doc_title"],
        "section_path": row["section_path"],
        "chunk_type": row["chunk_type"],
        "page": loc.get("page", row.get("page_start")),
        "bbox": loc.get("bbox"),
        "block_index": loc.get("block_index"),
        "text": row["text"],
        "context_header": row["context_header"],
        "evidence_level": row.get("evidence_level", "unknown"),
        "url": make_url(row["doc_id"], loc),
    }
    assert tuple(detail) == DETAIL_KEYS, "返回键集合必须与 DETAIL_KEYS 完全一致"
    return detail


def citation_to_dict(c: Citation) -> dict[str, Any]:
    return asdict(c)
