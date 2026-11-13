"""chunk 展示字段的索引期快照（docs/03 §T5 的 ChunkMetaIndex）。

为什么要它：检索时只需要 chunk 的展示与过滤字段，没必要重读 8MB 的
chunks.jsonl 并重新 json 解析。建索引时把这批字段抽成 chunk_meta.jsonl，
运行期 O(1) 取单条。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

META_FIELDS = (
    "chunk_id",
    "doc_id",
    "doc_title",
    "doc_type",
    "publisher",
    "publish_year",
    "lang",
    "section_path",
    "chunk_type",
    "text",
    "context_header",
    "evidence_level",
    "rec_strength",
    "page_start",
    "page_end",
)


def pick_meta(row: Mapping[str, Any]) -> dict[str, Any]:
    """从 chunks.jsonl 的一行里挑出展示字段。"""
    meta = {k: row.get(k) for k in META_FIELDS}
    meta["chunk_id"] = str(row.get("chunk_id") or "")
    meta["doc_id"] = str(row.get("doc_id") or "")
    meta["doc_title"] = str(row.get("doc_title") or "")
    meta["doc_type"] = str(row.get("doc_type") or "guideline")
    meta["publisher"] = str(row.get("publisher") or "")
    meta["publish_year"] = int(row.get("publish_year") or 0)
    meta["lang"] = str(row.get("lang") or "zh")
    meta["section_path"] = [str(s) for s in (row.get("section_path") or [])]
    meta["chunk_type"] = str(row.get("chunk_type") or "paragraph")
    meta["text"] = str(row.get("text") or "")
    meta["context_header"] = str(row.get("context_header") or "")
    meta["evidence_level"] = str(row.get("evidence_level") or "unknown")
    meta["rec_strength"] = str(row.get("rec_strength") or "unknown")
    meta["page_start"] = int(row.get("page_start") or 0)
    meta["page_end"] = int(row.get("page_end") or 0)
    return meta


def write_chunk_meta(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(pick_meta(row), ensure_ascii=False) + "\n")
    return len(rows)


class ChunkMetaIndex:
    """chunk_id -> 展示字段 的只读内存表。"""

    def __init__(self, rows: Mapping[str, dict[str, Any]]) -> None:
        self._rows = dict(rows)

    @classmethod
    def from_jsonl(cls, path: Path) -> ChunkMetaIndex:
        if not path.exists():
            raise FileNotFoundError(f"chunk meta not found: {path} (run s2_vectorize.py first)")
        rows: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows[str(obj["chunk_id"])] = obj
        return cls(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[str]:
        return iter(self._rows)

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        return self._rows.get(chunk_id)

    def ids(self) -> list[str]:
        return list(self._rows)
