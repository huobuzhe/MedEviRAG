"""T6 落盘：PG 不可用时优雅降级 + jsonl 幂等。"""

from __future__ import annotations

import json
from pathlib import Path

from medevirag.ingest import store


def test_try_load_without_dsn() -> None:
    conn, reason = store.try_load(None)
    assert conn is None
    assert "跳过落库" in reason


def test_try_load_bad_dsn_degrades() -> None:
    conn, reason = store.try_load("postgresql://nobody@127.0.0.1:1/none")
    assert conn is None
    assert "跳过落库" in reason


def test_write_jsonl_sorted_and_idempotent(tmp_path: Path) -> None:
    rows = [
        {"chunk_id": "b#1", "text": "乙"},
        {"chunk_id": "a#1", "text": "甲"},
    ]
    path = tmp_path / "chunks.jsonl"
    assert store.write_jsonl(path, rows) == 2
    first = path.read_bytes()
    store.write_jsonl(path, list(reversed(rows)))
    assert path.read_bytes() == first  # 顺序无关 -> 字节一致
    line = path.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(line)["chunk_id"] == "a#1"
