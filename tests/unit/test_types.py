"""Chunk 的硬约束与 RefusalReason 的 15 值（docs/01 §T4 验收 A11）。"""

from __future__ import annotations

import pytest

from medevirag.common.types import Chunk, Locator, RefusalReason


def make(**overrides: object) -> Chunk:
    base: dict[str, object] = {
        "chunk_id": "CN-HTN-2024#3.2.1#c02",
        "parent_id": "CN-HTN-2024#3.2.1",
        "doc_id": "CN-HTN-2024",
        "doc_title": "中国高血压防治指南（2024年修订版）",
        "doc_type": "guideline",
        "publisher": "中国高血压联盟",
        "publish_year": 2024,
        "lang": "zh",
        "section_path": ["3 血压测量", "3.2 诊室血压"],
        "section_level": 3,
        "page_start": 12,
        "page_end": 12,
        "chunk_type": "recommendation",
        "text": "推荐诊室血压≥140/90 mmHg 诊断为高血压。",
        "context_header": "《中国高血压防治指南（2024年修订版）》 > 3.2.1 诊室血压",
        "evidence_level": "A",
        "rec_strength": "I",
        "grade_system": "GRADE",
        "token_len": 38,
        "content_hash": Chunk.compute_hash("推荐诊室血压>=140/90"),
        "locator": Locator(page=12, block_index=3, bbox=(72.0, 300.0, 520.0, 336.0)),
    }
    base.update(overrides)
    return Chunk(**base)  # type: ignore[arg-type]


def test_valid_chunk() -> None:
    assert make().token_len == 38


def test_hash_prefix_required() -> None:
    with pytest.raises(ValueError):
        make(content_hash="deadbeef")


def test_page_order() -> None:
    with pytest.raises(ValueError):
        make(page_start=12, page_end=10)


def test_single_page_needs_bbox() -> None:
    with pytest.raises(ValueError):
        make(locator=Locator(page=12, block_index=3, bbox=None))


def test_table_must_not_span_pages() -> None:
    with pytest.raises(ValueError):
        make(chunk_type="table", page_start=12, page_end=13)


def test_refusal_reason_count() -> None:
    assert len(list(RefusalReason)) == 15


def test_to_dict_roundtrip() -> None:
    d = make().to_dict()
    assert len(d) == 21
    assert d["locator"]["page"] == 12  # type: ignore[index]
