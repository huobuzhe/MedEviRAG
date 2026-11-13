"""T4 切分：表格不跨页（3 条）+ context_header（3 条）+ 硬约束。"""

from __future__ import annotations

import pytest

from medevirag.ingest import chunker
from medevirag.ingest.chunker import Unit


def _table_text(rows: int) -> str:
    lines = ["| 食物 | 份量 |", "| --- | --- |"]
    lines += ["| 谷类" + str(i) + " | " + str(i * 10) + "g |" for i in range(rows)]
    return "\n".join(lines)


def test_table_not_cross_page(make_block, make_section, fake_count) -> None:
    """跨 2 页的剂量表必须拆成 2 个 chunk，且各自 page_start == page_end。"""
    blocks = [
        make_block(page=5, block_index=0, text=_table_text(30), kind="table"),
        make_block(page=6, block_index=0, text=_table_text(30), kind="table"),
    ]
    sec = make_section(page_start=5, page_end=6)
    units = chunker.to_units(blocks, fake_count)
    items = chunker.chunk_section(sec, units, fake_count)
    items = chunker.expand_overflow(items, fake_count)
    assert items
    for item in items:
        assert item["page_start"] == item["page_end"], item
    assert {item["page_start"] for item in items} == {5, 6}


def test_table_split_rows_keeps_header(make_section, fake_count) -> None:
    """按行切表时每一片都要重带表头，续片带「（续）」标记。"""
    unit = Unit(_table_text(60), "table", 7, (1.0, 2.0, 3.0, 4.0))
    pieces = chunker.split_table_by_rows(unit, fake_count)
    assert len(pieces) >= 2
    for piece in pieces:
        assert piece.page == 7
        assert piece.text.startswith("| 食物 | 份量 |")
    assert any("（续）" in p.text for p in pieces[1:])


def test_table_never_exceeds_hard_cap(make_section, fake_count) -> None:
    """单行就超硬上限的畸形表也必须被 _hard_cap 切开，不留超长块。"""
    huge_row = "| 超长行 | " + "数据" * 300 + " |"
    unit = Unit("| 食物 | 份量 |\n| --- | --- |\n" + huge_row, "table", 9, (1.0, 2.0, 3.0, 4.0))
    pieces = chunker.split_table_by_rows(unit, fake_count)
    assert pieces
    assert all(fake_count(p.text) <= chunker.MAX_TOKENS for p in pieces)


@pytest.mark.parametrize(
    ("path", "expect"),
    [
        (["三、诊断", "3.2.1 诊室血压"], "《高血压指南》 > 3.2.1 诊室血压"),
        (["一、前言"], "《高血压指南》 > 一、前言"),
        ([], "《高血压指南》"),
    ],
)
def test_context_header(path: list[str], expect: str) -> None:
    assert chunker.context_header("高血压指南", path) == expect


def test_context_header_empty_path_has_no_arrow() -> None:
    """section_path 为空时退化为《标题》，不得出现 " > "。"""
    assert " > " not in chunker.context_header("某标准", [])


def test_build_chunk_id_and_hash(fake_count) -> None:
    assert chunker.build_chunk_id("doc1", "3.2.1", 1) == "doc1#3.2.1#c01"
    h1 = chunker.content_hash("3.2.1  诊室血压\n")
    h2 = chunker.content_hash("3.2.1 诊室血压")
    assert h1 == h2  # 归一化后空白不影响 hash
    assert h1.startswith("sha256:")


def test_assert_hard_constraints_rejects_cross_page_table(make_block, make_section, fake_count) -> None:
    bad = {
        "chunk_id": "doc#t#c01",
        "chunk_type": "table",
        "text": "| a | b |",
        "token_len": 10,
        "page_start": 3,
        "page_end": 4,
        "context_header": "《X》 > 表",
        "locator": {"page": 3, "block_index": 0, "bbox": [1.0, 2.0, 3.0, 4.0]},
    }
    with pytest.raises(AssertionError):
        chunker.assert_hard_constraints(bad)


def test_short_groups_rebalanced_into_target_range(fake_count) -> None:
    """单节内 6 个 120 token 的短单元应被重新分配，而不是切成 6 个碎片。"""
    units = [Unit("正文" * 60, "body", 2, (1.0, 2.0, 3.0, 4.0)) for _ in range(6)]
    groups = [[u] for u in units]
    assert all(fake_count(u.text) < chunker.MIN_TOKENS for u in units)
    merged = chunker._rebalance(groups, fake_count)
    assert len(merged) < len(groups)
    assert all(fake_count(chunker._group_text(g)) >= chunker.MIN_TOKENS for g in merged)


def test_merge_tail_absorbs_short_last_group(fake_count) -> None:
    """尾组过短时并入前一组，前提是合并后不超 TARGET_MAX。"""
    prev = [Unit("正文" * 150, "body", 2, (1.0, 2.0, 3.0, 4.0))]
    tail = [Unit("短句" * 10, "body", 2, (1.0, 2.0, 3.0, 4.0))]
    merged = chunker.merge_tail([prev, tail], fake_count)
    assert len(merged) == 1
