"""回链定位：表格块真实矩形、block_index 透传、单页合并块 bbox 并集。

这三个断言直接守住 S1 返工的两个缺陷：表格块 bbox 恒 (0,0,0,0)（无法高亮）
与 locator.block_index 恒 0（无法按块回跳）。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from medevirag.ingest import chunker
from medevirag.ingest.chunker import Unit
from medevirag.ingest.parser import _rect_tuple, extract_blocks

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_table_pdf(path: Path) -> None:
    """造一份带真实表格线的单页 PDF：1 段正文 + 1 张 2x2 表格。"""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "正文段落：高血压患者应限制钠盐摄入。", fontsize=12)
    x0, y0, x1, y1 = 72.0, 100.0, 400.0, 160.0
    page.draw_line((x0, y0), (x1, y0))
    page.draw_line((x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2))
    page.draw_line((x0, y1), (x1, y1))
    for x in (x0, (x0 + x1) / 2, x1):
        page.draw_line((x, y0), (x, y1))
    page.insert_text((x0 + 6, y0 + 20), "分类", fontsize=11)
    page.insert_text(((x0 + x1) / 2 + 6, y0 + 20), "收缩压", fontsize=11)
    page.insert_text((x0 + 6, y0 + 50), "正常血压", fontsize=11)
    page.insert_text(((x0 + x1) / 2 + 6, y0 + 50), "<120", fontsize=11)
    doc.save(str(path))
    doc.close()


def test_rect_tuple_rejects_degenerate() -> None:
    """退化矩形不算有效矩形——(0,0,0,0) 必须被挡掉，不允许流进 locator。"""
    assert _rect_tuple((0.0, 0.0, 0.0, 0.0)) is None
    assert _rect_tuple(None) is None
    assert _rect_tuple((72.04, 100.06, 400.0, 160.0)) == (72.0, 100.1, 400.0, 160.0)


def test_table_block_bbox_is_real(tmp_path: Path) -> None:
    """表格块必须带真实表格矩形（返工缺陷 1）。"""
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "t.pdf"
    _make_table_pdf(pdf)
    blocks, stats = extract_blocks(pdf, "t")
    tables = [b for b in blocks if b.kind == "table"]
    assert tables, "合成 PDF 应至少解析出 1 张表格"
    rect = fitz.Rect(*tables[0].bbox)
    assert rect.width > 0 and rect.height > 0, f"表格块 bbox 退化：{tables[0].bbox}"
    assert rect.width < 590 and rect.height < 830, "表格块不应退化成整页矩形"
    assert stats["n_tables"] >= 1


def test_locator_block_index_transparent(make_block, fake_count) -> None:
    """版面块的 block_index 与 bbox 必须一起透传到 locator（返工缺陷 1 后半）。"""
    blocks = [
        make_block(page=2, block_index=7, text="甲" * 300, bbox=(72.0, 100.0, 500.0, 160.0)),
    ]
    units = chunker.to_units(blocks, fake_count)
    assert units[0].block_index == 7
    loc = chunker._locator_of(units)
    assert loc["block_index"] == 7
    assert loc["page"] == 2
    assert loc["bbox"] == [72.0, 100.0, 500.0, 160.0]


def test_single_page_merged_bbox_is_union(make_block, fake_count) -> None:
    """同页多块合并的 chunk，bbox 取并集，高亮才能盖住整块（返工缺陷 2）。"""
    blocks = [
        make_block(page=3, block_index=0, text="甲" * 120, bbox=(72.0, 100.0, 300.0, 116.0)),
        make_block(page=3, block_index=1, text="乙" * 120, bbox=(72.0, 120.0, 500.0, 180.0)),
        make_block(page=3, block_index=2, text="丙" * 120, bbox=(72.0, 184.0, 480.0, 240.0)),
    ]
    units = chunker.to_units(blocks, fake_count)
    loc = chunker._locator_of(units)
    assert loc["bbox"] == [72.0, 100.0, 500.0, 240.0]
    assert loc["block_index"] == 0
    assert loc["page"] == 3


def test_cross_page_bbox_keeps_first(make_block, fake_count) -> None:
    """跨页块只能定位首段：bbox 保持首块矩形，不跨页拼接。"""
    units = [
        Unit("甲" * 120, "body", 3, (72.0, 600.0, 500.0, 700.0), 4),
        Unit("乙" * 120, "body", 4, (72.0, 80.0, 500.0, 200.0), 1),
    ]
    loc = chunker._locator_of(units)
    assert loc["page"] == 3
    assert loc["bbox"] == [72.0, 600.0, 500.0, 700.0]
    assert loc["block_index"] == 4


def test_chunk_section_locator_nonzero(
    make_block, make_section: Callable[..., object], fake_count
) -> None:
    """端到端经过 chunk_section 后 locator 仍是真实值，不被写回 0。"""
    blocks = [
        make_block(page=6, block_index=11, text="甲" * 200, bbox=(72.0, 100.0, 500.0, 160.0)),
    ]
    sec = make_section(section_no="3.2.1", title="诊室血压", page_start=6, page_end=6)
    chunks = chunker.chunk_section(sec, chunker.to_units(blocks, fake_count), fake_count)
    assert chunks
    loc = chunks[0]["locator"]
    assert loc["block_index"] == 11
    assert loc["bbox"] == [72.0, 100.0, 500.0, 160.0]
