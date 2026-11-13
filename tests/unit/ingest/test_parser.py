"""T2 解析：表格渲染、文本质量判定、CER 计算、扫描件判定。"""

from __future__ import annotations

from pathlib import Path

import pytest

from medevirag.ingest.parser import (
    LayoutBlock,
    char_accuracy,
    evaluate_parse_quality,
    render_table,
    text_quality,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "ingest"


def test_render_table_pipes() -> None:
    text = render_table([["食物", "份量"], ["谷类", "50g"]])
    assert text.splitlines()[0] == "| 食物 | 份量 |"
    assert text.splitlines()[1] == "|---|---|"


def test_render_table_handles_none() -> None:
    text = render_table([[None, "x"]])
    assert "|  | x |" in text


def test_text_quality_flags_garbage() -> None:
    good = text_quality("高血压患者应限制钠盐摄入" * 20)
    assert good["is_garbage"] is False
    bad = text_quality("*rfP1ttm11:~m!ituANftt1" * 20)
    assert bad["is_garbage"] is True


def test_char_accuracy_identical_is_one() -> None:
    assert char_accuracy("高血压 诊断标准", "高血压诊断标准") == pytest.approx(1.0)


def test_char_accuracy_partial() -> None:
    got = char_accuracy("高血压诊断标准", "高血压治疗标准")
    assert 0.5 < got < 1.0


def test_char_accuracy_empty_ref() -> None:
    assert char_accuracy("", "任意") == 0.0


def test_evaluate_parse_quality_on_fixture() -> None:
    """夹具页文本与参考一致时逐页准确率应为 1，且整体达标。"""
    ref = (FIXTURES / "0108_135352_975_p3_ref.txt").read_text(encoding="utf-8")
    res = evaluate_parse_quality("0108_135352_975", {3: ref}, {3: ref})
    assert res["passed"] is True
    assert res["per_page"][3] == pytest.approx(1.0)


def test_fixture_blocks_rebuild_layout_block() -> None:
    """夹具块 JSON 能还原成 LayoutBlock，供离线回归使用。"""
    import json

    rows = json.loads((FIXTURES / "sample_blocks.json").read_text(encoding="utf-8"))
    blocks = [LayoutBlock(**r) for r in rows]
    assert [b.kind for b in blocks] == ["body", "body", "table"]
    assert blocks[1].bbox is not None and len(blocks[1].bbox) == 4
