# S5 T6：judges 单测（veto / 跨族断言 / 解析健壮性）。

from __future__ import annotations

import pytest

from medevirag.eval.judges import RUBRIC_V1, aggregate, assert_cross_family, parse_judge_json


def test_rubric_12_items_3_dims():
    dims = {it.dim for it in RUBRIC_V1}
    assert dims == {"faithfulness", "evidence_recall", "refusal"}
    assert len(RUBRIC_V1) == 12


def test_veto_item_zeroes_dim():
    s, p = aggregate("faithfulness", {"F1": True, "F2": False, "F3": True, "F4": True, "F5": True})
    assert s == 0.0 and p is False


@pytest.mark.parametrize("dim,atomic,score", [
    ("faithfulness", {"F1": True, "F2": True, "F3": True, "F4": True, "F5": True}, 1.0),
    ("faithfulness", {"F1": True, "F2": True, "F3": False, "F4": True, "F5": True}, 9 / 11),
    ("refusal", {"R1": True, "R2": True, "R3": False, "R4": True}, 8 / 10),
])
def test_aggregate_weights(dim, atomic, score):
    s, p = aggregate(dim, atomic)
    assert abs(s - score) < 1e-9


def test_cross_family_violation_raises():
    with pytest.raises(SystemExit):
        assert_cross_family("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-chat")


def test_cross_family_ok():
    assert_cross_family("deepseek-ai/DeepSeek-V4.1-Flash", "qwen3.8-flash")


def test_parse_tolerates_fences_and_chatter():
    raw = "前置说明```json\n{\"items\": [{\"item_id\": \"F1\", \"ok\": true}]}\n```后缀"
    obj = parse_judge_json(raw)
    assert obj["items"][0]["item_id"] == "F1"

