# P1 单元测试：RuleJudge 不是桩、veto 生效、扰动检验指标可信。
import pytest

from medevirag.eval.judge_validation import _case, _perturbed, _score, run_validation
from medevirag.eval.rule_judge import RuleJudgeLLM

BASE = {
    "case": _case("base"),
    "answer": "诊室血压≥140/90 mmHg 可诊断高血压，需非同日三次测量确认。[1]",
    "evidence": [
        {"chunk_id": "CN-HTN-2024#3.2.1#c01",
         "text": "诊室血压≥140/90 mmHg，需非同日三次测量确认",
         "section_path": ["3.2.1"], "doc_title": "指南"},
    ],
    "status": "answered",
    "cmap": {"1": "CN-HTN-2024#3.2.1#c01"},
}


def test_rule_judge_is_not_an_all_true_stub():
    j = RuleJudgeLLM()
    base = _score(j, BASE, "faithfulness")
    bad = _score(j, _perturbed("numeric_corrupt", 0), "faithfulness")
    assert base > 0
    assert bad < base
    assert j.calls >= 2


def test_numeric_corruption_triggers_f2_veto():
    j = RuleJudgeLLM()
    assert _score(j, _perturbed("numeric_corrupt", 1), "faithfulness") == 0.0


def test_wrong_refusal_triggers_r1_veto():
    j = RuleJudgeLLM()
    assert _score(j, _perturbed("wrong_refusal", 0), "refusal") == 0.0


def test_invariance_under_length_position_and_format():
    j = RuleJudgeLLM()
    base = _score(j, BASE, "faithfulness")
    for kind in ("length_padding", "evidence_reorder", "format_markdown"):
        assert abs(_score(j, _perturbed(kind, 0), "faithfulness") - base) <= 0.05


def test_validation_report_has_strong_agreement():
    v = run_validation(repeats=3)
    assert v["n_pairs"] == 21
    assert v["direction_agreement"] >= 0.75
    assert v["invariance_agreement"] >= 0.75
    assert v["veto_catch_rate"] >= 0.75
    assert v["judge_error_rate"] == 0.0
