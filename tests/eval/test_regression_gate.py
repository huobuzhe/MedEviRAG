# S5 T10/T11：回归门禁反向测试——注入回归必须被拦（docs/06 §6.1）。

from __future__ import annotations

from medevirag.common.config import get_settings
from medevirag.eval.regression import compare


def _mk_run(faithfulness: float, recall: float, sha: str = "s1") -> dict:
    row = {"case_id": "T-01", "must_refuse": False, "status": "answered",
           "refusal_reason": None, "gold_refuse_kind": None,
           "retrieved": ["a"], "graded": {"a": 2},
           "total_citation_count": 1, "dangling_citation_count": 0,
           "sentences": [], "correct": True, "judge_error": False,
           "judge": {"faithfulness": faithfulness}}
    run = {"metrics": {"frozen_set_sha256": sha, "faithfulness": faithfulness,
                       "recall_at_10": recall,
                       "refusal_recall": 0.9, "over_refusal_rate": 0.05,
                       "mrr": 0.7, "ndcg_at_10": 0.75,
                       "citation_landing_rate": 0.95,
                       "sentence_support_rate": 0.9,
                       "refusal_reason_accuracy": 0.9},
           "cases": [row]}
    run["by_case"] = {c["case_id"]: c for c in run["cases"]}   # load_run 加工字段
    return run


def test_identical_runs_pass():
    cfg = get_settings()
    _, problems = compare(_mk_run(0.9, 0.9), _mk_run(0.9, 0.9), cfg)
    assert problems == []


def test_hard_drop_triggers():
    cfg = get_settings()
    # 忠实度 0.9 -> 0.6，跌幅 0.3 > hard_drop 0.05
    _, problems = compare(_mk_run(0.9, 0.9), _mk_run(0.6, 0.9), cfg)
    assert any("HARD_DROP" in p for p in problems), problems


def test_threshold_fail_triggers():
    cfg = get_settings()
    # 忠实度 0.5 < 阈值 0.80
    _, problems = compare(_mk_run(0.9, 0.9), _mk_run(0.5, 0.9), cfg)
    assert any("THRESHOLD_FAIL" in p for p in problems), problems


def test_freeze_mismatch_blocks_comparison():
    cfg = get_settings()
    _, problems = compare(_mk_run(0.9, 0.9, sha="a"), _mk_run(0.9, 0.9, sha="b"), cfg)
    assert any("FREEZE_MISMATCH" in p for p in problems)


def test_over_refusal_direction_not_inverted():
    cfg = get_settings()
    # 过度拒答 0.05 -> 0.20：方向变坏且超过 hard_drop
    base = _mk_run(0.9, 0.9)
    base["metrics"]["over_refusal_rate"] = 0.05
    cur = _mk_run(0.9, 0.9)
    cur["metrics"]["over_refusal_rate"] = 0.20
    _, problems = compare(base, cur, cfg)
    assert any("over_refusal_rate" in p for p in problems), problems


def test_noise_band_not_flagged():
    cfg = get_settings()
    # 0.9 -> 0.89：跌幅 0.01 < max_drop 0.02，不报警
    _, problems = compare(_mk_run(0.9, 0.9), _mk_run(0.89, 0.9), cfg)
    assert not any("HARD_DROP" in p for p in problems)

