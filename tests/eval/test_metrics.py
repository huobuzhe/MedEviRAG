# S5 T8：metrics 单测（docs/06 §6）。分母口径与空值语义逐条锁死。

from __future__ import annotations

import pytest

from medevirag.eval import metrics as M


def test_recall_denominator_is_gold_not_k():
    # 2 gold 命中 1 = 0.5；用 k 当分母会错算成 0.1（docs §5.2）
    assert M.recall_at_k(["a", "b"], ["b"] + ["x"] * 9, 10) == 0.5


def test_recall_empty_gold_raises():
    with pytest.raises(ValueError):
        M.recall_at_k([], ["a"])


def test_mrr_rank_inversion():
    assert M.mrr(["g"], ["x", "g", "y"]) == 0.5
    assert M.mrr(["g"], ["x"] * 10) == 0.0


def test_ndcg_gold_outranks_sibling():
    g = {"gold": 2, "sib": 1, "junk": 0}
    best = M.ndcg_at_k(g, ["gold", "sib", "junk"], 10)
    worse = M.ndcg_at_k(g, ["sib", "gold", "junk"], 10)
    assert best > worse > 0.0


def test_landing_rate_undefined_without_citations():
    with pytest.raises(ValueError):
        M.citation_landing_rate(0, 0)
    assert M.citation_landing_rate(1, 4) == 0.75


def test_support_rate_only_counts_cited_sentences():
    rows = [{"cited_chunk_ids": (), "text": "模板句"},
            {"cited_chunk_ids": ("a",), "landing_ok": True, "entailed": True},
            {"cited_chunk_ids": ("b",), "landing_ok": True, "entailed": False}]
    assert M.sentence_support_rate(rows) == 0.5


def test_refusal_metrics_direction():
    rows = [{"must_refuse": True, "status": "refused"},
            {"must_refuse": True, "status": "answered"},
            {"must_refuse": False, "status": "refused"},
            {"must_refuse": False, "status": "answered"}]
    assert M.refusal_recall(rows) == 0.5
    assert M.over_refusal_rate(rows) == 0.5


def test_reason_accuracy():
    rows = [{"status": "refused", "must_refuse": True, "refusal_reason": "off_domain", "gold_refuse_kind": "out_of_scope"},
            {"status": "refused", "must_refuse": True, "refusal_reason": "low_support", "gold_refuse_kind": "insufficient_evidence"},
            {"status": "refused", "must_refuse": True, "refusal_reason": "low_support", "gold_refuse_kind": "out_of_scope"}]
    # REASON_TO_KIND 映射后：off_domain->out_of_scope 命中；low_support->insufficient 命中；第三条错配
    assert M.refusal_reason_accuracy(rows) == pytest.approx(2 / 3)


def test_ece_perfect_and_bad():
    assert M.expected_calibration_error([1.0, 1.0], [True, True]) == 0.0
    # conf 0.4 组全错、conf 0.9 组全对：ECE = 0.5*|0-0.4| + 0.5*|1-0.9| = 0.25
    e = M.expected_calibration_error([0.4] * 5 + [0.9] * 5, [False] * 5 + [True] * 5)
    assert abs(e - 0.25) < 1e-9


def test_mean_or_none_never_zero():
    assert M.mean_or_none([]) is None
    assert M.mean_or_none([1.0, 2.0]) == 1.5


def test_bootstrap_ci_covers_point_estimate():
    flags = [True] * 8 + [False] * 2
    lo, hi = M.bootstrap_ci(flags)
    assert lo <= 0.8 <= hi

