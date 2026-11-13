# P0 单元测试：judge 校准的统计量与防自欺校验。
import pytest

cal = pytest.importorskip("medevirag.eval.calibration",
                           reason="请先按操作手册落盘 src/medevirag/eval/calibration.py")


def test_spearman_perfect_and_reverse():
    assert round(cal.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]), 4) == 1.0
    assert round(cal.spearman([1, 2, 3, 4], [4, 3, 2, 1]), 4) == -1.0


def test_ranks_handle_ties_by_average_rank():
    assert cal._ranks([1, 1, 3]) == [1.5, 1.5, 3.0]


def test_cohen_kappa_known_value():
    a = [1, 1, 1, 0, 0]
    b = [1, 1, 0, 0, 0]
    assert abs(cal.cohen_kappa_binary(a, b) - 0.6153846153846154) < 1e-9


def test_severe_disagreement_rate_counts_gap_two_or_more():
    assert cal.severe_disagreement_rate([5, 3, 1], [3, 3, 1]) == pytest.approx(1 / 3)


def test_reject_auto_generated_annotations():
    rows = [{"case_id": "c" + str(i), "annotator": "auto",
             "faithfulness": 5, "evidence_recall": 5, "refusal": 5}
            for i in range(cal.EXPECTED_N)]
    problems = cal.validate_labels(rows)
    assert any("AI 代标" in p for p in problems)


def test_reject_offline_stub_judge_scores():
    problems = cal.validate_scores([{
        "case_id": "c1", "dim": "faithfulness", "score": 1.0,
        "judge_model": "offline_stub", "llm_mode": "fake"}])
    assert any("离线桩" in p for p in problems)


def test_severe_rate_rejects_unequal_length():
    with pytest.raises(ValueError):
        cal.severe_disagreement_rate([1, 2], [1])
