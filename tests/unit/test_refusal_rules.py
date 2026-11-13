"""A4 门禁 + 17 条规则正反例（docs/05 §6）。"""

from __future__ import annotations

import pytest
from tests.fixtures.builders import build_conf, build_ctx, load_cases

from medevirag.rag.refusal import refusal_config
from medevirag.rag.refusal_rules import RULE_ORDER, evaluate_non_score, evaluate_score

CASES = load_cases()
RULE_IDS = [f"R-{i:02d}" for i in range(1, 18)]
CFG = refusal_config()


def _final_hit(case):
    ctx = build_ctx(case)
    hit = evaluate_non_score(ctx, CFG)
    if hit is not None and hit.action == "refuse":
        return hit
    bd = build_conf(case)
    hit2 = evaluate_score(ctx, bd, CFG)
    return hit2 if hit2 is not None else hit


@pytest.mark.parametrize("case",
                         [c for c in CASES if c["expect"] == "refused"],
                         ids=[c["case_id"] for c in CASES if c["expect"] == "refused"])
def test_should_refuse(case):
    hit = _final_hit(case)
    assert hit is not None and hit.action == "refuse", \
        f"{case['case_id']} 应被 {case['gold_rule_id']} 拒绝"
    assert hit.rule_id == case["gold_rule_id"], \
        f"{case['case_id']} 期望 {case['gold_rule_id']}，实际 {hit.rule_id}"
    assert hit.reason.value == case["gold_reason"]


@pytest.mark.parametrize("case",
                         [c for c in CASES if c["expect"] == "answered"],
                         ids=[c["case_id"] for c in CASES if c["expect"] == "answered"])
def test_should_not_refuse(case):
    hit = _final_hit(case)
    refused = hit is not None and hit.action == "refuse"
    assert not refused, \
        f"{case['case_id']} 不该被拒，却被 {hit.rule_id if hit else None} 拒了"
    if case.get("expect_hit"):          # degrade 规则（R-06/R-07）仍须触发
        assert hit is not None and hit.rule_id == case["expect_hit"]


def test_rule_coverage():
    """A4：17 条规则每条至少一个用例。"""
    covered = {c.get("gold_rule_id") for c in CASES if c.get("gold_rule_id")}
    covered |= {c.get("expect_hit") for c in CASES if c.get("expect_hit")}
    missing = [r for r in RULE_IDS if r not in covered]
    assert not missing, f"无用例的规则: {missing}"
    assert len(covered) == 17


def test_rule_order_differs_from_numbering():
    """RULE_ORDER 与编号故意不同（docs 风险 4）；被顺手对齐会让 reason 漂移。"""
    nums = [int(fn.__name__.split("_r")[1][:2]) for fn in RULE_ORDER]
    assert nums != sorted(nums), "RULE_ORDER 被对齐编号了，这会让 reason 漂移"


def test_first_hit_wins_reason_unique():
    """同一输入在执行序上 reason 唯一（首个命中即返回）。"""
    case = next(c for c in CASES if c["case_id"] == "R12-P1")
    hit = _final_hit(case)
    assert hit is not None and hit.rule_id == "R-12"   # 急诊第一，不被其他规则覆盖
