# P3 单元测试：协议统一、注册表、版本门禁、回流隔离。
from medevirag.eval.protocol import (EvalRecord, TargetRegistry, compare_versions,
                                    evaluate_target, reflow_bad_cases)
from medevirag.eval.targets import AIChefTarget, MedEviRAGTarget


def _cases():
    return [
        {"case_id": "r1", "target": "medevirag_rag", "question": "诊室血压阈值是多少？",
         "must_refuse": False, "top_k": 10},
        {"case_id": "c1", "target": "ai_chef_agent", "question": "安排低钠食谱",
         "kind": "single_tool"},
    ]


def test_registry_and_duplicate_guard():
    reg = TargetRegistry()
    reg.register(MedEviRAGTarget())
    reg.register(AIChefTarget())
    assert reg.names() == ["ai_chef_agent", "medevirag_rag"]
    try:
        reg.register(MedEviRAGTarget())
    except ValueError as e:
        assert "重复注册" in str(e)
    else:
        raise AssertionError("重复注册应被拒绝")


def test_both_targets_emit_the_same_protocol_shape():
    reg = TargetRegistry()
    reg.register(MedEviRAGTarget())
    reg.register(AIChefTarget())
    for c in _cases():
        r = reg.get(c["target"]).run(c)
        assert isinstance(r, EvalRecord)
        assert set(r.__dataclass_fields__) == set(EvalRecord.__dataclass_fields__)
        assert r.target == c["target"] and r.case_id == c["case_id"]


def test_evaluate_target_and_version_gate():
    cases = [c for c in _cases() if c["target"] == "medevirag_rag"] * 3
    base = evaluate_target(MedEviRAGTarget(), cases)
    cur = evaluate_target(MedEviRAGTarget(degraded=True), cases)
    assert base["success_rate"] == 1.0 and cur["success_rate"] == 0.0
    cmp_ = compare_versions(base, cur)
    assert cmp_["gate"] == "BLOCKED"
    assert any("SUCCESS_DROP" in f for f in cmp_["flags"])


def test_reflow_is_quarantined():
    cases = [c for c in _cases() if c["target"] == "medevirag_rag"] * 2
    rep = evaluate_target(MedEviRAGTarget(degraded=True), cases)
    ref = reflow_bad_cases([rep], "unit_test")
    assert ref["n"] == 2 and ref["quarantine"] is True
    import json
    from pathlib import Path
    rows = [json.loads(l) for l in Path(ref["path"]).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert all(r["quarantine"] is True for r in rows)
    assert all("人工审核" in r["note"] for r in rows)
