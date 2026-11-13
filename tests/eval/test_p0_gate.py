# P0 单元测试：门禁的噪声带、防桩标记与 run 作废。
import json

import pytest

gate = pytest.importorskip("medevirag.eval.gate",
                            reason="请先按操作手册落盘 src/medevirag/eval/gate.py")
from medevirag.common.config import get_settings


def _metrics(**kw):
    m = {
        "run_id": "unit", "frozen_set_sha256": "abc",
        "llm_mode": "real", "trust_level": "publishable",
        "judge_error_rate": 0.0,
        "recall_at_10": 0.88, "mrr": 0.70, "ndcg_at_10": 0.75,
        "citation_landing_rate": 0.95, "sentence_support_rate": 0.90,
        "faithfulness": 0.90, "refusal_recall": 0.90,
        "over_refusal_rate": 0.05, "refusal_reason_accuracy": 0.90,
    }
    m.update(kw)
    return m


def _write(tmp_path, name, metrics):
    p = tmp_path / (name + ".json")
    p.write_text(json.dumps({"metrics": metrics, "cases": []}), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _isolate_result(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "RESULT_PATH", tmp_path / "gate_result.json")


def test_baseline_identical_passes(tmp_path):
    cfg = get_settings()
    base = _write(tmp_path, "base", _metrics())
    cur = _write(tmp_path, "cur", _metrics())
    r = gate.evaluate(cur, base, cfg)
    assert r["status"] == "PASS"
    assert r["publishable"] is True


def test_hard_drop_is_blocked(tmp_path):
    cfg = get_settings()
    base = _write(tmp_path, "base", _metrics())
    cur = _write(tmp_path, "cur", _metrics(recall_at_10=0.70))
    r = gate.evaluate(cur, base, cfg)
    assert r["status"] == "BLOCKED"
    assert r["exit_code"] == 2
    assert any("HARD_DROP" in p for p in r["problems"])
    assert any("THRESHOLD_FAIL" in p for p in r["problems"])


def test_noise_band_is_recorded_but_not_blocked(tmp_path):
    cfg = get_settings()
    base = _write(tmp_path, "base", _metrics())
    cur = _write(tmp_path, "cur", _metrics(recall_at_10=0.855))
    r = gate.evaluate(cur, base, cfg)
    assert r["status"] == "PASS"
    assert len(r["noise_band"]) == 1
    assert any("NOISE_BAND" in n for n in r["notes"])


def test_fake_run_is_not_publishable(tmp_path):
    r = gate.evaluate(_write(tmp_path, "cur", _metrics(
        llm_mode="fake", trust_level="pipeline_selfcheck")), None, get_settings())
    assert r["publishable"] is False
    assert any("PUBLISH_BLOCKED" in n for n in r["notes"])
    assert r["status"] == "PASS"


def test_high_judge_error_rate_invalidates_run(tmp_path):
    r = gate.evaluate(_write(tmp_path, "cur", _metrics(judge_error_rate=0.10)),
                      None, get_settings())
    assert r["status"] == "BLOCKED"
    assert any("RUN_INVALID_JUDGE_ERROR" in p for p in r["problems"])
