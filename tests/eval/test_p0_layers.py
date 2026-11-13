# P0 单元测试：评测集分层、schema 自检与冻结指纹。
import copy
import json

import pytest

layers = pytest.importorskip("medevirag.eval.layers",
                              reason="请先按操作手册落盘 src/medevirag/eval/layers.py")


def _raw():
    return layers.read_raw()


def test_real_set_passes_schema_but_reports_split_warning():
    problems, warnings = layers.validate_cases(_raw())
    assert problems == []
    assert any("SPLIT_DISTRIBUTION_MISMATCH" in w for w in warnings)


def test_validate_catches_three_classic_schema_errors():
    rows = copy.deepcopy(_raw()[:3])
    rows[0]["must_refuse"] = "False"
    rows[1]["case_type"] = "should_refuse"
    del rows[2]["case_id"]
    problems, _ = layers.validate_cases(rows)
    assert any("bool" in p for p in problems)
    assert any("不一致" in p for p in problems)
    assert any("缺字段" in p for p in problems)


def test_pick_is_deterministic_and_respects_quota():
    cases = layers.load_cases(layers.SET_PATH)
    a = [c.case_id for c in layers._pick(cases, 80, "same-salt")]
    b = [c.case_id for c in layers._pick(cases, 80, "same-salt")]
    assert a == b
    assert len(a) == 80 and len(set(a)) == 80


def test_build_layers_nesting_and_tamper_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(layers, "LAYER_DIR", tmp_path)
    monkeypatch.setattr(layers, "META_PATH", tmp_path / "layers_meta.json")
    meta = layers.build_layers()
    assert meta["layers"]["smoke_20"]["n"] == 20
    assert meta["layers"]["core_80"]["n"] == 80
    assert meta["layers"]["full_160"]["n"] == 160

    def ids(name):
        return {json.loads(ln)["case_id"]
                for ln in (tmp_path / (name + ".jsonl")).read_text(
                    encoding="utf-8").splitlines() if ln.strip()}

    assert ids("smoke_20") <= ids("core_80") <= ids("full_160")
    assert layers.verify_layers()["ok"]

    p = tmp_path / "core_80.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    assert not layers.verify_layers()["ok"]
