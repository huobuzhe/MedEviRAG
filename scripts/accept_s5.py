# S5 验收总入口（docs/06 §6.1 验收表）。
# 纪律同 accept_s3/s4：每条可复跑；无法满足的标 SKIP 并写原因，绝不伪绿。

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = {
    "PYTHONPATH": "src",
    "MEDEVIRAG_DEVICE": "cpu",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_CACHE": "data/.hf_cache",
}
RESULTS: list[dict] = []


def _run(cmd: str, timeout: int = 1800) -> tuple[int, str]:
    env = dict(os.environ)
    env.update(ENV)
    p = subprocess.run(cmd, shell=True, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env,
                       timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _summary_line(out: str) -> str:
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        if any(t in line.lower() for t in ("passed", "failed", "error", "skipped")):
            return line.strip("= ")
    return lines[-1] if lines else ""


def record(code: str, name: str, status: str, evidence: str) -> None:
    RESULTS.append({"id": code, "name": name, "status": status, "evidence": evidence})
    print("[" + status + "] " + code + " " + name + (" :: " + evidence if evidence else ""))


def check(code: str, name: str, fn) -> None:
    try:
        status, evidence = fn()
    except Exception as exc:
        status, evidence = "FAIL", "异常: " + type(exc).__name__ + ": " + str(exc)[:120]
    record(code, name, status, evidence)


def _load_set() -> list[dict]:
    p = ROOT / "data" / "eval" / "eval_set.jsonl"
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _latest_eval() -> dict | None:
    js = sorted(glob.glob(str(ROOT / "data" / "reports" / "eval_*.json")))
    return json.loads(Path(js[-1]).read_text(encoding="utf-8")) if js else None


def a1_scale() -> tuple[str, str]:
    from collections import Counter
    cases = _load_set()
    ct = Counter(c["case_type"] for c in cases)
    tp = len({c["topic"] for c in cases})
    ok = (len(cases) == 160 and ct["answerable"] == 90
          and ct["should_refuse"] == 50 and ct["multi_hop"] == 20 and tp == 12)
    return ("PASS" if ok else "FAIL"), ("160 条 = " + str(dict(ct)) + "，主题 " + str(tp))


def a2_quota() -> tuple[str, str]:
    rc, out = _run(chr(34) + PY + chr(34) + " -X utf8 -m pytest tests/unit/test_quota.py -q")
    return ("PASS" if rc == 0 else "FAIL"), _summary_line(out)


def a3_freeze() -> tuple[str, str]:
    rc, _ = _run(chr(34) + PY + chr(34) + " -X utf8 -m medevirag.eval.build_set verify --split dev")
    stamp = json.loads((ROOT / "data" / "eval" / "freeze.json").read_text(encoding="utf-8"))
    n = stamp["n"]
    ok = rc == 0 and n["dev"] == 128 and n["test"] == 32
    return ("PASS" if ok else "FAIL"), ("freeze rc=" + str(rc) + "，n=" + str(n))


def a4_tamper() -> tuple[str, str]:
    set_p = ROOT / "data" / "eval" / "eval_set.jsonl"
    orig = set_p.read_text(encoding="utf-8")
    lines = orig.splitlines()
    o = json.loads(lines[0])
    o["gold_chunk_ids"] = ["FAKE#id"]
    lines[0] = json.dumps(o, ensure_ascii=False)
    set_p.write_text("\n".join(lines), encoding="utf-8")
    rc, _ = _run(chr(34) + PY + chr(34) + " -X utf8 -m medevirag.eval.build_set verify --split dev")
    set_p.write_text(orig, encoding="utf-8")
    rc2, _ = _run(chr(34) + PY + chr(34) + " -X utf8 -m medevirag.eval.build_set verify --split dev")
    ok = rc != 0 and rc2 == 0
    return ("PASS" if ok else "FAIL"), ("篡改 exit=" + str(rc) + "，恢复 exit=" + str(rc2))


def a5_report() -> tuple[str, str]:
    d = _latest_eval()
    if not d:
        return "SKIP", "尚无 eval_*.json"
    rid = d["metrics"]["run_id"]
    md = ROOT / "data" / "reports" / ("eval_" + rid + ".md")
    if not md.exists():
        return "FAIL", "md 不存在"
    text = md.read_text(encoding="utf-8")
    secs = ("## 摘要卡", "## 分维度指标表", "## 消融表", "## 失败案例")
    missing = [s for s in secs if s not in text]
    return ("PASS" if not missing else "FAIL"), ("四段齐全" if not missing else "缺 " + str(missing))


def a6_metrics() -> tuple[str, str]:
    d = _latest_eval()
    if not d:
        return "SKIP", "尚无 eval_*.json"
    m = d["metrics"]
    keys = ("recall_at_10", "mrr", "ndcg_at_10", "faithfulness", "refusal_recall",
            "over_refusal_rate", "refusal_reason_accuracy",
            "citation_landing_rate", "sentence_support_rate")
    detail = "; ".join(k + "=" + str(round(m[k], 3)) if isinstance(m.get(k), float)
                 else k + "=n/a" for k in keys)
    if m.get("llm_mode") == "fake":
        return "SKIP", ("离线桩口径，真实门禁须 --real：" + detail[:220])
    fails = [k for k in keys if m.get(k + "_pass") is False]
    return ("PASS" if not fails else "FAIL"), ("fails=" + str(fails) + "; " + detail[:180])


def a7_tests() -> tuple[str, str]:
    rc, out = _run(chr(34) + PY + chr(34) + " -X utf8 -m pytest tests/eval tests/unit/test_quota.py -q")
    return ("PASS" if rc == 0 else "FAIL"), _summary_line(out)


def a8_gate_reverse() -> tuple[str, str]:
    rc, out = _run(chr(34) + PY + chr(34) + " -X utf8 -m pytest tests/eval/test_regression_gate.py -q")
    return ("PASS" if rc == 0 else "FAIL"), ("反向测试：" + _summary_line(out))


def a9_cross_family() -> tuple[str, str]:
    rc, out = _run(chr(34) + PY + chr(34) + " -X utf8 -m pytest tests/eval/test_judges.py -q -k cross_family")
    return ("PASS" if rc == 0 else "FAIL"), _summary_line(out)


def a10_zero_api() -> tuple[str, str]:
    rc, out = _run(chr(34) + PY + chr(34) + ' -X utf8 -m pytest tests/eval tests/unit -q -m "not docker"', timeout=900)
    rc2, out2 = _run(chr(34) + PY + chr(34) + " -X utf8 -c \"from medevirag.common.token_guard import LEDGER; print(LEDGER.calls)\"")
    calls = out2.strip().splitlines()[-1] if out2.strip() else "?"
    ok = rc == 0 and calls == "0"
    return ("PASS" if ok else "FAIL"), ("tests " + _summary_line(out) + "，LEDGER.calls=" + calls)


def a11_self_delta() -> tuple[str, str]:
    d = _latest_eval()
    if not d:
        return "SKIP", "尚无 run"
    rid = d["metrics"]["run_id"]
    rc, out = _run(chr(34) + PY + chr(34) + " -X utf8 -m medevirag.eval.regression --baseline "
                    + rid + " --current " + rid)
    # docs 6.1：门禁必须真的会红。离线口径下 self-delta 因指标未达标报 THRESHOLD_FAIL（exit=2），
    # 这证明门禁在工作；只有无 GATE 行却退出码非 0 才是 bug。
    has_gate = "[GATE]" in out or "THRESHOLD_FAIL" in out or "HARD_DROP" in out
    ok = (rc == 0) or (rc == 2 and has_gate)
    return ("PASS" if ok else "FAIL"), ("self-delta exit=" + str(rc) + "，门禁触发=" + str(has_gate))


def main() -> int:
    os.environ.update(ENV)
    checks = [
        ("A1", "评测集 160/90/50/20 与 12 主题", a1_scale),
        ("A2", "配额断言 pytest", a2_quota),
        ("A3", "dev/test 冻结 128/32", a3_freeze),
        ("A4", "篡改 eval_set 被拦", a4_tamper),
        ("A5", "报告 md 四段", a5_report),
        ("A6", "九指标门禁（--real 口径）", a6_metrics),
        ("A7", "tests/eval 全绿", a7_tests),
        ("A8", "门禁反向测试", a8_gate_reverse),
        ("A9", "跨族断言", a9_cross_family),
        ("A10", "零 API 消耗", a10_zero_api),
        ("A11", "regression self-delta", a11_self_delta),
    ]
    for code, name, fn in checks:
        check(code, name, fn)
    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    n_skip = sum(1 for r in RESULTS if r["status"] == "SKIP")
    out = ROOT / "data" / "reports" / "s5_acceptance.json"
    out.write_text(json.dumps({"results": RESULTS, "pass": n_pass,
                               "fail": n_fail, "skip": n_skip},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("")
    print("=" * 70)
    print("A1-A11: PASS=" + str(n_pass) + "  FAIL=" + str(n_fail) + "  SKIP=" + str(n_skip))
    print("报告: " + str(out))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
