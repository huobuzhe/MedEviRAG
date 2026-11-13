"""S4 验收总入口（docs/05 §6，A1-A13）。

与 accept_s3.py 同构：每条门禁落成可复跑判定；无法满足的标 SKIP 并写明原因。
矩阵指标来自 54 条夹具（fixture_level=true），非评测集指标——S5 建集后须重新分层抽样。
"""

from __future__ import annotations

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
}
RESULTS: list[dict] = []


def _run(cmd: str, timeout: int = 1800) -> tuple[int, str]:
    env = dict(os.environ)
    env.update(ENV)
    p = subprocess.run(cmd, shell=True, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env,
                       timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


_SUMMARY_TOKENS = ("passed", "failed", "error", "skipped", "deselected")


def _summary_line(out: str) -> str:
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        low = line.lower()
        if any(t in low for t in _SUMMARY_TOKENS):
            return line.strip("= ")
    return lines[-1] if lines else ""


def record(code: str, name: str, status: str, evidence: str) -> None:
    RESULTS.append({"id": code, "name": name, "status": status, "evidence": evidence})
    print(f"[{status}] {code} {name}" + (f" :: {evidence}" if evidence else ""))


def check(code: str, name: str, fn) -> None:
    try:
        status, evidence = fn()
    except Exception as exc:
        status, evidence = "FAIL", f"异常: {type(exc).__name__}: {exc}"
    record(code, name, status, evidence)


def _matrix() -> dict | None:
    p = ROOT / "data" / "reports" / "refusal_matrix.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _fresh_matrix() -> dict:
    rc, out = _run(f'"{PY}" -X utf8 -m medevirag.cli refusal-matrix', timeout=600)
    if rc != 0:
        raise RuntimeError(f"refusal-matrix exit={rc}: {out[-300:]}")
    return _matrix() or {}


def a1() -> tuple[str, str]:
    m = _fresh_matrix()
    v = m["refusal_recall"]
    return ("PASS" if v >= 0.85 else "FAIL"), f"拒答召回 {v:.3f} (>=0.85)，夹具级指标"


def a2() -> tuple[str, str]:
    m = _fresh_matrix()
    v = m["over_refusal_rate"]
    return ("PASS" if v <= 0.12 else "FAIL"), f"过度拒答率 {v:.3f} (<=0.12)"


def a3() -> tuple[str, str]:
    m = _fresh_matrix()
    v = m["reason_accuracy"]
    return ("PASS" if v >= 0.80 else "FAIL"), f"理由正确率 {v:.3f} (>=0.80)"


def a4() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/unit/test_refusal_rules.py::test_rule_coverage -q')
    return ("PASS" if rc == 0 else "FAIL"), f"规则覆盖 17/17: {_summary_line(out)}"


def a5() -> tuple[str, str]:
    n = sum(1 for _ in (ROOT / "tests" / "fixtures" / "refusal_cases.jsonl").open(encoding="utf-8") if _.strip())
    return ("PASS" if n >= 54 else "FAIL"), f"用例总数 {n} (>=54)"


def a6() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/unit/test_confidence.py -q -k monotonic')
    return ("PASS" if rc == 0 else "FAIL"), f"单调性: {_summary_line(out)}"


def a7() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/unit/test_confidence.py -q')
    return ("PASS" if rc == 0 else "FAIL"), f"四分量可分解: {_summary_line(out)}"


def a8() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/unit/test_refusal_templates.py -q')
    return ("PASS" if rc == 0 else "FAIL"), f"17 模板含免责声明: {_summary_line(out)}"


def a9() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/unit/test_refusal_rules.py tests/unit/test_decide_node.py -q -k unknown')
    return ("PASS" if rc == 0 else "FAIL"), f"unknown 双分支: {_summary_line(out)}"


def a10() -> tuple[str, str]:
    import time

    t0 = time.time()
    rc, out = _run(f'"{PY}" -X utf8 -m pytest -q -m "not docker"', timeout=600)
    dt = time.time() - t0
    ok = rc == 0 and dt <= 60
    return ("PASS" if ok else "FAIL"), f"无 Docker 全绿 exit={rc}，耗时 {dt:.0f}s (<=60s)"


def a11() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/unit/test_decide_node.py -q -k precedence')
    return ("PASS" if rc == 0 else "FAIL"), f"R-01 优先于 R-13: {_summary_line(out)}"


def a12() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/unit/test_decide_node.py -q')
    return ("PASS" if rc == 0 else "FAIL"), f"拒答不带引用: {_summary_line(out)}"


def a13() -> tuple[str, str]:
    """零 API 消耗：先跑全量单测再查 LEDGER。"""
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/unit tests/integration -q', timeout=900)
    rc2, out2 = _run('"' + PY + '" -X utf8 -c "from medevirag.common.token_guard import LEDGER; print(LEDGER.calls)"')
    calls = out2.strip().splitlines()[-1] if out2.strip() else "?"
    ok = rc == 0 and calls == "0"
    return ("PASS" if ok else "FAIL"), f"测试全绿={rc == 0}，LEDGER.calls={calls} (须=0)"


def main() -> int:
    os.environ.update(ENV)
    checks = [
        ("A1", "拒答召回 >= 0.85", a1),
        ("A2", "过度拒答率 <= 0.12", a2),
        ("A3", "拒答理由正确率 >= 0.80", a3),
        ("A4", "规则覆盖 17/17", a4),
        ("A5", "用例总数 >= 54", a5),
        ("A6", "置信度单调性", a6),
        ("A7", "复合分可分解", a7),
        ("A8", "模板免责声明 17/17", a8),
        ("A9", "unknown 降级双分支", a9),
        ("A10", "无 Docker CI 全绿 <=60s", a10),
        ("A11", "判定顺序 R-01 > R-13", a11),
        ("A12", "拒答不带引用", a12),
        ("A13", "零 API 消耗", a13),
    ]
    for code, name, fn in checks:
        check(code, name, fn)

    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    n_skip = sum(1 for r in RESULTS if r["status"] == "SKIP")
    out = ROOT / "data" / "reports" / "s4_acceptance.json"
    out.write_text(json.dumps({"results": RESULTS, "pass": n_pass,
                                "fail": n_fail, "skip": n_skip},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"A1-A13: PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}")
    print(f"报告: {out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
