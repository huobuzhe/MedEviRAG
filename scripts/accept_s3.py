"""S3 验收总入口（docs/04 §6，A1-A14）。

设计原则：每条门禁都必须落成"可复跑的判定"，而不是人工看数。
无法在本机满足条件的门禁（需真实 API / 需断网）标 SKIP 并写明原因，绝不伪绿。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
# MEDEVIRAG_DEVICE=cpu：子进程强制走 CPU。
# 原因（实测）：每个 pytest 子进程都会在 GPU 上建一份 bge-m3 + reranker 的 CUDA 上下文，
# 多个子进程叠加时 Windows 上会 access violation —— 表现为「汇总行写着 passed 但退出码非 0」，
# 而 A8/A10 会因此被误判成 FAIL。A1/A2/A12 的真机延迟另有单独门禁（真跑 GPU 的扫描脚本产出），
# 所以这里降级到 CPU 不会放松任何一项指标的判定口径。
# 线程约束同理：每个子进程会拉起 torch / sentence-transformers / Milvus Lite 多套线程池，
# 线程数严重超订时 torch 的 embedding 前向会在 Windows 上 access violation（原生崩溃，
# 无 Python traceback，退出码非 0 但汇总行已打印 passed）。限死线程数后复跑稳定。
ENV = {
    "PYTHONPATH": "src",
    "MEDEVIRAG_DEVICE": "cpu",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

RESULTS: list[dict] = []


def _run(cmd: str, timeout: int = 1800) -> tuple[int, str]:
    import os

    env = dict(os.environ)
    env.update(ENV)
    p = subprocess.run(cmd, shell=True, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env,
                       timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


_SUMMARY_TOKENS = ("passed", "failed", "error", "skipped", "no tests ran", "deselected")


def _summary_line(out: str) -> str:
    """从 pytest 输出里挑出真正的汇总行，避免抓到 warning / 日志行。"""
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        low = line.lower()
        if any(t in low for t in _SUMMARY_TOKENS) and ("=" not in line or "warnings" in low):
            return line.strip("= ")
    return lines[-1] if lines else ""


def record(code: str, name: str, status: str, evidence: str) -> None:
    RESULTS.append({"id": code, "name": name, "status": status, "evidence": evidence})
    mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
    print(f"[{mark}] {code} {name}" + (f" :: {evidence}" if evidence else ""))


def check(code: str, name: str, fn) -> None:
    try:
        status, evidence = fn()
    except Exception as exc:
        status, evidence = "FAIL", f"异常: {type(exc).__name__}: {exc}"
    record(code, name, status, evidence)


# --------------------------------------------------------------------------- #
# A1：重排器确定性
# --------------------------------------------------------------------------- #
def a1() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/unit/test_reranker.py -q')
    ok = rc == 0
    return ("PASS" if ok else "FAIL"), _summary_line(out)


# --------------------------------------------------------------------------- #
# A2/A3/A4：重排延迟与净收益（读 s3_rerank_sweep.json）
# --------------------------------------------------------------------------- #
def _sweep() -> dict | None:
    p = ROOT / "data" / "reports" / "s3_rerank_sweep.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def a2() -> tuple[str, str]:
    d = _sweep()
    if not d:
        return "FAIL", "缺少 data/reports/s3_rerank_sweep.json"
    k20 = d["by_k"]["20"]["latency_p95_ms"]
    k50 = d["by_k"]["50"]["latency_p95_ms"]
    ok = k20 <= 300 and k50 <= 800
    return ("PASS" if ok else "FAIL"), f"K=20 P95={k20}ms (≤300), K=50 P95={k50}ms (≤800)"


def a3() -> tuple[str, str]:
    d = _sweep()
    if not d:
        return "FAIL", "缺少 sweep 报告"
    v = d["verdict"]
    # 门禁看"是否被采用"：ΔMRR≥0.10 → 采用；否则必须走 R2 否决分支并留档
    denied = not v["rerank_approved"]
    if denied:
        enabled = _rerank_enabled()
        ok = enabled is False
        return ("PASS" if ok else "FAIL"), (
            f"ΔMRR={v['delta_mrr']}<0.10 → R2 否决重排；retrieval.yaml rerank.enabled={enabled} "
            f"(应为 False)；直通 Top8 已生效" + ("" if ok else " ← 配置未同步"))
    ok = v["delta_mrr"] >= 0.10
    return ("PASS" if ok else "FAIL"), f"ΔMRR={v['delta_mrr']}≥0.10 → 采用重排"


def _rerank_enabled() -> bool:
    from medevirag.index.paths import load_retrieval_config

    return bool((load_retrieval_config().get("rerank") or {}).get("enabled", False))


def a4() -> tuple[str, str]:
    d = _sweep()
    if not d:
        return "FAIL", "缺少 sweep 报告"
    ks = set(d["by_k"]) == {"10", "20", "50"}
    v = d["verdict"]
    md = (ROOT / "data" / "reports" / "s3_rerank_sweep.md")
    reason_ok = md.exists() and "否决" in md.read_text(encoding="utf-8")
    ok = ks and v.get("chosen_k") in (10, 20, 50) and reason_ok
    return ("PASS" if ok else "FAIL"), (
        f"三档齐全={ks}, 选定K={v.get('chosen_k')}, 报告含结论理由={reason_ok}")


# --------------------------------------------------------------------------- #
# A5/A6/A12：grounding 评测
# --------------------------------------------------------------------------- #
def _grounding() -> dict | None:
    p = ROOT / "data" / "reports" / "s3_grounding.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def a5() -> tuple[str, str]:
    d = _grounding()
    if not d:
        return "FAIL", "缺少 data/reports/s3_grounding.json"
    v = d["landing_acc_marker_level"]
    if not d.get("gate_applicable", False):
        return "SKIP", (f"离线桩 landing={v}（链路已跑通）；真实模型合规率须 --real 才能作门禁，"
                        f"本机无 API key")
    return ("PASS" if v >= 0.90 else "FAIL"), f"marker 级 landing={v}≥0.90"


def a6() -> tuple[str, str]:
    d = _grounding()
    if not d:
        return "FAIL", "缺少 data/reports/s3_grounding.json"
    v = d["support_rate_mean"]
    if not d.get("gate_applicable", False):
        return "SKIP", f"离线桩 support={v}（链路已跑通）；真实句级支撑率须 --real 才能作门禁"
    return ("PASS" if v >= 0.85 else "FAIL"), f"support_rate_mean={v}≥0.85"


def a12() -> tuple[str, str]:
    d = _grounding()
    if not d:
        return "FAIL", "缺少 data/reports/s3_grounding.json"
    v = d["latency_p95_ms"]
    return ("PASS" if v <= 6000 else "FAIL"), f"图内 P95={v}ms≤6000ms"


# --------------------------------------------------------------------------- #
# A7/A8/A10/A11/A13/A14：pytest
# --------------------------------------------------------------------------- #
def _pytest(sel: str, label: str):
    def fn() -> tuple[str, str]:
        rc, out = _run(f'"{PY}" -X utf8 -m pytest {sel} -q')
        return ("PASS" if rc == 0 else "FAIL"), f"{label}: {_summary_line(out)}"
    return fn


def a7() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/integration/test_graph.py -q -k illegal')
    return ("PASS" if rc == 0 else "FAIL"), f"[C9] 注入被拦: {_summary_line(out)}"


def a8() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/integration/test_graph.py -q -k "retry"')
    return ("PASS" if rc == 0 else "FAIL"), f"重试硬上限: {_summary_line(out)}"


def a9() -> tuple[str, str]:
    """用测试里的示例 chunk_id 校验 12 键（真实 chunk_id 不含 ASCII 前缀，见 docs 说明）。"""
    code = (
        "import json;from medevirag.rag import citations as C;"
        "row={'chunk_id':'CN-HTN-2024#3.2.1#c02','doc_id':'CN-HTN-2024','doc_title':'指南',"
        "'section_path':['a'],'chunk_type':'recommendation','locator':{'page':27,'bbox':[1,2,3,4],'block_index':0},"
        "'page_start':27,'text':'x','context_header':'h','evidence_level':'A'};"
        "C._chunk_index=lambda:{row['chunk_id']:row};"
        "d=C.get_citation_detail(row['chunk_id']);"
        "print(json.dumps({'keys':len(d),'same':list(d)==list(C.DETAIL_KEYS),'page_int':isinstance(d['page'],int),'bbox4':isinstance(d['bbox'],list) and len(d['bbox'])==4}))"
    )
    rc, out = _run(f'"{PY}" -X utf8 -c "{code}"')
    try:
        info = json.loads(out.strip().splitlines()[-1])
        ok = rc == 0 and info["keys"] == 12 and info["same"] and info["page_int"] and info["bbox4"]
        return ("PASS" if ok else "FAIL"), json.dumps(info, ensure_ascii=False)
    except Exception:
        return "FAIL", out.strip()[-300:]


def a10() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/integration/test_graph.py -q -k "eight or trace"')
    return ("PASS" if rc == 0 else "FAIL"), f"8 节点 + trace 8 键: {_summary_line(out)}"


def a11() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/integration/test_graph.py -q -k "out_of_scope"')
    return ("PASS" if rc == 0 else "FAIL"), f"越界 0 次 LLM: {_summary_line(out)}"


def a13() -> tuple[str, str]:
    """离线全绿：不使用真实 API、不依赖 Docker。"""
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/unit tests/integration -q', timeout=3600)
    return ("PASS" if rc == 0 else "FAIL"), f"unit+integration: {_summary_line(out)}"


def a14() -> tuple[str, str]:
    rc, out = _run(f'"{PY}" -X utf8 -m pytest tests/unit/test_budget.py -q')
    return ("PASS" if rc == 0 else "FAIL"), f"预算护栏: {_summary_line(out)}"


def main() -> int:
    import os

    os.environ["PYTHONPATH"] = "src"
    checks = [
        ("A1", "重排器确定性", a1),
        ("A2", "重排延迟不超预算", a2),
        ("A3", "重排净收益（不足则 R2 否决并留档）", a3),
        ("A4", "候选档位决策有数据支撑", a4),
        ("A5", "引用落点正确率", a5),
        ("A6", "句级支撑率", a6),
        ("A7", "verifier 拦得住注入的非法 citation", a7),
        ("A8", "重试有硬上限", a8),
        ("A9", "引用回链字段完整（12 键）", a9),
        ("A10", "8 节点图 + 埋点", a10),
        ("A11", "越界零 LLM 调用", a11),
        ("A12", "图内 P95 ≤ 6s", a12),
        ("A13", "无 Docker / 无 API 全绿", a13),
        ("A14", "预算护栏生效", a14),
    ]
    for code, name, fn in checks:
        check(code, name, fn)

    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    n_skip = sum(1 for r in RESULTS if r["status"] == "SKIP")

    out = ROOT / "data" / "reports" / "s3_acceptance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": RESULTS, "pass": n_pass,
                               "fail": n_fail, "skip": n_skip}, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"A1-A14: PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}")
    if n_skip:
        print("SKIP 原因：" + "；".join(f"{r['id']}={r['evidence']}" for r in RESULTS if r["status"] == "SKIP"))
    print(f"报告: {out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
