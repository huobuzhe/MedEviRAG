# P3 一键：两个业务接同一套协议 -> 正常跑一次做基线 -> 劣化跑一次做对比 -> 门禁 -> 看板 -> bad case 回流。
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from medevirag.eval.protocol import (REPORT_DIR, TargetRegistry, compare_versions,
                                    evaluate_target, reflow_bad_cases, render_dashboard)
from medevirag.eval.targets import AIChefTarget, MedEviRAGTarget

UNIFIED_CASES = ROOT / "data/eval/unified_cases.jsonl"


def build_cases() -> list[dict]:
    cases = []
    for i in range(12):
        cases.append({"case_id": "rag_" + str(i).zfill(2),
                      "target": "medevirag_rag",
                      "question": "高血压诊断标准第" + str(i + 1) + "问：诊室血压阈值是多少？",
                      "must_refuse": False, "top_k": 10})
    kinds = ("single_tool", "multi_tool", "param_error", "tool_failure",
             "loop_risk", "hitl_required")
    for i in range(12):
        cases.append({"case_id": "chef_" + str(i).zfill(2),
                      "target": "ai_chef_agent",
                      "question": "高血压患者第" + str(i + 1) + "问：安排低钠食谱",
                      "kind": kinds[i % len(kinds)]})
    UNIFIED_CASES.parent.mkdir(parents=True, exist_ok=True)
    UNIFIED_CASES.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n"
                                     for c in cases), encoding="utf-8")
    return cases


def run_all(degraded: bool, tag: str) -> list[dict]:
    reg = TargetRegistry()
    reg.register(MedEviRAGTarget(degraded=degraded))
    reg.register(AIChefTarget(degraded=degraded))
    cases = build_cases()
    out = []
    for name in reg.names():
        subset = [c for c in cases if c["target"] == name]
        out.append(evaluate_target(reg.get(name), subset, tag=tag))
    return out


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = run_all(degraded=False, tag="baseline")
    cur = run_all(degraded=True, tag="current")
    (REPORT_DIR / "unified_baseline.json").write_text(
        json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "unified_current.json").write_text(
        json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    comparisons = [compare_versions(b, c) for b, c in zip(base, cur)]
    reflow = reflow_bad_cases(cur, "p3_unified_current")
    dash = render_dashboard(cur, comparisons, reflow)
    (REPORT_DIR / "unified_dashboard.md").write_text(dash, encoding="utf-8")
    summary = {
        "baseline": [{k: b[k] for k in ("target", "case_count", "success_rate",
                                        "error_rate", "total_tokens")} for b in base],
        "current": [{k: c[k] for k in ("target", "case_count", "success_rate",
                                       "error_rate", "total_tokens")} for c in cur],
        "comparisons": comparisons,
        "reflow": reflow,
        "targets_registered": 2,
        "protocol_fields": ["case_id", "target", "version", "input", "output",
                            "success", "metrics", "trace", "cost_tokens",
                            "latency_ms", "status", "error"],
    }
    (REPORT_DIR / "unified_p3_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for b, c, cmp_ in zip(base, cur, comparisons):
        print("[p3] " + b["target"] + ": success " + str(b["success_rate"])
              + " -> " + str(c["success_rate"])
              + " | error " + str(b["error_rate"]) + " -> " + str(c["error_rate"])
              + " | tokens " + str(b["total_tokens"]) + " -> " + str(c["total_tokens"])
              + " | gate=" + cmp_["gate"])
    print("[p3] reflow candidates=" + str(reflow["n"])
          + " quarantine=" + str(reflow["quarantine"]))
    print("[p3] dashboard -> data/reports/unified_dashboard.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
