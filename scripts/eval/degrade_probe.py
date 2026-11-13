# P0 劣化实验：把检索 top_k 从 10 改成 1，真跑两次，证明门禁会拦截。
from __future__ import annotations

import json
import sys
from pathlib import Path

from run_core import REPORT_DIR, run_pipeline
from medevirag.eval.gate import evaluate

EVIDENCE_PATH = REPORT_DIR / "p0_degrade_probe.json"
KEYS = ("recall_at_10", "hit_at_10", "mrr", "ndcg_at_10")


def main() -> int:
    base = run_pipeline("core_80", 10, "base")
    deg = run_pipeline("core_80", 1, "deg")
    g = evaluate(deg["path"], base["path"])
    before = {k: base["metrics"].get(k) for k in KEYS}
    after = {k: deg["metrics"].get(k) for k in KEYS}
    blocked = g["status"] == "BLOCKED"
    evidence = {
        "purpose": "证明改动检索参数导致质量下降时，CI 门禁会真的拦住，而不是只报个数字",
        "layer": "core_80",
        "baseline_run_id": base["run_id"],
        "degraded_run_id": deg["run_id"],
        "baseline_top_k": 10,
        "degraded_top_k": 1,
        "baseline_metrics": before,
        "degraded_metrics": after,
        "gate_status": g["status"],
        "gate_problems": g["problems"],
        "publishable": g["publishable"],
        "note": "两次均为 pipeline_selfcheck（offline judge 桩）；本实验只验证门禁链路，"
                "检索指标来自确定性被测实现，不可当作系统质量结论",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    for k in KEYS:
        print("[probe] " + k + ": " + str(before[k]) + " -> " + str(after[k]))
    for p in g["problems"]:
        print("[GATE]", p)
    print("[probe] gate=" + g["status"]
          + " | evidence -> " + str(EVIDENCE_PATH))
    return 0 if blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
