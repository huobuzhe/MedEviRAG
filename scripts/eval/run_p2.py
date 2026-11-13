# P2 一键：跑 Agent 轨迹评测 + 六组消融，落盘任务集、逐条轨迹与对比表。
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from medevirag.agent.reactor import ReActAgent
from medevirag.eval.agent_eval import CONFIGS, build_tasks, run_ablation

EVAL_DIR = ROOT / "data/eval"
REPORT_DIR = ROOT / "data/reports"


def dump_tasks_and_traces() -> tuple[int, int]:
    tasks = build_tasks()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "agent_tasks.jsonl").write_text(
        "".join(json.dumps(asdict(t), ensure_ascii=False) + "\n" for t in tasks),
        encoding="utf-8")
    agent = ReActAgent()
    lines = []
    for t in tasks:
        tr = agent.run(t)
        lines.append(json.dumps({
            "task_id": tr.task_id, "kind": tr.kind,
            "called_tools": tr.called_tools, "errors": tr.errors,
            "steps": len(tr.steps), "tokens": tr.tokens,
            "stopped_by": tr.stopped_by, "loop_blocked": tr.loop_blocked,
            "budget_exceeded": tr.budget_exceeded,
            "hitl_triggered": tr.hitl_triggered, "recovered": tr.recovered,
            "final_answer": tr.final_answer,
        }, ensure_ascii=False))
    (REPORT_DIR / "agent_traces.jsonl").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    return len(tasks), len(lines)


def main() -> int:
    nt, ntr = dump_tasks_and_traces()
    r = run_ablation()
    full = r["full_config"]
    print("[p2] tasks=" + str(nt) + " traces=" + str(ntr))
    print("[p2-full] task_success_rate=" + str(full["task_success_rate"])
          + " tool_selection_accuracy=" + str(full["tool_selection_accuracy"])
          + " param_accuracy=" + str(full["param_accuracy"]))
    print("[p2-full] avg_steps=" + str(full["avg_steps"])
          + " p95_steps=" + str(full["p95_steps"])
          + " total_tokens=" + str(full["total_tokens"])
          + " cost_yuan=" + str(full["estimated_cost_yuan"]))
    print("[p2-full] loop_detection_rate=" + str(full["loop_detection_rate"])
          + " error_recovery_rate=" + str(full["error_recovery_rate"])
          + " hitl_trigger_accuracy=" + str(full["hitl_trigger_accuracy"]))
    for name, m in r["metrics_by_config"].items():
        if name == "full":
            continue
        print("[p2-abl] " + name + ": success " + str(full["task_success_rate"])
              + " -> " + str(m["task_success_rate"])
              + " | tokens " + str(full["total_tokens"]) + " -> " + str(m["total_tokens"]))
    print("[p2] configs=" + str(len(CONFIGS))
          + " | report -> data/reports/agent_eval.json, agent_ablation.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
