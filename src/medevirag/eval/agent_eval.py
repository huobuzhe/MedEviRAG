# P2：Agent 轨迹评测。RAG 只看答案，Agent 必须看整条轨迹（工具/参数/步数/成本/恢复/审批/死循环）。
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medevirag.agent.reactor import AgentTrace, ReActAgent

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "data/reports"
PRICE_PER_1K_TOKENS = 0.001

KINDS = ("single_tool", "multi_tool", "param_error", "tool_failure",
         "loop_risk", "hitl_required", "budget_risk", "out_of_scope")
EXPECTED: dict[str, tuple[str, ...]] = {
    "single_tool": ("retrieve_evidence",),
    "multi_tool": ("search_local_recipes", "calculate_nutrition", "verify_citation"),
    "param_error": ("calculate_nutrition",),
    "tool_failure": ("drug_food_interaction",),
    "loop_risk": ("retrieve_evidence",),
    "hitl_required": ("drug_food_interaction",),
    "budget_risk": ("retrieve_evidence",),
    "out_of_scope": (),
}
DANGEROUS_TOOL = "drug_food_interaction"


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    kind: str
    question: str
    expected_tools: tuple[str, ...]
    requires_hitl: bool = False
    expect_recovery: bool = False
    expect_loop_block: bool = False
    expect_budget_stop: bool = False


def build_tasks(per_kind: int = 5) -> list[AgentTask]:
    out: list[AgentTask] = []
    for kind in KINDS:
        for i in range(per_kind):
            out.append(AgentTask(
                task_id=kind + "_" + str(i).zfill(2),
                kind=kind,
                question="高血压患者第" + str(i + 1) + "问：怎么安排低钠饮食与用药？",
                expected_tools=EXPECTED[kind],
                requires_hitl=(kind in ("hitl_required", "tool_failure")),
                expect_recovery=(kind in ("param_error", "tool_failure")),
                expect_loop_block=(kind == "loop_risk"),
                expect_budget_stop=(kind == "budget_risk"),
            ))
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _handled(task: AgentTask, tr: AgentTrace) -> bool:
    got = set(tr.called_tools)
    want = set(task.expected_tools)
    if task.kind == "out_of_scope":
        return not tr.called_tools and "超范围" in tr.final_answer
    if task.expect_recovery:
        return bool(tr.recovered) and bool(got & want)
    if task.expect_loop_block:
        return bool(tr.loop_blocked)
    if task.expect_budget_stop:
        return bool(tr.budget_exceeded)
    if task.requires_hitl:
        return bool(tr.hitl_triggered) and bool(got & want)
    if not got & want:
        return False
    if tr.stopped_by not in ("finished", "budget"):
        return False
    return not any(e.startswith("PARAM_INVALID") for e in tr.errors)


def compute_metrics(tasks: list[AgentTask], traces: list[AgentTrace]) -> dict[str, Any]:
    n = len(traces)
    handled = [_handled(t, r) for t, r in zip(tasks, traces)]
    sel = [_jaccard(set(r.called_tools), set(t.expected_tools))
           for t, r in zip(tasks, traces)]
    calls = sum(len(r.called_tools) for r in traces)
    bad_params = sum(1 for r in traces for e in r.errors if e.startswith("PARAM_INVALID"))
    steps = sorted(len(r.steps) for r in traces)
    loop_tasks = [(t, r) for t, r in zip(tasks, traces) if t.expect_loop_block]
    rec_tasks = [(t, r) for t, r in zip(tasks, traces) if t.expect_recovery]
    hitl_ok = sum(1 for t, r in zip(tasks, traces)
                  if (r.hitl_triggered == t.requires_hitl))
    total_tokens = sum(r.tokens for r in traces)
    by_kind: dict[str, Any] = {}
    for kind in KINDS:
        pairs = [(t, r) for t, r in zip(tasks, traces) if t.kind == kind]
        by_kind[kind] = {
            "n": len(pairs),
            "handled": round(sum(1 for t, r in pairs if _handled(t, r)) / len(pairs), 4),
            "mean_steps": round(sum(len(r.steps) for _, r in pairs) / len(pairs), 2),
            "mean_tokens": round(sum(r.tokens for _, r in pairs) / len(pairs), 1),
        }
    return {
        "n_tasks": n,
        "task_success_rate": round(sum(handled) / n, 4),
        "tool_selection_accuracy": round(sum(sel) / n, 4),
        "param_accuracy": round(1 - bad_params / max(1, calls), 4),
        "total_tool_calls": calls,
        "avg_steps": round(sum(steps) / n, 2),
        "p95_steps": steps[min(n - 1, int(n * 0.95))],
        "total_tokens": total_tokens,
        "estimated_cost_yuan": round(total_tokens / 1000 * PRICE_PER_1K_TOKENS, 6),
        "loop_detection_rate": round(sum(1 for _, r in loop_tasks if r.loop_blocked)
                                     / max(1, len(loop_tasks)), 4),
        "error_recovery_rate": round(sum(1 for _, r in rec_tasks if r.recovered)
                                     / max(1, len(rec_tasks)), 4),
        "hitl_trigger_accuracy": round(hitl_ok / n, 4),
        "by_kind": by_kind,
        "criteria": "风险类任务的成功 = 对应机制正确处置（拦住/恢复/审批）；普通任务 = 工具命中且无未处理参数错误",
    }


CONFIGS: dict[str, dict[str, bool]] = {
    "full": {},
    "no_loop_detection": {"loop_detection": False},
    "no_budget_control": {"budget_control": False},
    "no_hitl": {"hitl": False},
    "no_error_recovery": {"error_recovery": False},
    "bare": {"loop_detection": False, "budget_control": False,
             "hitl": False, "error_recovery": False},
}


def run_one_config(tasks: list[AgentTask], cfg: dict[str, bool]) -> dict[str, Any]:
    agent = ReActAgent(**cfg)
    traces = [agent.run(t) for t in tasks]
    return compute_metrics(tasks, traces)


def run_ablation(per_kind: int = 5) -> dict[str, Any]:
    tasks = build_tasks(per_kind)
    results = {name: run_one_config(tasks, cfg) for name, cfg in CONFIGS.items()}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tasks": len(tasks),
        "configs": {k: v for k, v in CONFIGS.items()},
        "metrics_by_config": results,
        "full_config": results["full"],
        "note": "确定性脚本规划器（ScriptedPlanner）产出动作序列，用来量化评测与防护机制，"
                "不是模型能力评估；接真实 LLM 时替换 plan() 即可复用全部指标与门禁。",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "agent_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = ("task_success_rate", "tool_selection_accuracy", "param_accuracy",
            "avg_steps", "total_tokens", "loop_detection_rate",
            "error_recovery_rate", "hitl_trigger_accuracy")
    lines = ["# P2 Agent 轨迹评测 · 消融表", "",
             "| 配置 | " + " | ".join(keys) + " |",
             "|---|" + "---|" * len(keys)]
    for name, m in results.items():
        lines.append("| " + name + " | " + " | ".join(str(m[k]) for k in keys) + " |")
    lines += ["", "> 口径：" + results["full"]["criteria"],
              "> 成本按 " + str(PRICE_PER_1K_TOKENS) + " 元/1k tokens 估算，只用于相对比较。"]
    (REPORT_DIR / "agent_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
