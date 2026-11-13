"""P2：手写 ReAct 循环。不依赖 LangGraph/LangChain，把循环、工具、预算、死循环、恢复、HITL 全部显式写出来。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from medevirag.agent.tools import ToolRegistry

TOKEN_COST = {"retrieve_evidence": 130, "search_local_recipes": 90,
              "calculate_nutrition": 70, "verify_citation": 80,
              "drug_food_interaction": 110}


@dataclass(frozen=True)
class Action:
    tool: str
    args: dict[str, Any]


@dataclass
class Step:
    idx: int
    thought: str
    action: Action | None
    observation: str
    error: str = ""
    hitl: bool = False
    tokens: int = 0


@dataclass
class AgentTrace:
    task_id: str
    kind: str
    steps: list[Step] = field(default_factory=list)
    called_tools: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    final_answer: str = ""
    stopped_by: str = "finished"
    tokens: int = 0
    loop_blocked: bool = False
    budget_exceeded: bool = False
    hitl_triggered: bool = False
    recovery_attempted: bool = False
    recovered: bool = False


class ScriptedPlanner:
    """确定性动作规划器：按任务类型产出动作序列。接真实 LLM 时只需替换 plan()。"""

    def plan(self, task: Any) -> list[Action]:
        k = task.kind
        if k == "single_tool":
            return [Action("retrieve_evidence", {"query": task.question, "top_k": 10})]
        if k == "multi_tool":
            return [Action("search_local_recipes", {"query": task.question, "k": 5}),
                    Action("calculate_nutrition", {"ingredients": ["番茄", "鸡蛋"]}),
                    Action("verify_citation", {"answer": "低钠菜谱[1]",
                                              "evidence": ["CN-HTN#c01"]})]
        if k == "param_error":
            return [Action("calculate_nutrition", {"ingredients": "番茄"}),
                    Action("calculate_nutrition", {"ingredients": ["番茄", "鸡蛋"]})]
        if k == "tool_failure":
            return [Action("drug_food_interaction", {"drug": "华法林"}),
                    Action("drug_food_interaction", {"drug": "华法林", "food": "菠菜"})]
        if k == "loop_risk":
            a = Action("retrieve_evidence", {"query": task.question, "top_k": 10})
            return [a, a, a, a]
        if k == "hitl_required":
            return [Action("drug_food_interaction", {"drug": "华法林", "food": "菠菜"})]
        if k == "budget_risk":
            return [Action("retrieve_evidence", {"query": task.question + str(i), "top_k": 10})
                    for i in range(6)]
        return []


class ReActAgent:
    def __init__(self, registry: ToolRegistry | None = None,
                 planner: ScriptedPlanner | None = None,
                 max_steps: int = 8, budget_tokens: int = 600,
                 loop_detection: bool = True, budget_control: bool = True,
                 hitl: bool = True, error_recovery: bool = True,
                 approver: Callable[[Any, str], bool] | None = None) -> None:
        self.registry = registry or ToolRegistry()
        self.planner = planner or ScriptedPlanner()
        self.max_steps = max_steps
        self.budget_tokens = budget_tokens
        self.loop_detection = loop_detection
        self.budget_control = budget_control
        self.hitl = hitl
        self.error_recovery = error_recovery
        self.approver = approver or (lambda task, tool: True)

    def run(self, task: Any) -> AgentTrace:
        tr = AgentTrace(task_id=task.task_id, kind=task.kind)
        last_key = ""
        for i, action in enumerate(self.planner.plan(task)):
            if len(tr.steps) >= self.max_steps:
                tr.stopped_by = "max_steps"
                break
            cost = TOKEN_COST.get(action.tool, 100)
            if self.budget_control and tr.tokens + cost > self.budget_tokens:
                tr.budget_exceeded = True
                tr.stopped_by = "budget"
                break
            key = action.tool + ":" + json.dumps(action.args, ensure_ascii=False, sort_keys=True)
            if self.loop_detection and key == last_key:
                tr.loop_blocked = True
                tr.steps.append(Step(i, "检测到与上一轮完全相同的动作，注入纠正提示并跳过",
                                     action, "LOOP_BLOCKED", tokens=0))
                continue
            last_key = key
            hitl_flag = False
            if self.hitl and self.registry.is_dangerous(action.tool):
                tr.hitl_triggered = True
                hitl_flag = True
                if not self.approver(task, action.tool):
                    tr.stopped_by = "hitl_rejected"
                    break
            result = self.registry.call(action.tool, action.args)
            tr.tokens += cost
            tr.called_tools.append(action.tool)
            if not result.ok:
                tr.errors.append(result.error)
                tr.steps.append(Step(i, "工具调用失败，把错误原文放回 observation 供下一步纠正",
                                     action, result.error, error=result.error,
                                     hitl=hitl_flag, tokens=cost))
                if not self.error_recovery:
                    tr.stopped_by = "tool_error_no_recovery"
                    break
                tr.recovery_attempted = True
                continue
            obs = json.dumps(result.data, ensure_ascii=False)[:160]
            tr.steps.append(Step(i, "按计划调用 " + action.tool, action, obs,
                                 hitl=hitl_flag, tokens=cost))
            if tr.errors and result.ok:
                tr.recovered = True
        if tr.stopped_by == "finished" and not tr.called_tools:
            tr.final_answer = "【超范围】该问题不在本系统支持范围内，已拒答。"
        elif tr.stopped_by == "finished":
            tr.final_answer = "已根据 " + "、".join(sorted(set(tr.called_tools))) + " 的结果给出回答。"
        else:
            tr.final_answer = "【未完成】终止原因：" + tr.stopped_by
        return tr
