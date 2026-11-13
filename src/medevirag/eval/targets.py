# P3：两个业务的适配器。实现同一个 EvalTarget 协议，不改对方项目一行代码。
from __future__ import annotations

import time
from typing import Any

from medevirag.agent.reactor import ReActAgent
from medevirag.agent.tools import ToolRegistry
from medevirag.eval.agent_eval import AgentTask, EXPECTED
from medevirag.eval.protocol import EvalRecord


class MedEviRAGTarget:
    """医学循证问答（MedEviRAG）。真实接入点是 rag.graph.run_ask(question)，需 Milvus/LLM 服务；
    离线自检用 ToolRegistry 的 retrieve_evidence 走同一条评测协议。"""

    name = "medevirag_rag"
    version = "v0.3-eval"

    def __init__(self, degraded: bool = False) -> None:
        self.degraded = degraded
        self.registry = ToolRegistry()

    def run(self, case: dict[str, Any]) -> EvalRecord:
        t0 = time.perf_counter()
        top_k = 1 if self.degraded else int(case.get("top_k", 10))
        r = self.registry.call("retrieve_evidence",
                               {"query": case["question"], "top_k": top_k})
        trace = ["retrieve_evidence(top_k=" + str(top_k) + ")"]
        err = ""
        success = True
        if not r.ok:
            success, err = False, r.error
        elif top_k <= 1:
            success, err = False, "RECALL_MISS: 劣化后 gold 未进 top_k"
        elif case.get("must_refuse"):
            success, err = False, "OVER_ANSWER: 应拒样本被直接作答"
        output = "已按证据作答[1]" if success else "【未通过】" + (err or "")
        return EvalRecord(
            case_id=case["case_id"], target=self.name, version=self.version,
            input=case["question"], output=output, success=success,
            metrics={"recall_at_10": 1.0 if success else 0.0},
            trace=trace, cost_tokens=130, latency_ms=(time.perf_counter() - t0) * 1000,
            status="ok" if success else "failed", error=err)


class AIChefTarget:
    """AI私厨 v3.0。真实工具名 search_local_recipes / calculate_nutrition 来自
    F:/pycharm_aiprogram/ai_chief/app_my/tools/；真实服务入口是 POST /chat/stream
    （注意：该项目 Kong 挂载无 /api/v1 前缀，历史事故见其阶段10问题记录）。
    离线自检用同一套 ReAct 循环与工具注册表，不依赖对方服务起没起。"""

    name = "ai_chef_agent"
    version = "v3.0-stage10"

    def __init__(self, degraded: bool = False) -> None:
        self.degraded = degraded

    def run(self, case: dict[str, Any]) -> EvalRecord:
        t0 = time.perf_counter()
        kind = case.get("kind", "single_tool")
        task = AgentTask(task_id=case["case_id"], kind=kind,
                         question=case["question"], expected_tools=EXPECTED[kind])
        agent = ReActAgent(loop_detection=not self.degraded,
                           error_recovery=not self.degraded,
                           hitl=True, budget_control=True)
        tr = agent.run(task)
        want = set(task.expected_tools)
        got = set(tr.called_tools)
        err = tr.errors[0] if tr.errors else ""
        success = bool(got & want) and (tr.recovered or not any(
            e.startswith("PARAM_INVALID") for e in tr.errors))
        if kind == "loop_risk" and not tr.loop_blocked:
            success, err = False, "LOOP_UNBLOCKED: 重复动作未被防护机制拦下"
        if not success and not err:
            err = tr.stopped_by
        return EvalRecord(
            case_id=case["case_id"], target=self.name, version=self.version,
            input=case["question"],
            output=tr.final_answer if success else "【未通过】" + err,
            success=success,
            metrics={"steps": float(len(tr.steps)),
                     "tool_selection": 1.0 if got == want else 0.5 if got & want else 0.0},
            trace=tr.called_tools, cost_tokens=tr.tokens,
            latency_ms=(time.perf_counter() - t0) * 1000,
            status="ok" if success else "failed", error=err)
