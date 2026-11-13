"""T8 LangGraph 图：8 节点 + 条件边 + 埋点（docs/04 §T8，ADR-08）。

为什么用图而不是串行函数：需要分支（越界直接拒，省一次 LLM 调用）、可中断回退
（verify 失败回 generate）、每步可观测。串行函数做这三件事会变成 if-else 面条且没有统一 trace。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from .nodes import (
    RAGState,
    node_decide,
    node_filter,
    node_generate,
    node_rerank,
    node_retrieve,
    node_rewrite,
    node_route,
    node_verify,
    timed,
)
from .refusal_templates import PARTIAL_NOTES

_APP = None

BUSINESS_NODES = ("route", "rewrite", "retrieve", "filter", "rerank", "generate", "verify", "decide")


def after_route(s: RAGState) -> str:
    """越界 / 个体化用药 → 直接 decide，0 次 LLM 调用。"""
    return "proceed" if s.get("route_kind") == "clinical" else "refuse"


def after_rerank(s: RAGState) -> str:
    """过滤 + 重排后一个候选都没有 → 不浪费一次生成。"""
    return "generate" if s.get("reranked") else "refuse"


def after_verify(s: RAGState) -> str:
    """重试只允许 1 次：第二轮 verify 无论好坏都进 decide。"""
    if s.get("must_retry") and int(s.get("retry_count", 0)) < 1:
        return "retry"
    return "decide"


def build_graph() -> Any:
    g = StateGraph(RAGState)
    for name, fn in (
        ("route", node_route),
        ("rewrite", node_rewrite),
        ("retrieve", node_retrieve),
        ("filter", node_filter),
        ("rerank", node_rerank),
        ("generate", node_generate),
        ("verify", node_verify),
        ("decide", node_decide),
    ):
        g.add_node(name, timed(name, fn))

    g.set_entry_point("route")
    g.add_conditional_edges("route", after_route, {"proceed": "rewrite", "refuse": "decide"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "filter")
    g.add_edge("filter", "rerank")
    g.add_conditional_edges("rerank", after_rerank, {"generate": "generate", "refuse": "decide"})
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, {"retry": "generate", "decide": "decide"})
    # S4 docs §T7：decide 之后按 status 汇入 END（不新增节点，保持八节点）。
    # 只写条件边，不与 add_edge 并存（重复入边会让重试计数被偷偷加倍）。
    g.add_conditional_edges("decide", lambda s: s["status"],
                            {"answered": END, "partial": END, "refused": END})
    return g.compile()


def run_ask(question: str) -> dict[str, Any]:
    global _APP
    if _APP is None:
        _APP = build_graph()
    init: RAGState = {
        "question": question,
        "retry_count": 0,
        "degraded": False,
        "must_retry": False,
        "trace": {},
        "status": "answered",
    }
    out = _APP.invoke(init)
    # S5：评测 runner 需要的原始字段直接透出（docs/06 §T9 的 row 契约）
    citation_report = out.get("citation_report")
    marker_map = out.get("marker_map") or {}
    evidence = out.get("evidence") or {}
    context_chunks = [dict(e, chunk_id=e["chunk_id"]) for e in evidence.values()]
    sentences = []
    if citation_report is not None:
        sentences = [
            {"text": s.text, "cited_chunk_ids": list(s.cited_chunk_ids),
             "landing_ok": s.landing_ok, "entailed": s.entailed}
            for s in citation_report.sentences
        ]
    return {
        "answer": out.get("answer", ""),
        "citations": out.get("citations", []),
        "status": out.get("status"),
        "confidence": out.get("confidence"),
        "refusal_reason": out.get("refusal_reason"),
        "disclaimer": "本回答仅供技术验证，不构成诊疗建议。",
        "trace": out.get("trace", {}),
        "used_citations": out.get("used_citations", []),
        "illegal_citations": out.get("illegal_citations", []),
        "support_rate": out.get("support_rate", 0.0),
        # S4：审计与前端需要的判定细节
        "rule_id": out.get("rule_id"),
        "confidence_breakdown": out.get("confidence_breakdown", {}),
        "degrade_note": out.get("degrade_note"),
        "partial_note": PARTIAL_NOTES.get(out.get("degrade_note") or "", "")
        if out.get("status") == "partial" else "",
        # S5 runner 契约字段
        "retrieved_chunk_ids": [getattr(c, "chunk_id", "") if not isinstance(c, dict) else c["chunk_id"] for c in (out.get("reranked") or [])],
        "context_chunks": context_chunks,
        "sentences": sentences,
        "citation_map": marker_map,
        "dangling_citation_count": int(getattr(citation_report, "dangling_citation_count", 0) or 0),
        "total_citation_count": int(getattr(citation_report, "total_citation_count", 0) or 0),
    }
