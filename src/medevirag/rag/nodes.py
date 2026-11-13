"""T8 LangGraph 节点 + RAGState（docs/04 §T8）。

8 个节点与 00 §4.2 一一对应；证据装配（父节点回填 + 子块编号）不单独成节点，作为纯函数
在 node_generate 内调用。图结构与 doc 完全一致，渲染耗时并入 generate 埋点。

所有重资源对象（retriever/reranker/llm/judge/chunk_meta）走模块级单例，避免每问重加载
（重排器加载约 1.8s，否则 P95 直接崩）。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, TypedDict

from medevirag.common.llm import FakeLLM

log = logging.getLogger(__name__)


class RAGState(TypedDict, total=False):
    question: str
    route_kind: str             # S4: clinical | off_domain | non_evidence | individual（键名避免与节点同名）
    route_score: float          # 机械路由置信度（命中即 1.0）
    topic_covered: bool         # 主题是否在词表内（R-02 输入）
    queries: list[str]
    candidates: list[dict[str, Any]]      # S2 融合候选（≤50）
    filtered: list[dict[str, Any]]        # 元数据硬过滤后
    reranked: list[dict[str, Any]]        # Top8
    evidence: dict[str, dict[str, Any]]   # {"C1": Evidence-as-dict}
    marker_map: dict[str, str]  # {"C1": chunk_id}
    evidence_text: str
    raw: str
    answer: str
    used_citations: list[str]
    insufficient_evidence: bool
    parse_ok: bool
    citations: list[dict[str, Any]]
    illegal_citations: list[str]
    support_rate: float
    status: str                 # answered | partial | refused
    confidence: float | None    # S4 填充，S3 恒为 None
    refusal_reason: str | None
    citation_report: object     # S4: refusal.build_citation_report 产出的 CitationReport
    rule_id: str | None         # S4: 命中的规则编号（审计用）
    rule_detail: dict[str, Any]           # S4: 规则命中片段（渲染拒答文案的插槽）
    confidence_breakdown: dict[str, Any]  # S4: 四分量拆解（落库 qa_log）
    degrade_note: str | None    # S4: partial 降级提示键（PARTIAL_NOTES）
    must_retry: bool
    degraded: bool
    retry_count: int
    rerank_applied: bool        # false = R2 否决分支（直通 RRF Top8）
    trace: dict[str, int]


def timed(name: str, fn: Callable[[RAGState], dict[str, Any]]) -> Callable[[RAGState], dict[str, Any]]:
    """节点级耗时埋点。trace 必须基于父状态拷贝，否则节点间互相覆盖。"""

    def wrapper(state: RAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        patch = fn(state) or {}
        ms = int((time.perf_counter() - t0) * 1000)
        trace = dict(state.get("trace") or {})
        trace[name] = ms
        patch["trace"] = trace
        log.info("node_done", extra={"node": name, "ms": ms})
        return patch

    return wrapper


INDIVIDUAL_HINTS = ("我该吃多少", "我的剂量", "可以停药吗", "我能不能吃", "我是不是该")
"""S3 遗留的粗词表；S4 起 node_route 不再使用（改由规则正则驱动），保留仅为兼容外部引用。"""


def _rerank_cfg() -> dict[str, Any]:
    """读 configs/retrieval.yaml 的 rerank 段（缺省视为关闭）。"""
    from medevirag.index.paths import load_retrieval_config

    return dict(load_retrieval_config().get("rerank") or {})


def _rerank_enabled() -> bool:
    return bool(_rerank_cfg().get("enabled", False))


def _rerank_top_n() -> int:
    return int(_rerank_cfg().get("top_n", 8))


# --------------------------------------------------------------------------- #
# 模块级单例
# --------------------------------------------------------------------------- #
_retriever = None
_reranker = None
_llm = None
_judge = None
_chunk_meta = None
_parent_text: dict[str, str] = {}


def get_retriever() -> Any:
    global _retriever
    if _retriever is None:
        from medevirag.index.hybrid import HybridRetriever

        _retriever = HybridRetriever()
    return _retriever


def get_reranker() -> Any:
    global _reranker
    if _reranker is None:
        from medevirag.rag.reranker import Reranker

        _reranker = Reranker()
    return _reranker


def get_llm() -> Any:
    """返回 chat 接口的 LLM（docs/04 §T5 口径）。

    common/llm.FakeLLM 接口是 complete(req)->LLMResponse，与 chat 不一致；
    这里用适配器包一层，保证节点代码只依赖 chat 协议。生产默认走 FakeLLM（ADR-06 双轨）。

    2026-11-13 新增 MEDEVIRAG_LLM_PROVIDER 开关：未设置 / fake / off 时完全走下面
    原来的 FakeLLM 分支（离线测试、CI 门禁、pipeline_selfcheck 不受影响）；设为
    siliconflow / dashscope 等 models.yaml 里的 provider 名时，改用
    common/openai_chat.OpenAICompatChat 真实调用。关掉环境变量即回滚。
    """

    global _llm
    if _llm is None:
        import os

        provider = os.environ.get("MEDEVIRAG_LLM_PROVIDER", "").strip().lower()
        if provider and provider not in ("fake", "off", "none"):
            from medevirag.common.openai_chat import OpenAICompatChat

            _llm = OpenAICompatChat(provider)
            return _llm

        fake = FakeLLM()

        class _Adapter:
            _fake: FakeLLM  # 测试里通过 get_llm()._fake 读 calls

            def __init__(self, fake_llm: FakeLLM) -> None:
                self._fake = fake_llm

            def chat(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
                from medevirag.common.llm import LLMRequest

                resp = self._fake.complete(LLMRequest(system=system, user=user, temperature=temperature, max_tokens=max_tokens))
                return resp.text

        _llm = _Adapter(fake)
    return _llm


def get_judge() -> Any:
    """默认离线 LexicalJudge（符合 ADR-06，无需联网）。"""
    global _judge
    if _judge is None:
        from medevirag.rag.verifier import LexicalJudge

        _judge = LexicalJudge()
    return _judge


def get_chunk_meta() -> dict[str, dict[str, Any]]:
    """chunk_id -> 证据装配所需的元数据。

    为什么要合并两个来源：索引层的 chunk_meta.jsonl 只存展示快照，
    **不含 locator 与 parent_id**；而 T4 的父节点聚合与 T6 的引用回链（page/bbox）
    都必须有这两个字段。若不补，citation 的 url 会退化成 page=0&bbox= 的空深链。
    因此这里以 chunk_meta 为底，用 chunks.jsonl（已 lru_cache）补齐 pub 字段。
    """
    global _chunk_meta
    if _chunk_meta is None:
        from medevirag.index.chunk_meta import ChunkMetaIndex
        from medevirag.index.paths import CHUNK_META_PATH
        from medevirag.rag.citations import _chunk_index

        base = ChunkMetaIndex.from_jsonl(CHUNK_META_PATH)
        full = _chunk_index()
        merged: dict[str, dict[str, Any]] = {}
        for cid, row in full.items():
            meta = dict(base.get(cid) or {})
            meta.update({k: row.get(k) for k in
                         ("parent_id", "locator", "page_start", "page_end")})
            merged[cid] = meta
        _chunk_meta = merged
    return _chunk_meta


def get_parent_text() -> dict[str, str]:
    """父节点全文索引（small-to-big 回填用）。

    真实语料 parent_id 均不进 chunks.jsonl（父块不落索引），故返回空 dict，
    走 render_evidence_blocks 的"父节点不可得"降级分支。如未来引入章节树可在此填充。
    """
    return _parent_text


# --------------------------------------------------------------------------- #
# 8 个业务节点
# --------------------------------------------------------------------------- #
def node_route(state: RAGState) -> dict[str, Any]:
    """S4 扩成 docs/05 的 5 值路由：clinical | off_domain | non_evidence | individual。

    individual 只由 R-08~R-12 的正则决定（与规则表同一真值来源）：以前用
    INDIVIDUAL_HINTS 粗词表，route='individual' 但正则不命中时 decide 里
    没有规则可归因，会被空候选误判成 low_relevance——理由错配直接打穿 A3。
    机械命中记 route_score=1.0（R-01 要求 >=0.60）。
    """
    from medevirag.rag.refusal_rules import (
        DIAGNOSIS_RE,
        DOSAGE_RE,
        EMERGENCY_RE,
        NON_EVIDENCE_RE,
        PRESCRIBE_RE,
        SPECIAL_POP_RE,
        is_topic_covered,
    )

    q = state["question"].strip()
    covered = is_topic_covered(q)

    # ★ 偏差 D8：individual 只由**规则会命中的组合**触发。SPECIAL 单独命中
    # （如"孕妇可以吃对乙酰氨基酚吗"）是合法循证问题，走 clinical；否则
    # route=individual 但 R-08~R-12 都不命中，空候选会让它被归因成
    # low_relevance——理由错配直接打穿 A3。
    indiv = (EMERGENCY_RE.search(q) or DOSAGE_RE.search(q) or PRESCRIBE_RE.search(q)
             or DIAGNOSIS_RE.search(q)
             or (SPECIAL_POP_RE.search(q) and (DOSAGE_RE.search(q) or PRESCRIBE_RE.search(q))))
    if indiv:
        return {"route_kind": "individual", "route_score": 1.0,
                "topic_covered": covered}
    if NON_EVIDENCE_RE.search(q):
        return {"route_kind": "non_evidence", "route_score": 1.0,
                "topic_covered": covered}
    if len(q) < 4 or not any("\u4e00" <= ch <= "\u9fff" for ch in q):
        return {"route_kind": "off_domain", "route_score": 1.0,
                "topic_covered": covered}
    return {"route_kind": "clinical", "route_score": 0.0,
            "topic_covered": covered}


def node_rewrite(state: RAGState) -> dict[str, Any]:
    from medevirag.rag.rewrite import build_queries

    return {"queries": build_queries(state["question"], max_q=3)}


def node_retrieve(state: RAGState) -> dict[str, Any]:
    """多路改写时先各自召回再 RRF；单查询直接复用。候选保留 RetrievedChunk 对象，
    因为下游 apply_hard_filters 与 Reranker 都按属性访问（转 dict 会 AttributeError）。"""
    from medevirag.index.fusion import rrf_fuse

    retriever = get_retriever()
    queries = state.get("queries") or [state["question"]]
    per_query = [retriever.retrieve(q, topk=50) for q in queries]
    merged = rrf_fuse(per_query, k=60, topk=50) if len(per_query) > 1 else per_query[0]
    return {"candidates": list(merged)}


def node_filter(state: RAGState) -> dict[str, Any]:
    from medevirag.index.filters import apply_hard_filters

    cands = state.get("candidates") or []
    return {"filtered": apply_hard_filters(list(cands))}  # type: ignore[arg-type]


def node_rerank(state: RAGState) -> dict[str, Any]:
    """按 configs/retrieval.yaml 的 rerank.enabled 决定是否精排。

    实测 dev60 上 ΔMRR=0.0557 < 0.10 门禁，故默认 enabled=false：直通 RRF 顺序取 Top8
    （docs/04 §T2 R2 否决分支）。开关为 true 时走 T1 的 cross-encoder 重排。
    """
    filtered = state.get("filtered") or []
    top_n = _rerank_top_n()
    if not _rerank_enabled():
        return {"reranked": list(filtered[:top_n]), "rerank_applied": False}
    reranker = get_reranker()
    top = reranker.rerank(state["question"], filtered, top_n=top_n)
    return {"reranked": list(top), "rerank_applied": True}


def node_generate(state: RAGState) -> dict[str, Any]:
    from medevirag.rag.context import build_evidence, marker_map, render_evidence_blocks
    from medevirag.rag.generator import generate

    evs = build_evidence(state.get("reranked") or [], get_chunk_meta(), max_n=8)
    evidence = {e.marker: e.__dict__ for e in evs}
    evidence_text = render_evidence_blocks(evs, get_parent_text())

    degraded = bool(state.get("must_retry"))
    res = generate(get_llm(), state["question"], evidence_text, degraded=degraded)
    return {
        "evidence": evidence,
        "marker_map": marker_map(evs),
        "evidence_text": evidence_text,
        "raw": res.raw,
        "answer": res.answer,
        "used_citations": res.used_citations,
        "insufficient_evidence": res.insufficient_evidence,
        "parse_ok": res.parse_ok,
        "degraded": degraded,
        "retry_count": int(state.get("retry_count", 0)) + (1 if degraded else 0),
        "must_retry": False,
    }


def node_verify(state: RAGState) -> dict[str, Any]:
    from medevirag.rag.citations import build_citations, citation_to_dict
    from medevirag.rag.refusal import build_citation_report
    from medevirag.rag.verifier import verify

    ev = state.get("evidence") or {}
    res = verify(state.get("answer") or "", ev, get_judge())
    # S4: 把 VerifyResult 适配成 docs/05 §2.2 的 CitationReport，供 decide 消费。
    # 注意用 cleaned_answer 之前先取 res.sentences（校验器自己的逐句结果）。
    report = build_citation_report(
        state.get("answer") or "", res, state.get("marker_map") or {})
    return {
        "answer": res.cleaned_answer,
        "citations": [citation_to_dict(c) for c in build_citations(res.cleaned_answer, ev)],
        "illegal_citations": res.illegal_markers,
        "support_rate": res.support_rate,
        "citation_report": report,
        "must_retry": res.must_retry or not state.get("parse_ok", True),
    }


def node_decide(state: RAGState) -> dict[str, Any]:
    """S4 三级判定链（docs/05 §T7）：非分数规则 -> 置信度 -> 分数规则 + 阈值。

    decide 是纯函数：不读网络、不调 LLM，可 100% 单测。refusal_reason 显式
    .value 转字符串（Enum 直落 PG 会变成 'RefusalReason.INDIV_DOSAGE'）。
    拒答时清空 citations 并渲染模板文案——拒答带引用卡片会被用户误读为
    "有出处就有答案"。
    """
    from medevirag.common.config import get_settings
    from medevirag.rag.confidence import CitationReport
    from medevirag.rag.refusal import decide, refusal_config
    from medevirag.rag.refusal_rules import AskContext
    from medevirag.rag.refusal_templates import render

    s = get_settings()
    ctx = AskContext(
        question=state["question"],
        route=state.get("route_kind", "clinical"),
        route_score=float(state.get("route_score", 0.0)),
        topic_covered=bool(state.get("topic_covered", True)),
        candidates=list(state.get("reranked") or [])[:8],  # 切片必须在节点里做
        citation=state.get("citation_report") or CitationReport(),  # type: ignore[arg-type]
        now_year=s.refusal.now_year,
    )
    res = decide(ctx, s.confidence.weights, s.confidence.platt, refusal_config(),
                 rerank_applied=bool(state.get("rerank_applied", False)))

    # ★ S4 偏差 D5：docs/05 没给"模型自报证据不足"（insufficient_evidence）的归因，
    # 但 S3 已有该通道且 FakeLLM 剧本会触发。docs §T7 明确 insufficient 优先：
    # 只要没被**具体规则**（非 R-CONF）拒掉，就统一归因 LOW_RELEVANCE / R-INSUFF。
    # 具体规则命中说明有更准的成因（如 low_support），不该被自报覆盖。
    if (state.get("insufficient_evidence")
            and (res.status != "refused" or res.rule_id == "R-CONF")):
        from medevirag.common.types import DecideResult, RefusalReason
        res = DecideResult("refused", res.confidence, res.breakdown,
                           RefusalReason.LOW_RELEVANCE, "R-INSUFF",
                           "low_relevance", None,
                           {"why": "生成模型自报现有证据不足以回答该问题"})

    out: dict[str, Any] = {
        "status": res.status,
        "confidence": round(res.confidence, 4),
        "refusal_reason": res.refusal_reason.value if res.refusal_reason else None,
        "rule_id": res.rule_id,
        "confidence_breakdown": res.breakdown,
        "degrade_note": res.note,
    }
    if res.status == "refused":
        out["answer"] = render(res.template_key or "low_relevance",
                               state["question"], res.detail or state.get("rule_detail") or {})
        out["citations"] = []
    return out
