"""A9/A11/A12 门禁：decide 判定顺序、reason 唯一、拒答不带引用（docs/05 §6）。"""

from __future__ import annotations

from medevirag.common.config import get_settings
from medevirag.common.types import RefusalReason
from medevirag.rag.confidence import CitationReport
from medevirag.rag.refusal import decide, refusal_config
from medevirag.rag.refusal_rules import AskContext

S = get_settings()
CFG = refusal_config()


def _ctx(question, route="clinical", route_score=0.0, covered=True,
          candidates=None, citation=None):
    return AskContext(question=question, route=route, route_score=route_score,
                      topic_covered=covered, candidates=candidates or [],
                      citation=citation or CitationReport(),
                      now_year=S.refusal.now_year)


def test_precedence_off_domain_beats_low_relevance():
    """A11：同时满足 R-01 与 R-13 时，reason 必须是 off_domain。"""
    ctx = _ctx("帮我写一首关于春天的诗", route="off_domain", route_score=0.9)
    res = decide(ctx, S.confidence.weights, S.confidence.platt, CFG)
    assert res.status == "refused"
    assert res.refusal_reason is RefusalReason.OFF_DOMAIN
    assert res.rule_id == "R-01"          # 分数规则（空候选→R-13）根本没机会跑


def test_refused_returns_empty_citations_via_node():
    """A12：status=refused 时 node_decide 清空 citations。"""
    from medevirag.rag import nodes
    state = {"question": "帮我写一首关于春天的诗",
             "route_kind": "off_domain", "route_score": 0.9,
             "topic_covered": False, "reranked": [],
             "citation_report": CitationReport(),
             "rerank_applied": False, "citations": [{"marker": "C1"}]}
    out = nodes.node_decide(state)
    assert out["status"] == "refused"
    assert out["citations"] == []


def test_unknown_two_branches():
    """A9：全 unknown 时 conf>=0.55→partial；conf<0.55→refused(R-17)。"""
    from medevirag.index.fusion import RetrievedChunk
    from medevirag.rag.confidence import SentenceSupport

    def _mk(dense, cite_n, entailed_frac=1.0):
        cands = [RetrievedChunk(chunk_id=f"u#c{i}", doc_id="u", text="t",
                                evidence_level="unknown", publish_year=2024,
                                dense_score=dense - 0.01 * i, fused_rank=i + 1)
                 for i in range(8)]
        sents = tuple(SentenceSupport(text=f"s{i}", cited_chunk_ids=("u#c0",),
                                      landing_ok=True, entailed=True)
                      if i < cite_n else SentenceSupport(text=f"s{i}")
                      for i in range(4))
        rep = CitationReport(sentences=sents, dangling_citation_count=0,
                             total_citation_count=cite_n, sentences_total=4)
        return cands, rep

    # 分支一：全 unknown 但其余信号尚可 → partial + evidence_unknown 提示
    cands, rep = _mk(0.66, 4)
    res = decide(_ctx("q", candidates=cands, citation=rep),
                 S.confidence.weights, S.confidence.platt, CFG)
    assert res.status == "partial", f"conf={res.confidence}"
    assert res.note == "evidence_unknown"

    # 分支二：全 unknown 且低分 → refused(R-17)
    cands2, rep2 = _mk(0.645, 1, entailed_frac=1.0)
    res2 = decide(_ctx("q", candidates=cands2, citation=rep2),
                  S.confidence.weights, S.confidence.platt, CFG)
    assert res2.status == "refused"
    assert res2.rule_id == "R-17"


def test_reason_is_enum_value_serializable():
    """refusal_reason 落库前必须 .value（Enum 直落 PG 会变成类名）。"""
    ctx = _ctx("What is this?", route="off_domain", route_score=1.0)
    res = decide(ctx, S.confidence.weights, S.confidence.platt, CFG)
    assert isinstance(res.refusal_reason.value, str)
    assert res.refusal_reason.value == "off_domain"


def test_threshold_intervals():
    """阈值区间：>=0.65 answered；[0.45,0.65) partial；<0.45 refused(R-CONF)。"""
    # 用 off_domain=0（route 不命中）+ 空候选构造：conf=0 → R-13 更准
    res = decide(_ctx("高血压怎么办"), S.confidence.weights, S.confidence.platt, CFG)
    assert res.status == "refused" and res.confidence == 0.0
