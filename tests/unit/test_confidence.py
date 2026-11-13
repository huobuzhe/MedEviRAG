"""A6/A7 门禁：置信度单调性 + 可分解性（docs/05 §6）。"""

from __future__ import annotations

import pytest

from medevirag.rag.confidence import (
    CitationReport,
    SentenceSupport,
    compute_confidence,
    evidence_weight,
    platt,
)


def _fake_cands(n=8, dense=0.75):
    from medevirag.index.fusion import RetrievedChunk
    return [RetrievedChunk(chunk_id=f"d#c{i}", doc_id="d", text="t",
                           evidence_level="A" if i < 4 else "B",
                           publish_year=2024, dense_score=dense - 0.01 * i,
                           fused_rank=i + 1) for i in range(n)]


def _cite(n_cited=4, n_total=4):
    sents = tuple(SentenceSupport(text=f"s{i}", cited_chunk_ids=("d#c0",),
                                 landing_ok=True, entailed=True)
                  if i < n_cited else SentenceSupport(text=f"s{i}")
                  for i in range(n_total))
    return CitationReport(sentences=sents, dangling_citation_count=0,
                          total_citation_count=n_cited, sentences_total=n_total)


def test_platt_monotonic_increasing():
    assert platt(0.0, 0.670, 0.060) < platt(0.5, 0.670, 0.060) < platt(1.0, 0.670, 0.060)


def test_components_in_unit_interval():
    from medevirag.common.config import get_settings
    s = get_settings()
    bd = compute_confidence(_fake_cands(), ["d#c0"], _cite(), s.confidence.weights,
                            s.confidence.platt, rerank_applied=False)
    for name in ("rank_norm", "citation_coverage", "evidence_weight", "margin_norm", "score"):
        v = getattr(bd, name)
        assert 0.0 <= v <= 1.0, (name, v)


def test_score_equals_weighted_sum():
    from medevirag.common.config import get_settings
    s = get_settings()
    bd = compute_confidence(_fake_cands(), ["d#c0"], _cite(), s.confidence.weights,
                            s.confidence.platt, rerank_applied=False)
    w = s.confidence.weights
    expected = (w["rank"] * bd.rank_norm + w["citation"] * bd.citation_coverage
                + w["evidence"] * bd.evidence_weight + w["margin"] * bd.margin_norm)
    assert abs(expected - bd.score) < 1e-6


def test_no_candidates_zero_confidence():
    from medevirag.common.config import get_settings
    s = get_settings()
    bd = compute_confidence([], [], _cite(), s.confidence.weights,
                            s.confidence.platt, rerank_applied=False)
    assert bd.score == 0.0 and bd.unknown_level_ratio == 1.0


def test_evidence_weight_levels():
    assert evidence_weight(["A"]) == pytest.approx(1.00)
    assert evidence_weight(["unknown"]) == pytest.approx(0.45)
    assert evidence_weight([]) == 0.0


def test_confidence_monotonic_across_statuses():
    """A6：refused < partial < answered 三段递增（自造三组，不依赖夹具 expect）。"""
    from medevirag.common.config import get_settings
    from medevirag.rag.confidence import compute_confidence
    s = get_settings()
    w, pc = s.confidence.weights, s.confidence.platt

    def _conf(dense, cite_n, total, levels):
        from medevirag.index.fusion import RetrievedChunk
        cands = [RetrievedChunk(chunk_id=f"k#c{i}", doc_id="k", text="t",
                                evidence_level=levels[min(i, len(levels) - 1)],
                                publish_year=2024, dense_score=dense - 0.01 * i,
                                fused_rank=i + 1) for i in range(8)]
        sents = tuple(SentenceSupport(text=f"s{i}", cited_chunk_ids=("k#c0",),
                                      landing_ok=True, entailed=True)
                      if i < cite_n else SentenceSupport(text=f"s{i}")
                      for i in range(total))
        rep = CitationReport(sentences=sents, dangling_citation_count=0,
                             total_citation_count=cite_n, sentences_total=total)
        cited = [cid for ss in sents for cid in ss.cited_chunk_ids]
        return compute_confidence(cands, cited, rep, w, pc, rerank_applied=False)

    # refused：低分 + 无引用（rank 低被分数规则拒）
    low = _conf(0.55, 0, 4, ["unknown"] * 8)
    # partial：中等信号，conf 落 [0.45, 0.65)
    mid = _conf(0.64, 2, 4, ["A", "B"] * 4)
    # answered：强信号
    high = _conf(0.80, 4, 4, ["A"] * 8)
    assert low.score < mid.score < high.score, \
        f"单调性破坏: {low.score:.3f} < {mid.score:.3f} < {high.score:.3f}"
    assert mid.score < 0.65, f"mid 必须落在 partial 区间，实际 {mid.score:.3f}"

