"""T5 融合单测：RRF 的共识语义、稳定性、幂等，以及加权和的量纲敏感（docs/03 §T5）。"""

from __future__ import annotations

from medevirag.index.fusion import RetrievedChunk, rrf_fuse, weighted_fuse


def rc(
    cid: str,
    *,
    dense: float = 0.0,
    bm25: float = 0.0,
    dense_rank: int = 0,
    bm25_rank: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        doc_id=cid.split("#")[0],
        text=f"text of {cid}",
        dense_score=dense,
        bm25_score=bm25,
        dense_rank=dense_rank,
        bm25_rank=bm25_rank,
    )


def cids(chunks: list[RetrievedChunk]) -> list[str]:
    return [c.chunk_id for c in chunks]


def test_rrf_both_route_wins_and_is_stable() -> None:
    """两路都召回的块一定排在只被一路召回的块前面（RRF 的共识语义）。

    a = dense#1 + bm25#2，b = dense#2 + bm25#1，两者分数完全相同，
    按 (-fused_score, chunk_id) 的兜底键 a 在前——这就是稳定性的来源。
    """
    dense = [rc("a", dense=0.9, dense_rank=1), rc("b", dense=0.8, dense_rank=2), rc("c", dense_rank=3)]
    bm25 = [rc("b", bm25=20.0, bm25_rank=1), rc("a", bm25=10.0, bm25_rank=2), rc("d", bm25_rank=3)]
    out = rrf_fuse([dense, bm25], k=60, topk=10)
    assert cids(out) == ["a", "b", "c", "d"]
    assert out[0].source == "dense+bm25"
    assert out[1].source == "dense+bm25"
    assert out[2].source == "dense"
    assert out[3].source == "bm25"
    # 两路共识的分数（1/61+1/62）严格大于只被一路命中的 1/61
    assert out[0].fused_score > out[2].fused_score
    assert out[0].fused_score == out[1].fused_score
    for i, chunk in enumerate(out, start=1):
        assert chunk.fused_rank == i


def test_rrf_is_idempotent() -> None:
    dense = [rc("a", dense=0.9), rc("b", dense=0.8)]
    bm25 = [rc("b", bm25=20.0), rc("a", bm25=10.0)]
    first = rrf_fuse([dense, bm25], k=60, topk=10)
    second = rrf_fuse([dense, bm25], k=60, topk=10)
    assert cids(first) == cids(second)
    assert [c.fused_score for c in first] == [c.fused_score for c in second]


def test_rrf_tie_broken_by_chunk_id() -> None:
    """两路各召回一个不同块、名次相同 → 同分，必须按 chunk_id 兜底而不是 dict 顺序。"""
    dense = [rc("zzz", dense=0.9)]
    bm25 = [rc("aaa", bm25=9.0)]
    out = rrf_fuse([dense, bm25], k=60, topk=10)
    assert cids(out) == ["aaa", "zzz"]
    assert out[0].fused_score == out[1].fused_score


def test_rrf_truncates_after_sorting() -> None:
    dense = [rc(f"d{i}", dense=1.0 - i * 0.01) for i in range(10)]
    out = rrf_fuse([dense], k=60, topk=3)
    assert cids(out) == ["d0", "d1", "d2"]
    assert out[-1].fused_rank == 3


def test_rrf_k_change_shrinks_the_gap() -> None:
    """k 越大，1/(k+rank) 的差距越小——这是 RRF 只用名次的直接后果。"""
    dense = [rc("a", dense=0.9)]
    bm25 = [rc("b", bm25=9.0), rc("c", bm25=8.0)]
    small = rrf_fuse([dense, bm25], k=1, topk=3)
    large = rrf_fuse([dense, bm25], k=1000, topk=3)
    gap_small = small[0].fused_score - small[-1].fused_score
    gap_large = large[0].fused_score - large[-1].fused_score
    assert gap_large < gap_small


def test_rrf_keeps_scores_from_both_routes() -> None:
    dense = [rc("a", dense=0.9, dense_rank=1)]
    bm25 = [rc("a", bm25=12.5, bm25_rank=1)]
    out = rrf_fuse([dense, bm25], k=60, topk=10)
    assert out[0].dense_score == 0.9
    assert out[0].bm25_score == 12.5
    assert out[0].dense_rank == 1
    assert out[0].bm25_rank == 1


def test_weighted_sum_is_scale_sensitive() -> None:
    """同一块在两路各自是冠军 → min-max 后都是 1.0，加权和 = 1.0。

    这条正是 RRF 存在的理由：BM25 绝对分数（20.0）与余弦（0.9）不可比，
    只有归一化之后才能加权，而归一化本身又抹掉了跨 query 的强度差异。
    """
    dense = [rc("a", dense=0.9), rc("b", dense=0.5)]
    bm25 = [rc("a", bm25=20.0), rc("c", bm25=5.0)]
    out = weighted_fuse(dense, bm25, alpha=0.5, topk=10)
    assert abs(out[0].fused_score - 1.0) < 1e-9
    assert out[0].chunk_id == "a"
    assert out[0].source == "dense+bm25"


def test_weighted_sum_alpha_extremes() -> None:
    dense = [rc("a", dense=0.9), rc("b", dense=0.5)]
    bm25 = [rc("b", bm25=20.0), rc("c", bm25=5.0)]
    dense_only = weighted_fuse(dense, bm25, alpha=1.0, topk=10)
    bm25_only = weighted_fuse(dense, bm25, alpha=0.0, topk=10)
    assert dense_only[0].chunk_id == "a"
    assert bm25_only[0].chunk_id == "b"


def test_weighted_sum_all_equal_scores() -> None:
    """全同分时 min-max 给全 1.0（不除零、不拆并列），顺序回落到 chunk_id。"""
    dense = [rc("b", dense=0.5), rc("a", dense=0.5)]
    out = weighted_fuse(dense, [], alpha=0.5, topk=10)
    assert cids(out) == ["a", "b"]
    assert all(abs(c.fused_score - 0.5) < 1e-9 for c in out)


def test_empty_runs() -> None:
    assert rrf_fuse([[], []], k=60, topk=10) == []
    assert weighted_fuse([], [], alpha=0.5, topk=10) == []
