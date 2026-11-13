"""T5-a：RRF 与加权和融合（docs/03 §T5）。

为什么需要融合而不是直接比分数：BM25 分数域是 [0,+inf)（常见 3~35），
余弦是 [-1,1]，量纲不可比。RRF 只用名次，天然免疫量纲问题，且免调参。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["RetrievedChunk", "rrf_fuse", "weighted_fuse"]


@dataclass(frozen=True)
class RetrievedChunk:
    """一路召回的中间结果，也是最终返回的一项。

    字段顺序与 docs/03 §T5 一致：前段是展示字段，后段是三路分数与名次。
    """

    chunk_id: str
    doc_id: str
    text: str
    context_header: str = ""
    chunk_type: str = "paragraph"
    doc_title: str = ""
    doc_type: str = "guideline"
    publish_year: int = 0
    lang: str = "zh"
    evidence_level: str = "unknown"
    rec_strength: str = "unknown"
    section_path: tuple[str, ...] = field(default_factory=tuple)
    page_start: int = 0
    page_end: int = 0
    dense_score: float = 0.0
    bm25_score: float = 0.0
    dense_rank: int = 0
    bm25_rank: int = 0
    fused_score: float = 0.0
    fused_rank: int = 0
    source: str = ""
    # S3 重排层追加（docs/04 §T1）：dataclasses.replace 需要字段已声明。
    rerank_score: float = 0.0
    rerank_rank: int = 0


def _stable_key(chunk: RetrievedChunk) -> tuple[float, str]:
    """同分按 chunk_id 兜底：不许依赖 dict/set 迭代顺序（A12 的确定性要求）。"""
    return (-chunk.fused_score, chunk.chunk_id)


def rrf_fuse(
    runs: Sequence[Sequence[RetrievedChunk]],
    k: int = 60,
    topk: int = 50,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion：score = sum(1/(k + rank))。

    rank 从 1 开始；某路没召回到的块不贡献分数，但会保留另一路的分数。
    """
    acc: dict[str, float] = {}
    keep: dict[str, RetrievedChunk] = {}
    route: dict[str, list[str]] = {}
    for name, run in zip(("dense", "bm25", "extra"), runs, strict=False):
        for rank, chunk in enumerate(run, start=1):
            cid = chunk.chunk_id
            acc[cid] = acc.get(cid, 0.0) + 1.0 / (float(k) + float(rank))
            prev = keep.get(cid)
            if prev is None:
                keep[cid] = chunk
            else:
                keep[cid] = dataclasses.replace(
                    prev,
                    dense_score=max(prev.dense_score, chunk.dense_score),
                    bm25_score=max(prev.bm25_score, chunk.bm25_score),
                    dense_rank=prev.dense_rank or chunk.dense_rank,
                    bm25_rank=prev.bm25_rank or chunk.bm25_rank,
                )
            route.setdefault(cid, [])
            if name not in route[cid]:
                route[cid].append(name)

    merged: list[RetrievedChunk] = []
    for cid, score in acc.items():
        merged.append(dataclasses.replace(keep[cid], fused_score=score, source="+".join(route[cid])))
    merged.sort(key=_stable_key)
    merged = merged[: max(0, int(topk))]
    return [dataclasses.replace(c, fused_rank=i) for i, c in enumerate(merged, start=1)]


def _minmax(values: Mapping[str, float]) -> dict[str, float]:
    """min-max 归一化；全同分时统一给 1.0（避免除零，也避免把并列拆开）。"""
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if abs(hi - lo) < 1e-12:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def weighted_fuse(
    dense: Sequence[RetrievedChunk],
    bm25: Sequence[RetrievedChunk],
    alpha: float = 0.5,
    topk: int = 50,
) -> list[RetrievedChunk]:
    """加权和：min-max 归一化后 alpha*dense + (1-alpha)*bm25。

    作为 RRF 的对照实验（ADR-02）：alpha 是调参旋钮，RRF 不需要。
    """
    d_norm = _minmax({c.chunk_id: c.dense_score for c in dense})
    b_norm = _minmax({c.chunk_id: c.bm25_score for c in bm25})
    keep: dict[str, RetrievedChunk] = {}
    for chunk in list(dense) + list(bm25):
        keep.setdefault(chunk.chunk_id, chunk)
    score: dict[str, float] = {}
    for cid in keep:
        score[cid] = float(alpha) * d_norm.get(cid, 0.0) + (1.0 - float(alpha)) * b_norm.get(cid, 0.0)
    merged = [
        dataclasses.replace(
            keep[cid],
            fused_score=value,
            source="+".join(
                [n for n, pool in (("dense", d_norm), ("bm25", b_norm)) if cid in pool]
            ),
        )
        for cid, value in score.items()
    ]
    merged.sort(key=_stable_key)
    merged = merged[: max(0, int(topk))]
    return [dataclasses.replace(c, fused_rank=i) for i, c in enumerate(merged, start=1)]
