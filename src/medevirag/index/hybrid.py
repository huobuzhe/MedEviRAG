"""T5-b：混合召回编排（docs/03 §T5）。

一条 query 的路径：
    pool.submit(_dense_run) + pool.submit(_bm25_run)  -> 并发两路
    rrf_fuse([dense, bm25], k=rrf_k)                  -> 融合
    apply_hard_filters(...)                           -> post-filter（ADR-03）

单例纪律：Embedder（2.2GB 显存）与两套索引在 HybridRetriever 里各建一次，
四组消融复用同一实例，只换融合函数（docs/03 §T5 + 本阶段约束 5）。
"""

from __future__ import annotations

import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from medevirag.index import milvus_store
from medevirag.index.bm25_store import BM25Store, ChineseTokenizer
from medevirag.index.chunk_meta import ChunkMetaIndex
from medevirag.index.embedder import Embedder, InMemoryLRUCache
from medevirag.index.filters import FilterRules, apply_hard_filters
from medevirag.index.fusion import RetrievedChunk, rrf_fuse, weighted_fuse
from medevirag.index.paths import (
    BM25_DIR,
    CHUNK_META_PATH,
    INDEX_META_PATH,
    ROOT,
    load_retrieval_config,
)

__all__ = ["HybridRetriever", "make_chunk"]

LOGGER = logging.getLogger("medevirag.index.hybrid")


def make_chunk(
    chunk_id: str,
    meta: dict[str, Any] | None,
    *,
    dense_score: float = 0.0,
    bm25_score: float = 0.0,
    dense_rank: int = 0,
    bm25_rank: int = 0,
    source: str = "",
) -> RetrievedChunk:
    """把 chunk_meta 的一行变成 RetrievedChunk（meta 缺失时只留 chunk_id）。"""
    m = meta or {}
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=str(m.get("doc_id") or ""),
        text=str(m.get("text") or ""),
        context_header=str(m.get("context_header") or ""),
        chunk_type=str(m.get("chunk_type") or "paragraph"),
        doc_title=str(m.get("doc_title") or ""),
        doc_type=str(m.get("doc_type") or "guideline"),
        publish_year=int(m.get("publish_year") or 0),
        lang=str(m.get("lang") or "zh"),
        evidence_level=str(m.get("evidence_level") or "unknown"),
        rec_strength=str(m.get("rec_strength") or "unknown"),
        section_path=tuple(str(s) for s in (m.get("section_path") or [])),
        page_start=int(m.get("page_start") or 0),
        page_end=int(m.get("page_end") or 0),
        dense_score=float(dense_score),
        bm25_score=float(bm25_score),
        dense_rank=int(dense_rank),
        bm25_rank=int(bm25_rank),
        source=source,
    )


class HybridRetriever:
    """两路召回 + 融合 + 硬过滤。所有重量级对象都在这里惰性建一次。"""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        milvus_uri: str | None = None,
        load_embedder: bool = True,
    ) -> None:
        self.cfg = cfg or load_retrieval_config(milvus_uri=milvus_uri)
        self.milvus_uri = str(self.cfg.get("milvus_uri") or milvus_uri or "")
        self.collection = str(self.cfg["index"]["collection"])
        self.ef = int(self.cfg["index"]["hnsw"]["ef"])
        self.dense_topk = int(self.cfg["dense_topk"])
        self.bm25_topk = int(self.cfg["bm25_topk"])
        self.rrf_k = int(self.cfg["rrf_k"])
        self.topk = int(self.cfg.get("topk", 10))
        self.alpha = float(self.cfg.get("weighted_alpha", 0.5))
        self.rules = FilterRules.from_config(self.cfg)
        self._pool = ThreadPoolExecutor(max_workers=2)

        self.meta = ChunkMetaIndex.from_jsonl(CHUNK_META_PATH)
        self.embedder = (
            Embedder(
                ROOT / str(self.cfg.get("emb_model_dir") or "models/bge-m3"),
                device=str(self.cfg.get("emb_device") or "cuda"),
                batch_size=int(self.cfg.get("emb_batch_size") or 16),
                use_fp16=bool(self.cfg.get("emb_fp16", True)),
                cache=InMemoryLRUCache(),
            )
            if load_embedder
            else None
        )

        bm25_cfg = dict(self.cfg["bm25"])
        tokenizer = ChineseTokenizer(bm25_cfg.get("userdict"), bm25_cfg.get("stopwords"))
        self.bm25 = BM25Store(
            BM25_DIR,
            tokenizer,
            k1=float(bm25_cfg.get("k1", 1.5)),
            b=float(bm25_cfg.get("b", 0.75)),
            delta=float(bm25_cfg.get("delta", 0.5)),
            method=str(bm25_cfg.get("method", "lucene")),
        )
        self.bm25.load()

        if not INDEX_META_PATH.exists():
            raise FileNotFoundError(f"index meta not found: {INDEX_META_PATH} (run s2_vectorize.py first)")
        self.client = milvus_store.open_milvus_client(self.milvus_uri)
        if not self.client.has_collection(self.collection):
            raise RuntimeError(
                f"milvus collection '{self.collection}' not found at {self.milvus_uri} "
                "(run s2_build_index.py first)"
            )
        self.client.load_collection(self.collection)

    # ------------------------------------------------------------------ 生命周期
    def close(self) -> None:
        self._pool.shutdown(wait=False)
        try:
            self.client.close()
        except Exception:  # 关闭失败不影响结果，但不能让进程挂住
            LOGGER.debug("milvus client close failed", exc_info=True)

    def __enter__(self) -> HybridRetriever:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ 两路
    def _dense_run(self, query: str, topk: int | None = None, *, ef: int | None = None) -> list[RetrievedChunk]:
        """dense 单路召回（消融直接复用这个方法，不另写检索逻辑）。"""
        if self.embedder is None:
            raise RuntimeError("embedder 未加载：dense 路径不可用")
        vec = self.embedder.encode_one(query)
        hits = milvus_store.search_dense(
            self.client,
            vec,
            topk=int(topk or self.dense_topk),
            expr=None,
            collection=self.collection,
            ef=int(ef if ef is not None else self.ef),
        )
        out: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            out.append(
                make_chunk(
                    hit.chunk_id,
                    self.meta.get(hit.chunk_id),
                    dense_score=hit.score,
                    dense_rank=rank,
                    source="dense",
                )
            )
        return out

    def _bm25_run(self, query: str, topk: int | None = None) -> list[RetrievedChunk]:
        """BM25 单路召回。"""
        pairs = self.bm25.search(query, topk=int(topk or self.bm25_topk))
        out: list[RetrievedChunk] = []
        for rank, (cid, score) in enumerate(pairs, start=1):
            out.append(
                make_chunk(cid, self.meta.get(cid), bm25_score=score, bm25_rank=rank, source="bm25")
            )
        return out

    # ------------------------------------------------------------------ 对外
    def runs(self, query: str) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """并发跑两路，返回 (dense, bm25) 原始召回列表。"""
        f_dense = self._pool.submit(self._dense_run, query)
        f_bm25 = self._pool.submit(self._bm25_run, query)
        return f_dense.result(), f_bm25.result()

    def retrieve(
        self,
        query: str,
        topk: int | None = None,
        *,
        variant: str = "rrf",
        apply_filters: bool = True,
        ef: int | None = None,
        stats: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """端到端：两路并发 -> 融合 -> 硬过滤。

        variant 决定融合函数（rrf / weighted-sum / dense-only / bm25-only），
        消融四组共用这一条路径，保证对比公平。
        """
        limit = int(topk if topk is not None else self.topk)
        pool_size = max(limit, self.dense_topk)
        if variant == "dense-only":
            dense = self._dense_run(query, pool_size, ef=ef)
            fused = [
                dataclasses.replace(c, fused_score=c.dense_score, fused_rank=i)
                for i, c in enumerate(dense[:pool_size], start=1)
            ]
        elif variant == "bm25-only":
            bm25 = self._bm25_run(query, pool_size)
            fused = [
                dataclasses.replace(c, fused_score=c.bm25_score, fused_rank=i)
                for i, c in enumerate(bm25[:pool_size], start=1)
            ]
        else:
            dense, bm25 = self.runs(query) if ef is None else (
                self._dense_run(query, self.dense_topk, ef=ef),
                self._bm25_run(query, self.bm25_topk),
            )
            if variant == "weighted-sum":
                fused = weighted_fuse(dense, bm25, alpha=self.alpha, topk=pool_size)
            else:
                fused = rrf_fuse([dense, bm25], k=self.rrf_k, topk=pool_size)
        if apply_filters:
            fused = apply_hard_filters(fused, self.rules, stats=stats)
        return fused[:limit]

    def dense(self, query: str, topk: int | None = None) -> list[RetrievedChunk]:
        return self._dense_run(query, topk)

    def bm25_only(self, query: str, topk: int | None = None) -> list[RetrievedChunk]:
        return self._bm25_run(query, topk)
