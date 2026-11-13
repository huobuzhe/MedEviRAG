"""S2 索引层：embedder / Milvus / BM25 / RRF 融合 / 硬过滤。

模块职责一览：
    paths        路径常量与 retrieval.yaml 读取（common/ 冻结，故自建）
    embedder     bge-m3 封装 + query 缓存
    chunk_meta   展示字段快照
    milvus_store Milvus Lite/服务器双实现
    bm25_store   jieba 分词 + bm25s 倒排
    fusion       RRF / 加权和
    filters      元数据硬过滤（post-filter，ADR-03）
    hybrid       两路并发召回 + 融合 + 过滤的编排
"""

from __future__ import annotations

__all__ = ["paths"]
