"""S2 索引层路径常量与 retrieval 配置读取（docs/03 §T2/T5）。

docs/03 §T5 的骨架假设 `medevirag.common.paths.INDEX_DIR` 与
`medevirag.common.config.load_retrieval_config` 已存在，但 S0 冻结的 common/ 里没有
（且本阶段禁改 common/*），所以这里自建唯一一份：路径常量 + retrieval.yaml 读取 +
扁平化后的配置字典（骨架里用的 `cfg["dense_topk"]` 这类扁平 key）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
EVAL_DIR = DATA_DIR / "eval"
REPORTS_DIR = DATA_DIR / "reports"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
CHUNK_IDS_PATH = INDEX_DIR / "chunk_ids.jsonl"
CHUNK_META_PATH = INDEX_DIR / "chunk_meta.jsonl"
EMB_PATH = INDEX_DIR / "chunks_emb.npy"
INDEX_META_PATH = INDEX_DIR / "meta.json"
BM25_DIR = INDEX_DIR / "bm25"
RETRIEVAL_CFG_PATH = ROOT / "configs" / "retrieval.yaml"

DENSE_DIM = 1024
DEFAULT_MILVUS_URI = "./data/indexes/milvus_lite.db"

INDEX_DEFAULTS: dict[str, Any] = {
    "collection": "chunks",
    "dense_dim": DENSE_DIM,
    "metric": "IP",
    "hnsw": {"M": 16, "efConstruction": 200, "ef": 64},
}

BM25_DEFAULTS: dict[str, Any] = {
    "k1": 1.5,
    "b": 0.75,
    "delta": 0.5,
    "method": "lucene",
    "tokenizer": "jieba",
    "userdict": "data/processed/medical_terms.txt",
    "stopwords": "data/processed/stopwords_zh.txt",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return dict(data)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """一层深合并：嵌套 dict 合并，标量整体覆盖。"""
    out = dict(base)
    for key, val in over.items():
        old = out.get(key)
        if isinstance(old, dict) and isinstance(val, dict):
            out[key] = _merge(dict(old), dict(val))
        else:
            out[key] = val
    return out


def _embedding_section() -> dict[str, Any]:
    """读 S0 冻结的 embedding 配置（models.yaml），不在本阶段硬编码模型路径。"""
    from medevirag.common.config import get_settings

    emb = get_settings().embedding
    return {
        "emb_model_dir": str(emb.model_path),
        "emb_device": str(emb.device),
        "emb_batch_size": int(emb.batch_size),
        "emb_dim": int(emb.dim),
        "emb_max_length": int(emb.max_length),
    }


def load_retrieval_config(path: Path | None = None, milvus_uri: str | None = None) -> dict[str, Any]:
    """读 configs/retrieval.yaml，产出 `HybridRetriever` 用的扁平配置。

    保留 S0 已有的 `retrieval:` / `filters:` 段（不改四个冻结数），
    追加顶层 `index:` / `bm25:` 两段；本函数负责把 `retrieval.dense_topk`、
    `retrieval.sparse_topk` 摊平到 `dense_topk` / `bm25_topk`。
    """
    cfg_path = path or RETRIEVAL_CFG_PATH
    raw = _read_yaml(cfg_path)
    retrieval = dict(raw.get("retrieval") or {})
    filters = dict(raw.get("filters") or {})

    dense_topk = int(retrieval.get("dense_topk", 50))
    bm25_topk = int(retrieval.get("sparse_topk", 50))
    topk = int(retrieval.get("topk", 10))
    rrf_k = int(retrieval.get("rrf_k", 60))
    fusion = str(retrieval.get("fusion", "rrf"))
    weighted_alpha = float(retrieval.get("weighted_alpha", 0.5))

    out: dict[str, Any] = {
        "topk": topk,
        "dense_topk": dense_topk,
        "bm25_topk": bm25_topk,
        "sparse_topk": bm25_topk,
        "rrf_k": rrf_k,
        "fusion": fusion,
        "weighted_alpha": weighted_alpha,
        "filters": filters,
        "index": _merge(INDEX_DEFAULTS, dict(raw.get("index") or {})),
        "bm25": _merge(BM25_DEFAULTS, dict(raw.get("bm25") or {})),
        "milvus_uri": milvus_uri or str(raw.get("milvus_uri") or DEFAULT_MILVUS_URI),
        # S3 追加：重排开关（docs/04 §T2）。缺省关闭 = R2 否决分支。
        "rerank": {"enabled": False, "top_k": 20, "top_n": 8,
                   **dict(raw.get("rerank") or {})},
    }
    out.update(_embedding_section())
    return out


def resolve(p: str | Path) -> Path:
    """相对路径按项目根解析。"""
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def ensure_dirs() -> None:
    for d in (INDEX_DIR, EVAL_DIR, REPORTS_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)
