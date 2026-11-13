"""T3：Milvus 存储层（Lite 与服务器版同一套代码，docs/03 §T3）。

两条实现路径只差一个 uri：
    Lite    -> uri 是本地文件路径（./data/indexes/milvus_lite.db）
    服务器  -> uri 以 http(s):// 开头

实测坑（pymilvus 2.6.3 + milvus-lite 3.2.2）：
* search 的过滤参数关键字是 `filter=`，不是 `expr=`；
* 主键要按字段名取 `h["chunk_id"]`，`h["id"]` 会 KeyError；
* create_collection 之后必须 `load_collection`，否则 search 报 released。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from medevirag.index.paths import DENSE_DIM

__all__ = [
    "DenseHit",
    "SCALAR_FIELDS",
    "build_filter_expr",
    "ensure_collection",
    "insert_chunks",
    "is_remote_uri",
    "open_milvus_client",
    "search_dense",
]

COLLECTION = "chunks"

# 标量字段：全部建 INVERTED 索引，支撑 T6 的 post-filter 与 T3 的 pre-filter 对照
SCALAR_FIELDS: dict[str, str] = {
    "doc_id": "VARCHAR",
    "doc_type": "VARCHAR",
    "lang": "VARCHAR",
    "evidence_level": "VARCHAR",
    "section_path": "VARCHAR",
    "publish_year": "INT32",
}

_VARCHAR_MAX = 512
_CHUNK_ID_MAX = 128
_EXPR_MAX_LEN = 4096
_SAFE_STR = re.compile(r"^[A-Za-z0-9_\-./#]+$")


@dataclass(frozen=True)
class DenseHit:
    """一条向量召回结果：只要 id 与分数，展示字段由 ChunkMetaIndex 提供。"""

    chunk_id: str
    score: float


def is_remote_uri(uri: str) -> bool:
    return uri.startswith(("http://", "https://", "tcp://"))


def open_milvus_client(uri: str) -> Any:
    """按 uri 分流：http(s) 走服务器版，其余按 Lite 本地文件处理。"""
    from pymilvus import MilvusClient

    if is_remote_uri(uri):
        return MilvusClient(uri=uri)
    return MilvusClient(uri=str(uri))


def ensure_collection(
    client: Any,
    collection: str = COLLECTION,
    *,
    drop: bool = False,
    metric: str = "IP",
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 200,
    dense_dim: int = DENSE_DIM,
) -> None:
    """建集合 + 向量索引 + 标量 INVERTED 索引，最后必须 load。"""
    from pymilvus import DataType

    if client.has_collection(collection):
        if not drop:
            client.load_collection(collection)
            return
        client.drop_collection(collection)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=_CHUNK_ID_MAX)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dense_dim)
    for name, ftype in SCALAR_FIELDS.items():
        if ftype == "INT32":
            schema.add_field(name, DataType.INT32)
        else:
            schema.add_field(name, DataType.VARCHAR, max_length=_VARCHAR_MAX)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_type="HNSW",
        metric_type=metric,
        params={"M": int(hnsw_m), "efConstruction": int(hnsw_ef_construction)},
    )
    for name in SCALAR_FIELDS:
        index_params.add_index(field_name=name, index_type="INVERTED")

    client.create_collection(collection_name=collection, schema=schema, index_params=index_params)
    client.load_collection(collection)


def insert_chunks(
    client: Any,
    chunk_ids: list[str],
    vectors: Any,
    metas: list[dict[str, Any]],
    *,
    collection: str = COLLECTION,
    batch: int = 2000,
) -> int:
    """分批插入并 flush；行字段顺序与 ensure_collection 的 schema 一致。"""
    total = 0
    for start in range(0, len(chunk_ids), batch):
        stop = min(start + batch, len(chunk_ids))
        rows: list[dict[str, Any]] = []
        for i in range(start, stop):
            meta = metas[i]
            vec = vectors[i]
            rows.append(
                {
                    "chunk_id": str(chunk_ids[i]),
                    "dense": [float(x) for x in vec],
                    "doc_id": str(meta.get("doc_id") or ""),
                    "doc_type": str(meta.get("doc_type") or ""),
                    "lang": str(meta.get("lang") or ""),
                    "evidence_level": str(meta.get("evidence_level") or "unknown"),
                    "section_path": " > ".join(str(s) for s in (meta.get("section_path") or [])),
                    "publish_year": int(meta.get("publish_year") or 0),
                }
            )
        client.insert(collection_name=collection, data=rows)
        total += len(rows)
    client.flush(collection)
    return total


def _q(value: str) -> str:
    """字符串字面量转义：安全字符集之外的统一走 json 双引号。"""
    import json

    if _SAFE_STR.match(value):
        return f'"{value}"'
    return json.dumps(value, ensure_ascii=False)


def build_filter_expr(
    *,
    doc_type_allow: list[str] | tuple[str, ...] | None = None,
    lang_allow: list[str] | tuple[str, ...] | None = None,
    publish_year_min: int | None = None,
    evidence_level_exclude: list[str] | tuple[str, ...] | None = None,
) -> str:
    """把过滤条件拼成 Milvus 表达式；空条件返回空串（表示不过滤）。"""
    parts: list[str] = []
    if doc_type_allow:
        parts.append("doc_type in [" + ", ".join(_q(t) for t in doc_type_allow) + "]")
    if lang_allow:
        parts.append("lang in [" + ", ".join(_q(t) for t in lang_allow) + "]")
    if publish_year_min is not None and int(publish_year_min) > 0:
        parts.append(f"publish_year >= {int(publish_year_min)}")
    if evidence_level_exclude:
        parts.append(
            "evidence_level not in [" + ", ".join(_q(t) for t in evidence_level_exclude) + "]"
        )
    expr = " and ".join(parts)
    if len(expr) > _EXPR_MAX_LEN:
        raise ValueError(f"filter expr too long ({len(expr)} > {_EXPR_MAX_LEN}): {expr[:120]}...")
    return expr


def search_dense(
    client: Any,
    vector: Any,
    *,
    topk: int = 50,
    expr: str | None = None,
    collection: str = COLLECTION,
    ef: int = 64,
) -> list[DenseHit]:
    """单条 query 的向量检索。ef 是 search 参数（不是建索引参数）。"""
    res = client.search(
        collection_name=collection,
        data=[[float(x) for x in vector]],
        limit=int(topk),
        filter=expr or "",
        output_fields=["chunk_id"],
        search_params={"metric_type": "IP", "params": {"ef": int(ef)}},
    )
    hits: list[DenseHit] = []
    for one in res[0] if res else []:
        hits.append(DenseHit(str(one["chunk_id"]), float(one["distance"])))
    return hits
