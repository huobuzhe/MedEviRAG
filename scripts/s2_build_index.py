#!/usr/bin/env python
"""T3+T4：把向量装进 Milvus，把正文装进 BM25 倒排。

用法：
    cd F:/MedEviRAG && PYTHONPATH=src ./.venv/Scripts/python.exe -X utf8 scripts/s2_build_index.py
    # 服务器版：MILVUS_URI=http://127.0.0.1:19530 ... scripts/s2_build_index.py --skip-bm25

产物：
    data/indexes/milvus_lite.db/      Milvus Lite 库（uri 是本地文件）
    data/indexes/bm25/                bm25s 倒排 + chunk_ids.jsonl + meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medevirag.index import milvus_store  # noqa: E402
from medevirag.index.bm25_store import BM25Store, ChineseTokenizer  # noqa: E402
from medevirag.index.chunk_meta import ChunkMetaIndex  # noqa: E402
from medevirag.index.paths import (  # noqa: E402
    BM25_DIR,
    CHUNK_IDS_PATH,
    CHUNK_META_PATH,
    EMB_PATH,
    INDEX_META_PATH,
    load_retrieval_config,
)


def load_chunk_ids() -> list[str]:
    ids: list[str] = []
    with CHUNK_IDS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ids.append(str(json.loads(line)["chunk_id"]))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milvus-uri", default="")
    parser.add_argument("--skip-bm25", action="store_true", help="只建 Milvus（服务器版对照时用）")
    parser.add_argument("--skip-milvus", action="store_true", help="只建 BM25")
    parser.add_argument("--drop", action="store_true", help="先删集合再重建")
    args = parser.parse_args()

    for p in (CHUNK_IDS_PATH, EMB_PATH, CHUNK_META_PATH, INDEX_META_PATH):
        if not p.exists():
            print(f"FAIL: missing {p} (run scripts/s2_vectorize.py first)")
            return 1

    cfg = load_retrieval_config(milvus_uri=args.milvus_uri or None)
    uri = os.environ.get("MILVUS_URI") or str(cfg["milvus_uri"])
    chunk_ids = load_chunk_ids()
    meta = ChunkMetaIndex.from_jsonl(CHUNK_META_PATH)
    index_meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
    n = len(chunk_ids)
    print(f"N={n} meta_index={len(meta)} uri={uri}")
    if index_meta["N"] != n:
        print(f"FAIL: meta.json N={index_meta['N']} != chunk_ids {n}")
        return 1

    if not args.skip_milvus:
        vecs = np.load(EMB_PATH, mmap_mode="r")
        client = milvus_store.open_milvus_client(uri)
        t0 = time.perf_counter()
        milvus_store.ensure_collection(
            client,
            cfg["index"]["collection"],
            drop=args.drop,
            metric=str(cfg["index"]["metric"]),
            hnsw_m=int(cfg["index"]["hnsw"]["M"]),
            hnsw_ef_construction=int(cfg["index"]["hnsw"]["efConstruction"]),
            dense_dim=int(cfg["index"]["dense_dim"]),
        )
        print(f"collection ready in {time.perf_counter() - t0:.2f}s "
              f"(M={cfg['index']['hnsw']['M']} efConstruction={cfg['index']['hnsw']['efConstruction']} "
              f"metric={cfg['index']['metric']})")
        metas = [meta.get(cid) or {} for cid in chunk_ids]
        t1 = time.perf_counter()
        inserted = milvus_store.insert_chunks(
            client, chunk_ids, vecs, metas, collection=cfg["index"]["collection"]
        )
        client.flush(cfg["index"]["collection"])
        print(f"inserted {inserted} rows in {time.perf_counter() - t1:.2f}s; "
              f"num_entities={client.get_collection_stats(cfg['index']['collection'])}")
        idx = client.describe_index(collection_name=cfg["index"]["collection"], index_name="dense")
        print("dense index: " + json.dumps(idx, ensure_ascii=False, default=str))
        client.close()

    if not args.skip_bm25:
        bm25_cfg = dict(cfg["bm25"])
        tokenizer = ChineseTokenizer(bm25_cfg.get("userdict"), bm25_cfg.get("stopwords"))
        store = BM25Store(
            BM25_DIR,
            tokenizer,
            k1=float(bm25_cfg["k1"]),
            b=float(bm25_cfg["b"]),
            delta=float(bm25_cfg["delta"]),
            method=str(bm25_cfg["method"]),
        )
        texts = [(meta.get(cid) or {}).get("text") or "" for cid in chunk_ids]
        t2 = time.perf_counter()
        store.build(chunk_ids, texts)
        store.save()
        print(f"bm25 built in {time.perf_counter() - t2:.2f}s; saved -> {BM25_DIR}")
        print(json.dumps(json.loads((BM25_DIR / "meta.json").read_text(encoding="utf-8")),
                         ensure_ascii=False))
        sample = "基层高血压防治管理指南推荐的降压目标是多少"
        print("bm25 sample tokens: " + str(tokenizer(sample)))
        print("bm25 sample top3: " + str(store.search(sample, topk=3)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
