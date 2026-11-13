#!/usr/bin/env python
"""T2：全量向量化，产出 chunks_emb.npy / chunk_ids.jsonl / chunk_meta.jsonl / meta.json。

用法：
    cd F:/MedEviRAG && PYTHONPATH=src ./.venv/Scripts/python.exe -X utf8 scripts/s2_vectorize.py

三条纪律：
1. 按 chunk_id 升序固化顺序（chunk_ids.jsonl 是唯一顺序来源，A2 靠它对齐三处 N）；
2. 相同 content_hash 只编码一次（uniq 去重），其余行按 hash 回填；
3. 先 open_memmap 预分配再写，写完必须 flush —— 否则后续 np.load 读到全 0。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medevirag.index.chunk_meta import write_chunk_meta  # noqa: E402
from medevirag.index.embedder import DENSE_DIM, Embedder  # noqa: E402
from medevirag.index.paths import (  # noqa: E402
    CHUNK_IDS_PATH,
    CHUNK_META_PATH,
    CHUNKS_PATH,
    EMB_PATH,
    INDEX_META_PATH,
    ensure_dirs,
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_chunks(path: Path) -> list[dict]:
    """逐行解析（按 \n 切字节，行内换行是转义过的，不能用 splitlines）。"""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: str(r["chunk_id"]))
    for row in rows:
        if not str(row.get("text") or "").strip():
            raise ValueError(f"empty text in chunk {row.get('chunk_id')}")
    return rows


def text_key(row: dict) -> str:
    return str(row.get("content_hash") or hashlib.sha256(str(row["text"]).encode()).hexdigest())


def main() -> int:
    ensure_dirs()
    rows = load_chunks(CHUNKS_PATH)
    n = len(rows)
    print(f"chunks={n} from {CHUNKS_PATH}")

    chunk_ids = [str(r["chunk_id"]) for r in rows]
    uniq: dict[str, str] = {}
    for row in rows:
        key = text_key(row)
        if key not in uniq:
            uniq[key] = str(row["text"])
    keys = list(uniq)
    print(f"unique_texts={len(keys)} (dedup ratio {(1 - len(keys) / n) * 100:.2f}%)")

    from medevirag.common.config import get_settings

    cfg = get_settings()
    model_dir = ROOT / cfg.embedding.model_path
    embedder = Embedder(
        model_dir,
        device=cfg.embedding.device,
        batch_size=cfg.embedding.batch_size,
        use_fp16=True,
    )
    print(
        f"embedder ready: model={model_dir.name} device={cfg.embedding.device} "
        f"fp16={embedder.fp16_active} batch={embedder.batch_size}"
    )

    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(str(EMB_PATH), mode="w+", dtype=np.float32, shape=(n, DENSE_DIM))
    vec_by_hash: dict[str, np.ndarray] = {}
    step = 4096
    for start in range(0, len(keys), step):
        batch_keys = keys[start : start + step]
        vecs = embedder.encode([uniq[k] for k in batch_keys])
        for i, key in enumerate(batch_keys):
            vec_by_hash[key] = vecs[i]
        print(f"  encoded {min(start + step, len(keys))}/{len(keys)}", flush=True)

    for idx, row in enumerate(rows):
        matrix[idx] = vec_by_hash[text_key(row)]
    matrix.flush()
    del matrix

    check = np.load(str(EMB_PATH), mmap_mode="r")
    arr = np.asarray(check, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1)
    worst = float(np.max(np.abs(norms - 1.0)))
    print(f"emb shape={check.shape} dtype={check.dtype} max|norm-1|={worst:.3e}")
    assert check.shape == (n, DENSE_DIM), check.shape
    assert worst <= 1e-5, f"norm check failed: {worst}"

    with CHUNK_IDS_PATH.open("w", encoding="utf-8") as fh:
        for cid in chunk_ids:
            fh.write(json.dumps({"chunk_id": cid}, ensure_ascii=False) + "\n")
    write_chunk_meta(CHUNK_META_PATH, rows)

    meta = {
        "N": n,
        "n_unique_texts": len(keys),
        "dense_dim": DENSE_DIM,
        "model_id": model_dir.name,
        "max_token_len": max(int(r.get("token_len") or 0) for r in rows),
        "chunks_sha256": sha256_file(CHUNKS_PATH),
        "emb_sha256": sha256_file(EMB_PATH),
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    INDEX_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("meta -> " + str(INDEX_META_PATH))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
