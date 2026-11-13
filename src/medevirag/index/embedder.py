"""T1：bge-m3 向量化封装（docs/03 §T1）。

三条纪律：
1. 权重只从本地 `models/bge-m3` 加载，不联网；
2. 输出永远是 float32 且 L2 范数为 1（.half() 之后必须 astype(float32) 回写）；
3. query 侧走缓存：命中不碰 GPU，`encode_queries` 只把 miss 送进模型。
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from medevirag.index.paths import DENSE_DIM

FloatArray = NDArray[np.float32]

__all__ = [
    "DENSE_DIM",
    "Embedder",
    "InMemoryLRUCache",
    "QueryEmbedCache",
    "normalize_text",
]


def normalize_text(text: str) -> str:
    """空白归一化：缓存 key 与入库文本必须用同一套归一化。"""
    return " ".join(text.strip().split())


@runtime_checkable
class QueryEmbedCache(Protocol):
    """query 向量缓存接口（生产可换 Redis/文件，只要满足这四个方法）。"""

    def get(self, key: str) -> FloatArray | None: ...

    def set(self, key: str, vec: FloatArray) -> None: ...

    def mget(self, keys: Sequence[str]) -> list[FloatArray | None]: ...

    def mset(self, items: Sequence[tuple[str, FloatArray]]) -> None: ...


class InMemoryLRUCache:
    """容量受限的 LRU，带 ttl 与命中率计数（A12 的确定性靠它不出错）。"""

    def __init__(self, capacity: int = 2048, ttl_s: float = 86400.0) -> None:
        self.capacity = int(capacity)
        self.ttl_s = float(ttl_s)
        self.hits = 0
        self.misses = 0
        self._store: OrderedDict[str, tuple[float, FloatArray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._store)

    def _clock(self) -> float:
        import time

        return time.time()

    def get(self, key: str) -> FloatArray | None:
        item = self._store.get(key)
        if item is None:
            self.misses += 1
            return None
        stamp, vec = item
        if self.ttl_s > 0 and self._clock() - stamp > self.ttl_s:
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return vec

    def set(self, key: str, vec: FloatArray) -> None:
        self._store[key] = (self._clock(), np.asarray(vec, dtype=np.float32))
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def mget(self, keys: Sequence[str]) -> list[FloatArray | None]:
        return [self.get(k) for k in keys]

    def mset(self, items: Sequence[tuple[str, FloatArray]]) -> None:
        for key, vec in items:
            self.set(key, vec)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class Embedder:
    """bge-m3 单例封装：`encode` 是纯函数，`encode_queries` 带缓存。"""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        batch_size: int = 32,
        use_fp16: bool = True,
        cache: QueryEmbedCache | None = None,
    ) -> None:
        # 延迟导入：torch/ST 只在真正构造模型时付出代价（unit test 会 monkeypatch 掉 ST）
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_dir = Path(model_dir)
        self.model_id = self.model_dir.name
        # 环境变量可强制降级设备：并发跑多个进程时每个进程都会建一份 CUDA 上下文，
        # Windows 下会直接 access violation（无 traceback、退出码非 0）。
        # 测试与离线 CI 用 MEDEVIRAG_DEVICE=cpu 把显存占用归零，生产默认仍跟随配置。
        _forced = (os.environ.get("MEDEVIRAG_DEVICE") or "").strip().lower()
        if _forced == "cpu":
            device = "cpu"
        elif _forced == "cuda" and torch.cuda.is_available():
            device = "cuda"
        self.device = device
        self.batch_size = int(batch_size)
        self.use_fp16 = bool(use_fp16)
        self.cache = cache
        self.fp16_active = self.use_fp16 and str(device).startswith("cuda")
        # low_cpu_mem_usage=True 必须：默认加载路径会先把整份 fp32 state_dict 读进内存并逐层
        # full-materialize 参数，本机提交内存吃紧时直接 access violation（exit 139，无 traceback），
        # 见 docs/03 §T1 与返工记录。torch_dtype 直接按目标精度加载，省掉一次 fp32 全量常驻。
        self._model: Any = SentenceTransformer(
            str(self.model_dir),
            device=device,
            model_kwargs={
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16 if self.fp16_active else torch.float32,
            },
        )
        # bge-m3 官方上限 8192；这里只是安全冗余，切分粒度仍由 S1 的 480 token 决定
        self._model.max_seq_length = 8192
        if self.fp16_active:
            self._model.half()

    # ------------------------------------------------------------------ 内部
    def _key(self, text: str) -> str:
        return _md5("emb|" + self.model_id + "|" + normalize_text(text))

    def _raw_encode(self, texts: list[str]) -> FloatArray:
        out = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        arr = np.asarray(out, dtype=np.float32)  # .half() 输出 float16，必须在这里回 float32
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms < 1e-6):
            raise ValueError("embedder produced a zero vector; input may be empty string")
        arr = arr / norms
        return np.ascontiguousarray(arr, dtype=np.float32)

    # ------------------------------------------------------------------ 公开
    def encode(self, texts: Sequence[str]) -> FloatArray:
        """纯函数式编码：不做缓存（建索引用它，保证幂等）。"""
        if not texts:
            return np.zeros((0, DENSE_DIM), dtype=np.float32)
        return self._raw_encode([normalize_text(t) for t in texts])

    def encode_queries(self, queries: Sequence[str]) -> FloatArray:
        """只把缓存 miss 的 query 送 GPU。"""
        if not queries:
            return np.zeros((0, DENSE_DIM), dtype=np.float32)
        keys = [self._key(q) for q in queries]
        out = np.zeros((len(queries), DENSE_DIM), dtype=np.float32)
        miss_idx: list[int]
        if self.cache is None:
            miss_idx = list(range(len(queries)))
        else:
            cached = self.cache.mget(keys)
            miss_idx = []
            for i, vec in enumerate(cached):
                if vec is None:
                    miss_idx.append(i)
                else:
                    out[i] = vec
        if miss_idx:
            fresh = self._raw_encode([normalize_text(queries[i]) for i in miss_idx])
            assert fresh.shape == (len(miss_idx), DENSE_DIM), fresh.shape
            out[miss_idx] = fresh
            if self.cache is not None:
                self.cache.mset([(keys[i], fresh[j]) for j, i in enumerate(miss_idx)])
        return out

    def encode_one(self, text: str) -> FloatArray:
        vec = self.encode_queries([text])[0]
        return np.asarray(vec, dtype=np.float32)
