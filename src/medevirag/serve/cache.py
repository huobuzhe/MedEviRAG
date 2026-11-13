# S6 T4：两级缓存（docs/07 §T4）。
# 本机无 Redis：_MemCache 实现同样的 get/setex/语义，接口与 aioredis 对齐；
# 部署到有 Redis 的环境时自动切换（serve_fallback.redis_url 非空即用真 Redis）。

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any


def _emb_key(model_id: str, text: str) -> str:
    return "ck_emb:" + hashlib.md5(("emb|" + model_id + "|" + text).encode("utf-8")).hexdigest()


def _norm(q: str) -> str:
    # 折叠空白/大小写；不做同义改写（会误命中）
    return " ".join(q.strip().lower().split())


def _ans_key(question: str, config_hash: str, filters: dict[str, Any] | None) -> str:
    fsig = json.dumps(filters or {}, sort_keys=True, ensure_ascii=False)
    raw = "ans|" + _norm(question) + "|" + config_hash + "|" + fsig
    return "ck_ans:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


class _MemCache:
    # 进程内 dict + TTL（无 Redis 降级）。容量上限防膨胀。

    def __init__(self, max_items: int = 100_000) -> None:
        self._d: dict[str, tuple[float, bytes]] = {}
        self._max = max_items

    async def get(self, key: str) -> bytes | None:
        item = self._d.get(key)
        if item is None:
            return None
        expires, val = item
        if expires and expires < time.time():
            self._d.pop(key, None)
            return None
        return val

    async def setex(self, key: str, ttl_s: int, value: bytes) -> None:
        if len(self._d) >= self._max:
            self._d.pop(next(iter(self._d)), None)   # FIFO 淘汰
        self._d[key] = (time.time() + ttl_s, value)

    async def aclose(self) -> None:
        self._d.clear()


@dataclass
class CacheStats:
    emb_hit: int = 0
    emb_miss: int = 0
    ans_hit: int = 0
    ans_miss: int = 0

    def snapshot(self) -> dict[str, float]:
        emb_total = self.emb_hit + self.emb_miss
        ans_total = self.ans_hit + self.ans_miss
        return {
            "emb_hit_rate": round(self.emb_hit / emb_total, 4) if emb_total else 0.0,
            "ans_hit_rate": round(self.ans_hit / ans_total, 4) if ans_total else 0.0,
            "emb_requests": emb_total, "ans_requests": ans_total,
        }


class TwoLevelCache:
    # 答案必须整体缓存 answer+citations+status+confidence+disclaimer+refusal_reason，
    # 拆开缓存会造成"答案命中、引用过期"的回链断裂（docs §T4 第一坑）。

    def __init__(self, backend: Any, embedding_ttl_s: int, answer_ttl_s: int,
                 config_hash: str) -> None:
        self.r = backend
        self.emb_ttl = embedding_ttl_s
        self.ans_ttl = answer_ttl_s
        self.config_hash = config_hash
        self.stats = CacheStats()

    async def get_answer(self, question: str, filters: dict[str, Any] | None) -> dict[str, Any] | None:
        raw = await self.r.get(_ans_key(question, self.config_hash, filters))
        if raw is None:
            self.stats.ans_miss += 1
            return None
        self.stats.ans_hit += 1
        return dict(json.loads(raw))

    async def set_answer(self, question: str, filters: dict[str, Any] | None, payload: dict[str, Any]) -> None:
        # 空答案（refused）也缓存：拒答是确定性输出，能把必拒答压到 <50ms；
        # TTL 减半，避免规则表更新后继续拒答（docs §T4）。
        ttl = self.ans_ttl if payload.get("status") != "refused" else min(300, self.ans_ttl)
        await self.r.setex(_ans_key(question, self.config_hash, filters),
                           ttl,
                           json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    async def get_embedding(self, model_id: str, text: str) -> list[float] | None:
        raw = await self.r.get(_emb_key(model_id, text))
        if raw is None:
            self.stats.emb_miss += 1
            return None
        self.stats.emb_hit += 1
        return list(json.loads(raw))

    async def set_embedding(self, model_id: str, text: str, vec: Any) -> None:
        # 6 位小数足够检索用，体积降约三成
        await self.r.setex(_emb_key(model_id, text), self.emb_ttl,
                           json.dumps([round(float(x), 6) for x in vec]).encode("utf-8"))
