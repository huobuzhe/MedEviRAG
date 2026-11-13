# S6 T6：并发与时延工程（docs/07 §T6）。
# 三重闸门 + 一次降级：微批 / GPU 单飞信号量 / 等待超时降级跳重排。

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RerankOutcome:
    order: list[int]
    skipped: bool
    waited_ms: int
    batch_size: int


class RerankGate:
    """GPU 重排闸门：单飞 + 微批 + 超时降级。

    注意（docs §T6 坑 4）：攒批窗口 20ms 在单并发时是纯损失（P50 +20ms），
    但 20 并发时省下数秒 P95——取舍写进配置注释，不当魔数。
    """

    def __init__(self, reranker: Any, batch_size: int = 8, max_wait_ms: int = 400) -> None:
        self._rr = reranker
        self._bs = batch_size
        self._max_wait = max_wait_ms / 1000.0
        self._lock = asyncio.Lock()
        self._queue: list[tuple[asyncio.Future[list[int]], str, list[Any]]] = []
        self._worker: asyncio.Task[None] | None = None

    async def rerank(self, query: str, candidates: list[Any]) -> RerankOutcome:
        t0 = time.perf_counter()
        fut: asyncio.Future[list[int]] = asyncio.get_running_loop().create_future()
        self._queue.append((fut, query, candidates))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._drain())
        try:
            order = await asyncio.wait_for(fut, timeout=self._max_wait + 0.6)
            return RerankOutcome(order=order, skipped=False,
                                 waited_ms=int((time.perf_counter() - t0) * 1000),
                                 batch_size=self._bs)
        except asyncio.TimeoutError:
            # 排队超预算：跳过重排，用 RRF 融合顺序兜底（质量略降，P95 保住）
            return RerankOutcome(order=list(range(len(candidates))), skipped=True,
                                 waited_ms=int((time.perf_counter() - t0) * 1000),
                                 batch_size=len(candidates))

    async def _drain(self) -> None:
        async with self._lock:      # GPU 上串行：一次只跑一个前向
            await asyncio.sleep(0.02)   # 20ms 攒批窗口
            batch, self._queue = self._queue[: self._bs], self._queue[self._bs:]
            if not batch:
                return
            try:
                results = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: [self._rr.rerank(q, c, len(c))
                                   for _, q, c in batch])
                for (fut, _, _), order in zip(batch, results, strict=False):
                    if not fut.done():
                        fut.set_result(order)
            except Exception as exc:
                for fut, _, _ in batch:
                    if not fut.done():
                        fut.set_exception(exc)
            if self._queue:
                self._worker = asyncio.create_task(self._drain())


class InflightGate:
    """进程内并发上限：超出直接 503，不排队堆积（docs §T6 思路 3）。"""

    def __init__(self, max_inflight: int = 32) -> None:
        self._sem = asyncio.Semaphore(max_inflight)

    async def acquire(self) -> bool:
        return self._sem.locked() is False or not self._sem._value <= 0
