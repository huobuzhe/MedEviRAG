# S6 T5：审计日志（docs/07 §T5）。
# 合规要求可回溯每一次回答；无 PG 时降级写 data/logs/qa_log.jsonl，写失败不拖垮响应。

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("medevirag.serve.audit")


@dataclass
class QaLogRecord:
    trace_id: str
    question: str
    answer: str
    status: str
    confidence: float
    refusal_reason: str | None
    latency_ms: int
    retrieval_ms: int
    rerank_ms: int
    generation_ms: int
    retrieved_chunk_ids: list[str]
    config_hash: str
    cache: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return d


class FileAuditSink:
    # 无 PG 降级：追加写 jsonl。写失败只记日志（审计不许 500）。

    def __init__(self, audit_dir: str) -> None:
        self.path = Path(audit_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = (self.path / "qa_log.jsonl").open("a", encoding="utf-8")

    async def write(self, rec: QaLogRecord) -> None:
        # 同步小写入直接做，jsonl 追加一行 <1ms
        self.file.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        self.file.flush()

    async def close(self) -> None:
        self.file.close()


class PgAuditSink:
    # 生产路径：INSERT qa_log（docs/00 §6.2 表结构）

    _INSERT = """
    INSERT INTO qa_log (trace_id, question, answer, status, confidence, refusal_reason,
                        latency_ms, retrieval_ms, rerank_ms, generation_ms,
                        retrieved_chunk_ids, created_at)
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb, now())
    """

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    async def write(self, rec: QaLogRecord) -> None:
        await self.pool.execute(
            self._INSERT, rec.trace_id, rec.question, rec.answer, rec.status,
            rec.confidence, rec.refusal_reason, rec.latency_ms,
            rec.retrieval_ms, rec.rerank_ms, rec.generation_ms,
            json.dumps(rec.retrieved_chunk_ids[:200], ensure_ascii=False))

    async def close(self) -> None:
        await self.pool.close()


_pending: set[asyncio.Task[None]] = set()   # 防 GC 回收导致日志丢失（docs §T5 坑 1）


def fire_and_forget(sink: Any, rec: QaLogRecord) -> None:
    async def _w() -> None:
        try:
            await sink.write(rec)
        except Exception:
            log.exception("qa_log write failed", extra={"trace_id": rec.trace_id})

    task = asyncio.create_task(_w())
    _pending.add(task)
    task.add_done_callback(_pending.discard)
