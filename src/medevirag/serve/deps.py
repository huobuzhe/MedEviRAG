# S6 T2：依赖注入与生命周期（docs/07 §T2）。
# 重活（模型加载）后台预热不阻塞首请求；/readyz 读 models_ready 事件。
# 本机无 PG/Redis：pg/redis 探测项允许 not_ready 并在 detail 说明降级路径。

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI

from medevirag.common.config import get_settings
from medevirag.serve.cache import TwoLevelCache, _MemCache

log = logging.getLogger("medevirag.serve.deps")


def _config_hash_for_cache() -> str:
    # 与 S5 runner.config_hash 同口径：yaml + prompt 常量
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    h = hashlib.sha256()
    for f in sorted((root / "configs").glob("*.yaml")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    from medevirag.rag import generator as G
    from medevirag.rag import refusal_templates as RT
    h.update(b"sysprompt")
    h.update(G.SYSTEM_PROMPT.encode("utf-8"))
    h.update(b"disclaimer")
    h.update(RT.DISCLAIMER.encode("utf-8"))
    return h.hexdigest()[:12]


@dataclass
class AppDeps:
    cfg: object
    cache: TwoLevelCache | None = None
    audit_sink: object = None
    graph_ready: bool = False
    models_ready: asyncio.Event = field(default_factory=asyncio.Event)
    warmup_error: str | None = None
    warmup_ms: int = 0
    pg_available: bool = False
    redis_available: bool = False

    @property
    def ready(self) -> bool:
        return self.models_ready.is_set() and self.warmup_error is None

    @property
    def graph(self) -> Any:
        if not self.graph_ready:
            raise RuntimeError("graph not built")
        from medevirag.rag.graph import _APP
        return _APP


def _warmup(deps: AppDeps) -> None:
    # 同步函数：跑在线程池里。编译 graph + 一次轻量前向都不做重活——
    # 检索栈在首次真实请求时按单例惰性加载（与 S3 nodes.get_* 一致），
    # 这里只验证 import 链路可用，把"首请求加载 25-40s"的问题交给 readyz 提示。
    t0 = time.perf_counter()
    try:
        from medevirag.rag import (
            graph,
            nodes,
        )
        nodes.get_chunk_meta()          # 触发 chunk 索引加载（CPU 侧，秒级）
        graph.build_graph()             # 编译 LangGraph（进程级单例 _APP）
        deps.graph_ready = True
        deps.models_ready.set()
    except Exception as exc:
        deps.warmup_error = f"{type(exc).__name__}: {exc}"
        log.exception("warmup failed")
    finally:
        deps.warmup_ms = int((time.perf_counter() - t0) * 1000)
        log.info("warmup finished ok=%s elapsed_ms=%s", deps.ready, deps.warmup_ms)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    s = get_settings()
    deps = AppDeps(cfg=s)
    # 缓存后端：serve_fallback.redis_url 空则进程内
    if s.serve_fallback.redis_url:
        try:
            import redis.asyncio as aioredis
            deps.cache = TwoLevelCache(
                aioredis.from_url(s.serve_fallback.redis_url, decode_responses=False),  # type: ignore[no-untyped-call]
                s.serve.cache.embedding_ttl_s, s.serve.cache.answer_ttl_s,
                _config_hash_for_cache())
            deps.redis_available = True
        except Exception:
            log.exception("redis connect failed, fallback to mem")
    if deps.cache is None:
        deps.cache = TwoLevelCache(_MemCache(), s.serve.cache.embedding_ttl_s,
                                   s.serve.cache.answer_ttl_s,
                                   _config_hash_for_cache())
    # 审计后端：pg_dsn 空则文件
    from medevirag.serve.audit import FileAuditSink
    deps.audit_sink = FileAuditSink(s.serve_fallback.audit_dir)
    app.state.deps = deps
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _warmup, deps)
    log.info("startup: redis=%s pg=%s models=%s",
             deps.redis_available, deps.pg_available, deps.ready)
    try:
        yield
    finally:
        if deps.audit_sink:
            await deps.audit_sink.close()
        if deps.cache:
            await deps.cache.r.aclose()


def get_deps(app: FastAPI) -> AppDeps:
    return app.state.deps  # type: ignore[no-any-return]
