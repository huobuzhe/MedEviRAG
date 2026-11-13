# S6 T3：五条路由 + app 装配（docs/07 §T3）。
# /healthz 不碰依赖；/readyz 四项并发探测；/v1/ask 走 S3-S4 图；错误码稳定枚举。

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi import status as http_status

from medevirag.serve.schemas import (
    AskRequest,
    AskResponse,
    CheckResult,
    CitationDetail,
    CitationOut,
    ErrorBody,
    FeedbackRequest,
    ReadyReport,
    TraceOut,
)

router = APIRouter()


async def _check(name: str, coro: Any, timeout: float = 2.0) -> CheckResult:
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(coro, timeout)
        return CheckResult(ok=True, ms=int((time.perf_counter() - t0) * 1000))
    except asyncio.TimeoutError:
        return CheckResult(ok=False, ms=int(timeout * 1000),
                           detail=f"{name} timeout >{timeout}s")
    except Exception as exc:
        return CheckResult(ok=False, ms=int((time.perf_counter() - t0) * 1000),
                           detail=f"{type(exc).__name__}: {exc}"[:200])


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    d = request.app.state.deps
    body = {"status": "ok"}
    if d is not None and d.cache is not None:
        body["cache_stats"] = d.cache.stats.snapshot()
    return body


@router.get("/readyz", response_model=ReadyReport,
            responses={503: {"model": ReadyReport}})
async def readyz(request: Request, response: Response) -> ReadyReport:
    d = request.app.state.deps

    async def _models() -> None:
        if not d.models_ready.is_set():
            raise RuntimeError(d.warmup_error or "warmup in progress")

    async def _pg() -> None:
        # 无 PG（本机降级路径）：审计走文件，报 ok=False 但 detail 说明降级
        if not d.pg_available:
            raise RuntimeError("pg not configured (audit -> file sink)")

    async def _redis() -> None:
        if not d.redis_available:
            raise RuntimeError("redis not configured (mem cache fallback)")

    async def _milvus() -> None:
        # Milvus Lite 本地文件，同步探测丢线程池
        def _h() -> None:
            from medevirag.index.paths import INDEX_META_PATH
            if not INDEX_META_PATH.exists():
                raise RuntimeError("milvus lite index not built")
        await asyncio.get_running_loop().run_in_executor(None, _h)

    pairs = {"postgres": _pg(), "redis": _redis(), "milvus": _milvus(),
             "models": _models()}
    results = dict(zip(pairs.keys(),
                       await asyncio.gather(*(_check(k, c) for k, c in pairs.items()),
                                            return_exceptions=False), strict=False))
    failed = [k for k, v in results.items() if not v.ok]
    if failed:
        response.status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyReport(status="not_ready" if failed else "ready",
                       checks=results, failed=failed)


def _coerce_citation(c: dict[str, Any]) -> CitationOut:
    return CitationOut(
        chunk_id=str(c.get("chunk_id", "")),
        doc_id=str(c.get("doc_id", "")),
        doc_title=str(c.get("doc_title", "")),
        section_path=[str(x) for x in (c.get("section_path") or [])],
        page=int(c.get("page", 0) or 0),
        quote=str(c.get("quote", ""))[:300],
        url=c.get("url"),
        evidence_level=str(c.get("evidence_level", "unknown")),
    )


@router.post("/v1/ask", response_model=AskResponse,
             responses={400: {"model": ErrorBody}, 503: {"model": ErrorBody},
                        504: {"model": ErrorBody}, 502: {"model": ErrorBody}})
async def ask(req: AskRequest, request: Request) -> AskResponse:
    d = request.app.state.deps
    trace_id = getattr(request.state, "trace_id", uuid.uuid4().hex[:16])
    t0 = time.perf_counter()
    if not d.ready:
        raise HTTPException(503, detail={
            "code": "BACKEND_UNAVAILABLE",
            "message": d.warmup_error or "models not warmed up"})

    filters = dict(req.filters) if req.filters else None
    cached = None
    if req.use_cache and d.cache is not None:
        cached = await d.cache.get_answer(req.question, filters)
    if cached is not None:
        # 缓存体已含完整 trace（写入时固定 cache="hit"），只更新 trace_id 与 total_ms
        body = dict(cached)
        tr = dict(body.get("trace") or {})
        tr["trace_id"] = trace_id
        tr["total_ms"] = int((time.perf_counter() - t0) * 1000)
        body["trace"] = tr
        return AskResponse(**body)

    # 走 S3-S4 图（进程级单例；节点内部再用模块级单例）
    from medevirag.rag.graph import run_ask
    loop = asyncio.get_running_loop()
    out = await loop.run_in_executor(None, run_ask, req.question)
    total_ms = int((time.perf_counter() - t0) * 1000)
    trace_nodes = {k: float(v) for k, v in (out.get("trace") or {}).items()}

    resp = AskResponse(
        trace=TraceOut(trace_id=trace_id, cache="miss_answer", total_ms=total_ms,
                       retrieval_ms=int(trace_nodes.get("retrieve", 0)),
                       rerank_ms=int(trace_nodes.get("rerank", 0)),
                       generation_ms=int(trace_nodes.get("generate", 0)),
                       nodes=trace_nodes),
        status=out["status"],
        answer=out["answer"],
        citations=[_coerce_citation(c) for c in (out.get("citations") or [])],
        confidence=float(out.get("confidence") or 0.0),
        refusal_reason=out.get("refusal_reason"),
        disclaimer=out.get("disclaimer") or "本回答仅供技术验证，不构成诊疗建议。",
    )
    if d.cache is not None:
        await d.cache.set_answer(req.question, filters, {
            "status": resp.status, "answer": resp.answer,
            "citations": [c.model_dump() for c in resp.citations],
            "confidence": resp.confidence,
            "refusal_reason": resp.refusal_reason,
            "disclaimer": resp.disclaimer,
            "trace": {"trace_id": trace_id, "cache": "hit", "total_ms": 0,
                      "retrieval_ms": 0, "rerank_ms": 0, "generation_ms": 0,
                      "nodes": {}},
        })
    # 审计：fire-and-forget，失败不拖垮响应
    from medevirag.serve.audit import QaLogRecord, fire_and_forget
    fire_and_forget(d.audit_sink, QaLogRecord(
        trace_id=trace_id, question=req.question, answer=resp.answer,
        status=resp.status, confidence=resp.confidence,
        refusal_reason=resp.refusal_reason, latency_ms=total_ms,
        retrieval_ms=int(trace_nodes.get("retrieve", 0)),
        rerank_ms=int(trace_nodes.get("rerank", 0)),
        generation_ms=int(trace_nodes.get("generate", 0)),
        retrieved_chunk_ids=[c.chunk_id for c in resp.citations][:200],
        config_hash=d.cache.config_hash if d.cache else "-",
        cache="miss_answer"))
    return resp


@router.get("/v1/citations/{chunk_id}", response_model=CitationDetail,
             responses={404: {"model": ErrorBody}})
async def citation_detail(chunk_id: str, request: Request) -> CitationDetail:
    from medevirag.rag.citations import get_citation_detail
    loop = asyncio.get_running_loop()
    try:
        d = await loop.run_in_executor(None, get_citation_detail, chunk_id)
    except KeyError:
        raise HTTPException(404, detail={
            "code": "NOT_FOUND", "message": f"chunk_id 不存在: {chunk_id}",
            "chunk_id": chunk_id}) from None
    except ValueError as exc:
        raise HTTPException(404, detail={
            "code": "NOT_FOUND", "message": str(exc)[:120]}) from exc
    row = {
        "chunk_id": d["chunk_id"], "parent_id": "",
        "doc_title": d["doc_title"],
        "section_path": d["section_path"],
        "chunk_type": d["chunk_type"], "text": d["text"],
        "context_header": d["context_header"],
        "page_start": int(d["page"] or 0), "page_end": int(d["page"] or 0),
        "locator": {"page": d["page"], "block_index": d.get("block_index"),
                    "bbox": d.get("bbox")},
        "evidence_level": d["evidence_level"], "rec_strength": "unknown",
    }
    return CitationDetail(**row)


@router.post("/v1/feedback", status_code=201)
async def feedback(req: FeedbackRequest, request: Request) -> dict[str, Any]:
    d = request.app.state.deps
    # 无 PG：落文件审计（同一目录 feedback.jsonl）
    import json as _json
    from datetime import datetime, timezone
    from pathlib import Path
    sink = d.audit_sink
    line = _json.dumps({
        "trace_id": req.trace_id, "rating": req.rating, "comment": req.comment,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        ensure_ascii=False) + "\n"
    Path(getattr(sink, "path", Path("data/logs"))).parent.mkdir(parents=True, exist_ok=True)
    fb = Path(str(getattr(sink, "path", Path("data/logs/qa_log.jsonl")))).parent / "feedback.jsonl"
    fb.open("a", encoding="utf-8").write(line)
    return {"id": int(time.time())}
