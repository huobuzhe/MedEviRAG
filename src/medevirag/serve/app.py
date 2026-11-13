# S6 T3：app 装配（docs/07 §T3）。trace 中间件 + 三个异常处理器 + CORS。
# uvicorn medevirag.serve.app:app --port 8000 --workers 1

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from medevirag.common.config import get_settings
from medevirag.serve.deps import lifespan
from medevirag.serve.routes import router
from medevirag.serve.schemas import ErrorBody

log = logging.getLogger("medevirag.serve.app")


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="MedEviRAG API", version="0.6.0",
        description="中文医学指南循证问答 · 全链路可溯源",
        lifespan=lifespan, docs_url="/docs", openapi_url="/openapi.json")

    @app.middleware("http")
    async def _trace(request: Request, call_next: Any) -> Any:
        tid = request.headers.get("x-trace-id") or uuid.uuid4().hex[:16]
        request.state.trace_id = tid
        t0 = time.perf_counter()
        resp = await call_next(request)
        resp.headers["x-trace-id"] = tid
        resp.headers["x-elapsed-ms"] = str(int((time.perf_counter() - t0) * 1000))
        return resp

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({
            "code": "INVALID_REQUEST",
            "message": str(exc.errors()[:2])[:200],
            "trace_id": getattr(request.state, "trace_id", "-")}, status_code=400)

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException) -> JSONResponse:
        detail: dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {}
        body = ErrorBody(code=detail.get("code", "INTERNAL"),
                         message=detail.get("message", str(exc.detail))[:200],
                         trace_id=getattr(request.state, "trace_id", "-"))
        return JSONResponse(body.model_dump(mode="json"), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _internal(request: Request, exc: Exception) -> JSONResponse:
        tid = getattr(request.state, "trace_id", "-")
        log.exception("unhandled error trace_id=%s", tid)
        return JSONResponse({
            "code": "INTERNAL",
            "message": "internal error, see trace_id in logs",
            "trace_id": tid}, status_code=500)

    app.include_router(router)
    # trace 中间件先注册（后进先出），再挂 CORS
    app.add_middleware(CORSMiddleware,
                       allow_origins=s.serve.cors_origins,
                       allow_methods=["*"], allow_headers=["*"])
    return app


app = create_app()
