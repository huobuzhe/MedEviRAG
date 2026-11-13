# S6 T1：Pydantic v2 服务契约（docs/07 §T1）。
# status 与 S4 common/types.py 同源；错误码有限枚举，内部细节只进日志。

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Status = Literal["answered", "partial", "refused"]


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"          # 400
    NOT_FOUND = "NOT_FOUND"                      # 404
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"  # 503
    LLM_UPSTREAM_ERROR = "LLM_UPSTREAM_ERROR"    # 502
    TIMEOUT = "TIMEOUT"                          # 504
    INTERNAL = "INTERNAL"                       # 500


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=50, ge=10, le=100)
    filters: dict[str, str | int] | None = None
    use_cache: bool = True

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question is blank")
        return v


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: list[str]
    page: int
    quote: str
    url: str | None = None
    evidence_level: str = "unknown"


class TraceOut(BaseModel):
    trace_id: str
    cache: Literal["hit", "miss_answer", "bypass"] = "bypass"
    retrieval_ms: int = 0
    rerank_ms: int = 0
    generation_ms: int = 0
    total_ms: int = 0
    nodes: dict[str, float] = Field(default_factory=dict)


class AskResponse(BaseModel):
    trace: TraceOut
    status: Status
    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    refusal_reason: str | None = None
    disclaimer: str


class CitationDetail(BaseModel):
    chunk_id: str
    parent_id: str
    doc_title: str
    section_path: list[str]
    chunk_type: str
    text: str
    context_header: str
    page_start: int
    page_end: int
    locator: dict[str, Any]
    evidence_level: str
    rec_strength: str


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=500)


class CheckResult(BaseModel):
    ok: bool
    ms: int
    detail: str | None = None


class ReadyReport(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckResult]
    failed: list[str] = Field(default_factory=list)


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    trace_id: str
