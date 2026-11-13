"""结构化 JSON 日志 + contextvars trace_id（docs/01 §T3）。

为什么不用标准 logging 的默认格式：一条检索链路有 6 个节点，若日志是自由文本，
线上"这次请求哪一步慢"只能靠肉眼拼；JSON 一行一个事件，latency_ms 可直接聚合。

trace_id 用 contextvars 而非全局变量：FastAPI 同一进程并发处理多个请求，
全局变量会让两个请求的 trace_id 串台。
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")

_CONFIGURED = False


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def bind_trace(trace_id: str | None = None) -> str:
    """绑定 trace_id，返回实际使用的值。"""
    tid = trace_id or new_trace_id()
    _trace_id.set(tid)
    return tid


def reset_trace() -> None:
    _trace_id.set("-")


def current_trace_id() -> str:
    return _trace_id.get()


class JsonFormatter(logging.Formatter):
    """把 LogRecord 渲染成单行 JSON；extra 里的字段平铺进顶层。"""

    _RESERVED = frozenset(
        {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "trace_id": getattr(record, "trace_id", current_trace_id()),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """控制台可读格式，字段追加在行尾。"""

    def format(self, record: logging.LogRecord) -> str:
        head = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.getMessage()}"
        extras = [
            f"{k}={v}"
            for k, v in record.__dict__.items()
            if k not in JsonFormatter._RESERVED and not k.startswith("_") and k != "trace_id"
        ]
        tid = getattr(record, "trace_id", current_trace_id())
        tail = " ".join([f"trace={tid}", *extras])
        return f"{head} | {tail}" if tail else head


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """幂等地配置根 logger；重复调用不会叠加 handler。"""
    global _CONFIGURED
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """临时给本作用域内的日志追加字段（不改全局配置）。"""

    token = _trace_id.set(fields.pop("trace_id", current_trace_id()))
    try:
        yield
    finally:
        _trace_id.reset(token)
