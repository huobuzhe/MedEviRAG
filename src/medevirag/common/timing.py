"""节点计时上下文管理器（docs/01 §T3）。

用法::

    with node_timer("dense_search") as t:
        hits = ...
    # 日志已输出 {"event": "node_done", "node": "dense_search", "latency_ms": 12.3}
    latency = t.latency_ms

为什么单独一个 contextmanager 而不是散落的 time.perf_counter：S6 的 P95 预算被拆成
800+600+3500+1000ms 四段，每段都必须能从日志里直接聚合出来。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from medevirag.common.logging import get_logger

_log = get_logger(__name__)


@dataclass
class Timer:
    """计时结果；即使代码在 with 块里抛异常也会被记录。"""

    node: str
    latency_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    failed: bool = False


@contextmanager
def node_timer(node: str, **extra: Any) -> Iterator[Timer]:
    timer = Timer(node=node, extra=dict(extra))
    started = time.perf_counter()
    try:
        yield timer
    except Exception:
        timer.failed = True
        timer.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        _log.exception("node_failed", extra={"node": node, "latency_ms": timer.latency_ms, **timer.extra})
        raise
    else:
        timer.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        _log.info("node_done", extra={"node": node, "latency_ms": timer.latency_ms, **timer.extra})


def timed(node: str, **extra: Any) -> Any:
    """函数装饰器版本，保持原返回值。"""

    def wrapper(fn: Any) -> Any:
        def inner(*args: Any, **kwargs: Any) -> Any:
            with node_timer(node, **extra):
                return fn(*args, **kwargs)

        inner.__name__ = getattr(fn, "__name__", node)
        inner.__doc__ = getattr(fn, "__doc__", None)
        return inner

    return wrapper
