"""直连/代理可切换的 HTTP 探测与 UTF-8 安全 POST（docs/01 §T6，00 §13.2）。

两个真实踩过的坑写进实现里：

1. 代理必须显式指定：SiliconFlow 走 Clash 返回 HTTP 000（完全不通），DeepSeek 官方
   必须走代理。若依赖环境变量 HTTPS_PROXY，httpx 会把 proxy=None 当成"读环境变量"，
   于是"直连"配置反而全部走子代理。
2. curl 里塞中文 JSON 走 -d '...' 会乱码（实测模型回阿拉伯语）。Python 侧统一
   json.dumps(ensure_ascii=False).encode("utf-8") 并显式声明 charset。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from medevirag.common.errors import ProviderError
from medevirag.common.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    status: int
    latency_ms: float
    detail: str = ""


def probe_endpoint(
    url: str,
    *,
    proxy: str | None = None,
    timeout: float = 10.0,
    headers: dict[str, str] | None = None,
) -> ProbeResult:
    """探一个 URL 是否可达；proxy=None 表示**显式直连**（清空环境代理）。"""
    handlers: list[Any] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    else:
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, method="GET", headers=headers or {})
    started = time.perf_counter()
    try:
        with opener.open(req, timeout=timeout) as resp:
            latency = round((time.perf_counter() - started) * 1000.0, 1)
            return ProbeResult(ok=True, status=resp.status, latency_ms=latency)
    except urllib.error.HTTPError as exc:
        latency = round((time.perf_counter() - started) * 1000.0, 1)
        # 401/404 也说明网络可达：可达性与鉴权是两件事
        return ProbeResult(ok=True, status=exc.code, latency_ms=latency, detail=str(exc.reason))
    except Exception as exc:  # 探测工具要吞掉一切网络异常并如实上报
        latency = round((time.perf_counter() - started) * 1000.0, 1)
        return ProbeResult(ok=False, status=0, latency_ms=latency, detail=f"{type(exc).__name__}: {exc}")


def post_json(
    url: str,
    body: dict[str, Any],
    *,
    api_key: str | None = None,
    proxy: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """UTF-8 安全的 JSON POST；proxy=None 表示显式直连。"""
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    handlers: list[Any] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    else:
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with opener.open(req, timeout=timeout) as resp:
            parsed: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return parsed
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"HTTP {exc.code}: {detail[:500]}", url=url, status=exc.code) from exc
    except Exception as exc:  # 统一转成 ProviderError，交给上层决定重试还是降级
        raise ProviderError(f"{type(exc).__name__}: {exc}", url=url) from exc
