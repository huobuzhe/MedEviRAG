"""统一错误体系（docs/01 §T5）。

每个异常都带 code / retryable / to_dict()：
- retryable 让上层决定重试还是直接降级（例如 LLM 超时可重试，配置错不可重试）；
- code 是稳定字符串，日志与告警按它聚合，不按 message 聚合。
"""

from __future__ import annotations

from typing import Any


class MedEviRAGError(Exception):
    """所有自定义异常的基类。"""

    code = "internal_error"
    retryable = False
    http_status = 500

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "context": self.context,
        }

    def __str__(self) -> str:
        if not self.context:
            return self.message
        detail = " ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} [{detail}]"


class ConfigError(MedEviRAGError):
    """配置缺失、取值非法、provider 未定义。"""

    code = "config_error"
    retryable = False
    http_status = 500


class ProviderError(MedEviRAGError):
    """LLM / embedding / reranker 等外部依赖调用失败。"""

    code = "provider_error"
    retryable = True
    http_status = 502


class ProviderAuthError(ProviderError):
    """Key 无效或无权限；重试无意义。"""

    code = "provider_auth_error"
    retryable = False
    http_status = 502


class ProviderRateLimitError(ProviderError):
    """命中限流或配额；退避后可重试。"""

    code = "provider_rate_limit"
    retryable = True
    http_status = 429


class StoreError(MedEviRAGError):
    """Milvus / PostgreSQL / Redis 等存储层失败。"""

    code = "store_error"
    retryable = True
    http_status = 503


class HealthCheckError(MedEviRAGError):
    """doctor 或 /healthz 自检未通过。"""

    code = "health_check_error"
    retryable = False
    http_status = 500


class BudgetExceeded(MedEviRAGError):
    """token 预算超限；属于主动熔断而不是故障。"""

    code = "budget_exceeded"
    retryable = False
    http_status = 429


class RetrievalError(MedEviRAGError):
    """召回阶段无可用结果或索引缺失。"""

    code = "retrieval_error"
    retryable = False
    http_status = 200


class GenerationError(MedEviRAGError):
    """生成阶段失败（空响应、引用越界等）。"""

    code = "generation_error"
    retryable = True
    http_status = 502
