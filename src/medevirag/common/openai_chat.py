"""OpenAI 兼容 provider 的 chat 客户端（新增文件，不改动原项目默认行为）。

只在环境变量 MEDEVIRAG_LLM_PROVIDER 显式指定 provider 时，由
rag/nodes.py::get_llm() 使用；未设置时仍走 common/llm.FakeLLM（ADR-06 双轨），
所有离线测试、CI 评测门禁与 pipeline_selfcheck 报告保持确定性不变。

配置来源：configs/models.yaml 的 llm.providers[provider]（get_settings().provider()）。
密钥来源：优先进程环境变量 <api_key_env>，其次项目根 .env 文件；只按变量名取值，
绝不打印、记录或返回密钥明文。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from medevirag.common.config import get_settings
from medevirag.common.llm_client import post_json

# src/medevirag/common/openai_chat.py -> 上溯三级 -> 项目根
_ROOT = Path(__file__).resolve().parents[3]


def _env_value(name: str) -> str:
    """按变量名取密钥：先 os.environ，再项目根 .env（不打印任何值）。

    Settings 用 env_prefix="MEDEVIRAG__" + extra="ignore"，.env 里不带前缀的
    SILICONFLOW_API_KEY 不会进入 os.environ，所以这里必须自己补读一次。
    """
    val = os.environ.get(name)
    if val:
        return val.strip()
    env_file = _ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip() == name:
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    return ""


class OpenAICompatChat:
    """实现 rag/generator.LLMClient 协议：chat(system, user, *, temperature, max_tokens) -> str。"""

    def __init__(self, provider: str | None = None) -> None:
        settings = get_settings()
        cfg = settings.provider(provider)
        self.name = provider or settings.llm.default
        self.cfg = cfg
        self.api_key = _env_value(cfg.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"provider {self.name!r} 需要环境变量 {cfg.api_key_env}；"
                "请在项目根 .env 中填写后重启服务"
            )

    def chat(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        data = post_json(
            url,
            body,
            api_key=self.api_key,
            proxy=self.cfg.proxy,
            timeout=self.cfg.timeout_s,
        )
        try:
            return str(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"provider {self.name!r} 返回结构异常") from exc
