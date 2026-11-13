"""LLM 端口（Protocol）与确定性桩 FakeLLM（docs/01 §T6）。

为什么用 Protocol 而不是抽象基类：领域代码只需要"能 complete(req)"这一件事，
Protocol 是结构化类型，S3 的真实 HTTP 客户端不需要继承任何基类就能满足它，
也不用把 medevirag 的类型反向导入到 provider 适配层。

为什么桩必须确定性：单测断言"同问同答"要有依据。指纹取 sha256(system\x00user)[:12]，
既保证输入相同输出相同，又保证输入不同输出不同（否则桩退化成常量，测不出问题）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, Protocol

ScriptMode = Literal["normal", "bad_citation", "refuse", "malformed_json"]
"""四个剧本：正常作答 / 引用越界 / 应拒答 / 返回残缺 JSON。

用 Literal 而非 str，是为了让 FakeLLM(mode="badcition") 这类拼错在 mypy 阶段就报错，
而不是静默落到 normal 分支让"非法引用"用例假绿。
"""


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "fake"
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    """端口：S3 的 HTTP 客户端与 FakeLLM 都满足它，调用方只依赖这个协议。"""

    def complete(self, req: LLMRequest) -> LLMResponse: ...


@dataclass
class FakeLLM:
    """确定性桩：不发任何 HTTP，答案里嵌输入指纹。"""

    mode: ScriptMode = "normal"
    model: str = "fake-deterministic"
    calls: list[LLMRequest] = field(default_factory=list)

    def _key(self, req: LLMRequest) -> str:
        raw = (req.system + "\x00" + req.user).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(req)
        key = self._key(req)
        if self.mode == "refuse":
            payload = {"answer": "", "status": "refused", "refusal_reason": "off_domain"}
        elif self.mode == "bad_citation":
            payload = {"answer": f"推荐使用氨氯地平 5mg qd [C9]。key={key}", "status": "answered"}
        elif self.mode == "malformed_json":
            text = f'{{"answer": "缺少右括号 key={key}"'
            return LLMResponse(
                text=text,
                prompt_tokens=max(1, len(req.user) // 4),
                completion_tokens=8,
                model=self.model,
            )
        else:
            payload = {"answer": f"固定答案 key={key} [C1]", "status": "answered"}
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return LLMResponse(
            text=text,
            prompt_tokens=max(1, len(req.user) // 4),
            completion_tokens=max(1, len(text) // 4),
            model=self.model,
        )
