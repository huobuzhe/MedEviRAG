"""T5 生成器：三段 Prompt + 强制 JSON（docs/04 §T5）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

SYSTEM_PROMPT = """你是循证医学问答系统的答案生成器，服务于临床医生。必须遵守以下硬约束：
1. 只能使用【证据块】中出现的内容作答；禁止使用你的先验医学知识、经验或常识补充。
2. 每一个事实性断言（含数字、阈值、剂量、分期、推荐等级）后必须紧跟至少一个引用标记，
   格式严格为 [C1]、[C2]……编号只能取自本次【证据块】中实际出现的编号。
3. 禁止无出处表述，例如"一般认为""临床上常用""通常情况下"。
4. 若【证据块】不足以回答问题的核心部分，必须把 insufficient_evidence 置为 true，
   answer 只写"现有证据不足以回答该问题"，不得编造，也不得用部分证据拼凑完整答案。
5. 只输出一个 JSON 对象。不要输出 Markdown 代码围栏，不要输出解释、前言、后记。
6. 使用简体中文，不超过 6 句；先结论后依据；涉及用药时只复述证据原文给出的剂量，
   不得换算或推算。"""

OUTPUT_CONTRACT = """【输出格式契约】
严格输出如下 JSON（键名不可改、不可增删）：
{"answer": "中文答案正文，引用标记形如 [C1]", "used_citations": ["C1", "C3"],
 "insufficient_evidence": false}
- used_citations 必须与 answer 中实际出现的标记集合完全一致（不多、不少、不重复）。
- insufficient_evidence 只能是 true 或 false。"""


@dataclass
class GenerationResult:
    answer: str
    used_citations: list[str]
    insufficient_evidence: bool
    raw: str
    parse_ok: bool


class LLMClient(Protocol):
    """docs/04 §T5 的端口：系统与用户提示 + 采样参数，返回纯文本。"""

    def chat(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str: ...


def build_user_prompt(question: str, evidence_text: str, degraded: bool = False) -> str:
    extra = ""
    if degraded:
        extra = (
            "\n【重试约束】上一轮输出未通过引用校验。本轮只允许引用 [C1]、[C2]、[C3]；"
            "每个句子都必须以引用标记结尾；不得连续两句共用同一标记。\n"
        )
    return f"【问题】\n{question}\n\n【证据块】\n{evidence_text}\n\n{OUTPUT_CONTRACT}{extra}"


TRIPLE = "\u0060\u0060\u0060"  # 三反引号，写成转义避免源码里出现字面围栏


def strip_code_fence(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines() if not ln.strip().startswith(TRIPLE)]
    return "\n".join(lines).strip()


_MARKER = re.compile(r"[\[［【]\s*[Cc]\s*(\d{1,2})\s*[\]］】]")


def normalize_marker(raw: str) -> str:
    text = raw if raw.strip().startswith(("[", "［", "【")) else f"[{raw}]"
    hit = _MARKER.search(text)
    return f"C{hit.group(1)}" if hit else ""


def parse_generation(raw: str) -> GenerationResult:
    txt = strip_code_fence(raw)
    obj = None
    candidates = [txt]
    if "{" in txt and "}" in txt:
        candidates.append(txt[txt.find("{"): txt.rfind("}") + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            break
        except Exception:
            continue
    if not isinstance(obj, dict):
        return GenerationResult("", [], True, raw, parse_ok=False)
    answer = str(obj.get("answer", "")).strip()
    used = [normalize_marker(str(x)) for x in (obj.get("used_citations") or [])]
    used = [m for m in dict.fromkeys(used) if m]
    return GenerationResult(
        answer=answer,
        used_citations=used,
        insufficient_evidence=bool(obj.get("insufficient_evidence", False)),
        raw=raw,
        parse_ok=True,
    )


def generate(llm: LLMClient, question: str, evidence_text: str,
             degraded: bool = False) -> GenerationResult:
    raw = llm.chat(
        SYSTEM_PROMPT,
        build_user_prompt(question, evidence_text, degraded),
        temperature=0.0,
        max_tokens=800,
    )
    return parse_generation(raw)
