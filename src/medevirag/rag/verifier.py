"""T7 两级引用校验 + 降级（docs/04 §T7，ADR-04 落地）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .citations import markers_in, split_sentences

MIN_ENTAIL = 0.5
SUPPORT_THRESHOLD = 0.85
MIN_ANSWER_CHARS = 20


@dataclass
class SentenceCheck:
    index: int
    sentence: str
    markers: list[str]
    legal_markers: list[str] = field(default_factory=list)
    illegal_markers: list[str] = field(default_factory=list)
    entail_score: float = 0.0
    supported: bool = False


@dataclass
class VerifyResult:
    sentences: list[SentenceCheck]
    illegal_markers: list[str]
    landing_ok: bool
    support_rate: float
    unsupported: list[str]
    cleaned_answer: str
    must_retry: bool


class EntailmentJudge(Protocol):
    def judge(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class LexicalJudge:
    """离线回退：中文 bigram 覆盖率。阈值 0.5 与 LLMJudge 在 dev 集上校准对齐。"""

    def judge(self, pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for sent, src in pairs:
            sent_c = re.sub(r"[^\u4e00-\u9fff0-9A-Za-z.]", "", sent)
            grams = {sent_c[i: i + 2] for i in range(len(sent_c) - 1)} or {sent_c}
            hit = sum(1 for g in grams if g in src)
            out.append(hit / max(1, len(grams)))
        return out


class LLMJudge:
    PROMPT = (
        "判断【句子】中的每个断言是否被【原文】支持。"
        "只输出 JSON 数组，元素为 0~1 的浮点，顺序与输入一致。"
        "原文没有明确支持、或与原文矛盾，一律给 0.0。"
    )

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def judge(self, pairs: list[tuple[str, str]]) -> list[float]:
        payload = "\n".join(f"【句子】{s}\n【原文】{src}\n---" for s, src in pairs)
        raw = self.llm.chat(self.PROMPT, payload, temperature=0.0, max_tokens=200)
        try:
            vals = [float(v) for v in json.loads(raw)]
        except Exception:
            return [0.0] * len(pairs)
        vals += [0.0] * max(0, len(pairs) - len(vals))
        return vals[: len(pairs)]


def check_sentences(answer: str, evidence: dict[str, dict[str, Any]],
                    judge: EntailmentJudge,
                    min_entail: float = MIN_ENTAIL) -> list[SentenceCheck]:
    checks: list[SentenceCheck] = []
    for i, sent in enumerate(split_sentences(answer)):
        ms = markers_in(sent)
        if not ms:
            continue
        ck = SentenceCheck(index=i, sentence=sent, markers=ms)
        ck.legal_markers = [m for m in ms if m in evidence]
        ck.illegal_markers = [m for m in ms if m not in evidence]
        src = "\n".join(evidence[m]["text"] for m in ck.legal_markers) if ck.legal_markers else ""
        ck.entail_score = judge.judge([(sent, src)])[0] if src else 0.0
        ck.supported = bool(src) and ck.entail_score >= min_entail
        checks.append(ck)
    return checks


def verify(answer: str, evidence: dict[str, dict[str, Any]], judge: EntailmentJudge,
           support_threshold: float = SUPPORT_THRESHOLD) -> VerifyResult:
    checks = check_sentences(answer, evidence, judge)
    illegal = sorted({m for c in checks for m in c.illegal_markers})
    cited = len(checks)
    support_rate = sum(1 for c in checks if c.supported) / cited if cited else 0.0

    kept: list[str] = []
    for ck in checks:
        if ck.illegal_markers and not ck.legal_markers:
            continue
        if ck.illegal_markers:
            # 只删非法标记，保留该句的合法标记（降级策略第 1 行）
            cleaned = ck.sentence
            for mk in ck.illegal_markers:
                cleaned = re.sub(rf"[\[［【]\s*[Cc]\s*{mk[1:]}\s*[\]］】]", "", cleaned)
        else:
            cleaned = ck.sentence
        kept.append(cleaned.strip())

    cleaned_answer = "\n".join(s for s in kept if s).strip()
    deleted_ratio = (len(checks) - len(kept)) / len(checks) if checks else 0.0
    must_retry = (cleaned_answer and len(cleaned_answer) < MIN_ANSWER_CHARS) or deleted_ratio > 0.5
    if must_retry:
        cleaned_answer = ""

    return VerifyResult(
        sentences=checks,
        illegal_markers=illegal,
        landing_ok=not illegal,
        support_rate=support_rate,
        unsupported=[c.sentence for c in checks if c.markers and not c.supported],
        cleaned_answer=cleaned_answer,
        must_retry=must_retry,
    )
