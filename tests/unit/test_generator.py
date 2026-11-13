"""T5 生成器单测：JSON 契约 / 围栏剥离 / 标记归一化。"""

from __future__ import annotations

from medevirag.rag.generator import (
    OUTPUT_CONTRACT,
    SYSTEM_PROMPT,
    build_user_prompt,
    generate,
    normalize_marker,
    parse_generation,
    strip_code_fence,
)

_FENCE = "\u0060\u0060\u0060"


class _StubLLM:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def chat(self, system, user, *, temperature, max_tokens):
        self.calls.append({"system": system, "user": user,
                           "temperature": temperature, "max_tokens": max_tokens})
        return self.text


def test_system_prompt_has_six_constraints():
    for token in ("禁止使用你的先验", "引用标记", "insufficient_evidence",
                  "只输出一个 JSON", "不超过 6 句"):
        assert token in SYSTEM_PROMPT
    assert "used_citations" in OUTPUT_CONTRACT


def test_strip_code_fence_removes_fence_lines():
    raw = _FENCE + "json\n{\"a\": 1}\n" + _FENCE
    assert strip_code_fence(raw) == "{\"a\": 1}"


def test_normalize_marker_variants():
    assert normalize_marker("[C 1]") == "C1"
    assert normalize_marker("【c3】") == "C3"
    assert normalize_marker("C12") == "C12"
    assert normalize_marker("不是引用") == ""


def test_parse_generation_ok():
    r = parse_generation('{"answer":"结论 [C1]","used_citations":["C1"],"insufficient_evidence":false}')
    assert r.parse_ok and r.answer == "结论 [C1]"
    assert r.used_citations == ["C1"]
    assert r.insufficient_evidence is False


def test_parse_generation_dedups_and_normalizes_citations():
    r = parse_generation('{"answer":"a [C1] b [C2]","used_citations":["[C 1]","【c2】","C1"]}')
    assert r.used_citations == ["C1", "C2"]


def test_parse_generation_broken_json_marks_not_ok():
    r = parse_generation('{"answer": "缺右括号')
    assert r.parse_ok is False
    assert r.answer == ""


def test_parse_generation_recovers_from_surrounding_noise():
    r = parse_generation('模型说：{"answer":"x [C1]","used_citations":["C1"]} 以上。')
    assert r.parse_ok and r.answer == "x [C1]"


def test_build_user_prompt_degraded_adds_retry_constraint():
    plain = build_user_prompt("q", "ev")
    degraded = build_user_prompt("q", "ev", degraded=True)
    assert "重试约束" not in plain
    assert "重试约束" in degraded and "只允许引用 [C1]" in degraded


def test_generate_uses_zero_temperature_and_800_tokens():
    llm = _StubLLM('{"answer":"x [C1]","used_citations":["C1"]}')
    r = generate(llm, "q", "ev")
    assert r.answer == "x [C1]"
    assert llm.calls[0]["temperature"] == 0.0
    assert llm.calls[0]["max_tokens"] == 800
    assert llm.calls[0]["system"] == SYSTEM_PROMPT
