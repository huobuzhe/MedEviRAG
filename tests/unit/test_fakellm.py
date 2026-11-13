"""FakeLLM 的确定性与四剧本可辨识（docs/01 §T6 验收 A9）。"""

from __future__ import annotations

import json

import pytest

from medevirag.common.llm import FakeLLM, LLMRequest
from medevirag.common.token_guard import TokenLedger

REQ = LLMRequest(system="你是医学助手", user="高血压一线用药？")


def test_same_input_same_output() -> None:
    llm = FakeLLM()
    assert llm.complete(REQ).text == llm.complete(REQ).text


def test_different_input_different_output() -> None:
    llm = FakeLLM()
    other = LLMRequest(system=REQ.system, user="糖尿病一线用药？")
    assert llm.complete(REQ).text != llm.complete(other).text


def test_four_modes_distinguishable() -> None:
    texts = {mode: FakeLLM(mode=mode).complete(REQ).text for mode in (
        "normal", "bad_citation", "refuse", "malformed_json",
    )}
    assert len(set(texts.values())) == 4
    assert "[C9]" in texts["bad_citation"]


def test_refuse_payload_parsable() -> None:
    payload = json.loads(FakeLLM(mode="refuse").complete(REQ).text)
    assert payload["status"] == "refused"
    assert payload["answer"] == ""


def test_malformed_json_is_broken_by_design() -> None:
    text = FakeLLM(mode="malformed_json").complete(REQ).text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_no_real_api_call() -> None:
    llm = FakeLLM()
    llm.complete(REQ)
    assert len(llm.calls) == 1


def test_ledger_starts_empty() -> None:
    ledger = TokenLedger()
    assert ledger.calls == 0 and ledger.total == 0


def test_ledger_hard_limit_raises() -> None:
    from medevirag.common.errors import BudgetExceeded

    ledger = TokenLedger()
    ledger.add({"total_tokens": 10}, node="gen", warn_at=100, hard_at=50)
    with pytest.raises(BudgetExceeded):
        ledger.add({"total_tokens": 100}, node="gen", warn_at=100, hard_at=50)
