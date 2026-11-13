"""A14 门禁：预算护栏（docs/04 §A14）。

超 per_run_warn 打 WARNING；超 max_total_tokens 抛 BudgetExceeded。
TokenLedger.add(usage, *, node, warn_at, hard_at) 的 warn_at/hard_at 是必填关键字参数。
"""

from __future__ import annotations

import logging

import pytest

from medevirag.common.errors import BudgetExceeded
from medevirag.common.token_guard import TokenLedger

WARN_AT = 200_000
HARD_AT = 2_000_000


def test_warn_threshold_logs_warning(caplog):
    ledger = TokenLedger()
    with caplog.at_level(logging.WARNING, logger="medevirag.budget"):
        ledger.add({"total_tokens": 250_000}, node="generate",
                   warn_at=WARN_AT, hard_at=HARD_AT)
    assert ledger.total == 250_000
    assert ledger.calls == 1
    assert any("token_budget_warning" in r.getMessage() or "token_budget_warning" in str(r.msg)
               for r in caplog.records if r.levelno >= logging.WARNING)


def test_no_warning_below_threshold(caplog):
    ledger = TokenLedger()
    with caplog.at_level(logging.WARNING, logger="medevirag.budget"):
        ledger.add({"total_tokens": 1000}, node="generate", warn_at=WARN_AT, hard_at=HARD_AT)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_hard_threshold_raises():
    ledger = TokenLedger()
    with pytest.raises(BudgetExceeded):
        ledger.add({"total_tokens": HARD_AT + 1}, node="generate",
                   warn_at=WARN_AT, hard_at=HARD_AT)


def test_accumulates_and_tracks_per_node():
    ledger = TokenLedger()
    ledger.add({"total_tokens": 100}, node="generate", warn_at=WARN_AT, hard_at=HARD_AT)
    ledger.add({"total_tokens": 200}, node="verify", warn_at=WARN_AT, hard_at=HARD_AT)
    assert ledger.total == 300
    assert ledger.calls == 2
    assert ledger.per_node["generate"] == 100
    assert ledger.per_node["verify"] == 200


def test_cumulative_crosses_hard_threshold():
    ledger = TokenLedger()
    ledger.add({"total_tokens": HARD_AT - 10}, node="generate", warn_at=WARN_AT, hard_at=HARD_AT)
    with pytest.raises(BudgetExceeded):
        ledger.add({"total_tokens": 100}, node="verify", warn_at=WARN_AT, hard_at=HARD_AT)
