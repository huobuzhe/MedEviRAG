# S5 T2：配额矩阵自检（docs/06 §T2）。手改配额会在 CI 炸出来。

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_quota():
    p = Path(__file__).resolve().parents[2] / "data" / "eval" / "quota.py"
    spec = importlib.util.spec_from_file_location("quota_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


Q = _load_quota()


def test_quota_totals():
    Q.assert_quota()


def test_topic_code_covers_all_topics():
    assert set(Q.TOPIC_CODE) == set(Q.TOPIC_QUOTA)
    assert len(Q.TOPIC_CODE) == 12


def test_refuse_quota_sums_to_50():
    assert sum(Q.REFUSE_QUOTA.values()) == 50
