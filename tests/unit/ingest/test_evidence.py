"""T4 evidence：GRADE / 牛津 / 指南自定 三类识别 + 反例。"""

from __future__ import annotations

import pytest

from medevirag.ingest.evidence import coverage_report, extract_evidence

POSITIVE = [
    ("本研究证据等级：A，推荐用于一线治疗", "A", "GRADE"),
    ("证据 等级：B 级，推荐强度：I", "B", "GRADE"),
    ("推荐强度：Ⅱa，证据级别：1", "A", "牛津"),
    ("证据水平：2（Level of evidence）", "B", "牛津"),
    ("推荐类别：I 类推荐，证据等级：C", "C", None),
]


@pytest.mark.parametrize(("text", "level", "system"), POSITIVE)
def test_extract_evidence_positive(text: str, level: str, system: str | None) -> None:
    got = extract_evidence(text)
    if level in ("A", "B", "C", "D"):
        assert got["evidence_level"] == level
    if system is not None:
        assert got["grade_system"] == system


def test_extract_evidence_rec_strength() -> None:
    got = extract_evidence("推荐强度：Ⅱa。在合并糖尿病患者中应考虑使用。")
    assert got["rec_strength"] == "IIa"


@pytest.mark.parametrize(
    "text",
    [
        "我院为A类定点医院",  # 反例：A 类定点不是证据等级
        "本研究纳入 120 例患者，随访 12 个月",
        "推荐患者每日食盐摄入量不超过 5g",
        "",
    ],
)
def test_extract_evidence_negative(text: str) -> None:
    """无证据标识时必须写 unknown，不能猜。"""
    got = extract_evidence(text)
    assert got["evidence_level"] == "unknown"
    assert got["rec_strength"] == "unknown"
    assert got["grade_system"] == "unknown"


def test_coverage_report_counts() -> None:
    rows = [
        {"evidence_level": "A", "rec_strength": "I", "doc_id": "d1"},
        {"evidence_level": "unknown", "rec_strength": "unknown", "doc_id": "d2"},
    ]
    rep = coverage_report(rows)
    assert rep["chunks"] == 2
    assert rep["evidence_level_hits"] == 1
    assert rep["evidence_level_coverage"] == pytest.approx(0.5)
