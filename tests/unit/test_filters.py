"""T6 元数据硬过滤单测（docs/03 §T6，A8）。

四条规则各自独立可放宽；unknown 证据等级必须 100% 存活；
候选不足 min_keep 时逐级放宽且 relaxed 非空。
"""

from __future__ import annotations

import pytest

from medevirag.index.filters import (
    FilterRules,
    _drop_reason,
    apply_hard_filters,
    explain_drops,
    relax,
)
from medevirag.index.fusion import RetrievedChunk


def rc(
    cid: str,
    *,
    lang: str = "zh",
    doc_type: str = "guideline",
    publish_year: int = 2023,
    evidence_level: str = "unknown",
    fused_rank: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        doc_id=cid.split("#")[0],
        text=f"text of {cid}",
        lang=lang,
        doc_type=doc_type,
        publish_year=publish_year,
        evidence_level=evidence_level,
        fused_rank=fused_rank,
    )


def test_rules_from_config_reads_filters_section() -> None:
    rules = FilterRules.from_config({
        "filters": {
            "publish_year_min": 2015,
            "lang_allow": ["zh", "en"],
            "doc_type_allow": ["guideline", "consensus", "paper", "manual"],
            "evidence_level_exclude": ["D"],
            "min_keep": 8,
        }
    })
    assert rules.publish_year_min == 2015
    assert rules.evidence_level_exclude == ("D",)
    assert rules.min_keep == 8


def test_rules_from_config_derives_exclude_from_min_evidence_level() -> None:
    """S0 只冻结了 min_evidence_level="C" → 只排除严格低于 C 的 D。"""
    rules = FilterRules.from_config({"filters": {"min_evidence_level": "C"}})
    assert rules.evidence_level_exclude == ("D",)


def test_rule_lang_drops_non_allowed() -> None:
    rules = FilterRules(min_keep=0)
    assert _drop_reason(rc("x", lang="ja"), rules, set()) == "lang"
    assert _drop_reason(rc("y", lang="en"), rules, set()) is None


def test_rule_doc_type_drops_outside_allowlist() -> None:
    rules = FilterRules(min_keep=0)
    assert _drop_reason(rc("x", doc_type="news"), rules, set()) == "doc_type"
    assert _drop_reason(rc("y", doc_type="consensus"), rules, set()) is None


def test_rule_publish_year_drops_too_old_but_keeps_unknown_zero() -> None:
    rules = FilterRules(publish_year_min=2015, min_keep=0)
    assert _drop_reason(rc("x", publish_year=2010), rules, set()) == "publish_year"
    assert _drop_reason(rc("y", publish_year=2015), rules, set()) is None
    # 年份未知（0）不能当成"太旧"淘汰
    assert _drop_reason(rc("z", publish_year=0), rules, set()) is None


def test_rule_evidence_level_excludes_d_only() -> None:
    rules = FilterRules(evidence_level_exclude=("D",), min_keep=0)
    assert _drop_reason(rc("x", evidence_level="D"), rules, set()) == "evidence_level"
    assert _drop_reason(rc("y", evidence_level="C"), rules, set()) is None


def test_unknown_evidence_level_survives_100_percent() -> None:
    """A8 的核心断言：unknown 不因证据等级被淘汰。"""
    chunks = [rc(f"u{i}", evidence_level="unknown") for i in range(50)]
    rules = FilterRules(min_keep=0)
    kept = apply_hard_filters(chunks, rules)
    assert len(kept) == 50
    assert all(c.evidence_level == "unknown" for c in kept)


def test_unknown_survives_even_under_a_strict_floor() -> None:
    """即使把阈值抬到 A（只留 A），unknown 仍全部存活——只看"明确低于"。"""
    chunks = [rc("a1", evidence_level="A"), rc("u1", evidence_level="unknown"), rc("d1", evidence_level="D")]
    rules = FilterRules(evidence_level_exclude=("B", "C", "D"), min_keep=0)
    kept = apply_hard_filters(chunks, rules)
    assert [c.chunk_id for c in kept] == ["a1", "u1"]


def test_apply_is_order_preserving_and_keeps_fused_rank() -> None:
    chunks = [rc(f"c{i}", fused_rank=i + 1) for i in range(10)]
    kept = apply_hard_filters(chunks, FilterRules(min_keep=0))
    assert [c.chunk_id for c in kept] == [c.chunk_id for c in chunks]
    assert [c.fused_rank for c in kept] == list(range(1, 11))


def test_relax_triggers_when_candidates_below_min_keep() -> None:
    """候选 < min_keep 时 relaxed 非空（A8 的第三条要求）。"""
    chunks = [rc("a", lang="ja"), rc("b", doc_type="news"), rc("c", publish_year=2000)]
    rules = FilterRules(min_keep=8)
    stats: dict[str, object] = {}
    kept = apply_hard_filters(chunks, rules, stats=stats)
    assert stats["relaxed"], stats
    assert stats["relaxed"] == ["doc_type", "evidence_level", "publish_year"]
    # 三轮放宽全部走完，lang 不在 _RELAX_ORDER 里（docs/03 §T6 只定义三个可放宽规则），
    # 所以 lang="ja" 的 a 仍被淘汰，b/c 被救回。
    assert [c.chunk_id for c in kept] == ["b", "c"]


def test_relax_order_is_evidence_then_doc_type_then_publish_year() -> None:
    """逐级放宽：先证据等级，再 doc_type，最后年份；一旦达标立即停。"""
    chunks = [rc(f"d{i}", evidence_level="D") for i in range(10)]
    rules = FilterRules(min_keep=8, evidence_level_exclude=("D",))
    stats: dict[str, object] = {}
    kept = apply_hard_filters(chunks, rules, stats=stats)
    assert len(kept) == 10
    assert stats["relaxed"] == ["evidence_level"]


def test_no_relax_when_enough_candidates() -> None:
    chunks = [rc(f"c{i}") for i in range(10)]
    stats: dict[str, object] = {}
    apply_hard_filters(chunks, FilterRules(min_keep=8), stats=stats)
    assert stats["relaxed"] == []


def test_stats_counts_drops_by_reason() -> None:
    chunks = [
        rc("ok1"),
        rc("old", publish_year=2001),
        rc("en_bad", lang="ja"),
        rc("news", doc_type="news"),
        rc("d", evidence_level="D"),
    ]
    stats: dict[str, object] = {}
    apply_hard_filters(chunks, FilterRules(min_keep=0), stats=stats)
    assert stats["input"] == 5
    assert stats["kept"] == 1
    assert stats["dropped"] == {"publish_year": 1, "lang": 1, "doc_type": 1, "evidence_level": 1}


def test_explain_drops_does_not_filter() -> None:
    chunks = [rc("ok"), rc("old", publish_year=2001)]
    counted = explain_drops(chunks, FilterRules(min_keep=0))
    assert counted["publish_year"] == 1
    assert len(chunks) == 2


def test_relax_helper_returns_list() -> None:
    chunks = [rc("a")]
    out = relax(chunks, FilterRules(min_keep=8))
    assert isinstance(out, list)
    assert out


@pytest.mark.parametrize("level", ["A", "B", "C", "unknown"])
def test_non_excluded_levels_all_kept(level: str) -> None:
    chunks = [rc("x", evidence_level=level)]
    assert len(apply_hard_filters(chunks, FilterRules(evidence_level_exclude=("D",), min_keep=0))) == 1
