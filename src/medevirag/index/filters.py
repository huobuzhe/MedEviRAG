"""T6：元数据硬过滤（post-filter，ADR-03；docs/03 §T6）。

位置选择是实测结论：pre-filter 在窄条件下会把候选池打到 0 条
（docs/03 §T6 的量化例子：doc_type=guideline AND publish_year>=2015 AND
evidence_level=A 在 20000 条里命中 0 条），post-filter 则是在融合之后
按需淘汰，配合"候选不足 min_keep 就逐级放宽"的兜底。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from medevirag.common.types import EVIDENCE_LEVEL_ORDER
from medevirag.index.fusion import RetrievedChunk

__all__ = ["FilterRules", "apply_hard_filters", "relax", "explain_drops"]

LOGGER = logging.getLogger("medevirag.index.filters")

Verdict = tuple[bool, str]

_RELAX_ORDER: tuple[str, ...] = ("evidence_level", "doc_type", "publish_year")


@dataclass(frozen=True)
class FilterRules:
    """四条硬规则 + 一个放宽下限（A8）。"""

    publish_year_min: int = 2015
    lang_allow: tuple[str, ...] = ("zh", "en")
    doc_type_allow: tuple[str, ...] = ("guideline", "consensus", "paper", "manual")
    evidence_level_exclude: tuple[str, ...] = ("D",)
    min_keep: int = 8

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> FilterRules:
        """从 retrieval.yaml 的 filters 段构造。

        S0 只冻结了 `max_publish_age_years` 与 `min_evidence_level` 两个键，
        这里把它们翻译成本阶段的四条规则：
            年份下限 = docs/03 §T6 冻结的 2015（不用 this_year-age，避免结果随日历漂移）
            证据等级排除 = 严格低于 min_evidence_level 的等级（"C" -> 只排 "D"）
        显式写了 publish_year_min / evidence_level_exclude 时以显式值为准。
        """
        sec = dict(cfg.get("filters") or cfg or {})
        # 年份下限用 docs/03 §T6 冻结的 2015，而不是 this_year - max_publish_age_years：
        # 后者会随日历漂移（2026 年算出来是 2016），评测结果就不可复现了。
        publish_year_min = int(sec.get("publish_year_min", 2015))

        explicit = sec.get("evidence_level_exclude")
        if explicit:
            exclude = tuple(str(x) for x in explicit)
        else:
            floor = str(sec.get("min_evidence_level") or "C")
            floor_rank = EVIDENCE_LEVEL_ORDER.get(floor, 2)
            exclude = tuple(
                level for level, rank in EVIDENCE_LEVEL_ORDER.items() if rank < floor_rank and rank > 0
            )
        return cls(
            publish_year_min=publish_year_min,
            lang_allow=tuple(str(x) for x in sec.get("lang_allow") or ("zh", "en")),
            doc_type_allow=tuple(
                str(x) for x in sec.get("doc_type_allow") or ("guideline", "consensus", "paper", "manual")
            ),
            evidence_level_exclude=exclude,
            min_keep=int(sec.get("min_keep", 8)),
        )


def _drop_reason(chunk: RetrievedChunk, rules: FilterRules, relaxed: set[str]) -> str | None:
    """返回丢弃原因；None 表示保留。四条规则各自独立可放宽。"""
    if "lang" not in relaxed and chunk.lang and chunk.lang not in rules.lang_allow:
        return "lang"
    if "doc_type" not in relaxed and chunk.doc_type and chunk.doc_type not in rules.doc_type_allow:
        return "doc_type"
    # publish_year=0 表示未知，放行（不能把未知当"太旧"淘汰）
    if "publish_year" not in relaxed and chunk.publish_year and chunk.publish_year < rules.publish_year_min:
        return "publish_year"
    # 只有"明确低于阈值"才淘汰：unknown 永不因证据等级被丢（A8 的 100% 存活）
    if "evidence_level" not in relaxed and chunk.evidence_level in rules.evidence_level_exclude:
        return "evidence_level"
    return None


def apply_hard_filters(
    chunks: list[RetrievedChunk],
    rules: FilterRules | None = None,
    cfg: dict[str, Any] | None = None,
    *,
    stats: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    """保序过滤；候选不足 min_keep 时按 _RELAX_ORDER 逐级放宽并打 WARNING。

    stats 若传入，会把 relaxed 集合与按原因分类的丢弃计数写进去，
    供 data/reports/s2_ablation.json 记录（docs/03 §T6 要求）。
    """
    use = rules or FilterRules.from_config(cfg or {})
    relaxed: set[str] = set()
    dropped_total: dict[str, int] = {}
    kept: list[RetrievedChunk] = []

    for name in (None, *_RELAX_ORDER):
        if name is not None:
            relaxed.add(name)
            LOGGER.warning("filter relaxed: %s (候选不足 min_keep=%d)", name, use.min_keep)
        drops = {"publish_year": 0, "lang": 0, "doc_type": 0, "evidence_level": 0}
        kept = []
        for chunk in chunks:
            reason = _drop_reason(chunk, use, relaxed)
            if reason is None:
                kept.append(chunk)
            else:
                drops[reason] += 1
                dropped_total[reason] = dropped_total.get(reason, 0) + 1
        if len(kept) >= use.min_keep or name == _RELAX_ORDER[-1]:
            break

    if stats is not None:
        stats["relaxed"] = sorted(relaxed)
        stats["dropped"] = dict(dropped_total)
        stats["kept"] = len(kept)
        stats["input"] = len(chunks)
    return kept


def relax(chunks: list[RetrievedChunk], rules: FilterRules | None = None) -> list[RetrievedChunk]:
    """只做放宽计数，不返回放宽明细；给 CLI 打日志用。"""
    return apply_hard_filters(chunks, rules)


def explain_drops(chunks: list[RetrievedChunk], rules: FilterRules | None = None) -> dict[str, int]:
    """按原因统计丢弃数（不过滤，只解释），给报告用。"""
    use = rules or FilterRules()
    out = {"publish_year": 0, "lang": 0, "doc_type": 0, "evidence_level": 0}
    for chunk in chunks:
        reason = _drop_reason(chunk, use, set())
        if reason is not None:
            out[reason] += 1
    return out
