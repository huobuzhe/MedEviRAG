"""T2 复合置信度（docs/05 §T2）：rank_norm / citation_coverage / evidence_weight / margin_norm。

conf = 0.35*rank + 0.30*citation + 0.20*evidence + 0.15*margin，四分量与 score 均在 [0,1]。

★ S4 偏差 D1（改这里之前必须知道）：S3 已按 R2 否决重排（retrieval.yaml rerank.enabled=false，
实测 dMRR=0.0557 < 0.10），候选的 rerank_score 恒为 0.0。docs 假定的 rerank logit 直接套用
会让 rank/margin 全部退化成常数。故 compute_confidence 在 rerank_applied=False 时改用
dense 余弦作排序信号，Platt 参数在 dev60（60 题，gold_top1=44，常数基线 logloss 0.5799）上
网格标定（精度只到一档网格；样本仅 60 条，不做连续拟合，防阈值漂移）：
  rank   : Platt(dense_top1, mid=0.670, scale=0.060)  -> logloss 0.5029
  margin : Platt(rank_norm(top1)-rank_norm(top5), mid=0.1385, scale=0.1693) -> logloss 0.5138
         （优于裸 dense gap 的 0.6330 与 mean-top5 gap 的 0.5664，故取归一化后分差口径）
若将来重排启用（rerank_applied=True），rank/margin 自动回到 docs 的 rerank logit 参数
（mid=2.0/scale=1.5 与 mid=1.0/scale=1.2），其余分量不变。

为什么不能拿向量相似度直接当置信度（docs §T2 四条，别推翻）：
  1. 相似度衡量"主题接近"，不是"能回答"；
  2. mean pooling 有 length bias / hubness，长 chunk 的相似度基线系统性偏高；
  3. 相似度跨 query 不可比，单一全局阈值必然一边狂拒一边狂放；
  4. 只有 cross-encoder 见过 query 与 doc 的交互。
本项目的折中：把 dense 信号用 gold 命中做二分类标定后再用（Platt 参数来自最小化 logloss），
使它成为一个可解释的概率，而不是裸相似度。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# docs/05 §T2 的等级权重
EVIDENCE_WEIGHT: dict[str, float] = {"A": 1.00, "B": 0.75, "C": 0.50, "D": 0.25, "unknown": 0.45}
UNKNOWN_LEVEL = 0.45
WORST_LOGIT = -11.0   # 候选不足 5 条时 top5 兜底；别改成 r1 自己（D=0 会白送 0.045 分）


def _platt(x: float, mid: float, scale: float) -> float:
    """把无界 logit 压到 (0,1)。scale 越小越陡。"""
    return 1.0 / (1.0 + math.exp(-(x - mid) / scale))


platt = _platt  # 标定脚本与夹具 builders 复用同一实现，避免两套公式漂移


def rank_norm(top1_score: float, mid: float = 2.0, scale: float = 1.5) -> float:
    return _platt(top1_score, mid, scale)


def margin_norm(top1_score: float, top5_score: float,
                mid: float = 1.0, scale: float = 1.2) -> float:
    return _platt(top1_score - top5_score, mid, scale)


@dataclass(frozen=True)
class SentenceSupport:
    """docs/05 §2.2 契约；S3 无此类型，由 refusal.py 适配产出。"""

    text: str
    cited_chunk_ids: tuple[str, ...] = ()
    landing_ok: bool = True
    entailed: bool = False


@dataclass(frozen=True)
class CitationReport:
    """docs/05 §2.2 契约的适配版（偏差 D2）。

    S3 的 verifier 产出 VerifyResult/SentenceCheck；这里保留 docs 的字段名，
    另加 sentences_total（断言句总数，已排除模板句）供 citation_coverage 作分母。
    """

    sentences: tuple[SentenceSupport, ...] = ()
    dangling_citation_count: int = 0
    total_citation_count: int = 0
    sentences_total: int = 0


@dataclass(frozen=True)
class ConfidenceBreakdown:
    rank_norm: float
    citation_coverage: float
    evidence_weight: float
    margin_norm: float
    score: float
    unknown_level_ratio: float     # 供 R-17
    year_unknown_ratio: float      # 供 R-07
    margin_gap: float = 0.0        # 供 R-14：rn(top1)-rn(top5)，见偏差 D10

    def as_dict(self) -> dict[str, float]:
        return {
            "rank_norm": round(self.rank_norm, 4),
            "citation_coverage": round(self.citation_coverage, 4),
            "evidence_weight": round(self.evidence_weight, 4),
            "margin_norm": round(self.margin_norm, 4),
            "score": round(self.score, 4),
        }


def citation_coverage(report: CitationReport) -> float:
    """断言句中被有效引用覆盖的比例；落点校验失败的句子不算覆盖。

    分母优先用 sentences_total（调用方已排除"建议就医"这类模板句），
    缺省退回 len(sentences)。这是 docs §T2 的硬要求。
    """
    sentences = [s for s in report.sentences if s.text.strip()]
    denom = report.sentences_total or len(sentences)
    if not denom:
        return 0.0
    covered = sum(1 for s in sentences if s.landing_ok and s.cited_chunk_ids)
    return covered / denom


def evidence_weight(cited_levels: Sequence[str]) -> float:
    """按引用次数加权的证据等级均值；无引用直接 0（没有证据就没有置信度）。"""
    if not cited_levels:
        return 0.0
    return sum(EVIDENCE_WEIGHT.get(lv, UNKNOWN_LEVEL) for lv in cited_levels) / len(cited_levels)


def compute_confidence(
    candidates: Sequence[Any],
    cited_chunk_ids: Sequence[str],
    report: CitationReport,
    weights: dict[str, float],
    platt_cfg: dict[str, dict[str, float]],
    *,
    rerank_applied: bool = False,
) -> ConfidenceBreakdown:
    """单条答案的复合置信度。candidates 已按名次升序（第 0 个是 Top1）。

    信号口径按 rerank_applied 切换（偏差 D1）：启用取 rerank_score，未启用取 dense_score；
    两套 Platt 参数都从配置读，未启用时走 rank_dense / margin_dense 子键。
    """
    if not candidates:
        return ConfidenceBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0)

    def _sig(c: Any) -> float:
        if rerank_applied:
            return float(getattr(c, "rerank_score", 0.0))
        return float(getattr(c, "dense_score", 0.0))

    rcfg = platt_cfg["rank"] if rerank_applied else platt_cfg.get("rank_dense", platt_cfg["rank"])
    s1 = _sig(candidates[0])
    s5 = _sig(candidates[4]) if len(candidates) >= 5 else WORST_LOGIT

    rn = rank_norm(s1, **rcfg)
    if rerank_applied:
        mn = margin_norm(s1, s5, **platt_cfg["margin"])
    else:
        mcfg = platt_cfg.get("margin_dense", platt_cfg["margin"])
        mn = _platt(rn - rank_norm(s5, **rcfg), mcfg["mid"], mcfg["scale"])
    cc = citation_coverage(report)

    by_id = {c.chunk_id: c for c in candidates}
    levels = [by_id[cid].evidence_level for cid in cited_chunk_ids if cid in by_id]
    ew = evidence_weight(levels)

    score = (weights["rank"] * rn + weights["citation"] * cc
             + weights["evidence"] * ew + weights["margin"] * mn)

    top8 = list(candidates[:8])
    unknown_level_ratio = sum(1 for c in top8 if c.evidence_level == "unknown") / len(top8)
    year_unknown_ratio = sum(1 for c in top8 if not c.publish_year) / len(top8)

    margin_gap = rn - rank_norm(s5, **rcfg)
    return ConfidenceBreakdown(rn, cc, ew, mn, min(1.0, max(0.0, score)),
                               unknown_level_ratio, year_unknown_ratio, margin_gap)
