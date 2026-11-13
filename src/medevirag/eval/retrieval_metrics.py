"""T7：检索指标（docs/03 §T7）。

一个必须说清的点：recall_at_k 的分母是 len(gold)，不是 k。
多跳题有 2 个 gold、命中 1 个时得分必须是 0.5；用 k 做分母会算成 0.1（或把命中率虚高）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["evaluate", "mrr", "ndcg_at_k", "recall_at_k"]


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    """命中数 / gold 数（分母是 len(gold)）。"""
    golds = {str(g) for g in gold}
    if not golds:
        return 0.0
    top = {str(c) for c in list(ranked)[: int(k)]}
    return len(golds & top) / len(golds)


def mrr(ranked: Sequence[str], gold: Sequence[str]) -> float:
    """第一个命中名次的倒数；没有命中返回 0.0。"""
    golds = {str(g) for g in gold}
    if not golds:
        return 0.0
    for rank, cid in enumerate(ranked, start=1):
        if str(cid) in golds:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int = 10) -> float:
    """DCG/IDCG，gains 里 gold=2、near_miss=1，未标注=0。"""
    top = [str(c) for c in list(ranked)[: int(k)]]
    dcg = 0.0
    for i, cid in enumerate(top, start=1):
        g = float(gains.get(cid, 0.0))
        if g:
            dcg += (2.0**g - 1.0) / math.log2(i + 1.0)
    ideal = sorted((float(v) for v in gains.values()), reverse=True)[: int(k)]
    idcg = 0.0
    for i, g in enumerate(ideal, start=1):
        if g:
            idcg += (2.0**g - 1.0) / math.log2(i + 1.0)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def gains_of(case: Mapping[str, Any]) -> dict[str, float]:
    """一条标注题的增益表：gold=2，near_miss=1。"""
    out: dict[str, float] = {}
    for cid in case.get("gold_chunk_ids") or []:
        out[str(cid)] = 2.0
    for cid in case.get("near_miss_ids") or []:
        out.setdefault(str(cid), 1.0)
    return out


def evaluate(
    runs: Mapping[str, Sequence[str]],
    cases: Sequence[Mapping[str, Any]],
    k: int = 10,
) -> dict[str, float]:
    """对一批题算 Recall@k / MRR / nDCG@k 的均值（保留 4 位）。"""
    if not cases:
        return {"recall@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0}
    n = 0
    sum_r = sum_m = sum_n = 0.0
    for case in cases:
        qid = str(case["qid"])
        ranked = [str(c) for c in runs.get(qid, [])]
        gold = [str(c) for c in case.get("gold_chunk_ids") or []]
        if not gold:
            continue
        n += 1
        sum_r += recall_at_k(ranked, gold, k)
        sum_m += mrr(ranked, gold)
        sum_n += ndcg_at_k(ranked, gains_of(case), k)
    if n == 0:
        return {"recall@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0}
    return {
        f"recall@{k}": round(sum_r / n, 4),
        "mrr": round(sum_m / n, 4),
        f"ndcg@{k}": round(sum_n / n, 4),
    }


def evaluate_by_case_type(
    runs: Mapping[str, Sequence[str]],
    cases: Sequence[Mapping[str, Any]],
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """按 case_type 分组算（消融报告的三张分组小表）。"""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        groups.setdefault(str(case.get("case_type") or "unknown"), []).append(case)
    return {name: evaluate(runs, items, k) for name, items in sorted(groups.items())}
