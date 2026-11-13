# S5 T8：00 §9 全部指标（docs/06 §T8）。
# 纪律：空分母一律 None（渲染 n/a），绝不填 0；upper-bound 指标显式标注方向。

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from statistics import mean
from typing import Any


def recall_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int = 10) -> float:
    # 主口径：gold 集合在 Top-K 的覆盖率。分母是 len(gold) 不是 k。
    golds = {str(g) for g in gold}
    if not golds:
        raise ValueError("gold_chunk_ids 为空：每条题必须至少 1 个 gold")
    top = {str(c) for c in list(retrieved)[:k]}
    return len(golds & top) / len(golds)


def hit_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int = 10) -> float:
    golds = {str(g) for g in gold}
    return 1.0 if golds & set(list(retrieved)[:k]) else 0.0


def mrr(gold: Sequence[str], retrieved: Sequence[str]) -> float:
    golds = {str(g) for g in gold}
    for rank, cid in enumerate(retrieved, start=1):
        if str(cid) in golds:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(graded: dict[str, int], retrieved: Sequence[str], k: int = 10) -> float:
    # graded: chunk_id -> rel(2/1/0)。2=gold，1=同父兄弟块，0=其他。
    dcg = sum((2 ** graded.get(str(c), 0) - 1) / math.log2(i + 2)
              for i, c in enumerate(list(retrieved)[:k]))
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def citation_landing_rate(dangling: int, total: int) -> float:
    if total == 0:
        raise ValueError("答案里没有引用：落点正确率无定义")
    return (total - dangling) / total


def sentence_support_rate(sentences: Sequence[dict[str, Any]]) -> float:
    # 只统计含引用的句子；一条引用都没有 -> ValueError，由 runner 记 fail
    cited = [s for s in sentences if s.get("cited_chunk_ids")]
    if not cited:
        raise ValueError("无含引用的句子")
    return sum(1 for s in cited if s.get("landing_ok") and s.get("entailed")) / len(cited)


def refusal_recall(rows: Sequence[dict[str, Any]]) -> float:
    den = sum(1 for r in rows if r["must_refuse"])
    if den == 0:
        raise ValueError("没有应拒答题")
    return sum(1 for r in rows if r["must_refuse"] and r["status"] == "refused") / den


def over_refusal_rate(rows: Sequence[dict[str, Any]]) -> float:
    den = sum(1 for r in rows if not r["must_refuse"])
    if den == 0:
        raise ValueError("没有可答题")
    return sum(1 for r in rows if not r["must_refuse"] and r["status"] == "refused") / den


def refusal_reason_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    # S4 枚举先映射回 docs 4 类 refuse_kind 再比对（词表口径对齐）
    # docs §6.1：分母 = 应拒且被判 refused 的题（误拒题不参与 reason 比对，
    # 否则 FakeLLM 全误拒时 reason_acc 被 None gold 拉成 0，口径失真）
    refused = [r for r in rows if r["status"] == "refused" and r.get("must_refuse")]
    if not refused:
        raise ValueError("没有应拒且被拒的样本")
    hit = sum(1 for r in refused
              if REASON_TO_KIND.get(str(r.get("refusal_reason")), r.get("refusal_reason"))
              == r.get("gold_refuse_kind"))
    return hit / len(refused)


def bootstrap_ci(flags: Sequence[bool], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 7) -> tuple[float, float]:
    rng, n = random.Random(seed), len(flags)
    if n == 0:
        return (float("nan"), float("nan"))
    means = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def expected_calibration_error(confs: Sequence[float], corrects: Sequence[bool],
                               n_bins: int = 10) -> float:
    # S4 遗留的正式标定：ECE = Σ (n_b/N)·|acc_b − conf_b|
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, ok in zip(confs, corrects, strict=False):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, ok))
    total = len(confs)
    if total == 0:
        return 0.0
    return sum(len(b) / total * abs(sum(ok for _, ok in b) / len(b)
                                     - sum(c for c, _ in b) / len(b))
               for b in bins if b)


def mean_or_none(xs: Sequence[float]) -> float | None:
    # 空分母返回 None（渲染成 n/a），绝不填 0
    return mean(xs) if xs else None


# S4 RefusalReason(15) -> docs/06 refuse_kind(4) 的映射（reason_accuracy 的分母口径）
REASON_TO_KIND = {
    "off_domain": "out_of_scope",
    "uncovered_topic": "out_of_scope",
    "non_evidence_request": "out_of_scope",
    "unsafe_action_request": "out_of_scope",
    "stale_evidence": "stale",
    "individualized_dosage": "personalized",
    "individualized_prescription": "personalized",
    "individualized_diagnosis": "personalized",
    "special_population": "personalized",
    "emergency": "personalized",
    "low_relevance": "insufficient_evidence",
    "no_margin": "insufficient_evidence",
    "citation_ungrounded": "insufficient_evidence",
    "low_support": "insufficient_evidence",
    "evidence_level_unknown": "insufficient_evidence",
}

HIGHER_IS_BETTER = {
    "recall_at_10": True, "mrr": True, "ndcg_at_10": True,
    "citation_landing_rate": True, "sentence_support_rate": True,
    "faithfulness": True, "refusal_recall": True,
    "refusal_reason_accuracy": True,
    "over_refusal_rate": False,      # 上限型：越低越好
    "judge_error_rate": False,
}


def compute_all(rows: list[dict[str, Any]], cfg: Any) -> dict[str, Any]:
    # rows 每题一条（runner 产出）。judge_error 的题不进任何分母，单列。
    ok_rows = [r for r in rows if not r.get("judge_error")]
    m: dict[str, Any] = {"n": len(ok_rows), "n_total": len(rows),
               "judge_error_rate": round(1 - len(ok_rows) / max(1, len(rows)), 4)}
    # 检索指标只统计有 gold 的题（应拒题 gold 为空，强行算会 ValueError 且无意义）
    ans_rows = [r for r in ok_rows if r["gold_chunk_ids"]]
    m["recall_at_10"] = mean_or_none([recall_at_k(r["gold_chunk_ids"], r["retrieved"], 10) for r in ans_rows])
    m["hit_at_10"] = mean_or_none([hit_at_k(r["gold_chunk_ids"], r["retrieved"], 10) for r in ans_rows])
    m["mrr"] = mean_or_none([mrr(r["gold_chunk_ids"], r["retrieved"]) for r in ans_rows])
    m["ndcg_at_10"] = mean_or_none([ndcg_at_k(r["graded"], r["retrieved"], 10) for r in ans_rows])
    m["faithfulness"] = mean_or_none([r["judge"]["faithfulness"] for r in ok_rows
                                      if "faithfulness" in r["judge"]])
    m["evidence_recall"] = mean_or_none([r["judge"]["evidence_recall"] for r in ok_rows
                                         if "evidence_recall" in r["judge"]])
    m["refusal_dim"] = mean_or_none([r["judge"]["refusal"] for r in ok_rows
                                     if "refusal" in r["judge"]])
    land = [
        citation_landing_rate(r["dangling_citation_count"], r["total_citation_count"])
        for r in ok_rows if r["total_citation_count"] > 0 and not r["must_refuse"]]
    m["citation_landing_rate"] = mean_or_none(land)
    sup = [
        sentence_support_rate(r["sentences"])
        for r in ok_rows if any(s.get("cited_chunk_ids") for s in r["sentences"])
                    and not r["must_refuse"]]
    m["sentence_support_rate"] = mean_or_none(sup)
    if any(r["must_refuse"] for r in ok_rows):
        m["refusal_recall"] = refusal_recall(ok_rows)
    else:
        m["refusal_recall"] = None
    if any(not r["must_refuse"] for r in ok_rows):
        m["over_refusal_rate"] = over_refusal_rate(ok_rows)
    else:
        m["over_refusal_rate"] = None
    refused_rows = [r for r in ok_rows if r["status"] == "refused"]
    if refused_rows:
        m["refusal_reason_accuracy"] = refusal_reason_accuracy(ok_rows)
    else:
        m["refusal_reason_accuracy"] = None
    m["ece"] = expected_calibration_error([float(r.get("confidence", 0.0)) for r in ok_rows],
                                          [bool(r.get("correct")) for r in ok_rows])
    m["partial_rate"] = mean_or_none([1.0 if r["status"] == "partial" else 0.0 for r in ok_rows])
    # CI 与门禁判定
    for key, higher in HIGHER_IS_BETTER.items():
        if key not in m or m[key] is None or key == "judge_error_rate":
            continue
        flags = [bool(r["metric_flags"][key]) for r in ok_rows if key in r.get("metric_flags", {})]
        if flags:
            lo, hi = bootstrap_ci(flags)
            m[f"{key}_ci95"] = [round(lo, 3), round(hi, 3)]
        th = getattr(cfg.eval.thresholds, key, None)
        if th is not None:
            m[f"{key}_pass"] = (m[key] >= th) if higher else (m[key] <= th)
    return m

