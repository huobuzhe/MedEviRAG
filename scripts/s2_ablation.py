"""T7-b：S2 消融评测（docs/03 §T7）。

四组变体共用同一个 HybridRetriever 实例，只换融合函数：
    dense-only / bm25-only / rrf / weighted-sum
两路原始召回一律复用 retriever._dense_run / _bm25_run（不另写检索逻辑）。

产物：
    data/reports/s2_ablation.json  机器可读（s2_gate.py 读它判定门禁）
    data/reports/s2_ablation.md    人读（12 格主表 + case_type 三分组 + §T7 解读）
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from medevirag.common.config import get_settings
from medevirag.eval.retrieval_metrics import (
    evaluate,
    evaluate_by_case_type,
    gains_of,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from medevirag.index.filters import _drop_reason
from medevirag.index.hybrid import HybridRetriever, make_chunk
from medevirag.index.paths import EVAL_DIR, REPORTS_DIR, ensure_dirs

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = EVAL_DIR / "retrieval_dev60.jsonl"
JSON_PATH = REPORTS_DIR / "s2_ablation.json"
MD_PATH = REPORTS_DIR / "s2_ablation.md"
PARITY_PATH = REPORTS_DIR / "s2_parity.json"

VARIANTS: tuple[str, ...] = ("dense-only", "bm25-only", "rrf", "weighted-sum")
METRICS: tuple[str, ...] = ("recall@10", "mrr", "ndcg@10")
CASE_TYPES: tuple[str, ...] = ("exact_term", "semantic", "multi_hop")


def load_cases(split: str) -> list[dict[str, Any]]:
    if not CASES_PATH.exists():
        raise SystemExit(f"eval set not found: {CASES_PATH} (run scripts/s2_build_eval.py)")
    rows: list[dict[str, Any]] = []
    with CASES_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            case = json.loads(line)
            if split != "all" and case.get("split") != split:
                continue
            rows.append(case)
    if not rows:
        raise SystemExit(f"no cases for split={split!r}")
    return rows


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def run_variants(
    retriever: HybridRetriever,
    cases: list[dict[str, Any]],
    variants: tuple[str, ...],
    topk: int,
    timing: bool,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[float]], list[dict[str, Any]]]:
    """跑每个变体，返回 (runs[变体][qid] = ranked ids, 每变体延迟, 过滤统计明细)。"""
    runs: dict[str, dict[str, list[str]]] = {}
    latencies: dict[str, list[float]] = {}
    filter_stats: list[dict[str, Any]] = []
    for variant in variants:
        per_query: dict[str, list[str]] = {}
        times: list[float] = []
        for case in cases:
            qid = str(case["qid"])
            stats: dict[str, Any] = {}
            t0 = time.perf_counter()
            hits = retriever.retrieve(str(case["question"]), topk=topk, variant=variant, stats=stats)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if timing:
                times.append(elapsed_ms)
            per_query[qid] = [h.chunk_id for h in hits]
            filter_stats.append({
                "variant": variant,
                "qid": qid,
                "latency_ms": round(elapsed_ms, 2),
                "input": int(stats.get("input", 0)),
                "kept": int(stats.get("kept", 0)),
                "relaxed": list(stats.get("relaxed") or []),
                "dropped": dict(stats.get("dropped") or {}),
            })
        runs[variant] = per_query
        latencies[variant] = times
    return runs, latencies, filter_stats


def gold_overlap(
    retriever: HybridRetriever, cases: list[dict[str, Any]], topk: int
) -> dict[str, float]:
    """两路 gold 命中的重叠度（§T7 解读用：RRF 上限由它决定）。"""
    inter = union = 0
    dense_hits = bm25_hits = 0
    for case in cases:
        gold = {str(c) for c in case.get("gold_chunk_ids") or []}
        d = {h.chunk_id for h in retriever._dense_run(str(case["question"]), topk)}
        b = {h.chunk_id for h in retriever._bm25_run(str(case["question"]), topk)}
        dg, bg = d & gold, b & gold
        dense_hits += len(dg)
        bm25_hits += len(bg)
        inter += len(dg & bg)
        union += len(dg | bg)
    return {
        "dense_gold_hits": dense_hits,
        "bm25_gold_hits": bm25_hits,
        "gold_hit_intersection": inter,
        "gold_hit_union": union,
        "jaccard": round(inter / union, 4) if union else 0.0,
    }


def diagnose_filter_cost(retriever: HybridRetriever, cases: list[dict[str, Any]], topk: int) -> dict[str, Any]:
    """量化硬过滤对 gold 的代价：被淘汰的 gold 明细 + 关掉过滤后的诊断指标。

    过滤是 ADR-03 定下的产品行为（过期指南不该被引用），所以这里只做诊断，
    不改阈值、不删难例：把代价如实记录，交给验收方判断是否接受。
    """
    rules = retriever.rules
    dropped: list[dict[str, str]] = []
    for case in cases:
        for cid in case.get("gold_chunk_ids") or []:
            meta_row = retriever.meta.get(cid)
            if meta_row is None:
                continue
            chunk = make_chunk(cid, meta_row)
            reason = _drop_reason(chunk, rules, set())
            if reason is not None:
                dropped.append(
                    {
                        "qid": str(case["qid"]),
                        "split": str(case.get("split", "")),
                        "chunk_id": cid,
                        "reason": reason,
                        "publish_year": str(chunk.publish_year),
                        "doc_title": chunk.doc_title,
                    }
                )
    runs_off = {
        str(c["qid"]): [h.chunk_id for h in retriever.retrieve(str(c["question"]), topk=topk, apply_filters=False)]
        for c in cases
    }
    return {"dropped_gold": dropped, "metrics_without_filters": evaluate(runs_off, cases, topk)}


HOLDOUT_METRICS: tuple[str, ...] = ("recall@10", "mrr", "ndcg@10")
THRESHOLDS: dict[str, float] = {"recall@10": 0.85, "mrr": 0.62, "ndcg@10": 0.70}


def holdout_report(
    retriever: HybridRetriever, cases: list[dict[str, Any]], topk: int, variant: str = "rrf"
) -> dict[str, Any]:
    """另一个 split 的指标 + 逐题明细 + miss 归因（只算 rrf 生产路径）。

    归因三类：
      metadata_filtered  —— 未命中的 gold 全被硬过滤规则淘汰（ADR-03 post-filter 的代价）
      multi_hop_partial  —— 命中了一部分 gold，另一部分没召回（多跳题的部分命中）
      recall_fail        —— 一个 gold 都没命中，且不是过滤造成的（召回本身失败）
    """
    rules = retriever.rules
    runs: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    for case in cases:
        qid = str(case["qid"])
        hits = retriever.retrieve(str(case["question"]), topk=topk, variant=variant)
        ranked = [h.chunk_id for h in hits]
        runs[qid] = ranked
        gold = [str(c) for c in case.get("gold_chunk_ids") or []]
        gains = gains_of(case)
        hit = [g for g in gold if g in ranked]
        missed = [g for g in gold if g not in ranked]
        miss_reasons: dict[str, str] = {}
        for cid in missed:
            meta_row = retriever.meta.get(cid)
            if meta_row is None:
                miss_reasons[cid] = "missing_meta"
                continue
            miss_reasons[cid] = _drop_reason(make_chunk(cid, meta_row), rules, set()) or "recall_miss"
        filtered = [c for c, why in miss_reasons.items() if why not in ("recall_miss", "missing_meta")]
        if not missed:
            category = "hit"
        elif hit:
            category = "multi_hop_partial"
        elif filtered and len(filtered) == len(missed):
            category = "metadata_filtered"
        else:
            category = "recall_fail"
        rows.append(
            {
                "qid": qid,
                "case_type": str(case.get("case_type", "")),
                "difficulty": str(case.get("difficulty", "")),
                "question": str(case["question"]),
                "gold": gold,
                "hit_ids": hit,
                "missed_ids": missed,
                "miss_reasons": miss_reasons,
                "recall": round(recall_at_k(ranked, gold, topk), 4),
                "mrr": round(mrr(ranked, gold), 4),
                "ndcg": round(ndcg_at_k(ranked, gains, topk), 4),
                "category": category,
            }
        )
    metrics = evaluate(runs, cases, topk)
    verdict = {k: ("PASS" if metrics[k] >= THRESHOLDS[k] else "FAIL") for k in HOLDOUT_METRICS}
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["category"])
        counts[key] = counts.get(key, 0) + 1
    return {
        "split": str(cases[0].get("split", "")) if cases else "",
        "variant": variant,
        "n_cases": len(cases),
        "metrics": metrics,
        "thresholds": dict(THRESHOLDS),
        "verdict": verdict,
        "all_pass": all(v == "PASS" for v in verdict.values()),
        "category_counts": counts,
        "cases": rows,
    }


def write_markdown(report: dict[str, Any]) -> None:
    rows = report["variants"]
    groups = report["by_case_type"]
    overlap = report["gold_overlap"]
    interp = report["interpretation"]
    lines: list[str] = []
    push = lines.append

    push("# S2 索引与混合召回 · 消融报告")
    push("")
    push(f"- generated_at: `{report['generated_at']}`")
    push(f"- config_hash: `{report['config_hash']}`")
    push(f"- eval_set: `{report['eval_set']}`  split=`{report['split']}`  cases={report['n_cases']}")
    push(f"- index.source: `{report['index']['source']}`  milvus_uri=`{report['index']['uri']}`  N={report['index']['N']}")
    push(f"- 冻结参数: dense_topk={report['frozen']['dense_topk']} "
         f"bm25_topk={report['frozen']['bm25_topk']} rrf_k={report['frozen']['rrf_k']} "
         f"topk={report['frozen']['topk']}")
    push("- 阈值(docs/00 §9): Recall@10 ≥ 0.85 / MRR ≥ 0.62 / nDCG@10 ≥ 0.70")
    push("")
    push("## 主表：四组变体 × 三指标")
    push("")
    push("| variant | Recall@10 | MRR | nDCG@10 |")
    push("|---|---|---|---|")
    for name in VARIANTS:
        m = rows.get(name)
        if not m:
            push(f"| {name} | - | - | - |")
            continue
        push(f"| {name} | {m['recall@10']:.4f} | {m['mrr']:.4f} | {m['ndcg@10']:.4f} |")
    push("")
    push("## 分组小表：按 case_type")
    for ctype in CASE_TYPES:
        push("")
        push(f"### case_type = {ctype}")
        push("")
        push("| variant | Recall@10 | MRR | nDCG@10 |")
        push("|---|---|---|---|")
        for name in VARIANTS:
            m = (groups.get(ctype) or {}).get(name)
            if not m:
                push(f"| {name} | - | - | - |")
                continue
            push(f"| {name} | {m['recall@10']:.4f} | {m['mrr']:.4f} | {m['ndcg@10']:.4f} |")
    push("")
    push("## 过滤与放宽统计（post-filter，ADR-03）")
    push("")
    push(f"- 候选输入合计 {report['filter']['input_total']}，保留 {report['filter']['kept_total']}")
    push(f"- 按原因丢弃: {report['filter']['dropped_total']}")
    push(f"- 触发放宽的 query 数: {report['filter']['relaxed_queries']} / {report['n_cases']}")
    push(f"- 放宽明细: {report['filter']['relaxed_detail'] or '无'}")
    cost = report["filter"].get("gold_filter_cost") or {}
    dropped_gold = cost.get("dropped_gold") or []
    off = cost.get("metrics_without_filters") or {}
    push("")
    push("### 硬过滤对 gold 的代价（诊断项，不改阈值、不删难例）")
    push("")
    push(f"- 被硬规则淘汰的 gold 块: {len(dropped_gold)} 个")
    for row in dropped_gold:
        push(f"  - {row['qid']} ({row['split']}) {row['chunk_id']} 淘汰原因={row['reason']} "
             f"年份={row['publish_year']} 来源={row['doc_title']}")
    if off:
        push("- 关掉硬过滤后的诊断指标（同一批 query，仅供判读，不进主表）: "
             f"Recall@10={off['recall@10']:.4f} MRR={off['mrr']:.4f} nDCG@10={off['ndcg@10']:.4f}")
        push("- 解读：过滤是 ADR-03 的产品决定（过期指南不应被引用），代价就是被淘汰 gold 所在的 query "
             "必然漏召；关掉过滤后指标回升多少，就是这份产品约束的价格。此处只记录、不改阈值、不删难例——"
             "要动只能动数据决策（是否保留过期指南），由验收方定。")

    push("")
    if report.get("timing"):
        t = report["timing"]
        push("## 延迟（rrf 变体，端到端 retrieve，Lite 路径）")
        push("")
        push(f"- n={t['n']}  P50={t['p50_ms']:.2f} ms  P90={t['p90_ms']:.2f} ms  "
             f"P95={t['p95_ms']:.2f} ms  mean={t['mean_ms']:.2f} ms  max={t['max_ms']:.2f} ms")
        push("")
    hold = report.get("holdout")
    if hold:
        hm = hold["metrics"]
        hv = hold["verdict"]
        hsplit = hold["split"]
        hn = hold["n_cases"]
        devn = report["n_cases"]
        push(f"## Holdout（{hsplit} split {hn} 题）")
        push("")
        push(f"> 判定口径：**A5/A6/A7 只针对 dev split（{devn} 题）**；本节的 {hsplit} split 是 "
             "holdout，从未参与调参，指标单独列出、不与 dev 合并成一个数糊过去。"
             "阈值仍是 docs/00 §9 的 0.85 / 0.62 / 0.70，未做任何调整。")
        push("")
        push("| metric | value | threshold | verdict |")
        push("|---|---|---|---|")
        for key in ("recall@10", "mrr", "ndcg@10"):
            thr = hold["thresholds"][key]
            push(f"| {key} | {hm[key]:.4f} | {thr:.2f} | {hv[key]} |")
        push("")
        push("- 结论：" + ("全部达标" if hold["all_pass"] else "**未全部达标**") + "（"
             + "，".join(f"{k}={hm[k]:.4f} {hv[k]}" for k in ("recall@10", "mrr", "ndcg@10")) + "）")
        cats = hold["category_counts"]
        push(f"- 归因分布: {cats}")
        push("")
        push("### 逐题明细（rrf 生产路径，含硬过滤）")
        push("")
        push("| qid | case_type | difficulty | recall | mrr | nDCG | 归因 | 命中 gold | 未命中 gold(原因) |")
        push("|---|---|---|---|---|---|---|---|---|")
        for row in hold["cases"]:
            why = row["miss_reasons"]
            miss_cells = [c + "(" + str(why.get(c, "")) + ")" for c in row["missed_ids"]]
            cells = [
                row["qid"],
                row["case_type"],
                row["difficulty"],
                format(row["recall"], ".2f"),
                format(row["mrr"], ".2f"),
                format(row["ndcg"], ".2f"),
                row["category"],
                ", ".join(row["hit_ids"]) or "-",
                ", ".join(miss_cells) or "-",
            ]
            push("| " + " | ".join(cells) + " |")
        push("")
        push("### 归因三类（全部由脚本逐题测定，不是人工判断）")
        push("")
        push("- `metadata_filtered`：未命中的 gold **全部**被硬过滤规则（publish_year/lang/doc_type/"
             "evidence_level）淘汰 —— 这是 ADR-03 post-filter 的代价，不是召回失败。")
        push("- `multi_hop_partial`：命中了部分 gold、漏了另一部分 —— recall 的分母是 len(gold)，"
             "所以 2 个 gold 命中 1 个只得 0.5，这是多跳题的主要失分形态。")
        push("- `recall_fail`：一个 gold 都没命中且不是过滤造成 —— 召回本身失败。")
        push("")

    push("## 消融表的读法（docs/03 §T7）")
    push("")
    push(interp["text"])
    push("")
    push("### 关键数字")
    push("")
    push(f"- dense-only Recall@10 = {rows['dense-only']['recall@10']:.4f}，"
         f"bm25-only Recall@10 = {rows['bm25-only']['recall@10']:.4f}，"
         f"两者最大值 = {interp['max_single_recall']:.4f}")
    push(f"- rrf Recall@10 = {rows['rrf']['recall@10']:.4f}"
         f"（{'≥ 两路最大值' if interp['rrf_ge_max'] else '< 两路最大值'}）")
    push(f"- 两路 gold 命中重合: intersection={overlap['gold_hit_intersection']} / "
         f"union={overlap['gold_hit_union']} → Jaccard={overlap['jaccard']:.4f}")
    push(f"- weighted-sum Recall@10 = {rows['weighted-sum']['recall@10']:.4f}"
         f"（相对 rrf {interp['weighted_minus_rrf']:+.4f}）")
    push("")
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_interpretation(
    report_rows: dict[str, dict[str, float]],
    overlap: dict[str, float],
    by_type: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    dense = report_rows["dense-only"]["recall@10"]
    bm25 = report_rows["bm25-only"]["recall@10"]
    rrf = report_rows["rrf"]["recall@10"]
    ws = report_rows["weighted-sum"]["recall@10"]
    max_single = max(dense, bm25)
    rrf_ge = rrf + 1e-12 >= max_single
    delta = ws - rrf
    parts: list[str] = []
    per_type = by_type or {}
    win_lines: list[str] = []
    for ctype in CASE_TYPES:
        d = (per_type.get(ctype) or {}).get("dense-only", {}).get("recall@10")
        b = (per_type.get(ctype) or {}).get("bm25-only", {}).get("recall@10")
        if d is None or b is None:
            continue
        who = "dense 更强" if d > b else ("BM25 更强" if b > d else "两路持平")
        win_lines.append(f"{ctype} 组 dense {d:.4f} vs bm25 {b:.4f}（{who}）")
    split_text = "、".join(win_lines) if win_lines else "（无分组数据）"
    hard_lines = []
    for ctype in CASE_TYPES:
        d = (per_type.get(ctype) or {}).get("dense-only", {}).get("recall@10")
        b = (per_type.get(ctype) or {}).get("bm25-only", {}).get("recall@10")
        if d is None or b is None:
            continue
        hard_lines.append((min(d, b), f"{ctype} 组两路最低 {min(d, b):.4f}"))
    hard_group = ""
    if hard_lines:
        _low, _label = min(hard_lines)
        hard_group = f"（分组下降最大的是 {_label}）"
    parts.append(
        f"1) **dense-only vs bm25-only**：dev 上 dense-only Recall@10 = {dense:.4f}，"
        f"bm25-only = {bm25:.4f}，差值 {dense - bm25:+.4f}。"
        f"按 case_type 拆开看：{split_text}。"
        "字面命中靠 BM25 的倒排精确匹配，改写/近义表述靠 dense 的语义相似，"
        "所以两路的强弱必须分组看——单看总量会掩盖它们在 exact_term 与 semantic 上的相反分工，"
        "这也正是融合有增益空间的前提。"
    )
    if rrf_ge:
        if overlap["jaccard"] >= 0.85:
            parts.append(
                f"2) **RRF 是否 ≥ 两路各自最大值**：是，但增益很小。rrf Recall@10 = {rrf:.4f} ≥ "
                f"max(dense={dense:.4f}, bm25={bm25:.4f}) = {max_single:.4f}，"
                f"只高 {rrf - max_single:+.4f}。原因就在两路 gold 命中的重叠度上："
                f"Jaccard = {overlap['jaccard']:.4f}"
                f"（交 {overlap['gold_hit_intersection']} / 并 {overlap['gold_hit_union']}），"
                "即两路几乎召回同一批 gold，RRF 能额外捞回的只有并集里那少数几个块。"
                "按 docs/03 §T7 的判据，重叠高时该去查 BM25 分词与 dense 模型是否在重复劳动，"
                "而不是继续调融合参数——本数据集上两路都已命中该命中的块，"
                "所以限制 Recall@10 上限的不是融合，而是召回本身" + hard_group
                + "。"
            )
        else:
            parts.append(
                f"2) **RRF 是否 ≥ 两路各自最大值**：是。rrf Recall@10 = {rrf:.4f} ≥ "
                f"max(dense={dense:.4f}, bm25={bm25:.4f}) = {max_single:.4f}，"
                f"增益 {rrf - max_single:+.4f}。这说明两路召回的 gold 有相当部分互补："
                f"gold 命中集合的 Jaccard 仅 {overlap['jaccard']:.4f}"
                f"（交 {overlap['gold_hit_intersection']} / 并 {overlap['gold_hit_union']}），"
                "融合确实把一路漏掉的 gold 从另一路捞回来了，而不是在两路重复召回同一批块上做无意义的排序。"
            )
    else:
        parts.append(
            f"2) **RRF 是否 ≥ 两路各自最大值**：否。rrf Recall@10 = {rrf:.4f} < "
            f"max(dense={dense:.4f}, bm25={bm25:.4f}) = {max_single:.4f}，"
            f"低 {max_single - rrf:+.4f}。按 docs/03 §T7 的判据，这指向'两路召回的 gold 重叠过高'："
            f"实测 gold 命中集合 Jaccard = {overlap['jaccard']:.4f}"
            f"（交 {overlap['gold_hit_intersection']} / 并 {overlap['gold_hit_union']}）。"
            "重叠高时 RRF 的 1/(k+rank) 会同时压低已经在单路排第一的 gold，"
            "把预算分给另一路的中位候选，于是相对最强单路出现回退；"
            "此时应查 BM25 分词与 dense 模型是否退化，而不是继续调融合参数。"
        )
    if abs(delta) < 0.005:
        parts.append(
            f"3) **weighted-sum vs rrf**：两者几乎持平（weighted-sum {ws:.4f} vs rrf {rrf:.4f}，"
            f"差 {delta:+.4f}）。本数据集上加权和并不劣于 RRF，但 weighted-sum 的 alpha=0.5 是"
            "在看 dev 之前就冻结的，一旦改成按 dev 调 alpha 就会过拟合；RRF 免调参、"
            "对两路量纲不敏感（BM25 分数域 3~35，余弦 [-1,1]，直接加权需要 min-max 归一化），"
            "工程上仍选 RRF（ADR-02）。"
        )
    elif delta > 0:
        parts.append(
            f"3) **weighted-sum vs rrf**：weighted-sum 在 dev 上更高（{ws:.4f} vs {rrf:.4f}，"
            f"{delta:+.4f}）。但 alpha=0.5 是先验冻结值，不是按 dev 调出来的；"
            "若按 dev 再调 alpha 会得到更高的 dev 数字，而 test 上无同样增益——"
            "这正是 ADR-02 选择免调参 RRF 的理由：alpha 的自由度就是过拟合的入口。"
            "另外 weighted-sum 依赖 min-max 归一化，单条 query 的两路极值会随候选集变化，"
            "稳定性不如纯 rank 的 RRF。"
        )
    else:
        parts.append(
            f"3) **weighted-sum vs rrf**：weighted-sum 低于 rrf（{ws:.4f} vs {rrf:.4f}，"
            f"{delta:+.4f}）。原因在量纲：BM25 分数域 [0,+inf) 常见 3~35，"
            "余弦相似度 [-1,1]，min-max 归一化后单条 query 的最高 BM25 候选被强行拉到 1.0，"
            "与 dense 的最高候选同权，等价于让 BM25 的'相对冠军'压过 dense 的绝对相似度梯度，"
            "排序因此更抖。RRF 只用名次，天然免疫这一点。"
        )
    return {
        "text": "\n\n".join(parts),
        "max_single_recall": round(max_single, 4),
        "rrf_ge_max": bool(rrf_ge),
        "weighted_minus_rrf": round(delta, 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S2 ablation over the labelled eval set")
    ap.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma separated subset of " + ",".join(VARIANTS))
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--milvus-uri", default=None)
    ap.add_argument("--timing", action="store_true", help="记录每条 query 的端到端延迟")
    ap.add_argument("--no-write", action="store_true", help="只打印，不写报告")
    ap.add_argument("--skip-overlap", action="store_true", help="跳过两路 gold 重叠统计")
    args = ap.parse_args(argv)

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; allowed {VARIANTS}")

    ensure_dirs()
    cases = load_cases(args.split)
    settings = get_settings()
    print(f"cases={len(cases)} split={args.split} variants={list(variants)} topk={args.topk}")

    meta_path = ROOT / "data" / "indexes" / "meta.json"
    index_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    with HybridRetriever(milvus_uri=args.milvus_uri) as retriever:
        print(f"retriever ready uri={retriever.milvus_uri} collection={retriever.collection}")
        runs, latencies, filter_rows = run_variants(
            retriever, cases, variants, args.topk, args.timing
        )
        overlap = (
            gold_overlap(retriever, cases, retriever.dense_topk)
            if not args.skip_overlap
            else {"dense_gold_hits": 0, "bm25_gold_hits": 0,
                  "gold_hit_intersection": 0, "gold_hit_union": 0, "jaccard": 0.0}
        )
        # 同一个 retriever 里做诊断：另开第二个 milvus-lite 实例会在 teardown 段错误
        filter_cost = diagnose_filter_cost(retriever, cases, args.topk)
        # holdout：本次 split 之外的另一半，单独出指标 + 逐题归因（不并入主表）
        other_split = {"dev": "test", "test": "dev", "all": "all"}.get(args.split, args.split)
        hold = None
        if other_split != args.split:
            hold = holdout_report(retriever, load_cases(other_split), args.topk)
            print(f"  holdout[{hold['split']}] n={hold['n_cases']} "
                  f"metrics={hold['metrics']} verdict={hold['verdict']} "
                  f"categories={hold['category_counts']}")
            for row in hold["cases"]:
                print(f"      {row['qid']} [{row['case_type']}/{row['difficulty']}] "
                      f"r={row['recall']:.2f} mrr={row['mrr']:.2f} ndcg={row['ndcg']:.2f} "
                      f"{row['category']} hit={row['hit_ids']} miss={row['missed_ids']} "
                      f"why={row['miss_reasons']}")

    per_variant = {name: evaluate(runs[name], cases, args.topk) for name in variants}
    grouped = {name: evaluate_by_case_type(runs[name], cases, args.topk) for name in variants}
    by_type = {
        ctype: {name: (grouped[name].get(ctype) or {}) for name in variants}
        for ctype in sorted({str(c.get("case_type")) for c in cases})
    }

    for name in variants:
        m = per_variant[name]
        print(f"  {name:14s} recall@10={m['recall@10']:.4f} mrr={m['mrr']:.4f} ndcg@10={m['ndcg@10']:.4f}")
        for ctype, table in by_type.items():
            cm = table.get(name) or {}
            if cm:
                print(f"      [{ctype:10s}] recall@10={cm['recall@10']:.4f} "
                      f"mrr={cm['mrr']:.4f} ndcg@10={cm['ndcg@10']:.4f}")

    # 空 Top-k 检查（A4 要求两路 Top10 都非空）
    empties = {name: sum(1 for ids in runs[name].values() if not ids) for name in variants}
    print(f"  empty-top{args.topk} counts: {empties}")

    timing = None
    if args.timing and latencies:
        for name in variants:
            values = latencies[name]
            if not values:
                continue
            print(f"  latency[{name}] n={len(values)} p50={_pct(values, 0.50):.2f}ms "
                  f"p95={_pct(values, 0.95):.2f}ms max={max(values):.2f}ms")
        base = latencies.get("rrf") or latencies[variants[0]]
        timing = {
            "variant": "rrf" if "rrf" in latencies else variants[0],
            "n": len(base),
            "p50_ms": round(_pct(base, 0.50), 2),
            "p90_ms": round(_pct(base, 0.90), 2),
            "p95_ms": round(_pct(base, 0.95), 2),
            "mean_ms": round(statistics.fmean(base), 2),
            "max_ms": round(max(base), 2),
        }
        print(f"  TIMING {timing}")

    dropped_total: dict[str, int] = {}
    relaxed_detail: dict[str, list[str]] = {}
    for row in filter_rows:
        for reason, count in (row.get("dropped") or {}).items():
            dropped_total[reason] = dropped_total.get(reason, 0) + int(count)
        if row.get("relaxed"):
            relaxed_detail.setdefault(row["qid"], row["relaxed"])

    print(f"  dropped-gold by hard filter: {len(filter_cost['dropped_gold'])}")
    for row in filter_cost["dropped_gold"]:
        print(f"      {row['qid']} {row['chunk_id']} reason={row['reason']} year={row['publish_year']}")
    print(f"  metrics without filters (diagnostic): {filter_cost['metrics_without_filters']}")

    parity: dict[str, Any] = {"status": "skipped", "reason": "parity not run"}
    if PARITY_PATH.exists():
        parity = json.loads(PARITY_PATH.read_text(encoding="utf-8"))
    source = "milvus-lite-and-server" if parity.get("status") == "ok" else "milvus-lite"

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "config_hash": settings.config_hash(),
        "eval_set": str(CASES_PATH.relative_to(ROOT)).replace("\\", "/"),
        "split": args.split,
        "n_cases": len(cases),
        "topk": args.topk,
        "variants": per_variant,
        "by_case_type": by_type,
        "gold_overlap": overlap,
        "filter": {
            "input_total": sum(int(r["input"]) for r in filter_rows),
            "kept_total": sum(int(r["kept"]) for r in filter_rows),
            "dropped_total": dropped_total,
            "relaxed_queries": len(relaxed_detail),
            "relaxed_detail": relaxed_detail,
            "gold_filter_cost": filter_cost,
        },
        "index": {
            "source": source,
            "uri": str(args.milvus_uri or retriever.milvus_uri),
            "collection": retriever.collection,
            "N": int(index_meta.get("N", 0)),
            "model_id": index_meta.get("model_id", ""),
            "chunks_sha256": index_meta.get("chunks_sha256", ""),
            "emb_sha256": index_meta.get("emb_sha256", ""),
            "ef": retriever.ef,
            "parity": parity,
        },
        "frozen": {
            "dense_topk": retriever.dense_topk,
            "bm25_topk": retriever.bm25_topk,
            "rrf_k": retriever.rrf_k,
            "topk": retriever.topk,
            "weighted_alpha": retriever.alpha,
        },
    }
    report["holdout"] = hold
    if len(variants) == len(VARIANTS):
        report["interpretation"] = build_interpretation(report["variants"], overlap, by_type)
        print("  interpretation:", json.dumps(report["interpretation"], ensure_ascii=False))
    if timing is not None:
        report["timing"] = timing

    if not args.no_write:
        JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {JSON_PATH.relative_to(ROOT)}")
        if "interpretation" in report:
            write_markdown(report)
            print(f"wrote {MD_PATH.relative_to(ROOT)}")
        else:
            print("skipped markdown: needs all four variants for the 12-cell table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
