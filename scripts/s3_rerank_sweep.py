"""T2 重排收益/延迟扫描（docs/04 §T2，A2/A3/A4 门禁数据源）。

对比 B0（无重排，取 RRF Top8）与 Bk（RRF TopK → 重排 → Top8），三档 K=10/20/50。
判定：ΔMRR < 0.02 且 T > 200ms → "只增延迟"；取 ΔMRR/(T/1000) 最大者为默认 K；
ΔMRR(B3)-ΔMRR(B2) < 0.01 → 收敛到 K=20。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"


def _p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return float(s[idx])


def _mrr_recall(ranked_ids: list[str], gold: set[str], k: int = 8) -> tuple[float, float]:
    top = ranked_ids[:k]
    for i, cid in enumerate(top, start=1):
        if cid in gold:
            return 1.0 / i, 1.0
    return 0.0, 0.0


def sweep(cases: list[dict], retriever, reranker, ks=(10, 20, 50), topn: int = 8) -> dict:
    by_k: dict[int, dict] = {}
    base = {"mrr@8": [], "recall@8": [], "latency_ms": []}
    for k in ks:
        by_k[k] = {"mrr@8": [], "recall@8": [], "latency_ms": []}

    for case in cases:
        gold = set(str(x) for x in (case.get("gold_chunk_ids") or []))
        cands = retriever.retrieve(case["question"], topk=50)

        base_ids = [c.chunk_id for c in cands[:topn]]
        m, r = _mrr_recall(base_ids, gold, topn)
        base["mrr@8"].append(m)
        base["recall@8"].append(r)

        for k in ks:
            pool = cands[:k]
            t0 = time.perf_counter()
            top = reranker.rerank(case["question"], pool, top_n=topn)
            ms = (time.perf_counter() - t0) * 1000.0
            ids = [c.chunk_id for c in top]
            m, r = _mrr_recall(ids, gold, topn)
            by_k[k]["mrr@8"].append(m)
            by_k[k]["recall@8"].append(r)
            by_k[k]["latency_ms"].append(ms)

    def agg(d: dict) -> dict:
        return {
            "mrr@8": round(statistics.fmean(d["mrr@8"]), 4) if d["mrr@8"] else 0.0,
            "recall@8": round(statistics.fmean(d["recall@8"]), 4) if d["recall@8"] else 0.0,
            "latency_p50_ms": round(_p95(d["latency_ms"]) if False else (statistics.median(d["latency_ms"]) if d["latency_ms"] else 0.0), 1),
            "latency_p95_ms": round(_p95(d["latency_ms"]), 1),
        }

    base_agg = agg(base)
    base_agg.pop("latency_p50_ms", None)
    result = {"n_cases": len(cases), "topn": topn,
              "baseline_no_rerank": base_agg, "by_k": {}}
    for k in ks:
        a = agg(by_k[k])
        a["delta_mrr"] = round(a["mrr@8"] - base_agg["mrr@8"], 4)
        result["by_k"][str(k)] = a
    return result


def decide(result: dict, ks=(10, 20, 50)) -> dict:
    by_k = result["by_k"]
    ranking = []
    for k in ks:
        a = by_k[str(k)]
        eff = a["delta_mrr"] / max(1e-6, a["latency_p95_ms"] / 1000.0)
        ranking.append((eff, k, a))
    ranking.sort(reverse=True)
    best_eff, best_k, best = ranking[0]
    notes = []
    if best["delta_mrr"] < 0.02 and best["latency_p95_ms"] > 200:
        notes.append("只增延迟")
    if len(ks) >= 3 and by_k[str(ks[2])]["delta_mrr"] - by_k[str(ks[1])]["delta_mrr"] < 0.01:
        best_k = ks[1]
        notes.append("K=" + str(ks[1]) + " 与 K=" + str(ks[2]) + " 收益差 < 0.01，收敛到较小档")
    selected = by_k[str(best_k)]
    approved = selected["delta_mrr"] >= 0.10
    return {"chosen_k": best_k, "delta_mrr": selected["delta_mrr"],
            "latency_p95_ms": selected["latency_p95_ms"],
            "rerank_approved": approved, "notes": notes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=str(ROOT / "data" / "eval" / "retrieval_dev60.jsonl"))
    ap.add_argument("--ks", default="10,20,50")
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 为全量）")
    ap.add_argument("--out", default=str(REPORTS / "s3_rerank_sweep.json"))
    args = ap.parse_args()

    from medevirag.index.hybrid import HybridRetriever
    from medevirag.rag.reranker import Reranker

    with open(args.dev, encoding="utf-8") as fh:
        cases = [json.loads(ln) for ln in fh if ln.strip()]
    if args.limit:
        cases = cases[: args.limit]
    ks = tuple(int(x) for x in args.ks.split(","))

    with HybridRetriever() as retriever:
        reranker = Reranker()
        result = sweep(cases, retriever, reranker, ks=ks, topn=args.topn)
    verdict = decide(result, ks)
    result["verdict"] = verdict
    REPORTS.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# S3 重排档位扫描（K=10/20/50）", "",
          "基线（无重排，RRF Top8）：MRR@8=" + str(result["baseline_no_rerank"]["mrr@8"])
          + "  Recall@8=" + str(result["baseline_no_rerank"]["recall@8"]), "",
          "| K | MRR@8 | Recall@8 | ΔMRR | P50(ms) | P95(ms) |", "|---|---|---|---|---|---|"]
    for k in ks:
        a = result["by_k"][str(k)]
        md.append("| " + str(k) + " | " + str(a["mrr@8"]) + " | " + str(a["recall@8"]) + " | "
                  + str(a["delta_mrr"]) + " | " + str(a["latency_p50_ms"]) + " | " + str(a["latency_p95_ms"]) + " |")
    md += ["", "## 结论", "",
           "- 选定 K = **" + str(verdict["chosen_k"]) + "**（ΔMRR=" + str(verdict["delta_mrr"])
           + "，P95=" + str(verdict["latency_p95_ms"]) + "ms）",
           "- 重排是否采用：" + ("**采用**（ΔMRR ≥ 0.10）" if verdict["rerank_approved"] else "**R2 否决**（ΔMRR < 0.10，node_rerank 直通 Top8，留档不凑指标）")]
    if verdict["notes"]:
        md.append("- 判定备注：" + "；".join(verdict["notes"]))
    Path(args.out).with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"chosen_k": verdict["chosen_k"], "delta_mrr": verdict["delta_mrr"],
                      "p95": verdict["latency_p95_ms"], "approved": verdict["rerank_approved"],
                      "n": result["n_cases"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
