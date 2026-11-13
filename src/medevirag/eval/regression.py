# S5 T10：回归门禁（docs/06 §T10）。双 run 逐题比对 + 阈值/跌幅双门禁。

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from medevirag.common.config import get_settings

ROOT = Path(__file__).resolve().parents[3]

GATED = ("recall_at_10", "mrr", "ndcg_at_10", "citation_landing_rate",
         "sentence_support_rate", "faithfulness", "refusal_recall",
         "over_refusal_rate", "refusal_reason_accuracy")


def load_run(path: Path) -> dict[str, Any]:
    d = dict(json.loads(path.read_text(encoding="utf-8")))
    d["by_case"] = {str(c["case_id"]): c for c in d["cases"]}
    return d


def attribute(cid: str, b: dict[str, Any], c: dict[str, Any]) -> str:
    # 三分类归因：用可观测差异定，不猜
    if b["retrieved"][:10] != c["retrieved"][:10]:
        return "retrieval: Top-10 顺序或成员变了"
    if (b["total_citation_count"] - b["dangling_citation_count"]) != \
       (c["total_citation_count"] - c["dangling_citation_count"]):
        return "generation: 引用落点数变了"
    if b["status"] != c["status"]:
        return "refusal: " + str(b["status"]) + " -> " + str(c["status"])
    return "judge: 检索与生成产物一致，差异来自判分"


def compare(base: dict[str, Any], cur: dict[str, Any], cfg: Any) -> tuple[list[dict[str, Any]], list[str]]:
    problems: list[str] = []
    if base["metrics"]["frozen_set_sha256"] != cur["metrics"]["frozen_set_sha256"]:
        problems.append("FREEZE_MISMATCH: 两次 run 用的评测集指纹不同，不可比较")

    # 1) 绝对阈值门禁（上限型方向相反）
    for key in GATED:
        v = cur["metrics"].get(key)
        th = getattr(cfg.eval.thresholds, key, None)
        if v is None or th is None:
            continue
        higher = key != "over_refusal_rate"
        if (higher and v < th) or (not higher and v > th):
            problems.append("THRESHOLD_FAIL " + key + "=" + str(v)
                            + " 目标" + (">=" if higher else "<=") + str(th))

    # 2) 相对跌幅门禁（分两档：噪声带 vs 硬跌）
    delta = []
    for key in GATED:
        b = base["metrics"].get(key)
        c = cur["metrics"].get(key)
        if b is None or c is None:
            continue
        d = round(c - b, 4)
        better = (d >= 0) if key != "over_refusal_rate" else (d <= 0)
        if not better and abs(d) > cfg.eval.regression.hard_drop:
            problems.append("HARD_DROP " + key + ": " + str(b) + " -> " + str(c)
                            + " (delta=" + str(d) + ")")
        delta.append({"metric": key, "base": b, "cur": c, "delta": d,
                      "status": "ok" if better else "worse"})

    # 3) 逐题 diff：只挑方向变化的
    diff = []
    for cid in sorted(set(base["by_case"]) & set(cur["by_case"])):
        b = base["by_case"][cid]
        c = cur["by_case"][cid]
        if b["correct"] == c["correct"]:
            continue
        diff.append({
            "case_id": cid,
            "direction": "regressed" if b["correct"] else "fixed",
            "base": b["judge"], "cur": c["judge"],
            "base_status": b["status"], "cur_status": c["status"],
            "base_refusal_reason": b.get("refusal_reason"),
            "cur_refusal_reason": c.get("refusal_reason"),
            "hint": attribute(cid, b, c),
        })
    return diff, problems


def main(base_id: str, cur_id: str) -> int:
    cfg = get_settings()
    rd = ROOT / cfg.eval.reports_dir
    base = load_run(rd / ("eval_" + base_id + ".json"))
    cur = load_run(rd / ("eval_" + cur_id + ".json"))
    diff, problems = compare(base, cur, cfg)
    from medevirag.eval.report import render_delta_md
    md = render_delta_md(base_id, cur_id, diff, problems)
    out = rd / ("delta_" + base_id + "_vs_" + cur_id + ".md")
    out.write_text(md, encoding="utf-8")
    for p in problems:
        print("[GATE]", p, file=sys.stderr)
    print("[delta] " + str(len(diff)) + " 题方向变化，报告 " + str(out))
    return 2 if problems else 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    a = ap.parse_args()
    raise SystemExit(main(a.baseline, a.current))
