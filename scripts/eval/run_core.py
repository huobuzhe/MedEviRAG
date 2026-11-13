# P0 一键评测闭环：分层校验 -> 跑评测 -> 出报告 -> 过门禁。
# 默认 pipeline_selfcheck 模式：用确定性被测实现验证「评测管线本身」，不产出质量结论。
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from medevirag.common.config import get_settings
from medevirag.eval import metrics as M
from medevirag.eval.build_set import load_cases
from medevirag.eval.gate import evaluate as gate_evaluate
from medevirag.eval.layers import LAYER_DIR, verify_layers
from medevirag.eval.report import render_markdown
from medevirag.eval.runner import OfflineJudgeLLM, make_run_id, run_one

REPORT_DIR = ROOT / "data/reports"
DISTRACTOR = "DISTRACTOR#"
REFUSAL_ANSWER = "【就医提示】该问题需由医师或药师确认，本系统不提供具体建议。"


def _config_fingerprint() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "configs").glob("*.yaml")):
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _answer_for(case: Any, top_k: int) -> dict[str, Any]:
    if case.must_refuse:
        return {
            "answer": REFUSAL_ANSWER,
            "retrieved_chunk_ids": [],
            "context_chunks": [],
            "status": "refused",
            "refusal_reason": case.refuse_kind or "out_of_scope",
            "confidence": 0.90,
            "dangling_citation_count": 0,
            "total_citation_count": 0,
            "sentences": [],
            "citation_map": {},
        }
    gold = list(case.gold_chunk_ids)
    if top_k <= 1:
        retrieved = [DISTRACTOR + case.case_id + "#0"]
    else:
        pads = [DISTRACTOR + case.case_id + "#" + str(i)
                for i in range(max(0, top_k - len(gold)))]
        retrieved = (gold + pads)[:top_k]
    context = [{"chunk_id": g, "text": "证据片段", "section_path": [],
                "doc_title": "指南"} for g in gold]
    cmap = {str(i + 1): g for i, g in enumerate(gold)}
    return {
        "answer": case.gold_answer or "依据指南给出回答。",
        "retrieved_chunk_ids": retrieved,
        "context_chunks": context,
        "status": "answered",
        "refusal_reason": None,
        "confidence": 0.80,
        "dangling_citation_count": 0,
        "total_citation_count": len(gold),
        "sentences": [{"text": "答案句", "cited_chunk_ids": gold,
                       "landing_ok": True, "entailed": True}],
        "citation_map": cmap,
    }


def make_ask(case: Any, top_k: int):
    def ask(question: str) -> dict[str, Any]:
        return _answer_for(case, top_k)
    return ask


def _unique_path(run_id: str) -> Path:
    p = REPORT_DIR / ("eval_" + run_id + ".json")
    i = 2
    while p.exists():
        p = REPORT_DIR / ("eval_" + run_id + "_" + str(i) + ".json")
        i += 1
    return p


def run_pipeline(layer: str = "core_80", top_k: int = 10,
                 tag: str = "base") -> dict[str, Any]:
    v = verify_layers()
    if not v["ok"]:
        raise SystemExit("分层校验失败: " + "; ".join(v["changed"]))
    cfg = get_settings()
    path = LAYER_DIR / (layer + ".jsonl")
    if not path.exists():
        raise SystemExit("分层文件不存在: " + str(path) + "，先跑 layers build")
    cases = load_cases(path)
    judge = OfflineJudgeLLM()
    rows = [run_one(c, make_ask(c, top_k), judge, cfg) for c in cases]
    m = M.compute_all(rows, cfg)
    chash = _config_fingerprint()
    run_id = make_run_id("p0" + tag, chash, layer + "_k" + str(top_k))
    m.update({
        "run_id": run_id,
        "git_sha": "p0" + tag,
        "config_hash": chash,
        "split": layer,
        "seed": cfg.eval.sample_seed,
        "frozen_set_sha256": v["meta"]["freeze_set_sha256"],
        "llm_mode": "fake",
        "trust_level": "pipeline_selfcheck",
        "layer": layer,
        "top_k": top_k,
        "judge_model": "offline_stub",
        "tokens_used": 0,
        "api_calls": getattr(judge, "calls", 0),
    })
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _unique_path(run_id)
    run_id = out.stem[len("eval_"):]
    m["run_id"] = run_id
    out.write_text(json.dumps({"metrics": m, "cases": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / ("eval_" + run_id + ".md")).write_text(
        render_markdown(m, rows), encoding="utf-8")
    scores: list[str] = []
    for r in rows:
        for dim, score in r["judge"].items():
            scores.append(json.dumps({
                "case_id": r["case_id"], "dim": dim, "score": score,
                "judge_model": "offline_stub", "llm_mode": "fake",
            }, ensure_ascii=False))
    (REPORT_DIR / ("eval_" + run_id + "_judge_scores.jsonl")).write_text(
        "\n".join(scores) + "\n", encoding="utf-8")
    return {"run_id": run_id, "path": out, "metrics": m, "n": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(prog="run_core")
    ap.add_argument("--layer", default="core_80",
                    choices=["smoke_20", "core_80", "full_160"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--baseline", default=None, help="基线 run_id，传入后执行双 run 门禁")
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    r = run_pipeline(a.layer, a.top_k, a.tag)
    m = r["metrics"]
    print("[eval] " + r["run_id"] + " n=" + str(m["n"])
          + " recall@10=" + str(m.get("recall_at_10"))
          + " mrr=" + str(m.get("mrr"))
          + " refusal_recall=" + str(m.get("refusal_recall")))
    print("[eval] report -> " + str(r["path"]))
    if a.no_gate:
        return 0
    base = None
    if a.baseline:
        base = REPORT_DIR / ("eval_" + a.baseline + ".json")
        if not base.exists():
            raise SystemExit("基线报告不存在: " + str(base))
    g = gate_evaluate(r["path"], base)
    for n in g["notes"]:
        print("[NOTE]", n)
    for p in g["problems"]:
        print("[GATE]", p)
    print("[gate] " + g["status"] + " publishable=" + str(g["publishable"]))
    return g["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
