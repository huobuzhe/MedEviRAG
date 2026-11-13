# S5 T9：评测 runner（docs/06 §T9）。只经 rag 公开接口驱动，不复制节点逻辑。

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medevirag.common.config import get_settings
from medevirag.eval import metrics as M
from medevirag.eval.build_set import load_cases, verify_freeze
from medevirag.eval.judges import assert_cross_family, judge_dim

ROOT = Path(__file__).resolve().parents[3]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            text=True, cwd=str(ROOT), stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def config_hash() -> str:
    # 配置 + prompt 常量一起进 hash：改 prompt 必须改变 run_id
    import hashlib
    h = hashlib.sha256()
    for f in sorted((ROOT / "configs").glob("*.yaml")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    from medevirag.rag import generator as G
    from medevirag.rag import refusal_templates as RT
    h.update(b"sysprompt")
    h.update(G.SYSTEM_PROMPT.encode("utf-8"))
    h.update(b"disclaimer")
    h.update(RT.DISCLAIMER.encode("utf-8"))
    return h.hexdigest()[:12]


def make_run_id(sha: str, chash: str, split: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    return ts + "_" + sha + "_" + chash + "_" + split


class OfflineJudgeLLM:
    # 离线判分桩（无 API key 时的降级）：不动网络。
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, system: str, user: str, *, temperature: float = 0.0,
             max_tokens: int = 512) -> str:
        self.calls += 1
        want = re.findall(r"- ([FER]\d)", user)
        items = [{"item_id": i, "ok": True, "why": "offline"} for i in want]
        return json.dumps({"items": items}, ensure_ascii=False)


def run_one(case: Any, ask_fn: Any, judge_llm: Any, cfg: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    resp = ask_fn(case.question)
    latency_ms = (time.perf_counter() - t0) * 1000

    retrieved = list(resp.get("retrieved_chunk_ids") or [])
    graded = {c: 2 for c in case.gold_chunk_ids}
    context = resp.get("context_chunks") or []
    gold_pref = {g.split("#")[0] for g in case.gold_chunk_ids}
    for e in context:
        cid = e.get("chunk_id", "")
        if cid not in graded and any(cid.startswith(p) for p in gold_pref):
            graded[cid] = 1

    citation_map = resp.get("citation_map") or {}
    row = {
        "case_id": case.case_id, "case_type": case.case_type,
        "must_refuse": case.must_refuse,
        "gold_chunk_ids": list(case.gold_chunk_ids),
        "gold_refuse_kind": case.refuse_kind,
        "question": case.question,
        "answer": resp.get("answer", ""),
        "retrieved": retrieved,
        "graded": graded,
        "status": resp.get("status"),
        "refusal_reason": resp.get("refusal_reason"),
        "confidence": float(resp.get("confidence") or 0.0),
        "dangling_citation_count": int(resp.get("dangling_citation_count") or 0),
        "total_citation_count": int(resp.get("total_citation_count") or 0),
        "sentences": resp.get("sentences") or [],
        "latency_ms": latency_ms,
        "judge": {}, "judge_error": False,
    }
    for dim in ("faithfulness", "evidence_recall", "refusal"):
        js = judge_dim(case, row["answer"], context, dim, judge_llm,
                       cfg.eval.judge, row["status"], citation_map)
        if js.error:
            row["judge_error"] = True
            row["judge_error_msg"] = js.error
        else:
            row["judge"][dim] = js.score
    row["correct"] = bool(row["judge"].get("faithfulness", 0.0) >= 0.8
                          and row["judge"].get("refusal", 0.0) >= 0.8)
    row["metric_flags"] = {
        "recall_at_10": bool(set(case.gold_chunk_ids) & set(retrieved[:10])),
        "mrr": M.mrr(case.gold_chunk_ids, retrieved) > 0,
        "ndcg_at_10": M.ndcg_at_k(graded, retrieved, 10) >= 0.7,
        "faithfulness": row["judge"].get("faithfulness", 0.0) >= 0.8,
        "refusal_recall": (row["status"] == "refused") if case.must_refuse else True,
    }
    return row


def main(split: str = "dev", allow_test: bool = False, limit: int | None = None,
         ask_fn: Any = None, judge_llm: Any = None) -> dict[str, Any]:
    cfg = get_settings()
    set_path = ROOT / cfg.eval.set_path
    cases = load_cases(set_path)
    verify_freeze(cases, ROOT / cfg.eval.freeze_path, split)
    if split == "test" and not allow_test:
        raise SystemExit("test 子集被冻结：必须显式 --allow-test")
    subset = [c for c in cases if c.split == split]
    if limit:
        subset = subset[:limit]

    if ask_fn is None:
        from medevirag.rag.graph import run_ask as _ask
        ask_fn = _ask
    if judge_llm is None:
        judge_llm = OfflineJudgeLLM()

    assert_cross_family(cfg.eval.generator.model, cfg.eval.judge.model)
    sha, chash = git_sha(), config_hash()
    run_id = make_run_id(sha, chash, split)
    rows = [run_one(c, ask_fn, judge_llm, cfg) for c in subset]
    m = M.compute_all(rows, cfg)
    m.update({"run_id": run_id, "git_sha": sha, "config_hash": chash, "split": split,
              "seed": cfg.eval.sample_seed,
              "frozen_set_sha256": json.loads(
                  (ROOT / cfg.eval.freeze_path).read_text(encoding="utf-8"))["set_sha256"],
              "llm_mode": "fake",
              "judge_model": cfg.eval.judge.model,
              "tokens_used": 0,
              "api_calls": getattr(judge_llm, "calls", 0)})

    out_dir = ROOT / cfg.eval.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("eval_" + run_id + ".json")).write_text(
        json.dumps({"metrics": m, "cases": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    from medevirag.eval.report import render_markdown
    (out_dir / ("eval_" + run_id + ".md")).write_text(
        render_markdown(m, rows), encoding="utf-8")
    print("[eval] " + run_id + " n=" + str(m["n"])
          + " faithfulness=" + str(m.get("faithfulness"))
          + " recall@10=" + str(m.get("recall_at_10")))
    return m


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(split=a.split, allow_test=a.allow_test, limit=a.limit)
