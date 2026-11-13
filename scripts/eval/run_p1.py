# P1 一键闭环：用非桩 RuleJudge 真跑评测 -> 导出可供人评校准的分数 -> 做扰动检验。
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from medevirag.common.config import get_settings
from medevirag.eval import metrics as M
from medevirag.eval.build_set import load_cases
from medevirag.eval.judge_validation import run_validation
from medevirag.eval.layers import LAYER_DIR, verify_layers
from medevirag.eval.rule_judge import RuleJudgeLLM
from medevirag.eval.runner import run_one
from run_core import REPORT_DIR, make_ask

SCORES_PATH = REPORT_DIR / "judge_scores_30.jsonl"
SHEET_PATH = ROOT / "data/eval/human_labels_30_sheet.jsonl"


def _sheet_ids() -> list[str]:
    if not SHEET_PATH.exists():
        from make_label_sheet import GUIDE_PATH, GUIDE
        from make_label_sheet import make_sheet
        rows = make_sheet()
        SHEET_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                      for r in rows), encoding="utf-8")
        GUIDE_PATH.write_text(GUIDE, encoding="utf-8")
    return [json.loads(ln)["case_id"] for ln in
            SHEET_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run_real_eval(layer: str = "core_80", top_k: int = 10) -> dict:
    v = verify_layers()
    if not v["ok"]:
        raise SystemExit("分层校验失败: " + "; ".join(v["changed"]))
    cfg = get_settings()
    cases = load_cases(LAYER_DIR / (layer + ".jsonl"))
    judge = RuleJudgeLLM()
    rows = []
    for c in cases:
        judge.bind(c)
        rows.append(run_one(c, make_ask(c, top_k), judge, cfg))
    m = M.compute_all(rows, cfg)
    run_id = "p1_rule_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    m.update({
        "run_id": run_id, "git_sha": "p1", "config_hash": "rule-v1",
        "split": layer, "seed": cfg.eval.sample_seed,
        "frozen_set_sha256": v["meta"]["freeze_set_sha256"],
        "llm_mode": "rule", "trust_level": "rule_judge_selfcheck",
        "layer": layer, "top_k": top_k,
        "judge_model": "rule_judge_v1", "tokens_used": 0,
        "api_calls": judge.calls,
    })
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / ("eval_" + run_id + ".json")
    out.write_text(json.dumps({"metrics": m, "cases": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    wanted = set(_sheet_ids())
    lines = []
    for r in rows:
        if r["case_id"] not in wanted:
            continue
        for dim, score in r["judge"].items():
            lines.append(json.dumps({"case_id": r["case_id"], "dim": dim,
                                     "score": score, "judge_model": "rule_judge_v1",
                                     "llm_mode": "rule"}, ensure_ascii=False))
    SCORES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"run_id": run_id, "metrics": m, "n": len(rows),
            "scores_lines": len(lines), "path": out}


def main() -> int:
    r = run_real_eval()
    m = r["metrics"]
    print("[p1-real] " + r["run_id"] + " n=" + str(m["n"])
          + " faithfulness=" + str(m.get("faithfulness"))
          + " evidence_recall=" + str(m.get("evidence_recall"))
          + " refusal_dim=" + str(m.get("refusal_dim"))
          + " judge_error_rate=" + str(m.get("judge_error_rate")))
    print("[p1-real] judge_scores_30.jsonl lines=" + str(r["scores_lines"]))
    v = run_validation()
    print("[p1-valid] n_pairs=" + str(v["n_pairs"])
          + " direction_agreement=" + str(v["direction_agreement"])
          + " invariance_agreement=" + str(v["invariance_agreement"])
          + " veto_catch_rate=" + str(v["veto_catch_rate"]))
    print("[p1-valid] bias_gap=" + str(v["bias_gap"]))
    print("[p1-valid] report -> data/reports/judge_validation.json")
    print("[p1-note] 人评 30 条仍需真人标注；标完另存 human_labels_30.jsonl 后，"
          "calibration 可直接用本步导出的分数算出 rho/kappa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
