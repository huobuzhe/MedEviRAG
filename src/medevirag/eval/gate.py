# P0 统一门禁外壳：复用 regression.compare，补三件原内核未兑现的检查。
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medevirag.common.config import get_settings
from medevirag.eval.regression import GATED, compare, load_run

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "data/reports"
RESULT_PATH = REPORT_DIR / "p0_gate_result.json"
JUDGE_KEYS = (
    "faithfulness", "evidence_recall", "refusal_dim",
    "citation_landing_rate", "sentence_support_rate", "refusal_reason_accuracy",
)


def evaluate(cur_path: Path, base_path: Path | None = None,
             cfg: Any = None) -> dict[str, Any]:
    cfg = cfg or get_settings()
    cur = load_run(cur_path)
    base = load_run(base_path) if base_path else None
    metrics = cur["metrics"]
    problems: list[str] = []
    notes: list[str] = []

    llm_mode = str(metrics.get("llm_mode", "unknown"))
    trust_level = str(metrics.get("trust_level", "unknown"))
    publishable = llm_mode == "real"
    if not publishable:
        notes.append("PUBLISH_BLOCKED: 本次为 " + llm_mode
                     + " / " + trust_level
                     + "，judge 类指标不可作为发布或简历数字")

    err = metrics.get("judge_error_rate")
    err_max = cfg.eval.thresholds.judge_error_rate
    if err is not None and err > err_max:
        problems.append("RUN_INVALID_JUDGE_ERROR: " + str(err)
                        + " 超过上限 " + str(err_max) + "，本次 run 作废")

    noise_band: list[dict[str, Any]] = []
    diff_count = 0
    if base is not None:
        diff, probs = compare(base, cur, cfg)
        problems.extend(probs)
        diff_count = len(diff)
        max_drop = cfg.eval.regression.max_drop
        hard_drop = cfg.eval.regression.hard_drop
        for key in GATED:
            b = base["metrics"].get(key)
            c = metrics.get(key)
            if b is None or c is None:
                continue
            d = round(c - b, 4)
            worse = (d < 0) if key != "over_refusal_rate" else (d > 0)
            if worse and max_drop < abs(d) <= hard_drop:
                rec = {"metric": key, "base": b, "cur": c, "delta": d}
                noise_band.append(rec)
                notes.append("NOISE_BAND " + key + ": " + str(b) + " -> " + str(c)
                             + " (delta=" + str(d) + ") 落在噪声带，记录但不拦截")

    status = "BLOCKED" if problems else "PASS"
    result: dict[str, Any] = {
        "status": status,
        "exit_code": 2 if problems else 0,
        "current": str(cur_path),
        "baseline": str(base_path) if base_path else None,
        "run_id": metrics.get("run_id"),
        "llm_mode": llm_mode,
        "trust_level": trust_level,
        "publishable": publishable,
        "frozen_set_sha256": metrics.get("frozen_set_sha256"),
        "problems": problems,
        "notes": notes,
        "noise_band": noise_band,
        "diff_count": diff_count,
        "gated_metrics": {k: metrics.get(k) for k in GATED},
        "judge_keys_unusable_when_fake": list(JUDGE_KEYS),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(prog="gate")
    ap.add_argument("--current", required=True, help="run_id 或报告文件绝对路径")
    ap.add_argument("--baseline", default=None, help="基线 run_id 或报告文件绝对路径")
    a = ap.parse_args()

    def resolve(x: str) -> Path:
        p = Path(x)
        return p if p.is_absolute() or p.exists() else REPORT_DIR / ("eval_" + x + ".json")

    r = evaluate(resolve(a.current), resolve(a.baseline) if a.baseline else None)
    for n in r["notes"]:
        print("[NOTE]", n)
    for p in r["problems"]:
        print("[GATE]", p)
    print("[gate] status=" + r["status"]
          + " publishable=" + str(r["publishable"])
          + " -> " + str(RESULT_PATH))
    return r["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
