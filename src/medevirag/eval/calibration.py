# P0 judge 校准：人评 vs judge 的一致性。人评只能由人填写，AI 不得代标。
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[3]
LABELS_PATH = ROOT / "data/eval/human_labels_30.jsonl"
SCORES_PATH = ROOT / "data/reports/judge_scores_30.jsonl"
REPORT_DIR = ROOT / "data/reports"
DIMS = ("faithfulness", "evidence_recall", "refusal")
EXPECTED_N = 30
FORBIDDEN_ANNOTATORS = {"", "auto", "ai", "llm", "gpt", "model", "offline", "stub"}


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return 0.0 if da == 0 or db == 0 else num / (da * db)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("spearman 需要至少 2 对且等长")
    return _pearson(_ranks(xs), _ranks(ys))


def cohen_kappa_binary(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("cohen_kappa_binary 需要非空且等长")
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca.get(k, 0) / n) * (cb.get(k, 0) / n) for k in (0, 1))
    return 0.0 if pe == 1.0 else (po - pe) / (1.0 - pe)


def severe_disagreement_rate(hs: Sequence[float], js: Sequence[float],
                             gap: float = 2.0) -> float:
    if len(hs) != len(js) or not hs:
        raise ValueError("需要非空且等长")
    return sum(1 for h, j in zip(hs, js) if abs(h - j) >= gap) / len(hs)


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit("缺少文件 " + str(path))
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def validate_labels(rows: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    if len(rows) != EXPECTED_N:
        problems.append("人评样本数应为 " + str(EXPECTED_N) + "，实际 " + str(len(rows)))
    ids = [str(r.get("case_id", "")) for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("case_id 有重复")
    for i, r in enumerate(rows, 1):
        who = str(r.get("annotator", "")).strip().lower()
        if who in FORBIDDEN_ANNOTATORS:
            problems.append("row " + str(i) + ": annotator 缺失或为自动来源，人评不得由 AI 代标")
        for dim in DIMS:
            v = r.get(dim)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (1 <= v <= 5):
                problems.append("row " + str(i) + ": " + dim + " 必须是 1-5 的人工评分")
    return problems


def validate_scores(rows: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    for i, r in enumerate(rows, 1):
        mode = str(r.get("llm_mode", "")).strip().lower()
        model = str(r.get("judge_model", "")).strip().lower()
        if mode in ("fake", "offline", "selfcheck", "") or "stub" in model or model in ("", "offline"):
            problems.append("row " + str(i) + ": judge 分数来自离线桩，不可用于校准")
        if not isinstance(r.get("score"), (int, float)) or isinstance(r.get("score"), bool):
            problems.append("row " + str(i) + ": score 必须是数字")
    return problems


def calibrate(labels_path: Path = LABELS_PATH,
              scores_path: Path = SCORES_PATH) -> dict[str, Any]:
    labels = _read(labels_path)
    scores = _read(scores_path)
    p = validate_labels(labels) + validate_scores(scores)
    if p:
        raise SystemExit("校准输入不合格:\n  " + "\n  ".join(p))
    by_id: dict[str, dict[str, float]] = {}
    for s in scores:
        dim = str(s["dim"])
        if dim not in DIMS:
            continue
        by_id.setdefault(str(s["case_id"]), {})[dim] = float(s["score"])
    missing = [str(r["case_id"]) for r in labels if str(r["case_id"]) not in by_id]
    if missing:
        raise SystemExit("judge 分数缺少对应 case: " + ",".join(missing[:5]))
    dims: dict[str, Any] = {}
    for dim in DIMS:
        hs = [float(r[dim]) for r in labels]
        js = [by_id[str(r["case_id"])][dim] for r in labels]
        dims[dim] = {
            "spearman_rho": round(spearman(hs, js), 4),
            "cohen_kappa_pass": round(cohen_kappa_binary(
                [1 if h >= 4 else 0 for h in hs],
                [1 if j >= 4 else 0 for j in js]), 4),
            "severe_disagreement_rate": round(severe_disagreement_rate(hs, js), 4),
            "binary_agreement": round(sum(
                1 for h, j in zip(hs, js) if (h >= 4) == (j >= 4)) / len(hs), 4),
            "n": len(hs),
        }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels_path": str(labels_path),
        "scores_path": str(scores_path),
        "expected_n": EXPECTED_N,
        "dims": dims,
        "targets": {"spearman_rho": 0.75, "severe_disagreement_rate": 0.10,
                    "binary_agreement": 0.90},
        "note": "本报告的 rho 只对本次人评与本次 judge 有效；换 judge prompt 或换模型必须重标重算。",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "judge_calibration.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(prog="calibration")
    ap.add_argument("--labels", default=str(LABELS_PATH))
    ap.add_argument("--scores", default=str(SCORES_PATH))
    a = ap.parse_args()
    r = calibrate(Path(a.labels), Path(a.scores))
    for dim, v in r["dims"].items():
        print("[calib] " + dim + ": rho=" + str(v["spearman_rho"])
              + " kappa=" + str(v["cohen_kappa_pass"])
              + " severe=" + str(v["severe_disagreement_rate"]))
    print("[calib] report -> " + str(REPORT_DIR / "judge_calibration.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
