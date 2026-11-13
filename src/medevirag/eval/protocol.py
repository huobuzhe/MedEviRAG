# P3：统一评测协议。不同业务只要实现 EvalTarget，就能共用同一套指标、基线、门禁与看板。
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "data/reports"
EVAL_DIR = ROOT / "data/eval/layers"


@dataclass
class EvalRecord:
    case_id: str
    target: str
    version: str
    input: str
    output: str
    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    cost_tokens: int = 0
    latency_ms: float = 0.0
    status: str = "ok"
    error: str = ""


class EvalTarget(Protocol):
    name: str
    version: str

    def run(self, case: dict[str, Any]) -> EvalRecord: ...


class TargetRegistry:
    def __init__(self) -> None:
        self._targets: dict[str, EvalTarget] = {}

    def register(self, target: EvalTarget) -> None:
        if target.name in self._targets:
            raise ValueError("重复注册 target: " + target.name)
        self._targets[target.name] = target

    def names(self) -> list[str]:
        return sorted(self._targets)

    def get(self, name: str) -> EvalTarget:
        return self._targets[name]


def evaluate_target(target: EvalTarget, cases: list[dict[str, Any]],
                    tag: str = "run") -> dict[str, Any]:
    records = [target.run(c) for c in cases]
    ok = sum(1 for r in records if r.success)
    tokens = sum(r.cost_tokens for r in records)
    lat = sorted(r.latency_ms for r in records)
    return {
        "target": target.name,
        "version": target.version,
        "run_id": target.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                  + "_" + tag,
        "case_count": len(records),
        "success_rate": round(ok / max(1, len(records)), 4),
        "error_rate": round(sum(1 for r in records if r.error) / max(1, len(records)), 4),
        "total_tokens": tokens,
        "estimated_cost_yuan": round(tokens / 1000 * 0.001, 6),
        "avg_latency_ms": round(sum(lat) / max(1, len(lat)), 4),
        "p95_latency_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 4) if lat else 0.0,
        "records": [asdict(r) for r in records],
    }


def compare_versions(base: dict[str, Any], cur: dict[str, Any],
                     max_drop: float = 0.02) -> dict[str, Any]:
    keys = ("success_rate", "error_rate", "total_tokens", "avg_latency_ms")
    deltas = {}
    flags = []
    for k in keys:
        b, c = base.get(k, 0), cur.get(k, 0)
        deltas[k] = round(c - b, 4)
        if k == "success_rate" and c < b - max_drop:
            flags.append("SUCCESS_DROP " + k + ": " + str(b) + " -> " + str(c))
        if k == "error_rate" and c > b + max_drop:
            flags.append("ERROR_RISE " + k + ": " + str(b) + " -> " + str(c))
    return {"target": cur["target"], "baseline": base["run_id"],
            "current": cur["run_id"], "deltas": deltas,
            "flags": flags, "gate": "BLOCKED" if flags else "PASS"}


def reflow_bad_cases(reports: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    """失败样本回流成候选集；带 quarantine 标记，未经人工审核不得进正式评测集（防污染）。"""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "reflow_candidates.jsonl"
    rows = []
    for rep in reports:
        for r in rep["records"]:
            if r["success"]:
                continue
            rows.append({
                "candidate_id": "rf_" + rep["target"] + "_" + r["case_id"],
                "source_target": rep["target"], "source_run_id": rep["run_id"],
                "question": r["input"], "failure_reason": r["error"] or r["status"],
                "quarantine": True,
                "note": "候选样本，须人工审核并重新冻结后才可进入评测集；禁止直接当 few-shot",
            })
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                   encoding="utf-8")
    return {"n": len(rows), "path": str(out), "source_run_id": run_id,
            "quarantine": True}


def render_dashboard(reports: list[dict[str, Any]], comparisons: list[dict[str, Any]],
                     reflow: dict[str, Any]) -> str:
    lines = ["# 统一评测看板（P3）", "",
             "生成时间：" + datetime.now(timezone.utc).isoformat(timespec="seconds"), "",
             "## 各业务本次指标", "",
             "| target | version | case_count | success_rate | error_rate | tokens | p95_latency_ms |",
             "|---|---|---|---|---|---|---|"]
    for r in reports:
        lines.append("| " + r["target"] + " | " + r["version"] + " | "
                     + str(r["case_count"]) + " | " + str(r["success_rate"]) + " | "
                     + str(r["error_rate"]) + " | " + str(r["total_tokens"]) + " | "
                     + str(r["p95_latency_ms"]) + " |")
    lines += ["", "## 版本对比（统一基线）", "",
              "| target | baseline | current | gate | flags |", "|---|---|---|---|---|"]
    for c in comparisons:
        lines.append("| " + c["target"] + " | " + c["baseline"] + " | " + c["current"]
                     + " | " + c["gate"] + " | " + ("; ".join(c["flags"]) or "无") + " |")
    lines += ["", "## bad case 回流", "",
              "- 候选样本数：" + str(reflow["n"]),
              "- 文件：`" + reflow["path"] + "`",
              "- 隔离状态：`quarantine=True`，须人工审核后再冻结，禁止直接当 few-shot（防污染）", "",
              "> 说明：两个业务共用同一套 EvalRecord 字段、同一份基线、同一套门禁与同一张看板；",
              "> 这就是「复用/服务化」的中台思维，但项目命名仍不叫中台。"]
    return "\n".join(lines) + "\n"
