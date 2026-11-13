"""T8：S2 门禁。读 data/reports/s2_ablation.json 判定是否达标。

阈值与 docs/00 §9 逐字一致：Recall@10 >= 0.85 / MRR >= 0.62 / nDCG@10 >= 0.70。
任何一项不达标 → return 1（供 accept_s2.py 与 CI 判定）。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from medevirag.index.paths import REPORTS_DIR

ABLATION_PATH = REPORTS_DIR / "s2_ablation.json"
MD_PATH = REPORTS_DIR / "s2_ablation.md"

THRESHOLDS: dict[str, float] = {"recall@10": 0.85, "mrr": 0.62, "ndcg@10": 0.70}
REQUIRED_VARIANTS: tuple[str, ...] = ("dense-only", "bm25-only", "rrf", "weighted-sum")
GATED_VARIANTS: tuple[str, ...] = ("rrf",)
SOURCE_WITH_SERVER = "milvus-lite-and-server"
SOURCE_LITE_ONLY = "milvus-lite"


def check(report: dict[str, Any], require_server: bool) -> tuple[bool, list[str]]:
    problems: list[str] = []
    variants = report.get("variants") or {}
    for name in REQUIRED_VARIANTS:
        if name not in variants:
            problems.append(f"missing variant {name!r} in ablation report")
    for name in GATED_VARIANTS:
        metrics = variants.get(name) or {}
        for metric, floor in THRESHOLDS.items():
            value = metrics.get(metric)
            if value is None:
                problems.append(f"{name}: metric {metric} missing")
                continue
            if float(value) < floor:
                problems.append(f"{name}: {metric}={float(value):.4f} < {floor:.2f}")
    source = str((report.get("index") or {}).get("source") or "")
    allowed = (SOURCE_WITH_SERVER,) if require_server else (SOURCE_WITH_SERVER, SOURCE_LITE_ONLY)
    if source not in allowed:
        problems.append(
            f"index.source={source!r} not in {allowed!r} "
            "(docs/03 §T8 要求 milvus-lite-and-server；parity 被 skip 时退化为 milvus-lite)"
        )
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S2 quality gate")
    ap.add_argument("--report", default=str(ABLATION_PATH))
    ap.add_argument("--require-server", action="store_true",
                    help="连 index.source 也要求 milvus-lite-and-server（服务器版跑过 parity 时才该加）")
    args = ap.parse_args(argv)

    from pathlib import Path

    path = Path(args.report)
    if not path.exists():
        print(f"FAIL: ablation report not found: {path} (run scripts/s2_ablation.py first)")
        return 1
    report = json.loads(path.read_text(encoding="utf-8"))
    ok, problems = check(report, require_server=args.require_server)

    print(f"report: {path}")
    print(f"split={report.get('split')} n_cases={report.get('n_cases')} "
          f"generated_at={report.get('generated_at')} config_hash={report.get('config_hash')}")
    print("thresholds: " + ", ".join(f"{k}>={v}" for k, v in THRESHOLDS.items()))
    for name in REQUIRED_VARIANTS:
        m = (report.get("variants") or {}).get(name)
        if not m:
            print(f"  {name:14s} MISSING")
            continue
        flags = []
        for metric, floor in THRESHOLDS.items():
            mark = "ok" if float(m.get(metric, 0.0)) >= floor else "BELOW"
            flags.append(f"{metric}={float(m.get(metric, 0.0)):.4f}[{mark}]")
        gate_note = " (gated)" if name in GATED_VARIANTS else ""
        print(f"  {name:14s} " + "  ".join(flags) + gate_note)
    print(f"index.source = {(report.get('index') or {}).get('source')!r}")

    if MD_PATH.exists():
        md = MD_PATH.read_text(encoding="utf-8")
        need_markers = ["Recall@10", "MRR", "nDCG@10", "case_type = exact_term"]
        missing = [m for m in need_markers if m not in md]
        if missing:
            problems.append(f"s2_ablation.md missing sections: {missing}")
        if "generated_at" not in md or "config_hash" not in md:
            problems.append("s2_ablation.md missing generated_at/config_hash header")

    if ok:
        print("GATE: PASS")
        return 0
    print("GATE: FAIL")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
