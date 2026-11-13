"""T9 dev 集自测（docs/04 §T9，A5/A6/A12 门禁数据源）。

统计口径与 00 §9 一致：
- 落点正确率 = 合法标记数 / 全部标记数（marker 级；case 级仅诊断，门禁只看 marker 级）
- 句级支撑率 = run_ask 返回 support_rate 的均值

离线 / 在线两条路径（docs/04 §T9 + A13）：
- 默认 --offline：用 GroundedFake 桩。它从【证据块】里抽 C1 的原文当答案并附 [C1]，
  因此能真正走通「落点校验 + 句级蕴含 + 降级」整条链路（llm_mode="fake_grounded"）。
  但 gate_applicable=False —— 真实模型的合规率必须在 --real 下测，离线数据不能冒充门禁。
- --real：走真实 API（本机无 key → 标 llm_real_skipped 并跳过断言）。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"

# 证据行形如 [C1] 《doc_title》leaf（p1）：正文；正文可能含换行，故用 DOTALL + 非贪婪
_C1_LINE = re.compile(r"^\[C1\] 《.*?（p\d+）：(.+?)(?=^###|^\[C\d+\]|\Z)", re.M | re.S)


class GroundedFake:
    """离线链路自检桩：把证据块 C1 的原文当成答案，并附 [C1]。

    这不是「凑指标」——它只验证 checkpoint 链路本身；真实合规率必须 --real。
    """

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        self.calls += 1
        hit = _C1_LINE.search(user or "")
        body = (hit.group(1) if hit else "").strip()
        # 只取首句并压平换行：verifier 只保留"带引用的句子"，多句正文会被当无引用句丢弃，
        # 那是校验器的正确行为，不是桩该触发的路径。
        # 压平换行并把内部句末标点降级为逗号 → 保证是"单句"，且内容取自 C1 原文。
        # 这样 verifier 的"只保留带引用句子"与 min-20 字规则都不会误伤链路本身。
        body = body.replace("\n", " ").strip() if body else ""
        body = re.sub(r"[。！？!?；;]+", "，", body).strip("，").strip()
        body = body[:120].rstrip("，")
        if not body:
            return json.dumps({"answer": "现有证据不足以回答该问题",
                               "used_citations": [], "insufficient_evidence": True},
                              ensure_ascii=False)
        return json.dumps({"answer": f"{body} [C1]",
                           "used_citations": ["C1"],
                           "insufficient_evidence": False}, ensure_ascii=False)


def _p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(ROOT / "data" / "eval" / "s3_grounding_dev.jsonl"))
    ap.add_argument("--out", default=str(REPORTS / "s3_grounding.json"))
    ap.add_argument("--real", action="store_true", help="走真实 API（本机无 key，默认离线桩）")
    ap.add_argument("--latency", action="store_true", help="额外校验图内 P95 延迟（A12）")
    args = ap.parse_args()

    from medevirag.rag import graph, nodes

    llm_mode = "real" if args.real else "fake_grounded"
    stub = None
    if not args.real:
        stub = GroundedFake()
        nodes._llm = stub

    with open(args.set, encoding="utf-8") as fh:
        cases = [json.loads(ln) for ln in fh if ln.strip()]

    marker_total = marker_legal = case_ok = 0
    support_sum = support_n = 0
    illegal_samples: list[dict] = []
    latencies: list[float] = []

    for case in cases:
        t0 = time.perf_counter()
        out = graph.run_ask(case["question"])
        latencies.append((time.perf_counter() - t0) * 1000.0)

        cited = list(out.get("used_citations") or [])
        illegal = set(out.get("illegal_citations") or [])
        legal = [m for m in cited if m not in illegal]
        marker_total += len(cited)
        marker_legal += len(legal)
        if len(legal) == len(cited):
            case_ok += 1
        else:
            illegal_samples.append({"case_id": case.get("case_id"), "illegal": sorted(illegal)})
        # 句级支撑率只在"有引用句"的题上统计（docs: 含引用句子中被原文蕴含的比例）；
        # 证据不足题没有引用句，其 support_rate 恒为 0，计入会污染分母。
        if cited:
            support_sum += float(out.get("support_rate") or 0.0)
            support_n += 1

    llm_calls = stub.calls if stub is not None else 0
    report = {
        "n_cases": len(cases),
        "llm_mode": llm_mode,
        "gate_applicable": bool(args.real),
        "llm_real_skipped": (not args.real),
        "landing_acc_marker_level": round(marker_legal / max(1, marker_total), 4),
        "landing_acc_case_level": round(case_ok / max(1, len(cases)), 4),
        "support_rate_mean": round(support_sum / max(1, support_n), 4),
        "support_n_cases": support_n,
        "illegal_samples": illegal_samples[:20],
        "llm_calls": llm_calls,
        "tokens_used": 0,
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_p95_ms": round(_p95(latencies), 1),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in (
        "n_cases", "llm_mode", "gate_applicable", "landing_acc_marker_level",
        "support_rate_mean", "latency_p95_ms", "llm_calls")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
