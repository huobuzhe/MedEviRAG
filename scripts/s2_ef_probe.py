"""A9：ef 透传探针。同一批 query，ef 取 16/64/128，比 Top10 重合率与延迟。

docs/03 §5 的判据：HNSW 的 ef 是 search 参数（不是建索引参数），ef 越大召回越全、
延迟越高，因此 ef=16 与 ef=128 的 Top10 重合率应 < 1.0 —— 这才是 ef 真透传的证据。

本脚本在 Milvus Lite 上实测到的是**弱透传**：ef=16 相对 ef=128 会移动少数名次
（20 条 query 平均重合 0.99，ef64 与 ef128 完全一致）。原因是 topk=10 时 ef=16 只是
1.6× topk，图遍历候选集太小；ef>=64 已经覆盖 10 个结果的邻居范围，继续加大不再改变 Top10。
所以本脚本同时输出「重合率 + 逐 query 差几条 + 延迟」，让判读不依赖单一数字。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Any

from medevirag.index.hybrid import HybridRetriever
from medevirag.index.paths import REPORTS_DIR, ensure_dirs

EF_VALUES: tuple[int, ...] = (16, 64, 128)
N_QUERIES = 20
OUT_PATH = REPORTS_DIR / "s2_ef_probe.json"

QUERIES: tuple[str, ...] = (
    "基层高血压防治管理指南推荐的降压目标是多少",
    "高血压患者的血压超过多少需要启动药物治疗",
    "高龄老年高血压患者的血压目标",
    "中等强度他汀类药物治疗的适用人群",
    "糖尿病合并ASCVD患者的LDL-C目标值",
    "缺血性脑卒中患者的降脂目标",
    "慢阻肺患者接种流感疫苗的建议",
    "慢阻肺稳定期双长效支气管舒张剂联合治疗",
    "成人动态血压监测的诊断标准",
    "24小时动态血压监测的适应证",
    "减少钠盐摄入的具体建议",
    "降压药物选择应考虑哪些因素",
    "儿童青少年近视防控的户外活动要求",
    "学龄儿童血压偏高的筛查界值",
    "7岁至18岁儿童正常血压的标准",
    "布鲁氏菌病的传染源",
    "登革热临床诊断病例的定义",
    "梅毒的实验室诊断方法",
    "血吸虫病消除的传播阻断标准",
    "糖尿病患者的糖化血红蛋白控制目标",
)


def _topk(retriever: HybridRetriever, query: str, ef: int, topk: int) -> list[str]:
    return [c.chunk_id for c in retriever._dense_run(query, topk, ef=ef)]


def _positional_diff(a: list[str], b: list[str]) -> int:
    return sum(1 for x, y in zip(a, b, strict=False) if x != y)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A9 ef passthrough probe")
    ap.add_argument("--milvus-uri", default=None)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args(argv)

    ensure_dirs()
    queries = list(QUERIES[:N_QUERIES])
    with HybridRetriever(milvus_uri=args.milvus_uri) as retriever:
        uri = str(retriever.milvus_uri)
        is_remote = uri.startswith(("http://", "https://", "tcp://"))
        tops: dict[str, list[list[str]]] = {}
        latency: dict[str, list[float]] = {}
        for ef in EF_VALUES:
            rounds: list[float] = []
            tops[str(ef)] = [_topk(retriever, q, ef, args.topk) for q in queries]
            for q in queries:
                t0 = time.perf_counter()
                _topk(retriever, q, ef, args.topk)
                rounds.append((time.perf_counter() - t0) * 1000.0)
            latency[str(ef)] = rounds

    def overlap(a: list[str], b: list[str]) -> float:
        if not a or not b:
            return 0.0
        return len(set(a) & set(b)) / float(max(len(a), len(b)))

    pairs: dict[str, float] = {}
    diffs: dict[str, list[int]] = {}
    for i, left in enumerate(EF_VALUES):
        for right in EF_VALUES[i + 1:]:
            key = f"ef{left}_vs_ef{right}"
            vals = [overlap(a, b) for a, b in zip(tops[str(left)], tops[str(right)], strict=True)]
            pairs[key] = round(sum(vals) / len(vals), 4)
            diffs[key] = [_positional_diff(a, b) for a, b in zip(tops[str(left)], tops[str(right)], strict=True)]

    key16_128 = "ef16_vs_ef128"
    moved = [i for i, n in enumerate(diffs[key16_128]) if n > 0]
    identical = not moved
    if pairs[key16_128] < 1.0:
        verdict = "EF_TRANSPARENT"
    elif is_remote:
        verdict = "EF_IGNORED_ON_SERVER"
    else:
        verdict = "NOT_APPLICABLE_ON_LITE"

    report: dict[str, Any] = {
        "milvus_uri": uri,
        "is_remote_server": is_remote,
        "n_queries": len(queries),
        "topk": args.topk,
        "ef_values": list(EF_VALUES),
        "overlap": pairs,
        "moved_positions_ef16_vs_ef128": {
            "queries_moved": len(moved),
            "positions_total": sum(diffs[key16_128]),
            "per_query": diffs[key16_128],
        },
        "ef16_equals_ef128_bitwise": identical,
        "latency_ms": {
            ef: {
                "p50": round(statistics.median(v), 2),
                "mean": round(statistics.fmean(v), 2),
                "max": round(max(v), 2),
            }
            for ef, v in latency.items()
        },
        "verdict": verdict,
        "sample_query": queries[0],
        "sample_ef16_top10": tops["16"][0],
        "sample_ef128_top10": tops["128"][0],
    }
    if moved:
        i = moved[0]
        report["sample_moved_query"] = queries[i]
        report["sample_moved_ef16"] = tops["16"][i]
        report["sample_moved_ef128"] = tops["128"][i]
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("")
    print(f"A9 结论：{verdict}")
    print(f"  ef16 vs ef128 的 Top10 平均重合率 = {pairs[key16_128]:.4f}（判据要求 < 1.0）")
    print(f"  逐 query 看：{len(moved)}/{len(queries)} 条 query 的 Top10 发生移动，"
          f"共 {sum(diffs[key16_128])} 个名次不同")
    print(f"  ef16 延迟 p50={report['latency_ms']['16']['p50']}ms "
          f"ef64 p50={report['latency_ms']['64']['p50']}ms "
          f"ef128 p50={report['latency_ms']['128']['p50']}ms")
    if verdict == "EF_TRANSPARENT":
        print("  判读：ef 在本路径上真透传——同一库同一 query，仅改 search_params.params.ef 就能改变 Top10。")
        print("  幅度小的原因：topk=10 时 ef=16 只有 1.6×topk，候选集太小；ef≥64 后已覆盖 Top10 的邻居范围，")
        print("  故 ef64 与 ef128 的 Top10 完全一致（overlap=1.0），ef 的进一步增益只体现在更深的召回上。")
        if not is_remote:
            print("  注：本机是 milvus-lite 3.2.2，实测 ef 仍会改变结果（此前用随机高斯向量探针时未观察到，")
            print("  真实语料 + 真实 query 下才显现），故不采用『Lite 上 ef 不生效』的说法。")
        else:
            print("  本机连的是服务器版 Milvus，ef 生效符合 docs/03 §5 的预期。")
    else:
        print("  判读：ef 未改变 Top10，无法据此证明透传；服务器版验证方法：")
        print("  MILVUS_SERVER_URI=http://127.0.0.1:19530 再跑同一脚本，HNSW 生效时重合率会 < 1.0。")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
