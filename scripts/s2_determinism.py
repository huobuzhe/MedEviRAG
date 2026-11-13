"""A12：确定性检查。同一 query 两次 retrieve 的 chunk_id 序列必须逐位相同。

排序稳定性来源：rrf_fuse / weighted_fuse 的排序键是 (-fused_score, chunk_id)，
同分时按 chunk_id 兜底，不依赖 dict/set 迭代顺序。
"""

from __future__ import annotations

import argparse
import json
import sys

from medevirag.index.hybrid import HybridRetriever
from medevirag.index.paths import REPORTS_DIR, ensure_dirs

OUT_PATH = REPORTS_DIR / "s2_determinism.json"
QUERIES: tuple[str, ...] = (
    "基层高血压防治管理指南推荐的降压目标是多少",
    "高血压患者的血压超过多少需要启动药物治疗",
    "中等强度他汀类药物治疗的适用人群",
    "糖尿病合并ASCVD患者的LDL-C目标值",
    "慢阻肺患者接种流感疫苗的建议",
    "成人动态血压监测的诊断标准",
    "儿童青少年近视防控的户外活动要求",
    "布鲁氏菌病的传染源",
    "登革热临床诊断病例的定义",
    "糖尿病患者的糖化血红蛋白控制目标",
)
VARIANTS: tuple[str, ...] = ("rrf", "weighted-sum", "dense-only", "bm25-only")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A12 determinism check")
    ap.add_argument("--milvus-uri", default=None)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args(argv)

    ensure_dirs()
    detail = []
    all_same = True
    with HybridRetriever(milvus_uri=args.milvus_uri) as retriever:
        for variant in VARIANTS:
            for query in QUERIES:
                first = [c.chunk_id for c in retriever.retrieve(query, topk=args.topk, variant=variant)]
                second = [c.chunk_id for c in retriever.retrieve(query, topk=args.topk, variant=variant)]
                same = first == second
                all_same = all_same and same
                if not same:
                    detail.append({"variant": variant, "query": query, "a": first, "b": second})
    print(f"DETERMINISTIC = {all_same}  (variants={list(VARIANTS)} queries={len(QUERIES)})")
    if detail:
        print(json.dumps(detail, ensure_ascii=False, indent=2)[:4000])
    OUT_PATH.write_text(
        json.dumps(
            {"deterministic": all_same, "variants": list(VARIANTS), "n_queries": len(QUERIES), "mismatch": detail},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_PATH}")
    return 0 if all_same else 1


if __name__ == "__main__":
    sys.exit(main())
