#!/usr/bin/env python
"""S2 检索实例展示：dev 抽样逐条打印 question -> Top5 的 chunk_id/score/source。

source 取值：dense（只在 dense 一路出现）、bm25（只在 BM25 一路出现）、
dense+bm25（两路都召回，RRF 合并）。用来人工核对「题目问的东西是否真的被检索到」。
"""

from __future__ import annotations

import argparse
import json

from medevirag.index.hybrid import HybridRetriever
from medevirag.index.paths import EVAL_DIR

CASES_PATH = EVAL_DIR / "retrieval_dev60.jsonl"


def load_cases(split: str) -> list[dict]:
    with CASES_PATH.open(encoding="utf-8") as fh:
        cases = [json.loads(line) for line in fh if line.strip()]
    if split == "all":
        return cases
    return [c for c in cases if c.get("split") == split]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Show retrieval hits for labelled cases")
    ap.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases in the split")
    ap.add_argument("--milvus-uri", default=None)
    args = ap.parse_args(argv)

    cases = load_cases(args.split)
    if args.limit:
        cases = cases[: args.limit]
    with HybridRetriever(milvus_uri=args.milvus_uri) as retriever:
        for case in cases:
            gold = list(case.get("gold_chunk_ids") or [])
            print("=" * 88)
            print(f"{case['qid']} [{case['case_type']}/{case.get('difficulty')}] {case['question']}")
            print(f"  gold={gold}")
            hits = retriever.retrieve(str(case["question"]), topk=args.topk)
            for i, h in enumerate(hits, 1):
                mark = "GOLD" if h.chunk_id in gold else "    "
                title = (h.doc_title or "")[:24]
                print(f"  {i}. [{mark}] {h.chunk_id}  fused={h.fused_score:.5f} "
                      f"dense={h.dense_score:.4f} bm25={h.bm25_score:.3f} "
                      f"rank(d/b)={h.dense_rank}/{h.bm25_rank} src={h.source} | {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
