"""夹具构造器（docs/05 §T8）：从一行 JSONL 造 AskContext / ConfidenceBreakdown。

build_ctx 若把分数全填高值，R-13/R-14 永不触发——分数型用例会假绿。
所以每行的 top1/top5/years/levels 都从 case 字段读，不在这里拍脑袋。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medevirag.common.config import get_settings
from medevirag.index.fusion import RetrievedChunk
from medevirag.rag.confidence import CitationReport, ConfidenceBreakdown, SentenceSupport, compute_confidence
from medevirag.rag.refusal import refusal_config
from medevirag.rag.refusal_rules import AskContext, evaluate_non_score, evaluate_score

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "refusal_cases.jsonl"


def load_cases() -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in
            FIXTURES.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _candidates(case: dict[str, Any]) -> list[RetrievedChunk]:
    years = case.get("years") or [2024] * 8
    top1 = float(case.get("top1", 0.75))
    top5 = float(case.get("top5", 0.60))
    unknown_level = case.get("unknown_level", "none")
    top1_doc_type = case.get("top1_doc_type", "consensus")
    out = []
    for i in range(8):
        dense = top1 if i == 0 else (top5 if i == 4 else top1 - 0.01 * (i + 1))
        level = "unknown" if unknown_level == "all" else ("A" if i < 4 else "B")
        out.append(RetrievedChunk(
            chunk_id=f"{case['case_id']}#c{i:02d}", doc_id="d", text="t",
            context_header="h", chunk_type="paragraph", doc_title="t",
            doc_type=top1_doc_type if i == 0 else "consensus",
            publish_year=int(years[i]) if i < len(years) else 2024,
            evidence_level=level, dense_score=dense, fused_rank=i + 1,
        ))
    return out


def _citation(case: dict[str, Any], cands: list[RetrievedChunk]) -> CitationReport:
    dangling = int(case.get("dangling", 0))
    total = int(case.get("total", 3))
    entailed_frac = float(case.get("entailed_frac", 1.0))
    cite_frac = float(case.get("cite_frac", 1.0))
    n_sents = 4
    n_cited = max(1, int(round(n_sents * cite_frac)))
    sents = []
    for i in range(n_sents):
        cited = i < n_cited
        entailed = cited and (entailed_frac >= 1.0 or (i + 1) / n_cited <= entailed_frac)
        sents.append(SentenceSupport(
            text=f"句子{i}",
            cited_chunk_ids=(cands[0].chunk_id,) if cited else (),
            landing_ok=True, entailed=entailed,
        ))
    return CitationReport(sentences=tuple(sents), dangling_citation_count=dangling,
                          total_citation_count=total, sentences_total=n_sents)


def build_ctx(case: dict[str, Any]) -> AskContext:
    cands = _candidates(case)
    return AskContext(
        question=case["question"],
        route=case.get("route", "clinical"),
        route_score=1.0 if case.get("route") in ("off_domain", "non_evidence") else 0.0,
        topic_covered=bool(case.get("topic_covered", True)),
        candidates=cands,
        citation=_citation(case, cands),
        now_year=int(get_settings().refusal.now_year),
    )


def build_conf(case: dict[str, Any]) -> ConfidenceBreakdown:
    cands = _candidates(case)
    cite = _citation(case, cands)
    cited_ids = [cid for s in cite.sentences for cid in s.cited_chunk_ids]
    s = get_settings()
    return compute_confidence(cands, cited_ids, cite, s.confidence.weights,
                              s.confidence.platt, rerank_applied=False)


def run_matrix() -> dict[str, Any]:
    """跑全部夹具，返回矩阵与四项指标（夹具级指标，非评测集指标）。"""
    cfg = refusal_config()
    rows = []
    for case in load_cases():
        ctx = build_ctx(case)
        hit = evaluate_non_score(ctx, cfg)
        final = None
        if hit is not None and hit.action == "refuse":
            final = hit
        else:
            bd = build_conf(case)
            hit2 = evaluate_score(ctx, bd, cfg)
            if hit2 is not None:
                final = hit2
            elif hit is not None:
                final = hit
        refused = final is not None and final.action == "refuse"
        expect = case["expect"]
        # degrade 规则（R-06/R-07）按 docs 表不拒答：expect=answered 且命中 degrade 算 ok；
        # 规则可触发性由 expect_hit 字段单独断言（A4 覆盖计数用它）。
        degrade_hit = final is not None and final.action == "degrade"
        if expect == "refused":
            ok = refused
        else:
            ok = (not refused) and (not degrade_hit or case.get("expect_hit") == (final.rule_id if final else None))
        if case.get("expect_hit"):
            ok = ok and final is not None and final.rule_id == case["expect_hit"]
        reason_ok = bool(expect == "refused" and refused and final is not None
                         and final.reason.value == case.get("gold_reason")
                         and final.rule_id == case.get("gold_rule_id"))
        hit_ok = (not case.get("expect_hit")) or (final is not None and final.rule_id == case["expect_hit"])
        rows.append({"case_id": case["case_id"], "expect": expect, "ok": ok and hit_ok,
                     "refused": refused,
                     "rule_id": final.rule_id if final else None,
                     "reason": final.reason.value if final else None,
                     "reason_ok": reason_ok})

    should_refuse = [r for r in rows if r["expect"] == "refused"]
    should_answer = [r for r in rows if r["expect"] == "answered"]
    refused_correct = [r for r in should_refuse if r["refused"]]
    refused_right_reason = [r for r in refused_correct if r["reason_ok"]]
    over_refused = [r for r in should_answer if r["refused"]]

    covered = sorted({r["rule_id"] for r in rows if r["rule_id"]})
    all_rules = sorted({f"R-{i:02d}" for i in range(1, 18)} & {c.get("gold_rule_id") for c in load_cases() if c.get("gold_rule_id")})
    from collections import Counter
    over_attr = Counter(r["rule_id"] for r in over_refused)

    return {
        "n_cases": len(rows),
        "n_should_refuse": len(should_refuse),
        "n_should_answer": len(should_answer),
        "refusal_recall": round(len(refused_correct) / max(1, len(should_refuse)), 4),
        "over_refusal_rate": round(len(over_refused) / max(1, len(should_answer)), 4),
        "reason_accuracy": round(len(refused_right_reason) / max(1, len(refused_correct)), 4),
        "rules_covered": covered,
        "rules_expected": all_rules,
        "rules_cover_n": len(covered),
        "over_refused": over_refused,
        "over_refused_attribution": dict(over_attr),
        "failed": [r for r in rows if not r["ok"] or (r["expect"] == "refused" and r["refused"] and not r["reason_ok"])],
        "fixture_level": True,   # docs 风险 5：夹具级指标，非评测集指标
    }
