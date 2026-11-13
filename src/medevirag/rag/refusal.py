"""T7 薄编排层 decide()（docs/05 §T7）：规则 → 置信度 → 阈值 的三级判定链。

判定顺序严格三级：① 非分数规则（短路）→ ② 算 conf → ③ 分数型规则 + 阈值区间。
先规则后阈值，因为规则能给出具体 reason，阈值只能给出"证据不足"这种笼统结论；
反过来会出现"明明是越界问题却报证据不足"的理由错配，打穿拒答理由正确率 ≥0.80。

本模块禁止任何 I/O 与 LLM 调用（写 PG qa_log 由 serve/ 统一做），因此可 100% 单测。
"""

from __future__ import annotations

import re
from typing import Any

from medevirag.common.config import get_settings
from medevirag.common.types import DecideResult, RefusalReason, RuleHit, Status
from medevirag.rag.confidence import CitationReport, SentenceSupport, compute_confidence
from medevirag.rag.refusal_rules import AskContext, evaluate_non_score, evaluate_score

# citation_coverage 的分母必须排除模板句（docs §T2 硬要求）：降级文案越长，
# 不过滤的话覆盖率反而虚高。
TEMPLATE_SENTENCE_RE = re.compile(
    r"(建议就医|仅供参考|仅供技术验证|不构成诊疗建议|请咨询.{0,6}医生|请立即拨打)"
)


def count_assertion_sentences(answer: str) -> int:
    """断言句总数：按中文句末标点切句，排除模板句。空正文记 1（有正文才谈覆盖）。"""
    from medevirag.rag.citations import split_sentences

    sents = [s for s in split_sentences(answer or "") if s.strip()]
    sents = [s for s in sents if not TEMPLATE_SENTENCE_RE.search(s)]
    return len(sents)


def build_citation_report(answer: str, vres: Any, marker_map: dict[str, str]) -> CitationReport:
    """把 S3 的 verifier.VerifyResult 适配成 docs §2.2 的 CitationReport（偏差 D2）。

    - cited_chunk_ids：该句合法 [Cn] 经 marker_map 换成的 chunk_id（落点通过）；
    - landing_ok：该句没有任何非法标记；
    - entailed：S3 SentenceCheck.supported（legal + entail>=0.5 才 True）；
    - sentences_total：断言句总数（排除模板句），citation_coverage 的分母。
    """
    supports: list[SentenceSupport] = []
    total_marks = 0
    dangling = 0
    for ck in getattr(vres, "sentences", []) or []:
        cited = tuple(marker_map.get(m, "") for m in ck.legal_markers) if marker_map else ()
        cited = tuple(c for c in cited if c)
        total_marks += len(ck.legal_markers) + len(ck.illegal_markers)
        dangling += len(ck.illegal_markers)
        supports.append(SentenceSupport(
            text=ck.sentence,
            cited_chunk_ids=cited,
            landing_ok=not ck.illegal_markers,
            entailed=bool(ck.supported),
        ))
    return CitationReport(
        sentences=tuple(supports),
        dangling_citation_count=dangling,
        total_citation_count=total_marks,
        sentences_total=count_assertion_sentences(answer),
    )


def refusal_config() -> dict[str, Any]:
    """把 Settings 的 refusal 段 + confidence.thresholds 拼成 decide 的扁平 cfg。

    docs 的 eval.yaml 把 thresholds 放 confidence 段、规则阈值放 refusal 段，
    decide 只吃一个 dict——在唯一一处合并，避免两处读配置。
    """
    s = get_settings()
    cfg = s.refusal.model_dump()
    cfg["thresholds"] = dict(s.confidence.thresholds)
    return cfg


def decide(ctx: AskContext, weights: dict[str, float], platt: dict[str, Any], rcfg: dict[str, Any],
           rerank_applied: bool = False) -> DecideResult:
    """三级判定链（docs §T7 原样），返回可落库的 DecideResult。"""
    # ① 非分数规则：与 confidence 无关，refuse 命中即短路
    hit: RuleHit | None = evaluate_non_score(ctx, rcfg)
    if hit is not None and hit.action == "refuse":
        return DecideResult("refused", 0.0, {}, hit.reason, hit.rule_id,
                            hit.template_key, None)

    # ② 复合置信度（degrade 命中不影响公式，只影响下游 status 判定）
    cited = [cid for s in ctx.citation.sentences for cid in s.cited_chunk_ids]
    bd = compute_confidence(ctx.candidates, cited, ctx.citation, weights, platt,
                            rerank_applied=rerank_applied)

    # ③ 分数型规则
    score_hit = evaluate_score(ctx, bd, rcfg)
    if score_hit is not None and score_hit.action == "refuse":
        return DecideResult("refused", bd.score, bd.as_dict(), score_hit.reason,
                            score_hit.rule_id, score_hit.template_key, None)

    # ④ 阈值区间：conf 决定 answered / partial；degrade 规则强制压到 partial
    th = rcfg["thresholds"]
    note: str | None = None
    if hit is not None and hit.action == "degrade":
        note = hit.template_key            # "stale_degrade" / "year_unknown"
    if bd.unknown_level_ratio >= rcfg["unknown_level_ratio"] and bd.score < th["answered"]:
        note = note or "evidence_unknown"

    if bd.score >= th["answered"] and note is None:
        status: Status = "answered"
    elif bd.score >= th["partial"]:
        status = "partial"
        note = note or "low_confidence"
    else:
        return DecideResult("refused", bd.score, bd.as_dict(),
                            RefusalReason.LOW_RELEVANCE, "R-CONF", "low_relevance", None)

    rule_id = hit.rule_id if hit is not None else (score_hit.rule_id if score_hit else None)
    return DecideResult(status, bd.score, bd.as_dict(), None, rule_id, None, note)
