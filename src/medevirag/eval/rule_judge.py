# P1：可解释的规则判分器。实现与 judge_dim 兼容的 .chat()，但按 rubric 逐项真判分，
# 取代 OfflineJudgeLLM（全 ok=True 的桩）。零成本、完全可复现，用于验证评测管线与 judge 行为。
# 注意：它不是真实 LLM judge；真实 judge 需 API key，接入点见 scripts/eval/run_p1.py 的 --real 说明。
from __future__ import annotations

import json
import re
from typing import Any

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
ITEM_RE = re.compile(r"- ([FER]\d)")
EXTRAPOLATION = ("因此建议", "一般来说", "通常来说", "肯定能", "一定能", "必然")
CONTRADICTION = ("无需", "不必", "不用", "不需要")
REFUSAL_WORDS = ("不提供", "无法提供", "就医提示", "需由医师")


def _between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    tail = text.split(start, 1)[1]
    return tail.split(end, 1)[0].strip() if end in tail else tail.strip()


class RuleJudgeLLM:
    """按 rubric 逐项判定的确定性 judge；用 bind() 补齐 judge_dim 没传进 prompt 的 case 元数据。"""

    def __init__(self, normalize_format: bool = True, ignore_length: bool = True,
                 shuffle_invariant: bool = True) -> None:
        self.calls = 0
        self.normalize_format = normalize_format
        self.ignore_length = ignore_length
        self.shuffle_invariant = shuffle_invariant
        self.case: dict[str, Any] = {}

    def bind(self, case: Any) -> None:
        self.case = {
            "case_id": case.case_id,
            "must_refuse": bool(case.must_refuse),
            "refuse_kind": case.refuse_kind,
            "gold_chunk_ids": tuple(case.gold_chunk_ids),
            "gold_answer": case.gold_answer,
        }

    def chat(self, system: str, user: str, *, temperature: float = 0.0,
             max_tokens: int = 512) -> str:
        self.calls += 1
        answer_raw = _between(user, "【系统答案】", "【拒答状态】")
        status = _between(user, "【拒答状态】", "【引用标记")
        cmap_text = _between(user, "【引用标记与 chunk_id 对应】", "【证据原文")
        evidence = _between(user, "【证据原文", "【本维度要判的项】")
        head = user.split("【本维度要判的项】", 1)[1] if "【本维度要判的项】" in user else ""
        item_ids = ITEM_RE.findall(head)
        dim = ""
        for i in item_ids:
            dim = {"F": "faithfulness", "E": "evidence_recall", "R": "refusal"}[i[0]]
            break

        answer = answer_raw
        if self.normalize_format:
            answer = re.sub(r"[*_`#>\-]", "", answer)
        if self.ignore_length:
            answer = re.sub(r"\s+", " ", answer).strip()

        ans_nums = set(NUM_RE.findall(answer))
        ev_nums = set(NUM_RE.findall(evidence))
        has_evidence = bool(evidence.strip())
        gold_in_evidence = any(g in evidence for g in self.case.get("gold_chunk_ids", ()))
        cite_ids = set(re.findall(r"(\d+)\s*->", cmap_text))
        ans_cites = set(re.findall(r"\[(\d+)\]", answer))
        is_refused = status.startswith("refused")
        must_refuse = bool(self.case.get("must_refuse", False))

        checks: dict[str, bool] = {
            "F1": bool(answer) and ans_nums <= ev_nums and not any(w in answer for w in EXTRAPOLATION),
            "F2": ans_nums <= ev_nums,
            "F3": not (any(w in answer for w in CONTRADICTION) and "需" in evidence),
            "F4": not any(w in answer for w in EXTRAPOLATION),
            "F5": bool(ans_cites) and ans_cites <= cite_ids,
            "E1": has_evidence and gold_in_evidence,
            "E2": has_evidence and (bool(ev_nums) or bool(self.case.get("gold_answer"))),
            "E3": has_evidence,
            "R1": is_refused == must_refuse,
            "R2": (not is_refused) or any(w in answer for w in REFUSAL_WORDS),
            "R3": (not is_refused) or ("不提供" in answer and ("就医" in answer or "医师" in answer)),
            "R4": is_refused or not any(w in answer for w in REFUSAL_WORDS),
        }
        items = []
        for i in item_ids:
            ok = checks.get(i, False)
            items.append({"item_id": i, "ok": ok,
                          "why": "规则判定" if ok else "规则命中缺陷:" + dim})
        return json.dumps({"items": items}, ensure_ascii=False)


def score_from_items(dim: str, atomic: dict[str, bool]) -> float:
    """复用 judges.aggregate 的口径（veto 归零 + 加权平均），不复制实现。"""
    from medevirag.eval.judges import aggregate
    return aggregate(dim, atomic)[0]
