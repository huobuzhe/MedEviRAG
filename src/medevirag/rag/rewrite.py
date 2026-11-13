"""T3 查询改写：把临床问句扩成 1~3 条检索变体（docs/04 §T3）。

为什么需要：RRF 融合已经多路召回，但用户原句常常"只问结论不写限定词"（如"要不要吃药"）。
改写负责补回同义词 / 限定词，提高 BM25 精确命中率。代价是每多一条 query 就多一遍 fusion 开销。

什么时候不该改写（docs §T3 明确）：含剂量/阈值/单位、含缩写/药名/引号内原文、极短查询（≤6 字）。
这些情况下改写会改丢数值或让 BM25 词项失效，所以原样返回单条。
"""

from __future__ import annotations

import re

# 命中这些模式的问题通常已含精确检索所需的数值/缩写/原文，改写只会引入噪声。
PRECISE_PATTERNS = [
    r"\d+(\.\d+)?\s?(?:mg|g|ml|mmHg|mmol|mmol/|kg|次/|qd|bid|tid|\d+%|<\d+|>\d+|=)",  # 数值+单位
    r"\b(?:HFrEF|HFpEF|ACEI|ARB|CCB|eGFR|LDL|HbA1c|BMI|SBP|DBP|NYHA|CKD)\b",  # 缩写
    r"[<>]=?",  # 比较/阈值符
    r"[\"\u201c\u201d].{2,}[\"\u201c\u201d]",  # 引号内原文短语
]

TERM_DICT = {
    "血压": ["高血压", "血压控制"],
    "降压": ["降压治疗", "抗高血压"],
    "血糖": ["高血糖", "血糖控制"],
    "血栓": ["静脉血栓", "抗凝"],
    "心梗": ["急性心肌梗死", "心肌梗死"],
    "中风": ["卒中", "脑血管事件"],
    "吃药": ["药物治疗", "服药"],
    "能不能": ["是否适用", "适应证"],
}


def is_precise(question: str) -> bool:
    return any(re.search(p, question) for p in PRECISE_PATTERNS) or len(question.strip()) <= 6


def build_queries(question: str, max_q: int = 3) -> list[str]:
    """返回 1~3 条检索变体；精确查询只返回原句。"""
    q = question.strip()
    if not q:
        return [q]
    if is_precise(q) or max_q <= 1:
        return [q]

    seen = {q}
    out = [q]
    for key, syns in TERM_DICT.items():
        if key in q and len(out) < max_q:
            for syn in syns:
                variant = q.replace(key, syn)
                if variant not in seen:
                    seen.add(variant)
                    out.append(variant)
                    break
        if len(out) >= max_q:
            break
    return out[:max_q]
