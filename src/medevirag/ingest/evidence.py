"""证据等级与推荐强度抽取（docs/02 4.5 T5）。

正则 + 锚点窗口双闸：只有"证据等级：B"这类显式标注才算命中，且必须落在含临床
语境的窗口内。宁可漏，不可错——伪造出来的等级会直接污染 S5 的分级推荐与拒答。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

EVIDENCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("GRADE", re.compile(r"证据\s*(?:等级|级别|质量|体质量)\s*[:：]?\s*([ABCD])(?![A-Za-z0-9])")),
    ("牛津", re.compile(r"(?:证据\s*水平|证据\s*级别|Level\s+of\s+evidence)\s*[:：]?\s*([1-5])(?!\d)")),
    (
        "指南自定",
        re.compile(r"推荐\s*(?:强度|级别|类别|等级)\s*[:：]?\s*(III|IIa|IIb|IV|II|I)(?![A-Za-z0-9])"),
    ),
)

# 中文指南最常见的是把等级与强度合并写进括号："……（A，强推荐）""……（B，弱推荐）"。
# 这条必须单列：它一次给出两个字段，且没有"证据等级："这类提示词可依赖。
COMBO_PATTERN = re.compile(r"[（(]\s*([ABCD])\s*[，,]\s*(强推荐|弱推荐|推荐)\s*[）)]")
COMBO_STRENGTH = {"强推荐": "I", "弱推荐": "IIa", "推荐": "I"}

# "水平" 补进锚点：真实指南里 "证据水平：2" 是牛津体系的常见写法，
# 缺了它这条标注就只剩 "证据" 一个锚点，会被双闸误杀。
ANCHORS: tuple[str, ...] = (
    "推荐", "证据", "级别", "等级", "水平", "强度", "强推荐", "弱推荐",
    "患者", "治疗", "临床", "研究", "质量",
)
WINDOW = 120

# EvidenceLevel 字面量只有 A/B/C/D，牛津 1~5 级按 GRADE 的强弱顺序对齐到 A~D，
# 5 级（病例报告/专家意见）在 GRADE 里没有对应档位，归 unknown 而不是硬塞一个 D。
OXFORD_TO_LEVEL: dict[str, str] = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "unknown"}
# RecStrength 字面量只有 I/IIa/IIb/III，II 与 IV 无法无损映射，归 unknown。
REC_KEEP = frozenset({"I", "IIa", "IIb", "III"})


def _window_ok(text: str, start: int, end: int, min_anchors: int = 2) -> bool:
    """命中点前后 WINDOW 字符内至少出现 min_anchors 个不同锚点词，压掉目录与免责声明。"""
    lo = max(0, start - WINDOW)
    hi = min(len(text), end + WINDOW)
    window = text[lo:hi]
    hits = sum(1 for a in ANCHORS if a in window)
    return hits >= min_anchors


def extract_evidence(text: str) -> dict[str, str]:
    """返回 {evidence_level, rec_strength, grade_system}，缺失一律 "unknown"。"""
    level = "unknown"
    rec = "unknown"
    system = "unknown"
    if not text or not text.strip():
        return {"evidence_level": level, "rec_strength": rec, "grade_system": system}
    norm = unicodedata.normalize("NFKC", text)

    for m in COMBO_PATTERN.finditer(norm):
        if not _window_ok(norm, m.start(), m.end()):
            continue
        level = m.group(1)
        rec = COMBO_STRENGTH.get(m.group(2), "unknown")
        system = "指南自定"
        break

    for name, pat in EVIDENCE_PATTERNS:
        for m in pat.finditer(norm):
            if not _window_ok(norm, m.start(), m.end()):
                continue
            val = m.group(1)
            if name == "GRADE" and level == "unknown":
                level = val
                if system == "unknown":
                    system = "GRADE"
                break
            if name == "牛津" and level == "unknown":
                level = OXFORD_TO_LEVEL.get(val, "unknown")
                if level != "unknown" and system == "unknown":
                    system = "牛津"
                break
            if name == "指南自定" and rec == "unknown" and val in REC_KEEP:
                rec = val
                if system == "unknown":
                    system = "指南自定"
                break
    return {"evidence_level": level, "rec_strength": rec, "grade_system": system}


def coverage_report(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """统计可得率：整体 + 每篇。分母是 chunk 总数，分子是字段 != unknown 的块数。"""
    total = len(chunks)
    ev = sum(1 for c in chunks if c.get("evidence_level", "unknown") != "unknown")
    rec = sum(1 for c in chunks if c.get("rec_strength", "unknown") != "unknown")
    per_doc: dict[str, dict[str, Any]] = {}
    for c in chunks:
        doc_id = str(c.get("doc_id", ""))
        slot = per_doc.setdefault(doc_id, {"chunks": 0, "evidence_level_hits": 0, "coverage": 0.0})
        slot["chunks"] = int(slot["chunks"]) + 1
        if c.get("evidence_level", "unknown") != "unknown":
            slot["evidence_level_hits"] = int(slot["evidence_level_hits"]) + 1
    for slot in per_doc.values():
        n = int(slot["chunks"])
        slot["coverage"] = round(int(slot["evidence_level_hits"]) / n, 4) if n else 0.0
    return {
        "chunks": total,
        "evidence_level_hits": ev,
        "rec_strength_hits": rec,
        "evidence_level_coverage": round(ev / total, 4) if total else 0.0,
        "rec_strength_coverage": round(rec / total, 4) if total else 0.0,
        "per_doc": per_doc,
    }
