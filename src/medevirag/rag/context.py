"""T4 证据装配：small-to-big 回填（docs/04 §T4）。

核心约束：父节点全文进 prompt 作上下文，但**只对子块编号**。父节点不可作为引用来源，
否则落点校验无从谈起（模型会引用整节）。

按 parent_id 聚合：同一父节点的多个子块共用一次父节点全文，避免 Top8 里同一节出现 3 遍白烧 token。
编号顺序与 state["reranked"] 严格一致——这是落点校验归零的经典成因，必须靠 marker_map 落账。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_PARENT_CHARS = 1200          # 父节点截断预算（中文 1 字≈1 token 的保守估算）
MAX_EVIDENCE_CHARS = 6000        # 整个证据区预算


@dataclass
class Evidence:
    marker: str                  # "C1"
    chunk_id: str
    parent_id: str
    doc_id: str
    doc_title: str
    section_path: list[Any]
    leaf_title: str              # section_path[-1]
    page: int
    locator: dict[str, Any]
    text: str
    context_header: str
    evidence_level: str
    rerank_score: float


def build_evidence(top: list[Any], chunk_meta: dict[str, Any], max_n: int = 8) -> list[Evidence]:
    """top: T1 的 reranked 列表（dict 或 RetrievedChunk）；chunk_meta: chunk_id -> meta。"""
    evs: list[Evidence] = []
    for i, c in enumerate(top[:max_n], start=1):
        cid = c["chunk_id"] if isinstance(c, dict) else c.chunk_id
        m = chunk_meta.get(cid) or {}
        # chunk_meta 为 S2 展示字段快照；parent_text 由 nodes.get_parent_text() 另行提供。
        evs.append(
            Evidence(
                marker=f"C{i}",
                chunk_id=cid,
                parent_id=str(m.get("parent_id", "") or ""),
                doc_id=str(m.get("doc_id", "") or ""),
                doc_title=str(m.get("doc_title", "") or ""),
                section_path=list(m.get("section_path") or []),
                leaf_title=(m.get("section_path") or [])[-1] if (m.get("section_path") or []) else "",
                page=int((m.get("locator") or {}).get("page", m.get("page_start", 0)) or 0),
                locator=dict(m.get("locator") or {}),
                text=str(m.get("text", "") or ""),
                context_header=str(m.get("context_header", "") or ""),
                evidence_level=str(m.get("evidence_level", "unknown") or "unknown"),
                rerank_score=float(getattr(c, "rerank_score", 0.0) if not isinstance(c, dict) else c.get("rerank_score", 0.0)),
            )
        )
    return evs


def render_evidence_blocks(evs: list[Evidence], parent_text: dict[str, str]) -> str:
    """按 parent_id 聚合：父节点全文渲染一次（不可引用），子块各自编号（可引用）。"""
    groups: dict[str, list[Evidence]] = {}
    for e in evs:
        groups.setdefault(e.parent_id, []).append(e)

    out: list[str] = []
    budget = MAX_EVIDENCE_CHARS
    for gi, (pid, items) in enumerate(groups.items(), start=1):
        head = items[0]
        out.append(f"### 证据组 {gi}｜{' > '.join(head.section_path)}")
        ptext = parent_text.get(pid, "")
        if ptext:
            cut = ptext[:MAX_PARENT_CHARS]
            if len(cut) < len(ptext):
                cut += "……（父节点其余内容已截断）"
            block = f"<父节点全文，仅供理解上下文，不得作为引用来源>\n{cut}"
        else:
            block = "<父节点不可得，本节仅提供下列子块片段>"
        budget -= len(block)
        out.append(block)
        for e in items:
            line = f"[{e.marker}] 《{e.doc_title}》{e.leaf_title}（p{e.page}）：{e.text}"
            budget -= len(line)
            out.append(line)
        if budget <= 0:
            out.append("（证据区已达 token 预算上限，后续证据块已省略）")
            break
    return "\n\n".join(out)


def marker_map(evs: list[Evidence]) -> dict[str, str]:
    """编号 → chunk_id 的映射必须随 evidence 一起落进 state，禁止事后重算。"""
    return {e.marker: e.chunk_id for e in evs}
