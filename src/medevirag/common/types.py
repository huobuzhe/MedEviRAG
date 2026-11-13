"""全项目共享类型（docs/00 §6.1 契约，docs/01 §T4）。

约束住在类型里：Chunk.__post_init__ 把 00 §6.1 的四条硬约束写成断言，S1 每产一个
chunk 都在自查，而不是等到 S3 生成引用时才发现页码不对。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

DocType = Literal["guideline", "consensus", "paper", "manual"]
ChunkType = Literal["paragraph", "recommendation", "table", "figure_caption", "list"]
EvidenceLevel = Literal["A", "B", "C", "D", "unknown"]
RecStrength = Literal["I", "IIa", "IIb", "III", "unknown"]
GradeSystem = Literal["GRADE", "牛津", "指南自定", "unknown"]
Status = Literal["answered", "partial", "refused"]
Lang = Literal["zh", "en"]
HitSource = Literal["dense", "sparse", "fused", "rerank"]

EVIDENCE_LEVEL_ORDER: dict[str, int] = {"A": 4, "B": 3, "C": 2, "D": 1, "unknown": 0}
"""证据等级强弱序：用于 min_evidence_level 硬过滤与置信度加权。"""


class RefusalReason(str, Enum):
    """15 个拒答理由，与 docs/05 §T1 完全一致（替换 S3 之前的枚举）。

    17 条规则映射到这 15 个 reason（部分 reason 被多条规则共用，例如
    off_domain 同时覆盖 R-01 域外与 R-04 代开处方等不安全行为）。
    """

    OFF_DOMAIN = "off_domain"
    UNCOVERED_TOPIC = "uncovered_topic"
    NON_EVIDENCE_REQUEST = "non_evidence_request"
    UNSAFE_ACTION = "unsafe_action_request"
    STALE_EVIDENCE = "stale_evidence"
    INDIV_DOSAGE = "individualized_dosage"
    INDIV_PRESCRIBE = "individualized_prescription"
    INDIV_DIAGNOSIS = "individualized_diagnosis"
    SPECIAL_POPULATION = "special_population"
    EMERGENCY = "emergency"
    LOW_RELEVANCE = "low_relevance"
    NO_MARGIN = "no_margin"
    CITATION_UNGROUNDED = "citation_ungrounded"
    LOW_SUPPORT = "low_support"
    EVIDENCE_LEVEL_UNKNOWN = "evidence_level_unknown"


@dataclass(frozen=True)
class RuleHit:
    """一条规则命中后的结构化结果（docs/05 §T1）。

    action 决定后续处置：refuse 直接拒答；degrade 只降级置信度/加
    partial 备注，仍允许出答案。
    """

    rule_id: str
    reason: RefusalReason
    template_key: str
    action: Literal["refuse", "degrade"]
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecideResult:
    """decide() 的最终判定（docs/05 §T7）。

    breakdown 是四分量加权前的裸分量（rank/citation/evidence/margin），
    score 为加权后的综合置信度。note 记录降级原因（如 evidence_unknown、
    low_confidence、stale_degrade）。
    """

    status: Status
    confidence: float
    breakdown: dict[str, float]
    refusal_reason: RefusalReason | None = None
    rule_id: str | None = None
    template_key: str | None = None
    note: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    """命中规则的 detail（docs §T7 的 node 代码读 state['rule_detail'] 但没有写入处，
    这是 docs 的缺口；这里由 decide 原样带出，渲染拒答文案的插槽用，记偏差 D4）。"""


DISCLAIMER = (
    "本回答基于检索到的公开指南与文献自动生成，仅供医学专业人员参考，"
    "不能替代执业医师的临床判断，也不构成诊断或处方建议。"
)


@dataclass(frozen=True, slots=True)
class Locator:
    """回链原文的定位信息。"""

    page: int
    block_index: int
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    parent_id: str
    doc_id: str
    doc_title: str
    doc_type: DocType
    publisher: str
    publish_year: int
    lang: Lang
    section_path: list[str]
    section_level: int
    page_start: int
    page_end: int
    chunk_type: ChunkType
    text: str
    context_header: str
    evidence_level: EvidenceLevel
    rec_strength: RecStrength
    grade_system: GradeSystem
    token_len: int
    content_hash: str
    locator: Locator

    def __post_init__(self) -> None:
        if not self.content_hash.startswith("sha256:"):
            raise ValueError("content_hash 必须带 sha256: 前缀")
        if self.page_start > self.page_end:
            raise ValueError("page_start 不能大于 page_end")
        if self.page_start == self.page_end and self.locator.bbox is None:
            raise ValueError("page_start==page_end 时 bbox 不得为空，否则无法回链")
        if self.chunk_type in ("table", "recommendation") and self.page_start != self.page_end:
            raise ValueError("table/recommendation 禁止跨页合并")

    @staticmethod
    def compute_hash(text: str) -> str:
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "doc_type": self.doc_type,
            "publisher": self.publisher,
            "publish_year": self.publish_year,
            "lang": self.lang,
            "section_path": list(self.section_path),
            "section_level": self.section_level,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "context_header": self.context_header,
            "evidence_level": self.evidence_level,
            "rec_strength": self.rec_strength,
            "grade_system": self.grade_system,
            "token_len": self.token_len,
            "content_hash": self.content_hash,
            "locator": {
                "page": self.locator.page,
                "block_index": self.locator.block_index,
                "bbox": list(self.locator.bbox) if self.locator.bbox else None,
            },
        }


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """一路召回的命中记录；source 记录这条命中的最后处理阶段。"""

    chunk: Chunk
    score: float
    source: HitSource
    rank: int
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class Citation:
    """答案里一个 [Cn] 标记对应的原文段落。"""

    marker: str
    chunk_id: str
    doc_title: str
    section_path: list[str]
    page: int
    quote: str
    grounded: bool


@dataclass(frozen=True, slots=True)
class AnswerResult:
    question: str
    answer: str
    status: Status
    confidence: float
    citations: list[Citation] = field(default_factory=list)
    refusal_reason: RefusalReason | None = None
    timings: dict[str, float] = field(default_factory=dict)
    disclaimer: str = DISCLAIMER



# ------ S5 评测契约（docs/06 §T1）------

@dataclass(frozen=True, slots=True)
class EvalCase:
    """00 §6.2 eval_cases 八列 + jsonl 扩展字段。八列落 PG；扩展字段只留 jsonl。"""

    case_id: str
    question: str
    case_type: str
    gold_chunk_ids: tuple[str, ...]
    gold_answer: str = ""
    must_refuse: bool = False
    topic: str = ""
    difficulty: str = "medium"
    refuse_kind: str | None = None
    split: str = "dev"
    origin: str = "text_direct"
    annotators: tuple[str, ...] = ()
    rubric_version: str = "v1"

    def to_jsonl(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "question": self.question,
            "case_type": self.case_type,
            "gold_chunk_ids": list(self.gold_chunk_ids),
            "gold_answer": self.gold_answer, "must_refuse": self.must_refuse,
            "topic": self.topic, "difficulty": self.difficulty,
            "refuse_kind": self.refuse_kind, "split": self.split,
            "origin": self.origin, "annotators": list(self.annotators),
            "rubric_version": self.rubric_version,
        }

    @classmethod
    def from_jsonl(cls, d: dict[str, Any]) -> EvalCase:
        return cls(
            case_id=str(d["case_id"]), question=str(d["question"]),
            case_type=str(d["case_type"]),
            gold_chunk_ids=tuple(str(x) for x in d.get("gold_chunk_ids") or ()),
            gold_answer=str(d.get("gold_answer") or ""),
            must_refuse=bool(d.get("must_refuse", False)),
            topic=str(d.get("topic") or ""),
            difficulty=str(d.get("difficulty") or "medium"),
            refuse_kind=d.get("refuse_kind"),
            split=str(d.get("split") or "dev"),
            origin=str(d.get("origin") or "text_direct"),
            annotators=tuple(d.get("annotators") or ()),
            rubric_version=str(d.get("rubric_version") or "v1"),
        )


@dataclass(frozen=True, slots=True)
class JudgeScore:
    """三维 judge 的一维结果；error 非空则不进分母，单列 judge_error_rate。"""

    case_id: str
    dim: str
    score: float
    passed: bool
    atomic: dict[str, bool]
    judge_model: str
    rubric_version: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RunContext:
    """一次评测 run 的指纹与台账（run_id 可追溯）。"""

    run_id: str
    git_sha: str
    config_hash: str
    split: str
    llm_mode: str
    judge_model: str
    started_at: str
    seed: int
    tokens_used: int = 0
    api_calls: int = 0
