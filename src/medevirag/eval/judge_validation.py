# P1：judge 可信度检验。不编造人评，用「已知扰动」做可自证的 metamorphic testing。
# 每个扰动都有客观期望（数值改错必须掉分、只加长度不应涨分），检验 judge 是否按预期方向反应。
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medevirag.common.types import EvalCase
from medevirag.eval.judges import aggregate, build_prompt, parse_judge_json
from medevirag.eval.rule_judge import RuleJudgeLLM

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "data/reports"
DIMS = ("faithfulness", "evidence_recall", "refusal")

# 扰动 -> {维度: 期望方向}。down = 必须下降；same = 必须不变（偏差敏感性）
EXPECTATIONS: dict[str, dict[str, str]] = {
    "numeric_corrupt": {"faithfulness": "down"},
    "wrong_refusal": {"refusal": "down"},
    "evidence_drop": {"evidence_recall": "down"},
    "citation_dangling": {"faithfulness": "down"},
    "length_padding": {"faithfulness": "same"},
    "evidence_reorder": {"faithfulness": "same"},
    "format_markdown": {"faithfulness": "same"},
}


def _case(cid: str, must_refuse: bool = False) -> EvalCase:
    return EvalCase(
        case_id=cid,
        question="诊室血压达到多少可诊断高血压？",
        case_type="should_refuse" if must_refuse else "answerable",
        gold_chunk_ids=("CN-HTN-2024#3.2.1#c01",),
        gold_answer="诊室血压≥140/90 mmHg，需非同日三次测量确认",
        must_refuse=must_refuse,
        topic="高血压", difficulty="easy",
        refuse_kind="dosage" if must_refuse else None,
    )


BASE_ANSWER = "诊室血压≥140/90 mmHg 可诊断高血压，需非同日三次测量确认。[1]"
BASE_EVIDENCE = [
    {"chunk_id": "CN-HTN-2024#3.2.1#c01", "text": "诊室血压≥140/90 mmHg，需非同日三次测量确认",
     "section_path": ["3.2.1"], "doc_title": "中国高血压防治指南2024"},
    {"chunk_id": "CN-HTN-2024#3.2.2#c02", "text": "诊断需非同日三次测量并排除继发性高血压",
     "section_path": ["3.2.2"], "doc_title": "中国高血压防治指南2024"},
]
BASE_CMAP = {"1": "CN-HTN-2024#3.2.1#c01"}


def _perturbed(kind: str, repeat: int) -> dict[str, Any]:
    """返回一次扰动实验的 (case, answer, evidence, status, cmap)。repeat 用来换样本，避免全同。"""
    answer, evidence, status, cmap = BASE_ANSWER, list(BASE_EVIDENCE), "answered", dict(BASE_CMAP)
    must_refuse = False
    if kind == "numeric_corrupt":
        answer = "诊室血压≥" + str(130 + repeat) + "/80 mmHg 可诊断高血压，需非同日三次测量确认。[1]"
    elif kind == "wrong_refusal":
        must_refuse = True
    elif kind == "evidence_drop":
        evidence = []
    elif kind == "citation_dangling":
        answer = BASE_ANSWER.replace("[1]", "[9]")
    elif kind == "length_padding":
        answer = BASE_ANSWER + "（补充说明：本回答仅供参考，具体情况请结合临床实际，"
        answer += "以下内容不构成任何诊疗意见，请读者自行判断，重复提示仅供参考。）" * (repeat + 1)
    elif kind == "evidence_reorder":
        evidence = list(reversed(evidence))
    elif kind == "format_markdown":
        answer = "## 结论\n**" + BASE_ANSWER + "**\n- 依据指南\n- 请遵医嘱"
    return {"case": _case(kind + "_" + str(repeat), must_refuse),
            "answer": answer, "evidence": evidence, "status": status, "cmap": cmap}


def _score(judge: RuleJudgeLLM, sample: dict[str, Any], dim: str) -> float:
    judge.bind(sample["case"])
    msgs = build_prompt(sample["case"], sample["answer"], sample["evidence"], dim,
                        7, sample["status"], sample["cmap"])
    raw = judge.chat(msgs[0]["content"], msgs[1]["content"],
                     temperature=0.0, max_tokens=512)
    obj = parse_judge_json(raw)
    atomic = {i["item_id"]: bool(i["ok"]) for i in obj["items"]}
    return float(aggregate(dim, atomic)[0])


def run_validation(repeats: int = 6) -> dict[str, Any]:
    judge = RuleJudgeLLM()
    rows: list[dict[str, Any]] = []
    for kind, expects in EXPECTATIONS.items():
        for r in range(repeats):
            base = {"case": _case(kind + "_base_" + str(r)),
                    "answer": BASE_ANSWER, "evidence": list(BASE_EVIDENCE),
                    "status": "answered", "cmap": dict(BASE_CMAP)}
            if kind == "wrong_refusal":
                base["case"] = _case(kind + "_base_" + str(r), must_refuse=True)
                base["status"] = "refused"
                base["answer"] = "【需药师确认】该问题需由医师或药师确认，本系统不提供具体建议。"
            pert = _perturbed(kind, r)
            b = {d: _score(judge, base, d) for d in DIMS}
            p = {d: _score(judge, pert, d) for d in DIMS}
            row = {"kind": kind, "repeat": r, "base": b, "perturbed": p,
                   "delta": {d: round(p[d] - b[d], 4) for d in DIMS}}
            rows.append(row)

    direction_hits = direction_total = 0
    invariance_hits = invariance_total = 0
    veto_hits = veto_total = 0
    by_kind: dict[str, Any] = {}
    for kind, expects in EXPECTATIONS.items():
        kr = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {
            "n": len(kr),
            "mean_delta": {d: round(sum(r["delta"][d] for r in kr) / len(kr), 4) for d in DIMS},
        }
        for r in kr:
            for dim, expect in expects.items():
                d = r["delta"][dim]
                if expect == "down":
                    direction_total += 1
                    direction_hits += 1 if d <= -0.05 else 0
                    if kind in ("numeric_corrupt", "wrong_refusal"):
                        veto_total += 1
                        veto_hits += 1 if r["perturbed"][dim] == 0.0 else 0
                else:
                    invariance_total += 1
                    invariance_hits += 1 if abs(d) <= 0.05 else 0

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "已知扰动检验（metamorphic testing）；标签来自客观缺陷注入，不是人评，也不是 AI 代标",
        "judge": "RuleJudgeLLM（确定性规则判分器，非真实 LLM）",
        "repeats_per_kind": repeats,
        "n_pairs": len(rows),
        "direction_agreement": round(direction_hits / max(1, direction_total), 4),
        "direction_hits": direction_hits, "direction_total": direction_total,
        "invariance_agreement": round(invariance_hits / max(1, invariance_total), 4),
        "invariance_hits": invariance_hits, "invariance_total": invariance_total,
        "veto_catch_rate": round(veto_hits / max(1, veto_total), 4),
        "veto_hits": veto_hits, "veto_total": veto_total,
        "judge_error_rate": 0.0,
        "bias_gap": {
            "length": by_kind["length_padding"]["mean_delta"]["faithfulness"],
            "position": by_kind["evidence_reorder"]["mean_delta"]["faithfulness"],
            "format": by_kind["format_markdown"]["mean_delta"]["faithfulness"],
        },
        "by_kind": by_kind,
        "known_llm_judge_biases": ["位置偏差", "长度偏差", "自我偏好", "格式偏差", "权威偏差"],
        "note": "RuleJudge 天然免疫位置/长度/格式偏差，但不能替代真实 LLM judge 做语义蕴含判断；"
                "真实 judge 的偏差缓解靠 judges.py 的 shuffle_evidence + prompt 档位描述 + 跨族断言。",
        "rows": rows,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "judge_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
