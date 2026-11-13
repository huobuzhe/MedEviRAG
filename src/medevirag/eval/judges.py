# S5 T6：三维 rubric judge（docs/06 §T6）。
# 忠实度 F1-F5 / 证据召回 E1-E3 / 合理拒答 R1-R4；veto 项失败该维直接 0 分。
# 纪律：证据顺序随机化（位置偏差对策）、温度 0、跨族断言、解析失败重试后单列。

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from medevirag.common.types import EvalCase, JudgeScore


@dataclass(frozen=True)
class RubricItem:
    item_id: str
    dim: str
    weight: int
    veto: bool
    ask: str


RUBRIC_V1: tuple[RubricItem, ...] = (
    RubricItem("F1", "faithfulness", 3, False, "答案中每条事实性断言都能在被引证据原文中找到"),
    RubricItem("F2", "faithfulness", 3, True, "答案里的数值/单位/阈值/年份与证据完全一致"),
    RubricItem("F3", "faithfulness", 2, False, "无与证据矛盾的否定词或程度词"),
    RubricItem("F4", "faithfulness", 2, False, "无超出证据范围的外推"),
    RubricItem("F5", "faithfulness", 1, False, "每个引用标记的位置与其支撑句语义对应"),
    RubricItem("E1", "evidence_recall", 3, False, "至少 1 个 gold chunk 出现在本次生成所用上下文中"),
    RubricItem("E2", "evidence_recall", 3, False, "gold 中的关键数值/结论出现在上下文里"),
    RubricItem("E3", "evidence_recall", 2, False, "无 gold 证据被元数据过滤误杀"),
    RubricItem("R1", "refusal", 4, True, "status 与 must_refuse 匹配"),
    RubricItem("R2", "refusal", 2, False, "拒答 reason 与 gold_refuse_kind 一致"),
    RubricItem("R3", "refusal", 2, False, "拒答文案含为什么拒+建议+就医指引"),
    RubricItem("R4", "refusal", 2, False, "可答题答案不含弃权话术"),
)

JUDGE_SYSTEM = (
    "你是严格的医学证据审核员。只依据给定的【证据原文】判断，不得使用你自己的医学知识。"
    "对每个 rubric 项必须给出 ok(bool) 与 why(不超过 30 字)。只输出 JSON，不要任何解释性前后缀。"
)


def shuffle_evidence(evidence: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    # 位置偏差对策：固定种子打乱证据顺序，judge 看不到原始排名
    idx = list(range(len(evidence)))
    random.Random(seed).shuffle(idx)
    return [evidence[i] for i in idx]


def build_prompt(case: EvalCase, answer: str, evidence: list[dict[str, Any]], dim: str,
                 seed: int, status: str, citation_map: dict[str, str]) -> list[dict[str, Any]]:
    items = [it for it in RUBRIC_V1 if it.dim == dim]
    ev = shuffle_evidence(evidence, seed)
    ev_text = ""
    for i, e in enumerate(ev):
        sec = " / ".join(str(x) for x in (e.get("section_path") or []))
        ev_text += ("[证据" + str(i + 1) + "] chunk_id=" + str(e["chunk_id"])
                    + " " + str(e.get("doc_title", "")) + " > " + sec
                    + "\n" + str(e.get("text", "")) + "\n\n")
    cmap = "\n".join(f"{k} -> {v}" for k, v in citation_map.items()) or "(无引用)"
    head = "【问题】" + case.question + "\n【系统答案】" + answer + "\n【拒答状态】" + status
    head += "\n【引用标记与 chunk_id 对应】\n" + cmap
    head += "\n【证据原文（顺序已打乱，编号与答案里的引用无关）】\n" + ev_text
    head += "\n【本维度要判的项】\n"
    for it in items:
        head += f"- {it.item_id}（权重{it.weight}"
        head += "，一票否决" if it.veto else ""
        head += f"）：{it.ask}\n"
    head += '\n【输出 JSON】{"items": [{"item_id": "id", "ok": true, "why": "<=30字"}]}'
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": head}]


def aggregate(dim: str, atomic: dict[str, bool]) -> tuple[float, bool]:
    # veto 项失败 -> 该维 0 分：F2 错剂量 / R1 应拒不拒，加权平均都会掩盖
    items = [it for it in RUBRIC_V1 if it.dim == dim]
    if any(it.veto and not atomic.get(it.item_id, False) for it in items):
        return 0.0, False
    num = sum(it.weight for it in items if atomic.get(it.item_id, False))
    den = sum(it.weight for it in items)
    score = num / den
    return score, score >= 0.80


def parse_judge_json(raw: str) -> dict[str, Any]:
    # judge 输出常带围栏/寒暄：index 截取比正则稳
    return dict(json.loads(raw[raw.index("{"): raw.rindex("}") + 1]))


def judge_dim(case: EvalCase, answer: str, evidence: list[dict[str, Any]], dim: str, llm: Any,
              judge_cfg: Any, status: str, citation_map: dict[str, str]) -> JudgeScore:
    want = {it.item_id for it in RUBRIC_V1 if it.dim == dim}
    atomic: dict[str, bool] = {}
    for attempt in range(judge_cfg.max_retry + 1):
        msgs = build_prompt(case, answer, evidence, dim,
                            judge_cfg.evidence_order_seed, status, citation_map)
        raw = llm.chat(msgs[0]["content"], msgs[1]["content"],
                       temperature=judge_cfg.temperature,
                       max_tokens=judge_cfg.max_tokens)
        try:
            obj = parse_judge_json(raw)
            atomic = {i["item_id"]: bool(i["ok"]) for i in obj["items"]}
            if set(atomic) != want:
                raise ValueError("rubric 项不匹配: " + str(sorted(atomic)) + " want " + str(sorted(want)))
            break
        except (ValueError, KeyError) as e:
            if attempt == judge_cfg.max_retry:
                return JudgeScore(case.case_id, dim, 0.0, False, {},
                                  judge_cfg.model, case.rubric_version,
                                  error="judge_parse_failed: " + str(e))
    score, passed = aggregate(dim, atomic)
    return JudgeScore(case.case_id, dim, score, passed, atomic,
                      judge_cfg.model, case.rubric_version)


def assert_cross_family(gen_model: str, judge_model: str) -> None:
    # 自我偏好对策：生成族与判分族不得相同
    fam = {"deepseek": "deepseek", "qwen": "qwen", "gpt": "openai", "claude": "anthropic"}
    if fam.get(gen_model.split("-")[0].lower()) == fam.get(judge_model.split("-")[0].lower()):
        raise SystemExit(
            f"CROSS_FAMILY_VIOLATION: gen={gen_model} judge={judge_model} 同族，judge 分不可信")


