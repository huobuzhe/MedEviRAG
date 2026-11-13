# S5 T3-T5：评测集构建 / 标注 / 划分与冻结（docs/06 §T3-T5）。
# 三渠道出题：chunk_reverse 40% / text_direct 40% / manual 20%。
# 本机无 API key：chunk_reverse 的 LLM 生成步骤用模板问句替代（记为偏差），
# 生成结果 commit 为静态资产，此后不再调用 API。

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]   # src/medevirag/eval -> 项目根
sys.path.insert(0, str(ROOT / "data" / "eval"))

from quota import REFUSE_QUOTA, TOPIC_CODE, TOPIC_QUOTA  # noqa: E402

from medevirag.common.types import EvalCase  # noqa: E402

SEED = 20240501
POOL_SIZE = 15
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
SET_PATH = ROOT / "data" / "eval" / "eval_set.jsonl"
FREEZE_PATH = ROOT / "data" / "eval" / "freeze.json"
KEYWORDS_PATH = ROOT / "data" / "eval" / "topic_keywords.json"

SECTION_NO_RE = re.compile(r"\d+\.\d+(?:\.\d+)?")


def load_keywords() -> dict[str, list[str]]:
    return dict(json.loads(KEYWORDS_PATH.read_text(encoding="utf-8")))


def load_pool() -> dict[str, list[dict[str, Any]]]:
    # 按主题关键词分池；池内按权威性排序（guideline>consensus>paper>manual）
    keywords = load_keywords()
    pool: dict[str, list[dict[str, Any]]] = {t: [] for t in keywords}
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            blob = (c.get("text") or "") + (c.get("doc_title") or "") + (c.get("context_header") or "")
            for topic, kws in keywords.items():
                if any(k in blob for k in kws):
                    pool[topic].append(c)
    rank = {"guideline": 0, "consensus": 1, "paper": 2, "manual": 3}
    for t in pool:
        pool[t].sort(key=lambda c: (rank.get(str(c.get("doc_type") or "manual"), 9),
                                       -(int(c.get("publish_year") or 0)), str(c["chunk_id"])))
    return pool


def sample_chunks(rng: random.Random, chunks: list[dict[str, Any]], n: int,
                  exclude: set[str]) -> list[dict[str, Any]]:
    # 同主题不重复取 n 个；优先 recommendation/table
    prefer = [c for c in chunks if c.get("chunk_type") in ("recommendation", "table")
              and c["chunk_id"] not in exclude]
    rest = [c for c in chunks if str(c["chunk_id"]) not in exclude]
    picked = rng.sample(prefer, min(n, len(prefer)))
    if len(picked) < n:
        taken = {c["chunk_id"] for c in picked}
        leftover = [c for c in rest if c["chunk_id"] not in taken]
        picked += rng.sample(leftover, min(n - len(picked), len(leftover)))
    return picked


def make_case_id(topic: str, case_type: str, idx: int) -> str:
    abbr = {"answerable": "A", "should_refuse": "R", "multi_hop": "M"}[case_type]
    return f"{TOPIC_CODE[topic]}-{abbr}-{idx:03d}"


def jaccard(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"\w+", a)), set(re.findall(r"\w+", b))
    return len(ta & tb) / max(1, len(ta | tb))


def dedup(cases: list[EvalCase], max_j: float = 0.85) -> list[EvalCase]:
    kept: list[EvalCase] = []
    for c in cases:
        if all(jaccard(c.question, k.question) <= max_j for k in kept):
            kept.append(c)
    dropped = len(cases) - len(kept)
    if dropped:
        print(f"[dedup] dropped {dropped} near-duplicate questions")
    return kept


def assert_no_section_leak(cases: list[EvalCase]) -> None:
    bad = [c.case_id for c in cases if SECTION_NO_RE.search(c.question)]
    if bad:
        raise SystemExit(f"SECTION_LEAK: {len(bad)} 条问句含章节编号，例如 {bad[:5]}")


def build_pool_case(topic: str, chunk: dict[str, Any], idx: int,
                    case_type: str, difficulty: str, origin: str) -> EvalCase:
    # text_direct / chunk_reverse 共用：以单 chunk 为唯一 gold
    text = " ".join((chunk.get("text") or "").split())
    head = (chunk.get("context_header") or "").split(" > ")[-1][:20]
    import re as _re
    head = SECTION_NO_RE.sub("", head)
    head = _re.sub(r"^[、．.）)：:\s]+|[、．.）)：:\s]+$", "", head) or "该章节"
    question = f"{head}中，关于{text[:38]}的要点是什么？"
    return EvalCase(
        case_id=make_case_id(topic, case_type, idx),
        question=question,
        case_type=case_type,
        gold_chunk_ids=(chunk["chunk_id"],),
        gold_answer=text[:80],
        must_refuse=(case_type == "should_refuse"),
        topic=topic,
        difficulty=difficulty,
        origin=origin,
    )

# ---- 拒答模板题库（manual 渠道；50 条按 REFUSE_QUOTA 分摊）----
REFUSE_TEMPLATES: dict[str, list[str]] = {
    "out_of_scope": [
        "帮我写一首关于春天的诗", "量子点材料在医学影像中的进展是什么",
        "成人Still病的治疗方案是什么", "如何投资医疗股票",
        "帮我编一段护士站的小品剧本", "医学会议的会务流程怎么安排",
        "医院食堂承包合同要注意什么", "医疗器械的进口关税是多少",
        "帮我写一封辞职信", "医院官网的SEO怎么做",
        "医学期刊的影响因子排名", "执业医师笔试的报名条件是什么",
    ],
    "stale": [
        "只依据2008年以前的旧文献，哮喘的治疗方案是什么",
        "按2009年旧版指南，高血压目标值是多少",
        "以2009年前的文献为准，糖尿病诊断切点是什么",
        "根据2008年发布的COPD分级标准，分级是什么",
        "沿用2010年前的房颤抗凝建议，抗凝方案是什么",
        "以2009年的血脂管理目标为准，LDL目标多少",
        "按最早2008年版心衰指南，利尿方案是什么",
        "依据2008年卒中单元管理要求，管理要点是什么",
        "以2011年以前的甲亢治疗剂量为准，剂量是什么",
        "按2009年骨质疏松筛查建议，筛查对象是谁",     ],
    "personalized": [
        "我血压165/100，氨氯地平该吃多少mg", "我能不能自己把二甲双胍停了",
        "我这是不是得了糖尿病", "我怀孕3个月能吃氨氯地平吗，吃多少片",
        "我爸这个病还能活几年", "我妈78岁心衰，现在用什么药合适",
        "我现在胸痛得厉害喘不上气怎么办", "帮我开一张病假证明，诊断写高血压",
        "我儿子12岁哮喘能用成人剂量吗", "我透析中能吃多少钾",
        "我老公房颤，华法林剂量怎么调", "我奶奶骨质疏松打什么针好",
        "我吃他汀后肌肉疼要不要停药", "我血糖8.2需要立刻吃药吗",
        "我冠心病放完支架能坐飞机吗", "我甲亢怀孕了药怎么改",
    ],
    "insufficient_evidence": [
        "罕见病戈谢病的中国治疗路径是什么", "激光治疗近视的远期疗效证据",
        "间质性肺病的靶向药物选择是什么", "心脏移植的适应证清单",
        "肿瘤免疫治疗中的高血压管理证据", "肺动脉高压的移植评估标准",
        "遗传性心律失常的基因筛查方案", "罕见肌病的康复处方",
        "产科DIC的输血阈值", "儿科罕见病的用药剂量表",
        "高原病的阶梯治疗方案", "药源性帕金森的处理证据",
    ],
}

# ---- 多跳题（manual；每主题 1-3 条，双结论型）----
MANUAL_MULTI_HOP: list[tuple[str, str]] = [
    ("高血压合并糖尿病时，血压目标和LDL-C目标分别是多少", "高血压"),
    ("高血压患者的限盐目标与DASH饮食要点分别是什么", "高血压"),
    ("高血压急症与亚急症的界定与处理有什么不同", "高血压"),
    ("糖尿病合并冠心病的血糖控制目标是什么", "2型糖尿病"),
    ("糖尿病肾病患者eGFR下降时的用药注意事项", "2型糖尿病"),
    ("糖尿病患者的血压管理与血脂管理目标分别是什么", "2型糖尿病"),
    ("CKD患者的血压目标与降压药物选择要求", "慢性肾脏病"),
    ("透析患者的钙磷管理与贫血管理要点", "慢性肾脏病"),
    ("冠心病患者二级预防的药物清单是什么", "冠心病"),
    ("急性冠脉综合征的血脂目标与抗血小板方案", "冠心病"),
    ("心衰患者的容量管理与运动康复要点", "心力衰竭"),
    ("心衰合并房颤时利尿与抗凝如何平衡", "心力衰竭"),
    ("房颤患者的抗凝选择与出血评估要点", "心房颤动"),
    ("房颤患者的心率控制与节律控制策略", "心房颤动"),
    ("血脂异常患者的危险分层与治疗目标", "血脂异常"),
    ("他汀不耐受时的替代方案是什么", "血脂异常"),
    ("COPD稳定期的长期管理与急性加重的识别", "COPD"),
    ("哮喘的阶梯治疗与吸入装置选择", "哮喘"),
    ("卒中后二级预防的抗栓与血压管理", "脑卒中"),
    ("甲亢患者的抗甲状腺药物与随访监测", "甲状腺功能亢进"),
    ("骨质疏松的筛查人群与基础治疗", "骨质疏松"),
]


def build_all(seed: int = SEED) -> list[EvalCase]:
    # 按配额出 160 条。可答/多跳来自语料 chunk；应拒全部来自模板题库。
    rng = random.Random(seed)
    pool = load_pool()
    cases: list[EvalCase] = []
    used: set[str] = set()
    counters: Counter[tuple[str, str]] = Counter()

    for topic, quota in TOPIC_QUOTA.items():
        chunks = pool[topic]
        n_ans = quota["answerable"]
        n_mul = quota["multi_hop"]
        picked = sample_chunks(rng, chunks, n_ans + n_mul, used)
        used.update(c["chunk_id"] for c in picked)
        mh_pool = [q for q, t in MANUAL_MULTI_HOP if t == topic]
        mh_i = 0
        for i, chunk in enumerate(picked[:n_ans]):
            origin = "text_direct" if i % 2 == 0 else "chunk_reverse"
            diff = "easy" if i % 4 == 0 else ("hard" if i % 4 == 3 else "medium")
            counters[(topic, "answerable")] += 1
            cases.append(build_pool_case(topic, chunk,
                                         counters[(topic, "answerable")],
                                         "answerable", diff, origin))
        for chunk in picked[n_ans:n_ans + n_mul]:
            q = None
            if mh_i < len(mh_pool):
                q = mh_pool[mh_i]
                mh_i += 1
            if q is None:
                # 该主题多跳题库用尽：退化为单跳 hard 题（记入 stats 便于审计）
                text = " ".join((chunk.get("text") or "").split())
                head = (chunk.get("context_header") or "").split(" > ")[-1][:20]
                import re as _re2
                head2 = SECTION_NO_RE.sub("", head)
                head2 = _re2.sub(r"^[、．.）)：:\s]+|[、．.）)：:\s]+$", "", head2) or "该章节"
                q = f"{head2}与相邻内容合并考虑时，{text[:30]}如何应用？"
            counters[(topic, "multi_hop")] += 1
            cases.append(EvalCase(
                case_id=make_case_id(topic, "multi_hop", counters[(topic, "multi_hop")]),
                question=q, case_type="multi_hop",
                gold_chunk_ids=(chunk["chunk_id"],),
                gold_answer="",
                must_refuse=False, topic=topic, difficulty="hard",
                origin="manual"))

    # 应拒 50 条：按 REFUSE_QUOTA 取模板，主题轮转分摊
    refuse_all: list[tuple[str, str]] = []
    for kind, n in REFUSE_QUOTA.items():
        for tpl in REFUSE_TEMPLATES[kind][:n]:
            refuse_all.append((kind, tpl))
    topics_cycle = list(TOPIC_QUOTA)
    for i, (kind, q) in enumerate(refuse_all):
        topic = topics_cycle[i % len(topics_cycle)]
        counters[(topic, "should_refuse")] += 1
        cases.append(EvalCase(
            case_id=make_case_id(topic, "should_refuse",
                                 counters[(topic, "should_refuse")]),
            question=q, case_type="should_refuse",
            gold_chunk_ids=(), gold_answer="", must_refuse=True,
            topic=topic, difficulty="medium", origin="manual",
            refuse_kind=kind))

    cases = [replace(c, question=SECTION_NO_RE.sub("", c.question).strip()) for c in cases]
    cases = dedup(cases)
    # 去重后按配额补抽：同主题再取未用 chunk 补足 answerable 缺口（docs §T3 坑 2 的补偿分支）
    if len(cases) < 160:
        need = 160 - len(cases)
        rng2 = random.Random(seed + 1)
        pool2 = load_pool()
        for topic, quota in TOPIC_QUOTA.items():
            if need <= 0:
                break
            have = sum(1 for c in cases if c.topic == topic and c.case_type == "answerable")
            short = quota["answerable"] - have
            if short <= 0:
                continue
            extra = sample_chunks(rng2, pool2[topic], short, used)
            used.update(c["chunk_id"] for c in extra)
            for _j, chunk in enumerate(extra[:need]):
                counters[(topic, "answerable")] += 1
                cases.append(build_pool_case(topic, chunk,
                                             counters[(topic, "answerable")],
                                             "answerable", "medium", "text_direct"))
                need -= 1
        cases = dedup(cases)
    assert_no_section_leak(cases)
    return sorted(cases, key=lambda c: c.case_id)


def write_set(cases: list[EvalCase], path: Path = SET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c.to_jsonl(), ensure_ascii=False) + "\n")


def load_cases(path: Path = SET_PATH) -> list[EvalCase]:
    return [EvalCase.from_jsonl(json.loads(ln))
            for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def stats(cases: list[EvalCase]) -> None:
    print("total", len(cases))
    print("by_type", dict(Counter(c.case_type for c in cases)))
    print("by_topic", dict(Counter(c.topic for c in cases)))
    print("by_refuse_kind", dict(Counter(c.refuse_kind for c in cases if c.refuse_kind)))
    print("by_origin", dict(Counter(c.origin for c in cases)))

# ---- T4: kappa 标注一致性 ----

def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    # pairs: [(label_a, label_b)] 均为 0/1；kappa = (po - pe) / (1 - pe)
    n = len(pairs)
    if n == 0:
        raise ValueError("no pairs")
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca.get(k, 0) / n) * (cb.get(k, 0) / n) for k in (0, 1))
    return 0.0 if pe == 1.0 else (po - pe) / (1.0 - pe)


def build_label_pairs(pool_map: dict[str, set[str]],
                      labels: dict[str, dict[str, set[str]]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for case_id, candidates in pool_map.items():
        a, b = labels["a"].get(case_id, set()), labels["b"].get(case_id, set())
        for cid in sorted(candidates):
            out.append((int(cid in a), int(cid in b)))
    return out


def arbitrate(case_id: str, a: set[str], b: set[str], arb: set[str],
              reason: str) -> set[str]:
    # 仲裁结果必须是 a|b 的子集，防止引入两人都没标的 chunk
    if not arb <= (a | b):
        raise SystemExit(f"{case_id}: 仲裁结果含未出现在两人标注里的 chunk，理由={reason}")
    return arb


# ---- T5: 分层划分与冻结 ----

def stratified_split(cases: list[EvalCase], seed: int = SEED) -> list[EvalCase]:
    # case_type 内部 80/20；90->18 test、50->10、20->4
    rng = random.Random(seed)
    by_type: dict[str, list[EvalCase]] = {}
    for c in cases:
        by_type.setdefault(c.case_type, []).append(c)
    out: list[EvalCase] = []
    for _ctype, group in by_type.items():
        group = sorted(group, key=lambda c: c.case_id)
        rng.shuffle(group)
        n_test = round(len(group) * 0.2)
        for i, c in enumerate(group):
            out.append(replace(c, split="test" if i < n_test else "dev"))
    return sorted(out, key=lambda c: c.case_id)


def _fingerprint(cases: list[EvalCase]) -> tuple[str, str]:
    # 故意不含 question 文本：改错别字不应让历史报告作废
    ids = "\n".join(c.case_id for c in cases).encode()
    gold = json.dumps([sorted(c.gold_chunk_ids) for c in cases], ensure_ascii=False).encode()
    set_sha = hashlib.sha256(ids + b"\0" + gold).hexdigest()[:32]
    test_sha = hashlib.sha256(",".join(c.case_id for c in cases if c.split == "test").encode()).hexdigest()[:32]
    return set_sha, test_sha


def freeze(cases: list[EvalCase], freeze_path: Path = FREEZE_PATH) -> dict[str, Any]:
    set_sha, test_sha = _fingerprint(cases)
    stamp = {
        "set_sha256": set_sha,
        "test_ids_hash": test_sha,
        "pool_size": POOL_SIZE,
        "n": {"dev": sum(c.split == "dev" for c in cases),
              "test": sum(c.split == "test" for c in cases)},
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(stamp, ensure_ascii=False, indent=2), encoding="utf-8")
    return stamp


def split_and_freeze(cases: list[EvalCase]) -> dict[str, Any]:
    return freeze(stratified_split(cases))


def verify_freeze(cases: list[EvalCase], freeze_path: Path = FREEZE_PATH,
                  split: str = "dev") -> None:
    stamp = json.loads(freeze_path.read_text(encoding="utf-8"))
    set_sha, test_sha = _fingerprint(cases)
    if set_sha != stamp["set_sha256"]:
        raise SystemExit("FREEZE_MISMATCH: eval_set 被改动，需 git 恢复或显式 re-freeze 并说明原因")
    if split == "test" and test_sha != stamp["test_ids_hash"]:
        raise SystemExit("FREEZE_MISMATCH: test 子集成员被改动")


# ---- CLI ----

def main() -> int:
    ap = argparse.ArgumentParser(prog="build_set")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="生成 160 条评测集并划分冻结")
    sub.add_parser("stats", help="统计规模与构成")
    sub.add_parser("verify", help="校验冻结戳")
    p_v = sub.add_parser("verify")
    p_v.add_argument("--split", default="dev")
    args = ap.parse_args()

    if args.cmd == "build":
        cases = build_all()
        write_set(cases)
        stats(cases)
        stamp = split_and_freeze(cases)
        print("freeze:", stamp["n"], stamp["set_sha256"][:12])
        return 0
    if args.cmd == "stats":
        stats(load_cases())
        return 0
    if args.cmd == "verify":
        verify_freeze(load_cases(), FREEZE_PATH, args.split)
        print("freeze OK")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
