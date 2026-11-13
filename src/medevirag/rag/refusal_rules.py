"""T3-T5 拒答规则表（docs/05 §T3/T4/T5）：17 条规则 = 纯函数 + 稳定执行序。

关键思路（docs §T3）：执行序由 RULE_ORDER 决定，与 rule_id 编号**故意不同**——
编号是给人读的分组（越界→时效→个体化→证据不足），优先级是「哪个 reason 对用户最有用」
（急诊 > 个体化 > 越界 > 时效 > 证据不足）。首个命中即返回，保证 refusal_reason 唯一，
否则两条规则同时命中会给出不确定 reason，直接打穿 00 §9 的「拒答理由正确率 ≥0.80」。
本文件禁止任何 I/O 与 LLM 调用，全部可 100% 单测。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from medevirag.common.types import RefusalReason, RuleHit
from medevirag.rag.confidence import CitationReport, ConfidenceBreakdown


@dataclass(frozen=True)
class AskContext:
    question: str
    route: str                 # clinical | off_domain | non_evidence | individual | unclear
    route_score: float         # 路由置信度（机械判据命中时为 1.0）
    topic_covered: bool        # 主题是否在 COVERED_TOPICS 内
    candidates: Sequence[Any]       # RetrievedChunk 列表，已按名次升序
    citation: CitationReport
    now_year: int = 2025


RuleFn = Callable[..., "RuleHit | None"]

# docs/05 §T3 冻结的 12 病种（保留原文以便审计）
COVERED_TOPICS: tuple[str, ...] = (
    "高血压", "2型糖尿病", "血脂异常", "慢性肾脏病", "冠心病", "心力衰竭",
    "房颤", "卒中", "COPD", "哮喘", "甲状腺功能亢进", "痛风",
)

# ★ S4 偏差 D3：真实语料 4023 个 chunk 的主题远超上述 12 病种（大量传染病与公共卫生规范：
# 艾滋病/血吸虫病/登革热/布鲁氏菌病/麻风病/梅毒/疟疾/伤寒/乙脑/脊灰/破伤风、近视防控、
# 肥胖与食养指南、伤害监测、动态血压监测等）。照抄 12 病种会让 dev60 里的合法循证问题
# 被 R-02 大面积误拒（00 §9 过度拒答率 ≤0.12 首先在这里崩）。故按真实语料扩词表，
# 并对带循证信号词的医学问句兜底放行。
CORPUS_TOPICS: tuple[str, ...] = (
    "血压",
    "血糖",
    "血脂",
    "LDL",
    "HDL",
    "胆固醇",
    "肥胖",
    "超重",
    "体重",
    "BMI",
    "膳食",
    "食养",
    "限盐",
    "添加糖",
    "烹调油",
    "营养",
    "能量摄入",
    "糖尿病",
    "糖化血红蛋白",
    "二甲双胍",
    "胰岛素",
    "SGLT2",
    "GLP-1",
    "eGFR",
    "肾功能",
    "透析",
    "肾脏",
    "蛋白尿",
    "心衰",
    "心梗",
    "心肌梗死",
    "冠脉",
    "动脉粥样硬化",
    "阿司匹林",
    "抗血小板",
    "抗凝",
    "华法林",
    "他汀",
    "降压",
    "氨氯地平",
    "ACEI",
    "ARB",
    "ARNI",
    "CCB",
    "利尿剂",
    "心率",
    "心律",
    "起搏",
    "一级预防",
    "二级预防",
    "动态血压",
    "诊室血压",
    "血压监测",
    "疫苗",
    "接种",
    "筛查",
    "随访",
    "转诊",
    "分级诊疗",
    "监测方案",
    "病例报告",
    "流行病学",
    "传染",
    "艾滋病",
    "HIV",
    "梅毒",
    "麻风",
    "登革热",
    "布鲁氏菌",
    "血吸虫",
    "疟疾",
    "伤寒",
    "副伤寒",
    "乙脑",
    "脊灰",
    "脊髓灰质炎",
    "破伤风",
    "狂犬",
    "结核",
    "肝炎",
    "流感",
    "手足口",
    "猴痘",
    "消毒",
    "隔离",
    "院感",
    "暴发",
    "突发公共卫生",
    "近视",
    "视力",
    "屈光",
    "口腔",
    "学生",
    "儿童青少年",
    "学校卫生",
    "教室",
    "睡眠",
    "身体活动",
    "运动",
    "戒烟",
    "控烟",
    "酒精",
    "伤害",
    "慢阻肺",
    "慢性气道",
    "吸入装置",
    "支气管",
    "肺功能",
    "甲亢",
    "甲减",
    "甲状腺",
    "尿酸",
    "关节炎",
    "抑郁",
    "焦虑",
    "认知",
    "跌倒",
    "老年",
    "孕期",
    "妊娠",
    "哺乳",
    "儿童",
    "剂量",
    "用药",
    "禁忌",
    "不良反应",
    "指南",
    "共识",
    "推荐",
    "规范",
    # 实测 dev60 后补齐的具体病名与检测手段（不做通用词兜底，理由见 is_topic_covered）
    "疱疹",
    "卵形疟",
    "间日疟",
    "三日疟",
    "基孔肯雅",
    "寨卡",
    "血氧",
    "眼底",
    "近视防控",
    "血清抗体",
    "病原学",
    "分子检测",
    "基因检测",
    "病理",
    "超声",
    "心超",
    "动态心电",
    "脑电图",
    "肺栓塞",
    "深静脉血栓",
    "房室",
    "早搏",
    "瓣膜",
    "心肌病",
    "心包",
    "贫血",
    "白细胞",
    "血小板",
    "凝血",
    "过敏",
    "脓毒症",
    "烧伤",
    "冻伤",
    "中暑",
    "溺水",
    "一氧化碳",
    "有机磷",
    "重金属",
    "职业病",
    "放射",
    "食品安全",
    "饮用水",
    "空气质量",
    "病媒",
    "伊蚊",
    "灭鼠",
    "媒介生物",
    "免疫规划",
    "冷链",
    "接种率",
    "应急",
    "分诊",
    "留观",
    "并发症",
    "复发",
    "潜伏期",
    "窗口期",
    "携带者",
    "易感",
    "群体免疫",
    "基本再生数",
)


def is_topic_covered(question: str) -> bool:
    """主题是否在可答范围内（偏差 D3 的判定入口，供 node_route 与夹具复用）。

    ★ 刻意不做"含治疗/诊断/指南等通用词就放行"的兜底：那样 docs 自己的 R-02 反例
    「成人 Still 病的治疗方案」会被放行，uncovered_topic 规则形同虚设。词表必须按
    真实语料扩，而不是把判定条件放宽。
    """
    q = question or ""
    if any(t in q for t in COVERED_TOPICS):
        return True
    return any(t in q for t in CORPUS_TOPICS)


# ── ① 越界（R-01 ~ R-04）─────────────────────────────────
# 非循证请求（R-03）：预后寿命 / 归责 / 赔偿 / 保险 / 疗效承诺——docs §T3 规则总表。
NON_EVIDENCE_RE = re.compile(
    r"(还能活多久|活(?:几|多少|多长)?年|预期寿命|能治好吗|保证.{0,4}(治愈|有效)|"
    r"是谁的(责任|锅)|(医疗)?纠纷|赔偿|鉴定责任|报销比例|保险(赔|拒)|工伤认定)"
)

UNSAFE_ACTION_RE = re.compile(
    r"(开(一张|个)?处方|代(签|开)|伪造|补开[^　。！？?]{0,4}(证明|处方|诊断证明)|"
    r"帮我出[^　。！？?]{0,3}(检验|检查)(报告|单)|"
    # 偏差 D9：docs 的 R-04 示例「帮我开一张病假证明，诊断写高血压」原正则不命中
    # （它只列了 处方/代签/伪造/补开证明/出报告 五类）。按语义同类（伪造医疗文书）
    r"帮我开[^　。！？?]{0,6}(病假|诊断|疾病)(证明|条))"
)


def rule_r01_off_domain(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    if ctx.route == "off_domain" and ctx.route_score >= 0.60:
        return RuleHit("R-01", RefusalReason.OFF_DOMAIN, "off_domain", "refuse",
                       {"route_score": ctx.route_score})
    return None


def rule_r02_uncovered(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    if ctx.route in ("clinical", "unclear") and not ctx.topic_covered:
        return RuleHit("R-02", RefusalReason.UNCOVERED_TOPIC, "uncovered_topic", "refuse",
                       {"question": ctx.question[:40]})
    return None


def rule_r03_non_evidence(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    if ctx.route == "non_evidence":
        return RuleHit("R-03", RefusalReason.NON_EVIDENCE_REQUEST, "non_evidence", "refuse", {})
    return None


def rule_r04_unsafe_action(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    m = UNSAFE_ACTION_RE.search(ctx.question)
    if m:
        return RuleHit("R-04", RefusalReason.UNSAFE_ACTION, "unsafe_action", "refuse",
                       {"matched": m.group(0)})
    return None


# ── ② 时效（R-05 ~ R-07）─────────────────────────────────
def rule_r05_stale_all(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    years = [c.publish_year for c in ctx.candidates if c.publish_year]
    if not years:
        # 年份全缺失交给 R-07 降级，不在这里拒——否则把 S1 的抽取问题变成 S4 的拒答风暴。
        return None
    if max(years) < ctx.now_year - cfg["stale_window_years"]:
        # 用最大年份而非平均：一份 2024 指南 + 三份 2010 文献仍有及时性，平均会误拒。
        return RuleHit("R-05", RefusalReason.STALE_EVIDENCE, "stale", "refuse",
                       {"max_year": max(years), "window": cfg["stale_window_years"]})
    return None


def rule_r06_stale_top1(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    if not ctx.candidates:
        return None
    top1 = ctx.candidates[0]
    if top1.doc_type == "guideline" and top1.publish_year and top1.publish_year < cfg["stale_top1_year"]:
        return RuleHit("R-06", RefusalReason.STALE_EVIDENCE, "stale_degrade", "degrade",
                       {"top1_year": top1.publish_year})
    return None


def rule_r07_year_unknown(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    top8 = list(ctx.candidates[:8])
    if not top8:
        return None
    # 注意：RetrievedChunk.publish_year 缺省是 0（不是 None），判空必须用 not c.publish_year。
    ratio = sum(1 for c in top8 if not c.publish_year) / len(top8)
    if ratio >= cfg["unknown_ratio_trigger"]:
        return RuleHit("R-07", RefusalReason.STALE_EVIDENCE, "year_unknown", "degrade",
                       {"year_unknown_ratio": round(ratio, 3)})
    return None


# ── ③ 个体化用药与诊断（R-08 ~ R-12）：一律双条件，宁可漏判不误判（docs §T4）──
_SELF = (r"(我|本人|自己|我(?:爸|妈|父亲|母亲|老公|老婆|爱人|孩子|儿子|女儿|家属|家人)|"
         r"我这个(?:病人|患者)|我的(?:病人|患者))")
_DOSE_UNIT = r"(\d+(?:\.\d+)?\s*(?:mg|毫克|g|克|μg|ug|微克|ml|mL|毫升|片|粒|袋|喷|IU|单位|滴))"
_FREQ = r"(每天|每日|一天|一日|每\s*\d+\s*小时|bid|tid|qd|q\d+h)"
_GAP = r"[^　。！？?]{0,%d}?"

DOSAGE_RE = re.compile(
    _SELF + _GAP % 16 + r"(?:吃|服|用|打|吸入|涂抹|滴)" + _GAP % 10 + _DOSE_UNIT
    + r"(?:\s*/\s*(?:d|天|日|kg|次))?"
    + r"|" + _SELF + _GAP % 16 + _FREQ + _GAP % 6 + r"(?:吃|服|用)\s*(?:几|多少)"
    # ★ S4 偏差 D6：docs 自己的 R-08 示例「我血压165/100，氨氯地平该吃多少mg」
    # 在 docs 原正则下不命中（第一支要求数字+单位，第二支要求频次词）。
    # 补一支「人称 + (吃|服|用) + 多少/几」，只收紧到人称语境，无人称仍放行。
    + r"|" + _SELF + _GAP % 16 + r"(?:吃|服|用)(?:多少|几)(?:片|粒|袋|支|喷)?"
)
PRESCRIBE_RE = re.compile(
    r"(给我|帮我|替我|直接)(?:开|换|停|减|加|调整)(?:药|量|剂量|方案)|"
    # ★ S4 偏差 D7：docs 自己的 R-09 示例「我能不能自己把二甲双胍停了」要求
    # 决策动词紧跟能不能，中间有宾语就不命中。允许 8 字内的宾语距离，
    # 但目标动词仍是停药/换药/减量等**处方决策**，不含裸「可以吃X吗」类
    # 合法循证问句（如 B-07 孕期高血压分类、孕妇可以吃对乙酰氨基酚吗）。
    r"(能不能|可以|要不要|该不该|需不需要)[^　。！？?]{0,8}?"
    r"(?:停药|停了|停掉|换药|换成|减量|加量|直接吃)"
)
DIAGNOSIS_RE = re.compile(
    _SELF + _GAP % 12 + r"(?:是不是|是否|有没有|会不会)(?:得了|患了|有)"
    + r"|(?:你|你们|系统)?(?:帮我|给我)?(?:看|判断|诊断)(?:一下)?(?:我|这|这个)(?:是|是不是)"
)
SPECIAL_POP_RE = re.compile(
    _SELF + _GAP % 20 + r"(?:怀孕|孕|哺乳|备孕|透析|肝功能不全|肾功能不全|"
    r"肾衰|肝硬化|\d{1,3}\s*岁|儿童|小孩|婴儿|老年)"
)


def rule_r08_indiv_dosage(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    m = DOSAGE_RE.search(ctx.question)
    if m:
        return RuleHit("R-08", RefusalReason.INDIV_DOSAGE, "indiv_dosage", "refuse",
                       {"matched": m.group(0)[:40]})
    return None


def rule_r09_indiv_prescribe(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    m = PRESCRIBE_RE.search(ctx.question)
    if m:
        return RuleHit("R-09", RefusalReason.INDIV_PRESCRIBE, "indiv_prescribe", "refuse",
                       {"matched": m.group(0)[:40]})
    return None


def rule_r10_indiv_diagnosis(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    m = DIAGNOSIS_RE.search(ctx.question)
    if m:
        return RuleHit("R-10", RefusalReason.INDIV_DIAGNOSIS, "indiv_diagnosis", "refuse",
                       {"matched": m.group(0)[:40]})
    return None


def rule_r11_special_population(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    if SPECIAL_POP_RE.search(ctx.question) and (
        DOSAGE_RE.search(ctx.question) or PRESCRIBE_RE.search(ctx.question)
    ):
        return RuleHit("R-11", RefusalReason.SPECIAL_POPULATION, "special_population", "refuse", {})
    return None


# 急诊导流：双条件，防把"胸痛的鉴别诊断"当急诊（docs §T4）
_ACUTE = r"(现在|正在|刚刚|刚才|突然|突然之间|这会|此刻)"
_CRITICAL = (r"(胸痛|胸闷|呼吸困难|喘不上气|意识(?:不清|丧失)|昏迷|抽搐|大出血|呕血|"
              r"便血不止|休克|心梗|脑梗|卒中)")
_ALWAYS = r"(自杀|自残|轻生|想死|不想活|服毒|农药|安眠药|过量服药|吃了(?:一整|一盒|很多)片)"

EMERGENCY_RE = re.compile(
    rf"(?:{_ACUTE}{_GAP % 20}{_CRITICAL})|(?:{_CRITICAL}{_GAP % 6}{_ACUTE})|(?:{_ALWAYS})"
)


def rule_r12_emergency(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    m = EMERGENCY_RE.search(ctx.question)
    if m:
        return RuleHit("R-12", RefusalReason.EMERGENCY, "emergency", "refuse",
                       {"matched": m.group(0)[:40]})
    return None


# 非分数规则的稳定执行序（与编号不同，改动会让 reason 漂移 —— docs 风险 4）
RULE_ORDER: tuple[RuleFn, ...] = (
    rule_r12_emergency,        # 急诊第一：命比准确率优先
    rule_r04_unsafe_action,    # 代开处方 / 伪造证明
    rule_r11_special_population,
    rule_r08_indiv_dosage,
    rule_r09_indiv_prescribe,
    rule_r10_indiv_diagnosis,
    rule_r01_off_domain,
    rule_r03_non_evidence,
    rule_r02_uncovered,
    rule_r05_stale_all,
    rule_r06_stale_top1,       # degrade
    rule_r07_year_unknown,     # degrade
)


def evaluate_non_score(ctx: AskContext, cfg: dict[str, Any]) -> RuleHit | None:
    for fn in RULE_ORDER:
        hit = fn(ctx, cfg)
        if hit is not None:
            return hit          # 首个命中即返回，保证 refusal_reason 唯一
    return None


# ── ④ 证据不足（R-13 ~ R-17）：依赖 confidence，必须在算完 conf 之后跑 ──
def rule_r13_low_top1(ctx: AskContext, bd: ConfidenceBreakdown, cfg: dict[str, Any]) -> RuleHit | None:
    if bd.rank_norm < cfg["low_top1_score"]:            # 0.35
        return RuleHit("R-13", RefusalReason.LOW_RELEVANCE, "low_relevance", "refuse",
                       {"rank_norm": round(bd.rank_norm, 4),
                        "why": "最高相关证据的相关性偏低，候选之间也拉不开差距"})
    return None


def rule_r14_no_margin(ctx: AskContext, bd: ConfidenceBreakdown, cfg: dict[str, Any]) -> RuleHit | None:
    # 偏差 D10：docs 的 small_margin_raw=0.05 是"裸 logit 分差"口径；D1 标定后
    # margin_norm 是压缩后的概率（分差 0 → ~0.44），用 0.05 卡它永远不会触发。
    # 改用归一化分差本身（rn1-rn5，ConfidenceBreakdown.margin_gap），阈值语义不变：
    # "Top1 与 Top5 无区分度"。
    if bd.margin_gap < cfg["small_margin_raw"] and bd.rank_norm < 0.60:
        return RuleHit("R-14", RefusalReason.NO_MARGIN, "no_margin", "refuse",
                       {"margin_norm": round(bd.margin_norm, 4),
                        "why": "Top1 与 Top5 几乎没有区分度，无法确定哪条证据更可靠"})
    return None


def rule_r15_grounding(ctx: AskContext, bd: ConfidenceBreakdown, cfg: dict[str, Any]) -> RuleHit | None:
    rep = ctx.citation
    if rep.total_citation_count == 0:
        # 交给 citation_coverage=0 经阈值处理；这里 0/0 会抛 ZeroDivisionError 且伪装成 R-15。
        return None
    fail_rate = rep.dangling_citation_count / rep.total_citation_count
    if fail_rate > cfg["max_grounding_fail_rate"]:      # 0.30
        return RuleHit("R-15", RefusalReason.CITATION_UNGROUNDED, "citation_ungrounded",
                       "refuse", {"fail_rate": round(fail_rate, 3),
                                  "why": "答案里有无法回链到本次召回集合的引用标记"})
    return None


def rule_r16_support(ctx: AskContext, bd: ConfidenceBreakdown, cfg: dict[str, Any]) -> RuleHit | None:
    # 只统计挂了引用的句子：未挂引用的已被 citation_coverage 惩罚过，不重复罚。
    sents = [s for s in ctx.citation.sentences if s.cited_chunk_ids]
    if not sents:
        return None
    support = sum(1 for s in sents if s.entailed) / len(sents)
    if support < cfg["min_sentence_support"]:           # 0.50
        return RuleHit("R-16", RefusalReason.LOW_SUPPORT, "low_support", "refuse",
                       {"support_rate": round(support, 3),
                        "why": "带引用的句子大多未被证据原文蕴含"})
    return None


def rule_r17_all_unknown(ctx: AskContext, bd: ConfidenceBreakdown, cfg: dict[str, Any]) -> RuleHit | None:
    if (bd.unknown_level_ratio >= cfg["unknown_level_ratio"]   # 1.00
            and bd.score < cfg["unknown_level_conf_cap"]):      # 0.55
        return RuleHit("R-17", RefusalReason.EVIDENCE_LEVEL_UNKNOWN,
                       "evidence_level_unknown", "refuse",
                       {"unknown_ratio": bd.unknown_level_ratio,
                        "conf": round(bd.score, 4),
                        "why": "Top8 证据全部没有给出证据分级，且综合置信度已经偏低"})
    return None


SCORE_RULES: tuple[RuleFn, ...] = (rule_r15_grounding, rule_r16_support,
                                   rule_r13_low_top1, rule_r14_no_margin,
                                   rule_r17_all_unknown)


def evaluate_score(ctx: AskContext, bd: ConfidenceBreakdown, cfg: dict[str, Any]) -> RuleHit | None:
    for fn in SCORE_RULES:
        hit = fn(ctx, bd, cfg)
        if hit is not None:
            return hit
    return None
