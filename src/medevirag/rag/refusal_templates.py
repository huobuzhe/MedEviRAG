"""T6 拒答模板与降级提示（docs/05 §T6）。

纪律（docs 原话）：模板里不得出现具体数字断言（如"每天最多 4 g"），否则文案本身
就成了个体化建议，与 R-08 自相矛盾；拒答文案控制在 200 字内（急诊模板例外，要保命）；
render 用 raise ValueError 不用 assert（python -O 会关掉断言）。
"""

from __future__ import annotations

DISCLAIMER = ("以上内容仅供技术验证与学习参考，不构成诊疗建议；"
              "任何用药与治疗决定请咨询执业医师。")


def _render_off_domain(question: str, detail: dict[str, str]) -> str:
    return (
        "这个问题超出了我能回答的范围。我只依据已收录的临床指南、共识与部分开放获取文献作答，"
        "覆盖心血管、内分泌代谢、肾脏、呼吸、传染病与公共卫生等主题；你问的内容不在其中，"
        "我不做推测性回答。\n"
        "如果你要问的是某个具体疾病的诊断标准、治疗原则或药物适应证，可以换成那个问法再问一次。\n"
        + DISCLAIMER
    )


def _render_uncovered_topic(question: str, detail: dict[str, str]) -> str:
    return (
        "我查了本地语料，没有找到能支撑这个问题的指南或文献原文。语料没覆盖不等于临床上没有答案，"
        "只说明我这套系统没有可引用的证据，因此我不给结论。\n"
        "建议：在中华医学会各专科分会指南库、或 UpToDate / DynaMed 检索该主题；"
        "若是罕见病，可考虑到罕见病诊疗协作网医院就诊。\n"
        + DISCLAIMER
    )


def _render_stale(question: str, detail: dict[str, str]) -> str:
    year = detail.get("max_year", "未知")
    return (
        f"我检索到的证据最新只到 {year} 年，已经超出我设定的 10 年时效窗口。"
        "医学推荐变化很快，用一份过期证据回答当前问题，我不愿意这么做。\n"
        "建议：优先查该领域最新版指南（多数专科指南 3-5 年更新一版）；"
        "如果你需要，我可以只列出这些旧证据的原文与出处，时效性由你判断。\n"
        + DISCLAIMER
    )


def _render_indiv_dosage(question: str, detail: dict[str, str]) -> str:
    return (
        # docs §T6：拒答文案 ≤200 字；不得出现具体剂量数字，否则文案本身就成了个体化建议。
        "剂量取决于年龄、体重、肝肾功能、合并用药与实测值，这些我看不到也无法核实，"
        "所以我不能给可直接照做的数字。\n"
        "建议：带全部用药清单到专科门诊，由医生结合化验单调整；"
        "服药期间出现头晕、乏力、水肿等不适请尽快复诊，不要自行加减量。\n"
        "我能做的是：给出该药在指南里的常规剂量范围与需减量情形（不作个体化结论）。\n"
        + DISCLAIMER
    )


def _render_indiv_prescription(question: str, detail: dict[str, str]) -> str:
    return (
        "停药还是继续、换药还是加量，这个决定要由看过你完整病历的医生来做。"
        "我只有指南文本，看不到你的化验单与既往反应，替你定方向的风险远大于收益。\n"
        "建议：记下「药名 + 剂量 + 服用时长 + 最近化验值 + 症状」尽快复诊；"
        "若出现皮疹、呼吸困难、黑便等急性反应，立即停药并就医，不要等门诊。\n"
        + DISCLAIMER
    )


def _render_indiv_diagnosis(question: str, detail: dict[str, str]) -> str:
    return (
        "诊断要结合体格检查、化验与影像。我没有你的检查结果，不能也不该给出"
        "「你是不是得了某个病」的结论——猜错会让你漏掉真正的问题。\n"
        "建议：记下症状的出现时间、诱因、持续时长与伴随表现，到专科门诊面诊；"
        "若症状加重，优先去急诊而不是等门诊号。\n"
        "我能做的是：给出该病在指南里的诊断标准条目，你自行对照、由医生确认。\n"
        + DISCLAIMER
    )


def _render_emergency(question: str, detail: dict[str, str]) -> str:
    return (
        "你描述的情况可能是急症，先不要在这里等答案。\n"
        "请立即拨打 120，或让身边的人送你去最近医院的急诊科。"
        "如果意识不清、呼吸困难或大量出血，不要自行驾车。\n"
        "如果是药物过量或误服，带上药盒/药瓶一起去急诊，这对处理方案很关键。\n"
        "如果你此刻有伤害自己的想法，请联系当地心理援助热线或直接前往急诊，你不需要一个人扛。\n"
        + DISCLAIMER
    )


def _render_insufficient(question: str, detail: dict[str, str]) -> str:
    why = detail.get("why", "最高相关证据的相关性偏低，候选之间也拉不开差距")
    return (
        f"我检索到的证据不足以支撑一个可靠回答：{why}。"
        "与其给你一个听起来像答案的东西，不如直接说没有。\n"
        "建议：把问题拆得更具体（加上疾病名、人群、干预措施），或换用该领域的标准术语再问一次。\n"
        + DISCLAIMER
    )


def _render_evidence_unknown(question: str, detail: dict[str, str]) -> str:
    return (
        "我找到的原文推荐意见没有标注证据等级（GRADE unknown），"
        "来源本身没给出分级，我无法判断这条推荐的证据强度。\n"
        "建议：把下面列出的原文推荐意见当作「指南这么说」来看，而不是「高质量证据证明」；"
        "涉及治疗选择时请与专科医生确认。\n"
        + DISCLAIMER
    )


TEMPLATES = {
    # ① 越界
    "off_domain": _render_off_domain,
    "uncovered_topic": _render_uncovered_topic,
    "non_evidence": _render_uncovered_topic,
    "unsafe_action": _render_off_domain,
    # ② 时效
    "stale": _render_stale,
    "stale_degrade": _render_stale,
    "year_unknown": _render_evidence_unknown,
    # ③ 个体化
    "indiv_dosage": _render_indiv_dosage,
    "indiv_prescribe": _render_indiv_prescription,
    "indiv_diagnosis": _render_indiv_diagnosis,
    "special_population": _render_indiv_dosage,
    "emergency": _render_emergency,
    # ④ 证据不足
    "low_relevance": _render_insufficient,
    "no_margin": _render_insufficient,
    "citation_ungrounded": _render_insufficient,
    "low_support": _render_insufficient,
    "evidence_level_unknown": _render_evidence_unknown,
}


def render(key: str, question: str, detail: dict[str, str]) -> str:
    """渲染拒答文案；漏免责声明直接抛错（A8 门禁的实现基础）。"""
    if key not in TEMPLATES:
        raise KeyError(f"未注册的模板键: {key!r}（可选: {sorted(TEMPLATES)}）")
    text = TEMPLATES[key](question, detail)
    if DISCLAIMER not in text:
        raise ValueError(f"模板 {key} 漏了免责声明")
    return text


# partial 降级提示（不是拒答，是"照答但打折"）
PARTIAL_NOTES = {
    "low_confidence": "证据有限：我找到的原文相关度不高，以下内容请作为线索而非结论。",
    "evidence_unknown": ("该指南未给出分级证据（GRADE unknown），以下要点来自原文推荐意见，"
                         "请结合临床判断。"),
    "stale_degrade": "注意时效：最相关的指南版本较早，可能已有更新版。",
    "year_unknown": "部分来源未标注发布年份，时效性无法核实。",
}
