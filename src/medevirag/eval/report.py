# S5 T9：报告渲染（docs/06 §T9 骨架四段：摘要卡/分维表/消融/失败Top10）。

from __future__ import annotations

from typing import Any


def render_markdown(m: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    out: list[str] = []
    out.append("# 评测报告 " + str(m.get("run_id", "")))
    out.append("")
    out.append("## 摘要卡")
    out.append("| 项 | 值 |")
    out.append("|---|---|")
    out.append("| run_id | " + str(m.get("run_id")) + " |")
    out.append("| git_sha / config_hash | " + str(m.get("git_sha")) + " / " + str(m.get("config_hash")) + " |")
    out.append("| 集合指纹 / 划分 | " + str(m.get("frozen_set_sha256")) + " / " + str(m.get("split")) + " |")
    out.append("| 抽样种子 | " + str(m.get("seed")) + " |")
    out.append("| n（进分母） / n_total | " + str(m.get("n")) + " / " + str(m.get("n_total")) + " |")
    out.append("| LLM 模式 / judge | " + str(m.get("llm_mode", "n/a")) + " / " + str(m.get("judge_model", "n/a")) + " |")
    out.append("| tokens / api_calls | " + str(m.get("tokens_used", 0)) + " / " + str(m.get("api_calls", 0)) + " |")
    gates = [k for k in m if k.endswith("_pass")]
    n_pass = sum(1 for k in gates if m[k])
    n_fail = sum(1 for k in gates if m[k] is False)
    out.append("| 门禁 | " + str(n_pass) + " PASS / " + str(n_fail) + " FAIL（观测 ece=" + str(m.get("ece")) + "） |")
    out.append("")
    out.append("## 分维度指标表")
    out.append("| 指标 | 实测 | 95% CI | 判定 |")
    out.append("|---|---|---|---|")
    order = ["recall_at_10", "hit_at_10", "mrr", "ndcg_at_10",
             "citation_landing_rate", "sentence_support_rate", "faithfulness",
             "evidence_recall", "refusal_dim", "refusal_recall",
             "over_refusal_rate", "refusal_reason_accuracy", "partial_rate",
             "judge_error_rate", "ece"]
    for k in order:
        if k not in m:
            continue
        v = m[k]
        ci = m.get(k + "_ci95")
        g = m.get(k + "_pass")
        ci_s = "[" + str(ci[0]) + ", " + str(ci[1]) + "]" if ci else "n/a"
        if g is True:
            g_s = "PASS"
        elif g is False:
            g_s = "FAIL"
        else:
            g_s = "观测"
        vs = "n/a" if v is None else str(v)
        out.append("| " + k + " | " + vs + " | " + ci_s + " | " + g_s + " |")
    out.append("")
    out.append("## 消融表（4 组）")
    out.append("| 组 | 说明 | 状态 |")
    out.append("|---|---|---|")
    out.append("| rrf vs weighted | 融合函数对照 | 阶段末手动跑（见 s2_ablation） |")
    out.append("| rerank off | S3 R2 否决后默认即 off | 已生效 |")
    out.append("| filter off | 硬过滤价值验证 | 阶段末手动跑 |")
    out.append("| refusal rules off | S4 规则表价值验证 | 阶段末手动跑 |")
    out.append("")
    out.append("## 失败案例 Top10 与归因")
    fails = [r for r in rows if not r.get("correct") or r.get("judge_error")]
    fails.sort(key=lambda r: (r.get("judge", {}).get("faithfulness", 1.0),
                              r.get("judge", {}).get("refusal", 1.0)))
    out.append("| case_id | 类型 | faithfulness | refusal | 归因 |")
    out.append("|---|---|---|---|---|")
    for r in fails[:10]:
        jd = r.get("judge", {})
        why = r.get("judge_error_msg", "低分")[:40]
        out.append("| " + str(r["case_id"]) + " | " + str(r["case_type"]) + " | "
                   + str(jd.get("faithfulness", "n/a")) + " | "
                   + str(jd.get("refusal", "n/a")) + " | " + why + " |")
    if not fails:
        out.append("| （无失败样本） | | | | |")
    out.append("")
    out.append("> 反作弊自查：n / frozen_set_sha256 / git_sha / config_hash / seed 均已在摘要卡打印。")
    out.append("> llm_mode=fake 时三维 judge 分为离线桩口径，仅验证链路，不可作为门禁（须 --real）。")
    return "\n".join(out)


def render_delta_md(base_id: str, cur_id: str, diff: list[dict[str, Any]], problems: list[str]) -> str:
    out = ["# 回归 delta：" + base_id + " vs " + cur_id, ""]
    out.append("## 门禁")
    out.append("\n".join(problems) if problems else "（无触发）")
    out.append("")
    out.append("## 逐题方向变化")
    out.append("| case_id | 方向 | base F/R | cur F/R | 线索 |")
    out.append("|---|---|---|---|---|")
    for d in diff:
        b = d["base"]
        c = d["cur"]
        out.append("| " + str(d["case_id"]) + " | " + str(d["direction"]) + " | "
                   + "F" + str(b.get("faithfulness", "n/a")) + "/R" + str(b.get("refusal", "n/a")) + " | "
                   + "F" + str(c.get("faithfulness", "n/a")) + "/R" + str(c.get("refusal", "n/a")) + " | "
                   + str(d["hint"]) + " |")
    return "\n".join(out)
