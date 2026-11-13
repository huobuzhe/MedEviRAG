"""Build data/eval/retrieval_dev60.jsonl from real chunk evidence (S2 / T7).

Every gold_chunk_ids / near_miss_ids entry is validated against
data/processed/chunks.jsonl; any unknown id aborts with exit code 1.

Split rule (frozen by docs/03): qid suffix % 5 == 0 -> test (12 cases),
the remaining 48 cases form the dev split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
OUT = ROOT / "data" / "eval" / "retrieval_dev60.jsonl"

# (qid, case_type, difficulty, topic, question, gold, near_miss)
CASES: list[tuple[str, str, str, str, str, list[str], list[str]]] = [
    # ---------------- exact_term: question carries a literal surface term ----------------
    ("q001", "exact_term", "easy", "高血压/降压启动", "血压超过140/90 mmHg 的心血管病高危患者，是否应启动降压药物治疗？",
     ["0201_083620_622#h880#c22"], ["0201_083620_622#h880#c23"]),
    ("q002", "exact_term", "easy", "高血压/老年目标", "高龄老年高血压患者的血压目标可考虑为多少？",
     ["0201_083620_622#h880#c23"], ["0201_083620_622#h880#c22"]),
    ("q003", "exact_term", "easy", "血脂/他汀起始", "中等强度他汀类药物治疗能否作为降脂达标的起始治疗？",
     ["0201_083620_622#h1176#c32"], ["0201_083620_622#h1315#c35"]),
    ("q004", "exact_term", "medium", "血脂/高甘油三酯", "ASCVD 高危人群接受中等剂量他汀类药物治疗后如TG>2.3 mmol/L，应考虑给予什么治疗？",
     ["0201_083620_622#h1315#c35"], ["0201_083620_622#h1176#c32"]),
    ("q005", "exact_term", "medium", "血脂/糖尿病目标", "糖尿病合并ASCVD 患者的LDL-C 目标值是多少？",
     ["0201_084005_469#1.4#c57"], ["0201_084005_469#h572#c43"]),
    ("q006", "exact_term", "medium", "血脂/脑卒中目标", "动脉粥样硬化性缺血性脑卒中或TIA 患者的降脂治疗LDL-C 和非HDL-C 目标值是多少？",
     ["0201_084005_469#h1185#c59"], ["0201_084005_469#1.4#c57"]),
    ("q007", "exact_term", "easy", "高血压/基层降压目标", "国家基层高血压防治管理指南推荐的降压目标是多少？",
     ["1031_101832_436#h2#c17"], ["1031_101832_436#h2#c02"]),
    ("q008", "exact_term", "easy", "血脂/他汀剂量", "中等强度他汀类药物的日剂量分别是多少？",
     ["1031_101832_436#h2#c51"], ["1031_101832_436#h2#c49"]),
    ("q009", "exact_term", "medium", "高血压/阿司匹林", "糖尿病患者服用阿司匹林前，血压至少应稳定控制在多少以下？",
     ["1031_101832_436#h2#c55"], ["1031_101832_436#h2#c51"]),
    ("q010", "exact_term", "medium", "糖尿病/健康状况分层", "糖尿病患者按健康状况分层时的血压和血脂管理目标分别是什么？",
     ["1031_100011_575#8.3#c257"], ["1031_100011_575#(一)#c132"]),
    ("q011", "exact_term", "easy", "动态血压/疗效评估", "已经接受降压治疗的患者如何用24 h 动态血压监测评估降压疗效？",
     ["1031_101620_633#h5#c23"], ["1031_101620_633#h5#c03"]),
    ("q012", "exact_term", "easy", "动态血压/孕妇", "对于怀疑白大衣性高血压、隐蔽性高血压的孕妇推荐做什么检查？",
     ["1031_101620_633#h5#c29"], ["1031_101620_633#h5#c24"]),
    ("q013", "exact_term", "medium", "动态血压/成本效益", "动态血压监测每投入1 元能节省多少高血压及并发症医药费用？",
     ["1031_101620_633#h5#c32"], ["1031_101620_633#h5#c29"]),
    ("q014", "exact_term", "easy", "糖尿病/糖代谢分类", "空腹血糖受损的静脉血浆葡萄糖界值是多少？",
     ["1031_095205_290#h36#c147"], ["1031_095205_290#h36#c146"]),
    ("q015", "exact_term", "easy", "慢阻肺/疫苗接种", "慢阻肺患者接种流感疫苗和肺炎链球菌疫苗的推荐强度分别是什么？",
     ["0727_162437_802#(三)#c42"], ["0727_162602_559#h631#c13"]),
    ("q016", "exact_term", "medium", "慢阻肺/支气管舒张剂", "B 组慢阻肺患者（多症状少急性加重）推荐使用什么支气管舒张剂方案？",
     ["0727_162437_802#(六)#c38"], ["0727_162437_802#(一)#c48"]),
    ("q017", "exact_term", "easy", "慢阻肺/流感疫苗频次", "慢阻肺患者每年注射流感疫苗几次，什么季节注射？",
     ["0727_162437_802#(一)#c48"], ["0727_162437_802#(三)#c42"]),
    ("q018", "exact_term", "easy", "布病/传染源", "布鲁氏菌病的主要传染源和传播途径是什么？",
     ["0527_153210_594#(一)#c01"], ["0527_153210_594#(二)#c02"]),
    ("q019", "exact_term", "medium", "布病/抗菌治疗", "急性期布鲁氏菌病的推荐抗菌治疗方案（药物与疗程）是什么？",
     ["0527_153210_594#h24#c08"], ["0527_153210_594#(二)#c04"]),
    ("q020", "exact_term", "medium", "登革热/临床诊断", "登革热临床诊断病例对白细胞计数和血小板计数的要求是什么？",
     ["0527_153307_354#(1)#c02"], ["0527_153307_354#(一)#c03"]),
    # ---------------- semantic: paraphrase, the literal term is deliberately absent ----------------
    ("q021", "semantic", "easy", "近视/户外活动", "孩子多大年龄段是视力发育的关键期，家长怎样帮助孩子预防近视？",
     ["0108_135352_975#(一)#c03"], ["0108_135352_975#(三)#c08"]),
    ("q022", "semantic", "easy", "近视/视力监测", "学校应当多久为学生开展一次视力监测，发现异常如何处理？",
     ["0108_135352_975#(一)#c07"], ["0108_135352_975#(五)#c10"]),
    ("q023", "semantic", "medium", "血压/儿童青少年", "7 岁到18 岁儿童青少年什么样的读数算正常高值血压？",
     ["0108_140741_181#4.3.2#c03"], ["0108_140741_181#h16#c02"]),
    ("q024", "semantic", "easy", "肥胖/儿童判定", "学龄儿童怎样用体质指数判断属于超重还是肥胖？",
     ["0811_150232_451#4.3.2#c05"], ["0108_140741_181#4.3.2#c03"]),
    ("q025", "semantic", "medium", "体重管理/能量", "做体重管理时每天的能量摄入应该比平时降低多少？",
     ["1031_101439_126#三、#c08"], ["1031_101719_234#一、#c15"]),
    ("q026", "semantic", "easy", "膳食/坚果与油", "每天吃多少坚果、用多少烹调油比较合适？",
     ["1031_101719_234#一、#c15"], ["1031_101126_324#三、#c23"]),
    ("q027", "semantic", "easy", "膳食/盐与糖", "高脂血症的人每天食盐和添加糖应该限制在多少？",
     ["1031_101126_324#三、#c23"], ["1031_101719_234#一、#c15"]),
    ("q028", "semantic", "medium", "糖尿病/清淡饮食", "糖尿病人饮食上要控制哪些调味品，为什么饮酒不利？",
     ["1031_095205_290#三、#c13"], ["1031_095205_290#一、#c144"]),
    ("q029", "semantic", "medium", "糖尿病/血糖检测", "测定静脉血浆葡萄糖时对送检时间和血糖仪有什么要求？",
     ["1031_095848_337#(二)#c09"], ["1031_095848_337#(三)#c12"]),
    ("q030", "semantic", "medium", "糖尿病/二甲双胍", "服用二甲双胍的患者使用碘化对比剂做造影前后要注意什么？",
     ["1031_095848_337#(三)#c16"], ["1031_095848_337#(二)#c09"]),
    ("q031", "semantic", "medium", "糖尿病/营养治疗", "营养治疗能让糖尿病患者的糖化血红蛋白下降多少？",
     ["1031_100011_575#第七章#c54"], ["1031_100011_575#(一)#c28"]),
    ("q032", "semantic", "medium", "糖尿病/血压干预", "糖尿病患者的血压超过什么水平就需要开始干预？",
     ["1031_100011_575#(一)#c132"], ["1031_100011_575#8.3#c257"]),
    ("q033", "semantic", "medium", "糖尿病/降压药选择", "给糖尿病人选择降压药物时要综合考虑哪些方面？",
     ["1031_100011_575#(一)#c134"], ["1031_100011_575#(一)#c132"]),
    ("q034", "semantic", "medium", "梅毒/血清学试验", "梅毒螺旋体血清学试验在不同病期的敏感性和特异性如何？",
     ["0123_130133_511#7.4#c33"], ["0123_130133_511#7.4#c14"]),
    ("q035", "semantic", "medium", "梅毒/临床表现", "二期梅毒的皮肤损害有什么特点，病期多长？",
     ["0123_130133_511#4.2.2#c05"], ["0123_130133_511#4.5.3.2#c08"]),
    ("q036", "semantic", "medium", "麻风/分类", "按WHO简化分类法，少菌型与多菌型麻风病如何界定？",
     ["0123_125349_591#h30#c28"], ["0123_125349_591#h30#c22"]),
    ("q037", "semantic", "hard", "疱疹/血清学判断", "怎样用型特异性血清抗体区分初发性生殖器疱疹是原发还是非原发感染？",
     ["0123_130218_622#7.2#c21"], ["0123_130218_622#h5#c01"]),
    ("q038", "semantic", "easy", "血吸虫/曼氏分布", "曼氏血吸虫主要分布在哪些国家和地区，中间宿主是什么？",
     ["0621_143946_131#h3#c02"], ["0621_143946_131#h3#c01"]),
    ("q039", "semantic", "medium", "血吸虫/病畜治疗", "治疗血吸虫病病畜的首选药物是什么，不同家畜的剂量如何？",
     ["0621_144019_327#5.2.3#c21"], ["0621_144019_327#3.5#c89"]),
    ("q040", "semantic", "medium", "疟疾/间日疟治疗", "间日疟和卵形疟应该怎么治疗，治愈后还要做什么？",
     ["0531_112855_497#(一)#c09"], ["0531_112855_497#(一)#c08"]),
    # ---------------- multi_hop: the answer needs two distinct evidence chunks ----------------
    ("q041", "multi_hop", "medium", "高血压/分层目标",
     "一般高血压患者和高龄老年高血压患者的血压目标分别是多少？",
     ["1031_101832_436#h2#c17", "0201_083620_622#h880#c23"],
     ["1031_101832_436#h2#c02"]),
    ("q042", "multi_hop", "hard", "高血压/合并糖尿病",
     "高血压合并糖尿病时，血压目标和LDL-C 目标值分别是多少？",
     ["1031_101832_436#h2#c17", "1031_101832_436#h2#c53"],
     ["1031_101832_436#h2#c49"]),
    ("q043", "multi_hop", "medium", "慢阻肺/疫苗证据",
     "慢阻肺患者需要接种哪些疫苗，各自的推荐强度和证据来源如何？",
     ["0727_162437_802#(三)#c42", "0727_162602_559#h631#c13"],
     ["0727_162437_802#(一)#c48"]),
    ("q044", "multi_hop", "hard", "血脂/他汀安全",
     "他汀类药物有哪些不良反应和禁忌证，中等强度他汀的日剂量是多少？",
     ["0201_084005_469#h731#c49", "1031_101832_436#h2#c51"],
     ["0201_083620_622#h1176#c32"]),
    ("q045", "multi_hop", "hard", "血脂/目标对照",
     "糖尿病合并ASCVD 患者与单纯动脉粥样硬化性缺血性脑卒中患者的LDL-C 目标值分别是多少？",
     ["0201_084005_469#1.4#c57", "0201_084005_469#h1185#c59"],
     ["0201_084005_469#h572#c43"]),
    ("q046", "multi_hop", "medium", "布病/传播与治疗",
     "布鲁氏菌病是怎样传播的，急性期如何治疗？",
     ["0527_153210_594#(一)#c01", "0527_153210_594#h24#c08"],
     ["0527_153210_594#(二)#c02"]),
    ("q047", "multi_hop", "medium", "登革热/诊断与报告",
     "登革热临床诊断病例的条件和诊断后的报告时限分别是多少？",
     ["0527_153307_354#(1)#c02", "0527_153307_354#(一)#c03"],
     ["0527_153307_354#(一)#c06"]),
    ("q048", "multi_hop", "medium", "动态血压/价值与成本",
     "动态血压监测有什么临床价值，成本效益如何？",
     ["1031_101620_633#h5#c03", "1031_101620_633#h5#c32"],
     ["1031_101620_633#h5#c23"]),
    ("q049", "multi_hop", "hard", "糖尿病/检测与分类",
     "糖尿病常用的血糖监测指标有哪些，糖代谢状态如何分类？",
     ["1031_095848_337#(二)#c09", "1031_095205_290#h36#c147"],
     ["1031_095205_290#h36#c146"]),
    ("q050", "multi_hop", "medium", "血吸虫/治疗与随访",
     "血吸虫病病畜治疗用什么药，病人规范治疗后多久随访复查？",
     ["0621_144019_327#5.2.3#c21", "0621_143946_131#(一)#c08"],
     ["0621_144019_327#3.5#c89"]),
    ("q051", "multi_hop", "medium", "近视/家庭与学校",
     "家庭和学校在儿童青少年近视防控中各自应做什么？",
     ["0108_135352_975#(一)#c03", "0108_135352_975#(一)#c07"],
     ["0108_135352_975#(三)#c08"]),
    ("q052", "multi_hop", "hard", "血压/儿童界值",
     "儿童青少年血压的正常值和正常高值界值分别是如何判定的？",
     ["0108_140741_181#4.3.2#c03", "0108_140741_181#h16#c02"],
     ["0108_140741_181#5.2#c04"]),
    ("q053", "multi_hop", "medium", "膳食/盐糖油限量",
     "高脂血症人群每日食盐、添加糖的限量以及每日烹调油的推荐量分别是多少？",
     ["1031_101126_324#三、#c23", "1031_101719_234#一、#c15"],
     ["1031_101126_324#三、#c25"]),
    ("q054", "multi_hop", "medium", "体重管理/能量与BMI",
     "体重管理每日能量摄入如何控制，超重和肥胖如何用BMI 判定？",
     ["1031_101439_126#三、#c08", "0811_150232_451#4.3.2#c05"],
     ["1031_101719_234#一、#c15"]),
    ("q055", "multi_hop", "hard", "疟疾/治疗与突发疫情",
     "间日疟的治疗方案和疟疾突发疫情的判定标准分别是什么？",
     ["0531_112855_497#(一)#c09", "0531_112855_497#(一)#c14"],
     ["0531_112855_497#(一)#c08"]),
    ("q056", "multi_hop", "medium", "高血压/负担与获益",
     "我国高血压患病人数有多少，收缩压和舒张压降低能带来哪些获益？",
     ["1031_101832_436#h2#c02", "1031_101832_436#h2#c17"],
     ["1031_101832_436#h2#c03"]),
    ("q057", "multi_hop", "hard", "血脂/获益与分层",
     "LDL-C 每降低1 mmol/L 能带来什么获益，高血压患者的LDL-C 目标值如何分层？",
     ["1031_101832_436#h2#c49", "1031_101832_436#h2#c53"],
     ["1031_101832_436#h2#c51"]),
    ("q058", "multi_hop", "hard", "梅毒/血清学与胎传",
     "梅毒螺旋体血清学试验的临床意义是什么，胎传梅毒如何定义？",
     ["0123_130133_511#7.4#c33", "0123_130133_511#4.5.3.2#c08"],
     ["0123_130133_511#7.4#c14"]),
    ("q059", "multi_hop", "hard", "慢阻肺/分级诊疗",
     "慢阻肺分级诊疗流程中各级医疗机构的职责是什么，B 组患者如何用药？",
     ["0727_162750_927#h234#c07", "0727_162437_802#(六)#c38"],
     ["0727_162437_802#(三)#c42"]),
    ("q060", "multi_hop", "medium", "血吸虫/监测复核",
     "血吸虫病病例报告后如何复核与订正，监测工作对检测试剂有什么要求？",
     ["0531_112531_215#(一)#c03", "0531_112531_215#五、#c13"],
     ["0531_112531_215#(一)#c02"]),
]


def load_chunk_map() -> dict[str, dict[str, Any]]:
    if not CHUNKS.exists():
        raise SystemExit(f"chunks.jsonl not found: {CHUNKS}")
    out: dict[str, dict[str, Any]] = {}
    with CHUNKS.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[str(row["chunk_id"])] = row
    return out


def build(verbose: int) -> int:
    chunk_map = load_chunk_map()
    missing: list[str] = []
    for qid, _ct, _d, _t, _q, gold, near in CASES:
        for cid in list(gold) + list(near):
            if cid not in chunk_map:
                missing.append(f"{qid}:{cid}")
    if missing:
        print(f"FAIL gold/near_miss not found in chunks.jsonl: {len(missing)}")
        for item in missing:
            print("  missing", item)
        return 1
    if len(CASES) != 60:
        print(f"FAIL expected 60 cases, got {len(CASES)}")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for qid, ct, diff, topic, question, gold, near in sorted(CASES):
        rows.append({
            "qid": qid,
            "question": question,
            "case_type": ct,
            "gold_chunk_ids": list(gold),
            "near_miss_ids": list(near),
            "topic": topic,
            "difficulty": diff,
            "split": "test" if int(qid[1:]) % 5 == 0 else "dev",
        })
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    per_type: dict[str, int] = {}
    for r in rows:
        per_type[r["case_type"]] = per_type.get(r["case_type"], 0) + 1
    print(f"wrote {OUT.relative_to(ROOT)} rows={len(rows)}")
    print(f"case_type counts: {dict(sorted(per_type.items()))}")
    print(f"split: dev={len(dev)} test={len(test)}")
    print(f"gold id validation: checked {len(rows)} cases, missing=0")
    print(f"gold total={sum(len(r['gold_chunk_ids']) for r in rows)} "
          f"near_miss total={sum(len(r['near_miss_ids']) for r in rows)}")
    for row in rows[:verbose]:
        texts = [" ".join(chunk_map[c]["text"].split())[:80] for c in row["gold_chunk_ids"]]
        print(f"  {row['qid']} [{row['case_type']}] {row['question']}")
        print(f"    gold={row['gold_chunk_ids']}")
        for cid, txt in zip(row["gold_chunk_ids"], texts, strict=True):
            print(f"      {cid} :: {txt}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build the S2 labelled eval set")
    ap.add_argument("--verbose", type=int, default=0, help="print this many cases with gold text")
    args = ap.parse_args(argv)
    return build(args.verbose)


if __name__ == "__main__":
    sys.exit(main())
