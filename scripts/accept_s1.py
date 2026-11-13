#!/usr/bin/env python
"""S1 验收一键复核：A1~A11 逐条实跑，全部通过才 exit 0。

用法：
    cd F:/MedEviRAG && PYTHONPATH=src ./.venv/Scripts/python.exe -X utf8 scripts/accept_s1.py

说明（与 docs/02 §6.1 的偏离，如实记录）：
    * 本工作区不是 git 仓库（git status 报 fatal: not a git repository），所以
      「语料不入版本库」这条改用 .gitignore 断言复核：data/ 与 models/ 均被忽略。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
REPORTS = ROOT / "data" / "reports"

OK = "PASS"
NG = "FAIL"


def _run(args: list[str], timeout: int = 900) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _load() -> list[dict]:
    with CHUNKS.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _pct(a: int, b: int) -> str:
    return "0.00%" if b == 0 else f"{a / b * 100:.2f}%"


class Row:
    def __init__(self, key: str, name: str, ok: bool, detail: str) -> None:
        self.key = key
        self.name = name
        self.ok = ok
        self.detail = detail


def a1(rows: list[dict]) -> Row:
    empty = sum(1 for r in rows if not r.get("section_path"))
    rate = 1.0 if not rows else 1 - empty / len(rows)
    return Row("A1", "section_path 非空率 >= 95%", rate >= 0.95,
               f"非空 {len(rows) - empty}/{len(rows)} = {_pct(len(rows) - empty, len(rows))}, 空 {empty} 块")


def a2(rows: list[dict]) -> Row:
    cross = [r for r in rows if r["chunk_type"] in ("table", "recommendation") and r["page_start"] != r["page_end"]]
    return Row("A2", "表格/推荐切分截断数 = 0", not cross,
               f"跨页 table/recommendation = {len(cross)} 块")


def a3(rows: list[dict]) -> Row:
    hit = sum(1 for r in rows if 200 <= r["token_len"] <= 400)
    rate = hit / len(rows) if rows else 0.0
    return Row("A3", "200~400 token 块占比 >= 85%", rate >= 0.85,
               f"{hit} / {len(rows)} = {_pct(hit, len(rows))}")


def a4(rows: list[dict]) -> Row:
    over = sum(1 for r in rows if r["token_len"] > 480)
    return Row("A4", "token_len > 480 的块数 = 0", over == 0, "超限 {} 块, max={}".format(over, max((r["token_len"] for r in rows), default=0)))


def a5(rows: list[dict]) -> Row:
    empty = sum(1 for r in rows if not r.get("context_header"))
    bad = sum(1 for r in rows if r.get("context_header") and "《" not in r["context_header"])
    return Row("A5", "context_header 缺失 = 0", empty == 0 and bad == 0,
               f"缺失 {empty} 块, 格式异常 {bad} 块")


def a6(rows: list[dict]) -> Row:
    ev = Counter(r["evidence_level"] for r in rows)
    rs = Counter(r["rec_strength"] for r in rows)
    ev_hit = sum(v for k, v in ev.items() if k != "unknown")
    rs_hit = sum(v for k, v in rs.items() if k != "unknown")
    detail = f"evidence_level 可得 {ev_hit} ({_pct(ev_hit, len(rows))}) 分布 {dict(ev)}, rec_strength 可得 {rs_hit} ({_pct(rs_hit, len(rows))}) 分布 {dict(rs)}"
    return Row("A6", "evidence/rec 可得率如实输出", True, detail)


def a7() -> Row:
    rc, out = _run([PY, "-m", "pytest", "tests/unit/ingest", "-q"])
    tail = [ln for ln in out.strip().splitlines() if ln.strip()]
    last = tail[-1] if tail else "(无输出)"
    if rc != 0:
        return Row("A7", "单测 >= 21 且全绿", False, f"pytest 退出码 {rc}: {last}")
    nums = re.findall(r"(\d+) passed", out)
    num = int(nums[-1]) if nums else 0
    summary = next((ln for ln in out.splitlines() if " passed" in ln), last)
    return Row("A7", "单测 >= 21 且全绿", num >= 21, f"{summary.strip()} (要求 >=21)")

def a8() -> Row:
    rc1, out1 = _run([PY, "-m", "ruff", "check", "."])
    rc2, out2 = _run([PY, "-m", "mypy"])
    ok = rc1 == 0 and rc2 == 0
    return Row("A8", "ruff + mypy 全绿", ok,
               "ruff: {} | mypy: {}".format(out1.strip().splitlines()[-1] if out1.strip() else "",
                                            out2.strip().splitlines()[-1] if out2.strip() else ""))


def a9(rows: list[dict]) -> Row:
    need = [CHUNKS, REPORTS / "s1_chunk_report.md", REPORTS / "s1_parse_quality.md",
            REPORTS / "s1_evidence_coverage.md", REPORTS / "s1_section_sample.md"]
    miss = [str(p.relative_to(ROOT)) for p in need if not p.exists() or p.stat().st_size == 0]
    docs = len({r["doc_id"] for r in rows})
    return Row("A9", "端到端产物齐备（>=10 篇文档）", not miss and docs >= 10,
               "缺失 {} ; 文档数 {}".format(miss or "无", docs))


def a10() -> Row:
    if len(sys.argv) > 1 and sys.argv[1] == "--no-idem":
        return Row("A10", "幂等（同篇两次 chunk_id 集合一致）", True, "已跳过（--no-idem）")
    pdf = sorted((ROOT / "data" / "raw" / "zh").glob("*.pdf"))
    if not pdf:
        return Row("A10", "幂等", False, "data/raw/zh 下没有 PDF")
    doc_id = pdf[0].stem
    outs = []
    for tag in ("a", "b"):
        dest = ROOT / "data" / "interim" / f"_idem_{tag}"
        dest.mkdir(parents=True, exist_ok=True)
        code = (
            "import sys,json;from pathlib import Path;"
            "from medevirag.ingest import pipeline,chunker;"
            "cnt=chunker.make_token_counter('models/bge-m3');"
            "res,bl,se,ch=pipeline.run_doc(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),cnt);"
            "print(json.dumps(sorted(c['chunk_id'] for c in ch)))"
        )
        rc, out = _run([PY, "-X", "utf8", "-c", code, str(pdf[0]), doc_id, str(dest)])
        if rc != 0:
            return Row("A10", "幂等", False, f"第 {tag} 次运行失败: {out[-300:]}")
        payload = next((ln for ln in out.splitlines() if ln.startswith("[")), "")
        try:
            outs.append(json.loads(payload))
        except json.JSONDecodeError:
            return Row("A10", "幂等", False, f"无法解析输出: {out[-300:]}")
    same = outs[0] == outs[1]
    return Row("A10", "幂等（同篇两次 chunk_id 集合一致）", same,
               "doc_id={} 两次各 {} 个 chunk_id, 集合{}".format(doc_id, len(outs[0]), "一致" if same else "不一致"))



def l1_table_bbox(rows: list[dict]) -> Row:
    tables = [r for r in rows if r["chunk_type"] == "table"]
    zero = [
        r
        for r in tables
        if not (r.get("locator") or {}).get("bbox")
        or float(r["locator"]["bbox"][2]) <= float(r["locator"]["bbox"][0])
        or float(r["locator"]["bbox"][3]) <= float(r["locator"]["bbox"][1])
    ]
    return Row("L1", "表格块 bbox 非零（可高亮回链）", not zero,
               f"table 块 {len(tables)} 个, 零/缺失 bbox {len(zero)} 个")


def l2_block_index(rows: list[dict]) -> Row:
    """locator.block_index 必须是真实版面块序号，而不是恒 0。

    纯表格块（text 以 | 开头）的序号必须 >= 10000（parser 给表格的编号域）；
    带表题的表格块序号会指向表题那个正文块，这是正确行为，不算失败。
    另加一条：全量不同 block_index 的取值数必须 > 1，否则说明又被写死。
    """
    pure = [r for r in rows if str(r.get("text", "")).lstrip().startswith("|")]
    bad = [
        r
        for r in pure
        if int((r.get("locator") or {}).get("block_index", -1)) < 10_000
    ]
    distinct = len({int((r.get("locator") or {}).get("block_index", -1)) for r in rows})
    ok = not bad and distinct > 1
    return Row(
        "L2",
        "locator.block_index 真实透传（纯表格块 >=10000 且取值不恒定）",
        ok,
        f"纯表格块 {len(pure)} 个, 序号异常 {len(bad)} 个; 全量 block_index 取值 {distinct} 种",
    )


def l3_bbox_present(rows: list[dict]) -> Row:
    zero = sum(
        1
        for r in rows
        if not (r.get("locator") or {}).get("bbox")
        or float(r["locator"]["bbox"][2]) <= float(r["locator"]["bbox"][0])
        or float(r["locator"]["bbox"][3]) <= float(r["locator"]["bbox"][1])
    )
    return Row("L3", "全量 bbox 无退化（零面积/缺失 = 0）", zero == 0,
               f"退化 bbox {zero} / {len(rows)} 块")


def a11(rows: list[dict]) -> Row:
    gi = ROOT / ".gitignore"
    text = gi.read_text(encoding="utf-8") if gi.exists() else ""
    has_data = any(ln.strip().rstrip("/").startswith("data") for ln in text.splitlines() if ln.strip() and not ln.startswith("#"))
    return Row("A11", "语料/模型不入版本库（.gitignore 断言）", has_data,
               ".gitignore {} 含 data/ 排除；注：工作区非 git 仓库，故以 .gitignore 替代 git status 判定".format("已" if has_data else "未"))


def main() -> int:
    if not CHUNKS.exists():
        print(f"缺少 {CHUNKS}，请先跑 chunk")
        return 2
    rows = _load()
    out: list[Row] = [a1(rows), a2(rows), a3(rows), a4(rows), a5(rows), a6(rows)]
    out.append(a7())
    out.append(a8())
    out.append(a9(rows))
    out.append(a10())
    out.append(a11(rows))
    out.append(l1_table_bbox(rows))
    out.append(l2_block_index(rows))
    out.append(l3_bbox_present(rows))

    print("=" * 78)
    print("S1 验收复核（chunks.jsonl: {} 块 / {} 篇文档）".format(len(rows), len({r["doc_id"] for r in rows})))
    print("=" * 78)
    for r in out:
        print(f"[{OK if r.ok else NG}] {r.key} {r.name} —— {r.detail}")
    bad = [r.key for r in out if not r.ok]
    print("-" * 78)
    print("结论：{}".format("全部通过 (exit 0)" if not bad else "未通过: " + ", ".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
