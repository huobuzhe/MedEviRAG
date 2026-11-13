"""生成 tests/fixtures/chunks_small.jsonl（300 条真实 chunk 的子集）。

用途：A3/A8 的 Lite 集成测试与过滤单测要用真实结构的样本，
但不该把 4023 条全量 chunks.jsonl 拷进 tests/。
选取策略：从 chunks.jsonl 按 chunk_id 升序取 300 条，且尽量覆盖
四种 doc_type 与两种 lang（保证过滤规则有可测的反例）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "chunks.jsonl"
DST = ROOT / "tests" / "fixtures" / "chunks_small.jsonl"
LIMIT = 300


def main() -> int:
    rows: list[dict[str, object]] = []
    with SRC.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: str(r["chunk_id"]))

    picked: dict[str, dict[str, object]] = {}
    # ① 稀有样本优先：非中文块 + 有明确证据等级的块（过滤规则要能测到真实反例）
    for row in rows:
        cid = str(row["chunk_id"])
        rare = str(row.get("lang")) != "zh" or str(row.get("evidence_level")) != "unknown"
        if rare and len([k for k in picked if str(picked[k].get("lang")) != "zh"
                         or str(picked[k].get("evidence_level")) != "unknown"]) < 40:
            picked[cid] = row
    # ② 按 doc_type 各取一批，保证覆盖所有 doc_type
    per_type: dict[str, int] = {}
    for row in rows:
        dtype = str(row.get("doc_type"))
        if per_type.get(dtype, 0) < LIMIT // 8:
            picked.setdefault(str(row["chunk_id"]), row)
            per_type[dtype] = per_type.get(dtype, 0) + 1
    # ③ 补足到 LIMIT
    for row in rows:
        if len(picked) >= LIMIT:
            break
        picked.setdefault(str(row["chunk_id"]), row)

    out = [picked[k] for k in sorted(picked)]
    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    types: dict[str, int] = {}
    langs: dict[str, int] = {}
    years: dict[int, int] = {}
    levels: dict[str, int] = {}
    for row in out:
        types[str(row.get("doc_type"))] = types.get(str(row.get("doc_type")), 0) + 1
        langs[str(row.get("lang"))] = langs.get(str(row.get("lang")), 0) + 1
        year = int(row.get("publish_year") or 0)
        years[year] = years.get(year, 0) + 1
        levels[str(row.get("evidence_level"))] = levels.get(str(row.get("evidence_level")), 0) + 1
    print(f"wrote {DST} rows={len(out)}")
    print(f"  doc_type={types}")
    print(f"  lang={langs}")
    print(f"  evidence_level={levels}")
    print(f"  publish_year buckets={sorted(years.items())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
