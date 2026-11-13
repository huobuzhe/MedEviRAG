# P0 评测集分层与自检：从已冻结的 160 条全集派生三层，不改动原文件。
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medevirag.eval.build_set import load_cases

ROOT = Path(__file__).resolve().parents[3]
SET_PATH = ROOT / "data/eval/eval_set.jsonl"
FREEZE_PATH = ROOT / "data/eval/freeze.json"
LAYER_DIR = ROOT / "data/eval/layers"
META_PATH = LAYER_DIR / "layers_meta.json"
SALT = "p0-layers-v1"
LAYER_SIZES = {"smoke_20": 20, "core_80": 80, "full_160": 160}
REQUIRED_FIELDS = (
    "case_id", "question", "case_type", "gold_chunk_ids", "gold_answer",
    "must_refuse", "topic", "difficulty", "refuse_kind", "split",
    "origin", "annotators", "rubric_version",
)
VALID_TYPES = ("answerable", "multi_hop", "should_refuse")


def read_raw(path: Path = SET_PATH) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def validate_cases(raw: list[dict[str, Any]],
                   freeze_path: Path = FREEZE_PATH) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    for i, r in enumerate(raw, 1):
        missing = [k for k in REQUIRED_FIELDS if k not in r]
        if missing:
            problems.append("row " + str(i) + ": 缺字段 " + ",".join(missing))
            continue
        if not isinstance(r["case_id"], str) or not r["case_id"]:
            problems.append("row " + str(i) + ": case_id 必须是非空字符串")
        if not isinstance(r["question"], str) or not r["question"].strip():
            problems.append("row " + str(i) + ": question 为空")
        if r["case_type"] not in VALID_TYPES:
            problems.append("row " + str(i) + ": case_type 非法 " + str(r["case_type"]))
        if not isinstance(r["gold_chunk_ids"], list) or any(
                not isinstance(x, str) for x in r["gold_chunk_ids"]):
            problems.append("row " + str(i) + ": gold_chunk_ids 必须是字符串数组")
        if not isinstance(r["must_refuse"], bool):
            problems.append("row " + str(i) + ": must_refuse 必须是 bool，不能是字符串")
        should = r["case_type"] == "should_refuse"
        if should != r["must_refuse"]:
            problems.append("row " + str(i) + ": case_type 与 must_refuse 不一致")
        if not should and not r["gold_chunk_ids"]:
            problems.append("row " + str(i) + ": 可回答题必须有 gold_chunk_ids")
    ids = [str(r.get("case_id", "")) for r in raw]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        problems.append("case_id 重复: " + ",".join(sorted(dup)[:5]))
    stamp = json.loads(freeze_path.read_text(encoding="utf-8"))
    dist = dict(Counter(str(r.get("split")) for r in raw))
    expect = stamp.get("n", {})
    if any(dist.get(k, 0) != v for k, v in expect.items()):
        warnings.append(
            "SPLIT_DISTRIBUTION_MISMATCH: freeze.json 声称 " + str(expect)
            + "，实际 jsonl 是 " + str(dist)
            + "；P0 不信任原 split 字段，改由本模块按确定性算法重新分层")
    return problems, warnings


def _order_key(case_id: str, salt: str) -> str:
    return hashlib.sha256((salt + ":" + case_id).encode("utf-8")).hexdigest()


def _quota(groups: dict[str, list[Any]], n: int) -> dict[str, int]:
    total = sum(len(g) for g in groups.values())
    raw = {k: n * len(g) / max(1, total) for k, g in groups.items()}
    q = {k: int(v) for k, v in raw.items()}
    rest = n - sum(q.values())
    for k in sorted(raw, key=lambda k: (raw[k] - q[k], k), reverse=True)[:max(0, rest)]:
        q[k] += 1
    return q


def _pick(cases: list[Any], n: int, salt: str) -> list[Any]:
    groups: dict[str, list[Any]] = {}
    for c in cases:
        groups.setdefault(c.case_type, []).append(c)
    quota = _quota(groups, n)
    out: list[Any] = []
    for k in sorted(groups):
        g = sorted(groups[k], key=lambda c: _order_key(c.case_id, salt + ":" + k))
        out.extend(g[:quota[k]])
    return sorted(out, key=lambda c: c.case_id)


def build_layers() -> dict[str, Any]:
    cases = load_cases(SET_PATH)
    problems, warnings = validate_cases(read_raw(SET_PATH))
    if problems:
        raise SystemExit("评测集 schema 校验失败:\n  " + "\n  ".join(problems))
    full = sorted(cases, key=lambda c: c.case_id)
    core = _pick(full, LAYER_SIZES["core_80"], SALT + ":core")
    smoke = _pick(core, LAYER_SIZES["smoke_20"], SALT + ":smoke")
    layers = {"smoke_20": smoke, "core_80": core, "full_160": full}
    core_ids = {c.case_id for c in core}
    smoke_ids = {c.case_id for c in smoke}
    assert smoke_ids <= core_ids, "smoke_20 必须是 core_80 的子集"
    assert core_ids <= {c.case_id for c in full}, "core_80 必须是 full_160 的子集"
    LAYER_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "version": SALT,
        "source_set": "data/eval/eval_set.jsonl",
        "source_sha256": hashlib.sha256(SET_PATH.read_bytes()).hexdigest(),
        "freeze_set_sha256": json.loads(
            FREEZE_PATH.read_text(encoding="utf-8"))["set_sha256"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warnings": warnings,
        "layers": {},
    }
    for name, group in layers.items():
        p = LAYER_DIR / (name + ".jsonl")
        body = "".join(json.dumps(c.to_jsonl(), ensure_ascii=False) + "\n"
                       for c in group)
        p.write_text(body, encoding="utf-8")
        meta["layers"][name] = {
            "file": "data/eval/layers/" + name + ".jsonl",
            "n": len(group),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "by_type": dict(Counter(c.case_type for c in group)),
            "must_refuse": sum(1 for c in group if c.must_refuse),
        }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return meta


def verify_layers() -> dict[str, Any]:
    if not META_PATH.exists():
        raise SystemExit("未找到 layers_meta.json，请先执行 build")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    changed: list[str] = []
    if hashlib.sha256(SET_PATH.read_bytes()).hexdigest() != meta["source_sha256"]:
        changed.append("source eval_set.jsonl 已改动")
    for name, info in meta["layers"].items():
        p = LAYER_DIR / (name + ".jsonl")
        if not p.exists():
            changed.append(name + " 文件缺失")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != info["sha256"]:
            changed.append(name + " 内容被改动")
    return {"ok": not changed, "changed": changed, "meta": meta}


def main() -> int:
    ap = argparse.ArgumentParser(prog="layers")
    ap.add_argument("cmd", choices=["build", "verify", "stats"])
    a = ap.parse_args()
    if a.cmd == "build":
        meta = build_layers()
        for w in meta["warnings"]:
            print("[WARN]", w)
        for name, info in meta["layers"].items():
            print("[layers] " + name + ": n=" + str(info["n"])
                  + " by_type=" + str(info["by_type"])
                  + " must_refuse=" + str(info["must_refuse"]))
        print("[layers] meta -> " + str(META_PATH))
        return 0
    r = verify_layers()
    if a.cmd == "verify":
        for c in r["changed"]:
            print("[CHANGED]", c)
        print("[layers] verify " + ("OK" if r["ok"] else "FAIL"))
        return 0 if r["ok"] else 2
    for name, info in r["meta"]["layers"].items():
        print("[stats] " + name + ": n=" + str(info["n"])
              + " by_type=" + str(info["by_type"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
