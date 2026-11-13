"""A10：Lite vs 服务器版 parity（同一索引、同一批 query 的 Top10 比对）。

服务器版不可用时不算失败，但**必须留下 skip 的具体原因与错误原文**，
写进 data/reports/s2_parity.json；s2_ablation.py 读它决定报告里的 index.source。
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys

from medevirag.index.paths import EMB_PATH, REPORTS_DIR, ensure_dirs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARITY_PATH = os.path.join(str(REPORTS_DIR), "s2_parity.json")
LITE_URI = "./data/indexes/milvus_lite.db"
N_QUERIES = 10
MIN_OVERLAP = 9


def _docker_error() -> str:
    """取 docker daemon 不可用的原文，作为 skip 证据。"""
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=60, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"{type(exc).__name__}: {exc}"
    text = (proc.stderr or proc.stdout or "").strip()
    return " ".join(text.split())[:600]


def _write(report: dict[str, object], code: int) -> int:
    with open(PARITY_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"wrote {PARITY_PATH}")
    return code


def main() -> int:
    ensure_dirs()
    server_uri = os.environ.get("MILVUS_SERVER_URI") or None
    report: dict[str, object] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "skipped",
        "reason": "",
        "lite_uri": LITE_URI,
        "server_uri": server_uri,
        "n_queries": N_QUERIES,
        "min_overlap": MIN_OVERLAP,
        "ok_queries": 0,
        "detail": [],
    }

    # 本机 docker daemon 起不来，但允许连一个「预置」的服务器容器走 parity。
    # 先读 env，未设置时尝试连本机常见默认端口 19530（Milvus 默认），能连上就做 parity，
    # 连不上才走 skip 分支（docs/03 §6 允许）。
    if server_uri is None:
        from pymilvus import MilvusClient
        for candidate in ("http://127.0.0.1:19530",):
            try:
                probe = MilvusClient(uri=candidate)
                probe.get_load_state(collection_name="chunks")
                server_uri = candidate
                report["server_uri"] = candidate
                report["server_version"] = probe.get_server_version()
                print(f"A10 auto-detected pre-existing server: {candidate}")
                break
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        if server_uri is None:
            err = _docker_error()
            report["reason"] = (
                "no Milvus server reachable: MILVUS_SERVER_URI is unset, no pre-existing server on "
                "127.0.0.1:19530, and the docker daemon is down. "
                f"docker info stderr: {err}"
            )
            print("A10 SKIP:", report["reason"])
            return _write(report, 0)

    try:
        import numpy as np

        from medevirag.index.milvus_store import open_milvus_client, search_dense
    except ImportError as exc:
        report["reason"] = f"import failed: {type(exc).__name__}: {exc}"
        print("A10 SKIP:", report["reason"])
        return _write(report, 0)

    matrix = np.load(str(EMB_PATH), mmap_mode="r")
    lite = open_milvus_client(LITE_URI)
    server = open_milvus_client(server_uri)
    for client in (lite, server):
        client.load_collection("chunks")

    detail: list[dict[str, object]] = []
    ok = 0
    for i in range(N_QUERIES):
        vec = [float(x) for x in matrix[i * 7]]
        lite_ids = [h.chunk_id for h in search_dense(lite, vec, topk=10, collection="chunks", ef=64)]
        server_ids = [h.chunk_id for h in search_dense(server, vec, topk=10, collection="chunks", ef=64)]
        overlap = len(set(lite_ids) & set(server_ids))
        ok += 1 if overlap >= MIN_OVERLAP else 0
        detail.append({"i": i, "overlap": overlap, "lite_top3": lite_ids[:3], "server_top3": server_ids[:3]})
    report["detail"] = detail
    report["ok_queries"] = ok
    report["status"] = "ok" if ok >= N_QUERIES else "mismatch"
    report["reason"] = f"{ok}/{N_QUERIES} queries reached >={MIN_OVERLAP}/10 Top10 overlap"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return _write(report, 0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    sys.exit(main())
