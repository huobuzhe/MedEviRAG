"""doctor 自检（docs/01 §T9）。

设计要点：三态 OK/WARN/FAIL，**只有 FAIL 才非 0 退出**。
docker daemon 没起、代理不通，这些都是"降级可跑"而不是"环境坏了"——
把它们判 FAIL 会让 CI 因网络随机红灯，门禁就失去意义。
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from medevirag.common.config import get_settings

STATUS_OK, STATUS_WARN, STATUS_FAIL = "OK", "WARN", "FAIL"


@dataclass
class CheckRow:
    name: str
    status: str
    detail: str


def _run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.returncode, (proc.stdout or proc.stderr or "").strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, type(exc).__name__


def check_python() -> CheckRow:
    ver = platform.python_version()
    ok = sys.version_info[:2] == (3, 10)
    return CheckRow("python", STATUS_OK if ok else STATUS_WARN, f"{ver} {sys.executable}")


def check_gpu() -> CheckRow:
    try:
        import torch

        if not torch.cuda.is_available():
            return CheckRow("gpu", STATUS_WARN, f"torch {torch.__version__} cuda=False -> 用 CPU 跑 embedding")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        return CheckRow("gpu", STATUS_OK, f"{name}, {mem} MiB | torch {torch.__version__} cuda=True")
    except Exception as exc:  # 自检要吞掉一切异常并如实报告，不能自己先崩
        return CheckRow("gpu", STATUS_WARN, f"{type(exc).__name__}: {exc}")


def check_docker() -> CheckRow:
    if shutil.which("docker") is None:
        return CheckRow("docker", STATUS_WARN, "未安装 -> 降级 Milvus Lite")
    rc, out = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if rc != 0:
        return CheckRow("docker", STATUS_WARN, "daemon 未启动 -> 降级 Milvus Lite")
    return CheckRow("docker", STATUS_OK, f"daemon ok {out}")


def check_llm() -> CheckRow:
    from medevirag.common.llm_client import probe_endpoint

    cfg = get_settings()
    target = cfg.llm.providers[cfg.llm.default]
    res = probe_endpoint(f"{target.base_url}/models", proxy=target.proxy, timeout=8.0)
    via = "proxy" if target.proxy else "direct"
    detail = f"{target.base_url}/models -> HTTP {res.status} ({via}) {res.latency_ms}ms"
    return CheckRow(f"llm:{cfg.llm.default}", STATUS_OK if res.ok else STATUS_WARN, detail)


def check_disk() -> CheckRow:
    root = Path(__file__).resolve().parents[2]
    usage = shutil.disk_usage(root)
    free_gb = usage.free // (1024**3)
    status = STATUS_OK if free_gb >= 50 else STATUS_WARN
    return CheckRow("disk", status, f"{root.drive or root} 可用 {free_gb}GB (需要 50GB+)")


def check_config() -> CheckRow:
    cfg = get_settings()
    detail = (
        f"env={cfg.app.env} topk={cfg.retrieval.topk} "
        f"device={cfg.embedding.device} hash={cfg.config_hash()}"
    )
    return CheckRow("config", STATUS_OK, detail)


CHECKS = (check_python, check_gpu, check_docker, check_llm, check_disk, check_config)


def doctor(as_json: bool = False) -> int:
    rows = [fn() for fn in CHECKS]
    if as_json:
        print(json.dumps([r.__dict__ for r in rows], ensure_ascii=False, indent=2))
    else:
        width = max(len(r.name) for r in rows)
        print(f"{'检查项'.ljust(width)}  结果  详情")
        print("-" * (width + 74))
        for row in rows:
            print(f"{row.name.ljust(width)}  {row.status:<4}  {row.detail}")
    return 0 if all(r.status != STATUS_FAIL for r in rows) else 1


# ---------------------------------------------------------------- S1 采集与切分

def _count_tokens() -> Callable[[str], int]:
    from medevirag.ingest.chunker import make_token_counter

    cfg = get_settings()
    model_path = str(Path(__file__).resolve().parents[2] / cfg.embedding.model_path)
    return make_token_counter(model_path)


def _paths() -> tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    data = root / "data"
    return data / "raw" / "zh", data / "interim", data / "processed", data / "reports"


def cmd_fetch(args: argparse.Namespace) -> int:
    """T1：下载或本地登记语料，生成 data/raw/manifest.json。"""
    from medevirag.ingest import fetcher, pipeline

    raw_dir, _interim, _processed, reports = _paths()
    manifest_path = Path(args.manifest) if args.manifest else raw_dir.parent / "manifest.json"
    rows: list[dict[str, object]] = []
    if args.url:
        out = raw_dir
        out.mkdir(parents=True, exist_ok=True)
        client = fetcher.direct_client()
        doc_id = Path(args.url.rstrip("/")).stem or "remote"
        rec = fetcher.fetch_pdf(client, args.url, out, doc_id, min_delay=0.0, max_delay=0.0)
        rows.append(rec.to_dict())
    # 抓取清单里带每份文件的真实 URL，优先用它；--source 只作为兜底
    origin = pipeline.load_manifest(reports / "_scdc_manifest.json")
    for pdf in pipeline.discover_pdfs(raw_dir):
        url = str(origin.get(pdf.name, {}).get("url") or args.source or "")
        rec_local = fetcher.record_local(pdf.stem, pdf, url)
        rows.append(rec_local.to_dict())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json

    existing: list[dict[str, object]] = []
    if manifest_path.exists():
        try:
            data = _json.loads(manifest_path.read_text(encoding="utf-8"))
            existing = data if isinstance(data, list) else list(data.get("items", []))
        except _json.JSONDecodeError:
            existing = []
    merged: dict[str, dict[str, object]] = {}
    for r in existing + rows:
        key = str(r.get("file") or r.get("file_name") or r.get("doc_id") or len(merged))
        merged[key] = r
    manifest_path.write_text(
        _json.dumps({"items": sorted(merged.values(), key=lambda x: str(x.get("file", "")))},
                    ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )
    print("manifest -> " + str(manifest_path) + "  (" + str(len(merged)) + " 条)")
    (reports).mkdir(parents=True, exist_ok=True)
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    """T2/T3：只跑解析与章节树，落 interim，便于单独检查版面质量。"""
    from medevirag.ingest import chunker, parser, pipeline, sectionizer

    raw_dir, interim, _processed, _reports = _paths()
    count = chunker.make_token_counter(str(Path(args.tokenizer)))
    pdfs = pipeline.discover_pdfs(raw_dir)
    if args.only:
        pdfs = [p for p in pdfs if p.stem in set(args.only)]
    for pdf in pdfs:
        blocks, stats = parser.extract_blocks(pdf, pdf.stem)
        kept = sectionizer.merge_cross_page(blocks)
        headings, body_size = sectionizer.detect_headings(kept)
        sections = sectionizer.build_sections(pdf.stem, headings, kept)
        interim.mkdir(parents=True, exist_ok=True)
        pipeline._write_jsonl(
            interim / (pdf.stem + ".blocks.jsonl"),
            [pipeline.block_to_dict(b) for b in kept],
        )
        print(
            pdf.stem + ": pages=" + str(stats.get("page_count")) + " blocks=" + str(len(kept))
            + " headings=" + str(len(headings)) + " sections=" + str(len(sections))
            + " body_size=" + str(body_size)
        )
        _ = count
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    """T4~T6：全量切分，落 data/processed/chunks.jsonl 与四份报告。"""
    from medevirag.ingest import pipeline

    root = Path(__file__).resolve().parents[2]
    raw_dir, interim, processed, reports = _paths()
    count = _count_tokens()
    manifest = pipeline.load_manifest(raw_dir.parent / "_scdc_manifest.json")
    if not manifest:  # 抓取清单实际落在 data/reports/，也要能找到
        manifest = pipeline.load_manifest(reports / "_scdc_manifest.json")
    if not manifest:
        manifest = pipeline.load_manifest(raw_dir.parent / "manifest.json")
    batch = pipeline.run_batch(raw_dir, interim, processed, manifest, count, only=args.only or None)
    n = pipeline.write_chunks(processed / "chunks.jsonl", batch.chunks)
    pipeline.write_chunk_report(reports / "s1_chunk_report.md", batch.chunks, batch, raw_dir)
    pipeline.write_parse_quality(reports / "s1_parse_quality.md", batch.results)
    pipeline.write_evidence_coverage(reports / "s1_evidence_coverage.md", batch.chunks)
    pipeline.write_section_sample(reports / "s1_section_sample.md", batch.chunks)
    st = pipeline.stats_of(batch.chunks)
    print("docs ok=" + str(len(batch.ok_docs)) + " scanned=" + str(len(batch.scanned_docs))
          + " garbage=" + str(len(batch.garbage_docs)) + " error=" + str(len(batch.error_docs))
          + " empty=" + str(len(batch.empty_docs)))
    print("chunks=" + str(n) + " docs=" + str(st["docs"]))
    print("reports -> " + str(reports))
    _ = root
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """按 A1~A6 从已落盘的 chunks.jsonl 重算指标（不重新解析 PDF）。"""
    from medevirag.ingest import pipeline

    raw_dir, _interim, processed, reports = _paths()
    path = Path(args.chunks) if args.chunks else processed / "chunks.jsonl"
    rows = pipeline.load_chunks(path)
    st = pipeline.stats_of(rows)
    n = int(st["chunks"]) or 1
    print("chunks=" + str(st["chunks"]) + " docs=" + str(st["docs"]))
    print("token mean/median/p95 = " + str(st["token_mean"]) + " / " + str(st["token_median"])
          + " / " + str(st["token_p95"]))
    print("A1 section_path 非空 = " + pipeline._pct(int(st["section_path_ok"]), n))
    print("A3 200~400 占比 = " + pipeline._pct(int(st["in_range_200_400"]), n))
    print("A4 over480 = " + str(st["over_480"]))
    print("A5 context_header 空 = " + str(n - int(st["context_header_ok"])))
    pipeline.write_chunk_report(reports / "s1_chunk_report.md", rows, None, raw_dir)
    print("report -> " + str(reports / "s1_chunk_report.md"))
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """S2：建索引（向量化 + Milvus + BM25），可分别跳过。"""
    import numpy as np

    from medevirag.index import (
        bm25_store,
        chunk_meta,
        milvus_store,
    )
    from medevirag.index import (
        embedder as embedder_mod,
    )
    from medevirag.index.paths import (
        BM25_DIR,
        CHUNK_IDS_PATH,
        CHUNK_META_PATH,
        EMB_PATH,
        INDEX_META_PATH,
        ensure_dirs,
        load_retrieval_config,
    )
    from medevirag.ingest import pipeline

    ensure_dirs()
    cfg = load_retrieval_config(milvus_uri=args.milvus_uri or None)
    rows = pipeline.load_chunks(Path(args.chunks) if args.chunks else Path(cfg["chunks_path"]))
    rows.sort(key=lambda r: str(r["chunk_id"]))
    print("chunks=" + str(len(rows)))

    emb = embedder_mod.Embedder(
        cfg["emb_model_dir"],
        device=str(cfg["emb_device"]),
        batch_size=int(cfg["emb_batch_size"]),
        use_fp16=not args.no_fp16,
    )
    texts = [str(r["text"]) for r in rows]
    vectors = emb.encode(texts)
    np.save(EMB_PATH, vectors.astype(np.float32))
    CHUNK_IDS_PATH.write_text(
        "".join(
            json.dumps({"chunk_id": str(r["chunk_id"])}, ensure_ascii=False) + chr(10) for r in rows
        ),
        encoding="utf-8",
    )
    chunk_meta.write_chunk_meta(CHUNK_META_PATH, rows)
    print("emb=" + str(vectors.shape) + " -> " + str(EMB_PATH))

    if not args.skip_milvus:
        client = milvus_store.open_milvus_client(args.milvus_uri or cfg["milvus_uri"])
        milvus_store.ensure_collection(
            client,
            cfg["index"]["collection"],
            drop=args.drop,
            metric=cfg["index"]["metric"],
            hnsw_m=int(cfg["index"]["hnsw"]["M"]),
            hnsw_ef_construction=int(cfg["index"]["hnsw"]["efConstruction"]),
            dense_dim=int(cfg["index"]["dense_dim"]),
        )
        n = milvus_store.insert_chunks(
            client, [str(r["chunk_id"]) for r in rows], vectors, rows,
            collection=cfg["index"]["collection"],
        )
        print("milvus inserted=" + str(n))

    if not args.skip_bm25:
        tok = bm25_store.ChineseTokenizer(
            cfg["bm25"]["userdict"] or None, cfg["bm25"]["stopwords"] or None
        )
        store = bm25_store.BM25Store(
            BM25_DIR, tok, k1=float(cfg["bm25"]["k1"]), b=float(cfg["bm25"]["b"]),
            delta=float(cfg["bm25"]["delta"]), method=str(cfg["bm25"]["method"]),
        )
        store.build([str(r["chunk_id"]) for r in rows], texts)
        store.save()
        print("bm25 N=" + str(len(rows)) + " -> " + str(BM25_DIR))

    meta = {
        "N": len(rows),
        "dense_dim": int(vectors.shape[1]),
        "model_id": Path(cfg["emb_model_dir"]).name,
        "index_meta": str(INDEX_META_PATH.name),
    }
    print("index meta -> " + json.dumps(meta, ensure_ascii=False))
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    """S2：单条 query 检索，打印 Top-k 与两路来源。"""
    from medevirag.index.hybrid import HybridRetriever

    with HybridRetriever(milvus_uri=args.milvus_uri or None) as retriever:
        hits = retriever.retrieve(
            args.query, topk=args.topk, variant=args.variant, apply_filters=not args.no_filter
        )
    print("query=" + args.query + "  variant=" + args.variant + "  hits=" + str(len(hits)))
    for h in hits:
        print(
            "  " + h.chunk_id
            + "  fused=" + format(h.fused_score, ".5f")
            + "  dense=" + format(h.dense_score, ".4f")
            + "  bm25=" + format(h.bm25_score, ".4f")
            + "  src=" + (h.source or "-")
            + "  " + h.doc_title[:30]
        )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """S3：跑一次 8 节点图，打印答案 / 引用 / 状态 / 节点耗时。"""
    from medevirag.rag import nodes as _nodes
    from medevirag.rag.graph import run_ask

    out = run_ask(args.question)
    # 标注当前 LLM 兜底模式：本机无 API key 时一律 FakeLLM，答案不具医学意义，
    # 若不标注，partial/refused 会被误读成链路缺陷。
    mode = "fake-deterministic" if getattr(_nodes.get_llm(), "_fake", None) is not None else "real"
    if args.json:
        out["llm_mode"] = mode
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    print("llm_mode=" + mode + ("（无 API key：答案仅为链路自检，不具医学意义）"
                                if mode != "real" else ""))
    print("status=" + str(out.get("status")) + "  reason=" + str(out.get("refusal_reason")))
    print("answer=" + (out.get("answer") or "<空>"))
    for c in out.get("citations") or []:
        print("  [" + c["marker"] + "] " + c["chunk_id"] + "  p" + str(c["page"]) + "  " + c["url"])
    print("trace=" + json.dumps(out.get("trace") or {}, ensure_ascii=False))
    print(out.get("disclaimer", ""))
    return 0


def cmd_citation(args: argparse.Namespace) -> int:
    """S3：按 chunk_id 回链原文段落 + 页码 + bbox（A9 门禁）。"""
    from medevirag.rag.citations import get_citation_detail

    try:
        detail = get_citation_detail(args.chunk_id)
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(detail, ensure_ascii=False, indent=2))
    else:
        print(detail["chunk_id"] + "  p" + str(detail["page"]))
        print(detail["url"])
        print(detail["text"])
    return 0



def cmd_refusal_matrix(args: argparse.Namespace) -> int:
    """S4：跑 54 条夹具矩阵，打印 A1-A4 指标并落盘（夹具级指标，非评测集指标）。"""
    import json as _json

    from tests.fixtures.builders import run_matrix

    from medevirag.rag.refusal_templates import PARTIAL_NOTES  # noqa: F401

    m = run_matrix()
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(_json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[R1] 用例总数        {m['n_cases']}  (应拒 {m['n_should_refuse']} / 可答 {m['n_should_answer']})")
    cov = f"{m['rules_cover_n']}/17"
    print(f"[R1] 规则覆盖        {cov:<28} == 17     {'PASS' if m['rules_cover_n'] == 17 else 'FAIL'}")
    rr = m["refusal_recall"]
    print(f"[R2] 拒答召回        {rr:.3f}  ({m['n_should_refuse']} 应拒)      >= 0.85   {'PASS' if rr >= 0.85 else 'FAIL'}")
    print(f"[R3] 过度拒答率      {m['over_refusal_rate']:.3f}                   <= 0.12   {'PASS' if m['over_refusal_rate'] <= 0.12 else 'FAIL'}")
    print(f"[R4] 拒答理由正确率  {m['reason_accuracy']:.3f}                    >= 0.80   {'PASS' if m['reason_accuracy'] >= 0.80 else 'FAIL'}")
    print("[R5] 误拒归因 Top3  ", " ".join(f"{k}:{v}" for k, v in list(m["over_refused_attribution"].items())[:3]) or "-")
    print(f"matrix -> {args.out}")
    ok = (m["rules_cover_n"] == 17 and rr >= 0.85
          and m["over_refusal_rate"] <= 0.12 and m["reason_accuracy"] >= 0.80)
    return 0 if ok else 1

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medevirag", description="MedEviRAG 命令行")
    sub = parser.add_subparsers(dest="command", required=True)
    p_doctor = sub.add_parser("doctor", help="环境自检")
    p_doctor.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_fetch = sub.add_parser("fetch", help="S1 语料下载/登记")
    p_fetch.add_argument("--url", default="", help="单条 URL（留空则只登记本地文件）")
    p_fetch.add_argument("--source", default="", help="本地文件的来源页")
    p_fetch.add_argument("--manifest", default="", help="manifest 输出路径")

    p_parse = sub.add_parser("parse", help="S1 解析与章节树（只落 interim）")
    p_parse.add_argument("--only", nargs="*", default=[], help="只处理指定 doc_id")
    p_parse.add_argument("--tokenizer", default="models/bge-m3", help="tokenizer 目录")

    p_chunk = sub.add_parser("chunk", help="S1 全量切分并出报告")
    p_chunk.add_argument("--only", nargs="*", default=[], help="只处理指定 doc_id")

    p_report = sub.add_parser("report", help="S1 指标重算")
    p_report.add_argument("--chunks", default="", help="chunks.jsonl 路径")

    p_index = sub.add_parser("index", help="S2 建索引（向量化 + Milvus + BM25）")
    p_index.add_argument("--chunks", default="", help="chunks.jsonl 路径（默认取配置）")
    p_index.add_argument("--milvus-uri", default="", help="Milvus URI（默认 local Lite）")
    p_index.add_argument("--drop", action="store_true", help="重建集合前先删")
    p_index.add_argument("--skip-milvus", action="store_true", help="只做向量化 + BM25")
    p_index.add_argument("--skip-bm25", action="store_true", help="只做向量化 + Milvus")
    p_index.add_argument("--no-fp16", action="store_true", help="禁用半精度")

    p_retrieve = sub.add_parser("retrieve", help="S2 单条 query 检索")
    p_retrieve.add_argument("query", help="检索问句")
    p_retrieve.add_argument("--topk", type=int, default=10, help="返回条数")
    p_retrieve.add_argument(
        "--variant", default="rrf", choices=("rrf", "weighted-sum", "dense-only", "bm25-only")
    )
    p_retrieve.add_argument("--milvus-uri", default="", help="Milvus URI")
    p_retrieve.add_argument("--no-filter", action="store_true", help="关闭元数据硬过滤")

    p_ask = sub.add_parser("ask", help="S3 循证问答（8 节点图）")
    p_ask.add_argument("question", help="临床问句")
    p_ask.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_citation = sub.add_parser("citation", help="S3 引用回链（按 chunk_id 取原文与页码）")
    p_citation.add_argument("chunk_id", help="形如 CN-HTN-2024#3.2.1#c02")
    p_citation.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_matrix = sub.add_parser("refusal-matrix", help="S4 拒答规则矩阵（54 条夹具）")
    p_matrix.add_argument("--out", default="data/reports/refusal_matrix.json", help="矩阵 JSON 输出路径")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(as_json=args.json)
    if args.command == "fetch":
        return cmd_fetch(args)
    if args.command == "parse":
        return cmd_parse(args)
    if args.command == "chunk":
        return cmd_chunk(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "index":
        return cmd_index(args)
    if args.command == "retrieve":
        return cmd_retrieve(args)
    if args.command == "ask":
        return cmd_ask(args)
    if args.command == "citation":
        return cmd_citation(args)
    if args.command == "refusal-matrix":
        return cmd_refusal_matrix(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
