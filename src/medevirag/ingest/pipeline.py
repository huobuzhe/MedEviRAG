"""S1 编排入口：把 T1~T5 串成一条可复现的流水线（docs/02 4.6）。

一次 run_doc() 的产物固定三件：
    data/interim/<doc_id>.blocks.jsonl   版面块（带页码与 bbox，可回链）
    data/interim/<doc_id>.sections.json  章节树（含 level=0 虚拟 root）
    data/processed/chunks.jsonl          最终检索单元（Chunk 强类型落盘）

幂等性靠两点保证：chunk_id 由 (doc_id, section_no, seq) 决定，与运行次数无关；
text 排序后写入，同一输入的字节序列完全一致。
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medevirag.common.logging import get_logger
from medevirag.common.timing import node_timer
from medevirag.common.types import Chunk, Locator
from medevirag.ingest import chunker, evidence, metadata, parser, sectionizer

_log = get_logger(__name__)


@dataclass(slots=True)
class DocResult:
    """单篇文档的处理结果与诊断信息。"""

    doc_id: str
    status: str  # ok | scanned | garbage | error
    title: str = ""
    doc_type: str = "paper"
    publisher: str = "unknown"
    publish_year: int = 0
    lang: str = "zh"
    page_count: int = 0
    n_blocks: int = 0
    n_sections: int = 0
    n_chunks: int = 0
    note: str = ""
    stats: dict[str, Any] = field(default_factory=dict)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def block_to_dict(b: parser.LayoutBlock) -> dict[str, Any]:
    return {
        "page": b.page,
        "block_index": b.block_index,
        "kind": b.kind,
        "text": b.text,
        "bbox": list(b.bbox) if b.bbox else None,
        "size": round(float(b.size), 2),
        "font": b.font,
        "bold": bool(b.bold),
        "n_chars": int(b.n_chars),
    }


def section_to_dict(s: sectionizer.Section) -> dict[str, Any]:
    return {
        "section_id": s.section_id,
        "doc_id": s.doc_id,
        "section_no": s.section_no,
        "title": s.title,
        "level": s.level,
        "page_start": s.page_start,
        "page_end": s.page_end,
        "order_index": s.order_index,
        "parent_section_id": s.parent_section_id,
        "section_path": list(s.section_path),
        "n_blocks": len(s.block_ids),
    }


def _fallback_title(pdf_path: Path) -> str:
    """标题抽不到时的兜底：用文件名，至少保证 context_header 不为空。"""
    stem = pdf_path.stem
    return stem.replace("_", " ") if stem else "未命名文档"


def run_doc(
    pdf_path: Path,
    doc_id: str,
    out_interim: Path,
    count: Any,
    *,
    source_url: str = "",
    min_chunks: int = 1,
) -> tuple[DocResult, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """处理一篇 PDF；返回 (结果, blocks, sections, chunks)。"""
    res = DocResult(doc_id=doc_id, status="ok")
    try:
        with node_timer("parse", doc_id=doc_id):
            blocks, stats = parser.extract_blocks(pdf_path, doc_id)
    except Exception as exc:  # 单篇解析失败不能中断全量
        res.status = "error"
        res.note = "解析失败: " + str(exc)[:200]
        _log.warning("parse_failed", extra={"doc_id": doc_id, "err": str(exc)[:200]})
        return res, [], [], []

    res.stats = dict(stats)
    res.page_count = int(stats.get("page_count", 0))
    res.n_blocks = len(blocks)
    if stats.get("is_scanned"):
        res.status = "scanned"
        res.note = "扫描件（平均字符/页 < 150），跳过，不污染统计"
        return res, [], [], []
    if stats.get("is_garbage"):
        res.status = "garbage"
        res.note = "文本编码损坏（汉字被映射为替换码点），剔除以避免污染统计"
        return res, [], [], []

    first_page = "".join(b.text for b in blocks if b.page == 1)[:3000]
    title = metadata.extract_title(blocks) or _fallback_title(pdf_path)
    res.title = title
    res.doc_type = metadata.extract_doc_type(title, first_page)
    res.publisher = metadata.extract_publisher(title, first_page, pdf_path.name)
    year = metadata.extract_publish_year(title, pdf_path.name, first_page)
    res.publish_year = year if year is not None else 0
    lang = metadata.extract_lang(first_page)
    res.lang = lang if lang in ("zh", "en") else "zh"

    # 页眉页脚：先按 (页内相对位置 + 归一化文本) 统计跨页重复行，再统一剔除
    try:
        margins = parser.collect_margin_lines(pdf_path)
        reject = sectionizer.detect_repeated_margins(margins)
    except Exception as exc:  # 单篇失败不中断
        reject = set()
        _log.warning("margin_detect_failed", extra={"doc_id": doc_id, "err": str(exc)[:120]})

    kept = [b for b in blocks if not sectionizer.is_margin(b.text, reject)]
    kept = sectionizer.merge_cross_page(kept)

    headings, _body_size = sectionizer.detect_headings(kept)
    sections = sectionizer.build_sections(doc_id, headings, kept)
    res.n_sections = len(sections)

    by_id = {sectionizer.block_id_of(b): b for b in kept}
    sec_units: list[tuple[sectionizer.Section, list[chunker.Unit]]] = []
    for sec in sections:
        if sec.level == 0:  # 虚拟 root 只做容器，不产块
            continue
        sec_blocks = [by_id[bid] for bid in sec.block_ids if bid in by_id]
        if not sec_blocks:
            continue
        units = chunker.to_units(sec_blocks, count)
        if units:
            sec_units.append((sec, units))

    chunks: list[Chunk] = []
    raw_chunks: list[dict[str, Any]] = []
    for sec, item in chunker.chunk_stream(sec_units, count):
        # 无编号的节（封面/附录/期刊栏目）用 order_index 兜底，避免所有节都撞成 "#root#"
        no = sec.section_no or ("h" + str(sec.order_index))
        cid = chunker.build_chunk_id(doc_id, no, int(item["_seq"]))
        header = chunker.context_header(res.title, sec.section_path)
        ev = evidence.extract_evidence(item["text"])
        loc = item["locator"]
        bbox = tuple(loc["bbox"]) if loc.get("bbox") else None
        try:
            ch = Chunk(
                chunk_id=cid,
                parent_id=sec.section_id,
                doc_id=doc_id,
                doc_title=res.title,
                doc_type=res.doc_type,  # type: ignore[arg-type]
                publisher=res.publisher,
                publish_year=res.publish_year,
                lang=res.lang,  # type: ignore[arg-type]
                section_path=list(sec.section_path),
                section_level=sec.level,
                page_start=int(item["page_start"]),
                page_end=int(item["page_end"]),
                chunk_type=item["chunk_type"],
                text=item["text"],
                context_header=header,
                evidence_level=ev["evidence_level"],  # type: ignore[arg-type]
                rec_strength=ev["rec_strength"],  # type: ignore[arg-type]
                grade_system=ev["grade_system"],  # type: ignore[arg-type]
                token_len=int(item["token_len"]),
                content_hash=chunker.content_hash(item["text"]),
                locator=Locator(
                    page=int(loc["page"]),
                    block_index=int(loc.get("block_index", 0)),
                    bbox=bbox,
                ),
            )
            chunker.assert_hard_constraints(ch.to_dict())
        except (AssertionError, ValueError) as exc:
            _log.warning("chunk_rejected", extra={"doc_id": doc_id, "chunk_id": cid, "err": str(exc)[:160]})
            continue
        d = ch.to_dict()
        d["source_url"] = source_url
        chunks.append(ch)
        raw_chunks.append(d)


    res.n_chunks = len(chunks)
    if res.n_chunks < min_chunks:
        res.status = "empty"
        res.note = "未产出任何 chunk（可能全部无标题且无正文）"

    out_interim.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_interim / (doc_id + ".blocks.jsonl"), [block_to_dict(b) for b in kept])
    sections_doc = {
        "doc_id": doc_id,
        "page_offset": 0,
        "toc_trusted": False,
        "sections": [section_to_dict(s) for s in sections],
    }
    (out_interim / (doc_id + ".sections.json")).write_text(
        json.dumps(sections_doc, ensure_ascii=False, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    return res, [block_to_dict(b) for b in kept], [section_to_dict(s) for s in sections], raw_chunks


def discover_pdfs(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.glob("*.pdf") if p.is_file())


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """读抓取清单，按文件名建索引，供 pipeline 补 source_url/sha256。"""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("items", [])
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        name = str(r.get("file") or r.get("file_name") or "")
        if name:
            out[Path(name).name] = r
    return out


def stats_of(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """chunks 的聚合统计，报告与验收共用同一份实现，避免两套口径。"""
    import statistics

    toks = [int(r["token_len"]) for r in rows] if rows else []
    doc_ids = {str(r["doc_id"]) for r in rows}
    types: dict[str, int] = {}
    for r in rows:
        types[str(r["chunk_type"])] = types.get(str(r["chunk_type"]), 0) + 1
    sec_ok = sum(1 for r in rows if r.get("section_path"))
    hdr_ok = sum(1 for r in rows if r.get("context_header"))
    # 回链定位：bbox 退化与跨页块数量决定"引用落点可用率"（00 §9 要求 >= 0.90）
    zero_bbox = sum(1 for r in rows if _is_zero_bbox(r))
    cross_page = sum(
        1 for r in rows if int(r.get("page_start", 0)) != int(r.get("page_end", 0))
    )
    idx_zero = sum(1 for r in rows if int((r.get("locator") or {}).get("block_index", 0)) == 0)
    return {
        "chunks": len(rows),
        "docs": len(doc_ids),
        "token_mean": round(statistics.fmean(toks), 1) if toks else 0.0,
        "token_median": int(statistics.median(toks)) if toks else 0,
        "token_p95": int(sorted(toks)[max(0, int(len(toks) * 0.95) - 1)]) if toks else 0,
        "token_min": min(toks) if toks else 0,
        "token_max": max(toks) if toks else 0,
        "in_range_200_400": sum(1 for t in toks if 200 <= t <= 400),
        "over_480": sum(1 for t in toks if t > 480),
        "chunk_types": types,
        "section_path_ok": sec_ok,
        "context_header_ok": hdr_ok,
        "bbox_zero": zero_bbox,
        "bbox_ok": len(rows) - zero_bbox,
        "cross_page_chunks": cross_page,
        "locator_index_zero": idx_zero,
    }


def measure_bbox_coverage(
    pdf_dir: Path, rows: list[dict[str, Any]], *, limit: int = 120
) -> dict[str, Any]:
    """实测 bbox 的文字覆盖率：把 chunk.text 与 bbox 裁剪区内的 PDF 文字比长度。

    覆盖率低不代表 bbox 错——一个 chunk 常由多个版面块合并而成，而 locator 只能带
    一个矩形。这里量化它，让"引用落点可用率"有可复现的基线，而不是靠感觉。
    """
    import fitz  # 延迟导入：无 PDF 的纯逻辑单测不该拉起 MuPDF

    single = [r for r in rows if int(r["page_start"]) == int(r["page_end"])]
    others = [r for r in rows if int(r["page_start"]) != int(r["page_end"])]
    step = max(1, len(single) // max(1, limit))
    sample = [r for i, r in enumerate(sorted(single, key=lambda x: str(x["chunk_id"]))) if i % step == 0]
    sample = sample[:limit]
    ratios: list[float] = []
    per_type: dict[str, list[float]] = {}
    docs: dict[str, Any] = {}
    misses = 0
    for r in sample:
        doc_id = str(r["doc_id"])
        if doc_id not in docs:
            path = pdf_dir / (doc_id + ".pdf")
            docs[doc_id] = fitz.open(str(path)) if path.exists() else None
        doc = docs[doc_id]
        if doc is None:
            misses += 1
            continue
        bbox = (r.get("locator") or {}).get("bbox")
        if not bbox:
            misses += 1
            continue
        page = doc[int(r["page_start"]) - 1]
        clip = fitz.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        covered = "".join(str(page.get_text("text", clip=clip)).split())
        target = "".join(str(r["text"]).split())
        if not target:
            continue
        ratio = min(1.0, len(covered) / len(target))
        ratios.append(ratio)
        per_type.setdefault(str(r["chunk_type"]), []).append(ratio)
    for doc in docs.values():
        if doc is not None:
            _close_quietly(doc)
    return {
        "sampled": len(ratios),
        "skipped": misses,
        "single_page": len(single),
        "cross_page": len(others),
        "mean": round(sum(ratios) / len(ratios), 3) if ratios else 0.0,
        "median": round(sorted(ratios)[len(ratios) // 2], 3) if ratios else 0.0,
        "ge_095": sum(1 for x in ratios if x >= 0.95),
        "ge_080": sum(1 for x in ratios if x >= 0.80),
        "lt_050": sum(1 for x in ratios if x < 0.50),
        "ratios": [round(x, 3) for x in ratios],
        "by_type": {k: round(sum(v) / len(v), 3) for k, v in sorted(per_type.items())},
        "para_buckets": _bucket(per_type.get("paragraph", [])),
        "all_buckets": _bucket(ratios),
    }


def _bucket(ratios: list[float]) -> dict[str, int]:
    """把覆盖率分桶，让"合并块定位到什么程度"是一张表而不是一句均值。"""
    return {
        "=1.0": sum(1 for x in ratios if x >= 0.999),
        "0.95~1": sum(1 for x in ratios if 0.95 <= x < 0.999),
        "0.80~0.95": sum(1 for x in ratios if 0.80 <= x < 0.95),
        "0.50~0.80": sum(1 for x in ratios if 0.50 <= x < 0.80),
        "<0.50": sum(1 for x in ratios if x < 0.50),
    }


def _close_quietly(doc: Any) -> None:
    """关闭 PDF，损坏文件在报告阶段也不该抛错。"""
    with contextlib.suppress(Exception):
        doc.close()


def _is_zero_bbox(row: dict[str, Any]) -> bool:
    """bbox 缺失或退化成零面积，视为无法高亮回链。"""
    bbox = (row.get("locator") or {}).get("bbox")
    if not bbox:
        return True
    return bool(float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1]))


# ---------------------------------------------------------------- 批量与报告

@dataclass(slots=True)
class BatchResult:
    """一次全量处理的汇总，供 CLI 与报告共用。"""

    results: list[DocResult] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    ok_docs: list[str] = field(default_factory=list)
    scanned_docs: list[str] = field(default_factory=list)
    garbage_docs: list[str] = field(default_factory=list)
    error_docs: list[str] = field(default_factory=list)
    empty_docs: list[str] = field(default_factory=list)


def run_batch(
    raw_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
    manifest: dict[str, dict[str, Any]],
    count: Any,
    *,
    only: list[str] | None = None,
) -> BatchResult:
    """遍历语料目录，逐篇处理并汇总。单篇失败只记录，不中断全量。"""
    batch = BatchResult()
    pdfs = discover_pdfs(raw_dir)
    if only:
        wanted = {Path(x).stem for x in only}
        pdfs = [p for p in pdfs if p.stem in wanted]
    for pdf in pdfs:
        doc_id = pdf.stem
        meta = manifest.get(pdf.name, {})
        res, _blocks, _sections, rows = run_doc(
            pdf, doc_id, interim_dir, count, source_url=str(meta.get("url", ""))
        )
        batch.results.append(res)
        bucket = {
            "ok": batch.ok_docs,
            "scanned": batch.scanned_docs,
            "garbage": batch.garbage_docs,
            "error": batch.error_docs,
            "empty": batch.empty_docs,
        }.get(res.status)
        if bucket is not None:
            bucket.append(doc_id)
        batch.chunks.extend(rows)
    (processed_dir / "chunks.jsonl").parent.mkdir(parents=True, exist_ok=True)
    return batch


def write_chunks(path: Path, rows: list[dict[str, Any]]) -> int:
    """按 chunk_id 排序落盘：同一输入两次运行得到的字节序列完全一致（A10）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: str(r["chunk_id"]))
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in ordered:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return len(ordered)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pct(a: int, b: int) -> str:
    return (str(round(100.0 * a / b, 2)) + "%") if b else "n/a"


def write_chunk_report(
    path: Path,
    rows: list[dict[str, Any]],
    batch: BatchResult | None = None,
    pdf_dir: Path | None = None,
) -> None:
    """data/reports/s1_chunk_report.md：A1~A6 的自证材料 + 回链定位量化基线。"""
    st = stats_of(rows)
    n = st["chunks"]
    n_docs = st["docs"]
    in_range = int(st["in_range_200_400"])
    lines: list[str] = []
    lines.append("# S1 切分报告（chunks）")
    lines.append("")
    lines.append("## 总量")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    lines.append("| chunk 总数 | " + str(n) + " |")
    lines.append("| 覆盖文档数 | " + str(n_docs) + " |")
    lines.append("| token 均值 | " + str(st["token_mean"]) + " |")
    lines.append("| token 中位数 | " + str(st["token_median"]) + " |")
    lines.append("| token P95 | " + str(st["token_p95"]) + " |")
    lines.append("| token 最小/最大 | " + str(st["token_min"]) + " / " + str(st["token_max"]) + " |")
    lines.append("")
    lines.append("## 验收指标")
    lines.append("")
    lines.append("| 项 | 指标 | 实测 | 结论 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        "| A1 | section_path 非空率 >= 95% | "
        + _pct(int(st["section_path_ok"]), n)
        + " ("
        + str(st["section_path_ok"])
        + "/"
        + str(n)
        + ") | "
        + ("PASS" if n and st["section_path_ok"] / n >= 0.95 else "FAIL")
        + " |"
    )
    tables = [r for r in rows if r["chunk_type"] == "table"]
    table_cross = [r for r in tables if r["page_start"] != r["page_end"]]
    lines.append(
        "| A2 | table 块跨页截断数 = 0 | "
        + str(len(table_cross))
        + " / "
        + str(len(tables))
        + " 个 table 块 | "
        + ("PASS" if not table_cross else "FAIL")
        + " |"
    )
    lines.append(
        "| A3 | 200~400 token 占比 >= 85% | "
        + _pct(in_range, n)
        + " ("
        + str(in_range)
        + "/"
        + str(n)
        + ") | "
        + ("PASS" if n and in_range / n >= 0.85 else "FAIL")
        + " |"
    )
    lines.append(
        "| A4 | token_len > 480 的块数 = 0 | "
        + str(st["over_480"])
        + " | "
        + ("PASS" if st["over_480"] == 0 else "FAIL")
        + " |"
    )
    lines.append(
        "| A5 | context_header 为空 = 0 | "
        + str(n - int(st["context_header_ok"]))
        + " | "
        + ("PASS" if st["context_header_ok"] == n else "FAIL")
        + " |"
    )
    ev = sum(1 for r in rows if str(r["evidence_level"]) != "unknown")
    rec = sum(1 for r in rows if str(r["rec_strength"]) != "unknown")
    lines.append(
        "| A6 | evidence_level 可得率（参考 20%~40%） | "
        + _pct(ev, n)
        + " | 如实报告 |"
    )
    lines.append(
        "| A6 | rec_strength 可得率（参考 30%~50%） | " + _pct(rec, n) + " | 如实报告 |"
    )
    lines.append("")
    lines.append("## chunk_type 分布")
    lines.append("")
    lines.append("| chunk_type | 数量 | 占比 |")
    lines.append("| --- | --- | --- |")
    for k in sorted(st["chunk_types"]):
        v = int(st["chunk_types"][k])
        lines.append("| " + k + " | " + str(v) + " | " + _pct(v, n) + " |")
    lines.append("")
    lines.append("")
    lines.append("## 回链定位（locator）")
    lines.append("")
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("| --- | --- | --- |")
    lines.append(
        "| bbox 退化（零面积/缺失）块数 | " + str(st["bbox_zero"]) + " | 退化即无法在 PDF 上高亮回链 |"
    )
    lines.append("| bbox 有效块数 | " + str(st["bbox_ok"]) + " | 含表格块的真实表格矩形 |")
    lines.append(
        "| locator.block_index == 0 的块数 | "
        + str(st["locator_index_zero"])
        + " | 版面块序号透传后的残留（首块序号本就是 0） |"
    )
    lines.append(
        "| 跨页块数 | "
        + str(st["cross_page_chunks"])
        + " | 跨页块 bbox 只能定位首段，无法覆盖后几页 |"
    )
    lines.append("")
    lines.append("bbox 取值规则：")
    lines.append("")
    lines.append("- 单页 chunk：取该页所有组成单元 bbox 的**并集**，高亮可盖住整块；")
    lines.append("- 跨页 chunk：只保留**首段**矩形，本文如实声明「只能定位首段」；")
    lines.append("- 表格块：pdfium find_tables() 的真实表格矩形；降级通道取 pdfplumber Table.bbox；")
    lines.append("  两者都拿不到时退化为整页矩形（绝不写零矩形）。")
    lines.append("")
    if pdf_dir is not None and pdf_dir.exists():
        cov = measure_bbox_coverage(pdf_dir, rows)
        lines.append("### bbox 文字覆盖率实测（抽样 " + str(cov["sampled"]) + " 个单页块）")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("| --- | --- |")
        lines.append("| 抽样块数 | " + str(cov["sampled"]) + " |")
        lines.append("| 无法定位（跳过） | " + str(cov["skipped"]) + " |")
        lines.append("| 覆盖率均值 | " + str(cov["mean"]) + " |")
        lines.append("| 覆盖率中位数 | " + str(cov["median"]) + " |")
        lines.append(
            "| 覆盖率 >= 0.95 占比 | "
            + _pct(int(cov["ge_095"]), int(cov["sampled"]))
            + " ("
            + str(cov["ge_095"])
            + "/"
            + str(cov["sampled"])
            + ") |"
        )
        lines.append(
            "| 覆盖率 >= 0.80 占比 | "
            + _pct(int(cov["ge_080"]), int(cov["sampled"]))
            + " ("
            + str(cov["ge_080"])
            + "/"
            + str(cov["sampled"])
            + ") |"
        )
        lines.append(
            "| 覆盖率 < 0.50 占比 | " + _pct(int(cov["lt_050"]), int(cov["sampled"])) + " |"
        )
        lines.append("")
        lines.append(
            "按 chunk_type 的覆盖率均值："
            + ", ".join(k + " " + str(v) for k, v in cov["by_type"].items())
        )
        lines.append("")
        lines.append(
            "覆盖率分桶（全部 " + str(cov["sampled"]) + " 个抽样块）："
            + ", ".join(f"{k} {v}" for k, v in cov["all_buckets"].items())
        )
        lines.append("")
        lines.append(
            "覆盖率分桶（抽样中的 paragraph 块，共 "
            + str(sum(cov["para_buckets"].values()))
            + " 个）："
            + ", ".join(f"{k} {v}" for k, v in cov["para_buckets"].items())
        )
        lines.append("")
        lines.append(
            "口径说明：单页块已改为并集 bbox；跨页块共 "
            + str(cov["cross_page"])
            + " 个，其 bbox 只覆盖首段，覆盖率必然偏低——这是设计取舍（不做多矩形），"
            + "S3/S5 引用落点以 locator.page 为准、bbox 仅用于首段高亮。"
        )
        lines.append("")
    lines.append("## 文档处理结果")
    lines.append("")
    if batch is not None:
        lines.append("| 状态 | 文档数 |")
        lines.append("| --- | --- |")
        lines.append("| ok | " + str(len(batch.ok_docs)) + " |")
        lines.append("| scanned（扫描件跳过） | " + str(len(batch.scanned_docs)) + " |")
        lines.append("| garbage（编码损坏剔除） | " + str(len(batch.garbage_docs)) + " |")
        lines.append("| empty（未产出块） | " + str(len(batch.empty_docs)) + " |")
        lines.append("| error（解析失败） | " + str(len(batch.error_docs)) + " |")
        lines.append("")
        lines.append("扫描件清单：" + (", ".join(batch.scanned_docs) or "无"))
        lines.append("")
        lines.append("编码损坏清单：" + (", ".join(batch.garbage_docs) or "无"))
        lines.append("")
        if batch.error_docs:
            lines.append("解析失败清单：" + ", ".join(batch.error_docs))
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


def write_parse_quality(path: Path, results: list[DocResult]) -> None:
    """data/reports/s1_parse_quality.md：解析层质量（字符数、空页、扫描件判定）。"""
    lines = ["# S1 解析质量报告", "", "| doc_id | 状态 | 页数 | 块数 | 节数 | 块数/页 | 空页 | 字符/页 | 说明 |"]
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for res in results:
        pc = int(res.page_count) or 0
        per_page = round(res.n_blocks / pc, 1) if pc else 0.0
        raw_empty = res.stats.get("empty_pages", 0)
        empty = len(raw_empty) if isinstance(raw_empty, list) else int(raw_empty)
        cpp = round(float(res.stats.get("chars_per_page_mean", 0.0)), 1)
        note = res.note.replace("|", "/") if res.note else ""
        lines.append(
            "| " + res.doc_id + " | " + res.status + " | " + str(pc) + " | " + str(res.n_blocks)
            + " | " + str(res.n_sections) + " | " + str(per_page) + " | " + str(empty)
            + " | " + str(cpp) + " | " + note + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


def write_evidence_coverage(path: Path, rows: list[dict[str, Any]]) -> None:
    """data/reports/s1_evidence_coverage.md：证据等级/推荐强度可得率（A6）。"""
    rep = evidence.coverage_report(rows)
    lines = ["# S1 证据等级覆盖报告", ""]
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    lines.append("| chunk 总数 | " + str(rep["chunks"]) + " |")
    lines.append(
        "| evidence_level 可得率 | " + _pct(int(rep["evidence_level_hits"]), int(rep["chunks"])) + " |"
    )
    lines.append(
        "| rec_strength 可得率 | " + _pct(int(rep["rec_strength_hits"]), int(rep["chunks"])) + " |"
    )
    lv: dict[str, int] = {}
    rc: dict[str, int] = {}
    gs: dict[str, int] = {}
    for r in rows:
        lv[str(r["evidence_level"])] = lv.get(str(r["evidence_level"]), 0) + 1
        rc[str(r["rec_strength"])] = rc.get(str(r["rec_strength"]), 0) + 1
        gs[str(r["grade_system"])] = gs.get(str(r["grade_system"]), 0) + 1
    lines.append("")
    lines.append("## evidence_level 分布")
    lines.append("")
    lines.append("| 值 | 数量 |")
    lines.append("| --- | --- |")
    for k in sorted(lv):
        lines.append("| " + k + " | " + str(lv[k]) + " |")
    lines.append("")
    lines.append("## rec_strength 分布")
    lines.append("")
    lines.append("| 值 | 数量 |")
    lines.append("| --- | --- |")
    for k in sorted(rc):
        lines.append("| " + k + " | " + str(rc[k]) + " |")
    lines.append("")
    lines.append("## grade_system 分布")
    lines.append("")
    lines.append("| 值 | 数量 |")
    lines.append("| --- | --- |")
    for k in sorted(gs):
        lines.append("| " + k + " | " + str(gs[k]) + " |")
    lines.append("")
    lines.append("## 每篇文档的 evidence_level 覆盖率")
    lines.append("")
    lines.append("| doc_id | chunk 数 | 命中数 | 覆盖率 |")
    lines.append("| --- | --- | --- | --- |")
    for doc_id in sorted(rep["per_doc"]):
        slot = rep["per_doc"][doc_id]
        lines.append(
            "| " + doc_id + " | " + str(slot["chunks"]) + " | " + str(slot["evidence_level_hits"])
            + " | " + _pct(int(slot["evidence_level_hits"]), int(slot["chunks"])) + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


def write_section_sample(path: Path, rows: list[dict[str, Any]], limit: int = 30) -> None:
    """data/reports/s1_section_sample.md：30 条跨文档抽样，人工可核对。"""
    picked: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for r in rows:
        d = str(r["doc_id"])
        if seen.get(d, 0) >= 2:
            continue
        seen[d] = seen.get(d, 0) + 1
        picked.append(r)
        if len(picked) >= limit:
            break
    lines = ["# S1 章节抽样核对表（30 条）", ""]
    lines.append("| # | doc_id | section_path | context_header | 页 | token | 类型 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(picked, start=1):
        sp = " > ".join(str(x) for x in r.get("section_path") or []) or "(空)"
        lines.append(
            "| " + str(i) + " | " + str(r["doc_id"]) + " | " + sp + " | " + str(r["context_header"])
            + " | " + str(r["page_start"]) + " | " + str(r["token_len"]) + " | " + str(r["chunk_type"]) + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
