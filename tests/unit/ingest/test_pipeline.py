"""T4/T5 编排：stats_of 口径、run_doc 的扫描件早退。"""

from __future__ import annotations

from medevirag.ingest import pipeline


def test_stats_of_metrics() -> None:
    rows = [
        {"token_len": 250, "chunk_type": "paragraph", "doc_id": "d1", "section_path": ["a"], "context_header": "h"},
        {"token_len": 500, "chunk_type": "table", "doc_id": "d1", "section_path": [], "context_header": "h"},
    ]
    st = pipeline.stats_of(rows)
    assert st["chunks"] == 2
    assert st["docs"] == 1
    assert st["token_max"] == 500
    assert st["in_range_200_400"] == 1
    assert st["over_480"] == 1
    assert st["section_path_ok"] == 1
    assert st["chunk_types"] == {"paragraph": 1, "table": 1}


def test_stats_of_empty() -> None:
    st = pipeline.stats_of([])
    assert st["chunks"] == 0
    assert st["token_mean"] == 0.0


def test_load_manifest_shape(tmp_path) -> None:
    p = tmp_path / "m.json"
    p.write_text('{"items": [{"file": "a.pdf", "url": "u"}]}', encoding="utf-8")
    got = pipeline.load_manifest(p)
    assert got["a.pdf"]["url"] == "u"


def test_load_manifest_missing(tmp_path) -> None:
    assert pipeline.load_manifest(tmp_path / "nope.json") == {}


def test_discover_pdfs_sorted(tmp_path) -> None:
    for name in ("b.pdf", "a.pdf", "c.txt"):
        (tmp_path / name).write_bytes(b"%PDF-1.7")
    got = [p.name for p in pipeline.discover_pdfs(tmp_path)]
    assert got == ["a.pdf", "b.pdf"]


def test_run_doc_rejects_scanned_pdf(monkeypatch, fake_count, tmp_path) -> None:
    """扫描件必须原地早退为 scanned，不产出 chunk，也不写 interim。"""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7")

    def fake_extract(_path, doc_id):
        return [], {
            "doc_id": doc_id,
            "page_count": 3,
            "chars_total": 0,
            "chars_per_page_mean": 0.0,
            "empty_pages": [1, 2, 3],
            "is_scanned": True,
        }

    monkeypatch.setattr(pipeline.parser, "extract_blocks", fake_extract)
    res, blocks, sections, chunks = pipeline.run_doc(pdf, "x", tmp_path / "it", fake_count)
    assert res.status == "scanned"
    assert blocks == [] and sections == [] and chunks == []


def test_run_doc_rejects_garbage_text(monkeypatch, fake_count, tmp_path) -> None:
    """可读性判据判为乱码的文档同样早退，不与扫描件混为一类。"""
    pdf = tmp_path / "y.pdf"
    pdf.write_bytes(b"%PDF-1.7")

    def fake_extract(_path, doc_id):
        return [], {
            "doc_id": doc_id,
            "page_count": 2,
            "chars_total": 800,
            "chars_per_page_mean": 400.0,
            "empty_pages": [],
            "is_scanned": False,
            "is_garbage": True,
        }

    monkeypatch.setattr(pipeline.parser, "extract_blocks", fake_extract)
    res, _b, _s, chunks = pipeline.run_doc(pdf, "y", tmp_path / "it", fake_count)
    assert res.status == "garbage"
    assert chunks == []


def test_run_doc_end_to_end_on_blocks(monkeypatch, fake_count, tmp_path) -> None:
    """正文+标题的合成文档要能走完 parse->section->chunk，且字段齐全。"""
    from medevirag.ingest import parser as p_mod

    body = "高血压患者应限制钠盐摄入。" * 40
    blocks = [
        p_mod.LayoutBlock(1, 0, "body", "一、饮食干预", (72.0, 80.0, 300.0, 96.0), 18.0, "SimSun", False, 7),
        p_mod.LayoutBlock(1, 1, "body", body, (72.0, 100.0, 500.0, 600.0), 12.0, "SimSun", False, len(body)),
    ]
    pdf = tmp_path / "z.pdf"
    pdf.write_bytes(b"%PDF-1.7")

    def fake_extract(_path, doc_id):
        return blocks, {
            "doc_id": doc_id,
            "page_count": 1,
            "chars_total": 2000,
            "chars_per_page_mean": 2000.0,
            "empty_pages": [],
            "is_scanned": False,
            "is_garbage": False,
        }

    monkeypatch.setattr(pipeline.parser, "extract_blocks", fake_extract)
    monkeypatch.setattr(pipeline.parser, "collect_margin_lines", lambda _p: {})
    res, _b, sections, chunks = pipeline.run_doc(pdf, "z", tmp_path / "it", fake_count)
    assert res.status == "ok"
    assert sections and chunks
    first = chunks[0]
    assert first["context_header"].startswith("《")
    assert isinstance(first["section_path"], list) and first["section_path"]
    assert first["content_hash"].startswith("sha256:")
    assert first["locator"]["page"] == 1
    assert (tmp_path / "it" / "z.sections.json").exists()
