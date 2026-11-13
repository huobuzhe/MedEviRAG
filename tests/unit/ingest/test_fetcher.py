"""T1 采集：文件名清洗、sha256 前缀、PDF 魔数校验。"""

from __future__ import annotations

from pathlib import Path

from medevirag.ingest import fetcher


def test_sanitize_file_name() -> None:
    assert fetcher.sanitize_file_name('a<b>c:d"e|f?g*h.pdf') == "a_b_c_d_e_f_g_h.pdf"


def test_sha256_of_has_prefix(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"%PDF-1.7 test")
    digest = fetcher.sha256_of(p)
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


def test_is_pdf(tmp_path: Path) -> None:
    good = tmp_path / "good.pdf"
    good.write_bytes(b"%PDF-1.7\n...")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"<html>404</html>")
    assert fetcher.is_pdf(good) is True
    assert fetcher.is_pdf(bad) is False


def test_default_headers_has_ua() -> None:
    h = fetcher.default_headers()
    assert h["User-Agent"] == fetcher.UA
