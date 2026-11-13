"""T6/A9 门禁：引用解析、回链 URL、get_citation_detail 12 键。

真实 chunk_id 含中文/特殊字符，不匹配 CHUNK_ID_RE；这里用示例 chunk_id 造 fixture，
通过 monkeypatch _chunk_index 提供数据，从而不依赖真实语料。
"""

from __future__ import annotations

import pytest

from medevirag.rag import citations as C

SAMPLE_ID = "CN-HTN-2024#3.2.1#c02"
SAMPLE_ROW = {
    "chunk_id": SAMPLE_ID,
    "doc_id": "CN-HTN-2024",
    "doc_title": "中国高血压防治指南（2024年修订版）",
    "section_path": ["第3章 诊断与评估", "3.2 血压测量", "3.2.1 诊室血压"],
    "chunk_type": "recommendation",
    "locator": {"page": 27, "bbox": [72, 318, 523, 402], "block_index": 4},
    "page_start": 27,
    "text": "诊室血压≥140/90 mmHg 可诊断高血压。",
    "context_header": "《中国高血压防治指南（2024年修订版）》 > 3.2.1 诊室血压",
    "evidence_level": "A",
}


def _evidence(marker="C1", chunk_id=SAMPLE_ID):
    return {
        marker: {
            "chunk_id": chunk_id,
            "doc_id": "CN-HTN-2024",
            "doc_title": SAMPLE_ROW["doc_title"],
            "section_path": SAMPLE_ROW["section_path"],
            "page": 27,
            "locator": SAMPLE_ROW["locator"],
            "text": SAMPLE_ROW["text"],
        }
    }


def test_markers_in_order_and_dedup():
    assert C.markers_in("A [C1] B 【c2】 C [C 3] D [C1]") == ["C1", "C2", "C3"]


def test_split_sentences_keeps_punctuation():
    parts = C.split_sentences("第一句。第二句！第三句？")
    assert parts == ["第一句。", "第二句！", "第三句？"]


def test_make_url_format():
    url = C.make_url("CN-HTN-2024", {"page": 27, "bbox": [72, 318, 523, 402]})
    assert url == "medevirag://doc/CN-HTN-2024?page=27&bbox=72,318,523,402"


def test_build_citations_only_known_markers():
    ev = _evidence()
    ev.update({"C9": dict(ev["C1"])})
    cs = C.build_citations("结论 [C1]。越界 [C9]。", {"C1": ev["C1"]})
    assert [c.marker for c in cs] == ["C1"]
    assert cs[0].url.startswith("medevirag://doc/CN-HTN-2024?page=27")


def test_get_citation_detail_keys_and_types(monkeypatch):
    monkeypatch.setattr(C, "_chunk_index", lambda: {SAMPLE_ID: SAMPLE_ROW})
    d = C.get_citation_detail(SAMPLE_ID)
    assert tuple(d) == C.DETAIL_KEYS
    assert isinstance(d["page"], int)
    assert isinstance(d["bbox"], list) and len(d["bbox"]) == 4
    assert d["url"].startswith("medevirag://doc/")


def test_get_citation_detail_rejects_bad_format(monkeypatch):
    monkeypatch.setattr(C, "_chunk_index", lambda: {})
    with pytest.raises(ValueError):
        C.get_citation_detail("NOT A CHUNK ID")


def test_get_citation_detail_missing_raises_keyerror(monkeypatch):
    monkeypatch.setattr(C, "_chunk_index", lambda: {SAMPLE_ID: SAMPLE_ROW})
    with pytest.raises(KeyError):
        C.get_citation_detail("CN-HTN-2024#9.9.9#c99")
