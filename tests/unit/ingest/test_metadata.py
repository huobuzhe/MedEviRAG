"""T5 元数据：doc_type / publisher / year / lang / title 抽取。"""

from __future__ import annotations

import pytest

from medevirag.ingest.metadata import (
    extract_doc_type,
    extract_lang,
    extract_publish_year,
    extract_publisher,
    extract_title,
)


@pytest.mark.parametrize(
    ("title", "expect"),
    [
        ("成人高血压食养指南（2023年版）", "guideline"),
        ("中国高血压防治指南", "guideline"),
        ("艾滋病诊疗专家共识", "consensus"),
        ("全国艾滋病检测技术规范（2020年修订版）", "manual"),
    ],
)
def test_extract_doc_type(title: str, expect: str) -> None:
    assert extract_doc_type(title, "") == expect


def test_extract_doc_type_fallback_paper() -> None:
    assert extract_doc_type("某某观察性研究", "abstract") == "paper"


def test_extract_publisher_whitelist() -> None:
    got = extract_publisher("成人高血压食养指南", "国家卫生健康委员会 发布", "x.pdf")
    assert got == "国家卫生健康委员会"


def test_extract_publisher_unknown() -> None:
    assert extract_publisher("无关文档", "", "a.pdf") == "unknown"


def test_extract_publish_year() -> None:
    assert extract_publish_year("（2023 年版）", "2020年修订") == 2023
    assert extract_publish_year("无年份") is None
    assert extract_publish_year("1850 年") is None  # 超出 1990~2026


def test_extract_lang() -> None:
    assert extract_lang("高血压诊断标准") == "zh"
    assert extract_lang("hypertension guideline") == "en"
    assert extract_lang("") == "unknown"


def test_extract_title_takes_largest_font(make_block) -> None:
    blocks = [
        make_block(page=1, block_index=0, text="封面装饰", size=8.0),
        make_block(page=1, block_index=1, text="成人高血压食养指南", size=22.0),
        make_block(page=1, block_index=2, text="国家卫生健康委", size=12.0),
    ]
    assert extract_title(blocks) == "成人高血压食养指南"
