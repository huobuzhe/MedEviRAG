"""T3 章节感知：编号识别（6 正例 + 反例）与页眉页脚剔除（3 条）。"""

from __future__ import annotations

import pytest

from medevirag.ingest.sectionizer import (
    build_sections,
    cn2int,
    detect_headings,
    detect_repeated_margins,
    is_pure_page_number,
    match_number,
    merge_cross_page,
)

NOISE_FIXTURES = [
    ("３．２．１　诊室血压", ("multi_arabic", "3.2.1", 3)),
    ("3 . 2 . 1 诊室血压", ("multi_arabic", "3.2.1", 3)),
    ("第三章 诊断与评估", ("cn_chapter", "第三章", 1)),
    ("（二）药物治疗", ("cn_paren", "(二)", 3)),
    ("一、一般人群", ("cn_dunhao", "一、", 2)),
    ("2.1 诊断标准", ("multi_arabic", "2.1", 2)),
]


@pytest.mark.parametrize(("text", "expect"), NOISE_FIXTURES)
def test_match_number_accepts(text: str, expect: tuple[str, str, int]) -> None:
    """全角、空格隔开的点分编号都要归一成同一个编号串。"""
    got = match_number(text)
    assert got is not None
    assert (got[0], got[1], got[2]) == expect


@pytest.mark.parametrize(
    "text",
    [
        "140/90 mmHg 为高血压诊断界值",  # 反例：数字是临床阈值，不是章节号
        "2024年修订版",
        "5mg 每日一次",
        "27.5%，其中18～44 岁人群患病率",
    ],
)
def test_match_number_rejects(text: str) -> None:
    assert match_number(text) is None


def test_cn2int() -> None:
    assert cn2int("三") == 3
    assert cn2int("十一") == 11
    assert cn2int("二十") == 20


def test_build_sections_root_and_page_end(make_block) -> None:
    """level=0 虚拟 root 必须存在；子节的 page_end 在栈弹出时回填。"""
    blocks = [
        make_block(page=1, block_index=0, text="一、前言", size=16.0),
        make_block(page=1, block_index=1, text="正文甲" * 20, size=10.0),
        make_block(page=3, block_index=0, text="二、方法", size=16.0),
        make_block(page=4, block_index=0, text="正文乙" * 20, size=10.0),
    ]
    headings, body_size = detect_headings(blocks)
    assert body_size == pytest.approx(10.0)
    sections = build_sections("doc", headings, blocks)
    assert sections[0].level == 0
    first = next(s for s in sections if s.section_no == "一、")
    assert first.page_start == 1
    assert first.page_end == 2  # 下一标题在第 3 页 -> 回填 3-1
    assert first.section_path[-1].startswith("一、")


def test_detect_repeated_margins_drops_page_numbers() -> None:
    """同键页眉页脚（"— 27 —" 与 "— 28 —"）必须被识别为噪声。"""
    lines = {
        1: [("— 27 —", 10.0, 20.0, 800.0)],
        2: [("— 28 —", 10.0, 20.0, 800.0)],
        3: [("— 29 —", 10.0, 20.0, 800.0)],
    }
    reject = detect_repeated_margins(lines, min_pages=3, min_ratio=0.30)
    assert reject, "重复页脚应被识别"
    assert is_pure_page_number("12") is True
    assert is_pure_page_number("第 12 页") is True


def test_merge_cross_page_joins_broken_sentence(make_block) -> None:
    """上一页结尾无标点、下一页开头是小写续写 -> 合并成一个块。"""
    blocks = [
        make_block(page=1, block_index=0, text="诊断标准为血压持续升高并"),
        make_block(page=2, block_index=0, text="伴随靶器官损害的证据。"),
    ]
    merged = merge_cross_page(blocks)
    assert len(merged) == 1
    assert merged[0].page_end if hasattr(merged[0], "page_end") else merged[0].page == 1
    assert "持续升高并伴随" in merged[0].text
