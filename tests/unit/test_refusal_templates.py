"""A8 门禁：17 个模板键渲染均含 DISCLAIMER；partial 提示键齐全（docs/05 §6）。"""

from __future__ import annotations

import re

import pytest

from medevirag.rag.refusal_templates import (
    DISCLAIMER,
    PARTIAL_NOTES,
    TEMPLATES,
    render,
)


def test_seventeen_template_keys():
    assert len(TEMPLATES) == 17


@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_every_template_has_disclaimer(key):
    text = render(key, "测试问题", {"max_year": 2009, "why": "相关性偏低"})
    assert DISCLAIMER in text


@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_templates_have_no_dose_numbers(key):
    """docs §T6：文案带剂量就成了个体化建议，与 R-08 自相矛盾。"""
    text = TEMPLATES[key]("q", {"max_year": 2009, "why": "x"})
    assert not re.search(r"\d+\s*(mg|毫克|克|片)", text), key


def test_refusal_text_budget():
    """拒答文案 ≤200 字（急诊例外，docs §T6）。"""
    for key, fn in TEMPLATES.items():
        text = fn("q", {"max_year": 2009, "why": "x" * 10})
        if key == "emergency":
            continue
        assert len(text) <= 200, (key, len(text))


def test_render_unknown_key_raises():
    with pytest.raises(KeyError):
        render("no_such_key", "q", {})


def test_partial_notes_four_keys():
    assert set(PARTIAL_NOTES) == {"low_confidence", "evidence_unknown",
                                  "stale_degrade", "year_unknown"}
