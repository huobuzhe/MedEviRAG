"""ingest 单测的共用夹具：**零网络、零 PG、零模型**。

token 计数用"每字符 1 token"的假计数器，只为让 chunker 的长度逻辑可断言；
真实 bge-m3 计数在端到端脚本里验证（模型加载 3s，不应拖进单测）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from medevirag.ingest.parser import LayoutBlock
from medevirag.ingest.sectionizer import Section

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture()
def fake_count() -> Callable[[str], int]:
    """1 字符 = 1 token，便于直接按字符数推 token_len。"""

    def count(s: str) -> int:
        return len(s)

    return count


@pytest.fixture()
def make_block() -> Callable[..., LayoutBlock]:
    def _make(
        page: int = 1,
        block_index: int = 0,
        text: str = "正文",
        kind: str = "body",
        size: float = 10.0,
        bold: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> LayoutBlock:
        return LayoutBlock(
            page=page,
            block_index=block_index,
            kind=kind,
            text=text,
            bbox=bbox or (10.0, 20.0, 500.0, 60.0),
            size=size,
            font="Song",
            bold=bold,
            n_chars=len(text),
        )

    return _make


@pytest.fixture()
def make_section() -> Callable[..., Section]:
    def _make(
        section_no: str = "3.2.1",
        title: str = "诊室血压",
        level: int = 3,
        section_path: list[str] | None = None,
        page_start: int = 1,
        page_end: int = 1,
    ) -> Section:
        path = section_path if section_path is not None else ["三、诊断", section_no + " " + title]
        return Section(
            section_id="doc#" + (section_no or "root"),
            doc_id="doc",
            section_no=section_no,
            title=title,
            level=level,
            page_start=page_start,
            page_end=page_end,
            order_index=1,
            parent_section_id="doc#root",
            section_path=path,
        )

    return _make
