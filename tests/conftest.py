"""pytest 全局夹具（docs/01 §T9）。

三条纪律写进夹具，而不是写进每个测试文件：
1. 默认不联网：任何用例想调真实 API 必须显式声明 network 标记；
2. docker 用例自动跳过：CI 与开发机都不需要 daemon；
3. FakeLLM 由夹具提供，保证用到桩的用例永远不烧 token。
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator

import pytest

from medevirag.common.llm import FakeLLM


def _docker_available() -> bool:
    """只检查二进制在不在，不启动 daemon：慢且可能挂住。"""
    return shutil.which("docker") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """没有 docker 就把 docker 标记的用例标成 skip，而不是让它们失败。"""
    if _docker_available():
        return
    skip_docker = pytest.mark.skip(reason="docker 不可用")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip_docker)


@pytest.fixture()
def fake_llm() -> FakeLLM:
    """默认剧本的确定性桩。"""
    return FakeLLM(mode="normal")


@pytest.fixture()
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """断网夹具：禁止真实 socket 连接，断言用例不偷偷联网。"""
    import socket

    def _blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError("测试禁止真实网络访问")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    yield
