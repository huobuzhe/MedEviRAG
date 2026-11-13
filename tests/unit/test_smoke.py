"""冒烟：包能导入、配置能加载、doctor 可调用。"""

from __future__ import annotations

import medevirag
from medevirag.common.config import get_settings


def test_version_present() -> None:
    assert medevirag.__version__


def test_settings_load() -> None:
    cfg = get_settings()
    assert cfg.app.name == "medevirag"
    assert cfg.retrieval.topk == 10


def test_config_hash_stable() -> None:
    cfg = get_settings()
    assert cfg.config_hash() == cfg.config_hash()
    assert cfg.config_hash().startswith("sha256:")


def test_subpackages_importable() -> None:
    import medevirag.common
    import medevirag.eval
    import medevirag.index
    import medevirag.ingest
    import medevirag.rag
    import medevirag.serve

    assert all([medevirag.common, medevirag.eval, medevirag.index])


def test_doctor_returns_zero() -> None:
    from medevirag.cli import doctor

    assert doctor(as_json=True) == 0
