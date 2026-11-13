"""A3：Milvus Lite 端到端检索（无 Docker）。

用 tests/fixtures/chunks_small.jsonl 的 300 条真实 chunk，在 tmp_path 建库，
向量用 FakeEmbedder（md5 播种的正态采样，确定性、零模型、零网络），
验证：建集合 → 插入 → load → search 全链路可用，且单次 search < 20ms。

注意：本用例**不用 pytest 标记**。pyproject 的 --strict-markers 只注册了
docker/gpu/slow/network，`integration` 标记未注册会直接被拒；
而带上 docker 标记又会被 conftest 判为"docker 可用"后真去连 daemon。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from medevirag.index.milvus_store import (
    build_filter_expr,
    ensure_collection,
    insert_chunks,
    open_milvus_client,
    search_dense,
)

DIM = 1024
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "chunks_small.jsonl"


def fake_vec(text: str, dim: int = DIM) -> np.ndarray:
    """md5 播种的正态采样 + L2 归一化：同文本必得同向量。"""
    seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / float(np.linalg.norm(vec))


def load_fixture() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with FIXTURE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def lite_index(tmp_path_factory: pytest.TempPathFactory) -> tuple[object, list[dict[str, object]]]:
    """module 级：300 条建库约 4s，不值得每个用例重建（tmp_path_factory 提供独立目录）。"""
    tmp_path = tmp_path_factory.mktemp("lite_index")
    rows = load_fixture()
    client = open_milvus_client(str(tmp_path / "lite.db"))
    ensure_collection(client, "chunks", dense_dim=DIM)
    ids = [str(r["chunk_id"]) for r in rows]
    vectors = np.stack([fake_vec(str(r["text"])) for r in rows])
    inserted = insert_chunks(client, ids, vectors, rows, collection="chunks", batch=200)
    assert inserted == len(rows)
    client.load_collection("chunks")
    return client, rows


def test_fixture_has_300_real_chunks() -> None:
    rows = load_fixture()
    assert len(rows) == 300
    for row in rows[:5]:
        for key in ("chunk_id", "doc_id", "doc_type", "text", "lang", "publish_year"):
            assert key in row, key
        assert str(row["text"]).strip()


def test_inserted_row_count_matches_fixture(lite_index: tuple[object, list[dict[str, object]]]) -> None:
    client, rows = lite_index
    stats = client.get_collection_stats("chunks")  # type: ignore[attr-defined]
    assert int(stats["row_count"]) == len(rows)


def test_search_returns_self_as_top1(lite_index: tuple[object, list[dict[str, object]]]) -> None:
    client, rows = lite_index
    for row in rows[:10]:
        vec = [float(x) for x in fake_vec(str(row["text"]))]
        hits = search_dense(client, vec, topk=10, collection="chunks", ef=64)
        assert len(hits) == 10
        assert hits[0].chunk_id == str(row["chunk_id"])
        assert hits[0].score > 0.999


def test_single_search_under_20ms(lite_index: tuple[object, list[dict[str, object]]]) -> None:
    """A3 的硬指标：单次 search < 20 ms（300 条 fixture，warm 后 30 次取 P50）。"""
    client, rows = lite_index
    queries = [[float(x) for x in fake_vec(str(r["text"]))] for r in rows[:30]]
    for vec in queries:  # warm
        search_dense(client, vec, topk=10, collection="chunks", ef=64)
    timings: list[float] = []
    for vec in queries:
        start = time.perf_counter()
        search_dense(client, vec, topk=10, collection="chunks", ef=64)
        timings.append((time.perf_counter() - start) * 1000.0)
    p50 = float(np.percentile(timings, 50))
    assert p50 < 20.0, f"P50 search latency {p50:.2f} ms >= 20 ms"
    print(f"\nA3 search latency: n={len(timings)} p50={p50:.2f}ms "
          f"p95={float(np.percentile(timings, 95)):.2f}ms max={max(timings):.2f}ms")


def test_filter_expr_narrows_candidates(lite_index: tuple[object, list[dict[str, object]]]) -> None:
    """元数据硬过滤在 Milvus 侧也能用（pre-filter 版），确认 expr 语法与字段类型正确。"""
    client, rows = lite_index
    expr = build_filter_expr(
        doc_type_allow=("guideline",),
        lang_allow=("zh",),
        publish_year_min=2015,
        evidence_level_exclude=("D",),
    )
    assert "doc_type in" in expr
    vec = [float(x) for x in fake_vec(str(rows[0]["text"]))]
    hits = search_dense(client, vec, topk=50, expr=expr, collection="chunks", ef=64)
    assert hits, "filtered search returned nothing"
    allowed = {str(r["chunk_id"]) for r in rows if str(r["doc_type"]) == "guideline"}
    assert {h.chunk_id for h in hits} <= allowed


def test_filtered_search_can_be_empty_not_error(lite_index: tuple[object, list[dict[str, object]]]) -> None:
    """窄条件打到 0 条不应抛异常（这正是 ADR-03 选 post-filter 的理由之一）。"""
    client, rows = lite_index
    expr = build_filter_expr(doc_type_allow=("nonexistent-type",))
    vec = [float(x) for x in fake_vec(str(rows[0]["text"]))]
    hits = search_dense(client, vec, topk=10, expr=expr, collection="chunks", ef=64)
    assert hits == []


def test_ef_is_accepted_on_lite(lite_index: tuple[object, list[dict[str, object]]]) -> None:
    """Lite 接受 ef 参数（但按 A9 的结论，它不改变结果——这里只保证不报错）。"""
    client, rows = lite_index
    vec = [float(x) for x in fake_vec(str(rows[0]["text"]))]
    for ef in (16, 64, 128):
        hits = search_dense(client, vec, topk=10, collection="chunks", ef=ef)
        assert len(hits) == 10
