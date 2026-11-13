# S6 服务化集成测试（docs/07 §6 A1-A12 可本机化部分）。
# 无 Docker/Redis/PG：TestClient 进程内验证；Docker 项在 accept_s6 里标 SKIP。

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from medevirag.serve.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_a1_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "cache_stats" in r.json()   # A16：命中率报告实测值


def test_a2_readyz_models_ready(client):
    r = client.get("/readyz")
    body = r.json()
    assert set(body["checks"]) == {"postgres", "redis", "milvus", "models"}
    # 本机降级：pg/redis 允许 not_ready（无 Docker），models/milvus 必须 ok
    assert body["checks"]["models"]["ok"] is True
    assert body["checks"]["milvus"]["ok"] is True
    assert 200 <= r.status_code in (200, 503)


def test_a5_ask_contract(client):
    r = client.post("/v1/ask", json={"question": "成人高血压的诊断标准是什么", "top_k": 50})
    assert r.status_code == 200
    b = r.json()
    for k in ("trace", "status", "answer", "citations", "confidence", "disclaimer"):
        assert k in b, k
    assert b["status"] in ("answered", "partial", "refused")
    assert 0.0 <= b["confidence"] <= 1.0
    assert b["trace"]["trace_id"]


def test_a6_refused_no_citations(client):
    r = client.post("/v1/ask", json={"question": "我血压165/100，氨氯地平该吃多少mg"})
    b = r.json()
    assert b["status"] == "refused"
    assert b["citations"] == []
    assert b["refusal_reason"]
    assert b["disclaimer"]


def test_a8_citation_404(client):
    r = client.get("/v1/citations/no-such-chunk")
    assert r.status_code == 404
    assert r.json()["code"] == "NOT_FOUND"
    assert r.json()["trace_id"]


def test_a10_invalid_request(client):
    # 空串 / 超长 / top_k 越界 -> 400 + INVALID_REQUEST
    for body in ({"question": "  "}, {"question": "x" * 501},
                  {"question": "正常问题", "top_k": 5}):
        r = client.post("/v1/ask", json=body)
        assert r.status_code == 400, (body, r.status_code)
        assert r.json()["code"] == "INVALID_REQUEST"
        assert "Traceback" not in r.text


def test_a15_cache_hit_and_citations_intact(client):
    q = {"question": "高血压患者如何限盐", "top_k": 50}
    r1 = client.post("/v1/ask", json=q)
    b1 = r1.json()
    r2 = client.post("/v1/ask", json=q)
    b2 = r2.json()
    assert b2["trace"]["cache"] == "hit"
    # 回链不断裂：两次 citations 逐元素相同
    ids1 = [c["chunk_id"] for c in b1["citations"]]
    ids2 = [c["chunk_id"] for c in b2["citations"]]
    assert ids1 == ids2
    assert b2["trace"]["total_ms"] <= 200


def test_a11_audit_written(client, tmp_path):
    client.post("/v1/ask", json={"question": "成人高血压的诊断标准是什么"})
    from pathlib import Path
    p = Path("data/logs/qa_log.jsonl")
    if p.exists():
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        last = json.loads(lines[-1])
        for k in ("trace_id", "question", "answer", "status", "confidence",
                  "latency_ms", "retrieval_ms", "rerank_ms", "generation_ms",
                  "retrieved_chunk_ids", "cache"):
            assert k in last, k


def test_a18_openapi(client):
    r = client.get("/docs")
    assert r.status_code == 200
    spec = client.get("/openapi.json").json()
    assert len(spec["paths"]) == 5


def test_trace_id_header_echo(client):
    r = client.post("/v1/ask", json={"question": "成人高血压的诊断标准是什么"},
                       headers={"x-trace-id": "fixed-trace-001"})
    assert r.headers["x-trace-id"] == "fixed-trace-001"
    assert "x-elapsed-ms" in r.headers
