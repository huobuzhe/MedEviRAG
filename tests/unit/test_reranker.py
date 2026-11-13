"""A1 门禁：重排器确定性（docs/04 §T1/A1）。

真模型加载约 1.8s，这里只跑一次然后复用；两次 rerank 结果必须逐位相同且逆序对为 0。
"""

from __future__ import annotations

import pytest

from medevirag.index.fusion import RetrievedChunk
from medevirag.rag.reranker import Reranker


def _chunks():
    texts = [
        "诊室血压≥140/90 mmHg 可诊断高血压。",
        "推荐非同日三次测量诊室血压，取平均值。",
        "高血压患者应限制钠盐摄入，每日不超过 5g。",
        "本段与高血压诊断标准无关，讨论的是血糖管理。",
    ]
    return [
        RetrievedChunk(chunk_id=f"d#s#c0{i}", doc_id="d", text=t, context_header="《指南》",
                       fused_rank=i + 1)
        for i, t in enumerate(texts)
    ]


@pytest.fixture(scope="module")
def reranker():
    return Reranker(batch_size=8, max_length=512)


def _inversions(scores: list[float]) -> int:
    return sum(1 for i in range(len(scores)) for j in range(i + 1, len(scores))
               if scores[i] < scores[j])


def test_rerank_deterministic(reranker):
    chunks = _chunks()
    a = reranker.rerank("诊室血压多少可以诊断高血压？", chunks, top_n=4)
    b = reranker.rerank("诊室血压多少可以诊断高血压？", chunks, top_n=4)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [round(c.rerank_score, 6) for c in a] == [round(c.rerank_score, 6) for c in b]


def test_rerank_scores_descending_no_inversions(reranker):
    chunks = _chunks()
    out = reranker.rerank("诊室血压多少可以诊断高血压？", chunks, top_n=4)
    scores = [c.rerank_score for c in out]
    assert _inversions(scores) == 0
    assert [c.rerank_rank for c in out] == list(range(1, len(out) + 1))


def test_rerank_respects_top_n(reranker):
    out = reranker.rerank("高血压", _chunks(), top_n=2)
    assert len(out) == 2


def test_default_device_honours_env_override(monkeypatch):
    """MEDEVIRAG_DEVICE 必须能强制降级到 cpu（docs/04 §11.5 D8）。

    为什么要有这条断言：并发跑多个进程时每个进程都会建一份 CUDA 上下文，
    Windows 下 torch 前向会 access violation —— 表现为「汇总行 passed 但退出码非 0」，
    曾把 A8/A10 误判成 FAIL。这个开关是验收脚本稳定性的支点，不能被无意改掉。
    """
    from medevirag.rag import reranker as R

    monkeypatch.setenv("MEDEVIRAG_DEVICE", "cpu")
    assert R._default_device() == "cpu"

    monkeypatch.setenv("MEDEVIRAG_DEVICE", "")
    import torch

    assert R._default_device() == ("cuda" if torch.cuda.is_available() else "cpu")


def test_embedder_device_forced_cpu(monkeypatch):
    """embedder 的设备覆盖与 reranker 同口径（D8 另一半）。"""

    from medevirag.index.embedder import Embedder

    monkeypatch.setenv("MEDEVIRAG_DEVICE", "cpu")
    emb = Embedder("models/bge-m3", device="cuda")
    assert emb.device == "cpu"
    assert emb.fp16_active is False, "cpu 上不得启用 fp16"
