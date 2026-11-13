"""A7/A8/A10/A11 门禁：8 节点图结构、埋点、越界零 LLM、注入非法引用被拦。

图里依赖真实检索（HybridRetriever）+ 本地重排器（默认关闭）/ FakeLLM。
单个用例 1~3s，模块级共享 retriever 与 app 单例。
"""

from __future__ import annotations

import pytest

from medevirag.rag import graph, nodes

EXPECTED_NODES = {"route", "rewrite", "retrieve", "filter", "rerank", "generate", "verify", "decide"}
TRACE_KEYS = {"route", "rewrite", "retrieve", "filter", "rerank", "generate", "verify", "decide"}


class _ScriptedLLM:
    """按剧本依次返回；记录调用次数（A8 要求 ≤2）。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat(self, system, user, *, temperature, max_tokens):
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[idx]


@pytest.fixture(scope="module")
def app():
    return graph.build_graph()


def test_graph_has_exactly_eight_business_nodes(app):
    assert set(graph.BUSINESS_NODES) == EXPECTED_NODES
    assert len(graph.BUSINESS_NODES) == 8
    assert set(app.get_graph().nodes) >= EXPECTED_NODES


def test_trace_covers_all_eight_nodes():
    out = graph.run_ask("诊室血压多少可以诊断高血压？")
    trace = out.get("trace") or {}
    assert set(trace) >= TRACE_KEYS, f"missing trace keys: {TRACE_KEYS - set(trace)}"
    assert all(isinstance(v, int) and v >= 0 for v in trace.values())


def test_out_of_scope_uses_zero_llm_calls(monkeypatch):
    calls = {"n": 0}

    class _Counter:
        def chat(self, system, user, *, temperature, max_tokens):
            calls["n"] += 1
            return "{}"

    monkeypatch.setattr(nodes, "_llm", _Counter())
    # 按 docs/04 §T8 的机械判据，越界 = 无中文 或 长度 < 4
    out = graph.run_ask("What is the capital of France?")
    assert calls["n"] == 0
    assert out["status"] == "refused"
    assert out["refusal_reason"] == "off_domain"   # S4 新枚举（docs/05 §T1）


def test_illegal_citation_injected_is_blocked(monkeypatch):
    from medevirag.rag import verifier as V

    # 强制注入含 [C9] 的剧本；判词走 LexicalJudge（离线）
    scripted = _ScriptedLLM(['{"answer":"诊室血压≥140/90 mmHg 可诊断高血压 [C9]。","used_citations":["C9"],"insufficient_evidence":false}'])
    monkeypatch.setattr(nodes, "_llm", scripted)
    monkeypatch.setattr(nodes, "_judge", V.LexicalJudge())

    out = graph.run_ask("诊室血压多少可以诊断高血压？")
    assert "[C9]" not in out["answer"]
    assert out["answer"].count("C9") == 0
    assert out["status"] in ("partial", "refused")


def test_retry_is_hard_capped_at_two_llm_calls(monkeypatch):
    from medevirag.rag import verifier as V

    # 两轮都返回非法引用 → 图内重试上限 1 次，总调用 ≤2
    bad = '{"answer":"诊室血压≥140/90 mmHg 可诊断高血压 [C9]。","used_citations":["C9"],"insufficient_evidence":false}'
    scripted = _ScriptedLLM([bad, bad])
    monkeypatch.setattr(nodes, "_llm", scripted)
    monkeypatch.setattr(nodes, "_judge", V.LexicalJudge())

    out = graph.run_ask("诊室血压多少可以诊断高血压？")
    assert scripted.calls <= 2
    assert "[C9]" not in out["answer"]
    assert out["status"] == "refused"


def test_insufficient_evidence_refuses_without_retry(monkeypatch):
    from medevirag.rag import verifier as V

    scripted = _ScriptedLLM(['{"answer":"现有证据不足以回答该问题","used_citations":[],"insufficient_evidence":true}'])
    monkeypatch.setattr(nodes, "_llm", scripted)
    monkeypatch.setattr(nodes, "_judge", V.LexicalJudge())

    out = graph.run_ask("诊室血压多少可以诊断高血压？")
    assert scripted.calls == 1
    assert out["status"] == "refused"
    assert out["refusal_reason"] in ("low_relevance", "low_support", "evidence_level_unknown")   # S4：insufficient 落到具体规则或 R-INSUFF


def test_run_ask_returns_disclaimer():
    out = graph.run_ask("诊室血压多少可以诊断高血压？")
    assert out["disclaimer"]
    assert "不构成诊疗建议" in out["disclaimer"]
