"""T7/A7/A8 门禁：落点校验、句级支撑、降级策略、重试硬上限。"""

from __future__ import annotations

from medevirag.rag.citations import CITE_RE
from medevirag.rag.verifier import (
    MIN_ANSWER_CHARS,
    LexicalJudge,
    check_sentences,
    verify,
)

EV = {
    "C1": {
        "chunk_id": "d#s#c01",
        "text": "推荐非同日三次测量诊室血压，收缩压≥140 mmHg 和/或舒张压≥90 mmHg 可诊断高血压。",
    }
}


def test_landing_check_flags_illegal_marker():
    judge = LexicalJudge()
    res = verify("诊室血压≥140/90 mmHg 可诊断高血压 [C9]。", EV, judge)
    assert "C9" in res.illegal_markers
    assert res.landing_ok is False


def test_illegal_only_sentence_is_deleted():
    judge = LexicalJudge()
    res = verify("完全无关的一句话 [C7]。另一种说法 [C8]。", EV, judge)
    assert res.cleaned_answer == ""
    assert res.must_retry is True


def test_partial_illegal_keeps_sentence_and_strips_marker():
    judge = LexicalJudge()
    ans = "推荐非同日三次测量诊室血压 [C1]，收缩压≥140 mmHg 可诊断高血压 [C9]。"
    res = verify(ans, EV, judge)
    assert "C9" not in res.cleaned_answer
    assert "C1" in res.cleaned_answer
    assert not CITE_RE.search(res.cleaned_answer.replace("[C1]", ""))
    assert res.illegal_markers == ["C9"]


def test_illegal_citation_appears_zero_times_after_cleanup():
    judge = LexicalJudge()
    ans = "诊室血压≥140/90 mmHg 可诊断高血压 [C9]。"
    res = verify(ans, EV, judge)
    assert "[C9]" not in res.cleaned_answer
    assert res.cleaned_answer.count("C9") == 0


def test_multi_sentence_deletion_over_half_triggers_retry():
    judge = LexicalJudge()
    ans = "第一句非法 [C7]。第二句合法且足够长以超过二十字符推荐非同日三次测量诊室血压 [C1]。第三句非法 [C8]。第四句非法 [C9]。"
    res = verify(ans, EV, judge)
    assert res.must_retry is True
    assert res.cleaned_answer == ""


def test_short_answer_after_cleanup_triggers_retry():
    judge = LexicalJudge()
    res = verify("短答 [C1]。另一段非法内容 [C7]。另一段非法内容 [C8]。", EV, judge)
    assert res.cleaned_answer == "" and res.must_retry is True


def test_support_rate_in_range_and_unsupported_reported():
    judge = LexicalJudge()
    res = verify("推荐每日饮用咖啡三杯可降低血压 [C1]。", EV, judge)
    assert 0.0 <= res.support_rate <= 1.0
    assert res.support_rate < 0.5
    assert res.unsupported


def test_check_sentences_skips_sentences_without_markers():
    judge = LexicalJudge()
    checks = check_sentences("没有引用的句子。有引用 [C1] 的句子。", EV, judge)
    assert len(checks) == 1 and checks[0].markers == ["C1"]


def test_min_answer_chars_constant():
    assert MIN_ANSWER_CHARS == 20
