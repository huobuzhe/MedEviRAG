"""T9 FakeLLM 剧本（docs/04 §T9）。

六个剧本字符串，供 verifier / generator / graph 单测注入，覆盖：
非法引用、无引用、支撑不足、代码围栏、证据不足、连续两轮非法。
"""

from __future__ import annotations

ILLEGAL_CITATION = '{"answer":"诊室血压≥140/90 mmHg 可诊断高血压 [C9]。推荐非同日三次测量 [C1]。","used_citations":["C1","C9"],"insufficient_evidence":false}'
NO_CITATION = '{"answer":"诊室血压≥140/90 mmHg 可诊断高血压。","used_citations":[],"insufficient_evidence":false}'
PARTIAL_SUPPORT = '{"answer":"推荐限盐每日 3g 以下 [C1]。","used_citations":["C1"],"insufficient_evidence":false}'
FENCED = '\u0060\u0060\u0060json\n{"answer":"诊室血压≥140/90 mmHg 可诊断高血压 [C1]","used_citations":["C1"],"insufficient_evidence":false}\n\u0060\u0060\u0060'
INSUFFICIENT = '{"answer":"现有证据不足以回答该问题","used_citations":[],"insufficient_evidence":true}'
ALL_ILLEGAL_TWICE = [ILLEGAL_CITATION, ILLEGAL_CITATION]
