"""token 预算护栏（docs/01 §T6，00 §13.1 第 3 条）。

为什么是进程级单例而不是每次请求 new 一个：预算是"整个开发周期"口径，
S4 的拒答链路与 S5 的评测脚本必须共享同一个计数器，否则各算各的等于没算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medevirag.common.errors import BudgetExceeded
from medevirag.common.logging import get_logger

log = get_logger("medevirag.budget")


@dataclass
class TokenLedger:
    """累计 token 用量；超硬上限抛 BudgetExceeded（主动熔断，不是故障）。"""

    total: int = 0
    calls: int = 0
    per_node: dict[str, int] = field(default_factory=dict)

    def add(self, usage: dict[str, Any], *, node: str, warn_at: int, hard_at: int) -> None:
        tokens = int(usage.get("total_tokens", 0))
        self.total += tokens
        self.calls += 1
        self.per_node[node] = self.per_node.get(node, 0) + tokens
        if self.total > hard_at:
            raise BudgetExceeded(
                f"{node}: 累计 {self.total} > 硬上限 {hard_at}",
                node=node,
                total=self.total,
                calls=self.calls,
            )
        if self.total > warn_at:
            log.warning(
                "token_budget_warning",
                extra={"node": node, "total": self.total, "calls": self.calls, "warn_at": warn_at},
            )

    def reset(self) -> None:
        self.total = 0
        self.calls = 0
        self.per_node.clear()


LEDGER = TokenLedger()
"""进程级单例；CI 断言 LEDGER.calls == 0 以证明全程没调过真实模型。"""
