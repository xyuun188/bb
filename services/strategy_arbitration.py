"""Strategy arbitration contract.

This layer records whether strategy accepts the AI intent before execution
checks. It does not talk to OKX, and it does not hide exchange errors behind
vague wording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.decision_state import DecisionStageStatus


@dataclass(slots=True)
class StrategyArbitrationResult:
    status: str
    reason: str
    data: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.status in {DecisionStageStatus.PASSED, DecisionStageStatus.COMPLETED}


def arbitrate_decision(decision: DecisionOutput) -> StrategyArbitrationResult:
    """Record the strategy view of an AI decision before execution checks."""

    action = decision.action.value
    data = {
        "symbol": decision.symbol,
        "action": action,
        "confidence": float(decision.confidence or 0.0),
        "position_size_pct": float(decision.position_size_pct or 0.0),
        "suggested_leverage": float(decision.suggested_leverage or 1.0),
    }
    if decision.is_entry:
        return StrategyArbitrationResult(
            status=DecisionStageStatus.PASSED,
            reason=(
                "策略仲裁已承认 AI 的开仓意图；后续只允许余额、仓位、行情严重过期、"
                "交易所错误、价格异常等硬问题拦截。"
            ),
            data={**data, "intent": "entry"},
        )
    if decision.is_exit:
        return StrategyArbitrationResult(
            status=DecisionStageStatus.PASSED,
            reason=(
                "策略仲裁已承认 AI 的平仓/减仓意图；后续只做仓位存在、重复订单、"
                "交易所错误和明确保护规则检查。"
            ),
            data={**data, "intent": "exit"},
        )
    return StrategyArbitrationResult(
        status=DecisionStageStatus.SKIPPED,
        reason="AI 最终选择观望，本轮没有交易意图，不进入 OKX 提交流程。",
        data={**data, "intent": "hold"},
    )
