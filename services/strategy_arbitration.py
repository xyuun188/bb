"""Lightweight strategy arbitration contract.

This first pass does not rewrite trading thresholds.  It records whether the
strategy layer accepts the AI's intent before execution-layer checks run.
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
    """Record the policy view of an AI decision without changing it."""

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
                "策略仲裁已承认 AI 的开仓意图；后续只允许余额、仓位不存在、"
                "行情严重过期、交易所错误等硬问题拦截。"
            ),
            data={**data, "intent": "entry"},
        )
    if decision.is_exit:
        return StrategyArbitrationResult(
            status=DecisionStageStatus.PASSED,
            reason=(
                "策略仲裁已承认 AI 的平仓/减仓意图；后续风控只做硬错误和重复订单保护。"
            ),
            data={**data, "intent": "exit"},
        )
    return StrategyArbitrationResult(
        status=DecisionStageStatus.SKIPPED,
        reason="AI 最终选择观望，本轮没有交易意图，不进入 OKX 提交流程。",
        data={**data, "intent": "hold"},
    )
