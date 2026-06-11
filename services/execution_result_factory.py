"""Factories for local execution-result objects."""

from __future__ import annotations

from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from core.safe_output import safe_error_text
from executor.base_executor import ExecutionResult, OrderStatus


class ExecutionResultFactory:
    """Build local execution results that mirror executor output shape."""

    def rejected(self, decision: DecisionOutput, error: Exception | str) -> ExecutionResult:
        message = safe_error_text(error)
        return ExecutionResult(
            order_id="rejected",
            symbol=decision.symbol,
            side=self.decision_side(decision),
            order_type="market",
            quantity=0.0,
            price=0.0,
            status=OrderStatus.REJECTED,
            raw_response={"error": message},
        )

    @staticmethod
    def decision_side(decision: DecisionOutput) -> str:
        action_value = getattr(decision.action, "value", decision.action)
        return {
            Action.LONG.value: "buy",
            Action.SHORT.value: "sell",
            Action.CLOSE_LONG.value: "sell",
            Action.CLOSE_SHORT.value: "buy",
        }.get(str(action_value), "hold")

    @staticmethod
    def action_label(action: Action | str | None) -> str:
        value: Any = getattr(action, "value", action)
        return {
            Action.LONG.value: "做多",
            Action.SHORT.value: "做空",
            Action.CLOSE_LONG.value: "平多",
            Action.CLOSE_SHORT.value: "平空",
            Action.HOLD.value: "观望",
        }.get(str(value), str(value or "未知"))
