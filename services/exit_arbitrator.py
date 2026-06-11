"""Explicit exit-policy arbitration.

The arbitrator gives one priority-ordered interpretation of an exit before the
individual guards run.  Guards still own their detailed checks; this component
decides which ordinary protections should yield to stronger exit intents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.exit_intent import ExitIntent, classify_exit_intent
from services.trading_params import DEFAULT_TRADING_PARAMS, ExitArbitrationParams


@dataclass(frozen=True, slots=True)
class ExitArbitrationResult:
    intent: ExitIntent
    priority: int
    bypass_partial_guard: bool = False
    bypass_cooldown: bool = False
    bypass_profit_precheck: bool = False
    bypass_fee_churn_guard: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "priority": self.priority,
            "bypass_partial_guard": self.bypass_partial_guard,
            "bypass_cooldown": self.bypass_cooldown,
            "bypass_profit_precheck": self.bypass_profit_precheck,
            "bypass_fee_churn_guard": self.bypass_fee_churn_guard,
            "reason": self.reason,
        }


class ExitArbitrator:
    """Resolve exit intent into ordered guard behavior."""

    def __init__(
        self,
        params: ExitArbitrationParams | None = None,
    ) -> None:
        self.params = params or DEFAULT_TRADING_PARAMS.exit_arbitration

    def arbitrate(self, decision: DecisionOutput) -> ExitArbitrationResult:
        intent = classify_exit_intent(decision)
        if intent == ExitIntent.HARD_RISK:
            result = ExitArbitrationResult(
                intent=intent,
                priority=self.params.hard_risk_priority,
                bypass_partial_guard=True,
                bypass_cooldown=True,
                bypass_profit_precheck=True,
                bypass_fee_churn_guard=True,
                reason="hard risk exits have top priority",
            )
        elif intent == ExitIntent.TREND_FAILURE:
            result = ExitArbitrationResult(
                intent=intent,
                priority=self.params.trend_failure_priority,
                bypass_partial_guard=True,
                bypass_cooldown=True,
                bypass_profit_precheck=True,
                bypass_fee_churn_guard=True,
                reason="trend failure exits outrank ordinary churn protection",
            )
        elif intent == ExitIntent.PREDICTIVE_DOWNSIDE:
            result = ExitArbitrationResult(
                intent=intent,
                priority=self.params.predictive_downside_priority,
                bypass_partial_guard=True,
                bypass_cooldown=True,
                bypass_profit_precheck=True,
                bypass_fee_churn_guard=True,
                reason="predictive downside exits protect capital before loss expands",
            )
        elif intent == ExitIntent.PROFIT_DRAWDOWN:
            result = ExitArbitrationResult(
                intent=intent,
                priority=self.params.profit_drawdown_priority,
                bypass_partial_guard=True,
                bypass_cooldown=True,
                bypass_profit_precheck=True,
                bypass_fee_churn_guard=True,
                reason="profit drawdown exits outrank ordinary profit-lock guards",
            )
        elif intent == ExitIntent.PROFIT_PROTECTION:
            result = ExitArbitrationResult(
                intent=intent,
                priority=self.params.profit_protection_priority,
                reason="ordinary profit protection must still pass profit and fee guards",
            )
        elif intent == ExitIntent.CAPITAL_ROTATION:
            result = ExitArbitrationResult(
                intent=intent,
                priority=self.params.capital_rotation_priority,
                reason="capital rotation remains subject to ordinary churn guards",
            )
        elif intent == ExitIntent.LOSS_REPAIR:
            result = ExitArbitrationResult(
                intent=intent,
                priority=self.params.loss_repair_priority,
                reason="loss repair remains subject to aggregate-position protection",
            )
        else:
            result = ExitArbitrationResult(
                intent=intent,
                priority=self.params.ordinary_priority,
                reason="ordinary exit uses the full guard chain",
            )

        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["exit_arbitration"] = result.to_dict()
        decision.raw_response = raw
        return result
