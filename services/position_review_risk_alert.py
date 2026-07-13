"""Observation-only alerts for governed position-review exits."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_brain.base_model import DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus

FloatParser = Callable[[Any, float], float]
TextShortener = Callable[[Any, int], str]
ActionLabeler = Callable[[Any], str]
ExecutionReasonProvider = Callable[[ExecutionResult], str]


class PositionReviewRiskAlertPolicy:
    """Format alerts after the unified dynamic exit contract has passed."""

    def __init__(
        self,
        *,
        float_parser: FloatParser,
        text_shortener: TextShortener,
        action_labeler: ActionLabeler,
    ) -> None:
        self._float_parser = float_parser
        self._text_shortener = text_shortener
        self._action_labeler = action_labeler

    def build_alert(
        self,
        decision: DecisionOutput,
        positions: list[dict[str, Any]],
    ) -> str | None:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        close_evidence = raw.get("close_evidence")
        if not isinstance(close_evidence, dict):
            return None
        dynamic_exit = close_evidence.get("dynamic_exit_policy")
        if (
            not decision.is_exit
            or not isinstance(dynamic_exit, dict)
            or dynamic_exit.get("eligible") is not True
        ):
            return None

        opinions = raw.get("opinions")
        opinions = opinions if isinstance(opinions, list) else []
        risk_opinion = next(
            (
                item
                for item in opinions
                if isinstance(item, dict) and item.get("model_name") == "risk_expert"
            ),
            {},
        )
        risk_action = str(risk_opinion.get("action") or "hold")
        risk_confidence = self._float_parser(risk_opinion.get("confidence"), 0.0)
        risk_reason = str(risk_opinion.get("reasoning") or "")

        position_bits = []
        for position in positions:
            side = str(position.get("side") or "unknown")
            position_bits.append(
                f"{side} entry={position.get('entry_price', '-')} "
                f"quantity={position.get('quantity', '-')} "
                f"unrealized={position.get('unrealized_pnl', 0)}"
            )
        position_text = "; ".join(position_bits) or "no matching position details"
        return (
            f"Governed position exit alert: {decision.symbol}; {position_text}. "
            f"Dynamic action={self._action_labeler(decision.action)}; "
            f"risk expert observation={self._action_labeler(risk_action)} "
            f"({risk_confidence:.0%}); reason={risk_reason or 'not provided'}."
        )

    def attach(self, decision: DecisionOutput, message: str) -> None:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["position_review_risk_alert"] = {
            "message": message,
            "planned_action": decision.action.value,
            "production_permission": False,
        }
        decision.raw_response = raw

    @staticmethod
    def alert_context(decision: DecisionOutput) -> dict[str, Any] | None:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        alert = raw.get("position_review_risk_alert")
        return alert if isinstance(alert, dict) else None

    def execution_result_text(
        self,
        decision: DecisionOutput,
        execution_result: ExecutionResult,
        execution_reason_provider: ExecutionReasonProvider,
    ) -> str:
        if execution_result.status == OrderStatus.FILLED:
            return (
                f"Execution filled: action={self._action_labeler(decision.action)}, "
                f"quantity={execution_result.quantity:g}, price={execution_result.price:g}, "
                f"status={execution_result.status.value}."
            )
        return (
            f"Execution incomplete: action={self._action_labeler(decision.action)}, "
            f"status={execution_result.status.value}, "
            f"reason={execution_reason_provider(execution_result)}"
        )

    def risk_event_detail(
        self,
        decision: DecisionOutput,
        alert: dict[str, Any],
        result_text: str | None,
    ) -> str:
        return (
            f"{alert.get('message')} System action={self._action_labeler(decision.action)}. "
            f"Execution result={result_text or 'not available'}"
        )
