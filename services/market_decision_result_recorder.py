"""Result payload helpers for market-analysis decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.entry_funnel_diagnostics import classify_entry_funnel_reason


def _action_value(decision_or_action: DecisionOutput | str) -> str:
    if isinstance(decision_or_action, DecisionOutput):
        return str(getattr(decision_or_action.action, "value", decision_or_action.action))
    return str(decision_or_action)


@dataclass(frozen=True, slots=True)
class MarketDecisionResultRecorder:
    """Append standardized market-analysis result rows."""

    def append_result(
        self,
        *,
        results: dict[str, Any],
        model_name: str,
        symbol: str,
        decision_or_action: DecisionOutput | str,
        model_mode: str,
        approved: bool = True,
        executed: bool = False,
        execution_status: str = "skipped",
        reason: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        if confidence is None and isinstance(decision_or_action, DecisionOutput):
            confidence = decision_or_action.confidence
        row: dict[str, Any] = {
            "model": model_name,
            "symbol": symbol,
            "action": _action_value(decision_or_action),
            "approved": approved,
            "executed": executed,
            "execution_status": execution_status,
            "reason": reason,
            "is_paper": model_mode == "paper",
        }
        if confidence is not None:
            row["confidence"] = confidence
        if isinstance(decision_or_action, DecisionOutput):
            raw = decision_or_action.raw_response if isinstance(decision_or_action.raw_response, dict) else {}
            funnel_reason = classify_entry_funnel_reason(
                raw=raw,
                action=row["action"],
                was_executed=bool(executed),
                has_order=execution_status not in {"skipped", "hold", "fast_prefilter"},
                reason=reason or decision_or_action.reasoning,
            )
            if funnel_reason:
                row["funnel_reason"] = funnel_reason
        results["decisions"].append(row)
        return row
