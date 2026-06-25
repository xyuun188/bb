"""Forced-exit classification policy.

Hard-risk exits are allowed to bypass several ordinary safety delays.  Keeping
the classifier here avoids scattering keyword and raw-payload checks across the
main trading orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.exit_intent import is_low_quality_release_without_hard_risk

FORCED_EXIT_UPPER_TERMS = (
    "STOP LOSS",
    "BLACK SWAN",
    "HARD STOP",
    "CRITICAL",
)
FORCED_EXIT_TEXT_TERMS = (
    "强制平仓",
    "硬止损",
    "极端风险",
    "黑天鹅",
    "熔断",
    "触发止损",
    "触发止盈",
    "止损触发",
    "止盈触发",
    "快速风控",
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(slots=True)
class ForcedExitPolicy:
    """Detect hard-risk exit decisions from structured flags and reason text."""

    def is_forced_exit(self, decision: DecisionOutput) -> bool:
        text = str(decision.reasoning or "").upper()
        raw_text = str(decision.reasoning or "")
        raw = _safe_dict(decision.raw_response)
        close_evidence = _safe_dict(raw.get("close_evidence"))
        position_review_alert = _safe_dict(raw.get("position_review_risk_alert"))
        if is_low_quality_release_without_hard_risk(raw):
            return False
        return (
            bool(raw.get("fast_risk_exit") or raw.get("forced_exit"))
            or bool(close_evidence.get("hard_risk") or close_evidence.get("forced_exit"))
            or bool(position_review_alert.get("force_exit"))
            or decision.model_name == "risk_engine"
            or any(term in text for term in FORCED_EXIT_UPPER_TERMS)
            or any(term in raw_text for term in FORCED_EXIT_TEXT_TERMS)
        )
