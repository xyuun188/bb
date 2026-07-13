"""Classify governed planned-stop exits that may bypass ordinary delays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(slots=True)
class ForcedExitPolicy:
    """Accept only the unified dynamic policy's current planned-stop fact."""

    def is_forced_exit(self, decision: DecisionOutput) -> bool:
        raw = _safe_dict(decision.raw_response)
        policy = _safe_dict(raw.get("dynamic_exit_policy"))
        provenance = _safe_dict(policy.get("policy_provenance"))
        return bool(
            policy.get("eligible") is True
            and policy.get("hard_risk") is True
            and policy.get("planned_stop_crossed") is True
            and provenance.get("source")
            == "current_position_fee_after_pnl_peak_planned_stop_and_market_returns"
        )
