"""Build HOLD records for position groups deferred from governed review."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class FastScanHoldPlan:
    reason: str
    raw_response: dict[str, Any]
    defer_count: int


@dataclass(frozen=True, slots=True)
class PositionReviewFastScanHoldPolicy:
    """Record deferred review without creating score-based permissions."""

    clock: Callable[[], datetime] = lambda: datetime.now(UTC)

    def plan(
        self,
        scan: dict[str, Any],
        *,
        previous_defer_count: int,
        urgent_exit: bool,
        portfolio_symbol_context: dict[str, Any] | None,
        agent_skill_dicts: list[dict[str, Any]],
        agent_skill_summary: dict[str, Any],
    ) -> FastScanHoldPlan:
        dynamic_exit = scan.get("dynamic_exit_policy")
        dynamic_exit = dynamic_exit if isinstance(dynamic_exit, dict) else {}
        eligible = bool(scan.get("dynamic_exit_eligible") is True and dynamic_exit.get("eligible") is True)
        close_fraction = _safe_float(dynamic_exit.get("close_fraction"), 0.0)
        defer_count = int(previous_defer_count or 0) + 1 if eligible or urgent_exit else 0
        reason = (
            "Governed dynamic exit review was deferred by the current model-call budget."
            if eligible
            else "No governed dynamic exit permission was present in the fast scan."
        )
        scan_reason = str(scan.get("reason") or "")
        if scan_reason:
            reason += f" Reason: {scan_reason}."

        portfolio_context = (
            portfolio_symbol_context if isinstance(portfolio_symbol_context, dict) else {}
        )
        raw_response: dict[str, Any] = {
            "analysis_type": "position_review",
            "position_fast_scan": {
                "skipped_llm": True,
                "dynamic_exit_eligible": eligible,
                "close_fraction": round(close_fraction, 8),
                "reason": scan_reason,
                "production_permission": False,
            },
            "agent_skills": {
                "version": 1,
                "phases": {
                    "position_fast_scan": {
                        "phase": "position_fast_scan",
                        "recorded_at": self.clock().isoformat(),
                        "note": "Resource deferral only; no trade permission is created.",
                        "skills": agent_skill_dicts,
                    }
                },
                "summary": agent_skill_summary,
            },
        }
        if portfolio_context:
            raw_response["portfolio_profit_observation"] = {
                **portfolio_context,
                "production_permission": False,
            }
        return FastScanHoldPlan(reason, raw_response, defer_count)
