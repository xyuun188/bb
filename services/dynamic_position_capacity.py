"""Read-only observation of current position-group usage.

Position count is not a production strategy gate. Entry permission and sizing are
owned by the fee-after-return contract and current account risk budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class DynamicCapacityDecision:
    hard_limit: None
    entry_limit: None
    open_group_count: int
    available_group_slots: None
    reason: str
    policy_provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hard_limit": self.hard_limit,
            "entry_limit": self.entry_limit,
            "open_group_count": self.open_group_count,
            "available_group_slots": self.available_group_slots,
            "reason": self.reason,
            "policy_provenance": self.policy_provenance,
        }


class DynamicPositionCapacityPolicy:
    """Report position groups without granting or denying production entries."""

    def evaluate(
        self,
        *,
        open_positions: list[dict[str, Any]],
        strategy_context: dict[str, Any] | None = None,
        market_regime: dict[str, Any] | None = None,
        account_equity: float | None = None,
        active_strategy_profile_id: str | None = None,
    ) -> DynamicCapacityDecision:
        del strategy_context, market_regime, active_strategy_profile_id
        open_groups = self._open_group_count(open_positions)
        generated_at = datetime.now(UTC).isoformat()
        provenance = {
            "source": "current_open_position_group_observation",
            "observation_window": "current_open_position_snapshot",
            "sample_count": open_groups,
            "generated_at": generated_at,
            "strategy_version": "2026-07-12.position-count-observation.v1",
            "fallback_reason": "" if account_equity is not None and account_equity > 0 else "account_equity_missing",
            "production_eligible": False,
            "production_permission": False,
        }
        return DynamicCapacityDecision(
            hard_limit=None,
            entry_limit=None,
            open_group_count=open_groups,
            available_group_slots=None,
            reason="position count is observation-only; dynamic risk sizing owns exposure",
            policy_provenance=provenance,
        )

    @staticmethod
    def _open_group_count(positions: list[dict[str, Any]]) -> int:
        groups = {
            (
                str(position.get("symbol") or "").upper(),
                str(position.get("side") or "").lower(),
            )
            for position in positions or []
            if isinstance(position, dict)
            and position.get("is_open", True) is not False
            and str(position.get("symbol") or "")
            and str(position.get("side") or "").lower() in {"long", "short"}
        }
        return len(groups)
