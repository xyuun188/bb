"""Exit position snapshot boundary.

ExitPolicy decides whether an exit may proceed.  This helper owns the exchange
snapshot refresh calls needed before that decision, keeping direct OKX sync
service calls out of the policy gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput


@dataclass(slots=True)
class ExitPositionSnapshotPolicy:
    """Refresh local exit context without running heavy full reconciliation per exit."""

    sync_service: Any
    reconcile_reason: str = "exit precheck"
    reconcile_timeout_seconds: float | None = None

    async def refresh_positions(
        self,
        open_positions: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if self.reconcile_timeout_seconds and self.reconcile_timeout_seconds > 0:
            await self.sync_service.reconcile_positions(
                self.reconcile_reason,
                timeout_seconds=self.reconcile_timeout_seconds,
                lock_wait_seconds=0.05,
                record_timeout_error=False,
            )
        exit_positions = await self.sync_service.get_open_positions_context()
        if open_positions is not None:
            open_positions[:] = exit_positions
        return exit_positions

    async def has_matching_exchange_position(
        self,
        model_name: str,
        decision: DecisionOutput,
    ) -> bool | None:
        return await self.sync_service.has_matching_exchange_exit_position(
            model_name,
            decision,
        )
