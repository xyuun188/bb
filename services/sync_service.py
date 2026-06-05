"""OKX and local-position synchronization boundary."""

from __future__ import annotations

from typing import Any


class OkxSyncService:
    """Thin synchronization facade around the legacy TradingService methods."""

    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator

    async def reconcile_positions(self, reason: str) -> Any:
        return await self.orchestrator._reconcile_exchange_positions_with_timeout(reason)

    async def get_open_positions_context(self) -> list[dict[str, Any]]:
        return await self.orchestrator._get_open_positions_context()

    async def refresh_position_prices(self, feature_vectors: dict[str, Any]) -> Any:
        return await self.orchestrator._refresh_db_position_prices(feature_vectors)

    async def has_matching_exchange_exit_position(self, model_name: str, decision: Any) -> bool:
        return await self.orchestrator._has_matching_exchange_exit_position(model_name, decision)
