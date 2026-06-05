"""Analysis loop service wrappers.

TradingService still owns the legacy implementation, but market analysis and
position review now have explicit service boundaries.  This keeps scheduling
and scope ownership out of ad-hoc call sites while the large orchestrator is
being split apart.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog


logger = structlog.get_logger(__name__)


class _ScopedAnalysisService:
    scope: str
    initial_delay_seconds: float

    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator

    async def run_once(self) -> dict[str, Any]:
        return await self.orchestrator.run_once(self.scope)

    async def loop(self, interval_seconds: float) -> None:
        await asyncio.sleep(self.initial_delay_seconds)
        while self.orchestrator._running:
            try:
                await self.run_once()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("analysis service loop error", scope=self.scope, error=str(exc))
                await asyncio.sleep(interval_seconds)


class MarketAnalysisService(_ScopedAnalysisService):
    scope = "market"
    initial_delay_seconds = 3.0


class PositionReviewService(_ScopedAnalysisService):
    scope = "position"
    initial_delay_seconds = 0.5
