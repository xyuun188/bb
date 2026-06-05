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

    async def review_open_positions(
        self,
        *,
        feature_vectors: dict[str, Any],
        results: dict[str, Any],
        round_decision_ids: set[int],
        open_positions: list[dict[str, Any]],
        position_entry_pause_reason: str | None,
        max_groups_override: int,
        claimed_analysis_symbols: list[str],
    ) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
        """Run the position-review pass and execute approved exit/add candidates."""

        orchestrator = self.orchestrator
        review_blocked_keys: set[tuple[str, str]] = set()

        orchestrator._set_loop_stage("enforce_sl_tp")
        sl_tp_results = await orchestrator._enforce_sl_tp(feature_vectors)
        for action in sl_tp_results:
            results["executions"].append({
                "model": action["model_name"],
                "symbol": action["symbol"],
                "action": f"auto_close_{action['trigger']}",
                "quantity": action["quantity"],
                "price": action["exit_price"],
                "status": action.get("status", "filled"),
            })

        orchestrator._set_loop_stage("review_open_positions")
        open_positions = await orchestrator.okx_sync_service.get_open_positions_context()
        review_candidates, review_blocked_keys = await orchestrator._review_open_positions(
            open_positions,
            feature_vectors,
            results=results,
            round_decision_ids=round_decision_ids,
            position_entry_pause_reason=position_entry_pause_reason,
            max_groups_override=max_groups_override,
        )
        if review_candidates:
            logger.info("review pass added execution candidates", count=len(review_candidates))

        for symbol, model_name, decision, assessment, decision_db_id in review_candidates:
            if not await orchestrator._try_claim_analysis_symbol(symbol, "position"):
                logger.info("position execution skipped because another analysis owns symbol", symbol=symbol)
                continue
            claimed_analysis_symbols.append(symbol)
            if decision_db_id is not None:
                round_decision_ids.add(decision_db_id)
            review_blocked_keys.add((model_name, orchestrator._normalize_position_symbol(symbol)))
            await orchestrator._execute_candidate(
                symbol,
                model_name,
                decision,
                assessment,
                decision_db_id,
                results,
                open_positions=open_positions,
            )

        return open_positions, review_blocked_keys
