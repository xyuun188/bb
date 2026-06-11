"""Analysis loop service boundaries.

Market analysis and position review have separate loop owners.  TradingService
still provides shared dependencies and low-level helpers, but scope scheduling,
position-review candidate execution, and SL/TP review sequencing live here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from core.safe_output import safe_error_text

logger = structlog.get_logger(__name__)

RunOnceProvider = Callable[[str], Awaitable[dict[str, Any]]]
RunningProvider = Callable[[], bool]
ReviewCandidate = tuple[str, str, Any, Any, int | None]


class _ScopedAnalysisService:
    scope: str
    initial_delay_seconds: float

    def __init__(
        self,
        *,
        run_once_provider: RunOnceProvider | None = None,
        is_running_provider: RunningProvider | None = None,
    ) -> None:
        self._run_once_provider = run_once_provider
        self._is_running_provider = is_running_provider

    async def run_once(self) -> dict[str, Any]:
        if self._run_once_provider is None:
            raise RuntimeError(f"{type(self).__name__} requires run_once_provider")
        return await self._run_once_provider(self.scope)

    def is_running(self) -> bool:
        if self._is_running_provider is None:
            raise RuntimeError(f"{type(self).__name__} requires is_running_provider")
        return bool(self._is_running_provider())

    async def loop(self, interval_seconds: float) -> None:
        await asyncio.sleep(self.initial_delay_seconds)
        while self.is_running():
            try:
                await self.run_once()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "analysis service loop error",
                    scope=self.scope,
                    error=safe_error_text(exc),
                )
                await asyncio.sleep(interval_seconds)


class MarketAnalysisService(_ScopedAnalysisService):
    scope = "market"
    initial_delay_seconds = 3.0


class PositionReviewService(_ScopedAnalysisService):
    scope = "position"
    initial_delay_seconds = 0.5

    def __init__(
        self,
        *,
        run_once_provider: RunOnceProvider | None = None,
        is_running_provider: RunningProvider | None = None,
        loop_stage_setter: Callable[[str], None] | None = None,
        sl_tp_enforcer: Callable[[dict[str, Any]], Awaitable[list[dict[str, Any]]]] | None = None,
        open_positions_context_provider: (
            Callable[[], Awaitable[list[dict[str, Any]]]] | None
        ) = None,
        position_reviewer: (
            Callable[..., Awaitable[tuple[list[ReviewCandidate], set[tuple[str, str]]]]] | None
        ) = None,
        analysis_symbol_claimer: Callable[[str, str], Awaitable[bool]] | None = None,
        symbol_normalizer: Callable[[str], str] | None = None,
        candidate_executor: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        super().__init__(
            run_once_provider=run_once_provider,
            is_running_provider=is_running_provider,
        )
        self.loop_stage_setter = loop_stage_setter
        self.sl_tp_enforcer = sl_tp_enforcer
        self.open_positions_context_provider = open_positions_context_provider
        self.position_reviewer = position_reviewer
        self.analysis_symbol_claimer = analysis_symbol_claimer
        self.symbol_normalizer = symbol_normalizer
        self.candidate_executor = candidate_executor

    def _required_loop_stage_setter(self) -> Callable[[str], None]:
        if self.loop_stage_setter is None:
            raise RuntimeError("PositionReviewService requires loop_stage_setter")
        return self.loop_stage_setter

    def _required_sl_tp_enforcer(
        self,
    ) -> Callable[[dict[str, Any]], Awaitable[list[dict[str, Any]]]]:
        if self.sl_tp_enforcer is None:
            raise RuntimeError("PositionReviewService requires sl_tp_enforcer")
        return self.sl_tp_enforcer

    def _required_open_positions_context_provider(
        self,
    ) -> Callable[[], Awaitable[list[dict[str, Any]]]]:
        if self.open_positions_context_provider is None:
            raise RuntimeError("PositionReviewService requires open_positions_context_provider")
        return self.open_positions_context_provider

    def _required_position_reviewer(
        self,
    ) -> Callable[..., Awaitable[tuple[list[ReviewCandidate], set[tuple[str, str]]]]]:
        if self.position_reviewer is None:
            raise RuntimeError("PositionReviewService requires position_reviewer")
        return self.position_reviewer

    def _required_analysis_symbol_claimer(
        self,
    ) -> Callable[[str, str], Awaitable[bool]]:
        if self.analysis_symbol_claimer is None:
            raise RuntimeError("PositionReviewService requires analysis_symbol_claimer")
        return self.analysis_symbol_claimer

    def _required_symbol_normalizer(self) -> Callable[[str], str]:
        if self.symbol_normalizer is None:
            raise RuntimeError("PositionReviewService requires symbol_normalizer")
        return self.symbol_normalizer

    def _required_candidate_executor(self) -> Callable[..., Awaitable[Any]]:
        if self.candidate_executor is None:
            raise RuntimeError("PositionReviewService requires candidate_executor")
        return self.candidate_executor

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

        set_loop_stage = self._required_loop_stage_setter()
        enforce_sl_tp = self._required_sl_tp_enforcer()
        get_open_positions_context = self._required_open_positions_context_provider()
        review_positions = self._required_position_reviewer()
        try_claim_analysis_symbol = self._required_analysis_symbol_claimer()
        normalize_symbol = self._required_symbol_normalizer()
        execute_candidate = self._required_candidate_executor()
        review_blocked_keys: set[tuple[str, str]] = set()

        set_loop_stage("enforce_sl_tp")
        sl_tp_results = await enforce_sl_tp(feature_vectors)
        for action in sl_tp_results:
            results["executions"].append(
                {
                    "model": action["model_name"],
                    "symbol": action["symbol"],
                    "action": f"auto_close_{action['trigger']}",
                    "quantity": action["quantity"],
                    "price": action["exit_price"],
                    "status": action.get("status", "filled"),
                }
            )

        set_loop_stage("review_open_positions")
        open_positions = await get_open_positions_context()
        review_candidates, review_blocked_keys = await review_positions(
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
            if not await try_claim_analysis_symbol(symbol, "position"):
                logger.info(
                    "position execution skipped because another analysis owns symbol", symbol=symbol
                )
                continue
            claimed_analysis_symbols.append(symbol)
            if decision_db_id is not None:
                round_decision_ids.add(decision_db_id)
            review_blocked_keys.add((model_name, normalize_symbol(symbol)))
            await execute_candidate(
                symbol,
                model_name,
                decision,
                assessment,
                decision_db_id,
                results,
                open_positions=open_positions,
            )

        return open_positions, review_blocked_keys
