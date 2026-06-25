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
IntervalProvider = Callable[[], float]
ReviewCandidate = tuple[str, str, Any, Any, int | None]
TimeoutProvider = Callable[[], float]
TimeBudgetProvider = Callable[[], float]


class _ScopedAnalysisService:
    scope: str
    initial_delay_seconds: float

    def __init__(
        self,
        *,
        run_once_provider: RunOnceProvider | None = None,
        is_running_provider: RunningProvider | None = None,
        time_budget_provider: TimeBudgetProvider | None = None,
    ) -> None:
        self._run_once_provider = run_once_provider
        self._is_running_provider = is_running_provider
        self._time_budget_provider = time_budget_provider

    async def run_once(self) -> dict[str, Any]:
        if self._run_once_provider is None:
            raise RuntimeError(f"{type(self).__name__} requires run_once_provider")
        return await self._run_once_provider(self.scope)

    def _time_budget_seconds(self) -> float | None:
        if self._time_budget_provider is None:
            return None
        try:
            return max(0.05, float(self._time_budget_provider()))
        except (TypeError, ValueError):
            return None

    def is_running(self) -> bool:
        if self._is_running_provider is None:
            raise RuntimeError(f"{type(self).__name__} requires is_running_provider")
        return bool(self._is_running_provider())

    @staticmethod
    def _resolve_interval(interval_seconds: float | IntervalProvider) -> float:
        if callable(interval_seconds):
            return max(1.0, float(interval_seconds()))
        return max(1.0, float(interval_seconds))

    async def loop(self, interval_seconds: float | IntervalProvider) -> None:
        await asyncio.sleep(self.initial_delay_seconds)
        while self.is_running():
            try:
                time_budget = self._time_budget_seconds()
                if time_budget is None:
                    await self.run_once()
                else:
                    await asyncio.wait_for(self.run_once(), timeout=time_budget)
                await asyncio.sleep(self._resolve_interval(interval_seconds))
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if not self.is_running() or (task is not None and task.cancelling()):
                    break
                logger.warning(
                    "analysis service round was cancelled while service remains running; continuing next loop",
                    scope=self.scope,
                )
                await asyncio.sleep(self._resolve_interval(interval_seconds))
            except TimeoutError:
                logger.error(
                    "analysis service loop timed out",
                    scope=self.scope,
                    timeout_seconds=self._time_budget_seconds(),
                )
                await asyncio.sleep(self._resolve_interval(interval_seconds))
            except Exception as exc:
                logger.error(
                    "analysis service loop error",
                    scope=self.scope,
                    error=safe_error_text(exc),
                )
                await asyncio.sleep(self._resolve_interval(interval_seconds))


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
        timeout_provider: TimeoutProvider | None = None,
        round_watchdog_provider: TimeBudgetProvider | None = None,
    ) -> None:
        super().__init__(
            run_once_provider=run_once_provider,
            is_running_provider=is_running_provider,
            time_budget_provider=round_watchdog_provider,
        )
        self.loop_stage_setter = loop_stage_setter
        self.sl_tp_enforcer = sl_tp_enforcer
        self.open_positions_context_provider = open_positions_context_provider
        self.position_reviewer = position_reviewer
        self.analysis_symbol_claimer = analysis_symbol_claimer
        self.symbol_normalizer = symbol_normalizer
        self.candidate_executor = candidate_executor
        self.timeout_provider = timeout_provider
        self.round_watchdog_provider = round_watchdog_provider

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

    def _timeout_seconds(self) -> float:
        if self.timeout_provider is None:
            return 30.0
        try:
            return max(1.0, float(self.timeout_provider()))
        except (TypeError, ValueError):
            return 30.0

    async def _wait_stage(
        self,
        stage: str,
        awaitable: Awaitable[Any],
        *,
        results: dict[str, Any],
    ) -> Any | None:
        timeout_seconds = self._timeout_seconds()
        try:
            return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
        except TimeoutError:
            diagnostic = {
                "stage": stage,
                "timeout_seconds": timeout_seconds,
                "message": f"持仓复盘阶段超时：{stage}，本轮跳过该阶段并继续下一轮。",
            }
            results.setdefault("position_review_diagnostics", []).append(diagnostic)
            results.setdefault("warnings", []).append(
                {
                    "model": "position_review",
                    "symbol": "ALL",
                    "warning": diagnostic["message"],
                }
            )
            logger.warning(
                "position review stage timed out",
                stage=stage,
                timeout_seconds=timeout_seconds,
            )
            return None
        except Exception as exc:
            diagnostic = {
                "stage": stage,
                "error": safe_error_text(exc),
                "message": f"持仓复盘阶段异常：{stage}，本轮跳过该阶段并继续下一轮。",
            }
            results.setdefault("position_review_diagnostics", []).append(diagnostic)
            results.setdefault("warnings", []).append(
                {
                    "model": "position_review",
                    "symbol": "ALL",
                    "warning": diagnostic["message"],
                }
            )
            logger.warning(
                "position review stage failed",
                stage=stage,
                error=safe_error_text(exc),
            )
            return None

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
        sl_tp_results = await self._wait_stage(
            "enforce_sl_tp",
            enforce_sl_tp(feature_vectors),
            results=results,
        )
        if sl_tp_results is None:
            sl_tp_results = []
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
        refreshed_open_positions = await self._wait_stage(
            "get_open_positions_context",
            get_open_positions_context(),
            results=results,
        )
        if isinstance(refreshed_open_positions, list):
            open_positions = refreshed_open_positions
        review_result = await self._wait_stage(
            "review_positions",
            review_positions(
                open_positions,
                feature_vectors,
                results=results,
                round_decision_ids=round_decision_ids,
                position_entry_pause_reason=position_entry_pause_reason,
                max_groups_override=max_groups_override,
            ),
            results=results,
        )
        if review_result is None:
            return open_positions, review_blocked_keys
        review_candidates, review_blocked_keys = review_result
        if review_candidates:
            logger.info("review pass added execution candidates", count=len(review_candidates))

        for symbol, model_name, decision, assessment, decision_db_id in review_candidates:
            claim_result = await self._wait_stage(
                f"claim_position_symbol:{symbol}",
                try_claim_analysis_symbol(symbol, "position"),
                results=results,
            )
            if not claim_result:
                logger.info(
                    "position execution skipped because another analysis owns symbol", symbol=symbol
                )
                continue
            claimed_analysis_symbols.append(symbol)
            if decision_db_id is not None:
                round_decision_ids.add(decision_db_id)
            review_blocked_keys.add((model_name, normalize_symbol(symbol)))
            await self._wait_stage(
                f"execute_position_candidate:{symbol}",
                execute_candidate(
                    symbol,
                    model_name,
                    decision,
                    assessment,
                    decision_db_id,
                    results,
                    open_positions=open_positions,
                ),
                results=results,
            )

        return open_positions, review_blocked_keys
