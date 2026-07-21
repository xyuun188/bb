"""Analysis loop service boundaries.

Market analysis and position review have separate loop owners.  TradingService
still provides shared dependencies and low-level helpers, but scope scheduling,
position-review candidate execution, and SL/TP review sequencing live here.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog

from core.safe_output import safe_error_text
from services.decision_state import DecisionStage, DecisionStageStatus

logger = structlog.get_logger(__name__)

RunOnceProvider = Callable[[str], Awaitable[dict[str, Any]]]
RunningProvider = Callable[[], bool]
IntervalProvider = Callable[[], float]
ReviewCandidate = tuple[str, str, Any, Any, int | None]
TimeoutProvider = Callable[[], float]
TimeBudgetProvider = Callable[[], float]
DecisionStageRecorder = Callable[..., Awaitable[dict[str, Any]]]
DecisionReasonMarker = Callable[[int, str], Awaitable[None]]


@dataclass(frozen=True)
class _PositionReviewStageSkip:
    stage: str
    kind: str
    message: str


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
        loop = asyncio.get_running_loop()
        next_round_at = loop.time()
        while self.is_running():
            try:
                await self._run_loop_round()
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if not self.is_running() or (task is not None and task.cancelling()):
                    break
                logger.warning(
                    "analysis service round was cancelled while service remains running; continuing next loop",
                    scope=self.scope,
                )
            except TimeoutError:
                logger.error(
                    "analysis service loop timed out",
                    scope=self.scope,
                    timeout_seconds=self._time_budget_seconds(),
                )
            except Exception as exc:
                logger.error(
                    "analysis service loop error",
                    scope=self.scope,
                    error=safe_error_text(exc),
                )
            if not self.is_running():
                break
            interval = self._resolve_interval(interval_seconds)
            next_round_at += interval
            now = loop.time()
            if next_round_at <= now:
                next_round_at = now
            await asyncio.sleep(max(next_round_at - now, 0.0))

    async def _run_loop_round(self) -> None:
        time_budget = self._time_budget_seconds()
        if time_budget is None:
            await self.run_once()
        else:
            await asyncio.wait_for(self.run_once(), timeout=time_budget)


class MarketAnalysisService(_ScopedAnalysisService):
    scope = "market"
    initial_delay_seconds = 3.0

    async def _run_loop_round(self) -> None:
        """Let the market pipeline enforce its own stage and symbol budgets.

        TradingService already limits feature fetches, model calls, candidate
        scheduling, and exchange execution independently.  A second outer
        ``asyncio.wait_for`` used to cancel all of that work together, turning
        one slow model or optional data source into a false whole-round failure.
        Keep the watchdog value for diagnostics and the internal market budget,
        but do not use it as a cancellation boundary here.
        """

        await self.run_once()


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
        decision_stage_recorder: DecisionStageRecorder | None = None,
        decision_reason_marker: DecisionReasonMarker | None = None,
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
        self.decision_stage_recorder = decision_stage_recorder
        self.decision_reason_marker = decision_reason_marker
        self.timeout_provider = timeout_provider
        self.round_watchdog_provider = round_watchdog_provider

    async def _run_loop_round(self) -> None:
        # Position review has its own round deadline and per-stage timeouts
        # inside TradingService.run_once()/review_open_positions().  Wrapping
        # the whole coroutine in asyncio.wait_for cancels the round before those
        # softer boundaries can persist skipped groups and execution handoffs.
        # Keep round_watchdog_provider for diagnostics and internal deadlines,
        # not as an outer cancellation boundary.
        await self.run_once()

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

    def _required_decision_stage_recorder(self) -> DecisionStageRecorder:
        if self.decision_stage_recorder is None:
            raise RuntimeError("PositionReviewService requires decision_stage_recorder")
        return self.decision_stage_recorder

    def _required_decision_reason_marker(self) -> DecisionReasonMarker:
        if self.decision_reason_marker is None:
            raise RuntimeError("PositionReviewService requires decision_reason_marker")
        return self.decision_reason_marker

    def _timeout_seconds(self) -> float:
        if self.timeout_provider is None:
            return 30.0
        try:
            return max(1.0, float(self.timeout_provider()))
        except (TypeError, ValueError):
            return 30.0

    @staticmethod
    def _callable_accepts_keyword(callback: Callable[..., Any], keyword: str) -> bool:
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD or name == keyword
            for name, parameter in signature.parameters.items()
        )

    @staticmethod
    def _close_unawaited(awaitable: Awaitable[Any]) -> None:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        elif isinstance(awaitable, asyncio.Future) and not awaitable.done():
            awaitable.cancel()

    def _resolve_round_deadline(self, explicit_deadline: float | None) -> float | None:
        if explicit_deadline is not None:
            try:
                return float(explicit_deadline)
            except (TypeError, ValueError):
                return None
        budget = self._time_budget_seconds()
        if budget is None:
            return None
        return asyncio.get_running_loop().time() + budget

    @staticmethod
    def _remaining_round_seconds(round_deadline: float | None) -> float | None:
        if round_deadline is None:
            return None
        return max(float(round_deadline) - asyncio.get_running_loop().time(), 0.0)

    def _stage_timeout_seconds(
        self,
        round_deadline: float | None,
        *,
        requested_timeout_seconds: float | None = None,
        deadline_grace_seconds: float = 0.0,
    ) -> float:
        stage_timeout = max(
            1.0,
            float(
                requested_timeout_seconds
                if requested_timeout_seconds is not None
                else self._timeout_seconds()
            ),
        )
        remaining = self._remaining_round_seconds(round_deadline)
        if remaining is None:
            return stage_timeout
        reserve_seconds = min(2.0, max(0.25, stage_timeout * 0.03), remaining * 0.25)
        deadline_grace_seconds = max(0.0, float(deadline_grace_seconds or 0.0))
        return max(0.0, min(stage_timeout, remaining - reserve_seconds + deadline_grace_seconds))

    def _review_positions_timeout_seconds(self, max_groups_override: int) -> float:
        stage_timeout = self._timeout_seconds()
        try:
            group_count = max(1, int(max_groups_override or 1))
        except (TypeError, ValueError):
            group_count = 1
        shared_overhead = max(4.0, min(30.0, group_count * 2.0))
        return max(stage_timeout, stage_timeout * group_count + shared_overhead)

    def _record_stage_skip(
        self,
        *,
        stage: str,
        kind: str,
        message: str,
        results: dict[str, Any],
        timeout_seconds: float | None = None,
        remaining_budget_seconds: float | None = None,
        error: str | None = None,
    ) -> _PositionReviewStageSkip:
        diagnostic: dict[str, Any] = {
            "stage": stage,
            "kind": kind,
            "message": message,
        }
        if timeout_seconds is not None:
            diagnostic["timeout_seconds"] = round(float(timeout_seconds), 3)
        if remaining_budget_seconds is not None:
            diagnostic["remaining_budget_seconds"] = round(float(remaining_budget_seconds), 3)
        if error:
            diagnostic["error"] = error
        results.setdefault("position_review_diagnostics", []).append(diagnostic)
        results.setdefault("warnings", []).append(
            {
                "model": "position_review",
                "symbol": "ALL",
                "warning": diagnostic["message"],
            }
        )
        return _PositionReviewStageSkip(stage=stage, kind=kind, message=message)

    async def _wait_stage(
        self,
        stage: str,
        awaitable: Awaitable[Any],
        *,
        results: dict[str, Any],
        round_deadline: float | None = None,
        requested_timeout_seconds: float | None = None,
        deadline_grace_seconds: float = 0.0,
    ) -> Any | _PositionReviewStageSkip:
        timeout_seconds = self._stage_timeout_seconds(
            round_deadline,
            requested_timeout_seconds=requested_timeout_seconds,
            deadline_grace_seconds=deadline_grace_seconds,
        )
        remaining = self._remaining_round_seconds(round_deadline)
        if round_deadline is not None and (
            timeout_seconds <= 0 or (remaining is not None and remaining <= 0)
        ):
            self._close_unawaited(awaitable)
            message = (
                f"持仓复盘本轮剩余时间不足：{stage} 已顺延到下一轮，"
                "系统会用最新持仓和行情重新复盘。"
            )
            skip = self._record_stage_skip(
                stage=stage,
                kind="position_review_round_budget_exhausted",
                message=message,
                results=results,
                timeout_seconds=0.0,
                remaining_budget_seconds=remaining,
            )
            logger.info(
                "position review round budget exhausted before stage",
                stage=stage,
                remaining_budget_seconds=remaining,
            )
            return skip
        try:
            return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
        except TimeoutError:
            stage_timeout = max(
                1.0,
                float(
                    requested_timeout_seconds
                    if requested_timeout_seconds is not None
                    else self._timeout_seconds()
                ),
            )
            budget_limited = timeout_seconds < (stage_timeout - 0.001)
            if budget_limited:
                message = (
                    f"持仓复盘本轮预算用尽：{stage} 没有在剩余时间内完成，"
                    "系统已停止继续消耗本轮时间并顺延到下一轮。"
                )
                kind = "position_review_round_budget_exhausted"
            else:
                message = f"持仓复盘阶段超时：{stage}，本轮跳过该阶段并继续下一轮。"
                kind = "position_review_stage_timeout"
            skip = self._record_stage_skip(
                stage=stage,
                kind=kind,
                message=message,
                results=results,
                timeout_seconds=timeout_seconds,
                remaining_budget_seconds=self._remaining_round_seconds(round_deadline),
            )
            logger.warning(
                "position review stage timed out",
                stage=stage,
                timeout_seconds=timeout_seconds,
                budget_limited=budget_limited,
            )
            return skip
        except Exception as exc:
            error = safe_error_text(exc)
            skip = self._record_stage_skip(
                stage=stage,
                kind="position_review_stage_error",
                message=f"持仓复盘阶段异常：{stage}，本轮跳过该阶段并继续下一轮。",
                results=results,
                error=error,
            )
            logger.warning(
                "position review stage failed",
                stage=stage,
                error=error,
            )
            return skip

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
        round_deadline_monotonic: float | None = None,
    ) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
        """Run the position-review pass and execute approved exit/add candidates."""

        set_loop_stage = self._required_loop_stage_setter()
        enforce_sl_tp = self._required_sl_tp_enforcer()
        get_open_positions_context = self._required_open_positions_context_provider()
        review_positions = self._required_position_reviewer()
        try_claim_analysis_symbol = self._required_analysis_symbol_claimer()
        normalize_symbol = self._required_symbol_normalizer()
        execute_candidate = self._required_candidate_executor()
        record_decision_stage = self._required_decision_stage_recorder()
        mark_decision_reason = self._required_decision_reason_marker()
        review_blocked_keys: set[tuple[str, str]] = set()
        round_deadline = self._resolve_round_deadline(round_deadline_monotonic)

        set_loop_stage("enforce_sl_tp")
        sl_tp_kwargs: dict[str, Any] = {}
        if self._callable_accepts_keyword(enforce_sl_tp, "open_positions"):
            sl_tp_kwargs["open_positions"] = open_positions
        sl_tp_results = await self._wait_stage(
            "enforce_sl_tp",
            enforce_sl_tp(feature_vectors, **sl_tp_kwargs),
            results=results,
            round_deadline=round_deadline,
        )
        if isinstance(sl_tp_results, _PositionReviewStageSkip):
            if sl_tp_results.kind == "position_review_round_budget_exhausted":
                return open_positions, review_blocked_keys
            sl_tp_results = []
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
        if sl_tp_results:
            refreshed_open_positions = await self._wait_stage(
                "get_open_positions_context",
                get_open_positions_context(),
                results=results,
                round_deadline=round_deadline,
            )
            if isinstance(refreshed_open_positions, _PositionReviewStageSkip):
                if refreshed_open_positions.kind == "position_review_round_budget_exhausted":
                    return open_positions, review_blocked_keys
            elif isinstance(refreshed_open_positions, list):
                open_positions = refreshed_open_positions
        elif open_positions:
            results.setdefault("position_review_diagnostics", []).append(
                {
                    "stage": "get_open_positions_context",
                    "kind": "reused_round_open_positions",
                    "message": (
                        "本轮持仓复盘没有触发快速平仓，继续复用轮次开始时已加载的持仓上下文，"
                        "避免重复同步等待 OKX 当前持仓。"
                    ),
                    "open_position_count": len(open_positions or []),
                }
            )
        else:
            refreshed_open_positions = await self._wait_stage(
                "get_open_positions_context",
                get_open_positions_context(),
                results=results,
                round_deadline=round_deadline,
            )
            if isinstance(refreshed_open_positions, _PositionReviewStageSkip):
                if refreshed_open_positions.kind == "position_review_round_budget_exhausted":
                    return open_positions, review_blocked_keys
            elif isinstance(refreshed_open_positions, list):
                open_positions = refreshed_open_positions
        review_kwargs = {
            "results": results,
            "round_decision_ids": round_decision_ids,
            "position_entry_pause_reason": position_entry_pause_reason,
            "max_groups_override": max_groups_override,
        }
        if self._callable_accepts_keyword(review_positions, "round_deadline_monotonic"):
            review_kwargs["round_deadline_monotonic"] = round_deadline
        review_result = await self._wait_stage(
            "review_positions",
            review_positions(
                open_positions,
                feature_vectors,
                **review_kwargs,
            ),
            results=results,
            round_deadline=round_deadline,
            requested_timeout_seconds=self._review_positions_timeout_seconds(
                max_groups_override
            ),
            deadline_grace_seconds=5.0,
        )
        if isinstance(review_result, _PositionReviewStageSkip):
            return open_positions, review_blocked_keys
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
                round_deadline=round_deadline,
            )
            if isinstance(claim_result, _PositionReviewStageSkip):
                reason = claim_result.message
                logger.info(
                    "position execution skipped because review stage did not finish",
                    symbol=symbol,
                    stage=claim_result.stage,
                    kind=claim_result.kind,
                    reason=reason,
                )
                if decision_db_id is not None:
                    await record_decision_stage(
                        decision_db_id,
                        decision,
                        DecisionStage.STRATEGY_ARBITRATION,
                        DecisionStageStatus.SKIPPED,
                        reason,
                        {
                            "skip_kind": claim_result.kind,
                            "selected_for_execution": False,
                        },
                    )
                    await mark_decision_reason(decision_db_id, reason)
                results.setdefault("decisions", []).append(
                    {
                        "model": model_name,
                        "symbol": symbol,
                        "action": getattr(getattr(decision, "action", None), "value", None),
                        "approved": True,
                        "confidence": getattr(decision, "confidence", 0.0),
                        "executed": False,
                        "execution_status": "skipped",
                        "reason": reason,
                    }
                )
                if claim_result.kind == "position_review_round_budget_exhausted":
                    break
                continue
            if not claim_result:
                reason = (
                    "持仓复盘生成了执行候选，但同一币种正在被另一条分析流程处理；"
                    "本轮不重复提交订单，下一轮会基于最新仓位和行情重新复盘。"
                )
                logger.info(
                    "position execution skipped because another analysis owns symbol",
                    symbol=symbol,
                    reason=reason,
                )
                if decision_db_id is not None:
                    await record_decision_stage(
                        decision_db_id,
                        decision,
                        DecisionStage.STRATEGY_ARBITRATION,
                        DecisionStageStatus.SKIPPED,
                        reason,
                        {
                            "skip_kind": "position_analysis_symbol_claimed",
                            "selected_for_execution": False,
                        },
                    )
                    await mark_decision_reason(decision_db_id, reason)
                results.setdefault("decisions", []).append(
                    {
                        "model": model_name,
                        "symbol": symbol,
                        "action": getattr(getattr(decision, "action", None), "value", None),
                        "approved": True,
                        "confidence": getattr(decision, "confidence", 0.0),
                        "executed": False,
                        "execution_status": "skipped",
                        "reason": reason,
                    }
                )
                continue
            claimed_analysis_symbols.append(symbol)
            if decision_db_id is not None:
                round_decision_ids.add(decision_db_id)
            review_blocked_keys.add((model_name, normalize_symbol(symbol)))
            execute_result = await self._wait_stage(
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
                round_deadline=round_deadline,
            )
            if isinstance(execute_result, _PositionReviewStageSkip):
                reason = execute_result.message
                if decision_db_id is not None:
                    await record_decision_stage(
                        decision_db_id,
                        decision,
                        DecisionStage.EXCHANGE_SUBMIT,
                        DecisionStageStatus.SKIPPED,
                        reason,
                        {
                            "skip_kind": execute_result.kind,
                            "selected_for_execution": True,
                        },
                    )
                    await mark_decision_reason(decision_db_id, reason)
                if execute_result.kind == "position_review_round_budget_exhausted":
                    break

        return open_positions, review_blocked_keys
