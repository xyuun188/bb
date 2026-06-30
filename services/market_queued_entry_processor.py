"""Queued market entry execution after candidate ranking/filtering."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog

from ai_brain.base_model import DecisionOutput
from core.safe_output import safe_error_text
from services.decision_state import (
    DecisionStage,
    DecisionStageStatus,
    append_decision_stage,
)
from services.market_decision_result_recorder import MarketDecisionResultRecorder

logger = structlog.get_logger(__name__)

NormalizeSymbol = Callable[[str], str]
AnalysisSymbolClaimer = Callable[[str, str], Awaitable[bool]]
CandidateSelectionAnnotator = Callable[..., dict[str, Any]]
DecisionRawResponseMarker = Callable[[int, dict[str, Any]], Awaitable[None]]
DecisionReasonMarker = Callable[[int, str], Awaitable[None]]
PendingExecutionMarker = Callable[[int, str], Awaitable[None]]
LoopStageSetter = Callable[[str], None]
CandidateExecutor = Callable[..., Awaitable[Any]]
FinalStateEnsurer = Callable[[int, str, str, DecisionOutput, dict[str, Any]], Awaitable[None]]
ModelExecutionModeProvider = Callable[[str], str]
CapacityReleaser = Callable[[str, DecisionOutput, dict[str, dict[Any, int]]], None]

QUEUED_ENTRY_EXECUTION_REASON = (
    "排序后进入执行：该信号不是即时强信号，但在本轮候选比较后通过机会评分、"
    "容量和风控筛选，正在进入下单前检查。"
)
QUEUED_ENTRY_PENDING_REASON = (
    "排序后进入执行：该信号不是即时强信号，但在本轮候选比较后通过机会评分、"
    "容量和风控筛选；正在进行下单前价格偏移、异常插针、保证金和 OKX 提交检查。"
)


@dataclass(frozen=True, slots=True)
class MarketQueuedEntryProcessResult:
    """Outcome for one queued entry candidate."""

    handled: bool
    claimed_symbol: str | None = None
    execution_attempted: bool = False
    execution_confirmed: bool = False
    execution_error: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class MarketQueuedEntryProcessor:
    """Execute ranked entry candidates after round-level filtering."""

    normalize_symbol: NormalizeSymbol
    analysis_symbol_claimer: AnalysisSymbolClaimer
    annotate_candidate_selection: CandidateSelectionAnnotator
    mark_decision_raw_response: DecisionRawResponseMarker
    mark_decision_reason: DecisionReasonMarker
    mark_decision_pending_execution: PendingExecutionMarker
    result_recorder: MarketDecisionResultRecorder
    model_execution_mode_provider: ModelExecutionModeProvider
    set_loop_stage: LoopStageSetter
    candidate_executor: CandidateExecutor
    final_state_ensurer: FinalStateEnsurer
    capacity_releaser: CapacityReleaser | None = None
    execution_confirmed_checker: Callable[[Any], bool] | None = None

    async def process(
        self,
        *,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        assessment: Any,
        decision_db_id: int | None,
        results: dict[str, Any],
        open_positions: list[dict[str, Any]],
        claimed_symbol_keys: set[str],
        staged_entry_counts: dict[str, dict[Any, int]] | None = None,
    ) -> MarketQueuedEntryProcessResult:
        normalized_symbol = self.normalize_symbol(symbol)
        claimed_this_call = normalized_symbol not in claimed_symbol_keys
        if claimed_this_call:
            claimed = await self.analysis_symbol_claimer(symbol, "market")
            if not claimed:
                reason = "该币种正在被另一条分析流程处理，本次开仓执行跳过，等待下一轮重新评估。"
                logger.info(
                    "entry execution skipped because another analysis owns symbol",
                    symbol=symbol,
                )
                if decision_db_id is not None:
                    decision.raw_response = append_decision_stage(
                        decision.raw_response if isinstance(decision.raw_response, dict) else {},
                        DecisionStage.STRATEGY_ARBITRATION,
                        DecisionStageStatus.SKIPPED,
                        reason,
                        {"skip_kind": "analysis_symbol_claimed"},
                    )
                    await self.mark_decision_raw_response(decision_db_id, decision.raw_response)
                    await self.mark_decision_reason(decision_db_id, reason)
                self.result_recorder.append_result(
                    results=results,
                    model_name=model_name,
                    symbol=symbol,
                    decision_or_action=decision,
                    model_mode=self.model_execution_mode_provider(model_name),
                    approved=True,
                    execution_status="skipped",
                    reason=reason,
                )
                return MarketQueuedEntryProcessResult(
                    handled=True,
                    reason=reason,
                )

        self.set_loop_stage(f"execute:{symbol}")
        raw_response = self.annotate_candidate_selection(
            decision,
            selected=True,
            reason=QUEUED_ENTRY_EXECUTION_REASON,
        )
        if decision_db_id is not None:
            await self.mark_decision_raw_response(decision_db_id, raw_response)
            await self.mark_decision_pending_execution(
                decision_db_id,
                QUEUED_ENTRY_PENDING_REASON,
            )

        try:
            execution_result = await self.candidate_executor(
                symbol,
                model_name,
                decision,
                assessment,
                decision_db_id,
                results,
                open_positions=open_positions,
            )
            execution_confirmed = self._execution_confirmed(execution_result)
            if not execution_confirmed:
                self._release_capacity(model_name, decision, staged_entry_counts)
            if decision_db_id is not None:
                await self.final_state_ensurer(
                    decision_db_id,
                    symbol,
                    model_name,
                    decision,
                    results,
                )
            return MarketQueuedEntryProcessResult(
                handled=True,
                claimed_symbol=symbol if claimed_this_call else None,
                execution_attempted=True,
                execution_confirmed=execution_confirmed,
                reason=QUEUED_ENTRY_EXECUTION_REASON,
            )
        except asyncio.CancelledError:
            self._release_capacity(model_name, decision, staged_entry_counts)
            reason = (
                "候选进入 OKX 下单流程后，本轮分析/执行任务被外层超时保护取消；"
                "系统已按未执行处理，下一轮会用最新行情重新评估。"
            )
            logger.error(
                "entry candidate execution cancelled",
                symbol=symbol,
                model=model_name,
                action=decision.action.value,
            )
            if decision_db_id is not None:
                decision.raw_response = append_decision_stage(
                    decision.raw_response if isinstance(decision.raw_response, dict) else {},
                    DecisionStage.EXCHANGE_SUBMIT,
                    DecisionStageStatus.FAILED,
                    reason,
                    {"skip_kind": "queued_entry_execution_cancelled"},
                )
                await self.mark_decision_raw_response(decision_db_id, decision.raw_response)
                await self.mark_decision_reason(decision_db_id, reason)
            self.result_recorder.append_result(
                results=results,
                model_name=model_name,
                symbol=symbol,
                decision_or_action=decision,
                model_mode=self.model_execution_mode_provider(model_name),
                approved=True,
                execution_status="error",
                reason=reason,
            )
            return MarketQueuedEntryProcessResult(
                handled=True,
                claimed_symbol=symbol if claimed_this_call else None,
                execution_attempted=True,
                execution_confirmed=False,
                execution_error="cancelled",
                reason=reason,
            )
        except Exception as exc:
            self._release_capacity(model_name, decision, staged_entry_counts)
            error_text = safe_error_text(exc, limit=160)
            reason = (
                "候选进入执行流程后异常中断："
                f"{error_text}。系统已跳过本次订单，下一轮会用最新行情重新评估。"
            )
            logger.error(
                "entry candidate execution crashed",
                symbol=symbol,
                model=model_name,
                action=decision.action.value,
                error=error_text,
            )
            if decision_db_id is not None:
                decision.raw_response = append_decision_stage(
                    decision.raw_response if isinstance(decision.raw_response, dict) else {},
                    DecisionStage.EXCHANGE_SUBMIT,
                    DecisionStageStatus.FAILED,
                    reason,
                    {"skip_kind": "queued_entry_execution_error", "error": error_text},
                )
                await self.mark_decision_raw_response(decision_db_id, decision.raw_response)
                await self.mark_decision_reason(decision_db_id, reason)
            self.result_recorder.append_result(
                results=results,
                model_name=model_name,
                symbol=symbol,
                decision_or_action=decision,
                model_mode=self.model_execution_mode_provider(model_name),
                approved=True,
                execution_status="error",
                reason=reason,
            )
            return MarketQueuedEntryProcessResult(
                handled=True,
                execution_attempted=True,
                execution_confirmed=False,
                execution_error=error_text,
                reason=reason,
            )

    def _release_capacity(
        self,
        model_name: str,
        decision: DecisionOutput,
        staged_entry_counts: dict[str, dict[Any, int]] | None,
    ) -> None:
        if self.capacity_releaser is not None and staged_entry_counts is not None:
            self.capacity_releaser(model_name, decision, staged_entry_counts)

    def _execution_confirmed(self, execution_result: Any) -> bool:
        if self.execution_confirmed_checker is not None:
            return bool(self.execution_confirmed_checker(execution_result))
        if execution_result is None:
            return False
        status = getattr(getattr(execution_result, "status", None), "value", None)
        if status is None:
            status = str(getattr(execution_result, "status", "") or "").lower()
        return bool(
            status == "filled"
            and str(getattr(execution_result, "exchange_order_id", "") or "").strip()
            and float(getattr(execution_result, "quantity", 0.0) or 0.0) > 0
        )
