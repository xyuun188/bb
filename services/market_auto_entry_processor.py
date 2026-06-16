"""Auto-scan market entry processing before order execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog

from ai_brain.base_model import DecisionOutput
from core.safe_output import safe_error_text
from services.entry_immediate_execution import EntryImmediateExecutionPlanner
from services.market_decision_result_recorder import MarketDecisionResultRecorder

logger = structlog.get_logger(__name__)

ScoreCandidate = Callable[[DecisionOutput, dict[str, Any] | None], float]
GateReason = Callable[[DecisionOutput], str | None]
CandidateSelectionAnnotator = Callable[..., dict[str, Any]]
DecisionRawResponseMarker = Callable[[int, dict[str, Any]], Awaitable[None]]
DecisionReasonMarker = Callable[[int, str], Awaitable[None]]
PendingExecutionMarker = Callable[[int, str], Awaitable[None]]
LoopStageSetter = Callable[[str], None]
CandidateExecutor = Callable[..., Awaitable[None]]
FinalStateEnsurer = Callable[[int, str, str, DecisionOutput, dict[str, Any]], Awaitable[None]]
MarketNoOpportunityClearer = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class MarketAutoEntryProcessResult:
    """Outcome of processing an auto-scan entry decision."""

    handled: bool
    execution_attempted: bool = False
    execution_error: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class MarketAutoEntryProcessor:
    """Apply entry gate, immediate planning, and execution handoff for auto scan."""

    score_candidate: ScoreCandidate
    gate_reason: GateReason
    immediate_execution: EntryImmediateExecutionPlanner
    annotate_candidate_selection: CandidateSelectionAnnotator
    mark_decision_raw_response: DecisionRawResponseMarker
    mark_decision_reason: DecisionReasonMarker
    mark_decision_pending_execution: PendingExecutionMarker
    result_recorder: MarketDecisionResultRecorder
    clear_market_no_opportunity_symbol: MarketNoOpportunityClearer
    set_loop_stage: LoopStageSetter
    candidate_executor: CandidateExecutor
    final_state_ensurer: FinalStateEnsurer

    async def process(
        self,
        *,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        assessment: Any,
        decision_db_id: int | None,
        results: dict[str, Any],
        model_mode: str,
        open_positions: list[dict[str, Any]],
        staged_entry_counts: dict[str, dict[Any, int]],
        strategy_mode_context: dict[str, Any] | None,
    ) -> MarketAutoEntryProcessResult:
        self.clear_market_no_opportunity_symbol(symbol)
        self.score_candidate(decision, strategy_mode_context)
        if decision_db_id is not None:
            await self.mark_decision_raw_response(decision_db_id, decision.raw_response)

        opportunity_reason = self.gate_reason(decision)
        if opportunity_reason:
            reason = self._entry_gate_skip_reason(opportunity_reason)
            await self._record_skip(
                symbol=symbol,
                model_name=model_name,
                decision=decision,
                decision_db_id=decision_db_id,
                results=results,
                model_mode=model_mode,
                reason=reason,
            )
            return MarketAutoEntryProcessResult(handled=True, reason=reason)

        immediate_plan = self.immediate_execution.plan(
            model_name=model_name,
            decision=decision,
            open_positions=open_positions,
            staged_entry_counts=staged_entry_counts,
        )
        if not immediate_plan.should_execute:
            reason = immediate_plan.reason
            await self._record_skip(
                symbol=symbol,
                model_name=model_name,
                decision=decision,
                decision_db_id=decision_db_id,
                results=results,
                model_mode=model_mode,
                reason=reason,
            )
            return MarketAutoEntryProcessResult(handled=True, reason=reason)

        raw_response = self.annotate_candidate_selection(
            decision,
            selected=True,
            reason=immediate_plan.reason,
        )
        if decision_db_id is not None:
            await self.mark_decision_raw_response(decision_db_id, raw_response)
            await self.mark_decision_pending_execution(decision_db_id, immediate_plan.reason)

        self.set_loop_stage(f"execute:{symbol}")
        try:
            await self.candidate_executor(
                symbol,
                model_name,
                decision,
                assessment,
                decision_db_id,
                results,
                open_positions=open_positions,
            )
            if decision_db_id is not None:
                await self.final_state_ensurer(
                    decision_db_id,
                    symbol,
                    model_name,
                    decision,
                    results,
                )
        except Exception as exc:
            error_text = safe_error_text(exc, limit=160)
            reason = self._execution_error_reason(
                error_text,
                is_strong_signal=immediate_plan.is_strong_signal,
            )
            logger.error(
                (
                    "immediate entry execution crashed"
                    if immediate_plan.is_strong_signal
                    else "entry execution crashed"
                ),
                symbol=symbol,
                model=model_name,
                action=decision.action.value,
                error=error_text,
            )
            if decision_db_id is not None:
                await self.mark_decision_reason(decision_db_id, reason)
            self.result_recorder.append_result(
                results=results,
                model_name=model_name,
                symbol=symbol,
                decision_or_action=decision,
                model_mode=model_mode,
                approved=True,
                execution_status="error",
                reason=reason,
            )
            return MarketAutoEntryProcessResult(
                handled=True,
                execution_attempted=True,
                execution_error=error_text,
                reason=reason,
            )

        return MarketAutoEntryProcessResult(
            handled=True,
            execution_attempted=True,
            reason=immediate_plan.reason,
        )

    @staticmethod
    def _entry_gate_skip_reason(gate_reason: str) -> str:
        """Preserve severe gate semantics without relabeling all skips as score failures."""

        reason = str(gate_reason or "").strip()
        if not reason:
            return "入场执行前检查未通过，本轮不提交 OKX 订单。"
        if any(
            token in reason
            for token in (
                "动态证据不足",
                "保持观望",
                "极小探针",
                "强冲突硬拦截",
                "硬拦截",
                "风控",
                "暂停新开仓",
                "持仓压力",
            )
        ):
            return reason
        return f"入场候选暂未满足执行条件：{reason}"

    async def _record_skip(
        self,
        *,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        decision_db_id: int | None,
        results: dict[str, Any],
        model_mode: str,
        reason: str,
    ) -> None:
        raw_response = self.annotate_candidate_selection(
            decision,
            selected=False,
            reason=reason,
        )
        if decision_db_id is not None:
            await self.mark_decision_raw_response(decision_db_id, raw_response)
            await self.mark_decision_reason(decision_db_id, reason)
        self.result_recorder.append_result(
            results=results,
            model_name=model_name,
            symbol=symbol,
            decision_or_action=decision,
            model_mode=model_mode,
            approved=True,
            execution_status="skipped",
            reason=reason,
        )

    @staticmethod
    def _execution_error_reason(error_text: str, *, is_strong_signal: bool) -> str:
        if is_strong_signal:
            return (
                "强信号已进入即时执行，但下单流程异常中断："
                f"{error_text}。系统已跳过本次订单，下一轮会重新分析。"
            )
        return (
            "开仓信号已进入即时执行，但下单流程异常中断："
            f"{error_text}。系统已跳过本次订单，下一轮会重新分析。"
        )
