"""Auto-scan market entry processing before order execution."""

from __future__ import annotations

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
from services.entry_immediate_execution import EntryImmediateExecutionPlanner
from services.market_decision_result_recorder import MarketDecisionResultRecorder

logger = structlog.get_logger(__name__)

ENTRY_EVIDENCE_SHADOW_ONLY_TIERS = frozenset({"weak_conflict_probe", "degraded_missing_probe"})
ENTRY_EVIDENCE_SHADOW_ONLY_REASON = (
    "动态证据仍处于弱证据学习档，本轮只记录影子样本和复盘数据，"
    "不提交 OKX 真实/模拟订单；需要更多同向模型证据或更高预期收益后再开仓。"
)

ScoreCandidate = Callable[[DecisionOutput, dict[str, Any] | None], float]
GateReason = Callable[[DecisionOutput], str | None]
CandidateSelectionAnnotator = Callable[..., dict[str, Any]]
DecisionRawResponseMarker = Callable[[int, dict[str, Any]], Awaitable[None]]
DecisionReasonMarker = Callable[[int, str], Awaitable[None]]
PendingExecutionMarker = Callable[[int, str], Awaitable[None]]
LoopStageSetter = Callable[[str], None]
CandidateExecutor = Callable[..., Awaitable[Any]]
FinalStateEnsurer = Callable[[int, str, str, DecisionOutput, dict[str, Any]], Awaitable[None]]
MarketNoOpportunityClearer = Callable[[str], None]
CapacityReleaser = Callable[[str, DecisionOutput, dict[str, dict[Any, int]]], None]


@dataclass(frozen=True, slots=True)
class MarketAutoEntryProcessResult:
    """Outcome of processing an auto-scan entry decision."""

    handled: bool
    execution_attempted: bool = False
    execution_confirmed: bool = False
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
        model_mode: str,
        open_positions: list[dict[str, Any]],
        staged_entry_counts: dict[str, dict[Any, int]],
        strategy_mode_context: dict[str, Any] | None,
    ) -> MarketAutoEntryProcessResult:
        self.clear_market_no_opportunity_symbol(symbol)
        self.score_candidate(decision, strategy_mode_context)
        shadow_only_reason = self._entry_evidence_shadow_only_reason(decision)
        if shadow_only_reason:
            await self._record_skip(
                symbol=symbol,
                model_name=model_name,
                decision=decision,
                decision_db_id=decision_db_id,
                results=results,
                model_mode=model_mode,
                reason=shadow_only_reason,
            )
            return MarketAutoEntryProcessResult(
                handled=True,
                execution_attempted=False,
                reason=shadow_only_reason,
            )
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
            return MarketAutoEntryProcessResult(
                handled=True,
                execution_attempted=True,
                execution_confirmed=execution_confirmed,
                reason=immediate_plan.reason,
            )
        except Exception as exc:
            self._release_capacity(model_name, decision, staged_entry_counts)
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
                execution_confirmed=False,
                execution_error=error_text,
                reason=reason,
            )

    def _release_capacity(
        self,
        model_name: str,
        decision: DecisionOutput,
        staged_entry_counts: dict[str, dict[Any, int]],
    ) -> None:
        if self.capacity_releaser is not None:
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

    @staticmethod
    def _entry_evidence_shadow_only_reason(decision: DecisionOutput) -> str | None:
        """Return the explicit observation-only reason for weak evidence entries."""

        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw, dict) else {}
        if not isinstance(opportunity, dict):
            return None
        evidence_score = opportunity.get("evidence_score")
        if not isinstance(evidence_score, dict):
            return None
        evidence_tier = str(evidence_score.get("tier") or "")
        if evidence_tier not in ENTRY_EVIDENCE_SHADOW_ONLY_TIERS:
            return None
        legacy_tradeable_probe = bool(evidence_score.get("tradeable_probe"))
        legacy_shadow_only = bool(evidence_score.get("shadow_only", True))
        raw["entry_evidence_shadow_only"] = {
            "applied": True,
            "stage_status": "skipped",
            "skip_kind": "entry_evidence_shadow_only",
            "shadow_only": True,
            "evidence_tier": evidence_tier,
            "evidence_score": evidence_score,
            "legacy_tradeable_probe": legacy_tradeable_probe,
            "legacy_shadow_only": legacy_shadow_only,
            "position_size_pct_before_execution": float(decision.position_size_pct or 0.0),
            "reason": ENTRY_EVIDENCE_SHADOW_ONLY_REASON,
        }
        raw = append_decision_stage(
            raw,
            DecisionStage.RISK_CHECK,
            DecisionStageStatus.SKIPPED,
            ENTRY_EVIDENCE_SHADOW_ONLY_REASON,
            {
                "skip_kind": "entry_evidence_shadow_only",
                "shadow_only": True,
                "evidence_tier": evidence_tier,
            },
        )
        decision.raw_response = raw
        return ENTRY_EVIDENCE_SHADOW_ONLY_REASON

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
        raw_response = append_decision_stage(
            raw_response,
            DecisionStage.RISK_CHECK,
            DecisionStageStatus.SKIPPED,
            reason,
            {"skip_kind": "entry_pre_execution_skip"},
        )
        decision.raw_response = raw_response
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
