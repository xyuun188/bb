"""Post-processing for position-review model decisions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog

from ai_brain.base_model import DecisionOutput
from core.safe_output import safe_error_text
from services.dynamic_exit_policy import apply_dynamic_exit
from services.entry_capacity import EntryCapacityPolicy
from services.position_review_entry_guard import PositionReviewEntryGuardPolicy
from services.position_review_result_recorder import PositionReviewResultRecorder
from services.position_review_risk_assessment import PositionReviewRiskAssessmentPolicy

logger = structlog.get_logger(__name__)

AccountBalanceProvider = Callable[[str], Awaitable[float]]
EntryRiskContractPreparer = Callable[
    [DecisionOutput, str, list[dict[str, Any]]],
    Awaitable[None],
]
CandidateExecutor = Callable[..., Awaitable[None]]
FinalStateEnsurer = Callable[
    [int, str, str, DecisionOutput, dict[str, Any] | None], Awaitable[None]
]
EntryCandidate = tuple[str, str, DecisionOutput, Any, int | None]


@dataclass(frozen=True, slots=True)
class PositionReviewProcessResult:
    """Outcome of processing one position-review decision."""

    handled: bool
    candidate: EntryCandidate | None = None
    executed_immediately: bool = False


@dataclass(frozen=True, slots=True)
class PositionReviewDecisionProcessor:
    """Apply guards, risk assessment, and immediate exit execution for reviews."""

    entry_guard: PositionReviewEntryGuardPolicy
    entry_capacity: EntryCapacityPolicy
    risk_assessment: PositionReviewRiskAssessmentPolicy
    result_recorder: PositionReviewResultRecorder
    candidate_executor: CandidateExecutor
    final_state_ensurer: FinalStateEnsurer
    account_balance_provider: AccountBalanceProvider
    entry_risk_contract_preparer: EntryRiskContractPreparer | None = None

    async def process(
        self,
        *,
        decision: DecisionOutput,
        model_name: str,
        symbol: str,
        model_mode: str,
        decision_db_id: int | None,
        open_positions: list[dict[str, Any]],
        feature_vector: Any,
        position_entry_pause_reason: str | None,
        risk_alert: str | None,
        results: dict[str, Any] | None,
    ) -> PositionReviewProcessResult:
        if decision.is_hold:
            await self.result_recorder.record_hold(
                decision=decision,
                model_name=model_name,
                decision_db_id=decision_db_id,
                risk_alert=risk_alert,
            )
            return PositionReviewProcessResult(handled=True)

        if await self._record_entry_precheck_block(
            decision=decision,
            model_name=model_name,
            symbol=symbol,
            model_mode=model_mode,
            decision_db_id=decision_db_id,
            open_positions=open_positions,
            position_entry_pause_reason=position_entry_pause_reason,
            risk_alert=risk_alert,
            results=results,
        ):
            return PositionReviewProcessResult(handled=True)

        if decision.is_entry:
            if self.entry_risk_contract_preparer is None:
                reason = "动态费后收益风险预算准备器不可用，本次持仓复盘 entry 失败关闭。"
                await self.result_recorder.record_skip(
                    decision=decision,
                    model_name=model_name,
                    symbol=symbol,
                    model_mode=model_mode,
                    reason=reason,
                    decision_db_id=decision_db_id,
                    results=results,
                    risk_alert=risk_alert,
                )
                return PositionReviewProcessResult(handled=True)
            try:
                await self.entry_risk_contract_preparer(
                    decision,
                    model_mode,
                    open_positions,
                )
            except Exception as exc:
                reason = (
                    "动态费后收益风险预算生成失败，本次持仓复盘 entry 失败关闭："
                    f"{safe_error_text(exc, limit=160)}"
                )
                await self.result_recorder.record_skip(
                    decision=decision,
                    model_name=model_name,
                    symbol=symbol,
                    model_mode=model_mode,
                    reason=reason,
                    decision_db_id=decision_db_id,
                    results=results,
                    risk_alert=risk_alert,
                )
                return PositionReviewProcessResult(handled=True)

        assessment = await self.risk_assessment.assess(
            decision=decision,
            model_name=model_name,
            open_positions=open_positions,
            feature_vector=feature_vector,
            account_balance_provider=self.account_balance_provider,
        )

        if not assessment.approved:
            rejection_reason = assessment.rejection_reason or "风控引擎拒绝本次持仓复盘决策"
            logger.info(
                "risk blocked close",
                model=model_name,
                symbol=symbol,
                reason=rejection_reason,
            )
            await self.result_recorder.record_skip(
                decision=decision,
                model_name=model_name,
                symbol=symbol,
                model_mode=model_mode,
                reason=rejection_reason,
                decision_db_id=decision_db_id,
                results=results,
                risk_alert=risk_alert,
            )
            return PositionReviewProcessResult(handled=True)

        executed = assessment.decision if assessment.decision else decision
        if executed is not decision and decision.raw_response and not executed.raw_response:
            executed.raw_response = decision.raw_response
            executed.feature_snapshot = executed.feature_snapshot or decision.feature_snapshot

        if executed.is_hold:
            await self.result_recorder.record_hold(
                decision=executed,
                model_name=model_name,
                decision_db_id=decision_db_id,
                risk_alert=risk_alert,
                after_risk_adjustment=True,
            )
            return PositionReviewProcessResult(handled=True)

        entry_guard = self.entry_guard.block_reason(
            executed,
            position_entry_pause_reason,
            after_risk_adjustment=True,
        )
        if entry_guard is not None:
            await self.result_recorder.record_entry_guard(
                decision=executed,
                model_name=model_name,
                symbol=symbol,
                model_mode=model_mode,
                decision_db_id=decision_db_id,
                results=results,
                guard=entry_guard,
            )
            return PositionReviewProcessResult(handled=True)

        if executed.is_exit:
            return await self._process_exit(
                executed=executed,
                assessment=assessment,
                model_name=model_name,
                symbol=symbol,
                model_mode=model_mode,
                decision_db_id=decision_db_id,
                open_positions=open_positions,
                risk_alert=risk_alert,
                results=results,
            )

        return PositionReviewProcessResult(
            handled=False,
            candidate=(symbol, model_name, executed, assessment, decision_db_id),
        )

    async def _record_entry_precheck_block(
        self,
        *,
        decision: DecisionOutput,
        model_name: str,
        symbol: str,
        model_mode: str,
        decision_db_id: int | None,
        open_positions: list[dict[str, Any]],
        position_entry_pause_reason: str | None,
        risk_alert: str | None,
        results: dict[str, Any] | None,
    ) -> bool:
        if not decision.is_entry:
            return False

        entry_guard = self.entry_guard.block_reason(
            decision,
            position_entry_pause_reason,
        )
        if entry_guard is not None:
            await self.result_recorder.record_entry_guard(
                decision=decision,
                model_name=model_name,
                symbol=symbol,
                model_mode=model_mode,
                decision_db_id=decision_db_id,
                results=results,
                guard=entry_guard,
            )
            return True

        capacity_reason = self.entry_capacity.reason(
            model_name,
            decision,
            open_positions,
            {"model_totals": {}, "symbol_side": {}, "side_totals": {}},
        )
        if capacity_reason:
            await self.result_recorder.record_skip(
                decision=decision,
                model_name=model_name,
                symbol=symbol,
                model_mode=model_mode,
                reason=capacity_reason,
                decision_db_id=decision_db_id,
                results=results,
                risk_alert=risk_alert,
            )
            return True
        return False

    async def _process_exit(
        self,
        *,
        executed: DecisionOutput,
        assessment: Any,
        model_name: str,
        symbol: str,
        model_mode: str,
        decision_db_id: int | None,
        open_positions: list[dict[str, Any]],
        risk_alert: str | None,
        results: dict[str, Any] | None,
    ) -> PositionReviewProcessResult:
        dynamic_exit = apply_dynamic_exit(executed, open_positions)
        if not dynamic_exit.eligible:
            await self.result_recorder.record_skip(
                decision=executed,
                model_name=model_name,
                symbol=symbol,
                model_mode=model_mode,
                reason=dynamic_exit.reason,
                decision_db_id=decision_db_id,
                results=results,
                risk_alert=risk_alert,
                append_result=True,
            )
            return PositionReviewProcessResult(handled=True)

        if results is None:
            return PositionReviewProcessResult(
                handled=False,
                candidate=(symbol, model_name, executed, assessment, decision_db_id),
            )

        logger.info(
            "review exit decision executing immediately",
            model=model_name,
            symbol=symbol,
            action=executed.action.value,
            decision_id=decision_db_id,
        )
        await self.candidate_executor(
            symbol,
            model_name,
            executed,
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
                executed,
                results,
            )
        return PositionReviewProcessResult(
            handled=True,
            executed_immediately=True,
        )
