"""Persistence helpers for position-review non-execution outcomes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.position_review_entry_guard import PositionReviewEntryGuardResult
from services.position_review_outcome import (
    PositionReviewOutcomePolicy,
    position_review_not_executed_reason,
)

DecisionReasonMarker = Callable[[int, str], Awaitable[None]]
DecisionRawResponseMarker = Callable[[int, dict[str, Any]], Awaitable[None]]
RiskResultLogger = Callable[[DecisionOutput, str, str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class PositionReviewResultRecorder:
    """Record hold, guard, and skipped outcomes for position review."""

    outcome_policy: PositionReviewOutcomePolicy
    decision_reason_marker: DecisionReasonMarker
    decision_raw_response_marker: DecisionRawResponseMarker
    risk_result_logger: RiskResultLogger

    async def record_hold(
        self,
        *,
        decision: DecisionOutput,
        model_name: str,
        decision_db_id: int | None,
        risk_alert: str | None,
        after_risk_adjustment: bool = False,
    ) -> None:
        if risk_alert:
            await self.risk_result_logger(
                decision,
                model_name,
                self.outcome_policy.hold_reason(
                    after_risk_adjustment=after_risk_adjustment,
                    for_alert=True,
                ),
            )
        if decision_db_id is not None:
            raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
            if raw:
                await self.decision_raw_response_marker(decision_db_id, raw)
            await self.decision_reason_marker(
                decision_db_id,
                self.outcome_policy.hold_reason(
                    after_risk_adjustment=after_risk_adjustment,
                ),
            )

    async def record_entry_guard(
        self,
        *,
        decision: DecisionOutput,
        model_name: str,
        symbol: str,
        model_mode: str,
        decision_db_id: int | None,
        results: dict[str, Any] | None,
        guard: PositionReviewEntryGuardResult,
    ) -> None:
        decision.raw_response = guard.raw_response
        if decision_db_id is not None:
            await self.decision_raw_response_marker(decision_db_id, guard.raw_response)
            await self.decision_reason_marker(decision_db_id, guard.reason)
        self.append_skipped_result(
            results=results,
            model_name=model_name,
            symbol=symbol,
            decision=decision,
            reason=guard.reason,
            model_mode=model_mode,
        )

    async def record_skip(
        self,
        *,
        decision: DecisionOutput,
        model_name: str,
        symbol: str,
        model_mode: str,
        reason: str,
        decision_db_id: int | None,
        results: dict[str, Any] | None,
        risk_alert: str | None = None,
        append_result: bool = False,
    ) -> None:
        if decision_db_id is not None:
            raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
            if raw:
                await self.decision_raw_response_marker(decision_db_id, raw)
            await self.decision_reason_marker(decision_db_id, reason)
        if risk_alert:
            await self.risk_result_logger(
                decision,
                model_name,
                position_review_not_executed_reason(reason),
            )
        if append_result:
            self.append_skipped_result(
                results=results,
                model_name=model_name,
                symbol=symbol,
                decision=decision,
                reason=reason,
                model_mode=model_mode,
            )

    def append_skipped_result(
        self,
        *,
        results: dict[str, Any] | None,
        model_name: str,
        symbol: str,
        decision: DecisionOutput,
        reason: str,
        model_mode: str,
    ) -> None:
        if results is None:
            return
        results["decisions"].append(
            self.outcome_policy.skipped_result(
                model_name=model_name,
                symbol=symbol,
                decision=decision,
                reason=reason,
                is_paper=(model_mode == "paper"),
            )
        )

    def append_fast_scan_result(
        self,
        *,
        results: dict[str, Any] | None,
        model_name: str,
        symbol: str,
        reason: str,
        model_mode: str,
    ) -> None:
        if results is None:
            return
        results["decisions"].append(
            self.outcome_policy.fast_scan_result(
                model_name=model_name,
                symbol=symbol,
                reason=reason,
                is_paper=(model_mode == "paper"),
            )
        )
