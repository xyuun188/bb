"""Direct market entry execution processing for non-auto scan mode."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.market_decision_result_recorder import MarketDecisionResultRecorder

CapacityReasonProvider = Callable[
    [str, DecisionOutput, list[dict[str, Any]], dict[str, dict[Any, int]]],
    str | None,
]
CapacityReserver = Callable[[str, DecisionOutput, dict[str, dict[Any, int]]], None]
CapacityReleaser = Callable[[str, DecisionOutput, dict[str, dict[Any, int]]], None]
CandidateSelectionAnnotator = Callable[..., dict[str, Any]]
DecisionRawResponseMarker = Callable[[int, dict[str, Any]], Awaitable[None]]
DecisionReasonMarker = Callable[[int, str], Awaitable[None]]
MarketNoOpportunityClearer = Callable[[str], None]
CandidateExecutor = Callable[..., Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class MarketDirectEntryProcessResult:
    """Outcome of processing a non-auto market entry."""

    handled: bool
    execution_attempted: bool = False
    execution_confirmed: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class MarketDirectEntryProcessor:
    """Apply capacity checks and execution handoff for direct entry decisions."""

    capacity_reason_provider: CapacityReasonProvider
    capacity_reserver: CapacityReserver
    annotate_candidate_selection: CandidateSelectionAnnotator
    mark_decision_raw_response: DecisionRawResponseMarker
    mark_decision_reason: DecisionReasonMarker
    result_recorder: MarketDecisionResultRecorder
    clear_market_no_opportunity_symbol: MarketNoOpportunityClearer
    candidate_executor: CandidateExecutor
    capacity_releaser: CapacityReleaser | None = None
    execution_confirmed_checker: Callable[[Any], bool] | None = None

    async def process(
        self,
        *,
        symbol: str,
        model_name: str,
        original_decision: DecisionOutput,
        executed: DecisionOutput,
        assessment: Any,
        decision_db_id: int | None,
        results: dict[str, Any],
        model_mode: str,
        open_positions: list[dict[str, Any]],
        staged_entry_counts: dict[str, dict[Any, int]],
    ) -> MarketDirectEntryProcessResult:
        capacity_reason = self.capacity_reason_provider(
            model_name,
            executed,
            open_positions,
            staged_entry_counts,
        )
        if capacity_reason:
            raw_response = self.annotate_candidate_selection(
                original_decision,
                selected=False,
                reason=capacity_reason,
            )
            if decision_db_id is not None:
                await self.mark_decision_raw_response(decision_db_id, raw_response)
                await self.mark_decision_reason(decision_db_id, capacity_reason)
            self.result_recorder.append_result(
                results=results,
                model_name=model_name,
                symbol=symbol,
                decision_or_action=executed,
                model_mode=model_mode,
                approved=True,
                execution_status="skipped",
                reason=capacity_reason,
            )
            return MarketDirectEntryProcessResult(
                handled=True,
                reason=capacity_reason,
            )

        self.capacity_reserver(model_name, executed, staged_entry_counts)
        self.clear_market_no_opportunity_symbol(symbol)
        execution_result = await self.candidate_executor(
            symbol,
            model_name,
            executed,
            assessment,
            decision_db_id,
            results,
            open_positions=open_positions,
        )
        execution_confirmed = self._execution_confirmed(execution_result)
        if not execution_confirmed and self.capacity_releaser is not None:
            self.capacity_releaser(model_name, executed, staged_entry_counts)
        return MarketDirectEntryProcessResult(
            handled=True,
            execution_attempted=True,
            execution_confirmed=execution_confirmed,
        )

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
