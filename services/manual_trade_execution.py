"""Execution handoff for manual one-shot trades after risk approval."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput
from executor.base_executor import ExecutionResult

DecisionLogger = Callable[..., Awaitable[int]]
DecisionCountIncrementer = Callable[[], None]
CandidateExecutor = Callable[..., Awaitable[ExecutionResult | None]]
PaperModeProvider = Callable[[], bool]

MANUAL_HOLD_REASON = "AI 选择观望，未提交订单。"
MANUAL_EXECUTION_EMPTY_REASON = "交易接口未返回执行结果，或执行前策略/风控检查未通过。"


@dataclass(frozen=True, slots=True)
class ManualTradeExecutionProcessor:
    """Record and execute an already-approved manual trade decision."""

    decision_logger: DecisionLogger
    decision_count_incrementer: DecisionCountIncrementer
    candidate_executor: CandidateExecutor
    is_paper_provider: PaperModeProvider

    async def execute(
        self,
        *,
        symbol: str,
        model_name: str,
        original_decision: DecisionOutput,
        assessment: Any,
        open_positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        executed = assessment.decision if assessment.decision else original_decision
        if (
            executed is not original_decision
            and original_decision.raw_response
            and not executed.raw_response
        ):
            executed.raw_response = original_decision.raw_response
            executed.feature_snapshot = (
                executed.feature_snapshot or original_decision.feature_snapshot
            )

        if executed.is_hold:
            return {
                "approved": True,
                "reason": MANUAL_HOLD_REASON,
            }

        decision_db_id = await self.decision_logger(
            executed,
            is_paper=self.is_paper_provider(),
        )
        self.decision_count_incrementer()

        manual_results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}
        execution_result = await self.candidate_executor(
            symbol,
            model_name,
            executed,
            assessment,
            decision_db_id,
            manual_results,
            open_positions=open_positions,
        )
        if execution_result is None:
            return {
                "approved": False,
                "rejection_reason": self._last_decision_reason(manual_results),
            }

        return {
            "approved": True,
            "execution": {
                "order_id": execution_result.order_id,
                "status": execution_result.status.value,
                "quantity": execution_result.quantity,
                "price": execution_result.price,
            },
        }

    @staticmethod
    def _last_decision_reason(results: dict[str, Any]) -> str:
        decisions = results.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            return MANUAL_EXECUTION_EMPTY_REASON
        last_decision = decisions[-1]
        if not isinstance(last_decision, dict):
            return MANUAL_EXECUTION_EMPTY_REASON
        reason = last_decision.get("reason")
        return str(reason or MANUAL_EXECUTION_EMPTY_REASON)
