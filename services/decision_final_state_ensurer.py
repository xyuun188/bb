"""Ensure attempted decisions leave a concrete final state."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy import func, select

from ai_brain.base_model import DecisionOutput
from core.safe_output import safe_error_text
from db.session import get_session_ctx
from models.decision import AIDecision
from models.trade import Order
from services.stale_entry_candidate_expirer import (
    is_pending_execution_reason,
    pending_execution_failed_reason,
)

logger = structlog.get_logger(__name__)

ExecutionReasonUnusableChecker = Callable[[Any], bool]
ExecutionReasonRecoverer = Callable[[Any], str | None]
ModelExecutionModeProvider = Callable[[str], str]
FlushCallback = Callable[[], Awaitable[None]]


@dataclass(slots=True)
class DecisionFinalStateEnsurer:
    """Finalize decisions that entered execution but did not produce an execution row."""

    execution_reason_unusable_checker: ExecutionReasonUnusableChecker
    execution_reason_recoverer: ExecutionReasonRecoverer
    model_execution_mode_provider: ModelExecutionModeProvider

    async def ensure(
        self,
        decision_id: int,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        results: dict[str, Any],
    ) -> None:
        """Load a decision row and ensure pending execution state is closed out."""
        try:
            async with get_session_ctx() as session:
                row = await session.get(AIDecision, int(decision_id))
                order_count = (
                    await session.execute(
                        select(func.count(Order.id)).where(Order.decision_id == int(decision_id))
                    )
                ).scalar() or 0
                await self.ensure_row(
                    row,
                    order_count=int(order_count),
                    symbol=symbol,
                    model_name=model_name,
                    decision=decision,
                    results=results,
                    flush_callback=session.flush,
                )
        except Exception as exc:
            logger.error(
                "failed to ensure decision final state",
                decision_id=decision_id,
                error=safe_error_text(exc),
            )

    async def ensure_row(
        self,
        row: Any,
        *,
        order_count: int,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        results: dict[str, Any],
        flush_callback: FlushCallback | None = None,
    ) -> None:
        """Finalize an already-loaded decision row."""
        if row is None or getattr(row, "was_executed", False):
            return

        reason = str(getattr(row, "execution_reason", "") or "")
        if order_count > 0:
            if is_pending_execution_reason(reason):
                row.execution_reason = "本地订单记录已生成，但成交或拒单状态还没有最终确认。请以执行记录中的最新订单状态为准。"
                if flush_callback is not None:
                    await flush_callback()
            return

        if getattr(row, "action", None) in {"close_long", "close_short"} and (
            not reason or self.execution_reason_unusable_checker(reason)
        ):
            recovered = self.execution_reason_recoverer(row)
            if recovered:
                row.execution_reason = recovered
                if flush_callback is not None:
                    await flush_callback()
                results.setdefault("decisions", []).append(
                    self._result_item(
                        symbol=symbol,
                        model_name=model_name,
                        decision=decision,
                        execution_status="skipped",
                        reason=recovered,
                    )
                )
            return

        if is_pending_execution_reason(reason):
            row.execution_reason = pending_execution_failed_reason(symbol, decision.action.value)
            if flush_callback is not None:
                await flush_callback()
            results.setdefault("decisions", []).append(
                self._result_item(
                    symbol=symbol,
                    model_name=model_name,
                    decision=decision,
                    execution_status="error",
                    reason=row.execution_reason,
                )
            )

    def _result_item(
        self,
        *,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        execution_status: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "model": model_name,
            "symbol": symbol,
            "action": decision.action.value,
            "approved": True,
            "confidence": decision.confidence,
            "executed": False,
            "execution_status": execution_status,
            "reason": reason,
            "is_paper": (self.model_execution_mode_provider(model_name) == "paper"),
        }
