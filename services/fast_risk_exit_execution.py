"""Execution handoff for fast-risk and profit-drawdown exits."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import structlog

from ai_brain.base_model import DecisionOutput
from core.safe_output import safe_error_text
from executor.base_executor import ExecutionResult, OrderStatus

logger = structlog.get_logger(__name__)

ModelExecutionModeProvider = Callable[[str], str]
DecisionLogger = Callable[..., Awaitable[int]]
DecisionCountIncrementer = Callable[[], None]
CandidateExecutor = Callable[..., Awaitable[ExecutionResult | None]]
ExecutionClassifier = Callable[[ExecutionResult | None], bool]
ProfitExitRecorder = Callable[[str, str, str], None]
RiskEventLogger = Callable[[str, str, str, str], Awaitable[None]]
RejectedExecutionFactory = Callable[[DecisionOutput, Exception], ExecutionResult]
TradeLogger = Callable[[ExecutionResult, str, DecisionOutput, int | None], Awaitable[None]]
DecisionReasonMarker = Callable[[int, str], Awaitable[None]]
ExecutionReasonProvider = Callable[[ExecutionResult], str]


@dataclass(frozen=True, slots=True)
class FastRiskExitExecutionResult:
    """Outcome of submitting one fast-risk close decision."""

    auto_close: dict[str, Any] | None = None
    skipped: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FastRiskExitExecutionProcessor:
    """Submit fast-risk exit decisions through the normal execution state machine."""

    model_execution_mode_provider: ModelExecutionModeProvider
    decision_logger: DecisionLogger
    decision_count_incrementer: DecisionCountIncrementer
    candidate_executor: CandidateExecutor
    exchange_confirmed_checker: ExecutionClassifier
    exit_progress_checker: ExecutionClassifier
    profit_exit_recorder: ProfitExitRecorder
    risk_event_logger: RiskEventLogger
    rejected_execution_factory: RejectedExecutionFactory
    trade_logger: TradeLogger
    decision_reason_marker: DecisionReasonMarker
    execution_reason_provider: ExecutionReasonProvider

    async def execute(
        self,
        *,
        model_name: str,
        symbol: str,
        side: str,
        position: dict[str, Any],
        decision: DecisionOutput,
        trigger: str,
        reason: str,
        close_fraction: float,
        entry_price: float,
        current_price: float,
    ) -> FastRiskExitExecutionResult:
        model_exec_mode = self.model_execution_mode_provider(model_name)
        decision_db_id = await self.decision_logger(
            decision,
            is_paper=(model_exec_mode == "paper"),
        )
        self.decision_count_incrementer()
        fast_results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}

        try:
            execution_result = await self.candidate_executor(
                symbol,
                model_name,
                decision,
                SimpleNamespace(warnings=[]),
                decision_db_id,
                fast_results,
                open_positions=[position],
                refresh_exit_positions=False,
            )
            if execution_result is None:
                logger.info(
                    "fast risk close skipped by execution service",
                    model=model_name,
                    symbol=symbol,
                    side=side,
                    trigger=trigger,
                )
                return FastRiskExitExecutionResult(skipped=True)

            exchange_confirmed = self.exchange_confirmed_checker(execution_result)
            exit_progress = self.exit_progress_checker(execution_result)
            if (exchange_confirmed or exit_progress) and str(trigger).startswith("profit_drawdown"):
                self.profit_exit_recorder(model_name, symbol, side)

            auto_close = {
                "model_name": model_name,
                "symbol": symbol,
                "side": side,
                "quantity": execution_result.quantity,
                "entry_price": entry_price,
                "exit_price": execution_result.price,
                "pnl": execution_result.pnl,
                "trigger": trigger,
                "close_fraction": close_fraction,
                "status": execution_result.status.value,
            }
            await self.risk_event_logger(
                "info" if trigger == "take_profit" else "warning",
                symbol,
                (
                    f"[{model_name}] {reason} 入场 {entry_price:.6g}，"
                    f"当前 {current_price:.6g}，处理仓位 {close_fraction:.0%}，"
                    f"订单状态 {execution_result.status.value}，"
                    f"PnL {execution_result.pnl:+.2f} USDT。"
                ),
                model_name,
            )
            return FastRiskExitExecutionResult(auto_close=auto_close)
        except Exception as exc:
            error_text = safe_error_text(exc, limit=160)
            logger.error("failed to execute fast risk close", error=error_text)
            rejected = self.rejected_execution_factory(decision, exc)
            await self.trade_logger(rejected, model_name, decision, decision_db_id)
            if decision_db_id is not None:
                await self.decision_reason_marker(
                    decision_db_id,
                    self.execution_reason_provider(rejected),
                )

            auto_close = {
                "model_name": model_name,
                "symbol": symbol,
                "side": side,
                "quantity": 0.0,
                "entry_price": entry_price,
                "exit_price": 0.0,
                "pnl": 0.0,
                "trigger": trigger,
                "close_fraction": close_fraction,
                "status": OrderStatus.REJECTED.value,
            }
            await self.risk_event_logger(
                "warning",
                symbol,
                f"[{model_name}] {reason} but OKX close submission failed: {error_text}",
                model_name,
            )
            return FastRiskExitExecutionResult(auto_close=auto_close, error=error_text)
