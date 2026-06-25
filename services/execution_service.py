"""Execution service boundary.

ExecutionService owns serialized execution and the order-submit state machine.
TradingService remains the orchestrator and dependency provider, but the
submit/confirm/local-sync flow physically lives here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from time import perf_counter
from typing import Any

import structlog

from ai_brain.base_model import DecisionOutput
from core.safe_output import safe_error_text
from core.symbols import normalize_trading_symbol
from executor.base_executor import ExecutionResult
from services.decision_state import DecisionStage, DecisionStageStatus
from services.strategy_arbitration import arbitrate_decision
from services.trading_policies import PolicyGateResult

logger = structlog.get_logger(__name__)

AGENT_SKILLS_TRADING_EFFECTS_ENABLED = True


class ExecutionService:
    def __init__(
        self,
        *,
        execution_lock: AbstractAsyncContextManager[Any] | None = None,
        risk_event_logger: Callable[..., Awaitable[None]] | None = None,
        model_execution_mode_provider: Callable[[str], str] | None = None,
        decision_stage_recorder: Callable[..., Awaitable[dict[str, Any]]] | None = None,
        decision_reason_marker: Callable[[int, str | None], Awaitable[None]] | None = None,
        decision_raw_response_marker: (
            Callable[[int, dict[str, Any] | None], Awaitable[None]] | None
        ) = None,
        position_review_alert_context_provider: (
            Callable[[DecisionOutput], dict[str, Any] | None] | None
        ) = None,
        position_review_risk_result_logger: Callable[..., Awaitable[None]] | None = None,
        duplicate_decision_order_reason_provider: (
            Callable[[int, DecisionOutput], Awaitable[str | None]] | None
        ) = None,
        okx_executor_provider: Callable[[str], Awaitable[Any]] | None = None,
        allocated_order_balance_provider: (
            Callable[[str, DecisionOutput], Awaitable[float | None]] | None
        ) = None,
        rejected_execution_result_factory: (
            Callable[[DecisionOutput, str], ExecutionResult] | None
        ) = None,
        execution_leverage_summary_attacher: (
            Callable[[DecisionOutput, ExecutionResult, float], None] | None
        ) = None,
        execution_reason_provider: Callable[[ExecutionResult | None], str] | None = None,
        pending_execution_marker: Callable[[int, str], Awaitable[None]] | None = None,
        untradable_exchange_error_checker: Callable[[str], bool] | None = None,
        untradable_symbol_rememberer: Callable[[str, str], None] | None = None,
        transient_entry_exchange_error_checker: Callable[[str], bool] | None = None,
        temporary_entry_block_rememberer: Callable[[str, str, float], None] | None = None,
        transient_entry_block_minutes_provider: Callable[[str], float] | None = None,
        trade_logger: (
            Callable[[ExecutionResult, str, DecisionOutput, int | None], Awaitable[None]] | None
        ) = None,
        exchange_confirmed_checker: Callable[[ExecutionResult | None], bool] | None = None,
        exit_progress_checker: Callable[[ExecutionResult | None], bool] | None = None,
        no_exchange_position_result_checker: Callable[[ExecutionResult], bool] | None = None,
        trade_count_incrementer: Callable[[], None] | None = None,
        position_execution_persister: (
            Callable[[str, DecisionOutput, ExecutionResult, str], Awaitable[None]] | None
        ) = None,
        open_positions_execution_applier: (
            Callable[[list[dict[str, Any]], str, DecisionOutput, ExecutionResult], None] | None
        ) = None,
        decision_executed_marker: Callable[[int, float], Awaitable[None]] | None = None,
        market_no_opportunity_symbol_clearer: Callable[[str], None] | None = None,
        account_update_persister: (
            Callable[[str, str, ExecutionResult], Awaitable[None]] | None
        ) = None,
        account_balance_provider: Callable[[str], Awaitable[float]] | None = None,
        decision_outcome_marker: Callable[[int, str, float], Awaitable[None]] | None = None,
        entry_policy_evaluator: (
            Callable[
                [DecisionOutput, str, str, list[dict[str, Any]] | None],
                Awaitable[PolicyGateResult],
            ]
            | None
        ) = None,
        exit_policy_evaluator: (
            Callable[
                [DecisionOutput, str, list[dict[str, Any]] | None],
                Awaitable[PolicyGateResult],
            ]
            | None
        ) = None,
        execution_skills_provider: Callable[..., list[Any]] | None = None,
        execution_skills_attacher: Callable[..., None] | None = None,
        execution_skills_block_reason_provider: Callable[..., str | None] | None = None,
        position_reconciler: Callable[[str], Awaitable[None]] | None = None,
        open_positions_context_provider: (
            Callable[[], Awaitable[list[dict[str, Any]]]] | None
        ) = None,
        matching_exit_local_position_checker: (
            Callable[[list[dict[str, Any]], str, DecisionOutput], bool] | None
        ) = None,
        matching_exit_exchange_position_checker: (
            Callable[[str, DecisionOutput], Awaitable[bool | None]] | None
        ) = None,
        exit_cooldown_recorder: Callable[[str, DecisionOutput], None] | None = None,
        trade_notional_recorder: Callable[[float], None] | None = None,
    ) -> None:
        self.execution_lock = execution_lock
        self.risk_event_logger = risk_event_logger
        self.model_execution_mode_provider = model_execution_mode_provider
        self.decision_stage_recorder = decision_stage_recorder
        self.decision_reason_marker = decision_reason_marker
        self.decision_raw_response_marker = decision_raw_response_marker
        self.position_review_alert_context_provider = position_review_alert_context_provider
        self.position_review_risk_result_logger = position_review_risk_result_logger
        self.duplicate_decision_order_reason_provider = duplicate_decision_order_reason_provider
        self.okx_executor_provider = okx_executor_provider
        self.allocated_order_balance_provider = allocated_order_balance_provider
        self.rejected_execution_result_factory = rejected_execution_result_factory
        self.execution_leverage_summary_attacher = execution_leverage_summary_attacher
        self.execution_reason_provider = execution_reason_provider
        self.pending_execution_marker = pending_execution_marker
        self.untradable_exchange_error_checker = untradable_exchange_error_checker
        self.untradable_symbol_rememberer = untradable_symbol_rememberer
        self.transient_entry_exchange_error_checker = transient_entry_exchange_error_checker
        self.temporary_entry_block_rememberer = temporary_entry_block_rememberer
        self.transient_entry_block_minutes_provider = transient_entry_block_minutes_provider
        self.trade_logger = trade_logger
        self.exchange_confirmed_checker = exchange_confirmed_checker
        self.exit_progress_checker = exit_progress_checker
        self.no_exchange_position_result_checker = no_exchange_position_result_checker
        self.trade_count_incrementer = trade_count_incrementer
        self.position_execution_persister = position_execution_persister
        self.open_positions_execution_applier = open_positions_execution_applier
        self.decision_executed_marker = decision_executed_marker
        self.market_no_opportunity_symbol_clearer = market_no_opportunity_symbol_clearer
        self.account_update_persister = account_update_persister
        self.account_balance_provider = account_balance_provider
        self.decision_outcome_marker = decision_outcome_marker
        self.entry_policy_evaluator = entry_policy_evaluator
        self.exit_policy_evaluator = exit_policy_evaluator
        self.execution_skills_provider = execution_skills_provider
        self.execution_skills_attacher = execution_skills_attacher
        self.execution_skills_block_reason_provider = execution_skills_block_reason_provider
        self.position_reconciler = position_reconciler
        self.open_positions_context_provider = open_positions_context_provider
        self.matching_exit_local_position_checker = matching_exit_local_position_checker
        self.matching_exit_exchange_position_checker = matching_exit_exchange_position_checker
        self.exit_cooldown_recorder = exit_cooldown_recorder
        self.trade_notional_recorder = trade_notional_recorder

    def _required_execution_lock(self) -> AbstractAsyncContextManager[Any]:
        if self.execution_lock is None:
            raise RuntimeError("ExecutionService requires execution_lock dependency")
        return self.execution_lock

    def _required_risk_event_logger(self) -> Callable[..., Awaitable[None]]:
        if self.risk_event_logger is None:
            raise RuntimeError("ExecutionService requires risk_event_logger dependency")
        return self.risk_event_logger

    def _required_model_execution_mode_provider(self) -> Callable[[str], str]:
        if self.model_execution_mode_provider is None:
            raise RuntimeError("ExecutionService requires model_execution_mode_provider dependency")
        return self.model_execution_mode_provider

    def _required_decision_stage_recorder(self) -> Callable[..., Awaitable[dict[str, Any]]]:
        if self.decision_stage_recorder is None:
            raise RuntimeError("ExecutionService requires decision_stage_recorder dependency")
        return self.decision_stage_recorder

    def _required_decision_reason_marker(
        self,
    ) -> Callable[[int, str | None], Awaitable[None]]:
        if self.decision_reason_marker is None:
            raise RuntimeError("ExecutionService requires decision_reason_marker dependency")
        return self.decision_reason_marker

    def _required_decision_raw_response_marker(
        self,
    ) -> Callable[[int, dict[str, Any] | None], Awaitable[None]]:
        if self.decision_raw_response_marker is None:
            raise RuntimeError("ExecutionService requires decision_raw_response_marker dependency")
        return self.decision_raw_response_marker

    def _required_position_review_alert_context_provider(
        self,
    ) -> Callable[[DecisionOutput], dict[str, Any] | None]:
        if self.position_review_alert_context_provider is None:
            raise RuntimeError(
                "ExecutionService requires position_review_alert_context_provider dependency"
            )
        return self.position_review_alert_context_provider

    def _required_position_review_risk_result_logger(self) -> Callable[..., Awaitable[None]]:
        if self.position_review_risk_result_logger is None:
            raise RuntimeError(
                "ExecutionService requires position_review_risk_result_logger dependency"
            )
        return self.position_review_risk_result_logger

    def _required_duplicate_decision_order_reason_provider(
        self,
    ) -> Callable[[int, DecisionOutput], Awaitable[str | None]]:
        if self.duplicate_decision_order_reason_provider is None:
            raise RuntimeError(
                "ExecutionService requires duplicate_decision_order_reason_provider dependency"
            )
        return self.duplicate_decision_order_reason_provider

    def _required_okx_executor_provider(self) -> Callable[[str], Awaitable[Any]]:
        if self.okx_executor_provider is None:
            raise RuntimeError("ExecutionService requires okx_executor_provider dependency")
        return self.okx_executor_provider

    def _required_allocated_order_balance_provider(
        self,
    ) -> Callable[[str, DecisionOutput], Awaitable[float | None]]:
        if self.allocated_order_balance_provider is None:
            raise RuntimeError(
                "ExecutionService requires allocated_order_balance_provider dependency"
            )
        return self.allocated_order_balance_provider

    def _required_rejected_execution_result_factory(
        self,
    ) -> Callable[[DecisionOutput, str], ExecutionResult]:
        if self.rejected_execution_result_factory is None:
            raise RuntimeError(
                "ExecutionService requires rejected_execution_result_factory dependency"
            )
        return self.rejected_execution_result_factory

    def _required_execution_leverage_summary_attacher(
        self,
    ) -> Callable[[DecisionOutput, ExecutionResult, float], None]:
        if self.execution_leverage_summary_attacher is None:
            raise RuntimeError(
                "ExecutionService requires execution_leverage_summary_attacher dependency"
            )
        return self.execution_leverage_summary_attacher

    def _required_execution_reason_provider(
        self,
    ) -> Callable[[ExecutionResult | None], str]:
        if self.execution_reason_provider is None:
            raise RuntimeError("ExecutionService requires execution_reason_provider dependency")
        return self.execution_reason_provider

    def _required_pending_execution_marker(self) -> Callable[[int, str], Awaitable[None]]:
        if self.pending_execution_marker is None:
            raise RuntimeError("ExecutionService requires pending_execution_marker dependency")
        return self.pending_execution_marker

    def _required_untradable_exchange_error_checker(self) -> Callable[[str], bool]:
        if self.untradable_exchange_error_checker is None:
            raise RuntimeError(
                "ExecutionService requires untradable_exchange_error_checker dependency"
            )
        return self.untradable_exchange_error_checker

    def _required_untradable_symbol_rememberer(self) -> Callable[[str, str], None]:
        if self.untradable_symbol_rememberer is None:
            raise RuntimeError("ExecutionService requires untradable_symbol_rememberer dependency")
        return self.untradable_symbol_rememberer

    def _required_transient_entry_exchange_error_checker(self) -> Callable[[str], bool]:
        if self.transient_entry_exchange_error_checker is None:
            raise RuntimeError(
                "ExecutionService requires transient_entry_exchange_error_checker dependency"
            )
        return self.transient_entry_exchange_error_checker

    def _required_temporary_entry_block_rememberer(self) -> Callable[[str, str, float], None]:
        if self.temporary_entry_block_rememberer is None:
            raise RuntimeError(
                "ExecutionService requires temporary_entry_block_rememberer dependency"
            )
        return self.temporary_entry_block_rememberer

    def _required_transient_entry_block_minutes_provider(self) -> Callable[[str], float]:
        if self.transient_entry_block_minutes_provider is None:
            raise RuntimeError(
                "ExecutionService requires transient_entry_block_minutes_provider dependency"
            )
        return self.transient_entry_block_minutes_provider

    def _required_trade_logger(
        self,
    ) -> Callable[[ExecutionResult, str, DecisionOutput, int | None], Awaitable[None]]:
        if self.trade_logger is None:
            raise RuntimeError("ExecutionService requires trade_logger dependency")
        return self.trade_logger

    def _required_exchange_confirmed_checker(
        self,
    ) -> Callable[[ExecutionResult | None], bool]:
        if self.exchange_confirmed_checker is None:
            raise RuntimeError("ExecutionService requires exchange_confirmed_checker dependency")
        return self.exchange_confirmed_checker

    def _required_exit_progress_checker(self) -> Callable[[ExecutionResult | None], bool]:
        if self.exit_progress_checker is None:
            raise RuntimeError("ExecutionService requires exit_progress_checker dependency")
        return self.exit_progress_checker

    def _required_no_exchange_position_result_checker(
        self,
    ) -> Callable[[ExecutionResult], bool]:
        if self.no_exchange_position_result_checker is None:
            raise RuntimeError(
                "ExecutionService requires no_exchange_position_result_checker dependency"
            )
        return self.no_exchange_position_result_checker

    def _required_trade_count_incrementer(self) -> Callable[[], None]:
        if self.trade_count_incrementer is None:
            raise RuntimeError("ExecutionService requires trade_count_incrementer dependency")
        return self.trade_count_incrementer

    def _required_position_execution_persister(
        self,
    ) -> Callable[[str, DecisionOutput, ExecutionResult, str], Awaitable[None]]:
        if self.position_execution_persister is None:
            raise RuntimeError("ExecutionService requires position_execution_persister dependency")
        return self.position_execution_persister

    def _required_open_positions_execution_applier(
        self,
    ) -> Callable[[list[dict[str, Any]], str, DecisionOutput, ExecutionResult], None]:
        if self.open_positions_execution_applier is None:
            raise RuntimeError(
                "ExecutionService requires open_positions_execution_applier dependency"
            )
        return self.open_positions_execution_applier

    def _required_decision_executed_marker(self) -> Callable[[int, float], Awaitable[None]]:
        if self.decision_executed_marker is None:
            raise RuntimeError("ExecutionService requires decision_executed_marker dependency")
        return self.decision_executed_marker

    def _required_market_no_opportunity_symbol_clearer(self) -> Callable[[str], None]:
        if self.market_no_opportunity_symbol_clearer is None:
            raise RuntimeError(
                "ExecutionService requires market_no_opportunity_symbol_clearer dependency"
            )
        return self.market_no_opportunity_symbol_clearer

    def _required_account_update_persister(
        self,
    ) -> Callable[[str, str, ExecutionResult], Awaitable[None]]:
        if self.account_update_persister is None:
            raise RuntimeError("ExecutionService requires account_update_persister dependency")
        return self.account_update_persister

    def _required_account_balance_provider(self) -> Callable[[str], Awaitable[float]]:
        if self.account_balance_provider is None:
            raise RuntimeError("ExecutionService requires account_balance_provider dependency")
        return self.account_balance_provider

    def _required_decision_outcome_marker(
        self,
    ) -> Callable[[int, str, float], Awaitable[None]]:
        if self.decision_outcome_marker is None:
            raise RuntimeError("ExecutionService requires decision_outcome_marker dependency")
        return self.decision_outcome_marker

    def _required_entry_policy_evaluator(
        self,
    ) -> Callable[
        [DecisionOutput, str, str, list[dict[str, Any]] | None],
        Awaitable[PolicyGateResult],
    ]:
        if self.entry_policy_evaluator is None:
            raise RuntimeError("ExecutionService requires entry_policy_evaluator dependency")
        return self.entry_policy_evaluator

    def _required_exit_policy_evaluator(
        self,
    ) -> Callable[
        [DecisionOutput, str, list[dict[str, Any]] | None],
        Awaitable[PolicyGateResult],
    ]:
        if self.exit_policy_evaluator is None:
            raise RuntimeError("ExecutionService requires exit_policy_evaluator dependency")
        return self.exit_policy_evaluator

    def _required_execution_skills_provider(self) -> Callable[..., list[Any]]:
        if self.execution_skills_provider is None:
            raise RuntimeError("ExecutionService requires execution_skills_provider dependency")
        return self.execution_skills_provider

    def _required_execution_skills_attacher(self) -> Callable[..., None]:
        if self.execution_skills_attacher is None:
            raise RuntimeError("ExecutionService requires execution_skills_attacher dependency")
        return self.execution_skills_attacher

    def _required_execution_skills_block_reason_provider(
        self,
    ) -> Callable[..., str | None]:
        if self.execution_skills_block_reason_provider is None:
            raise RuntimeError(
                "ExecutionService requires execution_skills_block_reason_provider dependency"
            )
        return self.execution_skills_block_reason_provider

    def _required_position_reconciler(self) -> Callable[[str], Awaitable[None]]:
        if self.position_reconciler is None:
            raise RuntimeError("ExecutionService requires position_reconciler dependency")
        return self.position_reconciler

    def _required_open_positions_context_provider(
        self,
    ) -> Callable[[], Awaitable[list[dict[str, Any]]]]:
        if self.open_positions_context_provider is None:
            raise RuntimeError(
                "ExecutionService requires open_positions_context_provider dependency"
            )
        return self.open_positions_context_provider

    def _required_matching_exit_local_position_checker(
        self,
    ) -> Callable[[list[dict[str, Any]], str, DecisionOutput], bool]:
        if self.matching_exit_local_position_checker is None:
            raise RuntimeError(
                "ExecutionService requires matching_exit_local_position_checker dependency"
            )
        return self.matching_exit_local_position_checker

    def _required_matching_exit_exchange_position_checker(
        self,
    ) -> Callable[[str, DecisionOutput], Awaitable[bool | None]]:
        if self.matching_exit_exchange_position_checker is None:
            raise RuntimeError(
                "ExecutionService requires matching_exit_exchange_position_checker dependency"
            )
        return self.matching_exit_exchange_position_checker

    def _required_exit_cooldown_recorder(self) -> Callable[[str, DecisionOutput], None]:
        if self.exit_cooldown_recorder is None:
            raise RuntimeError("ExecutionService requires exit_cooldown_recorder dependency")
        return self.exit_cooldown_recorder

    def _required_trade_notional_recorder(self) -> Callable[[float], None]:
        if self.trade_notional_recorder is None:
            raise RuntimeError("ExecutionService requires trade_notional_recorder dependency")
        return self.trade_notional_recorder

    async def execute_candidate(
        self,
        symbol: str,
        model_name: str,
        decision: Any,
        assessment: Any,
        decision_db_id: int | None,
        results: dict[str, Any],
        *,
        open_positions: list[dict[str, Any]] | None = None,
        refresh_exit_positions: bool = True,
    ) -> ExecutionResult | None:
        async with self._required_execution_lock():
            return await self.execute_candidate_locked(
                symbol,
                model_name,
                decision,
                assessment,
                decision_db_id,
                results,
                open_positions=open_positions,
                refresh_exit_positions=refresh_exit_positions,
            )

    async def execute_candidate_locked(
        self,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        assessment,
        decision_db_id: int | None,
        results: dict[str, Any],
        open_positions: list[dict] | None = None,
        refresh_exit_positions: bool = True,
    ) -> ExecutionResult | None:
        log_risk_event = self._required_risk_event_logger()
        get_model_execution_mode = self._required_model_execution_mode_provider()
        record_decision_stage = self._required_decision_stage_recorder()
        mark_decision_reason = self._required_decision_reason_marker()
        mark_decision_raw_response = self._required_decision_raw_response_marker()
        position_review_alert_context = self._required_position_review_alert_context_provider()
        log_position_review_risk_result = self._required_position_review_risk_result_logger()
        duplicate_decision_order_reason = self._required_duplicate_decision_order_reason_provider()
        get_okx_executor = self._required_okx_executor_provider()
        allocated_order_balance = self._required_allocated_order_balance_provider()
        rejected_execution_result = self._required_rejected_execution_result_factory()
        attach_execution_leverage_summary = self._required_execution_leverage_summary_attacher()
        execution_reason_from_result = self._required_execution_reason_provider()
        mark_decision_pending_execution = self._required_pending_execution_marker()
        is_untradable_exchange_error = self._required_untradable_exchange_error_checker()
        remember_untradable_symbol = self._required_untradable_symbol_rememberer()
        is_transient_entry_exchange_error = self._required_transient_entry_exchange_error_checker()
        remember_temporary_entry_block = self._required_temporary_entry_block_rememberer()
        transient_entry_block_minutes = self._required_transient_entry_block_minutes_provider()
        log_trade = self._required_trade_logger()
        is_exchange_confirmed_execution = self._required_exchange_confirmed_checker()
        is_exit_progress_execution = self._required_exit_progress_checker()
        result_has_no_exchange_position = self._required_no_exchange_position_result_checker()
        increment_trade_count = self._required_trade_count_incrementer()
        persist_position_from_execution = self._required_position_execution_persister()
        apply_execution_to_open_positions = self._required_open_positions_execution_applier()
        mark_decision_executed = self._required_decision_executed_marker()
        clear_market_no_opportunity_symbol = self._required_market_no_opportunity_symbol_clearer()
        persist_account_update = self._required_account_update_persister()
        get_account_balance = self._required_account_balance_provider()
        mark_decision_outcome = self._required_decision_outcome_marker()
        evaluate_entry_policy = self._required_entry_policy_evaluator()
        evaluate_exit_policy = self._required_exit_policy_evaluator()
        execution_skills = self._required_execution_skills_provider()
        attach_execution_skills = self._required_execution_skills_attacher()
        execution_skills_block_reason = self._required_execution_skills_block_reason_provider()
        reconcile_positions = self._required_position_reconciler()
        get_open_positions_context = self._required_open_positions_context_provider()
        has_matching_local_exit_position = self._required_matching_exit_local_position_checker()
        has_matching_exchange_exit_position = (
            self._required_matching_exit_exchange_position_checker()
        )
        remember_exit_cooldown = self._required_exit_cooldown_recorder()
        record_trade_notional = self._required_trade_notional_recorder()
        for warning in assessment.warnings:
            results["warnings"].append(
                {
                    "model": model_name,
                    "symbol": symbol,
                    "warning": warning,
                }
            )
            await log_risk_event("warning", symbol, warning, model_name)

        execution_result = None
        model_mode = get_model_execution_mode(model_name)
        stage_started_at: dict[str, float] = {}
        submitted_to_exchange = False

        def attach_execution_parameters(source: str) -> None:
            raw_response = decision.raw_response if isinstance(decision.raw_response, dict) else {}
            raw_response = dict(raw_response)
            raw_response["execution_parameters"] = {
                "source": source,
                "action": decision.action.value,
                "position_size_pct": float(decision.position_size_pct or 0.0),
                "suggested_leverage": float(decision.suggested_leverage or 1.0),
                "stop_loss_pct": float(decision.stop_loss_pct or 0.0),
                "take_profit_pct": float(decision.take_profit_pct or 0.0),
            }
            decision.raw_response = raw_response

        def compact_execution_value(value: Any, depth: int = 0) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if depth >= 4:
                return safe_error_text(value, limit=300)
            if isinstance(value, dict):
                return {
                    str(key): compact_execution_value(child, depth + 1)
                    for key, child in list(value.items())[:40]
                }
            if isinstance(value, list):
                return [compact_execution_value(child, depth + 1) for child in value[:25]]
            return safe_error_text(value, limit=300)

        def attach_execution_result_snapshot(
            source: str,
            *,
            exchange_confirmed: bool = False,
            exit_progress: bool = False,
        ) -> None:
            if execution_result is None:
                return
            status = getattr(execution_result, "status", None)
            raw_response = decision.raw_response if isinstance(decision.raw_response, dict) else {}
            raw_response = dict(raw_response)
            raw_response["execution_result"] = {
                "source": source,
                "order_id": getattr(execution_result, "order_id", None),
                "exchange_order_id": getattr(execution_result, "exchange_order_id", None),
                "status": getattr(status, "value", status),
                "quantity": float(getattr(execution_result, "quantity", 0.0) or 0.0),
                "price": float(getattr(execution_result, "price", 0.0) or 0.0),
                "fee": float(getattr(execution_result, "fee", 0.0) or 0.0),
                "pnl": float(getattr(execution_result, "pnl", 0.0) or 0.0),
                "exchange_confirmed": bool(exchange_confirmed),
                "exit_progress": bool(exit_progress),
                "raw_response": compact_execution_value(
                    getattr(execution_result, "raw_response", None)
                ),
            }
            decision.raw_response = raw_response

        async def mark_stage(
            stage: str,
            status: str,
            reason: str,
            data: dict[str, Any] | None = None,
        ) -> None:
            now = perf_counter()
            started_at = stage_started_at.get(stage, now)
            if status == DecisionStageStatus.PENDING:
                stage_started_at[stage] = now
                duration_sec = 0.0
            else:
                duration_sec = now - started_at
                stage_started_at[stage] = now

            payload = dict(data or {})
            payload.setdefault("model_mode", model_mode)
            payload.setdefault("symbol", symbol)
            payload.setdefault("model_name", model_name)
            try:
                await record_decision_stage(
                    decision_db_id,
                    decision,
                    stage,
                    status,
                    reason,
                    payload,
                    duration_sec=duration_sec,
                )
            except TypeError:
                await record_decision_stage(
                    decision_db_id,
                    decision,
                    stage,
                    status,
                    reason,
                    payload,
                )

        async def mark_blocked(reason: str, data: dict[str, Any] | None = None) -> None:
            await mark_stage(
                DecisionStage.RISK_CHECK,
                DecisionStageStatus.BLOCKED,
                reason,
                data,
            )

        async def mark_skipped(reason: str, data: dict[str, Any] | None = None) -> None:
            await mark_stage(
                DecisionStage.RISK_CHECK,
                DecisionStageStatus.SKIPPED,
                reason,
                data,
            )

        async def block_before_submit(policy_result: PolicyGateResult) -> ExecutionResult:
            reason = str(policy_result.reason or "策略或风控检查未通过，未提交 OKX 订单。")
            blocker = str(policy_result.blocker or "policy_gate")
            data = {"blocker": blocker}
            if isinstance(policy_result.data, dict):
                data.update(policy_result.data)
            stage_status = str(data.get("stage_status") or "").lower()
            is_skipped = (
                stage_status == DecisionStageStatus.SKIPPED
                or bool(data.get("shadow_only"))
                or bool(data.get("skip_kind"))
            )
            if is_skipped:
                await mark_skipped(reason, data)
            else:
                await mark_blocked(reason, data)
            final_stage_status = (
                DecisionStageStatus.SKIPPED if is_skipped else DecisionStageStatus.BLOCKED
            )
            raw_response = decision.raw_response if isinstance(decision.raw_response, dict) else {}
            raw_response = dict(raw_response)
            if decision.is_entry:
                opportunity = raw_response.get("opportunity_score")
                if not isinstance(opportunity, dict):
                    opportunity = {}
                opportunity = dict(opportunity)
                opportunity["selected_for_execution"] = False
                opportunity["selection_reason"] = reason
                opportunity["execution_final_state"] = final_stage_status
                opportunity["execution_final_blocker"] = blocker
                raw_response["opportunity_score"] = opportunity
            raw_response.update(
                {
                    "execution_skipped": is_skipped,
                    "execution_policy_terminal": True,
                    "policy_blocker": blocker,
                    "stage_status": final_stage_status,
                    "skip_kind": data.get("skip_kind") or blocker,
                    "reason": reason,
                }
            )
            raw_response.update(data)
            decision.raw_response = raw_response
            attach_execution_parameters("policy_blocked")
            if decision_db_id is not None:
                await mark_decision_reason(decision_db_id, reason)
                await mark_decision_raw_response(decision_db_id, decision.raw_response)
            await log_risk_event(
                "warning",
                symbol,
                f"[{model_name}] {reason}",
                model_name,
            )
            if position_review_alert_context(decision):
                await log_position_review_risk_result(
                    decision,
                    model_name,
                    f"未执行：{reason}",
                )
            results["decisions"].append(
                {
                    "model": model_name,
                    "symbol": symbol,
                    "action": decision.action.value,
                    "approved": True,
                    "confidence": decision.confidence,
                    "executed": False,
                    "execution_status": "skipped",
                    "reason": reason,
                    "is_paper": (model_mode == "paper"),
                }
            )
            result = rejected_execution_result(decision, reason)
            result.raw_response = raw_response
            return result

        request_symbol = normalize_trading_symbol(symbol)
        decision_symbol = normalize_trading_symbol(decision.symbol)
        if request_symbol and decision_symbol and request_symbol != decision_symbol:
            reason = (
                "执行链交易对不一致，系统已在提交 OKX 前拦截："
                f"流程交易对 {request_symbol}，决策交易对 {decision_symbol}。"
            )
            return await block_before_submit(
                PolicyGateResult.block(
                    "execution_symbol_mismatch",
                    reason,
                    {
                        "request_symbol": symbol,
                        "decision_symbol": decision.symbol,
                        "normalized_request_symbol": request_symbol,
                        "normalized_decision_symbol": decision_symbol,
                    },
                )
            )

        arbitration = arbitrate_decision(decision)
        await mark_stage(
            DecisionStage.STRATEGY_ARBITRATION,
            arbitration.status,
            arbitration.reason,
            arbitration.data,
        )
        if decision.is_entry or decision.is_exit:
            await mark_stage(
                DecisionStage.RISK_CHECK,
                DecisionStageStatus.PENDING,
                "已进入执行前严重风险检查。",
                {"mode": model_mode},
            )
        if decision_db_id is not None:
            duplicate_reason = await duplicate_decision_order_reason(decision_db_id, decision)
            if duplicate_reason:
                return await block_before_submit(
                    PolicyGateResult.block(
                        "duplicate_decision_order",
                        duplicate_reason,
                    )
                )

        if decision.is_exit:
            exit_policy_result = await evaluate_exit_policy(
                decision,
                model_name,
                open_positions,
                refresh_positions=refresh_exit_positions,
            )
            if not exit_policy_result.passed:
                return await block_before_submit(exit_policy_result)

        if decision.is_entry:
            entry_policy_result = await evaluate_entry_policy(
                decision,
                model_name,
                model_mode,
                open_positions,
            )
            if not entry_policy_result.passed:
                return await block_before_submit(entry_policy_result)
            if decision_db_id is not None:
                attach_execution_parameters("entry_policy_passed")
                await mark_decision_raw_response(decision_db_id, decision.raw_response)

        if decision.is_entry or decision.is_exit:
            await mark_stage(
                DecisionStage.RISK_CHECK,
                DecisionStageStatus.PASSED,
                "执行前严重风险检查通过，进入交易所提交阶段。",
                {"mode": model_mode},
            )
            await mark_stage(
                DecisionStage.EXCHANGE_SUBMIT,
                DecisionStageStatus.PENDING,
                "正在提交 OKX 订单并等待交易所返回结果。",
                {"mode": model_mode},
            )

        override_balance = None
        try:
            executor = await get_okx_executor(model_mode)
            ai_requested_leverage = float(decision.suggested_leverage or 1.0)
            override_balance = await allocated_order_balance(model_mode, decision)
            execution_agent_skills = execution_skills(
                decision=decision,
                model_mode=model_mode,
                override_balance=override_balance,
            )
            if execution_agent_skills:
                attach_execution_skills(
                    decision,
                    phase="execution_precheck",
                    skills=execution_agent_skills,
                    note="提交 OKX 前的 Agent/Skills 执行守门。",
                )
            if decision_db_id is not None:
                attach_execution_parameters("execution_precheck")
                await mark_decision_raw_response(decision_db_id, decision.raw_response)
            execution_guard_reason = (
                execution_skills_block_reason(execution_agent_skills, for_entry=True)
                if AGENT_SKILLS_TRADING_EFFECTS_ENABLED
                else None
            )
            if decision.is_entry and execution_guard_reason:
                await mark_stage(
                    DecisionStage.RISK_CHECK,
                    DecisionStageStatus.BLOCKED,
                    execution_guard_reason,
                    {"blocker": "execution_agent_skills"},
                )
                await mark_stage(
                    DecisionStage.EXCHANGE_SUBMIT,
                    DecisionStageStatus.SKIPPED,
                    "执行前守门模块拦截，未向 OKX 提交订单。",
                    {"blocker": "execution_agent_skills"},
                )
                execution_result = rejected_execution_result(
                    decision,
                    execution_guard_reason,
                )
            else:
                execution_timeout = 90.0 if decision.is_exit else 60.0
                submitted_to_exchange = True
                execution_result = await asyncio.wait_for(
                    executor.place_order(
                        decision,
                        account_id=model_name,
                        override_balance=override_balance,
                    ),
                    timeout=execution_timeout,
                )
            if submitted_to_exchange and (decision.is_entry or decision.is_exit):
                await mark_stage(
                    DecisionStage.EXCHANGE_SUBMIT,
                    DecisionStageStatus.PASSED,
                    "OKX 提交阶段已返回订单响应，进入成交确认阶段。",
                    {
                        "has_execution_result": execution_result is not None,
                        "status": getattr(getattr(execution_result, "status", None), "value", None),
                        "exchange_order_id": getattr(execution_result, "exchange_order_id", None),
                    },
                )
            if decision.is_entry and execution_result is not None:
                attach_execution_leverage_summary(
                    decision,
                    execution_result,
                    ai_requested_leverage,
                )
                attach_execution_parameters("exchange_result")
        except TimeoutError:
            logger.error(
                "decision execution timed out",
                model=model_name,
                symbol=symbol,
                action=decision.action.value,
                mode=model_mode,
            )
            execution_result = rejected_execution_result(
                decision,
                (
                    "OKX 下单或确认超时，系统没有拿到最终订单结果；"
                    "本轮按未执行处理，下一轮会继续复盘该仓位。"
                ),
            )
            if submitted_to_exchange:
                await mark_stage(
                    DecisionStage.EXCHANGE_SUBMIT,
                    DecisionStageStatus.FAILED,
                    "OKX 下单请求超时，未拿到交易所明确接收结果。",
                    {"error_type": "timeout"},
                )
            await mark_stage(
                DecisionStage.EXCHANGE_CONFIRM,
                DecisionStageStatus.FAILED,
                execution_reason_from_result(execution_result),
                {"error_type": "timeout"},
            )
            await log_risk_event(
                "warning",
                symbol,
                f"[{model_name}] OKX execution timed out",
                model_name,
            )
        except Exception as e:
            error_text = safe_error_text(e, limit=180)
            logger.error(
                "decision execution failed",
                model=model_name,
                symbol=symbol,
                action=decision.action.value,
                mode=model_mode,
                error=error_text,
            )
            execution_result = rejected_execution_result(decision, error_text)
            await mark_stage(
                DecisionStage.EXCHANGE_SUBMIT,
                DecisionStageStatus.FAILED,
                execution_reason_from_result(execution_result),
                {"error_type": "exception"},
            )
            await log_risk_event(
                "warning",
                symbol,
                f"[{model_name}] OKX execution failed: {error_text}",
                model_name,
            )

        if execution_result is None and decision.is_exit:
            retry_intro = (
                "平仓裁决已生成，但第一次提交没有返回 OKX 订单结果；"
                "系统立即同步 OKX 仓位并重试一次平仓，避免错过平仓时机。"
            )
            await mark_stage(
                DecisionStage.EXCHANGE_CONFIRM,
                DecisionStageStatus.FAILED,
                retry_intro,
                {"retry": "exit_missing_execution_result"},
            )
            if decision_db_id is not None:
                await mark_decision_pending_execution(decision_db_id, retry_intro)
            await log_risk_event(
                "warning",
                symbol,
                f"[{model_name}] {retry_intro}",
                model_name,
            )
            await reconcile_positions("exit missing execution result")
            exit_positions = await get_open_positions_context()
            if open_positions is not None:
                open_positions[:] = exit_positions
            local_has_position = has_matching_local_exit_position(
                exit_positions, model_name, decision
            )
            exchange_has_position = await has_matching_exchange_exit_position(
                model_name,
                decision,
            )
            if exchange_has_position is None and not local_has_position:
                execution_result = rejected_execution_result(
                    decision,
                    (
                        "平仓裁决已生成，但第一次提交没有返回订单结果；系统同步本地持仓后，"
                        "OKX 持仓快照暂时不可用，无法确认是否仍有可平仓仓位。"
                        "为避免把查询失败误判为无仓或重复提交平仓单，本轮等待下一轮同步确认。"
                    ),
                )
            elif local_has_position or exchange_has_position is True:
                try:
                    retry_executor = await get_okx_executor(model_mode)
                    execution_result = await asyncio.wait_for(
                        retry_executor.place_order(
                            decision,
                            account_id=model_name,
                            override_balance=override_balance,
                        ),
                        timeout=45.0,
                    )
                    if execution_result is not None:
                        raw = (
                            execution_result.raw_response
                            if isinstance(execution_result.raw_response, dict)
                            else {}
                        )
                        raw["exit_missing_result_retry"] = True
                        raw["retry_reason"] = retry_intro
                        execution_result.raw_response = raw
                except TimeoutError:
                    execution_result = rejected_execution_result(
                        decision,
                        (
                            "平仓重试仍然超时：系统已同步 OKX 仓位并重新提交平仓，"
                            "但 45 秒内仍没有拿到订单结果。请以 OKX 当前仓位和委托状态为准；"
                            "下一轮持仓复盘会继续优先处理该仓位。"
                        ),
                    )
                except Exception as e:
                    error_text = safe_error_text(e, limit=180)
                    execution_result = rejected_execution_result(
                        decision,
                        (
                            "平仓重试失败：第一次提交没有返回订单结果，系统同步 OKX 仓位后已尝试重提，"
                            f"但交易接口返回错误：{error_text}"
                        ),
                    )
            else:
                execution_result = rejected_execution_result(
                    decision,
                    (
                        "平仓裁决已生成，但第一次提交没有返回订单结果；系统随即同步 OKX 仓位，"
                        "发现本地和 OKX 都已经没有该方向可平仓位，因此没有重复提交平仓单。"
                    ),
                )

            if execution_result is None:
                execution_result = rejected_execution_result(
                    decision,
                    (
                        "平仓重试后交易接口仍未返回执行结果。系统已避免把该状态继续标记为等待；"
                        "下一轮持仓复盘会再次检查 OKX 实际仓位并重新处理。"
                    ),
                )

        missing_result_reason = None
        if execution_result:
            result_text = " ".join(
                str(part or "")
                for part in (
                    execution_result.raw_response,
                    execution_result.exchange_order_id,
                    execution_result.status.value,
                )
            )
            transient_entry_exchange_error = bool(
                decision.is_entry and is_transient_entry_exchange_error(result_text)
            )
            if decision.is_entry and is_untradable_exchange_error(result_text):
                remember_untradable_symbol(symbol, result_text)
            elif decision.is_exit and is_untradable_exchange_error(result_text):
                raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                raw["untradable_exit_execution_error"] = {"reason": result_text[:1000]}
                decision.raw_response = raw
                remember_exit_cooldown(model_name, decision)
            elif transient_entry_exchange_error:
                remember_temporary_entry_block(
                    symbol,
                    result_text,
                    transient_entry_block_minutes(result_text),
                )
            await log_trade(execution_result, model_name, decision, decision_db_id)
            exchange_confirmed = is_exchange_confirmed_execution(execution_result)
            exit_progress = is_exit_progress_execution(execution_result)
            confirm_reason = execution_reason_from_result(execution_result)
            if exchange_confirmed:
                await mark_stage(
                    DecisionStage.EXCHANGE_CONFIRM,
                    DecisionStageStatus.COMPLETED,
                    "OKX 已返回有效订单号并确认成交。",
                    {
                        "order_id": execution_result.order_id,
                        "exchange_order_id": execution_result.exchange_order_id,
                        "status": execution_result.status.value,
                        "price": execution_result.price,
                        "quantity": execution_result.quantity,
                    },
                )
            elif exit_progress:
                await mark_stage(
                    DecisionStage.EXCHANGE_CONFIRM,
                    DecisionStageStatus.PENDING,
                    confirm_reason,
                    {
                        "order_id": execution_result.order_id,
                        "exchange_order_id": execution_result.exchange_order_id,
                        "status": execution_result.status.value,
                    },
                )
            else:
                await mark_stage(
                    DecisionStage.EXCHANGE_CONFIRM,
                    (
                        DecisionStageStatus.SKIPPED
                        if transient_entry_exchange_error
                        else DecisionStageStatus.FAILED
                    ),
                    confirm_reason,
                    {
                        "order_id": execution_result.order_id,
                        "exchange_order_id": execution_result.exchange_order_id,
                        "status": execution_result.status.value,
                        "error_type": (
                            "transient_exchange_error"
                            if transient_entry_exchange_error
                            else "execution_not_confirmed"
                        ),
                    },
                )
            if (
                decision.is_exit
                and not exchange_confirmed
                and result_has_no_exchange_position(execution_result)
            ):
                await reconcile_positions("exit no-position result")
                if open_positions is not None:
                    open_positions[:] = await get_open_positions_context()
            if exchange_confirmed or exit_progress:
                increment_trade_count()
                await persist_position_from_execution(
                    model_name,
                    decision,
                    execution_result,
                    model_mode,
                )
                if open_positions is not None:
                    apply_execution_to_open_positions(
                        open_positions,
                        model_name,
                        decision,
                        execution_result,
                    )
                if decision.is_exit:
                    remember_exit_cooldown(model_name, decision)
                await mark_stage(
                    DecisionStage.LOCAL_SYNC,
                    DecisionStageStatus.COMPLETED,
                    "成交结果已写入本地订单/持仓记录。",
                    {
                        "exit_progress": bool(exit_progress),
                        "exchange_confirmed": bool(exchange_confirmed),
                    },
                )
            else:
                await mark_stage(
                    DecisionStage.LOCAL_SYNC,
                    DecisionStageStatus.SKIPPED,
                    "交易所未确认成交，本地未改动持仓。",
                    {
                        "exchange_confirmed": bool(exchange_confirmed),
                        "exit_progress": bool(exit_progress),
                    },
                )
            results["executions"].append(
                {
                    "model": model_name,
                    "symbol": symbol,
                    "action": decision.action.value,
                    "order_id": execution_result.order_id,
                    "status": execution_result.status.value,
                    "quantity": execution_result.quantity,
                    "price": execution_result.price,
                    "is_paper": (model_mode == "paper"),
                }
            )
            if exchange_confirmed:
                record_trade_notional(execution_result.price * execution_result.quantity)
            if decision_db_id is not None and exchange_confirmed:
                await mark_decision_executed(decision_db_id, execution_result.price)
                await mark_decision_reason(decision_db_id, confirm_reason)
                attach_execution_parameters("exchange_confirmed")
                attach_execution_result_snapshot(
                    "exchange_confirmed",
                    exchange_confirmed=exchange_confirmed,
                    exit_progress=exit_progress,
                )
                await mark_decision_raw_response(decision_db_id, decision.raw_response)
                if decision.is_entry:
                    clear_market_no_opportunity_symbol(symbol)
            elif decision_db_id is not None:
                await mark_decision_reason(
                    decision_db_id,
                    execution_reason_from_result(execution_result),
                )
                attach_execution_parameters("exchange_not_confirmed")
                attach_execution_result_snapshot(
                    "exchange_not_confirmed",
                    exchange_confirmed=exchange_confirmed,
                    exit_progress=exit_progress,
                )
                await mark_decision_raw_response(decision_db_id, decision.raw_response)
            if model_mode != "paper" and decision.is_exit and execution_result.pnl != 0.0:
                await persist_account_update(model_name, decision.model_name, execution_result)
                if decision_db_id is not None:
                    balance = await get_account_balance(model_name)
                    pnl_pct = execution_result.pnl / balance if balance > 0 else 0.0
                    outcome = (
                        "profit"
                        if execution_result.pnl > 0
                        else ("loss" if execution_result.pnl < 0 else "flat")
                    )
                    await mark_decision_outcome(decision_db_id, outcome, pnl_pct)
            if position_review_alert_context(decision):
                await log_position_review_risk_result(
                    decision,
                    model_name,
                    execution_result=execution_result,
                )
        else:
            missing_result_reason = (
                "交易接口未返回执行结果，系统没有拿到 OKX 订单号，也没有生成本地订单；"
                "本次裁决已按未执行处理。"
            )
            await mark_stage(
                DecisionStage.EXCHANGE_CONFIRM,
                DecisionStageStatus.FAILED,
                missing_result_reason,
                {"error_type": "missing_execution_result"},
            )
            await mark_stage(
                DecisionStage.LOCAL_SYNC,
                DecisionStageStatus.SKIPPED,
                "没有成交结果，本地持仓未改动。",
                {"error_type": "missing_execution_result"},
            )
            if decision_db_id is not None:
                await mark_decision_reason(decision_db_id, missing_result_reason)
                attach_execution_parameters("missing_execution_result")
                await mark_decision_raw_response(decision_db_id, decision.raw_response)
            if position_review_alert_context(decision):
                await log_position_review_risk_result(
                    decision,
                    model_name,
                    f"未执行：{missing_result_reason}",
                )

        results["decisions"].append(
            {
                "model": model_name,
                "symbol": symbol,
                "action": decision.action.value,
                "approved": True,
                "confidence": decision.confidence,
                "executed": is_exchange_confirmed_execution(execution_result),
                "execution_status": execution_result.status.value if execution_result else None,
                "reason": missing_result_reason,
                "is_paper": (model_mode == "paper"),
            }
        )
        return execution_result
