"""Execution service boundary.

ExecutionService owns serialized execution and the order-submit state machine.
TradingService remains the orchestrator and dependency provider, but the
submit/confirm/local-sync flow physically lives here.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, suppress
from time import perf_counter
from typing import Any

import structlog

from ai_brain.base_model import DecisionOutput
from core.safe_output import safe_error_text
from core.symbols import normalize_trading_symbol
from executor.base_executor import ExecutionResult
from services.decision_state import DecisionStage, DecisionStageStatus
from services.okx_error_classifier import is_okx_temporary_service_error
from services.paper_bootstrap_canary import PaperBootstrapCanaryPolicy
from services.paper_exploration import (
    assess_paper_exploration_entry,
    is_paper_exploration_decision,
)
from services.paper_training import (
    assess_paper_training_entry,
    attach_paper_training_order_identity,
    is_paper_training_decision,
)
from services.strategy_arbitration import arbitrate_decision
from services.trade_execution_contract import build_live_rules_canary_entry_contract
from services.trade_recommendation_contract import attach_trade_execution_result
from services.trading_policies import PolicyGateResult

logger = structlog.get_logger(__name__)

AGENT_SKILLS_TRADING_EFFECTS_ENABLED = True


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _governance_complete(value: dict[str, Any]) -> bool:
    return all(
        value.get(key) not in {None, ""}
        for key in (
            "source",
            "observation_window",
            "sample_count",
            "generated_at",
            "strategy_version",
        )
    )


def _rules_canary_entry_contract_result(
    decision: DecisionOutput,
    model_mode: str = "",
) -> PolicyGateResult | None:
    """Allow bounded live rules-canary entries before model promotion."""

    if str(model_mode or "").lower() == "paper":
        return None
    raw = _safe_dict(decision.raw_response)
    gate = _safe_dict(raw.get("production_trade_gate"))
    if gate.get("mode") != "live_rules_canary":
        return None

    contract, blockers = build_live_rules_canary_entry_contract(
        raw,
        entry_action=decision.action,
    )
    if blockers:
        return PolicyGateResult.block(
            "live_rules_canary_contract_incomplete",
            "Live rules canary contract is incomplete; entry fails closed.",
            {
                "stage_status": "blocked",
                "block_reasons": blockers,
                "production_trade_gate": gate,
                "live_rules_canary_contract": contract,
            },
        )
    return PolicyGateResult.allow(
        {
            "return_execution_contract": "live_rules_canary",
            "production_permission": True,
            "production_trade_gate": gate,
            "live_rules_canary_contract": contract,
        }
    )


def _return_entry_contract_result(
    decision: DecisionOutput,
    model_mode: str = "",
) -> PolicyGateResult:
    """Fail closed unless the dynamic fee-after-return contract is complete."""

    if not decision.is_entry:
        return PolicyGateResult.allow({"return_execution_contract": "not_entry"})

    rules_canary_result = _rules_canary_entry_contract_result(decision, model_mode)
    if rules_canary_result is not None:
        return rules_canary_result

    if PaperBootstrapCanaryPolicy.is_claimed(decision):
        assessment = PaperBootstrapCanaryPolicy.assess(decision, model_mode)
        if assessment.eligible:
            return PolicyGateResult.allow(
                {
                    "return_execution_contract": "paper_bootstrap_canary",
                    "production_permission": False,
                    "paper_bootstrap_canary": assessment.details,
                }
            )
        return PolicyGateResult.block(
            "paper_bootstrap_canary_contract_incomplete",
            assessment.reason,
            {
                "stage_status": "blocked",
                "paper_bootstrap_canary": assessment.details,
            },
        )

    if is_paper_exploration_decision(decision):
        assessment = assess_paper_exploration_entry(decision, model_mode)
        if assessment.eligible:
            return PolicyGateResult.allow(
                {
                    "return_execution_contract": "paper_exploration",
                    "production_permission": False,
                    "paper_exploration": assessment.to_dict(),
                }
            )
        return PolicyGateResult.block(
            "paper_exploration_contract_incomplete",
            assessment.reason,
            {
                "stage_status": "blocked",
                "paper_exploration": assessment.to_dict(),
            },
        )

    if is_paper_training_decision(decision):
        assessment = assess_paper_training_entry(decision, model_mode)
        if assessment.eligible:
            return PolicyGateResult.allow(
                {
                    "return_execution_contract": "paper_training",
                    "production_permission": False,
                    "paper_training": assessment.to_dict(),
                }
            )
        return PolicyGateResult.block(
            "paper_training_contract_incomplete",
            assessment.reason,
            {
                "stage_status": "blocked",
                "paper_training": assessment.to_dict(),
            },
        )

    raw = _safe_dict(decision.raw_response)
    candidate = _safe_dict(raw.get("authoritative_return_candidate"))
    side_evidence = _safe_dict(candidate.get("side_evidence"))
    opportunity = _safe_dict(raw.get("opportunity_score"))
    execution_cost = _safe_dict(opportunity.get("execution_cost"))
    pre_order_facts = _safe_dict(raw.get("pre_order_execution_facts"))
    cost_sizing_pass = _safe_dict(raw.get("execution_cost_sizing_pass"))
    sizing = _safe_dict(raw.get("profit_risk_sizing"))
    candidate_provenance = _safe_dict(side_evidence.get("policy_provenance"))
    sizing_provenance = _safe_dict(sizing.get("policy_provenance"))
    expected_net = _safe_float(side_evidence.get("expected_net_return_pct"), 0.0)
    return_lcb = _safe_float(side_evidence.get("return_lcb_pct"), 0.0)
    source_count = int(_safe_float(side_evidence.get("production_source_count"), 0.0))
    risk_budget = _safe_float(sizing.get("risk_budget_usdt"), 0.0)
    planned_loss = _safe_float(sizing.get("planned_stressed_loss_usdt"), 0.0)
    stress_fraction = _safe_float(sizing.get("stressed_loss_fraction"), 0.0)
    target_notional = _safe_float(sizing.get("target_notional_usdt"), 0.0)
    final_notional = _safe_float(sizing.get("final_notional_usdt"), 0.0)
    final_margin = _safe_float(sizing.get("final_margin_usdt"), 0.0)
    available_margin = _safe_float(sizing.get("available_margin_usdt"), 0.0)
    reasons: list[str] = []
    if candidate.get("production_eligible") is not True:
        reasons.append("authoritative_return_candidate_not_eligible")
    if side_evidence.get("production_eligible") is not True:
        reasons.append("selected_side_return_not_eligible")
    if expected_net <= 0 or return_lcb <= 0:
        reasons.append("fee_after_return_lower_bound_not_positive")
    if source_count <= 0:
        reasons.append("authoritative_return_samples_missing")
    if not _governance_complete(candidate_provenance):
        reasons.append("return_policy_provenance_incomplete")
    if sizing.get("production_eligible") is not True:
        reasons.append("dynamic_position_sizing_not_eligible")
    if execution_cost.get("order_size_complete") is not True:
        reasons.append("order_size_execution_cost_incomplete")
    if _safe_float(execution_cost.get("order_notional_usdt"), 0.0) + 1e-8 < final_notional:
        reasons.append("execution_cost_notional_below_final_order_notional")
    if pre_order_facts.get("production_eligible") is not True:
        reasons.append("pre_order_execution_facts_ineligible")
    if not str(pre_order_facts.get("input_fingerprint") or "").strip():
        reasons.append("pre_order_execution_facts_fingerprint_missing")
    if cost_sizing_pass.get("order_size_complete") is not True:
        reasons.append("order_size_sizing_pass_incomplete")
    if _safe_float(decision.position_size_pct, 0.0) <= 0:
        reasons.append("dynamic_position_size_zero")
    if not _governance_complete(sizing_provenance):
        reasons.append("sizing_policy_provenance_incomplete")
    if not str(sizing_provenance.get("contract_fingerprint") or "").strip():
        reasons.append("sizing_contract_fingerprint_missing")
    if risk_budget <= 0 or planned_loss <= 0 or planned_loss > risk_budget + 1e-8:
        reasons.append("sizing_risk_budget_invalid")
    if stress_fraction <= 0 or not math.isclose(
        planned_loss,
        final_notional * stress_fraction,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        reasons.append("sizing_stressed_loss_algebra_mismatch")
    if final_notional <= 0 or final_notional > target_notional + 1e-8:
        reasons.append("sizing_notional_target_invalid")
    if final_margin <= 0 or available_margin <= 0 or not math.isclose(
        final_margin,
        final_notional / max(_safe_float(decision.suggested_leverage, 1.0), 1.0),
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        reasons.append("sizing_margin_algebra_mismatch")
    if reasons:
        return PolicyGateResult.block(
            "dynamic_return_execution_contract_incomplete",
            "Dynamic fee-after-return execution contract is incomplete; entry fails closed.",
            {
                "stage_status": "blocked",
                "block_reasons": reasons,
                "expected_net_return_pct": expected_net,
                "return_lcb_pct": return_lcb,
                "production_source_count": source_count,
            },
        )
    return PolicyGateResult.allow(
        {
            "return_execution_contract": "complete",
            "expected_net_return_pct": expected_net,
            "return_lcb_pct": return_lcb,
            "production_source_count": source_count,
        }
    )


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
        position_protection_rebalancer: (
            Callable[[Any, DecisionOutput], Awaitable[dict[str, Any]]] | None
        ) = None,
        order_fact_recovery_trigger: Callable[[str], None] | None = None,
        open_positions_execution_applier: (
            Callable[[list[dict[str, Any]], str, DecisionOutput, ExecutionResult], None] | None
        ) = None,
        decision_executed_marker: Callable[[int, float], Awaitable[None]] | None = None,
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
        trade_notional_recorder: Callable[[float], None] | None = None,
        production_trade_gate_provider: (
            Callable[
                [DecisionOutput, str, str, list[dict[str, Any]] | None],
                Awaitable[dict[str, Any] | None],
            ]
            | None
        ) = None,
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
        self.trade_logger = trade_logger
        self.exchange_confirmed_checker = exchange_confirmed_checker
        self.exit_progress_checker = exit_progress_checker
        self.no_exchange_position_result_checker = no_exchange_position_result_checker
        self.trade_count_incrementer = trade_count_incrementer
        self.position_execution_persister = position_execution_persister
        self.position_protection_rebalancer = position_protection_rebalancer
        self.order_fact_recovery_trigger = order_fact_recovery_trigger
        self.open_positions_execution_applier = open_positions_execution_applier
        self.decision_executed_marker = decision_executed_marker
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
        self.trade_notional_recorder = trade_notional_recorder
        self.production_trade_gate_provider = production_trade_gate_provider

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

    def _required_position_protection_rebalancer(
        self,
    ) -> Callable[[Any, DecisionOutput], Awaitable[dict[str, Any]]]:
        if self.position_protection_rebalancer is None:
            raise RuntimeError(
                "ExecutionService requires position_protection_rebalancer dependency"
            )
        return self.position_protection_rebalancer

    def _trigger_order_fact_recovery(self, execution_mode: str) -> bool:
        """Request a non-blocking OKX fill recovery after local order persistence fails."""

        trigger = self.order_fact_recovery_trigger
        if trigger is None:
            return False
        try:
            trigger(execution_mode)
            return True
        except Exception as exc:  # pragma: no cover - defensive recovery boundary
            logger.warning(
                "failed to request order fact recovery",
                execution_mode=execution_mode,
                error=safe_error_text(exc, limit=180),
            )
            return False

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
        log_trade = self._required_trade_logger()
        is_exchange_confirmed_execution = self._required_exchange_confirmed_checker()
        is_exit_progress_execution = self._required_exit_progress_checker()
        result_has_no_exchange_position = self._required_no_exchange_position_result_checker()
        increment_trade_count = self._required_trade_count_incrementer()
        persist_position_from_execution = self._required_position_execution_persister()
        rebalance_position_protection = self._required_position_protection_rebalancer()
        apply_execution_to_open_positions = self._required_open_positions_execution_applier()
        mark_decision_executed = self._required_decision_executed_marker()
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

        def okx_exit_mismatch_summary(raw: Any) -> dict[str, Any] | None:
            if not isinstance(raw, dict):
                return None
            mismatch = raw.get("okx_exit_position_mismatch")
            if not isinstance(mismatch, dict):
                return None
            candidates = mismatch.get("candidates")
            if not isinstance(candidates, list):
                candidates = []
            return {
                "source": mismatch.get("source"),
                "decision_symbol": mismatch.get("decision_symbol"),
                "expected_okx_inst_id": mismatch.get("expected_okx_inst_id"),
                "okx_symbol": mismatch.get("okx_symbol"),
                "target_position_side": mismatch.get("target_position_side"),
                "exit_order_side": mismatch.get("exit_order_side"),
                "positions_returned": mismatch.get("positions_returned"),
                "matching_position_count": mismatch.get("matching_position_count"),
                "matching_contracts_total": mismatch.get("matching_contracts_total"),
                "nonzero_same_symbol_sides": mismatch.get("nonzero_same_symbol_sides"),
                "candidate_reasons": [
                    {
                        "symbol": item.get("symbol"),
                        "raw_symbol": item.get("raw_symbol"),
                        "side": item.get("side"),
                        "contracts": item.get("contracts"),
                        "reason": item.get("reason"),
                    }
                    for item in candidates[:5]
                    if isinstance(item, dict)
                ],
            }

        def attach_execution_result_snapshot(
            source: str,
            *,
            exchange_confirmed: bool = False,
            exit_progress: bool = False,
        ) -> None:
            if execution_result is None:
                return
            status = getattr(execution_result, "status", None)
            execution_raw = getattr(execution_result, "raw_response", None)
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
                "okx_exit_position_mismatch_summary": okx_exit_mismatch_summary(
                    execution_raw
                ),
                "raw_response": compact_execution_value(execution_raw),
            }
            decision.raw_response = raw_response
            if model_mode == "paper":
                attach_trade_execution_result(
                    decision,
                    execution_result,
                    source=source,
                    exchange_confirmed=exchange_confirmed,
                    exit_progress=exit_progress,
                )

        async def await_exchange_place_order(
            executor: Any,
            *,
            timeout_seconds: float,
            retry: bool = False,
        ) -> ExecutionResult | None:
            deadline = perf_counter() + max(float(timeout_seconds), 0.0)
            order_task = asyncio.create_task(
                executor.place_order(
                    decision,
                    account_id=model_name,
                    override_balance=override_balance,
                )
            )
            try:
                return await asyncio.wait_for(
                    asyncio.shield(order_task),
                    timeout=max(deadline - perf_counter(), 0.001),
                )
            except asyncio.CancelledError:
                if order_task.done():
                    return order_task.result()
                logger.warning(
                    "exchange place_order shielded from outer cancellation",
                    model=model_name,
                    symbol=symbol,
                    action=decision.action.value,
                    mode=model_mode,
                    retry=retry,
                )
                return await asyncio.wait_for(
                    asyncio.shield(order_task),
                    timeout=max(deadline - perf_counter(), 0.001),
                )
            except TimeoutError:
                if not order_task.done():
                    order_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await order_task
                raise

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
            if model_mode == "paper":
                attach_trade_execution_result(
                    decision,
                    result,
                    source=blocker,
                    exchange_confirmed=False,
                )
                if decision_db_id is not None:
                    await mark_decision_raw_response(decision_db_id, decision.raw_response)
            result.raw_response = decision.raw_response
            return result

        async def policy_evaluation_failed_result(
            *,
            blocker: str,
            reason: str,
            error_type: str,
            error: str | None = None,
        ) -> ExecutionResult:
            data: dict[str, Any] = {
                "blocker": blocker,
                "error_type": error_type,
                "mode": model_mode,
            }
            if error:
                data["error"] = error
            await mark_stage(
                DecisionStage.RISK_CHECK,
                DecisionStageStatus.FAILED,
                reason,
                data,
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
                opportunity["execution_final_state"] = DecisionStageStatus.FAILED
                opportunity["execution_final_blocker"] = blocker
                raw_response["opportunity_score"] = opportunity
            review = raw_response.get("high_risk_review")
            if isinstance(review, dict) and str(review.get("status") or "") == "pending":
                review = dict(review)
                review.update(
                    {
                        "status": (
                            "cancelled_blocked"
                            if error_type == "cancelled"
                            else "error_blocked"
                        ),
                        "approved": False,
                        "reason": reason,
                    }
                )
                if error:
                    review["error"] = error
                raw_response["high_risk_review"] = review
            raw_response.update(
                {
                    "execution_policy_terminal": True,
                    "policy_blocker": blocker,
                    "stage_status": DecisionStageStatus.FAILED,
                    "reason": reason,
                    **data,
                }
            )
            decision.raw_response = raw_response
            attach_execution_parameters("policy_evaluation_failed")
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
                    "execution_status": DecisionStageStatus.FAILED,
                    "reason": reason,
                    "is_paper": (model_mode == "paper"),
                }
            )
            result = rejected_execution_result(decision, reason)
            if model_mode == "paper":
                attach_trade_execution_result(
                    decision,
                    result,
                    source=blocker,
                    exchange_confirmed=False,
                )
                if decision_db_id is not None:
                    await mark_decision_raw_response(decision_db_id, decision.raw_response)
            result.raw_response = decision.raw_response
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
            paper_training_order_identity = attach_paper_training_order_identity(
                decision,
                decision_db_id,
                model_mode,
            )
            if paper_training_order_identity and decision_db_id is not None:
                await mark_decision_raw_response(decision_db_id, decision.raw_response)
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
            try:
                exit_policy_result = await evaluate_exit_policy(
                    decision,
                    model_name,
                    open_positions,
                    refresh_positions=refresh_exit_positions,
                )
            except asyncio.CancelledError:
                return await policy_evaluation_failed_result(
                    blocker="exit_policy_cancelled",
                    reason=(
                        "执行前平仓风控检查被外层超时保护取消，系统未提交 OKX 订单；"
                        "本轮按未执行处理，下一轮持仓复盘会重新检查最新仓位和行情。"
                    ),
                    error_type="cancelled",
                )
            except Exception as exc:
                error_text = safe_error_text(exc, limit=180)
                return await policy_evaluation_failed_result(
                    blocker="exit_policy_error",
                    reason=(
                        f"执行前平仓风控检查异常：{error_text}。"
                        "系统未提交 OKX 订单，下一轮会重新检查最新仓位和行情。"
                    ),
                    error_type="exception",
                    error=error_text,
                )
            if not exit_policy_result.passed:
                return await block_before_submit(exit_policy_result)

        if decision.is_entry:
            try:
                if self.production_trade_gate_provider is not None:
                    gate_payload = await self.production_trade_gate_provider(
                        decision,
                        model_name,
                        model_mode,
                        open_positions,
                    )
                    if isinstance(gate_payload, dict) and gate_payload:
                        raw = _safe_dict(decision.raw_response)
                        raw = dict(raw)
                        raw["production_trade_gate"] = gate_payload
                        decision.raw_response = raw
                        if str(model_mode or "").lower() != "paper" and gate_payload.get(
                            "can_trade"
                        ) is not True:
                            return await block_before_submit(
                                PolicyGateResult.block(
                                    "production_trade_gate",
                                    (
                                        "生产交易闸门未放行："
                                        f"{gate_payload.get('reason') or gate_payload.get('mode') or 'unknown'}"
                                    ),
                                    {
                                        "stage_status": "blocked",
                                        "skip_kind": "production_trade_gate",
                                        "production_trade_gate": gate_payload,
                                    },
                                )
                            )
                entry_policy_result = await evaluate_entry_policy(
                    decision,
                    model_name,
                    model_mode,
                    open_positions,
                )
            except asyncio.CancelledError:
                return await policy_evaluation_failed_result(
                    blocker="entry_policy_cancelled",
                    reason=(
                        "执行前开仓风控检查被外层超时保护取消，系统未提交 OKX 订单；"
                        "本轮按未执行处理，下一轮会用最新行情重新分析。"
                    ),
                    error_type="cancelled",
                )
            except Exception as exc:
                error_text = safe_error_text(exc, limit=180)
                return await policy_evaluation_failed_result(
                    blocker="entry_policy_error",
                    reason=(
                        f"执行前开仓风控检查异常：{error_text}。"
                        "系统未提交 OKX 订单，下一轮会用最新行情重新分析。"
                    ),
                    error_type="exception",
                    error=error_text,
                )
            if not entry_policy_result.passed:
                return await block_before_submit(entry_policy_result)
            return_contract_result = _return_entry_contract_result(decision, model_mode)
            if not return_contract_result.passed:
                return await block_before_submit(return_contract_result)
            if (
                return_contract_result.data.get("return_execution_contract")
                == "live_rules_canary"
            ):
                raw = dict(_safe_dict(decision.raw_response))
                raw["return_execution_contract"] = "live_rules_canary"
                raw["production_permission"] = True
                raw["live_rules_canary_contract"] = return_contract_result.data[
                    "live_rules_canary_contract"
                ]
                decision.raw_response = raw
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
            if decision.is_entry and decision_db_id is not None:
                await mark_decision_pending_execution(
                    decision_db_id,
                    "风控复核已通过，系统正在向 OKX 提交订单并等待交易所回报。",
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
                execution_result = await await_exchange_place_order(
                    executor,
                    timeout_seconds=execution_timeout,
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
            if decision.is_entry or decision.is_exit:
                await mark_stage(
                    DecisionStage.EXCHANGE_SUBMIT,
                    DecisionStageStatus.FAILED,
                    "OKX 下单请求超时，未拿到交易所明确接收结果。",
                    {
                        "error_type": "timeout",
                        "submitted_to_exchange": bool(submitted_to_exchange),
                    },
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
        except asyncio.CancelledError:
            reason = (
                "OKX 下单流程被外层超时保护取消，系统没有拿到最终订单结果；"
                "本轮按未执行处理，下一轮会用最新行情重新分析。"
            )
            logger.error(
                "decision execution cancelled",
                model=model_name,
                symbol=symbol,
                action=decision.action.value,
                mode=model_mode,
            )
            execution_result = rejected_execution_result(decision, reason)
            if decision.is_entry or decision.is_exit:
                await mark_stage(
                    DecisionStage.EXCHANGE_SUBMIT,
                    DecisionStageStatus.FAILED,
                    "OKX 下单流程被外层超时保护取消，未拿到交易所明确接收结果。",
                    {
                        "error_type": "cancelled",
                        "submitted_to_exchange": bool(submitted_to_exchange),
                    },
                )
            await mark_stage(
                DecisionStage.EXCHANGE_CONFIRM,
                DecisionStageStatus.FAILED,
                execution_reason_from_result(execution_result),
                {"error_type": "cancelled"},
            )
            await log_risk_event(
                "warning",
                symbol,
                f"[{model_name}] OKX execution cancelled by outer watchdog",
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
                    execution_result = await await_exchange_place_order(
                        retry_executor,
                        timeout_seconds=45.0,
                        retry=True,
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
                decision.is_entry and is_okx_temporary_service_error(result_text)
            )
            exchange_confirmed = is_exchange_confirmed_execution(execution_result)
            exit_progress = is_exit_progress_execution(execution_result)
            confirm_reason = execution_reason_from_result(execution_result)
            local_order_persisted = True
            local_order_persistence_error: str | None = None
            recovery_requested = False
            try:
                await log_trade(execution_result, model_name, decision, decision_db_id)
            except Exception as exc:
                local_order_persisted = False
                local_order_persistence_error = safe_error_text(exc, limit=180)
                if exchange_confirmed or exit_progress:
                    recovery_requested = self._trigger_order_fact_recovery(model_mode)
                raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                raw = dict(raw)
                raw["local_order_persistence"] = {
                    "status": "failed",
                    "error": local_order_persistence_error,
                    "exchange_order_id": getattr(execution_result, "exchange_order_id", None),
                    "recovery_requested": recovery_requested,
                }
                decision.raw_response = raw
                warning = (
                    (
                        "OKX 成交已确认，但本地订单事实写入失败；系统没有写入孤立持仓，"
                        "并已触发 OKX 成交事实补偿。"
                    )
                    if exchange_confirmed or exit_progress
                    else "本地订单执行记录写入失败；交易所没有确认成交，系统未改动本地持仓。"
                )
                results["warnings"].append(
                    {"model": model_name, "symbol": symbol, "warning": warning}
                )
                await log_risk_event("warning", symbol, warning, model_name)
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
                if local_order_persisted:
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
                    protection_rebalance: dict[str, Any] | None = None
                    if decision.is_exit and exchange_confirmed:
                        try:
                            protection_rebalance = await rebalance_position_protection(
                                executor,
                                decision,
                            )
                        except Exception as exc:
                            protection_rebalance = getattr(exc, "report", None)
                            if not isinstance(protection_rebalance, dict):
                                protection_rebalance = {
                                    "status": "failed",
                                    "verified": False,
                                    "error": safe_error_text(exc, limit=180),
                                }
                            raw = _safe_dict(decision.raw_response)
                            raw = dict(raw)
                            raw["post_exit_protection_rebalance"] = protection_rebalance
                            decision.raw_response = raw
                            warning = (
                                "OKX 退出成交已确认，但剩余持仓保护数量未能完成精确复核；"
                                "系统保留成交事实并让后续非硬风险退出 fail-closed。"
                            )
                            results["warnings"].append(
                                {"model": model_name, "symbol": symbol, "warning": warning}
                            )
                            await log_risk_event("warning", symbol, warning, model_name)
                        else:
                            raw = _safe_dict(decision.raw_response)
                            raw = dict(raw)
                            raw["post_exit_protection_rebalance"] = protection_rebalance
                            decision.raw_response = raw
                    await mark_stage(
                        DecisionStage.LOCAL_SYNC,
                        DecisionStageStatus.COMPLETED,
                        "成交结果已写入本地订单/持仓记录。",
                        {
                            "exit_progress": bool(exit_progress),
                            "exchange_confirmed": bool(exchange_confirmed),
                            "protection_rebalance_status": (
                                protection_rebalance.get("status")
                                if isinstance(protection_rebalance, dict)
                                else "not_applicable"
                            ),
                            "protection_rebalance_verified": (
                                protection_rebalance.get("verified")
                                if isinstance(protection_rebalance, dict)
                                else None
                            ),
                        },
                    )
                else:
                    await mark_stage(
                        DecisionStage.LOCAL_SYNC,
                        DecisionStageStatus.FAILED,
                        (
                            "OKX 成交已确认，但本地订单事实写入失败；系统未写入本地持仓，"
                            "正在通过 OKX 成交事实补偿恢复。"
                        ),
                        {
                            "exchange_confirmed": bool(exchange_confirmed),
                            "exit_progress": bool(exit_progress),
                            "local_order_persistence_error": local_order_persistence_error,
                            "recovery_requested": recovery_requested,
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
                if model_mode == "paper":
                    attach_trade_execution_result(
                        decision,
                        None,
                        source="missing_execution_result",
                        exchange_confirmed=False,
                    )
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
        return execution_result
