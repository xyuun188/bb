from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.authoritative_trade_outcome import build_authoritative_trade_outcome
from services.decision_state import DecisionStage, DecisionStageStatus
from services.execution_result_factory import ExecutionResultFactory
from services.execution_service import ExecutionService, _return_entry_contract_result
from services.normal_paper_trade import (
    NORMAL_PAPER_TRADE_SIZING_VERSION,
    build_normal_paper_trade_contract,
)
from services.okx_execution_slippage import build_okx_fill_mark_slippage
from services.okx_training_facts import build_okx_history_training_sample
from services.production_trade_gate import PRODUCTION_TRADE_GATE_VERSION
from services.trade_execution_contract import validate_entry_execution_contract
from services.trade_order_log_service import TradeOrderLogOutcome
from services.trading_policies import PolicyGateResult
from services.training_data_quality import annotate_training_payload
from tests.legacy_paper_contract_fixtures import (
    build_legacy_normal_paper_v4_trade_contract,
)
from tests.legacy_paper_contract_fixtures import (
    build_legacy_paper_training_contract as build_paper_training_contract,
)
from tests.normal_paper_test_fixtures import paper_quality_permissions


async def _noop_async(*_args: Any, **_kwargs: Any) -> Any:
    return None


def _test_execution_service(
    *,
    okx_executor_provider,
    entry_policy_evaluator=None,
    exit_policy_evaluator=None,
    production_trade_gate_provider=None,
    raw_updates: list[dict[str, Any] | None] | None = None,
    reasons: list[str | None] | None = None,
    stages: list[tuple[str, str, str]] | None = None,
    trade_logger=None,
    trade_count_incrementer=None,
    position_execution_persister=None,
    position_protection_rebalancer=None,
    order_fact_recovery_trigger=None,
    open_positions_execution_applier=None,
    decision_stage_recorder=None,
) -> ExecutionService:
    async def mark_reason(_decision_id: int, reason: str | None) -> None:
        if reasons is not None:
            reasons.append(reason)

    async def mark_raw(_decision_id: int, raw: dict[str, Any] | None) -> None:
        if raw_updates is not None:
            raw_updates.append(raw)

    async def record_stage(
        _decision_id: int | None,
        _decision: DecisionOutput,
        stage: str,
        status: str,
        reason: str,
        _data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stages is not None:
            stages.append((stage, status, reason))
        return _decision.raw_response if isinstance(_decision.raw_response, dict) else {}

    async def allow_entry(*_args: Any, **_kwargs: Any) -> PolicyGateResult:
        return PolicyGateResult.allow()

    return ExecutionService(
        execution_lock=asyncio.Lock(),
        risk_event_logger=_noop_async,
        model_execution_mode_provider=lambda _model: "paper",
        decision_stage_recorder=decision_stage_recorder or record_stage,
        decision_reason_marker=mark_reason,
        decision_raw_response_marker=mark_raw,
        position_review_alert_context_provider=lambda _decision: None,
        position_review_risk_result_logger=_noop_async,
        duplicate_decision_order_reason_provider=lambda *_args: _noop_async(),
        okx_executor_provider=okx_executor_provider,
        allocated_order_balance_provider=lambda *_args: _noop_async(),
        rejected_execution_result_factory=ExecutionResultFactory().rejected,
        execution_leverage_summary_attacher=lambda *_args: None,
        execution_reason_provider=lambda result: result.raw_response.get("error") if result else "",
        pending_execution_marker=_noop_async,
        trade_logger=trade_logger or _noop_async,
        exchange_confirmed_checker=lambda result: bool(
            result
            and result.status == OrderStatus.FILLED
            and result.exchange_order_id
        ),
        exit_progress_checker=lambda _result: False,
        no_exchange_position_result_checker=lambda _result: False,
        trade_count_incrementer=trade_count_incrementer or (lambda: None),
        position_execution_persister=position_execution_persister or _noop_async,
        position_protection_rebalancer=position_protection_rebalancer or _noop_async,
        order_fact_recovery_trigger=order_fact_recovery_trigger,
        open_positions_execution_applier=(
            open_positions_execution_applier or (lambda *_args: None)
        ),
        decision_executed_marker=_noop_async,
        account_update_persister=_noop_async,
        account_balance_provider=lambda _model: _noop_async(),
        decision_outcome_marker=_noop_async,
        entry_policy_evaluator=entry_policy_evaluator or allow_entry,
        exit_policy_evaluator=exit_policy_evaluator or allow_entry,
        production_trade_gate_provider=production_trade_gate_provider,
        execution_skills_provider=lambda **_kwargs: [],
        execution_skills_attacher=lambda *_args, **_kwargs: None,
        execution_skills_block_reason_provider=lambda *_args, **_kwargs: None,
        position_reconciler=_noop_async,
        open_positions_context_provider=lambda: _noop_async(),
        matching_exit_local_position_checker=lambda *_args: False,
        matching_exit_exchange_position_checker=lambda *_args: _noop_async(),
        trade_notional_recorder=lambda _notional: None,
    )


def _entry_decision(symbol: str = "SPK/USDT") -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol=symbol,
        action=Action.SHORT,
        confidence=0.8,
        reasoning="test",
        position_size_pct=0.05,
        suggested_leverage=3.0,
        raw_response={},
    )


def _profit_first_ready_position_review_decision() -> DecisionOutput:
    decision = _dynamic_return_ready_decision()
    raw = decision.raw_response
    raw.pop("production_trade_gate", None)
    raw["normal_paper_trade"] = build_normal_paper_trade_contract(
        symbol=decision.symbol,
        side="short",
        selection_reason="strategy_edge_selected",
        direction_support={
            "eligible": True,
            "selected_side": "short",
            "prediction_horizon_minutes": 30.0,
            "expected_net_return_pct": 0.8,
            "objective_net_return_pct": 0.4,
            "loss_probability": 0.25,
            "quant_evidence_families": ["local_ml"],
            "quant_quality_permissions": paper_quality_permissions(),
            "strong_expert_opposition": False,
        },
    )
    raw["opportunity_score"]["execution_cost"].update(
        {
            "total_pct": 0.08,
            "order_size_complete": True,
            "order_notional_usdt": 40.0,
        }
    )
    raw["profit_risk_sizing"].update(
        {
            "contract_version": NORMAL_PAPER_TRADE_SIZING_VERSION,
            "contract_lifecycle": "normal_paper_trade",
            "execution_scope": "paper_only",
            "production_permission": False,
            "production_eligible": True,
            "account_equity_usdt": 1000.0,
            "risk_budget_usdt": 0.5,
            "portfolio_risk_budget_usdt": 1.5,
            "planned_stressed_loss_usdt": 0.4,
            "stressed_loss_fraction": 0.01,
            "target_notional_usdt": 40.0,
            "final_notional_usdt": 40.0,
            "fill_notional_ceiling_usdt": 50.0,
            "minimum_order_notional_usdt": 1.0,
            "final_margin_usdt": 40.0,
            "final_leverage": 1.0,
            "model_requested_leverage": 1.0,
            "model_leverage_is_explicit": True,
            "dynamic_leverage_decision": {
                "version": "dynamic_leverage_allocator_v5",
                "final_integer_leverage": 1,
            },
            "leverage_tier_selection": {
                "production_eligible": True,
                "max_leverage": 20.0,
            },
        }
    )
    raw["execution_cost_sizing_pass"].update(
        {
            "order_size_complete": True,
            "impact_basis_notional_usdt": 40.0,
            "final_notional_usdt": 40.0,
        }
    )
    decision.position_size_pct = 0.04
    decision.suggested_leverage = 1.0
    return decision


def _dynamic_return_ready_decision() -> DecisionOutput:
    decision = _entry_decision("BTC/USDT")
    provenance = {
        "source": "authoritative_test_return",
        "observation_window": "test_window",
        "sample_count": 5,
        "generated_at": "2026-07-12T00:00:00+00:00",
        "strategy_version": "test.dynamic.v1",
        "fallback_reason": "",
    }
    decision.position_size_pct = 0.03
    decision.raw_response = {
        "production_trade_gate": {
            "version": PRODUCTION_TRADE_GATE_VERSION,
            "can_trade": True,
            "mode": "live_ml",
            "decision_authority": "model",
            "model_can_influence": True,
        },
        "authoritative_return_candidate": {
            "production_eligible": True,
            "side_evidence": {
                "production_eligible": True,
                "expected_net_return_pct": 0.8,
                "return_lcb_pct": 0.4,
                "production_source_count": 5,
                "policy_provenance": provenance,
            },
        },
        "opportunity_score": {
            "execution_cost": {
                "production_eligible": True,
                "order_size_complete": True,
                "order_notional_usdt": 90.0,
            },
        },
        "profit_risk_sizing": {
            "production_eligible": True,
            "available_margin_usdt": 1000.0,
            "position_size_pct": 0.03,
            "risk_budget_usdt": 2.0,
            "planned_stressed_loss_usdt": 0.9,
            "stressed_loss_fraction": 0.01,
            "target_notional_usdt": 200.0,
            "final_notional_usdt": 90.0,
            "final_margin_usdt": 30.0,
            "policy_provenance": {**provenance, "contract_fingerprint": "test-fingerprint"},
        },
        "pre_order_execution_facts": {
            "production_eligible": True,
            "input_fingerprint": "test-pre-order-fingerprint",
        },
        "execution_cost_sizing_pass": {
            "order_size_complete": True,
            "impact_basis_notional_usdt": 90.0,
            "final_notional_usdt": 90.0,
        },
    }
    return decision


def _live_rules_canary_ready_decision(*, max_notional: float = 100.0) -> DecisionOutput:
    decision = _dynamic_return_ready_decision()
    raw = decision.raw_response
    provenance = raw["profit_risk_sizing"]["policy_provenance"]
    final_notional = float(raw["profit_risk_sizing"]["final_notional_usdt"])
    raw["production_trade_gate"] = {
        "version": PRODUCTION_TRADE_GATE_VERSION,
        "mode": "live_rules_canary",
        "can_trade": True,
        "decision_authority": "rules",
        "model_can_influence": False,
        "risk": {
            "max_notional_usdt": max_notional,
            "max_open_positions": 1,
            "max_daily_loss_usdt": 3.0,
        },
    }
    raw["live_rules_canary_signal"] = {
        "version": "test-rules-canary-signal",
        "execution_scope": "live_rules_canary",
        "decision_authority": "rules",
        "model_can_influence": False,
        "production_eligible": True,
        "action": "short",
        "policy_provenance": provenance,
    }
    raw["model_shadow_decision"] = {
        "action": "long",
        "observation_only": True,
        "can_authorize_entry": False,
        "can_change_size_or_leverage": False,
    }
    raw["opportunity_score"]["execution_cost"].update(
        {
            "total_pct": 0.08,
            "order_notional_usdt": final_notional,
            "order_size_complete": True,
            "policy_provenance": provenance,
        }
    )
    raw["profit_risk_sizing"].update(
        {
            "contract_version": "test-rules-canary-sizing",
            "contract_lifecycle": "live_rules_canary",
            "execution_scope": "live_rules_canary",
            "production_permission": True,
            "decision_authority": "rules",
            "model_can_influence": False,
            "target_notional_usdt": final_notional,
            "target_inst_id": "BTC-USDT-SWAP",
            "target_price": 100.0,
            "selected_contract_spec": {
                "ctVal": "0.01",
                "ctMult": "1",
                "minSz": "1",
                "lotSz": "1",
            },
            "exchange_minimum_order": {
                "production_eligible": True,
                "minimum_notional_usdt": 1.0,
            },
            "exchange_min_notional_usdt": 1.0,
            "final_margin_usdt": final_notional,
            "final_leverage": 1.0,
            "leverage_tier_selection": {
                "production_eligible": True,
                "max_leverage": 20.0,
            },
        }
    )
    decision.position_size_pct = final_notional / 1000.0
    decision.suggested_leverage = 1.0
    return decision


def _paper_training_ready_decision() -> DecisionOutput:
    provenance = {
        "source": "paper_training_test",
        "observation_window": "current_test_entry",
        "sample_count": 1,
        "generated_at": "2026-07-22T00:00:00+00:00",
        "strategy_version": "paper-training-test.v1",
        "fallback_reason": "",
        "contract_fingerprint": "paper-training-sizing-fingerprint",
    }
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.2,
        reasoning="loss-tolerant paper training",
        position_size_pct=0.005,
        suggested_leverage=1.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        feature_snapshot={"current_price": 100.0, "close": 100.0},
        raw_response={},
    )
    decision.raw_response = {
        "paper_training": build_paper_training_contract(
            symbol=decision.symbol,
            selected_side="long",
            signal_source="local_ml_observation",
            expected_net_return_pct=-0.5,
            return_lcb_pct=-0.8,
            horizon_minutes=10.0,
        ),
        "paper_training_mode": "bootstrap",
        "opportunity_score": {
            "execution_cost": {
                "production_eligible": True,
                "order_size_complete": True,
                "order_notional_usdt": 5.0,
            }
        },
        "pre_order_execution_facts": {
            "production_eligible": True,
            "input_fingerprint": "paper-training-pre-order",
        },
        "execution_cost_sizing_pass": {
            "order_size_complete": True,
            "impact_basis_notional_usdt": 5.0,
            "final_notional_usdt": 5.0,
        },
        "profit_risk_sizing": {
            "contract_version": "2026-07-22.paper-training-sizing.v1",
            "contract_lifecycle": "paper_training",
            "execution_scope": "paper_only",
            "production_permission": False,
            "production_eligible": True,
            "account_equity_usdt": 1000.0,
            "available_margin_usdt": 1000.0,
            "position_size_pct": 0.005,
            "risk_budget_usdt": 0.1,
            "portfolio_risk_budget_usdt": 0.3,
            "current_portfolio_stressed_loss_usdt": 0.0,
            "planned_stressed_loss_usdt": 0.1,
            "stressed_loss_fraction": 0.02,
            "target_notional_usdt": 5.0,
            "final_notional_usdt": 5.0,
            "final_margin_usdt": 5.0,
            "final_leverage": 1.0,
            "policy_provenance": provenance,
        },
    }
    return decision


def test_dynamic_return_contract_accepts_complete_governed_entry() -> None:
    result = _return_entry_contract_result(_dynamic_return_ready_decision())
    assert result.passed is True
    assert result.data["return_execution_contract"] == "complete"
    assert result.data["production_permission"] is True


def test_live_ml_profit_contract_cannot_authorize_entry_without_trade_gate() -> None:
    decision = _dynamic_return_ready_decision()
    decision.raw_response.pop("production_trade_gate")

    result = _return_entry_contract_result(decision, "live")

    assert result.passed is False
    assert result.blocker == "production_trade_gate"
    assert result.data["gate_validation_reason"] == "production_trade_gate_missing"


def test_live_rules_canary_bypasses_model_promotion_return_distribution() -> None:
    decision = _live_rules_canary_ready_decision()

    result = _return_entry_contract_result(decision, "live")

    assert result.passed is True
    assert result.data["return_execution_contract"] == "live_rules_canary"
    assert result.data["production_permission"] is True


def test_live_rules_canary_respects_gate_notional_limit() -> None:
    decision = _live_rules_canary_ready_decision(max_notional=50.0)

    result = _return_entry_contract_result(decision, "live")

    assert result.passed is False
    assert result.blocker == "live_rules_canary_contract_incomplete"
    assert "rules_canary_order_notional_above_gate_limit" in result.data[
        "block_reasons"
    ]


@pytest.mark.asyncio
async def test_live_entry_without_trade_gate_provider_never_calls_okx() -> None:
    calls = 0

    async def okx_executor_provider(_mode: str) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("missing production gate must stop before OKX submit")

    service = _test_execution_service(okx_executor_provider=okx_executor_provider)
    service.model_execution_mode_provider = lambda _model: "live"

    result = await service.execute_candidate(
        "BTC/USDT",
        "ensemble_trader",
        _dynamic_return_ready_decision(),
        SimpleNamespace(warnings=[]),
        2600,
        {"warnings": [], "decisions": [], "executions": []},
        open_positions=[],
    )

    assert calls == 0
    assert result is not None and result.status == OrderStatus.REJECTED
    assert result.raw_response["policy_blocker"] == "production_trade_gate"
    assert result.raw_response["gate_validation_reason"] == (
        "production_trade_gate_provider_missing"
    )


@pytest.mark.asyncio
async def test_live_entry_with_empty_trade_gate_never_calls_okx() -> None:
    calls = 0

    async def okx_executor_provider(_mode: str) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("invalid production gate must stop before OKX submit")

    async def gate_provider(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}

    service = _test_execution_service(
        okx_executor_provider=okx_executor_provider,
        production_trade_gate_provider=gate_provider,
    )
    service.model_execution_mode_provider = lambda _model: "live"

    result = await service.execute_candidate(
        "BTC/USDT",
        "ensemble_trader",
        _dynamic_return_ready_decision(),
        SimpleNamespace(warnings=[]),
        2601,
        {"warnings": [], "decisions": [], "executions": []},
        open_positions=[],
    )

    assert calls == 0
    assert result is not None and result.status == OrderStatus.REJECTED
    assert result.raw_response["policy_blocker"] == "production_trade_gate"
    assert result.raw_response["gate_validation_reason"] == (
        "production_trade_gate_missing"
    )


@pytest.mark.asyncio
async def test_execution_service_attaches_trade_gate_before_entry_policy() -> None:
    raw_updates: list[dict[str, Any] | None] = []
    calls: list[str] = []

    async def okx_executor_provider(_mode: str) -> Any:
        calls.append("okx")
        raise AssertionError("entry policy block should happen before OKX submit")

    async def gate_provider(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("gate")
        return {
            "version": PRODUCTION_TRADE_GATE_VERSION,
            "can_trade": True,
            "mode": "live_rules_canary",
            "decision_authority": "rules",
            "model_can_influence": False,
            "risk": {"max_notional_usdt": 10.0},
        }

    async def entry_policy(
        decision: DecisionOutput,
        *_args: Any,
        **_kwargs: Any,
    ) -> PolicyGateResult:
        calls.append("entry_policy")
        assert decision.raw_response["production_trade_gate"]["mode"] == (
            "live_rules_canary"
        )
        return PolicyGateResult.block(
            "stop_after_gate_for_test",
            "stop after proving gate order",
            {"stage_status": "blocked"},
        )

    service = _test_execution_service(
        okx_executor_provider=okx_executor_provider,
        entry_policy_evaluator=entry_policy,
        production_trade_gate_provider=gate_provider,
        raw_updates=raw_updates,
    )
    service.model_execution_mode_provider = lambda _model: "live"
    results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}

    result = await service.execute_candidate(
        "SPK/USDT",
        "ensemble_trader",
        _entry_decision("SPK/USDT"),
        SimpleNamespace(warnings=[]),
        2601,
        results,
        open_positions=[],
    )

    assert calls == ["gate", "entry_policy"]
    assert result is not None
    assert result.raw_response["policy_blocker"] == "stop_after_gate_for_test"
    assert raw_updates[-1]["production_trade_gate"]["mode"] == "live_rules_canary"


@pytest.mark.asyncio
async def test_execution_service_persists_live_rules_canary_contract_before_submit() -> None:
    decision = _live_rules_canary_ready_decision()
    trade_gate = dict(decision.raw_response["production_trade_gate"])
    raw_updates: list[dict[str, Any] | None] = []

    class FilledExecutor:
        async def place_order(
            self,
            current: DecisionOutput,
            account_id: str | None = None,
            override_balance: float | None = None,
        ) -> ExecutionResult:
            del account_id, override_balance
            return ExecutionResult(
                order_id="local-rules-canary-entry",
                exchange_order_id="okx-rules-canary-entry",
                symbol=current.symbol,
                side="sell",
                order_type="market",
                quantity=0.9,
                price=100.0,
                status=OrderStatus.FILLED,
                raw_response={},
            )

    executor = FilledExecutor()

    async def executor_provider(_mode: str) -> FilledExecutor:
        return executor

    async def gate_provider(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return dict(trade_gate)

    service = _test_execution_service(
        okx_executor_provider=executor_provider,
        production_trade_gate_provider=gate_provider,
        raw_updates=raw_updates,
    )
    service.model_execution_mode_provider = lambda _model: "live"

    result = await service.execute_candidate(
        decision.symbol,
        decision.model_name,
        decision,
        SimpleNamespace(warnings=[]),
        2602,
        {"warnings": [], "decisions": [], "executions": []},
        open_positions=[],
    )

    assert result is not None and result.status == OrderStatus.FILLED
    contract = decision.raw_response["live_rules_canary_contract"]
    assert contract["contract_complete"] is True
    assert contract["execution_scope"] == "live_rules_canary"
    assert contract["decision_authority"] == "rules"
    assert contract["model_can_influence"] is False
    assert raw_updates[-1]["live_rules_canary_contract"] == contract


def test_legacy_normal_v4_entry_is_blocked_but_settlement_validation_remains_valid() -> None:
    decision = _profit_first_ready_position_review_decision()
    decision.raw_response["normal_paper_trade"] = (
        build_legacy_normal_paper_v4_trade_contract(
            symbol=decision.symbol,
            side="short",
            objective_net_return_pct=-0.1,
        )
    )

    contract, reasons = validate_entry_execution_contract(decision.raw_response)
    assert reasons == []
    assert contract["contract_lifecycle"] == "normal_paper_trade"

    entry_gate = _return_entry_contract_result(decision, "paper")
    assert entry_gate.passed is False
    assert entry_gate.blocker == "normal_paper_trade_contract_incomplete"
    assert "normal_paper_trade_version_invalid" in str(entry_gate.reason)


def test_quality_observation_contract_passes_entry_gate_only_at_one_x() -> None:
    decision = _profit_first_ready_position_review_decision()
    permission = paper_quality_permissions()["local_ml"]
    permission.update(
        {
            "paper_execution_permission": False,
            "paper_execution_reason": "fee_after_return_lcb_not_positive",
            "paper_execution_blockers": ["fee_after_return_lcb_not_positive"],
            "paper_execution_evidence": {"sample_count": 0},
        }
    )
    decision.raw_response["normal_paper_trade"] = build_normal_paper_trade_contract(
        symbol=decision.symbol,
        side="short",
        selection_reason="paper_quality_observation",
        direction_support={
            "eligible": True,
            "selected_side": "short",
            "prediction_horizon_minutes": 30.0,
            "expected_net_return_pct": 0.35,
            "objective_net_return_pct": -0.2,
            "loss_probability": 0.3,
            "quant_evidence_families": ["local_ml"],
            "quant_quality_permissions": {"local_ml": permission},
            "paper_quality_observation_only": True,
            "paper_quality_observation_reasons": [
                "fee_after_return_lcb_not_positive"
            ],
            "strong_expert_opposition": False,
        },
    )
    raw = decision.raw_response
    raw["opportunity_score"]["execution_cost"].update(
        {"order_notional_usdt": 8.0}
    )
    raw["profit_risk_sizing"].update(
        {
            "risk_budget_usdt": 0.1,
            "planned_stressed_loss_usdt": 0.08,
            "stressed_loss_fraction": 0.01,
            "target_notional_usdt": 8.0,
            "final_notional_usdt": 8.0,
            "fill_notional_ceiling_usdt": 10.0,
            "final_margin_usdt": 8.0,
            "final_leverage": 1.0,
            "model_requested_leverage": 1.0,
        }
    )
    raw["execution_cost_sizing_pass"].update(
        {
            "impact_basis_notional_usdt": 8.0,
            "final_notional_usdt": 8.0,
        }
    )

    contract, reasons = validate_entry_execution_contract(raw)
    assert reasons == []
    assert contract["selection_reason"] == "paper_quality_observation"
    assert _return_entry_contract_result(decision, "paper").passed is True

    raw["profit_risk_sizing"].update(
        {
            "final_margin_usdt": 4.0,
            "final_leverage": 2.0,
            "model_requested_leverage": 2.0,
        }
    )
    _contract, reasons = validate_entry_execution_contract(raw)
    assert "paper_quality_observation_leverage_not_one_x" in reasons


def test_legacy_paper_training_entry_is_blocked_but_history_remains_trainable() -> None:
    decision = _paper_training_ready_decision()
    contract, reasons = validate_entry_execution_contract(decision.raw_response)
    assert reasons == []
    assert contract["contract_lifecycle"] == "paper_training"
    entry_gate = _return_entry_contract_result(decision, "paper")
    assert entry_gate.passed is False
    assert entry_gate.blocker == "normal_paper_trade_contract_incomplete"
    assert "normal_paper_trade_version_invalid" in str(entry_gate.reason)

    entry_result = ExecutionResult(
        order_id="historical-local-entry",
        exchange_order_id="historical-okx-entry",
        symbol=decision.symbol,
        side="buy",
        order_type="market",
        quantity=1.0,
        price=100.0,
        status=OrderStatus.FILLED,
        raw_response={},
    )
    close_result = ExecutionResult(
        order_id="historical-local-close",
        exchange_order_id="historical-okx-close",
        symbol=decision.symbol,
        side="sell",
        order_type="market",
        quantity=1.0,
        price=95.0,
        status=OrderStatus.FILLED,
        raw_response={},
    )

    opened_at = datetime(2026, 7, 22, 1, tzinfo=UTC)
    history = SimpleNamespace(
        id=1,
        mode="paper",
        row_identity="paper|BTC-USDT-SWAP|paper-training-pos|long|1",
        inst_id="BTC-USDT-SWAP",
        symbol="BTC/USDT",
        pos_id="paper-training-pos",
        side="long",
        close_status="full",
        opened_at=opened_at,
        updated_at_okx=opened_at + timedelta(minutes=30),
        open_avg_px=entry_result.price,
        close_avg_px=close_result.price,
        open_max_pos=entry_result.quantity,
        leverage=1.0,
        realized_pnl=-5.1,
        pnl=-5.0,
        pnl_ratio=-0.051,
        funding_fee=0.0,
        fee=-0.1,
        entry_order_ids=[entry_result.exchange_order_id],
        close_order_ids=[close_result.exchange_order_id],
        linked_order_ids=[
            entry_result.exchange_order_id,
            close_result.exchange_order_id,
        ],
        position_ids=[7],
        evidence_gaps=[],
        raw_row={
            "instId": "BTC-USDT-SWAP",
            "posId": "paper-training-pos",
            "posSide": "long",
            "realizedPnl": "-5.1",
            "pnl": "-5.0",
            "fee": "-0.1",
            "fundingFee": "0",
            "pnlRatio": "-0.051",
            "_bb_contract_spec": {"ctVal": "1", "ctMult": "1", "lotSz": "1"},
            "_bb_contract_spec_source": "okx_public_instruments",
        },
        sync_status="synced",
    )
    orders = {
        entry_result.exchange_order_id: SimpleNamespace(
            exchange_order_id=entry_result.exchange_order_id,
            okx_inst_id="BTC-USDT-SWAP",
            side="buy",
            quantity=1.0,
            price=entry_result.price,
            fee=0.04,
            okx_fill_contracts=entry_result.quantity,
            okx_trade_ids="trade-entry",
            decision_id=321,
            okx_raw_fills={
                "fills_history_confirmed": True,
                "order_id": entry_result.exchange_order_id,
                "trade_ids": ["trade-entry"],
                "inst_id": "BTC-USDT-SWAP",
                "contracts": 1.0,
                "base_quantity": 1.0,
                "avg_price": entry_result.price,
                "fee_abs": 0.04,
                "contract_size": 1.0,
                "contract_size_verified": True,
                "contract_size_source": "okx_public_instruments",
                "execution_slippage": build_okx_fill_mark_slippage(
                    order_id=entry_result.exchange_order_id,
                    inst_id="BTC-USDT-SWAP",
                    side="buy",
                    contracts=1.0,
                    average_price=entry_result.price,
                    contract_size=1.0,
                    rows=[
                        {
                            "ordId": entry_result.exchange_order_id,
                            "instId": "BTC-USDT-SWAP",
                            "tradeId": "trade-entry",
                            "side": "buy",
                            "fillSz": "1",
                            "fillPx": str(entry_result.price),
                            "fillMarkPx": str(entry_result.price - 0.02),
                        }
                    ],
                ),
                "protection_submission": {
                    "exchange_confirmation_recorded": True,
                    "source_authority": (
                        "local_submit_plus_okx_create_order_response"
                    ),
                },
            },
        ),
        close_result.exchange_order_id: SimpleNamespace(
            exchange_order_id=close_result.exchange_order_id,
            okx_inst_id="BTC-USDT-SWAP",
            side="sell",
            quantity=1.0,
            price=close_result.price,
            fee=0.06,
            okx_fill_contracts=close_result.quantity,
            okx_trade_ids="trade-close",
            decision_id=322,
            okx_raw_fills={
                "fills_history_confirmed": True,
                "order_id": close_result.exchange_order_id,
                "trade_ids": ["trade-close"],
                "inst_id": "BTC-USDT-SWAP",
                "contracts": 1.0,
                "base_quantity": 1.0,
                "avg_price": close_result.price,
                "fee_abs": 0.06,
                "contract_size": 1.0,
                "contract_size_verified": True,
                "contract_size_source": "okx_public_instruments",
                "execution_slippage": build_okx_fill_mark_slippage(
                    order_id=close_result.exchange_order_id,
                    inst_id="BTC-USDT-SWAP",
                    side="sell",
                    contracts=1.0,
                    average_price=close_result.price,
                    contract_size=1.0,
                    rows=[
                        {
                            "ordId": close_result.exchange_order_id,
                            "instId": "BTC-USDT-SWAP",
                            "tradeId": "trade-close",
                            "side": "sell",
                            "fillSz": "1",
                            "fillPx": str(close_result.price),
                            "fillMarkPx": str(close_result.price + 0.03),
                        }
                    ],
                ),
                "protection_execution": {
                    "lifecycle_complete": True,
                    "source_authority": "okx_algo_history_plus_fills_history",
                    "actual_side": "sl",
                    "stop_loss_slippage_pct": 3.061224489795918,
                    "stop_loss_slippage_source": (
                        "okx_configured_stop_trigger_to_fills_vwap"
                    ),
                },
            },
        ),
    }
    sample = build_okx_history_training_sample(
        history,
        positions_by_id={
            7: SimpleNamespace(
                model_name=decision.model_name,
                stop_loss_price=98.0,
                take_profit_price=104.0,
            )
        },
        orders_by_exchange_id=orders,
        decision_raw_by_order_id={
            entry_result.exchange_order_id: decision.raw_response
        },
    )
    outcome = build_authoritative_trade_outcome(sample)
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[outcome],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert outcome["outcome_complete"] is True
    assert outcome["strategy_entry_kind"] == "normal_strategy_trade"
    assert len(payload["trade_samples"]) == 1
    labels = payload["trade_samples"][0]["profit_learning_labels"]
    assert labels["training_supervision_ready"] is True
    assert labels["realized_net_pnl_usdt"] == -5.1


def test_dynamic_return_contract_ignores_legacy_probe_fields_and_fails_closed() -> None:
    decision = _dynamic_return_ready_decision()
    decision.raw_response["opportunity_score"] = {
        "evidence_score": {"tradeable_probe": True, "shadow_only": False}
    }
    decision.raw_response["authoritative_return_candidate"]["side_evidence"][
        "policy_provenance"
    ] = {}
    result = _return_entry_contract_result(decision)
    assert result.passed is False
    assert result.blocker == "dynamic_return_execution_contract_incomplete"
    assert "return_policy_provenance_incomplete" in result.data["block_reasons"]


def test_dynamic_return_contract_accepts_persisted_eight_decimal_risk_algebra() -> None:
    decision = _dynamic_return_ready_decision()
    sizing = decision.raw_response["profit_risk_sizing"]
    sizing.update(
        {
            "planned_stressed_loss_usdt": 1.28138362,
            "stressed_loss_fraction": 0.0278756,
            "final_notional_usdt": 45.96792255,
            "final_margin_usdt": 15.32264085,
        }
    )
    decision.raw_response["opportunity_score"]["execution_cost"][
        "order_notional_usdt"
    ] = 45.96792255

    result = _return_entry_contract_result(decision)

    assert result.passed is True


@pytest.mark.asyncio
async def test_execution_service_blocks_symbol_mismatch_before_okx_submit() -> None:
    calls: dict[str, int] = {"okx": 0}
    raw_updates: list[dict[str, Any] | None] = []
    reasons: list[str | None] = []

    async def okx_executor_provider(_mode: str) -> Any:
        calls["okx"] += 1
        raise AssertionError("symbol mismatch must stop before OKX executor is requested")

    async def mark_reason(_decision_id: int, reason: str | None) -> None:
        reasons.append(reason)

    async def mark_raw(_decision_id: int, raw: dict[str, Any] | None) -> None:
        raw_updates.append(raw)

    async def allow_entry(*_args: Any, **_kwargs: Any) -> PolicyGateResult:
        return PolicyGateResult.allow()

    service = ExecutionService(
        execution_lock=asyncio.Lock(),
        risk_event_logger=_noop_async,
        model_execution_mode_provider=lambda _model: "paper",
        decision_stage_recorder=_noop_async,
        decision_reason_marker=mark_reason,
        decision_raw_response_marker=mark_raw,
        position_review_alert_context_provider=lambda _decision: None,
        position_review_risk_result_logger=_noop_async,
        duplicate_decision_order_reason_provider=lambda *_args: _noop_async(),
        okx_executor_provider=okx_executor_provider,
        allocated_order_balance_provider=lambda *_args: _noop_async(),
        rejected_execution_result_factory=ExecutionResultFactory().rejected,
        execution_leverage_summary_attacher=lambda *_args: None,
        execution_reason_provider=lambda result: result.raw_response.get("error") if result else "",
        pending_execution_marker=_noop_async,
        trade_logger=_noop_async,
        exchange_confirmed_checker=lambda _result: False,
        exit_progress_checker=lambda _result: False,
        no_exchange_position_result_checker=lambda _result: False,
        trade_count_incrementer=lambda: None,
        position_execution_persister=_noop_async,
        position_protection_rebalancer=_noop_async,
        open_positions_execution_applier=lambda *_args: None,
        decision_executed_marker=_noop_async,
        account_update_persister=_noop_async,
        account_balance_provider=lambda _model: _noop_async(),
        decision_outcome_marker=_noop_async,
        entry_policy_evaluator=allow_entry,
        exit_policy_evaluator=allow_entry,
        execution_skills_provider=lambda **_kwargs: [],
        execution_skills_attacher=lambda *_args, **_kwargs: None,
        execution_skills_block_reason_provider=lambda *_args, **_kwargs: None,
        position_reconciler=_noop_async,
        open_positions_context_provider=lambda: _noop_async(),
        matching_exit_local_position_checker=lambda *_args: False,
        matching_exit_exchange_position_checker=lambda *_args: _noop_async(),
        trade_notional_recorder=lambda _notional: None,
    )
    results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}

    result = await service.execute_candidate(
        "SAHARA/USDT",
        "ensemble_trader",
        _entry_decision("SPK/USDT"),
        SimpleNamespace(warnings=[]),
        132210,
        results,
        open_positions=[],
    )

    assert calls["okx"] == 0
    assert result is not None
    assert result.raw_response["policy_blocker"] == "execution_symbol_mismatch"
    assert result.raw_response["normalized_request_symbol"] == "SAHARA/USDT"
    assert result.raw_response["normalized_decision_symbol"] == "SPK/USDT"
    assert reasons and "执行链交易对不一致" in reasons[-1]
    assert raw_updates[-1]["policy_blocker"] == "execution_symbol_mismatch"
    assert raw_updates[-1]["trade_recommendation_contract"]["execution"]["source"] == (
        "execution_symbol_mismatch"
    )
    assert raw_updates[-1]["trade_recommendation_contract"]["execution"][
        "exchange_confirmed"
    ] is False
    assert results["decisions"][0]["execution_status"] == "skipped"


@pytest.mark.asyncio
async def test_execution_service_marks_entry_policy_cancellation_terminal_before_okx_submit() -> None:
    calls: dict[str, int] = {"okx": 0}
    raw_updates: list[dict[str, Any] | None] = []
    reasons: list[str | None] = []
    stages: list[tuple[str, str, str]] = []

    async def okx_executor_provider(_mode: str) -> Any:
        calls["okx"] += 1
        raise AssertionError("cancelled entry policy must stop before OKX executor")

    async def cancelled_entry(*_args: Any, **_kwargs: Any) -> PolicyGateResult:
        raise asyncio.CancelledError()

    service = _test_execution_service(
        okx_executor_provider=okx_executor_provider,
        entry_policy_evaluator=cancelled_entry,
        raw_updates=raw_updates,
        reasons=reasons,
        stages=stages,
    )
    decision = _entry_decision()
    decision.raw_response = {
        "high_risk_review": {
            "triggered": True,
            "status": "pending",
            "approved": None,
        }
    }
    results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}

    result = await service.execute_candidate(
        "SPK/USDT",
        "ensemble_trader",
        decision,
        SimpleNamespace(warnings=[]),
        132211,
        results,
        open_positions=[],
    )

    assert calls["okx"] == 0
    assert result is not None
    assert result.status == OrderStatus.REJECTED
    assert results["decisions"][0]["execution_status"] == DecisionStageStatus.FAILED
    assert reasons and "风控检查被外层超时保护取消" in str(reasons[-1])
    assert raw_updates
    final_raw = raw_updates[-1] or {}
    assert final_raw["policy_blocker"] == "entry_policy_cancelled"
    assert final_raw["stage_status"] == DecisionStageStatus.FAILED
    assert final_raw["high_risk_review"]["status"] == "cancelled_blocked"
    assert final_raw["high_risk_review"]["approved"] is False
    assert final_raw["trade_recommendation_contract"]["execution"]["source"] == (
        "entry_policy_cancelled"
    )
    assert any(
        stage == DecisionStage.RISK_CHECK and status == DecisionStageStatus.FAILED
        for stage, status, _reason in stages
    )


@pytest.mark.asyncio
async def test_paper_entry_attaches_recoverable_identity_before_okx_submit() -> None:
    raw_updates: list[dict[str, Any] | None] = []
    observed_identity: dict[str, Any] = {}

    class FilledExecutor:
        async def place_order(
            self,
            decision: DecisionOutput,
            account_id: str | None = None,
            override_balance: float | None = None,
        ) -> ExecutionResult:
            del account_id, override_balance
            identity = decision.raw_response.get("normal_paper_order_identity", {})
            observed_identity.update(identity)
            return ExecutionResult(
                order_id="local-normal-paper-entry",
                exchange_order_id="okx-normal-paper-entry",
                symbol=decision.symbol,
                side="sell",
                order_type="market",
                quantity=2.0,
                price=100.0,
                status=OrderStatus.FILLED,
                raw_response={},
            )

    async def okx_executor_provider(_mode: str) -> Any:
        return FilledExecutor()

    service = _test_execution_service(
        okx_executor_provider=okx_executor_provider,
        raw_updates=raw_updates,
    )
    decision = _profit_first_ready_position_review_decision()

    result = await service.execute_candidate(
        "BTC/USDT",
        "ensemble_trader",
        decision,
        SimpleNamespace(warnings=[]),
        991,
        {"warnings": [], "decisions": [], "executions": []},
        open_positions=[],
    )

    assert result is not None and result.status == OrderStatus.FILLED
    assert observed_identity["decision_id"] == 991
    assert observed_identity["client_order_id"] == "BBNP991"
    assert observed_identity["entry_type"] == "normal_strategy_trade"
    assert observed_identity["production_permission"] is False
    assert decision.raw_response["normal_paper_order_identity"] == observed_identity
    assert any(
        update and update.get("normal_paper_order_identity") == observed_identity
        for update in raw_updates
    )


@pytest.mark.asyncio
async def test_execution_service_recovers_when_confirmed_order_fact_write_fails() -> None:
    persisted_positions: list[str] = []
    recovery_requests: list[str] = []
    stages: list[tuple[str, str, str]] = []

    class FilledExecutor:
        async def place_order(
            self,
            decision: DecisionOutput,
            account_id: str | None = None,
            override_balance: float | None = None,
        ) -> ExecutionResult:
            return ExecutionResult(
                order_id="local-order-1",
                exchange_order_id="okx-order-1",
                symbol=decision.symbol,
                side="sell",
                order_type="market",
                quantity=2.0,
                price=100.0,
                status=OrderStatus.FILLED,
                raw_response={},
            )

    async def failed_trade_logger(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("database write failed")

    async def persist_position(*_args: Any, **_kwargs: Any) -> None:
        persisted_positions.append("called")

    async def okx_executor_provider(_mode: str) -> Any:
        return FilledExecutor()

    service = _test_execution_service(
        okx_executor_provider=okx_executor_provider,
        trade_logger=failed_trade_logger,
        position_execution_persister=persist_position,
        order_fact_recovery_trigger=lambda mode: recovery_requests.append(mode),
        stages=stages,
    )
    results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}
    decision = _profit_first_ready_position_review_decision()

    result = await service.execute_candidate(
        "BTC/USDT",
        "ensemble_trader",
        decision,
        SimpleNamespace(warnings=[]),
        992,
        results,
        open_positions=[],
    )

    assert result is not None
    assert result.status == OrderStatus.FILLED
    assert persisted_positions == []
    assert recovery_requests == ["paper"]
    assert decision.raw_response["local_order_persistence"]["status"] == "failed"
    assert decision.raw_response["local_order_persistence"]["recovery_requested"] is True
    assert results["warnings"]
    assert any(
        stage == DecisionStage.LOCAL_SYNC and status == DecisionStageStatus.FAILED
        for stage, status, _reason in stages
    )


@pytest.mark.asyncio
async def test_execution_service_requests_authoritative_facts_after_confirmed_write() -> None:
    recovery_requests: list[str] = []

    class FilledExecutor:
        async def place_order(
            self,
            decision: DecisionOutput,
            account_id: str | None = None,
            override_balance: float | None = None,
        ) -> ExecutionResult:
            return ExecutionResult(
                order_id="local-order-2",
                exchange_order_id="okx-order-2",
                symbol=decision.symbol,
                side="sell",
                order_type="market",
                quantity=2.0,
                price=100.0,
                status=OrderStatus.FILLED,
                raw_response={},
            )

    async def okx_executor_provider(_mode: str) -> Any:
        return FilledExecutor()

    service = _test_execution_service(
        okx_executor_provider=okx_executor_provider,
        order_fact_recovery_trigger=lambda mode: recovery_requests.append(mode),
    )
    results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}

    result = await service.execute_candidate(
        "BTC/USDT",
        "ensemble_trader",
        _profit_first_ready_position_review_decision(),
        SimpleNamespace(warnings=[]),
        993,
        results,
        open_positions=[],
    )

    assert result is not None
    assert result.status == OrderStatus.FILLED
    assert recovery_requests == ["paper"]


@pytest.mark.asyncio
async def test_duplicate_exchange_fact_callback_does_not_mutate_position_again() -> None:
    persisted_positions: list[str] = []
    applied_positions: list[str] = []
    trade_counts: list[str] = []
    recovery_requests: list[str] = []
    raw_updates: list[dict[str, Any] | None] = []

    class ReplayedFilledExecutor:
        async def place_order(
            self,
            decision: DecisionOutput,
            account_id: str | None = None,
            override_balance: float | None = None,
        ) -> ExecutionResult:
            del account_id, override_balance
            return ExecutionResult(
                order_id="local-replayed-order",
                exchange_order_id="okx-existing-order",
                symbol=decision.symbol,
                side="sell",
                order_type="market",
                quantity=2.0,
                price=100.0,
                status=OrderStatus.FILLED,
                raw_response={"info": {"ordId": "okx-existing-order"}},
            )

    async def okx_executor_provider(_mode: str) -> Any:
        return ReplayedFilledExecutor()

    async def reused_trade_logger(*_args: Any, **_kwargs: Any) -> TradeOrderLogOutcome:
        return TradeOrderLogOutcome(
            created=False,
            local_order_id=8059,
            decision_id=388831,
            exchange_order_id="okx-existing-order",
        )

    async def persist_position(*_args: Any, **_kwargs: Any) -> None:
        persisted_positions.append("called")

    service = _test_execution_service(
        okx_executor_provider=okx_executor_provider,
        trade_logger=reused_trade_logger,
        trade_count_incrementer=lambda: trade_counts.append("called"),
        position_execution_persister=persist_position,
        open_positions_execution_applier=lambda *_args: applied_positions.append("called"),
        order_fact_recovery_trigger=lambda mode: recovery_requests.append(mode),
        raw_updates=raw_updates,
    )
    decision = _profit_first_ready_position_review_decision()

    result = await service.execute_candidate(
        "BTC/USDT",
        "ensemble_trader",
        decision,
        SimpleNamespace(warnings=[]),
        388833,
        {"warnings": [], "decisions": [], "executions": []},
        open_positions=[],
    )

    assert result is not None and result.status == OrderStatus.FILLED
    assert persisted_positions == []
    assert applied_positions == []
    assert trade_counts == []
    assert recovery_requests == ["paper", "paper"]
    idempotency = decision.raw_response["exchange_order_fact_idempotency"]
    assert idempotency == {
        "version": "2026-08-24.exchange-order-fact-idempotency.v1",
        "status": "reused_existing_fact",
        "exchange_order_id": "okx-existing-order",
        "local_order_id": 8059,
        "authoritative_decision_id": 388831,
        "incoming_decision_id": 388833,
        "position_mutation_skipped": True,
        "recovery_requested": True,
    }
    assert raw_updates[-1]["exchange_order_fact_idempotency"] == idempotency


@pytest.mark.asyncio
async def test_existing_partial_entry_requests_recovery_without_claiming_new_execution() -> None:
    recovery_requests: list[str] = []
    persisted_positions: list[str] = []
    stages: list[tuple[str, str, str]] = []

    class ExistingPartialExecutor:
        async def place_order(
            self,
            decision: DecisionOutput,
            account_id: str | None = None,
            override_balance: float | None = None,
        ) -> ExecutionResult:
            del account_id, override_balance
            return ExecutionResult(
                order_id="existing-partial-entry",
                exchange_order_id="existing-partial-entry",
                symbol=decision.symbol,
                side="sell",
                order_type="market",
                quantity=2.0,
                price=100.0,
                status=OrderStatus.PARTIAL,
                raw_response={
                    "entry_tracking": True,
                    "entry_recovery_only": True,
                    "do_not_persist_order": True,
                },
            )

    async def okx_executor_provider(_mode: str) -> Any:
        return ExistingPartialExecutor()

    async def persist_position(*_args: Any, **_kwargs: Any) -> None:
        persisted_positions.append("called")

    service = _test_execution_service(
        okx_executor_provider=okx_executor_provider,
        position_execution_persister=persist_position,
        order_fact_recovery_trigger=lambda mode: recovery_requests.append(mode),
        stages=stages,
    )
    results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}

    result = await service.execute_candidate(
        "BTC/USDT",
        "ensemble_trader",
        _profit_first_ready_position_review_decision(),
        SimpleNamespace(warnings=[]),
        994,
        results,
        open_positions=[],
    )

    assert result is not None and result.status == OrderStatus.PARTIAL
    assert recovery_requests == ["paper"]
    assert persisted_positions == []
    assert results["decisions"][0]["executed"] is False
    assert any(
        stage == DecisionStage.EXCHANGE_CONFIRM and status == DecisionStageStatus.PENDING
        for stage, status, _reason in stages
    )


@pytest.mark.asyncio
async def test_confirmed_exit_rebalances_protection_after_position_persistence() -> None:
    calls: list[str] = []
    raw_updates: list[dict[str, Any] | None] = []

    class FilledExitExecutor:
        async def place_order(
            self,
            decision: DecisionOutput,
            account_id: str | None = None,
            override_balance: float | None = None,
        ) -> ExecutionResult:
            calls.append("exchange_fill")
            return ExecutionResult(
                order_id="local-exit-1",
                exchange_order_id="okx-exit-1",
                symbol=decision.symbol,
                side="buy",
                order_type="market",
                quantity=2.0,
                price=90.0,
                status=OrderStatus.FILLED,
                raw_response={"info": {"ordId": "okx-exit-1"}},
            )

    executor = FilledExitExecutor()

    async def okx_executor_provider(_mode: str) -> Any:
        return executor

    async def persist_position(*_args: Any, **_kwargs: Any) -> None:
        calls.append("persist_position")

    async def rebalance(received_executor: Any, decision: DecisionOutput) -> dict[str, Any]:
        assert received_executor is executor
        assert decision.action == Action.CLOSE_SHORT
        calls.append("rebalance_protection")
        return {"status": "repaired", "verified": True}

    service = _test_execution_service(
        okx_executor_provider=okx_executor_provider,
        position_execution_persister=persist_position,
        position_protection_rebalancer=rebalance,
        raw_updates=raw_updates,
    )
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="ETC/USDT",
        action=Action.CLOSE_SHORT,
        confidence=0.0,
        reasoning="dynamic exit",
        position_size_pct=0.5,
        suggested_leverage=1.0,
        raw_response={
            "dynamic_exit_policy": {
                "eligible": True,
                "close_fraction": 0.5,
                "policy_provenance": {
                    "source": "test",
                    "observation_window": "current_position",
                    "sample_count": 1,
                    "generated_at": "2026-07-15T00:00:00+00:00",
                    "strategy_version": "test",
                    "fallback_reason": "",
                },
            }
        },
    )
    results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}

    result = await service.execute_candidate(
        "ETC/USDT",
        "ensemble_trader",
        decision,
        SimpleNamespace(warnings=[]),
        89216,
        results,
        open_positions=[],
    )

    assert result is not None and result.status == OrderStatus.FILLED
    assert calls == ["exchange_fill", "persist_position", "rebalance_protection"]
    assert decision.raw_response["post_exit_protection_rebalance"] == {
        "status": "repaired",
        "verified": True,
    }
    assert raw_updates[-1]["post_exit_protection_rebalance"]["verified"] is True


@pytest.mark.asyncio
async def test_unknown_exit_result_does_not_immediately_submit_a_second_order() -> None:
    calls = 0

    class UnknownExitExecutor:
        async def place_order(
            self,
            _decision: DecisionOutput,
            account_id: str | None = None,
            override_balance: float | None = None,
        ) -> None:
            nonlocal calls
            calls += 1
            return None

    async def okx_executor_provider(_mode: str) -> Any:
        return UnknownExitExecutor()

    service = _test_execution_service(okx_executor_provider=okx_executor_provider)
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="ZAMA/USDT",
        action=Action.CLOSE_SHORT,
        confidence=1.0,
        reasoning="hard stop",
        position_size_pct=1.0,
        suggested_leverage=1.0,
        raw_response={
            "dynamic_exit_policy": {
                "eligible": True,
                "close_fraction": 1.0,
                "policy_provenance": {
                    "source": "test",
                    "observation_window": "current_position",
                    "sample_count": 1,
                    "generated_at": "2026-08-20T00:00:00+00:00",
                    "strategy_version": "test",
                    "fallback_reason": "",
                },
            }
        },
    )

    result = await service.execute_candidate(
        "ZAMA/USDT",
        "ensemble_trader",
        decision,
        SimpleNamespace(warnings=[]),
        369549,
        {"warnings": [], "decisions": [], "executions": []},
        open_positions=[],
    )

    assert calls == 1
    assert result is not None
    assert result.status == OrderStatus.OPEN
    assert result.order_id == "exit_submission_result_unknown"
    assert result.raw_response["execution_transport_unknown"] is True
    assert result.raw_response["do_not_persist_order"] is True


@pytest.mark.asyncio
async def test_execution_service_shields_exchange_submit_from_outer_timeout() -> None:
    calls: list[tuple[str, Any]] = []
    raw_updates: list[dict[str, Any] | None] = []
    stages: list[tuple[str, str, str]] = []
    reasons: list[str | None] = []

    class SlowExecutor:
        async def place_order(
            self,
            decision: DecisionOutput,
            account_id: str | None = None,
            override_balance: float | None = None,
        ) -> ExecutionResult:
            calls.append(("place_start", decision.symbol, account_id, override_balance))
            await asyncio.sleep(0.03)
            calls.append(("place_done", decision.symbol, account_id, override_balance))
            return ExecutionResult(
                order_id="local-order-1",
                exchange_order_id="okx-order-1",
                symbol=decision.symbol,
                side=decision.action.value,
                order_type="market",
                quantity=2.0,
                price=100.0,
                status=OrderStatus.FILLED,
                raw_response={},
            )

    async def okx_executor_provider(_mode: str) -> Any:
        return SlowExecutor()

    service = _test_execution_service(
        okx_executor_provider=okx_executor_provider,
        raw_updates=raw_updates,
        reasons=reasons,
        stages=stages,
    )
    results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}
    decision = _profit_first_ready_position_review_decision()

    result = await asyncio.wait_for(
        service.execute_candidate(
            "BTC/USDT",
            "ensemble_trader",
            decision,
            SimpleNamespace(warnings=[]),
            991,
            results,
            open_positions=[],
        ),
        timeout=0.01,
    )

    assert result is not None
    assert result.status == OrderStatus.FILLED
    assert result.exchange_order_id == "okx-order-1"
    assert ("place_start", "BTC/USDT", "ensemble_trader", None) in calls
    assert ("place_done", "BTC/USDT", "ensemble_trader", None) in calls
    assert results["executions"][0]["order_id"] == "local-order-1"
    assert results["decisions"][0]["executed"] is True
    assert any(stage == DecisionStage.LOCAL_SYNC and status == DecisionStageStatus.COMPLETED for stage, status, _reason in stages)
    assert not any("外层超时保护取消" in str(reason) for reason in reasons)
    assert not any("外层超时保护取消" in str(reason) for reason in reasons)
    assert not any("外层超时保护取消" in str(reason) for reason in reasons)


@pytest.mark.asyncio
async def test_execution_service_preserves_fill_when_cancelled_during_stage_recording() -> None:
    stages: list[tuple[str, str, str]] = []
    passed_stage_started = asyncio.Event()

    class ImmediateFillExecutor:
        async def place_order(
            self,
            decision: DecisionOutput,
            account_id: str | None = None,
            override_balance: float | None = None,
        ) -> ExecutionResult:
            return ExecutionResult(
                order_id="local-after-submit",
                exchange_order_id="okx-after-submit",
                symbol=decision.symbol,
                side=decision.action.value,
                order_type="market",
                quantity=2.0,
                price=100.0,
                status=OrderStatus.FILLED,
                raw_response={},
            )

    async def okx_executor_provider(_mode: str) -> Any:
        return ImmediateFillExecutor()

    async def delayed_stage_recorder(
        _decision_id: int | None,
        decision: DecisionOutput,
        stage: str,
        status: str,
        reason: str,
        _data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stages.append((stage, status, reason))
        if stage == DecisionStage.EXCHANGE_SUBMIT and status == DecisionStageStatus.PASSED:
            passed_stage_started.set()
            await asyncio.sleep(0.05)
        return decision.raw_response if isinstance(decision.raw_response, dict) else {}

    service = _test_execution_service(
        okx_executor_provider=okx_executor_provider,
        decision_stage_recorder=delayed_stage_recorder,
    )
    decision = _profit_first_ready_position_review_decision()
    results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}

    execution_task = asyncio.create_task(
        service.execute_candidate(
            "BTC/USDT",
            "ensemble_trader",
            decision,
            SimpleNamespace(warnings=[]),
            992,
            results,
            open_positions=[],
        )
    )
    await asyncio.wait_for(passed_stage_started.wait(), timeout=1.0)
    execution_task.cancel()
    result = await execution_task

    assert result is not None and result.status == OrderStatus.FILLED
    assert result.exchange_order_id == "okx-after-submit"
    assert result.raw_response["outer_cancellation_after_exchange_result"]["preserved"] is True
    assert results["decisions"][0]["executed"] is True
    assert not any(
        stage == DecisionStage.EXCHANGE_SUBMIT and status == DecisionStageStatus.FAILED
        for stage, status, _reason in stages
    )
