from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from risk_manager.engine import RiskEngine
from services.authoritative_trade_outcome import build_authoritative_trade_outcome
from services.decision_state import DecisionStage, DecisionStageStatus
from services.execution_result_factory import ExecutionResultFactory
from services.execution_service import ExecutionService, _return_entry_contract_result
from services.okx_training_facts import build_okx_history_training_sample
from services.paper_training import build_paper_training_contract
from services.trade_execution_contract import validate_entry_execution_contract
from services.trading_policies import PolicyGateResult
from services.training_data_quality import annotate_training_payload


async def _noop_async(*_args: Any, **_kwargs: Any) -> Any:
    return None


def _test_execution_service(
    *,
    okx_executor_provider,
    entry_policy_evaluator=None,
    exit_policy_evaluator=None,
    raw_updates: list[dict[str, Any] | None] | None = None,
    reasons: list[str | None] | None = None,
    stages: list[tuple[str, str, str]] | None = None,
    trade_logger=None,
    position_execution_persister=None,
    position_protection_rebalancer=None,
    order_fact_recovery_trigger=None,
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
        decision_stage_recorder=record_stage,
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
        trade_count_incrementer=lambda: None,
        position_execution_persister=position_execution_persister or _noop_async,
        position_protection_rebalancer=position_protection_rebalancer or _noop_async,
        order_fact_recovery_trigger=order_fact_recovery_trigger,
        open_positions_execution_applier=lambda *_args: None,
        decision_executed_marker=_noop_async,
        account_update_persister=_noop_async,
        account_balance_provider=lambda _model: _noop_async(),
        decision_outcome_marker=_noop_async,
        entry_policy_evaluator=entry_policy_evaluator or allow_entry,
        exit_policy_evaluator=exit_policy_evaluator or allow_entry,
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
    return _dynamic_return_ready_decision()


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


def test_live_rules_canary_bypasses_model_promotion_return_distribution() -> None:
    decision = _dynamic_return_ready_decision()
    decision.raw_response["production_trade_gate"] = {
        "version": "test",
        "mode": "live_rules_canary",
        "can_trade": True,
        "decision_authority": "rules",
        "model_can_influence": False,
        "risk": {"max_notional_usdt": 100.0},
    }

    result = _return_entry_contract_result(decision, "live")

    assert result.passed is True
    assert result.data["return_execution_contract"] == "live_rules_canary"
    assert result.data["production_permission"] is True


def test_live_rules_canary_respects_gate_notional_limit() -> None:
    decision = _dynamic_return_ready_decision()
    decision.raw_response["production_trade_gate"] = {
        "version": "test",
        "mode": "live_rules_canary",
        "can_trade": True,
        "decision_authority": "rules",
        "model_can_influence": False,
        "risk": {"max_notional_usdt": 50.0},
    }

    result = _return_entry_contract_result(decision, "live")

    assert result.passed is False
    assert result.blocker == "live_rules_canary_contract_incomplete"
    assert "rules_canary_order_notional_above_gate_limit" in result.data[
        "block_reasons"
    ]


@pytest.mark.asyncio
async def test_paper_training_entry_close_and_loss_reach_authoritative_training() -> None:
    decision = _paper_training_ready_decision()
    contract, reasons = validate_entry_execution_contract(decision.raw_response)
    assert reasons == []
    assert contract["contract_lifecycle"] == "paper_training"
    assert RiskEngine._dynamic_risk_contract_reason(decision) is None
    assert _return_entry_contract_result(decision, "paper").passed is True

    class FilledExecutor:
        async def place_order(
            self,
            current: DecisionOutput,
            account_id: str | None = None,
            override_balance: float | None = None,
        ) -> ExecutionResult:
            del account_id, override_balance
            is_entry = current.action == Action.LONG
            return ExecutionResult(
                order_id="local-entry" if is_entry else "local-close",
                exchange_order_id="okx-entry" if is_entry else "okx-close",
                symbol=current.symbol,
                side="buy" if is_entry else "sell",
                order_type="market",
                quantity=1.0,
                price=100.0 if is_entry else 95.0,
                status=OrderStatus.FILLED,
                raw_response={},
            )

    executor = FilledExecutor()

    async def executor_provider(_mode: str) -> FilledExecutor:
        return executor

    service = _test_execution_service(okx_executor_provider=executor_provider)
    entry_result = await service.execute_candidate(
        decision.symbol,
        decision.model_name,
        decision,
        SimpleNamespace(warnings=[]),
        321,
        {"warnings": [], "decisions": [], "executions": []},
        open_positions=[],
    )
    assert entry_result is not None and entry_result.status == OrderStatus.FILLED
    assert decision.raw_response["paper_training_order_identity"]["client_order_id"] == (
        "BBPT321"
    )

    close_decision = DecisionOutput(
        model_name=decision.model_name,
        symbol=decision.symbol,
        action=Action.CLOSE_LONG,
        confidence=0.0,
        reasoning="authoritative paper close",
        position_size_pct=1.0,
        suggested_leverage=1.0,
        raw_response={},
    )
    close_result = await service.execute_candidate(
        close_decision.symbol,
        close_decision.model_name,
        close_decision,
        SimpleNamespace(warnings=[]),
        322,
        {"warnings": [], "decisions": [], "executions": []},
        open_positions=[
            {
                "symbol": decision.symbol,
                "side": "long",
                "quantity": 1.0,
                "is_open": True,
            }
        ],
    )
    assert close_result is not None and close_result.status == OrderStatus.FILLED

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
        },
        sync_status="synced",
    )
    orders = {
        entry_result.exchange_order_id: SimpleNamespace(
            okx_fill_contracts=entry_result.quantity,
            okx_trade_ids="trade-entry",
            decision_id=321,
        ),
        close_result.exchange_order_id: SimpleNamespace(
            okx_fill_contracts=close_result.quantity,
            okx_trade_ids="trade-close",
            decision_id=322,
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
    assert outcome["strategy_entry_kind"] == "loss_tolerant_paper_training"
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
    assert not any("外层超时保护取消" in str(reason) for reason in reasons)
    assert not any("外层超时保护取消" in str(reason) for reason in reasons)
    assert not any("外层超时保护取消" in str(reason) for reason in reasons)
    assert not any("外层超时保护取消" in str(reason) for reason in reasons)
