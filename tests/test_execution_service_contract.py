from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.decision_state import DecisionStage, DecisionStageStatus
from services.execution_result_factory import ExecutionResultFactory
from services.execution_service import ExecutionService, _profit_first_entry_contract_result
from services.trading_policies import PolicyGateResult


async def _noop_async(*_args: Any, **_kwargs: Any) -> Any:
    return None


def _test_execution_service(
    *,
    okx_executor_provider,
    raw_updates: list[dict[str, Any] | None] | None = None,
    reasons: list[str | None] | None = None,
    stages: list[tuple[str, str, str]] | None = None,
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
        untradable_exchange_error_checker=lambda _text: False,
        untradable_symbol_rememberer=lambda *_args: None,
        transient_entry_exchange_error_checker=lambda _text: False,
        temporary_entry_block_rememberer=lambda *_args: None,
        transient_entry_block_minutes_provider=lambda _text: 5.0,
        trade_logger=_noop_async,
        exchange_confirmed_checker=lambda result: bool(
            result
            and result.status == OrderStatus.FILLED
            and result.exchange_order_id
        ),
        exit_progress_checker=lambda _result: False,
        no_exchange_position_result_checker=lambda _result: False,
        trade_count_incrementer=lambda: None,
        position_execution_persister=_noop_async,
        open_positions_execution_applier=lambda *_args: None,
        decision_executed_marker=_noop_async,
        market_no_opportunity_symbol_clearer=lambda _symbol: None,
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
        exit_cooldown_recorder=lambda *_args: None,
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
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.84,
        reasoning="position review add entry",
        position_size_pct=0.06,
        suggested_leverage=4.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        feature_snapshot={"close": 100.0},
        raw_response={
            "analysis_type": "position_review",
            "current_price": 100.0,
            "strategy_learning_context": {"strategy_profile_id": "balanced_probe"},
            "opportunity_score": {
                "score": 3.4,
                "side": "long",
                "expected_return_pct": 1.05,
                "expected_net_return_pct": 0.9,
                "fee_pct": 0.05,
                "slippage_pct": 0.04,
                "expected_loss_pct": 0.20,
                "profit_quality_ratio": 1.25,
                "reward_risk_ratio": 2.5,
                "server_profit_loss_probability": 0.38,
                "tail_risk_score": 0.62,
                "side_realized_pnl_usdt": 2.0,
                "ml_aligned": True,
                "local_profit_aligned": True,
                "timeseries_aligned": True,
                "expert_aligned": True,
                "evidence_score": {
                    "tier": "normal",
                    "effective_score": 88,
                    "components": [
                        {"source": "sentiment", "status": "aligned"},
                        {"source": "shadow_memory", "status": "aligned"},
                    ],
                },
            },
            "profit_risk_sizing": {
                "quality_tier": "high_profit",
                "position_size_pct": 0.06,
                "final_notional_usdt": 120.0,
                "planned_stop_loss_usdt": 2.8,
                "max_stop_loss_usdt": 4.0,
                "expected_profit_usdt": 1.08,
            },
        },
    )


def test_profit_first_entry_contract_late_attaches_position_review_plan_and_ladder() -> None:
    decision = _profit_first_ready_position_review_decision()

    result = _profit_first_entry_contract_result(decision)

    assert result.passed is True
    raw = decision.raw_response
    assert raw["profit_first_trade_plan"]["analysis_type"] == "position_review"
    assert raw["profit_first_trade_plan"]["is_complete_for_real_trade"] is True
    ladder = raw["profit_risk_sizing"]["profit_first_position_ladder"]
    assert ladder["lane"] == "meaningful_entry"
    assert ladder["late_attached_at"] == "execution_pre_submit"


def test_profit_first_entry_contract_blocks_incomplete_entry_before_submit() -> None:
    decision = _entry_decision("BTC/USDT")

    result = _profit_first_entry_contract_result(decision)

    assert result.passed is False
    assert result.blocker == "profit_first_trade_plan_incomplete"
    assert result.data["shadow_only"] is True
    assert result.data["profit_first_trade_plan"]["is_complete_for_real_trade"] is False


def test_profit_first_entry_contract_blocks_low_payoff_one_x_probe_before_submit() -> None:
    decision = _entry_decision("LINK/USDT")
    decision.position_size_pct = 0.006
    decision.suggested_leverage = 1.0
    decision.stop_loss_pct = 0.02
    decision.take_profit_pct = 0.04
    decision.raw_response = {
        "analysis_type": "entry_candidate",
        "current_price": 100.0,
        "strategy_learning_context": {"strategy_profile_id": "profile_1"},
        "opportunity_score": {
            "score": 1.4,
            "side": "short",
            "expected_net_return_pct": 0.2,
            "fee_pct": 0.05,
            "slippage_pct": 0.04,
            "expected_loss_pct": 0.2,
            "profit_quality_ratio": 0.5,
            "reward_risk_ratio": 0.9,
            "server_profit_loss_probability": 0.48,
            "tail_risk_score": 0.7,
            "ml_aligned": True,
            "local_profit_aligned": True,
            "server_profit_expected_return_pct": 0.4,
            "evidence_score": {"tier": "normal", "effective_score": 72.0},
        },
        "profit_risk_sizing": {
            "low_payoff_quality": True,
            "quality_tier": "base",
            "high_quality_entry": False,
            "position_size_pct": 0.006,
            "final_notional_usdt": 30.0,
            "planned_stop_loss_usdt": 1.5,
            "max_stop_loss_usdt": 1.5,
            "expected_profit_usdt": 0.12,
            "dynamic_leverage_decision": {
                "final_integer_leverage": 1,
                "limiting_factor": "risk_budget",
                "reasons": ["limited_by_risk_budget"],
            },
            "profit_first_position_ladder": {
                "lane": "tiny_probe",
                "adjusted_size_pct": 0.006,
            },
        },
    }

    result = _profit_first_entry_contract_result(decision)

    assert result.passed is False
    assert result.blocker == "profit_first_defensive_probe_shadow"
    assert result.data["shadow_only"] is True
    assert result.data["skip_kind"] == "profit_first_defensive_probe_shadow"
    assert result.data["dynamic_leverage_limiting_factor"] == "risk_budget"


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
        untradable_exchange_error_checker=lambda _text: False,
        untradable_symbol_rememberer=lambda *_args: None,
        transient_entry_exchange_error_checker=lambda _text: False,
        temporary_entry_block_rememberer=lambda *_args: None,
        transient_entry_block_minutes_provider=lambda _text: 5.0,
        trade_logger=_noop_async,
        exchange_confirmed_checker=lambda _result: False,
        exit_progress_checker=lambda _result: False,
        no_exchange_position_result_checker=lambda _result: False,
        trade_count_incrementer=lambda: None,
        position_execution_persister=_noop_async,
        open_positions_execution_applier=lambda *_args: None,
        decision_executed_marker=_noop_async,
        market_no_opportunity_symbol_clearer=lambda _symbol: None,
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
        exit_cooldown_recorder=lambda *_args: None,
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
    assert results["decisions"][0]["execution_status"] == "skipped"


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
