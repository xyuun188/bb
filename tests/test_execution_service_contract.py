from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.execution_result_factory import ExecutionResultFactory
from services.execution_service import ExecutionService, _profit_first_entry_contract_result
from services.trading_policies import PolicyGateResult


async def _noop_async(*_args: Any, **_kwargs: Any) -> Any:
    return None


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
