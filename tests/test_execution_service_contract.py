from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.decision_state import DecisionStage, DecisionStageStatus
from services.execution_result_factory import ExecutionResultFactory
from services.execution_service import ExecutionService, _return_entry_contract_result
from services.trading_policies import PolicyGateResult


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


def test_dynamic_return_contract_accepts_complete_governed_entry() -> None:
    result = _return_entry_contract_result(_dynamic_return_ready_decision())
    assert result.passed is True
    assert result.data["return_execution_contract"] == "complete"


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
