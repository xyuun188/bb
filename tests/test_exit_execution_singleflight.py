from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.exit_execution_singleflight import (
    EXIT_INTENT_KEY,
    PROFIT_LOCK_LEDGER_KEY,
    ExitExecutionSingleFlightService,
    preserve_exit_execution_intent,
)


class _Session:
    def __init__(self) -> None:
        self.flush_calls = 0

    async def flush(self) -> None:
        self.flush_calls += 1


class _Repo:
    def __init__(self, positions: list[Any]) -> None:
        self.positions = positions

    async def get_matching_open_positions_for_update(self, **kwargs: Any) -> list[Any]:
        return [
            position
            for position in self.positions
            if position.is_open
            and position.model_name == kwargs["model_name"]
            and position.symbol == kwargs["symbol"]
            and position.side == kwargs["side"]
            and position.execution_mode == kwargs["execution_mode"]
        ]

    async def get_positions_for_update(self, position_ids: tuple[int, ...]) -> list[Any]:
        return [position for position in self.positions if position.id in position_ids]


def _position(position_id: int = 6069) -> Any:
    return SimpleNamespace(
        id=position_id,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ZAMA/USDT",
        side="short",
        is_open=True,
        okx_pos_id="3843743943360217089",
        entry_exchange_order_id="3843000000000000000",
        created_at=datetime(2026, 8, 19, 23, 0, tzinfo=UTC),
        current_management_contract={"policy": "dynamic_exit"},
    )


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="position_review",
        symbol="ZAMA/USDT",
        action=Action.CLOSE_SHORT,
        confidence=1.0,
        reasoning="hard stop",
        position_size_pct=1.0,
    )


def _profit_lock_decision() -> DecisionOutput:
    decision = _decision()
    decision.position_size_pct = 0.4
    decision.raw_response = {
        "dynamic_exit_policy": {
            "eligible": True,
            "hard_risk": False,
            "close_fraction": 0.4,
            "target_close_fraction": 0.5,
            "lifecycle_closed_fraction": 0.1,
            "incremental_close_fraction": 0.4,
            "lifecycle_net_pnl_usdt": 12.0,
            "profit_lock_pressure": 0.4,
        }
    }
    return decision


def _service(
    repo: _Repo,
    session: _Session,
    now: list[datetime],
) -> ExitExecutionSingleFlightService:
    @asynccontextmanager
    async def session_context():
        yield session

    return ExitExecutionSingleFlightService(
        session_context_factory=session_context,
        trade_repo_factory=lambda _session: repo,
        now_provider=lambda: now[0],
    )


@pytest.mark.asyncio
async def test_persistent_exit_lease_blocks_duplicate_and_survives_service_restart() -> None:
    position = _position()
    repo = _Repo([position])
    session = _Session()
    now = [datetime(2026, 8, 20, 1, 0, tzinfo=UTC)]
    service = _service(repo, session, now)

    first = await service.acquire(
        model_name="ensemble_trader",
        execution_mode="paper",
        decision=_decision(),
        decision_id=369549,
    )
    duplicate = await service.acquire(
        model_name="ensemble_trader",
        execution_mode="paper",
        decision=_decision(),
        decision_id=369550,
    )

    assert first.acquired is True
    assert duplicate.acquired is False
    assert duplicate.state == "submitting"
    assert duplicate.retry_after_seconds == pytest.approx(150.0)

    result = ExecutionResult(
        order_id="okx_native_full_close_not_confirmed",
        symbol="ZAMA/USDT",
        side="buy",
        order_type="market",
        quantity=0.0,
        price=0.04381,
        status=OrderStatus.OPEN,
        raw_response={"exit_tracking": True, "okx_native_close_position": True},
    )
    await service.finish(first, result)

    restarted_service = _service(repo, session, now)
    after_restart = await restarted_service.acquire(
        model_name="ensemble_trader",
        execution_mode="paper",
        decision=_decision(),
        decision_id=369551,
    )
    assert after_restart.acquired is False
    assert after_restart.state == "submitted_unconfirmed"
    assert after_restart.retry_after_seconds == pytest.approx(120.0)
    assert position.current_management_contract["policy"] == "dynamic_exit"
    assert position.current_management_contract[EXIT_INTENT_KEY]["attempt_count"] == 1

    now[0] += timedelta(seconds=121)
    retry = await restarted_service.acquire(
        model_name="ensemble_trader",
        execution_mode="paper",
        decision=_decision(),
        decision_id=369552,
    )
    assert retry.acquired is True
    assert retry.attempt_count == 2


@pytest.mark.asyncio
async def test_temporary_exchange_error_uses_wait_state_instead_of_duplicate_submit() -> None:
    position = _position()
    repo = _Repo([position])
    session = _Session()
    now = [datetime(2026, 8, 20, 2, 0, tzinfo=UTC)]
    service = _service(repo, session, now)
    lease = await service.acquire(
        model_name="ensemble_trader",
        execution_mode="paper",
        decision=_decision(),
        decision_id=369600,
    )
    result = ExecutionResult(
        order_id="rejected",
        symbol="ZAMA/USDT",
        side="buy",
        order_type="market",
        quantity=0.0,
        price=0.0,
        status=OrderStatus.REJECTED,
        raw_response={
            "error": "OKX API error [50001]: Service temporarily unavailable."
        },
    )

    await service.finish(lease, result)
    blocked = await service.acquire(
        model_name="ensemble_trader",
        execution_mode="paper",
        decision=_decision(),
        decision_id=369601,
    )
    waiting = service.waiting_result(_decision(), blocked)

    assert blocked.acquired is False
    assert blocked.state == "exchange_temporarily_unavailable"
    assert blocked.retry_after_seconds == pytest.approx(60.0)
    assert waiting.status == OrderStatus.OPEN
    assert waiting.raw_response["exit_singleflight_wait"] is True


@pytest.mark.asyncio
async def test_no_local_position_is_terminal_skip_and_never_a_submit_lease() -> None:
    repo = _Repo([])
    session = _Session()
    now = [datetime(2026, 8, 20, 3, 0, tzinfo=UTC)]
    service = _service(repo, session, now)

    lease = await service.acquire(
        model_name="ensemble_trader",
        execution_mode="paper",
        decision=_decision(),
        decision_id=369700,
    )

    assert lease.acquired is False
    assert lease.state == "no_local_position"
    waiting = service.waiting_result(_decision(), lease)
    assert waiting.quantity == 0.0
    assert waiting.raw_response["do_not_persist_order"] is True
    assert "前一笔平仓已经完成" in waiting.raw_response["message"]


def test_position_contract_refresh_preserves_exit_lease_without_stale_policy_fields() -> None:
    previous = {
        "contract_version": "old",
        "stale_field": True,
        EXIT_INTENT_KEY: {
            "version": "2026-08-20.exit-singleflight.v1",
            "token": "lease-token",
            "state": "submitted_unconfirmed",
        },
        PROFIT_LOCK_LEDGER_KEY: {
            "version": "2026-09-19.profit-lock-exit-ledger.v1",
            "realized_quantity": 3.0,
        },
    }
    refreshed = {"contract_version": "new", "management_eligible": True}

    merged = preserve_exit_execution_intent(previous, refreshed)

    assert merged["contract_version"] == "new"
    assert merged["management_eligible"] is True
    assert "stale_field" not in merged
    assert merged[EXIT_INTENT_KEY] == previous[EXIT_INTENT_KEY]
    assert merged[EXIT_INTENT_KEY] is not previous[EXIT_INTENT_KEY]
    assert merged[PROFIT_LOCK_LEDGER_KEY] == previous[PROFIT_LOCK_LEDGER_KEY]
    assert merged[PROFIT_LOCK_LEDGER_KEY] is not previous[PROFIT_LOCK_LEDGER_KEY]


@pytest.mark.asyncio
async def test_confirmed_profit_lock_fill_updates_separate_idempotent_ledger() -> None:
    position = _position()
    position.current_management_contract["lifecycle_entry_quantity"] = 10.0
    repo = _Repo([position])
    session = _Session()
    now = [datetime(2026, 9, 20, 1, 0, tzinfo=UTC)]
    service = _service(repo, session, now)
    decision = _profit_lock_decision()

    lease = await service.acquire(
        model_name="ensemble_trader",
        execution_mode="paper",
        decision=decision,
        decision_id=509586,
    )
    result = ExecutionResult(
        order_id="profit-lock-fill",
        exchange_order_id="3938000000000000001",
        symbol="ZAMA/USDT",
        side="buy",
        order_type="market",
        quantity=4.0,
        price=0.04,
        status=OrderStatus.FILLED,
    )

    await service.finish(lease, result)
    await service.finish(lease, result)

    intent = position.current_management_contract[EXIT_INTENT_KEY]
    ledger = position.current_management_contract[PROFIT_LOCK_LEDGER_KEY]
    assert intent["profit_lock_exit"] is True
    assert intent["exit_reason_class"] == "profit_lock"
    assert intent["requested_close_fraction"] == pytest.approx(0.4)
    assert intent["target_close_fraction"] == pytest.approx(0.5)
    assert intent["lifecycle_closed_fraction"] == pytest.approx(0.1)
    assert intent["incremental_close_fraction"] == pytest.approx(0.4)
    assert ledger["realized_quantity"] == pytest.approx(4.0)
    assert ledger["realized_fraction"] == pytest.approx(0.4)
    assert ledger["last_decision_id"] == 509586
    assert ledger["last_exchange_order_id"] == "3938000000000000001"
    assert len(ledger["intent_fill_quantities"]) == 1


@pytest.mark.asyncio
async def test_risk_reduction_fill_does_not_update_profit_lock_ledger() -> None:
    position = _position()
    position.current_management_contract["lifecycle_entry_quantity"] = 10.0
    repo = _Repo([position])
    session = _Session()
    now = [datetime(2026, 9, 20, 2, 0, tzinfo=UTC)]
    service = _service(repo, session, now)

    lease = await service.acquire(
        model_name="ensemble_trader",
        execution_mode="paper",
        decision=_decision(),
        decision_id=509600,
    )
    result = ExecutionResult(
        order_id="risk-reduction-fill",
        exchange_order_id="3938000000000000002",
        symbol="ZAMA/USDT",
        side="buy",
        order_type="market",
        quantity=5.0,
        price=0.04,
        status=OrderStatus.FILLED,
    )

    await service.finish(lease, result)

    assert position.current_management_contract[EXIT_INTENT_KEY]["profit_lock_exit"] is False
    assert PROFIT_LOCK_LEDGER_KEY not in position.current_management_contract
