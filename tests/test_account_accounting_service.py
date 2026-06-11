from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

import services.account_accounting_service as accounting_module
from executor.base_executor import ExecutionResult, OrderStatus
from services.account_accounting_service import (
    AccountAccountingService,
    allocatable_balance_from_snapshot,
    balance_from_snapshot,
    tradeable_balance_from_snapshot,
)


def test_balance_snapshot_parsers_prefer_tradeable_free_then_equity() -> None:
    snapshot = {
        "free": 0.0,
        "equity": 125.0,
        "cash": 80.0,
        "total": 100.0,
        "allocatable": 90.0,
    }

    assert tradeable_balance_from_snapshot({"free": 25.0, "equity": 125.0}) == 25.0
    assert tradeable_balance_from_snapshot(snapshot) == 125.0
    assert allocatable_balance_from_snapshot(snapshot) == 125.0
    assert balance_from_snapshot({"free": 10.0, "cash": 70.0}) == 70.0
    assert tradeable_balance_from_snapshot(None) == 0.0


@pytest.mark.asyncio
async def test_account_balance_prefers_exchange_snapshot_then_allocation() -> None:
    async def snapshot(_mode: str) -> dict[str, Any] | None:
        return {"equity": 250.0, "free": 100.0}

    async def allocation_state(_mode: str) -> dict[str, Any]:
        return {"allocated": 500.0}

    service = AccountAccountingService(
        balance_snapshot_provider=snapshot,
        allocation_state_provider=allocation_state,
        model_execution_mode_provider=lambda _model_name: "paper",
    )

    assert await service.account_balance("ensemble_trader") == 250.0
    assert await service.allocated_order_balance("paper") == 100.0


@pytest.mark.asyncio
async def test_account_balance_falls_back_to_persisted_allocation() -> None:
    async def snapshot(_mode: str) -> dict[str, Any] | None:
        return None

    async def allocation_state(_mode: str) -> dict[str, Any]:
        return {"allocated": 500.0}

    service = AccountAccountingService(
        balance_snapshot_provider=snapshot,
        allocation_state_provider=allocation_state,
        model_execution_mode_provider=lambda _model_name: "live",
    )

    assert await service.account_balance("ensemble_trader") == 500.0
    assert await service.allocated_order_balance("live") == 0.0


@pytest.mark.asyncio
async def test_persist_accounting_updates_use_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    updates: list[tuple[str, float, float]] = []
    trade_results: list[tuple[str, bool]] = []
    unrealized_updates: list[tuple[str, float]] = []

    class FakeAccountRepository:
        def __init__(self, _session: Any) -> None:
            pass

        async def update_balance(
            self,
            model_name: str,
            balance_delta: float,
            realized_pnl_delta: float,
        ) -> None:
            updates.append((model_name, balance_delta, realized_pnl_delta))

        async def record_trade_result(self, model_name: str, is_win: bool) -> None:
            trade_results.append((model_name, is_win))

        async def update_unrealized_pnl(self, model_name: str, unrealized_pnl: float) -> None:
            unrealized_updates.append((model_name, unrealized_pnl))

    @asynccontextmanager
    async def session_factory():
        yield object()

    monkeypatch.setattr(accounting_module, "AccountRepository", FakeAccountRepository)
    service = AccountAccountingService(
        balance_snapshot_provider=lambda _mode: None,
        allocation_state_provider=lambda _mode: {},
        model_execution_mode_provider=lambda _model_name: "paper",
        session_factory=session_factory,
    )
    result = ExecutionResult(
        order_id="order-1",
        symbol="BTC/USDT",
        side="sell",
        order_type="market",
        quantity=1.0,
        price=100.0,
        status=OrderStatus.FILLED,
        pnl=12.5,
    )

    await service.persist_balance_delta("ensemble_trader", 3.0, 2.0)
    await service.persist_account_update("ensemble_trader", "ensemble_trader", result)
    await service.record_unrealized_pnl("ensemble_trader", 4.567)

    assert updates == [
        ("ensemble_trader", 3.0, 2.0),
        ("ensemble_trader", 12.5, 12.5),
    ]
    assert trade_results == [("ensemble_trader", True)]
    assert unrealized_updates == [("ensemble_trader", 4.57)]


@pytest.mark.asyncio
async def test_persist_balance_delta_skips_zero_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class FakeAccountRepository:
        def __init__(self, _session: Any) -> None:
            nonlocal calls
            calls += 1

    @asynccontextmanager
    async def session_factory():
        yield object()

    monkeypatch.setattr(accounting_module, "AccountRepository", FakeAccountRepository)
    service = AccountAccountingService(
        balance_snapshot_provider=lambda _mode: None,
        allocation_state_provider=lambda _mode: {},
        model_execution_mode_provider=lambda _model_name: "paper",
        session_factory=session_factory,
    )

    await service.persist_balance_delta("ensemble_trader", 0.0, 0.0)

    assert calls == 0
