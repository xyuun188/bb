from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

import services.execution_allocation_service as allocation_module
from services.execution_allocation_service import (
    ExecutionAllocationService,
    beijing_start_utc,
    snapshot_execution_equity,
    snapshot_free_balance,
)


def _position(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "model_name": "ensemble_trader",
        "execution_mode": "paper",
        "symbol": "BTC/USDT",
        "side": "long",
        "is_open": False,
        "quantity": 1.0,
        "entry_price": 100.0,
        "leverage": 1.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "closed_at": None,
        "created_at": datetime(2026, 6, 8, tzinfo=UTC),
        "entry_exchange_order_id": "entry-ok",
        "close_exchange_order_id": "close-ok",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_snapshot_helpers_preserve_allocation_state_legacy_priority() -> None:
    snapshot = {"free": 12.0, "total": 80.0, "equity": 90.0, "allocatable": 100.0}

    assert snapshot_free_balance(snapshot) == 12.0
    assert snapshot_execution_equity(snapshot, fallback_free=12.0) == 100.0
    assert snapshot_execution_equity({"free": 7.0}, fallback_free=7.0) == 7.0
    assert snapshot_execution_equity(None) == 0.0


def test_beijing_start_utc_uses_beijing_midnight() -> None:
    now = datetime(2026, 6, 8, 9, 30, tzinfo=UTC)

    assert beijing_start_utc(now) == datetime(2026, 6, 7, 16, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_execution_allocation_filters_open_positions_by_exchange_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions = [
        _position(
            is_open=True,
            symbol="BTC/USDT",
            side="long",
            quantity=2.0,
            entry_price=100.0,
            leverage=4.0,
            unrealized_pnl=5.0,
        ),
        _position(
            is_open=True,
            symbol="ETH/USDT",
            side="long",
            quantity=3.0,
            entry_price=50.0,
            leverage=3.0,
            unrealized_pnl=9.0,
        ),
        _position(
            is_open=False,
            realized_pnl=10.0,
            closed_at=datetime(2026, 6, 8, 2, 0, tzinfo=UTC),
        ),
        _position(
            is_open=False,
            realized_pnl=-3.0,
            closed_at=datetime(2026, 6, 7, 10, 0, tzinfo=UTC),
        ),
    ]
    baseline_calls: list[dict[str, Any]] = []

    class FakeTradeRepository:
        def __init__(self, _session: Any) -> None:
            pass

        async def get_position_records(self, **kwargs: Any) -> list[Any]:
            assert kwargs["execution_mode"] == "paper"
            assert kwargs["model_name"] == "ensemble_trader"
            assert kwargs["limit"] == 5000
            return positions

    class FakeExecutor:
        def __init__(self) -> None:
            self.soft_calls = 0
            self.strict_calls = 0

        async def get_positions(self) -> list[dict[str, Any]]:
            self.soft_calls += 1
            raise AssertionError("allocation must prefer strict OKX-native positions")

        async def get_positions_strict(self) -> list[dict[str, Any]]:
            self.strict_calls += 1
            return [{"symbol": "BTC/USDT", "side": "long", "is_open": True}]

    fake_executor = FakeExecutor()

    @asynccontextmanager
    async def session_factory():
        yield object()

    async def baseline_provider(_session: Any, **kwargs: Any) -> dict[str, Any]:
        baseline_calls.append(kwargs)
        return {
            "today_equity_pnl": 8.0,
            "today_equity_baseline": 101.0,
            "today_equity_baseline_total_pnl": 4.0,
            "today_equity_baseline_at": "2026-06-08T00:00:00+08:00",
            "today_equity_baseline_source": "unit",
            "today_snapshot_date": "2026-06-08",
        }

    async def balance_snapshot(_mode: str) -> dict[str, Any]:
        return {"free": 12.0, "allocatable": 100.0}

    monkeypatch.setattr(allocation_module, "TradeRepository", FakeTradeRepository)
    service = ExecutionAllocationService(
        balance_snapshot_provider=balance_snapshot,
        active_executor_provider=lambda _mode: fake_executor,
        exchange_position_open_checker=lambda payload: bool(payload.get("is_open")),
        symbol_normalizer=lambda symbol: str(symbol or "").upper(),
        session_factory=session_factory,
        equity_baseline_provider=baseline_provider,
        now_provider=lambda: datetime(2026, 6, 8, 9, 30, tzinfo=UTC),
    )

    state = await service.calculate("paper")

    assert state["allocated_balance"] == 100.0
    assert state["remaining_allocation"] == 12.0
    assert state["used_margin"] == 50.0
    assert state["unrealized_pnl"] == 5.0
    assert state["realized_profit"] == 10.0
    assert state["realized_loss"] == 3.0
    assert state["today_realized_profit"] == 10.0
    assert state["today_realized_loss"] == 0.0
    assert state["realized_pnl"] == 7.0
    assert state["total_pnl"] == 12.0
    assert state["today_equity_pnl"] == 8.0
    assert state["today_risk_pnl"] == 8.0
    assert state["today_equity_baseline"] == 101.0
    assert baseline_calls[0]["current_equity"] == 100.0
    assert baseline_calls[0]["total_pnl"] == 12.0
    assert baseline_calls[0]["positions"] == positions
    assert fake_executor.strict_calls == 1
    assert fake_executor.soft_calls == 0


@pytest.mark.asyncio
async def test_execution_allocation_excludes_open_positions_when_exchange_snapshot_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions = [
        _position(
            is_open=True, symbol="BTC/USDT", quantity=1.0, entry_price=100.0, unrealized_pnl=-4.0
        ),
        _position(
            is_open=False, realized_pnl=2.0, closed_at=datetime(2026, 6, 8, 1, 0, tzinfo=UTC)
        ),
    ]

    class FakeTradeRepository:
        def __init__(self, _session: Any) -> None:
            pass

        async def get_position_records(self, **_kwargs: Any) -> list[Any]:
            return positions

    @asynccontextmanager
    async def session_factory():
        yield object()

    async def failing_baseline(_session: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("baseline unavailable")

    async def balance_snapshot(_mode: str) -> dict[str, Any]:
        return {"free": 20.0}

    monkeypatch.setattr(allocation_module, "TradeRepository", FakeTradeRepository)
    service = ExecutionAllocationService(
        balance_snapshot_provider=balance_snapshot,
        active_executor_provider=lambda _mode: None,
        exchange_position_open_checker=lambda _payload: True,
        symbol_normalizer=lambda symbol: str(symbol or ""),
        session_factory=session_factory,
        equity_baseline_provider=failing_baseline,
        now_provider=lambda: datetime(2026, 6, 8, 9, 30, tzinfo=UTC),
    )

    state = await service.calculate("paper")

    assert state["used_margin"] == 0.0
    assert state["unrealized_pnl"] == 0.0
    assert state["realized_pnl"] == 2.0
    assert state["total_pnl"] == 2.0
    assert state["today_equity_pnl"] is None
    assert state["today_total_pnl"] is None
    assert state["today_risk_pnl"] is None


@pytest.mark.asyncio
async def test_execution_allocation_excludes_untrusted_closed_trade_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions = [
        _position(
            is_open=False,
            realized_pnl=10.0,
            closed_at=datetime(2026, 6, 8, 2, 0, tzinfo=UTC),
            entry_exchange_order_id="entry-ok",
            close_exchange_order_id="close-ok",
        ),
        _position(
            is_open=False,
            realized_pnl=99.0,
            closed_at=datetime(2026, 6, 8, 3, 0, tzinfo=UTC),
            entry_exchange_order_id="",
            close_exchange_order_id="",
        ),
        _position(
            is_open=False,
            realized_pnl=-25.0,
            closed_at=datetime(2026, 6, 8, 4, 0, tzinfo=UTC),
            entry_exchange_order_id="entry-dirty",
            close_exchange_order_id="manual_close:1",
        ),
    ]

    class FakeTradeRepository:
        def __init__(self, _session: Any) -> None:
            pass

        async def get_position_records(self, **_kwargs: Any) -> list[Any]:
            return positions

    @asynccontextmanager
    async def session_factory():
        yield object()

    async def baseline_provider(_session: Any, **kwargs: Any) -> dict[str, Any]:
        return {"today_equity_pnl": kwargs["total_pnl"]}

    async def balance_snapshot(_mode: str) -> dict[str, Any]:
        return {"free": 50.0, "allocatable": 100.0}

    monkeypatch.setattr(allocation_module, "TradeRepository", FakeTradeRepository)
    service = ExecutionAllocationService(
        balance_snapshot_provider=balance_snapshot,
        active_executor_provider=lambda _mode: None,
        exchange_position_open_checker=lambda _payload: True,
        symbol_normalizer=lambda symbol: str(symbol or ""),
        session_factory=session_factory,
        equity_baseline_provider=baseline_provider,
        now_provider=lambda: datetime(2026, 6, 8, 9, 30, tzinfo=UTC),
    )

    state = await service.calculate("paper")

    assert state["realized_profit"] == 10.0
    assert state["realized_loss"] == 0.0
    assert state["today_realized_profit"] == 10.0
    assert state["today_realized_loss"] == 0.0
    assert state["realized_pnl"] == 10.0
    assert state["total_pnl"] == 10.0
