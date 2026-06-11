from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from services.entry_fee_provider import EntryFeeProvider, proportional_fee


class _FakeResult:
    def __init__(self, value: Any) -> None:
        self.value = value

    def scalar_one_or_none(self) -> Any:
        return self.value


class _FakeSession:
    def __init__(self, *values: Any) -> None:
        self.values = list(values)
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> _FakeResult:
        self.statements.append(statement)
        if not self.values:
            return _FakeResult(None)
        return _FakeResult(self.values.pop(0))


def _position(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "model_name": "ensemble_trader",
        "execution_mode": "paper",
        "symbol": "BTC/USDT",
        "side": "long",
        "created_at": datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _order(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "fee": 2.0,
        "quantity": 4.0,
        "created_at": datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_proportional_fee_prorates_and_handles_bad_values():
    assert proportional_fee(2.0, close_qty=1.0, total_qty=4.0) == 0.5
    assert proportional_fee(-2.0, close_qty=2.0, total_qty=4.0) == 1.0
    assert proportional_fee(2.0, close_qty=10.0, total_qty=4.0) == 2.0
    assert proportional_fee(2.0, close_qty=1.0, total_qty=0.0) == 2.0
    assert proportional_fee("bad", close_qty=1.0, total_qty=4.0) == 0.0


@pytest.mark.asyncio
async def test_entry_fee_provider_uses_window_order_first():
    session = _FakeSession(_order(fee=3.0, quantity=6.0))

    fee = await EntryFeeProvider().entry_fee_for_position(session, _position(), close_qty=2.0)

    assert fee == 1.0
    assert len(session.statements) == 1


@pytest.mark.asyncio
async def test_entry_fee_provider_falls_back_to_previous_order_near_position_open():
    session = _FakeSession(None, _order(fee=4.0, quantity=8.0))

    fee = await EntryFeeProvider().entry_fee_for_position(session, _position(), close_qty=2.0)

    assert fee == 1.0
    assert len(session.statements) == 2


@pytest.mark.asyncio
async def test_entry_fee_provider_falls_back_to_latest_order_without_position_time():
    session = _FakeSession(_order(fee=1.5, quantity=3.0))

    fee = await EntryFeeProvider().entry_fee_for_position(
        session,
        _position(side="short", created_at=None),
        close_qty=1.0,
    )

    assert fee == 0.5
    assert len(session.statements) == 1


@pytest.mark.asyncio
async def test_entry_fee_provider_returns_zero_without_matching_order():
    session = _FakeSession(None, None, None)

    fee = await EntryFeeProvider().entry_fee_for_position(
        session,
        _position(created_at=datetime.now(UTC) - timedelta(minutes=3)),
        close_qty=1.0,
    )

    assert fee == 0.0
    assert len(session.statements) == 3
