from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from services.exchange_backed_position_provider import ExchangeBackedPositionProvider


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
        "id": 1,
        "model_name": "ensemble_trader",
        "execution_mode": "paper",
        "symbol": "BTC/USDT",
        "side": "long",
        "created_at": datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_exchange_backed_position_provider_matches_filled_exchange_orders():
    session = _FakeSession(100, None)

    ids = await ExchangeBackedPositionProvider().ids(
        session,
        [
            _position(id=10, side="long"),
            _position(id=11, side="short"),
        ],
    )

    assert ids == {10}
    assert len(session.statements) == 2
    first_statement = str(session.statements[0].compile(compile_kwargs={"literal_binds": True}))
    second_statement = str(session.statements[1].compile(compile_kwargs={"literal_binds": True}))
    assert "orders.side = 'buy'" in first_statement
    assert "orders.exchange_order_id IS NOT NULL" in first_statement
    assert "orders.exchange_order_id != ''" in first_statement
    assert "2026-06-08 11:59:30" in first_statement
    assert "2026-06-08 12:00:30" in first_statement
    assert "orders.side = 'sell'" in second_statement


@pytest.mark.asyncio
async def test_exchange_backed_position_provider_skips_missing_ids_and_time_window():
    session = _FakeSession(100)

    ids = await ExchangeBackedPositionProvider().ids(
        session,
        [
            _position(id=None),
            _position(id=12, created_at=None),
        ],
    )

    assert ids == {12}
    assert len(session.statements) == 1
    statement = str(session.statements[0].compile(compile_kwargs={"literal_binds": True}))
    assert "orders.created_at >=" not in statement
