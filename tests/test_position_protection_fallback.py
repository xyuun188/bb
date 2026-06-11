from types import SimpleNamespace
from typing import Any

import pytest

from services.position_protection_fallback import PositionProtectionFallbackPolicy


class _FakeResult:
    def __init__(self, value: Any) -> None:
        self.value = value

    def scalar_one_or_none(self) -> Any:
        return self.value


class _FakeSession:
    def __init__(self, *values: Any) -> None:
        self.values = list(values)
        self.execute_count = 0

    async def execute(self, _statement: Any) -> _FakeResult:
        self.execute_count += 1
        if not self.values:
            return _FakeResult(None)
        return _FakeResult(self.values.pop(0))


def _decision(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "id": 42,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.2,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_position_protection_fallback_recovers_long_prices_from_order_decision():
    session = _FakeSession(_decision(id=101, stop_loss_pct="0.05", take_profit_pct="0.2"))
    policy = PositionProtectionFallbackPolicy()

    result = await policy.protection_from_decision(
        session,
        symbol="BTC/USDT",
        side="long",
        entry_price=100.0,
        order=SimpleNamespace(decision_id=101),
    )

    assert result == {
        "stop_loss_price": 95.0,
        "take_profit_price": 120.0,
        "source": "latest_executed_entry_decision",
        "decision_id": 101,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.2,
    }
    assert session.execute_count == 1


@pytest.mark.asyncio
async def test_position_protection_fallback_recovers_short_prices_from_latest_decision():
    session = _FakeSession(_decision(id=202, stop_loss_pct=0.05, take_profit_pct=0.1))
    policy = PositionProtectionFallbackPolicy()

    result = await policy.protection_from_decision(
        session,
        symbol="ETH/USDT",
        side="short",
        entry_price=100.0,
    )

    assert result["stop_loss_price"] == 105.0
    assert result["take_profit_price"] == 90.0
    assert result["decision_id"] == 202
    assert session.execute_count == 1


@pytest.mark.asyncio
async def test_position_protection_fallback_uses_latest_decision_when_order_decision_missing():
    session = _FakeSession(None, _decision(id=303, stop_loss_pct=0.03, take_profit_pct=0.08))
    policy = PositionProtectionFallbackPolicy()

    result = await policy.protection_from_decision(
        session,
        symbol="SOL/USDT",
        side="long",
        entry_price=50.0,
        order=SimpleNamespace(decision_id=999),
    )

    assert result["stop_loss_price"] == 48.5
    assert result["take_profit_price"] == 54.0
    assert result["decision_id"] == 303
    assert session.execute_count == 2


@pytest.mark.asyncio
async def test_position_protection_fallback_returns_empty_for_invalid_or_missing_data():
    policy = PositionProtectionFallbackPolicy()

    assert (
        await policy.protection_from_decision(
            _FakeSession(_decision()),
            symbol="BTC/USDT",
            side="flat",
            entry_price=100.0,
        )
        == {}
    )
    assert (
        await policy.protection_from_decision(
            _FakeSession(_decision(stop_loss_pct=0.0, take_profit_pct=0.0)),
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
        )
        == {}
    )
    assert (
        await policy.protection_from_decision(
            _FakeSession(None),
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
        )
        == {}
    )
