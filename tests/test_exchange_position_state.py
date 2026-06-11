from typing import Any

import pytest

from services.exchange_position_state import (
    ExchangePositionStatePolicy,
    ExchangeProtectionMapProvider,
)


def _normalize(symbol: Any) -> str:
    value = str(symbol or "").split(":")[0]
    if value.endswith("-SWAP"):
        value = value[:-5]
    if "/" not in value and "-" in value:
        parts = value.split("-")
        if len(parts) >= 2:
            value = f"{parts[0]}/{parts[1]}"
    return value


def test_exchange_position_state_detects_open_position_shapes():
    policy = ExchangePositionStatePolicy()

    assert policy.is_open({"contracts": "1"})
    assert policy.is_open({"size": "-0.5"})
    assert policy.is_open({"info": {"pos": "2"}})
    assert not policy.is_open({"contracts": "0", "symbol": "BTC/USDT"})
    assert policy.is_open({"contracts": "bad", "symbol": "BTC/USDT"})
    assert not policy.is_open({"contracts": "bad"})


@pytest.mark.asyncio
async def test_exchange_protection_map_provider_keeps_latest_order_by_symbol_side():
    calls: list[str] = []

    class FakeExecutor:
        async def get_position_protection_orders(self, symbol):
            calls.append(symbol)
            return [
                {
                    "symbol": symbol,
                    "position_side": "long",
                    "stop_loss_price": 95.0,
                    "updated_at_ms": "10",
                },
                {
                    "symbol": symbol,
                    "position_side": "long",
                    "stop_loss_price": 96.0,
                    "updated_at_ms": "20",
                },
                {
                    "symbol": symbol,
                    "position_side": "flat",
                    "stop_loss_price": 1.0,
                    "updated_at_ms": "30",
                },
            ]

    provider = ExchangeProtectionMapProvider(
        symbol_normalizer=_normalize,
        position_open_checker=ExchangePositionStatePolicy().is_open,
    )

    result = await provider.fetch(
        FakeExecutor(),
        [
            {"symbol": "BTC-USDT-SWAP", "contracts": "1"},
            {"symbol": "BTC/USDT", "contracts": "0"},
        ],
    )

    assert calls == ["BTC/USDT"]
    assert result[("BTC/USDT", "long")]["stop_loss_price"] == 96.0
    assert ("BTC/USDT", "flat") not in result


@pytest.mark.asyncio
async def test_exchange_protection_map_provider_ignores_fetch_failures():
    class FakeExecutor:
        async def get_position_protection_orders(self, _symbol):
            raise RuntimeError("Authorization: Bearer secret-token failed")

    provider = ExchangeProtectionMapProvider(
        symbol_normalizer=_normalize,
        position_open_checker=ExchangePositionStatePolicy().is_open,
    )

    assert (
        await provider.fetch(FakeExecutor(), [{"symbol": "ETH-USDT-SWAP", "contracts": "1"}]) == {}
    )
