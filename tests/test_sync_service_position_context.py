from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.symbols import normalize_trading_symbol
from services.sync_service import normalized_open_position_context


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def test_normalized_open_position_context_uses_contract_value_and_stable_open_time() -> None:
    opened_at = datetime.now(UTC) - timedelta(hours=7)
    updated_at = datetime.now(UTC)

    context = normalized_open_position_context(
        {
            "model_name": "ensemble_trader",
            "symbol": "AUCTION/USDT:USDT",
            "side": "short",
            "quantity": 99.5,
            "contracts": 99.5,
            "entryPrice": 3.532,
            "markPrice": 3.528,
            "unrealizedPnl": 0.0398,
            "leverage": 10,
            "info": {
                "instId": "AUCTION-USDT-SWAP",
                "pos": "99.5",
                "ctVal": "0.1",
                "avgPx": "3.532",
                "markPx": "3.528",
                "upl": "0.0398",
                "cTime": str(int(opened_at.timestamp() * 1000)),
                "uTime": str(int(updated_at.timestamp() * 1000)),
            },
        },
        symbol_normalizer=normalize_trading_symbol,
        float_parser=_float,
    )

    assert context["symbol"] == "AUCTION/USDT"
    assert context["quantity"] == pytest.approx(9.95)
    assert context["base_quantity"] == pytest.approx(9.95)
    assert context["contracts"] == pytest.approx(99.5)
    assert context["contract_size"] == pytest.approx(0.1)
    assert context["notional"] == pytest.approx(35.1434)
    assert context["unrealized_pnl"] == pytest.approx(0.0398)
    assert str(context["created_at"]).startswith(opened_at.isoformat(timespec="seconds")[:19])
    assert not str(context["created_at"]).startswith(updated_at.isoformat(timespec="seconds")[:19])


def test_normalized_open_position_context_fallback_keeps_contracts_separate_from_base_quantity() -> (
    None
):
    context = normalized_open_position_context(
        {
            "model_name": "ensemble_trader",
            "symbol": "AUCTION/USDT",
            "side": "short",
            "entry_price": 3.532,
            "current_price": 3.528,
            "contracts": 99.5,
            "contract_size": 0.1,
            "created_at": "2026-06-23T00:00:00+00:00",
            "is_open": True,
        },
        symbol_normalizer=normalize_trading_symbol,
        float_parser=_float,
    )

    assert context["quantity"] == pytest.approx(9.95)
    assert context["base_quantity"] == pytest.approx(9.95)
    assert context["raw_quantity"] == pytest.approx(0.0)
    assert context["contracts"] == pytest.approx(99.5)
    assert context["notional"] == pytest.approx(35.1434)


def test_normalized_open_position_context_uses_contract_value_without_price_snapshot() -> None:
    context = normalized_open_position_context(
        {
            "model_name": "ensemble_trader",
            "symbol": "AUCTION/USDT:USDT",
            "side": "short",
            "contracts": "99.5",
            "entry_price": "3.532",
            "current_price": "3.532",
            "info": {
                "ctVal": "0.1",
                "pos": "99.5",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
        float_parser=_float,
    )

    assert context["quantity"] == pytest.approx(9.95)
    assert context["base_quantity"] == pytest.approx(9.95)
    assert context["raw_quantity"] == pytest.approx(0.0)
    assert context["contracts"] == pytest.approx(99.5)
    assert context["contract_size"] == pytest.approx(0.1)
    assert context["notional"] == pytest.approx(35.1434)
