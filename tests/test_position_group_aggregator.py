from __future__ import annotations

from services.position_group_aggregator import PositionGroupAggregator


def _normalize(symbol: str | None) -> str | None:
    if not symbol:
        return None
    return str(symbol).split(":", 1)[0]


def test_position_group_aggregator_weighted_values_and_earliest_open_time() -> None:
    aggregate = PositionGroupAggregator(_normalize).aggregate(
        [
            {
                "side": "long",
                "quantity": 2.0,
                "entry_price": 100.0,
                "current_price": 110.0,
                "unrealized_pnl": 4.0,
                "stop_loss": 94.0,
                "take_profit": 120.0,
                "leverage": 2.0,
                "created_at": "2026-06-10T02:00:00Z",
            },
            {
                "side": "long",
                "quantity": 1.0,
                "entry_price": 130.0,
                "current_price": 140.0,
                "unrealized_pnl": -1.0,
                "stop_loss_price": 118.0,
                "take_profit_price": 150.0,
                "leverage": 5.0,
                "created_at": "2026-06-10T01:00:00Z",
            },
            {
                "side": "short",
                "quantity": 5.0,
                "entry_price": 80.0,
                "unrealized_pnl": 99.0,
            },
        ],
        "ensemble_trader",
        "BTC/USDT:USDT",
        "long",
    )

    assert aggregate["model_name"] == "ensemble_trader"
    assert aggregate["symbol"] == "BTC/USDT"
    assert aggregate["side"] == "long"
    assert aggregate["quantity"] == 3.0
    assert aggregate["entry_price"] == 110.0
    assert aggregate["current_price"] == 120.0
    assert aggregate["notional"] == 330.0
    assert aggregate["unrealized_pnl"] == 3.0
    assert aggregate["stop_loss"] == 102.0
    assert aggregate["take_profit"] == 130.0
    assert aggregate["leverage"] == 3.0
    assert aggregate["created_at"] == "2026-06-10T01:00:00Z"
    assert aggregate["rows"] == 2


def test_position_group_aggregator_skips_invalid_fragments() -> None:
    aggregate = PositionGroupAggregator(_normalize).aggregate(
        [
            {"side": "long", "quantity": 0.0, "entry_price": 100.0},
            {"side": "long", "quantity": 1.0, "entry_price": 0.0},
        ],
        "",
        "SOL/USDT",
        "long",
    )

    assert aggregate == {}
