from __future__ import annotations

import pytest

from core.symbols import normalize_trading_symbol
from services.exchange_position_state import (
    exchange_position_display_valuation,
    exchange_snapshot_price,
    exchange_snapshot_unrealized,
    parse_exchange_position_snapshot,
)


def test_parse_okx_position_snapshot_prefers_info_markpx_upl_and_ctval() -> None:
    snapshot = parse_exchange_position_snapshot(
        {
            "symbol": "PROS/USDT:USDT",
            "side": "long",
            "contracts": 0,
            "markPrice": 0,
            "entryPrice": 0,
            "info": {
                "instId": "PROS-USDT-SWAP",
                "pos": "46",
                "ctVal": "1",
                "avgPx": "0.4054",
                "markPx": "0.4059",
                "last": "0.5547",
                "upl": "-0.82",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
    )

    assert snapshot is not None
    assert snapshot["symbol"] == "PROS/USDT"
    assert snapshot["side"] == "long"
    assert snapshot["mark_price"] == pytest.approx(0.4059)
    assert snapshot["last_price"] == pytest.approx(0.5547)
    assert snapshot["entry_price"] == pytest.approx(0.4054)
    assert snapshot["quantity"] == pytest.approx(46.0)
    assert snapshot["upl"] == pytest.approx(-0.82)
    assert exchange_snapshot_price(snapshot) == pytest.approx(0.4059)
    assert exchange_snapshot_unrealized(snapshot, "long") == pytest.approx(-0.82)


def test_exchange_position_display_valuation_does_not_use_stale_local_profit() -> None:
    valuation = exchange_position_display_valuation(
        {
            "mark_price": 0.4059,
            "last_price": 0.5547,
            "entry_price": 0.4054,
            "quantity": 46.0,
            "upl": -0.82,
        },
        "long",
        fallback_current_price=0.5547,
        fallback_unrealized_pnl=6.8678,
        fallback_entry_price=0.4054,
        fallback_quantity=46,
    )

    assert valuation["current_price"] == pytest.approx(0.4059)
    assert valuation["unrealized_pnl"] == pytest.approx(-0.82)
    assert valuation["pnl_source"] == "okx_position_upl"
