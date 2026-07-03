from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from core.symbols import normalize_trading_symbol
from services.sync_service import (
    _build_local_position_profit_first_metadata,
    _merge_local_position_candidates,
    _okx_close_fill_order_payload,
    normalized_open_position_context,
)


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
                "posId": "auction-pos",
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
    assert context["okx_inst_id"] == "AUCTION-USDT-SWAP"
    assert context["okx_pos_id"] == "auction-pos"
    assert context["notional"] == pytest.approx(35.1434)
    assert context["unrealized_pnl"] == pytest.approx(0.0398)
    assert str(context["created_at"]).startswith(opened_at.isoformat(timespec="seconds")[:19])
    assert not str(context["created_at"]).startswith(updated_at.isoformat(timespec="seconds")[:19])


def test_normalized_open_position_context_preserves_profit_first_binding_metadata() -> None:
    context = normalized_open_position_context(
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "contracts": "10",
            "entry_price": "100",
            "current_price": "101",
            "entry_exchange_order_id": "entry-a,entry-b",
            "entry_legs": [
                {"exchange_order_id": "entry-a", "profit_first_exit_plan_id": "pfep-a"},
                {"exchange_order_id": "entry-b", "profit_first_exit_plan_id": "pfep-b"},
            ],
            "profit_first_trade_plan": {"exit_plan_id": "pfep-a", "decision_lane": "meaningful_entry"},
            "profit_first_exit_plan": {"exit_plan_id": "pfep-a", "max_hold_minutes": 360},
            "profit_first_exit_plan_id": "pfep-a",
            "info": {"ctVal": "0.1", "instId": "BTC-USDT-SWAP", "posId": "btc-pos", "pos": "10"},
        },
        symbol_normalizer=normalize_trading_symbol,
        float_parser=_float,
    )

    assert context["entry_exchange_order_id"] == "entry-a,entry-b"
    assert len(context["entry_legs"]) == 2
    assert context["entry_legs"][1]["profit_first_exit_plan_id"] == "pfep-b"
    assert context["profit_first_exit_plan_id"] == "pfep-a"
    assert context["profit_first_trade_plan"]["decision_lane"] == "meaningful_entry"
    assert context["profit_first_exit_plan"]["max_hold_minutes"] == 360


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


def test_build_local_position_profit_first_metadata_attaches_entry_leg_plan_ids() -> None:
    position = SimpleNamespace(id=1, entry_exchange_order_id="entry-a,entry-b")
    order_a = SimpleNamespace(exchange_order_id="entry-a", decision_id=11)
    order_b = SimpleNamespace(exchange_order_id="entry-b", decision_id=12)
    decision_a = SimpleNamespace(
        raw_llm_response={
            "profit_first_trade_plan": {"exit_plan_id": "pfep-a", "decision_lane": "small"},
            "profit_first_exit_plan": {"exit_plan_id": "pfep-a"},
        }
    )
    decision_b = SimpleNamespace(
        raw_llm_response={
            "profit_first_trade_plan": {"exit_plan_id": "pfep-b", "decision_lane": "small"},
            "profit_first_exit_plan": {"exit_plan_id": "pfep-b"},
        }
    )

    metadata = _build_local_position_profit_first_metadata(
        [position],
        orders_by_exchange_id={"entry-a": order_a, "entry-b": order_b},
        decisions_by_id={11: decision_a, 12: decision_b},
    )

    assert metadata[1]["entry_exchange_order_id"] == "entry-a,entry-b"
    assert metadata[1]["entry_legs"] == [
        {"exchange_order_id": "entry-a", "profit_first_exit_plan_id": "pfep-a"},
        {"exchange_order_id": "entry-b", "profit_first_exit_plan_id": "pfep-b"},
    ]
    assert metadata[1].get("profit_first_exit_plan_id", "") == ""


def test_merge_local_position_candidates_keeps_all_entry_legs_when_symbol_side_collapses() -> None:
    merged = _merge_local_position_candidates(
        [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "created_at": "2026-07-03T01:00:00+00:00",
                "entry_exchange_order_id": "entry-a",
                "entry_legs": [
                    {"exchange_order_id": "entry-a", "profit_first_exit_plan_id": "pfep-a"}
                ],
                "profit_first_exit_plan_id": "pfep-a",
                "profit_first_trade_plan": {"exit_plan_id": "pfep-a"},
                "profit_first_exit_plan": {"exit_plan_id": "pfep-a"},
            },
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "created_at": "2026-07-03T02:00:00+00:00",
                "entry_exchange_order_id": "entry-b",
                "entry_legs": [
                    {"exchange_order_id": "entry-b", "profit_first_exit_plan_id": "pfep-b"}
                ],
                "profit_first_exit_plan_id": "pfep-b",
                "profit_first_trade_plan": {"exit_plan_id": "pfep-b"},
                "profit_first_exit_plan": {"exit_plan_id": "pfep-b"},
            },
        ],
        exchange_position={"symbol": "BTC-USDT-SWAP", "side": "long", "info": {"posId": "btc-pos"}},
    )

    assert merged["entry_exchange_order_id"] == "entry-a,entry-b"
    assert merged["created_at"] == "2026-07-03T01:00:00+00:00"
    assert {leg["exchange_order_id"] for leg in merged["entry_legs"]} == {"entry-a", "entry-b"}
    assert merged["profit_first_exit_plan_id"] == ""
    assert merged["profit_first_trade_plan"] == {}
    assert merged["profit_first_exit_plan"] == {}


def test_okx_close_fill_order_payload_persists_native_fill_fact() -> None:
    filled_at = datetime(2026, 7, 2, 1, 20, 53, tzinfo=UTC)

    payload = _okx_close_fill_order_payload(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="SAND/USDT",
        side="buy",
        quantity=1455.0,
        price=0.0498,
        fee=0.0289836,
        decision_id=123,
        close_order_id="3706182741831417856",
        filled_at=filled_at,
        close_fill={
            "source": "okx_fills_history",
            "pnl": -3.58808175,
            "contracts": 145.5,
            "quantity": 1455.0,
            "order_info": {
                "instId": "SAND-USDT-SWAP",
                "tradeId": "sand-close-trade",
                "fillSz": "145.5",
                "fillPnl": "-3.58808175",
                "ts": str(int(filled_at.timestamp() * 1000)),
            },
        },
        okx_inst_id="SAND-USDT-SWAP",
    )

    assert payload["okx_inst_id"] == "SAND-USDT-SWAP"
    assert payload["okx_sync_status"] == "okx_confirmed"
    assert payload["okx_fill_contracts"] == pytest.approx(145.5)
    assert payload["okx_fill_pnl"] == pytest.approx(-3.58808175)
    assert payload["okx_raw_fills"]["fills_history_confirmed"] is True
    assert payload["okx_raw_fills"]["order_id"] == "3706182741831417856"
    assert payload["okx_raw_fills"]["trade_ids"] == ["sand-close-trade"]
    assert payload["okx_raw_fills"]["rows"][0]["instId"] == "SAND-USDT-SWAP"
