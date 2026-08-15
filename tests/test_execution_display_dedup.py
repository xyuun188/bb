from datetime import UTC, datetime
from types import SimpleNamespace

from web_dashboard.api.trades import _deduplicate_execution_orders


def _row(**overrides):
    values = {
        "id": 1,
        "model_name": "ensemble_trader",
        "execution_mode": "paper",
        "symbol": "XPL/USDT",
        "side": "sell",
        "quantity": 420.0,
        "price": 0.0755,
        "filled_at": datetime(2026, 8, 14, 19, 32, tzinfo=UTC),
        "created_at": datetime(2026, 8, 14, 19, 32, tzinfo=UTC),
        "decision_id": 99,
        "exchange_order_id": "local-1",
        "okx_sync_status": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_local_and_okx_backfill_twin_is_displayed_once():
    local = _row(id=1)
    okx = _row(
        id=2,
        model_name="okx_authoritative_sync",
        decision_id=None,
        exchange_order_id="okx-1",
        okx_sync_status="okx_only_backfilled",
    )

    assert [row.id for row in _deduplicate_execution_orders([okx, local])] == [1]


def test_two_local_fills_are_not_merged():
    first = _row(id=1)
    second = _row(id=2, exchange_order_id="local-2", decision_id=100)

    assert {row.id for row in _deduplicate_execution_orders([first, second])} == {1, 2}


def test_different_price_is_not_merged():
    local = _row(id=1)
    okx = _row(
        id=2,
        model_name="okx_authoritative_sync",
        decision_id=None,
        exchange_order_id="okx-2",
        okx_sync_status="okx_only_backfilled",
        price=0.0756,
    )

    assert {row.id for row in _deduplicate_execution_orders([local, okx])} == {1, 2}


def test_same_exchange_execution_duplicate_is_displayed_once():
    first = _row(id=1, exchange_order_id="okx-duplicate", okx_sync_status="okx_confirmed")
    second = _row(id=2, exchange_order_id="okx-duplicate", okx_sync_status="okx_confirmed", decision_id=None)

    assert [row.id for row in _deduplicate_execution_orders([second, first])] == [1]
