from __future__ import annotations

from types import SimpleNamespace

from models.trade import Position
from services.okx_order_fact_sync import OKX_SYNC_EXECUTION_RESULT_CONFIRMED
from services.trade_fact_trust import (
    closed_position_trade_fact_untrusted_reason_with_orders,
    closed_position_trade_fact_trusted,
    closed_position_trade_fact_untrusted_reason,
    filter_trusted_closed_positions,
)


def test_closed_position_fact_requires_entry_and_close_order_links() -> None:
    trusted = SimpleNamespace(
        id=1,
        is_open=False,
        realized_pnl=1.2,
        entry_exchange_order_id="entry-ok",
        close_exchange_order_id="close-ok",
    )
    missing_entry = SimpleNamespace(
        id=2,
        is_open=False,
        realized_pnl=1.2,
        entry_exchange_order_id="",
        close_exchange_order_id="close-ok",
    )
    missing_close = SimpleNamespace(
        id=3,
        is_open=False,
        realized_pnl=1.2,
        entry_exchange_order_id="entry-ok",
        close_exchange_order_id=None,
    )
    manual_close = SimpleNamespace(
        id=4,
        is_open=False,
        realized_pnl=1.2,
        entry_exchange_order_id="entry-ok",
        close_exchange_order_id="manual_close:local-only",
    )

    assert closed_position_trade_fact_trusted(trusted) is True
    assert closed_position_trade_fact_untrusted_reason(missing_entry) == (
        "missing_entry_exchange_order_id"
    )
    assert closed_position_trade_fact_untrusted_reason(missing_close) == (
        "missing_close_exchange_order_id"
    )
    assert closed_position_trade_fact_untrusted_reason(manual_close) == (
        "manual_close_exchange_order_id"
    )


def test_legacy_in_memory_objects_without_fact_fields_remain_trusted() -> None:
    row = SimpleNamespace(id=1, is_open=False, realized_pnl=-2.0)

    assert closed_position_trade_fact_trusted(row) is True


def test_transient_position_without_explicit_fact_fields_remains_trusted() -> None:
    row = Position(
        id=9,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        realized_pnl=-2.0,
        is_open=False,
    )

    assert closed_position_trade_fact_trusted(row) is True


def test_filter_trusted_closed_positions_reports_quarantine_reasons() -> None:
    rows = [
        SimpleNamespace(
            id=1,
            is_open=False,
            realized_pnl=1.0,
            entry_exchange_order_id="entry-ok",
            close_exchange_order_id="close-ok",
        ),
        SimpleNamespace(
            id=2,
            is_open=False,
            realized_pnl=-1.0,
            entry_exchange_order_id="entry-ok",
            close_exchange_order_id="",
        ),
    ]

    trusted, report = filter_trusted_closed_positions(rows)

    assert [row.id for row in trusted] == [1]
    assert report["quarantined"] == 1
    assert report["reason_counts"] == {"missing_close_exchange_order_id": 1}
    assert report["position_ids"] == [2]


def test_execution_result_confirmed_orders_are_trusted_trade_facts() -> None:
    position = SimpleNamespace(
        id=1,
        is_open=False,
        realized_pnl=1.2,
        entry_exchange_order_id="entry-ok",
        close_exchange_order_id="close-ok",
    )
    orders = {
        "entry-ok": SimpleNamespace(
            exchange_order_id="entry-ok",
            okx_sync_status=OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
        ),
        "close-ok": SimpleNamespace(
            exchange_order_id="close-ok",
            okx_sync_status=OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
        ),
    }

    assert closed_position_trade_fact_untrusted_reason_with_orders(position, orders) is None
