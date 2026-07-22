from __future__ import annotations

import pytest

from services.okx_realized_pnl import gross_pnl_with_okx_override
from services.position_settlement import (
    SETTLEMENT_FORMULA,
    build_position_settlement_snapshot,
    proportional_signed_value,
)


@pytest.mark.parametrize(
    ("side", "entry_price", "exit_price", "quantity", "expected"),
    [
        ("long", 100.0, 110.0, 2.0, 20.0),
        ("long", 100.0, 90.0, 2.0, -20.0),
        ("short", 100.0, 90.0, 2.0, 20.0),
        ("short", 100.0, 110.0, 2.0, -20.0),
    ],
)
def test_long_and_short_price_pnl_direction_is_consistent(
    side: str,
    entry_price: float,
    exit_price: float,
    quantity: float,
    expected: float,
) -> None:
    gross, source = gross_pnl_with_okx_override(
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        close_qty=quantity,
    )

    assert gross == pytest.approx(expected)
    assert source == "local_price_formula"


def test_fee_after_settlement_uses_signed_funding_and_absolute_trade_fees() -> None:
    snapshot = build_position_settlement_snapshot(
        close_fill_pnl=20.0,
        entry_fee=-0.4,
        close_fee=-0.6,
        funding_fee=-0.25,
        status="reconciled",
        source="test",
    )

    assert snapshot.entry_fee == pytest.approx(0.4)
    assert snapshot.close_fee == pytest.approx(0.6)
    assert snapshot.funding_fee == pytest.approx(-0.25)
    assert snapshot.realized_pnl == pytest.approx(18.75)
    assert snapshot.as_position_payload()["settlement_raw"]["formula"] == SETTLEMENT_FORMULA


def test_partial_close_allocates_entry_fee_and_funding_once_by_closed_quantity() -> None:
    close_quantity = 2.0
    total_quantity = 5.0

    allocated_entry_fee = proportional_signed_value(1.0, close_quantity, total_quantity)
    allocated_funding = proportional_signed_value(-0.5, close_quantity, total_quantity)
    snapshot = build_position_settlement_snapshot(
        close_fill_pnl=20.0,
        entry_fee=allocated_entry_fee,
        close_fee=0.6,
        funding_fee=allocated_funding,
        status="settling",
        source="test_partial_close",
    )

    assert allocated_entry_fee == pytest.approx(0.4)
    assert allocated_funding == pytest.approx(-0.2)
    assert snapshot.realized_pnl == pytest.approx(18.8)
    assert total_quantity - close_quantity == pytest.approx(3.0)


def test_okx_fill_pnl_is_allocated_to_partial_close_without_leverage_multiplier() -> None:
    gross, source = gross_pnl_with_okx_override(
        side="long",
        entry_price=100.0,
        exit_price=110.0,
        close_qty=2.0,
        okx_payload={"native_close_fill": {"pnl": 50.0, "quantity": 5.0}},
        okx_total_qty=5.0,
    )

    assert gross == pytest.approx(20.0)
    assert source == "okx_fill_pnl"


def test_leverage_does_not_change_underlying_trade_pnl() -> None:
    one_x, _ = gross_pnl_with_okx_override(
        side="long", entry_price=100.0, exit_price=110.0, close_qty=2.0
    )
    five_x, _ = gross_pnl_with_okx_override(
        side="long", entry_price=100.0, exit_price=110.0, close_qty=2.0
    )

    assert one_x == pytest.approx(20.0)
    assert five_x == one_x
