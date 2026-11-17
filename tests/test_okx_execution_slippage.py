import pytest

from services.okx_execution_slippage import (
    OKX_FILL_MARK_SLIPPAGE_SOURCE,
    build_okx_fill_mark_slippage,
    signed_fill_mark_slippage_usdt,
)


def test_fill_mark_slippage_measures_adverse_buy_execution_from_all_rows() -> None:
    fact = build_okx_fill_mark_slippage(
        order_id="order-1",
        inst_id="BTC-USDT-SWAP",
        side="buy",
        contracts=3.0,
        average_price=100.66666666666667,
        contract_size=0.01,
        rows=[
            {
                "ordId": "order-1",
                "instId": "BTC-USDT-SWAP",
                "tradeId": "trade-1",
                "side": "buy",
                "fillSz": "1",
                "fillPx": "100",
                "fillMarkPx": "99",
            },
            {
                "ordId": "order-1",
                "instId": "BTC-USDT-SWAP",
                "tradeId": "trade-2",
                "side": "buy",
                "fillSz": "2",
                "fillPx": "101",
                "fillMarkPx": "100.5",
            },
        ],
    )

    assert fact["complete"] is True
    assert fact["source"] == OKX_FILL_MARK_SLIPPAGE_SOURCE
    assert fact["fill_mark_vwap"] == pytest.approx(100.0)
    assert fact["actual_notional_usdt"] == pytest.approx(3.02)
    assert fact["adverse_slippage_usdt"] == pytest.approx(0.02)
    assert fact["adverse_slippage_pct"] == pytest.approx(0.02 / 3.02 * 100.0)
    assert signed_fill_mark_slippage_usdt(fact) == pytest.approx(0.02)


def test_fill_mark_slippage_does_not_turn_sell_price_improvement_into_cost() -> None:
    fact = build_okx_fill_mark_slippage(
        order_id="order-2",
        inst_id="ETH-USDT-SWAP",
        side="sell",
        contracts=2.0,
        average_price=101.0,
        contract_size=0.1,
        rows=[
            {
                "ordId": "order-2",
                "instId": "ETH-USDT-SWAP",
                "tradeId": "trade-3",
                "side": "sell",
                "fillSz": "2",
                "fillPx": "101",
                "fillMarkPx": "100",
            }
        ],
    )

    assert fact["complete"] is True
    assert fact["adverse_slippage_usdt"] == 0.0
    assert fact["adverse_slippage_pct"] == 0.0
    assert signed_fill_mark_slippage_usdt(fact) == pytest.approx(-0.2)


@pytest.mark.parametrize("field,value", [
    ("complete", False),
    ("fill_mark_notional_usdt", None),
    ("fill_mark_notional_usdt", 5.0),
    ("fill_mark_vwap", float("nan")),
])
def test_signed_slippage_requires_verified_mark_notional(field, value) -> None:
    fact = build_okx_fill_mark_slippage(
        order_id="entry", inst_id="XPL-USDT-SWAP", side="sell",
        contracts=7.0, average_price=0.0435, contract_size=10.0,
        rows=[{
            "ordId": "entry", "instId": "XPL-USDT-SWAP", "side": "sell",
            "tradeId": "one", "fillSz": "7", "fillPx": "0.0435",
            "fillMarkPx": "0.1148",
        }],
    )
    assert signed_fill_mark_slippage_usdt(fact) == pytest.approx(4.991)
    fact[field] = value
    assert signed_fill_mark_slippage_usdt(fact) is None


def test_fill_mark_slippage_fails_closed_when_any_fill_mark_is_missing() -> None:
    fact = build_okx_fill_mark_slippage(
        order_id="order-3",
        inst_id="BTC-USDT-SWAP",
        side="buy",
        contracts=1.0,
        average_price=100.0,
        contract_size=0.01,
        rows=[
            {
                "ordId": "order-3",
                "instId": "BTC-USDT-SWAP",
                "tradeId": "trade-4",
                "side": "buy",
                "fillSz": "1",
                "fillPx": "100",
                "fillMarkPx": "",
            }
        ],
    )

    assert fact["complete"] is False
    assert fact["source"] == ""
    assert fact["reasons"] == ["fill_row_mark_price_invalid"]
    assert fact["contracts"] == 1.0
    assert fact["fill_vwap"] == 100.0
    assert fact["actual_notional_usdt"] == 1.0
    assert fact["fill_mark_vwap"] is None
    assert fact["trade_ids"] == ["trade-4"]
