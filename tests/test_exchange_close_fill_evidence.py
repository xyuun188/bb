from datetime import UTC, datetime

from services.exchange_close_fill_evidence import (
    normalize_external_close_fill_evidence,
)


def test_normalize_external_close_fill_evidence_carries_nested_okx_facts() -> None:
    payload = normalize_external_close_fill_evidence(
        {
            "order_id": "order-1",
            "price": 1.407,
            "fee": -0.0471345,
            "contract_size": 100.0,
            "contract_size_source": "okx_public_instruments",
            "source": "okx_fills_history",
            "order_info": {
                "fillSz": "0.67",
                "fillPx": "1.407",
                "instId": "XRP-USDT-SWAP",
                "tradeId": "trade-1",
            },
        }
    )

    assert payload["contracts"] == 0.67
    assert payload["base_quantity"] == 67.0
    assert payload["contract_size_verified"] is True
    assert payload["trade_ids"] == ["trade-1"]
    assert payload["inst_id"] == "XRP-USDT-SWAP"
    assert payload["avg_price"] == 1.407
    assert payload["fee_abs"] == 0.0471345
    assert payload["fills_history_confirmed"] is True


def test_normalize_external_close_fill_evidence_does_not_invent_missing_facts() -> None:
    payload = normalize_external_close_fill_evidence(
        {
            "order_id": "order-2",
            "timestamp": datetime(2026, 8, 30, tzinfo=UTC),
        }
    )

    assert payload["timestamp"] == "2026-08-30T00:00:00+00:00"
    assert "contracts" not in payload
    assert "base_quantity" not in payload
    assert "contract_size_verified" not in payload
    assert "trade_ids" not in payload
