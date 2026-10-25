from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from data_feed import okx_rest_client, okx_ws_client
from data_feed.okx_rest_client import OKXRestClient
from data_feed.okx_ws_client import OKXWebSocketClient
from executor import okx_executor
from executor.okx_executor import OKXExecutor

SOURCE_TIMESTAMP_MS = 1_783_990_800_000
CONTRACT_SPEC = {
    "instId": "BTC-USDT-SWAP",
    "instType": "SWAP",
    "uly": "BTC-USDT",
    "ctVal": "0.01",
    "ctMult": "1",
}


@pytest.fixture
def receipt_clock(monkeypatch):
    class Clock(datetime):
        current = datetime(2026, 10, 22, 12, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current.astimezone(tz) if tz is not None else cls.current.replace(tzinfo=None)

    for module in (okx_rest_client, okx_ws_client, okx_executor):
        monkeypatch.setattr(module, "datetime", Clock)
    return Clock


@pytest.mark.asyncio
async def test_orderbook_receipt_is_network_completion_and_survives_cache(
    monkeypatch, receipt_clock,
) -> None:
    client = OKXRestClient()
    completed_at = receipt_clock.current + timedelta(seconds=4)
    calls = 0

    async def fetch(method, *args, **kwargs):
        nonlocal calls
        calls += 1
        assert method == "fetch_order_book"
        receipt_clock.current = completed_at
        return {
            "timestamp": SOURCE_TIMESTAMP_MS,
            "bids": [[100.0, 2.0]],
            "asks": [[100.1, 3.0]],
        }

    monkeypatch.setattr(client, "_ccxt_call", fetch)
    first = await client.fetch_order_book_metrics("BTC/USDT", contract_spec=CONTRACT_SPEC)
    receipt_clock.current += timedelta(hours=1)
    cached = await client.fetch_order_book_metrics("BTC/USDT", contract_spec=CONTRACT_SPEC)

    assert first["orderbook_fact"]["received_at"] == completed_at.isoformat()
    assert first["orderbook_fact"]["source_timestamp_ms"] == SOURCE_TIMESTAMP_MS
    assert cached["orderbook_fact"] == first["orderbook_fact"]
    assert cached["orderbook_cached"] is True
    assert calls == 1


@pytest.mark.asyncio
async def test_funding_receipt_is_distinct_from_source_and_scheduled_event(
    monkeypatch, receipt_clock,
) -> None:
    client = OKXRestClient()
    completed_at = receipt_clock.current + timedelta(seconds=3)
    event_at = int((completed_at + timedelta(hours=4)).timestamp() * 1000)
    calls = 0

    async def fetch(method, *args, **kwargs):
        nonlocal calls
        calls += 1
        assert method == "fetch_funding_rate"
        receipt_clock.current = completed_at
        return {
            "fundingRate": 0.0001,
            "info": {
                "ts": str(SOURCE_TIMESTAMP_MS),
                "fundingTime": str(event_at),
                "nextFundingTime": str(event_at + 28_800_000),
            },
        }

    monkeypatch.setattr(client, "_ccxt_call", fetch)
    first = await client._cached_derivatives_route(
        "funding", "BTC/USDT", lambda: client.fetch_funding_rate("BTC/USDT"),
    )
    receipt_clock.current += timedelta(hours=1)
    cached = await client._cached_derivatives_route(
        "funding", "BTC/USDT", lambda: client.fetch_funding_rate("BTC/USDT"),
    )

    assert first["funding_data_available"] is True
    assert first["funding_rate_received_at"] == completed_at.isoformat()
    assert first["funding_rate_observed_at"] == str(SOURCE_TIMESTAMP_MS)
    assert first["funding_time"] == str(event_at)
    assert cached["funding_rate_received_at"] == first["funding_rate_received_at"]
    assert calls == 1


@pytest.mark.asyncio
async def test_failed_funding_request_has_no_received_clock(monkeypatch, receipt_clock) -> None:
    client = OKXRestClient()

    async def fetch(*args, **kwargs):
        raise ConnectionError("unavailable")

    monkeypatch.setattr(client, "_ccxt_call", fetch)
    result = await client.fetch_funding_rate("BTC/USDT")

    assert result["funding_data_available"] is False
    assert result.get("funding_rate_received_at") is None


@pytest.mark.asyncio
async def test_reference_route_clocks_are_preserved_on_cache_hit(
    monkeypatch, receipt_clock,
) -> None:
    client = OKXRestClient()
    completed_at = receipt_clock.current + timedelta(seconds=5)
    calls = []

    async def fetch(method, *args, **kwargs):
        calls.append(method)
        receipt_clock.current = completed_at
        if method == "publicGetPublicMarkPrice":
            return {"code": "0", "data": [{
                "instId": "BTC-USDT-SWAP",
                "markPx": "100",
                "ts": str(SOURCE_TIMESTAMP_MS),
            }]}
        assert method == "publicGetMarketIndexTickers"
        return {"code": "0", "data": [{
            "instId": "BTC-USDT",
            "idxPx": "100.1",
            "ts": str(SOURCE_TIMESTAMP_MS + 1),
        }]}

    monkeypatch.setattr(client, "_ccxt_call", fetch)
    result = await client.fetch_reference_prices("BTC/USDT", contract_spec=CONTRACT_SPEC)
    receipt_clock.current += timedelta(hours=1)
    cached = await client.fetch_reference_prices("BTC/USDT", contract_spec=CONTRACT_SPEC)

    assert result["mark_price_fact"]["received_at"] == completed_at.isoformat()
    assert result["index_price_fact"]["received_at"] == completed_at.isoformat()
    assert result["mark_price_fact"]["source_timestamp_ms"] == SOURCE_TIMESTAMP_MS
    assert result["index_price_fact"]["source_timestamp_ms"] == SOURCE_TIMESTAMP_MS + 1
    assert cached["reference_prices_cached"] is True
    assert cached["mark_price_fact"] == result["mark_price_fact"]
    assert cached["index_price_fact"] == result["index_price_fact"]
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_route", ["mark", "index"])
async def test_reference_market_batch_without_source_clock_stays_unavailable(
    monkeypatch,
    receipt_clock,
    missing_route: str,
) -> None:
    client = OKXRestClient()
    source_ms = int(receipt_clock.current.timestamp() * 1000)

    async def fetch(method, *_args, **_kwargs):
        missing_timestamp = "mark" if method == "publicGetPublicMarkPrice" else "index"
        mark = {
            "instId": "BTC-USDT-SWAP",
            "markPx": "100",
            "ts": str(source_ms),
        }
        index = {
            "instId": "BTC-USDT",
            "idxPx": "100.1",
            "ts": str(source_ms),
        }
        if missing_timestamp == missing_route:
            (mark if missing_route == "mark" else index).pop("ts")
        row = mark if missing_timestamp == "mark" else index
        return {"code": "0", "data": [row]}

    monkeypatch.setattr(client, "_ccxt_call", fetch)
    result = await client.fetch_reference_prices(
        "BTC/USDT",
        contract_spec=CONTRACT_SPEC,
    )

    fact_key = "mark_price_fact" if missing_route == "mark" else "index_price_fact"
    assert result["reference_prices_data_available"] is False
    assert result[fact_key]["source_timestamp_ms"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("has_receipt", [True, False])
async def test_reference_prices_require_ticker_receipt_when_routes_are_empty(
    monkeypatch, receipt_clock, has_receipt,
) -> None:
    client = OKXRestClient()
    ticker_received_at = (
        (receipt_clock.current - timedelta(seconds=10)).isoformat() if has_receipt else None
    )
    ticker_calls = 0

    async def fetch(*args, **kwargs):
        return {"data": []}

    async def ticker(*args, **kwargs):
        nonlocal ticker_calls
        ticker_calls += 1
        payload = {
            "last": 100.0,
            "timestamp": SOURCE_TIMESTAMP_MS,
        }
        if has_receipt:
            payload["received_at"] = ticker_received_at
        return payload

    monkeypatch.setattr(client, "_ccxt_call", fetch)
    monkeypatch.setattr(client, "fetch_ticker", ticker)
    result = await client.fetch_reference_prices("BTC/USDT", contract_spec=CONTRACT_SPEC)

    assert result["reference_prices_data_available"] is False
    assert "reference_prices_fallback" not in result
    assert result["mark_price"] == 0.0
    assert result["index_price"] == 0.0
    assert ticker_calls == 0


@pytest.mark.asyncio
async def test_reference_prices_do_not_promote_authoritative_ticker_to_reference_facts(
    monkeypatch, receipt_clock,
) -> None:
    client = OKXRestClient()
    authoritative_received_at = (receipt_clock.current + timedelta(seconds=7)).isoformat()
    fallback_calls: list[str] = []

    async def fetch(*args, **kwargs):
        return {"data": []}

    async def short_ticker(*args, **kwargs):
        assert kwargs.get("wait_for_completion") is False
        return {"ticker_refresh_in_background": True}

    async def authoritative(inst_id: str) -> dict:
        fallback_calls.append(inst_id)
        return {
            "id": inst_id,
            "last": 100.0,
            "bid": 99.9,
            "ask": 100.1,
            "timestamp": SOURCE_TIMESTAMP_MS,
            "received_at": authoritative_received_at,
            "info": {"instId": inst_id},
        }

    monkeypatch.setattr(client, "_ccxt_call", fetch)
    monkeypatch.setattr(client, "fetch_ticker", short_ticker)
    monkeypatch.setattr(client, "_fetch_authoritative_ticker_fallback", authoritative)

    result = await client.fetch_reference_prices("BTC/USDT", contract_spec=CONTRACT_SPEC)

    assert result["reference_prices_data_available"] is False
    assert "reference_prices_fallback" not in result
    assert fallback_calls == []
    assert result["mark_price"] == 0.0
    assert result["index_price"] == 0.0


@pytest.mark.asyncio
async def test_ticker_routes_capture_local_completion_not_exchange_ts(
    monkeypatch, receipt_clock,
) -> None:
    client = OKXRestClient()
    completed_at = receipt_clock.current + timedelta(seconds=3)

    async def fetch(*args, **kwargs):
        receipt_clock.current = completed_at
        return {"data": [{
            "instId": "BTC-USDT-SWAP",
            "last": "100",
            "ts": str(SOURCE_TIMESTAMP_MS),
        }]}

    monkeypatch.setattr(client, "_ccxt_call", fetch)
    single = await client.fetch_ticker("BTC/USDT")
    batch = await client.fetch_tickers(["BTC/USDT"])

    assert single["received_at"] == completed_at.isoformat()
    assert batch["BTC/USDT"]["received_at"] == completed_at.isoformat()
    assert single["timestamp"] == SOURCE_TIMESTAMP_MS
    assert batch["BTC/USDT"]["timestamp"] == SOURCE_TIMESTAMP_MS


@pytest.mark.asyncio
@pytest.mark.parametrize("channel,fact_key", [
    ("books5", "orderbook_fact"),
    ("mark-price", "mark_price_fact"),
])
async def test_ws_nested_facts_preserve_the_received_message_clock(
    channel, fact_key, receipt_clock,
) -> None:
    client = OKXWebSocketClient()
    received_at = receipt_clock.current.isoformat()
    row = {
        "bids": [["99.9", "2"]],
        "asks": [["100.1", "3"]],
        "markPx": "100",
        "ts": str(SOURCE_TIMESTAMP_MS),
    }

    await client._handle_message(json.dumps({
        "arg": {"channel": channel, "instId": "BTC-USDT-SWAP"},
        "data": [row],
    }))
    receipt_clock.current += timedelta(hours=1)
    cache = client.latest_orderbooks if channel == "books5" else client.latest_mark_prices
    payload = cache["BTC/USDT"]

    assert payload["received_at"] == received_at
    assert payload[fact_key]["received_at"] == received_at
    assert payload[fact_key]["source_timestamp_ms"] == SOURCE_TIMESTAMP_MS


@pytest.mark.asyncio
@pytest.mark.parametrize("source_ts", [str(SOURCE_TIMESTAMP_MS), None])
async def test_fee_receipt_and_source_observation_keep_separate_identities(
    source_ts, receipt_clock,
) -> None:
    completed_at = receipt_clock.current + timedelta(seconds=4)

    class FeeExchange:
        async def privateGetAccountFeeRates(self, params):
            assert params == {"instType": "SWAP"}
            receipt_clock.current = completed_at
            return {"data": [{"takerU": "-0.0005", "ts": source_ts}]}

    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    executor._exchange = FeeExchange()
    result = await executor.fetch_account_fee_snapshot()
    source_at = (
        datetime.fromtimestamp(SOURCE_TIMESTAMP_MS / 1000, UTC).isoformat()
        if source_ts is not None else None
    )

    assert result["account_fee_received_at"] == completed_at.isoformat()
    assert result["account_fee_source_observed_at"] == source_at
    assert result["fee_rate_observed_at"] == (source_at or completed_at.isoformat())
    assert result["policy_provenance"]["generated_at"] == completed_at.isoformat()
    assert result["policy_provenance"]["source_observed_at"] == source_at
