from __future__ import annotations

import pytest

from core.okx_instrument_filter import (
    is_supported_usdt_swap_instrument,
    supported_usdt_swap_instruments,
)
from executor.okx_executor import OKXExecutor


def _instrument(
    inst_id: str,
    *,
    open_type: str = "fix_price",
    inst_category: str = "1",
    uly: str | None = None,
) -> dict[str, str]:
    base = inst_id.removesuffix("-USDT-SWAP")
    return {
        "instType": "SWAP",
        "state": "live",
        "ctType": "linear",
        "settleCcy": "USDT",
        "instId": inst_id,
        "ctVal": "1",
        "minSz": "0.01",
        "tickSz": "0.0001",
        "openType": open_type,
        "instCategory": inst_category,
        "uly": uly or f"{base}-USDT",
    }


def test_okx_instrument_filter_keeps_only_crypto_usdt_swaps() -> None:
    assert is_supported_usdt_swap_instrument(_instrument("BTC-USDT-SWAP")) is True
    assert (
        is_supported_usdt_swap_instrument(
            _instrument("OPN-USDT-SWAP", open_type="pre_quote", inst_category="1")
        )
        is True
    )
    assert (
        is_supported_usdt_swap_instrument(
            _instrument("PLTR-USDT-SWAP", open_type="fix_price", inst_category="3")
        )
        is False
    )
    assert (
        is_supported_usdt_swap_instrument(
            _instrument("XAU-USDT-SWAP", open_type="fix_price", inst_category="4")
        )
        is False
    )
    missing_category = _instrument("MISSING-CATEGORY-USDT-SWAP")
    missing_category.pop("instCategory")
    assert is_supported_usdt_swap_instrument(missing_category) is False


def test_supported_instruments_dedupe_like_okx_executor_market_keys() -> None:
    instruments = [
        _instrument("LINEA-USDT-SWAP", uly="LINEA-USDT"),
        _instrument("SKY-USDT-SWAP", uly="LINEA-USDT"),
        _instrument("BTC-USDT-SWAP", uly="BTC-USDT"),
    ]

    supported = supported_usdt_swap_instruments(instruments)

    assert [item["instId"] for item in supported] == [
        "SKY-USDT-SWAP",
        "BTC-USDT-SWAP",
    ]


class _InstrumentFilteringCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.markets: dict[str, dict] = {}
        self.markets_by_id: dict[str, dict] = {}

    async def publicGetPublicInstruments(self, _params: dict) -> dict:
        return {
                "data": [
                    _instrument("BTC-USDT-SWAP"),
                    _instrument("PLTR-USDT-SWAP", open_type="fix_price", inst_category="3"),
                ]
            }

    def parse_markets(self, rows: list[dict]) -> dict[str, dict]:
        markets: dict[str, dict] = {}
        for row in rows:
            base = str(row["instId"]).removesuffix("-USDT-SWAP")
            symbol = f"{base}/USDT:USDT"
            markets[symbol] = {"symbol": symbol, "id": row["instId"], "info": row}
        return markets

    def set_markets(self, markets: dict[str, dict]) -> None:
        self.markets = dict(markets)
        self.markets_by_id = {market["id"]: market for market in markets.values()}


@pytest.mark.asyncio
async def test_okx_executor_market_loader_filters_unsupported_pre_quote_contracts() -> None:
    fake_ccxt = _InstrumentFilteringCcxt()
    executor = OKXExecutor(mode="paper")
    executor._connected = True
    executor._exchange = fake_ccxt

    await executor._load_usdt_swap_markets()

    assert "BTC/USDT:USDT" in fake_ccxt.markets
    assert "PLTR/USDT:USDT" not in fake_ccxt.markets
