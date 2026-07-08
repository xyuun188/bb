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
    ct_val: str = "1",
    min_sz: str = "0.01",
    lot_sz: str = "0.01",
) -> dict[str, str]:
    base = inst_id.removesuffix("-USDT-SWAP")
    return {
        "instType": "SWAP",
        "state": "live",
        "ctType": "linear",
        "settleCcy": "USDT",
        "instId": inst_id,
        "ctVal": ct_val,
        "minSz": min_sz,
        "lotSz": lot_sz,
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
            markets[symbol] = {
                "symbol": symbol,
                "id": row["instId"],
                "contractSize": 1.0,
                "info": {"instId": row["instId"]},
            }
        return markets

    def set_markets(self, markets: dict[str, dict]) -> None:
        self.markets = dict(markets)
        self.markets_by_id = {market["id"]: market for market in markets.values()}

    def amount_to_precision(self, _symbol: str, amount: float) -> str:
        return str(amount)


@pytest.mark.asyncio
async def test_okx_executor_market_loader_filters_unsupported_pre_quote_contracts() -> None:
    fake_ccxt = _InstrumentFilteringCcxt()
    executor = OKXExecutor(mode="paper")
    executor._connected = True
    executor._exchange = fake_ccxt

    await executor._load_usdt_swap_markets()

    assert "BTC/USDT:USDT" in fake_ccxt.markets
    assert "PLTR/USDT:USDT" not in fake_ccxt.markets


@pytest.mark.asyncio
async def test_okx_executor_market_loader_preserves_subunit_contract_size() -> None:
    class _SmallContractCcxt(_InstrumentFilteringCcxt):
        async def publicGetPublicInstruments(self, _params: dict) -> dict:
            return {
                "data": [
                    _instrument(
                        "BTC-USDT-SWAP",
                        ct_val="0.01",
                        min_sz="0.01",
                        lot_sz="0.01",
                    )
                ]
            }

    fake_ccxt = _SmallContractCcxt()
    executor = OKXExecutor(mode="paper")
    executor._connected = True
    executor._exchange = fake_ccxt

    await executor._load_usdt_swap_markets()

    market = fake_ccxt.markets["BTC/USDT:USDT"]
    assert market["contractSize"] == pytest.approx(0.01)
    assert market["info"]["ctVal"] == "0.01"
    assert executor._contract_size(market) == pytest.approx(0.01)

    contracts, base_quantity = executor._entry_order_amount(
        fake_ccxt,
        market,
        position_value=43.4,
        price=62000.0,
        balance=100.0,
        leverage=1.0,
    )

    assert contracts == pytest.approx(0.07)
    assert base_quantity == pytest.approx(0.0007)
