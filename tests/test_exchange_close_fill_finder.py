from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from services.exchange_close_fill_finder import ExchangeCloseFillFinder, order_fee_cost


class _FakeCcxt:
    def __init__(
        self,
        *,
        fills_history=None,
        instruments=None,
        fill_history_inst_id_error: Exception | None = None,
    ) -> None:
        self.fills_history = fills_history or []
        self.instruments = instruments if instruments is not None else []
        self.fill_history_inst_id_error = fill_history_inst_id_error
        self.fill_history_params: list[dict] = []
        self.instrument_params: list[dict] = []

    def market(self, _symbol):
        raise AssertionError("close-fill lookup must not depend on CCXT market metadata")

    async def fetch_closed_orders(self, *_args):
        raise AssertionError("close-fill lookup must not use CCXT fetch_closed_orders")

    async def fetch_my_trades(self, *_args):
        raise AssertionError("close-fill lookup must not use CCXT fetch_my_trades")

    async def privateGetTradeFillsHistory(self, params):
        self.fill_history_params.append(params)
        if self.fill_history_inst_id_error and params.get("instId"):
            raise self.fill_history_inst_id_error
        return {"data": self.fills_history}

    async def privateGetTradeFills(self, params):
        return {"data": self.fills_history}

    async def publicGetPublicInstruments(self, params):
        self.instrument_params.append(params)
        return {"data": self.instruments}


class _FakePaperOkx:
    def __init__(self, ccxt) -> None:
        self.ccxt = ccxt

    async def _get_ccxt(self):
        return self.ccxt

    def _to_swap_symbol(self, symbol):
        return f"{symbol}:SWAP"

    async def _with_retry(self, method, *args):
        return await method(*args)


def _position(**kwargs):
    defaults = {
        "symbol": "BTC/USDT",
        "side": "long",
        "quantity": 2.0,
        "created_at": datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _without_begin(params: list[dict]) -> list[dict]:
    return [{key: value for key, value in item.items() if key != "begin"} for item in params]


@pytest.mark.asyncio
async def test_exchange_close_fill_finder_uses_latest_native_okx_fill_candidate():
    timestamp = int(datetime(2026, 6, 8, 12, 10, tzinfo=UTC).timestamp() * 1000)
    ccxt = _FakeCcxt(
        instruments=[{"instId": "BTC-USDT-SWAP", "ctVal": "0.5", "ctMult": "1"}],
        fills_history=[
            {
                "instId": "BTC-USDT-SWAP",
                "ordId": "too-small",
                "side": "sell",
                "fillSz": "0.1",
                "fillPx": "111",
                "ts": str(timestamp),
            },
            {
                "instId": "BTC-USDT-SWAP",
                "ordId": "close-order-1",
                "side": "sell",
                "fillSz": "4",
                "fillPx": "112",
                "fee": "-0.25",
                "fillPnl": "24",
                "ts": str(timestamp + 1),
                "algoId": "algo-1",
            },
        ]
    )
    parser_calls = []
    finder = ExchangeCloseFillFinder(
        paper_okx_provider=lambda: _FakePaperOkx(ccxt),
        datetime_from_ms_parser=lambda value: parser_calls.append(value) or "parsed",
    )

    result = await finder.find(_position())

    assert result["order_id"] == "close-order-1"
    assert result["price"] == 112.0
    assert result["quantity"] == 2.0
    assert result["fee"] == 0.25
    assert result["pnl"] == 24.0
    assert result["source"] == "okx_fills_history"
    assert result["order_info"]["algoId"] == "algo-1"
    assert parser_calls == [timestamp + 1]


@pytest.mark.asyncio
async def test_exchange_close_fill_finder_prefers_native_quantity_match_over_latest_candidate():
    timestamp = int(datetime(2026, 6, 8, 12, 10, tzinfo=UTC).timestamp() * 1000)
    ccxt = _FakeCcxt(
        instruments=[{"instId": "BTC-USDT-SWAP", "ctVal": "1", "ctMult": "1"}],
        fills_history=[
            {
                "instId": "BTC-USDT-SWAP",
                "ordId": "reduced-ten",
                "side": "sell",
                "fillSz": "10",
                "fillPx": "3.85",
                "fillPnl": "15.4",
                "ts": str(timestamp),
            },
            {
                "instId": "BTC-USDT-SWAP",
                "ordId": "later-six",
                "side": "sell",
                "fillSz": "6",
                "fillPx": "4.26",
                "fillPnl": "11.7",
                "ts": str(timestamp + 60000),
            },
        ]
    )
    finder = ExchangeCloseFillFinder(paper_okx_provider=lambda: _FakePaperOkx(ccxt))

    result = await finder.find(_position(quantity=10.0, contract_size=1.0))

    assert result["order_id"] == "reduced-ten"
    assert result["price"] == 3.85
    assert result["quantity"] == 10.0


@pytest.mark.asyncio
async def test_exchange_close_fill_finder_uses_okx_ctval_for_contract_quantity_match():
    timestamp = int(datetime(2026, 6, 28, 14, 28, tzinfo=UTC).timestamp() * 1000)
    ccxt = _FakeCcxt(
        instruments=[
            {
                "instType": "SWAP",
                "instId": "FLOKI-USDT-SWAP",
                "ctVal": "100000",
                "settleCcy": "USDT",
                "state": "live",
            }
        ],
        fills_history=[
            {
                "instId": "FLOKI-USDT-SWAP",
                "ordId": "3696171269466329088",
                "side": "buy",
                "fillSz": "6",
                "fillPx": "0.00007123",
                "fee": "-0.0042",
                "fillPnl": "1.23",
                "ts": str(timestamp),
            }
        ],
    )
    finder = ExchangeCloseFillFinder(paper_okx_provider=lambda: _FakePaperOkx(ccxt))

    result = await finder.find(
        _position(
            symbol="FLOKI/USDT",
            okx_inst_id="FLOKI-USDT-SWAP",
            side="short",
            quantity=600000.0,
        )
    )

    assert ccxt.instrument_params == [{"instType": "SWAP"}]
    assert _without_begin(ccxt.fill_history_params) == [
        {"instType": "SWAP", "instId": "FLOKI-USDT-SWAP", "limit": "100"}
    ]
    assert result["order_id"] == "3696171269466329088"
    assert result["quantity"] == pytest.approx(600000.0)
    assert result["contracts"] == pytest.approx(6.0)
    assert result["contract_size"] == pytest.approx(100000.0)
    assert result["contract_size_source"] == "okx_public_instruments"


@pytest.mark.asyncio
async def test_exchange_close_fill_finder_ignores_ccxt_abstract_history_without_native_fills():
    ccxt = _FakeCcxt(
        instruments=[{"instId": "BTC-USDT-SWAP", "ctVal": "1", "ctMult": "1"}]
    )
    finder = ExchangeCloseFillFinder(paper_okx_provider=lambda: _FakePaperOkx(ccxt))

    result = await finder.find(_position(side="short", quantity=2.0))

    assert result == {}
    assert _without_begin(ccxt.fill_history_params) == [
        {"instType": "SWAP", "instId": "BTC-USDT-SWAP", "limit": "100"}
    ]


@pytest.mark.asyncio
async def test_exchange_close_fill_finder_reads_native_okx_fills_history_when_market_missing():
    timestamp = int(datetime(2026, 6, 26, 3, 20, tzinfo=UTC).timestamp() * 1000)
    ccxt = _FakeCcxt(
        instruments=[{"instId": "LAB-USDT-SWAP", "ctVal": "0.1", "ctMult": "1"}],
        fills_history=[
            {
                "instId": "LAB-USDT-SWAP",
                "ordId": "close-lab",
                "side": "sell",
                "fillSz": "4",
                "fillPx": "17.43",
                "fee": "-0.006972",
                "ts": str(timestamp),
            },
            {
                "instId": "LAB-USDT-SWAP",
                "ordId": "close-lab",
                "side": "sell",
                "fillSz": "5",
                "fillPx": "17.448",
                "fee": "-0.008724",
                "ts": str(timestamp + 1),
            },
            {
                "instId": "LAB-USDT-SWAP",
                "ordId": "entry-lab",
                "side": "buy",
                "fillSz": "9",
                "fillPx": "16.86",
                "ts": str(timestamp - 1000),
            },
        ],
    )
    finder = ExchangeCloseFillFinder(paper_okx_provider=lambda: _FakePaperOkx(ccxt))

    result = await finder.find(_position(symbol="LAB/USDT", side="long", quantity=0.9))

    assert _without_begin(ccxt.fill_history_params) == [
        {"instType": "SWAP", "instId": "LAB-USDT-SWAP", "limit": "100"}
    ]
    assert result["source"] == "okx_fills_history"
    assert result["order_id"] == "close-lab"
    assert result["price"] == pytest.approx(((4 * 17.43) + (5 * 17.448)) / 9)
    assert result["quantity"] == pytest.approx(0.9)
    assert result["contracts"] == pytest.approx(9.0)
    assert result["contract_size"] == pytest.approx(0.1)
    assert result["fee"] == pytest.approx(0.015696)


@pytest.mark.asyncio
async def test_exchange_close_fill_finder_retries_account_wide_history_for_offline_instrument():
    timestamp = int(datetime(2026, 6, 26, 3, 20, tzinfo=UTC).timestamp() * 1000)
    ccxt = _FakeCcxt(
        fill_history_inst_id_error=RuntimeError("51001 instrument does not exist"),
        fills_history=[
            {
                "instId": "LAB-USDT-SWAP",
                "ordId": "close-lab",
                "side": "sell",
                "fillSz": "9",
                "fillPx": "17.44",
                "fillPnl": "0.517",
                "ts": str(timestamp),
            },
            {
                "instId": "OTHER-USDT-SWAP",
                "ordId": "other",
                "side": "sell",
                "fillSz": "9",
                "fillPx": "99",
                "ts": str(timestamp),
            },
        ],
    )
    finder = ExchangeCloseFillFinder(paper_okx_provider=lambda: _FakePaperOkx(ccxt))

    result = await finder.find(_position(symbol="LAB/USDT", side="long", quantity=0.9))

    assert result == {}
    assert ccxt.fill_history_params == []


@pytest.mark.asyncio
async def test_exchange_close_fill_finder_returns_empty_when_okx_unavailable():
    assert await ExchangeCloseFillFinder(paper_okx_provider=lambda: None).find(_position()) == {}


@pytest.mark.asyncio
async def test_exchange_close_fill_finder_propagates_native_fetch_failure():
    class FailingPaperOkx(_FakePaperOkx):
        async def _with_retry(self, _method, *_args):
            raise RuntimeError("fetch failed")

    finder = ExchangeCloseFillFinder(
        paper_okx_provider=lambda: FailingPaperOkx(_FakeCcxt()),
    )

    with pytest.raises(RuntimeError, match="fetch failed"):
        await finder.find(_position())


def test_order_fee_cost_reads_fee_shapes():
    assert order_fee_cost({"fee": {"cost": "0.3"}}) == 0.3
    assert order_fee_cost({"info": {"fee": "-0.2"}}) == 0.2
    assert order_fee_cost({"info": {"fee": "bad"}}) == 0.0
