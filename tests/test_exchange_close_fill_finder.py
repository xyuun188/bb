from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from services.exchange_close_fill_finder import ExchangeCloseFillFinder, order_fee_cost


class _FakeCcxt:
    def __init__(self, *, closed_orders=None, trades=None, contract_size=0.5) -> None:
        self.closed_orders = closed_orders or []
        self.trades = trades or []
        self.contract_size = contract_size

    def market(self, _symbol):
        return {"contractSize": self.contract_size}

    async def fetch_closed_orders(self, *_args):
        return self.closed_orders

    async def fetch_my_trades(self, *_args):
        return self.trades


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


@pytest.mark.asyncio
async def test_exchange_close_fill_finder_uses_latest_closed_order_candidate():
    timestamp = int(datetime(2026, 6, 8, 12, 10, tzinfo=UTC).timestamp() * 1000)
    ccxt = _FakeCcxt(
        closed_orders=[
            {
                "id": "too-small",
                "side": "sell",
                "filled": "0.1",
                "average": "111",
                "reduceOnly": True,
                "timestamp": timestamp,
            },
            {
                "id": "close-order-1",
                "side": "sell",
                "filled": "4",
                "average": "112",
                "reduceOnly": True,
                "timestamp": timestamp + 1,
                "fee": {"cost": "0.25"},
                "info": {"pnl": "24", "ordType": "trigger", "algoId": "algo-1"},
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
    assert result["source"] == "closed_orders"
    assert result["order_type"] == "trigger"
    assert result["algo_id"] == "algo-1"
    assert result["order_info"]["algoId"] == "algo-1"
    assert parser_calls == [timestamp + 1]


@pytest.mark.asyncio
async def test_exchange_close_fill_finder_prefers_quantity_match_over_latest_candidate():
    timestamp = int(datetime(2026, 6, 8, 12, 10, tzinfo=UTC).timestamp() * 1000)
    ccxt = _FakeCcxt(
        closed_orders=[
            {
                "id": "reduced-ten",
                "side": "sell",
                "filled": "10",
                "average": "3.85",
                "reduceOnly": True,
                "timestamp": timestamp,
                "info": {"pnl": "15.4"},
            },
            {
                "id": "later-six",
                "side": "sell",
                "filled": "6",
                "average": "4.26",
                "reduceOnly": True,
                "timestamp": timestamp + 60000,
                "info": {"pnl": "11.7"},
            },
        ],
        contract_size=1.0,
    )
    finder = ExchangeCloseFillFinder(paper_okx_provider=lambda: _FakePaperOkx(ccxt))

    result = await finder.find(_position(quantity=10.0))

    assert result["order_id"] == "reduced-ten"
    assert result["price"] == 3.85
    assert result["quantity"] == 10.0


@pytest.mark.asyncio
async def test_exchange_close_fill_finder_groups_my_trades_when_orders_missing():
    timestamp = int(datetime(2026, 6, 8, 12, 10, tzinfo=UTC).timestamp() * 1000)
    ccxt = _FakeCcxt(
        trades=[
            {
                "id": "trade-1",
                "order": "order-2",
                "side": "buy",
                "amount": "2",
                "price": "90",
                "timestamp": timestamp,
                "fee": {"cost": "0.1"},
                "info": {"fillPnl": "3"},
            },
            {
                "id": "trade-2",
                "order": "order-2",
                "side": "buy",
                "amount": "2",
                "price": "92",
                "timestamp": timestamp + 2,
                "fee": {"cost": "0.2"},
                "info": {"fillPnl": "5"},
            },
        ]
    )
    finder = ExchangeCloseFillFinder(paper_okx_provider=lambda: _FakePaperOkx(ccxt))

    result = await finder.find(_position(side="short", quantity=2.0))

    assert result["order_id"] == "order-2"
    assert result["price"] == 91.0
    assert result["quantity"] == 2.0
    assert result["fee"] == 0.30000000000000004
    assert result["pnl"] == 8.0
    assert result["source"] == "my_trades"


@pytest.mark.asyncio
async def test_exchange_close_fill_finder_returns_empty_when_okx_unavailable_or_fetch_fails():
    assert await ExchangeCloseFillFinder(paper_okx_provider=lambda: None).find(_position()) == {}

    class FailingPaperOkx(_FakePaperOkx):
        async def _with_retry(self, _method, *_args):
            raise RuntimeError("fetch failed")

    finder = ExchangeCloseFillFinder(
        paper_okx_provider=lambda: FailingPaperOkx(_FakeCcxt()),
    )

    assert await finder.find(_position()) == {}


def test_order_fee_cost_reads_fee_shapes():
    assert order_fee_cost({"fee": {"cost": "0.3"}}) == 0.3
    assert order_fee_cost({"info": {"fee": "-0.2"}}) == 0.2
    assert order_fee_cost({"info": {"fee": "bad"}}) == 0.0
