from __future__ import annotations

import pytest

from ai_brain.base_model import Action, DecisionOutput
from core.exceptions import ExchangeAPIError
from executor.base_executor import OrderStatus
from executor.okx_executor import OKXExecutor


class _MissingLabMarketExchange:
    def __init__(self) -> None:
        self.markets = {"BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT"}}
        self.markets_by_id = {}
        self.fetch_position_calls: list[list[str] | None] = []
        self.contracts = 9.0
        self.native_order_requests: list[dict] = []
        self.create_order_calls: list[tuple] = []

    def market(self, symbol: str) -> dict:
        raise Exception(f"okx does not have market symbol {symbol}")

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        raise Exception(f"okx does not have market symbol {symbol}")

    async def fetch_ticker(self, symbol: str):
        raise Exception(f"okx does not have market symbol {symbol}")

    async def fetch_open_orders(self, symbol=None, *args, **kwargs):
        return []

    async def create_order(self, *args, **kwargs):
        self.create_order_calls.append((args, kwargs))
        raise AssertionError("CCXT create_order should not be used for native LAB exits")

    async def privatePostTradeOrder(self, params):
        self.native_order_requests.append(dict(params))
        size = float(params["sz"])
        self.contracts = max(self.contracts - size, 0.0)
        return {"code": "0", "data": [{"ordId": "LAB_NATIVE_REDUCE_1", "sCode": "0"}]}

    async def fetch_positions(self, symbols=None):
        self.fetch_position_calls.append(symbols)
        if symbols:
            raise Exception(f"okx does not have market symbol {symbols[0]}")
        return [
            {
                "symbol": "LAB-USDT-SWAP",
                "side": "long",
                "contracts": self.contracts,
                "markPrice": 17.787,
                "entryPrice": 16.865555555555556,
                "unrealizedPnl": 0.8292999999999989,
                "info": {
                    "instId": "LAB-USDT-SWAP",
                    "pos": str(self.contracts),
                    "avgPx": "16.8655555555555556",
                    "markPx": "17.787",
                    "upl": "0.8292999999999989",
                    "posSide": "net",
                },
            }
        ]

    async def publicGetPublicInstruments(self, params):
        return {
            "data": [
                {
                    "instType": "SWAP",
                    "state": "live",
                    "ctType": "linear",
                    "settleCcy": "USDT",
                    "instId": "BTC-USDT-SWAP",
                    "ctVal": "0.01",
                    "minSz": "1",
                    "tickSz": "0.1",
                }
            ]
        }

    def parse_markets(self, instruments):
        return [{"symbol": "BTC/USDT:USDT", "id": "BTC-USDT-SWAP"}]

    def set_markets(self, markets):
        self.markets = {market["symbol"]: market for market in markets}


class _ContractDeliveryLabExchange(_MissingLabMarketExchange):
    async def privatePostTradeOrder(self, params):
        self.native_order_requests.append(dict(params))
        raise ExchangeAPIError(
            'okx {"code":"1","data":[{"ordId":"","sCode":"51028",'
            '"sMsg":"Contract under delivery."}],"msg":"All operations failed"}'
        )


@pytest.mark.asyncio
async def test_get_positions_strict_falls_back_to_account_wide_when_market_missing() -> None:
    executor = OKXExecutor("paper")
    exchange = _MissingLabMarketExchange()
    executor._exchange = exchange
    executor._connected = True
    executor._markets_loaded = True

    positions = await executor.get_positions_strict("LAB/USDT")

    assert len(positions) == 1
    assert positions[0]["symbol"] == "LAB-USDT-SWAP"
    assert exchange.fetch_position_calls[0] == ["LAB/USDT:USDT"]
    assert exchange.fetch_position_calls[1] is None


@pytest.mark.asyncio
async def test_market_for_symbol_builds_synthetic_exit_market_from_position() -> None:
    executor = OKXExecutor("paper")
    exchange = _MissingLabMarketExchange()
    executor._exchange = exchange
    executor._connected = True
    executor._markets_loaded = True

    market = await executor._market_for_symbol("LAB-USDT-SWAP", app_symbol="LAB/USDT")

    assert market["id"] == "LAB-USDT-SWAP"
    assert market["symbol"] == "LAB-USDT-SWAP"
    assert market["contractSize"] == pytest.approx(0.1)
    assert market["limits"]["amount"]["min"] == pytest.approx(1.0)
    assert market["synthetic_from_position"] is True


@pytest.mark.asyncio
async def test_place_order_uses_native_reduce_order_for_position_only_market() -> None:
    executor = OKXExecutor("paper")
    exchange = _MissingLabMarketExchange()
    executor._exchange = exchange
    executor._connected = True
    executor._markets_loaded = True

    decision = DecisionOutput(
        model_name="position_review",
        symbol="LAB/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.95,
        reasoning="reduce half of the LAB position",
        position_size_pct=0.5,
    )

    result = await executor.place_order(decision)

    assert result.status == OrderStatus.FILLED
    assert result.exchange_order_id == "LAB_NATIVE_REDUCE_1"
    assert result.quantity == pytest.approx(0.45)
    assert exchange.contracts == pytest.approx(4.5)
    assert exchange.create_order_calls == []
    assert exchange.native_order_requests == [
        {
            "instId": "LAB-USDT-SWAP",
            "tdMode": "cross",
            "side": "sell",
            "ordType": "market",
            "sz": "4.5",
            "reduceOnly": "true",
        }
    ]
    assert result.raw_response["okx_native_reduce_market_order"] is True
    assert result.raw_response["canonical_exchange_symbol"] == "LAB/USDT"


@pytest.mark.asyncio
async def test_native_reduce_contract_delivery_error_pauses_repeated_submit() -> None:
    executor = OKXExecutor("paper")
    exchange = _ContractDeliveryLabExchange()
    executor._exchange = exchange
    executor._connected = True
    executor._markets_loaded = True

    decision = DecisionOutput(
        model_name="position_review",
        symbol="LAB/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.95,
        reasoning="close LAB while OKX is settling the contract",
        position_size_pct=1.0,
    )

    first = await executor.place_order(decision)
    second = await executor.place_order(decision)

    assert first.status == OrderStatus.REJECTED
    assert first.raw_response["okx_contract_delivery_cooldown"] is True
    assert first.raw_response["okx_contract_delivery_lock_hit"] is False
    assert second.status == OrderStatus.REJECTED
    assert second.raw_response["okx_contract_delivery_cooldown"] is True
    assert second.raw_response["okx_contract_delivery_lock_hit"] is True
    assert second.raw_response["do_not_persist_order"] is True
    assert len(exchange.native_order_requests) == 1
