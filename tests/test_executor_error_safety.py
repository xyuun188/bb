from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

import executor.okx_executor as okx_module
from ai_brain.base_model import Action, DecisionOutput
from core.exceptions import ExchangeAPIError, OrderPlacementError
from executor.base_executor import OrderStatus
from executor.okx_executor import OKXExecutor


class _FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def warning(self, message: str, **kwargs: Any) -> None:
        self.events.append(("warning", message, kwargs))


class _FailingBalanceCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text

    async def fetch_balance(self) -> dict[str, Any]:
        raise RuntimeError(self.error_text)


class _BalanceOnlyCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.instrument_calls = 0

    async def publicGetPublicInstruments(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.instrument_calls += 1
        raise AssertionError("balance-only snapshot must not load OKX instruments")

    async def fetch_balance(self) -> dict[str, Any]:
        return {
            "USDT": {"free": 12.0, "used": 3.0, "total": 15.0},
            "info": {"data": [{"details": [{"ccy": "USDT", "cashBal": "15", "eq": "16"}]}]},
        }


class _AliasMismatchMarketCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"
    markets = {"WLFI/USDT:USDT": {"symbol": "WLFI/USDT:USDT"}}
    markets_by_id: dict[str, Any] = {}

    def market(self, symbol: str) -> dict[str, Any]:
        if symbol != "WLFI/USDT:USDT":
            raise RuntimeError("bad symbol")
        return {
            "symbol": "WLFI/USDT:USDT",
            "id": "H-USDT-SWAP",
            "info": {"instId": "H-USDT-SWAP"},
        }


class _CcxtBalanceWouldLoadMarkets:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"
    markets = None

    def __init__(self) -> None:
        self.instrument_calls = 0
        self.markets_seen_by_fetch: Any = None

    async def publicGetPublicInstruments(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.instrument_calls += 1
        raise AssertionError("balance snapshot must not load OKX instruments")

    async def fetch_balance(self) -> dict[str, Any]:
        self.markets_seen_by_fetch = self.markets
        if self.markets is None:
            await self.publicGetPublicInstruments({"instType": "SWAP"})
        return {
            "USDT": {"free": 7.0, "used": 1.0, "total": 8.0},
            "info": {"data": [{"details": [{"ccy": "USDT", "cashBal": "8", "eq": "8"}]}]},
        }


class _NativeBalanceOnlyCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.instrument_calls = 0
        self.balance_calls = 0

    async def publicGetPublicInstruments(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.instrument_calls += 1
        raise AssertionError("native balance snapshot must not load OKX instruments")

    async def privateGetAccountBalance(self, params: dict[str, Any]) -> dict[str, Any]:
        self.balance_calls += 1
        assert params == {"ccy": "USDT"}
        return {
            "data": [
                {
                    "details": [
                        {
                            "ccy": "USDT",
                            "cashBal": "15",
                            "eq": "16",
                            "availBal": "12",
                            "frozenBal": "3",
                        }
                    ]
                }
            ]
        }


def _native_position_row(
    inst_id: str,
    *,
    pos: Any,
    pos_side: str = "long",
    ct_val: Any = "1",
    avg_px: Any = "100",
    mark_px: Any = "100",
    upl: Any = "0",
) -> dict[str, Any]:
    return {
        "instId": inst_id,
        "posSide": pos_side,
        "pos": str(pos),
        "ctVal": str(ct_val),
        "avgPx": str(avg_px),
        "markPx": str(mark_px),
        "upl": str(upl),
    }


def _filter_native_rows(rows: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    inst_id = str(params.get("instId") or "").strip().upper()
    return {
        "data": [
            row
            for row in rows
            if not inst_id or str(row.get("instId") or "").strip().upper() == inst_id
        ]
    }


async def _native_ticker(
    params: dict[str, Any],
    *,
    last: Any,
    bid: Any | None = None,
    ask: Any | None = None,
) -> dict[str, Any]:
    return {
        "data": [
            {
                "instId": str(params.get("instId") or "").strip().upper(),
                "last": str(last),
                "bidPx": str(bid if bid is not None else last),
                "askPx": str(ask if ask is not None else last),
                "ts": "1780000000000",
            }
        ]
    }


def _native_order_detail_response(
    order: dict[str, Any],
    params: dict[str, Any],
    *,
    default_inst_id: str,
) -> dict[str, Any]:
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    fee = order.get("fee") if isinstance(order.get("fee"), dict) else {}
    return {
        "data": [
            {
                "instId": str(info.get("instId") or default_inst_id).strip().upper(),
                "ordId": order.get("id") or info.get("ordId") or params.get("ordId"),
                "side": order.get("side") or info.get("side") or "buy",
                "ordType": order.get("type") or info.get("ordType") or "market",
                "state": info.get("state") or order.get("status") or "live",
                "sz": str(order.get("amount") or info.get("sz") or "0"),
                "accFillSz": str(order.get("filled") or info.get("accFillSz") or "0"),
                "avgPx": str(order.get("average") or info.get("avgPx") or "0"),
                "px": str(order.get("price") or info.get("px") or "0"),
                "fee": str(fee.get("cost") or "0"),
            }
        ]
    }


class _FailingCancelCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text

    async def privatePostTradeCancelOrder(self, _params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(self.error_text)

    async def cancel_order(self, _order_id: str, _symbol: str) -> dict[str, Any]:
        raise AssertionError("order cancellation must use OKX native privatePostTradeCancelOrder")


class _FailingOpenOrdersCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text

    async def privateGetTradeOrdersPending(self, _params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(self.error_text)


class _ReloadableMarketCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.markets: dict[str, dict[str, Any]] = {}
        self.reload_calls = 0

    async def publicGetPublicInstruments(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.reload_calls += 1
        return {
            "data": [
                {
                    "instType": "SWAP",
                    "state": "live",
                    "ctType": "linear",
                    "settleCcy": "USDT",
                    "instId": "USAR-USDT-SWAP",
                    "ctVal": "1",
                    "ctValCcy": "USAR",
                    "minSz": "1",
                    "lotSz": "1",
                    "tickSz": "0.01",
                    "uly": "USAR-USDT",
                }
            ]
        }

    def parse_markets(self, _items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "USAR/USDT:USDT",
                "id": "USAR-USDT-SWAP",
                "contractSize": 1.0,
                "limits": {"amount": {"min": 1.0}},
                "info": {"instId": "USAR-USDT-SWAP"},
            }
        ]

    def set_markets(self, markets: list[dict[str, Any]]) -> None:
        self.markets = {market["symbol"]: market for market in markets}

    def market(self, symbol: str) -> dict[str, Any]:
        if symbol not in self.markets:
            raise RuntimeError(f"okx does not have market symbol {symbol}")
        return self.markets[symbol]


class _FailingPositionsForExitCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
        }

    async def publicGetMarketTicker(self, params: dict[str, Any]) -> dict[str, Any]:
        return await _native_ticker(params, last=100.0)

    async def fetch_ticker(self, _symbol: str) -> dict[str, Any]:
        raise AssertionError("execution sizing must use OKX native ticker API")

    async def privateGetAccountPositions(self, _params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(self.error_text)


class _FailingPositionsAfterExitSubmitCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text
        self.position_calls = 0

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
        }

    def amount_to_precision(self, _symbol: str, amount: float) -> str:
        return str(float(amount))

    async def publicGetMarketTicker(self, params: dict[str, Any]) -> dict[str, Any]:
        return await _native_ticker(params, last=100.0)

    async def fetch_ticker(self, _symbol: str) -> dict[str, Any]:
        raise AssertionError("execution sizing must use OKX native ticker API")

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        self.position_calls += 1
        if self.position_calls == 1:
            return _filter_native_rows(
                [
                    _native_position_row(
                        "BTC-USDT-SWAP",
                        pos="2",
                        pos_side="long",
                        ct_val="1",
                        avg_px="100",
                        mark_px="100",
                    )
                ],
                params,
            )
        raise RuntimeError(self.error_text)

    async def privateGetTradeOrdersPending(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "id": "exit-1",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": quantity,
            "filled": 0.0,
            "price": price or 100.0,
            "average": 100.0,
            "status": "open",
            "info": {"state": "live", "ordId": "exit-1", "side": side},
        }

    async def privateGetTradeOrder(self, params: dict[str, Any]) -> dict[str, Any]:
        return _native_order_detail_response(
            {
                "id": "exit-1",
                "symbol": "BTC-USDT-SWAP",
                "side": "sell",
                "type": "market",
                "status": "live",
                "amount": "2",
                "filled": "0",
                "average": "100",
                "info": {"state": "live", "ordId": "exit-1", "instId": "BTC-USDT-SWAP"},
            },
            params,
            default_inst_id="BTC-USDT-SWAP",
        )

    async def fetch_order(self, _order_id: str, _symbol: str) -> dict[str, Any]:
        raise AssertionError("order confirmation must use OKX native privateGetTradeOrder")


class _LeverageUnknownAfterOpenOrderLimitCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
            "info": {"instId": "BTC-USDT-SWAP"},
        }

    async def fetch_market_leverage_tiers(self, _symbol: str) -> list[dict[str, Any]]:
        return [{"maxLeverage": 20}]

    async def privateGetAccountAdjustLeverageInfo(
        self,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        return {"data": [{"maxLever": "20"}]}

    async def fetch_leverage(
        self,
        _symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        raise RuntimeError(self.error_text)

    async def set_leverage(
        self,
        _leverage: int,
        _symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        raise RuntimeError(f"OKX 59670 open order limit: {self.error_text}")

    async def privateGetTradeOrdersPending(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def privateGetAccountPositions(self, _params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(self.error_text)


class _PrecisionEntryCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.create_calls: list[tuple[Any, ...]] = []

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
            "info": {"instId": "SHIB-USDT-SWAP"},
        }

    def amount_to_precision(self, _symbol: str, amount: float) -> str:
        return str(float(amount))

    def price_to_precision(self, _symbol: str, price: float) -> str:
        return f"{float(price):.9f}"

    async def publicGetMarketTicker(self, params: dict[str, Any]) -> dict[str, Any]:
        return await _native_ticker(
            params,
            last=0.000008789,
            bid=0.000008788,
            ask=0.00000879,
        )

    async def fetch_ticker(self, _symbol: str) -> dict[str, Any]:
        raise AssertionError("execution sizing must use OKX native ticker API")

    async def fetch_open_orders(self, _symbol: str | None = None) -> list[dict[str, Any]]:
        return []

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.create_calls.append((symbol, order_type, side, quantity, price, params))
        return {
            "id": "entry-shib",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": quantity,
            "filled": quantity,
            "price": price or 0.000008789,
            "average": 0.000008789,
            "status": "closed",
            "info": {"state": "filled", "ordId": "entry-shib", "side": side},
        }

    async def privateGetTradeOrder(self, params: dict[str, Any]) -> dict[str, Any]:
        return _native_order_detail_response(
            {
                "id": "entry-shib",
                "symbol": "SHIB-USDT-SWAP",
                "side": "buy",
                "type": "market",
                "status": "filled",
                "amount": "1",
                "filled": "1",
                "average": "0.000008789",
                "info": {
                    "state": "filled",
                    "ordId": "entry-shib",
                    "instId": "SHIB-USDT-SWAP",
                },
            },
            params,
            default_inst_id="SHIB-USDT-SWAP",
        )

    async def fetch_order(self, _order_id: str, _symbol: str) -> dict[str, Any]:
        raise AssertionError("order confirmation must use OKX native privateGetTradeOrder")


class _EntryMaxMarketSizeCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.create_calls: list[tuple[Any, ...]] = []
        self.orders: dict[str, dict[str, Any]] = {}

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
            "info": {"instId": "BTC-USDT-SWAP", "maxMktSz": "100", "lotSz": "1"},
        }

    def amount_to_precision(self, _symbol: str, amount: float) -> str:
        return str(float(amount))

    def price_to_precision(self, _symbol: str, price: float) -> str:
        return str(float(price))

    async def publicGetMarketTicker(self, params: dict[str, Any]) -> dict[str, Any]:
        return await _native_ticker(params, last=1.0, bid=0.999, ask=1.001)

    async def fetch_ticker(self, _symbol: str) -> dict[str, Any]:
        raise AssertionError("execution sizing must use OKX native ticker API")

    async def privateGetTradeOrdersPending(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def fetch_market_leverage_tiers(self, _symbol: str) -> list[dict[str, Any]]:
        return [{"maxLeverage": 20}]

    async def privateGetAccountAdjustLeverageInfo(
        self,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        return {"data": [{"maxLever": "20"}]}

    async def fetch_leverage(
        self,
        _symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        return {"longLeverage": 5, "shortLeverage": 5}

    async def set_leverage(
        self,
        leverage: int,
        _symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        return {"info": {"lever": str(leverage)}}

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.create_calls.append((symbol, order_type, side, quantity, price, params))
        order = {
            "id": "entry-max-market",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": quantity,
            "filled": quantity,
            "price": price or 1.0,
            "average": 1.0,
            "status": "closed",
            "info": {"state": "filled", "ordId": "entry-max-market", "side": side},
        }
        self.orders["entry-max-market"] = order
        return order

    async def privateGetTradeOrder(self, params: dict[str, Any]) -> dict[str, Any]:
        return _native_order_detail_response(
            self.orders[str(params.get("ordId"))],
            params,
            default_inst_id="BTC-USDT-SWAP",
        )

    async def fetch_order(self, order_id: str, _symbol: str) -> dict[str, Any]:
        raise AssertionError("order confirmation must use OKX native privateGetTradeOrder")


class _ExitMaxMarketSizeCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(
        self,
        position_contracts: float = 100.0,
        *,
        native_close_error: bool = False,
    ) -> None:
        self.position_contracts = position_contracts
        self.native_close_error = native_close_error
        self.create_calls: list[tuple[Any, ...]] = []
        self.close_position_calls: list[dict[str, Any]] = []
        self.orders: dict[str, dict[str, Any]] = {}

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
            "info": {"instId": "USAR-USDT-SWAP", "maxMktSz": "10", "lotSz": "1"},
        }

    def amount_to_precision(self, _symbol: str, amount: float) -> str:
        return str(float(amount))

    async def publicGetMarketTicker(self, params: dict[str, Any]) -> dict[str, Any]:
        return await _native_ticker(params, last=3.0, bid=3.0, ask=3.01)

    async def fetch_ticker(self, _symbol: str) -> dict[str, Any]:
        raise AssertionError("execution sizing must use OKX native ticker API")

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        return _filter_native_rows(
            [
                _native_position_row(
                    "USAR-USDT-SWAP",
                    pos=self.position_contracts,
                    pos_side="long",
                    ct_val="1",
                    avg_px="2.31",
                    mark_px="3.0",
                    upl="69",
                )
            ],
            params,
        )

    async def privateGetTradeOrdersPending(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if quantity > 10:
            raise AssertionError("exit market order must be split below maxMktSz")
        self.create_calls.append((symbol, order_type, side, quantity, price, dict(params)))
        self.position_contracts = max(self.position_contracts - quantity, 0.0)
        order_id = f"exit-{len(self.create_calls)}"
        order = {
            "id": order_id,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": quantity,
            "filled": quantity,
            "price": price or 3.0,
            "average": 3.0,
            "status": "closed",
            "fee": {"cost": quantity * 0.001},
            "info": {
                "state": "filled",
                "ordId": order_id,
                "side": side,
                "reduceOnly": "true",
            },
        }
        self.orders[order_id] = order
        return order

    async def privateGetTradeOrder(self, params: dict[str, Any]) -> dict[str, Any]:
        return _native_order_detail_response(
            self.orders[str(params.get("ordId"))],
            params,
            default_inst_id="USAR-USDT-SWAP",
        )

    async def privatePostTradeClosePosition(self, params: dict[str, Any]) -> dict[str, Any]:
        self.close_position_calls.append(dict(params))
        if self.native_close_error:
            raise ExchangeAPIError("native close-position unavailable")
        self.position_contracts = 0.0
        return {
            "code": "0",
            "data": [
                {
                    "clOrdId": "native-close-client",
                    "ordId": "native-close-order",
                    "sCode": "0",
                    "sMsg": "",
                }
            ],
        }


class _MarketsByIdAliasMismatchCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.markets = {}
        self.markets_by_id = {
            "SPK-USDT-SWAP": {
                "symbol": "SAHARA/USDT:USDT",
                "id": "SAHARA-USDT-SWAP",
                "info": {"instId": "SAHARA-USDT-SWAP"},
            }
        }

    def market(self, _symbol: str) -> dict[str, Any]:
        raise RuntimeError("okx does not have market symbol")

    async def privateGetAccountPositions(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}


class _PositionAliasMismatchCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"
    markets = {}
    markets_by_id: dict[str, Any] = {}

    def market(self, _symbol: str) -> dict[str, Any]:
        raise RuntimeError("okx does not have market symbol")

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        return _filter_native_rows(
            [
                _native_position_row(
                    "SPK-USDT-SWAP",
                    pos="-200",
                    pos_side="net",
                    ct_val="1",
                    avg_px="0.012",
                    mark_px="0.011",
                    upl="0.2",
                )
            ],
            params,
        )


class _ExitPositionInstIdOnlyCcxt(_ExitMaxMarketSizeCcxt):
    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
            "info": {"instId": "SPK-USDT-SWAP", "maxMktSz": "100", "lotSz": "1"},
        }

    async def publicGetMarketTicker(self, params: dict[str, Any]) -> dict[str, Any]:
        return await _native_ticker(params, last=0.013, bid=0.013, ask=0.0131)

    async def fetch_ticker(self, _symbol: str) -> dict[str, Any]:
        raise AssertionError("execution sizing must use OKX native ticker API")

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        return _filter_native_rows(
            [
                _native_position_row(
                    "SPK-USDT-SWAP",
                    pos=self.position_contracts,
                    pos_side="long",
                    ct_val="1",
                    avg_px="0.012",
                    mark_px="0.013",
                    upl="0.018",
                )
            ],
            params,
        )


class _ExitNoMatchingSideCcxt(_ExitPositionInstIdOnlyCcxt):
    def __init__(self, position_contracts: float = 7.0) -> None:
        super().__init__(position_contracts=position_contracts, native_close_error=True)

    async def privatePostTradeClosePosition(self, params: dict[str, Any]) -> dict[str, Any]:
        self.close_position_calls.append(dict(params))
        raise ExchangeAPIError("native close-position unavailable")

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        return _filter_native_rows(
            [
                _native_position_row(
                    "SPK-USDT-SWAP",
                    pos="7",
                    pos_side="short",
                    ct_val="1",
                    avg_px="0.012",
                    mark_px="0.013",
                    upl="-0.007",
                ),
                _native_position_row(
                    "HOME-USDT-SWAP",
                    pos="3",
                    pos_side="long",
                    ct_val="1",
                    avg_px="0.021",
                    mark_px="0.022",
                    upl="0.003",
                ),
            ],
            params,
        )


class _NativeReduceNoPositionCcxt(_ExitMaxMarketSizeCcxt):
    async def privateGetTradeOrdersPending(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
            "info": {"instId": "USAR-USDT-SWAP", "lotSz": "1"},
            "synthetic_from_position": True,
        }

    async def privatePostTradeClosePosition(self, params: dict[str, Any]) -> dict[str, Any]:
        self.close_position_calls.append(dict(params))
        raise ExchangeAPIError("native close-position unavailable")

    async def privatePostTradeOrder(self, params: dict[str, Any]) -> dict[str, Any]:
        raise ExchangeAPIError(
            "OKX 51169: You don't have any positions in this direction."
        )


class _NativeFullCloseFillsHistoryCcxt(_ExitMaxMarketSizeCcxt):
    async def privatePostTradeClosePosition(self, params: dict[str, Any]) -> dict[str, Any]:
        self.close_position_calls.append(dict(params))
        self.position_contracts = 0.0
        return {"code": "0", "data": [{"clOrdId": "native-close-client", "sCode": "0"}]}

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instId"] == "USAR-USDT-SWAP"
        return {
            "data": [
                {
                    "ordId": "native-fill-order",
                    "instId": "USAR-USDT-SWAP",
                    "side": "sell",
                    "fillSz": "100",
                    "fillPx": "3.01",
                    "fillPnl": "71",
                    "fee": "-0.1505",
                    "ts": str(int(datetime.now(UTC).timestamp() * 1000)),
                }
            ]
        }


class _NativeFullCloseFillPendingCcxt(_ExitMaxMarketSizeCcxt):
    async def privatePostTradeClosePosition(self, params: dict[str, Any]) -> dict[str, Any]:
        self.close_position_calls.append(dict(params))
        self.position_contracts = 0.0
        return {"code": "0", "data": [{"clOrdId": "native-close-client", "sCode": "0"}]}


class _NativeFullCloseAccountWideFillsCcxt(_NativeFullCloseFillsHistoryCcxt):
    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("instId"):
            raise RuntimeError("instrument-specific history unavailable")
        return {
            "data": [
                {
                    "ordId": "native-fill-offline-order",
                    "instId": "USAR-USDT-SWAP-OFF",
                    "side": "sell",
                    "fillSz": "100",
                    "fillPx": "3.02",
                    "fillPnl": "72",
                    "fee": "-0.151",
                    "ts": str(int(datetime.now(UTC).timestamp() * 1000)),
                }
            ]
        }


def _executor(exchange: Any) -> OKXExecutor:
    executor = OKXExecutor(mode="paper")
    executor._connected = True
    executor._exchange = exchange
    return executor


def _exit_decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning="test exit",
        position_size_pct=1.0,
        suggested_leverage=3.0,
        raw_response={},
        feature_snapshot={"current_price": 100.0},
    )


def _spk_exit_decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="SPK/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning="test spk exit",
        position_size_pct=1.0,
        suggested_leverage=3.0,
        raw_response={},
        feature_snapshot={"current_price": 0.013},
    )


def _entry_decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="test entry",
        position_size_pct=0.1,
        suggested_leverage=5.0,
        raw_response={},
        feature_snapshot={"current_price": 100.0},
    )


def _shib_entry_decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="SHIB/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="test shib entry",
        position_size_pct=0.1,
        suggested_leverage=3.0,
        stop_loss_pct=0.012,
        take_profit_pct=0.024,
        raw_response={},
        feature_snapshot={"current_price": 0.000008789},
    )


def _secret_bearing_error() -> tuple[str, str, str]:
    token = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"
    hidden_value = "plain-credential-value"
    return token, hidden_value, f"Authorization: Bearer {token} failed password={hidden_value}"


@pytest.mark.asyncio
async def test_okx_balance_snapshot_error_is_redacted() -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    result = await _executor(_FailingBalanceCcxt(error_text)).get_balance_snapshot()

    rendered = str(result)
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in result["error"]
    assert "password=***" in result["error"]


@pytest.mark.asyncio
async def test_okx_balance_snapshot_does_not_require_instrument_rules() -> None:
    exchange = _BalanceOnlyCcxt()
    result = await _executor(exchange).get_balance_snapshot()

    assert result["free"] == 12.0
    assert result["allocatable"] == 16.0
    assert exchange.instrument_calls == 0


@pytest.mark.asyncio
async def test_okx_resolve_swap_symbol_rejects_ccxt_alias_to_different_inst_id() -> None:
    executor = _executor(_AliasMismatchMarketCcxt())
    executor._markets_loaded = True

    with pytest.raises(
        ExchangeAPIError, match="requested WLFI/USDT, exchange instrument is H/USDT"
    ):
        await executor._resolve_swap_symbol("WLFI/USDT")


@pytest.mark.asyncio
async def test_okx_resolve_swap_symbol_ignores_mismatched_markets_by_id_alias() -> None:
    executor = _executor(_MarketsByIdAliasMismatchCcxt())
    executor._markets_loaded = True

    resolved = await executor._resolve_swap_symbol("SPK/USDT")

    assert resolved == "SPK/USDT:USDT"


@pytest.mark.asyncio
async def test_okx_position_symbol_matching_prefers_native_inst_id_over_ccxt_alias() -> None:
    executor = _executor(_PositionAliasMismatchCcxt())
    executor._markets_loaded = True

    positions = await executor.get_positions_strict("SPK/USDT")
    resolved = await executor._resolve_swap_symbol("SPK/USDT")

    assert len(positions) == 1
    assert positions[0]["info"]["instId"] == "SPK-USDT-SWAP"
    assert resolved == "SPK-USDT-SWAP"


@pytest.mark.asyncio
async def test_okx_balance_snapshot_prevents_ccxt_implicit_market_loading() -> None:
    exchange = _CcxtBalanceWouldLoadMarkets()
    result = await _executor(exchange).get_balance_snapshot()

    assert result["free"] == 7.0
    assert result["allocatable"] == 8.0
    assert exchange.instrument_calls == 0
    assert exchange.markets_seen_by_fetch == {}


@pytest.mark.asyncio
async def test_okx_native_balance_snapshot_avoids_ccxt_market_loading() -> None:
    exchange = _NativeBalanceOnlyCcxt()
    result = await _executor(exchange).get_balance_snapshot()

    assert result["free"] == 12.0
    assert result["used"] == 3.0
    assert result["total"] == 16.0
    assert result["cash"] == 15.0
    assert result["allocatable"] == 16.0
    assert exchange.balance_calls == 1
    assert exchange.instrument_calls == 0


@pytest.mark.asyncio
async def test_okx_cancel_replace_error_is_redacted() -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    result = await _executor(_FailingCancelCcxt(error_text))._cancel_stale_exit_order(
        _FailingCancelCcxt(error_text),
        {},
        "BTC/USDT:USDT",
        "order-1",
        30.0,
    )

    rendered = str(result)
    assert result["cancel_success"] is False
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in result["cancel_error"]
    assert "password=***" in result["cancel_error"]


@pytest.mark.asyncio
async def test_okx_open_orders_failure_is_logged_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    fake_logger = _FakeLogger()
    monkeypatch.setattr(okx_module, "logger", fake_logger)

    result = await _executor(_FailingOpenOrdersCcxt(error_text)).get_open_orders("BTC/USDT")

    assert result == []
    assert fake_logger.events
    level, message, fields = fake_logger.events[-1]
    assert level == "warning"
    assert message == "fetch open orders failed"
    assert fields["symbol"] == "BTC/USDT:USDT"
    rendered = str(fields)
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in fields["error"]
    assert "password=***" in fields["error"]


@pytest.mark.asyncio
async def test_okx_exit_position_lookup_failure_does_not_return_no_position() -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    executor = _executor(_FailingPositionsForExitCcxt(error_text))

    with pytest.raises(OrderPlacementError) as exc_info:
        await executor.place_order(_exit_decision(), account_id="ensemble_trader")

    message = str(exc_info.value)
    assert token not in message
    assert hidden_value not in message
    assert "Authorization: ***" in message
    assert "password=***" in message
    assert "no_position" not in message


@pytest.mark.asyncio
async def test_okx_exit_uses_inst_id_position_snapshot_when_top_level_symbol_missing() -> None:
    exchange = _ExitPositionInstIdOnlyCcxt(position_contracts=18.0)
    result = await _executor(exchange).place_order(
        _spk_exit_decision(),
        account_id="ensemble_trader",
    )

    assert result.status == okx_module.OrderStatus.FILLED
    assert result.order_id == "native-close-order"
    assert result.symbol == "SPK/USDT"
    assert exchange.close_position_calls == [
        {"instId": "SPK-USDT-SWAP", "mgnMode": "cross", "autoCxl": True, "posSide": "long"}
    ]
    assert exchange.create_calls == []


@pytest.mark.asyncio
async def test_okx_exit_no_position_includes_position_mismatch_diagnostics() -> None:
    exchange = _ExitNoMatchingSideCcxt(position_contracts=7.0)
    result = await _executor(exchange).place_order(
        _spk_exit_decision(),
        account_id="ensemble_trader",
    )

    assert result.status == okx_module.OrderStatus.REJECTED
    assert result.order_id == "no_position"
    assert exchange.create_calls == []
    diagnostics = result.raw_response["okx_exit_position_mismatch"]
    assert diagnostics["source"] == "pre_submit_position_lookup"
    assert diagnostics["decision_symbol"] == "SPK/USDT"
    assert diagnostics["expected_okx_inst_id"] == "SPK-USDT-SWAP"
    assert diagnostics["target_position_side"] == "long"
    assert diagnostics["exit_order_side"] == "sell"
    assert diagnostics["positions_returned"] == 1
    assert diagnostics["matching_position_count"] == 0
    assert diagnostics["nonzero_same_symbol_sides"] == ["short"]
    spk_candidate = diagnostics["candidates"][0]
    assert spk_candidate["raw_symbol"] == "SPK-USDT-SWAP"
    assert spk_candidate["matches_expected_symbol"] is True
    assert spk_candidate["matches_target_side"] is False
    assert spk_candidate["reason"] == "side_mismatch"


@pytest.mark.asyncio
async def test_okx_native_reduce_no_position_keeps_snapshot_diagnostics() -> None:
    exchange = _NativeReduceNoPositionCcxt(position_contracts=18.0)
    executor = _executor(exchange)
    decision = _exit_decision()
    decision.symbol = "USAR/USDT"
    decision.position_size_pct = 0.5

    result = await executor.place_order(decision, account_id="ensemble_trader")

    assert result.status == okx_module.OrderStatus.REJECTED
    assert result.order_id == "no_position"
    assert result.raw_response["okx_native_reduce_market_order"] is True
    diagnostics = result.raw_response["okx_exit_position_mismatch"]
    assert diagnostics["source"] == "native_reduce_no_position_rejection"
    assert diagnostics["decision_symbol"] == "USAR/USDT"
    assert diagnostics["target_position_side"] == "long"
    assert diagnostics["matching_position_count"] == 1
    assert diagnostics["matching_contracts_total"] == 18.0
    assert diagnostics["candidates"][0]["reason"] == "matches"


@pytest.mark.asyncio
async def test_okx_exit_after_submit_position_refresh_failure_is_tracked_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    exchange = _FailingPositionsAfterExitSubmitCcxt(error_text)
    executor = _executor(exchange)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(okx_module.asyncio, "sleep", no_sleep)

    result = await executor.place_order(_exit_decision(), account_id="ensemble_trader")

    assert result.order_id == "exit-1"
    assert result.status == okx_module.OrderStatus.PENDING
    assert result.raw_response is not None
    assert result.raw_response["position_snapshot_unknown"] is True
    assert result.raw_response["position_contracts_after"] is None
    assert result.raw_response["remaining_contracts"] is None
    rendered = str(result.raw_response)
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in rendered
    assert "password=***" in rendered


def test_okx_attached_protection_uses_market_price_precision() -> None:
    exchange = _PrecisionEntryCcxt()
    executor = _executor(exchange)
    decision = _shib_entry_decision()
    stop_loss, take_profit = executor._attached_sl_tp_prices(
        decision,
        0.000008789,
        ticker={"last": 0.000008789, "bid": 0.000008788, "ask": 0.00000879},
    )

    result = executor._format_attached_sl_tp_prices(
        exchange,
        "SHIB/USDT:USDT",
        decision,
        stop_loss,
        take_profit,
        0.000008789,
    )

    assert result["ok"] is True
    assert result["stop_loss_price"] == "0.000008683"
    assert result["take_profit_price"] == "0.000009001"


def test_okx_attached_protection_rejects_invalid_direction_after_precision() -> None:
    class RoundedToReferenceCcxt(_PrecisionEntryCcxt):
        def price_to_precision(self, _symbol: str, _price: float) -> str:
            return "0.000008789"

    exchange = RoundedToReferenceCcxt()
    executor = _executor(exchange)
    result = executor._format_attached_sl_tp_prices(
        exchange,
        "SHIB/USDT:USDT",
        _shib_entry_decision(),
        0.00000868,
        0.000009,
        0.000008789,
    )

    assert result["ok"] is False
    assert result["stop_loss_price"] == "0.000008789"
    assert result["take_profit_price"] == "0.000008789"


@pytest.mark.asyncio
async def test_okx_leverage_open_order_limit_with_unknown_actual_rejects_entry() -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    executor = _executor(_LeverageUnknownAfterOpenOrderLimitCcxt(error_text))

    result = await executor._set_leverage_if_needed(_entry_decision())

    rendered = str(result)
    assert result["ok"] is False
    assert result["target_leverage"] == 5
    assert result["actual_leverage"] is None
    assert "未知杠杆" in result["error"]
    assert "59670" in result["open_order_limit_error"]
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in rendered
    assert "password=***" in rendered


class _FloorAmountPrecisionCcxt:
    def amount_to_precision(self, _symbol: str, amount: float) -> str:
        return str(float(int(amount)))


def test_okx_amount_min_uses_raw_okx_min_size() -> None:
    executor = OKXExecutor(mode="paper")
    market = {
        "symbol": "DOGE/USDT:USDT",
        "limits": {"amount": {"min": 0.0}},
        "info": {"minSz": "5", "lotSz": "1"},
    }

    assert executor._amount_min(market) == 5.0


def test_okx_entry_amount_lifts_to_okx_raw_min_size() -> None:
    executor = OKXExecutor(mode="paper")
    market = {
        "symbol": "DOGE/USDT:USDT",
        "contractSize": 1.0,
        "limits": {"amount": {"min": 0.0}},
        "info": {"minSz": "5", "lotSz": "1"},
    }

    contracts, base_quantity = executor._entry_order_amount(
        _FloorAmountPrecisionCcxt(),
        market,
        position_value=400.0,
        price=100.0,
        balance=500.0,
        leverage=1.0,
    )

    assert contracts == 5.0
    assert base_quantity == 5.0


def test_okx_order_contracts_ceil_after_precision_rounds_below_minimum() -> None:
    executor = OKXExecutor(mode="paper")
    market = {
        "symbol": "ALT/USDT:USDT",
        "limits": {"amount": {"min": 0.0}},
        "info": {"minSz": "1.1", "lotSz": "0.1"},
    }

    contracts = executor._normalize_order_contracts(
        _FloorAmountPrecisionCcxt(), market, contracts=1.05, min_contracts=1.1
    )

    assert contracts == 1.1


@pytest.mark.asyncio
async def test_okx_market_lookup_reloads_when_new_swap_missing_from_cache() -> None:
    exchange = _ReloadableMarketCcxt()
    executor = _executor(exchange)
    executor._markets_loaded = True

    market = await executor._market_for_symbol("USAR/USDT:USDT")

    assert market["symbol"] == "USAR/USDT:USDT"
    assert exchange.reload_calls == 1


def test_okx_entry_rule_snapshot_reads_raw_market_max_size() -> None:
    executor = OKXExecutor(mode="paper")
    market = {
        "symbol": "SAHARA/USDT:USDT",
        "contractSize": 1.0,
        "limits": {"amount": {"min": 1.0}},
        "info": {"maxMktSz": "100", "lotSz": "1"},
    }

    snapshot = executor._entry_order_rule_snapshot(
        market,
        price=1.0,
        balance=100.0,
        leverage=5.0,
        planned_notional_usdt=200.0,
        final_contracts=200.0,
    )

    assert snapshot["amount_max_market_contracts"] == 100.0
    assert snapshot["market_order_within_max_size"] is False
    assert snapshot["pre_submit_valid"] is False


@pytest.mark.asyncio
async def test_okx_entry_caps_market_order_above_exchange_max_before_submit() -> None:
    exchange = _EntryMaxMarketSizeCcxt()
    executor = _executor(exchange)
    decision = _entry_decision()
    decision.position_size_pct = 0.4
    decision.suggested_leverage = 5.0

    result = await executor.place_order(decision, override_balance=100.0)

    assert result.status.value == "filled"
    assert result.quantity == 100.0
    assert [call[3] for call in exchange.create_calls] == [100.0]
    adjustment = result.raw_response["market_order_size_adjustment"]
    assert adjustment["applied"] is True
    assert adjustment["original_planned_order_contracts"] == 200.0
    assert adjustment["adjusted_order_contracts"] == 100.0
    assert adjustment["amount_max_market_contracts"] == 100.0
    assert result.raw_response["okx_order_rules"]["market_order_within_max_size"] is True
    assert result.raw_response["okx_order_rules"]["pre_submit_valid"] is True


@pytest.mark.asyncio
async def test_okx_exit_splits_market_order_above_exchange_max_size() -> None:
    exchange = _ExitMaxMarketSizeCcxt(position_contracts=100.0)
    executor = _executor(exchange)
    decision = _exit_decision()
    decision.symbol = "USAR/USDT"
    decision.position_size_pct = 0.45

    result = await executor.place_order(decision)

    assert result.status.value == "filled"
    assert result.quantity == 45.0
    assert [call[3] for call in exchange.create_calls] == [10.0, 10.0, 10.0, 10.0, 5.0]
    assert all(call[5]["reduceOnly"] is True for call in exchange.create_calls)
    assert result.raw_response["split_exit_order"] is True
    assert result.raw_response["amount_max_market_contracts"] == 10.0
    assert result.raw_response["position_contracts_before"] == 100.0
    assert result.raw_response["position_contracts_after"] == 55.0
    assert result.raw_response["requested_exit_contracts"] == 45.0


@pytest.mark.asyncio
async def test_okx_exit_splits_full_close_above_exchange_max_size() -> None:
    exchange = _ExitMaxMarketSizeCcxt(position_contracts=100.0)
    executor = _executor(exchange)
    decision = _exit_decision()
    decision.symbol = "USAR/USDT"
    decision.position_size_pct = 1.0

    result = await executor.place_order(decision)

    assert result.status.value == "filled"
    assert result.quantity == 100.0
    assert exchange.create_calls == []
    assert exchange.close_position_calls == [
        {"instId": "USAR-USDT-SWAP", "mgnMode": "cross", "autoCxl": True, "posSide": "long"}
    ]
    assert result.raw_response["okx_native_close_position"] is True
    assert result.raw_response["position_contracts_before"] == 100.0
    assert result.raw_response["position_contracts_after"] == 0.0
    assert result.raw_response["requested_exit_fraction"] == 1.0
    assert result.raw_response["requested_exit_contracts"] == 100.0


@pytest.mark.asyncio
async def test_okx_native_full_close_uses_fills_history_when_response_has_no_order_id() -> None:
    exchange = _NativeFullCloseFillsHistoryCcxt(position_contracts=100.0)
    executor = _executor(exchange)
    decision = _exit_decision()
    decision.symbol = "USAR/USDT"
    decision.position_size_pct = 1.0

    result = await executor.place_order(decision)

    assert result.status.value == "filled"
    assert result.order_id == "native-fill-order"
    assert result.exchange_order_id == "native-fill-order"
    assert result.quantity == 100.0
    assert result.price == 3.01
    assert result.fee == 0.1505
    assert result.pnl == 71.0
    assert result.raw_response["native_close_fill"]["order_id"] == "native-fill-order"
    assert result.raw_response["native_close_fill"]["source"] == (
        "okx_fills_history_after_native_close"
    )


@pytest.mark.asyncio
async def test_okx_native_full_close_without_fill_order_id_waits_for_backfill() -> None:
    exchange = _NativeFullCloseFillPendingCcxt(position_contracts=100.0)
    executor = _executor(exchange)
    decision = _exit_decision()
    decision.symbol = "USAR/USDT"
    decision.position_size_pct = 1.0

    result = await executor.place_order(decision)

    assert result.status == OrderStatus.PARTIAL
    assert result.exchange_order_id is None
    assert result.order_id == "native-close-client"
    assert result.quantity == 100.0
    assert result.raw_response["requires_okx_fill_backfill"] is True
    assert result.raw_response["position_contracts_after"] == 0.0


@pytest.mark.asyncio
async def test_okx_native_full_close_falls_back_to_account_wide_fills_history() -> None:
    exchange = _NativeFullCloseAccountWideFillsCcxt(position_contracts=100.0)
    executor = _executor(exchange)
    decision = _exit_decision()
    decision.symbol = "USAR/USDT"
    decision.position_size_pct = 1.0

    result = await executor.place_order(decision)

    assert result.status.value == "filled"
    assert result.exchange_order_id == "native-fill-offline-order"
    assert result.price == 3.02
    assert result.fee == 0.151
    assert result.pnl == 72.0


@pytest.mark.asyncio
async def test_okx_exit_full_close_falls_back_to_split_when_native_close_fails() -> None:
    exchange = _ExitMaxMarketSizeCcxt(
        position_contracts=100.0,
        native_close_error=True,
    )
    executor = _executor(exchange)
    decision = _exit_decision()
    decision.symbol = "USAR/USDT"
    decision.position_size_pct = 1.0

    result = await executor.place_order(decision)

    assert result.status.value == "filled"
    assert result.quantity == 100.0
    assert exchange.close_position_calls == [
        {"instId": "USAR-USDT-SWAP", "mgnMode": "cross", "autoCxl": True, "posSide": "long"}
    ]
    assert [call[3] for call in exchange.create_calls] == [10.0] * 10
    assert result.raw_response["split_exit_order"] is True
    assert result.raw_response["position_contracts_after"] == 0.0
