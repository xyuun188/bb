from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

import executor.okx_executor as okx_module
from ai_brain.base_model import Action, DecisionOutput
from core.exceptions import ExchangeAPIError, OrderPlacementError
from executor.base_executor import OrderStatus
from executor.okx_executor import OKXExecutor
from services.entry_profit_risk_sizing import reconcile_profit_risk_sizing


@pytest.mark.parametrize(
    ("value", "expected"),
    ((190.0, "190"), (10.0, "10"), (0.01, "0.01"), (1.2300, "1.23")),
)
def test_okx_number_format_preserves_integer_trailing_zeroes(
    value: float,
    expected: str,
) -> None:
    assert OKXExecutor._format_okx_number(value) == expected


class _FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def warning(self, message: str, **kwargs: Any) -> None:
        self.events.append(("warning", message, kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        self.events.append(("info", message, kwargs))

    def debug(self, message: str, **kwargs: Any) -> None:
        self.events.append(("debug", message, kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        self.events.append(("error", message, kwargs))


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


class _InitTimeSyncCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.options = dict(config.get("options") or {})
        self.markets: dict[str, Any] = {}
        self.load_time_difference_calls = 0

    def set_sandbox_mode(self, _enabled: bool) -> None:
        return None

    async def load_time_difference(self) -> int:
        self.load_time_difference_calls += 1
        return 123

    async def close(self) -> None:
        return None


class _TimestampExpiredOnceCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.position_calls = 0
        self.load_time_difference_calls = 0

    async def load_time_difference(self) -> int:
        self.load_time_difference_calls += 1
        return 123

    async def privateGetAccountPositions(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.position_calls += 1
        if self.position_calls == 1:
            raise ExchangeAPIError('okx {"msg":"Timestamp request expired","code":"50102"}')
        return {"data": []}


class _SystemErrorOnceCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.position_history_calls = 0

    async def privateGetAccountPositionsHistory(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.position_history_calls += 1
        if self.position_history_calls == 1:
            raise ExchangeAPIError(
                "OKX API error [50026]: System error. Try again later.",
                code="50026",
            )
        return {"data": []}


class _TemporaryServiceOnceCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.position_calls = 0

    async def privateGetAccountPositions(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.position_calls += 1
        if self.position_calls == 1:
            raise ExchangeAPIError(
                "OKX API error [50001]: Service temporarily unavailable. Please try again later.",
                code="50001",
            )
        return {"data": []}


class _TemporaryServiceCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.position_calls = 0

    async def privateGetAccountPositions(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.position_calls += 1
        raise ExchangeAPIError(
            "OKX API error [50001]: Service temporarily unavailable. Please try again later.",
            code="50001",
        )


def _native_position_row(
    inst_id: str,
    *,
    pos: Any,
    pos_side: str = "long",
    leverage: Any = "0",
    ct_val: Any = "1",
    avg_px: Any = "100",
    mark_px: Any = "100",
    upl: Any = "0",
) -> dict[str, Any]:
    return {
        "instId": inst_id,
        "posSide": pos_side,
        "pos": str(pos),
        "lever": str(leverage),
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


class _EntryInstrumentAvailabilityCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.fetch_leverage_calls: list[str] = []

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "id": okx_module.okx_inst_id_from_symbol(symbol),
            "info": {"instId": okx_module.okx_inst_id_from_symbol(symbol)},
        }

    async def fetch_leverage(
        self,
        symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        self.fetch_leverage_calls.append(symbol)
        if symbol.startswith("PI/"):
            raise ExchangeAPIError(
                "OKX API error [51001]: Instrument ID doesn't exist.",
                code="51001",
            )
        return {"longLeverage": 1.0, "shortLeverage": 1.0, "info": []}


class _FailingEntryInstrumentAvailabilityCcxt(_EntryInstrumentAvailabilityCcxt):
    async def fetch_leverage(
        self,
        symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        self.fetch_leverage_calls.append(symbol)
        raise TimeoutError("private leverage transport timed out")


class _TemporaryServiceEntryInstrumentAvailabilityCcxt(
    _EntryInstrumentAvailabilityCcxt
):
    async def fetch_leverage(
        self,
        symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        self.fetch_leverage_calls.append(symbol)
        raise ExchangeAPIError(
            "OKX API error [50001]: Service temporarily unavailable. Please try again later.",
            code="50001",
        )


class _IncompatibleEntryEnvironmentCcxt(_EntryInstrumentAvailabilityCcxt):
    @staticmethod
    def _instrument(uly: str) -> dict[str, str]:
        return {
            "instId": "BTC-USDT-SWAP",
            "uly": uly,
            "ctVal": "0.01",
            "ctMult": "1",
            "ctValCcy": "BTC",
            "settleCcy": "USDT",
            "ctType": "linear",
            "lotSz": "0.01",
            "minSz": "0.01",
            "tickSz": "0.1",
        }

    async def publicGetPublicInstruments(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": [self._instrument("BTC-USDT")]}

    async def executionGetPublicInstruments(
        self,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        return {"data": [self._instrument("TEST-USDT")]}

    async def publicGetMarketTicker(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": [{"last": "60000"}]}

    async def executionGetMarketTicker(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": [{"last": "60000"}]}


class _MissingMarketCacheCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"
    markets: dict[str, Any] = {}
    markets_by_id: dict[str, Any] = {}

    def market(self, _symbol: str) -> dict[str, Any]:
        raise ExchangeAPIError("OKX SDK market is not loaded: NEW-USDT:USDT")

    async def publicGetPublicInstruments(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": [
            {
                "instId": "BTC-USDT-SWAP",
                "instType": "SWAP",
                "instCategory": "1",
                "state": "live",
                "settleCcy": "USDT",
                "uly": "BTC-USDT",
                "ctType": "linear",
                "ctVal": "0.01",
                "ctValCcy": "BTC",
                "minSz": "1",
                "lotSz": "1",
                "tickSz": "0.1",
            }
        ]}

    async def executionGetPublicInstruments(self, _params: dict[str, Any]) -> dict[str, Any]:
        return await self.publicGetPublicInstruments(_params)

    def parse_markets(self, _instruments: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "BTC/USDT:USDT": {
                "symbol": "BTC/USDT:USDT",
                "id": "BTC-USDT-SWAP",
                "info": {"instId": "BTC-USDT-SWAP"},
            }
        }

    def set_markets(self, markets: dict[str, Any]) -> None:
        self.markets = markets
        self.markets_by_id = {"BTC-USDT-SWAP": markets["BTC/USDT:USDT"]}

    async def publicGetMarketTicker(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": [{"last": "60000"}]}

    async def executionGetMarketTicker(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": [{"last": "60000"}]}


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


class _ExistingPositionLeverageCcxt:
    urls = {"api": {"rest": "https://www.okx.com"}}
    hostname = "www.okx.com"

    def __init__(self) -> None:
        self.set_leverage_calls = 0

    def market(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}},
            "info": {"instId": "BTC-USDT-SWAP"},
        }

    async def fetch_market_leverage_tiers(self, _symbol: str) -> list[dict[str, Any]]:
        return [{"maxLeverage": 20}]

    async def fetch_leverage(
        self,
        _symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        return {"longLeverage": 1, "shortLeverage": 1, "info": []}

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        return _filter_native_rows(
            [
                _native_position_row(
                    "BTC-USDT-SWAP",
                    pos="2",
                    pos_side="long",
                    leverage="2",
                )
            ],
            params,
        )

    async def set_leverage(
        self,
        _leverage: int,
        _symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        self.set_leverage_calls += 1
        raise AssertionError("existing-position add-on must not mutate OKX leverage")


class _FractionalLeverageCcxt(_ExistingPositionLeverageCcxt):
    def __init__(self) -> None:
        super().__init__()
        self.fetch_leverage_calls = 0
        self.requested_leverages: list[int] = []

    async def fetch_leverage(
        self,
        _symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        self.fetch_leverage_calls += 1
        if self.fetch_leverage_calls == 1:
            return {"longLeverage": 0, "shortLeverage": 0, "info": []}
        return {"longLeverage": 1, "shortLeverage": 1, "info": []}

    async def privateGetAccountPositions(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def set_leverage(
        self,
        leverage: int,
        _symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        self.set_leverage_calls += 1
        self.requested_leverages.append(leverage)
        return {"code": "0", "data": [{"lever": str(leverage)}]}


class _AccountCappedLeverageCcxt(_ExistingPositionLeverageCcxt):
    def __init__(
        self,
        *,
        accepted_max: int = 50,
        recovery_error: str = "",
        verification_stale: bool = False,
    ) -> None:
        super().__init__()
        self.accepted_max = accepted_max
        self.recovery_error = recovery_error
        self.verification_stale = verification_stale
        self.actual_leverage = 1
        self.requested_leverages: list[int] = []

    async def fetch_market_leverage_tiers(self, _symbol: str) -> list[dict[str, Any]]:
        return [{"maxLeverage": 100}]

    async def fetch_leverage(
        self,
        _symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        actual = 1 if self.verification_stale else self.actual_leverage
        return {"longLeverage": actual, "shortLeverage": actual, "info": []}

    async def privateGetAccountPositions(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def set_leverage(
        self,
        leverage: int,
        _symbol: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        self.requested_leverages.append(leverage)
        if leverage > self.accepted_max:
            raise ExchangeAPIError(
                "OKX API error [59102]: Leverage exceeds the maximum limit. "
                "Please lower the leverage.",
                code="59102",
            )
        if self.recovery_error:
            raise RuntimeError(self.recovery_error)
        self.actual_leverage = leverage
        return {"code": "0", "data": [{"lever": str(leverage)}]}


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
            "info": {
                "state": "filled",
                "ordId": "entry-max-market",
                "side": side,
                "attachAlgoOrds": [
                    {
                        "attachAlgoId": "entry-max-market-oco",
                        "tpTriggerPx": "1.02",
                        "slTriggerPx": "0.98",
                    }
                ],
            },
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


class _MovingEntryTickerCcxt(_EntryMaxMarketSizeCcxt):
    def __init__(self) -> None:
        super().__init__()
        self.ticker_calls = 0

    async def publicGetMarketTicker(self, params: dict[str, Any]) -> dict[str, Any]:
        self.ticker_calls += 1
        price = 1.0 if self.ticker_calls == 1 else 1.2
        return await _native_ticker(
            params,
            last=price,
            bid=price - 0.001,
            ask=price + 0.001,
        )

    async def publicGetPublicPriceLimit(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "code": "0",
            "data": [
                {
                    "instId": params["instId"],
                    "buyLmt": "1.3",
                    "sellLmt": "1.1",
                    "ts": "1730000000001",
                }
            ],
        }


class _AttachedProtectionRejectedOnceCcxt(_EntryMaxMarketSizeCcxt):
    def __init__(
        self,
        *,
        error_code: str = "51052",
        error_message: str = "Your TP price should be lower than the primary order price.",
    ) -> None:
        super().__init__()
        self.error_code = error_code
        self.error_message = error_message

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if params.get("attachAlgoOrds"):
            self.create_calls.append((symbol, order_type, side, quantity, price, params))
            raise ExchangeAPIError(
                f"OKX API error [{self.error_code}]: {self.error_message}",
                code=self.error_code,
            )
        order = await super().create_order(
            symbol,
            order_type,
            side,
            quantity,
            price,
            params,
        )
        info = order.get("info") if isinstance(order.get("info"), dict) else {}
        info.pop("attachAlgoOrds", None)
        return order


class _InvalidEntryPriceLimitCcxt(_MovingEntryTickerCcxt):
    async def publicGetPublicPriceLimit(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "code": "0",
            "data": [
                {
                    "instId": params["instId"],
                    "buyLmt": "0",
                    "sellLmt": "0",
                    "ts": "1730000000001",
                }
            ],
        }


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


class _ExitPositionDeltaWithoutFillCcxt(_ExitMaxMarketSizeCcxt):
    """Simulate OKX reducing a position while the order ack reports no fill."""

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.create_calls.append((symbol, order_type, side, quantity, price, dict(params)))
        self.position_contracts = max(self.position_contracts - quantity, 0.0)
        order_id = "exit-live-no-fill"
        return {
            "id": order_id,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": quantity,
            "filled": 0.0,
            "price": price or 3.0,
            "average": 0.0,
            "status": "open",
            "fee": {"cost": 0.0},
            "info": {
                "state": "live",
                "ordId": order_id,
                "side": side,
                "accFillSz": "0",
                "reduceOnly": "true",
            },
        }

    async def privateGetTradeOrder(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "data": [
                {
                    "ordId": str(params.get("ordId")),
                    "instId": "USAR-USDT-SWAP",
                    "state": "live",
                    "sz": "10",
                    "accFillSz": "0",
                    "avgPx": "",
                }
            ]
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
        raise ExchangeAPIError("OKX 51169: You don't have any positions in this direction.")


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


class _NativeFullCloseUnconfirmedCcxt(_ExitMaxMarketSizeCcxt):
    async def privatePostTradeClosePosition(self, params: dict[str, Any]) -> dict[str, Any]:
        self.close_position_calls.append(dict(params))
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
    if not callable(getattr(exchange, "privateGetAccountBalance", None)) and not callable(
        getattr(exchange, "fetch_balance", None)
    ):

        async def fixed_balance_snapshot(_asset: str = "USDT") -> dict[str, Any]:
            return {
                "free": 1000.0,
                "used": 0.0,
                "total": 1000.0,
                "cash": 1000.0,
                "equity": 1000.0,
                "allocatable": 1000.0,
            }

        executor.get_balance_snapshot = fixed_balance_snapshot  # type: ignore[method-assign]
    return executor


@pytest.mark.asyncio
async def test_okx_initialize_enables_and_loads_time_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}

    def fake_exchange(
        mode: str,
        *,
        close_http_after_call: bool = False,
    ) -> _InitTimeSyncCcxt:
        exchange = _InitTimeSyncCcxt({"options": {"defaultType": "swap"}})
        exchange.mode = mode
        exchange.close_http_after_call = close_http_after_call
        created["exchange"] = exchange
        return exchange

    monkeypatch.setattr(okx_module, "OkxPerpetualSdkExchange", fake_exchange)
    monkeypatch.setattr(type(okx_module.settings), "is_okx_demo", lambda _self, _mode: False)

    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    try:
        await executor.initialize()
    finally:
        await executor.shutdown()

    exchange = created["exchange"]
    assert exchange.mode == "paper"
    assert exchange.close_http_after_call is False
    assert exchange.load_time_difference_calls == 1


@pytest.mark.asyncio
async def test_okx_with_retry_resyncs_time_difference_after_timestamp_expired() -> None:
    exchange = _TimestampExpiredOnceCcxt()
    executor = _executor(exchange)

    result = await executor._with_retry(
        exchange.privateGetAccountPositions,
        {"instType": "SWAP"},
    )

    assert result == {"data": []}
    assert exchange.position_calls == 2
    assert exchange.load_time_difference_calls == 1


@pytest.mark.asyncio
async def test_okx_with_retry_recovers_from_temporary_50026(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _SystemErrorOnceCcxt()
    executor = _executor(exchange)
    monkeypatch.setattr(okx_module, "RETRY_DELAY", 0.0)

    result = await executor._with_retry(
        exchange.privateGetAccountPositionsHistory,
        {"instType": "SWAP"},
    )

    assert result == {"data": []}
    assert exchange.position_history_calls == 2


@pytest.mark.asyncio
async def test_okx_with_retry_recovers_from_temporary_50001(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _TemporaryServiceOnceCcxt()
    executor = _executor(exchange)
    monkeypatch.setattr(okx_module, "RETRY_DELAY", 0.0)

    result = await executor._with_retry(
        exchange.privateGetAccountPositions,
        {"instType": "SWAP"},
    )

    assert result == {"data": []}
    assert exchange.position_calls == 2


@pytest.mark.asyncio
async def test_okx_temporary_service_circuit_stops_private_request_storm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _TemporaryServiceCcxt()
    executor = _executor(exchange)
    monkeypatch.setattr(okx_module, "RETRY_DELAY", 0.0)

    with pytest.raises(ExchangeAPIError, match="50001"):
        await executor._with_retry(
            exchange.privateGetAccountPositions,
            {"instType": "SWAP"},
        )
    first_call_count = exchange.position_calls

    with pytest.raises(ExchangeAPIError, match="circuit open"):
        await executor._with_retry(
            exchange.privateGetAccountPositions,
            {"instType": "SWAP"},
        )

    assert first_call_count == okx_module.MAX_RETRIES
    assert exchange.position_calls == first_call_count


@pytest.mark.asyncio
async def test_okx_temporary_service_circuit_tracks_unified_fetch_leverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _TemporaryServiceEntryInstrumentAvailabilityCcxt()
    executor = _executor(exchange)
    monkeypatch.setattr(okx_module, "RETRY_DELAY", 0.0)

    first = await executor.entry_instrument_availability("BTC/USDT")
    second = await executor.entry_instrument_availability("ETH/USDT")

    assert first["available"] is None
    assert first["analysis_only"] is True
    assert first["execution_verified"] is False
    assert first["reason"] == "okx_private_entry_instrument_temporarily_unverified"
    assert second["available"] is None
    assert second["error_code"] == "50001"
    assert "circuit open" in second["error"]
    assert exchange.fetch_leverage_calls == ["BTC/USDT:USDT"]


@pytest.mark.parametrize(
    "method_name",
    [
        "privateGetAccountPositions",
        "fetch_balance",
        "fetch_leverage",
        "fetch_positions",
        "fetch_open_orders",
        "fetch_orders",
        "fetch_order",
        "fetch_my_trades",
        "create_order",
        "cancel_order",
        "set_leverage",
    ],
)
def test_okx_private_api_classifier_covers_unified_account_methods(
    method_name: str,
) -> None:
    assert OKXExecutor._is_private_api_method(method_name) is True


@pytest.mark.parametrize("method_name", ["fetch_ticker", "fetch_order_book", "fetch_ohlcv"])
def test_okx_private_api_classifier_keeps_public_market_methods_open(
    method_name: str,
) -> None:
    assert OKXExecutor._is_private_api_method(method_name) is False


@pytest.mark.asyncio
async def test_okx_temporary_service_circuit_uses_one_recovery_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _TemporaryServiceCcxt()
    executor = _executor(exchange)
    monkeypatch.setattr(okx_module, "RETRY_DELAY", 0.0)

    with pytest.raises(ExchangeAPIError, match="50001"):
        await executor._with_retry(
            exchange.privateGetAccountPositions,
            {"instType": "SWAP"},
        )

    executor._private_api_circuit_open_until = okx_module.time.monotonic() - 1.0
    executor._private_api_circuit_probe_in_flight = True
    with pytest.raises(ExchangeAPIError, match="recovery probe in progress"):
        await executor._with_retry(
            exchange.privateGetAccountPositions,
            {"instType": "SWAP"},
        )
    executor._private_api_circuit_probe_in_flight = False

    async def recovered(_params: dict[str, Any]) -> dict[str, Any]:
        exchange.position_calls += 1
        return {"data": []}

    recovered.__name__ = "privateGetAccountPositions"
    result = await executor._with_retry(recovered, {"instType": "SWAP"})

    assert result == {"data": []}
    assert executor._private_api_circuit_open_until == 0.0
    assert executor._private_api_circuit_failure_count == 0


def test_okx_private_api_circuit_state_is_shared_per_mode() -> None:
    paper_first = OKXExecutor(mode="paper")
    paper_second = OKXExecutor(mode="paper")
    live = OKXExecutor(mode="live")

    paper_first._private_api_circuit_failure_count = 2
    paper_first._private_api_circuit_open_until = okx_module.time.monotonic() + 10.0

    assert paper_second.private_api_circuit_status()["state"] == "open"
    assert paper_second.private_api_circuit_status()["failure_count"] == 2
    assert live.private_api_circuit_status()["state"] == "closed"
    assert live.private_api_circuit_status()["failure_count"] == 0


def test_okx_private_api_circuit_elects_one_probe_across_instances() -> None:
    paper_first = OKXExecutor(mode="paper")
    paper_second = OKXExecutor(mode="paper")
    paper_first._private_api_circuit_open_until = okx_module.time.monotonic() - 1.0

    assert paper_first._enter_private_api_circuit("privateGetAccountBalance", tracked=True) is True
    with pytest.raises(ExchangeAPIError, match="recovery probe in progress"):
        paper_second._enter_private_api_circuit("privateGetAccountPositions", tracked=True)


def test_okx_retry_classifier_recognizes_busy_50013() -> None:
    assert (
        OKXExecutor._is_transient_system_error(
            "OKX API error [50013]: Systems are busy. Please try again later."
        )
        is True
    )


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
        stop_loss_pct=0.013,
        take_profit_pct=0.027,
        raw_response={
            "profit_risk_sizing": {
                "production_eligible": True,
                "available_margin_usdt": 100.0,
                "position_size_pct": 0.1,
                "risk_budget_usdt": 100.0,
                "portfolio_risk_budget_usdt": 100.0,
                "current_portfolio_stressed_loss_usdt": 0.0,
                "planned_stressed_loss_usdt": 0.65,
                "target_notional_usdt": 1000.0,
                "final_notional_usdt": 50.0,
                "final_margin_usdt": 10.0,
                "stressed_loss_fraction": 0.013,
                "expected_net_return_pct": 1.0,
                "leverage_tier_selection": {
                    "production_eligible": True,
                    "max_leverage": 20.0,
                    "mark_price": 100.0,
                    "contract_spec": {"ctVal": "1", "ctMult": "1"},
                    "current_position_notional_usdt": 0.0,
                    "current_position_contracts": 0.0,
                },
                "policy_provenance": {
                    "source": "test",
                    "observation_window": "test",
                    "sample_count": 1,
                    "generated_at": "2026-07-15T00:00:00+00:00",
                    "strategy_version": "test",
                    "fallback_reason": "",
                },
            }
        },
        feature_snapshot={"current_price": 100.0},
    )


@pytest.mark.asyncio
async def test_okx_entry_symbol_resolution_failure_returns_structured_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor(object())

    async def fail_symbol_resolution(_symbol: str) -> str:
        raise ExchangeAPIError("OKX instrument resolution failed", code="51001")

    monkeypatch.setattr(executor, "_resolve_swap_symbol", fail_symbol_resolution)

    result = await executor.place_order(_entry_decision())

    assert result.status == OrderStatus.REJECTED
    assert result.order_id == "okx_rejected"
    assert result.raw_response["execution_blocker"] == "okx_exchange_rejection"
    assert result.raw_response["okx_error_code"] == "51001"
    assert result.raw_response["leverage_check"] == {}


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
async def test_okx_resolve_swap_symbol_falls_back_to_native_id_when_market_cache_is_missing() -> None:
    executor = _executor(_MissingMarketCacheCcxt())
    executor._markets_loaded = True

    resolved = await executor._resolve_swap_symbol("NEW/USDT")

    assert resolved == "NEW/USDT:USDT"


@pytest.mark.asyncio
async def test_okx_resolve_swap_symbol_treats_51001_position_lookup_as_no_position() -> None:
    executor = _executor(_MissingMarketCacheCcxt())
    executor._markets_loaded = True

    async def missing_position(_symbol: str) -> list[dict[str, Any]]:
        raise ExchangeAPIError(
            "OKX API error [51001]: Instrument ID doesn't exist.",
            code="51001",
        )

    executor.get_positions_strict = missing_position  # type: ignore[method-assign]

    resolved = await executor._resolve_swap_symbol("NEW/USDT")

    assert resolved == "NEW/USDT:USDT"


@pytest.mark.asyncio
async def test_okx_resolve_swap_symbol_keeps_non_51001_position_errors_strict() -> None:
    executor = _executor(_MissingMarketCacheCcxt())
    executor._markets_loaded = True

    async def failed_position_lookup(_symbol: str) -> list[dict[str, Any]]:
        raise ExchangeAPIError(
            "OKX API error [50001]: Service temporarily unavailable.",
            code="50001",
        )

    executor.get_positions_strict = failed_position_lookup  # type: ignore[method-assign]

    with pytest.raises(ExchangeAPIError, match="50001"):
        await executor._resolve_swap_symbol("NEW/USDT")


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
async def test_okx_entry_instrument_availability_uses_private_account_and_cache() -> None:
    exchange = _EntryInstrumentAvailabilityCcxt()
    executor = _executor(exchange)

    btc = await executor.entry_instrument_availability("BTC/USDT")
    btc_cached = await executor.entry_instrument_availability("BTC/USDT")
    pi = await executor.entry_instrument_availability("PI/USDT")
    pi_cached = await executor.entry_instrument_availability("PI/USDT")

    assert btc["available"] is True
    assert btc["source"] == "okx_private_account_leverage_info"
    assert btc_cached["available"] is True
    assert btc_cached["cache_hit"] is True
    assert pi["available"] is False
    assert pi["reason"] == "okx_private_entry_instrument_unavailable"
    assert pi["error_code"] == "51001"
    assert pi_cached["cache_hit"] is True
    assert exchange.fetch_leverage_calls == ["BTC/USDT:USDT", "PI/USDT:USDT"]


@pytest.mark.asyncio
async def test_okx_entry_instrument_prefilter_does_not_retry_transport_failure() -> None:
    exchange = _FailingEntryInstrumentAvailabilityCcxt()
    result = await _executor(exchange).entry_instrument_availability("BTC/USDT")

    assert result["available"] is None
    assert result["analysis_only"] is True
    assert result["execution_verified"] is False
    assert result["reason"] == "okx_private_entry_instrument_probe_failed"
    assert exchange.fetch_leverage_calls == ["BTC/USDT:USDT"]


@pytest.mark.asyncio
async def test_okx_entry_instrument_prefilter_allows_demo_underlying_alias() -> None:
    exchange = _IncompatibleEntryEnvironmentCcxt()
    result = await _executor(exchange).entry_instrument_availability("BTC/USDT")

    assert result["available"] is True
    assert result["environment_compatibility"]["compatible"] is True
    assert result["environment_compatibility"]["blockers"] == []
    assert result["environment_compatibility"]["warnings"] == ["uly_alias_mismatch"]
    assert exchange.fetch_leverage_calls == ["BTC/USDT:USDT"]


@pytest.mark.asyncio
async def test_okx_entry_instrument_price_missing_uses_transient_cache_ttl() -> None:
    exchange = _EntryInstrumentAvailabilityCcxt()
    executor = _executor(exchange)

    result = await executor.entry_instrument_availability(
        "BTC/USDT",
        environment_compatibility={
            "checked": True,
            "compatible": False,
            "blockers": ["environment_price_missing"],
        },
    )

    assert result["available"] is False
    inst_id = okx_module.okx_inst_id_from_symbol("BTC/USDT")
    cached_result, _cached_at, ttl = executor._entry_instrument_availability_cache[inst_id]
    assert cached_result["reason"] == "okx_entry_live_execution_environment_incompatible"
    assert ttl == okx_module.OKX_ENTRY_ENVIRONMENT_COMPATIBILITY_CACHE_SECONDS


@pytest.mark.asyncio
async def test_okx_entry_instrument_shortlist_stops_after_ranked_target() -> None:
    exchange = _EntryInstrumentAvailabilityCcxt()
    executor = _executor(exchange)

    result = await executor.entry_instrument_availability_shortlist(
        ["PI/USDT", "BTC/USDT", "ETH/USDT", "SOL/USDT"],
        target_count=1,
        concurrency=4,
    )

    assert result["selected_symbols"] == ["BTC/USDT"]
    assert result["evaluated_count"] == 2
    assert result["probed_count"] == 2
    assert result["skipped_after_target_count"] == 2
    assert exchange.fetch_leverage_calls == ["PI/USDT:USDT", "BTC/USDT:USDT"]
    assert result["availability"]["ETH/USDT"]["available"] is None


@pytest.mark.asyncio
async def test_okx_expected_instrument_rejection_is_not_logged_as_exchange_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()
    monkeypatch.setattr(okx_module, "logger", fake_logger)

    result = await _executor(_EntryInstrumentAvailabilityCcxt()).entry_instrument_availability(
        "PI/USDT"
    )

    assert result["reason"] == "okx_private_entry_instrument_unavailable"
    assert not [event for event in fake_logger.events if event[0] == "error"]
    assert ("debug", "OKX SDK expected capability rejection") == tuple(fake_logger.events[-1][:2])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_code", "error_message"),
    [
        ("51001", "Instrument ID doesn't exist."),
        ("52000", "No market data available."),
    ],
)
async def test_okx_instrument_capability_rejection_does_not_pollute_system_errors(
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
    error_message: str,
) -> None:
    fake_logger = _FakeLogger()
    monkeypatch.setattr(okx_module, "logger", fake_logger)
    executor = _executor(object())

    async def rejected_call() -> dict[str, Any]:
        raise ExchangeAPIError(
            f"OKX API error [{error_code}]: {error_message}",
            code=error_code,
        )

    rejected_call.__name__ = "privateGetAccountPositionsHistory"
    with pytest.raises(ExchangeAPIError, match=error_code):
        await executor._with_retry(rejected_call)

    assert not [event for event in fake_logger.events if event[0] == "error"]
    assert any(
        level == "info"
        and message == "OKX SDK instrument capability rejection"
        and fields["error_code"] == error_code
        for level, message, fields in fake_logger.events
    )
    assert executor.private_api_circuit_status()["state"] == "closed"


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


@pytest.mark.asyncio
async def test_okx_existing_position_add_on_reuses_authoritative_leverage() -> None:
    exchange = _ExistingPositionLeverageCcxt()
    executor = _executor(exchange)
    decision = _entry_decision()

    result = await executor._set_leverage_if_needed(decision)

    assert result["ok"] is True
    assert result["skipped_set"] is True
    assert result["existing_position"] is True
    assert result["actual_leverage"] == 2
    assert result["target_leverage"] == 2
    assert decision.suggested_leverage == 2
    assert exchange.set_leverage_calls == 0

    planned_notional = decision.raw_response["profit_risk_sizing"]["final_notional_usdt"]
    reconciled = reconcile_profit_risk_sizing(
        decision,
        final_notional_usdt=planned_notional,
        final_leverage=result["actual_leverage"],
        source="test_okx_existing_position_actual_leverage",
    )
    sizing = decision.raw_response["profit_risk_sizing"]
    assert reconciled["eligible"] is True
    assert sizing["final_notional_usdt"] == pytest.approx(planned_notional)
    assert sizing["final_margin_usdt"] == pytest.approx(planned_notional / 2.0)


@pytest.mark.asyncio
async def test_okx_fractional_model_leverage_is_normalized_downward() -> None:
    exchange = _FractionalLeverageCcxt()
    executor = _executor(exchange)
    decision = _entry_decision()
    decision.suggested_leverage = 1.5
    decision.raw_response["profit_risk_sizing"]["model_requested_leverage"] = 1.5

    result = await executor._set_leverage_if_needed(decision)

    assert result["ok"] is True
    assert result["ai_requested_leverage"] == 1
    assert result["target_leverage"] == 1
    assert result["actual_leverage"] == 1
    assert exchange.requested_leverages == [1]
    assert decision.suggested_leverage == 1

    planned_notional = decision.raw_response["profit_risk_sizing"]["final_notional_usdt"]
    reconciled = reconcile_profit_risk_sizing(
        decision,
        final_notional_usdt=planned_notional,
        final_leverage=result["actual_leverage"],
        source="test_fractional_model_leverage",
    )
    assert reconciled["eligible"] is True
    assert "execution_leverage_exceeds_model_request" not in reconciled["reasons"]


@pytest.mark.asyncio
async def test_okx_59102_discovers_and_verifies_account_accepted_leverage() -> None:
    exchange = _AccountCappedLeverageCcxt(accepted_max=50)
    executor = _executor(exchange)
    decision = _entry_decision()
    decision.suggested_leverage = 60
    decision.raw_response["profit_risk_sizing"]["leverage_tier_selection"]["max_leverage"] = 100

    result = await executor._set_leverage_if_needed(decision)

    assert result["ok"] is True
    assert result["target_leverage"] == 50
    assert result["actual_leverage"] == 50
    assert result["okx_max_leverage"] == 50
    assert result["okx_leverage_tier_selection"]["max_leverage"] == 100
    assert result["leverage_limit_recovery"]["rejected_leverage"] == 60
    assert result["leverage_limit_recovery"]["accepted_leverage"] == 50
    assert len(result["leverage_limit_recovery"]["attempts"]) <= 7
    assert exchange.requested_leverages[0] == 60
    assert max(exchange.requested_leverages[1:]) < 60
    assert decision.suggested_leverage == 50


@pytest.mark.asyncio
async def test_okx_59102_recovery_fails_closed_on_unexpected_error() -> None:
    exchange = _AccountCappedLeverageCcxt(
        accepted_max=50,
        recovery_error="unexpected account endpoint failure",
    )
    executor = _executor(exchange)
    decision = _entry_decision()
    decision.suggested_leverage = 60
    decision.raw_response["profit_risk_sizing"]["leverage_tier_selection"]["max_leverage"] = 100

    result = await executor._set_leverage_if_needed(decision)

    assert result["ok"] is False
    assert result["leverage_limit_recovery"]["ok"] is False
    assert result["leverage_limit_recovery"]["reason"] == (
        "unexpected_error_during_account_leverage_recovery"
    )
    assert exchange.requested_leverages == [60, 31]
    assert decision.suggested_leverage == 60


@pytest.mark.asyncio
async def test_okx_59102_recovery_rejects_unverified_leverage() -> None:
    exchange = _AccountCappedLeverageCcxt(
        accepted_max=50,
        verification_stale=True,
    )
    executor = _executor(exchange)
    decision = _entry_decision()
    decision.suggested_leverage = 60
    decision.raw_response["profit_risk_sizing"]["leverage_tier_selection"]["max_leverage"] = 100

    result = await executor._set_leverage_if_needed(decision)

    assert result["ok"] is False
    assert result["leverage_limit_recovery"]["reason"] == (
        "account_leverage_recovery_verification_mismatch"
    )
    assert result["actual_leverage"] == 1
    assert decision.suggested_leverage == 60


@pytest.mark.asyncio
async def test_okx_pre_order_execution_facts_share_native_instrument_and_units() -> None:
    class _PreOrderFactsCcxt:
        urls = {"api": {"rest": "https://www.okx.com"}}
        hostname = "www.okx.com"

        def market(self, symbol: str) -> dict[str, Any]:
            return {
                "symbol": symbol,
                "id": "BTC-USDT-SWAP",
                "info": {"instId": "BTC-USDT-SWAP"},
            }

        async def publicGetMarketTicker(self, _params: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("paper execution facts must not use live ticker")

        async def executionGetMarketTicker(self, params: dict[str, Any]) -> dict[str, Any]:
            assert params["instId"] == "BTC-USDT-SWAP"
            return {
                "data": [
                    {
                        "instId": "BTC-USDT-SWAP",
                        "last": "100",
                        "bidPx": "99.9",
                        "askPx": "100.1",
                        "ts": "1780000000000",
                    }
                ]
            }

        async def fetch_order_book(self, _symbol: str) -> dict[str, Any]:
            raise AssertionError("paper execution facts must not use live order book")

        async def executionFetchOrderBook(self, symbol: str) -> dict[str, Any]:
            assert symbol == "BTC/USDT:USDT"
            return {
                "bids": [[99.9, 2.0]],
                "asks": [[100.1, 3.0]],
                "timestamp": 1780000000001,
            }

        async def publicGetPublicMarkPrice(self, _params: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("paper execution facts must not use live mark price")

        async def executionGetPublicMarkPrice(self, params: dict[str, Any]) -> dict[str, Any]:
            assert params["instId"] == "BTC-USDT-SWAP"
            return {"data": [{"instId": "BTC-USDT-SWAP", "markPx": "100.05", "ts": "2"}]}

        async def publicGetPublicInstruments(self, _params: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("paper execution facts must not use live contract rules")

        async def executionGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
            assert params == {"instType": "SWAP"}
            return {
                "data": [
                    {
                        "instId": "BTC-USDT-SWAP",
                        "instType": "SWAP",
                        "instCategory": "1",
                        "state": "live",
                        "settleCcy": "USDT",
                        "uly": "BTC-USDT",
                        "ctVal": "0.01",
                        "ctMult": "1",
                        "ctValCcy": "BTC",
                        "minSz": "1",
                        "lotSz": "1",
                        "tickSz": "0.1",
                    }
                ]
            }

        def parse_markets(self, instruments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
            return {
                "BTC/USDT:USDT": {
                    "symbol": "BTC/USDT:USDT",
                    "id": "BTC-USDT-SWAP",
                    "info": dict(instruments[0]),
                }
            }

        def set_markets(self, markets: dict[str, dict[str, Any]]) -> None:
            self.markets = markets
            self.markets_by_id = {"BTC-USDT-SWAP": markets["BTC/USDT:USDT"]}

        async def privateGetAccountFeeRates(self, params: dict[str, Any]) -> dict[str, Any]:
            assert params == {"instType": "SWAP"}
            return {"data": [{"taker": "-0.0005", "ts": "1780000000002"}]}

    executor = _executor(_PreOrderFactsCcxt())

    facts = await executor.pre_order_execution_facts("BTC/USDT", "long")

    assert facts["production_eligible"] is True
    assert facts["inst_id"] == "BTC-USDT-SWAP"
    snapshot = facts["feature_snapshot"]
    assert snapshot["contract_value_base"] == pytest.approx(0.01)
    assert snapshot["orderbook_bid_depth"] == pytest.approx(99.9 * 2.0 * 0.01)
    assert snapshot["orderbook_ask_depth"] == pytest.approx(100.1 * 3.0 * 0.01)
    assert snapshot["mark_price"] == pytest.approx(100.05)
    assert snapshot["taker_fee_rate"] == pytest.approx(0.0005)


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


def test_okx_entry_amount_below_raw_min_size_is_rejected_without_enlargement() -> None:
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

    assert contracts == 0.0
    assert base_quantity == 0.0


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
    decision.raw_response["profit_risk_sizing"].update(
        {
            "final_notional_usdt": 200.0,
            "final_margin_usdt": 40.0,
            "planned_stressed_loss_usdt": 2.6,
        }
    )

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
    submission = result.raw_response["protection_submission"]
    assert submission["state"] == "confirmed"
    assert submission["exchange_confirmation_recorded"] is True
    assert submission["algo_ids"] == ["entry-max-market-oco"]
    assert submission["client_submit_requested_at"]
    assert submission["exchange_confirmed_at"]


@pytest.mark.asyncio
async def test_okx_entry_reprices_attached_protection_immediately_before_submit() -> None:
    exchange = _MovingEntryTickerCcxt()
    executor = _executor(exchange)
    decision = _entry_decision()
    decision.feature_snapshot = {"current_price": 1.0}

    result = await executor.place_order(decision, override_balance=100.0)

    assert result.status.value == "filled"
    assert exchange.ticker_calls == 2
    assert exchange.create_calls[0][3] == pytest.approx(50.0 / 1.3)
    request_params = exchange.create_calls[0][5]
    protection = request_params["attachAlgoOrds"][0]
    assert float(protection["slTriggerPx"]) < 1.1
    assert float(protection["tpTriggerPx"]) > 1.3
    assert protection["slTriggerPxType"] == "last"
    assert protection["tpTriggerPxType"] == "last"
    refresh = result.raw_response["okx_order_rules"]["pre_submit_price_refresh"]
    assert refresh["previous_price"] == 1.0
    assert refresh["refreshed_price"] == 1.2
    assert refresh["source"] == "okx_native_ticker_immediately_before_submit"
    assert refresh["buy_price_limit"] == 1.3
    assert refresh["sell_price_limit"] == 1.1
    assert refresh["price_limit_timestamp_ms"] == 1730000000001
    rules = result.raw_response["okx_order_rules"]
    assert rules["fill_risk_price"] == 1.3
    assert rules["fill_risk_price_source"] == "okx_buy_price_limit"
    assert rules["maximum_fill_notional_usdt"] == pytest.approx(50.0)
    assert rules["pre_submit_valid"] is True


def test_okx_entry_fill_integrity_quarantines_cross_scale_price_and_underlying() -> None:
    executor = OKXExecutor(mode="paper")

    result = executor._entry_fill_integrity(
        order={
            "info": {
                "instId": "STRK-USDT-SWAP",
                "uly": "TSLA-USDT",
            }
        },
        okx_symbol="STRK-USDT-SWAP",
        reference_price=0.02309,
        execution_price=375.952409,
    )

    assert result["ok"] is False
    assert "execution_price_far_from_pre_submit_quote" in result["reasons"]
    assert "execution_underlying_mismatch" in result["reasons"]
    assert result["price_ratio"] > 100.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "error_code", "error_message"),
    [
        (Action.LONG, "51050", "Your TP price should be higher than the primary order price."),
        (Action.LONG, "51051", "Your SL price should be lower than the primary order price."),
        (Action.SHORT, "51052", "Your TP price should be lower than the primary order price."),
        (Action.SHORT, "51053", "Your SL price should be higher than the primary order price."),
    ],
)
async def test_okx_entry_recovers_valid_attached_protection_rejection_with_verified_oco(
    monkeypatch: pytest.MonkeyPatch,
    action: Action,
    error_code: str,
    error_message: str,
) -> None:
    exchange = _AttachedProtectionRejectedOnceCcxt(
        error_code=error_code,
        error_message=error_message,
    )
    executor = _executor(exchange)
    decision = _entry_decision()
    decision.action = action
    recovery_calls: list[dict[str, Any]] = []

    async def ensure_protection(**kwargs: Any) -> dict[str, Any]:
        recovery_calls.append(kwargs)
        return {
            "verified": True,
            "created": True,
            "created_algo_id": "standalone-oco-1",
            "status": "created_and_verified",
        }

    monkeypatch.setattr(executor, "_ensure_partial_entry_protection", ensure_protection)

    result = await executor.place_order(decision, override_balance=100.0)

    assert result.status == OrderStatus.FILLED
    assert len(exchange.create_calls) == 2
    first_params = exchange.create_calls[0][5]
    retry_params = exchange.create_calls[1][5]
    assert first_params["attachAlgoOrds"]
    assert "attachAlgoOrds" not in retry_params
    assert len(recovery_calls) == 1
    assert recovery_calls[0]["filled_contracts"] == pytest.approx(50.0)
    assert recovery_calls[0]["params"]["attachAlgoOrds"] == first_params["attachAlgoOrds"]

    recovery = result.raw_response["entry_attached_protection_recovery"]
    assert recovery["okx_error_code"] == error_code
    assert recovery["version"] == "2026-08-06.okx-attached-protection-recovery.v2"
    assert recovery["trigger"] == "okx_attached_protection_false_rejection"
    assert recovery["standalone_protection"]["verified"] is True
    submission = result.raw_response["protection_submission"]
    assert submission["exchange_confirmation_recorded"] is True
    assert submission["state"] == "confirmed"
    assert submission["algo_ids"] == ["standalone-oco-1"]
    assert submission["submission_path"] == (
        "standalone_oco_after_attached_protection_rejection"
    )


@pytest.mark.asyncio
async def test_okx_entry_does_not_recover_take_profit_code_for_wrong_direction() -> None:
    exchange = _AttachedProtectionRejectedOnceCcxt()
    executor = _executor(exchange)

    result = await executor.place_order(_entry_decision(), override_balance=100.0)

    assert result.status == OrderStatus.REJECTED
    assert result.raw_response["okx_error_code"] == "51052"
    assert "attached_protection_recovery" not in result.raw_response
    assert len(exchange.create_calls) == 1


@pytest.mark.asyncio
async def test_okx_entry_rechecks_fresh_available_margin_before_submit() -> None:
    exchange = _EntryMaxMarketSizeCcxt()
    executor = _executor(exchange)
    decision = _entry_decision()

    async def low_balance(_asset: str = "USDT") -> dict[str, Any]:
        return {"free": 9.0, "equity": 1000.0, "total": 1000.0}

    executor.get_balance_snapshot = low_balance  # type: ignore[method-assign]
    result = await executor.place_order(decision, override_balance=100.0)

    assert result.status.value == "rejected"
    assert result.raw_response["execution_blocker"] == "okx_pre_submit_available_margin"
    assert result.raw_response["available_margin_usdt"] == 9.0
    assert result.raw_response["required_margin_usdt"] > 9.0
    assert exchange.create_calls == []


@pytest.mark.asyncio
async def test_okx_entry_rejects_before_submit_when_native_price_limit_is_invalid() -> None:
    exchange = _InvalidEntryPriceLimitCcxt()
    executor = _executor(exchange)
    decision = _entry_decision()

    result = await executor.place_order(decision, override_balance=100.0)

    assert result.status.value == "rejected"
    assert result.raw_response["system_pre_submit_rejection"] is True
    assert result.raw_response["okx_rejection"] is False
    assert result.raw_response["execution_blocker"] == "okx_pre_submit_execution_quote_unavailable"
    assert "price limit has no positive bounds" in result.raw_response["raw_error"]
    assert exchange.create_calls == []


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
async def test_okx_exit_position_delta_without_fill_is_pending_backfill() -> None:
    exchange = _ExitPositionDeltaWithoutFillCcxt(position_contracts=100.0)
    executor = _executor(exchange)
    decision = _exit_decision()
    decision.symbol = "USAR/USDT"
    decision.position_size_pct = 0.05

    result = await executor.place_order(decision)

    assert result.status == OrderStatus.PARTIAL
    assert result.exchange_order_id == "exit-live-no-fill"
    assert result.quantity == pytest.approx(5.0)
    assert result.raw_response["requires_okx_fill_backfill"] is True
    assert result.raw_response["fill_confirmation_basis"] == (
        "okx_position_delta_pending_order_fill"
    )
    assert result.raw_response["exchange_reported_filled_contracts"] == pytest.approx(0.0)
    assert result.raw_response["position_contracts_before"] == pytest.approx(100.0)
    assert result.raw_response["position_contracts_after"] == pytest.approx(95.0)


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
async def test_okx_native_full_close_unconfirmed_is_single_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _NativeFullCloseUnconfirmedCcxt(position_contracts=100.0)
    executor = _executor(exchange)
    decision = _exit_decision()
    decision.symbol = "USAR/USDT"
    decision.position_size_pct = 1.0

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(okx_module.asyncio, "sleep", no_sleep)

    first, second = await asyncio.gather(
        executor.place_order(decision),
        executor.place_order(decision),
    )

    assert first.status == OrderStatus.OPEN
    assert first.order_id == "native-close-client"
    assert second.status == OrderStatus.OPEN
    assert second.order_id == "exit_singleflight_wait"
    assert second.raw_response["exit_singleflight_wait"] is True
    assert second.raw_response["do_not_persist_order"] is True
    assert len(exchange.close_position_calls) == 1


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
