"""Unified OKX SDK adapter for USDT perpetual swaps.

All OKX network calls in the trading stack should enter through this module.
The adapter intentionally exposes the small CCXT-like surface the existing
executor and sync services already use, but the transport is the official
``python-okx`` SDK and every trading instrument is forced to ``SWAP``.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal, InvalidOperation
from typing import Any

import structlog

from config.settings import settings
from core.exceptions import ConfigError, ExchangeAPIError
from core.safe_output import safe_error_text
from core.symbols import normalize_trading_symbol, okx_inst_id_from_symbol, symbol_from_okx_inst_id

logger = structlog.get_logger(__name__)

OKX_DOMAIN = "https://www.okx.com"
OKX_SWAP_INST_TYPE = "SWAP"
OKX_SWAP_SETTLE_CCY = "USDT"
OKX_CROSS_MARGIN_MODE = "cross"
OKX_SERVER_TIME_PATH = "/api/v5/public/time"
OKX_SERVER_TIME_SYNC_TTL_SECONDS = 30.0
OKX_WS_PUBLIC_URL = "wss://ws.okx.com:8443/ws/v5/public"
OKX_WS_BUSINESS_URL = "wss://ws.okx.com:8443/ws/v5/business"
OKX_WS_DEMO_URL = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"


def okx_sdk_flag_for_mode(mode: str) -> str:
    """Return python-okx flag: ``1`` for demo/simulated, ``0`` for live."""

    return "1" if settings.is_okx_demo(mode) else "0"


def okx_proxy_url() -> str | None:
    return (
        settings.okx_proxy
        or os.environ.get("OKX_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("ALL_PROXY")
        or os.environ.get("all_proxy")
    )


def raise_if_okx_error(
    result: Any,
    *,
    fallback: str = "OKX SDK API error",
    check_data_code: bool = False,
) -> None:
    """Raise a typed project error for failed OKX SDK responses."""

    if not isinstance(result, Mapping):
        raise ExchangeAPIError(f"{fallback}: unexpected response type {type(result).__name__}")
    code = str(result.get("code") or "")
    if code not in {"", "0"}:
        message = safe_error_text(result.get("msg") or fallback, limit=240)
        raise ExchangeAPIError(
            f"OKX API error [{safe_error_text(code, limit=40)}]: {message}",
            code=code,
            payload=dict(result),
        )
    if not check_data_code:
        return
    rows = result.get("data")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        item_code = str(row.get("sCode") or row.get("code") or "")
        if item_code in {"", "0"}:
            continue
        message = safe_error_text(row.get("sMsg") or row.get("msg") or fallback, limit=240)
        raise ExchangeAPIError(
            f"OKX API error [{safe_error_text(item_code, limit=40)}]: {message}",
            code=item_code,
            payload={"response": dict(result), "item": dict(row)},
        )


def _format_number(value: Any) -> str:
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return str(value)
    return format(decimal_value.normalize(), "f").rstrip("0").rstrip(".") or "0"


def _timestamp_from_epoch_ms(epoch_ms: int) -> str:
    return (
        datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _timeframe_to_okx_bar(timeframe: str) -> str:
    text = str(timeframe or "1h").strip()
    mapping = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "6h": "6H",
        "12h": "12H",
        "1d": "1D",
        "1w": "1W",
        "1M": "1M",
    }
    return mapping.get(text, mapping.get(text.lower(), text))


def _okx_bar_to_milliseconds(bar: str) -> int:
    text = str(bar or "").strip()
    unit = text[-1:].lower()
    try:
        amount = int(text[:-1] or "1")
    except ValueError:
        amount = 1
    if unit == "m":
        return amount * 60_000
    if unit == "h":
        return amount * 3_600_000
    if unit == "d":
        return amount * 86_400_000
    if unit == "w":
        return amount * 7 * 86_400_000
    return 60_000


def _normalize_bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    return ""


class OkxPublicWebSocketSdkStream:
    """Small async stream wrapper around python-okx public WebSocket SDK."""

    def __init__(self, url: str = OKX_WS_PUBLIC_URL) -> None:
        self.url = url
        self._client: Any | None = None
        self._consume_task: asyncio.Task[Any] | None = None
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def connect(self) -> None:
        from okx.websocket.WsPublicAsync import WsPublicAsync

        self._client = WsPublicAsync(self.url, debug=False)
        self._consume_task = await self._client.start()

    def _on_message(self, message: str) -> None:
        self._queue.put_nowait(message)

    async def send(self, payload: str) -> None:
        if self._client is None:
            raise ExchangeAPIError("OKX WebSocket SDK client is not connected")
        text = payload.decode() if isinstance(payload, bytes) else str(payload)
        if text == "ping":
            websocket = getattr(self._client, "websocket", None)
            if websocket is not None:
                await websocket.send(text)
            return
        try:
            message = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExchangeAPIError("OKX WebSocket SDK payload must be JSON") from exc
        op = str(message.get("op") or "")
        args = message.get("args") or []
        request_id = message.get("id")
        if op == "subscribe":
            await self._client.subscribe(args, self._on_message, id=request_id)
        elif op == "unsubscribe":
            await self._client.unsubscribe(args, self._on_message, id=request_id)
        else:
            await self._client.send(op, args, callback=self._on_message, id=request_id)

    async def recv(self) -> str:
        return await self._queue.get()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.stop()
        if self._consume_task is not None and not self._consume_task.done():
            self._consume_task.cancel()
        self._client = None
        self._consume_task = None


def normalize_swap_inst_id(value: Any, *, field: str = "instId", required: bool = True) -> str:
    """Normalize and enforce a USDT perpetual-swap OKX instrument id."""

    raw = str(value or "").strip().upper()
    if not raw:
        if required:
            raise ExchangeAPIError(f"OKX {field} is required for perpetual swap API call")
        return ""
    if raw.endswith("-USDT-SWAP"):
        return raw
    if field == "instId" and "-" in raw and "/" not in raw and ":" not in raw:
        raise ExchangeAPIError(f"Only OKX USDT perpetual swaps are supported, got {field}={raw!r}")
    inst_id = okx_inst_id_from_symbol(raw)
    if inst_id.endswith("-USDT-SWAP"):
        return inst_id
    raise ExchangeAPIError(f"Only OKX USDT perpetual swaps are supported, got {field}={raw!r}")


def _swap_params(params: Mapping[str, Any] | None, *, require_inst_id: bool = False) -> dict[str, Any]:
    source = dict(params or {})
    inst_type = str(source.get("instType") or OKX_SWAP_INST_TYPE).upper()
    if inst_type != OKX_SWAP_INST_TYPE:
        raise ExchangeAPIError(f"Only OKX SWAP instType is supported, got {inst_type!r}")
    source["instType"] = OKX_SWAP_INST_TYPE
    if source.get("instId") or require_inst_id:
        source["instId"] = normalize_swap_inst_id(
            source.get("instId"),
            required=require_inst_id,
        )
    return source


def _instrument_to_market(instrument: Mapping[str, Any]) -> dict[str, Any]:
    inst_id = str(instrument.get("instId") or "").strip().upper()
    symbol = symbol_from_okx_inst_id(inst_id)
    base = symbol.split("/")[0] if "/" in symbol else inst_id.split("-")[0]
    contract_size = _safe_float(instrument.get("ctVal"), 1.0) or 1.0
    amount_step = _safe_float(instrument.get("lotSz"), 0.0)
    min_size = _safe_float(instrument.get("minSz"), 0.0)
    max_market_size = _safe_float(instrument.get("maxMktSz"), 0.0)
    tick_size = _safe_float(instrument.get("tickSz"), 0.0)
    amount_min = max(value for value in (min_size, amount_step, 0.0) if value >= 0)
    return {
        "id": inst_id,
        "symbol": f"{base}/USDT:USDT",
        "base": base,
        "quote": "USDT",
        "settle": "USDT",
        "type": "swap",
        "swap": True,
        "linear": True,
        "contract": True,
        "active": str(instrument.get("state") or "").lower() == "live",
        "contractSize": contract_size,
        "precision": {
            "amount": amount_step or min_size or 0.0,
            "price": tick_size,
        },
        "limits": {
            "amount": {
                "min": amount_min,
                "max": max_market_size or None,
            }
        },
        "info": dict(instrument),
    }


class OkxPerpetualSdkExchange:
    """Async SDK-backed OKX exchange adapter for USDT perpetual swaps."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.markets: dict[str, dict[str, Any]] = {}
        self.markets_by_id: dict[str, list[dict[str, Any]]] = {}
        self.urls: dict[str, Any] = {"api": {"rest": OKX_DOMAIN}, "test": {"rest": OKX_DOMAIN}}
        self.hostname = "www.okx.com"
        self._account_api: Any | None = None
        self._trade_api: Any | None = None
        self._market_api: Any | None = None
        self._public_api: Any | None = None
        self._server_time_lock = threading.Lock()
        self._server_time_offset_ms: int | None = None
        self._server_time_synced_at: float = 0.0

    @property
    def account_api(self) -> Any:
        if self._account_api is None:
            from okx.Account import AccountAPI

            self._account_api = self._configure_private_api(AccountAPI(**self._private_kwargs()))
        return self._account_api

    @property
    def trade_api(self) -> Any:
        if self._trade_api is None:
            from okx.Trade import TradeAPI

            self._trade_api = self._configure_private_api(TradeAPI(**self._private_kwargs()))
        return self._trade_api

    @property
    def market_api(self) -> Any:
        if self._market_api is None:
            from okx.MarketData import MarketAPI

            self._market_api = MarketAPI(**self._public_kwargs())
        return self._market_api

    @property
    def public_api(self) -> Any:
        if self._public_api is None:
            from okx.PublicData import PublicAPI

            self._public_api = PublicAPI(**self._public_kwargs())
        return self._public_api

    def _public_kwargs(self) -> dict[str, Any]:
        proxy = okx_proxy_url()
        return {
            "flag": okx_sdk_flag_for_mode(self.mode),
            "domain": OKX_DOMAIN,
            "debug": False,
            "proxy": proxy,
        }

    def _private_kwargs(self) -> dict[str, Any]:
        creds = settings.get_okx_credentials(self.mode)
        if not creds.get("api_key") or not creds.get("api_secret"):
            raise ConfigError("OKX API credentials are not configured")
        if not creds.get("passphrase"):
            raise ConfigError("OKX API passphrase is not configured")
        proxy = okx_proxy_url()
        return {
            "api_key": creds.get("api_key", ""),
            "api_secret_key": creds.get("api_secret", ""),
            "passphrase": creds.get("passphrase", ""),
            "flag": okx_sdk_flag_for_mode(self.mode),
            "domain": OKX_DOMAIN,
            "debug": False,
            "proxy": proxy,
        }

    def _configure_private_api(self, api: Any) -> Any:
        # python-okx 0.4.x accepts a deprecated use_server_time argument but
        # leaves the instance flag disabled. Set it explicitly and override the
        # timestamp source with a cached OKX server-time offset so private
        # signed calls are not invalidated by host clock drift.
        api.use_server_time = True
        api._get_timestamp = lambda: self._server_timestamp(api)
        return api

    def _server_timestamp(self, api: Any) -> str:
        now_monotonic = time.monotonic()
        with self._server_time_lock:
            offset_is_stale = (
                self._server_time_offset_ms is None
                or now_monotonic - self._server_time_synced_at > OKX_SERVER_TIME_SYNC_TTL_SECONDS
            )
            if offset_is_stale:
                try:
                    response = api.get(f"{OKX_DOMAIN}{OKX_SERVER_TIME_PATH}")
                    response.raise_for_status()
                    payload = response.json()
                    rows = payload.get("data") if isinstance(payload, Mapping) else None
                    server_ms = int((rows or [{}])[0].get("ts"))
                    local_ms = int(time.time() * 1000)
                    self._server_time_offset_ms = server_ms - local_ms
                    self._server_time_synced_at = now_monotonic
                    if abs(self._server_time_offset_ms) > 5_000:
                        logger.warning(
                            "OKX SDK server time offset is high",
                            mode=self.mode,
                            offset_ms=self._server_time_offset_ms,
                        )
                except Exception as exc:
                    if self._server_time_offset_ms is None:
                        logger.warning(
                            "OKX SDK server time sync failed; falling back to local clock",
                            mode=self.mode,
                            error=safe_error_text(exc),
                        )
                        return _timestamp_from_epoch_ms(int(time.time() * 1000))
                    logger.warning(
                        "OKX SDK server time refresh failed; reusing cached offset",
                        mode=self.mode,
                        error=safe_error_text(exc),
                    )

            return _timestamp_from_epoch_ms(int(time.time() * 1000) + int(self._server_time_offset_ms or 0))

    async def _call_sdk(
        self,
        api_getter: Callable[[], Any],
        method_name: str,
        *,
        check_data_code: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        def _sync() -> dict[str, Any]:
            api = api_getter()
            method = getattr(api, method_name)
            result = method(**kwargs)
            raise_if_okx_error(
                result,
                fallback=f"OKX SDK {method_name} failed",
                check_data_code=check_data_code,
            )
            return dict(result)

        return await asyncio.to_thread(_sync)

    async def load_time_difference(self) -> int:
        return 0

    async def close(self) -> None:
        for api in (self._account_api, self._trade_api, self._market_api, self._public_api):
            closer = getattr(api, "close", None)
            if callable(closer):
                try:
                    await asyncio.to_thread(closer)
                except Exception as exc:
                    logger.debug("OKX SDK client close failed", error=safe_error_text(exc))

    def set_sandbox_mode(self, _enabled: bool) -> None:
        return None

    def parse_markets(self, instruments: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [_instrument_to_market(row) for row in instruments if isinstance(row, Mapping)]

    def set_markets(self, markets: list[dict[str, Any]] | dict[str, dict[str, Any]]) -> None:
        iterable = markets.values() if isinstance(markets, dict) else markets
        self.markets = {}
        self.markets_by_id = {}
        for market in iterable:
            if not isinstance(market, dict):
                continue
            symbol = str(market.get("symbol") or "").strip()
            inst_id = str(market.get("id") or (market.get("info") or {}).get("instId") or "").strip().upper()
            if symbol:
                self.markets[symbol] = market
            if inst_id:
                self.markets[inst_id] = market
                self.markets_by_id.setdefault(inst_id, []).append(market)

    def market(self, symbol: str) -> dict[str, Any]:
        key = str(symbol or "").strip()
        if key in self.markets:
            return self.markets[key]
        inst_id = normalize_swap_inst_id(key, required=False)
        if inst_id and inst_id in self.markets:
            return self.markets[inst_id]
        normalized = normalize_trading_symbol(key)
        if normalized:
            ccxt_symbol = f"{normalized}:USDT"
            if ccxt_symbol in self.markets:
                return self.markets[ccxt_symbol]
            inst_id = okx_inst_id_from_symbol(normalized)
            if inst_id and inst_id in self.markets:
                return self.markets[inst_id]
        raise ExchangeAPIError(f"OKX SDK market is not loaded: {symbol}")

    def amount_to_precision(self, symbol: str, amount: Any) -> str:
        return self._number_to_step(symbol, amount, "amount")

    def price_to_precision(self, symbol: str, price: Any) -> str:
        return self._number_to_step(symbol, price, "price")

    def _number_to_step(self, symbol: str, value: Any, kind: str) -> str:
        try:
            market = self.market(symbol)
        except Exception:
            return _format_number(value)
        precision = market.get("precision") if isinstance(market.get("precision"), dict) else {}
        step = _safe_float(precision.get(kind), 0.0)
        if step <= 0:
            return _format_number(value)
        try:
            decimal_value = Decimal(str(value))
            decimal_step = Decimal(str(step))
            units = (decimal_value / decimal_step).to_integral_value(rounding=ROUND_DOWN)
            rounded = units * decimal_step
            if rounded <= 0 and decimal_value > 0:
                rounded = (decimal_value / decimal_step).to_integral_value(
                    rounding=ROUND_CEILING
                ) * decimal_step
            return format(rounded.normalize(), "f")
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return _format_number(value)

    async def publicGetPublicInstruments(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params)
        return await self._call_sdk(
            lambda: self.public_api,
            "get_instruments",
            instType=OKX_SWAP_INST_TYPE,
            uly=str(params.get("uly") or ""),
            instId=str(params.get("instId") or ""),
            instFamily=str(params.get("instFamily") or ""),
        )

    async def publicGetMarketTicker(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params, require_inst_id=True)
        return await self._call_sdk(
            lambda: self.market_api,
            "get_ticker",
            instId=str(params["instId"]),
        )

    async def publicGetMarketTickers(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params)
        return await self._call_sdk(
            lambda: self.market_api,
            "get_tickers",
            instType=OKX_SWAP_INST_TYPE,
            uly=str(params.get("uly") or ""),
            instFamily=str(params.get("instFamily") or ""),
        )

    async def privateGetAccountBalance(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return await self._call_sdk(
            lambda: self.account_api,
            "get_account_balance",
            ccy=str((params or {}).get("ccy") or ""),
        )

    async def privateGetAccountFeeRates(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params)
        return await self._call_sdk(
            lambda: self.account_api,
            "get_fee_rates",
            instType=OKX_SWAP_INST_TYPE,
            instId="",
            uly=str(params.get("uly") or ""),
            category=str(params.get("category") or ""),
            instFamily=str(params.get("instFamily") or ""),
        )

    async def privateGetAccountPositions(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params)
        return await self._call_sdk(
            lambda: self.account_api,
            "get_positions",
            instType=OKX_SWAP_INST_TYPE,
            instId=str(params.get("instId") or ""),
            posId=str(params.get("posId") or ""),
        )

    async def privateGetAccountPositionsHistory(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params)
        return await self._call_sdk(
            lambda: self.account_api,
            "get_positions_history",
            instType=OKX_SWAP_INST_TYPE,
            instId=str(params.get("instId") or ""),
            mgnMode=str(params.get("mgnMode") or ""),
            type=str(params.get("type") or ""),
            posId=str(params.get("posId") or ""),
            after=str(params.get("after") or ""),
            before=str(params.get("before") or ""),
            limit=str(params.get("limit") or ""),
        )

    async def privateGetAccountBills(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params)
        return await self._call_sdk(
            lambda: self.account_api,
            "get_account_bills",
            instType=OKX_SWAP_INST_TYPE,
            ccy=str(params.get("ccy") or ""),
            mgnMode=str(params.get("mgnMode") or ""),
            ctType=str(params.get("ctType") or ""),
            type=str(params.get("type") or ""),
            subType=str(params.get("subType") or ""),
            after=str(params.get("after") or ""),
            before=str(params.get("before") or ""),
            limit=str(params.get("limit") or ""),
        )

    async def privateGetAccountBillsArchive(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params)
        return await self._call_sdk(
            lambda: self.account_api,
            "get_account_bills_archive",
            instType=OKX_SWAP_INST_TYPE,
            ccy=str(params.get("ccy") or ""),
            mgnMode=str(params.get("mgnMode") or ""),
            ctType=str(params.get("ctType") or ""),
            type=str(params.get("type") or ""),
            subType=str(params.get("subType") or ""),
            after=str(params.get("after") or ""),
            before=str(params.get("before") or ""),
            limit=str(params.get("limit") or ""),
            begin=str(params.get("begin") or ""),
            end=str(params.get("end") or ""),
        )

    async def privateGetTradeFillsHistory(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params)
        return await self._call_sdk(
            lambda: self.trade_api,
            "get_fills_history",
            instType=OKX_SWAP_INST_TYPE,
            instId=str(params.get("instId") or ""),
            ordId=str(params.get("ordId") or ""),
            after=str(params.get("after") or ""),
            before=str(params.get("before") or ""),
            limit=str(params.get("limit") or ""),
            uly=str(params.get("uly") or ""),
            instFamily=str(params.get("instFamily") or ""),
        )

    async def privateGetTradeOrder(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params, require_inst_id=True)
        return await self._call_sdk(
            lambda: self.trade_api,
            "get_order",
            instId=str(params["instId"]),
            ordId=str(params.get("ordId") or ""),
            clOrdId=str(params.get("clOrdId") or ""),
        )

    async def privateGetTradeOrdersPending(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params)
        return await self._call_sdk(
            lambda: self.trade_api,
            "get_order_list",
            instType=OKX_SWAP_INST_TYPE,
            instId=str(params.get("instId") or ""),
            ordType=str(params.get("ordType") or ""),
            state=str(params.get("state") or ""),
            after=str(params.get("after") or ""),
            before=str(params.get("before") or ""),
            limit=str(params.get("limit") or ""),
            uly=str(params.get("uly") or ""),
            instFamily=str(params.get("instFamily") or ""),
        )

    async def privateGetTradeOrdersHistory(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return await self._trade_orders_history(params, archive=False)

    async def privateGetTradeOrdersHistoryArchive(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return await self._trade_orders_history(params, archive=True)

    async def _trade_orders_history(
        self,
        params: Mapping[str, Any],
        *,
        archive: bool,
    ) -> dict[str, Any]:
        params = _swap_params(params)
        inst_id = str(params.get("instId") or "")
        order_id = str(params.get("ordId") or "")
        if order_id and inst_id:
            try:
                return await self.privateGetTradeOrder({"instId": inst_id, "ordId": order_id})
            except Exception as exc:
                logger.debug(
                    "OKX SDK order detail fallback to order history",
                    inst_id=inst_id,
                    order_id=order_id,
                    error=safe_error_text(exc),
                )
        method_name = "get_orders_history_archive" if archive else "get_orders_history"
        return await self._call_sdk(
            lambda: self.trade_api,
            method_name,
            instType=OKX_SWAP_INST_TYPE,
            instId=inst_id,
            ordType=str(params.get("ordType") or ""),
            state=str(params.get("state") or ""),
            after=str(params.get("after") or ""),
            before=str(params.get("before") or ""),
            begin=str(params.get("begin") or ""),
            end=str(params.get("end") or ""),
            limit=str(params.get("limit") or ""),
            uly=str(params.get("uly") or ""),
            instFamily=str(params.get("instFamily") or ""),
        )

    async def privateGetTradeOrdersAlgoPending(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params)
        return await self._call_sdk(
            lambda: self.trade_api,
            "order_algos_list",
            ordType=str(params.get("ordType") or ""),
            instType=OKX_SWAP_INST_TYPE,
            instId=str(params.get("instId") or ""),
            after=str(params.get("after") or ""),
            before=str(params.get("before") or ""),
            limit=str(params.get("limit") or ""),
        )

    async def privatePostTradeCancelOrder(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = _swap_params(params, require_inst_id=True)
        return await self._call_sdk(
            lambda: self.trade_api,
            "cancel_order",
            instId=str(params["instId"]),
            ordId=str(params.get("ordId") or ""),
            clOrdId=str(params.get("clOrdId") or ""),
        )

    async def privatePostTradeClosePosition(self, params: Mapping[str, Any]) -> dict[str, Any]:
        inst_id = normalize_swap_inst_id((params or {}).get("instId"), required=True)
        return await self._call_sdk(
            lambda: self.trade_api,
            "close_positions",
            check_data_code=True,
            instId=inst_id,
            mgnMode=str((params or {}).get("mgnMode") or OKX_CROSS_MARGIN_MODE),
            posSide=str((params or {}).get("posSide") or ""),
            ccy=str((params or {}).get("ccy") or ""),
            autoCxl=str((params or {}).get("autoCxl") or ""),
            clOrdId=str((params or {}).get("clOrdId") or ""),
            tag=str((params or {}).get("tag") or ""),
        )

    async def privatePostTradeOrder(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = dict(params or {})
        inst_id = normalize_swap_inst_id(params.get("instId"), required=True)
        return await self._call_sdk(
            lambda: self.trade_api,
            "place_order",
            check_data_code=True,
            instId=inst_id,
            tdMode=str(params.get("tdMode") or OKX_CROSS_MARGIN_MODE),
            side=str(params.get("side") or ""),
            ordType=str(params.get("ordType") or "market"),
            sz=str(params.get("sz") or ""),
            ccy=str(params.get("ccy") or ""),
            clOrdId=str(params.get("clOrdId") or ""),
            tag=str(params.get("tag") or ""),
            posSide=str(params.get("posSide") or params.get("positionSide") or ""),
            px=str(params.get("px") or ""),
            reduceOnly=_normalize_bool_text(params.get("reduceOnly")),
            tgtCcy=str(params.get("tgtCcy") or ""),
            stpMode=str(params.get("stpMode") or ""),
            attachAlgoOrds=params.get("attachAlgoOrds"),
        )

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = dict(params or {})
        inst_id = normalize_swap_inst_id(symbol, field="symbol", required=True)
        payload: dict[str, Any] = {
            "instId": inst_id,
            "tdMode": str(params.get("tdMode") or params.get("marginMode") or OKX_CROSS_MARGIN_MODE),
            "side": side,
            "ordType": order_type,
            "sz": _format_number(amount),
        }
        if price is not None and _safe_float(price, 0.0) > 0:
            payload["px"] = _format_number(price)
        if params.get("positionSide"):
            payload["posSide"] = params.get("positionSide")
        if params.get("posSide"):
            payload["posSide"] = params.get("posSide")
        if "reduceOnly" in params:
            payload["reduceOnly"] = _normalize_bool_text(params.get("reduceOnly"))
        if params.get("attachAlgoOrds") is not None:
            payload["attachAlgoOrds"] = params.get("attachAlgoOrds")
        response = await self.privatePostTradeOrder(payload)
        return self._order_submit_to_ccxt_shape(response, payload)

    def _order_submit_to_ccxt_shape(
        self,
        response: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        data = response.get("data") if isinstance(response, Mapping) else None
        first = data[0] if isinstance(data, list) and data else {}
        if not isinstance(first, Mapping):
            first = {}
        order_id = str(first.get("ordId") or first.get("clOrdId") or "").strip()
        return {
            "id": order_id,
            "clientOrderId": str(first.get("clOrdId") or payload.get("clOrdId") or ""),
            "symbol": str(payload.get("instId") or ""),
            "type": str(payload.get("ordType") or "").lower(),
            "side": str(payload.get("side") or "").lower(),
            "status": "open",
            "amount": _safe_float(payload.get("sz"), 0.0),
            "filled": 0.0,
            "remaining": _safe_float(payload.get("sz"), 0.0),
            "price": _safe_float(payload.get("px"), 0.0),
            "average": None,
            "fee": {},
            "info": dict(first) if first else dict(response),
        }

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> list[list[float]]:
        inst_id = normalize_swap_inst_id(symbol, field="symbol", required=True)
        bar = _timeframe_to_okx_bar(timeframe)
        response = await self._call_sdk(
            lambda: self.market_api,
            "get_candlesticks",
            instId=inst_id,
            bar=bar,
            limit=str(max(1, int(limit or 100))),
        )
        rows = response.get("data") if isinstance(response, Mapping) else []
        result: list[list[float]] = []
        for row in rows or []:
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                continue
            result.append(
                [
                    _safe_float(row[0]),
                    _safe_float(row[1]),
                    _safe_float(row[2]),
                    _safe_float(row[3]),
                    _safe_float(row[4]),
                    _safe_float(row[5]),
                ]
            )
        return list(reversed(result))

    async def fetch_funding_rate(self, symbol: str) -> dict[str, Any]:
        inst_id = normalize_swap_inst_id(symbol, field="symbol", required=True)
        response = await self._call_sdk(lambda: self.public_api, "get_funding_rate", instId=inst_id)
        rows = response.get("data") if isinstance(response, Mapping) else []
        row = rows[0] if isinstance(rows, list) and rows else {}
        if not isinstance(row, Mapping):
            row = {}
        return {
            "symbol": symbol_from_okx_inst_id(inst_id),
            "fundingRate": _safe_float(row.get("fundingRate"), 0.0),
            "fundingDatetime": row.get("fundingTime"),
            "nextFundingDatetime": row.get("nextFundingTime") or row.get("fundingTime"),
            "info": dict(row),
        }

    async def fetch_open_interest(self, symbol: str) -> dict[str, Any]:
        inst_id = normalize_swap_inst_id(symbol, field="symbol", required=True)
        response = await self._call_sdk(
            lambda: self.public_api,
            "get_open_interest",
            instType=OKX_SWAP_INST_TYPE,
            instId=inst_id,
        )
        rows = response.get("data") if isinstance(response, Mapping) else []
        row = rows[0] if isinstance(rows, list) and rows else {}
        if not isinstance(row, Mapping):
            row = {}
        amount = _safe_float(row.get("oi"), 0.0)
        value = _safe_float(row.get("oiCcy") or row.get("oiUsd"), 0.0)
        return {
            "symbol": symbol_from_okx_inst_id(inst_id),
            "openInterestAmount": amount,
            "openInterest": amount,
            "openInterestValue": value,
            "info": dict(row),
        }

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        inst_id = normalize_swap_inst_id(symbol, field="symbol", required=True)
        response = await self._call_sdk(
            lambda: self.market_api,
            "get_orderbook",
            instId=inst_id,
            sz=str(max(1, int(limit or 20))),
        )
        rows = response.get("data") if isinstance(response, Mapping) else []
        row = rows[0] if isinstance(rows, list) and rows else {}
        if not isinstance(row, Mapping):
            row = {}
        return {
            "symbol": symbol_from_okx_inst_id(inst_id),
            "bids": [[_safe_float(px), _safe_float(sz)] for px, sz, *_ in row.get("bids", [])],
            "asks": [[_safe_float(px), _safe_float(sz)] for px, sz, *_ in row.get("asks", [])],
            "timestamp": _safe_float(row.get("ts"), 0.0),
            "info": dict(row),
        }

    async def fetch_balance(self) -> dict[str, Any]:
        return self._balance_response_to_ccxt_shape(await self.privateGetAccountBalance({}), "USDT")

    async def fetch_leverage(self, symbol: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        inst_id = normalize_swap_inst_id(symbol, field="symbol", required=True)
        response = await self._call_sdk(
            lambda: self.account_api,
            "get_leverage",
            mgnMode=str(params.get("mgnMode") or params.get("tdMode") or OKX_CROSS_MARGIN_MODE),
            instId=inst_id,
        )
        rows = response.get("data") if isinstance(response, Mapping) else []
        lever = 0.0
        for row in rows or []:
            if isinstance(row, Mapping):
                lever = _safe_float(row.get("lever"), lever)
        return {
            "longLeverage": lever,
            "shortLeverage": lever,
            "info": list(rows or []),
        }

    async def set_leverage(
        self,
        leverage: float,
        symbol: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = dict(params or {})
        inst_id = normalize_swap_inst_id(symbol, field="symbol", required=True)
        return await self._call_sdk(
            lambda: self.account_api,
            "set_leverage",
            check_data_code=True,
            lever=str(int(max(1, round(float(leverage or 1))))),
            mgnMode=str(params.get("mgnMode") or params.get("tdMode") or OKX_CROSS_MARGIN_MODE),
            instId=inst_id,
            posSide=str(params.get("posSide") or params.get("positionSide") or ""),
        )

    async def fetch_market_leverage_tiers(self, symbol: str) -> list[dict[str, Any]]:
        inst_id = normalize_swap_inst_id(symbol, field="symbol", required=True)
        response = await self._call_sdk(
            lambda: self.public_api,
            "get_position_tiers",
            instType=OKX_SWAP_INST_TYPE,
            tdMode=OKX_CROSS_MARGIN_MODE,
            instId=inst_id,
        )
        rows = response.get("data") if isinstance(response, Mapping) else []
        tiers: list[dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, Mapping):
                continue
            max_lever = _safe_float(row.get("maxLever"), 0.0)
            tiers.append({**dict(row), "maxLeverage": max_lever, "info": dict(row)})
        return tiers

    @staticmethod
    def _balance_response_to_ccxt_shape(response: Mapping[str, Any], asset: str) -> dict[str, Any]:
        raw_detail: dict[str, Any] = {}
        data = response.get("data") if isinstance(response, Mapping) else None
        for item in data or []:
            if not isinstance(item, Mapping):
                continue
            for detail in item.get("details", []) or []:
                if isinstance(detail, Mapping) and detail.get("ccy") == asset:
                    raw_detail = dict(detail)
                    break
            if raw_detail:
                break
        cash = _safe_float(raw_detail.get("cashBal"), 0.0)
        equity = _safe_float(raw_detail.get("eq"), cash)
        used = _safe_float(raw_detail.get("frozenBal"), 0.0)
        available = (
            _safe_float(raw_detail.get("availBal"), 0.0)
            or _safe_float(raw_detail.get("availEq"), 0.0)
            or _safe_float(raw_detail.get("disEq"), 0.0)
            or max(equity - used, 0.0)
        )
        total = equity if equity > 0 else cash
        return {
            asset: {"free": available, "used": used, "total": total},
            "info": {"data": list(data or [])},
        }
