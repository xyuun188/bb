"""
OKX V5 WebSocket client for real-time market data.
Subscribes to ticker and K-line channels for configured symbols.

OKX V5 WebSocket Docs: https://www.okx.com/docs-v5/en/#websocket-api
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import structlog

from config.settings import settings
from core.exceptions import WebSocketConnectionError
from core.safe_output import safe_error_text
from core.symbols import normalize_trading_symbol
from data_feed.okx_ticker_volume import okx_swap_volume_fields, safe_float
from services.okx_perpetual_sdk import (
    OKX_WS_BUSINESS_URL,
    OKX_WS_DEMO_URL,
    OKX_WS_PUBLIC_URL,
    OkxPublicWebSocketSdkStream,
)

logger = structlog.get_logger(__name__)

# OKX WebSocket endpoints
# Public URL: tickers, instruments, funding-rate, etc.
WS_PUBLIC_URL = OKX_WS_PUBLIC_URL
# Business URL: candlesticks, mark-price-candle, index-candle (moved from public in June 2023)
WS_BUSINESS_URL = OKX_WS_BUSINESS_URL
# Demo URL: blocked in some regions, not used
WS_DEMO_URL = OKX_WS_DEMO_URL
WS_RECEIVE_TIMEOUT_SECONDS = 30.0
WS_MAX_CONSECUTIVE_RECEIVE_TIMEOUTS = 2
WS_TICKER_STALE_RECONNECT_SECONDS = 90.0
WS_RECONNECT_DELAY_SECONDS = 5.0
WS_TICKER_MIN_PROCESS_INTERVAL_SECONDS = 0.5
WS_SUBSCRIPTION_BATCH_SIZE = 100
WS_SUBSCRIPTION_MIN_INTERVAL_SECONDS = 0.5


class OKXWebSocketClient:
    """Async WebSocket client for OKX V5 public channels.

    Handles reconnection, heartbeats, and message routing.
    Public market data uses ws.okx.com regardless of demo/live mode.
    """

    def __init__(self) -> None:
        self._ws_url = WS_PUBLIC_URL  # Public data always from ws.okx.com
        self._ws: Any = None
        self.latest_orderbooks: dict[str, dict] = {}
        self.latest_mark_prices: dict[str, dict] = {}
        self.latest_index_prices: dict[str, dict] = {}
        self._running = False
        self._ticker_callbacks: list[Callable[..., Any]] = []
        self._kline_callbacks: list[Callable[..., Any]] = []
        self._latest_tickers: dict[str, dict[str, Any]] = {}
        self._latest_klines: dict[str, list[dict]] = {}
        self._message_count = 0
        self._last_message_time = 0.0
        self._last_ticker_time = 0.0
        self._connected_at = 0.0
        self._reconnect_count = 0
        self._last_processed_ticker_time: dict[str, float] = {}
        self._throttled_ticker_updates = 0
        self._subscription_request_lock = asyncio.Lock()
        self._last_subscription_request_at: float | None = None
        self._subscription_request_count = 0
        self._subscription_acknowledged_count = 0
        self._subscription_error_count = 0
        self._subscription_unattributed_error_count = 0
        self._subscription_expected_args: set[tuple[str, str]] = set()
        self._subscription_acknowledged_args: set[tuple[str, str]] = set()
        self._subscription_rejected_args: set[tuple[str, str]] = set()
        self._channel_message_counts: dict[str, int] = {}

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running

    @property
    def latest_tickers(self) -> dict[str, dict]:
        return self._latest_tickers

    def on_ticker(self, callback: Callable) -> None:
        """Register a callback for ticker updates: func(symbol, data)."""
        self._ticker_callbacks.append(callback)

    def on_kline(self, callback: Callable) -> None:
        """Register a callback for kline updates: func(symbol, timeframe, candles)."""
        self._kline_callbacks.append(callback)

    def _to_ws_inst_id(self, symbol: str) -> str:
        """Convert CCXT symbol format to OKX WS perpetual swap instrument ID.
        BTC/USDT -> BTC-USDT-SWAP (always use perpetual swap for accurate pricing)
        """
        base = symbol.split("/")[0].split(":")[0].split("-")[0]
        return f"{base}-USDT-SWAP"

    def _to_ws_index_id(self, symbol: str) -> str:
        """Convert a trading symbol to its OKX index instrument ID."""
        base = symbol.split("/")[0].split(":")[0].split("-")[0]
        return f"{base}-USDT"

    def _subscription_args(self, symbols: list[str]) -> list[dict[str, str]]:
        channels: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for symbol in symbols:
            swap_id = self._to_ws_inst_id(symbol)
            for channel, inst_id in (
                ("tickers", swap_id),
                ("books5", swap_id),
                ("mark-price", swap_id),
                ("index-tickers", self._to_ws_index_id(symbol)),
            ):
                key = (channel, inst_id)
                if key not in seen:
                    channels.append({"channel": channel, "instId": inst_id})
                    seen.add(key)
        return channels

    @staticmethod
    def _subscription_argument_key(argument: Any) -> tuple[str, str] | None:
        if not isinstance(argument, dict):
            return None
        channel = str(argument.get("channel") or "").strip()
        inst_id = str(argument.get("instId") or "").strip()
        return (channel, inst_id) if channel and inst_id else None

    def _reset_subscription_coverage(
        self,
        channels: list[dict[str, str]],
    ) -> None:
        self._subscription_expected_args = {
            key
            for channel in channels
            if (key := self._subscription_argument_key(channel)) is not None
        }
        self._subscription_acknowledged_args.clear()
        self._subscription_rejected_args.clear()
        self._subscription_unattributed_error_count = 0

    async def connect(self) -> None:
        """Establish WebSocket connection and subscribe to channels."""
        self._running = True
        logger.info("connecting to OKX WebSocket", url=self._ws_url)

        try:
            stream = OkxPublicWebSocketSdkStream(self._ws_url)
            await stream.connect()
            self._ws = stream
            self._connected_at = time.time()
            self._last_ticker_time = 0.0
            logger.info("OKX WebSocket connected")
            await self._subscribe()
        except Exception as e:
            error_text = safe_error_text(e)
            logger.error("OKX WebSocket connection failed", error=error_text)
            self._ws = None
            raise WebSocketConnectionError(f"Failed to connect: {error_text}") from e

    async def _subscribe(self) -> None:
        """Subscribe to ticker, orderbook, and mark-price channels for configured symbols."""
        symbols = getattr(self, "_subscribe_symbols", settings.symbols)
        
        # Build all channels
        all_channels = self._subscription_args(symbols)
        self._reset_subscription_coverage(all_channels)

        request_count = await self._send_subscription_operation("subscribe", all_channels)
        logger.info(
            "all subscriptions sent",
            symbols=len(symbols),
            total_channels=len(all_channels),
            request_count=request_count,
        )

    async def _send_subscription_request(
        self,
        operation: str,
        channels: list[dict[str, str]],
    ) -> None:
        async with self._subscription_request_lock:
            if self._last_subscription_request_at is not None:
                delay = (
                    self._last_subscription_request_at
                    + WS_SUBSCRIPTION_MIN_INTERVAL_SECONDS
                    - time.monotonic()
                )
                if delay > 0:
                    await asyncio.sleep(delay)
            if self._ws is None or not self._running:
                raise WebSocketConnectionError(
                    "Cannot send a subscription request while WebSocket is disconnected."
                )
            await self._ws.send(
                json.dumps({"op": operation, "args": channels})
            )
            self._last_subscription_request_at = time.monotonic()
            self._subscription_request_count += 1

    async def _send_subscription_operation(
        self,
        operation: str,
        channels: list[dict[str, str]],
    ) -> int:
        keys = {
            key
            for channel in channels
            if (key := self._subscription_argument_key(channel)) is not None
        }
        if operation == "subscribe":
            self._subscription_expected_args.update(keys)
            self._subscription_acknowledged_args.difference_update(keys)
            self._subscription_rejected_args.difference_update(keys)
        elif operation == "unsubscribe":
            self._subscription_expected_args.difference_update(keys)
            self._subscription_acknowledged_args.difference_update(keys)
            self._subscription_rejected_args.difference_update(keys)

        requests = 0
        for offset in range(0, len(channels), WS_SUBSCRIPTION_BATCH_SIZE):
            batch = channels[offset : offset + WS_SUBSCRIPTION_BATCH_SIZE]
            await self._send_subscription_request(operation, batch)
            requests += 1
        return requests

    async def _handle_message(self, raw: str) -> None:
        """Parse and route incoming WebSocket messages."""
        received_at = datetime.now(UTC).isoformat()
        self._message_count += 1
        self._last_message_time = time.time()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        if "event" in data:
            event = str(data.get("event") or "")
            code = str(data.get("code") or "")
            argument_key = self._subscription_argument_key(data.get("arg"))
            if event == "error" or (code and code != "0"):
                self._subscription_error_count += 1
                if argument_key in self._subscription_expected_args:
                    self._subscription_rejected_args.add(argument_key)
                    self._subscription_acknowledged_args.discard(argument_key)
                else:
                    self._subscription_unattributed_error_count += 1
                logger.error(
                    "OKX WebSocket operation rejected",
                    operation=event or "unknown",
                    code=code or "missing",
                    message=str(data.get("msg") or "")[:240],
                    arg=data.get("arg"),
                )
            elif event in {"subscribe", "unsubscribe"}:
                self._subscription_acknowledged_count += 1
                if (
                    event == "subscribe"
                    and argument_key in self._subscription_expected_args
                ):
                    self._subscription_acknowledged_args.add(argument_key)
                logger.debug(
                    "OKX WebSocket operation acknowledged",
                    operation=event,
                    arg=data.get("arg"),
                )
            else:
                logger.debug("ws event", ws_event=event, msg=data.get("msg", ""))
            return

        if "arg" not in data or "data" not in data:
            return

        channel = data["arg"].get("channel", "")
        inst_id = data["arg"].get("instId", "")
        if channel:
            self._channel_message_counts[channel] = (
                self._channel_message_counts.get(channel, 0) + 1
            )
        
        # Debug: log non-ticker channels
        if channel != "tickers":
            if channel == "books5":
                # books5 is emitted continuously for every subscribed symbol;
                # keep per-message diagnostics out of the production info log.
                logger.debug(
                    "books5 message received",
                    inst_id=inst_id,
                    data_count=len(data.get("data", [])),
                )

        if channel == "tickers":
            if data["data"]:
                self._last_ticker_time = time.time()
            for ticker in data["data"]:
                symbol = inst_id.replace("-SWAP", "").replace("-", "/")
                now = time.monotonic()
                last_processed = self._last_processed_ticker_time.get(symbol, 0.0)
                if (
                    now - last_processed
                    < WS_TICKER_MIN_PROCESS_INTERVAL_SECONDS
                ):
                    self._throttled_ticker_updates += 1
                    continue
                self._last_processed_ticker_time[symbol] = now
                last = safe_float(ticker.get("last"), 0.0)
                open24h = safe_float(ticker.get("open24h"), 0.0)
                change_pct = ((last - open24h) / open24h * 100) if open24h else 0
                volume_fields = okx_swap_volume_fields(ticker, last)
                parsed = {
                    "symbol": symbol,
                    "inst_id": inst_id,
                    "inst_type": "SWAP",
                    "last_price": last,
                    "bid": safe_float(ticker.get("bidPx"), 0.0),
                    "ask": safe_float(ticker.get("askPx"), 0.0),
                    "high_24h": safe_float(ticker.get("high24h"), 0.0),
                    "low_24h": safe_float(ticker.get("low24h"), 0.0),
                    "volume_24h": volume_fields["volume_24h_base"]
                    or volume_fields["volume_24h_contracts"],
                    **volume_fields,
                    "change_24h_pct": change_pct,
                    "timestamp": int(safe_float(ticker.get("ts"), 0.0)),
                    "source_timestamp_ms": int(safe_float(ticker.get("ts"), 0.0)),
                    "source_sequence": ticker.get("seqId"),
                    "received_at": received_at,
                    "source": "websocket",
                    "source_endpoint": "okx_ws_public",
                    "source_channel": channel,
                    "info": {"instId": inst_id, "instType": "SWAP", **dict(ticker)},
                }
                self._latest_tickers[symbol] = parsed
                for cb in self._ticker_callbacks:
                    try:
                        cb(symbol, parsed)
                    except Exception as exc:
                        logger.debug(
                            "ticker callback failed",
                            symbol=symbol,
                              error=safe_error_text(exc),
                          )

        elif channel == "books5":
            # Order book depth updates
            for book in data["data"]:
                symbol = inst_id.replace("-SWAP", "").replace("-", "/")
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                # OKX swap quantities are contracts. Keep price-weighted
                # contract depth until the instrument spec converts it to USDT.
                bid_depth_native = sum(safe_float(q, 0.0) for p, q, *_ in bids)
                ask_depth_native = sum(safe_float(q, 0.0) for p, q, *_ in asks)
                bid_depth_quote_contracts = sum(
                    safe_float(p, 0.0) * safe_float(q, 0.0) for p, q, *_ in bids
                )
                ask_depth_quote_contracts = sum(
                    safe_float(p, 0.0) * safe_float(q, 0.0) for p, q, *_ in asks
                )
                timestamp_ms = int(safe_float(book.get("ts"), 0.0))
                best_bid = safe_float(bids[0][0], 0.0) if bids else 0.0
                best_ask = safe_float(asks[0][0], 0.0) if asks else 0.0
                orderbook_fact = {
                    "inst_id": inst_id,
                    "inst_type": "SWAP",
                    "source_endpoint": "okx_ws_public",
                    "source_channel": "books5",
                    "source_timestamp_ms": timestamp_ms,
                    "received_at": received_at,
                    "bid": best_bid,
                    "ask": best_ask,
                    "bid_depth_quote_contracts": bid_depth_quote_contracts,
                    "ask_depth_quote_contracts": ask_depth_quote_contracts,
                }
                orderbook_data = {
                    "symbol": symbol,
                    "inst_id": inst_id,
                    "inst_type": "SWAP",
                    "bid_depth_native": bid_depth_native,
                    "ask_depth_native": ask_depth_native,
                    "bid_depth_usdt": 0.0,
                    "ask_depth_usdt": 0.0,
                    "bid_depth_quote_contracts": bid_depth_quote_contracts,
                    "ask_depth_quote_contracts": ask_depth_quote_contracts,
                    "orderbook_bid_depth": 0.0,
                    "orderbook_ask_depth": 0.0,
                    "timestamp": timestamp_ms,
                    "source_timestamp_ms": timestamp_ms,
                    "received_at": received_at,
                    "source": "websocket",
                    "source_endpoint": "okx_ws_public",
                    "source_channel": "books5",
                    "orderbook_fact": orderbook_fact,
                }
                # Store in cache for data service to pick up
                self.latest_orderbooks[symbol] = orderbook_data

        elif channel == "mark-price":
            # Mark price updates
            for mark in data["data"]:
                symbol = inst_id.replace("-SWAP", "").replace("-", "/")
                mark_price = safe_float(mark.get("markPx"), 0.0)
                timestamp_ms = int(safe_float(mark.get("ts"), 0.0))
                mark_price_data = {
                    "symbol": symbol,
                    "inst_id": inst_id,
                    "inst_type": "SWAP",
                    "mark_price": mark_price,
                    "timestamp": timestamp_ms,
                    "source_timestamp_ms": timestamp_ms,
                    "received_at": received_at,
                    "source": "websocket",
                    "source_endpoint": "okx_ws_public",
                    "source_channel": "mark-price",
                    "mark_price_fact": {
                        "inst_id": inst_id,
                        "inst_type": "SWAP",
                        "source_endpoint": "okx_ws_public",
                        "source_channel": "mark-price",
                        "source_timestamp_ms": timestamp_ms,
                        "received_at": received_at,
                        "price": mark_price,
                    },
                }
                # Store in cache for data service to pick up
                self.latest_mark_prices[symbol] = mark_price_data

        elif channel == "index-tickers":
            # Keep index prices as their own observed fact; never derive them
            # from a swap ticker or mark price.
            for index in data["data"]:
                index_id = str(index.get("instId") or inst_id).upper()
                index_symbol = normalize_trading_symbol(index_id)
                index_price = safe_float(index.get("idxPx"), 0.0)
                timestamp_ms = int(safe_float(index.get("ts"), 0.0))
                self.latest_index_prices[index_symbol] = {
                    "symbol": index_symbol,
                    "inst_id": index_id,
                    "inst_type": "INDEX",
                    "index_price": index_price,
                    "timestamp": timestamp_ms,
                    "source_timestamp_ms": timestamp_ms,
                    "received_at": received_at,
                    "source": "websocket",
                    "source_endpoint": "okx_ws_public",
                    "source_channel": "index-tickers",
                    "index_price_fact": {
                        "inst_id": index_id,
                        "inst_type": "INDEX",
                        "source_endpoint": "okx_ws_public",
                        "source_channel": "index-tickers",
                        "source_timestamp_ms": timestamp_ms,
                        "received_at": received_at,
                        "price": index_price,
                    },
                }

        elif channel and channel.startswith("candle"):
            timeframe = channel.replace("candle", "")
            candles = []
            for candle in data["data"]:
                candles.append(
                    {
                        "open_time": int(candle[0]),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    }
                )
            symbol = inst_id.replace("-SWAP", "").replace("-", "/")
            key = f"{symbol}:{timeframe}"
            self._latest_klines[key] = candles
            for cb in self._kline_callbacks:
                try:
                    cb(symbol, timeframe, candles)
                except Exception as exc:
                    logger.debug(
                        "kline callback failed",
                        symbol=symbol,
                        timeframe=timeframe,
                        error=safe_error_text(exc),
                    )

    async def listen(self) -> None:
        """Consume messages and keep reconnecting until explicitly stopped."""
        consecutive_receive_timeouts = 0
        while self._running:
            if self._ws is None:
                if not await self._reconnect_until_available():
                    return
                consecutive_receive_timeouts = 0
            while self._running and self._ws is not None:
                try:
                    raw = await asyncio.wait_for(
                        self._ws.recv(),
                        timeout=WS_RECEIVE_TIMEOUT_SECONDS,
                    )
                    await self._handle_message(raw)
                    consecutive_receive_timeouts = 0
                    ticker_reference_time = self._last_ticker_time or self._connected_at
                    ticker_age = (
                        time.time() - ticker_reference_time if ticker_reference_time else 0.0
                    )
                    if ticker_age > WS_TICKER_STALE_RECONNECT_SECONDS:
                        logger.warning(
                            "OKX ticker stream stale; reconnecting",
                            ticker_age_seconds=round(ticker_age, 3),
                            stale_after_seconds=WS_TICKER_STALE_RECONNECT_SECONDS,
                            messages_received=self._message_count,
                        )
                        break
                except TimeoutError:
                    consecutive_receive_timeouts += 1
                    if consecutive_receive_timeouts >= WS_MAX_CONSECUTIVE_RECEIVE_TIMEOUTS:
                        logger.warning(
                            "OKX WebSocket receive timeout threshold reached; reconnecting",
                            consecutive_timeouts=consecutive_receive_timeouts,
                            timeout_seconds=WS_RECEIVE_TIMEOUT_SECONDS,
                        )
                        break
                    try:
                        await self._ws.send("ping")
                    except Exception:
                        logger.warning("ping failed, reconnecting...")
                        break
                except Exception as e:
                    if "ConnectionClosed" in type(e).__name__ or hasattr(e, "code"):
                        logger.warning(
                            "websocket closed",
                            code=getattr(e, "code", None),
                            reason=getattr(e, "reason", None),
                        )
                    else:
                        logger.error("unexpected error in listen loop", error=safe_error_text(e))
                    break
            if not self._running:
                return
            await self._close_current_stream()

    async def _reconnect_until_available(self) -> bool:
        """Retry connection setup until it succeeds or shutdown is requested."""

        while self._running:
            logger.info("reconnecting in 5 seconds...")
            await asyncio.sleep(WS_RECONNECT_DELAY_SECONDS)
            if not self._running:
                return False
            self._reconnect_count += 1
            try:
                await self.connect()
            except WebSocketConnectionError as exc:
                logger.warning(
                    "OKX WebSocket reconnect failed; retrying",
                    reconnect_count=self._reconnect_count,
                    error=safe_error_text(exc),
                )
                continue
            return True
        return False

    async def _close_current_stream(self) -> None:
        stale_ws = self._ws
        self._ws = None
        if stale_ws is None:
            return
        try:
            await stale_ws.close()
        except Exception as exc:
            logger.debug(
                "stale OKX WebSocket close failed",
                error=safe_error_text(exc),
            )

    async def subscribe_symbol(self, symbol: str) -> None:
        """Dynamically subscribe to a new symbol's ticker channel."""
        if not self._ws or not self._running:
            logger.warning("cannot subscribe, ws not connected", symbol=symbol)
            return
        channels = self._subscription_args([symbol])
        await self._send_subscription_operation("subscribe", channels)
        current = list(getattr(self, "_subscribe_symbols", settings.symbols))
        normalized = normalize_trading_symbol(symbol)
        if normalized not in {normalize_trading_symbol(item) for item in current}:
            current.append(normalized)
        self._subscribe_symbols = current
        logger.info("subscribed to symbol", symbol=symbol, channels=len(channels))

    async def resubscribe_all(self, symbols: list[str]) -> None:
        """Resubscribe to a new list of symbols (unsub old, sub new)."""
        old_symbols = getattr(self, "_subscribe_symbols", settings.symbols)
        old_ids = self._subscription_args(old_symbols)
        new_ids = self._subscription_args(symbols)
        if not self._ws or not self._running:
            self._subscribe_symbols = symbols
            return
        # Unsubscribe old channels and subscribe replacements within the same
        # per-connection operation budget used during initial startup.
        if old_ids:
            await self._send_subscription_operation("unsubscribe", old_ids)
        # Clear old tickers
        old_symbols = set(getattr(self, "_subscribe_symbols", []))
        for sym in old_symbols:
            self._latest_tickers.pop(sym, None)
            normalized = normalize_trading_symbol(sym)
            self.latest_orderbooks.pop(normalized, None)
            self.latest_mark_prices.pop(normalized, None)
            self.latest_index_prices.pop(normalized, None)
        # Subscribe to new
        if new_ids:
            await self._send_subscription_operation("subscribe", new_ids)
        self._subscribe_symbols = symbols
        logger.info("resubscribed", old_count=len(old_ids), new_count=len(new_ids))

    async def unsubscribe_symbol(self, symbol: str) -> None:
        """Dynamically unsubscribe from a symbol's channels."""
        if not self._ws or not self._running:
            return
        channels = self._subscription_args([symbol])
        await self._send_subscription_operation("unsubscribe", channels)
        self._latest_tickers.pop(symbol, None)
        normalized = normalize_trading_symbol(symbol)
        self.latest_orderbooks.pop(normalized, None)
        self.latest_mark_prices.pop(normalized, None)
        self.latest_index_prices.pop(normalized, None)
        self._subscribe_symbols = [
            item
            for item in getattr(self, "_subscribe_symbols", [])
            if normalize_trading_symbol(item) != normalized
        ]
        logger.info("unsubscribed from symbol", symbol=symbol)

    async def close(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception as exc:
                logger.debug("OKX WebSocket close failed", error=safe_error_text(exc))
            self._ws = None
        logger.info("OKX WebSocket disconnected")

    def get_stats(self) -> dict[str, Any]:
        now = time.time()
        ticker_reference_time = self._last_ticker_time or self._connected_at
        last_ticker_seconds_ago = now - ticker_reference_time if ticker_reference_time else None
        return {
            "connected": self.is_connected,
            "messages_received": self._message_count,
            "tracked_symbols": len(self._latest_tickers),
            "last_message_seconds_ago": (
                now - self._last_message_time if self._last_message_time else None
            ),
            "last_ticker_seconds_ago": last_ticker_seconds_ago,
            "ticker_stream_stale": bool(
                last_ticker_seconds_ago is not None
                and last_ticker_seconds_ago > WS_TICKER_STALE_RECONNECT_SECONDS
            ),
            "ticker_stale_after_seconds": WS_TICKER_STALE_RECONNECT_SECONDS,
            "reconnect_count": self._reconnect_count,
            "throttled_ticker_updates": self._throttled_ticker_updates,
            "ticker_process_interval_seconds": WS_TICKER_MIN_PROCESS_INTERVAL_SECONDS,
            "subscription_request_count": self._subscription_request_count,
            "subscription_acknowledged_count": self._subscription_acknowledged_count,
            "subscription_error_count": self._subscription_error_count,
            "subscription_coverage": self._subscription_coverage_snapshot(),
            "channel_message_counts": dict(self._channel_message_counts),
        }

    def _subscription_coverage_snapshot(self) -> dict[str, Any]:
        acknowledged = self._subscription_acknowledged_args & self._subscription_expected_args
        rejected = self._subscription_rejected_args & self._subscription_expected_args
        missing = self._subscription_expected_args - acknowledged - rejected
        channels = sorted(
            {
                channel
                for channel, _inst_id in self._subscription_expected_args
            }
        )
        by_channel = {}
        for channel in channels:
            expected_args = {
                key
                for key in self._subscription_expected_args
                if key[0] == channel
            }
            acknowledged_args = expected_args & acknowledged
            rejected_args = expected_args & rejected
            by_channel[channel] = {
                "expected": len(expected_args),
                "acknowledged": len(acknowledged_args),
                "rejected": len(rejected_args),
                "missing_ack": len(
                    expected_args - acknowledged_args - rejected_args
                ),
            }
        return {
            "expected_argument_count": len(self._subscription_expected_args),
            "acknowledged_argument_count": len(acknowledged),
            "rejected_argument_count": len(rejected),
            "missing_ack_count": len(missing),
            "unattributed_error_count": self._subscription_unattributed_error_count,
            "by_channel": by_channel,
            "missing_ack_sample": [
                {"channel": channel, "inst_id": inst_id}
                for channel, inst_id in sorted(missing)[:20]
            ],
        }
