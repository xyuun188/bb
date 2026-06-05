"""
OKX V5 WebSocket client for real-time market data.
Subscribes to ticker and K-line channels for configured symbols.

OKX V5 WebSocket Docs: https://www.okx.com/docs-v5/en/#websocket-api
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

import structlog
import websockets

from config.settings import settings
from core.exceptions import WebSocketConnectionError

logger = structlog.get_logger(__name__)

# OKX WebSocket endpoints
# Public URL: tickers, instruments, funding-rate, etc.
WS_PUBLIC_URL = "wss://ws.okx.com:8443/ws/v5/public"
# Business URL: candlesticks, mark-price-candle, index-candle (moved from public in June 2023)
WS_BUSINESS_URL = "wss://ws.okx.com:8443/ws/v5/business"
# Demo URL: blocked in some regions, not used
WS_DEMO_URL = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"


class OKXWebSocketClient:
    """Async WebSocket client for OKX V5 public channels.

    Handles reconnection, heartbeats, and message routing.
    Public market data uses ws.okx.com regardless of demo/live mode.
    """

    def __init__(self) -> None:
        self._ws_url = WS_PUBLIC_URL  # Public data always from ws.okx.com
        self._ws = None
        self._running = False
        self._ticker_callbacks: list[Callable] = []
        self._kline_callbacks: list[Callable] = []
        self._latest_tickers: dict[str, dict] = {}
        self._latest_klines: dict[str, list[dict]] = {}
        self._message_count = 0
        self._last_message_time = 0.0

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

    async def connect(self) -> None:
        """Establish WebSocket connection and subscribe to channels."""
        self._running = True
        logger.info("connecting to OKX WebSocket", url=self._ws_url)

        try:
            self._ws = await websockets.connect(
                self._ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
                max_size=2 ** 20,
            )
            logger.info("OKX WebSocket connected")
            await self._subscribe()
        except Exception as e:
            logger.error("OKX WebSocket connection failed", error=str(e))
            self._running = False
            raise WebSocketConnectionError(f"Failed to connect: {e}") from e

    async def _subscribe(self) -> None:
        """Subscribe to ticker channels for configured symbols."""
        symbols = getattr(self, '_subscribe_symbols', settings.symbols)
        ticker_channels = []
        for symbol in symbols:
            inst_id = self._to_ws_inst_id(symbol)
            ticker_channels.append({"channel": "tickers", "instId": inst_id})

        sub_msg = {"op": "subscribe", "args": ticker_channels}
        await self._ws.send(json.dumps(sub_msg))
        logger.info("subscribed to channels", tickers=len(ticker_channels))

    async def _handle_message(self, raw: str) -> None:
        """Parse and route incoming WebSocket messages."""
        self._message_count += 1
        self._last_message_time = time.time()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        if "event" in data:
            # Subscription confirmation or error
            logger.debug("ws event", ws_event=data.get("event"), msg=data.get("msg", ""))
            return

        if "arg" not in data or "data" not in data:
            return

        channel = data["arg"].get("channel", "")
        inst_id = data["arg"].get("instId", "")

        if channel == "tickers":
            for ticker in data["data"]:
                symbol = inst_id.replace("-SWAP", "").replace("-", "/")
                last = float(ticker.get("last", 0))
                open24h = float(ticker.get("open24h", 0))
                change_pct = ((last - open24h) / open24h * 100) if open24h else 0
                parsed = {
                    "symbol": symbol,
                    "last_price": last,
                    "bid": float(ticker.get("bidPx", 0)),
                    "ask": float(ticker.get("askPx", 0)),
                    "high_24h": float(ticker.get("high24h", 0)),
                    "low_24h": float(ticker.get("low24h", 0)),
                    "volume_24h": float(ticker.get("vol24h", 0)),
                    "change_24h_pct": change_pct,
                    "timestamp": int(ticker.get("ts", 0)),
                }
                self._latest_tickers[symbol] = parsed
                for cb in self._ticker_callbacks:
                    try:
                        cb(symbol, parsed)
                    except Exception:
                        pass

        elif channel and channel.startswith("candle"):
            timeframe = channel.replace("candle", "")
            candles = []
            for candle in data["data"]:
                candles.append({
                    "open_time": int(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                })
            symbol = inst_id.replace("-SWAP", "").replace("-", "/")
            key = f"{symbol}:{timeframe}"
            self._latest_klines[key] = candles
            for cb in self._kline_callbacks:
                try:
                    cb(symbol, timeframe, candles)
                except Exception:
                    pass

    async def listen(self) -> None:
        """Main message loop. Blocks until disconnected or stopped."""
        if not self._ws:
            await self.connect()

        while self._running:
            try:
                raw = await asyncio.wait_for(
                    self._ws.recv(), timeout=30
                )
                await self._handle_message(raw)
            except asyncio.TimeoutError:
                # Send ping to keep alive
                try:
                    await self._ws.send("ping")
                except Exception:
                    logger.warning("ping failed, reconnecting...")
                    break
            except websockets.ConnectionClosed as e:
                logger.warning("websocket closed", code=e.code, reason=e.reason)
                break
            except Exception as e:
                logger.error("unexpected error in listen loop", error=str(e))
                break

        # Auto-reconnect
        if self._running:
            logger.info("reconnecting in 5 seconds...")
            await asyncio.sleep(5)
            await self.connect()
            asyncio.create_task(self.listen())

    async def subscribe_symbol(self, symbol: str) -> None:
        """Dynamically subscribe to a new symbol's ticker channel."""
        if not self._ws or not self._running:
            logger.warning("cannot subscribe, ws not connected", symbol=symbol)
            return
        inst_id = self._to_ws_inst_id(symbol)
        channels = [{"channel": "tickers", "instId": inst_id}]
        sub_msg = {"op": "subscribe", "args": channels}
        await self._ws.send(json.dumps(sub_msg))
        logger.info("subscribed to symbol", symbol=symbol, inst_id=inst_id)

    async def resubscribe_all(self, symbols: list[str]) -> None:
        """Resubscribe to a new list of symbols (unsub old, sub new)."""
        old_ids = [
            {"channel": "tickers", "instId": self._to_ws_inst_id(s)}
            for s in getattr(self, '_subscribe_symbols', settings.symbols)
        ]
        new_ids = [
            {"channel": "tickers", "instId": self._to_ws_inst_id(s)}
            for s in symbols
        ]
        if not self._ws or not self._running:
            self._subscribe_symbols = symbols
            return
        # Unsubscribe from old
        if old_ids:
            await self._ws.send(json.dumps({"op": "unsubscribe", "args": old_ids}))
        # Clear old tickers
        old_symbols = set(getattr(self, '_subscribe_symbols', []))
        for sym in old_symbols:
            self._latest_tickers.pop(sym, None)
        # Subscribe to new
        if new_ids:
            await self._ws.send(json.dumps({"op": "subscribe", "args": new_ids}))
        self._subscribe_symbols = symbols
        logger.info("resubscribed", old_count=len(old_ids), new_count=len(new_ids))

    async def unsubscribe_symbol(self, symbol: str) -> None:
        """Dynamically unsubscribe from a symbol's channels."""
        if not self._ws or not self._running:
            return
        inst_id = self._to_ws_inst_id(symbol)
        channels = [{"channel": "tickers", "instId": inst_id}]
        sub_msg = {"op": "unsubscribe", "args": channels}
        await self._ws.send(json.dumps(sub_msg))
        self._latest_tickers.pop(symbol, None)
        logger.info("unsubscribed from symbol", symbol=symbol)

    async def close(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("OKX WebSocket disconnected")

    def get_stats(self) -> dict[str, Any]:
        return {
            "connected": self.is_connected,
            "messages_received": self._message_count,
            "tracked_symbols": len(self._latest_tickers),
            "last_message_seconds_ago": (
                time.time() - self._last_message_time if self._last_message_time else None
            ),
        }
