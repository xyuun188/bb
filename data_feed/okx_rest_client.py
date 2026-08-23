"""
OKX REST client via the official OKX SDK for fallback and private API operations.
Used when WebSocket is unavailable or for account/trading endpoints.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import structlog

from config.settings import settings
from core.market_facts import normalize_okx_contract_spec
from core.okx_instrument_filter import supported_usdt_swap_instruments
from core.safe_output import safe_error_text
from core.symbols import (
    normalize_trading_symbol,
    okx_inst_id_from_symbol,
    symbol_from_okx_inst_id,
    symbol_from_okx_market,
)
from core.trading_mode import mode_manager
from data_feed.okx_ticker_volume import okx_swap_volume_fields
from services.okx_perpetual_sdk import OkxPerpetualSdkExchange

logger = structlog.get_logger(__name__)

OKX_REST_URL = "https://{hostname}"
OKX_HOSTNAME = "www.okx.com"
SUSPICIOUS_CONTRACT_BASE_TOKENS = ("TEST", "DEMO", "DUMMY", "MOCK", "SAMPLE")
DERIVATIVES_SNAPSHOT_ROUTE_TIMEOUT_SECONDS = 6.0
DERIVATIVES_ROUTE_TIMEOUT_SECONDS = 2.5
DERIVATIVES_ROUTE_CACHE_TTL_SECONDS = 20.0
DERIVATIVES_ROUTE_FAILURE_BACKOFF_SECONDS = 3.0
REFERENCE_PRICES_ROUTE_TIMEOUT_SECONDS = 1.5
REFERENCE_PRICES_FALLBACK_TIMEOUT_SECONDS = 0.35
REFERENCE_PRICES_CACHE_TTL_SECONDS = 15.0
REFERENCE_PRICES_RETRY_BACKOFF_SECONDS = 2.0
ORDERBOOK_ROUTE_TIMEOUT_SECONDS = 2.5
ORDERBOOK_CACHE_TTL_SECONDS = 10.0
TICKER_ROUTE_TIMEOUT_SECONDS = 2.5
# A caller joining an already-running ticker refresh gets a normal bounded
# wait.  This is deliberately independent from the first caller's short
# route budget so a completed shared request is not lost as a timeout stub.
TICKER_ROUTE_JOIN_TIMEOUT_SECONDS = 2.5
TICKER_ROUTE_FAILURE_BACKOFF_SECONDS = 3.0


def _consume_cancelled_task(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        return

def _is_suspicious_contract_base(base: str | None) -> bool:
    value = str(base or "").upper()
    return bool(value and any(token in value for token in SUSPICIOUS_CONTRACT_BASE_TOKENS))


class OKXRestClient:
    """Async REST client wrapping the official OKX SDK adapter."""

    def __init__(self) -> None:
        self._exchange: OkxPerpetualSdkExchange | None = None
        # Concurrent candidate fetches can all arrive before the first market
        # bootstrap completes. Serialize that one-time bootstrap so the shared
        # SDK session is not initialized and used concurrently.
        self._exchange_init_lock = asyncio.Lock()
        self._instrument_spec_lock = asyncio.Lock()
        self._exchange_ready = False
        self._reference_price_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._reference_price_tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._reference_price_failure_cache: dict[str, float] = {}
        self._ticker_tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._ticker_retry_at: dict[str, float] = {}
        self._orderbook_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._orderbook_tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._derivatives_route_cache: dict[
            tuple[str, str], tuple[float, dict[str, Any]]
        ] = {}
        self._derivatives_route_tasks: dict[
            tuple[str, str], asyncio.Task[dict[str, Any]]
        ] = {}
        self._derivatives_route_retry_at: dict[tuple[str, str], float] = {}

    async def _get_exchange(self) -> OkxPerpetualSdkExchange:
        if self._exchange is not None and self._exchange_ready:
            return self._exchange
        async with self._exchange_init_lock:
            if self._exchange is None:
                mode = mode_manager.mode.value
                self._exchange = OkxPerpetualSdkExchange(mode)
                self._ensure_rest_url()
            else:
                mode = str(getattr(self._exchange, "mode", "") or mode_manager.mode.value)

            if not self._exchange_ready:
                is_demo = settings.is_okx_demo(mode)
                await self._load_usdt_swap_markets()
                self._exchange_ready = True
                logger.info(
                    "OKX REST markets loaded",
                    mode=mode,
                    demo=is_demo,
                    symbols_count=len(self._exchange.markets),
                )

        return self._exchange

    def _ensure_rest_url(self) -> None:
        """Keep legacy URL guards harmless for the SDK-backed client."""
        if self._exchange is None:
            return
        urls = getattr(self._exchange, "urls", None)
        if not isinstance(urls, dict):
            return
        for key in ("api", "test"):
            value = urls.get(key)
            if not isinstance(value, dict):
                urls[key] = {"rest": OKX_REST_URL}
            elif not value.get("rest"):
                value["rest"] = OKX_REST_URL
        if not getattr(self._exchange, "hostname", None):
            self._exchange.hostname = OKX_HOSTNAME

    def _is_broken_rest_url_error(self, exc: Exception) -> bool:
        message = safe_error_text(exc)
        return "unsupported operand type(s) for +: 'NoneType' and 'str'" in message or (
            "NoneType" in message and "+:" in message and "str" in message
        )

    async def _ccxt_call(self, method_name: str, *args, **kwargs):
        for attempt in range(2):
            ex = await self._get_exchange()
            self._ensure_rest_url()
            method = getattr(ex, method_name)
            try:
                return await method(*args, **kwargs)
            except Exception as exc:
                if attempt == 0 and self._is_broken_rest_url_error(exc):
                    logger.warning(
                        "OKX REST URL state invalid; reinitializing SDK client",
                        method=method_name,
                        error=safe_error_text(exc),
                    )
                    await self.reinitialize()
                    continue
                raise

    async def _load_usdt_swap_markets(self) -> None:
        """Load only live linear USDT perpetual swaps.

        OKX demo can return test/preopen instruments with incomplete fields; filtering
        before building local market rules keeps public ticker/K-line calls usable.
        """
        if self._exchange is None:
            return
        self._ensure_rest_url()
        response = await self._exchange.publicGetPublicInstruments({"instType": "SWAP"})
        instruments = response.get("data", []) if isinstance(response, dict) else []
        filtered = supported_usdt_swap_instruments(instruments)
        markets = self._exchange.parse_markets(filtered)
        self._exchange.set_markets(markets)

    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch one ticker through a bounded per-instrument single-flight."""
        inst_id = okx_inst_id_from_symbol(symbol)
        if not inst_id:
            return {}
        now = time.monotonic()
        task = self._ticker_tasks.get(inst_id)
        joining_inflight = task is not None and not task.done()
        if task is None or task.done():
            retry_at = self._ticker_retry_at.get(inst_id, 0.0)
            if retry_at > now:
                return {
                    "ticker_refresh_in_background": True,
                    "ticker_retry_after_seconds": round(retry_at - now, 3),
                }
            task = asyncio.create_task(self._fetch_ticker_uncached(inst_id))
            self._ticker_tasks[inst_id] = task

            def _finish(finished: asyncio.Task[dict[str, Any]]) -> None:
                if self._ticker_tasks.get(inst_id) is finished:
                    self._ticker_tasks.pop(inst_id, None)
                try:
                    result = finished.result()
                except BaseException:
                    self._ticker_retry_at[inst_id] = (
                        time.monotonic() + TICKER_ROUTE_FAILURE_BACKOFF_SECONDS
                    )
                    return
                if not isinstance(result, dict) or not result:
                    self._ticker_retry_at[inst_id] = (
                        time.monotonic() + TICKER_ROUTE_FAILURE_BACKOFF_SECONDS
                    )
                else:
                    self._ticker_retry_at.pop(inst_id, None)

            task.add_done_callback(_finish)

        try:
            timeout_seconds = (
                TICKER_ROUTE_JOIN_TIMEOUT_SECONDS
                if joining_inflight
                else TICKER_ROUTE_TIMEOUT_SECONDS
            )
            return dict(
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=timeout_seconds,
                )
            )
        except TimeoutError:
            # Keep the shared request alive. A later analysis round must wait on
            # this task instead of issuing another request against the same SDK.
            # Do not mark the instrument as failed here: the in-flight request
            # may still complete successfully after this caller's short wait.
            # The done callback owns failure backoff once the shared task settles.
            return {
                "ticker_refresh_in_background": True,
                "ticker_timeout": True,
            }

    async def _fetch_ticker_uncached(self, inst_id: str) -> dict[str, Any]:
        response = await self._ccxt_call("publicGetMarketTicker", {"instId": inst_id})
        rows = response.get("data", []) if isinstance(response, dict) else []
        if not rows:
            return {}
        return self._native_ticker_to_ccxt_shape(rows[0])

    async def fetch_instrument_spec(self, symbol: str) -> dict[str, Any]:
        # Coalesce the rare fallback instrument request. The first caller may
        # bootstrap the SDK; later callers can then reuse the full native
        # instrument cache instead of issuing one request per symbol.
        async with self._instrument_spec_lock:
            return await self._fetch_instrument_spec_locked(symbol)

    async def _fetch_instrument_spec_locked(self, symbol: str) -> dict[str, Any]:
        """Return the exact OKX-native contract identity for one swap symbol."""

        inst_id = okx_inst_id_from_symbol(symbol)
        if not inst_id:
            return {}
        # The initial SWAP instruments bootstrap already contains the exact
        # native contract row. Reuse it instead of issuing one public request
        # per candidate, which otherwise creates a burst of instrument calls
        # and starves derivatives requests behind the same SDK session.
        exchange = self._exchange
        markets = getattr(exchange, "markets", {}) or {}
        market = markets.get(inst_id)
        if isinstance(market, dict):
            info = market.get("info")
            if isinstance(info, dict) and str(info.get("instId") or "").strip().upper() == inst_id:
                return {
                    key: info.get(key)
                    for key in (
                        "instId", "instType", "uly", "instFamily", "ctType",
                        "ctVal", "ctMult", "ctValCcy", "settleCcy", "lotSz",
                        "minSz", "tickSz", "state",
                    )
                } | {"source": "okx_public_instruments_cache"}
        response = await self._ccxt_call(
            "publicGetPublicInstruments",
            {"instType": "SWAP", "instId": inst_id},
        )
        rows = response.get("data", []) if isinstance(response, dict) else []
        supported = supported_usdt_swap_instruments(rows)
        row = next(
            (
                item
                for item in supported
                if str(item.get("instId") or "").strip().upper() == inst_id
            ),
            None,
        )
        if not isinstance(row, dict):
            return {}
        return {
            key: row.get(key)
            for key in (
                "instId",
                "instType",
                "uly",
                "instFamily",
                "ctType",
                "ctVal",
                "ctMult",
                "ctValCcy",
                "settleCcy",
                "lotSz",
                "minSz",
                "tickSz",
                "state",
            )
        } | {"source": "okx_public_instruments"}

    async def fetch_tickers(self, symbols: list[str] | None = None) -> dict:
        target_inst_ids = {
            inst_id for symbol in symbols or [] if (inst_id := okx_inst_id_from_symbol(symbol))
        }
        exchange = self._exchange
        if not target_inst_ids:
            exchange = await self._get_exchange()
        supported_inst_ids = target_inst_ids or self._loaded_market_inst_ids(exchange)
        response = await self._ccxt_call("publicGetMarketTickers", {"instType": "SWAP"})
        rows = response.get("data", []) if isinstance(response, dict) else []
        tickers: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            inst_id = str(row.get("instId") or "").strip().upper()
            if supported_inst_ids and inst_id not in supported_inst_ids:
                continue
            ticker = self._native_ticker_to_ccxt_shape(row)
            symbol = str(ticker.get("symbol") or "").strip()
            if symbol:
                tickers[symbol] = ticker
            if inst_id:
                tickers[inst_id] = ticker
        return tickers

    def _loaded_market_inst_ids(self, exchange: Any | None = None) -> set[str]:
        source = exchange or self._exchange
        if source is None:
            return set()
        markets = getattr(source, "markets", None) or {}
        inst_ids: set[str] = set()
        for market_id, market in markets.items():
            if not isinstance(market, dict):
                continue
            inst_id = str(market.get("id") or market_id or "").strip().upper()
            if inst_id:
                inst_ids.add(inst_id)
        return inst_ids

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> list[list[float]]:
        return await self._ccxt_call(
            "fetch_ohlcv",
            self._to_swap_symbol(symbol),
            timeframe,
            limit=limit,
        )

    async def fetch_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Return OKX perpetual funding data, normalized for feature building."""
        try:
            data = await self._ccxt_call("fetch_funding_rate", self._to_swap_symbol(symbol))
        except Exception as exc:
            logger.debug("fetch funding rate failed", symbol=symbol, error=safe_error_text(exc))
            return {
                "funding_rate": 0.0,
                "funding_data_available": False,
                "funding_interval_minutes": None,
                "funding_time": None,
                "next_funding_time": None,
            }

        info = data.get("info") or {}
        funding_time = data.get("fundingDatetime") or info.get("fundingTime")
        next_funding_time = data.get("nextFundingDatetime") or info.get("nextFundingTime")
        funding_time_ms = self._safe_float(funding_time)
        next_funding_time_ms = self._safe_float(next_funding_time)
        funding_interval_minutes = (
            (next_funding_time_ms - funding_time_ms) / 60_000.0
            if funding_time_ms > 0 and next_funding_time_ms > funding_time_ms
            else None
        )
        funding_rate_present = any(
            key in data or key in info
            for key in ("fundingRate", "funding_rate")
        )
        return {
            "funding_rate": self._safe_float(
                data.get("fundingRate") or data.get("funding_rate") or info.get("fundingRate")
            ),
            "funding_data_available": bool(
                funding_rate_present and funding_interval_minutes is not None
            ),
            "funding_interval_minutes": funding_interval_minutes,
            "funding_time": funding_time,
            "next_funding_time": next_funding_time,
            "funding_rate_observed_at": info.get("ts"),
        }

    async def fetch_open_interest(self, symbol: str) -> dict[str, Any]:
        """Return OKX perpetual open-interest data with safe defaults."""
        try:
            data = await self._ccxt_call("fetch_open_interest", self._to_swap_symbol(symbol))
        except Exception as exc:
            logger.debug("fetch open interest failed", symbol=symbol, error=safe_error_text(exc))
            return {"open_interest_contracts": 0.0, "open_interest_value": 0.0}

        info = data.get("info") or {}
        contracts = self._safe_float(
            data.get("openInterestAmount") or data.get("openInterest") or info.get("oi")
        )
        value = self._safe_float(
            data.get("openInterestValue")
            or data.get("quoteVolume")
            or info.get("oiCcy")
            or info.get("oiUsd")
        )
        return {
            "open_interest_contracts": contracts,
            "open_interest_value": value,
        }

    async def _cached_derivatives_route(
        self,
        route: str,
        symbol: str,
        fetcher: Any,
        *,
        timeout_seconds: float = DERIVATIVES_ROUTE_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Coalesce slow derivative routes and keep their refresh off the round path."""

        inst_id = okx_inst_id_from_symbol(symbol)
        key = (str(route), inst_id)
        now = time.monotonic()
        cached = self._derivatives_route_cache.get(key)
        if cached is not None and now - cached[0] <= DERIVATIVES_ROUTE_CACHE_TTL_SECONDS:
            return {**cached[1], f"{route}_cached": True}
        task = self._derivatives_route_tasks.get(key)
        if task is not None and not task.done():
            try:
                return dict(
                    await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=max(float(timeout_seconds or 0.0), 0.1),
                    )
                )
            except TimeoutError:
                return {
                    **(cached[1] if cached is not None else {}),
                    f"{route}_refresh_in_background": True,
                    f"{route}_timeout": True,
                }
        retry_at = self._derivatives_route_retry_at.get(key, 0.0)
        if retry_at > now:
            return {
                f"{route}_refresh_in_background": True,
                f"{route}_retry_after_seconds": round(retry_at - now, 3),
                **(cached[1] if cached is not None else {}),
            }

        if task is None or task.done():
            task = asyncio.create_task(fetcher())
            self._derivatives_route_tasks[key] = task

            def cleanup(finished: asyncio.Task[dict[str, Any]]) -> None:
                if self._derivatives_route_tasks.get(key) is finished:
                    self._derivatives_route_tasks.pop(key, None)
                try:
                    result = finished.result()
                except BaseException:
                    self._derivatives_route_retry_at[key] = (
                        time.monotonic() + DERIVATIVES_ROUTE_FAILURE_BACKOFF_SECONDS
                    )
                    return
                if isinstance(result, dict) and result:
                    self._derivatives_route_cache[key] = (time.monotonic(), dict(result))
                    self._derivatives_route_retry_at.pop(key, None)
                else:
                    self._derivatives_route_retry_at[key] = (
                        time.monotonic() + DERIVATIVES_ROUTE_FAILURE_BACKOFF_SECONDS
                    )

            task.add_done_callback(cleanup)
        try:
            return dict(
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=max(float(timeout_seconds or 0.0), 0.1),
                )
            )
        except TimeoutError:
            # Keep the single-flight task alive so a late OKX response can seed
            # the cache; do not cancel and immediately reissue the same route.
            self._derivatives_route_retry_at[key] = (
                time.monotonic() + DERIVATIVES_ROUTE_FAILURE_BACKOFF_SECONDS
            )
            if cached is not None:
                return {
                    **cached[1],
                    f"{route}_refresh_in_background": True,
                    f"{route}_timeout": True,
                }
            return {
                f"{route}_refresh_in_background": True,
                f"{route}_timeout": True,
            }
        except Exception as exc:
            self._derivatives_route_retry_at[key] = (
                time.monotonic() + DERIVATIVES_ROUTE_FAILURE_BACKOFF_SECONDS
            )
            logger.debug(
                "OKX derivative route failed",
                route=route,
                symbol=symbol,
                error=safe_error_text(exc),
            )
            return dict(cached[1]) if cached is not None else {}

    async def fetch_order_book_metrics(
        self,
        symbol: str,
        limit: int = 20,
        *,
        contract_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return top-book metrics without letting a slow route block the round."""
        normalized_spec = normalize_okx_contract_spec(contract_spec)
        inst_id = okx_inst_id_from_symbol(symbol)
        contract_value = self._safe_float(normalized_spec.get("contract_value"))
        contract_multiplier = self._safe_float(
            normalized_spec.get("contract_multiplier")
        )
        contract_spec_available = bool(
            contract_value > 0 and contract_multiplier > 0
        )
        cached = self._orderbook_cache.get(inst_id)
        if cached is not None and time.monotonic() - cached[0] <= ORDERBOOK_CACHE_TTL_SECONDS:
            if not contract_spec_available:
                return {
                    "spread_pct": float(cached[1].get("spread_pct") or 0.0),
                    "orderbook_bid_depth": 0.0,
                    "orderbook_ask_depth": 0.0,
                    "orderbook_imbalance": 0.0,
                    "orderbook_data_available": False,
                    "orderbook_unavailable_reason": "contract_spec_missing",
                }
            return {**cached[1], "orderbook_cached": True}

        async def fetch_and_normalize() -> dict[str, Any]:
            book = await self._ccxt_call(
                "fetch_order_book",
                self._to_swap_symbol(symbol),
                limit=limit,
            )
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            bid = self._safe_float(bids[0][0] if bids else 0.0)
            ask = self._safe_float(asks[0][0] if asks else 0.0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
            spread_pct = ((ask - bid) / mid * 100) if mid > 0 and ask >= bid else 0.0
            bid_depth = sum(
                self._safe_float(level[0]) * self._safe_float(level[1])
                for level in bids[:limit]
                if len(level) >= 2
            )
            ask_depth = sum(
                self._safe_float(level[0]) * self._safe_float(level[1])
                for level in asks[:limit]
                if len(level) >= 2
            )
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0
            contract_base_size = contract_value * contract_multiplier
            bid_depth_usdt = bid_depth * contract_base_size if contract_base_size > 0 else 0.0
            ask_depth_usdt = ask_depth * contract_base_size if contract_base_size > 0 else 0.0
            info = book.get("info") if isinstance(book.get("info"), dict) else {}
            source_timestamp_ms = int(
                self._safe_float(book.get("timestamp") or info.get("ts"))
            )
            payload = {
                "spread_pct": spread_pct,
                "orderbook_bid_depth": bid_depth_usdt,
                "orderbook_ask_depth": ask_depth_usdt,
                "orderbook_imbalance": imbalance,
                "orderbook_data_available": bool(
                    bid > 0
                    and ask > 0
                    and contract_spec_available
                    and bid_depth_usdt > 0
                    and ask_depth_usdt > 0
                ),
                "orderbook_unavailable_reason": (
                    None
                    if contract_spec_available
                    else "contract_spec_missing"
                ),
                "orderbook_fact": {
                    "inst_id": inst_id,
                    "inst_type": "SWAP",
                    "source_endpoint": "okx_rest_market_books",
                    "source_channel": "books",
                    "source_timestamp_ms": source_timestamp_ms,
                    "bid": bid,
                    "ask": ask,
                    "bid_depth_usdt": bid_depth_usdt,
                    "ask_depth_usdt": ask_depth_usdt,
                    "contract_spec_version": normalized_spec.get("spec_version"),
                },
            }
            if payload["orderbook_data_available"]:
                self._orderbook_cache[inst_id] = (time.monotonic(), payload)
            return payload

        task = self._orderbook_tasks.get(inst_id)
        if task is None or task.done():
            task = asyncio.create_task(fetch_and_normalize())
            self._orderbook_tasks[inst_id] = task

            def cleanup(finished: asyncio.Task[dict[str, Any]]) -> None:
                if self._orderbook_tasks.get(inst_id) is finished:
                    self._orderbook_tasks.pop(inst_id, None)
                _consume_cancelled_task(finished)

            task.add_done_callback(cleanup)
        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                timeout=ORDERBOOK_ROUTE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.debug(
                "OKX orderbook refresh still running; returning unavailable snapshot",
                symbol=symbol,
                timeout_seconds=ORDERBOOK_ROUTE_TIMEOUT_SECONDS,
            )
            return {
                "spread_pct": 0.0,
                "orderbook_bid_depth": 0.0,
                "orderbook_ask_depth": 0.0,
                "orderbook_imbalance": 0.0,
                "orderbook_data_available": False,
                "orderbook_refresh_in_background": True,
            }
        except Exception as exc:
            logger.debug("fetch order book failed", symbol=symbol, error=safe_error_text(exc))
            return {
                "spread_pct": 0.0,
                "orderbook_bid_depth": 0.0,
                "orderbook_ask_depth": 0.0,
                "orderbook_imbalance": 0.0,
                "orderbook_data_available": False,
            }

    async def fetch_reference_prices(
        self,
        symbol: str,
        *,
        contract_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return reference prices through a per-instrument single-flight refresh."""
        normalized_spec = normalize_okx_contract_spec(contract_spec)
        inst_id = okx_inst_id_from_symbol(symbol)
        cached = self._reference_price_cache.get(inst_id)
        if cached is not None and time.monotonic() - cached[0] <= REFERENCE_PRICES_CACHE_TTL_SECONDS:
            return {**cached[1], "reference_prices_cached": True}

        now = time.monotonic()
        retry_at = self._reference_price_failure_cache.get(inst_id, 0.0)
        if retry_at > now:
            return {
                "reference_prices_data_available": False,
                "reference_prices_retry_after_seconds": round(retry_at - now, 3),
                "reference_prices_refresh_in_background": True,
            }

        task = self._reference_price_tasks.get(inst_id)
        if task is None or task.done():
            task = asyncio.create_task(
                self._refresh_reference_prices(
                    symbol,
                    contract_spec=normalized_spec,
                )
            )
            self._reference_price_tasks[inst_id] = task

            def _finish(completed: asyncio.Task[dict[str, Any]]) -> None:
                if self._reference_price_tasks.get(inst_id) is completed:
                    self._reference_price_tasks.pop(inst_id, None)
                try:
                    result = completed.result()
                except BaseException:
                    self._reference_price_failure_cache[inst_id] = (
                        time.monotonic() + REFERENCE_PRICES_RETRY_BACKOFF_SECONDS
                    )
                    return
                if not result.get("mark_price") and not result.get("index_price"):
                    self._reference_price_failure_cache[inst_id] = (
                        time.monotonic() + REFERENCE_PRICES_RETRY_BACKOFF_SECONDS
                    )
                else:
                    self._reference_price_failure_cache.pop(inst_id, None)

            task.add_done_callback(_finish)
        result = await asyncio.shield(task)
        return dict(result)

    async def _refresh_reference_prices(
        self,
        symbol: str,
        *,
        contract_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_spec = normalize_okx_contract_spec(contract_spec)
        inst_id = okx_inst_id_from_symbol(symbol)
        uly = str(normalized_spec.get("uly") or inst_id.removesuffix("-SWAP")).upper()
        route_tasks = {
            "mark": asyncio.create_task(
                self._ccxt_call(
                    "publicGetPublicMarkPrice",
                    {"instType": "SWAP", "instId": inst_id},
                )
            ),
            "index": asyncio.create_task(
                self._ccxt_call("publicGetMarketIndexTickers", {"instId": uly})
            ),
        }
        task_names = {task: name for name, task in route_tasks.items()}
        done, pending = await asyncio.wait(
            set(route_tasks.values()),
            timeout=REFERENCE_PRICES_ROUTE_TIMEOUT_SECONDS,
        )
        for task in pending:
            task.cancel()
            task.add_done_callback(_consume_cancelled_task)
        results: dict[str, Any] = {}
        for task in done:
            try:
                results[task_names[task]] = task.result()
            except Exception:
                results[task_names[task]] = {}

        mark_result = results.get("mark")
        index_result = results.get("index")
        mark_rows = (
            mark_result.get("data", []) if isinstance(mark_result, dict) else []
        )
        index_rows = (
            index_result.get("data", []) if isinstance(index_result, dict) else []
        )
        mark_row = mark_rows[0] if mark_rows and isinstance(mark_rows[0], dict) else {}
        index_row = index_rows[0] if index_rows and isinstance(index_rows[0], dict) else {}
        mark_price = self._safe_float(mark_row.get("markPx"))
        index_price = self._safe_float(index_row.get("idxPx"))
        fallback_used = False
        if mark_price <= 0 or index_price <= 0:
            # A slow public mark/index endpoint must not block an analysis
            # round.  The ticker is an exchange-native, current price fallback
            # and is explicitly labelled so downstream training can distinguish
            # it from a verified mark/index pair.
            ticker_task: asyncio.Task[Any] | None = None
            try:
                ticker_task = asyncio.create_task(self.fetch_ticker(symbol))
                ticker_result = await asyncio.wait_for(
                    asyncio.shield(ticker_task),
                    timeout=REFERENCE_PRICES_FALLBACK_TIMEOUT_SECONDS,
                )
                ticker_info = ticker_result.get("info") or {}
                ticker_price = self._safe_float(ticker_result.get("last"))
                ticker_timestamp_ms = int(
                    self._safe_float(
                        ticker_result.get("timestamp")
                        or ticker_info.get("ts")
                    )
                )
                if ticker_price > 0:
                    if mark_price <= 0:
                        mark_price = ticker_price
                        mark_row = {
                            "instId": inst_id,
                            "instType": "SWAP",
                            "markPx": ticker_result.get("last"),
                            "ts": ticker_timestamp_ms,
                            "fallback": True,
                        }
                        fallback_used = True
                    if index_price <= 0:
                        index_price = ticker_price
                        index_row = {
                            "instId": uly,
                            "idxPx": ticker_result.get("last"),
                            "ts": ticker_timestamp_ms,
                            "fallback": True,
                        }
                        fallback_used = True
            except Exception:
                if ticker_task is not None and not ticker_task.done():
                    ticker_task.cancel()
                    ticker_task.add_done_callback(_consume_cancelled_task)

        payload = {
            "mark_price": mark_price,
            "index_price": index_price,
            "mark_price_fact": {
                "inst_id": str(mark_row.get("instId") or inst_id).upper(),
                "inst_type": str(mark_row.get("instType") or "SWAP").upper(),
                "source_endpoint": (
                    "okx_rest_market_ticker_fallback"
                    if mark_row.get("fallback")
                    else "okx_rest_public_mark_price"
                ),
                "source_channel": (
                    "tickers-fallback" if mark_row.get("fallback") else "mark-price"
                ),
                "source_timestamp_ms": int(self._safe_float(mark_row.get("ts"))),
                "price": mark_price,
            },
            "index_price_fact": {
                "inst_id": str(index_row.get("instId") or uly).upper(),
                "inst_type": "INDEX",
                "source_endpoint": (
                    "okx_rest_market_ticker_fallback"
                    if index_row.get("fallback")
                    else "okx_rest_market_index_tickers"
                ),
                "source_channel": (
                    "tickers-fallback"
                    if index_row.get("fallback")
                    else "index-tickers"
                ),
                "source_timestamp_ms": int(self._safe_float(index_row.get("ts"))),
                "price": index_price,
            },
        }
        if fallback_used:
            payload["reference_prices_fallback"] = "ticker_last"
        if mark_price > 0 or index_price > 0:
            self._reference_price_cache[inst_id] = (time.monotonic(), payload)
        return payload

    async def fetch_derivatives_snapshot(
        self,
        symbol: str,
        *,
        contract_spec: dict[str, Any] | None = None,
        timeout_seconds: float = DERIVATIVES_SNAPSHOT_ROUTE_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Fetch compact perpetual-swap features used by AI experts."""
        route_tasks = {
            "funding": asyncio.create_task(
                self._cached_derivatives_route(
                    "funding",
                    symbol,
                    lambda: self.fetch_funding_rate(symbol),
                )
            ),
            "open_interest": asyncio.create_task(
                self._cached_derivatives_route(
                    "open_interest",
                    symbol,
                    lambda: self.fetch_open_interest(symbol),
                )
            ),
            "orderbook": asyncio.create_task(
                self.fetch_order_book_metrics(symbol, contract_spec=contract_spec)
            ),
            "reference_prices": asyncio.create_task(
                self.fetch_reference_prices(symbol, contract_spec=contract_spec)
            ),
        }
        task_names = {task: name for name, task in route_tasks.items()}
        done, pending = await asyncio.wait(
            set(task_names),
            timeout=max(float(timeout_seconds or 0.0), 0.05),
        )
        timed_out_routes = sorted(task_names[task] for task in pending)
        for task in pending:
            task.cancel()
            task.add_done_callback(_consume_cancelled_task)
        if pending:
            await asyncio.sleep(0)

        merged: dict[str, Any] = {}
        failed_routes: list[str] = []
        completed_routes: list[str] = []
        for task in done:
            route_name = task_names[task]
            try:
                result = task.result()
            except Exception:
                failed_routes.append(route_name)
                continue
            if isinstance(result, dict):
                merged.update(result)
                if result.get(f"{route_name}_timeout"):
                    timed_out_routes.append(route_name)
                else:
                    completed_routes.append(route_name)
            else:
                failed_routes.append(route_name)
        timed_out_routes = sorted(set(timed_out_routes))

        merged["derivatives_route_status"] = {
            "completed": sorted(completed_routes),
            "failed": sorted(failed_routes),
            "timed_out": timed_out_routes,
        }
        merged["derivatives_snapshot_partial"] = bool(
            failed_routes
            or timed_out_routes
            or merged.get("orderbook_data_available") is False
            or merged.get("reference_prices_fallback")
        )
        if timed_out_routes:
            logger.warning(
                "OKX derivatives snapshot returned partial data within deadline",
                symbol=symbol,
                timed_out_routes=timed_out_routes,
                completed_routes=sorted(completed_routes),
                timeout_seconds=round(max(float(timeout_seconds or 0.0), 0.05), 3),
            )
        return merged

    async def fetch_balance(self) -> dict:
        return await self._ccxt_call("fetch_balance")

    async def fetch_positions(self, symbols: list[str] | None = None) -> list[dict]:
        target_inst_ids = self._target_inst_ids(symbols)
        params: dict[str, str] = {"instType": "SWAP"}
        if len(target_inst_ids) == 1:
            params["instId"] = next(iter(target_inst_ids))
        response = await self._ccxt_call("privateGetAccountPositions", params)
        rows = response.get("data", []) if isinstance(response, dict) else []
        position_rows = [
            row
            for row in rows
            if isinstance(row, dict)
            and self._native_position_row_is_open(row)
            and self._row_inst_id_matches(row, target_inst_ids)
        ]
        position_inst_ids = {
            str(row.get("instId") or "").strip().upper()
            for row in position_rows
            if str(row.get("instId") or "").strip()
        }
        spec_results = await asyncio.gather(
            *(self.fetch_instrument_spec(inst_id) for inst_id in position_inst_ids),
            return_exceptions=True,
        )
        specs = {
            inst_id: result
            for inst_id, result in zip(position_inst_ids, spec_results, strict=True)
            if isinstance(result, dict) and result
        }
        return [
            self._native_position_to_ccxt_shape(
                row,
                contract_spec=specs.get(
                    str(row.get("instId") or "").strip().upper(),
                    {},
                ),
            )
            for row in position_rows
        ]

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> dict:
        return await self._ccxt_call(
            "create_order", symbol, order_type, side, amount, price, params or {}
        )

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        inst_id = okx_inst_id_from_symbol(symbol)
        if not inst_id:
            return {}
        return await self._ccxt_call(
            "privatePostTradeCancelOrder",
            {"instId": inst_id, "ordId": order_id},
        )

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        inst_id = okx_inst_id_from_symbol(symbol)
        if not inst_id:
            return {}
        response = await self._ccxt_call(
            "privateGetTradeOrder",
            {"instId": inst_id, "ordId": order_id},
        )
        rows = response.get("data", []) if isinstance(response, dict) else []
        row = rows[0] if rows else {}
        return self._native_order_to_ccxt_shape(row) if isinstance(row, dict) and row else {}

    async def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        target_inst_ids = self._target_inst_ids([symbol] if symbol else None)
        params_list = [
            {"instType": "SWAP", "instId": inst_id, "limit": "100"}
            for inst_id in sorted(target_inst_ids)
        ] or [{"instType": "SWAP", "limit": "100"}]
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for params in params_list:
            response = await self._ccxt_call("privateGetTradeOrdersPending", params)
            for row in response.get("data", []) if isinstance(response, dict) else []:
                if not isinstance(row, dict) or not self._row_inst_id_matches(
                    row,
                    target_inst_ids,
                ):
                    continue
                order_id = str(row.get("ordId") or row.get("clOrdId") or "").strip()
                if not order_id or order_id in seen:
                    continue
                seen.add(order_id)
                rows.append(row)
        return [self._native_order_to_ccxt_shape(row) for row in rows]

    async def get_available_symbols(self) -> list[dict[str, Any]]:
        """Return active OKX USDT linear perpetual swaps."""
        try:
            ex = await self._get_exchange()
            tickers = {}
            try:
                tickers = await self.fetch_tickers()
            except Exception as exc:
                logger.warning(
                    "fetch all tickers failed, using market metadata only",
                    error=safe_error_text(exc),
                )
            symbols = []
            for market_id, market in ex.markets.items():
                if not self._is_usdt_linear_swap(market):
                    continue
                symbol = self._display_symbol(market)
                if not symbol:
                    continue
                if symbol.split("/")[0] in {"XAU", "XAG"}:
                    continue
                if _is_suspicious_contract_base(market.get("base", "")):
                    continue
                if market.get("base", "").endswith(("1", "2", "3")) and market.get("base", "")[
                    :-1
                ] in {"AMZN", "ASTER"}:
                    continue
                ticker = self._ticker_for_market(tickers, market, symbol)
                ticker_info = ticker.get("info") or {}
                info = market.get("info") or {}
                last = self._safe_float(ticker.get("last") or ticker_info.get("last"))
                volume_fields = okx_swap_volume_fields(ticker or info, last)
                volume = self._safe_float(
                    volume_fields.get("volume_24h_base")
                    or ticker.get("baseVolume")
                    or ticker_info.get("vol24h")
                    or info.get("vol24h")
                )
                base_volume = self._safe_float(volume_fields.get("volume_24h_base") or volume)
                open_24h = self._safe_float(
                    ticker.get("open") or ticker_info.get("open24h") or ticker_info.get("sodUtc8")
                )
                high_24h = self._safe_float(ticker.get("high") or ticker_info.get("high24h"))
                low_24h = self._safe_float(ticker.get("low") or ticker_info.get("low24h"))
                change_pct = self._safe_float(ticker.get("percentage"))
                if change_pct == 0 and open_24h:
                    change_pct = (last - open_24h) / open_24h * 100
                spread_pct = self._spread_pct(ticker, ticker_info, last)
                range_pct = (
                    ((high_24h - low_24h) / last * 100) if last and high_24h > low_24h else 0.0
                )
                activity_score = self._market_activity_score(
                    volume=volume,
                    base_volume=base_volume,
                    change_pct=change_pct,
                    range_pct=range_pct,
                    spread_pct=spread_pct,
                    base=market.get("base", ""),
                )
                symbols.append(
                    {
                        "symbol": symbol,
                        "base": market.get("base", ""),
                        "quote": "USDT",
                        "type": "swap",
                        "id": market.get("id", market_id),
                        "ccxt_symbol": market.get("symbol", market_id),
                        "volume_24h": volume,
                        "base_volume_24h": base_volume,
                        **volume_fields,
                        "change_24h_pct": change_pct,
                        "range_24h_pct": range_pct,
                        "spread_pct": spread_pct,
                        "activity_score": activity_score,
                    }
                )
            if symbols:
                return self._sort_symbols(symbols)
        except Exception as exc:
            logger.warning(
                "OKX symbol discovery failed; using fallback list",
                error=safe_error_text(exc),
            )

        # Fallback: common OKX USDT perpetual swaps.
        common = [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "BNB/USDT",
            "DOGE/USDT",
            "ADA/USDT",
            "AVAX/USDT",
            "LINK/USDT",
            "SUI/USDT",
            "LTC/USDT",
            "BCH/USDT",
            "DOT/USDT",
            "TRX/USDT",
            "TON/USDT",
            "APT/USDT",
            "ARB/USDT",
            "OP/USDT",
            "NEAR/USDT",
            "FIL/USDT",
        ]
        return [
            {
                "symbol": s,
                "base": s.split("/")[0],
                "quote": "USDT",
                "type": "swap",
                "id": f"{s.split('/')[0]}-USDT-SWAP",
                "ccxt_symbol": f"{s}:USDT",
            }
            for s in common
        ]

    def _is_usdt_linear_swap(self, market: dict[str, Any]) -> bool:
        return bool(
            market.get("active", False)
            and market.get("type") == "swap"
            and market.get("swap") is True
            and market.get("linear") is True
            and market.get("settle") == "USDT"
            and market.get("quote") == "USDT"
        )

    def _display_symbol(self, market: dict[str, Any]) -> str:
        return symbol_from_okx_market(market)

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _ticker_for_market(
        self,
        tickers: dict[str, Any],
        market: dict[str, Any],
        display_symbol: str,
    ) -> dict[str, Any]:
        if not tickers:
            return {}
        keys = [
            market.get("symbol"),
            market.get("id"),
            market.get("ccxt_symbol"),
            self._to_swap_symbol(display_symbol),
            display_symbol,
        ]
        for key in keys:
            if key and key in tickers and isinstance(tickers[key], dict):
                return tickers[key]
        market_id = str(market.get("id") or "")
        for ticker in tickers.values():
            if not isinstance(ticker, dict):
                continue
            info = ticker.get("info") or {}
            if info.get("instId") == market_id:
                return ticker
        return {}

    def _native_ticker_to_ccxt_shape(self, row: dict[str, Any]) -> dict[str, Any]:
        info = dict(row)
        inst_id = str(info.get("instId") or "").strip().upper()
        symbol = symbol_from_okx_inst_id(inst_id)
        last = self._safe_float(info.get("last"))
        open_24h = self._safe_float(info.get("open24h") or info.get("sodUtc8"))
        high_24h = self._safe_float(info.get("high24h"))
        low_24h = self._safe_float(info.get("low24h"))
        bid = self._safe_float(info.get("bidPx"))
        ask = self._safe_float(info.get("askPx"))
        base_volume = self._safe_float(info.get("volCcy24h"))
        contract_volume = self._safe_float(info.get("vol24h"))
        quote_volume = self._safe_float(info.get("volCcyQuote24h"))
        if quote_volume <= 0 and base_volume > 0 and last > 0:
            quote_volume = base_volume * last
        timestamp = int(self._safe_float(info.get("ts")))
        percentage = ((last - open_24h) / open_24h * 100) if last > 0 and open_24h > 0 else 0.0
        return {
            "symbol": symbol,
            "id": inst_id,
            "last": last,
            "last_price": last,
            "price": last,
            "close": last,
            "open": open_24h,
            "high": high_24h,
            "low": low_24h,
            "bid": bid,
            "ask": ask,
            "baseVolume": base_volume,
            "quoteVolume": quote_volume,
            "volume_24h_contracts": contract_volume,
            "volume_24h_base": base_volume,
            "volume_24h_quote": quote_volume,
            "percentage": percentage,
            "timestamp": timestamp,
            "info": info,
        }

    def _target_inst_ids(self, symbols: list[str] | None) -> set[str]:
        return {
            inst_id
            for symbol in symbols or []
            if symbol and (inst_id := okx_inst_id_from_symbol(symbol))
        }

    def _row_inst_id_matches(self, row: dict[str, Any], target_inst_ids: set[str]) -> bool:
        if not target_inst_ids:
            return True
        return str(row.get("instId") or "").strip().upper() in target_inst_ids

    def _native_position_row_is_open(self, row: dict[str, Any]) -> bool:
        return abs(self._safe_float(row.get("pos"))) > 0

    def _native_position_to_ccxt_shape(
        self,
        row: dict[str, Any],
        *,
        contract_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        info = dict(row)
        spec = contract_spec if isinstance(contract_spec, dict) else {}
        inst_id = str(info.get("instId") or "").strip().upper()
        pos = self._safe_float(info.get("pos"))
        pos_side = str(info.get("posSide") or "").lower().strip()
        if pos_side == "net":
            side = "short" if pos < 0 else "long" if pos > 0 else ""
        else:
            side = pos_side
        contract_size = self._safe_float(spec.get("ctVal") or info.get("ctVal")) * self._safe_float(
            spec.get("ctMult") or info.get("ctMult") or 1.0
        )
        return {
            "id": str(info.get("posId") or ""),
            "symbol": inst_id or symbol_from_okx_inst_id(inst_id),
            "side": side,
            "contracts": abs(pos),
            "contractSize": contract_size if contract_size > 0 else None,
            "contractSizeSource": (
                str(spec.get("source") or "okx_public_instruments")
                if spec
                else "private_position_row"
            ),
            "contractSpec": dict(spec),
            "markPrice": self._safe_float(info.get("markPx")),
            "entryPrice": self._safe_float(info.get("avgPx")),
            "unrealizedPnl": self._safe_float(info.get("upl")),
            "leverage": self._safe_float(info.get("lever")),
            "marginMode": info.get("mgnMode"),
            "notional": self._safe_float(info.get("notionalUsd") or info.get("notional")),
            "liquidationPrice": self._safe_float(info.get("liqPx")),
            "timestamp": self._safe_float(info.get("uTime") or info.get("cTime")),
            "info": info,
        }

    def _native_order_to_ccxt_shape(self, row: dict[str, Any]) -> dict[str, Any]:
        info = dict(row)
        order_id = str(info.get("ordId") or info.get("clOrdId") or "").strip()
        amount = self._safe_float(info.get("sz"))
        filled = self._safe_float(info.get("accFillSz") or info.get("fillSz"))
        return {
            "id": order_id,
            "clientOrderId": str(info.get("clOrdId") or ""),
            "symbol": str(info.get("instId") or "").strip().upper(),
            "type": str(info.get("ordType") or "").lower(),
            "side": str(info.get("side") or "").lower(),
            "status": self._native_order_status(str(info.get("state") or "")),
            "amount": amount,
            "filled": filled,
            "remaining": max(amount - filled, 0.0),
            "price": self._safe_float(info.get("px")),
            "average": self._safe_float(info.get("avgPx")) or None,
            "reduceOnly": self._native_bool(info.get("reduceOnly")),
            "timestamp": self._safe_float(info.get("cTime")),
            "lastTradeTimestamp": self._safe_float(info.get("uTime")),
            "info": info,
        }

    def _native_order_status(self, state: str) -> str:
        return {
            "live": "open",
            "partially_filled": "partially_filled",
            "filled": "closed",
            "canceled": "canceled",
            "cancelled": "canceled",
        }.get(str(state or "").lower(), state or "open")

    def _native_bool(self, value: Any) -> bool | None:
        text = str(value or "").lower().strip()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        return None

    def _spread_pct(self, ticker: dict[str, Any], info: dict[str, Any], last: float) -> float:
        bid = self._safe_float(ticker.get("bid") or info.get("bidPx"))
        ask = self._safe_float(ticker.get("ask") or info.get("askPx"))
        mid = (bid + ask) / 2 if bid and ask else last
        if not mid or not bid or not ask or ask < bid:
            return 0.0
        return (ask - bid) / mid * 100

    def _market_activity_score(
        self,
        *,
        volume: float,
        base_volume: float,
        change_pct: float,
        range_pct: float,
        spread_pct: float,
        base: str,
    ) -> float:
        """Rank liquid and active contracts before spending AI tokens."""
        liquidity = math.log10(max(volume, 0.0) + 1.0) * 13.0
        participation = math.log10(max(base_volume, 0.0) + 1.0) * 3.0
        movement = min(abs(change_pct), 12.0) * 2.0
        tradable_range = min(max(range_pct, 0.0), 18.0) * 1.4
        spread_penalty = min(max(spread_pct, 0.0), 1.0) * 35.0
        blue_chip_bonus = (
            14.0
            if base
            in {
                "BTC",
                "ETH",
                "SOL",
                "XRP",
                "BNB",
                "DOGE",
                "ADA",
                "AVAX",
                "LINK",
                "SUI",
                "LTC",
                "BCH",
                "DOT",
                "TRX",
                "TON",
                "NEAR",
                "APT",
                "OP",
                "ARB",
            }
            else 0.0
        )
        extreme_range_penalty = 18.0 if range_pct > 45.0 else 0.0
        tiny_price_penalty = 8.0 if base in {"SHIB", "FLOKI", "CAT", "PEPE"} else 0.0
        return (
            liquidity
            + participation
            + movement
            + tradable_range
            + blue_chip_bonus
            - spread_penalty
            - extreme_range_penalty
            - tiny_price_penalty
        )

    def _to_swap_symbol(self, symbol: str) -> str:
        normalized = (symbol or "").strip().upper().replace("_", "-")
        if not normalized or ":" in normalized:
            return symbol
        resolved = self._resolve_loaded_swap_symbol(symbol)
        if resolved:
            return resolved
        if normalized.endswith("-SWAP"):
            base = normalized.removesuffix("-USDT-SWAP")
            return f"{base}/USDT:USDT"
        if "/" in normalized:
            base, quote = normalized.split("/", 1)
            quote = quote.split(":")[0]
            if quote == "USDT":
                return f"{base}/USDT:USDT"
        if "-" in normalized:
            parts = normalized.split("-")
            if len(parts) >= 2 and parts[1] == "USDT":
                return f"{parts[0]}/USDT:USDT"
        return symbol

    def _resolve_loaded_swap_symbol(self, symbol: str) -> str | None:
        if self._exchange is None:
            return None
        markets_by_id = getattr(self._exchange, "markets_by_id", None) or {}
        native = normalize_trading_symbol(symbol).replace("/", "-")
        if not native:
            return None
        by_id = markets_by_id.get(f"{native}-SWAP")
        markets = by_id if isinstance(by_id, list) else [by_id] if by_id else []
        for market in markets:
            if isinstance(market, dict) and market.get("symbol"):
                return str(market["symbol"])
        return None

    def _sort_symbols(self, symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped = {s["symbol"]: s for s in symbols}
        return sorted(
            deduped.values(),
            key=lambda s: (
                -float(s.get("activity_score") or 0),
                -float(s.get("volume_24h") or 0),
                -abs(float(s.get("change_24h_pct") or 0)),
                s.get("symbol", ""),
            ),
        )

    async def reinitialize(self) -> None:
        """Close and recreate the exchange with current settings."""
        await self.close()
        self._exchange = None
        self._exchange_ready = False

    async def close(self) -> None:
        ticker_tasks = list(self._ticker_tasks.values())
        for task in ticker_tasks:
            if not task.done():
                task.cancel()
        if ticker_tasks:
            await asyncio.gather(*ticker_tasks, return_exceptions=True)
        self._ticker_tasks.clear()
        self._ticker_retry_at.clear()
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
        self._exchange_ready = False
