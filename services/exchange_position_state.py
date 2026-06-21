from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

import structlog

from core.safe_output import safe_error_text

logger = structlog.get_logger(__name__)


def _first_float(*values: Any, default: float | None = 0.0) -> float | None:
    for value in values:
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _first_positive_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        parsed = _first_float(value, default=None)
        if parsed is not None and parsed > 0:
            return parsed
    return default


def _first_nonzero_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        parsed = _first_float(value, default=None)
        if parsed is not None and parsed != 0:
            return parsed
    return default


def exchange_snapshot_price(snapshot: dict[str, Any]) -> float:
    return (
        _first_positive_float(
            snapshot.get("mark_price"),
            snapshot.get("last_price"),
            snapshot.get("index_price"),
            default=0.0,
        )
        or 0.0
    )


def exchange_snapshot_quantity(snapshot: dict[str, Any]) -> float:
    quantity = _first_positive_float(snapshot.get("quantity"), default=0.0)
    if quantity > 0:
        return quantity
    contracts = abs(_first_nonzero_float(snapshot.get("contracts"), default=0.0))
    contract_size = abs(_first_positive_float(snapshot.get("contract_size"), default=1.0) or 1.0)
    if contracts <= 0:
        return 0.0
    return contracts * contract_size


def exchange_snapshot_unrealized(snapshot: dict[str, Any], side: str) -> float:
    snapshot_upl = _first_float(snapshot.get("upl"), default=None)
    if snapshot_upl is not None:
        return snapshot_upl
    mark_price = exchange_snapshot_price(snapshot)
    entry_price = _first_positive_float(snapshot.get("entry_price"), default=0.0)
    quantity = exchange_snapshot_quantity(snapshot)
    if mark_price <= 0 or entry_price <= 0 or quantity <= 0:
        return 0.0
    if side == "short":
        return (entry_price - mark_price) * quantity
    return (mark_price - entry_price) * quantity


def exchange_position_display_valuation(
    snapshot: dict[str, Any],
    side: str,
    *,
    fallback_current_price: Any,
    fallback_unrealized_pnl: Any,
    fallback_entry_price: Any,
    fallback_quantity: Any,
) -> dict[str, Any]:
    current_price = exchange_snapshot_price(snapshot)
    entry_price = _first_positive_float(snapshot.get("entry_price"), default=0.0)
    quantity = exchange_snapshot_quantity(snapshot)
    snapshot_upl = _first_float(snapshot.get("upl"), default=None)
    fallback_pnl = _first_float(fallback_unrealized_pnl, default=0.0) or 0.0

    if current_price <= 0:
        current_price = _first_positive_float(fallback_current_price, default=0.0)
    if entry_price <= 0:
        entry_price = _first_positive_float(fallback_entry_price, default=0.0)
    if quantity <= 0:
        quantity = abs(_first_float(fallback_quantity, default=0.0) or 0.0)

    if snapshot_upl is not None:
        unrealized_pnl = snapshot_upl
        pnl_source = "okx_position_upl"
    elif current_price > 0 and entry_price > 0 and quantity > 0:
        if side == "short":
            unrealized_pnl = (entry_price - current_price) * quantity
        else:
            unrealized_pnl = (current_price - entry_price) * quantity
        pnl_source = "okx_position_mark_recomputed"
    else:
        unrealized_pnl = fallback_pnl
        pnl_source = "local_db"

    return {
        "current_price": current_price,
        "entry_price": entry_price,
        "quantity": quantity,
        "unrealized_pnl": unrealized_pnl,
        "pnl_source": pnl_source,
    }


def parse_exchange_position_snapshot(
    position: dict[str, Any],
    *,
    symbol_normalizer: Callable[[Any], str],
) -> dict[str, Any] | None:
    if not ExchangePositionStatePolicy.is_open(position):
        return None
    info = position.get("info") or {}
    symbol = symbol_normalizer(position.get("symbol") or info.get("instId"))
    side = str(position.get("side") or info.get("posSide") or "").lower()
    if not symbol or side not in {"long", "short"}:
        return None

    mark_price = _first_positive_float(
        position.get("markPrice"),
        position.get("mark_price"),
        info.get("markPx"),
        info.get("mark_price"),
        default=0.0,
    )
    last_price = _first_positive_float(
        position.get("lastPrice"),
        position.get("last_price"),
        info.get("last"),
        info.get("lastPx"),
        info.get("last_price"),
        default=0.0,
    )
    index_price = _first_positive_float(info.get("idxPx"), info.get("indexPx"), default=0.0)
    upl = _first_float(
        info.get("upl"),
        position.get("unrealizedPnl"),
        position.get("unrealized_pnl"),
        info.get("unrealizedPnl"),
        default=None,
    )
    if mark_price <= 0 and last_price <= 0 and index_price <= 0 and upl is None:
        return None

    contracts = abs(
        _first_nonzero_float(
            position.get("contracts"),
            position.get("size"),
            position.get("positionAmt"),
            info.get("pos"),
            info.get("qty"),
            default=0.0,
        )
        or 0.0
    )
    contract_size = abs(
        _first_positive_float(
            position.get("contractSize"),
            position.get("contract_size"),
            info.get("ctVal"),
            info.get("contractSize"),
            default=1.0,
        )
        or 1.0
    )
    quantity = abs(
        _first_positive_float(
            position.get("quantity"),
            position.get("baseVolume"),
            info.get("baseBal"),
            default=0.0,
        )
        or 0.0
    )
    if quantity <= 0 and contracts > 0:
        quantity = contracts * contract_size
    entry_price = _first_positive_float(
        position.get("entryPrice"),
        position.get("entry_price"),
        info.get("avgPx"),
        info.get("avg_price"),
        default=0.0,
    )
    margin_used = (
        _first_positive_float(position.get("initialMargin"), default=0.0)
        or _first_positive_float(position.get("margin"), default=0.0)
        or _first_positive_float(info.get("imr"), default=0.0)
        or _first_positive_float(info.get("margin"), default=0.0)
    )

    return {
        "symbol": symbol,
        "side": side,
        "mark_price": mark_price,
        "last_price": last_price,
        "index_price": index_price,
        "upl": upl,
        "entry_price": entry_price,
        "contracts": contracts,
        "contract_size": contract_size,
        "quantity": quantity,
        "margin_used": margin_used,
        "raw_symbol": position.get("symbol") or info.get("instId"),
    }


def _default_float_parser(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ExchangePositionStatePolicy:
    """Interpret exchange position snapshots without coupling callers to OKX shapes."""

    @staticmethod
    def is_open(position: dict[str, Any]) -> bool:
        info = position.get("info") or {}
        raw_size = (
            position.get("contracts")
            or position.get("size")
            or position.get("positionAmt")
            or info.get("pos")
            or info.get("qty")
            or 0
        )
        try:
            return abs(float(raw_size)) > 0
        except (TypeError, ValueError):
            return bool(position.get("symbol"))


@dataclass(slots=True)
class _ProtectionCacheEntry:
    orders: list[dict[str, Any]]
    expires_at: float


class ExchangeProtectionMapProvider:
    """Fetch OKX TP/SL algo orders keyed by normalized symbol and position side."""

    def __init__(
        self,
        *,
        symbol_normalizer: Callable[[Any], str],
        position_open_checker: Callable[[dict[str, Any]], bool],
        float_parser: Callable[[Any, float], float] | None = None,
        timeout_seconds: float = 2.5,
        cache_ttl_seconds: float = 30.0,
    ) -> None:
        self.symbol_normalizer = symbol_normalizer
        self.position_open_checker = position_open_checker
        self.float_parser = float_parser or _default_float_parser
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._cache: dict[str, _ProtectionCacheEntry] = {}

    async def fetch(
        self,
        executor: Any,
        exchange_positions: list[dict[str, Any]],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        protection_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        symbols = {
            self.symbol_normalizer(position.get("symbol"))
            for position in exchange_positions or []
            if self.position_open_checker(position)
        }
        symbols.discard("")

        protection_results = await asyncio.gather(
            *(self._fetch_symbol_orders(executor, symbol) for symbol in symbols),
            return_exceptions=False,
        )

        for _symbol, orders in protection_results:
            for order in orders or []:
                key = (
                    self.symbol_normalizer(order.get("symbol")),
                    str(order.get("position_side") or "").lower(),
                )
                if not key[0] or key[1] not in {"long", "short"}:
                    continue

                existing = protection_by_key.get(key)
                if existing and self.float_parser(
                    existing.get("updated_at_ms"), 0.0
                ) > self.float_parser(order.get("updated_at_ms"), 0.0):
                    continue
                protection_by_key[key] = order

        return protection_by_key

    async def _fetch_symbol_orders(self, executor: Any, symbol: str) -> tuple[str, list[dict]]:
        now = time.monotonic()
        cached = self._cache.get(symbol)
        if cached and cached.expires_at > now:
            return symbol, list(cached.orders)

        try:
            orders = await asyncio.wait_for(
                executor.get_position_protection_orders(symbol),
                timeout=self.timeout_seconds,
            )
            normalized_orders = list(orders or [])
            if self.cache_ttl_seconds > 0:
                self._cache[symbol] = _ProtectionCacheEntry(
                    orders=normalized_orders,
                    expires_at=now + self.cache_ttl_seconds,
                )
            return symbol, normalized_orders
        except TimeoutError:
            logger.warning(
                "timed out fetching OKX TP/SL protection orders",
                symbol=symbol,
            )
            if cached:
                return symbol, list(cached.orders)
            return symbol, []
        except Exception as exc:
            logger.warning(
                "failed to fetch OKX TP/SL protection orders",
                symbol=symbol,
                error=safe_error_text(exc),
            )
            if cached:
                return symbol, list(cached.orders)
            return symbol, []
