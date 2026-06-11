from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import structlog

from core.safe_output import safe_error_text

logger = structlog.get_logger(__name__)


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


class ExchangeProtectionMapProvider:
    """Fetch OKX TP/SL algo orders keyed by normalized symbol and position side."""

    def __init__(
        self,
        *,
        symbol_normalizer: Callable[[Any], str],
        position_open_checker: Callable[[dict[str, Any]], bool],
        float_parser: Callable[[Any, float], float] | None = None,
        timeout_seconds: float = 2.5,
    ) -> None:
        self.symbol_normalizer = symbol_normalizer
        self.position_open_checker = position_open_checker
        self.float_parser = float_parser or _default_float_parser
        self.timeout_seconds = timeout_seconds

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
        try:
            orders = await asyncio.wait_for(
                executor.get_position_protection_orders(symbol),
                timeout=self.timeout_seconds,
            )
            return symbol, orders or []
        except TimeoutError:
            logger.warning(
                "timed out fetching OKX TP/SL protection orders",
                symbol=symbol,
            )
            return symbol, []
        except Exception as exc:
            logger.warning(
                "failed to fetch OKX TP/SL protection orders",
                symbol=symbol,
                error=safe_error_text(exc),
            )
            return symbol, []
