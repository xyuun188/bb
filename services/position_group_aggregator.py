"""Aggregate same-symbol same-side position fragments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text

NormalizeSymbol = Callable[[str | None], str | None]
FloatParser = Callable[[Any, float], float]

logger = structlog.get_logger(__name__)


def _default_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class PositionGroupAggregator:
    """Aggregate open-position fragments for review, profit protection, and peak tracking."""

    normalize_symbol: NormalizeSymbol
    float_parser: FloatParser = _default_float
    default_model_name: str = ENSEMBLE_TRADER_NAME

    def aggregate(
        self,
        positions: list[dict] | None,
        model_name: str,
        symbol: str,
        side: str,
    ) -> dict[str, Any]:
        """Aggregate same-side fragments into one weighted position row."""

        rows = [p for p in (positions or []) if str(p.get("side") or "").lower() == side]
        if not rows:
            return {}

        total_qty = 0.0
        entry_value = 0.0
        current_value = 0.0
        unrealized = 0.0
        stop_value = 0.0
        stop_weight = 0.0
        take_profit_value = 0.0
        take_profit_weight = 0.0
        leverage_value = 0.0
        leverage_weight = 0.0
        created_at = None

        for position in rows:
            qty = abs(self.float_parser(position.get("quantity"), 0.0))
            entry = self.float_parser(position.get("entry_price"), 0.0)
            current = self.float_parser(position.get("current_price"), entry)
            if qty <= 0 or entry <= 0:
                continue
            total_qty += qty
            entry_value += entry * qty
            current_value += (current if current > 0 else entry) * qty
            unrealized += self.float_parser(position.get("unrealized_pnl"), 0.0)

            stop = self.float_parser(
                position.get("stop_loss") or position.get("stop_loss_price"),
                0.0,
            )
            if stop > 0:
                stop_value += stop * qty
                stop_weight += qty

            take_profit = self.float_parser(
                position.get("take_profit") or position.get("take_profit_price"),
                0.0,
            )
            if take_profit > 0:
                take_profit_value += take_profit * qty
                take_profit_weight += qty

            leverage = self.float_parser(position.get("leverage"), 0.0)
            if leverage > 0:
                leverage_value += leverage * qty
                leverage_weight += qty

            created_at = self._earliest_created_at(
                created_at,
                position.get("created_at"),
                symbol=symbol,
            )

        if total_qty <= 0:
            return {}

        entry_price = entry_value / total_qty
        current_price = current_value / total_qty if current_value > 0 else entry_price
        notional = entry_price * total_qty
        return {
            "model_name": model_name or self.default_model_name,
            "symbol": self.normalize_symbol(symbol) or symbol,
            "side": side,
            "quantity": total_qty,
            "entry_price": entry_price,
            "current_price": current_price,
            "notional": notional,
            "unrealized_pnl": unrealized,
            "stop_loss": stop_value / stop_weight if stop_weight > 0 else 0.0,
            "take_profit": (
                take_profit_value / take_profit_weight if take_profit_weight > 0 else 0.0
            ),
            "leverage": leverage_value / leverage_weight if leverage_weight > 0 else 1.0,
            "is_open": True,
            "created_at": created_at,
            "rows": len(rows),
        }

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    def _earliest_created_at(self, current: Any, candidate: Any, *, symbol: str) -> Any:
        if current is None:
            return candidate
        if candidate is None:
            return current
        try:
            current_time = self._parse_time(current)
            candidate_time = self._parse_time(candidate)
            if (
                current_time is not None
                and candidate_time is not None
                and candidate_time < current_time
            ):
                return candidate
        except (TypeError, ValueError) as exc:
            logger.debug(
                "failed to compare aggregated position open time",
                symbol=symbol,
                error=safe_error_text(exc),
            )
        return current
