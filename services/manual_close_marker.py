from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

MANUAL_CLOSE_EXCHANGE_ID_PREFIX = "manual_close:"
ORPHAN_QUARANTINE_EXCHANGE_ID_PREFIX = "okx_orphan_quarantine:"
MANUAL_CLOSE_LABEL = "手动平仓"


def is_manual_close_exchange_order_id(value: Any) -> bool:
    return str(value or "").startswith(MANUAL_CLOSE_EXCHANGE_ID_PREFIX)


def is_local_non_exchange_close_marker(value: Any) -> bool:
    text = str(value or "")
    return text.startswith(MANUAL_CLOSE_EXCHANGE_ID_PREFIX) or text.startswith(
        ORPHAN_QUARANTINE_EXCHANGE_ID_PREFIX
    )


def is_manual_close_order(order: Any) -> bool:
    return is_manual_close_exchange_order_id(getattr(order, "exchange_order_id", None))


def normalized_symbol(value: Any) -> str:
    return str(value or "").replace("-", "/").replace("_", "/").upper()


def close_order_side_for_position_side(side: Any) -> str:
    return "buy" if str(side or "").lower() == "short" else "sell"


def _aware_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def manual_close_order_matches_position(
    order: Any,
    position: Any,
    *,
    max_seconds: float = 15.0,
) -> bool:
    if not is_manual_close_order(order):
        return False
    if getattr(order, "model_name", None) != getattr(position, "model_name", None):
        return False
    if getattr(order, "execution_mode", None) != getattr(position, "execution_mode", None):
        return False
    if normalized_symbol(getattr(order, "symbol", None)) != normalized_symbol(
        getattr(position, "symbol", None)
    ):
        return False

    order_side = str(getattr(order, "side", "") or "").lower()
    if order_side in {"buy", "sell"} and order_side != close_order_side_for_position_side(
        getattr(position, "side", None)
    ):
        return False

    closed_at = _aware_utc(getattr(position, "closed_at", None))
    filled_at = _aware_utc(getattr(order, "filled_at", None) or getattr(order, "created_at", None))
    if closed_at and filled_at and abs((closed_at - filled_at).total_seconds()) > max_seconds:
        return False

    order_price = _safe_float(getattr(order, "price", None), 0.0)
    close_price = _safe_float(getattr(position, "current_price", None), 0.0)
    if order_price > 0 and close_price > 0:
        tolerance = max(abs(close_price) * 0.002, 1e-8)
        if abs(order_price - close_price) > tolerance:
            return False

    return True


def position_has_manual_close_order(position: Any, orders: list[Any]) -> bool:
    return any(manual_close_order_matches_position(order, position) for order in orders)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
