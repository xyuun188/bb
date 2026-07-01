from __future__ import annotations

from typing import Any

from sqlalchemy import inspect as sqlalchemy_inspect

from services.manual_close_marker import (
    ORPHAN_QUARANTINE_EXCHANGE_ID_PREFIX,
    is_manual_close_exchange_order_id,
)
from services.okx_order_fact_sync import (
    OKX_SYNC_CONFIRMED,
    OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
    OKX_SYNC_OKX_ONLY,
)

TRUSTED_OKX_ORDER_SYNC_STATUSES = {
    OKX_SYNC_CONFIRMED,
    OKX_SYNC_OKX_ONLY,
    OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return str(value or "").strip()


def split_exchange_order_ids(value: Any) -> list[str]:
    tokens = {_text(value)}
    if not next(iter(tokens), ""):
        return []
    for separator in (",", ";", "|", "\n", "\t", " "):
        pieces: set[str] = set()
        for token in tokens:
            pieces.update(part.strip() for part in token.split(separator) if part.strip())
        tokens = pieces
    return [token for token in tokens if token]


def _has_fact_link_fields(position: Any) -> bool:
    return hasattr(position, "entry_exchange_order_id") or hasattr(
        position, "close_exchange_order_id"
    )


def _should_enforce_fact_links(position: Any) -> bool:
    if not _has_fact_link_fields(position):
        return False
    try:
        state = sqlalchemy_inspect(position, raiseerr=False)
    except Exception:
        state = None
    if state is not None and getattr(state, "transient", False):
        data = getattr(position, "__dict__", {}) or {}
        return "entry_exchange_order_id" in data or "close_exchange_order_id" in data
    return True


def closed_position_trade_fact_untrusted_reason(position: Any) -> str | None:
    """Return why a closed position must not be used as a trusted trade fact.

    The SQLAlchemy Position model now carries OKX order-link fields.  Historical
    rows without those links can still be shown/audited, but they must not drive
    model training, strategy-learning weights, or PnL attribution.  Plain test
    doubles that do not expose these fields are treated as legacy in-memory
    objects so pure logic tests do not need database-only attributes.
    """

    if bool(getattr(position, "is_open", False)):
        return None
    if not _should_enforce_fact_links(position):
        return None
    if not _text(getattr(position, "entry_exchange_order_id", None)):
        return "missing_entry_exchange_order_id"
    close_exchange_order_id = getattr(position, "close_exchange_order_id", None)
    close_exchange_text = _text(close_exchange_order_id)
    if is_manual_close_exchange_order_id(close_exchange_order_id):
        return "manual_close_exchange_order_id"
    if close_exchange_text.startswith(ORPHAN_QUARANTINE_EXCHANGE_ID_PREFIX):
        return "orphan_quarantine_exchange_order_id"
    realized_pnl = _as_float(getattr(position, "realized_pnl", None), 0.0)
    if realized_pnl != 0.0 and not close_exchange_text:
        return "missing_close_exchange_order_id"
    return None


def closed_position_trade_fact_trusted(position: Any) -> bool:
    return closed_position_trade_fact_untrusted_reason(position) is None


def closed_position_trade_fact_untrusted_reason_with_orders(
    position: Any,
    orders_by_exchange_id: dict[str, Any],
) -> str | None:
    """Return why a closed position is not backed by OKX-confirmed orders."""

    reason = closed_position_trade_fact_untrusted_reason(position)
    if reason is not None:
        return reason
    if bool(getattr(position, "is_open", False)):
        return None
    if not _should_enforce_fact_links(position):
        return None

    entry_ids = split_exchange_order_ids(getattr(position, "entry_exchange_order_id", None))
    close_ids = split_exchange_order_ids(getattr(position, "close_exchange_order_id", None))
    for order_id in entry_ids:
        order = orders_by_exchange_id.get(order_id)
        if not _order_is_okx_confirmed_execution(order):
            return "entry_order_not_okx_confirmed"
    for order_id in close_ids:
        order = orders_by_exchange_id.get(order_id)
        if not _order_is_okx_confirmed_execution(order):
            return "close_order_not_okx_confirmed"
    return None


def closed_position_trade_fact_trusted_with_orders(
    position: Any,
    orders_by_exchange_id: dict[str, Any],
) -> bool:
    return closed_position_trade_fact_untrusted_reason_with_orders(
        position,
        orders_by_exchange_id,
    ) is None


def orders_by_exchange_id(orders: list[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for order in orders:
        for order_id in split_exchange_order_ids(getattr(order, "exchange_order_id", None)):
            result.setdefault(order_id, order)
    return result


def _order_is_okx_confirmed_execution(order: Any) -> bool:
    if order is None:
        return False
    sync_status = _text(getattr(order, "okx_sync_status", None))
    return sync_status in TRUSTED_OKX_ORDER_SYNC_STATUSES


def filter_trusted_closed_positions(rows: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    trusted: list[Any] = []
    reason_counts: dict[str, int] = {}
    quarantined_ids: list[int] = []
    for row in rows:
        reason = closed_position_trade_fact_untrusted_reason(row)
        if reason is None:
            trusted.append(row)
            continue
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        row_id = int(getattr(row, "id", 0) or 0)
        if row_id > 0:
            quarantined_ids.append(row_id)
    return trusted, {
        "checked": len(rows),
        "trusted": len(trusted),
        "quarantined": len(rows) - len(trusted),
        "reason_counts": reason_counts,
        "position_ids": quarantined_ids[:50],
    }
