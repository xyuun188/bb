from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def proportional_value(value: float | None, close_qty: float, total_qty: float) -> float:
    amount = safe_float(value, 0.0)
    close = safe_float(close_qty, 0.0)
    total = safe_float(total_qty, 0.0)
    if amount == 0.0 or close <= 0:
        return 0.0
    if total <= 0:
        return amount
    return amount * min(close / total, 1.0)


def okx_fill_pnl_from_payload(payload: Any) -> float | None:
    """Return OKX-native fillPnl when a close fill is available.

    OKX ``fillPnl`` is the exchange-authoritative realised price PnL for the
    close fill, separate from fees.  It is safer than reconstructing PnL from
    local FIFO slices when OKX average position price differs from local rows.
    """

    if not isinstance(payload, dict):
        return None
    native = payload.get("native_close_fill")
    if isinstance(native, dict):
        for key in ("pnl", "fillPnl", "fill_pnl"):
            if key in native and native.get(key) is not None:
                return safe_float(native.get(key), 0.0)
    for key in ("pnl", "fillPnl", "fill_pnl"):
        if key in payload and payload.get(key) is not None:
            return safe_float(payload.get(key), 0.0)
    info = payload.get("info")
    if isinstance(info, dict):
        for key in ("fillPnl", "pnl", "fill_pnl"):
            if key in info and info.get(key) is not None:
                return safe_float(info.get(key), 0.0)
    return None


def okx_fill_quantity_from_payload(payload: Any) -> float:
    if not isinstance(payload, dict):
        return 0.0
    native = payload.get("native_close_fill")
    if isinstance(native, dict):
        qty = safe_float(native.get("quantity"), 0.0)
        if qty > 0:
            return qty
    for key in ("quantity", "base_quantity", "filled_base_quantity"):
        qty = safe_float(payload.get(key), 0.0)
        if qty > 0:
            return qty
    return 0.0


def gross_pnl_with_okx_override(
    *,
    side: str,
    entry_price: float,
    exit_price: float,
    close_qty: float,
    okx_payload: Any = None,
    okx_total_qty: float | None = None,
) -> tuple[float, str]:
    """Calculate gross close PnL, preferring OKX-native fillPnl.

    Returns ``(gross_pnl, source)`` where source is either ``okx_fill_pnl`` or
    ``local_price_formula``.
    """

    okx_fill_pnl = okx_fill_pnl_from_payload(okx_payload)
    if okx_fill_pnl is not None:
        total_qty = safe_float(okx_total_qty, 0.0)
        if total_qty <= 0:
            total_qty = okx_fill_quantity_from_payload(okx_payload)
        return proportional_value(okx_fill_pnl, close_qty, total_qty), "okx_fill_pnl"

    if str(side or "").lower() == "short":
        return (safe_float(entry_price) - safe_float(exit_price)) * safe_float(close_qty), "local_price_formula"
    return (safe_float(exit_price) - safe_float(entry_price)) * safe_float(close_qty), "local_price_formula"
