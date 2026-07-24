"""Authoritative OKX fill-mark execution slippage facts."""

from __future__ import annotations

from math import isclose, isfinite
from typing import Any

OKX_FILL_MARK_SLIPPAGE_VERSION = "2026-07-24.okx-fill-mark-slippage.v2"
OKX_FILL_MARK_SLIPPAGE_SOURCE = "okx_fills_history_fill_mark_vwap"
OKX_ROUND_TRIP_SLIPPAGE_SOURCE = "okx_fills_history_fill_mark_round_trip"


def build_okx_fill_mark_slippage(
    *,
    order_id: Any,
    inst_id: Any,
    side: Any,
    contracts: Any,
    average_price: Any,
    contract_size: Any,
    rows: Any,
) -> dict[str, Any]:
    expected_order_id = str(order_id or "").strip()
    expected_inst_id = str(inst_id or "").strip().upper()
    expected_side = str(side or "").strip().lower()
    expected_contracts = _finite_float(contracts)
    expected_average_price = _finite_float(average_price)
    public_contract_size = _finite_float(contract_size)
    fill_rows = [dict(row) for row in rows or [] if isinstance(row, dict)]
    reasons: list[str] = []

    if not expected_order_id:
        reasons.append("order_id_missing")
    if not expected_inst_id.endswith("-SWAP"):
        reasons.append("instrument_id_invalid")
    if expected_side not in {"buy", "sell"}:
        reasons.append("fill_side_invalid")
    if expected_contracts is None or expected_contracts <= 0:
        reasons.append("fill_contracts_invalid")
    if expected_average_price is None or expected_average_price <= 0:
        reasons.append("fill_average_price_invalid")
    if public_contract_size is None or public_contract_size <= 0:
        reasons.append("public_contract_size_invalid")
    if not fill_rows:
        reasons.append("fill_rows_missing")

    row_contracts = 0.0
    mark_contracts = 0.0
    fill_price_value = 0.0
    mark_price_value = 0.0
    actual_notional_usdt = 0.0
    mark_notional_usdt = 0.0
    adverse_slippage_usdt = 0.0
    trade_ids: list[str] = []
    for row in fill_rows:
        row_order_id = str(row.get("ordId") or "").strip()
        row_inst_id = str(row.get("instId") or "").strip().upper()
        row_side = str(row.get("side") or "").strip().lower()
        trade_id = str(row.get("tradeId") or "").strip()
        row_size = _finite_float(row.get("fillSz") or row.get("sz"))
        fill_price = _finite_float(row.get("fillPx"))
        fill_mark_price = _finite_float(row.get("fillMarkPx"))
        if row_order_id != expected_order_id:
            reasons.append("fill_row_order_id_mismatch")
        if row_inst_id != expected_inst_id:
            reasons.append("fill_row_instrument_id_mismatch")
        if row_side != expected_side:
            reasons.append("fill_row_side_mismatch")
        if not trade_id:
            reasons.append("fill_row_trade_id_missing")
        if row_size is None or row_size <= 0:
            reasons.append("fill_row_contracts_invalid")
        if fill_price is None or fill_price <= 0:
            reasons.append("fill_row_price_invalid")
        if fill_mark_price is None or fill_mark_price <= 0:
            reasons.append("fill_row_mark_price_invalid")
        if trade_id:
            trade_ids.append(trade_id)
        if (
            row_size is None
            or row_size <= 0
            or fill_price is None
            or fill_price <= 0
            or public_contract_size is None
            or public_contract_size <= 0
        ):
            continue
        row_contracts += row_size
        fill_price_value += fill_price * row_size
        actual_notional_usdt += fill_price * row_size * public_contract_size
        if fill_mark_price is None or fill_mark_price <= 0:
            continue
        mark_contracts += row_size
        mark_price_value += fill_mark_price * row_size
        mark_notional_usdt += fill_mark_price * row_size * public_contract_size
        adverse_price_delta = (
            max(fill_price - fill_mark_price, 0.0)
            if expected_side == "buy"
            else max(fill_mark_price - fill_price, 0.0)
        )
        adverse_slippage_usdt += adverse_price_delta * row_size * public_contract_size

    if (
        expected_contracts is not None
        and expected_contracts > 0
        and not isclose(row_contracts, expected_contracts, rel_tol=1e-9, abs_tol=1e-12)
    ):
        reasons.append("fill_row_contract_total_mismatch")
    fill_vwap = fill_price_value / row_contracts if row_contracts > 0 else None
    fill_mark_vwap = mark_price_value / mark_contracts if mark_contracts > 0 else None
    if (
        fill_vwap is not None
        and expected_average_price is not None
        and expected_average_price > 0
        and not isclose(fill_vwap, expected_average_price, rel_tol=1e-9, abs_tol=1e-12)
    ):
        reasons.append("fill_row_vwap_mismatch")
    reasons = list(dict.fromkeys(reasons))
    complete = not reasons
    return {
        "version": OKX_FILL_MARK_SLIPPAGE_VERSION,
        "source": OKX_FILL_MARK_SLIPPAGE_SOURCE if complete else "",
        "complete": complete,
        "reasons": reasons,
        "order_id": expected_order_id or None,
        "inst_id": expected_inst_id or None,
        "side": expected_side or None,
        "trade_ids": sorted(set(trade_ids)),
        "fill_row_count": len(fill_rows),
        "contracts": _rounded(row_contracts),
        "contract_size": _rounded(public_contract_size),
        "fill_vwap": _rounded(fill_vwap),
        "fill_mark_vwap": _rounded(fill_mark_vwap),
        "actual_notional_usdt": _rounded(actual_notional_usdt),
        "fill_mark_notional_usdt": _rounded(mark_notional_usdt),
        "adverse_slippage_usdt": _rounded(adverse_slippage_usdt),
        "adverse_slippage_pct": _rounded(
            adverse_slippage_usdt / actual_notional_usdt * 100.0
            if actual_notional_usdt > 0
            else None
        ),
    }


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _rounded(value: float | None) -> float | None:
    return round(value, 12) if value is not None and isfinite(value) else None
