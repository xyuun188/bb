"""Normalize immutable OKX close-fill facts before decision persistence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def normalize_external_close_fill_evidence(
    close_fill: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Carry complete fills-history identity into a reconciliation decision.

    The exchange sync writes the decision before it creates or updates the
    local order row.  This function copies only facts already present in the
    supplied OKX payload (including its nested fills-history row); it never
    invents a fill when the exchange did not provide one.
    """

    payload = {
        str(key): value.isoformat() if isinstance(value, datetime) else value
        for key, value in (close_fill or {}).items()
    }
    order_info = payload.get("order_info")
    order_info = order_info if isinstance(order_info, dict) else {}

    trade_ids = [
        str(value).strip()
        for value in (
            *(payload.get("trade_ids") or []),
            payload.get("trade_id"),
            payload.get("tradeId"),
            order_info.get("tradeId"),
        )
        if str(value or "").strip()
    ]
    if trade_ids:
        payload["trade_ids"] = list(dict.fromkeys(trade_ids))

    contracts = _safe_float(
        payload.get("contracts")
        or payload.get("fillSz")
        or order_info.get("fillSz")
        or order_info.get("accFillSz"),
        0.0,
    )
    contract_size = _safe_float(
        payload.get("contract_size")
        or payload.get("contractSize")
        or order_info.get("ctVal")
        or order_info.get("contractSize"),
        0.0,
    )
    if contracts > 0:
        payload.setdefault("contracts", contracts)
    if contract_size > 0:
        payload.setdefault("contract_size", contract_size)

    contract_size_source = str(payload.get("contract_size_source") or "").strip()
    if contract_size_source:
        payload.setdefault(
            "contract_size_verified",
            contract_size > 0 and contract_size_source == "okx_public_instruments",
        )

    base_quantity = _safe_float(
        payload.get("base_quantity")
        or payload.get("quantity")
        or (contracts * contract_size if contracts > 0 and contract_size > 0 else 0.0),
        0.0,
    )
    if base_quantity > 0:
        payload.setdefault("base_quantity", base_quantity)

    if not payload.get("avg_price"):
        avg_price = _safe_float(
            payload.get("price") or order_info.get("fillPx") or order_info.get("avgPx"),
            0.0,
        )
        if avg_price > 0:
            payload["avg_price"] = avg_price

    if payload.get("fee_abs") is None:
        fee_value = payload.get("fee")
        if fee_value is None:
            fee_value = order_info.get("fee")
        if fee_value is not None:
            payload["fee_abs"] = abs(_safe_float(fee_value, 0.0))

    if not payload.get("inst_id"):
        inst_id = str(order_info.get("instId") or "").strip()
        if inst_id:
            payload["inst_id"] = inst_id
    if payload.get("source") == "okx_fills_history":
        payload.setdefault("fills_history_confirmed", True)
    return payload

