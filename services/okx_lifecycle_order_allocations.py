"""Auditable allocation facts for OKX orders spanning position lifecycles."""

from __future__ import annotations

import math
from typing import Any

LIFECYCLE_ORDER_ALLOCATIONS_KEY = "_bb_lifecycle_order_allocations"
LIFECYCLE_ORDER_ALLOCATIONS_VERSION = "2026-08-18.okx-reversal-allocation.v1"
LIFECYCLE_ORDER_ALLOCATIONS_SOURCE = "okx_reversal_boundary_exact_contract_partition"
_VALID_ROLES = {"entry", "close"}


def build_lifecycle_order_allocation(
    *,
    order_id: str,
    allocated_contracts: float,
    order_contracts: float,
    boundary_at: str,
    peer_history_id: int,
    peer_role: str,
) -> dict[str, Any]:
    ratio = allocated_contracts / order_contracts
    return {
        "order_id": str(order_id),
        "allocated_contracts": float(allocated_contracts),
        "order_contracts": float(order_contracts),
        "allocation_ratio": float(ratio),
        "boundary_at": str(boundary_at),
        "peer_history_id": int(peer_history_id),
        "peer_role": str(peer_role),
    }


def build_lifecycle_order_allocation_document(
    *,
    entry: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    close: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    allocations = {
        role: {
            str(item.get("order_id") or "").strip(): dict(item)
            for item in items
            if str(item.get("order_id") or "").strip()
        }
        for role, items in (("entry", entry), ("close", close))
        if items
    }
    if not allocations:
        return {}
    return {
        "version": LIFECYCLE_ORDER_ALLOCATIONS_VERSION,
        "source": LIFECYCLE_ORDER_ALLOCATIONS_SOURCE,
        "allocations": allocations,
    }


def lifecycle_order_allocation(
    raw_row: dict[str, Any] | None,
    *,
    role: str,
    order_id: str,
    order_contracts: float | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return one validated allocation, or an explicit validation failure."""

    raw = raw_row if isinstance(raw_row, dict) else {}
    document = raw.get(LIFECYCLE_ORDER_ALLOCATIONS_KEY)
    if not isinstance(document, dict):
        return None, None
    if document.get("version") != LIFECYCLE_ORDER_ALLOCATIONS_VERSION:
        return None, "lifecycle_order_allocation_version_invalid"
    if document.get("source") != LIFECYCLE_ORDER_ALLOCATIONS_SOURCE:
        return None, "lifecycle_order_allocation_source_invalid"
    if role not in _VALID_ROLES:
        return None, "lifecycle_order_allocation_role_invalid"
    allocations = document.get("allocations")
    if not isinstance(allocations, dict):
        return None, "lifecycle_order_allocations_missing"
    role_allocations = allocations.get(role)
    if role_allocations is None:
        return None, None
    if not isinstance(role_allocations, dict):
        return None, "lifecycle_order_role_allocations_invalid"
    normalized_order_id = str(order_id or "").strip()
    item = role_allocations.get(normalized_order_id)
    if item is None:
        return None, None
    if not isinstance(item, dict):
        return None, "lifecycle_order_allocation_invalid"
    if str(item.get("order_id") or "").strip() != normalized_order_id:
        return None, "lifecycle_order_allocation_order_id_mismatch"
    try:
        allocated = float(item.get("allocated_contracts"))
        total = float(item.get("order_contracts"))
        ratio = float(item.get("allocation_ratio"))
        peer_history_id = int(item.get("peer_history_id"))
    except (TypeError, ValueError):
        return None, "lifecycle_order_allocation_numeric_fact_invalid"
    if not all(math.isfinite(value) for value in (allocated, total, ratio)):
        return None, "lifecycle_order_allocation_numeric_fact_invalid"
    if allocated <= 0 or total <= 0 or allocated > total:
        return None, "lifecycle_order_allocation_contracts_invalid"
    if not math.isclose(ratio, allocated / total, rel_tol=1e-9, abs_tol=1e-12):
        return None, "lifecycle_order_allocation_ratio_mismatch"
    if order_contracts is not None and not math.isclose(
        total,
        float(order_contracts),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        return None, "lifecycle_order_allocation_total_contracts_mismatch"
    if not str(item.get("boundary_at") or "").strip() or peer_history_id <= 0:
        return None, "lifecycle_order_allocation_lineage_missing"
    peer_role = str(item.get("peer_role") or "").strip()
    if peer_role not in _VALID_ROLES or peer_role == role:
        return None, "lifecycle_order_allocation_peer_role_invalid"
    return dict(item), None


def apply_lifecycle_order_allocation(
    fact: dict[str, Any],
    *,
    allocation: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    """Scale order-level economics to one lifecycle without duplicating facts."""

    ratio = float(allocation["allocation_ratio"])
    adjusted = dict(fact)
    adjusted["base_quantity"] = float(fact["base_quantity"]) * ratio
    adjusted["contracts"] = float(allocation["allocated_contracts"])
    adjusted["fee"] = float(fact["fee"]) * ratio
    fill_pnl = fact.get("fill_pnl")
    if fill_pnl is not None:
        # In a net-mode reversal, OKX fillPnl belongs to the contracts that
        # closed the old lifecycle. The newly opened lifecycle has no fill PnL.
        adjusted["fill_pnl"] = float(fill_pnl) if role == "close" else 0.0
    execution_slippage_usdt = fact.get("execution_slippage_usdt")
    if execution_slippage_usdt is not None:
        adjusted["execution_slippage_usdt"] = float(execution_slippage_usdt) * ratio
    adjusted["lifecycle_order_allocation"] = dict(allocation)
    return adjusted
