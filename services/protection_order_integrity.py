"""Quantity-aware OKX position protection audit and repair planning."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any

from core.symbols import normalize_trading_symbol, okx_inst_id_from_symbol

PROTECTION_INTEGRITY_VERSION = "2026-07-15.okx-protection-coverage.v2"


def _decimal(value: Any) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(0)
    return abs(number) if number.is_finite() else Decimal(0)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _key(symbol: Any, side: Any) -> tuple[str, str] | None:
    normalized = normalize_trading_symbol(symbol)
    normalized_side = str(side or "").lower()
    if not normalized or normalized_side not in {"long", "short"}:
        return None
    return normalized, normalized_side


def _position_row(position: dict[str, Any]) -> dict[str, Any] | None:
    info = _safe_dict(position.get("info"))
    key = _key(
        position.get("symbol") or info.get("instId"),
        position.get("side") or info.get("posSide"),
    )
    contracts = _decimal(position.get("contracts") or info.get("pos"))
    if key is None or contracts <= 0:
        return None
    return {
        "symbol": key[0],
        "side": key[1],
        "inst_id": str(info.get("instId") or okx_inst_id_from_symbol(key[0])).upper(),
        "contracts": str(contracts),
    }


def _order_row(order: dict[str, Any]) -> dict[str, Any] | None:
    raw = _safe_dict(order.get("raw"))
    info = _safe_dict(raw.get("info"))
    key = _key(order.get("symbol"), order.get("position_side"))
    algo_id = str(order.get("algo_id") or "")
    contracts = _decimal(order.get("contracts"))
    if key is None or not algo_id:
        return None
    return {
        "symbol": key[0],
        "side": key[1],
        "inst_id": str(
            order.get("inst_id")
            or info.get("instId")
            or okx_inst_id_from_symbol(key[0])
        ).upper(),
        "algo_id": algo_id,
        "contracts": str(contracts),
        "reduce_only": order.get("reduce_only") is True,
        "state": str(order.get("state") or ""),
        "order_type": str(order.get("order_type") or ""),
        "stop_loss_price": order.get("stop_loss_price"),
        "take_profit_price": order.get("take_profit_price"),
        "created_at_ms": order.get("created_at_ms"),
        "updated_at_ms": order.get("updated_at_ms"),
        "linked_order_id": order.get("linked_order_id"),
    }


def _pending_entry_keys(pending_orders: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for order in pending_orders:
        info = _safe_dict(order.get("info"))
        reduce_only = order.get("reduceOnly")
        if reduce_only in (None, ""):
            reduce_only = info.get("reduceOnly")
        if str(reduce_only or "").lower() in {"true", "1"}:
            continue
        close_side = str(order.get("side") or info.get("side") or "").lower()
        position_side = "long" if close_side == "buy" else "short" if close_side == "sell" else ""
        key = _key(order.get("symbol") or info.get("instId"), position_side)
        if key is not None:
            keys.add(key)
    return keys


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if value <= 0 or step <= 0:
        return Decimal(0)
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _observed_quantum(value: Decimal) -> Decimal:
    if value <= 0:
        return Decimal(0)
    exponent = value.normalize().as_tuple().exponent
    return Decimal(1).scaleb(exponent) if exponent < 0 else Decimal(1)


def _resize_actions(
    position: dict[str, Any],
    orders: list[dict[str, Any]],
    contract_spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    desired = _decimal(position.get("contracts"))
    exchange_step = _decimal(contract_spec.get("lotSz"))
    weights = [_decimal(order.get("contracts")) for order in orders]
    total = sum(weights, Decimal(0))
    blockers: list[str] = []
    if desired <= 0 or total <= 0:
        blockers.append("position_or_protection_quantity_missing")
    if exchange_step <= 0:
        blockers.append("okx_contract_lot_size_missing")
    if blockers:
        return [], blockers
    observed_steps = [
        value
        for value in [
            exchange_step,
            _observed_quantum(desired),
            *(_observed_quantum(weight) for weight in weights),
        ]
        if value > 0
    ]
    step = min(observed_steps)

    actions: list[dict[str, Any]] = []
    remaining = desired
    ordered = sorted(
        zip(orders, weights, strict=True),
        key=lambda item: (str(item[0].get("created_at_ms") or ""), item[0]["algo_id"]),
    )
    for index, (order, weight) in enumerate(ordered):
        is_last = index == len(ordered) - 1
        target = remaining if is_last else _floor_to_step(desired * weight / total, step)
        remaining -= target
        if target <= 0:
            actions.append(
                {
                    "action": "cancel",
                    "reason": "proportional_slice_below_exchange_minimum",
                    "inst_id": order["inst_id"],
                    "algo_id": order["algo_id"],
                    "old_contracts": order["contracts"],
                    "new_contracts": "0",
                    "rollback": _cancel_rollback(order),
                }
            )
        elif target != weight:
            actions.append(
                {
                    "action": "amend_size",
                    "reason": "match_current_position_contract_coverage",
                    "inst_id": order["inst_id"],
                    "algo_id": order["algo_id"],
                    "old_contracts": order["contracts"],
                    "new_contracts": str(target),
                    "rollback": {
                        "action": "amend_size",
                        "inst_id": order["inst_id"],
                        "algo_id": order["algo_id"],
                        "new_contracts": order["contracts"],
                    },
                }
            )
    if remaining != 0:
        return [], ["proportional_resize_cannot_match_exchange_lot_size"]
    return actions, []


def _cancel_rollback(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": "manual_recreate_from_backup",
        "inst_id": order["inst_id"],
        "algo_id": order["algo_id"],
        "contracts": order["contracts"],
        "order_type": order.get("order_type"),
        "reduce_only": order.get("reduce_only"),
        "stop_loss_price": order.get("stop_loss_price"),
        "take_profit_price": order.get("take_profit_price"),
        "linked_order_id": order.get("linked_order_id"),
    }


def audit_protection_order_integrity(
    positions: list[dict[str, Any]],
    protection_orders: list[dict[str, Any]],
    pending_orders: list[dict[str, Any]],
    contract_specs: dict[str, dict[str, Any]],
    *,
    pending_snapshot_complete: bool,
) -> dict[str, Any]:
    """Build a non-mutating, quantity-exact repair plan from OKX-native facts."""

    position_rows = [row for item in positions if (row := _position_row(item)) is not None]
    order_rows = [row for item in protection_orders if (row := _order_row(item)) is not None]
    positions_by_key = {
        (row["symbol"], row["side"]): row
        for row in position_rows
    }
    orders_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in order_rows:
        orders_by_key[(row["symbol"], row["side"])].append(row)
    pending_entry_keys = _pending_entry_keys(pending_orders)
    missing: list[list[str]] = []
    orphan: list[list[str]] = []
    split: list[list[str]] = []
    coverage_mismatch: list[dict[str, Any]] = []
    invalid_orders: list[dict[str, Any]] = []
    repair_actions: list[dict[str, Any]] = []
    repair_blockers: list[str] = []

    for key, position in sorted(positions_by_key.items()):
        orders = orders_by_key.get(key, [])
        if not orders:
            missing.append(list(key))
            repair_blockers.append(f"missing_protection:{key[0]}:{key[1]}")
            continue
        if len(orders) > 1:
            split.append(list(key))
        for order in orders:
            if (
                _decimal(order.get("contracts")) <= 0
                or order.get("reduce_only") is not True
                or not order.get("stop_loss_price")
            ):
                invalid_orders.append(order)
        desired = _decimal(position.get("contracts"))
        covered = sum((_decimal(order.get("contracts")) for order in orders), Decimal(0))
        if covered != desired:
            coverage_mismatch.append(
                {
                    "symbol": key[0],
                    "side": key[1],
                    "position_contracts": str(desired),
                    "protection_contracts": str(covered),
                    "order_count": len(orders),
                }
            )
            spec = _safe_dict(contract_specs.get(position["inst_id"]))
            actions, blockers = _resize_actions(position, orders, spec)
            repair_actions.extend(actions)
            repair_blockers.extend(f"{key[0]}:{key[1]}:{item}" for item in blockers)

    for key, orders in sorted(orders_by_key.items()):
        if key in positions_by_key:
            continue
        orphan.append(list(key))
        if not pending_snapshot_complete:
            repair_blockers.append(f"orphan_pending_snapshot_incomplete:{key[0]}:{key[1]}")
            continue
        if key in pending_entry_keys:
            repair_blockers.append(f"orphan_has_pending_entry:{key[0]}:{key[1]}")
            continue
        repair_actions.extend(
            {
                "action": "cancel",
                "reason": "no_position_and_no_pending_entry",
                "inst_id": order["inst_id"],
                "algo_id": order["algo_id"],
                "old_contracts": order["contracts"],
                "new_contracts": "0",
                "rollback": _cancel_rollback(order),
            }
            for order in orders
        )

    payload = {
        "contract_version": PROTECTION_INTEGRITY_VERSION,
        "audit_only": True,
        "position_count": len(position_rows),
        "protection_order_count": len(order_rows),
        "pending_order_count": len(pending_orders),
        "pending_snapshot_complete": pending_snapshot_complete,
        "positions": position_rows,
        "protection_orders": order_rows,
        "missing_keys": missing,
        "orphan_keys": orphan,
        "split_coverage_keys": split,
        "coverage_mismatches": coverage_mismatch,
        "invalid_orders": invalid_orders,
        "repair_actions": repair_actions,
        "rollback_actions": [
            action["rollback"]
            for action in reversed(repair_actions)
            if isinstance(action.get("rollback"), dict)
        ],
        "repair_blockers": list(dict.fromkeys(repair_blockers)),
    }
    payload["position_inventory_fingerprint"] = hashlib.sha256(
        json.dumps(position_rows, ensure_ascii=True, sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    payload["input_fingerprint"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    payload["repair_ready"] = not payload["repair_blockers"] and not missing and not invalid_orders
    return payload
