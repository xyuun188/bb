"""Recover local position links for confirmed OKX protection fills."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, or_, select

from core.symbols import normalize_trading_symbol, symbol_from_okx_inst_id
from models.trade import Order, Position
from services.position_settlement import (
    apply_position_settlement_snapshot,
    build_position_settlement_snapshot,
)

ORPHAN_QUARANTINE_CLOSE_PREFIX = "okx_orphan_quarantine:"
PROTECTION_POSITION_RECOVERY_SOURCE = "okx_protection_position_lifecycle_recovery"
PROTECTION_POSITION_RECOVERY_VERSION = "2026-08-01.okx-protection-position-recovery.v1"
QUANTITY_TOLERANCE_RATIO = 0.02


async def recover_protection_position_lifecycles(
    session: Any,
    *,
    orders: Iterable[Order],
    mode: str,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Link uniquely proven protection fills to open or quarantined positions."""

    checked_at = _aware_utc(now or datetime.now(UTC))
    candidates_by_order: dict[str, tuple[Order, dict[str, Any]]] = {}
    for order in orders:
        lifecycle = confirmed_protection_lifecycle(order)
        exchange_order_id = str(order.exchange_order_id or "").strip()
        if lifecycle is None or not exchange_order_id:
            continue
        candidates_by_order[exchange_order_id] = (order, lifecycle)
        raw = _dict(getattr(order, "okx_raw_fills", None))
        if _dict(raw.get("protection_execution")) != lifecycle:
            order.okx_raw_fills = {
                **raw,
                "protection_execution": dict(lifecycle),
            }
    if not candidates_by_order:
        return []

    symbols = {
        _position_symbol(order, lifecycle)
        for order, lifecycle in candidates_by_order.values()
        if _position_symbol(order, lifecycle)
    }
    inst_ids = {
        str(lifecycle.get("inst_id") or getattr(order, "okx_inst_id", "") or "")
        .strip()
        .upper()
        for order, lifecycle in candidates_by_order.values()
    }
    inst_ids.discard("")
    position_result = await session.execute(
        select(Position)
        .where(
            Position.execution_mode == ("live" if str(mode).lower() == "live" else "paper"),
            or_(Position.symbol.in_(symbols), Position.okx_inst_id.in_(inst_ids)),
            or_(
                Position.is_open.is_(True),
                and_(
                    Position.is_open.is_(False),
                    Position.close_exchange_order_id.like(
                        f"{ORPHAN_QUARANTINE_CLOSE_PREFIX}%"
                    ),
                ),
            ),
        )
        .with_for_update(skip_locked=True)
    )
    positions = list(position_result.scalars().all())
    if not positions:
        return []

    order_ids = set(candidates_by_order)
    linked_result = await session.execute(
        select(Position.close_exchange_order_id).where(
            Position.execution_mode
            == ("live" if str(mode).lower() == "live" else "paper"),
            Position.close_exchange_order_id.in_(order_ids),
        )
    )
    already_linked = {
        str(value or "").strip()
        for value in linked_result.scalars().all()
        if str(value or "").strip()
    }

    entry_order_ids = {
        token
        for position in positions
        for token in _split_exchange_order_ids(position.entry_exchange_order_id)
    }
    entry_orders: dict[str, Order] = {}
    if entry_order_ids:
        entry_result = await session.execute(
            select(Order).where(
                Order.execution_mode
                == ("live" if str(mode).lower() == "live" else "paper"),
                Order.exchange_order_id.in_(entry_order_ids),
            )
        )
        entry_orders = {
            str(order.exchange_order_id or "").strip(): order
            for order in entry_result.scalars().all()
            if str(order.exchange_order_id or "").strip()
        }

    recovered: list[dict[str, Any]] = []
    for exchange_order_id, (order, lifecycle) in candidates_by_order.items():
        if exchange_order_id in already_linked:
            continue
        matching = [
            position
            for position in positions
            if _position_matches_protection_order(position, order, lifecycle)
        ]
        if len(matching) != 1:
            continue
        position = matching[0]
        filled_at = _aware_utc(order.filled_at or getattr(order, "created_at", None))
        if filled_at is None:
            continue
        evidence = {
            "version": PROTECTION_POSITION_RECOVERY_VERSION,
            "source": PROTECTION_POSITION_RECOVERY_SOURCE,
            "recovered_at": checked_at.isoformat(),
            "position_id": int(position.id),
            "exchange_order_id": exchange_order_id,
            "entry_exchange_order_ids": sorted(
                _split_exchange_order_ids(position.entry_exchange_order_id)
            ),
            "okx_algo_id": str(lifecycle.get("algo_id") or ""),
            "quantity": _safe_float(order.quantity),
            "filled_at": filled_at.isoformat(),
            "training_policy": "exclude_until_manual_trust",
        }
        was_open = bool(position.is_open)
        position.close_exchange_order_id = exchange_order_id
        position.closed_at = filled_at
        position.current_price = _safe_float(order.price, position.current_price or 0.0)
        position.unrealized_pnl = 0.0
        position.updated_at = checked_at
        if was_open:
            position.is_open = False
            management = _dict(position.current_management_contract)
            entry_fee = _safe_float(
                management.get("entry_fee_usdt"),
                abs(_safe_float(position.entry_fee)),
            )
            close_fee = abs(_safe_float(order.fee))
            funding_fee = _safe_float(position.funding_fee)
            gross_pnl = _safe_float(order.okx_fill_pnl)
            snapshot = build_position_settlement_snapshot(
                close_fill_pnl=gross_pnl,
                entry_fee=entry_fee,
                close_fee=close_fee,
                funding_fee=funding_fee,
                status="settling",
                source=PROTECTION_POSITION_RECOVERY_SOURCE,
                synced_at=checked_at,
                raw=evidence,
            )
            apply_position_settlement_snapshot(position, snapshot)
        else:
            settlement_raw = _dict(position.settlement_raw)
            position.settlement_raw = {
                **settlement_raw,
                "protection_position_lifecycle_recovery": evidence,
            }

        if getattr(order, "decision_id", None) is None:
            linked_decision_ids = {
                int(entry_order.decision_id)
                for token in _split_exchange_order_ids(position.entry_exchange_order_id)
                if (entry_order := entry_orders.get(token)) is not None
                if getattr(entry_order, "decision_id", None) is not None
            }
            if len(linked_decision_ids) == 1:
                order.decision_id = linked_decision_ids.pop()

        recovered.append(
            {
                "kind": (
                    "open_protection_position_lifecycle_recovered"
                    if was_open
                    else "quarantined_protection_position_lifecycle_recovered"
                ),
                "position_id": int(position.id),
                "symbol": normalize_trading_symbol(position.symbol),
                "side": str(position.side or ""),
                "exchange_order_id": exchange_order_id,
                "okx_algo_id": str(lifecycle.get("algo_id") or ""),
            }
        )
    return recovered


def confirmed_protection_lifecycle(order: Order) -> dict[str, Any] | None:
    if str(getattr(order, "status", "") or "").lower() != "filled":
        return None
    raw = _dict(getattr(order, "okx_raw_fills", None))
    lifecycle = _dict(raw.get("protection_execution"))
    reduce_only = lifecycle.get("reduce_only")
    if reduce_only in (None, ""):
        reduce_only = _native_bool(_dict(lifecycle.get("algo_row")).get("reduceOnly"))
    exchange_order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
    generated_order_id = str(lifecycle.get("generated_order_id") or "").strip()
    if (
        raw.get("fills_history_confirmed") is not True
        or lifecycle.get("lifecycle_complete") is not True
        or not exchange_order_id
        or generated_order_id != exchange_order_id
        or str(lifecycle.get("source_authority") or "")
        != "okx_algo_history_plus_fills_history"
        or reduce_only is not True
        or not str(lifecycle.get("algo_id") or "").strip()
        or str(lifecycle.get("position_side") or "").lower() not in {"long", "short"}
        or str(lifecycle.get("close_side") or "").lower() not in {"buy", "sell"}
        or _safe_float(getattr(order, "quantity", None)) <= 0
    ):
        return None
    base_quantity = _safe_float(raw.get("base_quantity"))
    if base_quantity > 0 and not _quantity_matches(
        base_quantity,
        _safe_float(getattr(order, "quantity", None)),
    ):
        return None
    return {
        **lifecycle,
        "reduce_only": True,
    }


def _position_matches_protection_order(
    position: Position,
    order: Order,
    lifecycle: dict[str, Any],
) -> bool:
    if str(position.side or "").lower() != str(lifecycle.get("position_side") or "").lower():
        return False
    if str(order.side or "").lower() != str(lifecycle.get("close_side") or "").lower():
        return False
    if normalize_trading_symbol(position.symbol) != _position_symbol(order, lifecycle):
        return False
    if not _quantity_matches(position.quantity, order.quantity):
        return False
    algo_id = str(lifecycle.get("algo_id") or "").strip()
    if algo_id not in _position_protection_algo_ids(position):
        return False
    entry_ids = _split_exchange_order_ids(position.entry_exchange_order_id)
    if not entry_ids:
        return False
    original_ids = {
        str(item or "").strip()
        for item in _dict(position.current_management_contract).get(
            "original_entry_order_ids", []
        )
        if str(item or "").strip()
    }
    return not original_ids or bool(entry_ids & original_ids)


def _position_protection_algo_ids(position: Position) -> set[str]:
    contract = _dict(position.current_management_contract)
    orders = contract.get("protection_orders")
    if not isinstance(orders, list):
        return set()
    result: set[str] = set()
    for item in orders:
        row = _dict(item)
        raw = _dict(row.get("raw"))
        info = _dict(raw.get("info"))
        reduce_only = row.get("reduce_only")
        if reduce_only in (None, ""):
            reduce_only = row.get("reduceOnly")
        if reduce_only in (None, ""):
            reduce_only = raw.get("reduceOnly")
        if reduce_only in (None, ""):
            reduce_only = info.get("reduceOnly")
        if not _native_bool(reduce_only):
            continue
        for value in (
            row.get("algo_id"),
            row.get("algoId"),
            raw.get("algoId"),
            info.get("algoId"),
        ):
            token = str(value or "").strip()
            if token:
                result.add(token)
    return result


def _position_symbol(order: Order, lifecycle: dict[str, Any]) -> str:
    inst_id = str(lifecycle.get("inst_id") or getattr(order, "okx_inst_id", "") or "")
    return normalize_trading_symbol(
        symbol_from_okx_inst_id(inst_id) or getattr(order, "symbol", "")
    )


def _quantity_matches(left: Any, right: Any) -> bool:
    left_value = abs(_safe_float(left))
    right_value = abs(_safe_float(right))
    if left_value <= 0 or right_value <= 0:
        return False
    return abs(left_value - right_value) <= max(
        left_value * QUANTITY_TOLERANCE_RATIO,
        right_value * QUANTITY_TOLERANCE_RATIO,
        1e-8,
    )


def _split_exchange_order_ids(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value or "").replace(";", ",").split(",")
    return {str(item or "").strip() for item in values if str(item or "").strip()}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _native_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
