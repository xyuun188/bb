"""Repair missing local fragments for authoritative OKX close fills.

OKX can close one ``posId`` through several fills.  The order-fact sync may
persist every fill while an older position writer only creates some local
``Position`` rows.  This module closes that persistence gap without changing
exchange economics: a fragment is created only when the official lifecycle,
the close fill, contract size, direction, and entry order are all uniquely
proven.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from core.symbols import normalize_trading_symbol, symbol_from_okx_inst_id
from models.trade import OkxPositionHistory, Order, Position

REPAIR_SOURCE = "okx_authoritative_lifecycle_fragment_recovery"
REPAIR_VERSION = "2026-08-26.okx-lifecycle-fragment-repair.v1"
REPAIR_TRAINING_POLICY = "exclude_until_authoritative_settlement"
REPAIR_REASON = "missing_local_position_fragment_for_authoritative_close_fill"
ORDER_CONFIRMATION_STATES = {
    "okx_confirmed",
    "okx_order_detail_confirmed",
    "okx_execution_result_confirmed",
    "okx_only_backfilled",
}
POSITION_SETTLEMENT_SUPERSEDED = {
    "superseded_position_residual",
}
TIME_TOLERANCE_SECONDS = 120.0
CONTRACT_TOLERANCE_RATIO = 1e-6
MAX_LIFECYCLES_PER_PASS = 50


async def repair_missing_okx_lifecycle_fragments(
    session: Any,
    *,
    mode: str,
    position_history_rows: Iterable[dict[str, Any]],
    now: datetime | None = None,
    max_lifecycles: int = MAX_LIFECYCLES_PER_PASS,
) -> dict[str, Any]:
    """Create only uniquely proven local ``Position`` fragments.

    The operation is idempotent on ``(mode, posId, close_exchange_order_id)``.
    A lifecycle is changed only when all currently unlinked close fills in its
    bounded entry/close window exactly cover the missing official contracts.
    Ambiguous or incomplete lifecycles remain visible to the normal evidence
    gate and are deliberately not guessed.
    """

    selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
    checked_at = _aware_utc(now or datetime.now(UTC))
    history_rows = [row for row in position_history_rows if isinstance(row, dict)]
    if not history_rows:
        return _summary()

    position_result = await session.execute(
        select(Position).where(Position.execution_mode == selected_mode)
    )
    positions = list(position_result.scalars().all())
    order_result = await session.execute(
        select(Order).where(
            Order.execution_mode == selected_mode,
            Order.status == "filled",
            Order.exchange_order_id.is_not(None),
            Order.exchange_order_id != "",
            Order.filled_at.is_not(None),
            Order.filled_at >= checked_at - timedelta(days=90),
        )
    )
    orders = list(order_result.scalars().all())
    orders_by_id = {
        _text(order.exchange_order_id): order
        for order in orders
        if _text(order.exchange_order_id)
    }
    globally_linked_close_ids = {
        close_id
        for position in positions
        for close_id in _split_ids(getattr(position, "close_exchange_order_id", None))
    }

    created = 0
    skipped = 0
    samples: list[dict[str, Any]] = []
    seen_lifecycles: set[tuple[str, str, str, str, tuple[str, ...]]] = set()
    for row in history_rows:
        if created >= max(int(max_lifecycles or 1), 1):
            break
        lifecycle = _history_lifecycle(row, mode=selected_mode)
        if lifecycle is None:
            continue
        lifecycle_key, lifecycle_positions = _select_lifecycle_positions(
            lifecycle,
            positions,
            orders_by_id=orders_by_id,
        )
        if lifecycle_key is None or not lifecycle_positions:
            continue
        if lifecycle_key in seen_lifecycles:
            continue
        seen_lifecycles.add(lifecycle_key)
        repair = _plan_missing_fragments(
            lifecycle,
            lifecycle_positions,
            orders=orders,
            orders_by_id=orders_by_id,
            globally_linked_close_ids=globally_linked_close_ids,
        )
        if repair is None:
            continue
        missing_orders, entry_ids, entry_price, side, symbol, inst_id, pos_id = repair
        history_record = await _history_record(session, row)
        for order in missing_orders:
            exchange_order_id = _text(order.exchange_order_id)
            if not exchange_order_id or exchange_order_id in globally_linked_close_ids:
                skipped += 1
                continue
            contract_size = _order_contract_size(order, row)
            contracts = _order_contracts(order)
            quantity = _order_base_quantity(order, contract_size)
            filled_at = _aware_utc(order.filled_at or order.created_at)
            if contracts <= 0 or contract_size <= 0 or quantity <= 0 or filled_at is None:
                skipped += 1
                continue
            raw = _dict(getattr(order, "okx_raw_fills", None))
            evidence = {
                "version": REPAIR_VERSION,
                "source": REPAIR_SOURCE,
                "reason": REPAIR_REASON,
                "repaired_at": checked_at.isoformat(),
                "training_policy": REPAIR_TRAINING_POLICY,
                "authoritative": {
                    "okx_pos_id": pos_id,
                    "okx_inst_id": inst_id,
                    "close_exchange_order_id": exchange_order_id,
                    "entry_exchange_order_ids": list(entry_ids),
                    "close_contracts": contracts,
                    "contract_size": contract_size,
                    "quantity": quantity,
                    "close_price": _safe_float(order.price),
                    "close_fill_pnl": _safe_float(order.okx_fill_pnl),
                    "close_fee": abs(_safe_float(order.fee)),
                    "source": str(raw.get("source") or ""),
                    "fills_history_confirmed": raw.get("fills_history_confirmed") is True,
                    "contract_size_verified": raw.get("contract_size_verified") is True,
                },
            }
            position = Position(
                model_name="okx_authoritative_sync",
                execution_mode=selected_mode,
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                current_price=_safe_float(order.price),
                leverage=_history_leverage(row, lifecycle_positions),
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                close_fill_pnl=_safe_float(order.okx_fill_pnl),
                entry_fee=0.0,
                close_fee=abs(_safe_float(order.fee)),
                funding_fee=0.0,
                settlement_status="settling",
                settlement_source=REPAIR_SOURCE,
                settlement_synced_at=checked_at,
                settlement_raw=evidence,
                is_open=False,
                closed_at=filled_at,
                okx_inst_id=inst_id,
                okx_pos_id=pos_id,
                entry_exchange_order_id=",".join(entry_ids),
                close_exchange_order_id=exchange_order_id,
                created_at=_history_opened_at(row) or filled_at,
                updated_at=checked_at,
            )
            session.add(position)
            await session.flush()
            positions.append(position)
            globally_linked_close_ids.add(exchange_order_id)
            created += 1
            if history_record is not None:
                _merge_history_record_links(
                    history_record,
                    position_id=position.id,
                    entry_ids=entry_ids,
                    close_id=exchange_order_id,
                )
            row.setdefault("_dashboard_position_ids", [])
            row["_dashboard_position_ids"] = _merge_values(
                row.get("_dashboard_position_ids"),
                [str(position.id)],
            )
            row["_dashboard_close_order_ids"] = _merge_values(
                row.get("_dashboard_close_order_ids"),
                [exchange_order_id],
            )
            samples.append(
                {
                    "kind": "okx_missing_lifecycle_fragment_recovered",
                    "position_id": int(position.id),
                    "symbol": symbol,
                    "side": side,
                    "okx_pos_id": pos_id,
                    "exchange_order_id": exchange_order_id,
                    "contracts": contracts,
                    "quantity": quantity,
                }
            )
    if created or history_rows:
        await session.flush()
    return {
        "created_count": created,
        "skipped_count": skipped,
        "samples": samples[:10],
    }


def _summary() -> dict[str, Any]:
    return {"created_count": 0, "skipped_count": 0, "samples": []}


def _history_lifecycle(
    row: dict[str, Any],
    *,
    mode: str,
) -> tuple[tuple[str, str, str, str, tuple[str, ...]], dict[str, Any]] | None:
    inst_id = _text(row.get("instId") or row.get("inst_id")).upper()
    pos_id = _text(row.get("posId") or row.get("pos_id"))
    side = _history_side(row)
    opened_at = _history_opened_at(row)
    closed_at = _history_closed_at(row)
    close_contracts = _safe_float(row.get("closeTotalPos") or row.get("close_total_pos"))
    open_contracts = _safe_float(row.get("openMaxPos") or row.get("open_max_pos"))
    if not inst_id or not pos_id or side not in {"long", "short"}:
        return None
    if close_contracts <= 0 or open_contracts <= 0 or opened_at is None or closed_at is None:
        return None
    symbol = normalize_trading_symbol(symbol_from_okx_inst_id(inst_id) or inst_id)
    return (
        (mode, inst_id, pos_id, side, ()),
        {
            "mode": mode,
            "inst_id": inst_id,
            "pos_id": pos_id,
            "side": side,
            "symbol": symbol,
            "opened_at": opened_at,
            "closed_at": closed_at,
            "open_contracts": open_contracts,
            "close_contracts": close_contracts,
            "entry_price": _safe_float(row.get("openAvgPx") or row.get("open_avg_px")),
            "row": row,
        },
    )


def _select_lifecycle_positions(
    lifecycle: tuple[tuple[str, str, str, str, tuple[str, ...]], dict[str, Any]],
    positions: list[Position],
    *,
    orders_by_id: dict[str, Order],
) -> tuple[tuple[str, str, str, str, tuple[str, ...]] | None, list[Position]]:
    _base_key, facts = lifecycle
    candidates = [
        position
        for position in positions
        if not bool(getattr(position, "is_open", False))
        and _text(getattr(position, "okx_pos_id", "")) == facts["pos_id"]
        and _text(getattr(position, "okx_inst_id", "")).upper() == facts["inst_id"]
        and normalize_trading_symbol(getattr(position, "symbol", "")) == facts["symbol"]
        and str(getattr(position, "side", "") or "").lower() == facts["side"]
        and str(getattr(position, "settlement_status", "") or "")
        not in POSITION_SETTLEMENT_SUPERSEDED
    ]
    grouped: dict[tuple[str, ...], list[Position]] = {}
    for position in candidates:
        entry_ids = tuple(sorted(_split_ids(getattr(position, "entry_exchange_order_id", None))))
        if entry_ids:
            grouped.setdefault(entry_ids, []).append(position)
    if not grouped:
        return None, []
    scored: list[tuple[float, tuple[str, ...], list[Position]]] = []
    for entry_ids, group in grouped.items():
        entry_times = [
            _aware_utc(orders_by_id[order_id].filled_at or orders_by_id[order_id].created_at)
            for order_id in entry_ids
            if order_id in orders_by_id
        ]
        entry_times = [value for value in entry_times if value is not None]
        if not entry_times:
            continue
        opened_delta = abs((min(entry_times) - facts["opened_at"]).total_seconds())
        if opened_delta > 24 * 3600:
            continue
        entry_contracts = sum(
            _order_contracts(orders_by_id[order_id])
            for order_id in entry_ids
            if order_id in orders_by_id
        )
        contract_delta = abs(entry_contracts - facts["open_contracts"])
        score = opened_delta / 60.0 + contract_delta * 100.0
        scored.append((score, entry_ids, group))
    if not scored:
        return None, []
    scored.sort(key=lambda item: item[0])
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) <= 1e-9:
        return None, []
    chosen_score, entry_ids, group = scored[0]
    if chosen_score > 24 * 60.0 + 1e-6 and facts["open_contracts"] > 0:
        return None, []
    return (*_base_key[:4], entry_ids), group


def _plan_missing_fragments(
    lifecycle: tuple[tuple[str, str, str, str, tuple[str, ...]], dict[str, Any]],
    positions: list[Position],
    *,
    orders: list[Order],
    orders_by_id: dict[str, Order],
    globally_linked_close_ids: set[str],
) -> tuple[list[Order], tuple[str, ...], float, str, str, str, str] | None:
    _base_key, facts = lifecycle
    entry_ids = tuple(sorted(_split_ids(getattr(positions[0], "entry_exchange_order_id", None))))
    if not entry_ids:
        return None
    existing_close_ids = {
        close_id
        for position in positions
        for close_id in _split_ids(getattr(position, "close_exchange_order_id", None))
    }
    local_close_contracts = sum(
        _order_contracts(orders_by_id[close_id])
        for close_id in existing_close_ids
        if close_id in orders_by_id
    )
    expected_missing = facts["close_contracts"] - local_close_contracts
    tolerance = max(facts["close_contracts"] * CONTRACT_TOLERANCE_RATIO, 1e-8)
    if expected_missing <= tolerance:
        return None
    close_side = "buy" if facts["side"] == "short" else "sell"
    candidates: list[Order] = []
    for order in orders:
        exchange_order_id = _text(order.exchange_order_id)
        if not exchange_order_id or exchange_order_id in globally_linked_close_ids:
            continue
        if exchange_order_id in entry_ids or str(order.side or "").lower() != close_side:
            continue
        if _text(getattr(order, "okx_inst_id", "")).upper() != facts["inst_id"]:
            continue
        if not _order_is_authoritative(order, expected_side=close_side, position_side=facts["side"]):
            continue
        filled_at = _aware_utc(order.filled_at or order.created_at)
        if filled_at is None:
            continue
        if filled_at < facts["opened_at"] - timedelta(seconds=TIME_TOLERANCE_SECONDS):
            continue
        if filled_at > facts["closed_at"] + timedelta(seconds=TIME_TOLERANCE_SECONDS):
            continue
        candidates.append(order)
    if not candidates:
        return None
    candidate_contracts = sum(_order_contracts(order) for order in candidates)
    if abs(candidate_contracts - expected_missing) > tolerance:
        return None
    entry_price = facts["entry_price"]
    if entry_price <= 0:
        entry_price = _weighted_price(
            orders_by_id[entry_id] for entry_id in entry_ids if entry_id in orders_by_id
        )
    if entry_price <= 0:
        return None
    return (
        sorted(candidates, key=lambda item: _aware_utc(item.filled_at or item.created_at) or datetime.max.replace(tzinfo=UTC)),
        entry_ids,
        entry_price,
        facts["side"],
        facts["symbol"],
        facts["inst_id"],
        facts["pos_id"],
    )


def _order_is_authoritative(order: Order, *, expected_side: str, position_side: str) -> bool:
    if str(getattr(order, "status", "") or "").lower() != "filled":
        return False
    if str(getattr(order, "side", "") or "").lower() != expected_side:
        return False
    raw = _dict(getattr(order, "okx_raw_fills", None))
    if _order_contracts(order) <= 0:
        return False
    sync_status = _text(getattr(order, "okx_sync_status", "")).lower()
    if sync_status not in ORDER_CONFIRMATION_STATES and raw.get("fills_history_confirmed") is not True:
        return False
    if raw.get("fills_history_confirmed") is not True:
        return False
    if raw.get("contract_size_verified") is not True and _safe_float(raw.get("contract_size")) <= 0:
        return False
    lifecycle = _dict(raw.get("protection_execution"))
    if lifecycle:
        if lifecycle.get("lifecycle_complete") is not True:
            return False
        if lifecycle.get("reduce_only") not in (True, "true", 1, "1"):
            return False
        if _text(lifecycle.get("position_side")).lower() not in {"", position_side}:
            return False
    return True


def _order_contracts(order: Order) -> float:
    raw = _dict(getattr(order, "okx_raw_fills", None))
    return max(
        _safe_float(getattr(order, "okx_fill_contracts", None)),
        _safe_float(raw.get("contracts") or raw.get("filled_contracts")),
        0.0,
    )


def _order_contract_size(order: Order, row: dict[str, Any]) -> float:
    raw = _dict(getattr(order, "okx_raw_fills", None))
    direct = _safe_float(raw.get("contract_size"))
    if direct > 0:
        return direct
    spec = row.get("_bb_contract_spec")
    if isinstance(spec, dict):
        ct_val = _safe_float(spec.get("ctVal"))
        ct_mult = _safe_float(spec.get("ctMult"), 1.0) or 1.0
        if ct_val > 0:
            return ct_val * ct_mult
    return 0.0


def _order_base_quantity(order: Order, contract_size: float) -> float:
    contracts = _order_contracts(order)
    raw = _dict(getattr(order, "okx_raw_fills", None))
    base_quantity = _safe_float(raw.get("base_quantity"))
    expected = contracts * contract_size
    if base_quantity > 0 and expected > 0:
        if abs(base_quantity - expected) > max(expected * 0.02, 1e-8):
            return 0.0
        return base_quantity
    quantity = _safe_float(getattr(order, "quantity", None))
    if quantity > 0 and expected > 0 and abs(quantity - expected) <= max(expected * 0.02, 1e-8):
        return quantity
    return expected


async def _history_record(session: Any, row: dict[str, Any]) -> OkxPositionHistory | None:
    record_id = _safe_int(row.get("_dashboard_history_record_id"))
    if record_id <= 0:
        return None
    return await session.get(OkxPositionHistory, record_id)


def _merge_history_record_links(
    record: OkxPositionHistory,
    *,
    position_id: int | None,
    entry_ids: Iterable[str],
    close_id: str,
) -> None:
    record.position_ids = _merge_values(record.position_ids, [str(position_id)] if position_id else [])
    record.entry_order_ids = _merge_values(record.entry_order_ids, entry_ids)
    record.close_order_ids = _merge_values(record.close_order_ids, [close_id])
    record.linked_order_ids = _merge_values(
        record.linked_order_ids,
        [*entry_ids, close_id],
    )
    record.match_status = "okx_lifecycle_fragment_repair_pending_settlement"


def _history_side(row: dict[str, Any]) -> str:
    for key in ("direction", "side", "positionSide", "posSide"):
        value = _text(row.get(key)).lower()
        if value in {"long", "short"}:
            return value
    return ""


def _history_leverage(row: dict[str, Any], positions: list[Position]) -> float:
    value = _safe_float(row.get("lever") or row.get("leverage"), 0.0)
    if value > 0:
        return value
    return max(_safe_float(getattr(positions[0], "leverage", None), 1.0), 1.0)


def _history_opened_at(row: dict[str, Any]) -> datetime | None:
    return _datetime_from_ms(row.get("cTime") or row.get("createdTime") or row.get("openTime"))


def _history_closed_at(row: dict[str, Any]) -> datetime | None:
    return _datetime_from_ms(row.get("uTime") or row.get("updatedTime") or row.get("closeTime"))


def _weighted_price(orders: Iterable[Order]) -> float:
    weighted = 0.0
    total = 0.0
    for order in orders:
        quantity = _order_contracts(order)
        price = _safe_float(order.price)
        if quantity > 0 and price > 0:
            weighted += quantity * price
            total += quantity
    return weighted / total if total > 0 else 0.0


def _split_ids(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value or "").replace(";", ",").replace("|", ",").split(",")
    return {token for token in (_text(item) for item in values) if token}


def _merge_values(existing: Any, incoming: Iterable[Any]) -> list[str]:
    values = {_text(item) for item in (existing if isinstance(existing, (list, tuple, set)) else [])}
    values.update(_text(item) for item in incoming)
    values.discard("")
    return sorted(values)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _aware_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _datetime_from_ms(value: Any) -> datetime | None:
    timestamp = _safe_float(value, 0.0)
    if timestamp <= 0:
        return None
    if timestamp < 10_000_000_000:
        timestamp *= 1000
    try:
        return datetime.fromtimestamp(timestamp / 1000.0, UTC)
    except (OSError, OverflowError, ValueError):
        return None
