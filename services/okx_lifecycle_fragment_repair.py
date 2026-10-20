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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, text

from core.symbols import normalize_trading_symbol, symbol_from_okx_inst_id
from models.decision import AIDecision
from models.trade import OkxPositionHistory, Order, Position

REPAIR_SOURCE = "okx_authoritative_lifecycle_fragment_recovery"
REPAIR_VERSION = "2026-10-20.okx-lifecycle-recovery.v4"
REPAIR_TRAINING_POLICY = "exclude_until_authoritative_settlement"
REPAIR_REASON = "missing_local_position_for_authoritative_closed_lifecycle"
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


@dataclass(frozen=True, slots=True)
class MissingLifecyclePlan:
    history_record: OkxPositionHistory
    entry_order_ids: tuple[str, ...]
    close_order_ids: tuple[str, ...]
    decision_id: int
    model_name: str
    symbol: str
    side: str
    inst_id: str
    pos_id: str
    quantity: float
    entry_price: float
    close_price: float
    leverage: float
    opened_at: datetime
    closed_at: datetime


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

    # Hold the same transaction lock through commit so concurrent workers read
    # the previous repair's committed positions before planning another insert.
    if session.get_bind().dialect.name == "postgresql":
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:lock_key))"),
            {"lock_key": f"bb:okx_lifecycle_fragment_repair:{selected_mode}"},
        )
    position_result = await session.execute(
        select(Position)
        .where(Position.execution_mode == selected_mode)
        .execution_options(populate_existing=True)
    )
    all_positions = list(position_result.scalars().all())
    positions = [
        position for position in all_positions
        if position.settlement_status not in POSITION_SETTLEMENT_SUPERSEDED
    ]
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
    orders_by_id: dict[str, Order] = {}
    duplicate_order_ids: set[str] = set()
    for order in orders:
        order_id = _text(order.exchange_order_id)
        if not order_id or order_id in duplicate_order_ids:
            continue
        if order_id in orders_by_id:
            orders_by_id.pop(order_id, None)
            duplicate_order_ids.add(order_id)
            continue
        orders_by_id[order_id] = order
    globally_linked_close_ids = {
        close_id
        for position in all_positions
        for close_id in _split_ids(getattr(position, "close_exchange_order_id", None))
    }
    positions_by_close_id: dict[str, list[Position]] = {}
    for position in positions:
        if bool(position.is_open):
            continue
        for close_id in _split_ids(position.close_exchange_order_id):
            positions_by_close_id.setdefault(close_id, []).append(position)
    close_owners: dict[str, set[tuple[Any, ...]]] = {}
    entry_owners: dict[str, set[tuple[Any, ...]]] = {}
    lifecycle_row_counts: dict[tuple[Any, ...], int] = {}
    for row in history_rows:
        lifecycle = _history_lifecycle(row, mode=selected_mode)
        if lifecycle is None:
            continue
        _key, facts = lifecycle
        identity = (facts["inst_id"], facts["pos_id"], facts["side"], facts["opened_at"])
        lifecycle_row_counts[identity] = lifecycle_row_counts.get(identity, 0) + 1
        for entry_id in _split_ids(row.get("_dashboard_entry_order_ids")):
            entry_owners.setdefault(entry_id, set()).add(identity)
        for close_id in _split_ids(row.get("_dashboard_close_order_ids")):
            close_owners.setdefault(close_id, set()).add(identity)

    created = 0
    repaired_identity = 0
    repaired_order_links = 0
    skipped = 0
    samples: list[dict[str, Any]] = []
    seen_lifecycles: set[tuple[str, str, str, str, tuple[str, ...]]] = set()
    for row in history_rows:
        if created >= max(int(max_lifecycles or 1), 1):
            break
        lifecycle = _history_lifecycle(row, mode=selected_mode)
        if lifecycle is None:
            continue
        identity = (
            lifecycle[1]["inst_id"],
            lifecycle[1]["pos_id"],
            lifecycle[1]["side"],
            lifecycle[1]["opened_at"],
        )
        if lifecycle_row_counts.get(identity, 0) == 1:
            link_sample = await _repair_partial_lifecycle_order_links(
                session,
                lifecycle,
                positions=positions,
                orders=orders,
                orders_by_id=orders_by_id,
                entry_owners=entry_owners,
                close_owners=close_owners,
                now=checked_at,
            )
            if link_sample is not None:
                repaired_order_links += 1
                samples.append(link_sample)
                for order_id in _split_ids(row.get("_dashboard_close_order_ids")):
                    close_owners.setdefault(order_id, set()).add(identity)
        identity_samples = []
        if lifecycle_row_counts.get(identity, 0) == 1 and all(
            len(entry_owners.get(order_id, set())) == 1
            for order_id in _split_ids(row.get("_dashboard_entry_order_ids"))
        ):
            identity_samples = _repair_linked_lifecycle_identity(
                lifecycle, positions_by_close_id=positions_by_close_id,
                orders_by_id=orders_by_id, close_owners=close_owners, now=checked_at,
            )
        repaired_identity += len(identity_samples)
        samples.extend(identity_samples)
        lifecycle_key, lifecycle_positions = _select_lifecycle_positions(
            lifecycle,
            positions,
            orders_by_id=orders_by_id,
        )
        if lifecycle_key is None or not lifecycle_positions:
            full_lifecycle_plan = await _plan_missing_full_lifecycle(
                session,
                lifecycle,
                positions=all_positions,
                orders_by_id=orders_by_id,
                duplicate_order_ids=duplicate_order_ids,
                entry_owners=entry_owners,
                close_owners=close_owners,
                lifecycle_row_count=lifecycle_row_counts.get(identity, 0),
            )
            if full_lifecycle_plan is None:
                continue
            position = Position(
                model_name=full_lifecycle_plan.model_name,
                execution_mode=selected_mode,
                symbol=full_lifecycle_plan.symbol,
                side=full_lifecycle_plan.side,
                quantity=full_lifecycle_plan.quantity,
                entry_price=full_lifecycle_plan.entry_price,
                current_price=full_lifecycle_plan.close_price,
                leverage=full_lifecycle_plan.leverage,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                close_fill_pnl=0.0,
                entry_fee=0.0,
                close_fee=0.0,
                funding_fee=0.0,
                settlement_status="settling",
                settlement_source=REPAIR_SOURCE,
                settlement_synced_at=checked_at,
                settlement_raw={
                    "lifecycle_recovery": {
                        "version": REPAIR_VERSION,
                        "source": REPAIR_SOURCE,
                        "reason": REPAIR_REASON,
                        "repaired_at": checked_at.isoformat(),
                        "training_policy": REPAIR_TRAINING_POLICY,
                        "history_record_id": int(full_lifecycle_plan.history_record.id),
                        "decision_id": full_lifecycle_plan.decision_id,
                        "entry_order_ids": list(full_lifecycle_plan.entry_order_ids),
                        "close_order_ids": list(full_lifecycle_plan.close_order_ids),
                        "okx_inst_id": full_lifecycle_plan.inst_id,
                        "okx_pos_id": full_lifecycle_plan.pos_id,
                        "quantity": full_lifecycle_plan.quantity,
                    }
                },
                is_open=False,
                closed_at=full_lifecycle_plan.closed_at,
                okx_inst_id=full_lifecycle_plan.inst_id,
                okx_pos_id=full_lifecycle_plan.pos_id,
                entry_exchange_order_id=",".join(full_lifecycle_plan.entry_order_ids),
                close_exchange_order_id=",".join(full_lifecycle_plan.close_order_ids),
                created_at=full_lifecycle_plan.opened_at,
                updated_at=checked_at,
            )
            session.add(position)
            await session.flush()
            _merge_history_record_links(
                full_lifecycle_plan.history_record,
                position_id=position.id,
                entry_ids=full_lifecycle_plan.entry_order_ids,
                close_ids=full_lifecycle_plan.close_order_ids,
                match_status="okx_full_lifecycle_recovery_pending_settlement",
            )
            row["_dashboard_position_ids"] = _merge_values(
                row.get("_dashboard_position_ids"),
                [str(position.id)],
            )
            positions.append(position)
            all_positions.append(position)
            globally_linked_close_ids.update(full_lifecycle_plan.close_order_ids)
            created += 1
            samples.append(
                {
                    "kind": "okx_missing_full_lifecycle_recovered",
                    "position_id": int(position.id),
                    "symbol": full_lifecycle_plan.symbol,
                    "side": full_lifecycle_plan.side,
                    "okx_pos_id": full_lifecycle_plan.pos_id,
                    "entry_order_count": len(full_lifecycle_plan.entry_order_ids),
                    "close_order_count": len(full_lifecycle_plan.close_order_ids),
                    "quantity": full_lifecycle_plan.quantity,
                    "settlement_status": "settling",
                }
            )
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
                settlement_raw={"lifecycle_recovery": evidence},
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
            all_positions.append(position)
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
        "repaired_identity_count": repaired_identity,
        "repaired_order_link_count": repaired_order_links,
        "skipped_count": skipped,
        "samples": samples[:10],
    }


def _summary() -> dict[str, Any]:
    return {
        "created_count": 0,
        "repaired_identity_count": 0,
        "repaired_order_link_count": 0,
        "skipped_count": 0,
        "samples": [],
    }


async def _repair_partial_lifecycle_order_links(
    session: Any,
    lifecycle: tuple[tuple[str, str, str, str, tuple[str, ...]], dict[str, Any]],
    *,
    positions: list[Position],
    orders: list[Order],
    orders_by_id: dict[str, Order],
    entry_owners: dict[str, set[tuple[Any, ...]]],
    close_owners: dict[str, set[tuple[Any, ...]]],
    now: datetime,
) -> dict[str, Any] | None:
    """Refresh partial-history links only when fills conserve the live remainder."""
    _key, facts = lifecycle
    row = facts["row"]
    entry_ids = _split_ids(row.get("_dashboard_entry_order_ids"))
    if not entry_ids or facts["close_contracts"] >= facts["open_contracts"]:
        return None
    live_positions = [
        position for position in positions
        if position.is_open
        and _text(position.okx_inst_id).upper() == facts["inst_id"]
        and _text(position.okx_pos_id) == facts["pos_id"]
        and _text(position.side).lower() == facts["side"]
    ]
    if (
        len(live_positions) != 1
        or normalize_trading_symbol(live_positions[0].symbol) != facts["symbol"]
        or _split_ids(live_positions[0].entry_exchange_order_id) != entry_ids
    ):
        return None
    record = await _history_record(session, row)
    if (
        record is None
        or record.mode != facts["mode"]
        or record.row_identity != _text(row.get("_dashboard_history_row_identity"))
        or record.sync_status != "synced"
        or record.close_status != "partial"
        or _split_ids(record.entry_order_ids) != entry_ids
    ):
        return None
    owner = (facts["inst_id"], facts["pos_id"], facts["side"], facts["opened_at"])
    if any(entry_owners.get(order_id) != {owner} for order_id in entry_ids):
        return None
    entry_side = "sell" if facts["side"] == "short" else "buy"
    close_side = "buy" if facts["side"] == "short" else "sell"
    entry_orders = [orders_by_id.get(order_id) for order_id in sorted(entry_ids)]
    close_orders = [
        order for order in orders
        if _text(order.okx_inst_id).upper() == facts["inst_id"]
        and _text(order.side).lower() == close_side
        and order.filled_at is not None
        and facts["opened_at"] <= _aware_utc(order.filled_at)
        <= facts["closed_at"] + timedelta(milliseconds=1)
    ]
    close_ids = {_text(order.exchange_order_id) for order in close_orders}
    if (
        not close_ids
        or not _split_ids(record.close_order_ids).issubset(close_ids)
        or any(
            orders_by_id.get(_text(order.exchange_order_id)) is not order
            or entry_owners.get(_text(order.exchange_order_id), set()) - {owner}
            or close_owners.get(_text(order.exchange_order_id), set()) - {owner}
            for order in close_orders
        )
    ):
        return None
    for position in positions:
        if not _split_ids(position.close_exchange_order_id).intersection(close_ids):
            continue
        if (
            normalize_trading_symbol(position.symbol) != facts["symbol"]
            or _text(position.okx_inst_id).upper() != facts["inst_id"]
            or (_text(position.okx_pos_id) and _text(position.okx_pos_id) != facts["pos_id"])
            or _text(position.side).lower() != facts["side"]
            or _split_ids(position.entry_exchange_order_id) != entry_ids
        ):
            return None
    for group, side, target in (
        (entry_orders, entry_side, facts["open_contracts"]),
        (close_orders, close_side, facts["close_contracts"]),
    ):
        if any(
            order is None
            or _text(order.okx_inst_id).upper() != facts["inst_id"]
            or not _order_is_authoritative(order, expected_side=side, position_side=facts["side"])
            or _dict(order.okx_raw_fills).get("contract_size_verified") is not True
            for order in group
        ) or not _close_enough(sum(_order_contracts(order) for order in group), target):
            return None
    if any(order.filled_at is None for order in entry_orders):
        return None
    first_entry_at = min(_aware_utc(order.filled_at) for order in entry_orders)
    last_entry_at = max(_aware_utc(order.filled_at) for order in entry_orders)
    if (
        abs((first_entry_at - facts["opened_at"]).total_seconds()) > TIME_TOLERANCE_SECONDS
        or last_entry_at > min(_aware_utc(order.filled_at) for order in close_orders)
    ):
        return None
    all_orders = [*entry_orders, *close_orders]
    size = _order_contract_size(all_orders[0], row)
    remaining = (facts["open_contracts"] - facts["close_contracts"]) * size
    if (
        size <= 0
        or any(not _close_enough(_order_contract_size(order, row), size) for order in all_orders)
        or not _close_enough(_safe_float(live_positions[0].quantity), remaining)
        or not _price_close(_weighted_price(entry_orders), record.open_avg_px)
        or not _price_close(_weighted_price(close_orders), record.close_avg_px)
        or abs(sum(abs(_safe_float(order.fee)) for order in all_orders) - abs(record.fee))
        > max(abs(record.fee) * 1e-6, 1e-8)
    ):
        return None
    position_id = int(live_positions[0].id)
    missing_close_ids = close_ids - _split_ids(record.close_order_ids)
    if not missing_close_ids and str(position_id) in _split_ids(record.position_ids):
        return None
    raw = _dict(record.raw_row)
    record.raw_row = {
        **raw,
        "_bb_partial_lifecycle_link_recovery": {
            "source": REPAIR_SOURCE,
            "version": REPAIR_VERSION,
            "repaired_at": now.isoformat(),
            "previous_close_order_ids": sorted(_split_ids(record.close_order_ids)),
            "previous_position_ids": sorted(_split_ids(record.position_ids)),
            "added_close_order_ids": sorted(missing_close_ids),
        },
    }
    _merge_history_record_links(
        record,
        position_id=position_id,
        entry_ids=sorted(entry_ids),
        close_ids=sorted(close_ids),
        match_status=record.match_status,
    )
    row["_dashboard_close_order_ids"] = list(record.close_order_ids)
    row["_dashboard_position_ids"] = list(record.position_ids)
    return {
        "kind": "okx_partial_lifecycle_order_links_recovered",
        "symbol": facts["symbol"],
        "position_id": position_id,
        "added_close_order_ids": sorted(missing_close_ids),
    }


def _repair_linked_lifecycle_identity(
    lifecycle: tuple[tuple[str, str, str, str, tuple[str, ...]], dict[str, Any]],
    *,
    positions_by_close_id: dict[str, list[Position]],
    orders_by_id: dict[str, Order],
    close_owners: dict[str, set[tuple[Any, ...]]],
    now: datetime,
) -> list[dict[str, Any]]:
    """Restore identity and close time from a uniquely proven lifecycle."""
    _key, facts = lifecycle
    entry_ids = _split_ids(facts["row"].get("_dashboard_entry_order_ids"))
    close_ids = _split_ids(facts["row"].get("_dashboard_close_order_ids"))
    if not entry_ids or not close_ids or entry_ids.intersection(close_ids):
        return []
    if any(len(close_owners.get(order_id, set())) != 1 for order_id in close_ids):
        return []
    entry_side = "sell" if facts["side"] == "short" else "buy"
    close_side = "buy" if facts["side"] == "short" else "sell"
    for ids, side, target in (
        (entry_ids, entry_side, facts["open_contracts"]),
        (close_ids, close_side, facts["close_contracts"]),
    ):
        linked_orders = [orders_by_id.get(order_id) for order_id in ids]
        if any(
            order is None
            or _text(order.okx_inst_id).upper() != facts["inst_id"]
            or not _order_is_authoritative(order, expected_side=side, position_side=facts["side"])
            or order.filled_at is None
            or not (
                facts["opened_at"] - timedelta(seconds=TIME_TOLERANCE_SECONDS)
                <= _aware_utc(order.filled_at)
                <= facts["closed_at"] + timedelta(seconds=TIME_TOLERANCE_SECONDS)
            )
            for order in linked_orders
        ):
            return []
        if abs(sum(_order_contracts(order) for order in linked_orders) - target) > max(
            target * CONTRACT_TOLERANCE_RATIO, 1e-8
        ):
            return []
    first_entry_at = min(_aware_utc(orders_by_id[order_id].filled_at) for order_id in entry_ids)
    if abs((first_entry_at - facts["opened_at"]).total_seconds()) > TIME_TOLERANCE_SECONDS:
        return []
    if facts["entry_price"] <= 0:
        return []
    selected = {
        int(position.id): position
        for close_id in close_ids
        for position in positions_by_close_id.get(close_id, [])
    }
    samples = []
    for position in selected.values():
        position_close_ids = _split_ids(position.close_exchange_order_id)
        if (
            (_text(position.okx_pos_id) and _text(position.okx_pos_id) != facts["pos_id"])
            or (
                _text(position.okx_inst_id)
                and _text(position.okx_inst_id).upper() != facts["inst_id"]
            )
            or normalize_trading_symbol(position.symbol) != facts["symbol"]
            or str(position.side or "").lower() != facts["side"]
            or not position_close_ids
            or not position_close_ids.issubset(close_ids)
            or any(len(positions_by_close_id.get(order_id, [])) != 1 for order_id in position_close_ids)
        ):
            continue
        missing_identity = not _text(position.okx_pos_id) or not _text(position.okx_inst_id)
        if missing_identity and (
            _split_ids(position.entry_exchange_order_id) != entry_ids
            or not _close_enough(
                _safe_float(position.quantity),
                sum(
                    _order_base_quantity(
                        orders_by_id[order_id],
                        _order_contract_size(orders_by_id[order_id], facts["row"]),
                    )
                    for order_id in position_close_ids
                ),
            )
        ):
            continue
        identity_requires_repair = (
            missing_identity or _split_ids(position.entry_exchange_order_id) != entry_ids
        )
        authoritative_closed_at = max(
            _aware_utc(orders_by_id[order_id].filled_at) for order_id in position_close_ids
        )
        previous_closed_at = _aware_utc(position.closed_at)
        close_time_requires_repair = previous_closed_at != authoritative_closed_at
        if not identity_requires_repair and not close_time_requires_repair:
            continue
        raw = _dict(position.settlement_raw)
        evidence: dict[str, Any] = {}
        if identity_requires_repair:
            evidence["entry_identity_repair"] = {
                "source": REPAIR_SOURCE, "repaired_at": now.isoformat(),
                "history_record_id": facts["row"].get("_dashboard_history_record_id"),
                "previous_entry_order_ids": position.entry_exchange_order_id,
                "previous_created_at": _aware_utc(position.created_at).isoformat(),
                "previous_entry_price": position.entry_price,
                "previous_okx_pos_id": position.okx_pos_id,
                "previous_okx_inst_id": position.okx_inst_id,
                "authoritative_entry_order_ids": sorted(entry_ids),
                "authoritative_close_order_ids": sorted(position_close_ids),
            }
        if close_time_requires_repair:
            evidence["close_time_repair"] = {
                "source": REPAIR_SOURCE,
                "repaired_at": now.isoformat(),
                "history_record_id": facts["row"].get("_dashboard_history_record_id"),
                "previous_closed_at": previous_closed_at.isoformat() if previous_closed_at else None,
                "authoritative_closed_at": authoritative_closed_at.isoformat(),
                "authoritative_close_order_ids": sorted(position_close_ids),
            }
            position.closed_at = authoritative_closed_at
        position.settlement_raw = {
            **raw,
            **evidence,
            "next_settlement_retry_at": now.isoformat(),
        }
        position.entry_exchange_order_id = ",".join(sorted(entry_ids))
        position.okx_pos_id = facts["pos_id"]
        position.okx_inst_id = facts["inst_id"]
        position.created_at = facts["opened_at"]
        position.entry_price = facts["entry_price"]
        position.updated_at = now
        samples.append({
            "kind": (
                "okx_lifecycle_entry_identity_repaired"
                if identity_requires_repair else "okx_lifecycle_close_time_repaired"
            ),
            "position_id": int(position.id), "symbol": facts["symbol"],
        })
    return samples


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
    authoritative_entry_ids = tuple(
        sorted(_split_ids(facts["row"].get("_dashboard_entry_order_ids")))
    )
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
        if entry_ids and (
            not authoritative_entry_ids or entry_ids == authoritative_entry_ids
        ):
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
        if opened_delta > TIME_TOLERANCE_SECONDS:
            continue
        entry_contracts = sum(
            _order_contracts(orders_by_id[order_id])
            for order_id in entry_ids
            if order_id in orders_by_id
        )
        if not _close_enough(entry_contracts, facts["open_contracts"]):
            continue
        contract_delta = abs(entry_contracts - facts["open_contracts"])
        score = opened_delta / 60.0 + contract_delta * 100.0
        scored.append((score, entry_ids, group))
    if not scored:
        return None, []
    scored.sort(key=lambda item: item[0])
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) <= 1e-9:
        return None, []
    chosen_score, entry_ids, group = scored[0]
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
    close_ids: Iterable[str] = (),
    close_id: str = "",
    match_status: str = "okx_lifecycle_fragment_repair_pending_settlement",
) -> None:
    resolved_close_ids = _merge_values(close_ids, [close_id] if close_id else [])
    record.position_ids = _merge_values(record.position_ids, [str(position_id)] if position_id else [])
    record.entry_order_ids = _merge_values(record.entry_order_ids, entry_ids)
    record.close_order_ids = _merge_values(record.close_order_ids, resolved_close_ids)
    record.linked_order_ids = _merge_values(
        record.linked_order_ids,
        [*entry_ids, *resolved_close_ids],
    )
    record.match_status = match_status


async def _plan_missing_full_lifecycle(
    session: Any,
    lifecycle: tuple[tuple[str, str, str, str, tuple[str, ...]], dict[str, Any]],
    *,
    positions: list[Position],
    orders_by_id: dict[str, Order],
    duplicate_order_ids: set[str],
    entry_owners: dict[str, set[tuple[Any, ...]]],
    close_owners: dict[str, set[tuple[Any, ...]]],
    lifecycle_row_count: int,
) -> MissingLifecyclePlan | None:
    lifecycle_key, facts = lifecycle
    row = facts["row"]
    history_record = await _history_record(session, row)
    if (
        lifecycle_row_count != 1
        or history_record is None
        or str(history_record.mode or "").lower() != facts["mode"]
        or str(history_record.row_identity or "")
        != _text(row.get("_dashboard_history_row_identity"))
        or str(history_record.match_status or "") != "okx_account_position_history"
        or str(history_record.sync_status or "").lower() != "synced"
        or str(history_record.close_status or "").lower() != "full"
        or history_record.evidence_gaps
        or _split_ids(history_record.position_ids)
    ):
        return None

    entry_ids = tuple(sorted(_split_ids(row.get("_dashboard_entry_order_ids"))))
    close_ids = tuple(sorted(_split_ids(row.get("_dashboard_close_order_ids"))))
    if not entry_ids or not close_ids or set(entry_ids).intersection(close_ids):
        return None
    if (
        set(_split_ids(history_record.entry_order_ids)) != set(entry_ids)
        or set(_split_ids(history_record.close_order_ids)) != set(close_ids)
    ):
        return None
    owner_identity = (
        facts["inst_id"],
        facts["pos_id"],
        facts["side"],
        facts["opened_at"],
    )
    if any(
        len(entry_owners.get(order_id, set())) != 1
        or owner_identity not in entry_owners.get(order_id, set())
        for order_id in entry_ids
    ) or any(
        len(close_owners.get(order_id, set())) != 1
        or owner_identity not in close_owners.get(order_id, set())
        for order_id in close_ids
    ):
        return None

    open_contracts = _safe_float(facts["open_contracts"])
    close_contracts = _safe_float(facts["close_contracts"])
    if open_contracts <= 0 or not _close_enough(open_contracts, close_contracts):
        return None
    if (
        not _close_enough(_safe_float(history_record.open_max_pos), open_contracts)
        or not _close_enough(_safe_float(history_record.close_total_pos), close_contracts)
        or _safe_float(history_record.open_avg_px) <= 0
        or _safe_float(history_record.close_avg_px) <= 0
    ):
        return None

    all_order_ids = set(entry_ids) | set(close_ids)
    if all_order_ids.intersection(duplicate_order_ids):
        return None
    for position in positions:
        position_order_ids = _split_ids(position.entry_exchange_order_id) | _split_ids(
            position.close_exchange_order_id
        )
        if all_order_ids.intersection(position_order_ids):
            return None
        position_opened_at = getattr(position, "created_at", None)
        if (
            str(getattr(position, "okx_pos_id", "") or "").strip() == facts["pos_id"]
            and str(getattr(position, "okx_inst_id", "") or "").strip().upper()
            == facts["inst_id"]
            and str(getattr(position, "side", "") or "").lower() == facts["side"]
            and position_opened_at is not None
            and abs((_aware_utc(position_opened_at) - facts["opened_at"]).total_seconds())
            <= TIME_TOLERANCE_SECONDS
        ):
            return None

    entry_orders = tuple(orders_by_id.get(order_id) for order_id in entry_ids)
    close_orders = tuple(orders_by_id.get(order_id) for order_id in close_ids)
    if any(order is None for order in (*entry_orders, *close_orders)):
        return None
    resolved_entry_orders = tuple(order for order in entry_orders if order is not None)
    resolved_close_orders = tuple(order for order in close_orders if order is not None)
    entry_side = "sell" if facts["side"] == "short" else "buy"
    close_side = "buy" if facts["side"] == "short" else "sell"
    for order_group, expected_side in (
        (resolved_entry_orders, entry_side),
        (resolved_close_orders, close_side),
    ):
        if any(
            str(order.okx_inst_id or "").strip().upper() != facts["inst_id"]
            or not _order_is_authoritative(
                order,
                expected_side=expected_side,
                position_side=facts["side"],
            )
            or _dict(order.okx_raw_fills).get("contract_size_verified") is not True
            or order.filled_at is None
            or not (
                facts["opened_at"] - timedelta(seconds=TIME_TOLERANCE_SECONDS)
                <= _aware_utc(order.filled_at)
                <= facts["closed_at"] + timedelta(seconds=TIME_TOLERANCE_SECONDS)
            )
            for order in order_group
        ):
            return None

    decision_ids = {
        int(order.decision_id)
        for order in resolved_entry_orders
        if int(getattr(order, "decision_id", 0) or 0) > 0
    }
    if len(decision_ids) != 1 or any(
        not int(getattr(order, "decision_id", 0) or 0)
        for order in resolved_entry_orders
    ):
        return None
    decision_id = next(iter(decision_ids))
    decision = await session.get(AIDecision, decision_id)
    model_names = {str(order.model_name or "").strip() for order in resolved_entry_orders}
    if (
        decision is None
        or bool(decision.is_paper) != (facts["mode"] != "live")
        or not bool(decision.was_executed or decision.executed_at)
        or str(decision.action or "").lower() != facts["side"]
        or normalize_trading_symbol(decision.symbol) != facts["symbol"]
        or not str(decision.model_name or "").strip()
        or model_names != {str(decision.model_name).strip()}
    ):
        return None

    contract_sizes = [
        _order_contract_size(order, row)
        for order in (*resolved_entry_orders, *resolved_close_orders)
    ]
    if not contract_sizes or any(size <= 0 for size in contract_sizes):
        return None
    contract_size = contract_sizes[0]
    if any(not _close_enough(size, contract_size) for size in contract_sizes[1:]):
        return None
    quantity = open_contracts * contract_size
    entry_contracts = sum(_order_contracts(order) for order in resolved_entry_orders)
    close_contracts_from_orders = sum(_order_contracts(order) for order in resolved_close_orders)
    entry_quantity = sum(
        _order_base_quantity(order, contract_size) for order in resolved_entry_orders
    )
    close_quantity = sum(
        _order_base_quantity(order, contract_size) for order in resolved_close_orders
    )
    if (
        not _close_enough(entry_contracts, open_contracts)
        or not _close_enough(close_contracts_from_orders, close_contracts)
        or not _close_enough(entry_quantity, quantity)
        or not _close_enough(close_quantity, quantity)
    ):
        return None

    first_entry_at = min(
        _aware_utc(order.filled_at or order.created_at) for order in resolved_entry_orders
    )
    last_close_at = max(
        _aware_utc(order.filled_at or order.created_at) for order in resolved_close_orders
    )
    if (
        abs((first_entry_at - facts["opened_at"]).total_seconds()) > TIME_TOLERANCE_SECONDS
        or abs((last_close_at - facts["closed_at"]).total_seconds()) > TIME_TOLERANCE_SECONDS
    ):
        return None
    entry_price = _weighted_price(resolved_entry_orders)
    close_price = _weighted_price(resolved_close_orders)
    official_entry_price = _safe_float(history_record.open_avg_px)
    official_close_price = _safe_float(history_record.close_avg_px)
    if (
        entry_price <= 0
        or close_price <= 0
        or not _price_close(entry_price, official_entry_price)
        or not _price_close(close_price, official_close_price)
    ):
        return None

    return MissingLifecyclePlan(
        history_record=history_record,
        entry_order_ids=entry_ids,
        close_order_ids=close_ids,
        decision_id=decision_id,
        model_name=str(decision.model_name).strip(),
        symbol=facts["symbol"],
        side=facts["side"],
        inst_id=facts["inst_id"],
        pos_id=facts["pos_id"],
        quantity=quantity,
        entry_price=official_entry_price,
        close_price=official_close_price,
        leverage=max(_safe_float(row.get("lever") or row.get("leverage"), 1.0), 1.0),
        opened_at=facts["opened_at"],
        closed_at=facts["closed_at"],
    )


def _close_enough(left: float, right: float) -> bool:
    return abs(left - right) <= max(abs(left), abs(right), 1.0) * CONTRACT_TOLERANCE_RATIO


def _price_close(left: float, right: float) -> bool:
    return abs(left - right) <= max(abs(left), abs(right), 1e-8) * 0.002


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
