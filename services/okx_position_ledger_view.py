"""OKX-style historical position ledger views for Phase 3.

The dashboard should display grouped position lifecycles backed by OKX order
and fill facts, not raw local position fragments.  This module builds a
read-only view from the synced local OKX fact cache and marks evidence gaps
explicitly so they cannot be mistaken for clean training facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.symbols import normalize_trading_symbol, okx_inst_id_from_symbol, symbol_from_okx_inst_id
from models.trade import Order, Position
from services.okx_order_fact_sync import OKX_SYNC_CONFIRMED, OKX_SYNC_OKX_ONLY

NON_EXCHANGE_ORDER_TOKENS = {
    "-",
    "--",
    "0",
    "cancelled",
    "canceled",
    "error",
    "failed",
    "hold",
    "n/a",
    "na",
    "nan",
    "no-position",
    "no_position",
    "none",
    "null",
    "pending",
    "rejected",
}


@dataclass(slots=True)
class OkxLinkedFillRow:
    side: str
    quantity: float
    contracts: float
    contract_size: float
    price: float
    pnl: float | None
    pnl_pct: float | None
    fee: float
    order_id: str
    trade_id: str
    filled_at: datetime | None
    okx_confirmed: bool
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "quantity": _round(self.quantity),
            "contracts": _round(self.contracts),
            "contract_size": _round(self.contract_size),
            "price": _round(self.price),
            "pnl": _round_optional(self.pnl),
            "pnl_pct": _round_optional(self.pnl_pct),
            "fee": _round(self.fee),
            "order_id": self.order_id,
            "trade_id": self.trade_id,
            "filled_at": _iso(self.filled_at),
            "okx_confirmed": self.okx_confirmed,
            "source": self.source,
        }


@dataclass(slots=True)
class OkxPositionLedgerGroup:
    group_id: str
    symbol: str
    inst_id: str
    side: str
    leverage: float
    status: str
    status_label: str
    average_entry_price: float
    average_close_price: float
    realized_pnl: float
    realized_pnl_pct: float | None
    max_position_quantity: float
    closed_quantity: float
    opened_at: datetime | None
    closed_at: datetime | None
    position_ids: list[int] = field(default_factory=list)
    entry_order_ids: list[str] = field(default_factory=list)
    close_order_ids: list[str] = field(default_factory=list)
    linked_fills: list[OkxLinkedFillRow] = field(default_factory=list)
    evidence_complete: bool = False
    trainable: bool = False
    evidence_gaps: list[str] = field(default_factory=list)
    pnl_source: str = "position_realized_pnl"

    def as_dict(self, *, include_fills: bool = True) -> dict[str, Any]:
        payload = {
            "id": self.group_id,
            "group_id": self.group_id,
            "is_open": False,
            "symbol": self.symbol,
            "okx_inst_id": self.inst_id,
            "side": self.side,
            "leverage": _round(self.leverage),
            "position_status": self.status_label,
            "close_status": self.status,
            "close_status_label": self.status_label,
            "quantity": _round(self.closed_quantity),
            "max_position_quantity": _round(self.max_position_quantity),
            "closed_quantity": _round(self.closed_quantity),
            "entry_price": _round(self.average_entry_price),
            "current_price": _round(self.average_close_price),
            "average_entry_price": _round(self.average_entry_price),
            "average_close_price": _round(self.average_close_price),
            "realized_pnl": _round(self.realized_pnl),
            "realized_pnl_pct": _round_optional(self.realized_pnl_pct),
            "pnl_source": self.pnl_source,
            "opened_at": _iso(self.opened_at),
            "closed_at": _iso(self.closed_at),
            "position_ids": list(self.position_ids),
            "entry_order_ids": list(self.entry_order_ids),
            "close_order_ids": list(self.close_order_ids),
            "linked_order_count": len(self.linked_fills),
            "evidence_complete": self.evidence_complete,
            "trainable": self.trainable,
            "evidence_gaps": list(self.evidence_gaps),
            "ledger_source": "okx_native_grouped_cache",
        }
        if include_fills:
            payload["linked_fills"] = [row.as_dict() for row in self.linked_fills]
        return payload


@dataclass(slots=True)
class _LedgerPositionFragment:
    """Read-only split of one polluted local row into OKX lifecycle fragments."""

    id: int | None
    model_name: str
    execution_mode: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    leverage: float
    unrealized_pnl: float
    realized_pnl: float
    is_open: bool
    closed_at: datetime | None
    created_at: datetime | None
    okx_inst_id: str | None
    okx_pos_id: str | None
    entry_exchange_order_id: str | None
    close_exchange_order_id: str | None


def build_okx_position_ledger_groups(
    positions: list[Position],
    orders: list[Order],
) -> list[OkxPositionLedgerGroup]:
    """Build OKX-style grouped historical position rows from local OKX facts."""
    orders_by_id = _orders_by_exchange_id(orders)
    closed_positions = [
        position
        for position in positions
        if not bool(getattr(position, "is_open", False))
        and not _is_zero_quantity_residual(position)
    ]
    closed_positions = _split_polluted_sequential_lifecycle_positions(
        closed_positions,
        orders_by_id,
    )
    closed_positions = [
        position
        for position in closed_positions
        if not _is_superseded_position_residual(position, closed_positions)
    ]
    result: list[OkxPositionLedgerGroup] = []
    for key, rows in _group_closed_positions_by_lifecycle(closed_positions):
        rows = sorted(
            rows,
            key=lambda item: _as_utc(getattr(item, "created_at", None))
            or datetime.min.replace(tzinfo=UTC),
        )
        rows = _deduplicate_superseded_position_rows(rows)
        group = _build_group_from_positions(key, rows, orders_by_id)
        result.append(group)
    return sorted(
        result,
        key=lambda item: item.closed_at or item.opened_at or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )


def _group_closed_positions_by_lifecycle(
    positions: list[Position],
) -> list[tuple[tuple[str, str, str, str], list[Position]]]:
    """Group local fragments into OKX-style position lifecycles."""
    groups: list[tuple[tuple[str, str, str, str], list[Position]]] = []
    ordered_positions = sorted(
        positions,
        key=lambda item: (
            _position_base_key(item),
            _as_utc(getattr(item, "created_at", None)) or datetime.min.replace(tzinfo=UTC),
            _as_utc(getattr(item, "closed_at", None)) or datetime.min.replace(tzinfo=UTC),
            int(getattr(item, "id", 0) or 0),
        ),
    )
    for position in ordered_positions:
        for _key, rows in groups:
            if _position_belongs_to_lifecycle_group(position, rows):
                rows.append(position)
                break
        else:
            groups.append((_position_group_key(position), [position]))
    return groups


def _position_belongs_to_lifecycle_group(
    position: Position,
    rows: list[Position],
) -> bool:
    if not rows:
        return False
    if _position_base_key(position) != _position_base_key(rows[0]):
        return False

    position_pos_id = _position_pos_id(position)
    group_pos_ids = {_position_pos_id(row) for row in rows if _position_pos_id(row)}
    if position_pos_id and group_pos_ids:
        if position_pos_id not in group_pos_ids:
            return False
        return _position_order_sets_overlap(position, rows) or _position_time_window_matches_group(
            position,
            rows,
        )
    if position_pos_id and not group_pos_ids:
        return _position_order_sets_overlap(position, rows) or _position_time_window_matches_group(
            position,
            rows,
        )
    if group_pos_ids and not position_pos_id:
        return _position_order_sets_overlap(position, rows)
    if _position_order_sets_overlap(position, rows):
        return True
    return _position_time_window_matches_group(position, rows)


def _position_order_sets_overlap(position: Position, rows: list[Position]) -> bool:
    position_entry = set(_position_order_key(position, "entry_exchange_order_id"))
    position_close = set(_position_order_key(position, "close_exchange_order_id"))
    if not (position_entry or position_close):
        return False
    group_entry = {
        token for row in rows for token in _position_order_key(row, "entry_exchange_order_id")
    }
    group_close = {
        token for row in rows for token in _position_order_key(row, "close_exchange_order_id")
    }
    return bool((position_entry & group_entry) or (position_close & group_close))


def _position_time_window_matches_group(position: Position, rows: list[Position]) -> bool:
    opened = _as_utc(getattr(position, "created_at", None))
    closed = _as_utc(getattr(position, "closed_at", None))
    if opened is None or closed is None:
        return False
    group_opened = [
        value for row in rows if (value := _as_utc(getattr(row, "created_at", None))) is not None
    ]
    group_closed = [
        value for row in rows if (value := _as_utc(getattr(row, "closed_at", None))) is not None
    ]
    if not group_opened or not group_closed:
        return False
    opened_near = min(abs((opened - item).total_seconds()) for item in group_opened) <= 300
    closed_near = min(abs((closed - item).total_seconds()) for item in group_closed) <= 1800
    return bool(opened_near and closed_near)


def _split_polluted_sequential_lifecycle_positions(
    positions: list[Position],
    orders_by_id: dict[str, Order],
) -> list[Position]:
    result: list[Position] = []
    for position in positions:
        result.extend(_split_polluted_sequential_lifecycle_position(position, orders_by_id))
    return result


def _split_polluted_sequential_lifecycle_position(
    position: Position,
    orders_by_id: dict[str, Order],
) -> list[Position]:
    entry_ids = _position_order_key(position, "entry_exchange_order_id")
    close_ids = _position_order_key(position, "close_exchange_order_id")
    if len(entry_ids) < 2 or len(close_ids) < 2 or len(entry_ids) != len(close_ids):
        return [position]

    side = str(getattr(position, "side", "") or "").lower()
    expected_entry_side = "sell" if side == "short" else "buy" if side == "long" else ""
    expected_close_side = "buy" if side == "short" else "sell" if side == "long" else ""
    if not expected_entry_side or not expected_close_side:
        return [position]

    entry_orders = _matched_side_orders(entry_ids, orders_by_id, expected_entry_side)
    close_orders = _matched_side_orders(close_ids, orders_by_id, expected_close_side)
    if len(entry_orders) != len(entry_ids) or len(close_orders) != len(close_ids):
        return [position]

    pairs = _pair_sequential_entry_close_orders(entry_orders, close_orders)
    if len(pairs) < 2 or not _order_pairs_are_separate_lifecycles(pairs):
        return [position]
    if not _order_pairs_cover_position_quantity(position, pairs):
        return [position]

    return [_position_fragment_from_order_pair(position, entry, close) for entry, close in pairs]


def _matched_side_orders(
    order_ids: tuple[str, ...],
    orders_by_id: dict[str, Order],
    expected_side: str,
) -> list[Order]:
    orders: list[Order] = []
    for order_id in order_ids:
        order = orders_by_id.get(order_id)
        if order is None:
            continue
        if str(getattr(order, "side", "") or "").lower() != expected_side:
            continue
        if _order_time(order) is None:
            continue
        orders.append(order)
    return orders


def _pair_sequential_entry_close_orders(
    entry_orders: list[Order],
    close_orders: list[Order],
) -> list[tuple[Order, Order]]:
    remaining_closes = sorted(
        close_orders,
        key=lambda order: _order_time(order) or datetime.max.replace(tzinfo=UTC),
    )
    pairs: list[tuple[Order, Order]] = []
    for entry_order in sorted(
        entry_orders,
        key=lambda order: _order_time(order) or datetime.max.replace(tzinfo=UTC),
    ):
        entry_time = _order_time(entry_order)
        entry_quantity = _order_quantity(entry_order)
        if entry_time is None or entry_quantity <= 0:
            return []
        candidates = [
            close_order
            for close_order in remaining_closes
            if (close_time := _order_time(close_order)) is not None
            and close_time >= entry_time
            and _quantities_match(entry_quantity, _order_quantity(close_order))
        ]
        if not candidates:
            return []
        close_order = min(
            candidates,
            key=lambda order: _order_time(order) or datetime.max.replace(tzinfo=UTC),
        )
        pairs.append((entry_order, close_order))
        remaining_closes.remove(close_order)
    if remaining_closes:
        return []
    return sorted(
        pairs,
        key=lambda pair: _order_time(pair[0]) or datetime.max.replace(tzinfo=UTC),
    )


def _order_pairs_are_separate_lifecycles(pairs: list[tuple[Order, Order]]) -> bool:
    ordered = sorted(
        pairs,
        key=lambda pair: _order_time(pair[0]) or datetime.max.replace(tzinfo=UTC),
    )
    for previous, current in zip(ordered, ordered[1:], strict=False):
        previous_close_time = _order_time(previous[1])
        current_entry_time = _order_time(current[0])
        if previous_close_time is None or current_entry_time is None:
            return False
        if previous_close_time > current_entry_time:
            return False
    return True


def _order_pairs_cover_position_quantity(
    position: Position,
    pairs: list[tuple[Order, Order]],
) -> bool:
    position_quantity = abs(_safe_float(getattr(position, "quantity", None)))
    if position_quantity <= 0:
        return True
    paired_quantity = sum(
        min(_order_quantity(entry), _order_quantity(close)) for entry, close in pairs
    )
    return _quantities_match(position_quantity, paired_quantity, tolerance_ratio=0.05)


def _position_fragment_from_order_pair(
    position: Position,
    entry_order: Order,
    close_order: Order,
) -> Position:
    entry_quantity = _order_quantity(entry_order)
    close_quantity = _order_quantity(close_order)
    quantity = (
        min(entry_quantity, close_quantity)
        if entry_quantity > 0 and close_quantity > 0
        else max(entry_quantity, close_quantity)
    )
    entry_price = _safe_float(getattr(entry_order, "price", None), 0.0) or _safe_float(
        getattr(position, "entry_price", None),
        0.0,
    )
    close_price = _safe_float(getattr(close_order, "price", None), 0.0) or _safe_float(
        getattr(position, "current_price", None),
        0.0,
    )
    realized_pnl = _order_realized_pnl(close_order)
    if realized_pnl is None:
        realized_pnl = _estimated_pair_pnl(
            side=str(getattr(position, "side", "") or "").lower(),
            quantity=quantity,
            entry_price=entry_price,
            close_price=close_price,
        )
    return _LedgerPositionFragment(
        id=getattr(position, "id", None),
        model_name=str(getattr(position, "model_name", "") or ""),
        execution_mode=str(getattr(position, "execution_mode", "") or ""),
        symbol=str(getattr(position, "symbol", "") or ""),
        side=str(getattr(position, "side", "") or "").lower(),
        quantity=quantity,
        entry_price=entry_price,
        current_price=close_price or entry_price,
        leverage=_safe_float(getattr(position, "leverage", None), 1.0) or 1.0,
        unrealized_pnl=0.0,
        realized_pnl=realized_pnl or 0.0,
        is_open=False,
        closed_at=_order_time(close_order),
        created_at=_order_time(entry_order),
        okx_inst_id=getattr(position, "okx_inst_id", None)
        or getattr(entry_order, "okx_inst_id", None)
        or getattr(close_order, "okx_inst_id", None),
        okx_pos_id=getattr(position, "okx_pos_id", None),
        entry_exchange_order_id=str(getattr(entry_order, "exchange_order_id", "") or "").strip()
        or None,
        close_exchange_order_id=str(getattr(close_order, "exchange_order_id", "") or "").strip()
        or None,
    )


def _build_group_from_positions(
    key: tuple[str, str, str, str],
    positions: list[Position],
    orders_by_id: dict[str, Order],
) -> OkxPositionLedgerGroup:
    mode, symbol, side, lifecycle_open = key
    inst_id = _position_inst_id(positions[0]) or okx_inst_id_from_symbol(symbol) or ""
    position_ids = [int(pos.id) for pos in positions if getattr(pos, "id", None) is not None]
    opened_at_values = [_as_utc(pos.created_at) for pos in positions if _as_utc(pos.created_at)]
    closed_at_values = [_as_utc(pos.closed_at) for pos in positions if _as_utc(pos.closed_at)]
    opened_at = min(opened_at_values) if opened_at_values else None
    closed_at = max(closed_at_values) if closed_at_values else None

    entry_ids = _ordered_tokens(
        token
        for pos in positions
        for token in _split_exchange_order_ids(getattr(pos, "entry_exchange_order_id", None))
    )
    close_ids = _ordered_tokens(
        token
        for pos in positions
        for token in _split_exchange_order_ids(getattr(pos, "close_exchange_order_id", None))
    )

    all_order_ids = _ordered_tokens([*entry_ids, *close_ids])
    linked_orders = [
        orders_by_id[order_id] for order_id in all_order_ids if order_id in orders_by_id
    ]
    linked_fills = [_fill_row_from_order(order) for order in linked_orders]
    linked_fills = sorted(
        [row for row in linked_fills if row is not None],
        key=lambda row: row.filled_at or datetime.min.replace(tzinfo=UTC),
    )

    closed_quantity = sum(abs(_safe_float(getattr(pos, "quantity", None))) for pos in positions)
    max_quantity = max(
        [closed_quantity, *[abs(_safe_float(getattr(pos, "quantity", None))) for pos in positions]],
        default=closed_quantity,
    )
    entry_price = _weighted_average(
        (
            abs(_safe_float(getattr(pos, "quantity", None))),
            _safe_float(getattr(pos, "entry_price", None)),
        )
        for pos in positions
    )
    close_price = _weighted_average(
        (
            abs(_safe_float(getattr(pos, "quantity", None))),
            _safe_float(getattr(pos, "current_price", None)),
        )
        for pos in positions
    )
    if close_ids:
        close_price_from_orders = _weighted_average(
            (row.quantity, row.price)
            for row in linked_fills
            if row.order_id in close_ids and row.quantity > 0 and row.price > 0
        )
        if close_price_from_orders > 0:
            close_price = close_price_from_orders

    realized_pnl = sum(_safe_float(getattr(pos, "realized_pnl", None)) for pos in positions)
    if not realized_pnl:
        realized_pnl = sum(
            _safe_float(row.pnl)
            for row in linked_fills
            if row.order_id in close_ids and row.pnl is not None
        )
        pnl_source = "okx_fill_pnl" if realized_pnl else "position_realized_pnl"
    else:
        pnl_source = "position_realized_pnl"
    realized_pct = None
    entry_notional = closed_quantity * entry_price
    if entry_notional > 0:
        realized_pct = realized_pnl / entry_notional * 100.0

    leverage = max(
        (_safe_float(getattr(pos, "leverage", None), 0.0) for pos in positions), default=0.0
    )
    gaps: list[str] = []
    if not inst_id:
        gaps.append("missing_okx_inst_id")
    if not entry_ids:
        gaps.append("missing_entry_order_id")
    if not close_ids:
        gaps.append("missing_close_order_id")
    missing_orders = [order_id for order_id in all_order_ids if order_id not in orders_by_id]
    if missing_orders:
        gaps.append("missing_linked_order_rows")
    if any(not row.okx_confirmed for row in linked_fills):
        gaps.append("linked_order_not_okx_confirmed")
    evidence_complete = not gaps and bool(close_ids) and bool(entry_ids)
    status = "partial" if len(positions) > 1 and not evidence_complete else "full"
    status_label = "部分平仓" if status == "partial" else "全部平仓"

    group_id = _group_id(mode, inst_id or symbol, side, lifecycle_open, closed_at)
    return OkxPositionLedgerGroup(
        group_id=group_id,
        symbol=symbol_from_okx_inst_id(inst_id) or symbol,
        inst_id=inst_id,
        side=side,
        leverage=leverage or 1.0,
        status=status,
        status_label=status_label,
        average_entry_price=entry_price,
        average_close_price=close_price,
        realized_pnl=realized_pnl,
        realized_pnl_pct=realized_pct,
        max_position_quantity=max_quantity,
        closed_quantity=closed_quantity,
        opened_at=opened_at,
        closed_at=closed_at,
        position_ids=position_ids,
        entry_order_ids=entry_ids,
        close_order_ids=close_ids,
        linked_fills=linked_fills,
        evidence_complete=evidence_complete,
        trainable=evidence_complete,
        evidence_gaps=gaps,
        pnl_source=pnl_source,
    )


def _deduplicate_superseded_position_rows(positions: list[Position]) -> list[Position]:
    if len(positions) <= 1:
        return positions
    keep: list[Position] = []
    for bucket in _duplicate_position_buckets(positions).values():
        winner = max(bucket, key=_position_evidence_score)
        keep.append(winner)
    return sorted(
        keep,
        key=lambda item: _as_utc(getattr(item, "created_at", None))
        or datetime.min.replace(tzinfo=UTC),
    )


def _is_superseded_position_residual(position: Position, positions: list[Position]) -> bool:
    return any(
        other is not position and _position_supersedes(other, position) for other in positions
    )


def _duplicate_position_buckets(positions: list[Position]) -> dict[tuple[Any, ...], list[Position]]:
    buckets: dict[tuple[Any, ...], list[Position]] = {}
    for position in positions:
        buckets.setdefault(_duplicate_position_key(position), []).append(position)
    return buckets


def _duplicate_position_key(position: Position) -> tuple[Any, ...]:
    opened = _as_utc(getattr(position, "created_at", None))
    closed = _as_utc(getattr(position, "closed_at", None))
    opened_key = int(opened.timestamp()) if opened else 0
    closed_key = int(closed.timestamp()) if closed else 0
    return (
        _position_base_key(position),
        str(getattr(position, "okx_pos_id", "") or "").strip(),
        round(abs(_safe_float(getattr(position, "quantity", None))), 12),
        _position_order_key(position, "entry_exchange_order_id"),
        _position_order_key(position, "close_exchange_order_id"),
        opened_key,
        closed_key,
    )


def _position_supersedes(candidate: Position, other: Position) -> bool:
    candidate_entry = set(_position_order_key(candidate, "entry_exchange_order_id"))
    candidate_close = set(_position_order_key(candidate, "close_exchange_order_id"))
    other_entry = set(_position_order_key(other, "entry_exchange_order_id"))
    other_close = set(_position_order_key(other, "close_exchange_order_id"))
    if not _same_position_lifecycle(candidate, other):
        return False
    if (
        candidate_entry == other_entry
        and candidate_close == other_close
        and (candidate_entry or candidate_close)
    ):
        if not _same_position_quantity(candidate, other):
            return False
        return _position_evidence_score(candidate) > _position_evidence_score(other)
    if not (candidate_entry or candidate_close):
        return False
    if not (candidate_entry and candidate_close):
        return False
    entry_covers = not other_entry or candidate_entry.issuperset(other_entry)
    close_covers = not other_close or candidate_close.issuperset(other_close)
    strictly_more = candidate_entry != other_entry or candidate_close != other_close
    return bool(entry_covers and close_covers and strictly_more)


def _same_position_quantity(left: Position, right: Position) -> bool:
    return (
        abs(
            abs(_safe_float(getattr(left, "quantity", None)))
            - abs(_safe_float(getattr(right, "quantity", None)))
        )
        <= 1e-12
    )


def _same_position_lifecycle(left: Position, right: Position) -> bool:
    if _position_base_key(left) != _position_base_key(right):
        return False
    left_pos_id = str(getattr(left, "okx_pos_id", "") or "").strip()
    right_pos_id = str(getattr(right, "okx_pos_id", "") or "").strip()
    if left_pos_id and right_pos_id and left_pos_id != right_pos_id:
        return False
    left_opened = _as_utc(getattr(left, "created_at", None))
    right_opened = _as_utc(getattr(right, "created_at", None))
    left_closed = _as_utc(getattr(left, "closed_at", None))
    right_closed = _as_utc(getattr(right, "closed_at", None))
    if left_opened and right_opened and abs((left_opened - right_opened).total_seconds()) > 3:
        return False
    if left_closed and right_closed and abs((left_closed - right_closed).total_seconds()) > 3:
        return False
    return True


def _position_evidence_score(position: Position) -> tuple[int, int, int, int, float, int]:
    entry_ids = _position_order_key(position, "entry_exchange_order_id")
    close_ids = _position_order_key(position, "close_exchange_order_id")
    quantity = abs(_safe_float(getattr(position, "quantity", None)))
    realized = abs(_safe_float(getattr(position, "realized_pnl", None)))
    updated_at = _as_utc(getattr(position, "updated_at", None))
    updated_key = updated_at.timestamp() if updated_at else 0.0
    return (
        1 if entry_ids else 0,
        1 if close_ids else 0,
        1 if quantity > 1e-12 else 0,
        1 if realized > 1e-12 else 0,
        updated_key,
        int(getattr(position, "id", 0) or 0),
    )


def _is_zero_quantity_residual(position: Position) -> bool:
    return (
        not bool(getattr(position, "is_open", False))
        and abs(_safe_float(getattr(position, "quantity", None))) <= 1e-12
        and abs(_safe_float(getattr(position, "realized_pnl", None))) <= 1e-12
        and not _position_order_key(position, "entry_exchange_order_id")
        and not _position_order_key(position, "close_exchange_order_id")
    )


def _position_group_key(position: Position) -> tuple[str, str, str, str]:
    mode, symbol, side = _position_base_key(position)
    pos_id = _position_pos_id(position)
    if pos_id:
        return mode, symbol, side, f"okx_pos:{pos_id}"
    close_key = _position_order_key(position, "close_exchange_order_id")
    if close_key:
        return mode, symbol, side, f"close:{','.join(close_key)}"
    entry_key = _position_order_key(position, "entry_exchange_order_id")
    if entry_key:
        return mode, symbol, side, f"entry:{','.join(entry_key)}"
    opened_bucket = _position_open_bucket(position)
    if opened_bucket:
        return mode, symbol, side, f"opened:{opened_bucket}"
    position_id = getattr(position, "id", None)
    return mode, symbol, side, f"row:{position_id or ''}"


def _position_base_key(position: Position) -> tuple[str, str, str]:
    mode = str(getattr(position, "execution_mode", "") or "")
    inst_id = _position_inst_id(position)
    symbol = symbol_from_okx_inst_id(inst_id) or normalize_trading_symbol(
        getattr(position, "symbol", None)
    )
    side = str(getattr(position, "side", "") or "").lower()
    return mode, symbol, side


def _position_pos_id(position: Position) -> str:
    return str(getattr(position, "okx_pos_id", "") or "").strip()


def _position_order_key(position: Position, field_name: str) -> tuple[str, ...]:
    return tuple(sorted(_split_exchange_order_ids(getattr(position, field_name, None))))


def _position_open_bucket(position: Position) -> str:
    opened_at = _as_utc(getattr(position, "created_at", None))
    return opened_at.replace(microsecond=0).isoformat() if opened_at else ""


def _position_inst_id(position: Position) -> str:
    inst_id = str(getattr(position, "okx_inst_id", "") or "").strip().upper()
    if inst_id:
        return inst_id
    return okx_inst_id_from_symbol(getattr(position, "symbol", None)) or ""


def _orders_by_exchange_id(orders: list[Order]) -> dict[str, Order]:
    result: dict[str, Order] = {}
    for order in orders:
        for token in _split_exchange_order_ids(getattr(order, "exchange_order_id", None)):
            result.setdefault(token, order)
    return result


def _fill_row_from_order(order: Order) -> OkxLinkedFillRow | None:
    order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
    if not order_id:
        return None
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    contracts = _first_positive(
        raw.get("contracts"),
        raw.get("filled_contracts"),
        getattr(order, "okx_fill_contracts", None),
        default=0.0,
    )
    contract_size = _first_positive(raw.get("contract_size"), raw.get("contractSize"), default=0.0)
    quantity = _first_positive(
        raw.get("base_quantity"), getattr(order, "quantity", None), default=0.0
    )
    if quantity <= 0 and contracts > 0:
        quantity = contracts * (contract_size if contract_size > 0 else 1.0)
    price = _first_positive(
        raw.get("avg_price"), raw.get("average"), getattr(order, "price", None), default=0.0
    )
    fee = _first_positive(raw.get("fee_abs"), getattr(order, "fee", None), default=0.0)
    pnl = _safe_float(raw.get("fill_pnl"), None)
    if pnl is None:
        pnl = _safe_float(getattr(order, "okx_fill_pnl", None), None)
    timestamp = (
        _as_utc(raw.get("timestamp"))
        or _as_utc(getattr(order, "filled_at", None))
        or _as_utc(getattr(order, "created_at", None))
    )
    trade_ids = raw.get("trade_ids")
    if not isinstance(trade_ids, list):
        trade_ids = [
            token for token in str(getattr(order, "okx_trade_ids", "") or "").split(",") if token
        ]
    trade_id = ",".join(str(item) for item in trade_ids if str(item).strip())
    sync_status = str(getattr(order, "okx_sync_status", "") or "").strip()
    okx_confirmed = sync_status in {OKX_SYNC_CONFIRMED, OKX_SYNC_OKX_ONLY}
    source = "okx_raw_fills" if raw else "local_order_cache"
    if bool(raw.get("position_snapshot_confirmed")) and not bool(
        raw.get("fills_history_confirmed")
    ):
        source = "okx_current_position_snapshot"
    return OkxLinkedFillRow(
        side=str(getattr(order, "side", "") or "").lower(),
        quantity=quantity,
        contracts=contracts,
        contract_size=contract_size,
        price=price,
        pnl=pnl,
        pnl_pct=None,
        fee=fee,
        order_id=order_id,
        trade_id=trade_id,
        filled_at=timestamp,
        okx_confirmed=okx_confirmed,
        source=source,
    )


def _order_time(order: Order) -> datetime | None:
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    return (
        _as_utc(raw.get("timestamp"))
        or _as_utc(getattr(order, "filled_at", None))
        or _as_utc(getattr(order, "created_at", None))
    )


def _order_quantity(order: Order) -> float:
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    contracts = _first_positive(
        raw.get("contracts"),
        raw.get("filled_contracts"),
        getattr(order, "okx_fill_contracts", None),
        default=0.0,
    )
    contract_size = _first_positive(raw.get("contract_size"), raw.get("contractSize"), default=0.0)
    quantity = _first_positive(
        raw.get("base_quantity"),
        raw.get("filled_base_quantity"),
        getattr(order, "quantity", None),
        default=0.0,
    )
    if quantity <= 0 and contracts > 0:
        quantity = contracts * (contract_size if contract_size > 0 else 1.0)
    return quantity


def _order_realized_pnl(order: Order) -> float | None:
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    pnl = _safe_float(raw.get("fill_pnl"), None)
    if pnl is None:
        pnl = _safe_float(getattr(order, "okx_fill_pnl", None), None)
    return pnl


def _estimated_pair_pnl(
    *,
    side: str,
    quantity: float,
    entry_price: float,
    close_price: float,
) -> float:
    if quantity <= 0 or entry_price <= 0 or close_price <= 0:
        return 0.0
    if side == "short":
        return (entry_price - close_price) * quantity
    if side == "long":
        return (close_price - entry_price) * quantity
    return 0.0


def _quantities_match(
    left: float,
    right: float,
    *,
    tolerance_ratio: float = 0.001,
) -> bool:
    left = abs(_safe_float(left) or 0.0)
    right = abs(_safe_float(right) or 0.0)
    if left <= 0 or right <= 0:
        return left <= 1e-12 and right <= 1e-12
    tolerance = max(left, right) * tolerance_ratio
    return abs(left - right) <= max(tolerance, 1e-12)


def _split_exchange_order_ids(value: Any) -> list[str]:
    tokens = [str(value or "").strip()]
    if not tokens[0]:
        return []
    for separator in (",", ";", "|", "\n", "\t", " "):
        pieces: list[str] = []
        for token in tokens:
            pieces.extend(part.strip() for part in token.split(separator) if part.strip())
        tokens = pieces
    return _ordered_tokens(token for token in tokens if _is_exchange_order_token(token))


def _is_exchange_order_token(value: Any) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    return token.lower() not in NON_EXCHANGE_ORDER_TOKENS


def _ordered_tokens(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if token and token not in seen:
            result.append(token)
            seen.add(token)
    return result


def _weighted_average(values: Any) -> float:
    total = 0.0
    weight = 0.0
    for quantity, price in values:
        quantity = _safe_float(quantity)
        price = _safe_float(price)
        if quantity <= 0 or price <= 0:
            continue
        total += quantity * price
        weight += quantity
    return total / weight if weight > 0 else 0.0


def _first_positive(*values: Any, default: float = 0.0) -> float:
    for value in values:
        number = _safe_float(value)
        if number > 0:
            return number
    return default


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _as_utc(value).isoformat() if _as_utc(value) else None


def _round(value: Any) -> float:
    return round(float(_safe_float(value) or 0.0), 8)


def _round_optional(value: Any) -> float | None:
    if value is None:
        return None
    return _round(value)


def _group_id(
    mode: str,
    instrument: str,
    side: str,
    opened_at: str,
    closed_at: datetime | None,
) -> str:
    closed_text = closed_at.replace(microsecond=0).isoformat() if closed_at else ""
    raw = "|".join([mode, instrument, side, opened_at, closed_text])
    return raw.replace("/", "_").replace(":", "").replace("+", "")
