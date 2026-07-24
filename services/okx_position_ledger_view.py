"""OKX-style historical position ledger views for Phase 3.

The dashboard should display grouped position lifecycles backed by OKX order
and fill facts, not raw local position fragments.  This module builds a
read-only view from the synced local OKX fact cache and marks evidence gaps
explicitly so they cannot be mistaken for clean training facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from core.symbols import normalize_trading_symbol, okx_inst_id_from_symbol, symbol_from_okx_inst_id
from models.trade import Order, Position
from services.okx_order_fact_sync import (
    OKX_SYNC_CONFIRMED,
    OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
    OKX_SYNC_OKX_ONLY,
    OKX_SYNC_ORDER_DETAIL_CONFIRMED,
)

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
FUNDING_FEE_BILL_SUBTYPES = {"173", "174"}
FINAL_LEDGER_SETTLEMENT_STATUSES = frozenset(
    {"reconciled", "settled", "okx_position_history"}
)


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
    close_fill_pnl: float = 0.0
    entry_fee: float = 0.0
    close_fee: float = 0.0
    funding_fee: float = 0.0
    funding_bill_count: int = 0
    funding_fee_source: str = "none"
    settlement_status: str = ""
    settlement_source: str = ""
    realized_pnl_formula: str = "close_fill_pnl_plus_funding_fee_minus_entry_and_close_fees"

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
            "close_fill_pnl": _round(self.close_fill_pnl),
            "entry_fee": _round(self.entry_fee),
            "close_fee": _round(self.close_fee),
            "funding_fee": _round(self.funding_fee),
            "funding_bill_count": int(self.funding_bill_count),
            "funding_fee_source": self.funding_fee_source,
            "settlement_status": self.settlement_status,
            "settlement_source": self.settlement_source,
            "realized_pnl_formula": self.realized_pnl_formula,
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
    source_position_ids: list[int] = field(default_factory=list)
    strict_order_lifecycle: bool = False
    settlement_status: str = ""
    settlement_source: str = ""


@dataclass(frozen=True, slots=True)
class _LinkedOrderPnlComponents:
    close_fill_pnl: float
    entry_fee: float
    close_fee: float
    close_quantity: float
    entry_quantity: float
    quantity_match_source: str = ""
    quantity_mismatch: bool = False
    source: str = "okx_linked_order_net_pnl"


@dataclass(frozen=True, slots=True)
class _FundingFeeComponents:
    funding_fee: float
    bill_count: int
    source: str


@dataclass(frozen=True, slots=True)
class _StoredSettlementComponents:
    close_fill_pnl: float
    entry_fee: float
    close_fee: float
    funding_fee: float
    realized_pnl: float
    source: str
    status: str
    preferred: bool


@dataclass(frozen=True, slots=True)
class _OfficialPositionHistoryComponents:
    close_fill_pnl: float
    entry_fee: float
    close_fee: float
    funding_fee: float
    realized_pnl: float
    entry_price: float
    close_price: float
    closed_quantity: float
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    source: str = "okx_position_history_realized_pnl"


def build_okx_position_ledger_groups(
    positions: list[Position],
    orders: list[Order],
    account_bills: list[Any] | None = None,
    position_history_rows: list[dict[str, Any]] | None = None,
    include_order_lifecycle_fragments: bool = True,
    require_order_lifecycle_source_positions: bool = False,
) -> list[OkxPositionLedgerGroup]:
    """Build OKX-style grouped historical position rows from local OKX facts."""
    orders_by_id = _orders_by_exchange_id(orders)
    funding_bills = list(account_bills or [])
    official_position_history_rows = list(position_history_rows or [])
    closed_positions = [
        position
        for position in positions
        if not bool(getattr(position, "is_open", False))
        and not _is_zero_quantity_residual(position)
        and not _has_explicit_superseded_position_metadata(position)
    ]
    closed_positions = _split_polluted_sequential_lifecycle_positions(
        closed_positions,
        orders_by_id,
    )
    if include_order_lifecycle_fragments:
        closed_positions = _append_confirmed_order_lifecycle_fragments(
            closed_positions,
            orders_by_id,
            require_source_positions=require_order_lifecycle_source_positions,
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
        rows = _drop_rows_superseded_by_generated_lifecycle(rows)
        if not rows:
            continue
        group = _build_group_from_positions(
            key,
            rows,
            orders_by_id,
            funding_bills,
            official_position_history_rows,
        )
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
            -len(_position_order_key(item, "close_exchange_order_id")),
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
        if _same_okx_position_id_lifecycle(position, rows):
            return True
        if _position_lifecycle_anchor(position) and any(
            _position_lifecycle_anchor(row) for row in rows
        ):
            return _position_lifecycle_anchor_matches_group(position, rows)
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


def _same_okx_position_id_lifecycle(position: Position, rows: list[Position]) -> bool:
    if _is_generated_order_lifecycle_fragment(position) or any(
        _is_generated_order_lifecycle_fragment(row) for row in rows
    ):
        return False
    position_pos_id = _position_pos_id(position)
    if not position_pos_id:
        return False
    if not any(_position_pos_id(row) == position_pos_id for row in rows):
        return False
    if _position_order_sets_overlap(position, rows):
        return True
    position_anchor = _position_lifecycle_anchor(position)
    if position_anchor:
        return _position_lifecycle_anchor_matches_group(position, rows)
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


def _position_lifecycle_anchor(position: Position) -> tuple[str, ...]:
    close_ids = _position_order_key(position, "close_exchange_order_id")
    if close_ids:
        return ("close", *close_ids)
    entry_ids = _position_order_key(position, "entry_exchange_order_id")
    if entry_ids:
        return ("entry", *entry_ids)
    return ()


def _position_lifecycle_anchor_matches_group(
    position: Position,
    rows: list[Position],
) -> bool:
    if _is_generated_order_lifecycle_fragment(position) or any(
        _is_generated_order_lifecycle_fragment(row) for row in rows
    ):
        return _generated_order_lifecycle_matches_group(position, rows)
    anchor = _position_lifecycle_anchor(position)
    if not anchor:
        return _position_time_window_matches_group(position, rows)
    position_close = set(_position_order_key(position, "close_exchange_order_id"))
    group_close = {
        token for row in rows for token in _position_order_key(row, "close_exchange_order_id")
    }
    if position_close and group_close:
        return bool(position_close & group_close)
    row_anchors = {_position_lifecycle_anchor(row) for row in rows}
    if anchor in row_anchors:
        return True
    if _position_order_sets_overlap(position, rows):
        return True
    return _position_time_window_matches_group(position, rows)


def _is_generated_order_lifecycle_fragment(position: Position) -> bool:
    return isinstance(position, _LedgerPositionFragment) and getattr(position, "model_name", "") == "okx_authoritative_sync"


def _generated_order_lifecycle_matches_group(position: Position, rows: list[Position]) -> bool:
    position_close = set(_position_order_key(position, "close_exchange_order_id"))
    group_close = {
        token for row in rows for token in _position_order_key(row, "close_exchange_order_id")
    }
    if position_close and group_close:
        return bool(position_close & group_close)
    return _position_time_window_matches_group(position, rows)


def _append_confirmed_order_lifecycle_fragments(
    positions: list[Position],
    orders_by_id: dict[str, Order],
    *,
    require_source_positions: bool = False,
) -> list[Position]:
    result: list[Position] = list(positions)
    seen_anchors = {
        (
            _position_base_key(position),
            _position_order_key(position, "entry_exchange_order_id"),
            _position_order_key(position, "close_exchange_order_id"),
        )
        for position in positions
        if _position_order_key(position, "entry_exchange_order_id")
        and _position_order_key(position, "close_exchange_order_id")
    }
    for key, rows in _positions_by_base_key(positions).items():
        side = key[2]
        expected_entry_side = "sell" if side == "short" else "buy" if side == "long" else ""
        expected_close_side = "buy" if side == "short" else "sell" if side == "long" else ""
        if not expected_entry_side or not expected_close_side:
            continue
        relevant_orders = [
            order
            for order in orders_by_id.values()
            if _order_matches_position_base(order, key) and _order_is_okx_confirmed(order)
        ]
        entry_orders = [
            order
            for order in relevant_orders
            if str(getattr(order, "side", "") or "").lower() == expected_entry_side
            and _order_time(order) is not None
        ]
        close_orders = [
            order
            for order in relevant_orders
            if str(getattr(order, "side", "") or "").lower() == expected_close_side
            and _order_time(order) is not None
            and (_order_realized_pnl(order) is not None or _order_has_close_evidence(order))
        ]
        if not entry_orders or not close_orders:
            continue
        for fragment in _confirmed_order_lifecycle_fragments_for_base(
            key,
            rows,
            entry_orders=entry_orders,
            close_orders=close_orders,
        ):
            if require_source_positions and not getattr(fragment, "source_position_ids", None):
                continue
            anchor = (
                key,
                _position_order_key(fragment, "entry_exchange_order_id"),
                _position_order_key(fragment, "close_exchange_order_id"),
            )
            if anchor in seen_anchors:
                continue
            result.append(fragment)
            seen_anchors.add(anchor)
    return result


def _positions_by_base_key(positions: list[Position]) -> dict[tuple[str, str, str], list[Position]]:
    result: dict[tuple[str, str, str], list[Position]] = {}
    for position in positions:
        result.setdefault(_position_base_key(position), []).append(position)
    return result


def _order_matches_position_base(order: Order, key: tuple[str, str, str]) -> bool:
    mode, symbol, _side = key
    if str(getattr(order, "execution_mode", "") or "") != mode:
        return False
    order_inst_id = str(getattr(order, "okx_inst_id", "") or "").strip().upper()
    order_symbol = symbol_from_okx_inst_id(order_inst_id) or normalize_trading_symbol(
        getattr(order, "symbol", None)
    )
    return order_symbol == symbol


def _order_is_okx_confirmed(order: Order) -> bool:
    sync_status = str(getattr(order, "okx_sync_status", "") or "").strip()
    return sync_status in {
        OKX_SYNC_CONFIRMED,
        OKX_SYNC_OKX_ONLY,
        OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
        OKX_SYNC_ORDER_DETAIL_CONFIRMED,
    }


def _order_has_close_evidence(order: Order) -> bool:
    raw = getattr(order, "okx_raw_fills", None)
    raw = raw if isinstance(raw, dict) else {}
    return bool(
        raw.get("fills_history_confirmed")
        or raw.get("order_detail_confirmed")
        or raw.get("execution_result_confirmed")
    )


def _confirmed_order_lifecycle_fragments_for_base(
    key: tuple[str, str, str],
    positions: list[Position],
    *,
    entry_orders: list[Order],
    close_orders: list[Order],
) -> list[Position]:
    side = key[2]
    entry_side = "sell" if side == "short" else "buy" if side == "long" else ""
    close_side = "buy" if side == "short" else "sell" if side == "long" else ""
    if not entry_side or not close_side:
        return []
    order_stream = sorted(
        [*entry_orders, *close_orders],
        key=lambda order: _order_time(order) or datetime.max.replace(tzinfo=UTC),
    )
    active_entries: list[Order] = []
    active_closes: list[Order] = []
    open_quantity = 0.0
    max_quantity = 0.0
    fragments: list[Position] = []
    for order in order_stream:
        order_side = str(getattr(order, "side", "") or "").lower()
        quantity = _order_quantity(order)
        if quantity <= 0:
            continue
        if order_side == entry_side:
            if open_quantity <= max(quantity * 0.001, 1e-12):
                active_entries = []
                active_closes = []
                open_quantity = 0.0
                max_quantity = 0.0
            active_entries.append(order)
            open_quantity += quantity
            max_quantity = max(max_quantity, open_quantity)
            continue
        if order_side != close_side or open_quantity <= 0:
            continue
        active_closes.append(order)
        open_quantity -= min(quantity, open_quantity)
        if open_quantity > max(max_quantity * 0.01, 1e-12):
            continue
        fragment = _position_fragment_from_order_lifecycle(
            positions[0],
            positions,
            active_entries,
            active_closes,
            quantity=max_quantity,
            okx_pos_id=_okx_pos_id_for_order_lifecycle(
                positions,
                entry_orders=active_entries,
                close_orders=active_closes,
            ),
        )
        if fragment is not None:
            fragments.append(fragment)
        active_entries = []
        active_closes = []
        open_quantity = 0.0
        max_quantity = 0.0
    return fragments


def _position_fragment_from_order_lifecycle(
    position: Position,
    source_positions: list[Position],
    entry_orders: list[Order],
    close_orders: list[Order],
    *,
    quantity: float,
    okx_pos_id: str,
) -> Position | None:
    if not entry_orders or not close_orders or quantity <= 0:
        return None
    entry_price = _weighted_average(
        (_order_quantity(order), _safe_float(getattr(order, "price", None), 0.0))
        for order in entry_orders
    )
    close_price = _weighted_average(
        (_order_quantity(order), _safe_float(getattr(order, "price", None), 0.0))
        for order in close_orders
    )
    realized_pnl = sum(_order_realized_pnl(order) or 0.0 for order in close_orders)
    return _LedgerPositionFragment(
        id=getattr(position, "id", None),
        model_name="okx_authoritative_sync",
        execution_mode=str(getattr(position, "execution_mode", "") or ""),
        symbol=str(getattr(position, "symbol", "") or ""),
        side=str(getattr(position, "side", "") or "").lower(),
        quantity=quantity,
        entry_price=entry_price or _safe_float(getattr(position, "entry_price", None), 0.0) or 0.0,
        current_price=close_price
        or _safe_float(getattr(position, "current_price", None), 0.0)
        or 0.0,
        leverage=_safe_float(getattr(position, "leverage", None), 1.0) or 1.0,
        unrealized_pnl=0.0,
        realized_pnl=realized_pnl,
        is_open=False,
        closed_at=max(
            (_order_time(order) for order in close_orders if _order_time(order) is not None),
            default=None,
        ),
        created_at=min(
            (_order_time(order) for order in entry_orders if _order_time(order) is not None),
            default=None,
        ),
        okx_inst_id=getattr(position, "okx_inst_id", None)
        or getattr(entry_orders[0], "okx_inst_id", None)
        or getattr(close_orders[0], "okx_inst_id", None),
        okx_pos_id=okx_pos_id or getattr(position, "okx_pos_id", None),
        entry_exchange_order_id=",".join(
            str(getattr(order, "exchange_order_id", "") or "").strip()
            for order in entry_orders
            if str(getattr(order, "exchange_order_id", "") or "").strip()
        )
        or None,
        close_exchange_order_id=",".join(
            str(getattr(order, "exchange_order_id", "") or "").strip()
            for order in close_orders
            if str(getattr(order, "exchange_order_id", "") or "").strip()
        )
        or None,
        source_position_ids=_source_position_ids_for_order_lifecycle(
            source_positions,
            entry_orders=entry_orders,
            close_orders=close_orders,
        ),
        strict_order_lifecycle=True,
        settlement_status=_source_position_final_settlement_status(source_positions),
    )


def _okx_pos_id_for_order_lifecycle(
    positions: list[Position],
    *,
    entry_orders: list[Order],
    close_orders: list[Order],
) -> str:
    entry_ids = {
        str(getattr(order, "exchange_order_id", "") or "").strip() for order in entry_orders
    } - {""}
    close_ids = {
        str(getattr(order, "exchange_order_id", "") or "").strip() for order in close_orders
    } - {""}
    opened_at_values = [_order_time(order) for order in entry_orders]
    closed_at_values = [_order_time(order) for order in close_orders]
    opened_at = min((value for value in opened_at_values if value is not None), default=None)
    closed_at = max((value for value in closed_at_values if value is not None), default=None)
    scored: list[tuple[int, str]] = []
    for position in positions:
        pos_id = _position_pos_id(position)
        if not pos_id:
            continue
        score = 0
        position_entries = set(_position_order_key(position, "entry_exchange_order_id"))
        position_closes = set(_position_order_key(position, "close_exchange_order_id"))
        if position_entries & entry_ids:
            score += 3
        if position_closes & close_ids:
            score += 3
        position_opened = _as_utc(getattr(position, "created_at", None))
        position_closed = _as_utc(getattr(position, "closed_at", None))
        if opened_at and position_opened and abs((position_opened - opened_at).total_seconds()) <= 600:
            score += 1
        if closed_at and position_closed and abs((position_closed - closed_at).total_seconds()) <= 1800:
            score += 1
        if score > 0:
            scored.append((score, pos_id))
    if not scored:
        return ""
    return sorted(scored, key=lambda item: item[0], reverse=True)[0][1]


def _source_position_ids_for_order_lifecycle(
    positions: list[Position],
    *,
    entry_orders: list[Order],
    close_orders: list[Order],
) -> list[int]:
    entry_ids = {
        str(getattr(order, "exchange_order_id", "") or "").strip() for order in entry_orders
    } - {""}
    close_ids = {
        str(getattr(order, "exchange_order_id", "") or "").strip() for order in close_orders
    } - {""}
    source_ids: list[int] = []
    for position in positions:
        row_id = getattr(position, "id", None)
        if row_id is None:
            continue
        row_entries = set(_position_order_key(position, "entry_exchange_order_id"))
        row_closes = set(_position_order_key(position, "close_exchange_order_id"))
        if (row_entries and bool(row_entries & entry_ids)) or (
            row_closes and bool(row_closes & close_ids)
        ):
            source_ids.append(int(row_id))
    return _ordered_ints(source_ids)


def _source_position_final_settlement_status(positions: list[Position]) -> str:
    statuses = [
        str(getattr(position, "settlement_status", "") or "").strip()
        for position in positions
        if str(getattr(position, "settlement_status", "") or "").strip()
    ]
    if not statuses or any(status not in FINAL_LEDGER_SETTLEMENT_STATUSES for status in statuses):
        return ""
    for preferred in ("reconciled", "okx_position_history", "settled"):
        if preferred in statuses:
            return preferred
    return statuses[0]


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
        source_position_ids=[
            int(position.id)
            for position in [position]
            if getattr(position, "id", None) is not None
        ],
        settlement_status=(
            str(getattr(position, "settlement_status", "") or "").strip()
            if str(getattr(position, "settlement_status", "") or "").strip()
            in FINAL_LEDGER_SETTLEMENT_STATUSES
            else ""
        ),
    )


def _build_group_from_positions(
    key: tuple[str, str, str, str],
    positions: list[Position],
    orders_by_id: dict[str, Order],
    account_bills: list[Any],
    position_history_rows: list[dict[str, Any]],
) -> OkxPositionLedgerGroup:
    mode, symbol, side, lifecycle_open = key
    inst_id = _position_inst_id(positions[0]) or okx_inst_id_from_symbol(symbol) or ""
    metric_positions = _metric_positions_for_group(positions)
    position_ids = _position_ids_for_group(positions)
    order_positions = _strict_lifecycle_positions_for_group(positions) or positions
    opened_at_values = [
        _as_utc(pos.created_at) for pos in metric_positions if _as_utc(pos.created_at)
    ]
    closed_at_values = [
        _as_utc(pos.closed_at) for pos in metric_positions if _as_utc(pos.closed_at)
    ]
    opened_at = min(opened_at_values) if opened_at_values else None
    closed_at = max(closed_at_values) if closed_at_values else None

    entry_ids = _ordered_tokens(
        token
        for pos in order_positions
        for token in _split_exchange_order_ids(getattr(pos, "entry_exchange_order_id", None))
    )
    close_ids = _ordered_tokens(
        token
        for pos in order_positions
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

    closed_quantity = sum(
        abs(_safe_float(getattr(pos, "quantity", None))) for pos in metric_positions
    )
    max_quantity = max(
        [
            closed_quantity,
            *[
                abs(_safe_float(getattr(pos, "quantity", None)))
                for pos in metric_positions
            ],
        ],
        default=closed_quantity,
    )
    entry_price = _weighted_average(
        (
            abs(_safe_float(getattr(pos, "quantity", None))),
            _safe_float(getattr(pos, "entry_price", None)),
        )
        for pos in metric_positions
    )
    close_price = _weighted_average(
        (
            abs(_safe_float(getattr(pos, "quantity", None))),
            _safe_float(getattr(pos, "current_price", None)),
        )
        for pos in metric_positions
    )
    if close_ids:
        close_price_from_orders = _weighted_average(
            (row.quantity, row.price)
            for row in linked_fills
            if row.order_id in close_ids and row.quantity > 0 and row.price > 0
        )
        if close_price_from_orders > 0:
            close_price = close_price_from_orders

    realized_pnl = sum(
        _safe_float(getattr(pos, "realized_pnl", None)) for pos in metric_positions
    )
    stored_settlement = _stored_position_settlement_components(metric_positions)
    pnl_components = _confirmed_linked_order_pnl_components(
        linked_fills=linked_fills,
        entry_ids=entry_ids,
        close_ids=close_ids,
        closed_quantity=closed_quantity,
    )
    if pnl_components is not None and pnl_components.entry_quantity > max_quantity:
        max_quantity = pnl_components.entry_quantity
    funding_components = _funding_fee_components_for_group(
        account_bills=account_bills,
        mode=mode,
        inst_id=inst_id,
        side=side,
        opened_at=opened_at,
        closed_at=closed_at,
    )
    official_components = _official_position_history_components_for_group(
        position_history_rows=position_history_rows,
        mode=mode,
        inst_id=inst_id,
        side=side,
        positions=positions,
        linked_fills=linked_fills,
        entry_ids=entry_ids,
        close_ids=close_ids,
        closed_quantity=closed_quantity,
    )
    if official_components is not None:
        realized_pnl = official_components.realized_pnl
        pnl_source = official_components.source
        if official_components.opened_at is not None:
            opened_at = official_components.opened_at
        if official_components.closed_at is not None:
            closed_at = official_components.closed_at
        if official_components.entry_price > 0:
            entry_price = official_components.entry_price
        if official_components.close_price > 0:
            close_price = official_components.close_price
        if official_components.closed_quantity > 0:
            closed_quantity = official_components.closed_quantity
        max_quantity = max(max_quantity, closed_quantity)
        funding_components = _FundingFeeComponents(
            funding_fee=official_components.funding_fee,
            bill_count=funding_components.bill_count,
            source="okx_positions_history.fundingFee",
        )
    elif stored_settlement is not None and stored_settlement.preferred:
        realized_pnl = stored_settlement.realized_pnl
        pnl_source = stored_settlement.source
        funding_components = _FundingFeeComponents(
            funding_fee=stored_settlement.funding_fee,
            bill_count=funding_components.bill_count,
            source=(
                "position_settlement_snapshot"
                if abs(stored_settlement.funding_fee) > 1e-12
                else funding_components.source
            ),
        )
    elif pnl_components is not None:
        realized_pnl = (
            pnl_components.close_fill_pnl
            + funding_components.funding_fee
            - pnl_components.entry_fee
            - pnl_components.close_fee
        )
        pnl_source = pnl_components.source
    elif stored_settlement is not None:
        realized_pnl = stored_settlement.realized_pnl
        pnl_source = stored_settlement.source
        funding_components = _FundingFeeComponents(
            funding_fee=stored_settlement.funding_fee,
            bill_count=funding_components.bill_count,
            source=(
                "position_settlement_snapshot"
                if abs(stored_settlement.funding_fee) > 1e-12
                else funding_components.source
            ),
        )
    elif not realized_pnl:
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
        (_safe_float(getattr(pos, "leverage", None), 0.0) for pos in metric_positions),
        default=0.0,
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
    if pnl_components is None and entry_ids and close_ids:
        gaps.append("incomplete_okx_linked_order_pnl_components")
    if pnl_components is not None and pnl_components.quantity_mismatch:
        gaps.append("okx_fill_position_quantity_mismatch")
    evidence_complete = not gaps and bool(close_ids) and bool(entry_ids)
    partial_lifecycle = _group_represents_partial_lifecycle(
        closed_quantity=closed_quantity,
        pnl_components=pnl_components,
    )
    status = "partial" if partial_lifecycle or (len(positions) > 1 and not evidence_complete) else "full"
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
          close_fill_pnl=(
              official_components.close_fill_pnl
              if official_components is not None
              else
              stored_settlement.close_fill_pnl
              if stored_settlement is not None
              and (stored_settlement.preferred or pnl_components is None)
            else pnl_components.close_fill_pnl
            if pnl_components is not None
            else 0.0
          ),
          entry_fee=(
              official_components.entry_fee
              if official_components is not None
              else
              stored_settlement.entry_fee
              if stored_settlement is not None
              and (stored_settlement.preferred or pnl_components is None)
            else pnl_components.entry_fee
            if pnl_components is not None
            else 0.0
          ),
          close_fee=(
              official_components.close_fee
              if official_components is not None
              else
              stored_settlement.close_fee
              if stored_settlement is not None
              and (stored_settlement.preferred or pnl_components is None)
            else pnl_components.close_fee
            if pnl_components is not None
            else 0.0
        ),
        funding_fee=funding_components.funding_fee,
        funding_bill_count=funding_components.bill_count,
        funding_fee_source=funding_components.source,
        settlement_status=stored_settlement.status if stored_settlement is not None else "",
        settlement_source=(
            official_components.source
            if official_components is not None
            else stored_settlement.source
            if stored_settlement is not None
            else ""
        ),
    )


def _official_position_history_components_for_group(
    *,
    position_history_rows: list[dict[str, Any]],
    mode: str,
    inst_id: str,
    side: str,
    positions: list[Position],
    linked_fills: list[OkxLinkedFillRow],
    entry_ids: list[str],
    close_ids: list[str],
    closed_quantity: float,
) -> _OfficialPositionHistoryComponents | None:
    if not position_history_rows or not inst_id:
        return None
    group_pos_ids = {_position_pos_id(position) for position in positions if _position_pos_id(position)}
    best: tuple[int, dict[str, Any]] | None = None
    for row in position_history_rows:
        if not isinstance(row, dict):
            continue
        row_inst_id = str(row.get("instId") or row.get("inst_id") or "").strip().upper()
        if row_inst_id and row_inst_id != inst_id:
            continue
        row_pos_id = str(row.get("posId") or row.get("pos_id") or "").strip()
        if group_pos_ids and row_pos_id and row_pos_id not in group_pos_ids:
            continue
        realized_pnl = _first_present_float(
            row,
            ("realizedPnl", "realized_pnl", "realizedPnlInUsd", "realizedPnlUsd"),
        )
        if realized_pnl is None:
            continue
        row_closed_quantity = _first_present_float(
            row,
            ("closeTotalPos", "close_total_pos", "closedQuantity", "closed_quantity"),
        )
        row_closed_quantity_base = _official_closed_quantity_in_base_units(
            row_closed_quantity,
            linked_fills=linked_fills,
            close_ids=close_ids,
            local_closed_quantity=closed_quantity,
        )
        quantity_matches = bool(
            row_closed_quantity_base is None
            or row_closed_quantity_base <= 0
            or closed_quantity <= 0
            or _quantities_match(row_closed_quantity_base, closed_quantity, tolerance_ratio=0.02)
        )
        row_opened = _ms_timestamp_to_datetime(row.get("cTime") or row.get("createdTime"))
        row_updated = _ms_timestamp_to_datetime(row.get("uTime") or row.get("updatedTime"))
        opened_values = [
            value
            for position in positions
            if (value := _as_utc(getattr(position, "created_at", None))) is not None
        ]
        closed_values = [
            value
            for position in positions
            if (value := _as_utc(getattr(position, "closed_at", None))) is not None
        ]
        time_matches = True
        if row_opened and opened_values:
            time_matches = min(abs((row_opened - value).total_seconds()) for value in opened_values) <= 300
        if time_matches and row_updated and closed_values:
            time_matches = min(abs((row_updated - value).total_seconds()) for value in closed_values) <= 3600
        score = 0
        if row_pos_id and row_pos_id in group_pos_ids:
            score += 100
        if quantity_matches:
            score += 20
        if time_matches:
            score += 10
        if not quantity_matches and not time_matches:
            continue
        if best is None or score > best[0]:
            best = (score, row)
    if best is None:
        return None
    row = best[1]
    realized_pnl = _first_present_float(
        row,
        ("realizedPnl", "realized_pnl", "realizedPnlInUsd", "realizedPnlUsd"),
    )
    if realized_pnl is None:
        return None
    funding_fee = _first_present_float(row, ("fundingFee", "funding_fee")) or 0.0
    total_fee_abs = abs(_first_present_float(row, ("fee", "fees", "totalFee", "total_fee")) or 0.0)
    close_fill_pnl = _first_present_float(row, ("pnl", "closeFillPnl", "close_fill_pnl"))
    if close_fill_pnl is None:
        close_fill_pnl = realized_pnl - funding_fee + total_fee_abs
    entry_fee = sum(abs(_safe_float(fill.fee, 0.0) or 0.0) for fill in linked_fills if fill.order_id in entry_ids)
    if entry_fee > total_fee_abs:
        entry_fee = 0.0
    close_fee = max(total_fee_abs - entry_fee, 0.0)
    row_closed_quantity = _first_present_float(row, ("closeTotalPos", "closed_quantity"))
    closed_quantity_base = _official_closed_quantity_in_base_units(
        row_closed_quantity,
        linked_fills=linked_fills,
        close_ids=close_ids,
        local_closed_quantity=closed_quantity,
    ) or 0.0
    return _OfficialPositionHistoryComponents(
        close_fill_pnl=close_fill_pnl,
        entry_fee=entry_fee,
        close_fee=close_fee,
        funding_fee=funding_fee,
        realized_pnl=realized_pnl,
        entry_price=_first_present_float(row, ("openAvgPx", "avgPx", "entry_price")) or 0.0,
        close_price=_first_present_float(row, ("closeAvgPx", "close_price")) or 0.0,
        closed_quantity=closed_quantity_base,
        opened_at=_ms_timestamp_to_datetime(row.get("cTime") or row.get("createdTime")),
        closed_at=_ms_timestamp_to_datetime(row.get("uTime") or row.get("updatedTime")),
    )


def _official_closed_quantity_in_base_units(
    row_closed_quantity: float | None,
    *,
    linked_fills: list[OkxLinkedFillRow],
    close_ids: list[str],
    local_closed_quantity: float,
) -> float | None:
    if row_closed_quantity is None or row_closed_quantity <= 0:
        return row_closed_quantity
    close_rows = [row for row in linked_fills if row.order_id in close_ids]
    close_units = _linked_fill_unit_sums(close_rows)
    base_quantity = close_units.get("base_quantity", 0.0)
    contract_quantity = close_units.get("contracts", 0.0)
    if base_quantity > 0 and _quantities_match(
        row_closed_quantity,
        base_quantity,
        tolerance_ratio=0.02,
    ):
        return base_quantity
    if contract_quantity > 0 and _quantities_match(
        row_closed_quantity,
        contract_quantity,
        tolerance_ratio=0.02,
    ):
        if base_quantity > 0:
            return base_quantity
        contract_size = _weighted_average(
            (row.contracts, row.contract_size)
            for row in close_rows
            if row.contracts > 0 and row.contract_size > 0
        )
        if contract_size > 0:
            return row_closed_quantity * contract_size
    if local_closed_quantity > 0 and _quantities_match(
        row_closed_quantity,
        local_closed_quantity,
        tolerance_ratio=0.02,
    ):
        return local_closed_quantity
    return row_closed_quantity


def _group_represents_partial_lifecycle(
    *,
    closed_quantity: float,
    pnl_components: _LinkedOrderPnlComponents | None,
) -> bool:
    if pnl_components is None:
        return False
    entry_quantity = _safe_float(pnl_components.entry_quantity, 0.0) or 0.0
    closed_quantity = _safe_float(closed_quantity, 0.0) or 0.0
    return bool(
        entry_quantity > 0
        and closed_quantity > 0
        and closed_quantity < entry_quantity
        and not _quantities_match(closed_quantity, entry_quantity)
    )


def _confirmed_linked_order_pnl_components(
    *,
    linked_fills: list[OkxLinkedFillRow],
    entry_ids: list[str],
    close_ids: list[str],
    closed_quantity: float,
) -> _LinkedOrderPnlComponents | None:
    if not entry_ids or not close_ids:
        return None
    fills_by_order_id = {row.order_id: row for row in linked_fills if row.order_id}
    close_rows: list[OkxLinkedFillRow] = []
    for order_id in close_ids:
        row = fills_by_order_id.get(order_id)
        if row is None or not row.okx_confirmed or row.pnl is None or row.quantity <= 0:
            return None
        close_rows.append(row)
    entry_rows: list[OkxLinkedFillRow] = []
    for order_id in entry_ids:
        row = fills_by_order_id.get(order_id)
        if row is None or not row.okx_confirmed or row.quantity <= 0:
            continue
        entry_rows.append(row)

    close_units = _linked_fill_unit_sums(close_rows)
    entry_units = _linked_fill_unit_sums(entry_rows)
    close_quantity, entry_quantity, unit_source, quantity_mismatch = _best_pnl_quantity_units(
        closed_quantity=closed_quantity,
        close_units=close_units,
        entry_units=entry_units,
    )
    close_gross_pnl = sum(_safe_float(row.pnl, 0.0) or 0.0 for row in close_rows)
    close_fee = sum(abs(_safe_float(row.fee, 0.0) or 0.0) for row in close_rows)
    entry_fee = sum(abs(_safe_float(row.fee, 0.0) or 0.0) for row in entry_rows)
    if entry_quantity > 0:
        entry_fee *= min(max(close_quantity, 0.0) / entry_quantity, 1.0)
        source = "okx_linked_order_net_pnl"
    else:
        source = "okx_close_fill_net_pnl_partial"
    return _LinkedOrderPnlComponents(
        close_fill_pnl=close_gross_pnl,
        entry_fee=entry_fee,
        close_fee=close_fee,
        close_quantity=close_quantity,
        entry_quantity=entry_quantity,
        quantity_match_source=unit_source,
        quantity_mismatch=quantity_mismatch,
        source=source,
    )


def _stored_position_settlement_components(
    positions: list[Position],
) -> _StoredSettlementComponents | None:
    settlement_positions = [
        position
        for position in positions
        if str(getattr(position, "settlement_source", "") or "").strip()
        or str(getattr(position, "settlement_status", "") or "").strip()
        in FINAL_LEDGER_SETTLEMENT_STATUSES
    ]
    if not settlement_positions or len(settlement_positions) != len(positions):
        return None
    close_fill_pnl = sum(
        _safe_float(getattr(position, "close_fill_pnl", None), 0.0)
        for position in settlement_positions
    )
    entry_fee = sum(
        abs(_safe_float(getattr(position, "entry_fee", None), 0.0))
        for position in settlement_positions
    )
    close_fee = sum(
        abs(_safe_float(getattr(position, "close_fee", None), 0.0))
        for position in settlement_positions
    )
    funding_fee = sum(
        _safe_float(getattr(position, "funding_fee", None), 0.0)
        for position in settlement_positions
    )
    component_realized_pnl = close_fill_pnl + funding_fee - entry_fee - close_fee
    position_realized_pnl = sum(
        _safe_float(getattr(position, "realized_pnl", None), 0.0)
        for position in settlement_positions
    )
    has_component_values = any(
        abs(_safe_float(getattr(position, "close_fill_pnl", None), 0.0)) > 1e-12
        or abs(_safe_float(getattr(position, "entry_fee", None), 0.0)) > 1e-12
        or abs(_safe_float(getattr(position, "close_fee", None), 0.0)) > 1e-12
        or abs(_safe_float(getattr(position, "funding_fee", None), 0.0)) > 1e-12
        for position in settlement_positions
    )
    if has_component_values:
        realized_pnl = component_realized_pnl
    else:
        realized_pnl = position_realized_pnl
    status_values = {
        str(getattr(position, "settlement_status", "") or "").strip()
        for position in settlement_positions
    } - {""}
    source_values = [
        str(getattr(position, "settlement_source", "") or "").strip()
        for position in settlement_positions
        if str(getattr(position, "settlement_source", "") or "").strip()
    ]
    has_okx_authoritative_source = any(value.startswith("okx_") for value in source_values)
    preferred = bool(
        status_values
        and status_values.issubset(FINAL_LEDGER_SETTLEMENT_STATUSES)
        and has_okx_authoritative_source
    )
    status = ",".join(sorted(status_values)) if status_values else "provisional"
    if source_values:
        source = (
            "position_settlement_snapshot"
            if len(set(source_values)) != 1
            else f"position_settlement_snapshot:{source_values[0]}"
        )
    elif has_component_values:
        source = "position_settlement_snapshot"
    else:
        source = "position_realized_pnl"
    return _StoredSettlementComponents(
        close_fill_pnl=close_fill_pnl,
        entry_fee=entry_fee,
        close_fee=close_fee,
        funding_fee=funding_fee,
        realized_pnl=realized_pnl,
        source=source,
        status=status,
        preferred=preferred,
    )


def _linked_fill_unit_sums(rows: list[OkxLinkedFillRow]) -> dict[str, float]:
    base_quantity = sum(max(_safe_float(row.quantity, 0.0) or 0.0, 0.0) for row in rows)
    contracts = sum(max(_safe_float(row.contracts, 0.0) or 0.0, 0.0) for row in rows)
    if contracts <= 0:
        contracts = sum(
            (
                max(_safe_float(row.quantity, 0.0) or 0.0, 0.0)
                / max(_safe_float(row.contract_size, 0.0) or 0.0, 1.0)
            )
            for row in rows
            if (_safe_float(row.contract_size, 0.0) or 0.0) > 0
        )
    return {
        "base_quantity": base_quantity,
        "contracts": contracts,
    }


def _best_pnl_quantity_units(
    *,
    closed_quantity: float,
    close_units: dict[str, float],
    entry_units: dict[str, float],
) -> tuple[float, float, str, bool]:
    """Choose the quantity unit used to allocate entry fee for OKX PnL."""

    closed_quantity = max(_safe_float(closed_quantity, 0.0) or 0.0, 0.0)
    candidates = (
        (
            "base_quantity",
            max(close_units.get("base_quantity", 0.0), 0.0),
            max(entry_units.get("base_quantity", 0.0), 0.0),
        ),
        (
            "contracts",
            max(close_units.get("contracts", 0.0), 0.0),
            max(entry_units.get("contracts", 0.0), 0.0),
        ),
    )
    for name, close_quantity_value, entry_quantity_value in candidates:
        if (
            closed_quantity > 0
            and close_quantity_value > 0
            and _quantities_match(close_quantity_value, closed_quantity, tolerance_ratio=0.02)
        ):
            return close_quantity_value, entry_quantity_value, name, False

    for name, close_quantity_value, entry_quantity_value in candidates:
        if close_quantity_value > 0 and entry_quantity_value > 0:
            return close_quantity_value, entry_quantity_value, name, True

    for name, close_quantity_value, entry_quantity_value in candidates:
        if close_quantity_value > 0:
            return close_quantity_value, entry_quantity_value, name, True

    return closed_quantity, 0.0, "position_quantity", True


def _funding_fee_components_for_group(
    *,
    account_bills: list[Any],
    mode: str,
    inst_id: str,
    side: str,
    opened_at: datetime | None,
    closed_at: datetime | None,
) -> _FundingFeeComponents:
    if not account_bills or not inst_id or opened_at is None or closed_at is None:
        return _FundingFeeComponents(0.0, 0, "none")
    start = _as_utc(opened_at)
    end = _as_utc(closed_at)
    if start is None or end is None:
        return _FundingFeeComponents(0.0, 0, "none")
    normalized_mode = "live" if str(mode or "").lower() == "live" else "paper"
    total = 0.0
    count = 0
    for bill in account_bills:
        bill_mode = str(_bill_value(bill, "mode") or "").lower().strip()
        if bill_mode and bill_mode != normalized_mode:
            continue
        bill_inst_id = str(
            _bill_value(bill, "inst_id")
            or _bill_raw_value(bill, "instId")
            or ""
        ).upper().strip()
        if not bill_inst_id or bill_inst_id != inst_id:
            continue
        bill_ts = _as_utc(_bill_value(bill, "bill_ts") or _bill_value(bill, "timestamp"))
        if bill_ts is None:
            bill_ts = _datetime_from_ms(
                _bill_raw_value(bill, "ts")
                or _bill_raw_value(bill, "uTime")
                or _bill_raw_value(bill, "cTime")
            )
        if bill_ts is None or bill_ts < start or bill_ts > end:
            continue
        bill_pos_side = str(
            _bill_value(bill, "pos_side")
            or _bill_raw_value(bill, "posSide")
            or ""
        ).lower().strip()
        if bill_pos_side in {"long", "short"} and side and bill_pos_side != side:
            continue
        funding_fee = _bill_funding_fee(bill)
        if abs(funding_fee) <= 1e-12:
            continue
        total += funding_fee
        count += 1
    return _FundingFeeComponents(
        funding_fee=total,
        bill_count=count,
        source="okx_account_bills" if count else "none",
    )


def _bill_funding_fee(bill: Any) -> float:
    funding_fee = _safe_float(_bill_value(bill, "funding_fee"), 0.0) or 0.0
    if abs(funding_fee) > 1e-12:
        return funding_fee
    if not _bill_is_funding_fee(bill):
        return 0.0
    pnl = _safe_float(_bill_value(bill, "pnl") or _bill_raw_value(bill, "pnl"), 0.0) or 0.0
    if abs(pnl) > 1e-12:
        return pnl
    return _safe_float(
        _bill_value(bill, "balance_change")
        or _bill_raw_value(bill, "balChg")
        or _bill_raw_value(bill, "balanceChange"),
        0.0,
    ) or 0.0


def _bill_is_funding_fee(bill: Any) -> bool:
    sub_type = str(
        _bill_value(bill, "bill_sub_type")
        or _bill_raw_value(bill, "subType")
        or _bill_raw_value(bill, "billSubType")
        or ""
    ).strip()
    if sub_type in FUNDING_FEE_BILL_SUBTYPES:
        return True
    raw = _bill_raw(bill)
    for key in ("fundingFee", "funding_fee"):
        if key in raw and abs(_safe_float(raw.get(key), 0.0) or 0.0) > 1e-12:
            return True
    text = " ".join(
        str(
            _bill_value(bill, key)
            or _bill_raw_value(bill, key)
            or ""
        ).lower()
        for key in ("bill_type", "bill_sub_type", "type", "subType", "bizType", "desc")
    )
    return "funding" in text


def _bill_value(bill: Any, key: str) -> Any:
    if isinstance(bill, dict):
        return bill.get(key)
    return getattr(bill, key, None)


def _bill_raw_value(bill: Any, key: str) -> Any:
    return _bill_raw(bill).get(key)


def _bill_raw(bill: Any) -> dict[str, Any]:
    if isinstance(bill, dict):
        raw = bill.get("raw_bill") or bill.get("raw") or bill
    else:
        raw = getattr(bill, "raw_bill", None) or getattr(bill, "raw", None)
    return raw if isinstance(raw, dict) else {}


def _datetime_from_ms(value: Any) -> datetime | None:
    timestamp_ms = _safe_float(value, 0.0) or 0.0
    if timestamp_ms <= 0:
        return None
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000.0, UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _deduplicate_superseded_position_rows(positions: list[Position]) -> list[Position]:
    if len(positions) <= 1:
        return positions
    keep: list[Position] = []
    for bucket in _duplicate_position_buckets(positions).values():
        winner = max(bucket, key=_position_evidence_score)
        keep.append(winner)
    keep = _drop_superseded_authoritative_position_rows(keep)
    return sorted(
        keep,
        key=lambda item: _as_utc(getattr(item, "created_at", None))
        or datetime.min.replace(tzinfo=UTC),
    )


def _drop_rows_superseded_by_generated_lifecycle(positions: list[Position]) -> list[Position]:
    generated = [row for row in positions if _is_generated_order_lifecycle_fragment(row)]
    if not generated:
        return positions
    return [
        row
        for row in positions
        if _is_generated_order_lifecycle_fragment(row)
        or not any(_position_supersedes(candidate, row) for candidate in generated)
    ]


def _metric_positions_for_group(positions: list[Position]) -> list[Position]:
    authoritative_rows = [
        position
        for position in positions
        if _is_okx_authoritative_position_history_row(position)
    ]
    if not authoritative_rows:
        return positions
    all_close_ids = {
        token
        for position in positions
        for token in _position_order_key(position, "close_exchange_order_id")
    }
    authoritative_close_ids = {
        token
        for position in authoritative_rows
        for token in _position_order_key(position, "close_exchange_order_id")
    }
    if all_close_ids and authoritative_close_ids and not all_close_ids.issubset(
        authoritative_close_ids
    ):
        return positions
    latest_position_close = max(
        (
            value
            for position in positions
            if (value := _as_utc(getattr(position, "closed_at", None))) is not None
        ),
        default=None,
    )
    latest_authoritative_close = max(
        (
            value
            for position in authoritative_rows
            if (value := _as_utc(getattr(position, "closed_at", None))) is not None
        ),
        default=None,
    )
    if (
        latest_position_close is not None
        and latest_authoritative_close is not None
        and latest_position_close > latest_authoritative_close + timedelta(seconds=60)
    ):
        return positions
    return authoritative_rows


def _strict_lifecycle_positions_for_group(positions: list[Position]) -> list[Position]:
    return [
        position
        for position in positions
        if bool(getattr(position, "strict_order_lifecycle", False))
    ]


def _position_ids_for_group(positions: list[Position]) -> list[int]:
    ids: list[int] = []
    for position in positions:
        source_ids = getattr(position, "source_position_ids", None)
        if isinstance(source_ids, list):
            ids.extend(int(item) for item in source_ids if item is not None)
        elif getattr(position, "id", None) is not None:
            ids.append(int(position.id))
    return _ordered_ints(ids)


def _drop_superseded_authoritative_position_rows(positions: list[Position]) -> list[Position]:
    authoritative_rows = [
        position
        for position in positions
        if _is_okx_authoritative_position_history_row(position)
    ]
    if not authoritative_rows:
        return positions
    return [
        position
        for position in positions
        if not any(
            candidate is not position
            and _authoritative_position_history_row_supersedes(candidate, position)
            for candidate in authoritative_rows
        )
    ]


def _authoritative_position_history_row_supersedes(
    candidate: Position,
    other: Position,
) -> bool:
    """Return True when an OKX final lifecycle row covers an older fragment.

    OKX position-history sync can observe partial-close snapshots before the
    final lifecycle snapshot.  The final row carries the OKX UI's realizedPnl
    for the whole historical position, so older local/partial rows must remain
    audit evidence only and must not be summed into the ledger PnL.
    """

    if _is_generated_order_lifecycle_fragment(other) and not _is_generated_order_lifecycle_fragment(candidate):
        return False
    if not _is_okx_authoritative_position_history_row(candidate):
        return False
    if _position_base_key(candidate) != _position_base_key(other):
        return False

    candidate_pos_id = _position_pos_id(candidate)
    other_pos_id = _position_pos_id(other)
    if candidate_pos_id and other_pos_id and candidate_pos_id != other_pos_id:
        return False

    candidate_entry = set(_position_order_key(candidate, "entry_exchange_order_id"))
    candidate_close = set(_position_order_key(candidate, "close_exchange_order_id"))
    other_entry = set(_position_order_key(other, "entry_exchange_order_id"))
    other_close = set(_position_order_key(other, "close_exchange_order_id"))
    if _is_generated_order_lifecycle_fragment(candidate) and other_close:
        candidate_close_covers = candidate_close.issuperset(other_close)
        candidate_entry_overlaps = not other_entry or bool(candidate_entry & other_entry)
        if candidate_close_covers and candidate_entry_overlaps:
            return True
    if not candidate_close:
        return False
    if other_close and not other_close.issubset(candidate_close):
        return False
    if other_entry and candidate_entry and not other_entry.issubset(candidate_entry):
        close_covered = bool(other_close and other_close.issubset(candidate_close))
        if not close_covered:
            return False
    if other_entry and not candidate_entry:
        return False

    candidate_opened = _as_utc(getattr(candidate, "created_at", None))
    other_opened = _as_utc(getattr(other, "created_at", None))
    candidate_closed = _as_utc(getattr(candidate, "closed_at", None))
    other_closed = _as_utc(getattr(other, "closed_at", None))
    order_coverage_proves_lifecycle = bool(
        other_close
        and other_close.issubset(candidate_close)
        and (
            not other_entry
            or not candidate_entry
            or bool(other_entry & candidate_entry)
        )
    )
    if candidate_opened and other_opened and not order_coverage_proves_lifecycle:
        if abs((candidate_opened - other_opened).total_seconds()) > 60:
            return False
    elif not (
        (candidate_entry and other_entry and bool(candidate_entry & other_entry))
        or (candidate_close and other_close and bool(candidate_close & other_close))
    ):
        return False
    if candidate_closed and other_closed and candidate_closed < other_closed:
        if abs((candidate_closed - other_closed).total_seconds()) > 60:
            return False

    candidate_quantity = abs(_safe_float(getattr(candidate, "quantity", None)))
    other_quantity = abs(_safe_float(getattr(other, "quantity", None)))
    if other_quantity > 0 and candidate_quantity + max(other_quantity * 0.001, 1e-12) < other_quantity:
        return False

    strictly_more_complete = candidate_close != other_close or (
        candidate_entry != other_entry and bool(candidate_close)
    )
    return strictly_more_complete


def _is_okx_authoritative_position_history_row(position: Position) -> bool:
    return (
        str(getattr(position, "model_name", "") or "").strip() == "okx_authoritative_sync"
        and bool(_position_pos_id(position))
        and not bool(getattr(position, "is_open", False))
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
    if _is_generated_order_lifecycle_fragment(other) and not _is_generated_order_lifecycle_fragment(candidate):
        return False
    candidate_entry = set(_position_order_key(candidate, "entry_exchange_order_id"))
    candidate_close = set(_position_order_key(candidate, "close_exchange_order_id"))
    other_entry = set(_position_order_key(other, "entry_exchange_order_id"))
    other_close = set(_position_order_key(other, "close_exchange_order_id"))
    if _is_generated_order_lifecycle_fragment(candidate) and _is_generated_order_lifecycle_fragment(other):
        return bool(
            candidate_close
            and other_close
            and candidate_close.issuperset(other_close)
            and candidate_close != other_close
        )
    if not _same_position_lifecycle(candidate, other) and not _position_order_lifecycle_covers(
        candidate_entry=candidate_entry,
        candidate_close=candidate_close,
        other_entry=other_entry,
        other_close=other_close,
    ):
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


def _position_order_lifecycle_covers(
    *,
    candidate_entry: set[str],
    candidate_close: set[str],
    other_entry: set[str],
    other_close: set[str],
) -> bool:
    if not candidate_entry or not candidate_close:
        return False
    if other_entry and not other_entry.issubset(candidate_entry):
        return False
    if other_close and not other_close.issubset(candidate_close):
        return False
    return bool(other_entry or other_close)


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


def _has_explicit_superseded_position_metadata(position: Position) -> bool:
    if str(getattr(position, "settlement_status", "") or "") == (
        "superseded_position_residual"
    ):
        return True
    raw = getattr(position, "settlement_raw", None)
    raw = raw if isinstance(raw, dict) else {}
    return bool(
        str(raw.get("reason") or "")
        == "duplicate_local_open_position_for_same_okx_pos_id"
        and _safe_float(raw.get("canonical_position_id")) > 0
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
    if contracts > 0 and contract_size > 0:
        quantity = contracts * contract_size
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
    okx_confirmed = sync_status in {
        OKX_SYNC_CONFIRMED,
        OKX_SYNC_OKX_ONLY,
        OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
        OKX_SYNC_ORDER_DETAIL_CONFIRMED,
    }
    source = "okx_raw_fills" if raw else "local_order_cache"
    if bool(raw.get("position_snapshot_confirmed")) and not bool(
        raw.get("fills_history_confirmed")
    ):
        source = "okx_current_position_snapshot"
    elif bool(raw.get("order_detail_confirmed")) and not bool(
        raw.get("fills_history_confirmed")
    ):
        source = "okx_order_detail"
    elif bool(raw.get("execution_result_confirmed")) and not bool(
        raw.get("fills_history_confirmed")
    ):
        source = "okx_execution_result"
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
    if contracts > 0 and contract_size > 0:
        quantity = contracts * contract_size
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


def _ordered_ints(values: Any) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        result.append(item)
        seen.add(item)
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


def _first_present_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        value = _safe_float(row.get(key), None)
        if value is not None:
            return value
    return None


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


def _ms_timestamp_to_datetime(value: Any) -> datetime | None:
    number = _safe_float(value, None)
    if number is None or number <= 0:
        return None
    if number > 10_000_000_000:
        number /= 1000.0
    try:
        return datetime.fromtimestamp(number, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None


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
