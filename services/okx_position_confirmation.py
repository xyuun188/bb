"""Strict OKX current-position confirmation helpers.

This module deliberately distinguishes a current open position from a fill
history fact.  A match here can prove that an entry is exchange-backed for the
current ledger, but it must not be treated as a closed trade or realized-PnL
training sample.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from core.symbols import normalize_trading_symbol, okx_inst_id_from_symbol, symbol_from_okx_inst_id
from services.exchange_position_state import parse_exchange_position_snapshot

POSITION_CONFIRM_PRICE_TOLERANCE_RATIO = 0.01
POSITION_CONFIRM_QUANTITY_TOLERANCE_RATIO = 0.05


@dataclass(frozen=True, slots=True)
class OkxCurrentPositionEntryConfirmation:
    exchange_order_id: str
    inst_id: str
    symbol: str
    side: str
    pos_side: str
    pos_id: str
    trade_id: str
    contracts: float
    contract_size: float
    base_quantity: float
    avg_price: float
    mark_price: float
    upl: float | None
    fee_abs: float | None
    timestamp: datetime | None
    local_position_id: int | None
    match_source: str
    raw_position: dict[str, Any]

    def as_raw_payload(self) -> dict[str, Any]:
        return {
            "order_id": self.exchange_order_id,
            "inst_id": self.inst_id,
            "symbol": self.symbol,
            "pos_side": self.pos_side,
            "pos_id": self.pos_id,
            "position_trade_id": self.trade_id,
            "contracts": self.contracts,
            "contract_size": self.contract_size or None,
            "base_quantity": self.base_quantity,
            "avg_price": self.avg_price,
            "mark_price": self.mark_price,
            "upl": self.upl,
            "fee_abs": self.fee_abs,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "local_position_id": self.local_position_id,
            "match_source": self.match_source,
            "position_snapshot_confirmed": True,
            "fills_history_confirmed": False,
            "position_rows": [dict(self.raw_position)] if self.raw_position else [],
            "rows": [],
        }


def find_current_position_entry_confirmation(
    order: Any,
    *,
    exchange_order_id: str,
    exchange_positions: Iterable[dict[str, Any]],
    local_positions: Iterable[Any],
    contract_sizes: dict[str, float] | None = None,
) -> OkxCurrentPositionEntryConfirmation | None:
    """Return a strict OKX current-position match for a local entry order."""

    order_id = str(exchange_order_id or "").strip()
    if not order_id:
        return None
    order_side = str(getattr(order, "side", "") or "").lower().strip()
    expected_side = _entry_position_side(order_side)
    if expected_side not in {"long", "short"}:
        return None

    linked_positions = [
        position
        for position in local_positions or ()
        if bool(getattr(position, "is_open", False))
        and order_id in _split_exchange_order_ids(
            getattr(position, "entry_exchange_order_id", None)
        )
        and str(getattr(position, "side", "") or "").lower().strip() == expected_side
    ]
    if not linked_positions:
        return None

    sizes = {str(key or "").strip().upper(): _safe_float(value, 0.0) for key, value in (contract_sizes or {}).items()}
    candidate_rows: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for row in exchange_positions or ():
        if not isinstance(row, dict):
            continue
        snapshot = parse_exchange_position_snapshot(
            row,
            symbol_normalizer=normalize_trading_symbol,
        )
        if not snapshot:
            continue
        if str(snapshot.get("side") or "").lower().strip() != expected_side:
            continue
        inst_id = _position_row_inst_id(row, snapshot)
        if not inst_id:
            continue
        candidate_rows.append((row, snapshot, inst_id))

    for local_position in linked_positions:
        local_inst_id = _local_position_inst_id(local_position) or _order_inst_id(order)
        local_pos_id = str(getattr(local_position, "okx_pos_id", "") or "").strip()
        local_price_refs = [
            _safe_float(getattr(order, "price", None), 0.0),
            _safe_float(getattr(local_position, "entry_price", None), 0.0),
        ]
        local_quantity_refs = [
            abs(_safe_float(getattr(order, "quantity", None), 0.0)),
            abs(_safe_float(getattr(local_position, "quantity", None), 0.0)),
        ]
        for row, snapshot, inst_id in candidate_rows:
            if local_inst_id and inst_id != local_inst_id:
                continue
            pos_id = _position_row_pos_id(row)
            pos_id_matches = bool(local_pos_id and pos_id and local_pos_id == pos_id)
            if local_pos_id and not pos_id_matches:
                continue

            avg_price = _safe_float(snapshot.get("entry_price"), 0.0)
            if avg_price > 0 and any(value > 0 for value in local_price_refs):
                if not any(
                    _relative_close_enough(value, avg_price, POSITION_CONFIRM_PRICE_TOLERANCE_RATIO)
                    for value in local_price_refs
                    if value > 0
                ):
                    continue

            contracts = abs(_safe_float(snapshot.get("contracts"), 0.0))
            contract_size = _contract_size_for_position(
                inst_id,
                row,
                snapshot,
                contract_sizes=sizes,
            )
            base_quantity = contracts * contract_size if contracts > 0 and contract_size > 0 else 0.0
            quantity_matches = False
            if base_quantity > 0 and any(value > 0 for value in local_quantity_refs):
                quantity_matches = any(
                    _relative_close_enough(
                        value,
                        base_quantity,
                        POSITION_CONFIRM_QUANTITY_TOLERANCE_RATIO,
                    )
                    for value in local_quantity_refs
                    if value > 0
                )
                if not quantity_matches and not pos_id_matches:
                    continue
            elif not pos_id_matches:
                continue

            info = _safe_dict(row.get("info"))
            symbol = symbol_from_okx_inst_id(inst_id) or normalize_trading_symbol(
                snapshot.get("symbol") or inst_id
            )
            return OkxCurrentPositionEntryConfirmation(
                exchange_order_id=order_id,
                inst_id=inst_id,
                symbol=symbol,
                side=order_side,
                pos_side=str(info.get("posSide") or snapshot.get("raw_pos_side") or "").lower().strip(),
                pos_id=pos_id,
                trade_id=str(info.get("tradeId") or row.get("tradeId") or "").strip(),
                contracts=contracts,
                contract_size=contract_size,
                base_quantity=base_quantity or max(local_quantity_refs),
                avg_price=avg_price,
                mark_price=_safe_float(snapshot.get("mark_price"), 0.0)
                or _safe_float(snapshot.get("last_price"), 0.0),
                upl=_safe_float_or_none(snapshot.get("upl")),
                fee_abs=_abs_float_or_none(info.get("fee") or row.get("fee")),
                timestamp=_datetime_from_ms(
                    row.get("timestamp") or info.get("uTime") or info.get("cTime")
                ),
                local_position_id=int(getattr(local_position, "id", 0) or 0) or None,
                match_source=(
                    "okx_current_position_pos_id"
                    if pos_id_matches
                    else "okx_current_position_quantity_price"
                ),
                raw_position=_raw_position_row(row),
            )
    return None


def order_has_current_position_snapshot_confirmation(
    order: Any,
    *,
    exchange_positions: Iterable[dict[str, Any]],
) -> bool:
    """Return True when a prior OKX current-position confirmation still matches."""

    raw = getattr(order, "okx_raw_fills", None)
    if not isinstance(raw, dict):
        return False
    if raw.get("position_snapshot_confirmed") is not True:
        return False
    if raw.get("fills_history_confirmed") is not False:
        return False
    inst_id = str(raw.get("inst_id") or getattr(order, "okx_inst_id", "") or "").strip().upper()
    pos_id = str(raw.get("pos_id") or "").strip()
    if not inst_id:
        return False
    expected_side = _entry_position_side(str(getattr(order, "side", "") or "").lower().strip())
    for row in exchange_positions or ():
        if not isinstance(row, dict):
            continue
        snapshot = parse_exchange_position_snapshot(
            row,
            symbol_normalizer=normalize_trading_symbol,
        )
        if not snapshot:
            continue
        row_inst_id = _position_row_inst_id(row, snapshot)
        if row_inst_id != inst_id:
            continue
        if expected_side and str(snapshot.get("side") or "").lower().strip() != expected_side:
            continue
        row_pos_id = _position_row_pos_id(row)
        if pos_id and row_pos_id and row_pos_id != pos_id:
            continue
        return True
    return False


def _entry_position_side(order_side: str) -> str:
    if order_side == "buy":
        return "long"
    if order_side == "sell":
        return "short"
    return ""


def _local_position_inst_id(position: Any) -> str:
    inst_id = str(getattr(position, "okx_inst_id", "") or "").strip().upper()
    if inst_id:
        return inst_id
    return okx_inst_id_from_symbol(getattr(position, "symbol", None)) or ""


def _order_inst_id(order: Any) -> str:
    inst_id = str(getattr(order, "okx_inst_id", "") or "").strip().upper()
    if inst_id:
        return inst_id
    return okx_inst_id_from_symbol(getattr(order, "symbol", None)) or ""


def _position_row_inst_id(row: dict[str, Any], snapshot: dict[str, Any]) -> str:
    info = _safe_dict(row.get("info"))
    for value in (
        info.get("instId"),
        snapshot.get("raw_symbol"),
        row.get("symbol"),
        snapshot.get("symbol"),
    ):
        text = str(value or "").strip().upper()
        if text.endswith("-SWAP"):
            return text
        inst_id = okx_inst_id_from_symbol(text)
        if inst_id:
            return inst_id
    return ""


def _position_row_pos_id(row: dict[str, Any]) -> str:
    info = _safe_dict(row.get("info"))
    return str(info.get("posId") or row.get("id") or row.get("posId") or "").strip()


def _contract_size_for_position(
    inst_id: str,
    row: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    contract_sizes: dict[str, float],
) -> float:
    size = _safe_float(contract_sizes.get(inst_id), 0.0)
    if size > 0:
        return size
    info = _safe_dict(row.get("info"))
    for value in (
        snapshot.get("contract_size"),
        row.get("contractSize"),
        row.get("contract_size"),
        info.get("ctVal"),
        info.get("contractSize"),
    ):
        size = _safe_float(value, 0.0)
        if size > 0:
            return size
    return 0.0


def _split_exchange_order_ids(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    tokens = {text}
    for separator in (",", ";", "|", "\n", "\t", " "):
        pieces: set[str] = set()
        for token in tokens:
            pieces.update(part.strip() for part in token.split(separator) if part.strip())
        tokens = pieces
    return {token for token in tokens if token}


def _relative_close_enough(left: float, right: float, tolerance: float) -> bool:
    denominator = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denominator <= tolerance


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _raw_position_row(row: dict[str, Any]) -> dict[str, Any]:
    info = _safe_dict(row.get("info"))
    if info:
        return dict(info)
    return dict(row)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _abs_float_or_none(value: Any) -> float | None:
    parsed = _safe_float_or_none(value)
    return abs(parsed) if parsed is not None else None


def _datetime_from_ms(value: Any) -> datetime | None:
    timestamp_ms = _safe_float(value, 0.0)
    if timestamp_ms <= 0:
        return None
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000.0, UTC)
    except (OSError, OverflowError, ValueError):
        return None
