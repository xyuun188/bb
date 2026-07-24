"""Persistent OKX positions-history mirror.

OKX positions-history rows are the authoritative lifecycle facts for closed
position display and training labels. This module keeps those rows in a
dedicated table so dashboard reads do not depend on live OKX availability and
do not mix local Position fragments with exchange history semantics.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from core.symbols import normalize_trading_symbol, symbol_from_okx_inst_id
from models.trade import OkxPositionHistory


def okx_position_history_row_identity(row: dict[str, Any], *, mode: str | None = None) -> str:
    inst_id = _inst_id(row)
    pos_id = _text(row.get("posId") or row.get("pos_id"))
    pos_side = _text(row.get("posSide") or row.get("pos_side")).lower()
    c_time = _text(row.get("cTime") or row.get("createdTime") or row.get("openTime"))
    u_time = _text(row.get("uTime") or row.get("updatedTime") or row.get("closeTime"))
    lifecycle_time = c_time or u_time
    return "|".join(
        [
            _text(mode),
            inst_id,
            pos_id,
            pos_side,
            lifecycle_time,
        ]
    )


async def upsert_okx_position_history_row(
    session: Any,
    row: dict[str, Any],
    *,
    mode: str,
    source: str,
    entry_order_ids: Iterable[Any] | None = None,
    close_order_ids: Iterable[Any] | None = None,
    position_ids: Iterable[Any] | None = None,
    match_status: str | None = None,
    evidence_gaps: Iterable[Any] | None = None,
    synced_at: datetime | None = None,
    last_sync_error: str | None = None,
) -> OkxPositionHistory | None:
    if not isinstance(row, dict):
        return None
    identity = okx_position_history_row_identity(row, mode=mode)
    if not identity.strip("|"):
        return None
    existing_result = await session.execute(
        select(OkxPositionHistory).where(
            OkxPositionHistory.mode == mode,
            OkxPositionHistory.row_identity == identity,
        )
    )
    existing = existing_result.scalars().first()
    payload = _payload_from_row(
        row,
        mode=mode,
        row_identity=identity,
        source=source,
        entry_order_ids=entry_order_ids,
        close_order_ids=close_order_ids,
        position_ids=position_ids,
        match_status=match_status,
        evidence_gaps=evidence_gaps,
        synced_at=synced_at,
        last_sync_error=last_sync_error,
    )
    if existing is None:
        record = OkxPositionHistory(**payload)
        session.add(record)
        return record
    _apply_payload(existing, payload)
    return existing


async def load_okx_position_history_records(
    session: Any,
    *,
    mode: str | None,
    limit: int = 5000,
) -> list[OkxPositionHistory]:
    stmt = select(OkxPositionHistory).order_by(
        OkxPositionHistory.updated_at_okx.desc().nullslast(),
        OkxPositionHistory.opened_at.desc().nullslast(),
        OkxPositionHistory.id.desc(),
    )
    if mode:
        stmt = stmt.where(OkxPositionHistory.mode == mode)
    safe_limit = max(1, int(limit or 1))
    result = await session.execute(stmt.limit(min(safe_limit * 4, 20000)))
    records: list[OkxPositionHistory] = []
    seen_lifecycles: set[str] = set()
    for record in result.scalars().all():
        stable_identity = okx_position_history_row_identity(
            dict(record.raw_row or {}),
            mode=record.mode,
        )
        if stable_identity in seen_lifecycles:
            continue
        seen_lifecycles.add(stable_identity)
        records.append(record)
        if len(records) >= safe_limit:
            break
    return records


def okx_position_history_records_to_rows(
    records: Iterable[OkxPositionHistory],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        raw = dict(record.raw_row or {})
        raw.setdefault("instId", record.inst_id)
        raw.setdefault("posId", record.pos_id or "")
        raw.setdefault("posSide", record.pos_side or "")
        raw.setdefault("type", record.close_type or "")
        if record.opened_at and not _text(raw.get("cTime") or raw.get("createdTime")):
            raw["cTime"] = str(int(_as_utc(record.opened_at).timestamp() * 1000))
        if record.updated_at_okx and not _text(raw.get("uTime") or raw.get("updatedTime")):
            raw["uTime"] = str(int(_as_utc(record.updated_at_okx).timestamp() * 1000))
        raw.setdefault("openAvgPx", str(record.open_avg_px))
        raw.setdefault("closeAvgPx", str(record.close_avg_px))
        raw.setdefault("openMaxPos", str(record.open_max_pos))
        raw.setdefault("closeTotalPos", str(record.close_total_pos))
        raw.setdefault("lever", str(record.leverage))
        raw.setdefault("realizedPnl", str(record.realized_pnl))
        raw.setdefault("pnl", str(record.pnl))
        raw.setdefault("fundingFee", str(record.funding_fee))
        raw["_dashboard_history_record_id"] = record.id
        raw["_dashboard_history_row_identity"] = record.row_identity
        raw["_dashboard_entry_order_ids"] = _list_from_raw(record.entry_order_ids)
        raw["_dashboard_close_order_ids"] = _list_from_raw(record.close_order_ids)
        raw["_dashboard_linked_order_ids"] = _list_from_raw(record.linked_order_ids)
        raw["_dashboard_position_ids"] = _list_from_raw(record.position_ids)
        raw["_dashboard_history_source"] = record.source
        rows.append(raw)
    return rows


def _payload_from_row(
    row: dict[str, Any],
    *,
    mode: str,
    row_identity: str,
    source: str,
    entry_order_ids: Iterable[Any] | None,
    close_order_ids: Iterable[Any] | None,
    position_ids: Iterable[Any] | None,
    match_status: str | None,
    evidence_gaps: Iterable[Any] | None,
    synced_at: datetime | None,
    last_sync_error: str | None,
) -> dict[str, Any]:
    inst_id = _inst_id(row)
    symbol = symbol_from_okx_inst_id(inst_id) or normalize_trading_symbol(inst_id)
    close_type = _text(row.get("type") or row.get("closeType") or row.get("close_type"))
    opened_at = _datetime_from_ms(row.get("cTime") or row.get("createdTime") or row.get("openTime"))
    updated_at = _datetime_from_ms(row.get("uTime") or row.get("updatedTime") or row.get("closeTime"))
    open_max_pos = _safe_float(row.get("openMaxPos") or row.get("open_max_pos"), 0.0)
    close_total_pos = _safe_float(row.get("closeTotalPos") or row.get("close_total_pos"), 0.0)
    close_status = _close_status(close_type, open_max_pos=open_max_pos, close_total_pos=close_total_pos)
    entry_ids = _clean_list(entry_order_ids)
    close_ids = _clean_list(close_order_ids)
    linked_ids = _merge_lists(entry_ids, close_ids)
    return {
        "mode": mode,
        "row_identity": row_identity,
        "inst_id": inst_id,
        "symbol": symbol,
        "pos_id": _text(row.get("posId") or row.get("pos_id")) or None,
        "pos_side": _text(row.get("posSide") or row.get("pos_side")) or None,
        "side": _side(row),
        "close_type": close_type or None,
        "close_status": close_status,
        "opened_at": opened_at,
        "updated_at_okx": updated_at,
        "open_avg_px": _safe_float(row.get("openAvgPx") or row.get("open_avg_px"), 0.0),
        "close_avg_px": _safe_float(row.get("closeAvgPx") or row.get("close_avg_px"), 0.0),
        "open_max_pos": open_max_pos,
        "close_total_pos": close_total_pos,
        "leverage": _safe_float(row.get("lever") or row.get("leverage"), 1.0) or 1.0,
        "realized_pnl": _safe_float(row.get("realizedPnl") or row.get("realized_pnl"), 0.0),
        "pnl": _safe_float(row.get("pnl") or row.get("closePnl") or row.get("close_pnl"), 0.0),
        "pnl_ratio": _safe_float(row.get("pnlRatio") or row.get("pnl_ratio"), None),
        "funding_fee": _safe_float(row.get("fundingFee") or row.get("funding_fee"), 0.0),
        "fee": _safe_float(row.get("fee") or row.get("totalFee") or row.get("total_fee"), 0.0),
        "entry_order_ids": entry_ids,
        "close_order_ids": close_ids,
        "linked_order_ids": linked_ids,
        "position_ids": _clean_list(position_ids),
        "match_status": _text(match_status) or "unmatched",
        "evidence_gaps": _clean_list(evidence_gaps),
        "source": source,
        "raw_row": dict(row),
        "sync_status": "error" if last_sync_error else "synced",
        "last_sync_error": last_sync_error,
        "synced_at": synced_at or datetime.now(UTC),
    }


def _apply_payload(record: OkxPositionHistory, payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if key in {"entry_order_ids", "close_order_ids", "linked_order_ids", "position_ids"}:
            existing = getattr(record, key, None)
            setattr(record, key, _merge_lists(_list_from_raw(existing), _list_from_raw(value)))
        elif key == "raw_row":
            existing_raw = dict(getattr(record, "raw_row", None) or {})
            incoming_raw = dict(value or {})
            for raw_key, raw_value in existing_raw.items():
                if raw_key.startswith("_bb_") and raw_key not in incoming_raw:
                    incoming_raw[raw_key] = raw_value
            record.raw_row = incoming_raw
        elif key != "evidence_gaps":
            setattr(record, key, value)
    evidence_gaps = _clean_list(payload.get("evidence_gaps"))
    if _list_from_raw(record.entry_order_ids):
        evidence_gaps = [
            gap for gap in evidence_gaps if gap != "missing_position_history_entry_orders"
        ]
    if _list_from_raw(record.close_order_ids):
        evidence_gaps = [
            gap for gap in evidence_gaps if gap != "missing_position_history_close_orders"
        ]
    record.evidence_gaps = evidence_gaps


def _inst_id(row: dict[str, Any]) -> str:
    return _text(row.get("instId") or row.get("inst_id")).upper()


def _side(row: dict[str, Any]) -> str:
    side = _text(row.get("posSide") or row.get("pos_side")).lower()
    if side in {"long", "short"}:
        return side
    direction = _text(row.get("direction") or row.get("side")).lower()
    if direction in {"long", "short"}:
        return direction
    open_side = _text(row.get("openSide") or row.get("openOrdSide")).lower()
    if open_side == "sell":
        return "short"
    if open_side == "buy":
        return "long"
    open_price = _safe_float(row.get("openAvgPx"), 0.0)
    close_price = _safe_float(row.get("closeAvgPx"), 0.0)
    pnl = _safe_float(row.get("pnl") or row.get("realizedPnl"), None)
    if open_price > 0 and close_price > 0 and pnl is not None and close_price != open_price:
        return "long" if (close_price > open_price) == (pnl >= 0) else "short"
    return ""


def _close_status(close_type: str, *, open_max_pos: float, close_total_pos: float) -> str:
    if close_type == "1":
        return "partial"
    if close_type == "2":
        return "full"
    if open_max_pos > 0 and close_total_pos > 0 and close_total_pos < open_max_pos:
        return "partial"
    return "full"


def _datetime_from_ms(value: Any) -> datetime | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    if number < 10_000_000_000:
        number *= 1000
    return datetime.fromtimestamp(number / 1000, tz=UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_float(value: Any, default: float | None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number_text(value: Any) -> str:
    number = _safe_float(value, None)
    if number is None:
        return _text(value)
    return f"{number:.12g}"


def _clean_list(values: Iterable[Any] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        for token in _list_from_raw(value):
            if token and token not in seen:
                result.append(token)
                seen.add(token)
    return result


def _list_from_raw(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        pieces: list[str] = []
        for item in value:
            pieces.extend(_list_from_raw(item))
        return pieces
    text = str(value or "").strip()
    if not text:
        return []
    tokens = {text}
    for separator in (",", ";", "|", "\n", "\t", " "):
        next_tokens: set[str] = set()
        for token in tokens:
            next_tokens.update(piece.strip() for piece in token.split(separator) if piece.strip())
        tokens = next_tokens
    return sorted(tokens)


def _merge_lists(*items: Iterable[Any] | None) -> list[str]:
    return _clean_list(value for item in items for value in (item or []))
