from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from db.session import get_session_ctx
from models.learning import TradeReflection
from models.trade import Position
from services.manual_close_marker import is_manual_close_exchange_order_id
from services.trade_fact_trust import closed_position_trade_fact_untrusted_reason

REPAIR_REFLECTION_SOURCES = {
    "missing_closed_position_repair",
    "okx_native_full_close_fill_correction",
    "okx_order_pair_repair",
    "okx_orphan_position_quarantine",
    "okx_position_link_repair",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _is_repair_reflection(reflection: TradeReflection) -> bool:
    source = str(getattr(reflection, "source", "") or "").strip().lower()
    if source in REPAIR_REFLECTION_SOURCES:
        return True
    lessons = getattr(reflection, "expert_lessons", None)
    if isinstance(lessons, dict):
        lesson_source = str(lessons.get("source") or "").strip().lower()
        training_policy = str(lessons.get("training_policy") or "").strip().lower()
        if lesson_source in REPAIR_REFLECTION_SOURCES:
            return True
        if "exclude_until_manual_trust" in training_policy:
            return True
    return False


def _classification_for_position(
    position: Position,
    *,
    repair_position_ids: set[int],
) -> tuple[str, str]:
    position_id = _safe_int(getattr(position, "id", None))
    trust_reason = closed_position_trade_fact_untrusted_reason(position)
    if position_id in repair_position_ids:
        return "quarantined", "historical_repair_provenance"
    if trust_reason:
        return "quarantined", trust_reason
    close_order_id = str(getattr(position, "close_exchange_order_id", "") or "").strip()
    if close_order_id and is_manual_close_exchange_order_id(close_order_id):
        return "quarantined", "manual_close_exchange_order_id"
    return "trainable", "okx_backed_closed_trade_fact"


@dataclass(frozen=True)
class HistoricalTradeFactAuditService:
    """Read-only audit for closed trade facts before they enter training."""

    lookback_days: int = 180
    limit: int = 2000

    async def report(self) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        lookback_days = max(int(self.lookback_days or 0), 1)
        limit = max(min(int(self.limit or 0), 10000), 1)
        since = started_at - timedelta(days=lookback_days)

        async with get_session_ctx() as session:
            positions = list(
                (
                    await session.execute(
                        select(Position)
                        .where(
                            Position.is_open.is_(False),
                            Position.closed_at.is_not(None),
                            Position.closed_at >= since,
                        )
                        .order_by(Position.closed_at.desc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            position_ids = [_safe_int(getattr(position, "id", None)) for position in positions]
            repair_position_ids: set[int] = set()
            if position_ids:
                reflections = list(
                    (
                        await session.execute(
                            select(TradeReflection).where(
                                TradeReflection.position_id.in_(position_ids)
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                repair_position_ids = {
                    _safe_int(getattr(reflection, "position_id", None))
                    for reflection in reflections
                    if _is_repair_reflection(reflection)
                }

        status_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        symbol_reason_counts: Counter[tuple[str, str]] = Counter()
        mode_counts: Counter[str] = Counter()
        samples: list[dict[str, Any]] = []
        repairable_count = 0
        manual_close_marker_count = 0
        missing_entry_count = 0
        missing_close_count = 0

        for position in positions:
            classification, reason = _classification_for_position(
                position,
                repair_position_ids=repair_position_ids,
            )
            status_counts[classification] += 1
            reason_counts[reason] += 1
            symbol = str(getattr(position, "symbol", "") or "")
            side = str(getattr(position, "side", "") or "")
            mode = str(getattr(position, "execution_mode", "") or "unknown")
            symbol_reason_counts[(symbol, reason)] += 1
            mode_counts[mode] += 1

            if reason in {"missing_entry_exchange_order_id", "missing_close_exchange_order_id"}:
                repairable_count += 1
            if reason == "missing_entry_exchange_order_id":
                missing_entry_count += 1
            if reason == "missing_close_exchange_order_id":
                missing_close_count += 1
            if reason == "manual_close_exchange_order_id":
                manual_close_marker_count += 1

            if classification != "trainable" and len(samples) < 30:
                samples.append(
                    {
                        "position_id": _safe_int(getattr(position, "id", None)),
                        "symbol": symbol,
                        "side": side,
                        "execution_mode": mode,
                        "reason": reason,
                        "realized_pnl": round(_safe_float(getattr(position, "realized_pnl", 0.0)), 8),
                        "closed_at": _iso(getattr(position, "closed_at", None)),
                        "entry_exchange_order_id": getattr(position, "entry_exchange_order_id", None),
                        "close_exchange_order_id": getattr(position, "close_exchange_order_id", None),
                    }
                )

        checked = len(positions)
        trainable_count = int(status_counts.get("trainable", 0))
        quarantined_count = checked - trainable_count
        status = "clean" if quarantined_count == 0 else "dirty"
        return {
            "status": status,
            "read_only": True,
            "audit_only": True,
            "raw_records_preserved": True,
            "cleanup_mode": "quarantine_not_delete",
            "training_policy": "clean_training_view_only",
            "lookback_days": lookback_days,
            "limit": limit,
            "checked_closed_positions": checked,
            "trainable_closed_positions": trainable_count,
            "quarantined_closed_positions": quarantined_count,
            "repairable_candidate_count": repairable_count,
            "manual_close_marker_count": manual_close_marker_count,
            "missing_entry_link_count": missing_entry_count,
            "missing_close_link_count": missing_close_count,
            "historical_repair_provenance_count": len(repair_position_ids),
            "reason_counts": dict(reason_counts),
            "status_counts": dict(status_counts),
            "execution_mode_counts": dict(mode_counts),
            "top_symbol_reasons": [
                {"symbol": symbol, "reason": reason, "count": count}
                for (symbol, reason), count in symbol_reason_counts.most_common(20)
            ],
            "samples": samples,
            "checked_at": datetime.now(UTC).isoformat(),
            "duration_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 6),
            "can_delete_history": False,
            "can_apply_repair": False,
            "requires_backup_for_apply": True,
            "repair_entrypoints": [
                "scripts/repair_okx_history_position_reconciliation.py",
                "scripts/repair_missing_position_links_from_okx_fills.py",
            ],
        }
