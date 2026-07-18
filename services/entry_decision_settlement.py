"""Backfill entry-decision outcomes from finalized OKX position history."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select

from models.decision import AIDecision
from models.trade import OkxPositionHistory, Order, Position
from services.position_settlement import final_settlement_status_values

ENTRY_DECISION_SETTLEMENT_VERSION = "2026-07-18.entry-decision-settlement.v1"
DEFAULT_BACKFILL_LOOKBACK_HOURS = 24 * 14
DEFAULT_BACKFILL_LIMIT = 500


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _exchange_order_ids(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = re.split(r"[,;|\s]+", str(value or ""))
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _expected_entry_action(position: Position) -> str:
    side = str(getattr(position, "side", "") or "").strip().lower()
    return side if side in {"long", "short"} else ""


def _outcome(realized_pnl: float) -> str:
    if realized_pnl > 1e-12:
        return "profit"
    if realized_pnl < -1e-12:
        return "loss"
    return "flat"


def _history_is_authoritative(history: OkxPositionHistory | None) -> bool:
    if history is None:
        return False
    return bool(
        str(getattr(history, "close_status", "") or "").lower() == "full"
        and str(getattr(history, "sync_status", "") or "").lower() == "synced"
        and not list(getattr(history, "evidence_gaps", None) or [])
        and _safe_float(getattr(history, "pnl_ratio", None)) is not None
    )


async def _load_history(
    session: Any,
    *,
    position: Position,
) -> OkxPositionHistory | None:
    pos_id = str(getattr(position, "okx_pos_id", "") or "").strip()
    if not pos_id:
        return None
    result = await session.execute(
        select(OkxPositionHistory)
        .where(
            OkxPositionHistory.mode == str(position.execution_mode or "paper"),
            OkxPositionHistory.pos_id == pos_id,
        )
        .order_by(
            OkxPositionHistory.updated_at_okx.desc().nullslast(),
            OkxPositionHistory.id.desc(),
        )
        .limit(2)
    )
    rows = list(result.scalars().all())
    return rows[0] if len(rows) == 1 else None


async def sync_settled_entry_decision_outcome(
    session: Any,
    *,
    position: Position,
    history: OkxPositionHistory | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Apply one exact, authoritative position lifecycle to its entry decision."""

    checked_at = now or datetime.now(UTC)
    if (
        bool(getattr(position, "is_open", True))
        or str(getattr(position, "settlement_status", "") or "")
        not in final_settlement_status_values()
    ):
        return {"changed": False, "reason": "position_not_finally_settled"}

    entry_order_ids = _exchange_order_ids(getattr(position, "entry_exchange_order_id", None))
    if not entry_order_ids:
        return {"changed": False, "reason": "entry_exchange_order_id_missing"}

    order_result = await session.execute(
        select(Order).where(
            Order.execution_mode == str(position.execution_mode or "paper"),
            Order.exchange_order_id.in_(entry_order_ids),
        )
    )
    orders = list(order_result.scalars().all())
    decision_ids = sorted(
        {
            int(order.decision_id)
            for order in orders
            if getattr(order, "decision_id", None) is not None
            and str(getattr(order, "status", "") or "").lower() == "filled"
        }
    )
    if len(decision_ids) != 1:
        return {
            "changed": False,
            "reason": "entry_decision_link_not_unique",
            "decision_ids": decision_ids,
        }

    decision = await session.get(AIDecision, decision_ids[0])
    expected_action = _expected_entry_action(position)
    if (
        decision is None
        or not bool(decision.was_executed)
        or str(decision.action or "").lower() != expected_action
        or bool(decision.is_paper) != (str(position.execution_mode or "paper") == "paper")
    ):
        return {
            "changed": False,
            "reason": "entry_decision_identity_mismatch",
            "decision_id": decision_ids[0],
        }

    authoritative_history = history or await _load_history(session, position=position)
    if not _history_is_authoritative(authoritative_history):
        return {
            "changed": False,
            "reason": "authoritative_position_history_incomplete",
            "decision_id": int(decision.id),
        }

    realized_pnl = _safe_float(getattr(authoritative_history, "realized_pnl", None))
    pnl_ratio = _safe_float(getattr(authoritative_history, "pnl_ratio", None))
    if realized_pnl is None or pnl_ratio is None:
        return {
            "changed": False,
            "reason": "authoritative_outcome_values_missing",
            "decision_id": int(decision.id),
        }

    outcome = _outcome(realized_pnl)
    outcome_pnl_pct = pnl_ratio * 100.0
    raw = _safe_dict(decision.raw_llm_response)
    existing_contract = _safe_dict(raw.get("authoritative_settlement_outcome"))
    contract = {
        "version": ENTRY_DECISION_SETTLEMENT_VERSION,
        "authority": "okx_position_history",
        "position_id": int(position.id),
        "history_id": int(authoritative_history.id),
        "okx_pos_id": str(authoritative_history.pos_id or ""),
        "entry_order_ids": entry_order_ids,
        "outcome": outcome,
        "outcome_pnl_pct": outcome_pnl_pct,
        "realized_pnl_usdt": realized_pnl,
        "settlement_status": str(position.settlement_status or ""),
        "synced_at": checked_at.isoformat(),
    }
    stable_contract = {key: value for key, value in contract.items() if key != "synced_at"}
    stable_existing_contract = {
        key: value for key, value in existing_contract.items() if key != "synced_at"
    }
    changed = bool(
        str(decision.outcome or "") != outcome
        or _safe_float(decision.outcome_pnl_pct) != outcome_pnl_pct
        or stable_existing_contract != stable_contract
    )
    if not changed:
        return {
            "changed": False,
            "reason": "decision_outcome_already_authoritative",
            "decision_id": int(decision.id),
        }

    decision.outcome = outcome
    decision.outcome_pnl_pct = outcome_pnl_pct
    raw["authoritative_settlement_outcome"] = contract
    decision.raw_llm_response = raw
    decision.updated_at = checked_at
    return {
        "changed": True,
        "reason": "decision_outcome_synced_from_okx_position_history",
        "decision_id": int(decision.id),
        "position_id": int(position.id),
        "outcome": outcome,
        "outcome_pnl_pct": outcome_pnl_pct,
    }


async def backfill_settled_entry_decision_outcomes(
    session: Any,
    *,
    mode: str,
    now: datetime | None = None,
    lookback_hours: int = DEFAULT_BACKFILL_LOOKBACK_HOURS,
    limit: int = DEFAULT_BACKFILL_LIMIT,
) -> list[dict[str, Any]]:
    """Repair recent final positions whose exact entry decision missed settlement."""

    checked_at = now or datetime.now(UTC)
    since = checked_at - timedelta(hours=max(int(lookback_hours or 1), 1))
    result = await session.execute(
        select(Position)
        .join(
            Order,
            Order.exchange_order_id == Position.entry_exchange_order_id,
        )
        .join(AIDecision, AIDecision.id == Order.decision_id)
        .where(
            Position.execution_mode == ("live" if mode == "live" else "paper"),
            Position.is_open.is_(False),
            Position.closed_at.is_not(None),
            Position.closed_at >= since,
            Position.settlement_status.in_(final_settlement_status_values()),
            Position.entry_exchange_order_id.is_not(None),
            Order.execution_mode == ("live" if mode == "live" else "paper"),
            Order.status == "filled",
            AIDecision.was_executed.is_(True),
            AIDecision.action.in_(("long", "short")),
            or_(
                AIDecision.outcome.is_(None),
                AIDecision.outcome == "",
                AIDecision.outcome_pnl_pct.is_(None),
            ),
        )
        .order_by(Position.closed_at.desc(), Position.id.desc())
        .limit(max(1, min(int(limit or 1), 2000)))
    )
    changes: list[dict[str, Any]] = []
    for position in result.scalars().unique().all():
        sync_result = await sync_settled_entry_decision_outcome(
            session,
            position=position,
            now=checked_at,
        )
        if sync_result.get("changed") is True:
            changes.append(sync_result)
    if changes:
        await session.flush()
    return changes
