"""Resolve one authoritative decision for an OKX-reconciled exit fill."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import String, cast, select

from models.decision import AIDecision
from models.trade import Order, Position

RECONCILE_ORIGIN_SYSTEM_EXECUTION = "system_execution"
EXIT_LINEAGE_VERSION = "2026-07-15.okx-exit-decision-lineage.v1"
EXIT_ACTIONS = {"close_long", "close_short", "exit_long", "exit_short"}
SUPERSEDED_POSITION_STATUS = "superseded_position_residual"
SUPERSEDED_POSITION_REASONS = {
    "duplicate_local_open_position_for_same_okx_pos_id",
    "duplicate_local_closed_position_for_same_okx_lifecycle",
    "duplicate_local_closed_position_for_same_okx_close_order",
    "duplicate_local_closed_position_for_same_okx_lifecycle_exact_projection",
}


class ExitDecisionLineageAmbiguous(RuntimeError):
    """More than one production decision claims the same OKX close order."""


@dataclass(frozen=True, slots=True)
class ExitDecisionLineageResolution:
    authoritative: AIDecision | None
    linked_order: Order | None
    superseded: tuple[AIDecision, ...]
    matched_decision_ids: tuple[int, ...]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested_order_ids(value: Any) -> set[str]:
    payload = _safe_dict(value)
    return {
        str(candidate or "").strip()
        for candidate in (
            payload.get("exchange_order_id"),
            payload.get("order_id"),
            payload.get("ordId"),
        )
        if str(candidate or "").strip()
    }


def decision_exit_exchange_order_ids(decision: Any) -> set[str]:
    """Read only execution/close paths; entry lineage cannot claim an exit fill."""

    raw = _safe_dict(getattr(decision, "raw_llm_response", None))
    execution = _safe_dict(raw.get("execution_result"))
    execution_raw = _safe_dict(execution.get("raw_response"))
    execution_info = _safe_dict(execution_raw.get("info"))
    return set().union(
        _nested_order_ids(execution),
        _nested_order_ids(raw.get("native_close_fill")),
        _nested_order_ids(raw.get("close_fill")),
        _nested_order_ids(execution.get("native_close_fill")),
        _nested_order_ids(execution_raw),
        _nested_order_ids(execution_info),
    )


def choose_exit_decision_lineage(
    decisions: Sequence[AIDecision],
    *,
    close_order_id: str,
    linked_decision: AIDecision | None = None,
    linked_order: Order | None = None,
) -> ExitDecisionLineageResolution:
    """Prefer the original production decision over synthetic reconciliation rows."""

    order_id = str(close_order_id or "").strip()
    exact = [
        decision
        for decision in decisions
        if order_id and order_id in decision_exit_exchange_order_ids(decision)
    ]
    by_id = {int(decision.id): decision for decision in exact}
    if linked_decision is not None:
        by_id.setdefault(int(linked_decision.id), linked_decision)
    candidates = list(by_id.values())
    production = [
        decision
        for decision in candidates
        if _safe_dict(decision.raw_llm_response).get("system_sync") is not True
    ]
    if len(production) > 1:
        raise ExitDecisionLineageAmbiguous(
            f"Multiple production decisions claim OKX close order {order_id}: "
            + ",".join(str(decision.id) for decision in production)
        )
    authoritative = production[0] if production else None
    if authoritative is None and linked_decision is not None:
        authoritative = linked_decision
    if authoritative is None:
        system_rows = [
            decision
            for decision in candidates
            if _safe_dict(decision.raw_llm_response).get("system_sync") is True
        ]
        if len(system_rows) > 1:
            raise ExitDecisionLineageAmbiguous(
                f"Multiple reconciliation decisions claim OKX close order {order_id}: "
                + ",".join(str(decision.id) for decision in system_rows)
            )
        authoritative = system_rows[0] if system_rows else None
    superseded = tuple(
        decision
        for decision in candidates
        if authoritative is not None
        and int(decision.id) != int(authoritative.id)
        and _safe_dict(decision.raw_llm_response).get("system_sync") is True
        and (
            bool(getattr(decision, "was_executed", False))
            or getattr(decision, "executed_at", None) is not None
            or getattr(decision, "execution_price", None) is not None
            or getattr(decision, "outcome", None) is not None
            or getattr(decision, "outcome_pnl_pct", None) is not None
            or (
                linked_order is not None
                and int(getattr(linked_order, "decision_id", 0) or 0) == int(decision.id)
            )
        )
    )
    return ExitDecisionLineageResolution(
        authoritative=authoritative,
        linked_order=linked_order,
        superseded=superseded,
        matched_decision_ids=tuple(sorted(by_id)),
    )


async def load_exit_decision_lineage(
    session: Any,
    *,
    model_name: str,
    symbol: str,
    action: str,
    is_paper: bool,
    execution_mode: str,
    close_order_id: str,
) -> ExitDecisionLineageResolution:
    """Load exact order/JSON links without using a guessed time window."""

    order_id = str(close_order_id or "").strip()
    if not order_id:
        return ExitDecisionLineageResolution(None, None, (), ())
    order_result = await session.execute(
        select(Order)
        .where(
            Order.execution_mode == execution_mode,
            Order.exchange_order_id == order_id,
        )
        .limit(1)
    )
    linked_order = order_result.scalar_one_or_none()
    linked_decision = None
    if linked_order is not None and getattr(linked_order, "decision_id", None):
        linked_decision = await session.get(AIDecision, int(linked_order.decision_id))

    decisions_result = await session.execute(
        select(AIDecision).where(
            AIDecision.model_name == model_name,
            AIDecision.symbol == symbol,
            AIDecision.action == action,
            AIDecision.is_paper.is_(is_paper),
            AIDecision.raw_llm_response.is_not(None),
            cast(AIDecision.raw_llm_response, String).contains(order_id),
        )
    )
    decisions = list(decisions_result.scalars().all())
    return choose_exit_decision_lineage(
        decisions,
        close_order_id=order_id,
        linked_decision=linked_decision,
        linked_order=linked_order,
    )


async def recover_exit_decision_lineage_from_order_fact(
    session: Any,
    *,
    order: Order,
    close_fill: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Repair a close order whose local decision link lost a concurrency race.

    The exchange order id is the only join key. This helper deliberately skips
    entry actions and refuses to infer a position from a time window.
    """

    close_order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
    if not close_order_id:
        return None
    linked_decision_id = int(getattr(order, "decision_id", 0) or 0)
    if not linked_decision_id:
        return None
    linked_decision = await session.get(AIDecision, linked_decision_id)
    action = str(getattr(linked_decision, "action", "") or "").strip().lower()
    if action not in EXIT_ACTIONS:
        return None
    expected_action = (
        "close_short"
        if str(getattr(order, "side", "") or "").strip().lower() == "buy"
        else "close_long"
    )
    if action != expected_action:
        return None
    resolution = await load_exit_decision_lineage(
        session,
        model_name=str(getattr(order, "model_name", "") or ""),
        symbol=str(getattr(order, "symbol", "") or ""),
        action=action,
        is_paper=str(getattr(order, "execution_mode", "") or "").lower() != "live",
        execution_mode=str(getattr(order, "execution_mode", "") or "paper"),
        close_order_id=close_order_id,
    )
    authoritative = resolution.authoritative
    if authoritative is None:
        return None
    if int(getattr(order, "decision_id", 0) or 0) == int(authoritative.id) and not resolution.superseded:
        return None

    raw_fill = dict(close_fill or {})
    if not raw_fill:
        stored = getattr(order, "okx_raw_fills", None)
        if isinstance(stored, dict):
            raw_fill = dict(stored)
    raw_fill["order_id"] = close_order_id
    position_result = await session.execute(
        select(Position).where(
            Position.execution_mode == getattr(order, "execution_mode", "paper"),
            Position.symbol == getattr(order, "symbol", ""),
            Position.is_open.is_(False),
            Position.close_exchange_order_id.is_not(None),
            cast(Position.close_exchange_order_id, String).contains(close_order_id),
        )
    )
    positions = [
        position
        for position in position_result.scalars().all()
        if close_order_id
        in {
            item.strip()
            for item in str(getattr(position, "close_exchange_order_id", "") or "").split(",")
            if item.strip()
        }
        and str(getattr(position, "settlement_status", "") or "").strip()
        != SUPERSEDED_POSITION_STATUS
        and str(_safe_dict(getattr(position, "settlement_raw", None)).get("reason") or "").strip()
        not in SUPERSEDED_POSITION_REASONS
    ]
    exact_positions = [
        position
        for position in positions
        if str(getattr(position, "close_exchange_order_id", "") or "").strip()
        == close_order_id
    ]
    if len(exact_positions) > 1:
        raise ExitDecisionLineageAmbiguous(
            f"Multiple closed positions claim OKX close order {close_order_id}"
        )
    if not exact_positions and len(positions) > 1:
        raise ExitDecisionLineageAmbiguous(
            "Multiple closed positions contain OKX close order "
            f"{close_order_id} without one exact lifecycle slice"
        )
    settlement_positions = exact_positions or positions
    realized_pnl = (
        sum(float(getattr(position, "realized_pnl", 0.0) or 0.0) for position in settlement_positions)
        if settlement_positions
        else float(
            raw_fill.get("realized_pnl")
            or raw_fill.get("pnl")
            or raw_fill.get("fill_pnl")
            or 0.0
        )
    )
    entry_notional = (
        sum(
            abs(float(getattr(position, "entry_price", 0.0) or 0.0))
            * abs(float(getattr(position, "quantity", 0.0) or 0.0))
            for position in settlement_positions
        )
        or float(raw_fill.get("entry_notional") or 0.0)
    )
    closed_at = next(
        (
            position.closed_at
            for position in settlement_positions
            if isinstance(getattr(position, "closed_at", None), datetime)
        ),
        getattr(order, "filled_at", None) or now or datetime.now(UTC),
    )
    exit_price = float(
        raw_fill.get("avg_price")
        or raw_fill.get("price")
        or getattr(order, "price", 0.0)
        or 0.0
    )
    if exit_price <= 0:
        return None
    return apply_exit_decision_lineage(
        resolution,
        close_order_id=close_order_id,
        close_fill=raw_fill,
        reconcile_origin="external_okx_sync",
        exit_price=exit_price,
        realized_pnl=realized_pnl,
        closed_at=closed_at,
        entry_notional=entry_notional,
    )


def apply_exit_decision_lineage(
    resolution: ExitDecisionLineageResolution,
    *,
    close_order_id: str,
    close_fill: dict[str, Any],
    reconcile_origin: str,
    exit_price: float,
    realized_pnl: float,
    closed_at: datetime,
    entry_notional: float,
) -> dict[str, Any] | None:
    """Complete the original decision and retire duplicate reconciliation rows."""

    decision = resolution.authoritative
    if decision is None:
        return None
    raw = dict(_safe_dict(decision.raw_llm_response))
    original_system_sync = raw.get("system_sync") is True
    authoritative_origin = (
        reconcile_origin if original_system_sync else RECONCILE_ORIGIN_SYSTEM_EXECUTION
    )
    execution = dict(_safe_dict(raw.get("execution_result")))
    execution.update(
        {
            "source": "okx_authoritative_reconcile",
            "order_id": close_order_id,
            "exchange_order_id": close_order_id,
            "status": "filled",
            "price": exit_price,
            "pnl": realized_pnl,
            "exchange_confirmed": True,
        }
    )
    raw["execution_result"] = execution
    raw["authoritative_close_reconciliation"] = {
        "version": EXIT_LINEAGE_VERSION,
        "source": "okx_fills_history_exact_order_id",
        "close_order_id": close_order_id,
        "reconcile_origin": authoritative_origin,
        "close_fill": dict(close_fill),
        "superseded_decision_ids": [int(row.id) for row in resolution.superseded],
    }
    if not original_system_sync:
        raw["reconcile_origin"] = RECONCILE_ORIGIN_SYSTEM_EXECUTION
    decision.raw_llm_response = raw
    decision.was_executed = True
    decision.execution_reason = None
    decision.executed_at = closed_at
    decision.execution_price = exit_price
    decision.outcome = "profit" if realized_pnl > 0 else "loss" if realized_pnl < 0 else "flat"
    decision.outcome_pnl_pct = realized_pnl / entry_notional * 100 if entry_notional > 0 else 0.0

    if resolution.linked_order is not None:
        resolution.linked_order.decision_id = int(decision.id)
    for duplicate in resolution.superseded:
        duplicate_raw = dict(_safe_dict(duplicate.raw_llm_response))
        duplicate_raw["reconciliation_superseded"] = {
            "version": EXIT_LINEAGE_VERSION,
            "source": "exact_okx_close_order_identity",
            "close_order_id": close_order_id,
            "authoritative_decision_id": int(decision.id),
            "previous_execution": {
                "executed_at": duplicate.executed_at.isoformat()
                if isinstance(duplicate.executed_at, datetime)
                else duplicate.executed_at,
                "execution_price": duplicate.execution_price,
                "outcome": duplicate.outcome,
                "outcome_pnl_pct": duplicate.outcome_pnl_pct,
            },
        }
        duplicate.raw_llm_response = duplicate_raw
        duplicate.was_executed = False
        duplicate.execution_reason = (
            f"Superseded by authoritative exit decision {decision.id} for OKX order "
            f"{close_order_id}."
        )
        duplicate.executed_at = None
        duplicate.execution_price = None
        duplicate.outcome = None
        duplicate.outcome_pnl_pct = None
    return {
        "version": EXIT_LINEAGE_VERSION,
        "authoritative_decision_id": int(decision.id),
        "reused_original_decision": not original_system_sync,
        "superseded_decision_ids": [int(row.id) for row in resolution.superseded],
        "close_order_id": close_order_id,
    }
