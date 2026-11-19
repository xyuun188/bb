"""Resolve one authoritative decision for an OKX-reconciled exit fill."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import String, cast, select

from models.decision import AIDecision
from models.trade import Order, Position
from services.decision_state import DecisionStage, DecisionStageStatus, append_decision_stage
from services.position_settlement import is_final_settlement_status

RECONCILE_ORIGIN_SYSTEM_EXECUTION = "system_execution"
EXIT_LINEAGE_VERSION = "2026-11-19.okx-exit-decision-lineage.v3"
EXIT_ACTIONS = {"close_long", "close_short", "exit_long", "exit_short"}
UNLINKED_ORDER_MODEL = "okx_authoritative_sync"
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _relative_close_enough(left: float, right: float, tolerance_ratio: float) -> bool:
    tolerance = max(abs(left), abs(right), 1e-12) * max(tolerance_ratio, 0.0)
    return abs(left - right) <= tolerance


def _close_fill_target_quantity(
    raw_fill: dict[str, Any],
    order: Any,
) -> float:
    """Return the authoritative base quantity available for close aggregation."""

    for key in ("base_quantity", "filled_base_quantity"):
        value = _safe_float(raw_fill.get(key), 0.0)
        if value > 0:
            return value
    contracts = _safe_float(
        raw_fill.get("contracts") or raw_fill.get("filled_contracts"),
        0.0,
    )
    contract_size = _safe_float(
        raw_fill.get("contract_size") or raw_fill.get("contractSize"),
        0.0,
    )
    if contracts > 0 and contract_size > 0:
        return contracts * contract_size
    if contracts > 0:
        return contracts
    return _safe_float(getattr(order, "quantity", None), 0.0)


def _close_position_lineage_key(position: Any) -> tuple[str, ...]:
    """Identify a local lifecycle projection without using the close order id."""

    return (
        str(getattr(position, "execution_mode", "") or "").strip().lower(),
        str(getattr(position, "symbol", "") or "").strip().upper(),
        str(getattr(position, "side", "") or "").strip().lower(),
        str(getattr(position, "okx_pos_id", "") or "").strip(),
        str(getattr(position, "entry_exchange_order_id", "") or "").strip(),
        f"{_safe_float(getattr(position, 'entry_price', None), 0.0):.12g}",
    )


def _can_aggregate_close_positions(
    positions: list[Any],
    *,
    target_quantity: float,
) -> bool:
    """Allow one close fill to settle distinct local fragments only when exact.

    A single exchange close order can cover several local fragments.  It is safe
    to aggregate only when every fragment has a distinct lifecycle projection
    and the authoritative fill quantity covers their sum.  Duplicate rows or
    quantity mismatches remain ambiguous and therefore stay fail-closed.
    """

    if len(positions) <= 1 or target_quantity <= 0:
        return len(positions) <= 1
    lineage_keys = [_close_position_lineage_key(position) for position in positions]
    if any(not key[3] and not key[4] for key in lineage_keys):
        return False
    if len(set(lineage_keys)) != len(lineage_keys):
        return False
    quantities = [
        _safe_float(getattr(position, "quantity", None), 0.0)
        for position in positions
    ]
    if any(quantity <= 0 for quantity in quantities):
        return False
    return _relative_close_enough(sum(quantities), target_quantity, 0.02)


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


def _observes_existing_exit_order(decision: Any) -> bool:
    raw = _safe_dict(getattr(decision, "raw_llm_response", None))
    execution_raw = _safe_dict(_safe_dict(raw.get("execution_result")).get("raw_response"))
    return execution_raw.get("exit_tracking") is True and (
        execution_raw.get("existing_exit_order") is True
        or execution_raw.get("exit_replace_attempted") is True
    )


def decision_exit_exchange_order_ids(decision: Any) -> set[str]:
    """Read only execution/close paths; entry lineage cannot claim an exit fill."""

    raw = _safe_dict(getattr(decision, "raw_llm_response", None))
    execution = _safe_dict(raw.get("execution_result"))
    execution_raw = _safe_dict(execution.get("raw_response"))
    execution_info = _safe_dict(execution_raw.get("info"))
    close_ids = set().union(
        _nested_order_ids(raw.get("native_close_fill")),
        _nested_order_ids(raw.get("close_fill")),
        _nested_order_ids(execution.get("native_close_fill")),
    )
    # Inspecting another order (including a triggered OCO) does not submit it.
    # Keep explicit reconciled close facts, but never promote an observer's
    # copied order detail into ownership of the exchange execution.
    if _observes_existing_exit_order(decision):
        return close_ids
    return close_ids.union(
        _nested_order_ids(execution),
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
    if linked_decision is not None and (
        not _observes_existing_exit_order(linked_decision)
        or order_id in decision_exit_exchange_order_ids(linked_decision)
    ):
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
    if (
        authoritative is None
        and linked_decision is not None
        and int(linked_decision.id) in by_id
    ):
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


def _ioc_close_fill_receipt(decision: AIDecision, order: Order) -> dict[str, Any]:
    stored = _safe_dict(order.okx_raw_fills)
    execution = _safe_dict(_safe_dict(decision.raw_llm_response).get("execution_result"))
    raw = _safe_dict(execution.get("raw_response"))
    request = _safe_dict(raw.get("request_params"))
    order_id = str(order.exchange_order_id or "")
    position_side = "long" if order.side == "sell" else "short"
    if (
        stored.get("fills_history_confirmed") is True
        and stored.get("fills_pagination_complete") is True
        and stored.get("contract_size_verified") is True
        and stored.get("trade_ids")
        and stored.get("order_id") == order_id
        and stored.get("inst_id") == order.okx_inst_id == request.get("instId")
        and stored.get("pos_side") in {"net", position_side}
        and request.get("posSide", "net") == stored.get("pos_side")
        and order.side == request.get("side")
        and _safe_float(order.quantity) > 0
        and _relative_close_enough(
            _safe_float(stored.get("base_quantity")), _safe_float(order.quantity), 1e-8,
        )
        and 0 < _safe_float(stored.get("contracts")) <= _safe_float(request.get("sz"))
        and execution.get("exchange_order_id") == order_id == raw.get("submitted_order_id")
        and request.get("ordType") == "ioc"
        and str(request.get("reduceOnly")).lower() == "true"
        and not _observes_existing_exit_order(decision)
    ):
        return request
    return {}


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

    orphan = (
        linked_order is not None
        and linked_order.decision_id is None
        and model_name == UNLINKED_ORDER_MODEL
    )
    query = select(AIDecision).where(
        AIDecision.symbol == symbol,
        AIDecision.action == action,
        AIDecision.is_paper.is_(is_paper),
        AIDecision.raw_llm_response.is_not(None),
        cast(AIDecision.raw_llm_response, String).contains(order_id),
    )
    if orphan:
        stored = _safe_dict(linked_order.okx_raw_fills)
        filled_at = linked_order.filled_at
        if not (
            isinstance(filled_at, datetime)
            and stored.get("fills_history_confirmed") is True
            and stored.get("fills_pagination_complete") is True
            and stored.get("contract_size_verified") is True
        ):
            return ExitDecisionLineageResolution(None, linked_order, (), ())
        query = query.where(
            AIDecision.created_at >= filled_at - timedelta(hours=1),
            AIDecision.created_at <= filled_at + timedelta(minutes=1),
        )
    else:
        query = query.where(AIDecision.model_name == model_name)
    decisions_result = await session.execute(query)
    decisions = list(decisions_result.scalars().all())
    if orphan:
        decisions = [
            decision for decision in decisions
            if _ioc_close_fill_receipt(decision, linked_order)
        ]
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
    """Repair an authoritative close order with a missing or stale decision link.

    The exchange order id is the join key. The time bound only limits the
    search for an unlinked IOC receipt.
    """

    close_order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
    if not close_order_id:
        return None
    linked_decision_id = int(getattr(order, "decision_id", 0) or 0)
    side = str(getattr(order, "side", "") or "").strip().lower()
    if side not in {"buy", "sell"}:
        return None
    expected_action = "close_short" if side == "buy" else "close_long"
    model_name = str(getattr(order, "model_name", "") or "")
    if linked_decision_id:
        linked_decision = await session.get(AIDecision, linked_decision_id)
        if getattr(linked_decision, "action", None) != expected_action:
            return None
    elif model_name != UNLINKED_ORDER_MODEL:
        return None
    resolution = await load_exit_decision_lineage(
        session,
        model_name=model_name,
        symbol=str(getattr(order, "symbol", "") or ""),
        action=expected_action,
        is_paper=str(getattr(order, "execution_mode", "") or "").lower() != "live",
        execution_mode=str(getattr(order, "execution_mode", "") or "paper"),
        close_order_id=close_order_id,
    )
    authoritative = resolution.authoritative
    if authoritative is None:
        return None
    if (
        linked_decision_id == int(authoritative.id)
        and not resolution.superseded
        and (
            authoritative.was_executed
            or not _ioc_close_fill_receipt(authoritative, order)
        )
    ):
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
    target_quantity = _close_fill_target_quantity(raw_fill, order)
    candidate_positions = exact_positions or positions
    if len(candidate_positions) > 1 and not _can_aggregate_close_positions(
        candidate_positions,
        target_quantity=target_quantity,
    ):
        qualifier = "claim" if exact_positions else "contain"
        raise ExitDecisionLineageAmbiguous(
            f"aggregate_mismatch target={round(target_quantity, 8)} "
            f"qty={[round(_safe_float(getattr(item, 'quantity', None), 0.0), 8) for item in candidate_positions]} "
            f"order={close_order_id} ids={[int(getattr(item, 'id', 0) or 0) for item in candidate_positions]} "
            f"kind={qualifier}"
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
        settlement_ready=bool(settlement_positions) and all(
            is_final_settlement_status(position.settlement_status)
            for position in settlement_positions
        ),
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
    settlement_ready: bool = True,
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
    order = resolution.linked_order
    ioc_request = (
        _ioc_close_fill_receipt(decision, order)
        if order is not None and not decision.was_executed
        else {}
    )
    partial = bool(ioc_request) and not _relative_close_enough(
        _safe_float(order.okx_fill_contracts), _safe_float(ioc_request.get("sz")), 1e-9,
    )
    execution.update(
        {
            "source": "okx_authoritative_reconcile",
            "order_id": close_order_id,
            "exchange_order_id": close_order_id,
            "status": "partial" if partial else "filled",
            "price": exit_price,
            "pnl": realized_pnl,
            "exchange_confirmed": True,
        }
    )
    if ioc_request:
        evidence = _safe_dict(order.okx_raw_fills)
        execution.update(
            quantity=order.quantity,
            fee=order.fee,
            exit_progress=True,
        )
        raw = append_decision_stage(
            raw,
            DecisionStage.EXCHANGE_CONFIRM,
            DecisionStageStatus.COMPLETED,
            "OKX 已按原订单号确认部分平仓。" if partial else "OKX 已按原订单号确认平仓成交。",
            {"exchange_order_id": close_order_id, "quantity": order.quantity},
        )
        raw = append_decision_stage(
            raw,
            DecisionStage.LOCAL_SYNC,
            DecisionStageStatus.COMPLETED,
            "已补回成交对应的原始平仓决策。",
            {"exchange_order_id": close_order_id},
        )
        order.model_name = decision.model_name
        if partial:
            order.status = "partial"
            details = _safe_dict(_safe_dict(execution.get("raw_response")).get("order_detail"))
            native = _safe_dict(details.get("info"))
            if (
                native.get("ordId") == close_order_id
                and native.get("instId") == order.okx_inst_id
                and native.get("side") == order.side
                and native.get("state") == "canceled"
                and _relative_close_enough(
                    _safe_float(native.get("accFillSz")), _safe_float(order.okx_fill_contracts), 1e-9,
                )
            ):
                order.okx_state = "canceled"
            elif order.okx_state == "filled":
                order.okx_state = "partially_filled"
        close_fill = {**evidence, **close_fill}
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
    if ioc_request and not settlement_ready:
        decision.outcome = None
        decision.outcome_pnl_pct = None
    else:
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
