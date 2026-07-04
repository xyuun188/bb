"""Reconcile filled OKX order pairs back into local position history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from ai_brain.base_model import Action, DecisionOutput
from core.symbols import (
    normalize_trading_symbol,
    okx_inst_id_from_payload,
    okx_inst_id_from_symbol,
    symbol_from_okx_inst_id,
    trading_symbol_variants,
)
from models.decision import AIDecision
from models.trade import Order, Position
from services.position_settlement import (
    build_position_settlement_snapshot,
    settlement_payload_fields,
)

FILLED_STATUS = "filled"
ORDER_MATCH_WINDOW = timedelta(minutes=10)
PRICE_TOLERANCE_RATIO = 0.002
QUANTITY_TOLERANCE_RATIO = 0.02
POSITION_COVERAGE_DUPLICATE_RATIO = 0.80


@dataclass(frozen=True, slots=True)
class MissingClosedPositionPlan:
    """A deterministic plan to create one missing closed local position."""

    model_name: str
    execution_mode: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    leverage: float
    stop_loss_price: float | None
    take_profit_price: float | None
    realized_pnl: float
    gross_pnl: float
    entry_fee_allocated: float
    close_fee_allocated: float
    created_at: datetime
    closed_at: datetime
    entry_order_id: int
    close_order_id: int
    okx_inst_id: str | None
    okx_pos_id: str | None
    entry_exchange_order_id: str | None
    close_exchange_order_id: str | None


@dataclass(frozen=True, slots=True)
class ReconciledClosedPosition:
    """Result of applying a missing-position repair plan."""

    position: Position
    plan: MissingClosedPositionPlan


async def plan_missing_closed_position(
    session: Any,
    close_order: Order,
) -> MissingClosedPositionPlan | None:
    """Build a repair plan when a filled close order lacks a local position row."""

    close_quantity = _safe_float(close_order.quantity)
    close_price = _safe_float(close_order.price)
    if str(close_order.status or "").lower() != FILLED_STATUS:
        return None
    if close_quantity <= 0 or close_price <= 0:
        return None

    close_decision = await _decision_for_order(session, close_order)
    close_action = _decision_action(close_decision)
    if close_action not in {Action.CLOSE_LONG, Action.CLOSE_SHORT}:
        return None

    side = "long" if close_action == Action.CLOSE_LONG else "short"
    close_side = _close_order_side(side)
    if str(close_order.side or "").lower() != close_side:
        return None

    entry_order = await _matching_entry_order(session, close_order, side)
    if entry_order is None:
        return None

    entry_price = _safe_float(entry_order.price)
    if entry_price <= 0:
        return None
    created_at = _order_time(entry_order)
    closed_at = _order_time(close_order)
    if created_at is None or closed_at is None:
        return None
    if created_at > closed_at:
        return None

    entry_decision = await _decision_for_order(session, entry_order)
    entry_okx_inst_id = _order_okx_inst_id(entry_order, entry_decision)
    close_okx_inst_id = _order_okx_inst_id(close_order, close_decision)
    okx_inst_id = close_okx_inst_id or entry_okx_inst_id
    if entry_okx_inst_id and close_okx_inst_id and entry_okx_inst_id != close_okx_inst_id:
        return None

    symbol = (
        symbol_from_okx_inst_id(okx_inst_id)
        or normalize_trading_symbol(close_order.symbol or entry_order.symbol)
    )
    quantity = min(close_quantity, _safe_float(entry_order.quantity))
    if quantity <= 0:
        return None

    duplicate = await _matching_closed_position_exists(
        session,
        model_name=str(close_order.model_name or entry_order.model_name or ""),
        execution_mode=str(close_order.execution_mode or entry_order.execution_mode or ""),
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        exit_price=close_price,
        created_at=created_at,
        closed_at=closed_at,
    )
    if duplicate:
        return None

    leverage = _safe_float(getattr(entry_decision, "suggested_leverage", None), 1.0) or 1.0
    stop_loss_pct = _safe_float(getattr(entry_decision, "stop_loss_pct", None), 0.0)
    take_profit_pct = _safe_float(getattr(entry_decision, "take_profit_pct", None), 0.0)
    stop_loss_price, take_profit_price = _protection_prices(
        side=side,
        entry_price=entry_price,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    entry_fee = _allocated_fee(entry_order, quantity)
    close_fee = _allocated_fee(close_order, quantity)
    gross_pnl = _gross_pnl(side, entry_price, close_price, quantity)
    realized_pnl = gross_pnl - entry_fee - close_fee

    return MissingClosedPositionPlan(
        model_name=str(close_order.model_name or entry_order.model_name or "ensemble_trader"),
        execution_mode=str(close_order.execution_mode or entry_order.execution_mode or "paper"),
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        exit_price=close_price,
        leverage=leverage,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        realized_pnl=realized_pnl,
        gross_pnl=gross_pnl,
        entry_fee_allocated=entry_fee,
        close_fee_allocated=close_fee,
        created_at=created_at,
        closed_at=closed_at,
        entry_order_id=int(entry_order.id),
        close_order_id=int(close_order.id),
        okx_inst_id=okx_inst_id or okx_inst_id_from_symbol(symbol) or None,
        okx_pos_id=None,
        entry_exchange_order_id=entry_order.exchange_order_id,
        close_exchange_order_id=close_order.exchange_order_id,
    )


async def apply_missing_closed_position_plan(
    session: Any,
    plan: MissingClosedPositionPlan,
) -> Position:
    """Create the missing closed position described by the plan."""

    settlement = build_position_settlement_snapshot(
        close_fill_pnl=plan.gross_pnl,
        entry_fee=plan.entry_fee_allocated,
        close_fee=plan.close_fee_allocated,
        funding_fee=0.0,
        status="provisional",
        source="missing_closed_position_repair",
        synced_at=plan.closed_at,
        raw={
            "entry_order_id": plan.entry_order_id,
            "close_order_id": plan.close_order_id,
            "entry_exchange_order_id": plan.entry_exchange_order_id,
            "close_exchange_order_id": plan.close_exchange_order_id,
            "funding_fee_source": "not_available_from_order_pair",
        },
    )
    position = Position(
        model_name=plan.model_name,
        execution_mode=plan.execution_mode,
        symbol=plan.symbol,
        side=plan.side,
        quantity=plan.quantity,
        entry_price=plan.entry_price,
        current_price=plan.exit_price,
        leverage=plan.leverage,
        unrealized_pnl=0.0,
        **settlement_payload_fields(settlement),
        stop_loss_price=plan.stop_loss_price,
        take_profit_price=plan.take_profit_price,
        is_open=False,
        closed_at=plan.closed_at,
        created_at=plan.created_at,
        okx_inst_id=plan.okx_inst_id,
        okx_pos_id=plan.okx_pos_id,
        entry_exchange_order_id=plan.entry_exchange_order_id,
        close_exchange_order_id=plan.close_exchange_order_id,
    )
    session.add(position)
    await session.flush()
    return position


async def reconcile_missing_closed_position_for_exit(
    session: Any,
    *,
    model_name: str,
    execution_mode: str,
    decision: DecisionOutput,
    result: Any,
) -> ReconciledClosedPosition | None:
    """Recover one missing closed position after an exchange-confirmed exit."""

    if not decision.is_exit:
        return None
    close_order = await _close_order_for_execution(
        session,
        model_name=model_name,
        execution_mode=execution_mode,
        result=result,
        decision=decision,
    )
    if close_order is None:
        return None
    plan = await plan_missing_closed_position(session, close_order)
    if plan is None:
        return None
    position = await apply_missing_closed_position_plan(session, plan)
    return ReconciledClosedPosition(position=position, plan=plan)


async def _close_order_for_execution(
    session: Any,
    *,
    model_name: str,
    execution_mode: str,
    result: Any,
    decision: DecisionOutput,
) -> Order | None:
    exchange_order_id = str(getattr(result, "exchange_order_id", None) or "").strip()
    if exchange_order_id:
        row = await session.execute(
            select(Order)
            .where(
                Order.execution_mode == execution_mode,
                Order.exchange_order_id == exchange_order_id,
            )
            .limit(1)
        )
        order = row.scalar_one_or_none()
        if order is not None:
            return order

    side = _close_order_side("long" if decision.action == Action.CLOSE_LONG else "short")
    result_time = _ensure_aware(getattr(result, "timestamp", None))
    symbol_variants = trading_symbol_variants(getattr(result, "symbol", ""))
    stmt = select(Order).where(
        Order.model_name == model_name,
        Order.execution_mode == execution_mode,
        Order.symbol.in_(symbol_variants),
        Order.side == side,
        Order.status == FILLED_STATUS,
    )
    if result_time is not None:
        stmt = stmt.where(
            Order.filled_at >= result_time - ORDER_MATCH_WINDOW,
            Order.filled_at <= result_time + ORDER_MATCH_WINDOW,
        )
    rows = await session.execute(stmt.order_by(Order.filled_at.desc(), Order.created_at.desc()))
    for order in rows.scalars().all():
        if _quantities_close(
            _safe_float(order.quantity), _safe_float(getattr(result, "quantity", 0.0))
        ):
            return order
    return None


async def _matching_entry_order(session: Any, close_order: Order, side: str) -> Order | None:
    entry_action = Action.LONG if side == "long" else Action.SHORT
    entry_side = "buy" if side == "long" else "sell"
    symbol_variants = trading_symbol_variants(close_order.symbol)
    close_time = _order_time(close_order)
    if close_time is None:
        return None
    close_quantity = _safe_float(close_order.quantity)

    rows = await session.execute(
        select(Order)
        .where(
            Order.model_name == close_order.model_name,
            Order.execution_mode == close_order.execution_mode,
            Order.symbol.in_(symbol_variants),
            Order.side == entry_side,
            Order.status == FILLED_STATUS,
            Order.exchange_order_id.is_not(None),
            Order.exchange_order_id != "",
            Order.filled_at <= close_time,
        )
        .order_by(Order.filled_at.desc(), Order.created_at.desc())
        .limit(20)
    )
    fallback: Order | None = None
    for order in rows.scalars().all():
        if not _quantity_covers(_safe_float(order.quantity), close_quantity):
            continue
        decision = await _decision_for_order(session, order)
        if _decision_action(decision) == entry_action:
            return order
        if decision is None and fallback is None:
            fallback = order
    return fallback


async def _matching_closed_position_exists(
    session: Any,
    *,
    model_name: str,
    execution_mode: str,
    symbol: str,
    side: str,
    quantity: float,
    entry_price: float,
    exit_price: float,
    created_at: datetime,
    closed_at: datetime,
) -> bool:
    symbol_variants = trading_symbol_variants(symbol)
    rows = await session.execute(
        select(Position).where(
            Position.model_name == model_name,
            Position.execution_mode == execution_mode,
            Position.symbol.in_(symbol_variants),
            Position.side == side,
            Position.is_open.is_(False),
            Position.closed_at >= closed_at - ORDER_MATCH_WINDOW,
            Position.closed_at <= closed_at + ORDER_MATCH_WINDOW,
        )
    )
    price_matched_positions = [
        position
        for position in rows.scalars().all()
        if _prices_close(_safe_float(position.current_price), exit_price)
    ]
    if any(
        _quantities_close(_safe_float(position.quantity), quantity)
        for position in price_matched_positions
    ):
        return True
    covered_quantity = sum(
        max(_safe_float(position.quantity), 0.0) for position in price_matched_positions
    )
    return covered_quantity >= quantity * POSITION_COVERAGE_DUPLICATE_RATIO


async def _decision_for_order(session: Any, order: Order) -> AIDecision | None:
    decision_id = getattr(order, "decision_id", None)
    if not decision_id:
        return None
    return await session.get(AIDecision, int(decision_id))


def _order_okx_inst_id(order: Order, decision: AIDecision | None) -> str:
    payloads: list[dict[str, Any]] = []
    raw = getattr(decision, "raw_llm_response", None) if decision is not None else None
    if isinstance(raw, dict):
        payloads.append(raw)
        execution_result = raw.get("execution_result")
        if isinstance(execution_result, dict):
            payloads.append(execution_result)
    for payload in payloads:
        inst_id = okx_inst_id_from_payload(payload, include_fallback=False)
        if inst_id:
            return inst_id
    return okx_inst_id_from_symbol(getattr(order, "symbol", None))


def _decision_action(decision: AIDecision | None) -> Action | None:
    if decision is None:
        return None
    try:
        return Action.from_string(str(decision.action or ""))
    except Exception:
        return None


def _close_order_side(side: str) -> str:
    return "sell" if side == "long" else "buy"


def _order_time(order: Order) -> datetime | None:
    return _ensure_aware(order.filled_at or order.created_at)


def _ensure_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _quantity_covers(entry_quantity: float, close_quantity: float) -> bool:
    if entry_quantity <= 0 or close_quantity <= 0:
        return False
    return (
        entry_quantity + max(entry_quantity, close_quantity) * QUANTITY_TOLERANCE_RATIO
        >= close_quantity
    )


def _quantities_close(left: float, right: float) -> bool:
    tolerance = max(abs(left), abs(right), 1.0) * QUANTITY_TOLERANCE_RATIO
    return abs(left - right) <= tolerance


def _prices_close(left: float, right: float) -> bool:
    tolerance = max(abs(left), abs(right), 1e-9) * PRICE_TOLERANCE_RATIO
    return abs(left - right) <= tolerance


def _allocated_fee(order: Order, quantity: float) -> float:
    order_quantity = _safe_float(order.quantity)
    if order_quantity <= 0:
        return 0.0
    return _safe_float(order.fee) * min(max(quantity / order_quantity, 0.0), 1.0)


def _gross_pnl(side: str, entry_price: float, exit_price: float, quantity: float) -> float:
    if side == "short":
        return (entry_price - exit_price) * quantity
    return (exit_price - entry_price) * quantity


def _protection_prices(
    *,
    side: str,
    entry_price: float,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> tuple[float | None, float | None]:
    stop_loss_price = None
    take_profit_price = None
    if stop_loss_pct > 0:
        stop_loss_price = (
            entry_price * (1 - stop_loss_pct)
            if side == "long"
            else entry_price * (1 + stop_loss_pct)
        )
    if take_profit_pct > 0:
        take_profit_price = (
            entry_price * (1 + take_profit_pct)
            if side == "long"
            else entry_price * (1 - take_profit_pct)
        )
    return stop_loss_price, take_profit_price
