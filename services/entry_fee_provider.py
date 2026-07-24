from __future__ import annotations

from typing import Any

from sqlalchemy import select

from executor.base_executor import OrderStatus
from models.trade import Order
from services.current_position_management import (
    current_position_management_contract_complete,
)


def proportional_fee(fee: float | None, close_qty: float, total_qty: float) -> float:
    try:
        fee_value = abs(float(fee or 0.0))
        close_value = float(close_qty or 0.0)
        total_value = float(total_qty or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if fee_value <= 0 or close_value <= 0:
        return 0.0
    if total_value <= 0:
        return fee_value
    return fee_value * min(close_value / total_value, 1.0)


class EntryFeeProvider:
    """Read exact linked entry-fill fees for a closing position."""

    @staticmethod
    def proportional_fee(fee: float | None, close_qty: float, total_qty: float) -> float:
        return proportional_fee(fee, close_qty, total_qty)

    async def entry_fee_for_position(self, session: Any, position: Any, close_qty: float) -> float:
        entry_order_ids = _split_exchange_order_ids(
            getattr(position, "entry_exchange_order_id", None)
        )
        orders: list[Any] = []
        for order_id in entry_order_ids:
            order = await self._first_order(
                session,
                select(Order)
                .where(
                    Order.execution_mode == position.execution_mode,
                    Order.exchange_order_id == order_id,
                    Order.status == OrderStatus.FILLED.value,
                )
                .limit(1),
            )
            if order is None or not _has_authoritative_fee_fact(order, order_id=order_id):
                orders = []
                break
            orders.append(order)
        if orders and len(orders) == len(entry_order_ids):
            total_fee = sum(
                abs(float(getattr(order, "okx_raw_fills", {}).get("fee_abs") or 0.0))
                for order in orders
            )
            total_quantity = sum(abs(float(getattr(order, "quantity", 0.0) or 0.0)) for order in orders)
            return proportional_fee(total_fee, close_qty, total_quantity)

        if current_position_management_contract_complete(position):
            return proportional_fee(
                getattr(position, "entry_fee", None),
                close_qty,
                getattr(position, "quantity", None),
            )
        return 0.0

    @staticmethod
    async def _first_order(session: Any, statement: Any) -> Any | None:
        result = await session.execute(statement)
        return result.scalar_one_or_none()


def _split_exchange_order_ids(value: Any) -> list[str]:
    tokens = {str(value or "").strip()}
    for separator in (",", ";", "|", "\n", "\t", " "):
        tokens = {
            part.strip()
            for token in tokens
            for part in token.split(separator)
            if part.strip()
        }
    return sorted(token for token in tokens if token)


def _has_authoritative_fee_fact(order: Any, *, order_id: str) -> bool:
    raw = getattr(order, "okx_raw_fills", None)
    if not isinstance(raw, dict):
        return False
    exchange_confirmation = bool(
        raw.get("fills_history_confirmed") is True
        or (
            raw.get("order_detail_confirmed") is True
            and str(raw.get("source") or "").strip() == "okx_order_detail"
            and raw.get("contract_size_verified") is True
            and str(raw.get("contract_size_source") or "").strip()
            == "okx_public_instruments"
        )
        or (
            raw.get("execution_result_confirmed") is True
            and raw.get("contract_size_verified") is True
            and str(raw.get("contract_size_source") or "").strip()
            == "okx_public_instruments"
        )
    )
    return bool(
        exchange_confirmation
        and str(raw.get("order_id") or "").strip() == order_id
        and raw.get("fee_abs") is not None
        and abs(float(getattr(order, "quantity", 0.0) or 0.0)) > 0
    )
