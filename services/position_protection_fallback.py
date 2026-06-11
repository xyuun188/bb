from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import select

from ai_brain.base_model import Action
from models.decision import AIDecision
from models.trade import Order


def _default_float_parser(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class PositionProtectionFallbackPolicy:
    """Recover TP/SL prices from the executed entry decision when OKX omits them."""

    def __init__(
        self,
        float_parser: Callable[[Any, float], float] | None = None,
    ) -> None:
        self.float_parser = float_parser or _default_float_parser

    async def protection_from_decision(
        self,
        session: Any,
        *,
        symbol: str,
        side: str,
        entry_price: float,
        order: Order | None = None,
    ) -> dict[str, Any]:
        if entry_price <= 0 or side not in {"long", "short"}:
            return {}

        decision = await self._find_decision(session, symbol=symbol, side=side, order=order)
        if decision is None:
            return {}

        stop_loss_pct = self.float_parser(getattr(decision, "stop_loss_pct", 0.0), 0.0)
        take_profit_pct = self.float_parser(getattr(decision, "take_profit_pct", 0.0), 0.0)
        if stop_loss_pct <= 0 and take_profit_pct <= 0:
            return {}

        stop_loss = self._price_from_pct(
            entry_price=entry_price,
            side=side,
            pct=stop_loss_pct,
            kind="stop_loss",
        )
        take_profit = self._price_from_pct(
            entry_price=entry_price,
            side=side,
            pct=take_profit_pct,
            kind="take_profit",
        )

        return {
            "stop_loss_price": stop_loss if stop_loss > 0 else None,
            "take_profit_price": take_profit if take_profit > 0 else None,
            "source": "latest_executed_entry_decision",
            "decision_id": getattr(decision, "id", None),
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }

    async def _find_decision(
        self,
        session: Any,
        *,
        symbol: str,
        side: str,
        order: Order | None,
    ) -> AIDecision | None:
        if order is not None and getattr(order, "decision_id", None):
            result = await session.execute(
                select(AIDecision).where(AIDecision.id == order.decision_id).limit(1)
            )
            decision = result.scalar_one_or_none()
            if decision is not None:
                return decision

        action_value = Action.LONG.value if side == "long" else Action.SHORT.value
        result = await session.execute(
            select(AIDecision)
            .where(
                AIDecision.symbol == symbol,
                AIDecision.action == action_value,
                AIDecision.was_executed.is_(True),
            )
            .order_by(AIDecision.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _price_from_pct(
        *,
        entry_price: float,
        side: str,
        pct: float,
        kind: str,
    ) -> float:
        if pct <= 0:
            return 0.0
        if side == "long":
            return entry_price * (1 - pct) if kind == "stop_loss" else entry_price * (1 + pct)
        return entry_price * (1 + pct) if kind == "stop_loss" else entry_price * (1 - pct)
