from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import select

from models.decision import AIDecision
from models.trade import Order
from services.trade_execution_contract import validate_production_entry_contract


def _default_float_parser(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class PositionProtectionFallbackPolicy:
    """Recover only an exact order's governed dynamic stop plan."""

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

        decision = await self._find_decision(session, order=order)
        if decision is None:
            return {}

        raw = getattr(decision, "raw_llm_response", None)
        raw = raw if isinstance(raw, dict) else {}
        sizing = raw.get("profit_risk_sizing")
        sizing = sizing if isinstance(sizing, dict) else {}
        provenance = sizing.get("policy_provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        _, contract_blockers = validate_production_entry_contract(raw)
        stop_loss_pct = self.float_parser(sizing.get("stressed_loss_fraction"), 0.0)
        if contract_blockers or stop_loss_pct <= 0:
            return {}

        stop_loss = self._price_from_pct(
            entry_price=entry_price,
            side=side,
            pct=stop_loss_pct,
            kind="stop_loss",
        )
        return {
            "stop_loss_price": stop_loss if stop_loss > 0 else None,
            "take_profit_price": None,
            "source": "exact_order_dynamic_risk_plan",
            "decision_id": getattr(decision, "id", None),
            "stop_loss_pct": stop_loss_pct,
            "policy_provenance": provenance,
        }

    async def _find_decision(
        self,
        session: Any,
        *,
        order: Order | None,
    ) -> AIDecision | None:
        if order is None or not getattr(order, "decision_id", None):
            return None
        result = await session.execute(
            select(AIDecision).where(AIDecision.id == order.decision_id).limit(1)
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
