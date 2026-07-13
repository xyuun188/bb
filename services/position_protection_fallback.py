from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import select

from models.decision import AIDecision
from models.trade import Order


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
        return_policy = raw.get("production_return_policy")
        return_policy = return_policy if isinstance(return_policy, dict) else {}
        sizing = raw.get("profit_risk_sizing")
        sizing = sizing if isinstance(sizing, dict) else {}
        provenance = sizing.get("policy_provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        required_provenance = (
            "source",
            "observation_window",
            "sample_count",
            "generated_at",
            "strategy_version",
            "fallback_reason",
        )
        provenance_complete = bool(
            all(key in provenance for key in required_provenance)
            and str(provenance.get("source") or "").strip()
            and str(provenance.get("observation_window") or "").strip()
            and self.float_parser(provenance.get("sample_count"), 0.0) > 0
            and str(provenance.get("generated_at") or "").strip()
            and str(provenance.get("strategy_version") or "").strip()
            and not str(provenance.get("fallback_reason") or "").strip()
        )
        stop_loss_pct = self.float_parser(sizing.get("stress_stop_loss_pct"), 0.0)
        if (
            return_policy.get("eligible") is not True
            or sizing.get("production_eligible") is not True
            or not provenance_complete
            or stop_loss_pct <= 0
        ):
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
