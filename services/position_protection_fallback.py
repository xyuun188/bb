from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import select

from models.decision import AIDecision
from models.trade import Order
from services.normal_paper_trade import normal_paper_trade_contract_reasons
from services.trade_execution_contract import validate_entry_execution_contract


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
        normal_trade = raw.get("normal_paper_trade")
        normal_trade = normal_trade if isinstance(normal_trade, dict) else {}
        normal_trade_valid = not normal_paper_trade_contract_reasons(normal_trade)

        submitted = self._submitted_protection_prices(raw)
        if normal_trade_valid and submitted:
            return {
                **submitted,
                "source": "exact_order_submitted_dynamic_protection",
                "decision_id": getattr(decision, "id", None),
                "policy_provenance": {
                    "source": "persisted_execution_result_request_params",
                    "observation_window": "entry_submit",
                    "sample_count": 1,
                    "generated_at": getattr(decision, "executed_at", None),
                    "strategy_version": normal_trade.get("version"),
                },
            }

        sizing = raw.get("profit_risk_sizing")
        sizing = sizing if isinstance(sizing, dict) else {}
        provenance = sizing.get("policy_provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        _, contract_blockers = validate_entry_execution_contract(raw)
        stop_loss_pct = self.float_parser(
            getattr(decision, "stop_loss_pct", None)
            or sizing.get("stressed_loss_fraction"),
            0.0,
        )
        take_profit_pct = self.float_parser(
            getattr(decision, "take_profit_pct", None),
            0.0,
        )
        if contract_blockers or stop_loss_pct <= 0:
            return {}
        if not normal_trade_valid or take_profit_pct <= 0:
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
        stop_loss, take_profit = self._prices_from_decision(
            decision,
            entry_price=entry_price,
            side=side,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        return {
            "stop_loss_price": stop_loss if stop_loss > 0 else None,
            "take_profit_price": take_profit if take_profit > 0 else None,
            "source": "exact_order_dynamic_protection_plan",
            "decision_id": getattr(decision, "id", None),
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
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

    def _prices_from_decision(
        self,
        decision: AIDecision,
        *,
        entry_price: float,
        side: str,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> tuple[float, float]:
        refs = [entry_price]
        snapshot = getattr(decision, "feature_snapshot", None)
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        for key in ("current_price", "close", "bid", "ask", "last", "last_price"):
            value = self.float_parser(snapshot.get(key), 0.0)
            if value > 0:
                refs.append(value)
        low_ref = min(refs)
        high_ref = max(refs)
        if side == "long":
            return (
                low_ref * (1 - stop_loss_pct),
                high_ref * (1 + take_profit_pct),
            )
        return (
            high_ref * (1 + stop_loss_pct),
            low_ref * (1 - take_profit_pct),
        )

    def _submitted_protection_prices(self, raw: dict[str, Any]) -> dict[str, float]:
        execution = raw.get("execution_result")
        execution = execution if isinstance(execution, dict) else {}
        execution_raw = execution.get("raw_response")
        execution_raw = execution_raw if isinstance(execution_raw, dict) else {}
        params = execution_raw.get("request_params")
        params = params if isinstance(params, dict) else {}
        attached = params.get("attachAlgoOrds")
        attached_row = attached[0] if isinstance(attached, list) and attached else {}
        attached_row = attached_row if isinstance(attached_row, dict) else {}
        stop_loss = self.float_parser(attached_row.get("slTriggerPx"), 0.0)
        take_profit = self.float_parser(attached_row.get("tpTriggerPx"), 0.0)
        if stop_loss <= 0 or take_profit <= 0:
            return {}
        return {
            "stop_loss_price": stop_loss,
            "take_profit_price": take_profit,
        }
