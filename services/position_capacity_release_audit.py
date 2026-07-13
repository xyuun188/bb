"""Read-only audit of hard capacity and dynamic exit readiness."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

from sqlalchemy import select

from core.symbols import normalize_trading_symbol
from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.trade import Order, Position
from services.dynamic_position_capacity import DynamicPositionCapacityPolicy

EXIT_ACTIONS = {"close_long", "close_short", "exit_long", "exit_short"}
FILLED_STATUSES = {"filled", "closed"}
PROVENANCE_FIELDS = (
    "source",
    "observation_window",
    "sample_count",
    "generated_at",
    "strategy_version",
    "fallback_reason",
)


class PositionCapacityReleaseAuditService:
    """Audit current hard capacity and recent dynamic exits without mutation."""

    def __init__(
        self,
        *,
        lookback_hours: int = 24,
        limit: int = 500,
        capacity_policy: DynamicPositionCapacityPolicy | None = None,
    ) -> None:
        self.lookback_hours = max(int(lookback_hours or 24), 1)
        self.limit = max(1, min(int(limit or 500), 5000))
        self.capacity_policy = capacity_policy or DynamicPositionCapacityPolicy()

    async def report(self) -> dict[str, Any]:
        since = datetime.now(UTC) - timedelta(hours=self.lookback_hours)
        since_naive = since.replace(tzinfo=None)
        async with get_read_session_ctx() as session:
            positions = list(
                (await session.execute(select(Position).where(Position.is_open.is_(True))))
                .scalars()
                .all()
            )
            decisions = list(
                (
                    await session.execute(
                        select(AIDecision)
                        .where(AIDecision.created_at >= since_naive)
                        .order_by(AIDecision.created_at.desc())
                        .limit(self.limit)
                    )
                )
                .scalars()
                .all()
            )
            decision_ids = [
                int(decision.id)
                for decision in decisions
                if decision.id and _action(decision) in EXIT_ACTIONS
            ]
            orders = (
                list(
                    (
                        await session.execute(
                            select(Order)
                            .where(Order.decision_id.in_(decision_ids))
                            .order_by(Order.created_at.desc())
                            .limit(self.limit)
                        )
                    )
                    .scalars()
                    .all()
                )
                if decision_ids
                else []
            )
        return self._summarize(positions, decisions, orders)

    def _summarize(
        self,
        positions: list[Position],
        decisions: list[AIDecision],
        orders: list[Order],
    ) -> dict[str, Any]:
        position_rows = [self._position_row(position) for position in positions]
        capacity = self.capacity_policy.evaluate(open_positions=position_rows).as_dict()
        orders_by_decision: dict[int, list[Order]] = {}
        for order in orders:
            decision_id = int(getattr(order, "decision_id", 0) or 0)
            if decision_id:
                orders_by_decision.setdefault(decision_id, []).append(order)

        exit_rows = [
            self._exit_row(decision, orders_by_decision.get(int(decision.id or 0), []))
            for decision in decisions
            if _action(decision) in EXIT_ACTIONS
        ]
        incomplete_positions = [
            row for row in position_rows if not row["position_economics_complete"]
        ]
        executed_exit_gaps = [
            row
            for row in exit_rows
            if row["executed"] and not row["dynamic_exit_contract_complete"]
        ]
        return {
            "read_only": True,
            "audit_only": True,
            "live_exit_mutation": False,
            "live_entry_mutation": False,
            "live_sizing_mutation": False,
            "can_force_close": False,
            "can_bypass_risk_controls": False,
            "lookback_hours": self.lookback_hours,
            "checked_decisions": len(decisions),
            "open_position_count": len(position_rows),
            "open_group_count": capacity["open_group_count"],
            "side_counts": dict(Counter(row["side"] or "unknown" for row in position_rows)),
            "capacity": capacity,
            "position_economics_complete_count": len(position_rows) - len(incomplete_positions),
            "position_economics_incomplete_count": len(incomplete_positions),
            "position_economics_incomplete": incomplete_positions[:50],
            "dynamic_exit_decision_count": len(exit_rows),
            "executed_dynamic_exit_count": sum(row["executed"] for row in exit_rows),
            "executed_dynamic_exit_contract_gap_count": len(executed_exit_gaps),
            "executed_dynamic_exit_contract_gaps": executed_exit_gaps[:50],
            "dynamic_exit_decisions": exit_rows[:50],
            "policy": {
                "capacity_source": "configured_exchange_account_position_group_limit",
                "strategy_learning_cannot_expand_capacity": True,
                "position_economics_required_for_dynamic_exit": True,
                "dynamic_exit_provenance_required": True,
                "filled_order_link_required_for_executed_exit": True,
            },
        }

    @staticmethod
    def _position_row(position: Position) -> dict[str, Any]:
        current_price = _safe_float(getattr(position, "current_price", None))
        entry_price = _safe_float(getattr(position, "entry_price", None))
        quantity = abs(_safe_float(getattr(position, "quantity", None)))
        notional = abs(quantity * current_price) if quantity > 0 and current_price > 0 else 0.0
        entry_fee = max(_safe_float(getattr(position, "entry_fee", None)), 0.0)
        stop_price = max(_safe_float(getattr(position, "stop_loss_price", None)), 0.0)
        stop_distance = (
            abs(entry_price - stop_price) / entry_price
            if entry_price > 0 and stop_price > 0
            else 0.0
        )
        economics_complete = bool(
            quantity > 0
            and entry_price > 0
            and current_price > 0
            and notional > 0
            and entry_fee > 0
            and stop_distance > 0
        )
        return {
            "id": int(getattr(position, "id", 0) or 0),
            "model_name": str(getattr(position, "model_name", "") or ""),
            "symbol": normalize_trading_symbol(getattr(position, "symbol", "") or ""),
            "side": str(getattr(position, "side", "") or "").lower(),
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": current_price,
            "notional_usdt": round(notional, 8),
            "unrealized_pnl_usdt": _safe_float(getattr(position, "unrealized_pnl", None)),
            "has_execution_fee": entry_fee > 0,
            "has_stop_distance": stop_distance > 0,
            "position_economics_complete": economics_complete,
            "created_at": _iso(getattr(position, "created_at", None)),
        }

    @staticmethod
    def _exit_row(decision: AIDecision, orders: list[Order]) -> dict[str, Any]:
        raw = _safe_dict(getattr(decision, "raw_llm_response", None))
        policy = _safe_dict(raw.get("dynamic_exit_policy"))
        provenance = _safe_dict(policy.get("policy_provenance"))
        filled_count = sum(_order_status(order) in FILLED_STATUSES for order in orders)
        executed = bool(getattr(decision, "was_executed", False)) or filled_count > 0
        complete = bool(
            policy.get("eligible") is True
            and _safe_float(policy.get("close_fraction")) > 0
            and _provenance_complete(provenance)
            and (not executed or filled_count > 0)
        )
        return {
            "decision_id": int(getattr(decision, "id", 0) or 0),
            "symbol": normalize_trading_symbol(getattr(decision, "symbol", "") or ""),
            "action": _action(decision),
            "executed": executed,
            "filled_order_count": filled_count,
            "close_fraction": _safe_float(policy.get("close_fraction")),
            "hard_risk": bool(policy.get("hard_risk")),
            "position_sample_count": _safe_int(provenance.get("sample_count")),
            "dynamic_exit_contract_complete": complete,
            "created_at": _iso(getattr(decision, "created_at", None)),
        }


def _provenance_complete(value: Any) -> bool:
    provenance = _safe_dict(value)
    if any(key not in provenance for key in PROVENANCE_FIELDS):
        return False
    return bool(
        str(provenance.get("source") or "").strip()
        and str(provenance.get("observation_window") or "").strip()
        and _safe_int(provenance.get("sample_count")) > 0
        and str(provenance.get("generated_at") or "").strip()
        and str(provenance.get("strategy_version") or "").strip()
        and not str(provenance.get("fallback_reason") or "").strip()
    )


def _action(row: Any) -> str:
    value = getattr(row, "action", "")
    return str(getattr(value, "value", value) or "").lower()


def _order_status(order: Any) -> str:
    value = getattr(order, "status", "")
    return str(getattr(value, "value", value) or "").lower()


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _iso(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()
