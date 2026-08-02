"""Read-only audit of hard capacity and dynamic exit readiness."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from math import isfinite
from types import SimpleNamespace
from typing import Any

from sqlalchemy import or_, select

from core.symbols import normalize_trading_symbol
from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.trade import Order, Position
from services.current_position_management import (
    current_position_management_contract_complete,
)
from services.dynamic_position_capacity import DynamicPositionCapacityPolicy
from services.exchange_exit_decision_lineage import decision_exit_exchange_order_ids
from services.trade_execution_contract import (
    AUTHORITATIVE_FILL_SYNC_GRACE_SECONDS,
    classify_exit_execution_contract,
)

EXIT_ACTIONS = {"close_long", "close_short", "exit_long", "exit_short"}


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
            exit_order_ids = {
                exchange_order_id
                for decision in decisions
                if _action(decision) in EXIT_ACTIONS
                for exchange_order_id in decision_exit_exchange_order_ids(decision)
            }
            orders = (
                list(
                    (
                        await session.execute(
                            select(Order)
                            .where(
                                or_(
                                    Order.decision_id.in_(decision_ids)
                                    if decision_ids
                                    else False,
                                    Order.exchange_order_id.in_(sorted(exit_order_ids))
                                    if exit_order_ids
                                    else False,
                                )
                            )
                            .order_by(Order.created_at.desc())
                            .limit(self.limit)
                        )
                    )
                    .scalars()
                    .all()
                )
                if decision_ids or exit_order_ids
                else []
            )
        return self._summarize(positions, decisions, orders)

    def _summarize(
        self,
        positions: list[Position],
        decisions: list[AIDecision],
        orders: list[Order],
    ) -> dict[str, Any]:
        fragment_rows = [self._position_row(position) for position in positions]
        position_rows = self._position_group_rows(positions)
        capacity = self.capacity_policy.evaluate(open_positions=fragment_rows).as_dict()
        orders_by_decision: dict[int, list[Order]] = {}
        orders_by_exchange_id: dict[str, list[Order]] = {}
        for order in orders:
            decision_id = int(getattr(order, "decision_id", 0) or 0)
            if decision_id:
                orders_by_decision.setdefault(decision_id, []).append(order)
            for exchange_order_id in _order_exchange_ids(order):
                orders_by_exchange_id.setdefault(exchange_order_id, []).append(order)

        exit_rows = []
        for decision in decisions:
            if _action(decision) not in EXIT_ACTIONS:
                continue
            related_orders = list(orders_by_decision.get(int(decision.id or 0), []))
            for exchange_order_id in decision_exit_exchange_order_ids(decision):
                related_orders.extend(orders_by_exchange_id.get(exchange_order_id, []))
            deduplicated_orders = list({id(order): order for order in related_orders}.values())
            exit_rows.append(self._exit_row(decision, deduplicated_orders))
        pending_positions = [
            row
            for row in position_rows
            if not row["position_economics_complete"]
            and row.get("authoritative_entry_fact_sync_pending") is True
        ]
        incomplete_positions = [
            row
            for row in position_rows
            if not row["position_economics_complete"]
            and row.get("authoritative_entry_fact_sync_pending") is not True
        ]
        executed_exit_gaps = [
            row
            for row in exit_rows
            if row["executed"] and not row["exit_contract_complete"]
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
            "open_position_count": len(fragment_rows),
            "open_position_group_count": len(position_rows),
            "open_group_count": capacity["open_group_count"],
            "side_counts": dict(Counter(row["side"] or "unknown" for row in fragment_rows)),
            "capacity": capacity,
            "position_economics_complete_count": (
                len(position_rows) - len(incomplete_positions) - len(pending_positions)
            ),
            "position_economics_pending_count": len(pending_positions),
            "position_economics_pending": pending_positions[:50],
            "position_economics_incomplete_count": len(incomplete_positions),
            "position_economics_incomplete": incomplete_positions[:50],
            "dynamic_exit_decision_count": len(exit_rows),
            "executed_dynamic_exit_count": sum(
                row["executed"] and row["exit_contract_kind"] == "dynamic_exit"
                for row in exit_rows
            ),
            "executed_exchange_protection_exit_count": sum(
                row["executed"]
                and row["exit_contract_kind"] == "okx_exchange_protection"
                for row in exit_rows
            ),
            "executed_dynamic_exit_contract_gap_count": len(executed_exit_gaps),
            "executed_dynamic_exit_contract_gaps": executed_exit_gaps[:50],
            "dynamic_exit_decisions": exit_rows[:50],
            "policy": {
                "capacity_source": "configured_exchange_account_position_group_limit",
                "strategy_learning_cannot_expand_capacity": True,
                "position_economics_required_for_dynamic_exit": True,
                "authoritative_fill_sync_grace_seconds": (
                    AUTHORITATIVE_FILL_SYNC_GRACE_SECONDS
                ),
                "dynamic_exit_provenance_required": True,
                "filled_order_link_required_for_executed_exit": True,
            },
        }

    @classmethod
    def _position_group_rows(cls, positions: list[Position]) -> list[dict[str, Any]]:
        """Audit management economics against the exchange net-position scope.

        A single OKX net position can have multiple local persistence fragments.
        Its protection order and management contract belong to the net position,
        not to every fragment independently.
        """

        grouped: dict[tuple[str, str], list[Position]] = {}
        for position in positions:
            key = (
                normalize_trading_symbol(getattr(position, "symbol", "") or ""),
                str(getattr(position, "side", "") or "").lower(),
            )
            grouped.setdefault(key, []).append(position)

        rows: list[dict[str, Any]] = []
        for (_symbol, _side), fragments in grouped.items():
            contracts = [
                _safe_dict(getattr(fragment, "current_management_contract", None))
                for fragment in fragments
            ]
            management = next(
                (
                    contract
                    for contract in contracts
                    if contract.get("position_scope") == "exchange_net_position_group"
                ),
                contracts[0] if contracts else {},
            )
            local_quantity = sum(
                abs(_safe_float(getattr(fragment, "quantity", None)))
                for fragment in fragments
            )
            contract_quantity = abs(_safe_float(management.get("quantity")))
            quantity_matches = bool(
                contract_quantity > 0
                and abs(local_quantity - contract_quantity)
                <= max(contract_quantity * 0.001, 1e-8)
            )
            aggregate = SimpleNamespace(
                id=min((int(getattr(fragment, "id", 0) or 0) for fragment in fragments), default=0),
                model_name=getattr(fragments[0], "model_name", "") if fragments else "",
                symbol=management.get("symbol") or getattr(fragments[0], "symbol", ""),
                side=management.get("side") or getattr(fragments[0], "side", ""),
                quantity=contract_quantity or local_quantity,
                entry_price=_safe_float(management.get("entry_price")),
                current_price=_safe_float(management.get("current_price")),
                entry_fee=_safe_float(management.get("entry_fee_usdt")),
                stop_loss_price=_safe_float(management.get("stop_loss_price")),
                take_profit_price=_safe_float(management.get("take_profit_price")),
                unrealized_pnl=sum(
                    _safe_float(getattr(fragment, "unrealized_pnl", None))
                    for fragment in fragments
                ),
                current_management_contract=management,
                created_at=min(
                    (getattr(fragment, "created_at", None) for fragment in fragments),
                    default=None,
                ),
            )
            row = cls._position_row(aggregate)
            row["position_fragment_count"] = len(fragments)
            row["position_fragment_ids"] = [
                int(getattr(fragment, "id", 0) or 0)
                for fragment in fragments
                if int(getattr(fragment, "id", 0) or 0) > 0
            ]
            row["local_fragment_quantity"] = round(local_quantity, 8)
            row["local_fragment_quantity_matches_contract"] = quantity_matches
            if not quantity_matches:
                row["position_economics_complete"] = False
            rows.append(row)
        return rows

    @staticmethod
    def _position_row(position: Position) -> dict[str, Any]:
        current_price = _safe_float(getattr(position, "current_price", None))
        entry_price = _safe_float(getattr(position, "entry_price", None))
        quantity = abs(_safe_float(getattr(position, "quantity", None)))
        notional = abs(quantity * current_price) if quantity > 0 and current_price > 0 else 0.0
        entry_fee = max(_safe_float(getattr(position, "entry_fee", None)), 0.0)
        management_contract = _safe_dict(
            getattr(position, "current_management_contract", None)
        )
        management_complete = current_position_management_contract_complete(
            position,
            management_contract,
        )
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
            and management_complete
            and stop_distance > 0
        )
        blockers = {
            str(reason or "").strip()
            for reason in management_contract.get("blockers", [])
            if str(reason or "").strip()
        }
        created_at = _as_utc(getattr(position, "created_at", None))
        sync_deadline = (
            created_at + timedelta(seconds=AUTHORITATIVE_FILL_SYNC_GRACE_SECONDS)
            if created_at is not None
            else None
        )
        authoritative_entry_fact_sync_pending = bool(
            not economics_complete
            and blockers == {"authoritative_entry_fee_evidence_incomplete"}
            and management_contract.get("protection_evidence_complete") is True
            and stop_distance > 0
            and sync_deadline is not None
            and datetime.now(UTC) <= sync_deadline
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
            "entry_fee_usdt": entry_fee,
            "has_authoritative_execution_fee": management_contract.get(
                "entry_fee_evidence_complete"
            )
            is True,
            "has_stop_distance": stop_distance > 0,
            "current_management_contract_complete": management_complete,
            "current_management_contract": management_contract,
            "authoritative_entry_fact_sync_pending": (
                authoritative_entry_fact_sync_pending
            ),
            "authoritative_entry_fact_sync_deadline_at": (
                sync_deadline.isoformat()
                if authoritative_entry_fact_sync_pending and sync_deadline is not None
                else None
            ),
            "original_entry_contract_status": management_contract.get(
                "original_entry_contract_status"
            ),
            "position_economics_complete": economics_complete,
            "created_at": _iso(getattr(position, "created_at", None)),
        }

    @staticmethod
    def _exit_row(decision: AIDecision, orders: list[Order]) -> dict[str, Any]:
        classified = classify_exit_execution_contract(decision, orders)
        kind = str(classified.get("contract_kind") or "dynamic_exit")
        complete = classified.get("contract_complete") is True
        return {
            "decision_id": int(getattr(decision, "id", 0) or 0),
            "symbol": normalize_trading_symbol(getattr(decision, "symbol", "") or ""),
            "action": _action(decision),
            "executed": classified.get("executed") is True,
            "filled_order_count": _safe_int(classified.get("filled_order_count")),
            "close_fraction": classified.get("close_fraction"),
            "hard_risk": bool(classified.get("hard_risk")),
            "position_sample_count": _safe_int(
                _safe_dict(
                    _safe_dict(getattr(decision, "raw_llm_response", None)).get(
                        "dynamic_exit_policy"
                    )
                ).get("policy_provenance", {}).get("sample_count")
            ),
            "exit_contract_kind": kind,
            "exit_contract_complete": complete,
            "dynamic_exit_contract_complete": bool(kind == "dynamic_exit" and complete),
            "contract_reasons": list(classified.get("reasons") or []),
            "created_at": _iso(getattr(decision, "created_at", None)),
        }


def _action(row: Any) -> str:
    value = getattr(row, "action", "")
    return str(getattr(value, "value", value) or "").lower()


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _order_exchange_ids(order: Any) -> set[str]:
    value = str(getattr(order, "exchange_order_id", "") or "").strip()
    if not value:
        return set()
    return {
        token.strip()
        for separator_normalized in value.replace(";", ",").replace("|", ",").split(",")
        if (token := separator_normalized.strip())
    }


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


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
