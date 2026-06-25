"""Read-only audit for capacity release and position rotation closure."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from core.symbols import normalize_trading_symbol
from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.trade import Order, Position
from services.dynamic_position_capacity import DynamicPositionCapacityPolicy
from services.position_quality import PositionQualityScorer

DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_LIMIT = 500
OLD_PROFIT_MIN_HOLD_HOURS = 3.0
OLD_PROFIT_MAX_PNL_RATIO = 0.008
OLD_PROFIT_MIN_FEE_MULTIPLE = 1.0


class PositionCapacityReleaseAuditService:
    """Summarize whether capacity release signals close the loop.

    The audit is deliberately read-only. It does not create close decisions,
    change sizing, alter position capacity, or override risk controls.
    """

    def __init__(
        self,
        *,
        lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
        limit: int = DEFAULT_LIMIT,
        quality_scorer: PositionQualityScorer | None = None,
        capacity_policy: DynamicPositionCapacityPolicy | None = None,
    ) -> None:
        self.lookback_hours = max(int(lookback_hours or DEFAULT_LOOKBACK_HOURS), 1)
        self.limit = max(1, min(int(limit or DEFAULT_LIMIT), 5000))
        self.quality_scorer = quality_scorer or PositionQualityScorer()
        self.capacity_policy = capacity_policy or DynamicPositionCapacityPolicy(
            quality_scorer=self.quality_scorer
        )

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
            decision_ids = [int(decision.id) for decision in decisions if decision.id]
            linked_orders = []
            if decision_ids:
                linked_orders = list(
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

        open_rows = self._position_rows(positions)
        capacity = self.capacity_policy.evaluate(open_positions=open_rows).as_dict()
        release_decisions = [
            self._release_decision_row(decision)
            for decision in decisions
            if self._is_release_decision(decision)
        ]
        release_decisions = [row for row in release_decisions if row]
        orders_by_decision: dict[int, list[Order]] = {}
        for order in linked_orders:
            decision_id = int(getattr(order, "decision_id", 0) or 0)
            if decision_id:
                orders_by_decision.setdefault(decision_id, []).append(order)

        for row in release_decisions:
            decision_orders = orders_by_decision.get(int(row["decision_id"]), [])
            row["linked_order_count"] = len(decision_orders)
            row["linked_order_statuses"] = dict(
                Counter(str(getattr(order, "status", "") or "unknown") for order in decision_orders)
            )
            row["has_filled_order"] = any(
                str(getattr(order, "status", "") or "").lower() == "filled"
                for order in decision_orders
            )

        unclosed_release_decisions = [
            row
            for row in release_decisions
            if not row["was_executed"] and not row["has_filled_order"]
        ]
        crowded_blocks = [
            self._crowded_block_row(decision)
            for decision in decisions
            if self._contains_crowded_side_cap(decision)
        ]
        crowded_blocks = [row for row in crowded_blocks if row]
        current_release_candidates = [
            row for row in open_rows if row.get("position_quality", {}).get("should_release")
        ]
        old_profit_candidates = [
            row for row in open_rows if self._is_old_profit_rotation_candidate(row)
        ]

        return {
            "read_only": True,
            "audit_only": True,
            "live_exit_mutation": False,
            "live_entry_mutation": False,
            "live_sizing_mutation": False,
            "can_force_close": False,
            "can_close_winners": False,
            "can_bypass_risk_controls": False,
            "lookback_hours": self.lookback_hours,
            "checked_decisions": len(decisions),
            "open_position_count": len(open_rows),
            "open_group_count": self._open_group_count(open_rows),
            "side_counts": dict(Counter(str(row.get("side") or "unknown") for row in open_rows)),
            "quality_bucket_counts": dict(
                Counter(
                    str(row.get("position_quality", {}).get("bucket") or "unknown")
                    for row in open_rows
                )
            ),
            "capacity": capacity,
            "current_release_candidate_count": len(current_release_candidates),
            "current_release_candidates": current_release_candidates[:12],
            "old_profit_rotation_candidate_count": len(old_profit_candidates),
            "old_profit_rotation_candidates": old_profit_candidates[:12],
            "release_decision_count": len(release_decisions),
            "executed_release_decision_count": sum(
                1 for row in release_decisions if row["was_executed"] or row["has_filled_order"]
            ),
            "unclosed_release_decision_count": len(unclosed_release_decisions),
            "unclosed_release_decisions": unclosed_release_decisions[:12],
            "release_decision_action_counts": dict(
                Counter(str(row.get("action") or "unknown") for row in release_decisions)
            ),
            "crowded_block_count": len(crowded_blocks),
            "crowded_blocks": crowded_blocks[:12],
            "diagnostic_boundary": (
                "Read-only Phase 2 capacity-release audit. It can identify release "
                "gaps and old profitable capacity candidates, but cannot close, "
                "resize, force entries, or bypass risk controls."
            ),
        }

    def _position_rows(self, positions: list[Position]) -> list[dict[str, Any]]:
        base_rows = [self._position_row(position) for position in positions]
        group_counts = Counter(
            (
                str(row.get("symbol") or ""),
                str(row.get("side") or ""),
            )
            for row in base_rows
        )
        rows: list[dict[str, Any]] = []
        for row in base_rows:
            quality = self.quality_scorer.score(
                row,
                same_symbol_side_parts=group_counts[
                    (
                        str(row.get("symbol") or ""),
                        str(row.get("side") or ""),
                    )
                ],
            )
            rows.append({**row, "position_quality": quality.as_dict()})
        return rows

    @staticmethod
    def _position_row(position: Position) -> dict[str, Any]:
        current_price = _safe_float(getattr(position, "current_price", None))
        entry_price = _safe_float(getattr(position, "entry_price", None))
        quantity = abs(_safe_float(getattr(position, "quantity", None)))
        price = current_price if current_price > 0 else entry_price
        return {
            "id": int(getattr(position, "id", 0) or 0),
            "model_name": str(getattr(position, "model_name", "") or ""),
            "symbol": normalize_trading_symbol(getattr(position, "symbol", "") or ""),
            "side": str(getattr(position, "side", "") or "").lower(),
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": current_price,
            "leverage": _safe_float(getattr(position, "leverage", None), 1.0),
            "unrealized_pnl": _safe_float(getattr(position, "unrealized_pnl", None)),
            "realized_pnl": _safe_float(getattr(position, "realized_pnl", None)),
            "notional": abs(quantity * price) if quantity > 0 and price > 0 else 0.0,
            "is_open": bool(getattr(position, "is_open", True)),
            "created_at": _iso(getattr(position, "created_at", None)),
            "updated_at": _iso(getattr(position, "updated_at", None)),
        }

    @staticmethod
    def _is_release_decision(decision: AIDecision) -> bool:
        raw = _safe_dict(getattr(decision, "raw_llm_response", None))
        policy = _safe_dict(raw.get("position_release_policy"))
        action = str(getattr(decision, "action", "") or "").lower()
        return bool(
            policy
            or raw.get("exit_intent") == "capital_rotation"
            or action in {"close_long", "close_short"}
            and raw.get("analysis_type") == "position_review"
        )

    @staticmethod
    def _release_decision_row(decision: AIDecision) -> dict[str, Any]:
        raw = _safe_dict(getattr(decision, "raw_llm_response", None))
        policy = _safe_dict(raw.get("position_release_policy"))
        quality = _safe_dict(raw.get("position_quality"))
        return {
            "decision_id": int(getattr(decision, "id", 0) or 0),
            "symbol": normalize_trading_symbol(getattr(decision, "symbol", "") or ""),
            "action": str(getattr(decision, "action", "") or "").lower(),
            "created_at": _iso(getattr(decision, "created_at", None)),
            "was_executed": bool(getattr(decision, "was_executed", False)),
            "execution_reason": str(getattr(decision, "execution_reason", "") or "")[:260],
            "exit_intent": raw.get("exit_intent"),
            "release_policy": {
                "source": policy.get("source"),
                "forced": bool(policy.get("forced")),
                "exit_score": _round_optional(policy.get("exit_score")),
                "release_fraction": _round_optional(policy.get("release_fraction")),
                "release_reason": str(policy.get("release_reason") or "")[:260],
                "scan_reason": str(policy.get("scan_reason") or "")[:260],
            },
            "position_quality": quality,
        }

    @staticmethod
    def _contains_crowded_side_cap(decision: AIDecision) -> bool:
        raw = getattr(decision, "raw_llm_response", None)
        reason = str(getattr(decision, "execution_reason", "") or "")
        try:
            text = json.dumps(raw, ensure_ascii=False) if isinstance(raw, dict) else str(raw or "")
        except TypeError:
            text = str(raw or "")
        return "crowded_side_cap" in text or "crowded_side_cap" in reason

    @staticmethod
    def _crowded_block_row(decision: AIDecision) -> dict[str, Any]:
        raw = _safe_dict(getattr(decision, "raw_llm_response", None))
        gate = _safe_dict(raw.get("entry_execution_gate"))
        evidence = _safe_dict(_safe_dict(raw.get("opportunity_score")).get("evidence_score"))
        return {
            "decision_id": int(getattr(decision, "id", 0) or 0),
            "symbol": normalize_trading_symbol(getattr(decision, "symbol", "") or ""),
            "action": str(getattr(decision, "action", "") or "").lower(),
            "created_at": _iso(getattr(decision, "created_at", None)),
            "execution_reason": str(getattr(decision, "execution_reason", "") or "")[:260],
            "gate_status": gate.get("status"),
            "gate_reason": gate.get("reason") or gate.get("block_reason"),
            "evidence_tier": evidence.get("tier") or raw.get("evidence_tier"),
        }

    @staticmethod
    def _open_group_count(rows: list[dict[str, Any]]) -> int:
        groups = {
            (
                str(row.get("model_name") or ""),
                str(row.get("symbol") or ""),
                str(row.get("side") or ""),
            )
            for row in rows
            if row.get("is_open", True) is not False
        }
        return len(groups)

    @staticmethod
    def _is_old_profit_rotation_candidate(row: dict[str, Any]) -> bool:
        quality = _safe_dict(row.get("position_quality"))
        hold_hours = _safe_float(quality.get("hold_hours"))
        pnl_ratio = _safe_float(quality.get("pnl_ratio"))
        unrealized = _safe_float(row.get("unrealized_pnl"))
        estimated_fee = _safe_float(quality.get("estimated_round_trip_fee"))
        if hold_hours < OLD_PROFIT_MIN_HOLD_HOURS:
            return False
        if unrealized <= 0:
            return False
        if pnl_ratio > OLD_PROFIT_MAX_PNL_RATIO:
            return False
        return unrealized >= estimated_fee * OLD_PROFIT_MIN_FEE_MULTIPLE


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_optional(value: Any) -> float | None:
    if value is None:
        return None
    return round(_safe_float(value), 6)


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return None
