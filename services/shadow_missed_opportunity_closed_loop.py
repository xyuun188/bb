"""Observation-only fee-after return review for missed shadow opportunities."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from services.dynamic_policy_values import empirical_policy_value
from services.profit_supervision import shadow_fee_after_return_labels
from services.trade_execution_contract import (
    summarize_trade_execution_contract,
    validate_entry_execution_contract,
)

DEFAULT_REPORT_WINDOW_HOURS = 24
DEFAULT_REPORT_LIMIT = 200


class ShadowMissedOpportunityClosedLoopService:
    def __init__(self, session_context_factory: Any | None = None) -> None:
        self._session_context_factory = session_context_factory

    async def report(self, *, hours: int = DEFAULT_REPORT_WINDOW_HOURS, limit: int = DEFAULT_REPORT_LIMIT) -> dict[str, Any]:
        from sqlalchemy import select

        from db.session import get_read_session_ctx
        from models.decision import AIDecision
        from models.learning import ShadowBacktest
        from models.trade import Order

        capped_hours = max(1, min(int(hours or DEFAULT_REPORT_WINDOW_HOURS), 168))
        capped_limit = max(1, min(int(limit or DEFAULT_REPORT_LIMIT), 1000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        session_factory = self._session_context_factory or get_read_session_ctx
        async with session_factory() as session:
            result = await session.execute(
                select(
                    ShadowBacktest.id, ShadowBacktest.created_at, ShadowBacktest.symbol,
                    ShadowBacktest.status, ShadowBacktest.missed_opportunity,
                    ShadowBacktest.best_action, ShadowBacktest.horizon_minutes,
                    ShadowBacktest.long_return_pct, ShadowBacktest.short_return_pct,
                ).order_by(ShadowBacktest.id.desc()).limit(capped_limit)
            )
            shadows = [dict(row) for row in result.mappings().all()]
            result = await session.execute(
                select(
                    AIDecision.id, AIDecision.created_at, AIDecision.symbol,
                    AIDecision.action, AIDecision.was_executed,
                ).order_by(AIDecision.id.desc()).limit(capped_limit)
            )
            decisions = [dict(row) for row in result.mappings().all()]
        shadows = [row for row in shadows if _row_in_window(row, since)]
        decisions = [row for row in decisions if _row_in_window(row, since)]
        # Keep the same bounded windows, but only load JSON used by the report.
        # Pending shadows and unexecuted decisions need no large payload at all.
        payload_specs = (
            (ShadowBacktest, "feature_snapshot", {
                row["id"]: row for row in shadows
                if row["status"] == "completed" and row["missed_opportunity"]
            }),
            (AIDecision, "raw_llm_response", {
                row["id"]: row for row in decisions
                if row["action"].lower() in {"long", "short"} and row["was_executed"]
            }),
        )
        for model, field, rows_by_id in payload_specs:
            record_ids = list(rows_by_id)
            for offset in range(0, len(record_ids), 5):
                async with session_factory() as session:
                    result = await session.execute(
                        select(model.id, getattr(model, field)).where(
                            model.id.in_(record_ids[offset : offset + 5])
                        )
                    )
                    for row in result.mappings().all():
                        rows_by_id[row["id"]][field] = row[field]
        executed_entry_ids = [
            row["id"] for row in decisions
            if row["action"].lower() in {"long", "short"} and row["was_executed"]
        ]
        orders = []
        if executed_entry_ids:
            async with session_factory() as session:
                result = await session.execute(
                    select(
                        Order.decision_id, Order.execution_mode, Order.symbol,
                        Order.status, Order.quantity, Order.price,
                        Order.exchange_order_id, Order.filled_at, Order.created_at,
                        Order.okx_fill_contracts, Order.okx_sync_status, Order.okx_raw_fills,
                    ).where(Order.decision_id.in_(executed_entry_ids))
                )
                orders = [dict(row) for row in result.mappings().all()]
        report = summarize_shadow_missed_opportunities(
            shadows, decisions=decisions, orders=orders,
        )
        report["window_hours"] = capped_hours
        report["query_policy"] = {"online_safe": True, "ordered_by_primary_key": True, "row_limit": capped_limit}
        return report


def summarize_shadow_missed_opportunities(
    shadows: Sequence[Any], *, decisions: Sequence[Any] | None = None,
    orders: Sequence[Any] | None = None,
) -> dict[str, Any]:
    completed = [row for row in shadows if str(_row_get(row, "status") or "") == "completed"]
    missed = [row for row in completed if bool(_row_get(row, "missed_opportunity"))]
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in missed:
        symbol = str(_row_get(row, "symbol") or "")
        side = str(_row_get(row, "best_action") or "").lower()
        value = _side_return(row, side)
        if symbol and side in {"long", "short"} and value is not None:
            groups[(symbol, side)].append(value)

    observations: list[dict[str, Any]] = []
    for (symbol, side), returns in sorted(groups.items()):
        lower = empirical_policy_value(
            "missed_shadow_return_lower_hinge",
            returns,
            selector="lower_hinge",
            observation_window="completed_missed_shadow_fee_after_returns",
        )
        observations.append(
            {
                "symbol": symbol,
                "side": side,
                "sample_count": len(returns),
                "average_return_pct": round(sum(returns) / len(returns), 8),
                "return_lower_hinge_pct": lower.value,
                "return_distribution_provenance": lower.to_dict(),
                "observation_only": True,
                "can_authorize_entry": False,
                "can_change_size_or_leverage": False,
            }
        )

    executed_contract_gaps = _executed_return_contract_gaps(decisions or [], orders=orders)
    return {
        "audit_only": True,
        "read_only": True,
        "live_entry_mutation": False,
        "can_bypass_risk_controls": False,
        "global_missed_count_can_drive_entries": False,
        "usable_group_count": 0,
        "summary": {
            "completed_count": len(completed),
            "missed_count": len(missed),
            "group_count": len(groups),
            "observe_only_count": len(observations),
            "executed_return_contract_gap_count": len(executed_contract_gaps),
        },
        "return_observations": observations[:20],
        "executed_return_contract_gaps": executed_contract_gaps[:20],
        "blocked_reason_counts": dict(Counter(row["reason"] for row in executed_contract_gaps)),
        "safety_rules": [
            "missed_opportunity_is_observation_only",
            "production_entry_requires_current_positive_return_lcb",
            "production_entry_requires_live_cost_and_risk_provenance",
        ],
    }


def _executed_return_contract_gaps(
    decisions: Sequence[Any], *, orders: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    if orders is not None:
        entries = [
            row for row in decisions
            if str(_row_get(row, "action") or "").lower() in {"long", "short"}
            and bool(_row_get(row, "was_executed"))
        ]
        audit = summarize_trade_execution_contract(entries, orders=orders)
        return [
            {
                "decision_id": row["decision_id"],
                "reason": f"executed_without_complete_{row['contract_lifecycle']}_entry_contract",
                "contract_lifecycle": row["contract_lifecycle"],
                "contract_blockers": row["effective_reasons"],
            }
            for row in audit["entry_contracts"]
            if row["effective_reasons"]
        ]
    gaps: list[dict[str, Any]] = []
    for row in decisions:
        action = str(_row_get(row, "action") or "").lower()
        if action not in {"long", "short"} or not bool(_row_get(row, "was_executed")):
            continue
        raw = _safe_dict(_row_get(row, "raw_llm_response"))
        contract, contract_blockers = validate_entry_execution_contract(
            raw,
            allow_historical_settlement=True,
        )
        if contract_blockers:
            lifecycle = str(contract.get("contract_lifecycle") or "unknown")
            gaps.append(
                {
                    "decision_id": _row_get(row, "id"),
                    "reason": f"executed_without_complete_{lifecycle}_entry_contract",
                    "contract_lifecycle": lifecycle,
                    "contract_blockers": contract_blockers,
                }
            )
    return gaps


def _side_return(row: Any, side: str) -> float | None:
    if side not in {"long", "short"}:
        return None
    fee_after = shadow_fee_after_return_labels(
        {
            "horizon_minutes": _row_get(row, "horizon_minutes"),
            "long_return_pct": _row_get(row, "long_return_pct"),
            "short_return_pct": _row_get(row, "short_return_pct"),
            "features": _row_get(row, "feature_snapshot") or {},
        }
    )
    return _safe_float(
        fee_after.get(f"{side}_net_return_after_all_cost_pct"),
        None,
    )


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    return row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_in_window(row: Any, since: datetime) -> bool:
    created_at = _row_get(row, "created_at")
    if not isinstance(created_at, datetime):
        return True
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return created_at.astimezone(UTC) >= since
