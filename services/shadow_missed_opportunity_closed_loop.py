"""Observation-only fee-after return review for missed shadow opportunities."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from services.dynamic_policy_values import empirical_policy_value
from services.profit_supervision import shadow_fee_after_return_labels
from services.trade_execution_contract import validate_entry_execution_contract

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

        capped_hours = max(1, min(int(hours or DEFAULT_REPORT_WINDOW_HOURS), 168))
        capped_limit = max(1, min(int(limit or DEFAULT_REPORT_LIMIT), 1000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        session_factory = self._session_context_factory or get_read_session_ctx
        async with session_factory() as session:
            shadows = list(
                (await session.execute(select(ShadowBacktest).order_by(ShadowBacktest.id.desc()).limit(capped_limit))).scalars().all()
            )
            decisions = list(
                (await session.execute(select(AIDecision).order_by(AIDecision.id.desc()).limit(capped_limit))).scalars().all()
            )
        report = summarize_shadow_missed_opportunities(
            [row for row in shadows if _row_in_window(row, since)],
            decisions=[row for row in decisions if _row_in_window(row, since)],
        )
        report["window_hours"] = capped_hours
        report["query_policy"] = {"online_safe": True, "ordered_by_primary_key": True, "row_limit": capped_limit}
        return report


def summarize_shadow_missed_opportunities(shadows: Sequence[Any], *, decisions: Sequence[Any] | None = None) -> dict[str, Any]:
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

    executed_contract_gaps = _executed_return_contract_gaps(decisions or [])
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


def _executed_return_contract_gaps(decisions: Sequence[Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for row in decisions:
        action = str(_row_get(row, "action") or "").lower()
        if action not in {"long", "short"} or not bool(_row_get(row, "was_executed")):
            continue
        raw = _safe_dict(_row_get(row, "raw_llm_response"))
        contract, contract_blockers = validate_entry_execution_contract(raw)
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
