"""Read-only root-cause audit for the production return contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

from sqlalchemy import select

from db.session import get_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest
from services.ml_signal_service import MLSignalService

ENTRY_ACTIONS = {"long", "short", "open_long", "open_short", "buy", "sell"}
PROVENANCE_FIELDS = {
    "source",
    "observation_window",
    "sample_count",
    "generated_at",
    "strategy_version",
    "fallback_reason",
}


class StrategySignalRootCauseAuditService:
    """Explain return, cost, risk-budget and provenance gaps without mutation."""

    def __init__(
        self,
        *,
        lookback_hours: int = 24,
        limit: int = 500,
        ml_status_provider: Callable[[], dict[str, Any]] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.lookback_hours = max(1, int(lookback_hours or 24))
        self.limit = max(1, min(int(limit or 500), 2000))
        self._ml_status_provider = ml_status_provider or MLSignalService().status
        self._now = now or (lambda: datetime.now(UTC))

    async def report(self) -> dict[str, Any]:
        since = self._now()
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)
        since = (since.astimezone(UTC) - timedelta(hours=self.lookback_hours)).replace(tzinfo=None)
        async with get_session_ctx() as session:
            decisions = list(
                (
                    await session.execute(
                        select(AIDecision)
                        .where(AIDecision.created_at >= since)
                        .order_by(AIDecision.created_at.desc())
                        .limit(self.limit)
                    )
                )
                .scalars()
                .all()
            )
            shadows = list(
                (
                    await session.execute(
                        select(ShadowBacktest)
                        .where(ShadowBacktest.created_at >= since, ShadowBacktest.status == "completed")
                        .order_by(ShadowBacktest.created_at.desc())
                        .limit(self.limit)
                    )
                )
                .scalars()
                .all()
            )
        try:
            ml_status = self._ml_status_provider()
        except Exception as exc:  # pragma: no cover
            ml_status = {"available": False, "status": "error", "error": str(exc)[:180]}
        return self.summarize(decisions=decisions, shadows=shadows, ml_status=ml_status)

    def summarize(
        self,
        *,
        decisions: list[AIDecision],
        shadows: list[ShadowBacktest],
        ml_status: dict[str, Any],
    ) -> dict[str, Any]:
        entries = [row for row in decisions if str(row.action or "").lower() in ENTRY_ACTIONS]
        blocker_counts: Counter[str] = Counter()
        expected_returns: list[float] = []
        return_lcbs: list[float] = []
        ready_count = 0
        for row in entries:
            raw = _safe_dict(getattr(row, "raw_llm_response", None))
            policy = _safe_dict(raw.get("production_return_policy"))
            opportunity = _safe_dict(raw.get("opportunity_score"))
            cost = _safe_dict(opportunity.get("execution_cost"))
            sizing = _safe_dict(raw.get("profit_risk_sizing"))
            expected = _maybe_float(policy.get("expected_net_return_pct"))
            lcb = _maybe_float(policy.get("return_lcb_pct"))
            if expected is not None:
                expected_returns.append(expected)
            if lcb is not None:
                return_lcbs.append(lcb)
            reasons = _contract_blockers(policy, opportunity, cost, sizing)
            blocker_counts.update(reasons)
            if not reasons:
                ready_count += 1

        causes = [
            {
                "code": code,
                "severity": "warning",
                "count": count,
                "message": _cause_message(code),
            }
            for code, count in blocker_counts.most_common()
        ]
        ml_readiness = _safe_dict(ml_status.get("readiness"))
        shadow_missed_count = sum(bool(getattr(row, "missed_opportunity", False)) for row in shadows)
        return {
            "status": "warning" if causes else "ok",
            "summary": (
                "Production return contract gaps were found."
                if causes
                else "Production return contracts are complete in the audit window."
            ),
            "audit_only": True,
            "read_only": True,
            "live_entry_mutation": False,
            "live_sizing_mutation": False,
            "live_leverage_mutation": False,
            "can_force_open": False,
            "can_override_thresholds": False,
            "can_change_ml_readiness": False,
            "can_bypass_risk_controls": False,
            "entry_decision_count": len(entries),
            "high_quality_entry_count": ready_count,
            "production_return_ready_count": ready_count,
            "production_return_blocked_count": len(entries) - ready_count,
            "contract_blocker_counts": dict(blocker_counts),
            "expected_net_return_distribution": _distribution(expected_returns),
            "return_lcb_distribution": _distribution(return_lcbs),
            "ml": {
                "status": ml_status.get("status") or "unknown",
                "allow_live_position_influence": bool(
                    ml_status.get("allow_live_position_influence")
                ),
                "readiness": ml_readiness,
            },
            "server_profit": {"diagnostic_only": True},
            "shadow_missed_opportunity": {
                "missed_count": shadow_missed_count,
                "observation_only": True,
                "can_authorize_entry": False,
            },
            "expected_net_component_stats": {},
            "scheduler": {
                "read_only": True,
                "audit_only": True,
                "sample_count": len(decisions),
                "can_force_open": False,
                "can_override_thresholds": False,
                "can_bypass_risk_controls": False,
            },
            "root_causes": causes,
            "next_actions": [
                "Restore authoritative fee-after return, live cost, risk-budget and provenance inputs before production execution."
            ]
            if causes
            else ["Continue observing realized fee-after returns and left-tail behavior."],
        }


def _contract_blockers(
    policy: dict[str, Any],
    opportunity: dict[str, Any],
    cost: dict[str, Any],
    sizing: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if policy.get("eligible") is not True:
        reasons.append("production_return_policy_ineligible")
    if _safe_float(policy.get("expected_net_return_pct")) <= 0:
        reasons.append("fee_after_expected_return_not_positive")
    if _safe_float(policy.get("return_lcb_pct")) <= 0:
        reasons.append("fee_after_return_lcb_not_positive")
    if int(_safe_float(policy.get("production_source_count"))) <= 0:
        reasons.append("production_return_distribution_missing")
    if not _complete_provenance(policy.get("policy_provenance")):
        reasons.append("production_return_provenance_incomplete")
    if opportunity.get("production_eligible") is not True:
        reasons.append("opportunity_distribution_ineligible")
    if cost.get("production_eligible") is not True or _safe_float(cost.get("total_pct")) <= 0:
        reasons.append("live_execution_cost_incomplete")
    if not _complete_provenance(cost.get("policy_provenance")):
        reasons.append("execution_cost_provenance_incomplete")
    if sizing.get("production_eligible") is not True:
        reasons.append("dynamic_risk_budget_ineligible")
    if not _complete_provenance(sizing.get("policy_provenance")):
        reasons.append("dynamic_risk_budget_provenance_incomplete")
    return reasons


def _complete_provenance(value: Any) -> bool:
    payload = _safe_dict(value)
    return bool(
        PROVENANCE_FIELDS.issubset(payload)
        and str(payload.get("source") or "")
        and str(payload.get("observation_window") or "")
        and _safe_float(payload.get("sample_count")) > 0
        and str(payload.get("generated_at") or "")
        and str(payload.get("strategy_version") or "")
        and not str(payload.get("fallback_reason") or "")
    )


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 8),
        "median": round(ordered[len(ordered) // 2], 8),
        "max": round(ordered[-1], 8),
        "avg": round(sum(ordered) / len(ordered), 8),
    }


def _cause_message(code: str) -> str:
    return {
        "production_return_policy_ineligible": "The production return policy is absent or ineligible.",
        "fee_after_expected_return_not_positive": "Fee-after expected return is not positive.",
        "fee_after_return_lcb_not_positive": "Fee-after return lower confidence bound is not positive.",
        "production_return_distribution_missing": "No production-eligible return observations are available.",
        "production_return_provenance_incomplete": "Return-policy provenance is incomplete.",
        "opportunity_distribution_ineligible": "The opportunity return distribution is not production eligible.",
        "live_execution_cost_incomplete": "Live execution cost is unavailable or incomplete.",
        "execution_cost_provenance_incomplete": "Execution-cost provenance is incomplete.",
        "dynamic_risk_budget_ineligible": "The dynamic risk budget is ineligible.",
        "dynamic_risk_budget_provenance_incomplete": "Risk-budget provenance is incomplete.",
    }.get(code, code)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    result = _maybe_float(value)
    return default if result is None else result


def _maybe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None
