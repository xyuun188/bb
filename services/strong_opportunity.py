"""Read-only audit of production return opportunities."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

from sqlalchemy import select

from core.symbols import normalize_trading_symbol
from db.session import get_read_session_ctx
from models.decision import AIDecision

DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_LIMIT = 500
ENTRY_ACTIONS = {"long", "short", "open_long", "open_short", "buy", "sell"}
PROVENANCE_FIELDS = {
    "source",
    "observation_window",
    "sample_count",
    "generated_at",
    "strategy_version",
    "fallback_reason",
}


@dataclass(frozen=True, slots=True)
class StrongOpportunityCandidate:
    decision_id: int
    symbol: str
    side: str
    created_at: str | None
    action: str
    executed: bool
    strong_opportunity: bool
    shadow_only: bool
    stage: str
    block_reasons: tuple[str, ...]
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "block_reasons": list(self.block_reasons),
            "can_bypass_risk_controls": False,
            "can_force_open": False,
            "can_apply_live_sizing": False,
        }


class StrongOpportunityService:
    """Audit the same dynamic return contract used by execution."""

    def __init__(self, *, lookback_hours: int = DEFAULT_LOOKBACK_HOURS, limit: int = DEFAULT_LIMIT):
        self.lookback_hours = max(int(lookback_hours or DEFAULT_LOOKBACK_HOURS), 1)
        self.limit = max(1, min(int(limit or DEFAULT_LIMIT), 5000))

    async def report(self) -> dict[str, Any]:
        since = (datetime.now(UTC) - timedelta(hours=self.lookback_hours)).replace(tzinfo=None)
        async with get_read_session_ctx() as session:
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
        entries = [row for row in decisions if str(row.action or "").lower() in ENTRY_ACTIONS]
        candidates = [self._classify(row) for row in entries]
        strong = [row for row in candidates if row.strong_opportunity]
        near = [
            row
            for row in candidates
            if not row.strong_opportunity
            and _safe_float(row.metrics.get("expected_net_return_pct"), 0.0) > 0
        ]
        blockers = Counter(reason for row in candidates for reason in row.block_reasons)
        return {
            "read_only": True,
            "audit_only": True,
            "live_entry_mutation": False,
            "live_sizing_mutation": False,
            "can_bypass_risk_controls": False,
            "can_force_open": False,
            "can_apply_live_sizing": False,
            "lookback_hours": self.lookback_hours,
            "checked_decisions": len(decisions),
            "entry_decisions": len(entries),
            "strong_candidate_count": len(strong),
            "executed_strong_candidate_count": sum(row.executed for row in strong),
            "near_miss_count": len(near),
            "blocker_counts": dict(blockers),
            "side_counts": dict(Counter(row.side or "unknown" for row in candidates)),
            "strong_candidates": [row.as_dict() for row in strong[:20]],
            "near_misses": [row.as_dict() for row in near[:20]],
            "contract": {
                "optimization_target": "realized_fee_after_return",
                "requires_positive_return_lcb": True,
                "requires_live_execution_cost": True,
                "requires_dynamic_risk_budget": True,
                "requires_complete_provenance": True,
                "fixed_strategy_thresholds": [],
            },
        }

    def _classify(self, decision: AIDecision) -> StrongOpportunityCandidate:
        raw = _safe_dict(decision.raw_llm_response)
        policy = _safe_dict(raw.get("production_return_policy"))
        opportunity = _safe_dict(raw.get("opportunity_score"))
        cost = _safe_dict(opportunity.get("execution_cost"))
        sizing = _safe_dict(raw.get("profit_risk_sizing"))
        metrics = {
            "expected_net_return_pct": _safe_float(policy.get("expected_net_return_pct")),
            "return_lcb_pct": _safe_float(policy.get("return_lcb_pct")),
            "production_source_count": int(_safe_float(policy.get("production_source_count"))),
            "position_size_pct": _safe_float(policy.get("position_size_pct")),
            "execution_cost_pct": _safe_float(cost.get("total_pct")),
            "side": _entry_side(decision.action),
        }
        reasons = _contract_blockers(policy, opportunity, cost, sizing)
        strong = not reasons
        return StrongOpportunityCandidate(
            decision_id=int(decision.id or 0),
            symbol=normalize_trading_symbol(decision.symbol),
            side=metrics["side"],
            created_at=_iso(decision.created_at),
            action=str(decision.action or "").lower(),
            executed=bool(decision.was_executed),
            strong_opportunity=strong,
            shadow_only=not strong,
            stage="production_return_ready" if strong else "observe_only",
            block_reasons=tuple(reasons),
            metrics=metrics,
        )


def _contract_blockers(policy: dict[str, Any], opportunity: dict[str, Any], cost: dict[str, Any], sizing: dict[str, Any]) -> list[str]:
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


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _entry_side(action: Any) -> str:
    value = str(action or "").lower()
    return "long" if value in {"long", "open_long", "buy"} else "short" if value in {"short", "open_short", "sell"} else ""


def _iso(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)).isoformat()
