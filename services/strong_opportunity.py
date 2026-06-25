"""Read-only strong opportunity classifier for Phase 2 audits."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from core.symbols import normalize_trading_symbol
from db.session import get_read_session_ctx
from models.decision import AIDecision

DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_LIMIT = 500
ENTRY_ACTIONS = {"long", "short", "open_long", "open_short", "buy", "sell"}
MIN_EXPECTED_NET_PCT = 0.8
MIN_PROFIT_QUALITY_RATIO = 1.05
MAX_LOSS_PROBABILITY = 0.42
MAX_TAIL_RISK_SCORE = 0.72
MIN_ALIGNED_SOURCES = 2
MIN_EFFECTIVE_SCORE = 0.62
TRADEABLE_TIERS = {"small", "normal", "medium", "strong", "quality_override"}
BLOCKED_TIERS = {"blocked", "shadow_only", "weak_conflict_probe"}


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
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "side": self.side,
            "created_at": self.created_at,
            "action": self.action,
            "executed": self.executed,
            "strong_opportunity": self.strong_opportunity,
            "shadow_only": self.shadow_only,
            "stage": self.stage,
            "block_reasons": list(self.block_reasons),
            "metrics": self.metrics,
            "can_bypass_risk_controls": False,
            "can_force_open": False,
            "can_apply_live_sizing": False,
        }


class StrongOpportunityService:
    """Classify entry decisions into auditable strong-opportunity candidates.

    This service is intentionally read-only. It only explains whether a recent
    entry decision already satisfies the Phase 2 strong-opportunity shape; it
    does not feed execution, sizing, leverage, or risk gates.
    """

    def __init__(
        self,
        *,
        lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
        limit: int = DEFAULT_LIMIT,
    ) -> None:
        self.lookback_hours = max(int(lookback_hours or DEFAULT_LOOKBACK_HOURS), 1)
        self.limit = max(1, min(int(limit or DEFAULT_LIMIT), 5000))

    async def report(self) -> dict[str, Any]:
        since = datetime.now(UTC) - timedelta(hours=self.lookback_hours)
        since_naive = since.replace(tzinfo=None)
        async with get_read_session_ctx() as session:
            rows = await session.execute(
                select(AIDecision)
                .where(AIDecision.created_at >= since_naive)
                .order_by(AIDecision.created_at.desc())
                .limit(self.limit)
            )
            decisions = list(rows.scalars().all())

        entry_decisions = [
            decision for decision in decisions if _entry_action(getattr(decision, "action", ""))
        ]
        candidates = [self._classify(decision) for decision in entry_decisions]
        strong = [candidate for candidate in candidates if candidate.strong_opportunity]
        near_miss = [
            candidate
            for candidate in candidates
            if not candidate.strong_opportunity and _has_positive_shape(candidate)
        ]
        blocker_counts = Counter(
            reason for candidate in candidates for reason in candidate.block_reasons
        )
        tier_counts = Counter(
            str(candidate.metrics.get("evidence_tier") or "missing") for candidate in candidates
        )
        side_counts = Counter(candidate.side or "unknown" for candidate in candidates)
        executed_strong_count = sum(1 for candidate in strong if candidate.executed)
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
            "entry_decisions": len(entry_decisions),
            "strong_candidate_count": len(strong),
            "executed_strong_candidate_count": executed_strong_count,
            "near_miss_count": len(near_miss),
            "blocker_counts": dict(blocker_counts.most_common(12)),
            "evidence_tier_counts": dict(tier_counts.most_common(12)),
            "side_counts": dict(side_counts.most_common(8)),
            "strong_candidates": [candidate.as_dict() for candidate in strong[:20]],
            "near_misses": [candidate.as_dict() for candidate in near_miss[:20]],
            "thresholds": {
                "min_expected_net_pct": MIN_EXPECTED_NET_PCT,
                "min_profit_quality_ratio": MIN_PROFIT_QUALITY_RATIO,
                "max_loss_probability": MAX_LOSS_PROBABILITY,
                "max_tail_risk_score": MAX_TAIL_RISK_SCORE,
                "min_aligned_sources": MIN_ALIGNED_SOURCES,
                "min_effective_score": MIN_EFFECTIVE_SCORE,
                "tradeable_tiers": sorted(TRADEABLE_TIERS),
                "blocked_tiers": sorted(BLOCKED_TIERS),
            },
            "diagnostic_boundary": (
                "Read-only Phase 2 strong-opportunity audit. A strong candidate "
                "cannot bypass entry evidence, profit quality, ML readiness, OKX "
                "rules, sizing, leverage, or risk controls."
            ),
        }

    def _classify(self, decision: AIDecision) -> StrongOpportunityCandidate:
        raw = _safe_dict(getattr(decision, "raw_llm_response", None))
        opportunity = _safe_dict(raw.get("opportunity_score"))
        side = _entry_side(getattr(decision, "action", None), opportunity)
        side_evidence = _selected_side_evidence(raw, side)
        evidence = _safe_dict(opportunity.get("evidence_score"))
        metrics = _metrics_for_decision(decision, opportunity, evidence, side_evidence, side)
        block_reasons = tuple(_block_reasons(metrics))
        strong = not block_reasons
        return StrongOpportunityCandidate(
            decision_id=int(getattr(decision, "id", 0) or 0),
            symbol=normalize_trading_symbol(getattr(decision, "symbol", "")),
            side=side,
            created_at=_iso(getattr(decision, "created_at", None)),
            action=str(getattr(decision, "action", "") or "").lower(),
            executed=bool(getattr(decision, "was_executed", False)),
            strong_opportunity=strong,
            shadow_only=not strong,
            stage="strong_opportunity" if strong else "blocked_or_observe",
            block_reasons=block_reasons,
            metrics=metrics,
        )


def _metrics_for_decision(
    decision: AIDecision,
    opportunity: dict[str, Any],
    evidence: dict[str, Any],
    side_evidence: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    aligned_sources = _aligned_sources(opportunity, evidence, side_evidence)
    major_opposites = _safe_list(evidence.get("major_opposites"))
    strong_opposites = _safe_list(evidence.get("strong_opposites"))
    expected_net = _safe_float(
        side_evidence.get("expected_net_return_pct"),
        _safe_float(opportunity.get("expected_net_return_pct"), 0.0),
    )
    profit_quality = _safe_float(
        side_evidence.get("profit_quality_ratio"),
        _safe_float(opportunity.get("profit_quality_ratio"), 0.0),
    )
    loss_probability = _safe_float(
        side_evidence.get(
            "loss_probability",
            opportunity.get("server_profit_loss_probability", opportunity.get("loss_probability")),
        ),
        1.0,
    )
    tail_risk = _safe_float(
        side_evidence.get("tail_risk_score", opportunity.get("tail_risk_score")),
        1.0,
    )
    effective_score = _safe_float(
        evidence.get("effective_score", opportunity.get("score")),
        0.0,
    )
    evidence_tier = str(
        evidence.get("tier") or opportunity.get("evidence_tier") or side_evidence.get("tier") or ""
    ).lower()
    high_risk_review = _safe_dict(
        _safe_dict(getattr(decision, "raw_llm_response", None)).get("high_risk_review")
    )
    return {
        "expected_net_return_pct": round(expected_net, 6),
        "profit_quality_ratio": round(profit_quality, 6),
        "loss_probability": round(loss_probability, 6),
        "tail_risk_score": round(tail_risk, 6),
        "effective_score": round(effective_score, 6),
        "evidence_tier": evidence_tier,
        "aligned_sources": aligned_sources,
        "aligned_source_count": len(set(aligned_sources)),
        "major_opposites": major_opposites,
        "strong_opposites": strong_opposites,
        "hard_block": bool(evidence.get("hard_block")),
        "shadow_only": bool(evidence.get("shadow_only")),
        "side": side,
        "metrics_source": "entry_candidate_evidence" if side_evidence else "opportunity_score",
        "server_profit_expected_return_pct": _round_optional(
            side_evidence.get(
                "server_profit_expected_return_pct",
                opportunity.get("server_profit_expected_return_pct"),
            )
        ),
        "high_risk_review_status": high_risk_review.get("status") or "",
        "high_risk_review_approved": high_risk_review.get("approved"),
    }


def _block_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if metrics.get("side") not in {"long", "short"}:
        reasons.append("missing_entry_side")
    if _safe_float(metrics.get("expected_net_return_pct")) < MIN_EXPECTED_NET_PCT:
        reasons.append("expected_net_below_strong_threshold")
    if _safe_float(metrics.get("profit_quality_ratio")) < MIN_PROFIT_QUALITY_RATIO:
        reasons.append("profit_quality_below_strong_threshold")
    if _safe_float(metrics.get("loss_probability"), 1.0) > MAX_LOSS_PROBABILITY:
        reasons.append("loss_probability_above_strong_threshold")
    if _safe_float(metrics.get("tail_risk_score"), 1.0) > MAX_TAIL_RISK_SCORE:
        reasons.append("tail_risk_above_strong_threshold")
    if int(metrics.get("aligned_source_count") or 0) < MIN_ALIGNED_SOURCES:
        reasons.append("aligned_sources_below_strong_threshold")
    if _safe_float(metrics.get("effective_score")) < MIN_EFFECTIVE_SCORE:
        reasons.append("effective_score_below_strong_threshold")
    tier = str(metrics.get("evidence_tier") or "").lower()
    if tier in BLOCKED_TIERS:
        reasons.append("evidence_tier_not_tradeable_strong")
    if metrics.get("hard_block"):
        reasons.append("evidence_hard_block")
    if metrics.get("shadow_only"):
        reasons.append("evidence_shadow_only")
    if metrics.get("major_opposites"):
        reasons.append("major_opposites_present")
    if metrics.get("strong_opposites"):
        reasons.append("strong_opposites_present")
    if metrics.get("high_risk_review_approved") is False:
        reasons.append("high_risk_review_not_approved")
    return reasons


def _has_positive_shape(candidate: StrongOpportunityCandidate) -> bool:
    metrics = candidate.metrics
    return bool(
        _safe_float(metrics.get("expected_net_return_pct")) > 0
        and _safe_float(metrics.get("profit_quality_ratio")) > 0.65
        and _safe_float(metrics.get("loss_probability"), 1.0) < 0.65
    )


def _aligned_sources(
    opportunity: dict[str, Any],
    evidence: dict[str, Any],
    side_evidence: dict[str, Any],
) -> list[str]:
    sources: list[str] = []
    for source in _safe_list(evidence.get("aligned_support_sources")):
        text = str(source or "").strip()
        if text and text not in sources:
            sources.append(text)
    explicit_count = int(_safe_float(side_evidence.get("aligned_source_count"), 0.0))
    for key, source in (
        ("local_profit_aligned", "local_profit"),
        ("ml_aligned", "ml"),
        ("timeseries_aligned", "timeseries"),
        ("expert_aligned", "expert"),
        ("server_profit_aligned", "server_profit"),
    ):
        if opportunity.get(key) and source not in sources:
            sources.append(source)
    while len(sources) < explicit_count:
        sources.append(f"side_evidence_{len(sources) + 1}")
    return sources


def _entry_action(action: Any) -> bool:
    return str(action or "").lower().strip() in ENTRY_ACTIONS


def _entry_side(action: Any, opportunity: dict[str, Any]) -> str:
    value = str(action or "").lower().strip()
    if value in {"long", "open_long", "buy"}:
        return "long"
    if value in {"short", "open_short", "sell"}:
        return "short"
    return str(opportunity.get("side") or "").lower().strip()


def _selected_side_evidence(raw: dict[str, Any], side: str) -> dict[str, Any]:
    evidence = _safe_dict(raw.get("entry_candidate_evidence"))
    side_payload = _safe_dict(evidence.get(side))
    if side_payload:
        return side_payload
    if str(evidence.get("side") or "").lower() == side:
        return evidence
    return {}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()
