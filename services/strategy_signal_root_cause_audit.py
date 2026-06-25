"""Read-only root-cause audit for entry signal quality.

This service explains why entry candidates are not becoming high-quality
tradeable signals. It never changes thresholds, sizing, leverage, orders, or
model readiness.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from db.session import get_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest
from services.ml_signal_service import MLSignalService

HIGH_QUALITY_ENTRY_TIERS = {"exploration", "small", "medium", "normal"}
WEAK_ENTRY_TIERS = {"weak_conflict_probe", "degraded_missing_probe"}
MODEL_UNUSABLE_STATUSES = {"ignored", "missing", "unknown"}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    clean = sorted(float(value) for value in values if _maybe_float(value) is not None)
    if not clean:
        return {"count": 0}

    def percentile(ratio: float) -> float:
        index = min(max(int((len(clean) - 1) * ratio), 0), len(clean) - 1)
        return round(clean[index], 6)

    return {
        "count": len(clean),
        "min": round(clean[0], 6),
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "max": round(clean[-1], 6),
        "avg": round(sum(clean) / len(clean), 6),
    }


def _iso(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _decision_raw(decision: AIDecision) -> dict[str, Any]:
    return _safe_dict(getattr(decision, "raw_llm_response", None))


def _opportunity(decision: AIDecision) -> dict[str, Any]:
    return _safe_dict(_decision_raw(decision).get("opportunity_score"))


def _evidence(decision: AIDecision) -> dict[str, Any]:
    return _safe_dict(_opportunity(decision).get("evidence_score"))


def _evidence_tier(decision: AIDecision) -> str:
    return str(_evidence(decision).get("tier") or "").strip()


def _expected_net(decision: AIDecision) -> float | None:
    return _maybe_float(_opportunity(decision).get("expected_net_return_pct"))


def _analysis_type(decision: AIDecision) -> str:
    raw = _decision_raw(decision)
    return str(
        getattr(decision, "analysis_type", None) or raw.get("analysis_type") or "unknown"
    ).lower()


def _component_key(item: dict[str, Any]) -> str:
    return str(item.get("key") or item.get("source") or "unknown")


def _ml_readiness_summary(status: dict[str, Any]) -> dict[str, Any]:
    readiness = _safe_dict(status.get("readiness"))
    metrics = _safe_dict(readiness.get("metrics"))
    blocking = _safe_list(readiness.get("blocking_reasons"))
    thresholds = _safe_dict(readiness.get("thresholds"))
    if not metrics:
        metrics = {
            key: status.get(key)
            for key in (
                "sample_count",
                "test_count",
                "dirty_sample_ratio",
                "long_accuracy",
                "short_accuracy",
                "long_pr_auc",
                "short_pr_auc",
                "top_long_avg_return_pct",
                "bottom_long_avg_return_pct",
                "top_short_avg_return_pct",
                "bottom_short_avg_return_pct",
                "training_data_version",
                "required_training_data_version",
            )
            if key in status
        }
    return {
        "available": bool(status.get("available")),
        "status": status.get("status"),
        "readiness_state": status.get("readiness_state") or readiness.get("state"),
        "allow_live_position_influence": bool(
            status.get("allow_live_position_influence")
            or readiness.get("allow_live_position_influence")
        ),
        "advisory_enabled": bool(status.get("advisory_enabled")),
        "blocking_reason_codes": [
            str(item.get("code"))
            for item in blocking
            if isinstance(item, dict) and item.get("code")
        ],
        "blocking_reasons": [item for item in blocking if isinstance(item, dict)][:12],
        "thresholds": thresholds,
        "metrics": metrics,
    }


class StrategySignalRootCauseAuditService:
    """Aggregate ML/server-profit/shadow blockers without mutating live trading."""

    def __init__(
        self,
        *,
        lookback_hours: int = 24,
        limit: int = 500,
        ml_status_provider: Callable[[], dict[str, Any]] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.lookback_hours = max(1, int(lookback_hours or 24))
        self.limit = max(50, min(int(limit or 500), 2000))
        self._ml_status_provider = ml_status_provider or MLSignalService().status
        self._now = now or (lambda: datetime.now(UTC))

    async def report(self) -> dict[str, Any]:
        now = self._now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        since = now.astimezone(UTC) - timedelta(hours=self.lookback_hours)
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
                        .where(
                            ShadowBacktest.created_at >= since,
                            ShadowBacktest.status == "completed",
                        )
                        .order_by(ShadowBacktest.created_at.desc())
                        .limit(self.limit)
                    )
                )
                .scalars()
                .all()
            )
        try:
            ml_status = self._ml_status_provider()
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            ml_status = {
                "available": False,
                "status": "error",
                "readiness_state": "error",
                "allow_live_position_influence": False,
                "error": str(exc)[:180],
            }
        return self.summarize(decisions=decisions, shadows=shadows, ml_status=ml_status)

    def summarize(
        self,
        *,
        decisions: list[AIDecision],
        shadows: list[ShadowBacktest],
        ml_status: dict[str, Any],
    ) -> dict[str, Any]:
        entry_decisions = [
            row for row in decisions if str(row.action or "").lower() in {"long", "short"}
        ]
        market_entry_decisions = [
            row for row in entry_decisions if _analysis_type(row) in {"market", "market_scan"}
        ]
        symbols = Counter(str(row.symbol or "unknown") for row in entry_decisions)
        top_symbol_count = symbols.most_common(1)[0][1] if symbols else 0
        symbol_concentration_ratio = (
            round(top_symbol_count / len(entry_decisions), 6) if entry_decisions else 0.0
        )

        tier_counts = Counter(_evidence_tier(row) or "missing" for row in entry_decisions)
        high_quality_count = sum(tier_counts.get(tier, 0) for tier in HIGH_QUALITY_ENTRY_TIERS)
        weak_count = sum(tier_counts.get(tier, 0) for tier in WEAK_ENTRY_TIERS)

        evidence_status_counts: dict[str, Counter[str]] = {}
        component_contributions: dict[str, list[float]] = {}
        component_available_counts: dict[str, int] = {}
        component_block_reasons: dict[str, Counter[str]] = {}
        advisory_wait_reasons: Counter[str] = Counter()
        relief_counts: Counter[str] = Counter()
        score_gaps: list[float] = []
        expected_net_values: list[float] = []
        profit_quality_values: list[float] = []
        loss_probability_values: list[float] = []
        tail_risk_values: list[float] = []
        server_expected_values: list[float] = []

        for row in entry_decisions:
            opportunity = _opportunity(row)
            evidence = _safe_dict(opportunity.get("evidence_score"))
            expected_net = _expected_net(row)
            if expected_net is not None:
                expected_net_values.append(expected_net)
            score = _maybe_float(opportunity.get("score"))
            min_score = _maybe_float(opportunity.get("min_score_required"))
            if score is not None and min_score is not None:
                score_gaps.append(score - min_score)
            for target, key in (
                (profit_quality_values, "profit_quality_ratio"),
                (loss_probability_values, "server_profit_loss_probability"),
                (tail_risk_values, "tail_risk_score"),
                (server_expected_values, "server_profit_expected_return_pct"),
            ):
                value = _maybe_float(opportunity.get(key))
                if value is not None:
                    target.append(value)
            for reason in _safe_list(evidence.get("advisory_wait_reasons")):
                advisory_wait_reasons[str(reason)[:160]] += 1
            for relief_key in (
                "positive_net_probe_relief",
                "memory_missed_opportunity_relief",
                "strong_positive_net_relief",
                "short_probe_relief",
                "missing_key_degraded_relief",
            ):
                relief = _safe_dict(evidence.get(relief_key))
                if relief.get("applied"):
                    relief_counts[relief_key] += 1
                    if relief.get("tradeable_probe"):
                        relief_counts[f"{relief_key}:tradeable"] += 1
                    if relief.get("shadow_only"):
                        relief_counts[f"{relief_key}:shadow_only"] += 1

            for component in _safe_list(evidence.get("components")):
                if not isinstance(component, dict):
                    continue
                source = str(component.get("source") or "unknown")
                status = str(component.get("status") or "unknown")
                evidence_status_counts.setdefault(source, Counter())[status] += 1

            breakdown = _safe_dict(opportunity.get("expected_net_breakdown"))
            for component in _safe_list(breakdown.get("components")):
                if not isinstance(component, dict):
                    continue
                key = _component_key(component)
                contribution = _maybe_float(component.get("contribution_pct"))
                if contribution is not None:
                    component_contributions.setdefault(key, []).append(contribution)
                if component.get("available"):
                    component_available_counts[key] = component_available_counts.get(key, 0) + 1
                for reason in _safe_list(component.get("blocked_reasons")):
                    component_block_reasons.setdefault(key, Counter())[str(reason)[:120]] += 1

        component_stats = {
            key: {
                **_distribution(values),
                "positive_count": sum(1 for value in values if value > 0),
                "negative_count": sum(1 for value in values if value < 0),
                "zero_count": sum(1 for value in values if value == 0),
                "available_count": component_available_counts.get(key, 0),
                "top_blocked_reasons": [
                    {"reason": reason, "count": count}
                    for reason, count in component_block_reasons.get(key, Counter()).most_common(5)
                ],
            }
            for key, values in sorted(component_contributions.items())
        }

        ml_component_counts = dict(evidence_status_counts.get("ml", Counter()))
        ml_total = sum(ml_component_counts.values())
        ml_usable = sum(
            count
            for status, count in ml_component_counts.items()
            if status not in MODEL_UNUSABLE_STATUSES
        )
        ml_usable_rate = round(ml_usable / ml_total, 6) if ml_total else 0.0
        server_component_counts = dict(evidence_status_counts.get("server_profit", Counter()))
        server_negative_or_opposite_count = sum(
            int(server_component_counts.get(status, 0))
            for status in ("opposite", "weak_opposite", "ignored_negative_expected")
        )
        server_aligned_count = int(server_component_counts.get("aligned", 0))

        completed_shadow_count = len(shadows)
        missed_shadows = [row for row in shadows if bool(row.missed_opportunity)]
        missed_ratio = (
            round(len(missed_shadows) / completed_shadow_count, 6)
            if completed_shadow_count
            else 0.0
        )
        missed_by_best_action = Counter(str(row.best_action or "unknown") for row in missed_shadows)
        missed_by_symbol = Counter(str(row.symbol or "unknown") for row in missed_shadows)
        shadow_tradeable_relief_count = sum(
            count for key, count in relief_counts.items() if key.endswith(":tradeable")
        )

        ml_readiness = _ml_readiness_summary(ml_status)
        root_causes = self._root_causes(
            entry_count=len(entry_decisions),
            market_entry_count=len(market_entry_decisions),
            high_quality_count=high_quality_count,
            unique_symbol_count=len(symbols),
            symbol_concentration_ratio=symbol_concentration_ratio,
            ml_total=ml_total,
            ml_usable_rate=ml_usable_rate,
            server_negative_or_opposite_count=server_negative_or_opposite_count,
            server_aligned_count=server_aligned_count,
            missed_shadow_count=len(missed_shadows),
            missed_ratio=missed_ratio,
            shadow_tradeable_relief_count=shadow_tradeable_relief_count,
            positive_expected_net_count=sum(1 for value in expected_net_values if value > 0),
            weak_count=weak_count,
            score_gap_distribution=_distribution(score_gaps),
            ml_readiness=ml_readiness,
        )
        status = "warning" if root_causes else "ok"
        return {
            "status": status,
            "summary": (
                "Entry signal chain still has unresolved quality blockers."
                if root_causes
                else "Entry signal chain has no current blocking root-cause pattern."
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
            "window_hours": self.lookback_hours,
            "sampled_decision_limit": self.limit,
            "sampled_shadow_limit": self.limit,
            "decision_count": len(decisions),
            "entry_decision_count": len(entry_decisions),
            "market_entry_decision_count": len(market_entry_decisions),
            "action_counts": dict(Counter(str(row.action or "unknown") for row in decisions)),
            "entry_symbol_summary": {
                "unique_symbol_count": len(symbols),
                "top_symbol_ratio": symbol_concentration_ratio,
                "top_symbols": [
                    {"symbol": symbol, "count": count} for symbol, count in symbols.most_common(10)
                ],
            },
            "evidence_tier_counts": dict(tier_counts),
            "high_quality_entry_count": high_quality_count,
            "weak_entry_count": weak_count,
            "expected_net_distribution": _distribution(expected_net_values),
            "positive_expected_net_count": sum(1 for value in expected_net_values if value > 0),
            "negative_expected_net_count": sum(1 for value in expected_net_values if value < 0),
            "score_gap_distribution": _distribution(score_gaps),
            "profit_quality_distribution": _distribution(profit_quality_values),
            "loss_probability_distribution": _distribution(loss_probability_values),
            "tail_risk_distribution": _distribution(tail_risk_values),
            "expected_net_component_stats": component_stats,
            "evidence_component_status_counts": {
                key: dict(value) for key, value in sorted(evidence_status_counts.items())
            },
            "advisory_wait_reason_counts": dict(advisory_wait_reasons.most_common(12)),
            "entry_evidence_relief_counts": dict(relief_counts.most_common(12)),
            "ml": {
                "readiness": ml_readiness,
                "component_status_counts": ml_component_counts,
                "usable_rate": ml_usable_rate,
                "usable_count": ml_usable,
                "total_count": ml_total,
            },
            "server_profit": {
                "component_status_counts": server_component_counts,
                "selected_expected_return_distribution": _distribution(server_expected_values),
                "negative_or_opposite_count": server_negative_or_opposite_count,
                "aligned_count": server_aligned_count,
            },
            "shadow_missed_opportunity": {
                "completed_count": completed_shadow_count,
                "missed_count": len(missed_shadows),
                "missed_ratio": missed_ratio,
                "missed_by_best_action": dict(missed_by_best_action.most_common(8)),
                "missed_by_symbol": dict(missed_by_symbol.most_common(8)),
                "tradeable_relief_count": shadow_tradeable_relief_count,
            },
            "root_causes": root_causes,
            "next_actions": self._next_actions(root_causes),
            "diagnostic_boundary": (
                "Read-only Stage 5 audit. This report can explain blockers, but it must not "
                "open positions, change thresholds, change sizing, change leverage, or mark ML ready."
            ),
        }

    def _root_causes(
        self,
        *,
        entry_count: int,
        market_entry_count: int,
        high_quality_count: int,
        unique_symbol_count: int,
        symbol_concentration_ratio: float,
        ml_total: int,
        ml_usable_rate: float,
        server_negative_or_opposite_count: int,
        server_aligned_count: int,
        missed_shadow_count: int,
        missed_ratio: float,
        shadow_tradeable_relief_count: int,
        positive_expected_net_count: int,
        weak_count: int,
        score_gap_distribution: dict[str, Any],
        ml_readiness: dict[str, Any],
    ) -> list[dict[str, Any]]:
        causes: list[dict[str, Any]] = []
        if entry_count == 0:
            causes.append(
                {
                    "code": "no_entry_candidates",
                    "severity": "warning",
                    "message": "No long/short entry candidates were recorded in the audit window.",
                    "count": 0,
                }
            )
        if entry_count >= 10 and ml_total >= 10 and ml_usable_rate < 0.25:
            causes.append(
                {
                    "code": "ml_not_contributing",
                    "severity": "warning",
                    "message": "ML components are mostly ignored or missing in entry evidence.",
                    "count": ml_total,
                    "rate": ml_usable_rate,
                }
            )
        ml_blocking_codes = {
            str(code) for code in _safe_list(ml_readiness.get("blocking_reason_codes")) if code
        }
        metrics = _safe_dict(ml_readiness.get("metrics"))
        thresholds = _safe_dict(ml_readiness.get("thresholds"))
        top_return_blockers = sorted(
            code for code in ml_blocking_codes if code.endswith("_top_return_below_threshold")
        )
        if top_return_blockers:
            causes.append(
                {
                    "code": "ml_top_return_not_profitable",
                    "severity": "warning",
                    "message": (
                        "ML top-score buckets are not profitable enough for live influence."
                    ),
                    "blocking_reason_codes": top_return_blockers,
                    "readiness_state": ml_readiness.get("readiness_state"),
                    "allow_live_position_influence": bool(
                        ml_readiness.get("allow_live_position_influence")
                    ),
                    "top_long_avg_return_pct": _maybe_float(metrics.get("top_long_avg_return_pct")),
                    "top_short_avg_return_pct": _maybe_float(
                        metrics.get("top_short_avg_return_pct")
                    ),
                    "bottom_long_avg_return_pct": _maybe_float(
                        metrics.get("bottom_long_avg_return_pct")
                    ),
                    "bottom_short_avg_return_pct": _maybe_float(
                        metrics.get("bottom_short_avg_return_pct")
                    ),
                    "required_min_top_return_pct": _maybe_float(
                        thresholds.get("min_top_return_pct")
                    ),
                }
            )
        if entry_count >= 10 and server_negative_or_opposite_count > server_aligned_count:
            causes.append(
                {
                    "code": "server_profit_negative_or_opposite",
                    "severity": "warning",
                    "message": "Server-profit evidence is more often opposite or negative than aligned.",
                    "count": server_negative_or_opposite_count,
                    "aligned_count": server_aligned_count,
                }
            )
        if entry_count >= 20 and high_quality_count == 0:
            causes.append(
                {
                    "code": "high_quality_entry_gap",
                    "severity": "warning",
                    "message": "Entry candidates exist, but none reached exploration/small/medium/normal evidence tiers.",
                    "count": entry_count,
                }
            )
        if entry_count >= 20 and unique_symbol_count <= 5 and market_entry_count >= 10:
            causes.append(
                {
                    "code": "candidate_symbol_concentration",
                    "severity": "warning",
                    "message": "Market entry candidates are concentrated in too few symbols.",
                    "unique_symbol_count": unique_symbol_count,
                    "top_symbol_ratio": symbol_concentration_ratio,
                }
            )
        if (
            missed_shadow_count >= 20
            and missed_ratio >= 0.35
            and shadow_tradeable_relief_count == 0
        ):
            causes.append(
                {
                    "code": "shadow_missed_not_convertible",
                    "severity": "warning",
                    "message": "Shadow missed opportunities are frequent, but they are not converting into tradeable same-side evidence.",
                    "count": missed_shadow_count,
                    "missed_ratio": missed_ratio,
                }
            )
        score_gap_avg = _safe_float(score_gap_distribution.get("avg"), 0.0)
        if entry_count >= 10 and positive_expected_net_count > 0 and high_quality_count == 0:
            causes.append(
                {
                    "code": "positive_ev_still_below_evidence_quality",
                    "severity": "warning",
                    "message": "Some candidates have positive expected net, but evidence quality or opportunity score still blocks them.",
                    "positive_expected_net_count": positive_expected_net_count,
                    "score_gap_avg": round(score_gap_avg, 6),
                }
            )
        if weak_count >= max(5, int(entry_count * 0.5)):
            causes.append(
                {
                    "code": "weak_evidence_dominates",
                    "severity": "warning",
                    "message": "Most entry candidates remain weak-conflict or degraded-missing probes.",
                    "count": weak_count,
                }
            )
        return causes

    @staticmethod
    def _next_actions(root_causes: list[dict[str, Any]]) -> list[str]:
        if not root_causes:
            return [
                "Continue observing closed-position net PnL before changing sizing or leverage.",
                "Keep ML readiness and server-profit evidence under normal audit cadence.",
            ]
        by_code = {str(item.get("code")) for item in root_causes}
        actions: list[str] = []
        if "ml_not_contributing" in by_code:
            actions.append(
                "Fix ML training quality/readiness first; do not hard-set ML ready or relax entry gates."
            )
        if "ml_top_return_not_profitable" in by_code:
            actions.append(
                "Keep ML in observation until top-score buckets show positive fee-adjusted returns; inspect labels, window selection, and dirty samples before retraining."
            )
        if "server_profit_negative_or_opposite" in by_code:
            actions.append(
                "Audit server-profit labels, OKX fact integrity, fees, slippage, and side mapping before trusting its contribution."
            )
        if "shadow_missed_not_convertible" in by_code:
            actions.append(
                "Convert shadow missed opportunities only through same-symbol same-side repeated evidence with positive risk quality."
            )
        if "candidate_symbol_concentration" in by_code:
            actions.append(
                "Inspect scan/ranker/budget filters to find why market candidates concentrate in too few symbols."
            )
        if (
            "high_quality_entry_gap" in by_code
            or "positive_ev_still_below_evidence_quality" in by_code
        ):
            actions.append(
                "Inspect expected-net breakdown, evidence components, profit quality, loss probability, and tail risk before changing thresholds."
            )
        if "no_entry_candidates" in by_code:
            actions.append(
                "Check market scan, analysis budget, feature coverage, and AI decision throughput before strategy tuning."
            )
        return actions
