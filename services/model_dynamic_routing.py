from __future__ import annotations

from collections import Counter
from typing import Any

ALL_ENTRY_EXPERTS = (
    "trend_expert",
    "momentum_expert",
    "sentiment_expert",
    "position_expert",
    "risk_expert",
)
CORE_ENTRY_EXPERTS = ("trend_expert", "momentum_expert", "risk_expert")
MANDATORY_SAFETY_EXPERTS = ("risk_expert",)
NON_CORE_ENTRY_EXPERTS = tuple(name for name in ALL_ENTRY_EXPERTS if name not in CORE_ENTRY_EXPERTS)
WEAK_HEALTH_STATES = {"reduce", "shadow_only", "disable", "replace"}
VALID_ROUTE_STAGES = {"shadow", "canary", "live"}


class ModelDynamicRoutingService:
    def __init__(self, session_context_factory: Any | None = None) -> None:
        self._session_context_factory = session_context_factory

    async def report(self, *, hours: int = 72, limit: int = 1200) -> dict[str, Any]:
        from datetime import UTC, datetime, timedelta

        from sqlalchemy import select

        from db.session import get_read_session_ctx
        from models.decision import AIDecision

        capped_hours = max(1, min(int(hours or 72), 168))
        capped_limit = max(50, min(int(limit or 1200), 5000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        session_factory = self._session_context_factory or get_read_session_ctx
        async with session_factory() as session:
            result = await session.execute(
                select(AIDecision)
                .where(AIDecision.created_at >= since)
                .order_by(AIDecision.created_at.desc())
                .limit(capped_limit)
            )
        report = summarize_dynamic_model_routing(list(result.scalars().all()))
        report["window_hours"] = capped_hours
        return report


def plan_dynamic_model_route(
    features: Any,
    context: dict[str, Any] | None = None,
    *,
    model_health: dict[str, Any] | None = None,
    competition: dict[str, Any] | None = None,
    feature_coverage: dict[str, Any] | None = None,
    requested_stage: str = "shadow",
    training_governance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only expert routing plan.

    C5 starts as shadow-only routing. The returned selected/skipped experts are
    advisory until competition baseline, readiness, feature coverage, and canary
    controls all allow live route mutation.
    """

    ctx = context if isinstance(context, dict) else {}
    health = model_health if isinstance(model_health, dict) else {}
    competition_report = competition if isinstance(competition, dict) else {}
    coverage = feature_coverage if isinstance(feature_coverage, dict) else {}
    governance = _route_governance(ctx, training_governance)

    expert_reasons: dict[str, list[str]] = {name: [] for name in ALL_ENTRY_EXPERTS}
    selected: set[str] = set(CORE_ENTRY_EXPERTS)
    skipped: set[str] = set()
    for name in CORE_ENTRY_EXPERTS:
        if name != "risk_expert":
            expert_reasons[name].append("core_entry_expert")
    expert_reasons["risk_expert"].append("mandatory_safety_expert")

    candidate_quality = str(ctx.get("candidate_quality") or "unknown").lower()
    high_quality = candidate_quality in {"high", "strong", "premium"}
    low_quality = candidate_quality in {"low", "weak", "ordinary", "normal", "unknown"}
    event_required = _event_or_sentiment_required(features)
    high_risk = _high_risk_market(features, ctx)

    if high_quality:
        selected.update(ALL_ENTRY_EXPERTS)
        for name in ALL_ENTRY_EXPERTS:
            expert_reasons[name].append("high_quality_candidate")
    elif low_quality:
        for name in NON_CORE_ENTRY_EXPERTS:
            skipped.add(name)
            expert_reasons[name].append("low_quality_candidate_shadow_reduction")

    if event_required:
        selected.add("sentiment_expert")
        skipped.discard("sentiment_expert")
        expert_reasons["sentiment_expert"].append("event_or_sentiment_evidence")

    if _has_open_position_context(ctx):
        selected.add("position_expert")
        skipped.discard("position_expert")
        expert_reasons["position_expert"].append("position_risk_review_required")

    if high_risk:
        selected.add("risk_expert")
        skipped.discard("risk_expert")
        expert_reasons["risk_expert"].append("high_risk_market")

    components = _safe_dict(health.get("components"))
    for name in ALL_ENTRY_EXPERTS:
        state = str(_safe_dict(components.get(name)).get("recommended_state") or "").lower()
        if state in WEAK_HEALTH_STATES:
            if name in MANDATORY_SAFETY_EXPERTS:
                expert_reasons[name].append("mandatory_safety_kept_despite_health")
                selected.add(name)
                skipped.discard(name)
            elif name == "sentiment_expert" and event_required:
                expert_reasons[name].append("health_weak_but_event_required")
            elif name in selected and not high_quality:
                selected.discard(name)
                skipped.add(name)
                expert_reasons[name].append(f"health_state_{state}_shadow_only")

    requested = str(requested_stage or governance.get("model_stage") or "shadow").lower()
    if requested not in VALID_ROUTE_STAGES:
        requested = "shadow"
    canary_blockers = _canary_route_blockers(ctx, competition_report, coverage, governance)
    live_blockers = _live_route_blockers(ctx, competition_report, coverage, governance)
    blocking_reasons = live_blockers if requested == "live" else canary_blockers
    if requested == "live":
        mode = "live_blocked" if live_blockers else "live_ready"
    else:
        mode = "shadow_only" if canary_blockers else "canary_ready"
    selected_list = [name for name in ALL_ENTRY_EXPERTS if name in selected]
    skipped_list = [name for name in ALL_ENTRY_EXPERTS if name in skipped and name not in selected]
    return {
        "audit_only": True,
        "mode": mode,
        "applied_to_live_calls": False,
        "live_route_mutation": False,
        "can_apply_live_route": False,
        "requested_stage": requested,
        "canary_ready": not bool(canary_blockers),
        "live_ready": not bool(live_blockers),
        "canary_blocking_reasons": canary_blockers,
        "live_blocking_reasons": live_blockers,
        "selected_experts": selected_list,
        "skipped_experts": skipped_list,
        "mandatory_safety_experts": list(MANDATORY_SAFETY_EXPERTS),
        "estimated_call_reduction": max(len(ALL_ENTRY_EXPERTS) - len(selected_list), 0),
        "blocking_reasons": blocking_reasons,
        "expert_reasons": {name: reasons for name, reasons in expert_reasons.items() if reasons},
        "routing_basis": {
            "candidate_quality": candidate_quality or "unknown",
            "event_required": event_required,
            "high_risk_market": high_risk,
            "feature_coverage_status": coverage.get("status"),
        },
        "training_governance": governance,
        "safety_rules": [
            "initial_dynamic_routing_shadow_or_canary_only",
            "risk_expert_never_skipped_for_latency",
            "baseline_required_before_live_route_mutation",
            "missing_features_block_live_route_mutation",
            "walk_forward_required_before_live_route_mutation",
        ],
    }


def summarize_dynamic_model_routing(decisions: list[Any]) -> dict[str, Any]:
    route_plans: list[dict[str, Any]] = []
    blocking_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    weak_evidence_executed_count = 0
    negative_executed_count = 0
    unsafe_live_mutation_attempts = 0
    estimated_call_reduction = 0
    live_ready_count = 0
    live_blocked_count = 0
    for decision in decisions:
        raw = _safe_dict(_row_get(decision, "raw_llm_response"))
        route = _safe_dict(raw.get("dynamic_model_routing"))
        if not route:
            continue
        route_plans.append(route)
        for reason in _safe_list(route.get("blocking_reasons")):
            blocking_counts[str(reason)] += 1
        for name in _safe_list(route.get("skipped_experts")):
            skipped_counts[str(name)] += 1
        for name in _safe_list(route.get("selected_experts")):
            selected_counts[str(name)] += 1
        estimated_call_reduction += int(_safe_float(route.get("estimated_call_reduction"), 0.0))
        if bool(route.get("live_ready")):
            live_ready_count += 1
        if route.get("mode") == "live_blocked" or _safe_list(route.get("live_blocking_reasons")):
            live_blocked_count += 1
        if bool(route.get("applied_to_live_calls")) or bool(route.get("live_route_mutation")):
            unsafe_live_mutation_attempts += 1
        if bool(_row_get(decision, "was_executed")) and str(_row_get(decision, "action") or "") in {
            "long",
            "short",
        }:
            tier = str(
                _safe_dict(_safe_dict(raw.get("opportunity_score")).get("evidence_score")).get(
                    "tier"
                )
                or ""
            )
            if tier in {"weak_conflict_probe", "degraded_missing_probe"}:
                weak_evidence_executed_count += 1
            if _safe_float(_row_get(decision, "outcome_pnl_pct"), 0.0) < 0:
                negative_executed_count += 1
    route_plan_count = len(route_plans)
    mode_counts = Counter(str(route.get("mode") or "unknown") for route in route_plans)
    return {
        "audit_only": True,
        "live_route_mutation": False,
        "can_apply_live_route": False,
        "summary": {
            "route_plan_count": route_plan_count,
            "shadow_only_count": int(mode_counts.get("shadow_only") or 0),
            "canary_ready_count": int(mode_counts.get("canary_ready") or 0),
            "live_ready_count": live_ready_count,
            "live_blocked_count": live_blocked_count,
            "estimated_call_reduction": estimated_call_reduction,
            "unsafe_live_mutation_attempts": unsafe_live_mutation_attempts,
        },
        "blocking_reason_counts": dict(blocking_counts),
        "selected_expert_counts": dict(selected_counts),
        "skipped_expert_counts": dict(skipped_counts),
        "safety_observations": {
            "weak_evidence_executed_count": weak_evidence_executed_count,
            "negative_executed_count": negative_executed_count,
            "route_change_increased_weak_or_fast_loss": False,
        },
        "recent_routes": route_plans[:20],
        "safety_rules": [
            "routing_report_is_read_only",
            "unsafe_live_mutation_is_never_honored",
            "weak_evidence_and_fast_loss_require_observation_before_canary",
            "live_route_requires_walk_forward_and_manual_enablement",
        ],
    }


def _route_governance(
    context: dict[str, Any],
    training_governance: dict[str, Any] | None,
) -> dict[str, Any]:
    source = training_governance if isinstance(training_governance, dict) else {}
    if not source:
        source = _safe_dict(context.get("phase3_training_governance"))
    evaluation_policy = _safe_dict(source.get("evaluation_policy"))
    return {
        "training_mode": str(source.get("training_mode") or "shadow").lower(),
        "model_stage": str(source.get("model_stage") or "shadow").lower(),
        "promotion_flow": source.get("promotion_flow")
        or evaluation_policy.get("promotion_flow")
        or "shadow_to_canary_to_live",
        "live_mutation": bool(source.get("live_mutation") or evaluation_policy.get("live_mutation")),
        "requires_walk_forward": bool(evaluation_policy.get("requires_walk_forward", True)),
        "evaluation_policy": evaluation_policy,
    }


def _canary_route_blockers(
    context: dict[str, Any],
    competition: dict[str, Any],
    coverage: dict[str, Any],
    governance: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if _ml_readiness_blocks_route(context):
        blockers.append("ml_readiness_blocks_live_route")
    if competition.get("can_apply_live_weight") is False:
        blockers.append("competition_not_live_applicable")
    if "baseline_missing" in {
        str(item) for item in _safe_list(competition.get("blocking_reasons"))
    }:
        blockers.append("competition_baseline_missing")
    baseline = _safe_dict(competition.get("baseline"))
    if int(_safe_float(baseline.get("sample_count"), 0.0)) <= 0 and not competition.get(
        "blocking_reasons"
    ):
        blockers.append("competition_baseline_missing")
    if coverage.get("status") in {"warning", "critical"} and _safe_list(
        coverage.get("missing_features")
    ):
        blockers.append("feature_coverage_missing")
    if governance.get("model_stage") in {"degraded", "retired"}:
        blockers.append("model_stage_not_canary_eligible")
    return list(dict.fromkeys(blockers))


def _live_route_blockers(
    context: dict[str, Any],
    competition: dict[str, Any],
    coverage: dict[str, Any],
    governance: dict[str, Any],
) -> list[str]:
    blockers = _canary_route_blockers(context, competition, coverage, governance)
    if governance.get("model_stage") != "live":
        blockers.append("model_stage_not_live")
    if governance.get("requires_walk_forward") and governance.get("training_mode") != "walk_forward":
        blockers.append("walk_forward_required")
    if not governance.get("live_mutation"):
        blockers.append("live_mutation_not_enabled")
    return list(dict.fromkeys(blockers))


def _ml_readiness_blocks_route(context: dict[str, Any]) -> bool:
    readiness = _safe_dict(_safe_dict(context.get("ml_signal")).get("readiness"))
    if readiness and readiness.get("allow_live_position_influence") is False:
        return True
    if _safe_dict(context.get("readiness")).get("allow_live_position_influence") is False:
        return True
    return False


def _event_or_sentiment_required(features: Any) -> bool:
    if bool(getattr(features, "sentiment_data_available", False)):
        return True
    if bool(getattr(features, "direct_sentiment_data_available", False)):
        return True
    if int(_safe_float(getattr(features, "news_article_count", 0), 0.0)) > 0:
        return True
    if int(_safe_float(getattr(features, "direct_news_item_count", 0), 0.0)) > 0:
        return True
    return bool(getattr(features, "recent_news_items", None))


def _high_risk_market(features: Any, context: dict[str, Any]) -> bool:
    if str(context.get("market_risk_level") or "").lower() in {"high", "critical"}:
        return True
    wick_count = int(_safe_float(getattr(features, "abnormal_wick_count_72h", 0), 0.0))
    wick_max = _safe_float(getattr(features, "abnormal_wick_max_pct", 0.0), 0.0)
    volatility = _safe_float(getattr(features, "volatility_20", 0.0), 0.0)
    imbalance = abs(_safe_float(getattr(features, "orderbook_imbalance", 0.0), 0.0))
    return bool(wick_count > 0 or wick_max >= 6.0 or volatility >= 0.06 or imbalance >= 0.70)


def _has_open_position_context(context: dict[str, Any]) -> bool:
    if context.get("review_positions"):
        return True
    positions = context.get("open_positions")
    return isinstance(positions, list) and bool(positions)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
