from __future__ import annotations

from collections import Counter
from typing import Any

from services.trade_execution_contract import validate_entry_execution_contract

ALL_ENTRY_EXPERTS = (
    "trend_expert",
    "momentum_expert",
    "sentiment_expert",
    "position_expert",
    "risk_expert",
)
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

    del features, health
    expert_reasons = {
        name: ["full_governed_expert_set_preserved"] for name in ALL_ENTRY_EXPERTS
    }

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
    selected_list = list(ALL_ENTRY_EXPERTS)
    skipped_list: list[str] = []
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
        "mandatory_safety_experts": ["risk_expert"],
        "estimated_call_reduction": 0,
        "blocking_reasons": blocking_reasons,
        "expert_reasons": {name: reasons for name, reasons in expert_reasons.items() if reasons},
        "routing_basis": {
            "candidate_quality": str(ctx.get("candidate_quality") or "unknown"),
            "feature_coverage_status": coverage.get("status"),
        },
        "training_governance": governance,
        "safety_rules": [
            "routing_report_observation_only",
            "all_governed_experts_preserved",
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
    ineligible_return_contract_executed_count = 0
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
            _, contract_blockers = validate_entry_execution_contract(raw)
            if contract_blockers:
                ineligible_return_contract_executed_count += 1
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
            "ineligible_return_contract_executed_count": (
                ineligible_return_contract_executed_count
            ),
            "negative_executed_count": negative_executed_count,
            "route_change_increased_return_contract_gaps": False,
        },
        "recent_routes": route_plans[:20],
        "safety_rules": [
            "routing_report_is_read_only",
            "unsafe_live_mutation_is_never_honored",
            "ineligible_return_contracts_require_observation_before_canary",
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
    if _paper_canary_readiness_blocks_route(context):
        blockers.append("paper_canary_readiness_blocked")
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
    if _ml_readiness_blocks_route(context):
        blockers.append("ml_readiness_blocks_live_route")
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


def _paper_canary_readiness_blocks_route(context: dict[str, Any]) -> bool:
    readiness = _safe_dict(_safe_dict(context.get("ml_signal")).get("readiness"))
    paper = _safe_dict(readiness.get("paper_canary"))
    if paper:
        return paper.get("authorized") is not True
    return bool(readiness and readiness.get("allow_live_position_influence") is False)


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
