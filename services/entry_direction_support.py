"""Independent directional confirmation for executable entries.

Quant models may propose a side, but correlated outputs from the same runtime
family count once.  Executable entries also require directional confirmation
from the expert analysis; all-HOLD analysis remains Shadow-only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from math import isfinite
from typing import Any

INDEPENDENT_DIRECTION_SUPPORT_VERSION = (
    "2026-07-27.independent-direction-support.v2"
)
MIN_ALIGNED_EXPERT_COUNT = 2
MIN_INDEPENDENT_SUPPORT_GROUP_COUNT = 2
UNPROMOTED_MODEL_INTERVENTION_SCOPE = "bounded_unpromoted_model_intervention"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _quant_family(source: str) -> str:
    normalized = str(source or "").strip().lower()
    if normalized == "local_ml":
        return "local_ml"
    if normalized in {"server_profit", "timeseries", "sentiment"}:
        return "local_ai_tools"
    return normalized


def _expert_group(item: dict[str, Any]) -> str:
    source_group = str(item.get("source_group") or "").strip()
    if source_group:
        return source_group
    provider = str(item.get("provider_model") or "").strip()
    if provider:
        return f"llm:{provider}"
    return f"expert:{item.get('model_name') or 'unknown'}"


def _weight(value: Any) -> float:
    parsed = _float(value)
    return parsed if parsed is not None and parsed > 0.0 else 1.0


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        (_float(item.get(key)), _weight(item.get("continuous_weight_multiplier")))
        for item in rows
    ]
    eligible = [(value, weight) for value, weight in values if value is not None]
    total_weight = sum(weight for _, weight in eligible)
    if total_weight <= 0.0:
        return None
    return sum(float(value) * weight for value, weight in eligible) / total_weight


def _quant_family_summaries(
    evidence_rows: list[dict[str, Any]],
    execution_cost_pct: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in evidence_rows:
        if not isinstance(item, dict):
            continue
        family = _quant_family(str(item.get("source") or ""))
        if not family:
            continue
        raw_expected = _float(item.get("raw_expected_return_pct"))
        objective_expected = _float(item.get("objective_expected_return_pct"))
        horizon = _float(item.get("horizon_minutes"))
        if raw_expected is None or objective_expected is None or not horizon or horizon <= 0:
            continue
        grouped.setdefault(family, []).append(item)

    summaries: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        raw_expected = _weighted_mean(rows, "raw_expected_return_pct")
        objective_expected = _weighted_mean(rows, "objective_expected_return_pct")
        loss_probability = _weighted_mean(
            [
                {
                    **item,
                    "tail_loss_probability": _dict(
                        item.get("return_distribution_contract")
                    ).get("tail_loss_probability"),
                }
                for item in rows
            ],
            "tail_loss_probability",
        )
        horizons = [
            value
            for item in rows
            if (value := _float(item.get("horizon_minutes"))) is not None and value > 0
        ]
        if raw_expected is None or objective_expected is None or not horizons:
            continue
        summaries.append(
            {
                "family": family,
                "sources": sorted(
                    {str(item.get("source") or "").strip() for item in rows}
                ),
                "raw_expected_return_pct": raw_expected,
                "objective_expected_return_pct": objective_expected,
                "expected_net_return_pct": raw_expected - execution_cost_pct,
                "objective_net_return_pct": objective_expected - execution_cost_pct,
                "loss_probability": loss_probability,
                "horizon_minutes": min(horizons),
            }
        )
    return summaries


def summarize_unpromoted_quantitative_evidence(
    direction_competition: dict[str, Any] | None,
    selected_side: str,
    *,
    execution_cost_pct: float | None,
) -> dict[str, Any]:
    """Build the cost-complete model summary that experts may inspect pre-decision."""

    side = str(selected_side or "").lower()
    competition = _dict(direction_competition)
    evidence_rows = _dict(competition.get(side)).get("evidence")
    if not isinstance(evidence_rows, list):
        evidence_rows = []

    parsed_cost = _float(execution_cost_pct)
    execution_cost_complete = bool(parsed_cost is not None and parsed_cost > 0.0)
    applied_cost = float(parsed_cost or 0.0)
    family_summaries = _quant_family_summaries(evidence_rows, applied_cost)
    positive_families = [
        item
        for item in family_summaries
        if (_float(item.get("expected_net_return_pct")) or 0.0) > 0.0
    ]
    loss_probabilities = [
        float(value)
        for item in family_summaries
        if (value := _float(item.get("loss_probability"))) is not None
    ]
    horizons = [
        float(item["horizon_minutes"])
        for item in family_summaries
        if (_float(item.get("horizon_minutes")) or 0.0) > 0.0
    ]
    expected_net_return_pct = (
        sum(float(item["raw_expected_return_pct"]) for item in family_summaries)
        / len(family_summaries)
        - applied_cost
        if family_summaries
        else None
    )
    objective_net_return_pct = (
        sum(float(item["objective_expected_return_pct"]) for item in family_summaries)
        / len(family_summaries)
        - applied_cost
        if family_summaries
        else None
    )
    return {
        "version": INDEPENDENT_DIRECTION_SUPPORT_VERSION,
        "scope": UNPROMOTED_MODEL_INTERVENTION_SCOPE,
        "selected_side": side if side in {"long", "short"} else "neutral",
        "execution_cost_pct": parsed_cost,
        "execution_cost_complete": execution_cost_complete,
        "expected_net_return_pct": expected_net_return_pct,
        "objective_net_return_pct": objective_net_return_pct,
        "loss_probability": (
            sum(loss_probabilities) / len(loss_probabilities)
            if loss_probabilities
            else None
        ),
        "prediction_horizon_minutes": min(horizons) if horizons else None,
        "positive_quant_sources": sorted(
            {
                source
                for item in positive_families
                for source in item.get("sources") or []
            }
        ),
        "quant_evidence_families": sorted(
            str(item["family"]) for item in positive_families
        ),
        "quant_family_summaries": family_summaries,
        "diagnostic_only": True,
        "production_permission": False,
    }


def _fingerprint_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "version",
            "support_scope",
            "eligible",
            "selected_side",
            "execution_cost_pct",
            "execution_cost_complete",
            "expected_net_return_pct",
            "objective_net_return_pct",
            "loss_probability",
            "prediction_horizon_minutes",
            "positive_quant_sources",
            "quant_evidence_families",
            "quant_family_summaries",
            "aligned_expert_count",
            "opposition_expert_count",
            "hold_expert_count",
            "auditable_expert_count",
            "aligned_expert_groups",
            "opposition_expert_groups",
            "independent_support_groups",
            "independent_support_group_count",
            "blocking_reasons",
            "production_permission",
        )
    }


def assess_directional_entry_support(
    direction_competition: dict[str, Any] | None,
    expert_opinions: list[dict[str, Any]] | None,
    selected_side: str,
    *,
    support_scope: str = "governed_return_candidate",
    execution_cost_pct: float | None = None,
) -> dict[str, Any]:
    """Require positive quant evidence plus non-HOLD expert confirmation."""

    side = str(selected_side or "").lower()
    opposite_side = "short" if side == "long" else "long"
    unpromoted_scope = support_scope == UNPROMOTED_MODEL_INTERVENTION_SCOPE
    quantitative = summarize_unpromoted_quantitative_evidence(
        direction_competition,
        side,
        execution_cost_pct=execution_cost_pct,
    )
    parsed_cost = _float(quantitative.get("execution_cost_pct"))
    execution_cost_complete = quantitative.get("execution_cost_complete") is True
    family_summaries = list(quantitative.get("quant_family_summaries") or [])
    if unpromoted_scope:
        positive_families = [
            item
            for item in family_summaries
            if (_float(item.get("expected_net_return_pct")) or 0.0) > 0.0
        ]
    else:
        positive_families = [
            item
            for item in family_summaries
            if (_float(item.get("raw_expected_return_pct")) or 0.0) > 0.0
            and (_float(item.get("objective_expected_return_pct")) or 0.0) > 0.0
        ]
    positive_quant_sources = sorted(
        {
            source
            for item in positive_families
            for source in item.get("sources") or []
        }
    )
    quant_families = sorted(str(item["family"]) for item in positive_families)
    expected_net_return_pct = _float(quantitative.get("expected_net_return_pct"))
    objective_net_return_pct = _float(quantitative.get("objective_net_return_pct"))
    loss_probability = _float(quantitative.get("loss_probability"))
    prediction_horizon_minutes = _float(
        quantitative.get("prediction_horizon_minutes")
    )

    auditable_experts = [
        item
        for item in expert_opinions or []
        if isinstance(item, dict)
        and (_float(item.get("effective_weight")) or 0.0) > 0.0
        and item.get("trace_only_fallback") is not True
        and str(item.get("reasoning") or "").strip()
        and str(item.get("action") or "").lower() in {"long", "short", "hold"}
    ]
    aligned = [
        item for item in auditable_experts if str(item.get("action") or "").lower() == side
    ]
    opposition = [
        item
        for item in auditable_experts
        if str(item.get("action") or "").lower() == opposite_side
    ]
    holds = [
        item
        for item in auditable_experts
        if str(item.get("action") or "").lower() == "hold"
    ]
    aligned_groups = sorted({_expert_group(item) for item in aligned})
    opposition_groups = sorted({_expert_group(item) for item in opposition})
    support_groups = sorted(
        {f"quant:{family}" for family in quant_families}
        | {f"expert:{group}" for group in aligned_groups}
    )

    blockers: list[str] = []
    if side not in {"long", "short"}:
        blockers.append("direction_support_side_missing")
    if unpromoted_scope and not execution_cost_complete:
        blockers.append("direction_support_execution_cost_incomplete")
    if unpromoted_scope and (expected_net_return_pct is None or expected_net_return_pct <= 0.0):
        blockers.append("direction_support_expected_net_return_not_positive")
    if not quant_families:
        blockers.append("direction_support_positive_quant_evidence_missing")
    if len(auditable_experts) < 3:
        blockers.append("direction_support_expert_analysis_incomplete")
    if auditable_experts and len(holds) == len(auditable_experts):
        blockers.append("direction_support_experts_all_hold")
    if len(aligned) < MIN_ALIGNED_EXPERT_COUNT:
        blockers.append("direction_support_aligned_experts_insufficient")
    if len(aligned) <= len(opposition):
        blockers.append("direction_support_expert_opposition_not_resolved")
    if len(support_groups) < MIN_INDEPENDENT_SUPPORT_GROUP_COUNT:
        blockers.append("direction_support_independent_groups_insufficient")

    result = {
        "version": INDEPENDENT_DIRECTION_SUPPORT_VERSION,
        "support_scope": support_scope,
        "eligible": not blockers,
        "reason": (
            "independent_positive_direction_support_ready"
            if not blockers
            else blockers[0]
        ),
        "selected_side": side if side in {"long", "short"} else "neutral",
        "execution_cost_pct": parsed_cost,
        "execution_cost_complete": execution_cost_complete,
        "expected_net_return_pct": expected_net_return_pct,
        "objective_net_return_pct": objective_net_return_pct,
        "loss_probability": loss_probability,
        "prediction_horizon_minutes": prediction_horizon_minutes,
        "positive_quant_sources": positive_quant_sources,
        "quant_evidence_families": quant_families,
        "quant_family_summaries": family_summaries,
        "aligned_expert_count": len(aligned),
        "opposition_expert_count": len(opposition),
        "hold_expert_count": len(holds),
        "auditable_expert_count": len(auditable_experts),
        "aligned_expert_groups": aligned_groups,
        "opposition_expert_groups": opposition_groups,
        "independent_support_groups": support_groups,
        "independent_support_group_count": len(support_groups),
        "blocking_reasons": list(dict.fromkeys(blockers)),
        "production_permission": False,
        "policy_provenance": {
            "source": "family_balanced_model_return_distributions_and_live_execution_cost",
            "observation_window": "current_pre_order_direction_support",
            "sample_count": len(family_summaries),
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": INDEPENDENT_DIRECTION_SUPPORT_VERSION,
            "fallback_reason": "",
        },
    }
    result["contract_fingerprint"] = _fingerprint(_fingerprint_payload(result))
    return result


def assess_unpromoted_model_intervention_support(
    direction_competition: dict[str, Any] | None,
    expert_opinions: list[dict[str, Any]] | None,
    selected_side: str,
    *,
    execution_cost_pct: float | None,
) -> dict[str, Any]:
    return assess_directional_entry_support(
        direction_competition,
        expert_opinions,
        selected_side,
        support_scope=UNPROMOTED_MODEL_INTERVENTION_SCOPE,
        execution_cost_pct=execution_cost_pct,
    )


def directional_entry_support_reasons(value: Any, selected_side: str) -> list[str]:
    support = _dict(value)
    reasons: list[str] = []
    if support.get("version") != INDEPENDENT_DIRECTION_SUPPORT_VERSION:
        reasons.append("direction_support_version_invalid")
    if support.get("eligible") is not True:
        reasons.append("direction_support_not_eligible")
    if support.get("selected_side") != str(selected_side or "").lower():
        reasons.append("direction_support_side_mismatch")
    if support.get("support_scope") == UNPROMOTED_MODEL_INTERVENTION_SCOPE:
        if support.get("execution_cost_complete") is not True:
            reasons.append("direction_support_execution_cost_incomplete")
        expected_net = _float(support.get("expected_net_return_pct"))
        if expected_net is None or expected_net <= 0.0:
            reasons.append("direction_support_expected_net_return_not_positive")
    if not support.get("quant_evidence_families"):
        reasons.append("direction_support_positive_quant_evidence_missing")
    if int(support.get("aligned_expert_count") or 0) < MIN_ALIGNED_EXPERT_COUNT:
        reasons.append("direction_support_aligned_experts_insufficient")
    if int(support.get("aligned_expert_count") or 0) <= int(
        support.get("opposition_expert_count") or 0
    ):
        reasons.append("direction_support_expert_opposition_not_resolved")
    if int(support.get("independent_support_group_count") or 0) < (
        MIN_INDEPENDENT_SUPPORT_GROUP_COUNT
    ):
        reasons.append("direction_support_independent_groups_insufficient")
    if support.get("blocking_reasons"):
        reasons.append("direction_support_contains_blockers")
    if support.get("contract_fingerprint") != _fingerprint(
        _fingerprint_payload(support)
    ):
        reasons.append("direction_support_fingerprint_mismatch")
    return list(dict.fromkeys(reasons))
