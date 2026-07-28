"""Auditable model direction and independent expert-conflict assessment."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from math import isfinite
from typing import Any

INDEPENDENT_DIRECTION_SUPPORT_VERSION = "2026-07-28.paper-model-direction.v5"
PAPER_MODEL_TRADE_SCOPE = "paper_model_trade"
MIN_GOVERNED_ALIGNED_EXPERT_COUNT = 2
MIN_GOVERNED_INDEPENDENT_SUPPORT_GROUP_COUNT = 2


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
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for item in evidence_rows:
        if not isinstance(item, dict):
            continue
        if item.get("decision_eligible") is not True:
            continue
        family = _quant_family(str(item.get("source") or ""))
        if not family:
            continue
        raw_expected = _float(item.get("raw_expected_return_pct"))
        objective_expected = _float(item.get("objective_expected_return_pct"))
        horizon = _float(item.get("horizon_minutes"))
        if raw_expected is None or objective_expected is None or not horizon or horizon <= 0:
            continue
        grouped.setdefault((family, horizon), []).append(item)

    summaries: list[dict[str, Any]] = []
    for (family, horizon), rows in sorted(grouped.items()):
        raw_expected = _weighted_mean(rows, "raw_expected_return_pct")
        objective_expected = _weighted_mean(rows, "objective_expected_return_pct")
        explicit_net_expected = _weighted_mean(rows, "expected_net_return_pct")
        loss_probability = _weighted_mean(
            [
                {
                    **item,
                    "resolved_loss_probability": (
                        item.get("loss_probability")
                        if _float(item.get("loss_probability")) is not None
                        else _dict(item.get("return_distribution_contract")).get(
                            "tail_loss_probability"
                        )
                    ),
                }
                for item in rows
            ],
            "resolved_loss_probability",
        )
        if raw_expected is None or objective_expected is None:
            continue
        summaries.append(
            {
                "family": family,
                "sources": sorted(
                    {str(item.get("source") or "").strip() for item in rows}
                ),
                "raw_expected_return_pct": raw_expected,
                "objective_expected_return_pct": objective_expected,
                "expected_net_return_pct": (
                    explicit_net_expected
                    if explicit_net_expected is not None
                    else raw_expected - execution_cost_pct
                ),
                "objective_net_return_pct": objective_expected - execution_cost_pct,
                "loss_probability": loss_probability,
                "horizon_minutes": horizon,
            }
        )
    return summaries


def summarize_paper_quantitative_evidence(
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
    by_horizon: dict[float, list[dict[str, Any]]] = {}
    for item in family_summaries:
        horizon = _float(item.get("horizon_minutes"))
        if horizon is not None and horizon > 0.0:
            by_horizon.setdefault(horizon, []).append(item)
    horizon_candidates = [
        {
            "horizon_minutes": horizon,
            "family_summaries": rows,
            "expected_net_return_pct": sum(
                float(item["expected_net_return_pct"]) for item in rows
            )
            / len(rows),
            "objective_net_return_pct": sum(
                float(item["objective_net_return_pct"]) for item in rows
            )
            / len(rows),
            "loss_probability": _weighted_mean(
                [
                    {
                        "loss_probability": item.get("loss_probability"),
                        "continuous_weight_multiplier": 1.0,
                    }
                    for item in rows
                ],
                "loss_probability",
            ),
        }
        for horizon, rows in sorted(by_horizon.items())
        if rows
    ]
    cohort_selection = _dict(competition.get("horizon_cohort_selection"))
    cohort_horizon = _float(
        cohort_selection.get("selected_horizon_minutes")
        or competition.get("selected_horizon_minutes")
    )
    selected_horizon = next(
        (
            item
            for item in horizon_candidates
            if _float(item.get("horizon_minutes")) == cohort_horizon
        ),
        None,
    )
    if selected_horizon is None and cohort_horizon is None:
        selected_horizon = max(
            horizon_candidates,
            key=lambda item: (
                float(item["expected_net_return_pct"]),
                float(item["objective_net_return_pct"]),
                len(item["family_summaries"]),
                -float(item["loss_probability"] or 1.0),
                -float(item["horizon_minutes"]),
            ),
        ) if horizon_candidates else None
    family_summaries = list(
        _dict(selected_horizon).get("family_summaries") or []
    )
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
    expected_net_return_pct = (
        sum(float(item["expected_net_return_pct"]) for item in family_summaries)
        / len(family_summaries)
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
        "scope": PAPER_MODEL_TRADE_SCOPE,
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
        "prediction_horizon_minutes": _dict(selected_horizon).get(
            "horizon_minutes"
        ),
        "available_prediction_horizons": [
            item.get("horizon_minutes")
            for item in cohort_selection.get("available_horizon_groups") or []
            if _float(_dict(item).get("horizon_minutes")) is not None
        ] or sorted(by_horizon),
        "horizon_selection_policy": (
            cohort_selection.get("selection_reason")
            or "best_fee_after_return_coherent_horizon"
        ),
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
            "available_prediction_horizons",
            "horizon_selection_policy",
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
            "strong_expert_opposition",
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
    """Require one auditable model direction and block only strong opposition."""

    side = str(selected_side or "").lower()
    opposite_side = "short" if side == "long" else "long"
    paper_scope = support_scope == PAPER_MODEL_TRADE_SCOPE
    quantitative = summarize_paper_quantitative_evidence(
        direction_competition,
        side,
        execution_cost_pct=execution_cost_pct,
    )
    parsed_cost = _float(quantitative.get("execution_cost_pct"))
    execution_cost_complete = quantitative.get("execution_cost_complete") is True
    family_summaries = list(quantitative.get("quant_family_summaries") or [])
    if paper_scope:
        directional_families = [
            item
            for item in family_summaries
            if (_float(item.get("expected_net_return_pct")) or 0.0) > 0.0
        ]
    else:
        directional_families = [
            item
            for item in family_summaries
            if (_float(item.get("raw_expected_return_pct")) or 0.0) > 0.0
            and (_float(item.get("objective_expected_return_pct")) or 0.0) > 0.0
        ]
    quantitative_sources = sorted(
        {
            source
            for item in directional_families
            for source in item.get("sources") or []
        }
    )
    quant_families = sorted(str(item["family"]) for item in directional_families)
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
    strong_expert_opposition = bool(
        len(opposition_groups) >= 2 and len(opposition) > len(aligned)
    )

    blockers: list[str] = []
    if side not in {"long", "short"}:
        blockers.append("direction_support_side_missing")
    if paper_scope and not execution_cost_complete:
        blockers.append("direction_support_execution_cost_incomplete")
    if paper_scope and (
        expected_net_return_pct is None or expected_net_return_pct <= 0.0
    ):
        blockers.append("direction_support_expected_net_not_positive")
    if not quant_families:
        blockers.append("direction_support_quant_evidence_missing")
    if prediction_horizon_minutes is None or prediction_horizon_minutes <= 0.0:
        blockers.append("direction_support_prediction_horizon_missing")
    if paper_scope:
        if strong_expert_opposition:
            blockers.append("direction_support_strong_expert_opposition")
    else:
        if len(auditable_experts) < 3:
            blockers.append("direction_support_expert_analysis_incomplete")
        if auditable_experts and len(holds) == len(auditable_experts):
            blockers.append("direction_support_experts_all_hold")
        if len(aligned) < MIN_GOVERNED_ALIGNED_EXPERT_COUNT:
            blockers.append("direction_support_aligned_experts_insufficient")
        if len(aligned) <= len(opposition):
            blockers.append("direction_support_expert_opposition_not_resolved")
        if len(support_groups) < MIN_GOVERNED_INDEPENDENT_SUPPORT_GROUP_COUNT:
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
        "available_prediction_horizons": list(
            quantitative.get("available_prediction_horizons") or []
        ),
        "horizon_selection_policy": quantitative.get(
            "horizon_selection_policy"
        ),
        "positive_quant_sources": quantitative_sources,
        "quantitative_sources": quantitative_sources,
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
        "strong_expert_opposition": strong_expert_opposition,
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


def assess_paper_model_trade_support(
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
        support_scope=PAPER_MODEL_TRADE_SCOPE,
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
    if support.get("support_scope") == PAPER_MODEL_TRADE_SCOPE:
        if support.get("execution_cost_complete") is not True:
            reasons.append("direction_support_execution_cost_incomplete")
        expected_net = _float(support.get("expected_net_return_pct"))
        if expected_net is None or expected_net <= 0.0:
            reasons.append("direction_support_expected_net_not_positive")
    if not support.get("quant_evidence_families"):
        reasons.append("direction_support_quant_evidence_missing")
    horizon = _float(support.get("prediction_horizon_minutes"))
    if horizon is None or horizon <= 0.0:
        reasons.append("direction_support_prediction_horizon_missing")
    if support.get("support_scope") == PAPER_MODEL_TRADE_SCOPE:
        if support.get("strong_expert_opposition") is True:
            reasons.append("direction_support_strong_expert_opposition")
    else:
        if int(support.get("aligned_expert_count") or 0) < (
            MIN_GOVERNED_ALIGNED_EXPERT_COUNT
        ):
            reasons.append("direction_support_aligned_experts_insufficient")
        if int(support.get("aligned_expert_count") or 0) <= int(
            support.get("opposition_expert_count") or 0
        ):
            reasons.append("direction_support_expert_opposition_not_resolved")
        if int(support.get("independent_support_group_count") or 0) < (
            MIN_GOVERNED_INDEPENDENT_SUPPORT_GROUP_COUNT
        ):
            reasons.append("direction_support_independent_groups_insufficient")
    if support.get("blocking_reasons"):
        reasons.append("direction_support_contains_blockers")
    if support.get("contract_fingerprint") != _fingerprint(
        _fingerprint_payload(support)
    ):
        reasons.append("direction_support_fingerprint_mismatch")
    return list(dict.fromkeys(reasons))
