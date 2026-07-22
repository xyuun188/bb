"""Observation of long-vs-short gross market opportunity from governed models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.entry_signal_extraction import (
    first_tool_payload,
    signal_available,
    signal_production_eligibility,
    signal_return_distribution,
    signal_return_distribution_eligibility,
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _side_summary(values: list[dict[str, Any]]) -> dict[str, Any]:
    eligible_objective = [
        float(item["objective_expected_return_pct"])
        for item in values
        if item.get("production_eligible") is True
        and _safe_float(item.get("objective_expected_return_pct")) is not None
    ]
    eligible_raw = [
        float(item["raw_expected_return_pct"])
        for item in values
        if item.get("production_eligible") is True
        and _safe_float(item.get("raw_expected_return_pct")) is not None
    ]
    return {
        "score": (
            sum(eligible_objective) / len(eligible_objective)
            if eligible_objective
            else 0.0
        ),
        "raw_expected_return_pct": (
            sum(eligible_raw) / len(eligible_raw) if eligible_raw else None
        ),
        "objective_expected_return_pct": (
            sum(eligible_objective) / len(eligible_objective)
            if eligible_objective
            else None
        ),
        "production_source_count": len(eligible_objective),
        "evidence": values,
    }


def _training_side_summary(values: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize directional observations without requiring promotion permission."""

    objective_values = [
        float(item["objective_expected_return_pct"])
        for item in values
        if _safe_float(item.get("objective_expected_return_pct")) is not None
    ]
    raw_values = [
        float(item["raw_expected_return_pct"])
        for item in values
        if _safe_float(item.get("raw_expected_return_pct")) is not None
    ]
    horizon_values = [
        float(item["horizon_minutes"])
        for item in values
        if (_safe_float(item.get("horizon_minutes")) or 0.0) > 0
    ]
    selected = objective_values or raw_values
    return {
        "score": sum(selected) / len(selected) if selected else None,
        "objective_expected_return_pct": (
            sum(objective_values) / len(objective_values)
            if objective_values
            else None
        ),
        "raw_expected_return_pct": (
            sum(raw_values) / len(raw_values) if raw_values else None
        ),
        "horizon_minutes": min(horizon_values) if horizon_values else None,
        "horizon_source_count": len(horizon_values),
        "observation_count": len(selected),
    }


def _enforce_aggregate_contract_consistency(
    evidence: dict[str, list[dict[str, Any]]],
) -> list[str]:
    eligible = [
        item
        for side in ("long", "short")
        for item in evidence[side]
        if item.get("production_eligible") is True
    ]
    signatures = {
        (
            item.get("objective_version"),
            item.get("label_version"),
            item.get("cost_model_version"),
            item.get("profit_supervision_version"),
            item.get("horizon_minutes"),
        )
        for item in eligible
    }
    if len(signatures) <= 1:
        return []
    fields = (
        "objective_version",
        "label_version",
        "cost_model_version",
        "profit_supervision_version",
        "horizon_minutes",
    )
    blockers = [
        f"direction_competition_{field}_mismatch"
        for index, field in enumerate(fields)
        if len({signature[index] for signature in signatures}) > 1
    ]
    for item in eligible:
        item["production_eligible"] = False
        item["observation_only"] = True
        item["eligibility_reason"] = blockers[0]
    return blockers


@dataclass(frozen=True, slots=True)
class EntryDirectionCompetitionPolicy:
    """Compare only governed gross market-opportunity observations.

    This context may guide the model toward the better side, but it cannot grant
    execution permission. The selected side must still pass the live realized-net
    return, cost, validity, sizing, account, and exchange contracts.
    """

    def context(
        self,
        feature_vector: Any,
        ml_signal_context: dict[str, Any] | None,
        local_ai_tools_context: dict[str, Any] | None,
        market_regime: dict[str, Any] | None,
        strategy_mode: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del feature_vector, market_regime, strategy_mode
        evidence = {"long": [], "short": []}
        self._append_local_ml(evidence, ml_signal_context)
        self._append_server_tool(
            evidence,
            local_ai_tools_context,
            key="server_profit",
            aliases=("profit_prediction", "profit_model", "server_profit", "server_profit_model"),
        )
        self._append_server_tool(
            evidence,
            local_ai_tools_context,
            key="timeseries",
            aliases=(
                "time_series_prediction",
                "timeseries_prediction",
                "sequence_prediction",
                "timeseries",
                "time_series",
            ),
        )
        aggregate_blockers = _enforce_aggregate_contract_consistency(evidence)
        long_side = _side_summary(evidence["long"])
        short_side = _side_summary(evidence["short"])
        long_training = _training_side_summary(evidence["long"])
        short_training = _training_side_summary(evidence["short"])
        long_score = float(long_side["score"])
        short_score = float(short_side["score"])
        source_count = int(long_side["production_source_count"]) + int(
            short_side["production_source_count"]
        )
        preferred_side = (
            "neutral"
            if source_count <= 0 or long_score == short_score
            else "long"
            if long_score > short_score
            else "short"
        )
        training_scores = {
            "long": long_training.get("score"),
            "short": short_training.get("score"),
        }
        available_training_scores = {
            side: score
            for side, score in training_scores.items()
            if _safe_float(score) is not None
            and (_safe_float(
                (long_training if side == "long" else short_training).get(
                    "horizon_minutes"
                )
            ) or 0.0)
            > 0
        }
        training_preferred_side = "neutral"
        if available_training_scores:
            training_preferred_side = max(
                available_training_scores,
                key=lambda side: float(available_training_scores[side]),
            )
            if len(available_training_scores) == 2 and (
                available_training_scores["long"] == available_training_scores["short"]
            ):
                training_preferred_side = "neutral"
        return {
            "enabled": bool(source_count),
            "preferred_side": preferred_side,
            "score_gap": abs(long_score - short_score),
            "long": long_side,
            "short": short_side,
            "training_preferred_side": training_preferred_side,
            "training_long": long_training,
            "training_short": short_training,
            "training_permission": False,
            "production_source_count": source_count,
            "production_permission": False,
            "policy": "governed_gross_market_observation_only_no_fixed_gap",
            "aggregate_blockers": aggregate_blockers,
            "policy_provenance": {
                "source": "live_influence_gross_market_models",
                "observation_window": "current_decision_model_outputs",
                "sample_count": source_count,
                "strategy_version": "2026-07-14.gross-market-direction-observation.v2",
                "fallback_reason": "" if source_count else "governed_return_models_unavailable",
            },
        }

    @staticmethod
    def _append_local_ml(
        evidence: dict[str, list[dict[str, Any]]],
        ml_signal_context: dict[str, Any] | None,
    ) -> None:
        signal = _safe_dict(ml_signal_context)
        predictions = _safe_list(signal.get("predictions"))
        primary = _safe_dict(predictions[0] if predictions else {})
        influence = _safe_dict(signal.get("influence_policy"))
        eligibility = signal_production_eligibility(signal)
        for side in ("long", "short"):
            side_policy = _safe_dict(influence.get(side))
            distribution_eligibility = signal_return_distribution_eligibility(
                signal,
                side,
            )
            contract = signal_return_distribution(signal, side)
            eligible = bool(
                eligibility.get("eligible") is True
                and side_policy.get("enabled") is True
                and distribution_eligibility.get("eligible") is True
            )
            evidence[side].append(
                {
                    "source": "local_ml",
                    "side": side,
                    "available": bool(primary),
                    "production_eligible": eligible,
                    "observation_only": bool(contract and not eligible),
                    "eligibility_reason": (
                        "live_influence_and_side_readiness_confirmed"
                        if eligible
                        else str(
                            eligibility.get("reason")
                            or "local_ml_production_governance_incomplete"
                        )
                    ),
                    "raw_expected_return_pct": contract.get(
                        "raw_expected_return_pct"
                    ),
                    "objective_expected_return_pct": contract.get(
                        "objective_expected_return_pct"
                    ),
                    "horizon_minutes": contract.get("horizon_minutes"),
                    "objective_version": contract.get("objective_version"),
                    "label_version": contract.get("label_version"),
                    "cost_model_version": contract.get("cost_model_version"),
                    "profit_supervision_version": contract.get(
                        "profit_supervision_version"
                    ),
                    "return_distribution_contract": contract,
                }
            )

    @staticmethod
    def _append_server_tool(
        evidence: dict[str, list[dict[str, Any]]],
        local_ai_tools_context: dict[str, Any] | None,
        *,
        key: str,
        aliases: tuple[str, ...],
    ) -> None:
        tools = _safe_dict(local_ai_tools_context)
        payload = first_tool_payload({"local_ai_tools": tools}, *aliases)
        eligibility = signal_production_eligibility(payload)
        for side in ("long", "short"):
            distribution_eligibility = signal_return_distribution_eligibility(
                payload,
                side,
            )
            contract = signal_return_distribution(payload, side)
            eligible = bool(
                eligibility.get("eligible") is True
                and distribution_eligibility.get("eligible") is True
            )
            evidence[side].append(
                {
                    "source": key,
                    "side": side,
                    "available": signal_available(payload),
                    "production_eligible": eligible,
                    "observation_only": bool(contract and not eligible),
                    "eligibility_reason": eligibility.get("reason"),
                    "raw_expected_return_pct": contract.get(
                        "raw_expected_return_pct"
                    ),
                    "objective_expected_return_pct": contract.get(
                        "objective_expected_return_pct"
                    ),
                    "horizon_minutes": contract.get("horizon_minutes"),
                    "objective_version": contract.get("objective_version"),
                    "label_version": contract.get("label_version"),
                    "cost_model_version": contract.get("cost_model_version"),
                    "profit_supervision_version": contract.get(
                        "profit_supervision_version"
                    ),
                    "return_distribution_contract": contract,
                    "route_mode": payload.get("route_mode"),
                    "model": payload.get("primary_model") or payload.get("model"),
                }
            )
