"""Authoritative fee-after-return opportunity aggregation.

Governed models are preferred. When governance has not promoted any model, a
trained runtime prediction may form a recovery distribution only while its
return objective and prediction-quality contracts remain intact. AI confidence,
expert votes, shadow memory, and other advisory context cannot alter expected
net return, ranking, sizing, leverage, or execution permission.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite, sqrt
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_signal_extraction import (
    directional_expected_return_pct,
    first_tool_payload,
    payload_side,
    safe_dict,
    safe_float,
    safe_list,
    signal_available,
    signal_production_eligibility,
    signal_runtime_recovery_eligibility,
)
from services.execution_cost_model import execution_cost_estimate

NormalizeSymbol = Callable[[str | None], str]
DecisionAnnotator = Callable[[DecisionOutput], None]


def _finite(value: Any) -> float | None:
    number = safe_float(value, float("nan"))
    return number if isfinite(number) else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sampling_uncertainty(values: list[float], center: float) -> float:
    if len(values) <= 1:
        return abs(values[0] - center) if values else 0.0
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return sqrt(max(variance, 0.0) / len(values))


_LOCAL_ML_RECOVERY_PERFORMANCE_BLOCKERS = {
    f"{side}_{suffix}"
    for side in ("long", "short")
    for suffix in (
        "top_return_not_above_bottom",
        "top_return_lcb_not_positive",
        "top_profit_factor_not_above_one",
        "top_tail_loss_not_improved",
    )
}


def _local_ml_runtime_recovery_eligible(
    signal: dict[str, Any],
    *,
    primary: dict[str, Any],
    value: float | None,
    horizon_minutes: float | None,
) -> bool:
    """Keep artifact/data contracts while allowing current-return recovery."""

    readiness = safe_dict(signal.get("readiness"))
    provenance = safe_dict(readiness.get("policy_provenance"))
    blockers = safe_list(readiness.get("blocking_reasons"))
    blocker_codes = {
        str(safe_dict(blocker).get("code") or "") for blocker in blockers
    }
    return bool(
        signal.get("available") is True
        and primary
        and value is not None
        and horizon_minutes is not None
        and horizon_minutes > 0
        and safe_float(signal.get("trained_sample_count"), 0.0) > 0
        and str(signal.get("model_version") or "").strip()
        and safe_float(provenance.get("sample_count"), 0.0) > 0
        and safe_float(provenance.get("test_sample_count"), 0.0) > 0
        and not str(provenance.get("fallback_reason") or "").strip()
        and blocker_codes.issubset(_LOCAL_ML_RECOVERY_PERFORMANCE_BLOCKERS)
    )


def _loss_probability(payload: dict[str, Any], side: str) -> float | None:
    candidates = (
        payload.get(f"{side}_loss_probability"),
        payload.get("loss_probability"),
        payload.get("tail_loss_probability"),
    )
    for value in candidates:
        number = _finite(value)
        if number is not None:
            return min(max(number, 0.0), 1.0)
    return None


@dataclass(slots=True)
class EntryOpportunityScoringPolicy:
    """Build and rank the current production return distribution."""

    normalize_symbol: NormalizeSymbol
    annotate_decision_source: DecisionAnnotator

    def _local_ml_component(
        self,
        raw: dict[str, Any],
        side: str,
    ) -> dict[str, Any]:
        signal = safe_dict(raw.get("ml_signal"))
        predictions = safe_list(signal.get("predictions"))
        primary = safe_dict(predictions[0] if predictions else {})
        influence = safe_dict(signal.get("influence_policy"))
        side_policy = safe_dict(influence.get(side))
        production_eligible = bool(
            signal.get("allow_live_position_influence") is True
            and signal.get("influence_enabled") is True
            and side_policy.get("enabled") is True
            and primary
        )
        value = _finite(primary.get(f"{side}_expected_return_pct"))
        lower_bound = _finite(primary.get(f"{side}_lower_quantile_return_pct"))
        tail_probability = _finite(primary.get(f"{side}_tail_loss_probability"))
        horizon_minutes = _finite(
            primary.get("horizon_minutes", signal.get("primary_horizon_minutes"))
        )
        if value is None:
            production_eligible = False
        recovery_observation_eligible = bool(
            not production_eligible
            and _local_ml_runtime_recovery_eligible(
                signal,
                primary=primary,
                value=value,
                horizon_minutes=horizon_minutes,
            )
        )
        return {
            "key": "local_ml",
            "available": bool(primary),
            "production_eligible": production_eligible,
            "recovery_observation_eligible": recovery_observation_eligible,
            "eligibility_reason": (
                "live_influence_and_side_readiness_confirmed"
                if production_eligible
                else "runtime_prediction_available_for_recovery_distribution"
                if recovery_observation_eligible
                else "local_ml_production_governance_incomplete"
            ),
            "side": side,
            "raw_return_pct": value,
            "lower_bound_return_pct": lower_bound,
            "loss_probability": (
                min(max(tail_probability, 0.0), 1.0)
                if tail_probability is not None
                else None
            ),
            "horizon_minutes": horizon_minutes,
        }

    @staticmethod
    def _server_component(
        raw: dict[str, Any],
        *,
        key: str,
        side: str,
        aliases: tuple[str, ...],
    ) -> dict[str, Any]:
        payload = first_tool_payload(raw, *aliases)
        eligibility = signal_production_eligibility(payload)
        value = _finite(directional_expected_return_pct(payload, side))
        lower_bound = _finite(
            payload.get(
                f"{side}_lower_bound_return_pct",
                payload.get(f"{side}_lower_quantile_return_pct"),
            )
        )
        horizon_minutes = _finite(
            payload.get("horizon_minutes", payload.get("primary_horizon_minutes"))
        )
        production_eligible = bool(eligibility.get("eligible") and value is not None)
        recovery_eligibility = signal_runtime_recovery_eligibility(payload)
        recovery_observation_eligible = bool(
            not production_eligible
            and recovery_eligibility.get("eligible") is True
            and value is not None
            and horizon_minutes is not None
            and horizon_minutes > 0
        )
        return {
            "key": key,
            "available": signal_available(payload),
            "production_eligible": production_eligible,
            "recovery_observation_eligible": recovery_observation_eligible,
            "eligibility_reason": (
                eligibility.get("reason")
                if not recovery_observation_eligible
                else recovery_eligibility.get("reason")
            ),
            "side": payload_side(payload) or "unknown",
            "raw_return_pct": value,
            "lower_bound_return_pct": lower_bound,
            "loss_probability": _loss_probability(payload, side),
            "horizon_minutes": horizon_minutes,
        }

    def score_candidate(
        self,
        decision: DecisionOutput,
        strategy: dict[str, Any] | None = None,
    ) -> float:
        if not decision.is_entry:
            return float("-inf")

        side = "long" if decision.action == Action.LONG else "short"
        raw = safe_dict(decision.raw_response)
        execution_cost = execution_cost_estimate(
            decision.feature_snapshot if isinstance(decision.feature_snapshot, dict) else {}
        )
        components = [
            self._local_ml_component(raw, side),
            self._server_component(
                raw,
                key="server_profit",
                side=side,
                aliases=(
                    "profit_prediction",
                    "profit_model",
                    "server_profit",
                    "server_profit_model",
                    "profit",
                ),
            ),
            self._server_component(
                raw,
                key="timeseries",
                side=side,
                aliases=(
                    "time_series_prediction",
                    "timeseries_prediction",
                    "sequence_prediction",
                    "timeseries",
                    "time_series",
                ),
            ),
        ]
        governed_components = [
            component for component in components if component["production_eligible"]
        ]
        recovery_components = [
            component
            for component in components
            if component.get("recovery_observation_eligible") is True
        ]
        distribution_mode = (
            "governed_models"
            if governed_components
            else "runtime_recovery"
            if recovery_components
            else "unavailable"
        )
        selected_components = governed_components or recovery_components
        for component in components:
            component["included_in_return_distribution"] = component in selected_components

        observations = [
            float(component["raw_return_pct"])
            for component in selected_components
            if component.get("raw_return_pct") is not None
        ]
        lower_bounds = [
            float(component["lower_bound_return_pct"])
            for component in selected_components
            if component.get("lower_bound_return_pct") is not None
        ]
        loss_probabilities = [
            float(component["loss_probability"])
            for component in selected_components
            if component.get("loss_probability") is not None
        ]
        horizons = [
            float(component["horizon_minutes"])
            for component in selected_components
            if _finite(component.get("horizon_minutes")) is not None
            and float(component["horizon_minutes"]) > 0
        ]
        valid_for_seconds = min(horizons) * 60.0 if horizons else 0.0
        cost_pct = execution_cost.total_pct if execution_cost.production_eligible else 0.0
        gross_return = _mean(observations) if observations else 0.0
        expected_net = gross_return - cost_pct if observations else 0.0
        uncertainty = _sampling_uncertainty(observations, gross_return)
        if lower_bounds:
            uncertainty = max(uncertainty, gross_return - min(lower_bounds))
        downside_observations = [max(-value, 0.0) for value in observations]
        expected_loss = (
            _mean(downside_observations) + cost_pct if observations else 0.0
        )
        return_lcb = expected_net - uncertainty
        score = return_lcb - expected_loss
        loss_probability = _mean(loss_probabilities) if loss_probabilities else 1.0
        tail_risk = max(loss_probabilities) if loss_probabilities else 1.0
        profit_quality = (
            expected_net / max(expected_loss, cost_pct, 1e-12)
            if expected_net > 0 and execution_cost.production_eligible
            else 0.0
        )
        generated_at = datetime.now(UTC).isoformat()
        provenance = {
            "source": (
                "explicitly_production_eligible_return_models_and_live_execution_cost"
                if distribution_mode == "governed_models"
                else "trained_runtime_recovery_predictions_and_live_execution_cost"
                if distribution_mode == "runtime_recovery"
                else "return_distribution_unavailable"
            ),
            "observation_window": "current_decision_model_outputs_and_orderbook_snapshot",
            "sample_count": len(observations),
            "generated_at": generated_at,
            "strategy_version": "2026-07-12.authoritative-return-opportunity.v1",
            "fallback_reason": (
                ""
                if observations and execution_cost.production_eligible and valid_for_seconds > 0
                else "production_return_cost_or_validity_distribution_missing"
            ),
            "valid_for_seconds": round(valid_for_seconds, 8),
            "return_distribution_mode": distribution_mode,
        }
        raw["opportunity_score"] = {
            "score": round(score, 8),
            "side": side,
            "expected_gross_return_pct": round(gross_return, 8),
            "expected_net_return_pct": round(expected_net, 8),
            "return_lcb_pct": round(return_lcb, 8),
            "return_uncertainty_pct": round(uncertainty, 8),
            "expected_loss_pct": round(expected_loss, 8),
            "profit_quality_ratio": round(profit_quality, 8),
            "server_profit_loss_probability": round(loss_probability, 8),
            "tail_risk_score": round(tail_risk, 8),
            "score_policy": "fee_after_return_lcb_minus_expected_downside",
            "return_distribution_mode": distribution_mode,
            "execution_cost": execution_cost.to_dict(),
            "expected_net_breakdown": {
                "formula": "mean(selected_runtime_returns)-live_execution_cost",
                "unit": "pct",
                "components": components,
                "net_pct": round(expected_net, 8),
                "model_gross_pct": round(gross_return, 8),
                "observed_not_in_formula": {
                    "ai_confidence": safe_float(decision.confidence, 0.0),
                    "experts": safe_list(raw.get("experts")),
                    "memory_feedback": safe_dict(raw.get("memory_feedback")),
                    "sentiment": first_tool_payload(
                        raw,
                        "sentiment_analysis",
                        "sentiment_prediction",
                        "sentiment_model",
                        "sentiment",
                    ),
                },
            },
            "production_eligible": bool(
                observations and execution_cost.production_eligible and valid_for_seconds > 0
            ),
            "policy_provenance": provenance,
            "strategy_context_observation_only": safe_dict(strategy),
        }
        decision.raw_response = raw
        self.annotate_decision_source(decision)
        return score
