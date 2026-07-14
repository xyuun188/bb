"""Authoritative realized-net-return opportunity aggregation.

Only governed models may contribute gross market-opportunity observations.
Shadow and recovery predictions remain visible observations with zero production
weight. Production realized-net return combines the governed market distribution,
current executable cost, counterfactual cost uncertainty, and authoritative OKX
trade slippage calibration. AI confidence, expert votes, and memory cannot alter
that distribution or grant production permission.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite, sqrt
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_signal_extraction import (
    first_tool_payload,
    payload_side,
    safe_dict,
    safe_float,
    safe_list,
    signal_available,
    signal_production_eligibility,
)
from services.execution_cost_model import execution_cost_estimate
from services.profit_supervision import (
    PRODUCTION_RETURN_COMBINATION_VERSION,
    PROFIT_SUPERVISION_VERSION,
)

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


def _distribution_ready(
    distribution: dict[str, Any],
    *fields: str,
) -> bool:
    return bool(
        distribution
        and all(_finite(distribution.get(field)) is not None for field in fields)
    )


def _unique_distribution_rows(
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        signature = tuple(row.get(field) for field in fields)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(row)
    return unique


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
            and primary.get("profit_supervision_version")
            == PROFIT_SUPERVISION_VERSION
            and primary.get("return_semantics")
            == "gross_market_opportunity_before_execution"
            and primary.get(f"{side}_market_distribution_ready") is True
        )
        value = _finite(primary.get(f"{side}_market_expected_return_pct"))
        lower_bound = _finite(
            primary.get(
                f"{side}_market_lower_hinge_return_pct",
                primary.get(f"{side}_lower_quantile_return_pct"),
            )
        )
        tail_probability = _finite(primary.get(f"{side}_tail_loss_probability"))
        horizon_minutes = _finite(
            primary.get("horizon_minutes", signal.get("primary_horizon_minutes"))
        )
        if value is None:
            production_eligible = False
        observation_only = bool(not production_eligible and primary and value is not None)
        cost_distribution = safe_dict(
            safe_dict(primary.get("counterfactual_execution_cost_distribution")).get(
                side
            )
        )
        if cost_distribution.get("distribution_ready") is not True:
            production_eligible = False
        actual_calibration = safe_dict(
            safe_dict(primary.get("actual_trade_calibration")).get(side)
        )
        return {
            "key": "local_ml",
            "available": bool(primary),
            "production_eligible": production_eligible,
            "observation_only": observation_only,
            "eligibility_reason": (
                "live_influence_and_side_readiness_confirmed"
                if production_eligible
                else "runtime_prediction_observation_only"
                if observation_only
                else "local_ml_production_governance_incomplete"
            ),
            "side": side,
            "raw_market_return_pct": value,
            "raw_return_pct": value,
            "lower_bound_return_pct": lower_bound,
            "loss_probability": (
                min(max(tail_probability, 0.0), 1.0)
                if tail_probability is not None
                else None
            ),
            "horizon_minutes": horizon_minutes,
            "counterfactual_execution_cost_distribution": cost_distribution,
            "actual_trade_calibration": actual_calibration,
            "profit_supervision_version": primary.get(
                "profit_supervision_version"
            ),
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
        value = _finite(
            payload.get(
                f"{side}_market_expected_return_pct",
                payload.get("market_expected_return_pct"),
            )
        )
        lower_bound = _finite(
            payload.get(
                f"{side}_lower_bound_return_pct",
                payload.get(f"{side}_lower_quantile_return_pct"),
            )
        )
        horizon_minutes = _finite(
            payload.get("horizon_minutes", payload.get("primary_horizon_minutes"))
        )
        production_eligible = bool(
            eligibility.get("eligible")
            and value is not None
            and payload.get("profit_supervision_version")
            == PROFIT_SUPERVISION_VERSION
            and payload.get("return_semantics")
            == "gross_market_opportunity_before_execution"
        )
        observation_only = bool(
            not production_eligible
            and signal_available(payload)
            and value is not None
            and horizon_minutes is not None
            and horizon_minutes > 0
        )
        return {
            "key": key,
            "available": signal_available(payload),
            "production_eligible": production_eligible,
            "observation_only": observation_only,
            "eligibility_reason": (
                eligibility.get("reason")
                if not observation_only
                else "runtime_prediction_observation_only"
            ),
            "side": payload_side(payload) or "unknown",
            "raw_market_return_pct": value,
            "raw_return_pct": value,
            "lower_bound_return_pct": lower_bound,
            "loss_probability": _loss_probability(payload, side),
            "horizon_minutes": horizon_minutes,
            "counterfactual_execution_cost_distribution": safe_dict(
                safe_dict(
                    payload.get("counterfactual_execution_cost_distribution")
                ).get(side)
            ),
            "actual_trade_calibration": safe_dict(
                safe_dict(payload.get("actual_trade_calibration")).get(side)
            ),
            "profit_supervision_version": payload.get(
                "profit_supervision_version"
            ),
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
        distribution_mode = (
            "governed_market_opportunity"
            if governed_components
            else "unavailable"
        )
        selected_components = governed_components
        production_weight = 1.0 / len(selected_components) if selected_components else 0.0
        for component in components:
            component["included_in_return_distribution"] = component in selected_components
            component["production_weight"] = (
                production_weight if component in selected_components else 0.0
            )

        observations = [
            float(component["raw_market_return_pct"])
            for component in selected_components
            if component.get("raw_market_return_pct") is not None
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
        cost_distributions = _unique_distribution_rows(
            [
                safe_dict(component.get("counterfactual_execution_cost_distribution"))
                for component in selected_components
                if _distribution_ready(
                    safe_dict(
                        component.get("counterfactual_execution_cost_distribution")
                    ),
                    "expected_pct",
                    "upper_tail_pct",
                    "uncertainty_pct",
                )
                and safe_dict(
                    component.get("counterfactual_execution_cost_distribution")
                ).get("source_authority")
                == "shadow_counterfactual_live_microstructure"
                and safe_dict(
                    component.get("counterfactual_execution_cost_distribution")
                ).get("distribution_ready")
                is True
            ],
            fields=(
                "expected_pct",
                "upper_tail_pct",
                "uncertainty_pct",
                "source_authority",
            ),
        )
        calibrations = [
            safe_dict(component.get("actual_trade_calibration"))
            for component in selected_components
            if safe_dict(component.get("actual_trade_calibration")).get(
                "source_authority"
            )
            == "okx_position_history"
            and safe_dict(component.get("actual_trade_calibration")).get("side")
            == side
            and safe_dict(component.get("actual_trade_calibration")).get(
                "profile_source"
            )
            in {"symbol_side", "global_side"}
            and (
                (
                    safe_dict(component.get("actual_trade_calibration")).get(
                        "profile_source"
                    )
                    == "global_side"
                    and safe_dict(component.get("actual_trade_calibration")).get(
                        "symbol"
                    )
                    == "*"
                )
                or (
                    safe_dict(component.get("actual_trade_calibration")).get(
                        "profile_source"
                    )
                    == "symbol_side"
                    and self.normalize_symbol(
                        str(
                            safe_dict(
                                component.get("actual_trade_calibration")
                            ).get("symbol")
                            or ""
                        )
                    )
                    == self.normalize_symbol(decision.symbol)
                )
            )
        ]
        if any(row.get("profile_source") == "symbol_side" for row in calibrations):
            calibrations = [
                row for row in calibrations if row.get("profile_source") == "symbol_side"
            ]
        calibrations = _unique_distribution_rows(
            [
                {
                    **row,
                    "_net_count": safe_dict(
                        row.get("net_return_after_cost_pct")
                    ).get("count"),
                    "_net_expected": safe_dict(
                        row.get("net_return_after_cost_pct")
                    ).get("expected"),
                    "_net_lower_hinge": safe_dict(
                        row.get("net_return_after_cost_pct")
                    ).get("lower_hinge"),
                    "_slippage_count": safe_dict(row.get("slippage_pct")).get(
                        "count"
                    ),
                    "_slippage_expected": safe_dict(
                        row.get("slippage_pct")
                    ).get("expected"),
                    "_slippage_upper_hinge": safe_dict(
                        row.get("slippage_pct")
                    ).get("upper_hinge"),
                }
                for row in calibrations
            ],
            fields=(
                "symbol",
                "side",
                "profile_source",
                "source_authority",
                "_net_count",
                "_net_expected",
                "_net_lower_hinge",
                "_slippage_count",
                "_slippage_expected",
                "_slippage_upper_hinge",
            ),
        )
        valid_calibrations: list[dict[str, Any]] = []
        for calibration in calibrations:
            net_distribution = safe_dict(
                calibration.get("net_return_after_cost_pct")
            )
            slippage_distribution = safe_dict(calibration.get("slippage_pct"))
            if (
                safe_float(net_distribution.get("count"), 0.0) > 0
                and safe_float(slippage_distribution.get("count"), 0.0) > 0
                and _distribution_ready(
                    net_distribution,
                    "expected",
                    "lower_hinge",
                )
                and _distribution_ready(
                    slippage_distribution,
                    "expected",
                    "upper_hinge",
                )
            ):
                valid_calibrations.append(calibration)

        gross_return = _mean(observations) if observations else None
        market_uncertainty = (
            _sampling_uncertainty(observations, gross_return)
            if observations and gross_return is not None
            else None
        )
        if market_uncertainty is not None and lower_bounds:
            market_uncertainty = max(
                market_uncertainty,
                gross_return - min(lower_bounds),
            )
        historical_cost_expected = (
            _mean([float(row["expected_pct"]) for row in cost_distributions])
            if cost_distributions
            else None
        )
        historical_cost_uncertainty = (
            max(
                max(
                    float(row["upper_tail_pct"]) - float(row["expected_pct"]),
                    float(row["uncertainty_pct"]),
                    0.0,
                )
                for row in cost_distributions
            )
            if cost_distributions
            else None
        )
        actual_net_expected = (
            _mean(
                [
                    float(
                        safe_dict(row.get("net_return_after_cost_pct"))["expected"]
                    )
                    for row in valid_calibrations
                ]
            )
            if valid_calibrations
            else None
        )
        actual_net_lower_hinge = (
            min(
                float(
                    safe_dict(row.get("net_return_after_cost_pct"))["lower_hinge"]
                )
                for row in valid_calibrations
            )
            if valid_calibrations
            else None
        )
        authoritative_slippage_expected = (
            _mean(
                [
                    float(safe_dict(row.get("slippage_pct"))["expected"])
                    for row in valid_calibrations
                ]
            )
            if valid_calibrations
            else None
        )
        authoritative_slippage_upper_hinge = (
            max(
                float(safe_dict(row.get("slippage_pct"))["upper_hinge"])
                for row in valid_calibrations
            )
            if valid_calibrations
            else None
        )
        live_cost_pct = execution_cost.total_pct if execution_cost.production_eligible else None
        slippage_tail_excess = (
            max(authoritative_slippage_upper_hinge - execution_cost.slippage_pct, 0.0)
            if authoritative_slippage_upper_hinge is not None
            and execution_cost.production_eligible
            else None
        )
        actual_calibration_uncertainty = (
            max(actual_net_expected - actual_net_lower_hinge, 0.0)
            if actual_net_expected is not None and actual_net_lower_hinge is not None
            else None
        )
        combination_ready = bool(
            observations
            and valid_for_seconds > 0
            and execution_cost.production_eligible
            and cost_distributions
            and valid_calibrations
            and gross_return is not None
            and market_uncertainty is not None
            and historical_cost_uncertainty is not None
            and slippage_tail_excess is not None
            and actual_calibration_uncertainty is not None
        )
        expected_net = (
            gross_return - live_cost_pct - slippage_tail_excess
            if combination_ready
            and gross_return is not None
            and live_cost_pct is not None
            and slippage_tail_excess is not None
            else None
        )
        uncertainty = (
            market_uncertainty
            + historical_cost_uncertainty
            + actual_calibration_uncertainty
            if combination_ready
            and market_uncertainty is not None
            and historical_cost_uncertainty is not None
            and actual_calibration_uncertainty is not None
            else None
        )
        return_lcb = (
            expected_net - uncertainty
            if expected_net is not None and uncertainty is not None
            else None
        )
        downside_observations = [max(-value, 0.0) for value in observations]
        expected_loss = (
            _mean(downside_observations)
            + max(-(actual_net_expected or 0.0), 0.0)
            if combination_ready
            else None
        )
        score = (
            return_lcb - expected_loss
            if return_lcb is not None and expected_loss is not None
            else float("-inf")
        )
        loss_probability = _mean(loss_probabilities) if loss_probabilities else 1.0
        tail_risk = max(loss_probabilities) if loss_probabilities else 1.0
        profit_quality = (
            expected_net / max(expected_loss + uncertainty, 1e-12)
            if expected_net is not None
            and expected_loss is not None
            and uncertainty is not None
            and expected_net > 0
            else None
        )
        generated_at = datetime.now(UTC).isoformat()
        blockers: list[str] = []
        if not observations:
            blockers.append("governed_market_opportunity_distribution_missing")
        if valid_for_seconds <= 0:
            blockers.append("governed_market_horizon_missing")
        if not execution_cost.production_eligible:
            blockers.append("live_execution_cost_distribution_missing")
        if not cost_distributions:
            blockers.append("counterfactual_execution_cost_distribution_missing")
        if not valid_calibrations:
            blockers.append("authoritative_realized_return_or_slippage_distribution_missing")
        provenance = {
            "source": (
                "governed_market_live_cost_and_okx_trade_calibration"
                if combination_ready
                else "return_distribution_unavailable"
            ),
            "observation_window": (
                "current_governed_model_outputs_orderbook_and_authoritative_trade_history"
            ),
            "sample_count": len(observations),
            "generated_at": generated_at,
            "strategy_version": PRODUCTION_RETURN_COMBINATION_VERSION,
            "fallback_reason": ",".join(blockers),
            "valid_for_seconds": round(valid_for_seconds, 8),
            "return_distribution_mode": distribution_mode,
            "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
            "return_combination_version": PRODUCTION_RETURN_COMBINATION_VERSION,
        }
        raw["opportunity_score"] = {
            "score": round(score, 8) if isfinite(score) else None,
            "side": side,
            "expected_gross_return_pct": (
                round(gross_return, 8) if gross_return is not None else None
            ),
            "expected_realized_net_return_pct": (
                round(expected_net, 8) if expected_net is not None else None
            ),
            "expected_net_return_pct": (
                round(expected_net, 8) if expected_net is not None else None
            ),
            "realized_net_lcb_pct": (
                round(return_lcb, 8) if return_lcb is not None else None
            ),
            "return_lcb_pct": (
                round(return_lcb, 8) if return_lcb is not None else None
            ),
            "return_uncertainty_pct": (
                round(uncertainty, 8) if uncertainty is not None else None
            ),
            "expected_loss_pct": (
                round(expected_loss, 8) if expected_loss is not None else None
            ),
            "profit_quality_ratio": (
                round(profit_quality, 8) if profit_quality is not None else None
            ),
            "server_profit_loss_probability": round(loss_probability, 8),
            "tail_risk_score": round(tail_risk, 8),
            "score_policy": "realized_net_lcb_minus_calibrated_downside",
            "return_distribution_mode": distribution_mode,
            "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
            "return_combination_version": PRODUCTION_RETURN_COMBINATION_VERSION,
            "execution_cost": execution_cost.to_dict(),
            "expected_net_breakdown": {
                "formula": (
                    "mean(governed_gross_market_returns)-live_execution_cost-"
                    "authoritative_slippage_tail_excess"
                ),
                "unit": "pct",
                "components": components,
                "net_pct": round(expected_net, 8) if expected_net is not None else None,
                "model_gross_pct": (
                    round(gross_return, 8) if gross_return is not None else None
                ),
                "live_execution_cost_pct": (
                    round(live_cost_pct, 8) if live_cost_pct is not None else None
                ),
                "historical_counterfactual_cost_expected_pct": (
                    round(historical_cost_expected, 8)
                    if historical_cost_expected is not None
                    else None
                ),
                "historical_counterfactual_cost_uncertainty_pct": (
                    round(historical_cost_uncertainty, 8)
                    if historical_cost_uncertainty is not None
                    else None
                ),
                "authoritative_slippage_expected_pct": (
                    round(authoritative_slippage_expected, 8)
                    if authoritative_slippage_expected is not None
                    else None
                ),
                "authoritative_slippage_upper_hinge_pct": (
                    round(authoritative_slippage_upper_hinge, 8)
                    if authoritative_slippage_upper_hinge is not None
                    else None
                ),
                "authoritative_slippage_tail_excess_pct": (
                    round(slippage_tail_excess, 8)
                    if slippage_tail_excess is not None
                    else None
                ),
                "authoritative_realized_net_expected_pct": (
                    round(actual_net_expected, 8)
                    if actual_net_expected is not None
                    else None
                ),
                "authoritative_realized_net_lower_hinge_pct": (
                    round(actual_net_lower_hinge, 8)
                    if actual_net_lower_hinge is not None
                    else None
                ),
                "market_uncertainty_pct": (
                    round(market_uncertainty, 8)
                    if market_uncertainty is not None
                    else None
                ),
                "actual_trade_calibration_uncertainty_pct": (
                    round(actual_calibration_uncertainty, 8)
                    if actual_calibration_uncertainty is not None
                    else None
                ),
                "counterfactual_cost_distribution_count": len(cost_distributions),
                "authoritative_trade_calibration_count": len(valid_calibrations),
                "cost_deduction_count": 1 if combination_ready else 0,
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
            "production_eligible": combination_ready,
            "policy_provenance": provenance,
            "strategy_context_observation_only": safe_dict(strategy),
        }
        decision.raw_response = raw
        self.annotate_decision_source(decision)
        return score
