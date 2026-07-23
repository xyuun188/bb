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
from math import isfinite
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
    signal_return_distribution,
    signal_return_distribution_eligibility,
)
from services.execution_cost_model import execution_cost_estimate
from services.model_strategy_blueprint import paper_strategy_authorization
from services.paper_bootstrap_canary import annotate_paper_bootstrap_opportunity
from services.profit_supervision import (
    PRODUCTION_RETURN_COMBINATION_VERSION,
    PROFIT_SUPERVISION_VERSION,
)
from services.return_objective import combine_production_return_distribution

NormalizeSymbol = Callable[[str | None], str]
DecisionAnnotator = Callable[[DecisionOutput], None]


def _finite(value: Any) -> float | None:
    number = safe_float(value, float("nan"))
    return number if isfinite(number) else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


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


def _paper_quant_multipliers(strategy: dict[str, Any] | None) -> dict[str, float]:
    context = safe_dict(strategy)
    if str(context.get("execution_mode") or "").lower() != "paper":
        return {}
    report = safe_dict(context.get("continuous_model_weights"))
    if report.get("applied") is not True:
        return {}
    rows = safe_dict(report.get("quant_source_weights"))
    weights: dict[str, float] = {}
    for name in ("local_ml", "server_profit", "timeseries"):
        value = _finite(safe_dict(rows.get(name)).get("effective_multiplier"))
        if value is not None and value > 0.0:
            weights[name] = value
    return weights


@dataclass(slots=True)
class EntryOpportunityScoringPolicy:
    """Build and rank the current production return distribution."""

    normalize_symbol: NormalizeSymbol
    annotate_decision_source: DecisionAnnotator

    def _local_ml_component(
        self,
        raw: dict[str, Any],
        side: str,
        *,
        strategy: dict[str, Any] | None,
        symbol: str,
    ) -> dict[str, Any]:
        signal = safe_dict(raw.get("ml_signal"))
        predictions = safe_list(signal.get("predictions"))
        primary = safe_dict(predictions[0] if predictions else {})
        influence = safe_dict(signal.get("influence_policy"))
        side_policy = safe_dict(influence.get(side))
        live_claimed = bool(
            signal.get("allow_live_position_influence") is True
            and signal.get("influence_enabled") is True
            and side_policy.get("enabled") is True
        )
        paper_authorization = paper_strategy_authorization(
            strategy,
            signal,
            symbol=symbol,
            side=side,
        )
        paper_claimed = paper_authorization.get("eligible") is True
        production_claimed = bool(live_claimed or paper_claimed)
        governance = signal_production_eligibility(signal)
        distribution_eligibility = signal_return_distribution_eligibility(
            signal,
            side,
        )
        contract = signal_return_distribution(signal, side)
        production_eligible = bool(
            production_claimed
            and (governance.get("eligible") is True or paper_claimed)
            and distribution_eligibility.get("eligible") is True
        )
        observation_only = bool(not production_eligible and primary and contract)
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
            "production_claimed": production_claimed,
            "production_eligible": production_eligible,
            "observation_only": observation_only,
            "eligibility_reason": (
                "active_trained_model_paper_strategy"
                if production_eligible and paper_claimed
                else "standardized_distribution_and_side_readiness_confirmed"
                if production_eligible
                else str(
                    distribution_eligibility.get("reason")
                    or governance.get("reason")
                    or "local_ml_production_governance_incomplete"
                )
            ),
            "side": side,
            "execution_scope": "paper_only" if paper_claimed else "live",
            "paper_strategy_authorization": paper_authorization,
            "return_distribution_contract": contract,
            "raw_market_return_pct": contract.get("raw_expected_return_pct"),
            "raw_return_pct": contract.get("raw_expected_return_pct"),
            "objective_expected_return_pct": contract.get(
                "objective_expected_return_pct"
            ),
            "lower_bound_return_pct": contract.get(
                "lower_quantile_return_pct"
            ),
            "loss_probability": contract.get("tail_loss_probability"),
            "horizon_minutes": contract.get("horizon_minutes"),
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
        governance = signal_production_eligibility(payload)
        distribution_eligibility = signal_return_distribution_eligibility(
            payload,
            side,
        )
        contract = signal_return_distribution(payload, side)
        production_claimed = bool(
            signal_available(payload)
            and str(payload.get("route_mode") or "").lower() == "live"
            and any(
                payload.get(field) is True
                for field in (
                    "live_mutation",
                    "live_influence",
                    "influence_enabled",
                    "allow_live_position_influence",
                )
            )
        )
        production_eligible = bool(
            production_claimed
            and governance.get("eligible") is True
            and distribution_eligibility.get("eligible") is True
        )
        observation_only = bool(
            not production_eligible
            and signal_available(payload)
            and contract
        )
        return {
            "key": key,
            "available": signal_available(payload),
            "production_claimed": production_claimed,
            "production_eligible": production_eligible,
            "observation_only": observation_only,
            "eligibility_reason": str(
                distribution_eligibility.get("reason")
                or governance.get("reason")
                or "server_model_production_governance_incomplete"
            ),
            "side": side,
            "reported_best_side": payload_side(payload) or "unknown",
            "return_distribution_contract": contract,
            "raw_market_return_pct": contract.get("raw_expected_return_pct"),
            "raw_return_pct": contract.get("raw_expected_return_pct"),
            "objective_expected_return_pct": contract.get(
                "objective_expected_return_pct"
            ),
            "lower_bound_return_pct": contract.get(
                "lower_quantile_return_pct"
            ),
            "loss_probability": contract.get("tail_loss_probability"),
            "horizon_minutes": contract.get("horizon_minutes"),
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
            self._local_ml_component(
                raw,
                side,
                strategy=strategy,
                symbol=decision.symbol,
            ),
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
        selected_components = [
            component for component in components if component["production_eligible"]
        ]
        claimed_components = [
            component for component in components if component["production_claimed"]
        ]
        claimed_contracts = [
            safe_dict(component.get("return_distribution_contract"))
            for component in claimed_components
            if safe_dict(component.get("return_distribution_contract"))
        ]
        quant_multipliers = _paper_quant_multipliers(strategy)
        claimed_model_weights = (
            [
                quant_multipliers.get(str(component.get("key") or ""), 1.0)
                for component in claimed_components
                if safe_dict(component.get("return_distribution_contract"))
            ]
            if quant_multipliers
            else None
        )
        input_blockers = []
        if len(claimed_contracts) != len(claimed_components):
            input_blockers.append("claimed_profit_target_distribution_missing")
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
                        row.get("net_return_after_all_cost_pct")
                    ).get("count"),
                    "_net_expected": safe_dict(
                        row.get("net_return_after_all_cost_pct")
                    ).get("expected"),
                    "_net_lower_hinge": safe_dict(
                        row.get("net_return_after_all_cost_pct")
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
        valid_calibrations = calibrations
        historical_cost_expected = (
            _mean([float(row["expected_pct"]) for row in cost_distributions])
            if cost_distributions
            else None
        )
        actual_net_expected = (
            _mean(
                [
                    float(
                        safe_dict(row.get("net_return_after_all_cost_pct"))[
                            "expected"
                        ]
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
                    safe_dict(row.get("net_return_after_all_cost_pct"))[
                        "lower_hinge"
                    ]
                )
                for row in valid_calibrations
            )
            if valid_calibrations
            else None
        )
        production_distribution = combine_production_return_distribution(
            side=side,
            model_contracts=claimed_contracts,
            live_execution_cost_pct=(
                execution_cost.total_pct
                if execution_cost.production_eligible
                else None
            ),
            live_slippage_pct=(
                execution_cost.slippage_pct
                if execution_cost.production_eligible
                else None
            ),
            counterfactual_cost_distributions=cost_distributions,
            actual_trade_calibrations=valid_calibrations,
            profit_supervision_version=PROFIT_SUPERVISION_VERSION,
            source_authority=(
                "governed_model_contracts_live_orderbook_and_okx_position_history"
            ),
            input_blockers=input_blockers,
            model_weights=claimed_model_weights,
        )
        combination_ready = production_distribution.get("production_eligible") is True
        distribution_mode = (
            "governed_market_opportunity" if combination_ready else "unavailable"
        )
        selected_weight_total = sum(
            quant_multipliers.get(str(component.get("key") or ""), 1.0)
            for component in selected_components
        )
        for component in components:
            included = bool(combination_ready and component in selected_components)
            component["included_in_return_distribution"] = included
            component_weight = quant_multipliers.get(
                str(component.get("key") or ""),
                1.0,
            )
            component["production_weight"] = (
                component_weight / selected_weight_total
                if included and selected_weight_total > 0.0
                else 0.0
            )
            if quant_multipliers:
                component["continuous_weight_multiplier"] = component_weight
            if component in selected_components and not combination_ready:
                component["production_eligible"] = False
                component["observation_only"] = True
                component["eligibility_reason"] = (
                    "aggregate_return_distribution_contract_blocked"
                )

        gross_distribution = safe_dict(
            production_distribution.get("gross_market_distribution")
        )
        transformations = safe_dict(production_distribution.get("transformations"))
        gross_return = _finite(gross_distribution.get("raw_expected_return_pct"))
        valid_for_seconds = (
            float(production_distribution["horizon_minutes"]) * 60.0
            if _finite(production_distribution.get("horizon_minutes")) is not None
            else 0.0
        )
        expected_net = (
            _finite(production_distribution.get("raw_expected_return_pct"))
            if combination_ready
            else None
        )
        return_lcb = (
            _finite(production_distribution.get("objective_expected_return_pct"))
            if combination_ready
            else None
        )
        uncertainty = (
            _finite(production_distribution.get("uncertainty_penalty_pct"))
            if combination_ready
            else None
        )
        expected_loss = (
            _finite(production_distribution.get("tail_loss_penalty_pct"))
            if combination_ready
            else None
        )
        score = return_lcb if return_lcb is not None else float("-inf")
        loss_probability = _finite(
            production_distribution.get("tail_loss_probability")
        )
        tail_risk = loss_probability
        profit_quality = (
            return_lcb / max((expected_loss or 0.0) + (uncertainty or 0.0), 1e-12)
            if expected_net is not None
            and return_lcb is not None
            and return_lcb > 0
            else None
        )
        generated_at = datetime.now(UTC).isoformat()
        blockers = list(production_distribution.get("blockers") or [])
        provenance = {
            "source": (
                "governed_market_live_cost_and_okx_trade_calibration"
                if combination_ready
                else "return_distribution_unavailable"
            ),
            "observation_window": (
                "current_governed_model_outputs_orderbook_and_authoritative_trade_history"
            ),
            "sample_count": len(selected_components) if combination_ready else 0,
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
            "server_profit_loss_probability": (
                round(loss_probability, 8) if loss_probability is not None else None
            ),
            "tail_risk_score": (
                round(tail_risk, 8) if tail_risk is not None else None
            ),
            "score_policy": "standardized_objective_expected_return",
            "return_distribution_mode": distribution_mode,
            "return_distribution_contract": production_distribution,
            "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
            "return_combination_version": PRODUCTION_RETURN_COMBINATION_VERSION,
            "execution_cost": execution_cost.to_dict(),
            "expected_net_breakdown": {
                "formula": (
                    "weighted_mean(governed_gross_market_returns)-live_execution_cost-"
                    if quant_multipliers
                    else "mean(governed_gross_market_returns)-live_execution_cost-"
                ) + (
                    "authoritative_slippage_tail_excess"
                ),
                "unit": "pct",
                "components": components,
                "net_pct": round(expected_net, 8) if expected_net is not None else None,
                "model_gross_pct": (
                    round(gross_return, 8) if gross_return is not None else None
                ),
                "live_execution_cost_pct": transformations.get(
                    "live_execution_cost_pct"
                ),
                "historical_counterfactual_cost_expected_pct": (
                    round(historical_cost_expected, 8)
                    if historical_cost_expected is not None
                    else None
                ),
                "historical_counterfactual_cost_uncertainty_pct": transformations.get(
                    "counterfactual_cost_uncertainty_pct"
                ),
                "authoritative_slippage_expected_pct": transformations.get(
                    "authoritative_slippage_expected_pct"
                ),
                "authoritative_slippage_upper_hinge_pct": transformations.get(
                    "authoritative_slippage_upper_hinge_pct"
                ),
                "authoritative_slippage_tail_excess_pct": transformations.get(
                    "authoritative_slippage_tail_excess_pct"
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
                "market_uncertainty_pct": transformations.get(
                    "market_dispersion_pct"
                ),
                "actual_trade_calibration_uncertainty_pct": transformations.get(
                    "actual_trade_calibration_uncertainty_pct"
                ),
                "counterfactual_cost_distribution_count": len(cost_distributions),
                "authoritative_trade_calibration_count": len(valid_calibrations),
                "cost_deduction_count": transformations.get(
                    "cost_deduction_count",
                    0,
                ),
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
        paper_canary_score = annotate_paper_bootstrap_opportunity(decision)
        self.annotate_decision_source(decision)
        return paper_canary_score if paper_canary_score is not None else score
