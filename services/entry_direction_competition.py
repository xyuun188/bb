"""Execution-scoped long-vs-short gross market opportunity competition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.entry_signal_extraction import (
    first_tool_payload,
    signal_available,
    signal_paper_eligibility,
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


def _weight(value: Any) -> float:
    parsed = _safe_float(value, 1.0)
    return parsed if parsed is not None and parsed > 0.0 else 1.0


def _weighted_mean(values: list[tuple[float, float]]) -> float | None:
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0.0:
        return None
    return sum(value * weight for value, weight in values) / total_weight


def _paper_quant_weights(strategy_mode: dict[str, Any] | None) -> dict[str, float]:
    strategy = _safe_dict(strategy_mode)
    if str(strategy.get("execution_mode") or "").lower() != "paper":
        return {}
    report = _safe_dict(strategy.get("continuous_model_weights"))
    if report.get("applied") is not True:
        return {}
    rows = _safe_dict(report.get("quant_source_weights"))
    return {
        name: _weight(_safe_dict(row).get("effective_multiplier"))
        for name, row in rows.items()
        if name in {"local_ml", "server_profit", "timeseries", "sentiment"}
    }


def _execution_scope(strategy_mode: dict[str, Any] | None) -> str:
    mode = str(_safe_dict(strategy_mode).get("execution_mode") or "live").lower()
    return "paper" if mode == "paper" else "live"


def _side_summary(values: list[dict[str, Any]]) -> dict[str, Any]:
    eligible_objective = [
        (
            float(item["objective_expected_return_pct"]),
            _weight(item.get("continuous_weight_multiplier")),
        )
        for item in values
        if item.get("decision_eligible") is True
        and _safe_float(item.get("objective_expected_return_pct")) is not None
    ]
    eligible_raw = [
        (
            float(item["raw_expected_return_pct"]),
            _weight(item.get("continuous_weight_multiplier")),
        )
        for item in values
        if item.get("decision_eligible") is True
        and _safe_float(item.get("raw_expected_return_pct")) is not None
    ]
    return {
        "score": _weighted_mean(eligible_objective),
        "raw_expected_return_pct": _weighted_mean(eligible_raw),
        "objective_expected_return_pct": _weighted_mean(eligible_objective),
        "decision_source_count": len(eligible_objective),
        "evidence": values,
    }


def _training_side_summary(values: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize directional observations without requiring promotion permission."""

    objective_values = [
        (
            float(item["objective_expected_return_pct"]),
            _weight(item.get("continuous_weight_multiplier")),
        )
        for item in values
        if item.get("paper_eligible") is True
        and _safe_float(item.get("objective_expected_return_pct")) is not None
    ]
    raw_values = [
        (
            float(item["raw_expected_return_pct"]),
            _weight(item.get("continuous_weight_multiplier")),
        )
        for item in values
        if item.get("paper_eligible") is True
        and _safe_float(item.get("raw_expected_return_pct")) is not None
    ]
    horizon_values = [
        float(item["horizon_minutes"])
        for item in values
        if item.get("paper_eligible") is True
        and (_safe_float(item.get("horizon_minutes")) or 0.0) > 0
    ]
    selected = objective_values or raw_values
    return {
        "score": _weighted_mean(selected),
        "objective_expected_return_pct": _weighted_mean(objective_values),
        "raw_expected_return_pct": _weighted_mean(raw_values),
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
        if item.get("decision_eligible") is True
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
        item["decision_eligible"] = False
        item["aggregate_eligible"] = False
        item["observation_only"] = True
        item["eligibility_reason"] = blockers[0]
    return blockers


@dataclass(frozen=True, slots=True)
class EntryDirectionCompetitionPolicy:
    """Compare gross market-opportunity observations for the active scope.

    This context may guide the model toward the better side, but it cannot grant
    execution permission. The selected side still passes the execution-scoped net
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
        del feature_vector, market_regime
        execution_scope = _execution_scope(strategy_mode)
        evidence = {"long": [], "short": []}
        self._append_local_ml(evidence, ml_signal_context, execution_scope)
        self._append_server_tool(
            evidence,
            local_ai_tools_context,
            key="server_profit",
            aliases=("profit_prediction", "profit_model", "server_profit", "server_profit_model"),
            execution_scope=execution_scope,
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
            execution_scope=execution_scope,
        )
        quant_weights = _paper_quant_weights(strategy_mode)
        if quant_weights:
            for side in ("long", "short"):
                for item in evidence[side]:
                    source = str(item.get("source") or "")
                    item["continuous_weight_multiplier"] = quant_weights.get(source, 1.0)
        aggregate_blockers = _enforce_aggregate_contract_consistency(evidence)
        long_side = _side_summary(evidence["long"])
        short_side = _side_summary(evidence["short"])
        long_training = _training_side_summary(evidence["long"])
        short_training = _training_side_summary(evidence["short"])
        long_score = _safe_float(long_side.get("score"))
        short_score = _safe_float(short_side.get("score"))
        source_count = int(long_side["decision_source_count"]) + int(
            short_side["decision_source_count"]
        )
        production_source_count = sum(
            1
            for side in ("long", "short")
            for item in evidence[side]
            if item.get("production_eligible") is True
            and item.get("decision_eligible") is True
        )
        available_scores = {
            side: score
            for side, score in (("long", long_score), ("short", short_score))
            if score is not None
        }
        preferred_side = "neutral"
        if available_scores:
            preferred_side = max(
                available_scores,
                key=lambda side: float(available_scores[side]),
            )
            if len(available_scores) == 2 and long_score == short_score:
                preferred_side = "neutral"
        training_scores = {
            "long": long_training.get("score"),
            "short": short_training.get("score"),
        }
        available_training_scores = {
            side: score
            for side, score in training_scores.items()
            if _safe_float(score) is not None
            and float(score) > 0.0
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
        result = {
            "enabled": bool(source_count),
            "execution_scope": execution_scope,
            "preferred_side": preferred_side,
            "score_gap": (
                abs(float(long_score) - float(short_score))
                if long_score is not None and short_score is not None
                else None
            ),
            "long": long_side,
            "short": short_side,
            "training_preferred_side": training_preferred_side,
            "training_long": long_training,
            "training_short": short_training,
            "training_permission": False,
            "decision_source_count": source_count,
            "paper_source_count": source_count if execution_scope == "paper" else 0,
            "production_source_count": production_source_count,
            "production_permission": False,
            "policy": "execution_scoped_gross_market_observation_only_no_fixed_gap",
            "aggregate_blockers": aggregate_blockers,
            "policy_provenance": {
                "source": f"{execution_scope}_eligible_gross_market_models",
                "observation_window": "current_decision_model_outputs",
                "sample_count": source_count,
                "strategy_version": "2026-07-14.gross-market-direction-observation.v2",
                "fallback_reason": "" if source_count else "eligible_return_models_unavailable",
            },
        }
        if quant_weights:
            result["continuous_model_weighting"] = {
                "applied": True,
                "execution_scope": "paper_only",
                "weights": quant_weights,
                "fallback": "none",
            }
        return result

    @staticmethod
    def _append_local_ml(
        evidence: dict[str, list[dict[str, Any]]],
        ml_signal_context: dict[str, Any] | None,
        execution_scope: str,
    ) -> None:
        signal = _safe_dict(ml_signal_context)
        predictions = _safe_list(signal.get("predictions"))
        primary = _safe_dict(predictions[0] if predictions else {})
        influence = _safe_dict(signal.get("influence_policy"))
        production = signal_production_eligibility(signal)
        multitask = _safe_dict(primary.get("multitask_prediction"))
        for side in ("long", "short"):
            side_policy = _safe_dict(influence.get(side))
            distribution_eligibility = signal_return_distribution_eligibility(
                signal,
                side,
            )
            contract = signal_return_distribution(signal, side)
            paper = signal_paper_eligibility(signal, side)
            side_multitask = _safe_dict(multitask.get(side))
            production_eligible = bool(
                production.get("eligible") is True
                and side_policy.get("enabled") is True
                and distribution_eligibility.get("eligible") is True
            )
            paper_eligible = bool(
                paper.get("eligible") is True
                and distribution_eligibility.get("eligible") is True
            )
            eligible = (
                paper_eligible if execution_scope == "paper" else production_eligible
            )
            evidence[side].append(
                {
                    "source": "local_ml",
                    "side": side,
                    "available": bool(primary),
                    "decision_eligible": eligible,
                    "paper_eligible": paper_eligible,
                    "production_eligible": production_eligible,
                    "observation_only": bool(contract and not eligible),
                    "eligibility_reason": (
                        "live_influence_and_side_readiness_confirmed"
                        if eligible
                        else str(
                            (
                                paper.get("reason")
                                if execution_scope == "paper"
                                else production.get("reason")
                            )
                            or "local_ml_production_governance_incomplete"
                        )
                    ),
                    "raw_expected_return_pct": contract.get(
                        "raw_expected_return_pct"
                    ),
                    "objective_expected_return_pct": contract.get(
                        "objective_expected_return_pct"
                    ),
                    "expected_net_return_pct": side_multitask.get(
                        "expected_net_return_pct"
                    ),
                    "loss_probability": side_multitask.get("loss_probability"),
                    "tail_loss_probability": side_multitask.get(
                        "tail_loss_probability"
                    ),
                    "expected_execution_cost_pct": side_multitask.get(
                        "expected_execution_cost_pct"
                    ),
                    "expected_mfe_pct": side_multitask.get("expected_mfe_pct"),
                    "expected_mae_pct": side_multitask.get("expected_mae_pct"),
                    "multitask_prediction_contract": side_multitask,
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
        execution_scope: str,
    ) -> None:
        tools = _safe_dict(local_ai_tools_context)
        payload = first_tool_payload({"local_ai_tools": tools}, *aliases)
        production = signal_production_eligibility(payload)
        for side in ("long", "short"):
            distribution_eligibility = signal_return_distribution_eligibility(
                payload,
                side,
            )
            contract = signal_return_distribution(payload, side)
            paper = signal_paper_eligibility(payload, side)
            production_eligible = bool(
                production.get("eligible") is True
                and distribution_eligibility.get("eligible") is True
            )
            paper_eligible = bool(
                paper.get("eligible") is True
                and distribution_eligibility.get("eligible") is True
            )
            eligible = (
                paper_eligible if execution_scope == "paper" else production_eligible
            )
            evidence[side].append(
                {
                    "source": key,
                    "side": side,
                    "available": signal_available(payload),
                    "decision_eligible": eligible,
                    "paper_eligible": paper_eligible,
                    "production_eligible": production_eligible,
                    "observation_only": bool(contract and not eligible),
                    "eligibility_reason": (
                        paper.get("reason")
                        if execution_scope == "paper"
                        else production.get("reason")
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
                    "route_mode": payload.get("route_mode"),
                    "model": payload.get("primary_model") or payload.get("model"),
                }
            )
