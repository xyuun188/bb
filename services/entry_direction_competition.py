"""Observation of long-vs-short returns from production-governed models only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.entry_signal_extraction import (
    directional_expected_return_pct,
    first_tool_payload,
    signal_available,
    signal_production_eligibility,
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
    eligible = [
        float(item["expected_return_pct"])
        for item in values
        if item.get("production_eligible") is True
        and _safe_float(item.get("expected_return_pct")) is not None
    ]
    return {
        "score": sum(eligible) / len(eligible) if eligible else 0.0,
        "expected_return_pct": sum(eligible) / len(eligible) if eligible else 0.0,
        "production_source_count": len(eligible),
        "evidence": values,
    }


@dataclass(frozen=True, slots=True)
class EntryDirectionCompetitionPolicy:
    """Compare only governed expected-return observations.

    This context may guide the model toward the better side, but it cannot grant
    execution permission. The selected side must still pass the live fee-after
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
        long_side = _side_summary(evidence["long"])
        short_side = _side_summary(evidence["short"])
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
        return {
            "enabled": bool(source_count),
            "preferred_side": preferred_side,
            "score_gap": abs(long_score - short_score),
            "long": long_side,
            "short": short_side,
            "production_source_count": source_count,
            "production_permission": False,
            "policy": "production_governed_expected_returns_only_no_fixed_gap",
            "policy_provenance": {
                "source": "live_influence_return_models",
                "observation_window": "current_decision_model_outputs",
                "sample_count": source_count,
                "strategy_version": "2026-07-12.return-direction-observation.v1",
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
            value = _safe_float(primary.get(f"{side}_expected_return_pct"))
            eligible = bool(
                eligibility.get("eligible") is True
                and side_policy.get("enabled") is True
                and value is not None
            )
            evidence[side].append(
                {
                    "source": "local_ml",
                    "side": side,
                    "available": bool(primary),
                    "production_eligible": eligible,
                    "eligibility_reason": (
                        "live_influence_and_side_readiness_confirmed"
                        if eligible
                        else str(
                            eligibility.get("reason")
                            or "local_ml_production_governance_incomplete"
                        )
                    ),
                    "expected_return_pct": value,
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
            value = _safe_float(directional_expected_return_pct(payload, side))
            eligible = bool(eligibility.get("eligible") and value is not None)
            evidence[side].append(
                {
                    "source": key,
                    "side": side,
                    "available": signal_available(payload),
                    "production_eligible": eligible,
                    "eligibility_reason": eligibility.get("reason"),
                    "expected_return_pct": value,
                    "route_mode": payload.get("route_mode"),
                    "model": payload.get("primary_model") or payload.get("model"),
                }
            )
