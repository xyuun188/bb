"""Paper-only continuous model weighting from cost-complete return evidence."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from services.model_expert_health import ModelExpertHealthService
from services.specialist_shadow_evaluation import SpecialistShadowEvaluationService

CONTINUOUS_MODEL_WEIGHT_VERSION = "2026-07-22.paper-continuous-model-weight.v1"
EXPERT_MODEL_NAMES = (
    "trend_expert",
    "momentum_expert",
    "sentiment_expert",
    "position_expert",
    "risk_expert",
)
QUANT_SOURCE_NAMES = (
    "local_ml",
    "server_profit",
    "timeseries",
    "sentiment",
)
QUANT_CONTRIBUTION_KEYS = {
    "local_ml": "ml_profit_model",
    "server_profit": "server_profit_model",
    "timeseries": "timeseries_model",
    "sentiment": "sentiment_model",
}
QUANT_SPECIALIST_TOOLS = {
    "server_profit": "profit_prediction",
    "timeseries": "time_series_prediction",
    "sentiment": "sentiment_analysis",
}
COLD_START_MULTIPLIER = 0.35
MINIMUM_OBSERVATION_MULTIPLIER = 0.10
MAXIMUM_OBSERVATION_MULTIPLIER = 1.40
SMOOTHING_RATE = 0.25
EVIDENCE_CACHE_SECONDS = 600.0


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _confidence(sample_count: int) -> float:
    return 1.0 - math.exp(-max(int(sample_count), 0) / 20.0)


def _profit_factor_signal(
    profit_factor: Any,
    *,
    profit: float,
    loss: float,
) -> float:
    parsed = _finite(profit_factor)
    if parsed is None:
        if profit > 0 and loss <= 0:
            return 1.0
        return 0.0
    return _clamp((parsed - 1.0) / max(parsed + 1.0, 1e-12), -1.0, 1.0)


def _return_evidence(
    *,
    sample_count: Any,
    average_return: Any,
    return_lcb: Any,
    profit_factor: Any,
    profit: Any,
    loss: Any,
    source: str,
    worst_return: Any = None,
    max_drawdown: Any = None,
) -> dict[str, Any]:
    count = max(int(_finite(sample_count) or 0), 0)
    average = _finite(average_return)
    lower_bound = _finite(return_lcb)
    gross_profit = max(_finite(profit) or 0.0, 0.0)
    gross_loss = max(_finite(loss) or 0.0, 0.0)
    worst = _finite(worst_return)
    drawdown = max(_finite(max_drawdown) or 0.0, 0.0)
    if count <= 0 or average is None or lower_bound is None:
        return {
            "source": source,
            "sample_count": count,
            "available": False,
            "confidence": 0.0,
            "return_quality": 0.0,
        }
    scale = max(
        abs(average),
        abs(lower_bound),
        (gross_profit + gross_loss) / max(count, 1),
        1e-12,
    )
    average_signal = _clamp(average / scale, -1.0, 1.0)
    lower_bound_signal = _clamp(lower_bound / scale, -1.0, 1.0)
    factor_signal = _profit_factor_signal(
        profit_factor,
        profit=gross_profit,
        loss=gross_loss,
    )
    tail_signal = _clamp((worst or 0.0) / scale, -1.0, 0.0)
    drawdown_signal = -_clamp(
        drawdown / max(gross_profit + gross_loss, scale),
        0.0,
        1.0,
    )
    quality = (
        0.35 * average_signal
        + 0.30 * lower_bound_signal
        + 0.20 * factor_signal
        + 0.10 * tail_signal
        + 0.05 * drawdown_signal
    )
    return {
        "source": source,
        "sample_count": count,
        "available": True,
        "confidence": round(_confidence(count), 8),
        "return_quality": round(_clamp(quality, -1.0, 1.0), 8),
        "average_return": average,
        "return_lcb": lower_bound,
        "profit_factor": _finite(profit_factor),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "worst_return": worst,
        "max_drawdown": drawdown,
        "tail_loss_signal": round(tail_signal, 8),
        "drawdown_signal": round(drawdown_signal, 8),
    }


def _actual_evidence(bucket: dict[str, Any], *, source: str) -> dict[str, Any]:
    return _return_evidence(
        sample_count=bucket.get("count"),
        average_return=bucket.get("avg_pnl"),
        return_lcb=bucket.get("pnl_lcb_usdt"),
        profit_factor=bucket.get("profit_factor"),
        profit=bucket.get("profit"),
        loss=bucket.get("loss"),
        worst_return=bucket.get("worst_pnl_usdt"),
        max_drawdown=bucket.get("max_drawdown_usdt"),
        source=source,
    )


def _specialist_evidence(
    specialist_report: dict[str, Any],
    *,
    tool_name: str,
) -> dict[str, Any]:
    rows = [
        _dict(row)
        for row in _list(specialist_report.get("models"))
        if str(_dict(row).get("tool") or "") == tool_name
    ]
    if not rows:
        return _return_evidence(
            sample_count=0,
            average_return=None,
            return_lcb=None,
            profit_factor=None,
            profit=0.0,
            loss=0.0,
            source=f"specialist_shadow:{tool_name}",
        )
    count = sum(max(int(_finite(row.get("direction_count")) or 0), 0) for row in rows)
    gross_profit = sum(max(_finite(row.get("gross_profit_return_pct")) or 0.0, 0.0) for row in rows)
    gross_loss = sum(max(_finite(row.get("gross_loss_return_pct")) or 0.0, 0.0) for row in rows)
    average = (
        sum(
            (_finite(row.get("avg_shadow_return_after_all_cost_pct")) or 0.0)
            * max(int(_finite(row.get("direction_count")) or 0), 0)
            for row in rows
        )
        / count
        if count > 0
        else None
    )
    lower_bounds = [
        value
        for row in rows
        if (value := _finite(row.get("shadow_return_lcb_pct"))) is not None
    ]
    worst_returns = [
        value
        for row in rows
        if (value := _finite(row.get("worst_realized_return_pct"))) is not None
    ]
    return _return_evidence(
        sample_count=count,
        average_return=average,
        return_lcb=min(lower_bounds) if lower_bounds else None,
        profit_factor=(gross_profit / gross_loss if gross_loss > 0 else None),
        profit=gross_profit,
        loss=gross_loss,
        worst_return=min(worst_returns) if worst_returns else None,
        source=f"specialist_shadow:{tool_name}",
    )


def _health_evidence(health_report: dict[str, Any], name: str) -> dict[str, Any]:
    component = _dict(_dict(health_report.get("components")).get(name))
    window = _dict(_dict(component.get("windows")).get("24h"))
    participation = max(int(_finite(window.get("participation_count")) or 0), 0)
    json_error_rate = _clamp(_finite(window.get("json_error_rate")) or 0.0, 0.0, 1.0)
    no_return_rate = _clamp(_finite(window.get("no_return_rate")) or 0.0, 0.0, 1.0)
    wrong_rate = _clamp(
        _finite(window.get("wrong_recommendation_rate")) or 0.0,
        0.0,
        1.0,
    )
    stability = _clamp(
        (1.0 - json_error_rate) * (1.0 - no_return_rate) * (1.0 - 0.5 * wrong_rate),
        0.20,
        1.0,
    )
    return {
        "sample_count": participation,
        "json_error_rate": round(json_error_rate, 8),
        "no_return_rate": round(no_return_rate, 8),
        "shadow_direction_error_rate": round(wrong_rate, 8),
        "stability_multiplier": round(stability, 8),
    }


def _combined_multiplier(
    *,
    actual: dict[str, Any],
    shadow: dict[str, Any],
    stability_multiplier: float,
    previous: float | None,
) -> dict[str, Any]:
    evidence = [item for item in (actual, shadow) if item.get("available") is True]
    combined_confidence = 1.0
    weighted_quality = 0.0
    confidence_total = 0.0
    for item in evidence:
        confidence = _clamp(_finite(item.get("confidence")) or 0.0, 0.0, 1.0)
        combined_confidence *= 1.0 - confidence
        weighted_quality += confidence * (_finite(item.get("return_quality")) or 0.0)
        confidence_total += confidence
    combined_confidence = 1.0 - combined_confidence
    quality = weighted_quality / confidence_total if confidence_total > 0 else 0.0
    evidence_target = _clamp(
        1.0 + 0.40 * quality,
        MINIMUM_OBSERVATION_MULTIPLIER,
        MAXIMUM_OBSERVATION_MULTIPLIER,
    )
    target = COLD_START_MULTIPLIER + combined_confidence * (
        evidence_target - COLD_START_MULTIPLIER
    )
    target = _clamp(
        target * _clamp(stability_multiplier, 0.20, 1.0),
        MINIMUM_OBSERVATION_MULTIPLIER,
        MAXIMUM_OBSERVATION_MULTIPLIER,
    )
    effective = (
        target
        if previous is None
        else previous + SMOOTHING_RATE * (target - previous)
    )
    return {
        "target_multiplier": round(target, 8),
        "previous_multiplier": round(previous, 8) if previous is not None else None,
        "effective_multiplier": round(
            _clamp(
                effective,
                MINIMUM_OBSERVATION_MULTIPLIER,
                MAXIMUM_OBSERVATION_MULTIPLIER,
            ),
            8,
        ),
        "combined_evidence_confidence": round(combined_confidence, 8),
        "combined_return_quality": round(quality, 8),
        "cold_start": not evidence,
    }


_NON_REGIME_LABELS = {
    "return_distribution_observation",
    "observation_unavailable",
    "unknown",
}


def market_regime_name(value: Any) -> str:
    """Return one shared market-state label from cross-section or sample features."""

    if isinstance(value, str):
        label = value.strip().lower()
        return "unknown" if not label or label in _NON_REGIME_LABELS else label
    market_regime = _dict(value)
    for key in ("regime", "state", "label", "name", "market_regime"):
        nested = market_regime.get(key)
        if isinstance(nested, dict):
            label = market_regime_name(nested)
        else:
            label = str(nested or "").strip().lower()
        if label and label not in _NON_REGIME_LABELS:
            return label
    mode = str(market_regime.get("mode") or "").strip().lower()
    if mode and mode not in _NON_REGIME_LABELS:
        return mode
    adx = _finite(market_regime.get("avg_adx_14"))
    if adx is None:
        adx = _finite(market_regime.get("adx_14"))
    directional_inputs = []
    for aggregate_key, sample_key in (
        ("avg_returns_20", "returns_20"),
        ("avg_price_vs_sma20", "price_vs_sma20"),
        ("avg_price_vs_sma50", "price_vs_sma50"),
    ):
        directional = _finite(market_regime.get(aggregate_key))
        if directional is None:
            directional = _finite(market_regime.get(sample_key))
        if directional is not None:
            directional_inputs.append(directional)
    if adx is None or not directional_inputs:
        return "unknown"
    direction_score = sum(directional_inputs) / len(directional_inputs)
    if adx >= 25.0:
        if direction_score > 0.001:
            return "trend_up"
        if direction_score < -0.001:
            return "trend_down"
        return "trend_flat"
    return "range_bound"


def continuous_model_weight_scenario(
    execution_mode: str,
    market_regime: dict[str, Any] | None,
) -> str:
    mode = "live" if str(execution_mode or "").lower() == "live" else "paper"
    return f"{mode}:{market_regime_name(market_regime)}"


class ContinuousModelWeightPolicy:
    """Build auditable, smoothed model multipliers without mutating live trading."""

    def __init__(self) -> None:
        self._previous_by_scenario: dict[str, dict[str, float]] = {}

    def build(
        self,
        *,
        execution_mode: str,
        market_regime: dict[str, Any] | None,
        health_report: dict[str, Any] | None,
        specialist_report: dict[str, Any] | None,
        contribution_performance: dict[str, Any] | None,
        generated_at: datetime | None = None,
    ) -> dict[str, Any]:
        mode = "live" if str(execution_mode or "").lower() == "live" else "paper"
        scenario = continuous_model_weight_scenario(mode, market_regime)
        now = generated_at or datetime.now(UTC)
        if mode != "paper":
            return {
                "version": CONTINUOUS_MODEL_WEIGHT_VERSION,
                "execution_scope": "paper_only",
                "applied": False,
                "live_weights_unchanged": True,
                "scenario": scenario,
                "generated_at": now.isoformat(),
                "expert_weights": {
                    name: {"effective_multiplier": 1.0, "reason": "live_path_unchanged"}
                    for name in EXPERT_MODEL_NAMES
                },
                "quant_source_weights": {
                    name: {"effective_multiplier": 1.0, "reason": "live_path_unchanged"}
                    for name in QUANT_SOURCE_NAMES
                },
                "weight_changes": [],
                "rollback": {"mode": "base_weights", "multiplier": 1.0},
            }

        health = _dict(health_report)
        specialist = _dict(specialist_report)
        contribution = _dict(contribution_performance)
        previous = self._previous_by_scenario.get(scenario, {})
        current: dict[str, float] = {}
        changes: list[dict[str, Any]] = []
        expert_weights: dict[str, Any] = {}
        for name in EXPERT_MODEL_NAMES:
            identity = f"expert:{name}"
            health_item = _health_evidence(health, name)
            actual = _actual_evidence(
                _dict(contribution.get(identity)),
                source=f"authoritative_trade:{identity}",
            )
            shadow = _return_evidence(
                sample_count=0,
                average_return=None,
                return_lcb=None,
                profit_factor=None,
                profit=0.0,
                loss=0.0,
                source=f"shadow_direction_health:{name}",
            )
            multiplier = _combined_multiplier(
                actual=actual,
                shadow=shadow,
                stability_multiplier=float(health_item["stability_multiplier"]),
                previous=previous.get(identity),
            )
            current[identity] = float(multiplier["effective_multiplier"])
            expert_weights[name] = {
                **multiplier,
                "actual_fee_after_return": actual,
                "shadow_health": health_item,
                "production_permission": False,
            }
            if multiplier["previous_multiplier"] is not None and not math.isclose(
                float(multiplier["previous_multiplier"]),
                float(multiplier["effective_multiplier"]),
                abs_tol=1e-10,
            ):
                changes.append(
                    {
                        "model": name,
                        "kind": "expert",
                        "before": multiplier["previous_multiplier"],
                        "after": multiplier["effective_multiplier"],
                        "target": multiplier["target_multiplier"],
                    }
                )

        quant_weights: dict[str, Any] = {}
        for source_name in QUANT_SOURCE_NAMES:
            identity = f"quant:{source_name}"
            contribution_key = QUANT_CONTRIBUTION_KEYS[source_name]
            actual = _actual_evidence(
                _dict(contribution.get(contribution_key)),
                source=f"authoritative_trade:{contribution_key}",
            )
            tool_name = QUANT_SPECIALIST_TOOLS.get(source_name)
            shadow = (
                _specialist_evidence(specialist, tool_name=tool_name)
                if tool_name
                else _return_evidence(
                    sample_count=0,
                    average_return=None,
                    return_lcb=None,
                    profit_factor=None,
                    profit=0.0,
                    loss=0.0,
                    source=f"specialist_shadow:{source_name}",
                )
            )
            multiplier = _combined_multiplier(
                actual=actual,
                shadow=shadow,
                stability_multiplier=1.0,
                previous=previous.get(identity),
            )
            current[identity] = float(multiplier["effective_multiplier"])
            quant_weights[source_name] = {
                **multiplier,
                "actual_fee_after_return": actual,
                "shadow_fee_after_return": shadow,
                "production_permission": False,
            }
            if multiplier["previous_multiplier"] is not None and not math.isclose(
                float(multiplier["previous_multiplier"]),
                float(multiplier["effective_multiplier"]),
                abs_tol=1e-10,
            ):
                changes.append(
                    {
                        "model": source_name,
                        "kind": "quant_source",
                        "before": multiplier["previous_multiplier"],
                        "after": multiplier["effective_multiplier"],
                        "target": multiplier["target_multiplier"],
                    }
                )
        self._previous_by_scenario[scenario] = current
        return {
            "version": CONTINUOUS_MODEL_WEIGHT_VERSION,
            "execution_scope": "paper_only",
            "applied": True,
            "live_weights_unchanged": True,
            "scenario": scenario,
            "generated_at": now.isoformat(),
            "objective": "fee_after_total_return_drawdown_and_tail_loss_not_win_rate",
            "expert_weights": expert_weights,
            "quant_source_weights": quant_weights,
            "weight_changes": changes,
            "same_provider_group_budget_required": True,
            "failed_models_remain_observable": True,
            "smoothing_rate": SMOOTHING_RATE,
            "cold_start_multiplier": COLD_START_MULTIPLIER,
            "rollback": {
                "mode": "base_weights",
                "multiplier": 1.0,
                "recomputable_from_saved_evidence": True,
            },
        }


class ContinuousModelWeightEvidenceService:
    """Cache paper model health and cost-complete specialist shadow evidence."""

    def __init__(
        self,
        *,
        health_service: ModelExpertHealthService | None = None,
        specialist_service: SpecialistShadowEvaluationService | None = None,
        cache_seconds: float = EVIDENCE_CACHE_SECONDS,
    ) -> None:
        self.health_service = health_service or ModelExpertHealthService()
        self.specialist_service = specialist_service or SpecialistShadowEvaluationService()
        self.cache_seconds = max(float(cache_seconds), 1.0)
        self._cache: dict[str, Any] = {}

    async def report(self, mode: str) -> dict[str, Any]:
        selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
        if selected_mode != "paper":
            return {
                "execution_scope": "paper_only",
                "available": False,
                "reason": "live_path_unchanged",
            }
        now = datetime.now(UTC)
        cached_at = self._cache.get("cached_at")
        cached_report = self._cache.get("report")
        if (
            isinstance(cached_at, datetime)
            and (now - cached_at).total_seconds() < self.cache_seconds
            and isinstance(cached_report, dict)
        ):
            return cached_report
        health = await self.health_service.report(hours=72, limit=1200, mode="paper")
        specialist = await self.specialist_service.report(hours=72, mode="paper")
        report = {
            "execution_scope": "paper_only",
            "available": True,
            "generated_at": now.isoformat(),
            "health": health,
            "specialist_shadow": specialist,
            "expires_at": (now + timedelta(seconds=self.cache_seconds)).isoformat(),
        }
        self._cache = {"cached_at": now, "report": report}
        return report
