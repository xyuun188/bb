"""Long-vs-short entry evidence competition policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.entry_signal_extraction import (
    directional_expected_return_pct,
    first_tool_payload,
    payload_side,
    signal_available,
    signal_production_eligible,
)
from services.entry_signal_extraction import (
    expected_return_pct as signal_expected_return_pct,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _empty_side() -> dict[str, Any]:
    return {
        "score": 0.0,
        "expected_return_pct": 0.0,
        "loss_probability": None,
        "evidence": [],
    }


def _tool_signal(
    local_ai_tools_context: dict[str, Any] | None,
    *keys: str,
) -> dict[str, Any]:
    tools = _safe_dict(local_ai_tools_context)
    if not tools:
        return {}
    return first_tool_payload({"local_ai_tools": tools}, *keys)


_LOCAL_AI_TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    "profit_prediction": (
        "profit_prediction",
        "profit_model",
        "server_profit",
        "server_profit_model",
        "profit",
    ),
    "time_series_prediction": (
        "time_series_prediction",
        "timeseries_prediction",
        "sequence_prediction",
        "timeseries",
        "time_series",
    ),
    "sentiment_analysis": (
        "sentiment_analysis",
        "sentiment_prediction",
        "sentiment_model",
        "sentiment",
    ),
    "exit_advice": (
        "exit_advice",
        "exit_model",
        "position_exit",
        "exit",
    ),
}


def _normalize_feature_coverage(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (int, float)):
        return {"ratio": float(value), "status": "reported"}
    return {"ratio": None, "status": "not_reported"}


def _tool_model_evidence(
    local_ai_tools_context: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for tool, aliases in _LOCAL_AI_TOOL_ALIASES.items():
        payload = _tool_signal(local_ai_tools_context, *aliases)
        if not payload:
            continue
        evidence[tool] = {
            "tool": tool,
            "available": signal_available(payload),
            "production_eligible": signal_production_eligible(payload),
            "status": payload.get("status"),
            "primary_model": payload.get("primary_model") or payload.get("model"),
            "challenger_model": payload.get("challenger_model"),
            "model_version": payload.get("model_version"),
            "route_mode": payload.get("route_mode"),
            "fallback_reason": payload.get("fallback_reason"),
            "feature_coverage": _normalize_feature_coverage(payload.get("feature_coverage")),
        }
    return evidence


@dataclass(frozen=True, slots=True)
class EntryDirectionCompetitionPolicy:
    """Build symbol-level long/short evidence without opening a trade by itself."""

    neutral_gap_threshold: float = 0.08

    def context(
        self,
        feature_vector: Any,
        ml_signal_context: dict[str, Any] | None,
        local_ai_tools_context: dict[str, Any] | None,
        market_regime: dict[str, Any] | None,
        strategy_mode: dict[str, Any] | None,
    ) -> dict[str, Any]:
        sides = {"long": _empty_side(), "short": _empty_side()}
        strategy = _safe_dict(strategy_mode)

        self._add_ml_evidence(sides, ml_signal_context, strategy)
        self._add_server_profit_evidence(sides, local_ai_tools_context, strategy)
        self._add_timeseries_evidence(sides, local_ai_tools_context, strategy)
        self._add_sentiment_evidence(sides, local_ai_tools_context)
        self._add_technical_evidence(sides, feature_vector)
        self._add_market_regime_penalties(sides, strategy)
        self._add_strategy_side_weights(sides, strategy)
        self._add_position_exposure_balance(sides, strategy)

        long_score = _safe_float(sides["long"]["score"], 0.0)
        short_score = _safe_float(sides["short"]["score"], 0.0)
        score_gap = abs(long_score - short_score)
        if score_gap < self.neutral_gap_threshold:
            preferred_side = "neutral"
        else:
            preferred_side = "long" if long_score > short_score else "short"

        for side in ("long", "short"):
            self._round_side(sides[side])

        regime = _safe_dict(market_regime)
        return {
            "enabled": True,
            "preferred_side": preferred_side,
            "score_gap": round(score_gap, 6),
            "long": sides["long"],
            "short": sides["short"],
            "market_regime_mode": regime.get("mode"),
            "local_ai_tools_model_evidence": _tool_model_evidence(local_ai_tools_context),
            "policy": (
                "Compare long and short independently for the current symbol. "
                "Portfolio exposure can discount sizing and risk, but it is not a "
                "mechanical reason to force the opposite side."
            ),
        }

    def _source_weight(self, strategy: dict[str, Any], source: str) -> float:
        performance = _safe_dict(strategy.get("model_contribution_performance"))
        bucket = _safe_dict(performance.get(source))
        if not bucket:
            return 1.0

        count = _safe_int(bucket.get("count"), 0)
        if count < 5:
            return 1.0

        pnl = _safe_float(bucket.get("pnl"), 0.0)
        profit_factor = _safe_float(bucket.get("profit_factor"), 1.0)
        multiplier = _safe_float(bucket.get("score_multiplier"), 1.0)
        state = str(bucket.get("state") or "").lower()
        if state == "degrade" or pnl < 0 or profit_factor < 0.85:
            if pnl <= -50.0 or profit_factor < 0.55:
                return min(multiplier, 0.15)
            return min(multiplier, 0.40)
        if state == "promote" and pnl > 0 and profit_factor >= 1.15:
            return max(min(multiplier, 1.45), 1.12)
        return max(min(multiplier, 1.15), 0.85)

    def _add(
        self,
        sides: dict[str, dict[str, Any]],
        side: str,
        score: float,
        note: str,
        *,
        expected: float | None = None,
        loss_probability: float | None = None,
    ) -> None:
        if side not in sides:
            return
        sides[side]["score"] += _safe_float(score, 0.0)
        if expected is not None:
            sides[side]["expected_return_pct"] += _safe_float(expected, 0.0)
        if loss_probability is not None:
            probability = min(max(_safe_float(loss_probability, 0.5), 0.0), 1.0)
            old = sides[side].get("loss_probability")
            sides[side]["loss_probability"] = (
                probability if old is None else max(float(old), probability)
            )
        if note:
            sides[side]["evidence"].append(str(note)[:120])

    def _add_ml_evidence(
        self,
        sides: dict[str, dict[str, Any]],
        ml_signal_context: dict[str, Any] | None,
        strategy: dict[str, Any],
    ) -> None:
        ml_signal = _safe_dict(ml_signal_context)
        predictions = (
            ml_signal.get("predictions") if isinstance(ml_signal.get("predictions"), list) else []
        )
        primary = predictions[0] if predictions and isinstance(predictions[0], dict) else {}
        if not primary or not bool(ml_signal.get("influence_enabled", True)):
            return

        weight = self._source_weight(strategy, "ml_profit_model")
        for side in ("long", "short"):
            expected = _safe_float(primary.get(f"{side}_expected_return_pct"), 0.0)
            lower_quantile = _safe_float(
                primary.get(f"{side}_lower_quantile_return_pct"), expected
            )
            tail_loss_probability = _safe_float(
                primary.get(f"{side}_tail_loss_probability"), 0.5
            )
            score = (expected * 0.65 + lower_quantile * 0.35) * weight
            self._add(
                sides,
                side,
                score,
                f"ML {side}: expected={expected:.3f}%, lower={lower_quantile:.3f}%, "
                f"tail={tail_loss_probability:.1%}, weight={weight:.2f}",
                expected=expected,
                loss_probability=tail_loss_probability,
            )

    def _add_server_profit_evidence(
        self,
        sides: dict[str, dict[str, Any]],
        local_ai_tools_context: dict[str, Any] | None,
        strategy: dict[str, Any],
    ) -> None:
        profit = _tool_signal(
            local_ai_tools_context,
            "profit_prediction",
            "profit_model",
            "server_profit",
            "server_profit_model",
            "profit",
        )
        if not signal_production_eligible(profit):
            return

        weight = self._source_weight(strategy, "server_profit_model")
        for side in ("long", "short"):
            expected = signal_expected_return_pct(profit, side)
            loss_probability = _safe_float(profit.get(f"{side}_loss_probability"), 0.5)
            quality = _safe_float(profit.get("profit_quality_score"), 0.0)
            score = (
                expected * 0.70 - max(loss_probability - 0.50, 0.0) * 0.42 + quality * 0.12
            ) * weight
            self._add(
                sides,
                side,
                score,
                (
                    f"Server profit {side}: expected={expected:.3f}%, "
                    f"loss_prob={loss_probability:.1%}, weight={weight:.2f}"
                ),
                expected=expected,
                loss_probability=loss_probability,
            )

    def _add_timeseries_evidence(
        self,
        sides: dict[str, dict[str, Any]],
        local_ai_tools_context: dict[str, Any] | None,
        strategy: dict[str, Any],
    ) -> None:
        prediction = _tool_signal(
            local_ai_tools_context,
            "time_series_prediction",
            "timeseries_prediction",
            "sequence_prediction",
            "timeseries",
            "time_series",
        )
        if not signal_production_eligible(prediction):
            return

        weight = self._source_weight(strategy, "timeseries_model")
        predicted_side = payload_side(prediction)
        expected = directional_expected_return_pct(prediction, predicted_side)
        if predicted_side not in {"long", "short"}:
            return

        self._add(
            sides,
            predicted_side,
            (expected * 0.60 + 0.08) * weight,
            f"Timeseries favors {predicted_side}: expected={expected:.3f}%, weight={weight:.2f}",
            expected=expected,
        )
        opposite = "short" if predicted_side == "long" else "long"
        self._add(
            sides,
            opposite,
            -abs(expected) * 0.25 * weight,
            f"Timeseries does not support {opposite}.",
        )

    def _add_sentiment_evidence(
        self,
        sides: dict[str, dict[str, Any]],
        local_ai_tools_context: dict[str, Any] | None,
    ) -> None:
        sentiment = _tool_signal(
            local_ai_tools_context,
            "sentiment_analysis",
            "sentiment_prediction",
            "sentiment_model",
            "sentiment",
        )
        if not signal_production_eligible(sentiment):
            return

        side = payload_side(sentiment)
        expected = signal_expected_return_pct(sentiment, side)
        score = _safe_float(sentiment.get("score", sentiment.get("sentiment_score")), 0.0)
        if side in {"long", "short"}:
            self._add(
                sides,
                side,
                expected * 0.25 + score * 0.08,
                f"Sentiment favors {side}.",
                expected=expected,
            )

    def _add_technical_evidence(
        self,
        sides: dict[str, dict[str, Any]],
        feature_vector: Any,
    ) -> None:
        returns_1 = _safe_float(getattr(feature_vector, "returns_1", 0.0), 0.0)
        returns_5 = _safe_float(getattr(feature_vector, "returns_5", 0.0), 0.0)
        returns_20 = _safe_float(getattr(feature_vector, "returns_20", 0.0), 0.0)
        price_vs_sma20 = _safe_float(getattr(feature_vector, "price_vs_sma20", 0.0), 0.0)
        price_vs_sma50 = _safe_float(getattr(feature_vector, "price_vs_sma50", 0.0), 0.0)
        adx_14 = _safe_float(getattr(feature_vector, "adx_14", 0.0), 0.0)

        momentum = returns_1 * 100.0 * 0.08 + returns_5 * 100.0 * 0.18 + returns_20 * 100.0 * 0.10
        ma_bias = (price_vs_sma20 + price_vs_sma50) * 0.06
        trend_strength = min(max((adx_14 - 14.0) / 28.0, 0.0), 1.0)
        self._add(
            sides,
            "long",
            momentum + ma_bias + max(momentum, 0.0) * trend_strength * 0.12,
            "Technical structure long-side score.",
        )
        self._add(
            sides,
            "short",
            -momentum - ma_bias + max(-momentum, 0.0) * trend_strength * 0.12,
            "Technical structure short-side score.",
        )

    def _add_market_regime_penalties(
        self,
        sides: dict[str, dict[str, Any]],
        strategy: dict[str, Any],
    ) -> None:
        soft_avoided = {
            str(side).lower() for side in _safe_list(strategy.get("soft_avoided_directions"))
        }
        for side in ("long", "short"):
            if side in soft_avoided:
                self._add(
                    sides,
                    side,
                    -0.10,
                    f"Market regime soft penalty: {side} needs stronger single-symbol evidence.",
                )

    def _add_strategy_side_weights(
        self,
        sides: dict[str, dict[str, Any]],
        strategy: dict[str, Any],
    ) -> None:
        weights = _safe_dict(strategy.get("side_weights"))
        if not weights:
            learning = _safe_dict(strategy.get("strategy_learning"))
            weights = _safe_dict(_safe_dict(learning.get("runtime")).get("side_weights"))
        if not weights:
            return

        for side in ("long", "short"):
            if side not in weights:
                continue
            weight = min(max(_safe_float(weights.get(side), 1.0), 0.25), 1.40)
            if abs(weight - 1.0) <= 1e-9:
                continue
            original = _safe_float(sides[side].get("score"), 0.0)
            if weight < 1.0 and original < 0:
                adjusted = original / max(weight, 0.25)
            elif weight > 1.0 and original < 0:
                adjusted = original / weight
            else:
                adjusted = original * weight
            sides[side]["score"] = adjusted
            sides[side]["evidence"].append(
                f"Strategy learning side weight {side}: {weight:.2f}x."
            )

    def _add_position_exposure_balance(
        self,
        sides: dict[str, dict[str, Any]],
        strategy: dict[str, Any],
    ) -> None:
        exposure = _safe_dict(strategy.get("position_exposure"))
        dominant_side = str(exposure.get("dominant_side") or "neutral").lower()
        net_ratio = abs(_safe_float(exposure.get("net_ratio"), 0.0))
        if dominant_side not in {"long", "short"} or net_ratio <= 0:
            return

        side_pnl = _safe_float(exposure.get(f"{dominant_side}_unrealized_pnl"), 0.0)
        same_side_penalty = net_ratio * 0.28
        if side_pnl < 0:
            same_side_penalty += min(abs(side_pnl) / 25.0, 0.75)
        self._add(
            sides,
            dominant_side,
            -same_side_penalty,
            (
                f"Portfolio already concentrated {dominant_side}; unrealized_pnl={side_pnl:.2f}U, "
                "new same-side entries need stronger edge."
            ),
        )

        opposite = "short" if dominant_side == "long" else "long"
        opposite_expected = _safe_float(sides[opposite].get("expected_return_pct"), 0.0)
        if opposite_expected > 0:
            self._add(
                sides,
                opposite,
                min(net_ratio * 0.025, 0.04),
                (
                    f"Portfolio is {dominant_side}-heavy; give {opposite} only a small balance "
                    "nudge because its expected return is already positive."
                ),
            )

    @staticmethod
    def _round_side(side: dict[str, Any]) -> None:
        side["score"] = round(_safe_float(side["score"], 0.0), 6)
        side["expected_return_pct"] = round(_safe_float(side["expected_return_pct"], 0.0), 6)
        if side.get("loss_probability") is not None:
            side["loss_probability"] = round(_safe_float(side["loss_probability"], 0.5), 6)
        side["evidence"] = _safe_list(side.get("evidence"))[:5]
