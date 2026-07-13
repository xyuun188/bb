"""Observation-only market cross-section for entry diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ai_brain.base_model import DecisionOutput

FeatureValidator = Callable[[Any], bool]


def _feature_float(feature: Any, name: str) -> float:
    try:
        return float(getattr(feature, name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True, slots=True)
class EntryMarketRegimeContextPolicy:
    """Expose current market breadth without direction permission."""

    is_valid_feature_vector: FeatureValidator

    def context(self, feature_vectors: dict[str, Any]) -> dict[str, Any]:
        rows = [
            feature
            for feature in (feature_vectors or {}).values()
            if self.is_valid_feature_vector(feature)
        ]
        count = len(rows)
        generated_at = datetime.now(UTC).isoformat()
        if not rows:
            return {
                "mode": "observation_unavailable",
                "sample_count": 0,
                "production_permission": False,
                "policy_provenance": {
                    "source": "current_market_cross_section",
                    "observation_window": "current_analysis_round",
                    "sample_count": 0,
                    "generated_at": generated_at,
                    "strategy_version": "2026-07-12.market-regime-observation.v1",
                    "fallback_reason": "valid_market_cross_section_missing",
                },
            }

        def average(name: str) -> float:
            return sum(_feature_float(row, name) for row in rows) / count

        return {
            "mode": "return_distribution_observation",
            "sample_count": count,
            "avg_returns_5": round(average("returns_5"), 8),
            "avg_returns_20": round(average("returns_20"), 8),
            "avg_price_vs_sma20": round(average("price_vs_sma20"), 8),
            "avg_price_vs_sma50": round(average("price_vs_sma50"), 8),
            "avg_adx_14": round(average("adx_14"), 8),
            "production_permission": False,
            "reason": "Market regime is observation-only; governed fee-after returns choose direction.",
            "policy_provenance": {
                "source": "current_market_cross_section",
                "observation_window": "current_analysis_round",
                "sample_count": count,
                "generated_at": generated_at,
                "strategy_version": "2026-07-12.market-regime-observation.v1",
                "fallback_reason": "",
            },
        }


@dataclass(frozen=True, slots=True)
class EntryMarketRegimePolicy:
    """Compatibility boundary with no production decision authority."""

    def reason(
        self,
        decision: DecisionOutput,
        market_regime: dict[str, Any],
    ) -> None:
        del decision, market_regime
        return None
