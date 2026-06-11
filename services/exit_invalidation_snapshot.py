"""Exit thesis-invalidation snapshot policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput

MinVolumeRatioProvider = Callable[[], float]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class ExitInvalidationSnapshotPolicy:
    """Detect clear thesis invalidation before allowing discretionary exits."""

    min_volume_ratio_provider: MinVolumeRatioProvider

    def snapshot(
        self,
        decision: DecisionOutput,
        target_side: str,
        entry_price: float,
        current_price: float,
    ) -> dict[str, Any]:
        feature_snapshot = decision.feature_snapshot or {}
        atr_14 = _safe_float(feature_snapshot.get("atr_14"), 0.0)
        ema_12 = _safe_float(feature_snapshot.get("ema_12"), 0.0)
        ema_26 = _safe_float(feature_snapshot.get("ema_26"), 0.0)
        returns_5 = _safe_float(feature_snapshot.get("returns_5"), 0.0)
        returns_20 = _safe_float(feature_snapshot.get("returns_20"), 0.0)
        volume_ratio = _safe_float(feature_snapshot.get("volume_ratio"), 0.0)
        price_vs_sma20 = _safe_float(feature_snapshot.get("price_vs_sma20"), 0.0)
        price_vs_sma50 = _safe_float(feature_snapshot.get("price_vs_sma50"), 0.0)

        atr_break = atr_14 * 1.2 if atr_14 > 0 else 0.0
        pct_break = abs(entry_price) * 0.012
        break_distance = max(atr_break, pct_break)

        if target_side == "long":
            key_break = current_price <= entry_price - break_distance or price_vs_sma20 <= -0.006
            trend_reversal = (
                ema_12 > 0
                and ema_26 > 0
                and ema_12 < ema_26
                and price_vs_sma20 < -0.003
                and price_vs_sma50 <= 0
            )
            momentum_bad = returns_5 <= -0.006 or returns_20 <= -0.012
        else:
            key_break = current_price >= entry_price + break_distance or price_vs_sma20 >= 0.006
            trend_reversal = (
                ema_12 > 0
                and ema_26 > 0
                and ema_12 > ema_26
                and price_vs_sma20 > 0.003
                and price_vs_sma50 >= 0
            )
            momentum_bad = returns_5 >= 0.006 or returns_20 >= 0.012

        volume_confirms = volume_ratio >= max(float(self.min_volume_ratio_provider() or 1.0), 1.2)
        severe = (
            (key_break and trend_reversal)
            or (key_break and momentum_bad and volume_confirms)
            or (trend_reversal and momentum_bad and volume_confirms)
        )
        reasons = []
        if key_break:
            reasons.append("key_break")
        if trend_reversal:
            reasons.append("trend_reversal")
        if momentum_bad:
            reasons.append("momentum_bad")
        if volume_confirms:
            reasons.append("volume_confirms")
        return {
            "severe": severe,
            "key_break": key_break,
            "trend_reversal": trend_reversal,
            "momentum_bad": momentum_bad,
            "volume_confirms": volume_confirms,
            "reason": ";".join(reasons) if reasons else "no severe invalidation",
        }
