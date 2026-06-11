"""Stress stop-loss calculation for entry risk sizing."""

from __future__ import annotations

from dataclasses import dataclass

ENTRY_STRESS_STOP_MIN_PCT = 0.018
ENTRY_STRESS_STOP_MAX_PCT = 0.080
ENTRY_LOW_QUALITY_STRESS_STOP_MIN_PCT = 0.050
ENTRY_ATR_STRESS_STOP_MULTIPLIER = 1.60


@dataclass(frozen=True, slots=True)
class EntryStressStopPolicy:
    """Calculate the stress stop used for planned-loss sizing caps."""

    min_stop_loss_pct: float = ENTRY_STRESS_STOP_MIN_PCT
    max_stop_loss_pct: float = ENTRY_STRESS_STOP_MAX_PCT
    low_quality_min_stop_loss_pct: float = ENTRY_LOW_QUALITY_STRESS_STOP_MIN_PCT
    tail_risk_multiplier: float = 0.075
    negative_expected_multiplier: float = 0.65
    atr_stop_multiplier: float = ENTRY_ATR_STRESS_STOP_MULTIPLIER

    def stress_stop_loss_pct(
        self,
        *,
        declared_stop_loss_pct: float,
        expected_loss_pct: float,
        tail_risk_score: float,
        raw_expected_return_pct: float,
        low_payoff_quality: bool,
        atr_pct: float = 0.0,
    ) -> float:
        """Return the stress stop-loss pct used for entry loss-budget calculations."""

        declared_stop_loss_pct = max(float(declared_stop_loss_pct or 0.0), 0.0)
        expected_loss_stop = (
            float(expected_loss_pct or 0.0) / 100.0 if expected_loss_pct > 0 else 0.0
        )
        tail_risk_stop = min(
            max(float(tail_risk_score or 0.0), 0.0) * self.tail_risk_multiplier,
            self.max_stop_loss_pct,
        )
        negative_expected_stop = (
            min(
                abs(float(raw_expected_return_pct or 0.0))
                / 100.0
                * self.negative_expected_multiplier,
                self.max_stop_loss_pct,
            )
            if raw_expected_return_pct < 0
            else 0.0
        )
        atr_stop = (
            min(max(float(atr_pct or 0.0), 0.0) * self.atr_stop_multiplier, self.max_stop_loss_pct)
            if atr_pct > 0
            else 0.0
        )
        quality_floor = (
            self.low_quality_min_stop_loss_pct if low_payoff_quality else self.min_stop_loss_pct
        )
        stress_stop = max(
            declared_stop_loss_pct,
            expected_loss_stop,
            tail_risk_stop,
            negative_expected_stop,
            atr_stop,
            quality_floor,
        )
        return min(max(stress_stop, declared_stop_loss_pct), self.max_stop_loss_pct)
