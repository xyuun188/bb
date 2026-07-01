"""Entry payoff quality classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.entry_sizing import evidence_is_low_payoff_quality
from services.trading_params import DEFAULT_TRADING_PARAMS

_PAYOFF_PARAMS = DEFAULT_TRADING_PARAMS.entry_payoff_quality


@dataclass(frozen=True, slots=True)
class EntryLowPayoffQualityPolicy:
    """Classify entry candidates that should be capped to small, defensive sizing."""

    min_expected_net_return_pct: float = _PAYOFF_PARAMS.min_expected_net_return_pct
    min_profit_quality_ratio: float = _PAYOFF_PARAMS.min_profit_quality_ratio
    max_small_win_big_loss_penalty: float = _PAYOFF_PARAMS.max_small_win_big_loss_penalty

    def is_low_payoff(
        self,
        *,
        score: float,
        min_score_required: float,
        expected_net_return_pct: float,
        profit_quality_ratio: float,
        raw_expected_return_pct: float,
        small_win_big_loss_penalty: float,
        hard_contribution_caution: bool,
        evidence_score: dict[str, Any],
        evidence_effective_score: float,
    ) -> bool:
        """Return True when payoff quality is too weak for normal entry sizing."""

        return bool(
            self.reasons(
                score=score,
                min_score_required=min_score_required,
                expected_net_return_pct=expected_net_return_pct,
                profit_quality_ratio=profit_quality_ratio,
                raw_expected_return_pct=raw_expected_return_pct,
                small_win_big_loss_penalty=small_win_big_loss_penalty,
                hard_contribution_caution=hard_contribution_caution,
                evidence_score=evidence_score,
                evidence_effective_score=evidence_effective_score,
            )
        )

    def reasons(
        self,
        *,
        score: float,
        min_score_required: float,
        expected_net_return_pct: float,
        profit_quality_ratio: float,
        raw_expected_return_pct: float,
        small_win_big_loss_penalty: float,
        hard_contribution_caution: bool,
        evidence_score: dict[str, Any],
        evidence_effective_score: float,
    ) -> list[str]:
        """Return stable reason codes explaining defensive low-payoff sizing."""

        reasons: list[str] = []
        if score < min_score_required:
            reasons.append("score_below_required")
        if expected_net_return_pct < self.min_expected_net_return_pct:
            reasons.append("expected_net_below_min")
        if profit_quality_ratio < self.min_profit_quality_ratio:
            reasons.append("profit_quality_below_min")
        strong_aggregate_quality = (
            expected_net_return_pct >= self.min_expected_net_return_pct
            and profit_quality_ratio >= self.min_profit_quality_ratio
            and score >= min_score_required
            and not evidence_is_low_payoff_quality(evidence_score, evidence_effective_score)
        )
        if raw_expected_return_pct < 0 and not strong_aggregate_quality:
            reasons.append("raw_expected_return_negative")
        if small_win_big_loss_penalty >= self.max_small_win_big_loss_penalty:
            reasons.append("small_win_big_loss_penalty_high")
        if hard_contribution_caution and not strong_aggregate_quality:
            reasons.append("hard_contribution_caution")
        if evidence_is_low_payoff_quality(evidence_score, evidence_effective_score):
            reasons.append("evidence_low_payoff_quality")
        return reasons
