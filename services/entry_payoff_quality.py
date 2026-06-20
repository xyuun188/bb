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
            score < min_score_required
            or expected_net_return_pct < self.min_expected_net_return_pct
            or profit_quality_ratio < self.min_profit_quality_ratio
            or raw_expected_return_pct < 0
            or small_win_big_loss_penalty >= self.max_small_win_big_loss_penalty
            or hard_contribution_caution
            or evidence_is_low_payoff_quality(evidence_score, evidence_effective_score)
        )
