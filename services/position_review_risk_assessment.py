"""Risk-engine adapter for position-review decisions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput

AccountBalanceProvider = Callable[[str], Awaitable[float]]


@dataclass(frozen=True, slots=True)
class PositionReviewRiskAssessmentPolicy:
    """Prepare the lightweight risk-engine call used by position review."""

    risk_engine: Any

    async def assess(
        self,
        *,
        decision: DecisionOutput,
        model_name: str,
        open_positions: list[dict[str, Any]],
        feature_vector: Any,
        account_balance_provider: AccountBalanceProvider,
    ) -> Any:
        model_positions = [
            position for position in open_positions if position.get("model_name") == model_name
        ]
        return self.risk_engine.assess(
            decision,
            current_positions=model_positions,
            account_balance=await account_balance_provider(model_name),
            headlines=getattr(feature_vector, "recent_headlines", []),
            sentiment_scores=[],
            price_change_1m=getattr(feature_vector, "returns_1", 0.0),
            volume_ratio=getattr(feature_vector, "volume_ratio", 1.0),
            adx_14=getattr(feature_vector, "adx_14", None),
        )
