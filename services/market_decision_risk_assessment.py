"""Hard-risk adapter for market-analysis decisions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput

AccountBalanceProvider = Callable[[str], Awaitable[float]]


@dataclass(frozen=True, slots=True)
class MarketDecisionRiskAssessmentPolicy:
    """Apply exchange/account risk without expert, score, or probe permissions."""

    risk_engine: Any
    account_balance_provider: AccountBalanceProvider

    async def assess(
        self,
        *,
        decision: DecisionOutput,
        model_name: str,
        open_positions: list[dict[str, Any]],
        feature_vector: Any,
        strategy_mode_context: dict[str, Any] | None = None,
    ) -> Any:
        del strategy_mode_context, feature_vector
        model_positions = [
            position for position in open_positions if position.get("model_name") == model_name
        ]
        return self.risk_engine.assess(
            decision,
            current_positions=model_positions,
            account_balance=await self.account_balance_provider(model_name),
        )
