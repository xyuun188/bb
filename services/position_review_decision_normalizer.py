"""Decision normalization rules for position-review analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_brain.base_model import Action, DecisionOutput

SymbolNormalizer = Callable[[str | None], str]


class PositionReviewDecisionNormalizer:
    """Normalizes position-review entry signals against existing positions."""

    def __init__(self, symbol_normalizer: SymbolNormalizer) -> None:
        self._symbol_normalizer = symbol_normalizer

    def normalize(
        self,
        decision: DecisionOutput,
        positions: list[dict[str, Any]],
    ) -> DecisionOutput:
        """Turn opposite entry signals into close-first actions during review."""

        if not decision.is_entry:
            return decision

        target_side = "long" if decision.action == Action.LONG else "short"
        decision_symbol = self._symbol_normalizer(decision.symbol)
        existing_sides = {
            str(pos.get("side") or "").lower()
            for pos in positions
            if self._symbol_normalizer(pos.get("symbol")) == decision_symbol
        }
        if target_side in existing_sides:
            decision.reasoning += " [持仓复盘：同方向信号，按加仓候选进入风控和仓位上限检查。]"
            return decision

        close_action = None
        if target_side == "long" and "short" in existing_sides:
            close_action = Action.CLOSE_SHORT
        elif target_side == "short" and "long" in existing_sides:
            close_action = Action.CLOSE_LONG

        if close_action is None:
            return decision

        return DecisionOutput(
            model_name=decision.model_name,
            symbol=decision.symbol,
            action=close_action,
            confidence=max(decision.confidence, 0.62),
            reasoning=(
                f"{decision.reasoning} [持仓复盘：发现反向开仓信号；"
                "先平掉现有仓位，不在同一订单里直接反手。]"
            ),
            position_size_pct=1.0,
            suggested_leverage=1.0,
            stop_loss_pct=decision.stop_loss_pct,
            take_profit_pct=decision.take_profit_pct,
            cross_check_for=decision.cross_check_for,
            raw_response=decision.raw_response,
            feature_snapshot=decision.feature_snapshot,
        )
