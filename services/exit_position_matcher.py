"""Exit-position matching helpers.

Exit validation needs the same symbol/side/model/quantity checks in multiple
places.  This component centralizes those checks while preserving the stricter
model matching used before exchange submission and the looser context matching
used for aggregate risk guards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class ExitPositionMatcher:
    normalize_symbol: Callable[[Any], str]

    def target_side(self, decision: DecisionOutput) -> str:
        if decision.action == Action.CLOSE_LONG:
            return "long"
        if decision.action == Action.CLOSE_SHORT:
            return "short"
        return ""

    def matching_positions(
        self,
        positions: list[dict[str, Any]] | None,
        model_name: str,
        decision: DecisionOutput,
        *,
        require_model_name: bool = True,
    ) -> list[dict[str, Any]]:
        if not decision.is_exit:
            return []
        target_side = self.target_side(decision)
        target_symbol = self.normalize_symbol(decision.symbol)
        matches: list[dict[str, Any]] = []
        for pos in positions or []:
            pos_model = str(pos.get("model_name") or "")
            if require_model_name:
                if pos_model != model_name:
                    continue
            elif pos_model and pos_model != model_name:
                continue
            if self.normalize_symbol(pos.get("symbol")) != target_symbol:
                continue
            if str(pos.get("side") or "").lower() != target_side:
                continue
            if pos.get("is_open", True) is False:
                continue
            quantity = _safe_float(
                pos.get("quantity") or pos.get("contracts") or pos.get("sz"),
                0.0,
            )
            if abs(quantity) > 0:
                matches.append(pos)
        return matches

    def has_matching_position(
        self,
        positions: list[dict[str, Any]] | None,
        model_name: str,
        decision: DecisionOutput,
    ) -> bool:
        if not decision.is_exit:
            return True
        return bool(
            self.matching_positions(
                positions,
                model_name,
                decision,
                require_model_name=True,
            )
        )
