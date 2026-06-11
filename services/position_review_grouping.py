"""Group open positions for position-review analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PositionReviewGroupKey = tuple[str, str]
PositionReviewGroupItem = tuple[PositionReviewGroupKey, list[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class PositionReviewGroupingPolicy:
    """Build model+symbol groups from open positions."""

    def group(
        self,
        open_positions: list[dict[str, Any]],
    ) -> dict[PositionReviewGroupKey, list[dict[str, Any]]]:
        grouped: dict[PositionReviewGroupKey, list[dict[str, Any]]] = {}
        for position in open_positions:
            model_name = str(position.get("model_name") or "")
            if not model_name:
                continue
            key = (model_name, str(position["symbol"]))
            grouped.setdefault(key, []).append(position)
        return grouped

    def items(self, open_positions: list[dict[str, Any]]) -> list[PositionReviewGroupItem]:
        return list(self.group(open_positions).items())
