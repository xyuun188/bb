"""Track deferred position-review groups across analysis rounds."""

from __future__ import annotations

from dataclasses import dataclass, field

PositionReviewKey = tuple[str, str]


@dataclass(slots=True)
class PositionReviewDeferTracker:
    """Remember how many consecutive rounds a group waited for slow review."""

    counts: dict[PositionReviewKey, int] = field(default_factory=dict)

    def count(self, key: PositionReviewKey) -> int:
        """Return the current defer count for a model/symbol group."""

        return int(self.counts.get(key, 0) or 0)

    def clear(self, key: PositionReviewKey) -> None:
        """Clear defer state when a group is selected or no longer needs slow review."""

        self.counts.pop(key, None)

    def clear_many(self, keys: set[PositionReviewKey]) -> None:
        """Clear defer state for all selected groups in the current batch."""

        for key in keys:
            self.clear(key)

    def apply_plan_count(self, key: PositionReviewKey, defer_count: int) -> None:
        """Store a planned defer count, or clear it when the plan resets to zero."""

        next_count = int(defer_count or 0)
        if next_count > 0:
            self.counts[key] = next_count
        else:
            self.clear(key)
