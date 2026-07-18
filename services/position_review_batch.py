"""Batch selection for governed position-review work."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from services.analysis_budget import (
    POSITION_REVIEW_MAX_GROUPS_PER_ROUND,
    POSITION_REVIEW_URGENT_EXIT_MAX_GROUPS_PER_ROUND,
)

PositionGroupItem = tuple[tuple[str, str], list[dict[str, Any]]]
PositionReviewDeferCountProvider = Callable[[tuple[str, str]], int]
UrgentExitChecker = Callable[[dict[str, Any] | None], bool]


@dataclass(frozen=True, slots=True)
class PositionReviewBatchSelection:
    selected_items: list[PositionGroupItem]
    skipped_items: list[PositionGroupItem]
    selected_keys: set[tuple[str, str]]
    next_cursor: int
    max_groups: int
    total_groups: int
    urgent_exit_count: int
    deferred_exit_count: int
    loss_watch_count: int
    profit_exit_count: int
    priority_selected_count: int

    @property
    def limited(self) -> bool:
        return bool(self.skipped_items)


@dataclass(frozen=True, slots=True)
class PositionReviewBatchPolicy:
    """Prioritize unified dynamic exits without score tiers or marker rules."""

    urgent_exit_checker: UrgentExitChecker
    max_groups_per_round: int = POSITION_REVIEW_MAX_GROUPS_PER_ROUND
    priority_max_groups_per_round: int | None = None
    urgent_exit_max_groups_per_round: int = POSITION_REVIEW_URGENT_EXIT_MAX_GROUPS_PER_ROUND

    def select(
        self,
        grouped_items: list[PositionGroupItem],
        fast_scan: dict[tuple[str, str], dict[str, Any]],
        *,
        max_groups_override: int | None = None,
        hard_max_groups_override: int | None = None,
        defer_count_provider: PositionReviewDeferCountProvider | None = None,
        position_entry_pause_reason: str | None = None,
        cursor: int = 0,
    ) -> PositionReviewBatchSelection:
        del position_entry_pause_reason
        sorted_items = sorted(
            grouped_items,
            key=lambda item: (-self._close_fraction(fast_scan.get(item[0], {})), item[0][1]),
        )
        total_groups = len(sorted_items)
        max_groups = max(1, int(max_groups_override or self.max_groups_per_round))
        urgent_items = [
            item for item in sorted_items if self.urgent_exit_checker(fast_scan.get(item[0], {}))
        ]
        if urgent_items:
            max_groups = max(
                max_groups,
                min(total_groups, self.urgent_exit_max_groups_per_round, len(urgent_items)),
            )
        if hard_max_groups_override is not None:
            max_groups = min(max_groups, max(1, int(hard_max_groups_override)))

        selected_items = self._unique_items(urgent_items)[:max_groups]
        selected_keys = {item[0] for item in selected_items}
        remaining = [item for item in sorted_items if item[0] not in selected_keys]
        remaining_slots = max(max_groups - len(selected_items), 0)
        next_cursor = cursor
        if remaining_slots and remaining:
            start = cursor % len(remaining)
            rotated = remaining[start:] + remaining[:start]
            selected_items.extend(rotated[:remaining_slots])
            next_cursor = (start + remaining_slots) % len(remaining)
            selected_keys = {item[0] for item in selected_items}

        skipped_items = [item for item in sorted_items if item[0] not in selected_keys]
        defer_count_provider = defer_count_provider or (lambda _key: 0)
        deferred_exit_count = sum(1 for item in urgent_items if defer_count_provider(item[0]) > 0)
        loss_watch_count = sum(
            1 for item in urgent_items if self._fee_after_pnl(fast_scan.get(item[0], {})) < 0.0
        )
        profit_exit_count = sum(
            1 for item in urgent_items if self._profit_retrace(fast_scan.get(item[0], {})) > 0.0
        )
        return PositionReviewBatchSelection(
            selected_items=selected_items,
            skipped_items=skipped_items,
            selected_keys=selected_keys,
            next_cursor=next_cursor,
            max_groups=max_groups,
            total_groups=total_groups,
            urgent_exit_count=len(urgent_items),
            deferred_exit_count=deferred_exit_count,
            loss_watch_count=loss_watch_count,
            profit_exit_count=profit_exit_count,
            priority_selected_count=sum(
                1 for item in selected_items if self._eligible(fast_scan.get(item[0], {}))
            ),
        )

    @staticmethod
    def _dynamic(scan: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(scan, dict):
            return {}
        policy = scan.get("dynamic_exit_policy")
        return policy if isinstance(policy, dict) else {}

    @classmethod
    def _eligible(cls, scan: dict[str, Any] | None) -> bool:
        return bool(
            isinstance(scan, dict)
            and scan.get("dynamic_exit_eligible") is True
            and cls._dynamic(scan).get("eligible") is True
        )

    @classmethod
    def _close_fraction(cls, scan: dict[str, Any] | None) -> float:
        try:
            return float(cls._dynamic(scan).get("close_fraction") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _fee_after_pnl(cls, scan: dict[str, Any] | None) -> float:
        try:
            return float(cls._dynamic(scan).get("fee_after_unrealized_pnl_usdt") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _profit_retrace(cls, scan: dict[str, Any] | None) -> float:
        try:
            return float(cls._dynamic(scan).get("profit_retrace_ratio") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _unique_items(items: list[PositionGroupItem]) -> list[PositionGroupItem]:
        selected: list[PositionGroupItem] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            if item[0] in seen:
                continue
            seen.add(item[0])
            selected.append(item)
        return selected
