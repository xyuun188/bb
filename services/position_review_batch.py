"""Batch selection for position-review AI work."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from services.analysis_budget import (
    POSITION_REVIEW_FAST_ADD_SCORE,
    POSITION_REVIEW_FAST_EXIT_SCORE,
    POSITION_REVIEW_MAX_GROUPS_PER_ROUND,
    POSITION_REVIEW_URGENT_EXIT_MAX_GROUPS_PER_ROUND,
)

PositionGroupItem = tuple[tuple[str, str], list[dict[str, Any]]]
PositionReviewDeferCountProvider = Callable[[tuple[str, str]], int]
UrgentExitChecker = Callable[[dict[str, Any] | None], bool]

PROFIT_EXIT_REVIEW_MARKERS = (
    "profit_retrace",
    "profit_lock_candidate",
    "portfolio_profit_protection_focus",
)


@dataclass(frozen=True, slots=True)
class PositionReviewBatchSelection:
    """Selected and skipped position-review groups for one round."""

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
    """Pick which position groups deserve slow AI review this round."""

    urgent_exit_checker: UrgentExitChecker
    max_groups_per_round: int = POSITION_REVIEW_MAX_GROUPS_PER_ROUND
    priority_max_groups_per_round: int | None = None
    urgent_exit_max_groups_per_round: int = POSITION_REVIEW_URGENT_EXIT_MAX_GROUPS_PER_ROUND
    fast_exit_score: float = POSITION_REVIEW_FAST_EXIT_SCORE
    fast_add_score: float = POSITION_REVIEW_FAST_ADD_SCORE
    profit_exit_markers: tuple[str, ...] = PROFIT_EXIT_REVIEW_MARKERS

    def select(
        self,
        grouped_items: list[PositionGroupItem],
        fast_scan: dict[tuple[str, str], dict[str, Any]],
        *,
        max_groups_override: int | None = None,
        defer_count_provider: PositionReviewDeferCountProvider | None = None,
        position_entry_pause_reason: str | None = None,
        cursor: int = 0,
    ) -> PositionReviewBatchSelection:
        sorted_items = sorted(
            grouped_items,
            key=lambda item: (
                -self._scan_score(fast_scan, item[0], "priority_score"),
                item[0][1],
            ),
        )
        total_groups = len(sorted_items)
        max_groups = max(1, int(max_groups_override or self.max_groups_per_round))
        if total_groups <= max_groups:
            selected_keys = {item[0] for item in sorted_items}
            return PositionReviewBatchSelection(
                selected_items=sorted_items,
                skipped_items=[],
                selected_keys=selected_keys,
                next_cursor=cursor,
                max_groups=max_groups,
                total_groups=total_groups,
                urgent_exit_count=0,
                deferred_exit_count=0,
                loss_watch_count=0,
                profit_exit_count=0,
                priority_selected_count=sum(
                    1
                    for item in sorted_items
                    if self._scan_score(fast_scan, item[0], "priority_score") >= self.fast_add_score
                ),
            )

        defer_count_provider = defer_count_provider or (lambda _key: 0)
        urgent_exit_items = [
            item for item in sorted_items if self.urgent_exit_checker(fast_scan.get(item[0], {}))
        ]
        urgent_keys = {item[0] for item in urgent_exit_items}
        deferred_exit_items = [
            item
            for item in sorted_items
            if item[0] not in urgent_keys
            and defer_count_provider(item[0]) >= 2
            and self._scan_score(fast_scan, item[0], "exit_score") >= self.fast_exit_score
        ]
        urgent_exit_items.extend(deferred_exit_items)
        urgent_keys = {item[0] for item in urgent_exit_items}
        if urgent_exit_items:
            max_groups = max(
                max_groups,
                min(
                    total_groups,
                    self.urgent_exit_max_groups_per_round,
                    len(urgent_exit_items) + 2,
                ),
            )

        loss_watch_items = [
            item
            for item in sorted_items
            if item[0] not in urgent_keys and "loss_watch" in self._scan_reason(fast_scan, item[0])
        ]
        loss_watch_keys = {item[0] for item in loss_watch_items}
        profit_exit_items = [
            item
            for item in sorted_items
            if item[0] not in urgent_keys
            and self._scan_score(fast_scan, item[0], "exit_score") >= self.fast_exit_score
            and any(
                marker in self._scan_reason(fast_scan, item[0])
                for marker in self.profit_exit_markers
            )
        ]
        profit_exit_keys = {item[0] for item in profit_exit_items}
        priority_items = [
            item
            for item in sorted_items
            if self._scan_score(fast_scan, item[0], "priority_score") >= self.fast_add_score
        ]
        priority_keys = {item[0] for item in priority_items}
        normal_items = [
            item
            for item in sorted_items
            if item[0] not in priority_keys
            and item[0] not in urgent_keys
            and item[0] not in loss_watch_keys
            and item[0] not in profit_exit_keys
        ]

        priority_slots = min(
            len(priority_items),
            (
                max_groups
                if self.priority_max_groups_per_round is None
                else max(0, min(max_groups, int(self.priority_max_groups_per_round)))
            ),
        )
        selected_items = self._unique_items(
            urgent_exit_items + profit_exit_items + loss_watch_items
        )
        selected_keys = {item[0] for item in selected_items}
        remaining_priority_slots = max(priority_slots - len(selected_items), 0)
        if remaining_priority_slots > 0:
            exit_items = [
                item
                for item in priority_items
                if item[0] not in selected_keys
                and self._scan_score(fast_scan, item[0], "exit_score") >= self.fast_exit_score
            ]
            exit_keys = {item[0] for item in exit_items}
            add_items = [
                item
                for item in priority_items
                if item[0] not in selected_keys
                and item[0] not in exit_keys
                and not position_entry_pause_reason
            ]
            selected_items.extend((exit_items + add_items)[:remaining_priority_slots])
            selected_items = self._unique_items(selected_items)
            selected_keys = {item[0] for item in selected_items}

        remaining_slots = max_groups - len(selected_items)
        next_cursor = cursor
        if remaining_slots > 0 and normal_items:
            start = cursor % len(normal_items)
            rotated = normal_items[start:] + normal_items[:start]
            selected_items.extend(rotated[:remaining_slots])
            next_cursor = (start + remaining_slots) % len(normal_items)
            selected_items = self._unique_items(selected_items)
            selected_keys = {item[0] for item in selected_items}

        if len(selected_items) < max_groups:
            fallback_items = [item for item in sorted_items if item[0] not in selected_keys]
            selected_items.extend(fallback_items[: max_groups - len(selected_items)])
            selected_items = self._unique_items(selected_items)
            selected_keys = {item[0] for item in selected_items}

        skipped_items = [item for item in sorted_items if item[0] not in selected_keys]
        return PositionReviewBatchSelection(
            selected_items=selected_items,
            skipped_items=skipped_items,
            selected_keys=selected_keys,
            next_cursor=next_cursor,
            max_groups=max_groups,
            total_groups=total_groups,
            urgent_exit_count=len(urgent_exit_items),
            deferred_exit_count=len(deferred_exit_items),
            loss_watch_count=len(loss_watch_items),
            profit_exit_count=len(profit_exit_items),
            priority_selected_count=sum(1 for item in selected_items if item[0] in priority_keys),
        )

    @staticmethod
    def _scan_score(
        fast_scan: dict[tuple[str, str], dict[str, Any]],
        key: tuple[str, str],
        field: str,
    ) -> float:
        try:
            return float(fast_scan.get(key, {}).get(field, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _scan_reason(
        fast_scan: dict[tuple[str, str], dict[str, Any]],
        key: tuple[str, str],
    ) -> str:
        return str(fast_scan.get(key, {}).get("reason") or "")

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
