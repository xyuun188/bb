"""Track unfinished market-analysis candidates across scheduler rounds."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

NormalizeSymbol = Callable[[Any], str]


@dataclass(slots=True)
class MarketAnalysisDeferredCandidate:
    symbol: str
    reason: str
    defer_count: int = 1
    first_deferred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_deferred_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class MarketAnalysisDeferTracker:
    """Keep an ordered retry queue without letting repeated failures own the head."""

    normalize_symbol: NormalizeSymbol
    max_candidates: int = 512
    candidates: dict[str, MarketAnalysisDeferredCandidate] = field(default_factory=dict)

    def begin(self, symbol: str, *, now: datetime | None = None) -> None:
        key = self.normalize_symbol(symbol)
        if not key or key in self.candidates:
            return
        observed_at = now or datetime.now(UTC)
        self.candidates[key] = MarketAnalysisDeferredCandidate(
            symbol=str(symbol),
            reason="scheduled",
            first_deferred_at=observed_at,
            last_deferred_at=observed_at,
        )
        self._trim()

    def begin_many(self, symbols: Iterable[str], *, now: datetime | None = None) -> None:
        for symbol in symbols:
            self.begin(symbol, now=now)

    def defer(self, symbol: str, reason: str, *, now: datetime | None = None) -> None:
        key = self.normalize_symbol(symbol)
        if not key:
            return
        observed_at = now or datetime.now(UTC)
        previous = self.candidates.pop(key, None)
        self.candidates[key] = MarketAnalysisDeferredCandidate(
            symbol=str(symbol),
            reason=str(reason or "unspecified"),
            defer_count=(previous.defer_count + 1 if previous is not None else 1),
            first_deferred_at=(previous.first_deferred_at if previous is not None else observed_at),
            last_deferred_at=observed_at,
        )

        self._trim()

    def _trim(self) -> None:
        while len(self.candidates) > max(int(self.max_candidates), 1):
            self.candidates.pop(next(iter(self.candidates)))

    def defer_many(
        self,
        symbols: Iterable[str],
        reason: str,
        *,
        now: datetime | None = None,
    ) -> None:
        for symbol in symbols:
            self.defer(symbol, reason, now=now)

    def complete(self, symbol: str) -> None:
        key = self.normalize_symbol(symbol)
        if key:
            self.candidates.pop(key, None)

    def retain(self, eligible_symbols: Iterable[str]) -> None:
        eligible = {key for symbol in eligible_symbols if (key := self.normalize_symbol(symbol))}
        for key in list(self.candidates):
            if key not in eligible:
                self.candidates.pop(key, None)

    def ordered_symbols(self) -> list[str]:
        return [candidate.symbol for candidate in self.candidates.values()]

    def snapshot(
        self,
        *,
        now: datetime | None = None,
        coverage_target_seconds: float = 30 * 60,
    ) -> dict[str, Any]:
        observed_at = now or datetime.now(UTC)
        target_seconds = max(float(coverage_target_seconds), 0.0)
        rows = list(self.candidates.values())
        reasons = Counter(candidate.reason for candidate in rows)
        waits = [
            max((observed_at - candidate.first_deferred_at).total_seconds(), 0.0)
            for candidate in rows
        ]
        overdue = [
            candidate.symbol
            for candidate, wait_seconds in zip(rows, waits, strict=True)
            if wait_seconds >= target_seconds
        ]
        return {
            "read_only": True,
            "is_entry_gate": False,
            "deferred_count": len(rows),
            "deferred_symbols": [candidate.symbol for candidate in rows[:50]],
            "reason_counts": [
                {"reason": reason, "count": count} for reason, count in sorted(reasons.items())
            ],
            "oldest_defer_count": rows[0].defer_count if rows else 0,
            "max_defer_count": max((candidate.defer_count for candidate in rows), default=0),
            "coverage_target_seconds": round(target_seconds, 3),
            "oldest_wait_seconds": round(max(waits, default=0.0), 3),
            "overdue_count": len(overdue),
            "overdue_symbols": overdue[:50],
            "pending_coverage_window_met": not overdue,
            "reason": (
                "Candidates that did not finish feature preparation, scheduling, model "
                "inference, or decision persistence remain ordered for a later round. "
                "A repeated failure moves to the queue tail so it cannot starve other work."
            ),
        }
