"""Track unfinished market-analysis candidates across scheduler rounds."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

NormalizeSymbol = Callable[[Any], str]


@dataclass(slots=True)
class MarketAnalysisDeferredCandidate:
    symbol: str
    reason: str
    defer_count: int = 1


@dataclass(slots=True)
class MarketAnalysisDeferTracker:
    """Keep an ordered retry queue without letting repeated failures own the head."""

    normalize_symbol: NormalizeSymbol
    max_candidates: int = 512
    candidates: dict[str, MarketAnalysisDeferredCandidate] = field(default_factory=dict)

    def defer(self, symbol: str, reason: str) -> None:
        key = self.normalize_symbol(symbol)
        if not key:
            return
        previous = self.candidates.pop(key, None)
        self.candidates[key] = MarketAnalysisDeferredCandidate(
            symbol=str(symbol),
            reason=str(reason or "unspecified"),
            defer_count=(previous.defer_count + 1 if previous is not None else 1),
        )
        while len(self.candidates) > max(int(self.max_candidates), 1):
            self.candidates.pop(next(iter(self.candidates)))

    def defer_many(self, symbols: Iterable[str], reason: str) -> None:
        for symbol in symbols:
            self.defer(symbol, reason)

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

    def snapshot(self) -> dict[str, Any]:
        rows = list(self.candidates.values())
        reasons = Counter(candidate.reason for candidate in rows)
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
            "reason": (
                "Candidates that did not finish feature preparation, scheduling, model "
                "inference, or decision persistence remain ordered for a later round. "
                "A repeated failure moves to the queue tail so it cannot starve other work."
            ),
        }
