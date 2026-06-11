"""Symbol-universe helpers for market-entry scans."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from config.settings import ENSEMBLE_TRADER_NAME

NormalizeSymbol = Callable[[Any], str | None]
ReasonProvider = Callable[[str | None], str | None]


@dataclass(frozen=True, slots=True)
class BlockedSymbol:
    symbol: str
    reason: str


@dataclass(frozen=True, slots=True)
class SymbolFilterResult:
    symbols: list[str]
    skipped: list[str]


@dataclass(frozen=True, slots=True)
class BlockedSymbolFilterResult:
    symbols: list[str]
    skipped: list[BlockedSymbol]


@dataclass(frozen=True, slots=True)
class EntrySymbolUniversePolicy:
    """Normalize, de-duplicate, and filter symbols before market analysis."""

    normalize_symbol: NormalizeSymbol

    def dedupe_symbols(self, symbols: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for symbol in symbols or []:
            normalized = self.normalize_symbol(symbol)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    def open_position_symbol_keys(self, open_positions: list[dict] | None) -> set[str]:
        return {
            normalized
            for normalized in (
                self.normalize_symbol(position.get("symbol"))
                for position in (open_positions or [])
                if position.get("symbol")
            )
            if normalized
        }

    def open_position_group_count(self, open_positions: list[dict] | None) -> int:
        groups = {
            (
                str(position.get("model_name") or ENSEMBLE_TRADER_NAME),
                self.normalize_symbol(position.get("symbol")),
                str(position.get("side") or "").lower(),
            )
            for position in (open_positions or [])
            if position.get("is_open", True)
            and self.normalize_symbol(position.get("symbol"))
            and str(position.get("side") or "").lower() in {"long", "short"}
        }
        return len(groups)

    def filter_open_position_market_symbols(
        self,
        symbols: list[str],
        open_positions: list[dict] | None,
    ) -> SymbolFilterResult:
        open_symbol_keys = self.open_position_symbol_keys(open_positions)
        if not open_symbol_keys:
            return SymbolFilterResult(symbols=list(symbols or []), skipped=[])

        filtered: list[str] = []
        skipped: list[str] = []
        for symbol in symbols or []:
            normalized = self.normalize_symbol(symbol)
            if normalized in open_symbol_keys:
                skipped.append(str(normalized))
                continue
            filtered.append(symbol)
        return SymbolFilterResult(symbols=filtered, skipped=skipped)

    def filter_unclaimed_market_symbols(
        self,
        symbols: list[str],
        active_symbols: set[str],
    ) -> SymbolFilterResult:
        filtered: list[str] = []
        skipped: list[str] = []
        for symbol in symbols or []:
            normalized = self.normalize_symbol(symbol)
            if normalized in active_symbols:
                skipped.append(str(normalized))
                continue
            filtered.append(symbol)
        return SymbolFilterResult(symbols=filtered, skipped=skipped)

    def filter_blocked_new_symbols(
        self,
        symbols: list[str],
        open_positions: list[dict],
        suspicious_reason: ReasonProvider,
        blocked_reason: ReasonProvider,
    ) -> BlockedSymbolFilterResult:
        open_symbol_keys = self.open_position_symbol_keys(open_positions)
        filtered: list[str] = []
        skipped: list[BlockedSymbol] = []
        for symbol in symbols or []:
            normalized = self.normalize_symbol(symbol)
            reason = suspicious_reason(normalized) or blocked_reason(normalized)
            if reason and normalized not in open_symbol_keys:
                skipped.append(BlockedSymbol(symbol=str(normalized or symbol), reason=reason))
                continue
            filtered.append(symbol)
        return BlockedSymbolFilterResult(symbols=filtered, skipped=skipped)
