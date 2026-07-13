"""Suspicious symbol guard for new entry decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

SymbolNormalizer = Callable[[str | None], str]


@dataclass(frozen=True, slots=True)
class EntrySuspiciousSymbolPolicy:
    """Reject only structurally invalid empty symbols."""

    normalize_symbol: SymbolNormalizer

    def reason(self, symbol: str | None) -> str | None:
        """Return a block reason for blank or suspicious entry symbols."""

        normalized = self.normalize_symbol(symbol).strip().upper()
        if not normalized:
            return "币种符号为空，跳过新开仓分析。"

        return None
