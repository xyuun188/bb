"""Suspicious symbol guard for new entry decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

SUSPICIOUS_NEW_SYMBOL_TOKENS = ("TEST", "DEMO", "DUMMY", "MOCK", "SAMPLE")

SymbolNormalizer = Callable[[str | None], str]


@dataclass(frozen=True, slots=True)
class EntrySuspiciousSymbolPolicy:
    """Block exchange test/demo instruments from new-entry analysis and execution."""

    normalize_symbol: SymbolNormalizer
    suspicious_tokens: tuple[str, ...] = SUSPICIOUS_NEW_SYMBOL_TOKENS

    def reason(self, symbol: str | None) -> str | None:
        """Return a block reason for blank or suspicious entry symbols."""

        normalized = self.normalize_symbol(symbol).strip().upper()
        if not normalized:
            return "币种符号为空，跳过新开仓分析。"

        base = normalized.split("/", 1)[0]
        if any(token in base for token in self.suspicious_tokens):
            return (
                f"{normalized} 看起来是测试/模拟/占位合约，"
                "不参与自动扫描、AI开仓分析或新开仓执行。"
            )
        return None
