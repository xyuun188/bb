"""Advisory symbol quarantine reason for recent realized-loss feedback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

NormalizeSymbol = Callable[[str | None], str | None]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class EntrySymbolProfitQuarantinePolicy:
    """Explain recent symbol-level realized-loss quarantine without blocking AI analysis."""

    normalize_symbol: NormalizeSymbol | None = None

    def reason(self, symbol: str, strategy: dict[str, Any] | None) -> str | None:
        """Return an advisory reason when recent all-side performance is in cooldown."""

        if not isinstance(strategy, dict):
            return None
        profiles = strategy.get("symbol_side_performance")
        if not isinstance(profiles, dict):
            return None

        symbol_key = self._symbol_key(symbol)
        profile = profiles.get(f"{symbol_key}|all")
        if not isinstance(profile, dict) or not profile.get("cooldown"):
            return None

        pnl = _safe_float(profile.get("pnl"), 0.0)
        losses = int(profile.get("losses") or 0)
        count = int(profile.get("count") or 0)
        reason = str(profile.get("cooldown_reason") or "近期真实盈亏表现偏弱")
        return (
            f"{symbol_key} 进入亏损隔离观察：最近 {count} 笔真实平仓累计 {pnl:.2f} U，"
            f"亏损 {losses} 笔。原因：{reason}。该提示不会直接拦截 AI 分析，"
            "但会作为历史亏损证据提醒决策链降低质量预期和仓位。"
        )

    def _symbol_key(self, symbol: str) -> str:
        if self.normalize_symbol is None:
            return str(symbol or "")
        return self.normalize_symbol(symbol) or str(symbol or "")
