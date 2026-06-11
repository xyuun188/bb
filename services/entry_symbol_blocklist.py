"""Temporary entry blocklist and exchange-error classification for entry safety."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from web_dashboard.api.text_sanitize import sanitize_text

NormalizeSymbol = Callable[[str | None], str | None]

UNTRADABLE_SYMBOL_BLOCK_HOURS = 24.0
TRANSIENT_ENTRY_BLOCK_MINUTES = 20.0
PRICE_GUARD_ENTRY_BLOCK_MINUTES = 8.0

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class EntrySymbolBlocklistPolicy:
    """Own temporary entry blocks and non-tradable exchange-error recognition."""

    normalize_symbol: NormalizeSymbol
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    blocked_symbols: dict[str, dict[str, Any]] = field(default_factory=dict)

    def is_untradable_exchange_error(self, text: Any) -> bool:
        value = str(text or "").lower()
        return (
            "51155" in value
            or "can't trade this pair" in value
            or "local compliance restrictions" in value
        )

    def is_transient_entry_exchange_error(self, text: Any) -> bool:
        value = str(text or "").lower()
        return (
            "51290" in value
            or "trading bot engine currently upgrading" in value
            or "engine currently upgrading" in value
            or ("open interest" in value and "platform" in value and "limit" in value)
            or "has reached the platform's limit" in value
            or ("try again later" in value and "okx" in value)
        )

    def transient_entry_block_minutes(self, text: Any) -> float:
        value = str(text or "").lower()
        if (
            "open interest" in value and "platform" in value and "limit" in value
        ) or "has reached the platform's limit" in value:
            return 45.0
        return TRANSIENT_ENTRY_BLOCK_MINUTES

    def is_entry_price_guard_skip(self, text: Any) -> bool:
        value = str(text or "")
        return (
            "下单前价格" in value
            or "避免追高" in value
            or "避免追空" in value
            or "行情变化太快" in value
            or "下单前没有重新拿到最新价格" in value
            or "下单前行情质量复核未通过" in value
        )

    def remember_temporary_entry_block(
        self,
        symbol: str | None,
        reason: Any,
        minutes: float = TRANSIENT_ENTRY_BLOCK_MINUTES,
    ) -> None:
        normalized = self._normalize(symbol)
        if not normalized:
            return
        until = self.clock() + timedelta(minutes=max(float(minutes or 0), 1.0))
        self.blocked_symbols[normalized] = {
            "until": until,
            "reason": (
                f"临时跳过新开仓：" f"{str(sanitize_text(reason) or '近期该币种开仓未成功')[:460]}"
            ),
        }
        logger.warning(
            "symbol temporarily blocked for new entries",
            symbol=normalized,
            until=until.isoformat(),
            reason=str(reason or "")[:220],
        )

    def remember_untradable_symbol(
        self,
        symbol: str | None,
        reason: Any,
        hours: float = UNTRADABLE_SYMBOL_BLOCK_HOURS,
    ) -> None:
        normalized = self._normalize(symbol)
        if not normalized:
            return
        until = self.clock() + timedelta(hours=max(float(hours or 0), 1.0))
        self.blocked_symbols[normalized] = {
            "until": until,
            "reason": str(sanitize_text(reason) or "OKX 提示该交易对当前不可交易")[:500],
        }
        logger.warning(
            "symbol temporarily blocked as untradable",
            symbol=normalized,
            until=until.isoformat(),
        )

    def blocked_symbol_reason(self, symbol: str | None) -> str | None:
        normalized = self._normalize(symbol)
        if not normalized:
            return None
        item = self.blocked_symbols.get(normalized)
        if not item:
            return None
        until = item.get("until")
        if isinstance(until, datetime) and until > self.clock():
            return str(sanitize_text(item.get("reason")) or "该交易对暂时不可交易")
        self.blocked_symbols.pop(normalized, None)
        return None

    def _normalize(self, symbol: str | None) -> str | None:
        return self.normalize_symbol(symbol)
