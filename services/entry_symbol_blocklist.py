"""Temporary entry blocklist and exchange-error classification for entry safety."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from ai_brain.base_model import Action, DecisionOutput
from services.okx_error_classifier import extract_okx_error, is_okx_temporary_service_error
from services.trading_params import DEFAULT_TRADING_PARAMS
from web_dashboard.api.text_sanitize import sanitize_text

NormalizeSymbol = Callable[[str | None], str | None]

UNTRADABLE_SYMBOL_BLOCK_HOURS = 24.0
TRANSIENT_ENTRY_BLOCK_MINUTES = 20.0
PRICE_GUARD_ENTRY_BLOCK_MINUTES = DEFAULT_TRADING_PARAMS.entry_price_guard.entry_block_minutes
EXCHANGE_RECOVERY_BLOCK_MINUTES = 20.0
EXCHANGE_RECOVERY_ERROR_CODES = frozenset({"50026", "59247"})

logger = structlog.get_logger(__name__)

UNTRADABLE_EXCHANGE_ERROR_MARKERS = (
    "51155",
    "51028",
    "contract under delivery",
    "can't trade this pair",
    "cannot trade this pair",
    "can not trade this pair",
    "local compliance restrictions",
    "not currently tradable",
    "not available for trading",
    "not available to trade",
    "trading unavailable",
    "trading is unavailable",
    "instrument suspended",
    "symbol suspended",
    "market suspended",
    "temporarily suspended",
    "currently not tradable",
    "currently unavailable for trading",
    "当前不可交易",
    "暂时不可交易",
    "交易对当前不可交易",
    "该交易对当前不可交易",
    "当前不支持交易",
    "暂停交易",
)


@dataclass(slots=True)
class EntrySymbolBlocklistPolicy:
    """Own temporary entry blocks and non-tradable exchange-error recognition."""

    normalize_symbol: NormalizeSymbol
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    blocked_symbols: dict[str, dict[str, Any]] = field(default_factory=dict)
    exchange_recovery_blocks: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict
    )

    def is_untradable_exchange_error(self, text: Any) -> bool:
        value = str(text or "").lower()
        return any(marker in value for marker in UNTRADABLE_EXCHANGE_ERROR_MARKERS)

    def is_transient_entry_exchange_error(self, text: Any) -> bool:
        value = str(text or "").lower()
        return (
            is_okx_temporary_service_error(text)
            or "51290" in value
            or "trading bot engine currently upgrading" in value
            or "engine currently upgrading" in value
            or ("open interest" in value and "platform" in value and "limit" in value)
            or "has reached the platform's limit" in value
            or ("try again later" in value and "okx" in value)
        )

    def transient_entry_block_minutes(self, text: Any) -> float:
        value = str(text or "").lower()
        if is_okx_temporary_service_error(text):
            return 12.0
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

    def is_exchange_recovery_error(self, value: Any) -> bool:
        code, message = extract_okx_error(value)
        text = f"{value or ''} {message or ''}".lower()
        return bool(
            code in EXCHANGE_RECOVERY_ERROR_CODES
            or "operation failed" in text
            or "system error. try again later" in text
        )

    def remember_exchange_rejection(
        self,
        decision: DecisionOutput,
        raw_response: Any,
        minutes: float = EXCHANGE_RECOVERY_BLOCK_MINUTES,
    ) -> None:
        raw = raw_response if isinstance(raw_response, dict) else {}
        if not raw.get("okx_rejection") or not self.is_exchange_recovery_error(raw):
            return
        key = self._recovery_key(decision)
        if key is None:
            return
        code = str(raw.get("okx_error_code") or extract_okx_error(raw)[0] or "").strip()
        leverage_check = raw.get("leverage_check")
        leverage_check = leverage_check if isinstance(leverage_check, dict) else {}
        actual_leverage = self._safe_positive_float(
            leverage_check.get("actual_leverage")
            or leverage_check.get("target_leverage")
        )
        until = self.clock() + timedelta(minutes=max(float(minutes or 0), 1.0))
        self.exchange_recovery_blocks[key] = {
            "until": until,
            "reason": (
                f"OKX 拒绝了该方向上一次相同状态的开仓请求"
                f"{f'（错误码 {code}）' if code else ''}。系统会等待候选证据或交易所仓位状态变化后再重试。"
            ),
            "candidate_fingerprint": self._candidate_fingerprint(decision),
            "position_leverage": actual_leverage,
            "okx_error_code": code or None,
        }
        logger.warning(
            "symbol side temporarily blocked after OKX rejection",
            symbol=key[0],
            side=key[1],
            okx_error_code=code or None,
            until=until.isoformat(),
        )

    def exchange_recovery_block_reason(
        self,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> str | None:
        key = self._recovery_key(decision)
        if key is None:
            return None
        item = self.exchange_recovery_blocks.get(key)
        if not item:
            return None
        until = item.get("until")
        if not isinstance(until, datetime) or until <= self.clock():
            self.exchange_recovery_blocks.pop(key, None)
            return None
        if item.get("candidate_fingerprint") != self._candidate_fingerprint(decision):
            self.exchange_recovery_blocks.pop(key, None)
            return None
        previous_leverage = self._safe_positive_float(item.get("position_leverage"))
        current_leverage = self._matching_position_leverage(decision, open_positions)
        if (
            previous_leverage is not None
            and current_leverage is not None
            and not math.isclose(previous_leverage, current_leverage, rel_tol=1e-6)
        ):
            self.exchange_recovery_blocks.pop(key, None)
            return None
        return str(sanitize_text(item.get("reason")) or "OKX 相同状态拒单仍在恢复冷却中")

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

    def _recovery_key(self, decision: DecisionOutput) -> tuple[str, str] | None:
        normalized = self._normalize(decision.symbol)
        side = (
            "long"
            if decision.action == Action.LONG
            else "short" if decision.action == Action.SHORT else ""
        )
        if not normalized or not side:
            return None
        return normalized, side

    def _candidate_fingerprint(self, decision: DecisionOutput) -> str:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        context_keys = (
            "strategy_mode",
            "strategy_learning_context",
            "opportunity_score",
            "profit_first_trade_plan",
            "profit_first_position_ladder",
            "entry_evidence",
            "dynamic_evidence",
            "market_regime",
            "selected_entry_metrics",
        )
        payload = {
            "action": decision.action.value,
            "confidence": self._stable_value(decision.confidence),
            "position_size_pct": self._stable_value(decision.position_size_pct),
            "suggested_leverage": self._stable_value(decision.suggested_leverage),
            "stop_loss_pct": self._stable_value(decision.stop_loss_pct),
            "take_profit_pct": self._stable_value(decision.take_profit_pct),
            "feature_snapshot": self._stable_value(decision.feature_snapshot or {}),
            "decision_context": {
                key: self._stable_value(raw.get(key)) for key in context_keys if key in raw
            },
        }
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]

    def _matching_position_leverage(
        self,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]] | None,
    ) -> float | None:
        key = self._recovery_key(decision)
        if key is None:
            return None
        symbol, side = key
        for position in open_positions or []:
            position_symbol = self._normalize(
                str(position.get("symbol") or (position.get("info") or {}).get("instId") or "")
            )
            position_side = str(
                position.get("side") or (position.get("info") or {}).get("posSide") or ""
            ).lower()
            if position_symbol == symbol and position_side == side:
                return self._safe_positive_float(
                    position.get("leverage")
                    or (position.get("info") or {}).get("lever")
                )
        return None

    @staticmethod
    def _safe_positive_float(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) and result > 0 else None

    @classmethod
    def _stable_value(cls, value: Any, *, depth: int = 0) -> Any:
        if depth > 6:
            return None
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                return None
            return float(f"{number:.4g}")
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key in sorted(value)[:120]:
                lowered = str(key).lower()
                if any(
                    marker in lowered
                    for marker in ("timestamp", "latency", "duration", "trace", "request_id")
                ):
                    continue
                result[str(key)] = cls._stable_value(value[key], depth=depth + 1)
            return result
        if isinstance(value, (list, tuple)):
            return [cls._stable_value(item, depth=depth + 1) for item in list(value)[:30]]
        return str(value)[:200]
