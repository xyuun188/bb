"""
Notification service — sends alerts via Telegram and/or DingTalk.
Notifies on: trade executions, risk events, model promotions, errors.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

import httpx
import structlog

from config.settings import settings
from core.safe_output import bounded_redacted_text, safe_error_text, safe_response_error_text
from core.url_safety import normalize_https_webhook_url

logger = structlog.get_logger(__name__)
_MAX_NOTIFICATION_TEXT_CHARS = 3500


class NotificationService:
    """Sends notifications to configured channels."""

    def __init__(
        self,
        config: Any = settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = config
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._telegram_bot_token = self._load_telegram_bot_token()
        self._telegram_chat_id = str(self._settings.telegram_chat_id or "").strip()
        self._telegram_enabled = bool(self._telegram_bot_token and self._telegram_chat_id)
        self._dingtalk_webhook_url = self._load_dingtalk_webhook_url()
        self._dingtalk_enabled = bool(self._dingtalk_webhook_url)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
                transport=self._transport,
            )
        return self._client

    def _load_telegram_bot_token(self) -> str:
        try:
            return self._normalize_telegram_bot_token(self._settings.telegram_bot_token)
        except ValueError as exc:
            logger.warning(
                "telegram notification configuration disabled",
                error=safe_error_text(exc, limit=180),
            )
            return ""

    def _load_dingtalk_webhook_url(self) -> str:
        try:
            return normalize_https_webhook_url(
                self._settings.dingtalk_webhook_url,
                field_name="DingTalk webhook URL",
            )
        except ValueError as exc:
            logger.warning(
                "dingtalk notification configuration disabled",
                error=safe_error_text(exc, limit=180),
            )
            return ""

    def _normalize_telegram_bot_token(self, value: Any) -> str:
        token = str(value or "").strip()
        if not token:
            return ""
        if len(token) > 256:
            raise ValueError("Telegram bot token is too long.")
        if "\\" in token or "/" in token or any(char.isspace() for char in token):
            raise ValueError("Telegram bot token must not contain whitespace or slashes.")
        if any(ord(char) < 32 or ord(char) == 127 for char in token):
            raise ValueError("Telegram bot token must not contain control characters.")
        return token

    async def send_trade_notification(
        self,
        model_name: str,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        reasoning: str,
    ) -> None:
        """Notify about a trade execution."""
        emoji = "🟢" if action in ("long", "buy") else "🔴" if action in ("short", "sell") else "⚪"
        message = (
            f"{emoji} Trade Executed\n"
            f"Model: {model_name}\n"
            f"Symbol: {symbol}\n"
            f"Action: {action}\n"
            f"Quantity: {quantity:.4f}\n"
            f"Price: {price:.4f}\n"
            f"Reason: {reasoning[:200]}"
        )
        await self._send(message)

    async def send_risk_alert(
        self, event_type: str, severity: str, symbol: str, details: str
    ) -> None:
        """Notify about a risk event."""
        emoji = "🚨" if severity == "critical" else "⚠️"
        message = (
            f"{emoji} Risk Alert [{severity.upper()}]\n"
            f"Type: {event_type}\n"
            f"Symbol: {symbol}\n"
            f"Details: {details[:300]}"
        )
        await self._send(message)

    async def send_model_promotion(self, old_model: str, new_model: str, reason: str) -> None:
        """Notify when the live model changes."""
        message = (
            f"🏆 Model Promotion\n" f"From: {old_model}\n" f"To: {new_model}\n" f"Reason: {reason}"
        )
        await self._send(message)

    async def send_error(self, component: str, error: str) -> None:
        """Notify about system errors."""
        message = f"❌ System Error\nComponent: {component}\nError: {error[:500]}"
        await self._send(message)

    async def send_daily_summary(self, summary: dict[str, Any]) -> None:
        """Send end-of-day trading summary."""
        parts = ["📊 Daily Trading Summary"]
        for model_name, stats in summary.items():
            parts.append(
                f"\n{model_name}:\n"
                f"  PnL: {stats.get('pnl', 0):.2f} USD\n"
                f"  Trades: {stats.get('trades', 0)}\n"
                f"  Win Rate: {stats.get('win_rate', 0)*100:.1f}%"
            )
        await self._send("\n".join(parts))

    async def _send(self, message: str) -> None:
        """Send message to all configured channels."""
        safe_message = self._safe_message(message)
        tasks: list[tuple[str, Awaitable[None]]] = []
        if self._telegram_enabled:
            tasks.append(("telegram", self._send_telegram(safe_message)))
        if self._dingtalk_enabled:
            tasks.append(("dingtalk", self._send_dingtalk(safe_message)))
        if not tasks:
            return
        results = await asyncio.gather(
            *(task for _, task in tasks),
            return_exceptions=True,
        )
        for (channel, _), result in zip(tasks, results, strict=False):
            if isinstance(result, Exception):
                logger.warning(
                    "notification channel send failed",
                    channel=channel,
                    error=safe_error_text(result, limit=180),
                )

    def _safe_message(self, message: str) -> str:
        return bounded_redacted_text(message, limit=_MAX_NOTIFICATION_TEXT_CHARS)

    async def _send_telegram(self, message: str) -> None:
        if not self._telegram_enabled:
            return
        url = f"https://api.telegram.org/bot{self._telegram_bot_token}/sendMessage"
        await self._post_json(
            "telegram",
            url,
            {
                "chat_id": self._telegram_chat_id,
                "text": message,
                "disable_web_page_preview": True,
            },
        )

    async def _send_dingtalk(self, message: str) -> None:
        if not self._dingtalk_enabled:
            return
        await self._post_json(
            "dingtalk",
            self._dingtalk_webhook_url,
            {"msgtype": "text", "text": {"content": message}},
        )

    async def _post_json(self, channel: str, url: str, payload: dict[str, Any]) -> None:
        client = await self._get_client()
        response = await client.post(url, json=payload)
        if response.is_success:
            return
        detail = safe_response_error_text(response, limit=300)
        message = f"{channel} notification request failed with HTTP {response.status_code}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
