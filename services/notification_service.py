"""
Notification service — sends alerts via Telegram and/or DingTalk.
Notifies on: trade executions, risk events, model promotions, errors.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)


class NotificationService:
    """Sends notifications to configured channels."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._telegram_enabled = bool(settings.telegram_bot_token and settings.telegram_chat_id)
        self._dingtalk_enabled = bool(settings.dingtalk_webhook_url)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        return self._client

    async def send_trade_notification(
        self, model_name: str, symbol: str, action: str, quantity: float, price: float, reasoning: str
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
            f"🏆 Model Promotion\n"
            f"From: {old_model}\n"
            f"To: {new_model}\n"
            f"Reason: {reason}"
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
        results = await asyncio.gather(
            self._send_telegram(message),
            self._send_dingtalk(message),
            return_exceptions=True,
        )

    async def _send_telegram(self, message: str) -> None:
        if not self._telegram_enabled:
            return
        try:
            client = await self._get_client()
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            await client.post(url, json={
                "chat_id": settings.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            })
        except Exception as e:
            logger.debug("telegram send failed", error=str(e))

    async def _send_dingtalk(self, message: str) -> None:
        if not self._dingtalk_enabled:
            return
        try:
            client = await self._get_client()
            await client.post(
                settings.dingtalk_webhook_url,
                json={"msgtype": "text", "text": {"content": message}},
            )
        except Exception as e:
            logger.debug("dingtalk send failed", error=str(e))

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

