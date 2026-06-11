from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import services.notification_service as notification_module
from services.notification_service import NotificationService


def _config(**overrides: Any) -> SimpleNamespace:
    values = {
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "dingtalk_webhook_url": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_notification_service_sends_redacted_payloads_without_real_network() -> None:
    requests: list[httpx.Request] = []
    telegram_token = "123456:" + ("A" * 28)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True}, request=request)

    service = NotificationService(
        _config(
            telegram_bot_token=telegram_token,
            telegram_chat_id="chat-1",
            dingtalk_webhook_url=(
                "https://oapi.dingtalk.com/robot/send?access_token=dingtalk-secret-value"
            ),
        ),
        transport=httpx.MockTransport(handler),
    )

    await service.send_error("executor", "password=plain-secret-value")
    await service.close()

    assert len(requests) == 2
    combined_body = b"\n".join(request.content for request in requests).decode()
    assert "plain-secret-value" not in combined_body
    assert "password=***" in combined_body
    assert "executor" in combined_body


@pytest.mark.asyncio
async def test_notification_service_failure_logs_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[dict[str, Any]] = []
    telegram_token = "123456:" + ("B" * 28)
    dingtalk_token = "dingtalk-secret-value"

    class FakeLogger:
        def warning(self, _message: str, **kwargs: Any) -> None:
            warnings.append(kwargs)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            text=(
                "failed url=https://api.telegram.org/bot"
                f"{telegram_token}/sendMessage access_token={dingtalk_token}"
            ),
            request=request,
        )

    monkeypatch.setattr(notification_module, "logger", FakeLogger())
    service = NotificationService(
        _config(
            telegram_bot_token=telegram_token,
            telegram_chat_id="chat-1",
            dingtalk_webhook_url=(
                f"https://oapi.dingtalk.com/robot/send?access_token={dingtalk_token}"
            ),
        ),
        transport=httpx.MockTransport(handler),
    )

    await service.send_error("executor", "token=message-secret-value")
    await service.close()

    rendered = str(warnings)
    assert len(warnings) == 2
    assert telegram_token not in rendered
    assert dingtalk_token not in rendered
    assert "message-secret-value" not in rendered
    assert "api.telegram.org/bot***" in rendered
    assert "access_token=***" in rendered


@pytest.mark.asyncio
async def test_notification_service_disables_unsafe_webhook_url() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request)

    service = NotificationService(
        _config(
            dingtalk_webhook_url="http://oapi.dingtalk.com/robot/send?access_token=secret",
        ),
        transport=httpx.MockTransport(handler),
    )

    await service.send_error("executor", "network down")
    await service.close()

    assert requests == []
