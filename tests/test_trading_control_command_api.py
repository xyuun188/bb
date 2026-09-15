from __future__ import annotations

from typing import Any

import httpx
import pytest

from config.settings import settings
from services.trading_control_command_queue import TradingControlCommandQueue
from web_dashboard.api import control
from web_dashboard.api import dashboard as dashboard_api
from web_dashboard.app import create_app


class _QueuedCloseCommands:
    async def enqueue_close_all(self, *, mode: str | None, reason: str | None) -> dict[str, Any]:
        return {
            "command_id": "close-all-1",
            "command_type": "close_all_positions",
            "status": "queued",
            "mode": mode,
            "reason": reason,
        }

    async def enqueue_close_position(
        self,
        *,
        position_id: int,
        mode: str | None,
        reason: str | None,
    ) -> dict[str, Any]:
        return {
            "command_id": "close-one-1",
            "command_type": "close_position",
            "status": "queued",
            "mode": mode,
            "position_id": position_id,
            "reason": reason,
        }

    async def get(self, command_id: str) -> dict[str, Any] | None:
        if command_id != "close-all-1":
            return None
        return {
            "command_id": command_id,
            "command_type": "close_all_positions",
            "status": "completed",
            "mode": "paper",
            "result": {"closed": 3, "failed": 0},
        }


@pytest.mark.asyncio
async def test_split_dashboard_enqueues_close_all_and_exposes_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "unit-dashboard-write-token"
    monkeypatch.setattr(settings, "dashboard_admin_api_key", token)
    monkeypatch.setattr(dashboard_api, "_trading_service", None)
    monkeypatch.setattr(control, "trading_control_commands", _QueuedCloseCommands())
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        queued = await client.post(
            "/api/positions/close-all",
            headers={"Authorization": f"Bearer {token}"},
            json={"mode": "paper", "reason": "test close all"},
        )
        result = await client.get(
            "/api/positions/close-commands/close-all-1",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert queued.status_code == 200
    assert queued.json()["status"] == "queued"
    assert queued.json()["command_id"] == "close-all-1"
    assert result.status_code == 200
    assert result.json()["result"] == {"closed": 3, "failed": 0}


@pytest.mark.asyncio
async def test_command_exception_requires_reconfirmation_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = TradingControlCommandQueue(worker_id="unit-worker")
    command = {
        "command_id": "close-all-unknown",
        "command_type": "close_all_positions",
        "mode": "paper",
        "reason": "test interruption",
    }
    marked: list[str] = []

    async def claim_once() -> dict[str, Any] | None:
        return command

    async def require_reconfirmation(command_id: str, exc: BaseException) -> None:
        assert isinstance(exc, RuntimeError)
        marked.append(command_id)

    async def must_not_complete(command_id: str, result: dict[str, Any]) -> None:
        raise AssertionError("interrupted command must never be completed or replayed")

    async def interrupted_close_all(**_: Any) -> dict[str, Any]:
        raise RuntimeError("connection ended after exchange submit")

    monkeypatch.setattr(queue, "_claim_next", claim_once)
    monkeypatch.setattr(queue, "_require_reconfirmation", require_reconfirmation)
    monkeypatch.setattr(queue, "_complete", must_not_complete)

    processed = await queue.process_once(
        close_all=interrupted_close_all,
        close_position=interrupted_close_all,
    )

    assert processed is True
    assert marked == ["close-all-unknown"]
