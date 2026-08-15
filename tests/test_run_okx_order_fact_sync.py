from __future__ import annotations

import pytest

from scripts import run_okx_order_fact_sync


@pytest.mark.asyncio
async def test_isolated_okx_order_fact_runner_passes_scope_and_closes_db(
    monkeypatch,
) -> None:
    closed: list[bool] = []
    captured: dict[str, object] = {}

    class FakeService:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def sync(self) -> dict[str, object]:
            return {"status": "ok", "confirmed_count": 12}

    async def fake_close_db() -> None:
        closed.append(True)

    monkeypatch.setattr(run_okx_order_fact_sync, "close_db", fake_close_db)
    result = await run_okx_order_fact_sync.run_once(
        mode="paper",
        lookback_hours=24,
        limit=100,
        timeout_seconds=120.0,
        service_factory=FakeService,
    )

    assert result == {"status": "ok", "confirmed_count": 12}
    assert captured == {
        "mode": "paper",
        "lookback_hours": 24,
        "limit": 100,
        "timeout_seconds": 120.0,
    }
    assert closed == [True]


@pytest.mark.asyncio
async def test_isolated_okx_order_fact_runner_defers_failure_and_closes_db(
    monkeypatch,
) -> None:
    closed: list[bool] = []

    class FailingService:
        async def sync(self) -> dict[str, object]:
            raise TimeoutError("QueuePool connection timed out")

    async def fake_close_db() -> None:
        closed.append(True)

    monkeypatch.setattr(run_okx_order_fact_sync, "close_db", fake_close_db)
    result = await run_okx_order_fact_sync.run_once(
        mode="paper",
        lookback_hours=24,
        limit=100,
        timeout_seconds=120.0,
        service_factory=lambda **_kwargs: FailingService(),
    )

    assert result["status"] == "deferred"
    assert result["okx_pull_available"] is False
    assert result["deferred_stages"] == ["isolated_process"]
    assert "QueuePool connection timed out" in result["error"]
    assert closed == [True]
