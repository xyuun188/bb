from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_local_ml_auto_train


def test_training_process_installs_memory_guard_before_project_imports() -> None:
    source = run_local_ml_auto_train.__file__
    assert source is not None
    text = Path(source).read_text(encoding="utf-8")
    assert "LOCAL_ML_TRAINING_MEMORY_LIMIT_BYTES" in text
    assert "resource.RLIMIT_AS" in text


@pytest.mark.asyncio
async def test_isolated_local_ml_runner_returns_structured_result(monkeypatch) -> None:
    closed: list[bool] = []

    class FakeService:
        async def maybe_auto_train(self, *, force: bool) -> dict[str, object]:
            return {"trained": True, "reason": "trained_shadow_activated", "force": force}

    async def fake_close_db() -> None:
        closed.append(True)

    monkeypatch.setattr(run_local_ml_auto_train, "close_db", fake_close_db)
    result = await run_local_ml_auto_train.run_once(
        force=True,
        service_factory=FakeService,
    )

    assert result == {
        "trained": True,
        "reason": "trained_shadow_activated",
        "force": True,
    }
    assert closed == [True]


@pytest.mark.asyncio
async def test_isolated_local_ml_runner_converts_pool_failure_for_retry(monkeypatch) -> None:
    class FailingService:
        async def maybe_auto_train(self, *, force: bool) -> dict[str, object]:
            raise TimeoutError("QueuePool connection timed out")

    async def fake_close_db() -> None:
        return None

    monkeypatch.setattr(run_local_ml_auto_train, "close_db", fake_close_db)
    result = await run_local_ml_auto_train.run_once(service_factory=FailingService)

    assert result["trained"] is False
    assert result["reason"] == "error"
    assert "QueuePool connection timed out" in result["error"]
