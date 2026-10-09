from __future__ import annotations

import json

import pytest

from scripts import run_local_ai_tools_auto_train as runner


@pytest.mark.asyncio
async def test_run_once_uses_shared_training_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    class FakeCoordinator:
        def __init__(self) -> None:
            self.local_ai_tools = self

        async def train_local_ai_tools(self, *, force: bool = False) -> dict[str, object]:
            calls.append(force)
            return {"trained": False, "reason": "not_due"}

        async def close(self) -> None:
            return None

    async def close_db() -> None:
        return None

    monkeypatch.setattr(runner, "IndependentTrainingCoordinator", FakeCoordinator)
    monkeypatch.setattr(runner, "close_db", close_db)

    result = await runner.run_once()

    assert result == {
        "trained": False,
        "reason": "not_due",
        "training_mode": "walk_forward",
        "live_routing_enabled": False,
    }
    assert calls == [False]


@pytest.mark.asyncio
async def test_run_once_returns_governed_error_without_duplicate_training_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCoordinator:
        def __init__(self) -> None:
            self.local_ai_tools = self

        async def train_local_ai_tools(self, *, force: bool = False) -> dict[str, object]:
            raise RuntimeError("training unavailable")

        async def close(self) -> None:
            return None

    async def close_db() -> None:
        return None

    monkeypatch.setattr(runner, "IndependentTrainingCoordinator", FakeCoordinator)
    monkeypatch.setattr(runner, "close_db", close_db)

    result = await runner.run_once()

    assert result["trained"] is False
    assert result["reason"] == "error"
    assert "training unavailable" in str(result["error"])


def test_main_emits_one_structured_result_frame(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_run_once() -> dict[str, object]:
        return {"trained": False, "reason": "not_due"}

    monkeypatch.setattr(runner, "run_once", fake_run_once)

    assert runner.main() == 0
    output = capsys.readouterr().out.strip()
    assert output.startswith(runner.LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX)
    assert json.loads(
        output.removeprefix(runner.LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX)
    ) == {"trained": False, "reason": "not_due"}


@pytest.mark.parametrize("reason", ["resource_memory", "resource_error", "load_samples_error"])
def test_main_returns_failure_for_all_training_resource_errors(monkeypatch, reason) -> None:
    async def fake_run_once():
        return {"trained": False, "reason": reason}

    monkeypatch.setattr(runner, "run_once", fake_run_once)
    assert runner.main() == 2
