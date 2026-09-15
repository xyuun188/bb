from __future__ import annotations

import pytest

from scripts import run_local_ai_tools_auto_train as runner


def test_result_from_output_reads_only_structured_frame() -> None:
    output = "progress\n" + runner.LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX + '{"trained":true,"reason":"ok"}\n'

    assert runner._result_from_output(output) == {"trained": True, "reason": "ok"}


@pytest.mark.asyncio
async def test_run_once_records_okx_block_without_invoking_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, object]] = []

    class FakeStore:
        def heartbeat(self, **kwargs: object) -> None:
            events.append(("heartbeat", kwargs))

        def record_check(self, **kwargs: object) -> None:
            events.append(("record_check", kwargs))

        def start_run(self, **kwargs: object) -> None:
            raise AssertionError("blocked OKX gate must not start training")

        def finish_check(self, **kwargs: object) -> None:
            events.append(("finish_check", kwargs))

        def record_exception(self, **kwargs: object) -> None:
            events.append(("record_exception", kwargs))

        def record_timeout(self, **kwargs: object) -> None:
            events.append(("record_timeout", kwargs))

    monkeypatch.setattr(runner, "STATE_STORE", FakeStore())
    monkeypatch.setattr(
        runner,
        "okx_training_refresh_gate",
        lambda: {"allowed": False, "reason": "okx_current_state_attention"},
    )
    monkeypatch.setattr(
        runner,
        "_run_shadow_trainer",
        lambda: (_ for _ in ()).throw(AssertionError("trainer must not run")),
    )

    result = await runner.run_once()

    assert result["reason"] == "okx_training_gate_blocked"
    assert result["live_routing_enabled"] is False
    assert [name for name, _ in events] == ["heartbeat", "record_check", "finish_check"]
    finish_payload = next(payload for name, payload in events if name == "finish_check")
    assert finish_payload["result"]["training_gate"]["allowed"] is False
