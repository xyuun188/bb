from __future__ import annotations

from typing import Any

import pytest

import services.strategy_learning as strategy_learning_module
from services.strategy_learning import StrategyLearningService


class FakeSession:
    def __init__(self) -> None:
        self.added: Any | None = None
        self.flush_count = 0

    def add(self, instance: Any) -> None:
        self.added = instance

    async def flush(self) -> None:
        self.flush_count += 1
        if self.added is not None:
            self.added.id = 456


class FakeSessionContext:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> FakeSession:
        return self.session

    async def __aexit__(self, *_args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_record_event_uses_unified_runtime_text_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    def fake_sanitize(value: Any) -> Any:
        calls.append(value)
        if isinstance(value, str):
            return f"unified:{value}"
        if isinstance(value, dict):
            return {"unified": value}
        return value

    session = FakeSession()
    monkeypatch.setattr(
        strategy_learning_module,
        "sanitize_runtime_text",
        fake_sanitize,
        raising=False,
    )
    monkeypatch.setattr(
        strategy_learning_module,
        "get_session_ctx",
        lambda: FakeSessionContext(session),
    )

    service = StrategyLearningService()
    event_id = await service.record_event(
        mode="paper",
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action="long",
        event_type="entry_blocked",
        event_status="blocked",
        reason="raw event reason",
        strategy_context={
            "scheduler_reason": "raw scheduler reason",
            "strategy_learning": {
                "runtime": {"side_weights": {"long": "raw side weight"}},
                "reason": "raw learning reason",
            },
            "market_regime": {"note": "raw market"},
        },
        raw_response={"integrity": "raw expert"},
        market_state={"note": "raw explicit market"},
        attribution={"note": "raw attribution"},
    )

    assert event_id == 456
    assert session.added is not None
    assert session.added.reason == "unified:raw event reason"
    assert session.added.scheduler_reason == "unified:raw scheduler reason"
    assert session.added.strategy_snapshot == {
        "unified": session.added.strategy_snapshot["unified"]
    }
    assert session.added.market_state == {"unified": {"note": "raw explicit market"}}
    assert session.added.side_weights is None
    assert session.added.expert_integrity == {"unified": session.added.expert_integrity["unified"]}
    assert session.added.attribution == {"unified": {"note": "raw attribution"}}
    assert "raw event reason" in calls
    assert session.flush_count == 1
