from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scripts import run_specialist_shadow_evaluation as runner


def test_evaluation_since_uses_requested_window(monkeypatch: pytest.MonkeyPatch) -> None:
    epoch_start = datetime(2026, 1, 1, tzinfo=UTC)
    now = datetime(2026, 8, 25, 12, tzinfo=UTC)
    monkeypatch.setattr(runner, "load_training_epoch_start", lambda: epoch_start)

    assert runner._evaluation_since(hours=168, now=now) == now - timedelta(days=7)


def test_evaluation_since_caps_excessive_window(monkeypatch: pytest.MonkeyPatch) -> None:
    epoch_start = datetime(2025, 1, 1, tzinfo=UTC)
    now = datetime(2026, 8, 25, 12, tzinfo=UTC)
    monkeypatch.setattr(runner, "load_training_epoch_start", lambda: epoch_start)

    assert runner._evaluation_since(hours=24 * 365, now=now) == now - timedelta(days=90)


@pytest.mark.asyncio
async def test_evaluation_loader_uses_compact_uncached_decision_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    epoch_start = datetime(2026, 1, 1, tzinfo=UTC)
    now = datetime(2026, 8, 25, 12, tzinfo=UTC)

    async def load_outcomes(**kwargs: object) -> list[dict[str, object]]:
        captured.update(kwargs)
        return [{"outcome_id": "ato:test"}]

    monkeypatch.setattr(runner, "load_training_epoch_start", lambda: epoch_start)
    monkeypatch.setattr(runner, "load_authoritative_trade_outcomes", load_outcomes)

    result = await runner._load_evaluation_trade_samples(hours=168, now=now)

    assert result == [{"outcome_id": "ato:test"}]
    assert captured == {
        "since": now - timedelta(days=7),
        "compact": True,
        "include_decision_evidence": True,
    }
