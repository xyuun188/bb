from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

import db.repositories.memory_repo as memory_repo_module
from db.repositories.memory_repo import MemoryRepository


class FakeScalarResult:
    def scalar_one_or_none(self) -> Any | None:
        return None

    def scalars(self) -> FakeScalarResult:
        return self

    def all(self) -> list[Any]:
        return []


class FakeSession:
    def __init__(self) -> None:
        self.added: Any | None = None
        self.flush_count = 0

    def add(self, instance: Any) -> None:
        self.added = instance

    async def flush(self) -> None:
        self.flush_count += 1

    async def execute(self, _stmt: Any) -> FakeScalarResult:
        return FakeScalarResult()


def _install_fake_sanitizer(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []

    def fake_sanitize(value: Any) -> Any:
        calls.append(value)
        if isinstance(value, str):
            return f"unified:{value}"
        if isinstance(value, dict):
            return {"unified": value}
        return value

    monkeypatch.setattr(
        memory_repo_module,
        "sanitize_runtime_text",
        fake_sanitize,
        raising=False,
    )
    return calls


@pytest.mark.asyncio
async def test_upsert_memory_uses_unified_runtime_text_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_sanitizer(monkeypatch)
    session = FakeSession()
    repo = MemoryRepository(session)  # type: ignore[arg-type]

    memory = await repo.upsert_memory(
        {
            "expert_name": "trend_expert",
            "expert_label": "Trend",
            "symbol": "BTC/USDT",
            "side": "long",
            "memory_type": "lesson",
            "market_pattern": "raw pattern",
            "lesson": "raw lesson",
            "recommended_action": "reduce_risk",
            "memory_key": "trend:BTC:long",
            "extra": {"note": "raw extra"},
        }
    )

    assert session.added is memory
    assert memory.lesson == "unified:raw lesson"
    assert memory.market_pattern == "unified:raw pattern"
    assert memory.recommended_action == "unified:reduce_risk"
    assert memory.extra == {"unified": {"note": "raw extra"}}
    assert "raw lesson" in calls
    assert session.flush_count == 1


@pytest.mark.asyncio
async def test_create_reflection_uses_unified_runtime_text_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sanitizer(monkeypatch)
    session = FakeSession()
    repo = MemoryRepository(session)  # type: ignore[arg-type]

    reflection = await repo.create_reflection(
        {
            "position_id": 0,
            "model_name": "ensemble_trader",
            "execution_mode": "paper",
            "symbol": "BTC/USDT",
            "side": "long",
            "mistake_summary": "raw mistake",
            "improvement_summary": "raw improvement",
            "expert_lessons": {"trend_expert": {"lesson": "raw expert lesson"}},
        }
    )

    assert session.added is reflection
    assert reflection is not None
    assert reflection.mistake_summary == "unified:raw mistake"
    assert reflection.improvement_summary == "unified:raw improvement"
    assert reflection.expert_lessons == {
        "unified": {"trend_expert": {"lesson": "raw expert lesson"}}
    }
    assert session.flush_count == 1


@pytest.mark.asyncio
async def test_shadow_backtest_uses_unified_runtime_text_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sanitizer(monkeypatch)
    session = FakeSession()
    repo = MemoryRepository(session)  # type: ignore[arg-type]

    row = await repo.create_shadow_backtest(
        {
            "decision_id": 7,
            "model_name": "ensemble_trader",
            "execution_mode": "paper",
            "symbol": "BTC/USDT",
            "decision_action": "hold",
            "entry_price": 100.0,
            "feature_snapshot": {"reason": "raw feature"},
            "raw_llm_response": {"reason": "raw llm"},
            "due_at": datetime(2026, 6, 23, tzinfo=UTC),
            "horizon_minutes": 10,
        }
    )

    assert session.added is row
    assert row.feature_snapshot == {"unified": {"reason": "raw feature"}}
    assert row.raw_llm_response == {"unified": {"reason": "raw llm"}}

    await repo.complete_shadow_backtest(
        row,
        actual_price=101.0,
        long_return_pct=1.0,
        short_return_pct=-1.0,
        best_action="long",
        missed_opportunity=True,
        note="raw completion note",
    )

    assert row.note == "unified:raw completion note"
    assert row.status == "completed"
    assert session.flush_count == 2
