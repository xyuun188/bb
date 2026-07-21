from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import db.repositories.memory_repo as memory_repo_module
from db.repositories.memory_repo import MemoryRepository
from models.learning import ExpertMemory
from services.authoritative_trade_outcome import (
    AUTHORITATIVE_TRADE_OUTCOME_AUTHORITY,
    AUTHORITATIVE_TRADE_OUTCOME_VERSION,
)


def _outcome_extra(*, position_id: int, outcome_id: str, pnl: float, return_pct: float) -> dict:
    return {
        "source": "authoritative_trade_outcome",
        "authority_level": AUTHORITATIVE_TRADE_OUTCOME_AUTHORITY,
        "outcome_version": AUTHORITATIVE_TRADE_OUTCOME_VERSION,
        "outcome_id": outcome_id,
        "production_evidence_eligible": True,
        "realized_pnl": pnl,
        "net_return_after_cost_pct": return_pct,
        "source_position_id": position_id,
    }


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


def test_memory_outcomes_merge_conflicting_evidence_by_fee_after_return() -> None:
    positive = memory_repo_module._merge_memory_outcomes(
        None,
        _outcome_extra(position_id=1, outcome_id="ato:1", pnl=1.0, return_pct=4.0),
    )
    merged = memory_repo_module._merge_memory_outcomes(
        positive,
        _outcome_extra(position_id=2, outcome_id="ato:2", pnl=-5.0, return_pct=-10.0),
    )

    outcome = merged["outcome_aggregation"]
    assert outcome["count"] == 2
    assert outcome["conflict"] is True
    assert outcome["total_realized_net_pnl_usdt"] == pytest.approx(-4.0)
    assert outcome["avg_net_return_pct"] == pytest.approx(-3.0)
    assert outcome["return_lcb_pct"] == pytest.approx(-10.0)
    assert outcome["worst_net_return_pct"] == pytest.approx(-10.0)
    assert outcome["squared_net_return_sum_pct2"] == pytest.approx(116.0)
    assert outcome["profit_factor"] == pytest.approx(0.2)
    assert outcome["return_unit"] == "percentage_points"
    assert outcome["source_outcome_ids"] == ["ato:1", "ato:2"]


def test_memory_outcome_aggregation_is_idempotent_per_position() -> None:
    first = memory_repo_module._merge_memory_outcomes(
        None,
        _outcome_extra(position_id=2, outcome_id="ato:2", pnl=-5.0, return_pct=-10.0),
    )
    repeated = memory_repo_module._merge_memory_outcomes(
        first,
        _outcome_extra(position_id=2, outcome_id="ato:2", pnl=-5.0, return_pct=-10.0),
    )

    assert repeated["outcome_aggregation"]["count"] == 1
    assert repeated["outcome_aggregation"]["total_realized_net_pnl_usdt"] == -5.0
    assert repeated["outcome_aggregation"]["source_position_ids"] == [2]
    assert repeated["outcome_aggregation"]["source_outcome_ids"] == ["ato:2"]


def test_memory_profit_factor_is_undefined_without_any_realized_loss() -> None:
    merged = memory_repo_module._merge_memory_outcomes(
        None,
        _outcome_extra(position_id=3, outcome_id="ato:3", pnl=5.0, return_pct=2.0),
    )

    assert merged["outcome_aggregation"]["profit_factor"] is None
    assert merged["outcome_aggregation"]["profit_factor_defined"] is False


def test_reflection_source_is_sanitized_to_database_contract() -> None:
    normalized = memory_repo_module._normalize_reflection_payload(
        {"source": "authoritative-settlement-backfill-source-name-that-is-too-long"}
    )

    assert len(normalized["source"]) == 40


def test_memory_outcomes_do_not_fallback_to_legacy_pnl_ratio() -> None:
    merged = memory_repo_module._merge_memory_outcomes(
        None,
        {"realized_pnl": -5.0, "pnl_pct": -0.10, "source_position_id": 2},
    )

    assert "outcome_aggregation" not in merged


@pytest.mark.asyncio
async def test_bulk_memory_lookup_limits_each_expert_inside_ranked_query() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(ExpertMemory.__table__.create)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as session:
            for expert_name in ("trend_expert", "risk_expert"):
                for index in range(50):
                    session.add(
                        ExpertMemory(
                            expert_name=expert_name,
                            expert_label=expert_name,
                            symbol="BTC/USDT",
                            side="long",
                            memory_type="shadow_missed_opportunity",
                            market_pattern=f"pattern-{index}",
                            lesson=f"lesson-{index}",
                            recommended_action="observation_only",
                            evidence_count=1,
                            memory_key=f"{expert_name}:{index}",
                            is_active=True,
                            extra={"authority_rank": 100 if index == 0 else 0},
                        )
                    )
            await session.commit()

            grouped = await MemoryRepository(session).get_relevant_memories_for_experts(
                ["trend_expert", "risk_expert"],
                "BTC/USDT",
                per_expert_limit=3,
            )

        assert {name: len(rows) for name, rows in grouped.items()} == {
            "trend_expert": 3,
            "risk_expert": 3,
        }
        assert all(rows[0].extra["authority_rank"] == 100 for rows in grouped.values())
    finally:
        await engine.dispose()


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
    assert "confidence_adjustment" not in memory.__table__.columns
    assert "position_size_multiplier" not in memory.__table__.columns
    assert "raw lesson" in calls
    assert session.flush_count == 1


@pytest.mark.asyncio
async def test_upsert_memory_rejects_removed_policy_fields() -> None:
    session = FakeSession()
    repo = MemoryRepository(session)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unsupported expert memory fields") as exc_info:
        await repo.upsert_memory(
            {
                "expert_name": "risk_expert",
                "memory_key": "risk:BTC:long",
                "confidence_adjustment": 0.9,
                "position_size_multiplier": 2.5,
            }
        )

    assert "confidence_adjustment" in str(exc_info.value)
    assert "position_size_multiplier" in str(exc_info.value)


@pytest.mark.asyncio
async def test_upsert_memory_updates_existing_observation_without_policy_fields() -> None:
    existing = SimpleNamespace(
        evidence_count=1,
        success_count=0,
        failure_count=0,
        confidence_score=0.5,
        lesson="old lesson",
        market_pattern="old pattern",
        recommended_action="old action",
        source_position_id=None,
        extra={},
        is_active=True,
        updated_at=None,
    )

    class ExistingScalarResult(FakeScalarResult):
        def scalar_one_or_none(self) -> Any:
            return existing

    class ExistingSession(FakeSession):
        async def execute(self, _stmt: Any) -> ExistingScalarResult:
            return ExistingScalarResult()

    repo = MemoryRepository(ExistingSession())  # type: ignore[arg-type]
    updated = await repo.upsert_memory(
        {
            "expert_name": "risk_expert",
            "memory_key": "risk:BTC:long",
            "lesson": "new observation",
            "market_pattern": "new pattern",
        }
    )

    assert updated is existing
    assert existing.lesson == "new observation"
    assert existing.market_pattern == "new pattern"
    assert existing.evidence_count == 2


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
