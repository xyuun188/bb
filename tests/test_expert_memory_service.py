from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

import services.expert_memory_service as expert_memory_module
from services.expert_memory_service import (
    ExpertMemoryService,
    _reflection_lifecycle_key,
    _reflection_position_rank,
    reflection_summary,
)


def _u(escaped: str) -> str:
    return escaped.encode("ascii").decode("unicode_escape")


MOJIBAKE_MARKERS = (
    _u("\\u934b"),
    _u("\\u951b"),
    _u("\\u95b3"),
    _u("\\u9429"),
    _u("\\u6d5c\\u5fd4\\u5d2f"),
    _u("\\u93c9\\u51ae\\u5678"),
)


def _assert_clean_chinese(text: str) -> None:
    assert all(marker not in text for marker in MOJIBAKE_MARKERS)
    assert any(token in text for token in ("做多", "做空", "亏损", "盈利", "权重", "持仓"))


def test_trade_reflection_templates_are_clean_chinese() -> None:
    pos = SimpleNamespace(
        symbol="BTC/USDT",
        side="long",
        leverage=3.0,
        created_at=datetime(2026, 6, 8, 1, 0, tzinfo=UTC),
        closed_at=datetime(2026, 6, 8, 1, 4, tzinfo=UTC),
    )

    mistake, improvement = reflection_summary(pos, "loss", -0.012, 4.0)

    _assert_clean_chinese(mistake)
    assert "不进入专家记忆、训练或晋升" in improvement


def test_authoritative_reflection_backfill_deduplicates_exchange_lifecycle() -> None:
    base = {
        "execution_mode": "paper",
        "symbol": "SYRUP/USDT",
        "side": "short",
        "entry_exchange_order_id": "entry-1",
        "close_exchange_order_id": "close-1",
        "settlement_status": "reconciled",
    }
    repaired = SimpleNamespace(
        id=4467,
        settlement_source="system_execution",
        **base,
    )
    authoritative = SimpleNamespace(
        id=4392,
        settlement_source="okx_position_history",
        **base,
    )

    assert _reflection_lifecycle_key(repaired) == _reflection_lifecycle_key(authoritative)
    assert _reflection_position_rank(authoritative) > _reflection_position_rank(repaired)


@pytest.mark.asyncio
async def test_expert_memory_context_loads_all_experts_in_one_query(monkeypatch) -> None:
    bulk_calls: list[tuple[list[str], str]] = []
    used_ids: list[int] = []
    memory = SimpleNamespace(
        id=11,
        expert_name="trend_expert",
        expert_label="trend",
        symbol="BTC/USDT",
        side="long",
        memory_type="authoritative_trade_outcome",
        market_pattern="pattern",
        lesson="lesson",
        recommended_action="observation_only",
        evidence_count=1,
        success_count=1,
        failure_count=0,
        confidence_score=0.5,
        extra={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    class FakeMemoryRepository:
        def __init__(self, _session: Any) -> None:
            pass

        async def get_relevant_memories_for_experts(
            self,
            expert_names: list[str],
            symbol: str,
        ) -> dict[str, list[Any]]:
            bulk_calls.append((expert_names, symbol))
            return {"trend_expert": [memory]}

        async def mark_memories_used(self, memory_ids: list[int]) -> None:
            used_ids.extend(memory_ids)

    @asynccontextmanager
    async def session_factory():
        yield object()

    monkeypatch.setattr(expert_memory_module, "MemoryRepository", FakeMemoryRepository)
    service = ExpertMemoryService(
        session_factory=session_factory,
        memory_enabled_provider=lambda: True,
        model_slots=[
            {"name": "trend_expert", "label": "trend"},
            {"name": "risk_expert", "label": "risk"},
        ],
    )

    context = await service.context("BTC/USDT")

    assert bulk_calls == [(["trend_expert", "risk_expert"], "BTC/USDT")]
    assert used_ids == [11]
    assert context["expert_memories"]["trend_expert"][0]["id"] == 11


@pytest.mark.asyncio
async def test_local_close_records_reflection_without_expert_memory(monkeypatch) -> None:
    created_reflections: list[dict[str, Any]] = []

    class FakeMemoryRepository:
        def __init__(self, _session: Any) -> None:
            pass

        async def create_reflection(self, data: dict[str, Any]) -> SimpleNamespace:
            created_reflections.append(data)
            return SimpleNamespace(id=321)

    monkeypatch.setattr(expert_memory_module, "MemoryRepository", FakeMemoryRepository)
    service = ExpertMemoryService(
        memory_enabled_provider=lambda: True,
        model_slots=[{"name": "trend_expert", "label": "趋势专家", "weight": 1.0}],
    )
    pos = SimpleNamespace(
        id=7,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
        entry_price=100.0,
        current_price=98.0,
        quantity=2.0,
        realized_pnl=-4.0,
        leverage=3.0,
        created_at=datetime.now(UTC) - timedelta(minutes=8),
        closed_at=datetime.now(UTC),
    )

    processed = await service.record_trade_reflection_in_session(
        object(),
        pos,
        exit_price=98.0,
        entry_fee=0.1,
        close_fee=0.1,
        source="unit_test",
    )

    assert processed is True
    assert len(created_reflections) == 1
    reflection = created_reflections[0]
    assert reflection["outcome"] == "loss"
    assert reflection["source"] == "unit_test"
    assert reflection["closed_at"] == pos.closed_at
    assert reflection["expert_lessons"] == {}
    _assert_clean_chinese(reflection["mistake_summary"])
    assert "不进入专家记忆、训练或晋升" in reflection["improvement_summary"]


@pytest.mark.asyncio
async def test_existing_local_reflection_does_not_create_expert_memory(monkeypatch) -> None:
    class FakeMemoryRepository:
        def __init__(self, _session: Any) -> None:
            pass

        async def create_reflection(self, _data: dict[str, Any]) -> None:
            return None

        async def get_reflection_by_position_id(self, position_id: int) -> SimpleNamespace:
            assert position_id == 7
            return SimpleNamespace(id=654)

    monkeypatch.setattr(expert_memory_module, "MemoryRepository", FakeMemoryRepository)
    service = ExpertMemoryService(
        memory_enabled_provider=lambda: True,
        model_slots=[{"name": "trend_expert", "label": "趋势专家", "weight": 1.0}],
    )
    pos = SimpleNamespace(
        id=7,
        model_name="okx_authoritative_sync",
        execution_mode="paper",
        symbol="PROS/USDT",
        side="short",
        entry_price=2.0,
        current_price=2.2,
        quantity=10.0,
        realized_pnl=-2.0,
        funding_fee=-0.1,
        leverage=3.0,
        settlement_status="okx_position_history",
        settlement_source="okx_position_history",
        created_at=datetime.now(UTC) - timedelta(minutes=8),
        closed_at=datetime.now(UTC),
    )

    processed = await service.record_trade_reflection_in_session(
        object(),
        pos,
        exit_price=2.2,
        entry_fee=0.1,
        close_fee=0.1,
        source="authoritative_settlement_backfill",
    )

    assert processed is True


@pytest.mark.asyncio
async def test_authoritative_outcome_backfill_runs_when_prompt_memory_is_disabled(
    monkeypatch,
) -> None:
    processed: list[str] = []
    loader_calls: list[dict] = []

    class SessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return False

    async def load_outcomes(**kwargs):
        loader_calls.append(kwargs)
        return [
            {
                "lifecycle_key": "paper|ICP|1",
                "settlement_fact_trusted": True,
                "outcome_complete": True,
            },
            {
                "lifecycle_key": "paper|ICP|2",
                "settlement_fact_trusted": True,
                "outcome_complete": False,
            },
        ]

    service = ExpertMemoryService(
        memory_enabled_provider=lambda: False,
        session_factory=SessionContext,
        authoritative_outcome_loader=load_outcomes,
    )

    async def record(_session, outcome):
        processed.append(outcome["lifecycle_key"])
        return True

    monkeypatch.setattr(service, "_record_authoritative_outcome_in_session", record)
    monkeypatch.setattr(
        "services.expert_memory_service.load_training_epoch_start",
        lambda: datetime(2026, 7, 24, tzinfo=UTC),
    )

    report = await service.backfill_trade_reflections("paper")

    assert report["status"] == "completed"
    assert report["scanned"] == 2
    assert report["eligible"] == 1
    assert report["processed"] == 1
    assert processed == ["paper|ICP|1"]
    assert loader_calls[0]["mode"] == "paper"
    assert loader_calls[0]["since"] == datetime(2026, 7, 24, tzinfo=UTC)
