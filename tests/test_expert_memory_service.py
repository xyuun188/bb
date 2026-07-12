from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

import services.expert_memory_service as expert_memory_module
from services.expert_memory_service import (
    ExpertMemoryService,
    _reflection_lifecycle_key,
    _reflection_position_rank,
    build_expert_lessons,
    dynamic_expert_weights_from_memories,
    reflection_pattern,
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


def test_dynamic_expert_weights_use_chinese_reasons() -> None:
    weights = dynamic_expert_weights_from_memories(
        {},
        [{"name": "trend_expert", "weight": 1.2}],
    )

    reason = weights["trend_expert"]["reason"]
    assert reason == "暂无足够历史样本，使用基础权重。"
    _assert_clean_chinese(reason)


def test_dynamic_expert_weights_reduce_losing_memory() -> None:
    weights = dynamic_expert_weights_from_memories(
        {
            "trend_expert": [
                {
                    "confidence_score": 0.9,
                    "evidence_count": 5,
                    "success_count": 0,
                    "failure_count": 4,
                    "confidence_adjustment": -0.12,
                }
            ]
        },
        [{"name": "trend_expert", "weight": 1.0}],
    )

    trend = weights["trend_expert"]
    assert trend["multiplier"] <= 0.9
    assert "权重降到" in trend["reason"]
    _assert_clean_chinese(trend["reason"])


def test_trade_reflection_templates_are_clean_chinese() -> None:
    pos = SimpleNamespace(
        symbol="BTC/USDT",
        side="long",
        leverage=3.0,
        created_at=datetime(2026, 6, 8, 1, 0, tzinfo=UTC),
        closed_at=datetime(2026, 6, 8, 1, 4, tzinfo=UTC),
    )

    pattern = reflection_pattern(pos, pnl_pct=-0.012, hold_minutes=4.0)
    mistake, improvement = reflection_summary(pos, "loss", -0.012, 4.0)
    lessons = build_expert_lessons(
        pos=pos,
        outcome="loss",
        pnl_pct=-0.012,
        hold_minutes=4.0,
        pattern=pattern,
        model_slots=[{"name": "trend_expert", "label": "趋势专家"}],
    )

    _assert_clean_chinese(pattern)
    _assert_clean_chinese(mistake)
    _assert_clean_chinese(improvement)
    assert lessons["trend_expert"]["expert_label"] == "趋势专家"
    for lesson in lessons.values():
        _assert_clean_chinese(lesson["lesson"])
        _assert_clean_chinese(lesson["market_pattern"])


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
        settlement_source="okx_order_fact_sync",
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
async def test_expert_memory_service_records_reflection_and_memories(monkeypatch) -> None:
    created_reflections: list[dict[str, Any]] = []
    upserted_memories: list[dict[str, Any]] = []

    class FakeMemoryRepository:
        def __init__(self, _session: Any) -> None:
            pass

        async def create_reflection(self, data: dict[str, Any]) -> SimpleNamespace:
            created_reflections.append(data)
            return SimpleNamespace(id=321)

        async def upsert_memory(self, data: dict[str, Any]) -> None:
            upserted_memories.append(data)

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

    await service.record_trade_reflection_in_session(
        object(),
        pos,
        exit_price=98.0,
        entry_fee=0.1,
        close_fee=0.1,
        gross_pnl=-3.8,
        source="unit_test",
        decision=None,
    )

    assert len(created_reflections) == 1
    assert len(upserted_memories) == 5
    reflection = created_reflections[0]
    assert reflection["outcome"] == "loss"
    assert reflection["source"] == "unit_test"
    assert reflection["closed_at"] == pos.closed_at
    _assert_clean_chinese(reflection["mistake_summary"])
    _assert_clean_chinese(reflection["improvement_summary"])
    assert all(memory["extra"]["reflection_id"] == 321 for memory in upserted_memories)
    assert all(
        memory["extra"]["net_return_after_cost_pct"] == pytest.approx(-2.0)
        for memory in upserted_memories
    )
    assert all(memory["extra"]["source_position_id"] == 7 for memory in upserted_memories)


@pytest.mark.asyncio
async def test_existing_reflection_still_refreshes_return_memories(monkeypatch) -> None:
    upserted_memories: list[dict[str, Any]] = []

    class FakeMemoryRepository:
        def __init__(self, _session: Any) -> None:
            pass

        async def create_reflection(self, _data: dict[str, Any]) -> None:
            return None

        async def get_reflection_by_position_id(self, position_id: int) -> SimpleNamespace:
            assert position_id == 7
            return SimpleNamespace(id=654)

        async def upsert_memory(self, data: dict[str, Any]) -> None:
            upserted_memories.append(data)

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
        gross_pnl=-1.7,
        source="authoritative_settlement_backfill",
    )

    assert processed is True
    assert len(upserted_memories) == 5
    assert all(row["extra"]["reflection_id"] == 654 for row in upserted_memories)
    assert all(row["extra"]["net_return_after_cost_pct"] == -10.0 for row in upserted_memories)
