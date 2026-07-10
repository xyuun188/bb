from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from services.model_contribution_performance import ModelContributionPerformanceService
from services.trading_service import TradingService


class _FakeRowsResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> _FakeRowsResult:
        return self

    def all(self) -> list[object]:
        return list(self._rows)


class _FakeSession:
    def __init__(self, rows_by_query: list[list[object]]) -> None:
        self._rows_by_query = list(rows_by_query)
        self.statements: list[object] = []

    async def execute(self, statement: object) -> _FakeRowsResult:
        self.statements.append(statement)
        return _FakeRowsResult(self._rows_by_query.pop(0))


def _decision(decision_id: int, raw: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=decision_id,
        action="long",
        raw_llm_response=raw,
    )


def _position(index: int, pnl: float, *, created_at: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        symbol="BTC/USDT",
        side="long",
        realized_pnl=pnl,
        created_at=created_at + timedelta(seconds=index),
    )


def _order(index: int, decision_id: int, *, created_at: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        symbol="BTC/USDT",
        decision_id=decision_id,
        filled_at=created_at + timedelta(seconds=index + 10),
        created_at=created_at + timedelta(seconds=index + 10),
    )


def test_model_contribution_build_stats_tracks_realized_source_pnl() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    raw = {
        "opportunity_score": {
            "ml_aligned": True,
            "local_profit_aligned": True,
            "timeseries_aligned": True,
            "expert_aligned": True,
            "evidence_score": {
                "components": [
                    {"source": "sentiment", "status": "aligned"},
                    {"source": "shadow_memory", "status": "aligned"},
                ]
            },
        }
    }
    positions = [_position(i, 3.0, created_at=now) for i in range(5)]
    orders = [_order(i, i + 1, created_at=now) for i in range(5)]
    decisions = {i + 1: _decision(i + 1, raw) for i in range(5)}

    stats = ModelContributionPerformanceService().build_stats(positions, orders, decisions)

    for source in (
        "ml_profit_model",
        "server_profit_model",
        "timeseries_model",
        "sentiment_model",
        "shadow_memory",
        "expert_alignment",
    ):
        bucket = stats[source]
        assert bucket["count"] == 5
        assert bucket["pnl"] == pytest.approx(15.0)
        assert bucket["state"] == "promote"
        assert bucket["score_multiplier"] > 1.0
    assert stats["ai_only_without_quant"]["count"] == 0


def test_model_contribution_build_stats_prefers_profit_first_plan_sources() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    raw = {
        "opportunity_score": {"ml_aligned": False, "local_profit_aligned": False},
        "profit_first_trade_plan": {
            "model_sources": ["decision_llm", "server_profit", "high_risk_review"],
            "model_contributions": [
                {"source": "decision_llm", "valid": True, "field_path": "decision.model_name"},
                {
                    "source": "server_profit",
                    "valid": True,
                    "field_path": "opportunity_score.expected_net_return_pct",
                },
                {"source": "high_risk_review", "valid": True, "field_path": "review.approved"},
            ],
        },
    }
    position = _position(1, 4.0, created_at=now)
    position.entry_exchange_order_id = "entry-ok"
    position.close_exchange_order_id = "close-ok"

    stats = ModelContributionPerformanceService().build_stats(
        [position],
        [_order(1, 1, created_at=now)],
        {1: _decision(1, raw)},
    )

    assert stats["decision_llm"]["count"] == 1
    assert stats["server_profit_model"]["count"] == 1
    assert stats["high_risk_review"]["count"] == 1
    assert stats["ml_profit_model"]["count"] == 0


def test_model_contribution_build_stats_excludes_untrusted_trade_facts() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    raw = {"opportunity_score": {"ml_aligned": True}}
    trusted = _position(1, 3.0, created_at=now)
    trusted.entry_exchange_order_id = "entry-ok"
    trusted.close_exchange_order_id = "close-ok"
    dirty = _position(2, 50.0, created_at=now)
    dirty.entry_exchange_order_id = "entry-dirty"
    dirty.close_exchange_order_id = ""
    orders = [_order(1, 1, created_at=now), _order(2, 2, created_at=now)]
    decisions = {1: _decision(1, raw), 2: _decision(2, raw)}

    stats = ModelContributionPerformanceService().build_stats(
        [trusted, dirty],
        orders,
        decisions,
    )

    assert stats["ml_profit_model"]["count"] == 1
    assert stats["ml_profit_model"]["pnl"] == pytest.approx(3.0)


def test_model_contribution_lineage_diagnoses_missing_order_decision_ids() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    position = _position(1, 3.0, created_at=now)
    position.entry_exchange_order_id = "entry-ok"
    position.close_exchange_order_id = "close-ok"
    order = _order(1, 0, created_at=now)
    order.decision_id = None

    diagnostics = ModelContributionPerformanceService().build_lineage_diagnostics(
        [position],
        [order],
        {},
    )

    assert diagnostics["total_closed_positions"] == 1
    assert diagnostics["filled_order_count"] == 1
    assert diagnostics["orders_with_decision_id"] == 0
    assert diagnostics["matched_position_count"] == 0
    assert diagnostics["reason"] == "filled_orders_missing_decision_id"
    assert diagnostics["ready_for_profit_learning"] is False


def test_model_contribution_matches_okx_symbol_variants() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    raw = {"opportunity_score": {"ml_aligned": True}}
    position = _position(1, 3.0, created_at=now)
    position.symbol = "BTC/USDT"
    position.entry_exchange_order_id = "entry-ok"
    position.close_exchange_order_id = "close-ok"
    order = _order(1, 7, created_at=now)
    order.symbol = "BTC-USDT-SWAP"
    decision = _decision(7, raw)

    service = ModelContributionPerformanceService()
    diagnostics = service.build_lineage_diagnostics([position], [order], {7: decision})
    stats = service.build_stats([position], [order], {7: decision})

    assert diagnostics["matched_position_count"] == 1
    assert diagnostics["reason"] == "ok"
    assert stats["ml_profit_model"]["count"] == 1


@pytest.mark.asyncio
async def test_model_contribution_recent_reads_exact_entry_orders_in_closed_position_window() -> None:
    now = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
    raw = {"opportunity_score": {"ml_aligned": True}}
    position = SimpleNamespace(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
        realized_pnl=3.0,
        is_open=False,
        created_at=now - timedelta(minutes=1),
        closed_at=now,
        entry_exchange_order_id="entry-a,entry-b",
        close_exchange_order_id="close-a",
    )
    order = _order(1, 1, created_at=now - timedelta(minutes=1))
    order.exchange_order_id = "entry-a"
    decision = _decision(1, raw)
    session = _FakeSession([[position], [], [order], [decision]])

    @asynccontextmanager
    async def session_factory():
        yield session

    service = ModelContributionPerformanceService(
        session_factory=session_factory,
        clock=lambda: now,
    )

    stats = await service.recent("paper")

    assert stats["ml_profit_model"]["count"] == 1
    assert len(session.statements) == 4
    manual_close_query = str(session.statements[1])
    entry_order_query = session.statements[2]
    assert "orders.filled_at BETWEEN" in manual_close_query
    assert "orders.created_at BETWEEN" in manual_close_query
    assert "orders.exchange_order_id IN" in str(entry_order_query)
    entry_order_params = entry_order_query.compile().params
    entry_id_lists = [value for value in entry_order_params.values() if isinstance(value, list)]
    assert {"entry-a", "entry-b"} in [set(value) for value in entry_id_lists]


@pytest.mark.asyncio
async def test_model_contribution_recent_cache_is_isolated_per_execution_mode() -> None:
    now = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
    session = _FakeSession([[], []])

    @asynccontextmanager
    async def session_factory():
        yield session

    service = ModelContributionPerformanceService(
        session_factory=session_factory,
        clock=lambda: now,
    )

    await service.recent("paper")
    await service.recent("paper")
    await service.recent("live")

    assert len(session.statements) == 2
    assert set(service._cache_by_mode) == {"paper", "live"}


def test_model_contribution_sources_fall_back_to_ai_only_without_quant() -> None:
    service = ModelContributionPerformanceService()

    sources = service.contribution_sources({}, {}, "short")

    assert sources == ["ai_only_without_quant"]


def test_model_contribution_score_adjustment_flags_negative_sources() -> None:
    service = ModelContributionPerformanceService()
    performance = {
        "ml_profit_model": {
            "label": "本地 ML 盈利模型",
            "count": 6,
            "pnl": -12.0,
            "profit_factor": 0.5,
            "score_multiplier": 0.7,
            "size_multiplier": 0.75,
            "state": "degrade",
            "reason": "recent losses",
        }
    }

    adjustment = service.score_adjustment(["ml_profit_model"], performance)

    assert adjustment["active"] is True
    assert adjustment["state"] == "degrade"
    assert adjustment["hard_caution"] is True
    assert adjustment["score_multiplier"] == pytest.approx(0.7)
    assert adjustment["size_multiplier"] == pytest.approx(0.75)
    assert adjustment["negative_sources"][0]["source"] == "ml_profit_model"


@pytest.mark.asyncio
async def test_trading_service_model_contribution_methods_delegate_to_service() -> None:
    service = object.__new__(TradingService)
    calls: list[tuple[str, object]] = []

    class FakeContributionService:
        async def recent(self, mode: str):
            calls.append(("recent", mode))
            return {"ml_profit_model": {"pnl": 1.0}}

        def contribution_sources(self, opportunity, raw, side):
            calls.append(("sources", side))
            return ["ml_profit_model"]

        def score_adjustment(self, sources, performance):
            calls.append(("adjust", tuple(sources)))
            return {"active": True, "score_adjustment": 0.1}

    service.model_contribution_performance_service = FakeContributionService()

    assert await service._recent_model_contribution_performance("paper") == {
        "ml_profit_model": {"pnl": 1.0}
    }
    assert service._decision_contribution_sources({}, {}, "long") == ["ml_profit_model"]
    assert service._model_contribution_score_adjustment(["ml_profit_model"], {}) == {
        "active": True,
        "score_adjustment": 0.1,
    }
    assert calls == [
        ("recent", "paper"),
        ("sources", "long"),
        ("adjust", ("ml_profit_model",)),
    ]
