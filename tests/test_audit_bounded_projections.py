from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from models.decision import AIDecision
from models.learning import ShadowBacktest
from models.trade import Order
from services.high_risk_review_audit import HighRiskReviewAuditService
from services.shadow_missed_opportunity_closed_loop import (
    ShadowMissedOpportunityClosedLoopService,
    summarize_shadow_missed_opportunities,
)
from tests.paper_canary_fixtures import complete_paper_canary_raw


@pytest_asyncio.fixture
async def audit_database():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(AIDecision.__table__.create)
        await connection.run_sync(ShadowBacktest.__table__.create)
        await connection.run_sync(Order.__table__.create)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_high_risk_audit_preserves_counts_without_loading_full_decisions(audit_database):
    now = datetime.now(UTC)
    values = []
    for record_id, action in enumerate(("hold", "long", "short"), start=1):
        values.append({
            "id": record_id, "model_name": "test", "symbol": "BTC/USDT",
            "action": action, "confidence": 0.8, "created_at": now,
            "was_executed": action == "short",
            "raw_llm_response": {"unrelated": "x" * 1000, "high_risk_review": {
                "triggered": True, "hard_review_required": True,
                "approved": False, "status": "error_blocked",
            }},
        })
    async with audit_database.begin() as connection:
        await connection.execute(AIDecision.__table__.insert(), values)

    class BoundedSession(AsyncSession):
        async def execute(self, statement, *args, **kwargs):
            for column in statement.selected_columns:
                assert str(column) != "ai_decisions.raw_llm_response"
                assert str(column) != "ai_decisions.feature_snapshot"
            return await super().execute(statement, *args, **kwargs)

    maker = async_sessionmaker(audit_database, class_=BoundedSession)
    report = await HighRiskReviewAuditService(session_context_factory=maker).report(hours=24)
    assert report["checked_decisions"] == 3
    assert report["entry_decision_count"] == 2
    assert report["review_payload_count"] == 2
    assert report["blocked_count"] == 2
    assert report["executed_without_required_review_count"] == 1


@pytest.mark.asyncio
async def test_missed_opportunity_audit_reads_payloads_only_for_used_rows(audit_database):
    now = datetime.now(UTC)
    features = {
        "bid": 99.99, "ask": 100.01, "orderbook_bid_depth": 100_000.0,
        "orderbook_ask_depth": 100_000.0, "taker_fee_rate": 0.0004,
        "funding_rate": 0.0, "funding_data_available": True,
        "funding_interval_minutes": 480.0,
    }
    shadows = [{
        "id": record_id, "model_name": "test",
        "symbol": "BTC/USDT", "status": status,
        "execution_mode": "paper", "due_at": now,
        "missed_opportunity": missed, "horizon_minutes": 30,
        "long_return_pct": 0.8, "short_return_pct": -0.8,
        "best_action": "long", "feature_snapshot": features,
        "created_at": now - timedelta(hours=48 if record_id == 4 else 0),
    } for record_id, (status, missed) in enumerate(
        (("completed", True), ("completed", False), ("pending", True),
         ("completed", True)), start=1,
    )]
    decisions = [{
        "id": record_id, "model_name": "test", "symbol": "BTC/USDT",
        "action": "long", "confidence": 0.8, "created_at": now,
        "was_executed": record_id != 3, "raw_llm_response": raw,
    } for record_id, raw in enumerate(
        (complete_paper_canary_raw(), {"live_ml_profit_contract": {"eligible": False}}, {}),
        start=1,
    )]
    async with audit_database.begin() as connection:
        await connection.execute(AIDecision.__table__.insert(), decisions)
        await connection.execute(ShadowBacktest.__table__.insert(), shadows)

    payload_ids = {}

    class BoundedSession(AsyncSession):
        async def execute(self, statement, *args, **kwargs):
            names = [str(column) for column in statement.selected_columns]
            assert "shadow_backtests.training_feature_snapshot" not in names
            if any(name in names for name in (
                "shadow_backtests.feature_snapshot", "ai_decisions.raw_llm_response",
            )):
                assert len(names) == 2
                params = statement.compile().params
                ids = next(value for value in params.values() if isinstance(value, list))
                payload_ids[names[1]] = ids
            return await super().execute(statement, *args, **kwargs)

    maker = async_sessionmaker(audit_database, class_=BoundedSession)

    @asynccontextmanager
    async def session_context():
        async with maker() as session:
            yield session

    report = await ShadowMissedOpportunityClosedLoopService(
        session_context_factory=session_context,
    ).report(hours=24)
    expected = summarize_shadow_missed_opportunities(
        shadows[:3], decisions=decisions, orders=[],
    )
    assert report["summary"] == expected["summary"]
    assert report["blocked_reason_counts"] == expected["blocked_reason_counts"]
    assert report["return_observations"][0]["average_return_pct"] == (
        expected["return_observations"][0]["average_return_pct"]
    )
    assert payload_ids == {
        "shadow_backtests.feature_snapshot": [1],
        "ai_decisions.raw_llm_response": [2, 1],
    }
