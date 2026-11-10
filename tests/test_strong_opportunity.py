from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.session import close_db, get_read_session_ctx, get_session_ctx, init_db
from models.decision import AIDecision
from services import strong_opportunity
from services.strong_opportunity import StrongOpportunityService


async def _reset_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    await close_db()
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{(tmp_path / 'strong.db').as_posix()}")
    await init_db()


def _provenance() -> dict[str, object]:
    return {
        "source": "test_live_distribution",
        "observation_window": "current_test_round",
        "sample_count": 3,
        "generated_at": "2026-07-12T00:00:00+00:00",
        "strategy_version": "test-v1",
        "fallback_reason": "",
    }


def _raw(*, complete: bool = True) -> dict[str, object]:
    provenance = _provenance()
    return {
        "live_ml_profit_contract": {
            "eligible": complete,
            "expected_net_return_pct": 1.2,
            "return_lcb_pct": 0.4,
            "production_source_count": 3,
            "position_size_pct": 0.1,
            "policy_provenance": provenance,
        },
        "opportunity_score": {
            "production_eligible": complete,
            "policy_provenance": provenance,
            "execution_cost": {
                "production_eligible": complete,
                "total_pct": 0.08,
                "policy_provenance": provenance,
            },
        },
        "profit_risk_sizing": {
            "production_eligible": complete,
            "risk_budget_usdt": 3.0,
            "planned_stressed_loss_usdt": 2.4,
            "stressed_loss_fraction": 0.02,
            "target_notional_usdt": 150.0,
            "final_notional_usdt": 120.0,
            "policy_provenance": provenance,
        },
    }


@pytest.mark.asyncio
async def test_report_identifies_complete_positive_return_contract(tmp_path, monkeypatch) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        async with get_session_ctx() as session:
            session.add(AIDecision(model_name="ensemble_trader", symbol="BTC/USDT", action="long", confidence=0.2, raw_llm_response=_raw(), was_executed=True, created_at=datetime.now(UTC) - timedelta(hours=1)))
        report = await StrongOpportunityService().report()
        assert report["strong_candidate_count"] == 1
        assert report["strong_candidates"][0]["stage"] == "live_ml_ready"
        assert report["contract"]["fixed_strategy_thresholds"] == []
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_positive_expectation_with_incomplete_cost_stays_observation_only(tmp_path, monkeypatch) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        raw = _raw()
        raw["opportunity_score"]["execution_cost"] = {}
        async with get_session_ctx() as session:
            session.add(AIDecision(model_name="ensemble_trader", symbol="ETH/USDT", action="long", confidence=0.99, raw_llm_response=raw, was_executed=False, created_at=datetime.now(UTC) - timedelta(hours=1)))
        report = await StrongOpportunityService().report()
        assert report["strong_candidate_count"] == 0
        assert report["near_miss_count"] == 1
        assert "live_execution_cost_incomplete" in report["near_misses"][0]["block_reasons"]
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_hold_window_does_not_load_payloads_or_older_entries(tmp_path, monkeypatch) -> None:
    await _reset_db(tmp_path, monkeypatch)
    queries = []

    class RecordingSession:
        def __init__(self, session):
            self.session = session

        async def execute(self, statement):
            queries.append(str(statement))
            return await self.session.execute(statement)

    @asynccontextmanager
    async def recording_session():
        async with get_read_session_ctx() as session:
            yield RecordingSession(session)

    try:
        now = datetime.now(UTC)
        async with get_session_ctx() as session:
            session.add_all([
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="BTC/USDT",
                    action="long",
                    confidence=0.2,
                    raw_llm_response=_raw(),
                    created_at=now - timedelta(hours=1),
                ),
                *[
                    AIDecision(
                        model_name="ensemble_trader",
                        symbol="ETH/USDT",
                        action="hold",
                        confidence=0.2,
                        raw_llm_response={"reasoning": "large transcript" * 1000},
                        feature_snapshot={"unused": "large snapshot" * 1000},
                        created_at=now - timedelta(minutes=minute),
                    )
                    for minute in (1, 2)
                ],
            ])
        monkeypatch.setattr(strong_opportunity, "get_read_session_ctx", recording_session)
        report = await StrongOpportunityService(limit=2).report()

        assert report["checked_decisions"] == 2
        assert report["entry_decisions"] == 0
        assert report["strong_candidate_count"] == 0
        assert len(queries) == 1
        assert "raw_llm_response" not in queries[0]
        assert "feature_snapshot" not in queries[0]
    finally:
        await close_db()
