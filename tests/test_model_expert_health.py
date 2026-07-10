from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.learning import ExpertMemory, ShadowBacktest, StrategyLearningEvent
from services.model_expert_health import ModelExpertHealthService, summarize_model_expert_health


async def _use_temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    await close_db()
    db_path = tmp_path / "model-health.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db()


def _decision(
    *,
    decision_id: int,
    action: str,
    hours_ago: float,
    pnl: float | None = None,
    executed: bool = False,
    position_size_pct: float = 0.05,
    raw: dict | None = None,
) -> SimpleNamespace:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    return SimpleNamespace(
        id=decision_id,
        model_name="ensemble_trader",
        action=action,
        was_executed=executed,
        outcome_pnl_pct=pnl,
        position_size_pct=position_size_pct,
        raw_llm_response=raw or {},
        created_at=now - timedelta(hours=hours_ago),
    )


def _shadow(
    *,
    decision_id: int,
    hours_ago: float,
    best_action: str,
    missed: bool = False,
) -> SimpleNamespace:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    return SimpleNamespace(
        decision_id=decision_id,
        status="completed",
        best_action=best_action,
        missed_opportunity=missed,
        created_at=now - timedelta(hours=hours_ago),
    )


def test_model_expert_health_report_marks_detractors_without_mutating_weights() -> None:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    raw_good = {
        "model_timings": [
            {
                "name": "trend_expert",
                "status": "completed",
                "provider_model": "qwen3-14b-trade",
                "duration_sec": 2.0,
                "action": "long",
                "confidence": 0.72,
            },
            {
                "name": "risk_expert",
                "status": "completed",
                "provider_model": "deepseek-r1-14b-risk",
                "duration_sec": 5.0,
                "action": "hold",
                "confidence": 0.64,
            },
        ],
        "experts": [
            {"expert_name": "trend_expert", "action": "long", "confidence": 0.72},
            {"expert_name": "risk_expert", "action": "hold", "confidence": 0.64},
        ],
    }
    raw_bad = {
        "model_timings": [
            {
                "name": "trend_expert",
                "status": "completed",
                "provider_model": "qwen3-14b-trade",
                "duration_sec": 2.8,
                "action": "short",
                "confidence": 0.69,
            },
            {
                "name": "risk_expert",
                "status": "failed",
                "provider_model": "deepseek-r1-14b-risk",
                "duration_sec": 18.0,
                "reason": "Could not extract valid JSON from response",
            },
        ],
        "experts": [
            {"expert_name": "trend_expert", "action": "short", "confidence": 0.69},
        ],
    }
    decisions = [
        _decision(decision_id=1, action="long", hours_ago=2, pnl=0.8, executed=True, raw=raw_good),
        _decision(decision_id=2, action="short", hours_ago=3, pnl=-1.4, executed=True, raw=raw_bad),
        _decision(decision_id=3, action="hold", hours_ago=4, executed=False, raw=raw_bad),
    ]
    shadows = [
        _shadow(decision_id=1, hours_ago=1.5, best_action="long"),
        _shadow(decision_id=2, hours_ago=2.5, best_action="long"),
        _shadow(decision_id=3, hours_ago=3.5, best_action="long", missed=True),
    ]

    report = summarize_model_expert_health(decisions, shadows, now=now)

    assert report["audit_only"] is True
    assert report["live_weight_mutation"] is False
    assert report["windows_hours"] == [24, 72]
    trend = report["components"]["trend_expert"]
    assert trend["type"] == "expert"
    assert trend["windows"]["24h"]["participation_count"] == 3
    assert trend["windows"]["24h"]["adopted_count"] == 2
    assert trend["windows"]["24h"]["adopted_net_pnl_pct"] == -0.6
    assert trend["windows"]["24h"]["wrong_recommendation_rate"] > 0
    assert trend["recommended_state"] == "reduce"
    assert "negative_adopted_pnl" in trend["state_reasons"]

    risk = report["components"]["risk_expert"]
    assert risk["windows"]["24h"]["json_error_count"] == 2
    assert risk["windows"]["24h"]["no_return_count"] == 2
    assert risk["recommended_state"] in {"shadow_only", "disable"}
    assert risk["stability"]["json_error_rate"] > 0
    assert report["summary"]["components"] >= 2


def test_model_expert_health_report_observes_insufficient_samples() -> None:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    decisions = [
        _decision(
            decision_id=1,
            action="long",
            hours_ago=1,
            pnl=0.4,
            executed=True,
            raw={
                "model_timings": [
                    {
                        "name": "sentiment_expert",
                        "status": "completed",
                        "duration_sec": 1.1,
                        "action": "long",
                    }
                ],
                "experts": [
                    {"expert_name": "sentiment_expert", "action": "long", "confidence": 0.58}
                ],
            },
        )
    ]

    report = summarize_model_expert_health(decisions, [], now=now)

    sentiment = report["components"]["sentiment_expert"]
    assert sentiment["recommended_state"] == "shadow_only"
    assert sentiment["evidence_state"] == "observing"
    assert "insufficient_samples" in sentiment["state_reasons"]


def test_model_expert_health_does_not_count_successful_independent_retry_as_json_error() -> None:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    timings = [
        {
            "name": name,
            "status": "completed",
            "stage": "expert_independent_provider",
            "provider_model": "deepseek-r1-14b-risk",
            "provider_independent_expert_mode": True,
            "batch_expert": False,
            "shared_batch_call": False,
            "batch_failure_status": "batch_fallback",
            "duration_sec": 4.2,
            "action": "hold",
            "confidence": 0.61,
            "reason": 'batch expert failed: Could not extract valid JSON from: {"experts":',
        }
        for name in ("sentiment_expert", "position_expert", "risk_expert")
    ]
    decisions = [
        _decision(
            decision_id=index,
            action="hold",
            hours_ago=index,
            executed=False,
            raw={
                "model_timings": timings,
                "experts": [
                    {"expert_name": item["name"], "action": "hold", "confidence": 0.61}
                    for item in timings
                ],
            },
        )
        for index in range(1, 4)
    ]

    report = summarize_model_expert_health(decisions, [], now=now)

    for name in ("sentiment_expert", "position_expert", "risk_expert"):
        component = report["components"][name]
        assert component["windows"]["24h"]["participation_count"] == 3
        assert component["windows"]["24h"]["json_error_count"] == 0
        assert component["windows"]["24h"]["no_return_count"] == 0
        assert component["stability"]["json_error_rate"] == 0.0


@pytest.mark.asyncio
async def test_model_health_report_projects_required_raw_fragments_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    now = datetime.now(UTC)
    async with get_session_ctx() as session:
        session.add_all(
            [
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="BTC/USDT",
                    action="long",
                    confidence=0.7,
                    was_executed=True,
                    outcome_pnl_pct=0.4,
                    raw_llm_response={
                        "model_timings": [
                            {
                                "name": "trend_expert",
                                "status": "completed",
                                "duration_sec": 0.4,
                                "action": "long",
                            }
                        ],
                        "unused_full_transcript": "x" * 100_000,
                    },
                    created_at=now,
                ),
                ShadowBacktest(
                    decision_id=1,
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    status="completed",
                    due_at=now,
                    best_action="long",
                    created_at=now,
                ),
                ExpertMemory(
                    expert_name="trend_expert",
                    memory_key="trend-expert-test",
                    evidence_count=3,
                    success_count=2,
                    failure_count=1,
                ),
                StrategyLearningEvent(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    event_type="execution_result",
                    attribution={"trend": {"source": "trend_expert"}},
                    strategy_snapshot={"unused": "x" * 100_000},
                    created_at=now,
                ),
            ]
        )

    try:
        report = await ModelExpertHealthService().report(hours=24, limit=20)
        async with get_session_ctx() as session:
            decision = await session.get(AIDecision, 1)
            assert decision is not None
            assert decision.model_health_snapshot_version == 1
            assert decision.model_health_timings == [
                {
                    "name": "trend_expert",
                    "status": "completed",
                    "duration_sec": 0.4,
                    "action": "long",
                }
            ]
            assert decision.model_health_experts is None
    finally:
        await close_db()

    trend = report["components"]["trend_expert"]
    assert trend["windows"]["24h"]["participation_count"] == 1
    assert trend["memory"]["evidence_count"] == 3
