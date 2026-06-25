from __future__ import annotations

from datetime import UTC, datetime

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from services.strong_opportunity import StrongOpportunityService


async def _reset_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'strong_opportunity.db').as_posix()}",
    )
    await init_db()


def _strong_raw() -> dict:
    return {
        "entry_candidate_evidence": {
            "long": {
                "expected_net_return_pct": 1.25,
                "profit_quality_ratio": 1.35,
                "loss_probability": 0.31,
                "tail_risk_score": 0.45,
                "aligned_source_count": 3,
            }
        },
        "opportunity_score": {
            "side": "long",
            "score": 0.78,
            "min_score_required": 0.58,
            "expected_net_return_pct": 1.2,
            "profit_quality_ratio": 1.3,
            "server_profit_loss_probability": 0.32,
            "tail_risk_score": 0.46,
            "evidence_score": {
                "tier": "medium",
                "effective_score": 0.78,
                "aligned_support_sources": ["local_profit", "timeseries", "expert"],
                "major_opposites": [],
                "strong_opposites": [],
                "hard_block": False,
                "shadow_only": False,
            },
        },
        "high_risk_review": {"approved": True, "status": "approved"},
    }


@pytest.mark.asyncio
async def test_strong_opportunity_report_identifies_shadow_candidate(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        async with get_session_ctx() as session:
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="USAR/USDT",
                    action="long",
                    confidence=0.86,
                    raw_llm_response=_strong_raw(),
                    was_executed=True,
                    created_at=datetime(2026, 6, 25, 1, 0, tzinfo=UTC),
                )
            )

        report = await StrongOpportunityService(lookback_hours=24).report()

        assert report["audit_only"] is True
        assert report["live_entry_mutation"] is False
        assert report["can_force_open"] is False
        assert report["can_apply_live_sizing"] is False
        assert report["strong_candidate_count"] == 1
        candidate = report["strong_candidates"][0]
        assert candidate["symbol"] == "USAR/USDT"
        assert candidate["strong_opportunity"] is True
        assert candidate["shadow_only"] is False
        assert candidate["can_bypass_risk_controls"] is False
        assert candidate["can_apply_live_sizing"] is False
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_strong_opportunity_report_explains_near_miss_blockers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        raw = _strong_raw()
        raw["entry_candidate_evidence"]["long"]["expected_net_return_pct"] = 0.25
        raw["entry_candidate_evidence"]["long"]["loss_probability"] = 0.58
        raw["opportunity_score"]["evidence_score"]["tier"] = "blocked"
        async with get_session_ctx() as session:
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="MASK/USDT",
                    action="long",
                    confidence=0.72,
                    raw_llm_response=raw,
                    was_executed=False,
                    created_at=datetime(2026, 6, 25, 1, 5, tzinfo=UTC),
                )
            )

        report = await StrongOpportunityService(lookback_hours=24).report()

        assert report["strong_candidate_count"] == 0
        assert report["near_miss_count"] == 1
        blockers = report["near_misses"][0]["block_reasons"]
        assert "expected_net_below_strong_threshold" in blockers
        assert "loss_probability_above_strong_threshold" in blockers
        assert "evidence_tier_not_tradeable_strong" in blockers
        assert report["blocker_counts"]["expected_net_below_strong_threshold"] == 1
    finally:
        await close_db()
