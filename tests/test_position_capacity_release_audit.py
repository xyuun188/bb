from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.trade import Order, Position
from services.position_capacity_release_audit import PositionCapacityReleaseAuditService


async def _reset_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'position_capacity_release.db').as_posix()}",
    )
    await init_db()


@pytest.mark.asyncio
async def test_position_capacity_release_report_tracks_unclosed_release_decision(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        created = datetime.now(UTC) - timedelta(hours=5)
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="live",
                    symbol="BZ/USDT",
                    side="short",
                    quantity=10,
                    entry_price=0.74,
                    current_price=0.739,
                    unrealized_pnl=0.01,
                    leverage=3,
                    is_open=True,
                    created_at=created,
                )
            )
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="BZ/USDT",
                    action="close_short",
                    confidence=0.94,
                    analysis_type="position",
                    was_executed=False,
                    raw_llm_response={
                        "analysis_type": "position_review",
                        "exit_intent": "capital_rotation",
                        "position_release_policy": {
                            "forced": True,
                            "source": "position_quality_capacity_release",
                            "exit_score": 94,
                            "release_fraction": 1,
                            "release_reason": "fee_drag_dominates",
                            "scan_reason": "quality_release_now",
                        },
                        "position_quality": {
                            "score": 24,
                            "bucket": "release_now",
                            "should_release": True,
                        },
                    },
                    created_at=datetime.now(UTC) - timedelta(minutes=5),
                )
            )

        report = await PositionCapacityReleaseAuditService(lookback_hours=24).report()

        assert report["audit_only"] is True
        assert report["live_exit_mutation"] is False
        assert report["can_force_close"] is False
        assert report["open_position_count"] == 1
        assert report["release_decision_count"] == 1
        assert report["unclosed_release_decision_count"] == 1
        assert report["unclosed_release_decisions"][0]["symbol"] == "BZ/USDT"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_capacity_release_report_counts_executed_release_decision(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="ATOM/USDT",
                action="close_short",
                confidence=0.94,
                analysis_type="position",
                was_executed=True,
                raw_llm_response={
                    "analysis_type": "position_review",
                    "exit_intent": "capital_rotation",
                    "position_release_policy": {
                        "forced": True,
                        "source": "position_quality_capacity_release",
                    },
                },
                created_at=datetime.now(UTC) - timedelta(minutes=4),
            )
            session.add(decision)
            await session.flush()
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="live",
                    symbol="ATOM/USDT",
                    side="buy",
                    order_type="market",
                    quantity=2,
                    price=5.1,
                    status="filled",
                    decision_id=decision.id,
                    created_at=datetime.now(UTC) - timedelta(minutes=3),
                )
            )

        report = await PositionCapacityReleaseAuditService(lookback_hours=24).report()

        assert report["release_decision_count"] == 1
        assert report["executed_release_decision_count"] == 1
        assert report["unclosed_release_decision_count"] == 0
    finally:
        await close_db()
