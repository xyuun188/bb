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


@pytest.mark.asyncio
async def test_position_capacity_release_report_tracks_native_close_pending_backfill(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        async with get_session_ctx() as session:
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="ZRO/USDT",
                    action="close_long",
                    confidence=0.94,
                    analysis_type="position",
                    was_executed=False,
                    execution_reason=(
                        "OKX 原生平仓已确认交易所仓位归零，但成交明细暂时还没有返回真实订单号。"
                    ),
                    raw_llm_response={
                        "analysis_type": "position_review",
                        "exit_intent": "capital_rotation",
                        "position_release_policy": {
                            "forced": True,
                            "source": "position_quality_capacity_release",
                        },
                        "execution_result": {
                            "source": "exchange_not_confirmed",
                            "status": "partial",
                            "exchange_confirmed": False,
                            "exit_progress": True,
                            "raw_response": {
                                "okx_native_close_position": True,
                                "requires_okx_fill_backfill": True,
                            },
                        },
                    },
                    created_at=datetime.now(UTC) - timedelta(minutes=4),
                )
            )

        report = await PositionCapacityReleaseAuditService(lookback_hours=24).report()

        assert report["release_decision_count"] == 1
        assert report["unclosed_release_decision_count"] == 0
        assert report["release_execution_state_counts"] == {
            "exit_progress_pending_backfill": 1
        }
        assert report["release_execution_block_counts"] == {
            "okx_fill_backfill_pending": 1
        }
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_capacity_release_report_recovers_legacy_native_backfill_snapshot(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        async with get_session_ctx() as session:
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="ZRO/USDT",
                    action="close_long",
                    confidence=0.94,
                    analysis_type="position",
                    was_executed=False,
                    execution_reason="OKX 平仓已部分成交，系统会继续同步最终成交结果。",
                    raw_llm_response={
                        "analysis_type": "position_review",
                        "exit_intent": "capital_rotation",
                        "decision_state_machine": {
                            "summary": {
                                "final_stage": "local_sync",
                                "final_status": "skipped",
                                "failed": True,
                            }
                        },
                        "position_release_policy": {
                            "forced": True,
                            "source": "position_quality_capacity_release",
                        },
                        "execution_result": {
                            "source": "exchange_not_confirmed",
                            "status": "partial",
                            "exchange_confirmed": False,
                            "exit_progress": False,
                            "raw_response": {
                                "okx_native_close_position": True,
                                "requires_okx_fill_backfill": True,
                            },
                        },
                    },
                    created_at=datetime.now(UTC) - timedelta(minutes=4),
                )
            )

        report = await PositionCapacityReleaseAuditService(lookback_hours=24).report()

        assert report["unclosed_release_decision_count"] == 0
        assert report["release_execution_state_counts"] == {
            "exit_progress_pending_backfill": 1
        }
        assert report["release_execution_block_counts"] == {
            "okx_fill_backfill_pending": 1
        }
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_capacity_release_report_excludes_protected_non_execution_from_unclosed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        async with get_session_ctx() as session:
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="LINK/USDT",
                    action="close_short",
                    confidence=0.94,
                    analysis_type="position",
                    was_executed=False,
                    execution_reason=(
                        "仓位轮动保护：LINK/USDT 当前扣费后预计净亏 -0.0179 USDT，"
                        "未触发硬止损、止盈、严重趋势失效或预测下行风险。"
                    ),
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
            )

        report = await PositionCapacityReleaseAuditService(lookback_hours=24).report()

        assert report["release_decision_count"] == 1
        assert report["protected_release_decision_count"] == 1
        assert report["unclosed_release_decision_count"] == 0
        assert report["release_execution_state_counts"] == {"protected_not_executed": 1}
        assert report["protected_release_decisions"][0]["execution_state"] == (
            "protected_not_executed"
        )
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_capacity_release_report_excludes_exchange_cooldown_from_unclosed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        async with get_session_ctx() as session:
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="LAB/USDT",
                    action="close_long",
                    confidence=0.94,
                    analysis_type="position",
                    was_executed=False,
                    execution_reason=(
                        "不可交易平仓冷却：LAB/USDT 做多 上一次平仓提交被 OKX 明确拒绝"
                        "为交易对不可用，系统暂停重复提交约 1388 秒。"
                    ),
                    raw_llm_response={
                        "analysis_type": "position_review",
                        "exit_intent": "hard_risk",
                    },
                    created_at=datetime.now(UTC) - timedelta(minutes=4),
                )
            )

        report = await PositionCapacityReleaseAuditService(lookback_hours=24).report()

        assert report["release_decision_count"] == 1
        assert report["exchange_blocked_release_decision_count"] == 1
        assert report["unclosed_release_decision_count"] == 0
        assert report["release_execution_state_counts"] == {"exchange_blocked": 1}
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_capacity_release_report_separates_reported_fill_link_gap(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        async with get_session_ctx() as session:
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="AI16Z/USDT",
                    action="close_long",
                    confidence=0.95,
                    analysis_type="position",
                    was_executed=False,
                    execution_reason="订单已成交。",
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
            )

        report = await PositionCapacityReleaseAuditService(lookback_hours=24).report()

        assert report["release_decision_count"] == 1
        assert report["execution_link_gap_release_decision_count"] == 1
        assert report["unclosed_release_decision_count"] == 0
        assert report["release_execution_state_counts"] == {"reported_executed_without_link": 1}
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_capacity_release_report_separates_stale_skipped_decisions(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        async with get_session_ctx() as session:
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="LAB/USDT",
                    action="close_long",
                    confidence=0.81,
                    analysis_type="position",
                    was_executed=False,
                    execution_reason=(
                        "AI信号已过有效期：AI平仓裁决完成到准备下单已经过去 359 秒，"
                        "超过允许 300 秒。为避免使用旧裁决下单，本次不执行，等待下一轮重新分析。"
                    ),
                    raw_llm_response={
                        "analysis_type": "position_review",
                        "exit_intent": "profit_drawdown",
                    },
                    created_at=datetime.now(UTC) - timedelta(minutes=4),
                )
            )

        report = await PositionCapacityReleaseAuditService(lookback_hours=24).report()

        assert report["release_decision_count"] == 1
        assert report["stale_release_decision_count"] == 1
        assert report["unclosed_release_decision_count"] == 0
        assert report["release_execution_state_counts"] == {"stale_skipped": 1}
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_capacity_release_report_counts_only_real_crowded_blocks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    AIDecision(
                        model_name="ensemble_trader",
                        symbol="SPK/USDT",
                        action="short",
                        confidence=0.91,
                        analysis_type="market",
                        was_executed=True,
                        execution_reason="订单已成交。",
                        raw_llm_response={
                            "crowded_side_cap": {"mode": "crowded_strong_override"},
                            "opportunity_score": {"evidence_score": {"tier": "exploration"}},
                        },
                        created_at=datetime.now(UTC) - timedelta(minutes=5),
                    ),
                    AIDecision(
                        model_name="ensemble_trader",
                        symbol="BZ/USDT",
                        action="short",
                        confidence=0.72,
                        analysis_type="market",
                        was_executed=False,
                        execution_reason="单边拥挤硬上限[crowded_side_cap]：本轮拒绝开仓。",
                        raw_llm_response={
                            "crowded_side_cap": {"mode": "crowded_block"},
                            "entry_execution_gate": {
                                "status": "blocked",
                                "reason": "crowded_side_cap",
                            },
                            "opportunity_score": {"evidence_score": {"tier": "small"}},
                        },
                        created_at=datetime.now(UTC) - timedelta(minutes=4),
                    ),
                ]
            )

        report = await PositionCapacityReleaseAuditService(lookback_hours=24).report()

        assert report["crowded_block_count"] == 1
        assert report["crowded_blocks"][0]["symbol"] == "BZ/USDT"
        assert report["crowded_blocks"][0]["crowded_mode"] == "crowded_block"
    finally:
        await close_db()
