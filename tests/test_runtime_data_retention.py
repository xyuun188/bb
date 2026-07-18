from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from config.settings import settings
from core.runtime_data_retention_contract import (
    RETENTION_MARKER_KEY,
    RUNTIME_DATA_RETENTION_VERSION,
)
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.learning import ShadowBacktest, StrategyLearningEvent
from models.trade import Order
from scripts import install_runtime_data_retention_timer as timer_script
from scripts import run_runtime_data_retention as runner_script
from services.runtime_data_retention import (
    RuntimeDataRetentionPolicy,
    RuntimeDataRetentionService,
)


async def _use_temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'runtime-retention.db').as_posix()}",
    )
    await init_db()


def _decision(*, created_at: datetime, executed: bool = False) -> AIDecision:
    return AIDecision(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action="long",
        confidence=0.72,
        position_size_pct=2.0,
        suggested_leverage=2.0,
        stop_loss_pct=1.0,
        take_profit_pct=2.0,
        feature_snapshot={"current_price": 100.0, "unused": "x" * 10_000},
        raw_llm_response={
            "model_timings": [{"model": "decision_maker", "latency_ms": 120}],
            "ml_signal": {"best_side": "long", "expected_return_pct": 0.2},
            "training_fact": {"spread_pct": 0.01},
            "transcript": "x" * 20_000,
        },
        analysis_type="market",
        is_paper=True,
        was_executed=executed,
        created_at=created_at,
    )


def _shadow(
    *,
    created_at: datetime,
    status: str,
    action: str = "long",
) -> ShadowBacktest:
    return ShadowBacktest(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        analysis_type="market",
        decision_action=action,
        decision_confidence=0.7,
        entry_price=100.0,
        feature_snapshot={
            "current_price": 100.0,
            "spread_pct": 0.01,
            "unused_transcript": "x" * 20_000,
        },
        raw_llm_response={"response": "x" * 20_000},
        status=status,
        due_at=created_at + timedelta(minutes=30),
        horizon_minutes=30,
        actual_price=101.0,
        long_return_pct=1.0,
        short_return_pct=-1.0,
        best_action="long",
        missed_opportunity=False,
        created_at=created_at,
    )


def _strategy_event(*, created_at: datetime, status: str) -> StrategyLearningEvent:
    return StrategyLearningEvent(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
        action="long",
        event_type="entry_decision",
        event_status=status,
        reason="retained scalar reason",
        strategy_snapshot={"profile": "x" * 4_000},
        market_state={"market": "x" * 4_000},
        side_weights={"long": 0.7},
        expert_integrity={"healthy": True},
        attribution={"source": "test"},
        created_at=created_at,
    )


@pytest.mark.asyncio
async def test_runtime_retention_is_dry_run_safe_and_apply_preserves_facts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    old = now - timedelta(days=30)
    protected_shadow_time = now - timedelta(days=8)

    candidate = _decision(created_at=old)
    executed = _decision(created_at=old, executed=True)
    linked = _decision(created_at=old)
    protected_shadow = _shadow(created_at=protected_shadow_time, status="completed")
    quarantined_shadow = _shadow(created_at=old, status="quarantined")
    terminal_event = _strategy_event(created_at=old, status="recorded")
    active_event = _strategy_event(created_at=old, status="active")
    async with get_session_ctx() as session:
        session.add_all(
            [
                candidate,
                executed,
                linked,
                protected_shadow,
                quarantined_shadow,
                terminal_event,
                active_event,
            ]
        )
        await session.flush()
        session.add(
            Order(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=1.0,
                status="filled",
                decision_id=linked.id,
            )
        )
        ids = {
            "candidate": candidate.id,
            "executed": executed.id,
            "linked": linked.id,
            "protected_shadow": protected_shadow.id,
            "quarantined_shadow": quarantined_shadow.id,
            "terminal_event": terminal_event.id,
            "active_event": active_event.id,
        }
        candidate_learning_snapshot = dict(candidate.decision_learning_snapshot or {})
        candidate_model_timings = list(candidate.model_health_timings or [])
        protected_shadow_raw = dict(protected_shadow.raw_llm_response or {})
        quarantined_labels = (
            quarantined_shadow.long_return_pct,
            quarantined_shadow.short_return_pct,
            quarantined_shadow.best_action,
            quarantined_shadow.training_feature_snapshot,
        )

    policy = RuntimeDataRetentionPolicy(
        batch_size=2,
        max_rows_per_table=10,
        batch_pause_seconds=0,
    )
    service = RuntimeDataRetentionService(policy)
    dry_run = await service.run(apply=False, now=now)
    assert dry_run["mode"] == "dry_run"
    assert dry_run["mutates_database"] is False
    assert dry_run["summary"]["eligible_rows"] == 3
    assert dry_run["summary"]["processed_rows"] == 0

    async with get_session_ctx() as session:
        untouched = await session.get(AIDecision, ids["candidate"])
        assert untouched is not None
        assert untouched.feature_snapshot is not None
        assert RETENTION_MARKER_KEY not in (untouched.raw_llm_response or {})

    applied = await service.run(apply=True, now=now)
    assert applied["mode"] == "apply"
    assert applied["summary"]["processed_rows"] == 3, applied
    assert all(
        section["bounded_by_max_rows"] is False
        for section in applied["sections"].values()
    )

    async with get_session_ctx() as session:
        compacted = await session.get(AIDecision, ids["candidate"])
        kept_executed = await session.get(AIDecision, ids["executed"])
        kept_linked = await session.get(AIDecision, ids["linked"])
        kept_shadow = await session.get(ShadowBacktest, ids["protected_shadow"])
        compacted_shadow = await session.get(ShadowBacktest, ids["quarantined_shadow"])
        compacted_event = await session.get(StrategyLearningEvent, ids["terminal_event"])
        kept_event = await session.get(StrategyLearningEvent, ids["active_event"])

        assert compacted is not None
        assert compacted.feature_snapshot is None
        assert compacted.runtime_payload_compaction_version == (
            RUNTIME_DATA_RETENTION_VERSION
        )
        assert compacted.runtime_payload_compacted_at is not None
        assert compacted.raw_llm_response[RETENTION_MARKER_KEY]["version"] == (
            RUNTIME_DATA_RETENTION_VERSION
        )
        assert compacted.decision_learning_snapshot == candidate_learning_snapshot
        assert compacted.model_health_timings == candidate_model_timings
        assert kept_executed is not None and kept_executed.feature_snapshot is not None
        assert kept_linked is not None and kept_linked.feature_snapshot is not None

        assert kept_shadow is not None
        assert kept_shadow.raw_llm_response == protected_shadow_raw
        assert compacted_shadow is not None
        assert compacted_shadow.runtime_payload_compaction_version == (
            RUNTIME_DATA_RETENTION_VERSION
        )
        assert compacted_shadow.runtime_payload_compacted_at is not None
        assert compacted_shadow.raw_llm_response[RETENTION_MARKER_KEY]["version"] == (
            RUNTIME_DATA_RETENTION_VERSION
        )
        assert compacted_shadow.feature_snapshot == quarantined_labels[3]
        assert (
            compacted_shadow.long_return_pct,
            compacted_shadow.short_return_pct,
            compacted_shadow.best_action,
            compacted_shadow.training_feature_snapshot,
        ) == quarantined_labels

        assert compacted_event is not None
        assert compacted_event.reason == "retained scalar reason"
        assert compacted_event.strategy_snapshot is None
        assert compacted_event.market_state is None
        assert compacted_event.side_weights is None
        assert compacted_event.expert_integrity is None
        assert compacted_event.attribution is None
        assert kept_event is not None and kept_event.strategy_snapshot is not None

    second_apply = await service.run(apply=True, now=now)
    assert second_apply["summary"]["eligible_rows"] == 0
    assert second_apply["summary"]["processed_rows"] == 0
    routine_report = await service.run(
        apply=False,
        measure_reclaimable_bytes=False,
        now=now,
    )
    assert routine_report["summary"]["reclaimable_bytes_measured"] is False
    assert routine_report["summary"]["estimated_reclaimable_bytes"] is None
    assert all(
        section["estimated_reclaimable_bytes"] is None
        for section in routine_report["sections"].values()
    )
    await close_db()


def test_runtime_retention_runner_requires_explicit_apply_confirmation() -> None:
    with pytest.raises(SystemExit):
        runner_script.parse_args(["--apply"])
    with pytest.raises(SystemExit):
        runner_script.parse_args(["--confirm", runner_script.APPLY_CONFIRMATION])

    args = runner_script.parse_args(
        ["--apply", "--confirm", runner_script.APPLY_CONFIRMATION]
    )
    assert args.apply is True


def test_runtime_retention_timer_runs_as_bb_with_low_priority_runtime_env() -> None:
    service = timer_script.render_service()
    timer = timer_script.render_timer()

    assert "User=bb" in service
    assert "Group=bb" in service
    assert "EnvironmentFile=/etc/bb/bb-runtime.env" in service
    assert "Nice=10" in service
    assert "IOSchedulingClass=idle" in service
    assert "CPUWeight=10" in service
    assert "run_runtime_data_retention.py --apply --confirm compact-runtime-data-v1" in service
    assert "--max-rows-per-table 5000" in service
    assert "--skip-byte-estimate" in service
    assert "OnCalendar=*-*-* 18:20:00 UTC" in timer
    assert "Persistent=true" in timer
    assert "Unit=bb-runtime-data-retention.service" in timer


def test_runtime_retention_timer_dry_run_does_not_connect(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("dry-run must not connect to remote server")

    monkeypatch.setattr(timer_script, "connect_remote_ssh", fail_connect)
    timer_script.install_timer(dry_run=True)
    output = capsys.readouterr().out
    assert "bb-runtime-data-retention.service" in output
    assert "bb-runtime-data-retention.timer" in output
