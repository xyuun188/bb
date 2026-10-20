from datetime import UTC, datetime, timedelta

import pytest

from scripts import run_strategy_replay_governance as worker
from scripts.install_strategy_replay_timer import render_service, render_timer
from services.strategy_replay_governance import (
    load_replay_governance,
    summarize_replay_governance,
    write_replay_governance,
)


def test_governance_report_keeps_rejections_and_detects_stale_worker(tmp_path):
    path = tmp_path / "latest.json"
    report = summarize_replay_governance({
        "schedule": {
            "historical_model_replay": {"status": "complete", "can_authorize_live": False},
            "candidates": [
                {"promotion": {"rejection_reasons": ["negative_exam"]}},
                {"promotion": {"rejection_reasons": ["negative_exam", "unstable"]}},
            ],
        },
        "champion_reconciled": True,
    })
    write_replay_governance(report, path=path)
    generated = datetime.fromisoformat(report["generated_at"])
    assert load_replay_governance(path=path, now=generated)["stale"] is False
    assert load_replay_governance(
        path=path, now=generated + timedelta(hours=1)
    )["stale"] is True
    assert report["candidate_rejection_counts"] == {"negative_exam": 2, "unstable": 1}
    assert report["submits_orders"] is False
    assert report["live_execution_permission"] is False
    assert report["profitability_status"] == "no_validated_fee_after_strategy"


def test_missing_or_corrupt_governance_is_observable(tmp_path):
    path = tmp_path / "latest.json"
    assert load_replay_governance(path=path)["status"] == "missing"
    path.write_text("{", encoding="utf-8")
    assert load_replay_governance(path=path)["stale"] is True


@pytest.mark.asyncio
async def test_worker_excludes_overlapping_run_and_releases_lease_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "latest.json"
    monkeypatch.setattr(worker, "replay_report_path", lambda: path)
    monkeypatch.setattr(
        worker, "write_replay_governance",
        lambda report: write_replay_governance(report, path=path),
    )
    calls = []

    class Service:
        async def evaluate_paper_champion(self):
            calls.append(True)
            duplicate = await worker.run_once()
            assert duplicate["status"] == "skipped"
            raise RuntimeError("failed evidence read")

    monkeypatch.setattr(worker, "StrategyLearningService", Service)
    first = await worker.run_once()
    second = await worker.run_once()
    assert first["status"] == second["status"] == "error"
    assert calls == [True, True]
    persisted = load_replay_governance(path=path, now=datetime.now(UTC))
    assert persisted["error_type"] == "RuntimeError"
    assert persisted["live_execution_permission"] is False


def test_timer_runs_separately_with_bounded_database_and_cpu_resources():
    service = render_service()
    assert "run_strategy_replay_governance.py" in service
    assert "DATABASE_POOL_SIZE=2" in service
    assert "DATABASE_MAX_OVERFLOW=0" in service
    assert "CPUQuota=50%" in service
    assert "MemoryMax=2G" in service
    assert "TimeoutStartSec=11min" in service
    assert "KillMode=control-group" in service
    assert "OnUnitInactiveSec=30min" in render_timer()


@pytest.mark.asyncio
async def test_worker_timeout_is_persisted_and_database_is_closed(tmp_path, monkeypatch):
    import asyncio

    path = tmp_path / "latest.json"
    closed = []
    monkeypatch.setattr(worker, "replay_report_path", lambda: path)
    monkeypatch.setattr(worker, "REPLAY_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        worker, "write_replay_governance",
        lambda report: write_replay_governance(report, path=path),
    )

    class Service:
        async def evaluate_paper_champion(self):
            await asyncio.Event().wait()

    async def close():
        closed.append(True)

    monkeypatch.setattr(worker, "StrategyLearningService", Service)
    monkeypatch.setattr(worker, "close_db", close)
    result = await worker.run_once()
    assert result["error_type"] == "TimeoutError"
    assert load_replay_governance(path=path)["status"] == "error"
    assert closed == [True]
