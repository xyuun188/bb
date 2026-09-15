import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from services.continuous_observation import (
    ContinuousObservationScheduler,
    ContinuousObservationStore,
    ContinuousObservationWorkerState,
    observation_metrics_ready,
)
from services.observability_contract import normalize_status, status_from_sections


def _metrics(**overrides):
    metrics = {
        "service_restart_count": 0,
        "dashboard_timeout_storm_count": 0,
        "max_analysis_interval_seconds": 120,
        "duplicate_analysis_count": 0,
        "unexplained_data_collection_timeout_count": 0,
        "unresolved_trade_contract_count": 0,
        "unassigned_fill_count": 0,
        "duplicate_funding_attribution_count": 0,
        "model_tunnel_unresolved_timeout_count": 0,
        "training_state_clear": True,
        "attribution_mismatch_count": 0,
    }
    metrics.update(overrides)
    return metrics


def test_observation_requires_real_elapsed_window_and_metrics(tmp_path):
    store = ContinuousObservationStore(tmp_path / "observation.json")
    started = datetime(2026, 8, 29, tzinfo=UTC)

    assert store.snapshot(now=started)["status"] == "not_started"
    observing = store.start(required_hours=24, now=started)
    assert observing["status"] == "observing"
    assert "duplicate_analysis_count" in observing["missing_metrics"]

    for minutes in range(0, 24 * 60 + 1, 5):
        store.record(_metrics(), now=started + timedelta(minutes=minutes))
    assert store.snapshot(now=started + timedelta(hours=23, minutes=59))["status"] == "observing"
    passed = store.snapshot(now=started + timedelta(hours=24))
    assert passed["status"] == "passed"


def test_observation_store_can_bind_a_72_hour_window_contract(tmp_path):
    store = ContinuousObservationStore(
        tmp_path / "observation-72.json", default_required_hours=72
    )

    assert store.snapshot()["required_hours"] == 72
    started = datetime(2026, 8, 29, tzinfo=UTC)
    snapshot = store.start(now=started, baseline_metrics=_metrics())

    assert snapshot["required_hours"] == 72
    assert snapshot["status"] == "observing"


def test_observation_blocks_on_failed_gate_and_never_fakes_zero(tmp_path):
    store = ContinuousObservationStore(tmp_path / "observation.json")
    started = datetime(2026, 8, 29, tzinfo=UTC)
    store.start(required_hours=72, now=started)
    store.record(
        _metrics(max_analysis_interval_seconds=240),
        now=started + timedelta(hours=72),
    )
    snapshot = store.snapshot(now=started + timedelta(hours=72))
    assert snapshot["status"] == "blocked"
    assert "max_analysis_interval_seconds" in snapshot["failed_metrics"]


def test_observation_blocks_on_sampling_gap_and_stale_latest_sample(tmp_path):
    store = ContinuousObservationStore(tmp_path / "observation.json")
    started = datetime(2026, 8, 29, tzinfo=UTC)
    store.start(required_hours=24, now=started, baseline_metrics=_metrics())
    store.record(_metrics(), now=started + timedelta(minutes=5))
    snapshot = store.record(_metrics(), now=started + timedelta(minutes=21))

    assert snapshot["status"] == "blocked"
    assert "sample_gap_exceeded" in snapshot["continuity_failures"]
    stale = store.snapshot(now=started + timedelta(minutes=40))
    assert "latest_sample_stale" in stale["continuity_failures"]


def test_collection_error_blocks_window_instead_of_observing_forever(tmp_path):
    store = ContinuousObservationStore(tmp_path / "observation.json")
    started = datetime(2026, 8, 29, tzinfo=UTC)
    store.start(required_hours=24, now=started, baseline_metrics=_metrics())

    snapshot = store.record(
        {"collection_errors": "model:timeout"},
        now=started + timedelta(minutes=5),
    )

    assert snapshot["status"] == "blocked"
    assert snapshot["blocked_reason"] == "collection_error:model:timeout"


def test_observation_metrics_ready_requires_complete_healthy_sample():
    assert observation_metrics_ready(_metrics()) is True
    assert observation_metrics_ready(_metrics(max_analysis_interval_seconds=None)) is False
    assert observation_metrics_ready(_metrics(training_state_clear=False)) is False
    assert observation_metrics_ready({**_metrics(), "collection_errors": "trade:timeout"}) is False


def test_observation_rejects_record_before_explicit_start(tmp_path):
    store = ContinuousObservationStore(tmp_path / "observation.json")
    try:
        store.record({}, now=datetime(2026, 8, 29, tzinfo=UTC))
    except RuntimeError as exc:
        assert str(exc) == "observation_window_not_started"
    else:
        raise AssertionError("record must require an explicit observation window")


def test_observation_statuses_are_valid_snapshot_states():
    assert normalize_status("passed") == "passed"
    assert normalize_status("observing") == "observing"
    assert normalize_status("not_started") == "not_started"


def test_runtime_model_states_are_not_misclassified_as_missing():
    for state in ("ready", "active", "live", "trained", "available", "configured"):
        assert normalize_status(state) == state
    status, degraded = status_from_sections(
        {
            "local_ml": {"status": "trained"},
            "local_ai_tools": {"status": "ready"},
        }
    )
    assert status == "ok"
    assert degraded == []


def test_observation_counter_metrics_are_relative_to_window_baseline(tmp_path):
    store = ContinuousObservationStore(tmp_path / "observation.json")
    started = datetime(2026, 8, 29, tzinfo=UTC)
    store.start(
        required_hours=24,
        now=started,
        baseline_metrics=_metrics(service_restart_count=3, dashboard_timeout_storm_count=2),
    )
    store.record(
        _metrics(service_restart_count=4, dashboard_timeout_storm_count=2),
        now=started + timedelta(minutes=5),
    )
    latest = store.snapshot(now=started + timedelta(minutes=5))["latest_metrics"]
    assert latest["service_restart_count"] == 1
    assert latest["dashboard_timeout_storm_count"] == 0


@pytest.mark.asyncio
async def test_scheduler_starts_both_windows_and_records_real_samples(tmp_path):
    metrics = _metrics()
    calls = 0

    async def collect():
        nonlocal calls
        calls += 1
        return metrics

    stores = {
        24: ContinuousObservationStore(tmp_path / "24.json"),
        72: ContinuousObservationStore(tmp_path / "72.json"),
    }
    scheduler = ContinuousObservationScheduler(
        stores,
        collect,
        interval_seconds=60,
        startup_delay_seconds=0,
    )
    await scheduler.start()
    try:
        snapshots = await scheduler.sample_once()
        assert set(snapshots) == {"24", "72"}
        assert calls >= 2  # one baseline plus one real sample
        assert stores[24].snapshot()["status"] == "observing"
        assert stores[24].snapshot()["sample_count"] == 1
        assert stores[72].snapshot()["sample_count"] == 1
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_does_not_recollect_baseline_for_existing_windows(tmp_path):
    calls = 0

    async def collect():
        nonlocal calls
        calls += 1
        return _metrics()

    stores = {
        24: ContinuousObservationStore(tmp_path / "24.json"),
        72: ContinuousObservationStore(tmp_path / "72.json"),
    }
    for hours, store in stores.items():
        store.start(required_hours=hours, baseline_metrics=_metrics())
    scheduler = ContinuousObservationScheduler(
        stores,
        collect,
        interval_seconds=60,
        startup_delay_seconds=60,
    )
    await scheduler.start()
    try:
        await asyncio.sleep(0)
        assert calls == 0
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_waits_for_healthy_baseline_then_starts_windows(tmp_path):
    samples = [
        {"collection_errors": "model:timeout"},
        _metrics(),
    ]

    async def collect():
        return samples.pop(0)

    stores = {
        24: ContinuousObservationStore(tmp_path / "24.json"),
        72: ContinuousObservationStore(tmp_path / "72.json"),
    }
    scheduler = ContinuousObservationScheduler(
        stores,
        collect,
        interval_seconds=60,
        startup_delay_seconds=60,
    )
    await scheduler.start()
    try:
        assert stores[24].snapshot()["status"] == "not_started"
        snapshots = await scheduler.sample_once()
        assert snapshots["24"]["status"] == "observing"
        assert snapshots["72"]["status"] == "observing"
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_recovers_blocked_window_with_fresh_full_window(tmp_path):
    metrics = _metrics()

    async def collect():
        return metrics

    store = ContinuousObservationStore(tmp_path / "24.json")
    started = datetime(2026, 8, 29, tzinfo=UTC)
    store.start(required_hours=24, now=started, baseline_metrics=metrics)
    store.record(
        {**metrics, "collection_errors": "analysis:timeout"},
        now=started + timedelta(minutes=5),
    )
    assert store.snapshot(now=started + timedelta(minutes=5))["status"] == "blocked"

    scheduler = ContinuousObservationScheduler(
        {24: store},
        collect,
        interval_seconds=60,
        startup_delay_seconds=60,
    )
    await scheduler.start()
    try:
        snapshot = (await scheduler.sample_once())["24"]
        assert snapshot["status"] == "observing"
        assert snapshot["reset_count"] == 1
        assert snapshot["last_reset_reason"] == "collection_error:analysis:timeout"
        assert snapshot["elapsed_hours"] < 0.01
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_persists_worker_liveness_and_sample_count(tmp_path):
    async def collect():
        return _metrics()

    stores = {
        24: ContinuousObservationStore(tmp_path / "24.json"),
        72: ContinuousObservationStore(tmp_path / "72.json"),
    }
    worker_path = tmp_path / "worker.json"
    scheduler = ContinuousObservationScheduler(
        stores,
        collect,
        interval_seconds=60,
        startup_delay_seconds=60,
        worker_state_path=worker_path,
    )
    await scheduler.start()
    try:
        running = ContinuousObservationWorkerState(worker_path).read()
        assert running["status"] == "running"
        assert running["last_heartbeat_at"]
        await scheduler.sample_once()
        sampled = ContinuousObservationWorkerState(worker_path).read()
        assert sampled["status"] == "running"
        assert sampled["sample_count"] == 1
        assert sampled["last_sample_at"]
    finally:
        await scheduler.stop()

    stopped = ContinuousObservationWorkerState(worker_path).read()
    assert stopped["status"] == "stopped"
    assert stopped["stopped_at"]
