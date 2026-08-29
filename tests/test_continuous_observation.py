from datetime import UTC, datetime, timedelta

from services.continuous_observation import ContinuousObservationStore
from services.observability_contract import normalize_status


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

    store.record(_metrics(), now=started + timedelta(hours=23, minutes=59))
    assert store.snapshot(now=started + timedelta(hours=23, minutes=59))["status"] == "observing"
    passed = store.snapshot(now=started + timedelta(hours=24))
    assert passed["status"] == "passed"


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
