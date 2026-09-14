import json
from datetime import UTC, datetime

import pytest

from services.training_epoch import (
    TRAINING_DATA_MIGRATION_VERSION,
    TRAINING_EPOCH_VERSION,
    load_training_data_migration,
    load_training_data_start,
    load_training_epoch_start,
    training_data_scope,
    write_training_data_migration,
    write_training_epoch,
)


def test_training_epoch_missing_marker_fails_closed(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="marker is missing"):
        load_training_epoch_start(tmp_path / "missing.json")


def test_training_epoch_round_trips_and_rejects_old_contract(tmp_path) -> None:
    path = tmp_path / "training_epoch.json"
    started_at = datetime(2026, 7, 24, 1, 2, 3, tzinfo=UTC)
    payload = write_training_epoch(path, started_at=started_at, reset_id="reset-1")

    assert payload["version"] == TRAINING_EPOCH_VERSION
    assert load_training_epoch_start(path) == started_at
    path.write_text('{"version":"old"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported"):
        load_training_epoch_start(path)


def test_approved_historical_migration_broadens_only_matching_epoch(tmp_path) -> None:
    epoch_path = tmp_path / "training_epoch.json"
    migration_path = tmp_path / "training_data_migration.json"
    epoch_start = datetime(2026, 9, 14, tzinfo=UTC)
    data_start = datetime(2026, 7, 27, tzinfo=UTC)
    write_training_epoch(epoch_path, started_at=epoch_start, reset_id="reset-1")

    write_training_data_migration(
        {
            "training_data_started_at": data_start.isoformat(),
            "source_fact_fingerprint": "a" * 64,
            "quality_report_sha256": "b" * 64,
            "approved_sample_counts": {"authoritative_trade": 42},
            "approved_sample_count_total": 42,
        },
        migration_path,
        epoch_path=epoch_path,
    )

    assert load_training_data_start(
        epoch_path=epoch_path, migration_path=migration_path
    ) == data_start
    scope = training_data_scope(epoch_path=epoch_path, migration_path=migration_path)
    assert scope["pre_epoch_data_training_allowed"] is True
    assert scope["historical_migration_status"] == "approved"
    assert scope["approved_sample_count_total"] == 42


def test_stale_historical_migration_fails_closed(tmp_path) -> None:
    epoch_path = tmp_path / "training_epoch.json"
    migration_path = tmp_path / "training_data_migration.json"
    epoch_start = datetime(2026, 9, 14, tzinfo=UTC)
    write_training_epoch(epoch_path, started_at=epoch_start, reset_id="reset-2")
    migration_path.write_text(
        json.dumps(
            {
                "version": TRAINING_DATA_MIGRATION_VERSION,
                "reset_id": "reset-1",
                "status": "approved",
                "training_data_started_at": "2026-07-27T00:00:00+00:00",
                "source_fact_fingerprint": "a" * 64,
                "quality_report_sha256": "b" * 64,
                "approved_sample_counts": {"authoritative_trade": 42},
                "approved_sample_count_total": 42,
                "live_routing_enabled": False,
            }
        ),
        encoding="utf-8",
    )

    assert load_training_data_migration(
        migration_path, epoch_path=epoch_path
    ) is None
    assert load_training_data_start(
        epoch_path=epoch_path, migration_path=migration_path
    ) == epoch_start
