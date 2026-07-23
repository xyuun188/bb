from datetime import UTC, datetime

import pytest

from services.training_epoch import (
    TRAINING_EPOCH_VERSION,
    load_training_epoch_start,
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
