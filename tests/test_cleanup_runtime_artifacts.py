from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.cleanup_runtime_artifacts import cleanup_runtime_artifacts, plan_cleanup

NOW = datetime(2026, 6, 8, 2, 0, tzinfo=UTC)


def _write_file(path: Path, size: int = 1, *, age_hours: float = 24.0) -> None:
    path.write_bytes(b"x" * size)
    timestamp = (NOW - timedelta(hours=age_hours)).timestamp()
    os.utime(path, (timestamp, timestamp))


def test_plan_cleanup_keeps_current_database_and_newest_backup(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    current_db = data_dir / "trading.db"
    newest_backup = data_dir / "trading.backup_before_expert_memory_cleanup_20260606_143133.db"
    old_backup = data_dir / "trading.backup_before_expert_memory_cleanup_20260606_140925.db"
    rotated_log = data_dir / "trading.log.1"
    codex_log = data_dir / "codex_run_stdout.log"
    active_log = data_dir / "trading.log"
    lock_file = data_dir / "paper_trading.lock"

    _write_file(current_db, 100, age_hours=48)
    _write_file(newest_backup, 90, age_hours=7)
    _write_file(old_backup, 80, age_hours=8)
    _write_file(rotated_log, 70, age_hours=8)
    _write_file(codex_log, 60, age_hours=8)
    _write_file(active_log, 50, age_hours=8)
    _write_file(lock_file, 1, age_hours=8)

    candidates = plan_cleanup(
        data_dir,
        keep_db_backups=1,
        min_age_hours=6,
        now=NOW,
    )

    names = {Path(item.path).name for item in candidates}
    assert names == {
        old_backup.name,
        rotated_log.name,
        codex_log.name,
    }
    assert current_db.name not in names
    assert newest_backup.name not in names
    assert active_log.name not in names
    assert lock_file.name not in names


def test_cleanup_runtime_artifacts_apply_deletes_only_candidates(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    keep = data_dir / "trading.backup_before_manual_20260606_150000.db"
    delete = data_dir / "trading.backup_before_manual_20260606_140000.db"
    active_db = data_dir / "trading.db"
    rotated_log = data_dir / "trading.log.2"

    _write_file(keep, 100, age_hours=8)
    _write_file(delete, 90, age_hours=8)
    _write_file(active_db, 80, age_hours=8)
    _write_file(rotated_log, 70, age_hours=8)

    result = cleanup_runtime_artifacts(
        data_dir,
        apply=True,
        keep_db_backups=1,
        min_age_hours=6,
        now=NOW,
    )

    assert result.dry_run is False
    assert result.deleted_count == 2
    assert result.deleted_bytes == 160
    assert keep.exists()
    assert active_db.exists()
    assert not delete.exists()
    assert not rotated_log.exists()
