from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DATA_DIR = Path("data")
PROTECTED_FILENAMES = {
    "paper_trading.lock",
    "position_profit_peaks.json",
    "trading.db",
    "trading.db-journal",
    "trading.db-shm",
    "trading.db-wal",
}
ROTATED_TRADING_LOG_RE = re.compile(r"^trading\.log\.\d+$")
CODEX_RUN_LOG_RE = re.compile(r"^codex_run_(stdout|stderr)\.log$")


@dataclass(frozen=True, slots=True)
class CleanupCandidate:
    path: str
    reason: str
    size_bytes: int
    age_hours: float


@dataclass(frozen=True, slots=True)
class CleanupResult:
    dry_run: bool
    deleted_count: int
    deleted_bytes: int
    candidates: list[CleanupCandidate]


def _utc_from_timestamp(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=UTC)


def _resolve_data_dir(data_dir: Path) -> Path:
    resolved = data_dir.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"data directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"data path is not a directory: {resolved}")
    return resolved


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _age_hours(path: Path, now: datetime) -> float:
    modified_at = _utc_from_timestamp(path.stat().st_mtime)
    return max((now - modified_at).total_seconds() / 3600.0, 0.0)


def _is_safe_file(path: Path, data_dir: Path) -> bool:
    if path.name in PROTECTED_FILENAMES:
        return False
    if not _is_inside(path, data_dir):
        return False
    if path.is_symlink() or not path.is_file():
        return False
    return True


def _candidate(
    path: Path,
    *,
    data_dir: Path,
    now: datetime,
    min_age_hours: float,
    reason: str,
) -> CleanupCandidate | None:
    if not _is_safe_file(path, data_dir):
        return None
    age = _age_hours(path, now)
    if age < min_age_hours:
        return None
    return CleanupCandidate(
        path=str(path),
        reason=reason,
        size_bytes=path.stat().st_size,
        age_hours=round(age, 2),
    )


def plan_cleanup(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    keep_db_backups: int = 1,
    min_age_hours: float = 6.0,
    now: datetime | None = None,
) -> list[CleanupCandidate]:
    """Return deletable runtime artifacts without deleting anything."""
    if keep_db_backups < 0:
        raise ValueError("keep_db_backups must be >= 0")
    if min_age_hours < 0:
        raise ValueError("min_age_hours must be >= 0")

    resolved_data_dir = _resolve_data_dir(data_dir)
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    candidates: list[CleanupCandidate] = []

    db_backups = sorted(
        (
            path
            for path in resolved_data_dir.glob("*.backup_before_*.db")
            if _is_safe_file(path, resolved_data_dir)
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for path in db_backups[keep_db_backups:]:
        item = _candidate(
            path,
            data_dir=resolved_data_dir,
            now=current_time,
            min_age_hours=min_age_hours,
            reason="old_database_backup",
        )
        if item is not None:
            candidates.append(item)

    for path in resolved_data_dir.iterdir():
        if not _is_safe_file(path, resolved_data_dir):
            continue
        reason = ""
        if ROTATED_TRADING_LOG_RE.match(path.name):
            reason = "rotated_trading_log"
        elif CODEX_RUN_LOG_RE.match(path.name):
            reason = "codex_run_log"
        if not reason:
            continue
        item = _candidate(
            path,
            data_dir=resolved_data_dir,
            now=current_time,
            min_age_hours=min_age_hours,
            reason=reason,
        )
        if item is not None:
            candidates.append(item)

    return sorted(candidates, key=lambda item: item.size_bytes, reverse=True)


def cleanup_runtime_artifacts(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    apply: bool = False,
    keep_db_backups: int = 1,
    min_age_hours: float = 6.0,
    now: datetime | None = None,
) -> CleanupResult:
    resolved_data_dir = _resolve_data_dir(data_dir)
    candidates = plan_cleanup(
        resolved_data_dir,
        keep_db_backups=keep_db_backups,
        min_age_hours=min_age_hours,
        now=now,
    )
    deleted_count = 0
    deleted_bytes = 0
    if apply:
        for item in candidates:
            path = Path(item.path)
            if not _is_safe_file(path, resolved_data_dir):
                continue
            os.remove(path)
            deleted_count += 1
            deleted_bytes += item.size_bytes
    return CleanupResult(
        dry_run=not apply,
        deleted_count=deleted_count,
        deleted_bytes=deleted_bytes,
        candidates=candidates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or delete safe runtime artifacts from the local data directory."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--apply", action="store_true", help="Delete planned files.")
    parser.add_argument(
        "--keep-db-backups",
        type=int,
        default=1,
        help="Number of newest *.backup_before_*.db files to keep.",
    )
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=6.0,
        help="Only select files at least this old.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    result = cleanup_runtime_artifacts(
        Path(args.data_dir),
        apply=bool(args.apply),
        keep_db_backups=int(args.keep_db_backups),
        min_age_hours=float(args.min_age_hours),
    )
    if args.json:
        print(
            json.dumps(
                {
                    **asdict(result),
                    "candidate_count": len(result.candidates),
                    "candidate_bytes": sum(item.size_bytes for item in result.candidates),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    total_bytes = sum(item.size_bytes for item in result.candidates)
    print(f"dry_run={str(result.dry_run).lower()}")
    print(f"candidate_count={len(result.candidates)}")
    print(f"candidate_bytes={total_bytes}")
    print(f"deleted_count={result.deleted_count}")
    print(f"deleted_bytes={result.deleted_bytes}")
    for item in result.candidates:
        print(f"{item.reason}\t{item.size_bytes}\t{item.age_hours}h\t{item.path}")


if __name__ == "__main__":
    main()
