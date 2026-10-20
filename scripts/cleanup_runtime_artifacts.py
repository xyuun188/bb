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
REPORT_ARCHIVES = {
    "phase3_paper_resume_observation_reports": "phase3-paper-resume-observation-*.json",
    "phase3_paper_resume_preflight_reports": "phase3-paper-resume-preflight-*.json",
    "phase3_stage_handoff_reports": "phase3-stage-handoff-*.json",
    "phase3_model_server_readiness_reports": "phase3-model-server-readiness-*.json",
    "phase3_rebuild_preflight_reports": "phase3-rebuild-preflight-*.json",
    "okx_daily_reconciliation_reports": "okx-daily-reconciliation-*.json",
    "runtime_data_retention_reports": "runtime-data-retention-*.json",
}
REPORT_RETENTION_HOURS = 30 * 24
MINIMUM_REPORT_ARCHIVES = 100


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


def _referenced_report_paths(data_dir: Path) -> set[Path] | None:
    """Keep report evidence reachable from current manifests and latest cards."""
    pending = [
        *data_dir.glob("*.json"),
        *(data_dir / name / "latest.json" for name in REPORT_ARCHIVES),
    ]
    visited: set[Path] = set()
    protected: set[Path] = set()
    while pending:
        manifest = pending.pop()
        if manifest in visited or not manifest.is_file():
            continue
        visited.add(manifest)
        if manifest.is_symlink() or not _is_inside(manifest, data_dir):
            return None
        try:
            if manifest.stat().st_size > 8 * 1024 * 1024:
                return None
            values = [json.loads(manifest.read_text(encoding="utf-8-sig"))]
        except (OSError, ValueError):
            return None
        while values:
            value = values.pop()
            if isinstance(value, dict):
                values.extend(value.values())
            elif isinstance(value, list):
                values.extend(value)
            elif isinstance(value, str) and value.endswith(".json"):
                path = Path(value)
                if not path.is_absolute():
                    path = data_dir.parent / path
                if _is_inside(path, data_dir) and path.is_file():
                    resolved = path.resolve()
                    if resolved not in protected:
                        protected.add(resolved)
                        pending.append(resolved)
    return protected


def _report_archive_candidates(
    data_dir: Path, *, now: datetime, min_age_hours: float,
) -> list[CleanupCandidate]:
    protected = _referenced_report_paths(data_dir)
    if protected is None:
        return []
    candidates = []
    for name, pattern in REPORT_ARCHIVES.items():
        directory = data_dir / name
        if directory.is_symlink() or not directory.is_dir():
            continue
        reports = sorted(
            (path for path in directory.glob(pattern) if _is_safe_file(path, data_dir)),
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
        for path in reports[MINIMUM_REPORT_ARCHIVES:]:
            if path.resolve() in protected:
                continue
            item = _candidate(
                path,
                data_dir=data_dir,
                now=now,
                min_age_hours=max(min_age_hours, REPORT_RETENTION_HOURS),
                reason="expired_generated_report",
            )
            if item is not None:
                candidates.append(item)
    return candidates


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

    candidates.extend(
        _report_archive_candidates(
            resolved_data_dir, now=current_time, min_age_hours=min_age_hours,
        )
    )
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
            if path.stat().st_size != item.size_bytes or _age_hours(
                path, (now or datetime.now(UTC)).astimezone(UTC),
            ) < max(
                min_age_hours,
                REPORT_RETENTION_HOURS if item.reason == "expired_generated_report" else 0,
            ):
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
