"""Read-only audit of frozen development identities, timing and source evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shlex
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sqlalchemy import select  # noqa: E402

from db.session import close_db, get_read_session_ctx  # noqa: E402
from models.learning import ShadowBacktest  # noqa: E402
from services.execution_cost_model import execution_cost_provenance_blocker  # noqa: E402
from services.ml_development_snapshot import load_development_snapshot  # noqa: E402
from services.ml_nested_development import _folds  # noqa: E402
from services.ml_signal_service import FEATURE_KEYS, build_training_frame  # noqa: E402
from services.ml_training_dataset import (  # noqa: E402
    shadow_quality_sample,
    shadow_training_columns,
    shadow_training_row_from_mapping,
)
from services.training_data_quality import assess_shadow_sample  # noqa: E402

CUTOFF = "2026-09-26T22:11:51.501105+00:00"
SNAPSHOT = Path("data/strategy_replay/frozen_development_v1.json")
EXPECTED_SNAPSHOT_SHA256 = "afced19c8876ec2fc8237b298f22248d326160e6bdc83a754df1bc431cc9cec3"
AUDIT_DEPENDENCIES = (
    "scripts/audit_development_evidence.py",
    "services/ml_development_snapshot.py",
    "services/ml_nested_development.py",
    "services/ml_signal_service.py",
    "services/ml_training_dataset.py",
    "services/training_data_quality.py",
    "services/profit_supervision.py",
    "services/execution_cost_model.py",
    "services/historical_shadow_contract.py",
    "core/training_contracts.py",
    "core/market_facts.py",
    "models/learning.py",
)
COMPARE_NUMERIC = list(dict.fromkeys([
    *FEATURE_KEYS, "long_return_pct", "short_return_pct",
    "long_execution_cost_pct", "short_execution_cost_pct",
]))


def audit_code_identity() -> dict[str, Any]:
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in AUDIT_DEPENDENCIES
    }
    return {
        "dependency_sha256": hashes,
        "sha256": hashlib.sha256(json.dumps(
            hashes, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
    }


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


def inspect_source_batch(
    frozen: pd.DataFrame, rows: list[Any],
) -> tuple[
    Counter, list[tuple[int, str]], list[float], list[float], dict[int, pd.Timestamp],
]:
    """Compare exact IDs, never replace a frozen row with a newer observation."""
    issues: Counter = Counter()
    source = {row.id: row for row in rows}
    if len(source) != len(rows):
        raise ValueError("duplicate source identity")
    rebuilt = build_training_frame(rows)
    rebuilt_by_id = {
        int(row["id"]): row for row in rebuilt.to_dict("records")
    } if not rebuilt.empty else {}
    identities = []
    feature_offsets = []
    creation_offsets = []
    schedule_starts = {}
    for expected in frozen.to_dict("records"):
        sample_id = int(expected["id"])
        row = source.get(sample_id)
        if row is None:
            issues["source_row_missing"] += 1
            continue
        features = row.feature_snapshot
        label_contract = features.get("training_label_contract") or {}
        identities.append((sample_id, str(
            label_contract.get("compact_fingerprint") or label_contract.get("label_fingerprint")
            or ""
        )))
        assessment = assess_shadow_sample(shadow_quality_sample(row))
        if assessment.exclude_from_training:
            issues["source_quality_rejected"] += 1
            for reason in assessment.reasons:
                issues[f"quality:{reason}"] += 1
        created = _timestamp(row.created_at)
        due = _timestamp(row.due_at)
        if created != _timestamp(expected["decision_timestamp"]):
            issues["source_decision_time_mismatch"] += 1
        if due != _timestamp(expected["label_timestamp"]):
            issues["source_label_time_mismatch"] += 1
        if row.horizon_minutes != expected["horizon_minutes"]:
            issues["source_horizon_mismatch"] += 1
        feature_at = _timestamp(features.get("timestamp"))
        # The frozen rows mix legacy insertion clocks and historical flooring.
        # Current deployed code cannot establish their original decision clock.
        issues["decision_time_provenance_unresolved"] += 1
        if created is None or due is None or feature_at is None:
            issues["source_timing_missing"] += 1
        else:
            schedule_start = due - pd.Timedelta(minutes=row.horizon_minutes)
            schedule_starts[sample_id] = schedule_start
            feature_offsets.append((schedule_start - feature_at).total_seconds())
            creation_offsets.append((created - schedule_start).total_seconds())
        if int(label_contract.get("horizon_minutes") or 0) != row.horizon_minutes:
            issues["label_contract_horizon_mismatch"] += 1
        if int(label_contract.get("shadow_backtest_id") or 0) != sample_id:
            issues["label_contract_source_id_mismatch"] += 1
        if _timestamp(label_contract.get("label_timestamp")) != due:
            issues["label_contract_timestamp_mismatch"] += 1
        if feature_at is None:
            issues["feature_observation_time_missing"] += 1
        cost_blocker = execution_cost_provenance_blocker(features)
        if cost_blocker:
            issues[f"cost_provenance:{cost_blocker}"] += 1
        if features.get("historical_market_path_only") is True:
            issues["historical_bar_close_availability_requires_review"] += 1
        actual = rebuilt_by_id.get(sample_id)
        if actual is None:
            issues["source_not_in_current_cost_complete_frame"] += 1
            continue
        if str(actual["decision_group"]) != str(expected["decision_group"]):
            issues["source_decision_group_mismatch"] += 1
        for column in COMPARE_NUMERIC:
            observed, original = float(actual[column]), float(expected[column])
            if not math.isfinite(observed) or not math.isclose(
                observed, original, rel_tol=1e-10, abs_tol=1e-10
            ):
                issues[f"source_value_mismatch:{column}"] += 1
    return issues, identities, feature_offsets, creation_offsets, schedule_starts


def inspect_snapshot(frame: pd.DataFrame, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Integrity is enforced by load_development_snapshot before this inspection."""
    delay_seconds = (
        frame["label_timestamp"] - frame["decision_timestamp"]
        - pd.to_timedelta(frame["horizon_minutes"], unit="m")
    ).dt.total_seconds().abs()
    folds = _folds(frame, 3)
    return {
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "development_input_sha256": snapshot["development_input_sha256"],
        "cutoff": snapshot["development_cutoff"],
        "row_count": len(frame),
        "decision_group_count": int(frame["decision_group"].nunique()),
        "all_labels_strictly_before_cutoff": bool(
            (frame["label_timestamp"] < pd.Timestamp(CUTOFF)).all()
        ),
        "frozen_decision_time_source": "stored_shadow_created_at_semantics_unverified",
        "frozen_database_creation_label_offset_seconds": {
            "count": int(len(delay_seconds)),
            "p50_absolute_deviation_seconds": float(delay_seconds.quantile(0.5)),
            "p95_absolute_deviation_seconds": float(delay_seconds.quantile(0.95)),
            "over_5_seconds_count": int((delay_seconds > 5).sum()),
            "due_minus_horizon_is_verified_decision_time": False,
        },
        "horizon_counts": {str(key): int(value) for key, value in
                           frame["horizon_minutes"].value_counts().sort_index().items()},
        "zero_weight_count": int((frame["sample_weight"] == 0).sum()),
        "purged_fold_count": len(folds),
        "purged_fold_boundaries": [boundary for _, _, boundary in folds],
        "sample_weight_policy": "frozen_values_integrity_checked_not_reestimated_by_page",
        "feature_component_availability": "requires_separate_component_lineage_review",
    }


async def audit(*, max_rows: int) -> dict[str, Any]:
    frame, snapshot = load_development_snapshot(SNAPSHOT, cutoff=CUTOFF)
    if snapshot["snapshot_sha256"] != EXPECTED_SNAPSHOT_SHA256:
        raise ValueError("unexpected frozen snapshot identity")
    checked = frame.iloc[:max_rows]
    issues: Counter = Counter()
    identities = []
    feature_ages: list[float] = []
    insert_delays: list[float] = []
    decision_anchors: dict[int, pd.Timestamp] = {}
    async with get_read_session_ctx(
        statement_timeout_ms=15_000,
        transaction_isolation="REPEATABLE READ READ ONLY",
    ) as session:
        for offset in range(0, len(checked), 200):
            page = checked.iloc[offset:offset + 200]
            result = await session.execute(
                select(*shadow_training_columns()).where(ShadowBacktest.id.in_(
                    page["id"].astype(int).tolist()
                ))
            )
            rows = [shadow_training_row_from_mapping(row) for row in result.mappings()]
            (
                page_issues, page_ids, page_ages, page_inserts, page_anchors
            ) = inspect_source_batch(page, rows)
            issues.update(page_issues)
            identities.extend(page_ids)
            feature_ages.extend(page_ages)
            insert_delays.extend(page_inserts)
            decision_anchors.update(page_anchors)
    fingerprint = hashlib.sha256(json.dumps(
        sorted(identities), separators=(",", ":"),
    ).encode()).hexdigest()
    future_observation_offsets = [-value for value in feature_ages if value < 0]
    nonnegative_feature_ages = [value for value in feature_ages if value >= 0]
    return {
        "audit_version": "development-evidence.v2",
        "audit_code_identity": audit_code_identity(),
        "audit_source_sha256": hashlib.sha256(
            await asyncio.to_thread(Path(__file__).read_bytes)
        ).hexdigest(),
        "generated_at": datetime.now(UTC).isoformat(),
        "read_only": True, "exam_labels_read": False,
        "production_permission": False,
        "snapshot": inspect_snapshot(frame, snapshot),
        "source_rows_requested": len(checked),
        "source_row_budget": max_rows,
        "source_rows_found": len(identities),
        "source_coverage_complete": len(checked) == len(frame) == len(identities),
        "schedule_start_minus_feature_observation_seconds": {
            "count": len(feature_ages),
            "negative_offset_count": len(future_observation_offsets),
            "maximum_negative_offset_magnitude_seconds": max(
                future_observation_offsets, default=0.0
            ),
            "p95_negative_offset_magnitude_seconds": (
                float(pd.Series(future_observation_offsets).quantile(0.95))
                if future_observation_offsets else 0.0
            ),
            "over_5_seconds_count": sum(value > 5 for value in nonnegative_feature_ages),
            "maximum_nonnegative_age_seconds": max(
                nonnegative_feature_ages, default=0.0
            ),
            "p50_nonnegative_age_seconds": (
                float(pd.Series(nonnegative_feature_ages).quantile(0.5))
                if nonnegative_feature_ages else 0.0
            ),
            "p95_nonnegative_age_seconds": (
                float(pd.Series(nonnegative_feature_ages).quantile(0.95))
                if nonnegative_feature_ages else 0.0
            ),
            "not_equivalent_to_label_horizon_deviation": True,
            "establishes_future_feature_leakage": False,
        },
        "stored_creation_minus_schedule_start_seconds": {
            "count": len(insert_delays),
            "negative_count": sum(value < 0 for value in insert_delays),
            "over_5_seconds_count": sum(value > 5 for value in insert_delays),
            "maximum_seconds": max(insert_delays, default=0.0),
            "p50_seconds": (
                float(pd.Series(insert_delays).quantile(0.5)) if insert_delays else 0.0
            ),
            "p95_seconds": (
                float(pd.Series(insert_delays).quantile(0.95)) if insert_delays else 0.0
            ),
        },
        "source_issue_counts": dict(sorted(issues.items())),
        "source_label_identity_sha256": fingerprint,
        "source_schedule_start_row_count": len(decision_anchors),
        "source_schedule_start_policy": "due_at_minus_horizon_is_not_verified_decision_time",
        "verified_decision_anchor_row_count": 0,
        "historical_bar_semantics": (
            "historical_builder_uses_close_but_source_timestamp_is_1m_bar_open;"
            "close_availability_is_at_least_bar_open_plus_1m"
        ),
        "frozen_creation_time_purged_fold_count": inspect_snapshot(
            frame, snapshot
        )["purged_fold_count"],
        "g2_acceptance": "pending_component_lineage_review" if not issues else "blocked_by_evidence",
        "profitability_acceptance": "not_evaluated",
    }


def audit_online(*, max_rows: int) -> dict[str, Any]:
    from core.remote_ssh import connect_remote_ssh, run_remote_text

    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    code_identity = audit_code_identity()
    identity = f"frozen-v1-source-r{max_rows}-{code_identity['sha256'][:12]}"
    remote_output = f"/data/bb/app/data/profitability_remediation/{identity}.json"
    command = [
        "systemd-run", "--quiet", "--wait", "--pipe", "--collect",
        f"--unit=bb-{identity}", "--property=WorkingDirectory=/data/bb/app",
        "--property=User=bb", "--property=EnvironmentFile=-/data/bb/app/.env",
        "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
        "--property=CPUQuota=100%", "--property=MemoryMax=1G",
        "--property=RuntimeMaxSec=300", "--property=TimeoutStopSec=10",
        "--property=KillMode=control-group", "--property=Nice=10",
        "--setenv=DATABASE_POOL_SIZE=1", "--setenv=DATABASE_MAX_OVERFLOW=0",
        "--setenv=OMP_NUM_THREADS=1", "--setenv=OPENBLAS_NUM_THREADS=1",
        "/data/bb/app/.venv/bin/python", "scripts/audit_development_evidence.py",
        "--max-rows", str(max_rows), "--output", remote_output,
    ]
    runner = """
import pathlib, subprocess, sys
p = pathlib.Path(sys.argv[1])
unit = sys.argv[2]
if not p.exists():
    active = subprocess.run(
        ['systemctl', 'is-active', '--quiet', unit], capture_output=True,
    )
    if active.returncode == 0:
        raise RuntimeError('audit already active; inspect named unit before resuming')
    result = subprocess.run(sys.argv[3:], capture_output=True)
    if result.returncode:
        raise RuntimeError('source audit failed; inspect named unit')
print(p.read_text())
"""
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh, shlex.join([
                "/data/bb/app/.venv/bin/python", "-c", runner,
                remote_output, f"bb-{identity}", *command,
            ]),
            timeout=330, max_output_chars=60_000,
        )
    finally:
        ssh.close()
    report = json.loads(output)
    if (
        report.get("audit_source_sha256") != source_hash
        or report.get("audit_code_identity") != code_identity
        or report.get("source_row_budget") != max_rows
    ):
        raise ValueError("online source audit code or budget mismatch")
    return report | {
        "remote_report_path": remote_output, "remote_unit": f"bb-{identity}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, choices=range(1, 20_001), default=10_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    async def run() -> dict[str, Any]:
        try:
            return await asyncio.wait_for(audit(max_rows=args.max_rows), timeout=270)
        finally:
            await close_db()

    report = audit_online(max_rows=args.max_rows) if args.online else asyncio.run(run())
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")  # noqa: ASYNC240
    print(encoded)


if __name__ == "__main__":
    main()
