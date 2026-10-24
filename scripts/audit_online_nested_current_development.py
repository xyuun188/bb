"""Run bounded nested development research on recent native fee-after rows."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import shlex
import sys
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

DEPENDENCIES = (
    "services/ml_nested_development.py",
    "services/ml_development_baselines.py",
    "services/ml_signal_service.py",
    "services/ml_training_dataset.py",
    "services/ml_training_contract.py",
    "services/training_data_quality.py",
)

REMOTE_SCRIPT = r'''
import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import func, select

from db.session import close_db, get_read_session_ctx
from models.learning import ShadowBacktest
from services.ml_development_baselines import _development_input_sha256
from services.ml_nested_development import build_nested_development_report
from services.ml_signal_service import (
    MIN_TRAINING_SAMPLE_COUNT,
    PRIMARY_PREDICTION_HORIZON_MINUTES,
    _decision_group_availability,
    build_training_frame,
    decision_group_partition,
)
from services.ml_training_dataset import shadow_training_columns, shadow_training_row_from_mapping

GROUP_LIMIT = __GROUP_LIMIT__
HOURS = __HOURS__
CUTOFF = datetime.fromisoformat(__CUTOFF__.replace("Z", "+00:00")).astimezone(UTC)
REVISION = __REVISION__
NATIVE_SOURCE = "shadow_entry_result_and_okx_native_1m_path"
DEPENDENCIES = (
    "services/ml_nested_development.py",
    "services/ml_development_baselines.py",
    "services/ml_signal_service.py",
    "services/ml_training_dataset.py",
    "services/ml_training_contract.py",
    "services/training_data_quality.py",
)


def _compact_evidence(evidence):
    keys = (
        "opportunity_count",
        "selected_count",
        "selected_decision_group_count",
        "avg_return_pct",
        "return_lcb_pct",
        "profit_factor",
        "average_return_per_opportunity_pct",
        "mean_selected_gross_return_pct",
        "mean_selected_execution_cost_pct",
    )
    return {key: evidence.get(key) for key in keys if key in evidence}


def _dependency_hashes():
    result = {}
    for name in DEPENDENCIES:
        digest = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        result[name] = digest
    return result


def _fold_coverage(frame, count):
    if frame.empty:
        return []
    primary = frame[
        frame["horizon_minutes"] == PRIMARY_PREDICTION_HORIZON_MINUTES
    ].copy()
    groups, bounds = _decision_group_availability(primary)
    initial = max(int(np.ceil(len(groups) * 0.5)), 1)
    result = []
    chunks = np.array_split(np.asarray(groups[initial:]), count)
    for fold_index, chunk in enumerate(chunks):
        validation_groups = chunk.tolist()
        if not validation_groups:
            result.append({
                "fold_index": fold_index,
                "status": "empty_validation_groups",
            })
            continue
        start = min(bounds[group]["decision_start"] for group in validation_groups)
        train_groups = {
            group for group in groups if bounds[group]["end"] < start
        }
        train = frame[
            frame["decision_group"].astype(str).isin(train_groups)
        ].copy()
        validation = frame[
            frame["decision_group"].astype(str).isin(validation_groups)
        ].copy()
        primary_train = train[
            train["horizon_minutes"] == PRIMARY_PREDICTION_HORIZON_MINUTES
        ]
        primary_validation = validation[
            validation["horizon_minutes"] == PRIMARY_PREDICTION_HORIZON_MINUTES
        ]
        inner_primary = train[
            train["horizon_minutes"] == PRIMARY_PREDICTION_HORIZON_MINUTES
        ]
        inner_groups, inner_bounds = _decision_group_availability(inner_primary)
        inner_initial = max(int(np.ceil(len(inner_groups) * 0.5)), 1)
        inner = []
        for inner_index, inner_chunk in enumerate(
            np.array_split(np.asarray(inner_groups[inner_initial:]), 2)
        ):
            inner_validation_groups = inner_chunk.tolist()
            if not inner_validation_groups:
                inner.append({
                    "fold_index": inner_index,
                    "primary_train_count": 0,
                    "primary_validation_count": 0,
                    "status": "empty_validation_groups",
                })
                continue
            inner_start = min(
                inner_bounds[group]["decision_start"]
                for group in inner_validation_groups
            )
            inner_train_groups = {
                group for group in inner_groups
                if inner_bounds[group]["end"] < inner_start
            }
            inner_train = train[
                train["decision_group"].astype(str).isin(inner_train_groups)
            ]
            inner_validation = train[
                train["decision_group"].astype(str).isin(inner_validation_groups)
            ]
            inner_train_count = int(
                (
                    inner_train["horizon_minutes"]
                    == PRIMARY_PREDICTION_HORIZON_MINUTES
                ).sum()
            )
            inner_validation_count = int(
                (
                    inner_validation["horizon_minutes"]
                    == PRIMARY_PREDICTION_HORIZON_MINUTES
                ).sum()
            )
            status = (
                "ready"
                if inner_train_count >= MIN_TRAINING_SAMPLE_COUNT
                and inner_validation_count > 0
                else "primary_horizon_coverage_insufficient"
            )
            inner.append({
                "fold_index": inner_index,
                "primary_train_count": inner_train_count,
                "primary_validation_count": inner_validation_count,
                "status": status,
            })
        status = (
            "ready"
            if int(len(primary_train)) >= MIN_TRAINING_SAMPLE_COUNT
            and not primary_validation.empty
            and all(item["status"] == "ready" for item in inner)
            else "primary_horizon_coverage_insufficient"
        )
        result.append({
            "fold_index": fold_index,
            "training_group_count": len(train_groups),
            "validation_group_count": len(validation_groups),
            "primary_train_count": int(len(primary_train)),
            "primary_validation_count": int(len(primary_validation)),
            "inner_folds": inner,
            "status": status,
        })
    return result


async def main():
    since = CUTOFF - timedelta(hours=HOURS)
    group_key = func.coalesce(
        func.nullif(ShadowBacktest.decision_id, 0),
        -ShadowBacktest.id,
    )
    filters = (
        ShadowBacktest.execution_mode == "paper",
        ShadowBacktest.status == "completed",
        ShadowBacktest.updated_at >= since,
        ShadowBacktest.updated_at < CUTOFF,
        ShadowBacktest.training_feature_snapshot[
            "training_market_fact_contract"
        ]["source"].as_string() == NATIVE_SOURCE,
        ShadowBacktest.training_feature_snapshot[
            "training_label_contract"
        ]["fee_after_complete"].as_boolean().is_(True),
    )
    selected_groups = (
        select(group_key.label("decision_group_key"))
        .where(*filters)
        .group_by(group_key)
        .order_by(func.max(ShadowBacktest.updated_at), group_key)
        .limit(GROUP_LIMIT)
        .subquery()
    )
    statement = (
        select(*shadow_training_columns())
        .where(
            *filters,
            group_key.in_(select(selected_groups.c.decision_group_key)),
        )
        .order_by(ShadowBacktest.updated_at, ShadowBacktest.id)
    )
    try:
        async with get_read_session_ctx(statement_timeout_ms=15000) as session:
            result = await session.execute(statement)
            rows = [
                shadow_training_row_from_mapping(row)
                for row in result.mappings()
            ]

        frame = build_training_frame(rows)
        nested = build_nested_development_report(frame)
        primary = frame[
            frame["horizon_minutes"] == PRIMARY_PREDICTION_HORIZON_MINUTES
        ].copy()
        partition = decision_group_partition(primary)
        legacy_partition = decision_group_partition(frame)
        coverage = _fold_coverage(partition.train, 3)
        groups = (
            sorted(
                frame["decision_group"].astype(str).unique().tolist(),
            )
            if not frame.empty
            else []
        )
        folds = []
        for index, fold in enumerate(nested.get("folds", [])):
            selection = fold.get("selection") or {}
            candidate_rows = selection.get("candidates") or []
            folds.append(
                {
                    "fold_index": index,
                    "train_primary_sample_count": fold.get(
                        "train_primary_sample_count"
                    ),
                    "validation_primary_sample_count": fold.get(
                        "validation_primary_sample_count"
                    ),
                    "decision_group_overlap_count": fold.get(
                        "decision_group_overlap_count"
                    ),
                    "chronological_label_disjoint": fold.get(
                        "chronological_label_disjoint"
                    ),
                    "selection_reason": selection.get("reason"),
                    "policy": selection.get("policy"),
                    "candidate_count": len(candidate_rows),
                    "eligible_candidate_count": sum(
                        row.get("inner_selection_eligible") is True
                        for row in candidate_rows
                    ),
                    "candidates": [
                        {
                            "alpha": row.get("alpha"),
                            "net_threshold_pct": row.get("net_threshold_pct"),
                            "inner_selection_eligible": row.get(
                                "inner_selection_eligible"
                            ),
                            "evidence": _compact_evidence(
                                row.get("evidence") or {}
                            ),
                        }
                        for row in candidate_rows
                    ],
                    "outer_evidence": _compact_evidence(
                        fold.get("outer_evidence") or {}
                    ),
                }
            )

        horizon_counts = Counter(
            str(int(value))
            for value in (
                frame["horizon_minutes"].tolist()
                if "horizon_minutes" in frame
                else []
            )
        )
        report = {
            "experiment": "bounded-native-fee-after-nested-development",
            "protocol_revision": REVISION,
            "generated_at": datetime.now(UTC).isoformat(),
            "window": {
                "since": since.isoformat(),
                "cutoff_exclusive": CUTOFF.isoformat(),
                "hours": HOURS,
                "decision_group_limit": GROUP_LIMIT,
                "source": NATIVE_SOURCE,
                "filter": "completed paper labels with fee_after_complete=true",
            },
            "input": {
                "source_row_count": len(rows),
                "training_frame_row_count": int(len(frame)),
                "decision_group_count": len(groups),
                "horizon_row_counts": dict(sorted(horizon_counts.items())),
                "development_input_sha256": nested.get(
                    "development_input_sha256"
                ),
                "legacy_all_horizon_development_sha256": _development_input_sha256(
                    legacy_partition.train
                ),
                "source_frame_sha256": _development_input_sha256(frame),
                "dependency_sha256": _dependency_hashes(),
                "outer_fold_coverage": coverage,
            },
            "research": {
                "status": nested.get("status"),
                "version": nested.get("version"),
                "fold_policy": "primary-horizon-label-time-only",
                "outer_partition": nested.get("outer_partition"),
                "search_space": nested.get("search_space"),
                "selected_policy_fold_count": nested.get(
                    "selected_policy_fold_count", 0
                ),
                "aggregate_outer_evidence": _compact_evidence(
                    nested.get("aggregate_outer_evidence") or {}
                ),
                "folds": folds,
            },
            "safety": {
                "exam_labels_evaluated": False,
                "artifact_persisted": False,
                "database_mutated": False,
                "promotion_authorized": False,
                "orders_submitted": False,
                "runtime_policy_changed": False,
                "output_is_development_only": True,
                "cpu_quota_percent": 25,
                "memory_limit_bytes": 1073741824,
                "runtime_limit_seconds": 300,
                "database_connection_limit": 1,
            },
        }
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, default=str))
    finally:
        await close_db()


asyncio.run(main())
'''


def _parse_cutoff(value: str) -> datetime:
    cutoff = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        raise ValueError("--cutoff must include a timezone")
    return cutoff.astimezone(UTC)


def _build_remote_command(
    limit: int,
    hours: int,
    cutoff: datetime,
    unit: str,
    revision: str = "v1",
) -> str:
    rendered = (
        REMOTE_SCRIPT.replace("__GROUP_LIMIT__", str(limit))
        .replace("__HOURS__", str(hours))
        .replace("__CUTOFF__", repr(cutoff.isoformat().replace("+00:00", "Z")))
        .replace("__REVISION__", repr(revision))
    )
    payload = base64.b64encode(rendered.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode({payload!r}))"
    return " ".join(
        (
            "systemd-run --quiet --wait --pipe --collect",
            f"--unit={shlex.quote(unit)}",
            "--property=WorkingDirectory=/data/bb/app",
            "--property=User=bb",
            "--property=CPUQuota=25%",
            "--property=MemoryMax=1G",
            "--property=RuntimeMaxSec=300s",
            "--property=KillMode=control-group",
            "--property=Nice=10",
            "--property=EnvironmentFile=-/data/bb/app/.env",
            "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
            "--setenv=OMP_NUM_THREADS=1",
            "--setenv=OPENBLAS_NUM_THREADS=1",
            "--setenv=MKL_NUM_THREADS=1",
            "--setenv=DATABASE_POOL_SIZE=1",
            "--setenv=DATABASE_MAX_OVERFLOW=0",
            "/data/bb/app/.venv/bin/python -c",
            shlex.quote(runner),
        )
    )


def _local_dependency_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in DEPENDENCIES
    }


def _parse_hashes(output: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in output.splitlines():
        digest, separator, name = line.strip().partition("  ")
        if separator and name:
            hashes[name] = digest
    return hashes


def _validate_expected_input_hash(
    report: dict[str, object], expected_hash: str | None
) -> None:
    if not expected_hash:
        return
    input_report = report.get("input")
    actual = (
        input_report.get("development_input_sha256")
        if isinstance(input_report, dict)
        else None
    )
    if actual != expected_hash:
        raise RuntimeError(
            "development input fingerprint changed; refusing same-window comparison"
        )


def _validate_expected_legacy_hash(
    report: dict[str, object], expected_hash: str | None
) -> None:
    if not expected_hash:
        return
    input_report = report.get("input")
    actual = (
        input_report.get("legacy_all_horizon_development_sha256")
        if isinstance(input_report, dict)
        else None
    )
    if actual != expected_hash:
        raise RuntimeError(
            "legacy development fingerprint changed; refusing same-window comparison"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum distinct decision groups to read.",
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--cutoff", help="Exclusive UTC/offset timestamp.")
    parser.add_argument(
        "--revision",
        choices=("v1", "v2", "v3"),
        default="v1",
    )
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--expected-legacy-sha256")
    args = parser.parse_args()
    limit = max(1, min(int(args.limit), 2500))
    hours = max(1, min(int(args.hours), 168))
    cutoff = _parse_cutoff(args.cutoff) if args.cutoff else datetime.now(UTC)
    unit = f"bb-nested-dev-{args.revision}-{cutoff.strftime('%Y%m%dT%H%M%SZ')}"
    result_dir = ROOT / "data" / "profitability_remediation"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / (
        f"online-nested-development-{args.revision}-"
        f"{cutoff.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {result_path}")

    expected_hashes = _local_dependency_hashes()
    ssh = connect_remote_ssh(ROOT, timeout=25)
    sftp = ssh.open_sftp()
    remote_result = (
        f"/data/bb/app/tmp/codex-nested-development/"
        f"result_{secrets.token_hex(12)}.json"
    )
    try:
        remote_hashes = run_remote_text(
            ssh,
            "cd /data/bb/app && sha256sum "
            + " ".join(shlex.quote(name) for name in DEPENDENCIES),
            timeout=30,
            max_output_chars=2000,
        )
        if _parse_hashes(remote_hashes) != expected_hashes:
            raise RuntimeError(
                "remote nested research dependencies do not match the local worktree"
            )
        unit_state = run_remote_text(
            ssh,
            f"systemctl show {shlex.quote(unit)} -p LoadState,ActiveState --value 2>/dev/null || true",
            timeout=20,
            max_output_chars=500,
        ).strip().splitlines()
        if any(
            value in {"active", "activating", "reloading", "deactivating"}
            for value in unit_state
        ):
            raise RuntimeError(f"matching remote research unit already exists: {unit}")
        command = (
            "set -e\numask 077\n"
            "install -d -m 0700 /data/bb/app/tmp/codex-nested-development\n"
            f"{_build_remote_command(limit, hours, cutoff, unit, args.revision)} "
            f"> {shlex.quote(remote_result)}"
        )
        run_remote_text(
            ssh,
            command,
            timeout=330,
            max_output_chars=3000,
        )
        with sftp.file(remote_result, "r") as remote_file:
            report = json.loads(remote_file.read().decode("utf-8"))
    finally:
        with suppress(OSError):
            sftp.remove(remote_result)
        sftp.close()
        ssh.close()

    if not isinstance(report, dict):
        raise ValueError("nested development audit output must be a JSON object")
    if report.get("safety", {}).get("exam_labels_evaluated") is not False:
        raise ValueError("development audit unexpectedly reports exam evaluation")
    _validate_expected_input_hash(report, args.expected_input_sha256)
    _validate_expected_legacy_hash(report, args.expected_legacy_sha256)
    remote_hashes = report.get("input", {}).get("dependency_sha256")
    if remote_hashes != expected_hashes:
        raise RuntimeError(
            "remote nested research dependencies do not match the local worktree"
        )
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    safe_print(json.dumps(report, ensure_ascii=False, indent=2))
    safe_print(f"saved_report={result_path}")


if __name__ == "__main__":
    main()
