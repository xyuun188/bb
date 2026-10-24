"""Run bounded horizon-aware development research on current native labels."""

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
    "services/ml_development_baselines.py",
    "services/ml_development_snapshot.py",
    "services/ml_horizon_development.py",
    "services/ml_nested_development.py",
    "services/ml_signal_service.py",
    "services/ml_training_contract.py",
    "services/ml_training_dataset.py",
    "services/training_data_quality.py",
)

REMOTE_SCRIPT = r"""
import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select

from db.session import close_db, get_read_session_ctx
from models.learning import ShadowBacktest
from services.ml_development_snapshot import make_development_snapshot, snapshot_frame
from services.ml_horizon_development import build_horizon_development_report
from services.ml_signal_service import build_training_frame
from services.ml_training_dataset import (
    shadow_training_columns,
    shadow_training_row_from_mapping,
)

GROUP_LIMIT = __GROUP_LIMIT__
HOURS = __HOURS__
CUTOFF = datetime.fromisoformat(__CUTOFF__.replace("Z", "+00:00")).astimezone(UTC)
REVISION = __REVISION__
NATIVE_SOURCE = "shadow_entry_result_and_okx_native_1m_path"
DEPENDENCIES = __DEPENDENCIES__


def _dependency_hashes():
    return {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in DEPENDENCIES
    }


def _compact_evidence(value):
    if not isinstance(value, dict):
        return {}
    keys = (
        "selected_count",
        "selected_decision_group_count",
        "avg_return_pct",
        "return_lcb_pct",
        "profit_factor",
        "average_return_per_opportunity_pct",
        "mean_selected_gross_return_pct",
        "mean_selected_execution_cost_pct",
    )
    return {key: value.get(key) for key in keys if key in value}


def _compact_fold(fold):
    selection = fold.get("selection") or {}
    candidates = selection.get("candidates") or []
    compact_candidates = []
    for candidate in candidates:
        compact_candidates.append(
            {
                "horizon_minutes": candidate.get("horizon_minutes"),
                "feature_policy": candidate.get("feature_policy"),
                "alpha": candidate.get("alpha"),
                "net_threshold_pct": candidate.get("net_threshold_pct"),
                "inner_selection_eligible": candidate.get("inner_selection_eligible"),
                "rejection_reasons": candidate.get("rejection_reasons"),
                "evidence": {
                    name: _compact_evidence(candidate.get("evidence", {}).get(name))
                    for name in ("all_opportunities", "nonoverlapping")
                },
            }
        )
    return {
        key: fold.get(key)
        for key in (
            "training_label_end",
            "validation_decision_start",
            "training_primary_sample_count",
            "validation_primary_sample_count",
            "decision_group_overlap_count",
            "chronological_label_disjoint",
        )
    } | {
        "selection_reason": selection.get("reason"),
        "selected_policy": selection.get("policy"),
        "candidate_count": len(compact_candidates),
        "eligible_candidate_count": sum(
            item.get("inner_selection_eligible") is True
            for item in candidates
        ),
        "candidate_rejection_counts": dict(
            Counter(
                reason
                for item in candidates
                for reason in (item.get("rejection_reasons") or [])
            )
        ),
        "candidates": compact_candidates,
        "outer_evidence": {
            name: _compact_evidence((fold.get("outer_evidence") or {}).get(name))
            for name in ("all_opportunities", "nonoverlapping")
        },
    }


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
        .where(*filters, group_key.in_(select(selected_groups.c.decision_group_key)))
        .order_by(ShadowBacktest.updated_at, ShadowBacktest.id)
    )
    try:
        async with get_read_session_ctx(
            statement_timeout_ms=15000,
            transaction_isolation="REPEATABLE READ READ ONLY",
        ) as session:
            result = await session.execute(statement)
            rows = [
                shadow_training_row_from_mapping(row)
                for row in result.mappings()
            ]
        frame = build_training_frame(rows)
        snapshot = make_development_snapshot(
            frame,
            cutoff=CUTOFF.isoformat(),
        )
        development = snapshot_frame(snapshot, cutoff=CUTOFF.isoformat())
        research = build_horizon_development_report(
            development,
            snapshot=snapshot,
        )
        horizon_counts = Counter(
            str(int(value))
            for value in development["horizon_minutes"].tolist()
        ) if not development.empty else Counter()
        complete_groups = (
            int(research.get("common_horizon_decision_group_count") or 0)
        )
        report = {
            "experiment": "bounded-native-fee-after-horizon-current-development",
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
                "development_row_count": int(len(development)),
                "decision_group_count": int(
                    development["decision_group"].nunique()
                ) if not development.empty else 0,
                "common_horizon_group_count": complete_groups,
                "horizon_row_counts": dict(sorted(horizon_counts.items())),
                "development_input_sha256": snapshot["development_input_sha256"],
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "dependency_sha256": _dependency_hashes(),
            },
            "research": {
                "status": research.get("status"),
                "version": research.get("version"),
                "search_space": research.get("search_space"),
                "selection_policy": research.get("selection_policy"),
                "selected_policy_fold_count": research.get(
                    "selected_policy_fold_count", 0
                ),
                "aggregate_outer_evidence": {
                    name: _compact_evidence(
                        (research.get("aggregate_outer_evidence") or {}).get(name)
                    )
                    for name in ("all_opportunities", "nonoverlapping")
                },
                "fixed_baselines_by_horizon": research.get(
                    "fixed_baselines_by_horizon"
                ),
                "folds": [_compact_fold(fold) for fold in research.get("folds", [])],
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
"""


def _local_dependency_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in DEPENDENCIES
    }


def _render_remote_script(
    *,
    limit: int,
    hours: int,
    cutoff: datetime,
    revision: str,
) -> str:
    return (
        REMOTE_SCRIPT.replace("__GROUP_LIMIT__", str(limit))
        .replace("__HOURS__", str(hours))
        .replace("__CUTOFF__", repr(cutoff.isoformat().replace("+00:00", "Z")))
        .replace("__REVISION__", repr(revision))
        .replace("__DEPENDENCIES__", repr(DEPENDENCIES))
    )


def _build_remote_command(
    *,
    limit: int,
    hours: int,
    cutoff: datetime,
    unit: str,
    revision: str,
) -> str:
    remote_script = _render_remote_script(
        limit=limit,
        hours=hours,
        cutoff=cutoff,
        revision=revision,
    )
    encoded = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode({encoded!r}))"
    return " ".join(
        (
            "systemd-run --quiet --wait --pipe --collect",
            f"--unit={shlex.quote(unit)}",
            "--property=WorkingDirectory=/data/bb/app",
            "--property=User=bb",
            "--property=EnvironmentFile=-/data/bb/app/.env",
            "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
            "--property=CPUQuota=25%",
            "--property=MemoryMax=1G",
            "--property=RuntimeMaxSec=300s",
            "--property=TimeoutStopSec=15s",
            "--property=KillMode=control-group",
            "--setenv=DATABASE_POOL_SIZE=1",
            "--setenv=DATABASE_MAX_OVERFLOW=0",
            "--setenv=OMP_NUM_THREADS=1",
            "--setenv=OPENBLAS_NUM_THREADS=1",
            "/data/bb/app/.venv/bin/python -c",
            shlex.quote(runner),
        )
    )


def _parse_cutoff(value: str | None) -> datetime:
    cutoff = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value
        else datetime.now(UTC)
    )
    if cutoff.tzinfo is None:
        raise ValueError("--cutoff must include a timezone")
    return cutoff.astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--cutoff")
    parser.add_argument("--revision", choices=("v1",), default="v1")
    args = parser.parse_args()
    limit = max(1, min(int(args.limit), 2500))
    hours = max(1, min(int(args.hours), 168))
    cutoff = _parse_cutoff(args.cutoff)
    unit = f"bb-horizon-dev-{args.revision}-{cutoff.strftime('%Y%m%dT%H%M%SZ')}"
    result_path = (
        ROOT
        / "data"
        / "profitability_remediation"
        / f"online-horizon-development-{args.revision}-"
        f"{cutoff.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {result_path}")

    expected_hashes = _local_dependency_hashes()
    ssh = connect_remote_ssh(ROOT, timeout=25)
    sftp = ssh.open_sftp()
    remote_result = (
        "/data/bb/app/tmp/codex-horizon-development/"
        f"result_{secrets.token_hex(12)}.json"
    )
    try:
        remote_hashes = run_remote_text(
            ssh,
            "cd /data/bb/app && sha256sum "
            + " ".join(shlex.quote(name) for name in DEPENDENCIES),
            timeout=30,
            max_output_chars=3000,
        )
        parsed = {}
        for line in remote_hashes.splitlines():
            pieces = line.split()
            if len(pieces) == 2:
                parsed[pieces[1]] = pieces[0]
        if parsed != expected_hashes:
            raise RuntimeError("remote horizon research dependencies do not match local")
        state = run_remote_text(
            ssh,
            f"systemctl show {shlex.quote(unit)} -p LoadState,ActiveState "
            "2>/dev/null || true",
            timeout=20,
            max_output_chars=500,
        ).strip().splitlines()
        if any(item in {"active", "activating", "reloading", "deactivating"} for item in state):
            raise RuntimeError(f"matching remote research unit already exists: {unit}")
        command = (
            "set -e\numask 077\n"
            "install -d -m 0700 /data/bb/app/tmp/codex-horizon-development\n"
            f"{_build_remote_command(limit=limit, hours=hours, cutoff=cutoff, unit=unit, revision=args.revision)} "
            f"> {shlex.quote(remote_result)}"
        )
        run_remote_text(ssh, command, timeout=330, max_output_chars=3000)
        with sftp.file(remote_result, "r") as remote_file:
            report = json.loads(remote_file.read().decode("utf-8"))
    finally:
        with suppress(OSError):
            sftp.remove(remote_result)
        sftp.close()
        ssh.close()

    if report.get("safety", {}).get("exam_labels_evaluated") is not False:
        raise RuntimeError("horizon development unexpectedly evaluated exam labels")
    if report.get("input", {}).get("dependency_sha256") != expected_hashes:
        raise RuntimeError("remote horizon report dependency identity mismatch")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    safe_print(json.dumps(report, ensure_ascii=False, indent=2))
    safe_print(f"saved_report={result_path}")


if __name__ == "__main__":
    main()
