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

from sqlalchemy import Integer, cast, extract, func, select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from models.learning import ShadowBacktest  # noqa: E402
from services.ml_training_dataset import shadow_training_columns  # noqa: E402

DEPENDENCIES = (
    "scripts/audit_online_horizon_current_development.py",
    "services/ml_development_baselines.py",
    "services/ml_development_snapshot.py",
    "services/ml_horizon_development.py",
    "services/liquidity_regime_development.py",
    "services/ml_nested_development.py",
    "services/ml_signal_service.py",
    "services/ml_training_contract.py",
    "services/ml_training_dataset.py",
    "services/training_data_quality.py",
)

NATIVE_SOURCE = "shadow_entry_result_and_okx_native_1m_path"


def _build_development_statement(
    *, limit: int, since: datetime, cutoff: datetime, revision: str,
):
    group_key = func.coalesce(
        func.nullif(ShadowBacktest.decision_id, 0), -ShadowBacktest.id,
    )
    quality_filters = (
        ShadowBacktest.training_feature_snapshot[
            "training_market_fact_contract"
        ]["source"].as_string() == NATIVE_SOURCE,
        ShadowBacktest.training_feature_snapshot[
            "training_label_contract"
        ]["fee_after_complete"].as_boolean().is_(True),
    )
    base_filters = (
        ShadowBacktest.execution_mode == "paper",
        ShadowBacktest.status == "completed",
        ShadowBacktest.updated_at < cutoff,
    )
    if revision != "v3":
        filters = (
            *base_filters, ShadowBacktest.updated_at >= since, *quality_filters,
        )
        selected_groups = (
            select(group_key.label("decision_group_key"))
            .where(*filters)
            .group_by(group_key)
            .order_by(func.max(ShadowBacktest.updated_at).desc(), group_key.desc())
            .limit(limit)
            .subquery()
        )
        return (
            select(*shadow_training_columns())
            .where(*filters, group_key.in_(select(selected_groups.c.decision_group_key)))
            .order_by(ShadowBacktest.updated_at, ShadowBacktest.id)
        )

    if limit < 1 or cutoff <= since:
        raise ValueError("time coverage requires a positive budget and window")
    filters = (
        *base_filters,
        ShadowBacktest.created_at >= since,
        ShadowBacktest.created_at < cutoff,
        ShadowBacktest.due_at < cutoff,
    )
    groups = (
        select(
            group_key.label("decision_group_key"),
            func.min(ShadowBacktest.created_at).label("decision_at"),
        )
        .where(*filters)
        .group_by(group_key)
        .cte("research_groups")
        .prefix_with("MATERIALIZED", dialect="postgresql")
    )
    bucket = cast(
        func.floor(
            (extract("epoch", groups.c.decision_at) - since.timestamp())
            * limit / (cutoff - since).total_seconds()
        ),
        Integer,
    )
    ranked = select(
        groups.c.decision_group_key,
        func.row_number().over(
            partition_by=bucket,
            order_by=(groups.c.decision_at.desc(), groups.c.decision_group_key.desc()),
        ).label("bucket_rank"),
    ).subquery()
    selected_groups = select(ranked.c.decision_group_key).where(ranked.c.bucket_rank == 1)
    # Bound identities before parsing snapshots. Source/quality rejection
    # cannot cherry-pick a different outcome from the same time bucket.
    selected_ids = (
        select(ShadowBacktest.id)
        .where(*filters, group_key.in_(selected_groups))
        .cte("research_ids")
        .prefix_with("MATERIALIZED", dialect="postgresql")
    )
    return (
        select(*shadow_training_columns())
        .where(ShadowBacktest.id.in_(select(selected_ids.c.id)), *quality_filters)
        .order_by(ShadowBacktest.created_at, ShadowBacktest.id)
    )


REMOTE_SCRIPT = r"""
import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from db.session import close_db, get_read_session_ctx
from scripts.audit_online_horizon_current_development import _build_development_statement
from services.ml_development_snapshot import make_development_snapshot, snapshot_frame
from services.ml_horizon_development import build_horizon_development_report
from services.liquidity_regime_development import build_liquidity_regime_development_report
from services.ml_development_snapshot import InsufficientDevelopmentDataError
from services.ml_signal_service import build_training_frame
from services.ml_training_dataset import (
    shadow_training_row_from_mapping,
)

GROUP_LIMIT = __GROUP_LIMIT__
HOURS = __HOURS__
CUTOFF = datetime.fromisoformat(__CUTOFF__.replace("Z", "+00:00")).astimezone(UTC)
REVISION = __REVISION__
FAMILY = __FAMILY__
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


def _as_dict(value):
    return value if isinstance(value, dict) else {}


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
            "training_sample_count",
            "training_positive_weight_sample_count",
            "validation_sample_count",
            "training_sample_count_by_horizon",
            "label_purged_training_sample_count",
            "decision_group_overlap_count",
            "chronological_label_disjoint",
        )
    } | {
        "selection_reason": selection.get("reason"),
        "selected_policy": selection.get("policy"),
        "inner_folds": selection.get("inner_folds", []),
        "horizon_coverage": selection.get("horizon_coverage", {}),
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
        "fixed_baselines_by_horizon": {
            str(horizon): {
                side: {
                    name: _compact_evidence(_as_dict(views).get(name))
                    for name in ("all_opportunities", "nonoverlapping")
                }
                for side, views in _as_dict(side_views).items()
            }
            for horizon, side_views in _as_dict(
                fold.get("fixed_baselines_by_horizon")
            ).items()
        },
    }


async def main():
    since = CUTOFF - timedelta(hours=HOURS)
    statement = _build_development_statement(
        limit=GROUP_LIMIT, since=since, cutoff=CUTOFF, revision=REVISION,
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
        try:
            snapshot = make_development_snapshot(frame, cutoff=CUTOFF.isoformat())
        except InsufficientDevelopmentDataError as exc:
            snapshot = {}
            development = frame.iloc[0:0]
            research = {
                "status": "insufficient_development_data",
                "reason": str(exc),
                "folds": [],
            }
        else:
            development = snapshot_frame(snapshot, cutoff=CUTOFF.isoformat())
            builder = (
                build_liquidity_regime_development_report
                if FAMILY == "liquidity_regime" else build_horizon_development_report
            )
            research = builder(development, snapshot=snapshot)
        horizon_counts = Counter(
            str(int(value))
            for value in development["horizon_minutes"].tolist()
        ) if not development.empty else Counter()
        complete_group_count = (
            int(research.get("common_horizon_decision_group_count") or 0)
        )
        report = {
            "experiment": (
                "bounded-native-fee-after-horizon-current-development" if FAMILY == "ridge"
                else "bounded-native-fee-after-liquidity-regime-current-development"
            ),
            "protocol_revision": REVISION,
            "generated_at": datetime.now(UTC).isoformat(),
            "window": {
                "since": since.isoformat(),
                "cutoff_exclusive": CUTOFF.isoformat(),
                "hours": HOURS,
                "decision_group_limit": GROUP_LIMIT,
                "sampling_policy": (
                    "latest_group_per_equal_duration_decision_time_bucket_before_quality_filter"
                    if REVISION == "v3" else "latest_settled_decision_groups"
                ),
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
                "common_horizon_group_count": complete_group_count,
                "horizon_row_counts": dict(sorted(horizon_counts.items())),
                "decision_time_start": (
                    development["decision_timestamp"].min().isoformat() if len(development) else None
                ),
                "decision_time_end": (
                    development["decision_timestamp"].max().isoformat() if len(development) else None
                ),
                "decision_time_span_hours": (
                    development["decision_timestamp"].max()
                    - development["decision_timestamp"].min()
                ).total_seconds() / 3600.0 if len(development) else 0,
                "decision_group_counts_by_day": {
                    str(day): int(count)
                    for day, count in development.assign(
                        day=development["decision_timestamp"].dt.strftime("%Y-%m-%d")
                    ).groupby("day")["decision_group"].nunique().items()
                } if len(development) else {},
                "development_input_sha256": snapshot.get("development_input_sha256"),
                "snapshot_sha256": snapshot.get("snapshot_sha256"),
                "dependency_sha256": _dependency_hashes(),
            },
            "research": research if FAMILY == "liquidity_regime" else {
                "status": research.get("status"),
                "reason": research.get("reason"),
                "version": research.get("version"),
                "search_space": research.get("search_space"),
                "selection_policy": research.get("selection_policy"),
                "purge_policy": research.get("purge_policy"),
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
                "runtime_limit_seconds": __RUNTIME_LIMIT_SECONDS__,
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
    family: str = "ridge",
) -> str:
    return (
        REMOTE_SCRIPT.replace("__GROUP_LIMIT__", str(limit))
        .replace("__HOURS__", str(hours))
        .replace(
            "__RUNTIME_LIMIT_SECONDS__",
            str(_remote_runtime_limit_seconds(limit)),
        )
        .replace("__CUTOFF__", repr(cutoff.isoformat().replace("+00:00", "Z")))
        .replace("__REVISION__", repr(revision))
        .replace("__FAMILY__", repr(family))
        .replace("__DEPENDENCIES__", repr(DEPENDENCIES))
    )


def _build_remote_command(
    *,
    limit: int,
    hours: int,
    cutoff: datetime,
    unit: str,
    revision: str,
    family: str = "ridge",
) -> str:
    remote_script = _render_remote_script(
        limit=limit,
        hours=hours,
        cutoff=cutoff,
        revision=revision,
        family=family,
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
            f"--property=RuntimeMaxSec={_remote_runtime_limit_seconds(limit)}s",
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


def _remote_runtime_limit_seconds(limit: int) -> int:
    return 300 if int(limit) <= 1000 else 900


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


def _parse_cutoff(value: str | None) -> datetime:
    cutoff = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value
        else datetime.now(UTC)
    )
    if cutoff.tzinfo is None:
        raise ValueError("--cutoff must include a timezone")
    return cutoff.astimezone(UTC)


def _register_protocol(path: Path, registration: dict[str, object]) -> dict[str, object]:
    """Seal the complete hypothesis before opening a source-data connection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if {key: value for key, value in existing.items() if key != "registered_at"} != registration:
            raise ValueError("registered development protocol cannot change")
        return existing
    sealed = {**registration, "registered_at": datetime.now(UTC).isoformat()}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(sealed, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return sealed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--hours", type=int)
    parser.add_argument("--cutoff")
    parser.add_argument("--revision", choices=("v1", "v2", "v3"), default="v3")
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--family", choices=("ridge", "liquidity_regime"), default="ridge")
    args = parser.parse_args()
    limit = max(1, min(int(args.limit), 2500))
    max_hours = 672 if args.revision == "v3" else 168
    default_hours = 672 if args.revision == "v3" else 24
    hours = max(1, min(int(args.hours or default_hours), max_hours))
    cutoff = _parse_cutoff(args.cutoff)
    if args.family == "liquidity_regime" and (not args.cutoff or args.revision != "v3"):
        raise ValueError("liquidity_regime requires explicit --cutoff and time-stratified v3 sampling")
    name = "horizon" if args.family == "ridge" else "liquidity-regime"
    unit = f"bb-{name}-dev-{args.revision}-{cutoff.strftime('%Y%m%dT%H%M%SZ')}"
    result_path = (
        ROOT
        / "data"
        / "profitability_remediation"
        / f"online-{name}-development-{args.revision}-"
        f"{cutoff.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {result_path}")

    expected_hashes = _local_dependency_hashes()
    registration = None
    if args.family == "liquidity_regime":
        from services.liquidity_regime_development import protocol

        registration = {
            "protocol": protocol(), "cutoff_exclusive": cutoff.isoformat(),
            "hours": hours, "decision_group_limit": limit, "sampling_revision": args.revision,
            "dependency_sha256": expected_hashes,
        }
        registration_path = result_path.with_suffix(".protocol.json")
        registration = _register_protocol(registration_path, registration)
        safe_print(f"registered_protocol={registration_path}")
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
            f"{_build_remote_command(limit=limit, hours=hours, cutoff=cutoff, unit=unit, revision=args.revision, family=args.family)} "
            f"> {shlex.quote(remote_result)}"
        )
        run_remote_text(
            ssh,
            command,
            timeout=_remote_runtime_limit_seconds(limit) + 30,
            max_output_chars=3000,
        )
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
    _validate_expected_input_hash(report, args.expected_input_sha256)
    if registration is not None:
        report["registered_protocol"] = registration
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.family == "liquidity_regime":
        safe_print(json.dumps({
            "input": report["input"],
            "status": report["research"]["status"],
            "reason": report["research"].get("reason"),
            "qualified_candidate_ids": report["research"].get("qualified_candidate_ids", []),
            "candidates": [{
                "id": item["policy"]["id"],
                "aggregate_capacity": item["aggregate_evidence"]["capacity"],
                "rejection_reasons": item["rejection_reasons"],
            } for item in report["research"].get("candidates", [])],
        }, ensure_ascii=False, indent=2))
    else:
        safe_print(json.dumps(report, ensure_ascii=False, indent=2))
    safe_print(f"saved_report={result_path}")


if __name__ == "__main__":
    main()
