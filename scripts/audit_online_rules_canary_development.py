"""Run bounded development evaluation for the fixed rules-canary policy."""

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
    "services/live_rules_canary_signal.py",
    "services/rules_canary_development.py",
    "services/ml_development_baselines.py",
    "services/ml_horizon_development.py",
    "services/ml_signal_service.py",
    "services/ml_training_dataset.py",
    "services/training_data_quality.py",
)

REMOTE_SCRIPT = r"""
import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select

from db.session import close_db, get_read_session_ctx
from models.learning import ShadowBacktest
from services.ml_signal_service import build_training_frame
from services.ml_training_dataset import (
    shadow_training_columns,
    shadow_training_row_from_mapping,
)
from services.rules_canary_development import (
    RULES_CANARY_PRIMARY_HORIZON_MINUTES,
    build_rules_canary_development_report,
    filter_rules_canary_decision_window,
)

LIMIT = __LIMIT__
HOURS = __HOURS__
DECISION_CUTOFF = datetime.fromisoformat(
    __DECISION_CUTOFF__.replace("Z", "+00:00")
).astimezone(UTC)
MATURITY_CUTOFF = datetime.fromisoformat(
    __MATURITY_CUTOFF__.replace("Z", "+00:00")
).astimezone(UTC)
DEPENDENCIES = __DEPENDENCIES__
NATIVE_SOURCE = "shadow_entry_result_and_okx_native_1m_path"


def dependency_hashes():
    return {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in DEPENDENCIES
    }


async def main():
    since = DECISION_CUTOFF - timedelta(hours=HOURS)
    group_key = func.coalesce(
        func.nullif(ShadowBacktest.decision_id, 0),
        -ShadowBacktest.id,
    )
    filters = (
        ShadowBacktest.execution_mode == "paper",
        ShadowBacktest.status == "completed",
        ShadowBacktest.updated_at >= since,
        ShadowBacktest.updated_at < MATURITY_CUTOFF,
        ShadowBacktest.horizon_minutes == RULES_CANARY_PRIMARY_HORIZON_MINUTES,
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
        .order_by(func.max(ShadowBacktest.created_at), group_key)
        .limit(LIMIT)
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
        frame, decision_window = filter_rules_canary_decision_window(
            frame,
            since=since,
            cutoff=DECISION_CUTOFF,
        )
        selected_ids = set(frame["id"].astype(int)) if not frame.empty else set()
        snapshots_by_id = {
            int(row.id): row.feature_snapshot
            for row in rows
            if int(row.id) in selected_ids and isinstance(row.feature_snapshot, dict)
        }
        report = build_rules_canary_development_report(
            frame,
            snapshots_by_id=snapshots_by_id,
        )
        report.update(
            {
                "experiment": "bounded-native-fee-after-fixed-rules-canary-development",
                "generated_at": datetime.now(UTC).isoformat(),
                "window": {
                    "since": since.isoformat(),
                    "decision_cutoff_exclusive": DECISION_CUTOFF.isoformat(),
                    "maturity_cutoff_exclusive": MATURITY_CUTOFF.isoformat(),
                    "hours": HOURS,
                    "decision_group_limit": LIMIT,
                    "source": NATIVE_SOURCE,
                    "filter": (
                        "completed paper labels with fee_after_complete=true; "
                        "fixed live-rules-canary signal; primary five-minute horizon; "
                        "decision time in [since, decision_cutoff); "
                        "updated_at before maturity_cutoff"
                    ),
                    "decision_timestamp_source": (
                        "training_feature_snapshot.original_decision_timestamp"
                    ),
                    "maturity_timestamp_source": "shadow_backtests.updated_at",
                },
                "input": {
                    "source_row_count": len(rows),
                    "training_frame_row_count": int(len(frame)),
                    "dependency_sha256": dependency_hashes(),
                },
                "decision_window_filter": decision_window,
                "database_mutated": False,
                "orders_submitted": False,
            }
        )
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, default=str))
    finally:
        await close_db()


asyncio.run(main())
"""


def _parse_cutoff(value: str) -> datetime:
    cutoff = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        raise ValueError("--cutoff must include a timezone")
    return cutoff.astimezone(UTC)


def _render_remote_script(
    *,
    limit: int,
    hours: int,
    cutoff: datetime,
    maturity_cutoff: datetime | None = None,
) -> str:
    maturity_cutoff = maturity_cutoff or cutoff
    return (
        REMOTE_SCRIPT.replace("__LIMIT__", str(limit))
        .replace("__HOURS__", str(hours))
        .replace(
            "__DECISION_CUTOFF__",
            repr(cutoff.isoformat().replace("+00:00", "Z")),
        )
        .replace(
            "__MATURITY_CUTOFF__",
            repr(maturity_cutoff.isoformat().replace("+00:00", "Z")),
        )
        .replace("__DEPENDENCIES__", repr(DEPENDENCIES))
    )


def _build_remote_command(
    *,
    limit: int,
    hours: int,
    cutoff: datetime,
    maturity_cutoff: datetime | None = None,
    unit: str,
) -> str:
    script = _render_remote_script(
        limit=limit,
        hours=hours,
        cutoff=cutoff,
        maturity_cutoff=maturity_cutoff,
    )
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode({encoded!r}))"
    return " ".join(
        (
            "systemd-run --quiet --wait --pipe --collect",
            f"--unit={shlex.quote(unit)}",
            "--property=WorkingDirectory=/data/bb/app",
            "--property=User=bb",
            "--property=CPUQuota=25%",
            "--property=MemoryMax=1G",
            "--property=RuntimeMaxSec=300s",
            "--property=TimeoutStopSec=15s",
            "--property=KillMode=control-group",
            "--property=EnvironmentFile=-/data/bb/app/.env",
            "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
            "--setenv=DATABASE_POOL_SIZE=1",
            "--setenv=DATABASE_MAX_OVERFLOW=0",
            "--setenv=OMP_NUM_THREADS=1",
            "--setenv=OPENBLAS_NUM_THREADS=1",
            "/data/bb/app/.venv/bin/python -c",
            shlex.quote(runner),
        )
    )


def _local_dependency_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in DEPENDENCIES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument(
        "--maturity-cutoff",
        help="exclusive label updated_at cutoff; defaults to --cutoff",
    )
    args = parser.parse_args()
    limit = max(1, min(int(args.limit), 2500))
    hours = max(1, min(int(args.hours), 168))
    cutoff = _parse_cutoff(args.cutoff)
    maturity_cutoff = (
        _parse_cutoff(args.maturity_cutoff)
        if args.maturity_cutoff
        else cutoff
    )
    if maturity_cutoff < cutoff:
        parser.error("--maturity-cutoff must be at or after --cutoff")
    decision_stamp = cutoff.strftime("%Y%m%dT%H%M%SZ")
    maturity_stamp = maturity_cutoff.strftime("%Y%m%dT%H%M%SZ")
    unit = f"bb-rules-canary-development-d{decision_stamp}-m{maturity_stamp}"
    result_path = (
        ROOT
        / "data"
        / "profitability_remediation"
        / f"online-rules-canary-development-d{decision_stamp}-m{maturity_stamp}.json"
    )
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {result_path}")

    expected_hashes = _local_dependency_hashes()
    ssh = connect_remote_ssh(ROOT, timeout=25)
    sftp = ssh.open_sftp()
    remote_result = (
        "/data/bb/app/tmp/codex-rules-canary-development/"
        f"result_{secrets.token_hex(12)}.json"
    )
    try:
        command = (
            "set -e\numask 077\n"
            "install -d -m 0700 /data/bb/app/tmp/codex-rules-canary-development\n"
            f"{_build_remote_command(limit=limit, hours=hours, cutoff=cutoff, maturity_cutoff=maturity_cutoff, unit=unit)}"
            f" > {shlex.quote(remote_result)}"
        )
        run_remote_text(ssh, command, timeout=320, max_output_chars=4000)
        with sftp.file(remote_result, "r") as remote_file:
            report = json.loads(remote_file.read().decode("utf-8"))
        if report.get("input", {}).get("dependency_sha256") != expected_hashes:
            raise RuntimeError("remote rules-canary dependencies differ from local")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        with suppress(OSError):
            sftp.remove(remote_result)
        sftp.close()
        ssh.close()
    safe_print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"saved_report={result_path}")


if __name__ == "__main__":
    main()
