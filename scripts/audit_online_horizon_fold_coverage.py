"""Project chronological fold coverage without selecting or reading returns."""

from __future__ import annotations

import argparse
import base64
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_SCRIPT = r"""
import asyncio
import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from db.session import get_read_session_ctx

CUTOFF = datetime.fromisoformat(__CUTOFF__.replace("Z", "+00:00")).astimezone(UTC)
HOURS = __HOURS__
GROUP_LIMIT = __GROUP_LIMIT__
HORIZONS = (5, 15, 60, 240)
NATIVE_SOURCE = "shadow_entry_result_and_okx_native_1m_path"
MIN_TRAINING = 16
FOLD_COUNT = 3


async def main():
    since = CUTOFF - timedelta(hours=HOURS)
    async with get_read_session_ctx(
        statement_timeout_ms=15000,
        transaction_isolation="REPEATABLE READ READ ONLY",
    ) as session:
        result = await session.execute(text('''
            WITH filtered AS (
                SELECT decision_id, horizon_minutes, created_at, updated_at, due_at
                FROM shadow_backtests
                WHERE execution_mode='paper'
                  AND status='completed'
                  AND analysis_type='market'
                  AND decision_id > 0
                  AND horizon_minutes IN (5, 15, 60, 240)
                  AND updated_at >= :since
                  AND updated_at < :cutoff
                  AND training_feature_snapshot
                      ->'training_market_fact_contract'->>'source'
                      = :native_source
                  AND training_feature_snapshot
                      ->'training_label_contract'->>'fee_after_complete'
                      = 'true'
            ),
            selected AS (
                SELECT decision_id, max(updated_at) AS max_updated_at
                FROM filtered
                GROUP BY decision_id
                ORDER BY max_updated_at, decision_id
                LIMIT :group_limit
            )
            SELECT f.decision_id, f.horizon_minutes, f.created_at, f.updated_at, f.due_at
            FROM filtered f
            JOIN selected s ON s.decision_id = f.decision_id
            ORDER BY f.created_at, f.decision_id, f.horizon_minutes
        '''), {
            "since": since,
            "cutoff": CUTOFF,
            "group_limit": GROUP_LIMIT,
            "native_source": NATIVE_SOURCE,
        })
        rows = [dict(row) for row in result.mappings()]

    groups = {}
    for row in rows:
        group = str(row["decision_id"])
        item = groups.setdefault(group, {"horizons": {}, "rows": []})
        item["horizons"][int(row["horizon_minutes"])] = row
        item["rows"].append(row)

    complete = []
    for group, item in groups.items():
        if set(HORIZONS) != set(item["horizons"]):
            continue
        rows_for_group = item["rows"]
        if len(rows_for_group) != len(HORIZONS):
            raise ValueError("duplicate decision-group/horizon rows in source projection")
        decision_timestamps = []
        label_ends = []
        for row in rows_for_group:
            created = row.get("created_at")
            due = row.get("due_at")
            if created is None or due is None:
                continue
            decision_timestamps.append(created)
            label_ends.append(due)
        if (
            len(decision_timestamps) != len(rows_for_group)
            or len(label_ends) != len(rows_for_group)
        ):
            continue
        complete.append({
            "group": group,
            "decision_start": min(decision_timestamps),
            "decision_end": max(decision_timestamps),
            "label_end": max(label_ends),
        })

    complete.sort(
        key=lambda item: (
            item["decision_start"],
            item["decision_end"],
            item["group"],
        )
    )
    initial = max((len(complete) + 1) // 2, 1)
    validation_groups = complete[initial:]
    chunks = []
    if validation_groups:
        chunk_size = (len(validation_groups) + FOLD_COUNT - 1) // FOLD_COUNT
        chunks = [
            validation_groups[index:index + chunk_size]
            for index in range(0, len(validation_groups), chunk_size)
        ]
    folds = []
    for index, validation in enumerate(chunks):
        if not validation:
            continue
        validation_start = min(item["decision_start"] for item in validation)
        train = [
            item for item in complete
            if item["label_end"] < validation_start
        ]
        folds.append({
            "fold_index": index,
            "validation_group_count": len(validation),
            "training_group_count": len(train),
            "training_primary_sample_count": len(train),
            "validation_primary_sample_count": len(validation),
            "training_label_end": max(
                (item["label_end"] for item in train), default=None
            ),
            "validation_decision_start": validation_start,
            "passes_minimum_training": len(train) >= MIN_TRAINING,
        })

    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "window": {
            "since": since.isoformat(),
            "cutoff_exclusive": CUTOFF.isoformat(),
            "hours": HOURS,
            "decision_group_limit": GROUP_LIMIT,
            "source": NATIVE_SOURCE,
        },
        "rows": len(rows),
        "complete_four_horizon_groups": len(complete),
        "initial_training_group_count_before_purge": initial,
        "validation_group_count_after_initial_split": len(validation_groups),
        "folds": folds,
        "all_folds_pass_minimum": bool(
            len(folds) == FOLD_COUNT
            and all(item["passes_minimum_training"] for item in folds)
        ),
        "safety": {
            "returns_read": False,
            "exam_labels_evaluated": False,
            "database_mutated": False,
            "orders_submitted": False,
        },
    }
    print(json.dumps(report, ensure_ascii=False, default=str))


asyncio.run(main())
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--cutoff")
    args = parser.parse_args()
    cutoff = (
        datetime.fromisoformat(args.cutoff.replace("Z", "+00:00"))
        if args.cutoff
        else datetime.now(UTC)
    ).astimezone(UTC)
    rendered = (
        REMOTE_SCRIPT.replace("__CUTOFF__", repr(cutoff.isoformat().replace("+00:00", "Z")))
        .replace("__HOURS__", str(max(1, min(args.hours, 168))))
        .replace("__GROUP_LIMIT__", str(max(1, min(args.limit, 2500))))
    )
    payload = base64.b64encode(rendered.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode({payload!r}))"
    command = " ".join((
        "systemd-run --quiet --wait --pipe --collect",
        "--unit=bb-horizon-fold-coverage-audit",
        "--property=WorkingDirectory=/data/bb/app",
        "--property=User=bb",
        "--property=EnvironmentFile=-/data/bb/app/.env",
        "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
        "--property=RuntimeMaxSec=90s",
        "--setenv=DATABASE_POOL_SIZE=1",
        "--setenv=DATABASE_MAX_OVERFLOW=0",
        "/data/bb/app/.venv/bin/python -c",
        shlex.quote(runner),
    ))
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(ssh, command, timeout=110, max_output_chars=20000)
    finally:
        ssh.close()
    safe_print(json.dumps(json.loads(output), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
