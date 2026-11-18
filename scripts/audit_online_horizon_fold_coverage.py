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


def _row_clocks(
    row: dict, *, since: datetime, cutoff: datetime,
) -> tuple[datetime, datetime] | None:
    try:
        decision = datetime.fromisoformat(row["original_decision_at"])
        available = datetime.fromisoformat(row["label_available_at"])
    except (TypeError, ValueError):
        return None
    if decision.utcoffset() is None or available.utcoffset() is None:
        return None
    if not (
        since <= decision <= row["created_at"] < cutoff
        and decision < row["due_at"] <= available < cutoff
    ):
        return None
    return decision.astimezone(UTC), available.astimezone(UTC)


REMOTE_SCRIPT = r"""
import asyncio
import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from db.session import get_read_session_ctx
from scripts.audit_online_horizon_fold_coverage import _row_clocks

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
                SELECT decision_id, horizon_minutes, created_at, due_at,
                       training_feature_snapshot->>'original_decision_timestamp'
                           AS original_decision_at,
                       coalesce(
                           training_feature_snapshot
                               #>>'{training_label_contract,labels,label_timestamp}',
                           training_feature_snapshot
                               #>>'{training_label_contract,label_timestamp}'
                       ) AS label_available_at
                FROM shadow_backtests
                WHERE execution_mode='paper'
                  AND status='completed'
                  AND analysis_type='market'
                  AND decision_id > 0
                  AND horizon_minutes IN (5, 15, 60, 240)
                  AND created_at >= :since
                  AND created_at < :cutoff
                  AND due_at < :cutoff
                  AND updated_at < :cutoff
                  AND training_feature_snapshot
                      ->'training_market_fact_contract'->>'source'
                      = :native_source
                  AND training_feature_snapshot
                      ->'training_label_contract'->>'fee_after_complete'
                      = 'true'
            ),
            selected AS (
                SELECT decision_id, min(created_at) AS first_created_at
                FROM filtered
                GROUP BY decision_id
                ORDER BY first_created_at, decision_id
                LIMIT :group_limit
            )
            SELECT f.*
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
    excluded_clock_rows = 0
    for row in rows:
        clocks = _row_clocks(row, since=since, cutoff=CUTOFF)
        if clocks is None:
            excluded_clock_rows += 1
            continue
        decision, available = clocks
        group = str(row["decision_id"])
        item = groups.setdefault(group, {})
        horizon = int(row["horizon_minutes"])
        if horizon in item:
            raise ValueError("duplicate decision-group/horizon rows in source projection")
        item[horizon] = (decision, available)

    complete = []
    for group, item in groups.items():
        if set(HORIZONS) != set(item):
            continue
        decision_timestamps, label_ends = zip(*item.values(), strict=True)
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
        "excluded_clock_rows": excluded_clock_rows,
        "projection": "declared_decision_and_label_availability_complete_groups",
        "full_training_eligibility_checked": False,
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
