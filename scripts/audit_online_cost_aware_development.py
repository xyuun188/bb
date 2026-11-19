"""Run a bounded, read-only cost-aware development baseline ledger online.

The ledger is development evidence only. It never selects or promotes a
candidate, never reads the independent exam, and never submits an order.
"""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import shlex
import sys
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_SCRIPT = r'''
import asyncio
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import Integer, cast, extract, func, select

from core.training_contracts import SHADOW_LABEL_HORIZONS_MINUTES
from db.session import close_db, get_read_session_ctx
from models.learning import ShadowBacktest
from services.shadow_development_split import split_shadow_development_rows

LIMIT = __LIMIT__
HOURS = __HOURS__


def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _dict(value):
    return value if isinstance(value, dict) else {}


def _metrics(values):
    clean = [value for value in values if value is not None]
    gains = sum(value for value in clean if value > 0)
    losses = -sum(value for value in clean if value < 0)
    return {
        "count": len(clean),
        "mean_net_return_pct": round(sum(clean) / len(clean), 8) if clean else None,
        "positive_count": sum(value > 0 for value in clean),
        "negative_count": sum(value < 0 for value in clean),
        "profit_factor": round(gains / losses, 8) if losses > 0 else None,
        "total_net_return_pct": round(sum(clean), 8) if clean else None,
    }


def _baseline(rows, key):
    return _metrics([row.get(key) for row in rows])


def _fold(rows):
    return {
        "rows": len(rows),
        "always_long": _baseline(rows, "long_net"),
        "always_short": _baseline(rows, "short_net"),
        "hindsight_best_action_upper_bound": _baseline(rows, "best_net"),
        "observed_decision_action": _metrics([
            row.get(f"{row['decision_action']}_net") for row in rows
        ]),
    }


def _sample_statement(since, now):
    horizons = SHADOW_LABEL_HORIZONS_MINUTES
    bucket_count = (LIMIT + len(horizons) - 1) // len(horizons)
    bucket = cast(
        func.floor(
            (extract("epoch", ShadowBacktest.created_at) - since.timestamp())
            * bucket_count / (now - since).total_seconds()
        ),
        Integer,
    )
    # Bound the read across decision time and horizons before fetching wide snapshots.
    ranked = (
        select(
            ShadowBacktest.id,
            ShadowBacktest.horizon_minutes,
            bucket.label("time_bucket"),
            func.row_number().over(
                partition_by=(ShadowBacktest.horizon_minutes, bucket),
                order_by=(ShadowBacktest.created_at.desc(), ShadowBacktest.id.desc()),
            ).label("bucket_rank"),
        )
        .where(
            ShadowBacktest.execution_mode == "paper",
            ShadowBacktest.analysis_type == "market",
            ShadowBacktest.status == "completed",
            ShadowBacktest.horizon_minutes.in_(horizons),
            ShadowBacktest.created_at >= since,
            ShadowBacktest.created_at < now,
            ShadowBacktest.updated_at < now,
            ShadowBacktest.training_feature_snapshot[
                "training_market_fact_contract"
            ]["source"].as_string()
            == "shadow_entry_result_and_okx_native_1m_path",
            ShadowBacktest.training_feature_snapshot[
                "training_label_contract"
            ]["fee_after_complete"].as_boolean()
            .is_(True),
        )
        .subquery()
    )
    ids = (
        select(ranked.c.id)
        .where(ranked.c.bucket_rank == 1)
        .order_by(ranked.c.time_bucket, ranked.c.horizon_minutes)
        .limit(LIMIT)
    )
    return select(
        ShadowBacktest.id,
        ShadowBacktest.decision_id,
        ShadowBacktest.symbol,
        ShadowBacktest.horizon_minutes,
        ShadowBacktest.decision_action,
        ShadowBacktest.updated_at,
        ShadowBacktest.training_feature_snapshot,
    ).where(ShadowBacktest.id.in_(ids))


async def main():
    now = datetime.now(UTC)
    since = now - timedelta(hours=HOURS)
    try:
        async with get_read_session_ctx(statement_timeout_ms=15000) as session:
            result = await session.execute(_sample_statement(since, now))
            raw_rows = list(result.mappings())
    finally:
        await close_db()

    rows = []
    for raw in raw_rows:
        snapshot = _dict(raw.get("training_feature_snapshot"))
        label = _dict(snapshot.get("training_label_contract"))
        long_net = _float(label.get("long_net_return_after_all_cost_pct"))
        short_net = _float(label.get("short_net_return_after_all_cost_pct"))
        action = str(raw.get("decision_action") or "hold").lower()
        rows.append(
            {
                "id": int(raw["id"]),
                "decision_id": int(raw["decision_id"]) if raw.get("decision_id") else None,
                "symbol": str(raw.get("symbol") or ""),
                "horizon_minutes": int(raw["horizon_minutes"]),
                "decision_action": action,
                "decision_at": snapshot.get("original_decision_timestamp"),
                "label_at": label.get("label_timestamp"),
                "updated_at": raw.get("updated_at"),
                "long_net": long_net,
                "short_net": short_net,
                "best_net": (
                    max(long_net, short_net, 0.0)
                    if long_net is not None and short_net is not None else None
                ),
            }
        )

    development_rows, validation_rows, timing = split_shadow_development_rows(rows)
    by_horizon = defaultdict(lambda: ([], []))
    missing_return_count = 0
    for fold_number, fold_rows in enumerate((development_rows, validation_rows)):
        for row in fold_rows:
            if row["best_net"] is None:
                missing_return_count += 1
                continue
            by_horizon[row["horizon_minutes"]][fold_number].append(row)
    horizons = {}
    for horizon in SHADOW_LABEL_HORIZONS_MINUTES:
        development, validation = by_horizon[horizon]
        ordered = development + validation
        horizons[str(horizon)] = {
            "rows": len(ordered),
            "distinct_decision_groups": len(
                {row["decision_id"] for row in ordered if row["decision_id"] is not None}
            ),
            "time_range": {
                "first": ordered[0]["decision_at"].isoformat() if ordered else None,
                "last": ordered[-1]["decision_at"].isoformat() if ordered else None,
            },
            "decision_action_counts": {
                action: sum(row["decision_action"] == action for row in ordered)
                for action in ("long", "short", "hold")
            },
            "development_fold": _fold(development),
            "validation_fold": _fold(validation),
        }

    print(
        json.dumps(
            {
                "checked_at": now.isoformat(),
                "development_window": {
                    "since": since.isoformat(),
                    "hours": HOURS,
                    "limit": LIMIT,
                    "selection_policy": "one_completed_row_per_horizon_and_decision_time_bucket",
                    "source": "native executable 1m path + fee-after-complete labels",
                    "exam_opened": False,
                },
                "row_count": len(rows),
                "distinct_decision_groups": timing["decision_group_count"],
                "split": timing,
                "invalid_outcome_row_count": missing_return_count,
                "horizons": horizons,
                "candidate_status": (
                    "no candidate selected; transparent baselines only"
                ),
            },
            ensure_ascii=False,
            default=str,
        )
    )


asyncio.run(main())
'''


def _build_remote_command(limit: int, hours: int) -> str:
    remote_script = (
        REMOTE_SCRIPT.replace("__LIMIT__", str(limit))
        .replace("__HOURS__", str(hours))
    )
    payload = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode({payload!r}))"
    return " ".join(
        (
            "systemd-run --quiet --wait --pipe --collect",
            "--property=WorkingDirectory=/data/bb/app",
            "--property=User=bb",
            "--property=CPUQuota=100%",
            "--property=MemoryMax=1G",
            "--property=RuntimeMaxSec=240s",
            "--property=EnvironmentFile=-/data/bb/app/.env",
            "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
            "/data/bb/app/.venv/bin/python -c",
            shlex.quote(runner),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--hours", type=int, default=6)
    args = parser.parse_args()
    limit = max(1, min(int(args.limit), 2500))
    hours = max(1, min(int(args.hours), 24))
    result_path = (
        f"/data/bb/app/tmp/codex-cost-ledger/result_{secrets.token_hex(12)}.json"
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    sftp = ssh.open_sftp()
    try:
        command = (
            "set -e\numask 077\n"
            "install -d -m 0700 /data/bb/app/tmp/codex-cost-ledger\n"
            f"{_build_remote_command(limit, hours)} > {shlex.quote(result_path)}"
        )
        run_remote_text(ssh, command, timeout=260, max_output_chars=4000)
        with sftp.file(result_path, "r") as remote_file:
            report = json.loads(remote_file.read().decode("utf-8"))
        if not isinstance(report, dict):
            raise ValueError("cost-aware development audit must return an object")
    finally:
        with suppress(OSError):
            sftp.remove(result_path)
        sftp.close()
        ssh.close()
    safe_print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
