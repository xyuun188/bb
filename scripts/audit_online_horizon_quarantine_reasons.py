"""Summarize bounded per-horizon quarantine reasons for native shadow labels."""

from __future__ import annotations

import argparse
import base64
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

HORIZONS = (5, 15, 60, 240)
NATIVE_SOURCE = "shadow_entry_result_and_okx_native_1m_path"

REMOTE_SCRIPT = r'''
import asyncio
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from db.session import get_read_session_ctx

HOURS = __HOURS__
ROW_LIMIT = __ROW_LIMIT__
MATURE_MINUTES = __MATURE_MINUTES__
HORIZONS = (5, 15, 60, 240)
NATIVE_SOURCE = "shadow_entry_result_and_okx_native_1m_path"


def _dict(value):
    return value if isinstance(value, dict) else {}


def _reason_tokens(note):
    text_value = str(note or "")
    marker = "[training_quarantine]"
    if marker not in text_value:
        return []
    suffix = text_value.split(marker, 1)[1].strip()
    return [item.strip() for item in suffix.split(",") if item.strip()]


def _state(row):
    status = str(row.get("status") or "")
    source = str(row.get("source") or "")
    fee_complete = row.get("fee_after_complete") is True
    native = source == NATIVE_SOURCE
    if status == "completed" and native and fee_complete:
        return "qualified"
    if status == "pending" and native:
        return "pending"
    if status == "quarantined":
        reasons = row.get("reasons") or []
        return "quarantined:" + ",".join(reasons[:4]) if reasons else "quarantined:unknown"
    if status == "completed":
        return "completed_non_native"
    return status or "unknown"


def _group_horizon_state(rows):
    """Collapse duplicate rows without letting a legacy row mask a clean one."""

    if not rows:
        return "missing"
    states = [_state(row) for row in rows]
    if "qualified" in states:
        return "qualified"
    if "pending" in states:
        return "pending"
    native_quarantines = [
        value
        for row, value in zip(rows, states, strict=True)
        if row.get("source") == NATIVE_SOURCE and value.startswith("quarantined:")
    ]
    if native_quarantines:
        return native_quarantines[0]
    return states[0]


async def main():
    mature_cutoff = datetime.now(UTC) - timedelta(minutes=MATURE_MINUTES)
    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "lookback_hours": HOURS,
        "row_limit": ROW_LIMIT,
        "source": NATIVE_SOURCE,
        "horizons": list(HORIZONS),
    }
    async with get_read_session_ctx(
        statement_timeout_ms=10000,
        transaction_isolation="REPEATABLE READ READ ONLY",
    ) as session:
        result = await session.execute(text("""
            SELECT id, decision_id, symbol, horizon_minutes, status, created_at, updated_at,
                   note,
                   feature_snapshot->'market_fact' AS market_fact,
                   feature_snapshot->'market_source_consistency'
                       AS market_source_consistency,
                   training_feature_snapshot
                       ->'training_market_fact_contract'->>'source' AS source,
                   training_feature_snapshot
                       ->'training_label_contract'->>'fee_after_complete'
                       AS fee_after_complete,
                   feature_snapshot
                       ->'market_data_quality'->>'code' AS quality_code,
                   feature_snapshot
                       ->'market_data_quality'->>'reason' AS quality_reason,
                   feature_snapshot
                       ->'market_data_quality'->'details'
                       ->>'market_fact_reason_codes' AS market_fact_reason_codes
            FROM shadow_backtests
            WHERE execution_mode='paper'
              AND analysis_type='market'
              AND decision_id > 0
              AND horizon_minutes IN (5, 15, 60, 240)
              AND label_version='2026-07-27.market-opportunity-multitask.v2'
              AND created_at >= now() - (:hours * interval '1 hour')
            ORDER BY created_at DESC, id DESC
            LIMIT :row_limit
        """), {"hours": HOURS, "row_limit": ROW_LIMIT})
        rows = []
        for raw in result.mappings():
            item = dict(raw)
            item["fee_after_complete"] = (
                str(item.get("fee_after_complete") or "").lower() == "true"
            )
            item["reasons"] = _reason_tokens(item.get("note"))
            item["state"] = _state(item)
            rows.append(item)

    by_horizon = defaultdict(Counter)
    quarantine_reasons = defaultdict(Counter)
    native_invalid_details = defaultdict(Counter)
    native_invalid_samples = []
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        horizon = int(row["horizon_minutes"])
        by_horizon[horizon][row["state"].split(":", 1)[0]] += 1
        if row["status"] == "quarantined":
            for reason in row["reasons"] or ["unknown"]:
                quarantine_reasons[horizon][reason] += 1
            if row.get("quality_code") == "native_market_fact_invalid":
                details = (
                    row.get("market_fact_reason_codes")
                    or row.get("quality_reason")
                    or "details_missing"
                )
                for reason in str(details).split(";"):
                    if reason.strip():
                        native_invalid_details[horizon][reason.strip()] += 1
                if len(native_invalid_samples) < 16:
                    fact = _dict(row.get("market_fact"))
                    consistency = _dict(
                        row.get("market_source_consistency")
                        or fact.get("source_consistency")
                    )
                    path = _dict(consistency.get("path"))
                    reconciliation = _dict(consistency.get("quote_reconciliation"))
                    native_invalid_samples.append(
                        {
                            "id": row.get("id"),
                            "decision_id": row.get("decision_id"),
                            "symbol": row.get("symbol"),
                            "horizon_minutes": horizon,
                            "created_at": row.get("created_at"),
                            "source_interface": fact.get("source_interface"),
                            "source_endpoint": fact.get("source_endpoint"),
                            "source_channel": fact.get("source_channel"),
                            "source_timestamp_ms": fact.get("source_timestamp_ms"),
                            "quality_code": row.get("quality_code"),
                            "quality_reason": row.get("quality_reason"),
                            "market_fact_reason_codes": row.get(
                                "market_fact_reason_codes"
                            ),
                            "consistency_status": consistency.get("status"),
                            "consistency_reasons": consistency.get("reasons"),
                            "consistency_assertions": consistency.get("assertions"),
                            "quote_reconciliation": {
                                key: reconciliation.get(key)
                                for key in (
                                    "time_span_ms",
                                    "gap",
                                    "tolerance",
                                    "max_interval_spread",
                                )
                                if key in reconciliation
                            },
                            "path": {
                                key: path.get(key)
                                for key in (
                                    "source",
                                    "bar_count",
                                    "missing_open_times_count",
                                    "observed_prices_within_path",
                                    "continuity_accepted_via",
                                )
                                if key in path
                            },
                        }
                    )
        groups[int(row["decision_id"])][horizon].append(row)

    pattern_counts = Counter()
    mature_pattern_counts = Counter()
    mature_groups = 0
    mature_complete_rows = 0
    qualified_groups = 0
    qualified_mature_groups = 0
    duplicate_group_horizon_count = 0
    duplicate_row_count = 0
    for decision_id, horizon_rows in groups.items():
        duplicate_group_horizon_count += sum(
            len(rows_for_horizon) > 1 for rows_for_horizon in horizon_rows.values()
        )
        duplicate_row_count += sum(
            max(len(rows_for_horizon) - 1, 0)
            for rows_for_horizon in horizon_rows.values()
        )
        pattern = tuple(
            _group_horizon_state(horizon_rows[horizon])
            if horizon in horizon_rows
            else "missing"
            for horizon in HORIZONS
        )
        pattern_counts[pattern] += 1
        if all(horizon in horizon_rows for horizon in HORIZONS):
            created = [
                min(
                    row.get("created_at")
                    for row in horizon_rows[horizon]
                    if row.get("created_at") is not None
                )
                for horizon in HORIZONS
                if any(
                    row.get("created_at") is not None
                    for row in horizon_rows[horizon]
                )
            ]
            mature = all(
                value is not None
                and value < mature_cutoff
                for value in created
            )
            if mature:
                mature_groups += 1
                mature_pattern_counts[pattern] += 1
                mature_complete_rows += 4
                if all(value == "qualified" for value in pattern):
                    qualified_mature_groups += 1
        if all(value == "qualified" for value in pattern):
            qualified_groups += 1

    report["rows_by_horizon_state"] = {
        str(horizon): dict(by_horizon[horizon]) for horizon in HORIZONS
    }
    report["quarantine_reasons_by_horizon"] = {
        str(horizon): dict(quarantine_reasons[horizon].most_common(20))
        for horizon in HORIZONS
    }
    report["native_market_fact_invalid_details_by_horizon"] = {
        str(horizon): dict(native_invalid_details[horizon].most_common(20))
        for horizon in HORIZONS
    }
    report["native_market_fact_invalid_samples"] = native_invalid_samples
    report["coverage"] = {
        "decision_group_count": len(groups),
        "all_four_qualified_group_count": qualified_groups,
        "mature_all_four_group_count": mature_groups,
        "mature_all_four_qualified_group_count": qualified_mature_groups,
        "mature_all_four_row_count": mature_complete_rows,
        "duplicate_group_horizon_count": duplicate_group_horizon_count,
        "duplicate_row_count": duplicate_row_count,
    }
    report["top_mature_state_patterns"] = [
        {
            "states": list(pattern),
            "groups": count,
        }
        for pattern, count in mature_pattern_counts.most_common(24)
    ]
    report["safety"] = {
        "database_mutated": False,
        "exam_labels_evaluated": False,
        "orders_submitted": False,
    }
    print(json.dumps(report, ensure_ascii=False, default=str))


asyncio.run(main())
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=12)
    parser.add_argument("--row-limit", type=int, default=8000)
    args = parser.parse_args()
    hours = max(1, min(int(args.hours), 72))
    row_limit = max(100, min(int(args.row_limit), 20000))
    remote_script = REMOTE_SCRIPT.replace("__HOURS__", str(hours))
    remote_script = remote_script.replace("__ROW_LIMIT__", str(row_limit))
    remote_script = remote_script.replace("__MATURE_MINUTES__", "242")
    remote_script = remote_script.replace(
        "from sqlalchemy import text",
        "from sqlalchemy import text\nfrom db.session import get_read_session_ctx",
    )
    encoded = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode({encoded!r}))"
    command = " ".join(
        (
            "systemd-run --quiet --wait --pipe --collect",
            "--unit=bb-horizon-quarantine-reasons-audit",
            "--property=WorkingDirectory=/data/bb/app",
            "--property=User=bb",
            "--property=EnvironmentFile=-/data/bb/app/.env",
            "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
            "--property=RuntimeMaxSec=90s",
            "--setenv=DATABASE_POOL_SIZE=1",
            "--setenv=DATABASE_MAX_OVERFLOW=0",
            "/data/bb/app/.venv/bin/python -c",
            shlex.quote(runner),
        )
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(ssh, command, timeout=110, max_output_chars=80000)
    finally:
        ssh.close()
    safe_print(json.dumps(json.loads(output), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
