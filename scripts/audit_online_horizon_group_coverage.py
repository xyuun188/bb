"""Summarize bounded online shadow-label coverage by decision and horizon."""

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

REMOTE_SCRIPT = r'''
import asyncio
import json
from datetime import UTC, datetime

from sqlalchemy import text

from db.session import get_read_session_ctx

HOURS = __HOURS__


async def main():
    report = {"checked_at": datetime.now(UTC).isoformat(), "lookback_hours": HOURS}
    async with get_read_session_ctx(
        statement_timeout_ms=10000,
        transaction_isolation="REPEATABLE READ READ ONLY",
    ) as session:
        by_horizon = await session.execute(text("""
            SELECT horizon_minutes, status,
                   training_feature_snapshot->'training_market_fact_contract'->>'source'
                       AS source,
                   training_feature_snapshot->'training_label_contract'->>'fee_after_complete'
                       AS fee_after_complete,
                   count(*) AS rows,
                   count(DISTINCT decision_id) AS decision_groups,
                   min(created_at) AS first_decision,
                   max(created_at) AS last_decision
            FROM shadow_backtests
            WHERE execution_mode='paper'
              AND analysis_type='market'
              AND decision_id > 0
              AND horizon_minutes IN (5, 15, 60, 240)
              AND label_version='2026-07-27.market-opportunity-multitask.v2'
              AND created_at >= now() - (:hours * interval '1 hour')
            GROUP BY horizon_minutes, status,
                     training_feature_snapshot
                         ->'training_market_fact_contract'->>'source',
                     training_feature_snapshot
                         ->'training_label_contract'->>'fee_after_complete'
            ORDER BY horizon_minutes, status, source, fee_after_complete
        """), {"hours": HOURS})
        report["rows_by_horizon_status_source"] = [
            dict(row) for row in by_horizon.mappings()
        ]

        coverage = await session.execute(text("""
            WITH source_rows AS (
                SELECT decision_id, created_at, horizon_minutes, status,
                       training_feature_snapshot
                           ->'training_market_fact_contract'->>'source' AS source,
                       training_feature_snapshot
                           ->'training_label_contract'->>'fee_after_complete'
                           AS fee_after_complete
                FROM shadow_backtests
                WHERE execution_mode='paper'
                  AND analysis_type='market'
                  AND decision_id > 0
                  AND horizon_minutes IN (5, 15, 60, 240)
                  AND label_version='2026-07-27.market-opportunity-multitask.v2'
                  AND created_at >= now() - (:hours * interval '1 hour')
            ), grouped AS (
                SELECT decision_id, min(created_at) AS created_at,
                       bool_or(horizon_minutes=5) AS seen_5,
                       bool_or(horizon_minutes=15) AS seen_15,
                       bool_or(horizon_minutes=60) AS seen_60,
                       bool_or(horizon_minutes=240) AS seen_240,
                       bool_or(horizon_minutes=5 AND status='completed'
                           AND source='shadow_entry_result_and_okx_native_1m_path'
                           AND fee_after_complete='true') AS qualified_5,
                       bool_or(horizon_minutes=15 AND status='completed'
                           AND source='shadow_entry_result_and_okx_native_1m_path'
                           AND fee_after_complete='true') AS qualified_15,
                       bool_or(horizon_minutes=60 AND status='completed'
                           AND source='shadow_entry_result_and_okx_native_1m_path'
                           AND fee_after_complete='true') AS qualified_60,
                       bool_or(horizon_minutes=240 AND status='completed'
                           AND source='shadow_entry_result_and_okx_native_1m_path'
                           AND fee_after_complete='true') AS qualified_240
                FROM source_rows
                GROUP BY decision_id
            ), masks AS (
                SELECT *,
                    concat_ws(',',
                        CASE WHEN seen_5 THEN '5' END,
                        CASE WHEN seen_15 THEN '15' END,
                        CASE WHEN seen_60 THEN '60' END,
                        CASE WHEN seen_240 THEN '240' END
                    ) AS observed_mask,
                    concat_ws(',',
                        CASE WHEN qualified_5 THEN '5' END,
                        CASE WHEN qualified_15 THEN '15' END,
                        CASE WHEN qualified_60 THEN '60' END,
                        CASE WHEN qualified_240 THEN '240' END
                    ) AS qualified_mask
                FROM grouped
            )
            SELECT observed_mask, qualified_mask,
                   (created_at < now() - interval '242 minutes') AS four_hour_mature,
                   count(*) AS decision_groups,
                   count(*) FILTER (WHERE seen_5 AND seen_15 AND seen_60 AND seen_240)
                       AS all_four_rows_observed,
                   count(*) FILTER (
                       WHERE qualified_5 AND qualified_15 AND qualified_60 AND qualified_240
                   ) AS all_four_rows_qualified,
                   min(created_at) AS first_decision,
                   max(created_at) AS last_decision
            FROM masks
            GROUP BY observed_mask, qualified_mask, four_hour_mature
            ORDER BY four_hour_mature DESC, observed_mask, qualified_mask
        """), {"hours": HOURS})
        report["coverage_masks"] = [
            dict(row) for row in coverage.mappings()
        ]

        samples = await session.execute(text("""
            WITH selected AS (
                SELECT decision_id, min(created_at) AS created_at
                FROM shadow_backtests
                WHERE execution_mode='paper'
                  AND analysis_type='market'
                  AND decision_id > 0
                  AND horizon_minutes IN (5, 15, 60, 240)
                  AND label_version='2026-07-27.market-opportunity-multitask.v2'
                  AND created_at >= now() - (:hours * interval '1 hour')
                GROUP BY decision_id
                HAVING count(DISTINCT horizon_minutes) < 4
                    OR bool_or(status='quarantined')
                ORDER BY min(created_at) DESC, decision_id DESC
                LIMIT 8
            )
            SELECT selected.decision_id, selected.created_at,
                   shadow.horizon_minutes, shadow.status,
                   shadow.training_feature_snapshot
                       ->'training_market_fact_contract'->>'source' AS source,
                   shadow.training_feature_snapshot
                       ->'training_label_contract'->>'fee_after_complete'
                       AS fee_after_complete
            FROM selected
            JOIN shadow_backtests shadow
              ON shadow.decision_id=selected.decision_id
            WHERE shadow.execution_mode='paper'
              AND shadow.analysis_type='market'
              AND shadow.horizon_minutes IN (5, 15, 60, 240)
              AND shadow.label_version='2026-07-27.market-opportunity-multitask.v2'
            ORDER BY selected.created_at DESC, selected.decision_id DESC,
                     shadow.horizon_minutes
        """), {"hours": HOURS})
        report["bounded_incomplete_group_samples"] = [
            dict(row) for row in samples.mappings()
        ]
    print(json.dumps(report, ensure_ascii=False, default=str))


asyncio.run(main())
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=12)
    args = parser.parse_args()
    hours = max(1, min(int(args.hours), 72))
    remote_script = REMOTE_SCRIPT.replace("__HOURS__", str(hours))
    encoded = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode({encoded!r}))"
    command = " ".join(
        (
            "systemd-run --quiet --wait --pipe --collect",
            "--unit=bb-horizon-coverage-audit",
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
    report = json.loads(output)
    safe_print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
