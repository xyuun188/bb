"""Run a bounded, read-only fast-exit development comparison online."""

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

DEPENDENCIES = ("services/fast_exit_shadow_replay.py",)
SCAN_BATCH_SIZE = 250
MAX_SCAN_MULTIPLIER = 20
MAX_SCAN_ROWS = 50_000

REMOTE_SCRIPT = r"""
import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from db.session import close_db, get_read_session_ctx
from models.learning import ShadowBacktest
from services.fast_exit_shadow_replay import build_fast_exit_shadow_report

HOURS = __HOURS__
LIMIT = __LIMIT__
SCAN_BATCH_SIZE = __SCAN_BATCH_SIZE__
MAX_SCAN_MULTIPLIER = __MAX_SCAN_MULTIPLIER__
MAX_SCAN_ROWS = __MAX_SCAN_ROWS__
CUTOFF = datetime.fromisoformat(__CUTOFF__.replace("Z", "+00:00")).astimezone(UTC)
DEPENDENCIES = __DEPENDENCIES__
NATIVE_SOURCE = "shadow_entry_result_and_okx_native_1m_path"


def dependency_hashes():
    return {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in DEPENDENCIES
    }


async def main():
    since = CUTOFF - timedelta(hours=HOURS)
    try:
        async with get_read_session_ctx(
            statement_timeout_ms=15000,
            transaction_isolation="REPEATABLE READ READ ONLY",
        ) as session:
            rows = []
            scanned = 0
            source_row_count = 0
            cursor_updated_at = None
            cursor_id = None
            scan_limit = min(max(LIMIT * MAX_SCAN_MULTIPLIER, SCAN_BATCH_SIZE), MAX_SCAN_ROWS)
            while scanned < scan_limit and len(rows) < LIMIT:
                filters = [
                    ShadowBacktest.execution_mode == "paper",
                    ShadowBacktest.analysis_type == "market",
                    ShadowBacktest.status == "completed",
                    ShadowBacktest.horizon_minutes == 5,
                    ShadowBacktest.updated_at >= since,
                    ShadowBacktest.updated_at < CUTOFF,
                    ShadowBacktest.training_feature_snapshot[
                        "training_market_fact_contract"
                    ]["source"].as_string()
                    == NATIVE_SOURCE,
                    ShadowBacktest.training_feature_snapshot[
                        "training_label_contract"
                    ]["fee_after_complete"].as_boolean()
                    .is_(True),
                    ShadowBacktest.training_feature_snapshot[
                        "training_label_contract"
                    ]["long_mfe_pct"].is_not(None),
                    ShadowBacktest.training_feature_snapshot[
                        "training_label_contract"
                    ]["short_mfe_pct"].is_not(None),
                    ShadowBacktest.raw_llm_response[
                        "paper_trade_selection"
                    ]["selected"].as_boolean()
                    .is_(True),
                ]
                if cursor_updated_at is not None and cursor_id is not None:
                    filters.append(
                        (ShadowBacktest.updated_at < cursor_updated_at)
                        | (
                            (ShadowBacktest.updated_at == cursor_updated_at)
                            & (ShadowBacktest.id < cursor_id)
                        )
                    )
                statement = (
                    select(
                        ShadowBacktest.id,
                        ShadowBacktest.decision_id,
                        ShadowBacktest.created_at,
                        ShadowBacktest.updated_at,
                        ShadowBacktest.symbol,
                        ShadowBacktest.training_feature_snapshot[
                            "training_label_contract"
                        ]["long_mfe_pct"].label("long_mfe_pct"),
                        ShadowBacktest.training_feature_snapshot[
                            "training_label_contract"
                        ]["short_mfe_pct"].label("short_mfe_pct"),
                        ShadowBacktest.training_feature_snapshot[
                            "training_label_contract"
                        ]["long_gross_return_pct"].label("long_gross_return_pct"),
                        ShadowBacktest.training_feature_snapshot[
                            "training_label_contract"
                        ]["short_gross_return_pct"].label("short_gross_return_pct"),
                        ShadowBacktest.training_feature_snapshot[
                            "training_label_contract"
                        ]["long_net_return_after_all_cost_pct"].label("long_net_return_pct"),
                        ShadowBacktest.training_feature_snapshot[
                            "training_label_contract"
                        ]["short_net_return_after_all_cost_pct"].label("short_net_return_pct"),
                        ShadowBacktest.raw_llm_response[
                            "paper_trade_selection"
                        ]["selected"].as_boolean().label("selection_selected"),
                        ShadowBacktest.raw_llm_response[
                            "paper_trade_selection"
                        ]["selected_side"].as_string().label("selected_side"),
                    )
                    .where(*filters)
                    .order_by(ShadowBacktest.updated_at.desc(), ShadowBacktest.id.desc())
                    .limit(min(SCAN_BATCH_SIZE, scan_limit - scanned))
                )
                result = await session.execute(statement)
                batch = list(result.mappings())
                if not batch:
                    break
                scanned += len(batch)
                source_row_count += len(batch)
                cursor_updated_at = batch[-1]["updated_at"]
                cursor_id = batch[-1]["id"]
                for row in batch:
                    decision_id = int(row.get("decision_id") or 0) or None
                    rows.append(
                        {
                            "id": row.get("id"),
                            "decision_id": decision_id,
                            "decision_at": row.get("created_at"),
                            "symbol": row.get("symbol"),
                            "selected_side": row.get("selected_side"),
                            "long_mfe_pct": row.get("long_mfe_pct"),
                            "short_mfe_pct": row.get("short_mfe_pct"),
                            "long_gross_return_pct": row.get("long_gross_return_pct"),
                            "short_gross_return_pct": row.get("short_gross_return_pct"),
                            "long_net_return_pct": row.get("long_net_return_pct"),
                            "short_net_return_pct": row.get("short_net_return_pct"),
                        }
                    )
                    if len(rows) >= LIMIT:
                        break
            rows.reverse()
        report = build_fast_exit_shadow_report(
            rows,
            window={
                "since": since.isoformat(),
                "cutoff_exclusive": CUTOFF.isoformat(),
                "hours": HOURS,
                "source": NATIVE_SOURCE,
                "horizon_minutes": 5,
                "decision_group_limit": LIMIT,
                "time_basis": (
                    "outcome_updated_at within [since, cutoff); updated_at is pagination key"
                ),
                "filter": (
                    "completed paper rows updated within the window, native 1m path provenance, complete "
                    "fee-after labels, and selected paper training side; all filters are applied in SQL"
                ),
                "independent_exam": False,
            },
        )
        report["checked_at"] = datetime.now(UTC).isoformat()
        report["input"].update(
            {
                "source_row_count": source_row_count,
                "selected_candidate_row_count": len(rows),
                "dependency_sha256": dependency_hashes(),
                "query_rows_scanned": scanned,
                "query_scan_limit": scan_limit,
                "query_scan_limit_reached": scanned >= scan_limit and len(rows) < LIMIT,
                "query_strategy": (
                    "updated_at_index_keyset_pages_with_sql_contract_filters"
                ),
            }
        )
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, default=str))
    finally:
        await close_db()


asyncio.run(main())
"""


def _render_remote_script(*, hours: int, limit: int, cutoff: datetime) -> str:
    return (
        REMOTE_SCRIPT.replace("__HOURS__", str(hours))
        .replace("__LIMIT__", str(limit))
        .replace("__SCAN_BATCH_SIZE__", str(SCAN_BATCH_SIZE))
        .replace("__MAX_SCAN_MULTIPLIER__", str(MAX_SCAN_MULTIPLIER))
        .replace("__MAX_SCAN_ROWS__", str(MAX_SCAN_ROWS))
        .replace("__CUTOFF__", repr(cutoff.isoformat().replace("+00:00", "Z")))
        .replace("__DEPENDENCIES__", repr(DEPENDENCIES))
    )


def _local_dependency_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in DEPENDENCIES
    }


def _build_remote_command(
    *,
    hours: int,
    limit: int,
    cutoff: datetime,
    unit: str,
) -> str:
    script = _render_remote_script(hours=hours, limit=limit, cutoff=cutoff)
    payload = base64.b64encode(script.encode("utf-8")).decode("ascii")
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
            "--property=TimeoutStopSec=15s",
            "--property=KillMode=control-group",
            "--property=EnvironmentFile=-/data/bb/app/.env",
            "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
            "--setenv=DATABASE_POOL_SIZE=1",
            "--setenv=DATABASE_MAX_OVERFLOW=0",
            "/data/bb/app/.venv/bin/python -c",
            shlex.quote(runner),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--cutoff", help="exclusive UTC cutoff; defaults to now")
    args = parser.parse_args()
    hours = int(args.hours)
    limit = int(args.limit)
    if not 1 <= hours <= 168:
        parser.error("--hours must be between 1 and 168")
    if not 1 <= limit <= 2500:
        parser.error("--limit must be between 1 and 2500")
    try:
        cutoff = (
            datetime.fromisoformat(args.cutoff.replace("Z", "+00:00"))
            if args.cutoff
            else datetime.now(UTC)
        )
    except ValueError as exc:
        parser.error(f"invalid --cutoff: {exc}")
    if cutoff.tzinfo is None:
        parser.error("--cutoff must include a timezone")
    cutoff = cutoff.astimezone(UTC)
    stamp = cutoff.strftime("%Y%m%dT%H%M%SZ")
    unit = f"bb-fast-exit-development-{stamp}"
    report_path = (
        ROOT
        / "data"
        / "profitability_remediation"
        / f"online-fast-exit-development-{stamp}.json"
    )
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    ssh = connect_remote_ssh(ROOT, timeout=25)
    sftp = ssh.open_sftp()
    remote_result = (
        "/data/bb/app/tmp/codex-fast-exit-development/"
        f"result_{secrets.token_hex(12)}.json"
    )
    try:
        command = (
            "set -e\numask 077\n"
            "install -d -m 0700 /data/bb/app/tmp/codex-fast-exit-development\n"
            f"{_build_remote_command(hours=hours, limit=limit, cutoff=cutoff, unit=unit)}"
            f" > {shlex.quote(remote_result)}"
        )
        run_remote_text(ssh, command, timeout=320, max_output_chars=4000)
        with sftp.file(remote_result, "r") as remote_file:
            report = json.loads(remote_file.read().decode("utf-8"))
        if report.get("input", {}).get("dependency_sha256") != _local_dependency_hashes():
            raise RuntimeError("online fast-exit replay dependencies differ from local")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        with suppress(OSError):
            sftp.remove(remote_result)
        sftp.close()
        ssh.close()
    safe_print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"saved_report={report_path}")


if __name__ == "__main__":
    main()
