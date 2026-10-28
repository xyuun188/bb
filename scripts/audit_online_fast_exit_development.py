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
    statement = (
        select(
            ShadowBacktest.id,
            ShadowBacktest.decision_id,
            ShadowBacktest.created_at,
            ShadowBacktest.symbol,
            ShadowBacktest.training_feature_snapshot,
            ShadowBacktest.long_return_pct,
            ShadowBacktest.short_return_pct,
            ShadowBacktest.raw_llm_response["paper_trade_selection"]["selected"].as_boolean()
            .label("selection_selected"),
            ShadowBacktest.raw_llm_response["paper_trade_selection"]["selected_side"].as_string()
            .label("selected_side"),
        )
        .where(
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
        )
        .order_by(ShadowBacktest.created_at, ShadowBacktest.id)
        .limit(LIMIT)
    )
    try:
        async with get_read_session_ctx(
            statement_timeout_ms=15000,
            transaction_isolation="REPEATABLE READ READ ONLY",
        ) as session:
            result = await session.execute(statement)
            source_rows = list(result.mappings())
        rows = []
        for row in source_rows:
            if row.get("selection_selected") is not True:
                continue
            snapshot = row.get("training_feature_snapshot")
            snapshot = snapshot if isinstance(snapshot, dict) else {}
            label = snapshot.get("training_label_contract")
            label = label if isinstance(label, dict) else {}
            decision_id = int(row.get("decision_id") or 0) or None
            rows.append(
                {
                    "id": row.get("id"),
                    "decision_id": decision_id,
                    "decision_at": row.get("created_at"),
                    "symbol": row.get("symbol"),
                    "selected_side": row.get("selected_side"),
                    "long_mfe_pct": label.get("long_mfe_pct"),
                    "short_mfe_pct": label.get("short_mfe_pct"),
                    "long_gross_return_pct": label.get("long_gross_return_pct"),
                    "short_gross_return_pct": label.get("short_gross_return_pct"),
                    "long_net_return_pct": label.get(
                        "long_net_return_after_all_cost_pct"
                    ),
                    "short_net_return_pct": label.get(
                        "short_net_return_after_all_cost_pct"
                    ),
                }
            )
        report = build_fast_exit_shadow_report(
            rows,
            window={
                "since": since.isoformat(),
                "cutoff_exclusive": CUTOFF.isoformat(),
                "hours": HOURS,
                "source": NATIVE_SOURCE,
                "horizon_minutes": 5,
                "decision_group_limit": LIMIT,
                "filter": (
                    "completed paper rows, native 1m path provenance, complete "
                    "fee-after labels, and selected paper training side"
                ),
                "independent_exam": False,
            },
        )
        report["checked_at"] = datetime.now(UTC).isoformat()
        report["input"].update(
            {
                "source_row_count": len(source_rows),
                "selected_candidate_row_count": len(rows),
                "dependency_sha256": dependency_hashes(),
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
