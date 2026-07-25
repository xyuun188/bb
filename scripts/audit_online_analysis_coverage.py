#!/usr/bin/env python3
"""Audit online market coverage and position-review continuity."""

# ruff: noqa: S608 - the remote template uses SQLAlchemy expressions, not raw SQL.

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402


def _remote_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _remote_script(*, remote_app_dir: str, window_minutes: int) -> str:
    return f"""
import asyncio
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.runtime_env_bootstrap import (
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

root = Path({remote_app_dir!r})
load_runtime_env_files(project_root=root)
drop_privileges_to_runtime_user_if_needed(project_root=root)

from sqlalchemy import func, select
from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.trade import Position

WINDOW_MINUTES = {max(int(window_minutes), 1)!r}


def service_status(name):
    output = subprocess.check_output(
        [
            "systemctl",
            "show",
            name,
            "--property=ActiveState",
            "--property=SubState",
            "--property=NRestarts",
            "--property=ActiveEnterTimestamp",
        ],
        text=True,
    )
    values = {{}}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    active_enter_at = values.get("ActiveEnterTimestamp")
    if active_enter_at:
        try:
            active_enter_at = subprocess.check_output(
                ["date", "--date", active_enter_at, "--iso-8601=seconds"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            active_enter_at = None
    return {{
        "active_state": values.get("ActiveState"),
        "sub_state": values.get("SubState"),
        "restart_count": int(values.get("NRestarts") or 0),
        "active_enter_at": active_enter_at or None,
    }}


def as_utc(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


async def run():
    now = datetime.now(UTC)
    since = now - timedelta(minutes=WINDOW_MINUTES)
    runtime_path = root / "data" / "trading_runtime_status.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    async with get_read_session_ctx() as session:
        rows = (
            await session.execute(
                select(
                    AIDecision.analysis_type,
                    AIDecision.symbol,
                    AIDecision.created_at,
                )
                .where(
                    AIDecision.is_paper.is_(True),
                    AIDecision.created_at >= since.replace(tzinfo=None),
                )
                .order_by(AIDecision.created_at.asc(), AIDecision.id.asc())
            )
        ).all()
        open_position_count = int(
            (
                await session.execute(
                    select(func.count(Position.id)).where(
                        Position.execution_mode == "paper",
                        Position.is_open.is_(True),
                    )
                )
            ).scalar()
            or 0
        )

    market_rows = [row for row in rows if str(row.analysis_type or "") == "market"]
    position_rows = [row for row in rows if str(row.analysis_type or "") == "position"]
    market_times = [as_utc(row.created_at) for row in market_rows]
    position_times = [as_utc(row.created_at) for row in position_rows]
    activity_points = [since, *market_times, now]
    max_activity_gap = max(
        (
            (current - previous).total_seconds()
            for previous, current in zip(activity_points, activity_points[1:])
        ),
        default=(now - since).total_seconds(),
    )
    market_symbol_counts = Counter(str(row.symbol or "") for row in market_rows)
    top_symbol, top_count = market_symbol_counts.most_common(1)[0] if market_symbol_counts else ("", 0)
    payload = {{
        "generated_at": now.isoformat(),
        "read_only": True,
        "window_minutes": WINDOW_MINUTES,
        "window_started_at": since.isoformat(),
        "mode": runtime.get("mode"),
        "paused": runtime.get("paused"),
        "services": {{
            "trading": service_status("bb-paper-trading.service"),
            "dashboard": service_status("bb-dashboard.service"),
        }},
        "coverage": runtime.get("market_analysis_deferred") or {{}},
        "market": {{
            "decision_count": len(market_rows),
            "distinct_symbol_count": len(market_symbol_counts),
            "first_decision_at": market_times[0].isoformat() if market_times else None,
            "last_decision_at": market_times[-1].isoformat() if market_times else None,
            "max_activity_gap_seconds": round(max_activity_gap, 3),
            "top_symbol": top_symbol or None,
            "top_symbol_count": top_count,
            "top_symbol_share": round(top_count / len(market_rows), 6) if market_rows else None,
            "symbol_counts": dict(market_symbol_counts.most_common()),
        }},
        "position_review": {{
            "open_position_count": open_position_count,
            "decision_count": len(position_rows),
            "distinct_symbol_count": len({{str(row.symbol or "") for row in position_rows}}),
            "first_decision_at": position_times[0].isoformat() if position_times else None,
            "last_decision_at": position_times[-1].isoformat() if position_times else None,
        }},
    }}
    print(json.dumps(payload, ensure_ascii=False, default=str))


asyncio.run(run())
"""


def _decode_remote_json(output: str) -> dict[str, Any]:
    for line in reversed(str(output or "").splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("online analysis coverage audit returned no JSON object")


def assess_coverage_report(
    report: dict[str, Any],
    *,
    minimum_window_minutes: int = 30,
    max_activity_gap_seconds: float = 10 * 60,
    candidate_selection_stale_seconds: float = 10 * 60,
    position_stale_seconds: float = 5 * 60,
    now: datetime | None = None,
) -> dict[str, Any]:
    observed_at = now or _parse_datetime(report.get("generated_at")) or datetime.now(UTC)
    blockers: list[str] = []
    window_minutes = int(report.get("window_minutes") or 0)
    window_started_at = _parse_datetime(report.get("window_started_at"))
    if window_minutes < max(int(minimum_window_minutes), 1):
        blockers.append("observation_window_too_short")
    if window_started_at is None:
        blockers.append("observation_window_start_missing")
    services = report.get("services") if isinstance(report.get("services"), dict) else {}
    for name in ("trading", "dashboard"):
        status = services.get(name) if isinstance(services.get(name), dict) else {}
        if status.get("active_state") != "active" or status.get("sub_state") != "running":
            blockers.append(f"{name}_service_not_running")
        if int(status.get("restart_count") or 0) > 0:
            blockers.append(f"{name}_service_restarted")
        active_enter_at = _parse_datetime(status.get("active_enter_at"))
        if (
            window_started_at is None
            or active_enter_at is None
            or active_enter_at > window_started_at
        ):
            blockers.append(f"{name}_service_continuity_unproven")
    if report.get("mode") != "paper":
        blockers.append("execution_mode_not_paper")
    if report.get("paused") is True:
        blockers.append("paper_trading_paused")
    elif report.get("paused") is not False:
        blockers.append("paper_trading_pause_state_unknown")

    coverage = report.get("coverage") if isinstance(report.get("coverage"), dict) else {}
    if coverage.get("monitoring_active") is not True:
        blockers.append("market_coverage_monitoring_inactive")
    if coverage.get("coverage_window_evaluable") is not True:
        blockers.append("market_coverage_window_not_evaluable")
    if coverage.get("coverage_window_met") is not True:
        blockers.append("market_coverage_window_not_met")
    if int(coverage.get("overdue_count") or 0) > 0:
        blockers.append("market_coverage_has_overdue_symbols")
    if coverage.get("monitoring_active") is True:
        if coverage.get("candidate_coverage_evidence_available") is not True:
            blockers.append("market_candidate_coverage_evidence_missing")
        selection_generated_at = _parse_datetime(coverage.get("candidate_selection_generated_at"))
        if selection_generated_at is None or (
            observed_at - selection_generated_at
        ).total_seconds() > max(float(candidate_selection_stale_seconds), 1.0):
            blockers.append("market_candidate_coverage_evidence_stale")
        if int(coverage.get("coverage_due_count") or 0) > 0:
            blockers.append("market_candidate_coverage_has_unresolved_due_symbols")
        candidate_target_seconds = float(
            coverage.get("candidate_coverage_target_seconds") or float("inf")
        )
        if candidate_target_seconds > max(window_minutes * 60, 1):
            blockers.append("market_candidate_coverage_target_exceeds_window")

    market = report.get("market") if isinstance(report.get("market"), dict) else {}
    if int(market.get("decision_count") or 0) <= 0:
        blockers.append("market_analysis_activity_missing")
    if float(market.get("max_activity_gap_seconds") or float("inf")) > max(
        float(max_activity_gap_seconds), 1.0
    ):
        blockers.append("market_analysis_activity_gap_too_large")

    position = (
        report.get("position_review") if isinstance(report.get("position_review"), dict) else {}
    )
    if int(position.get("open_position_count") or 0) > 0:
        if int(position.get("decision_count") or 0) <= 0:
            blockers.append("position_review_activity_missing")
        last_position_at = _parse_datetime(position.get("last_decision_at"))
        if last_position_at is None or (observed_at - last_position_at).total_seconds() > max(
            float(position_stale_seconds), 1.0
        ):
            blockers.append("position_review_activity_stale")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "checked_at": observed_at.isoformat(),
    }


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-app-dir", default="/data/bb/app")
    parser.add_argument("--window-minutes", type=int, default=30)
    parser.add_argument("--max-activity-gap-seconds", type=float, default=10 * 60)
    parser.add_argument("--candidate-selection-stale-seconds", type=float, default=10 * 60)
    parser.add_argument("--position-stale-seconds", type=float, default=5 * 60)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--allow-not-ready",
        action="store_true",
        help="Print blockers without returning a failing process status.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remote_script = _remote_script(
        remote_app_dir=args.remote_app_dir,
        window_minutes=args.window_minutes,
    )
    command = (
        f"cd {_remote_quote(args.remote_app_dir)} && "
        "PYBIN=python3; "
        "if [ -x .venv/bin/python ]; then PYBIN=.venv/bin/python; "
        "elif [ -x venv/bin/python ]; then PYBIN=venv/bin/python; fi; "
        "$PYBIN - <<'PY'\n"
        f"{remote_script}\nPY"
    )
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        report = _decode_remote_json(
            run_remote_text(
                ssh,
                command,
                timeout=max(int(args.timeout or 1), 1),
                check=True,
            )
        )
    finally:
        ssh.close()
    report["assessment"] = assess_coverage_report(
        report,
        max_activity_gap_seconds=args.max_activity_gap_seconds,
        candidate_selection_stale_seconds=args.candidate_selection_stale_seconds,
        position_stale_seconds=args.position_stale_seconds,
    )
    safe_print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if not args.allow_not_ready and report["assessment"]["ready"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
