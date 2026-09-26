"""Read-only online 48-hour PnL and system-audit inspection."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402


REMOTE_CODE = r"""
import asyncio
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

APP_ROOT = Path("/data/bb/app")
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

pid = int(
    subprocess.check_output(
        ["systemctl", "show", "--property=MainPID", "--value", "bb-dashboard.service"],
        text=True,
    ).strip()
    or "0"
)
for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
    key, separator, value = item.partition(b"=")
    if separator and key:
        os.environ[key.decode("utf-8", errors="surrogateescape")] = value.decode(
            "utf-8", errors="surrogateescape"
        )

from db.session import get_read_session_ctx
from models.trade import OkxPositionHistory
from sqlalchemy import select
from web_dashboard.api.system_audit import collect_system_audit_status

WINDOW_HOURS = __WINDOW_HOURS__
since = datetime.now(UTC) - timedelta(hours=WINDOW_HOURS)
beijing = ZoneInfo("Asia/Shanghai")


def _as_utc(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


async def main():
    async with get_read_session_ctx(statement_timeout_ms=30000) as session:
        rows = list(
            (
                await session.execute(
                    select(OkxPositionHistory).where(
                        OkxPositionHistory.mode == "paper",
                        OkxPositionHistory.updated_at_okx >= since,
                    )
                )
            )
            .scalars()
            .all()
        )

    daily = defaultdict(lambda: {
        "count": 0,
        "profit_count": 0,
        "loss_count": 0,
        "realized_pnl_usdt": 0.0,
        "by_side": defaultdict(float),
        "by_symbol": defaultdict(float),
    })
    for row in rows:
        updated = _as_utc(row.updated_at_okx)
        if updated is None:
            continue
        day = updated.astimezone(beijing).date().isoformat()
        pnl = float(row.realized_pnl or 0.0)
        side = str(row.side or row.pos_side or "unknown").lower()
        symbol = str(row.symbol or row.inst_id or "unknown")
        item = daily[day]
        item["count"] += 1
        item["profit_count"] += int(pnl > 0.0)
        item["loss_count"] += int(pnl < 0.0)
        item["realized_pnl_usdt"] += pnl
        item["by_side"][side] += pnl
        item["by_symbol"][symbol] += pnl

    def normalize(value):
        if isinstance(value, defaultdict):
            value = dict(value)
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if isinstance(value, float):
            return round(value, 8)
        return value

    audit = await collect_system_audit_status(
        record_history=False,
        source="manual_48h",
        fresh_required_audits=True,
    )
    cards = []
    for card in audit.get("cards") or []:
        if not isinstance(card, dict):
            continue
        if str(card.get("status") or "").lower() in {"ok", "normal", "healthy"}:
            continue
        details = card.get("details") if isinstance(card.get("details"), dict) else {}
        cards.append(
            {
                "key": card.get("key"),
                "status": card.get("status"),
                "severity": card.get("severity"),
                "summary": card.get("summary"),
                "blockers": details.get("blockers"),
                "effective_blockers": details.get("effective_blockers"),
                "warnings": details.get("warnings"),
                "issues": details.get("issues"),
                "runtime": details.get("runtime"),
                "paper_active": details.get("paper_active"),
                "can_open_new_entries": details.get("can_open_new_entries"),
                "requires_attention": details.get("requires_attention"),
                "position_economics_incomplete_count": details.get(
                    "position_economics_incomplete_count"
                ),
                "position_economics_pending_count": details.get(
                    "position_economics_pending_count"
                ),
                "executed_dynamic_exit_contract_gap_count": details.get(
                    "executed_dynamic_exit_contract_gap_count"
                ),
                "position_economics_incomplete": details.get(
                    "position_economics_incomplete"
                ),
            }
        )

    print(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "window_hours": WINDOW_HOURS,
                "window_start_utc": since.isoformat(),
                "authoritative_history_row_count": len(rows),
                "daily_beijing": normalize(dict(sorted(daily.items()))),
                "system_audit": {
                    "checked_at": audit.get("checked_at"),
                    "status": audit.get("status"),
                    "non_normal_cards": cards,
                },
            },
            ensure_ascii=False,
        )
    )


asyncio.run(main())
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=48)
    args = parser.parse_args()
    hours = max(1, min(int(args.hours or 48), 168))
    code = REMOTE_CODE.replace("__WINDOW_HOURS__", str(hours))
    command = (
        "cd /data/bb/app && runuser -u bb -- /bin/bash -lc "
        + shlex.quote(".venv/bin/python -c " + shlex.quote(code))
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh,
            command,
            timeout=120,
            check=False,
            max_output_chars=100_000,
        )
    finally:
        ssh.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
