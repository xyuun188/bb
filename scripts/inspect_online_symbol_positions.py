#!/usr/bin/env python3
"""Inspect online position/order lineage for one normalized trading symbol."""

# ruff: noqa: S608

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402


def _quote(value: str) -> str:
    return shlex.quote(str(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--exchange-order-id", action="append", default=[])
    args = parser.parse_args()
    remote_code = f"""  # noqa: S608
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0, "/data/bb/app")
from scripts.runtime_env_bootstrap import load_runtime_env_files
load_runtime_env_files(project_root=Path("/data/bb/app"))
from sqlalchemy import select
from db.session import get_read_session_ctx
from models.trade import OkxPositionHistory, Order, Position

async def main():
    symbol = {str(args.symbol).strip()!r}
    order_ids = {tuple(str(value).strip() for value in args.exchange_order_id if str(value).strip())!r}
    async with get_read_session_ctx() as session:
        positions = list((await session.execute(
            select(Position).where(Position.symbol == symbol).order_by(Position.created_at.desc(), Position.id.desc())
        )).scalars().all())
        orders = list((await session.execute(
            select(Order).where(Order.symbol == symbol).order_by(Order.filled_at.desc(), Order.id.desc())
        )).scalars().all())
        history = list((await session.execute(
            select(OkxPositionHistory)
            .where(OkxPositionHistory.inst_id == symbol.replace("/", "-") + "-SWAP")
            .order_by(OkxPositionHistory.updated_at_okx.desc())
        )).scalars().all())
    def position_row(row):
        return {{
            "id": row.id, "symbol": row.symbol, "side": row.side,
            "quantity": row.quantity, "entry_price": row.entry_price,
            "entry_exchange_order_id": row.entry_exchange_order_id,
            "close_exchange_order_id": row.close_exchange_order_id,
            "okx_inst_id": row.okx_inst_id, "okx_pos_id": row.okx_pos_id,
            "is_open": row.is_open, "created_at": str(row.created_at),
            "closed_at": str(row.closed_at), "realized_pnl": row.realized_pnl,
        }}
    def order_row(row):
        return {{
            "id": row.id, "side": row.side, "status": row.status,
            "quantity": row.quantity, "price": row.price,
            "decision_id": row.decision_id, "exchange_order_id": row.exchange_order_id,
            "okx_sync_status": row.okx_sync_status, "filled_at": str(row.filled_at),
            "created_at": str(row.created_at),
        }}
    def history_row(row):
        return {{
            "id": row.id,
            "inst_id": row.inst_id,
            "pos_id": row.pos_id,
            "pos_side": row.pos_side,
            "opened_at": str(row.opened_at),
            "updated_at_okx": str(row.updated_at_okx),
            "open_avg_px": row.open_avg_px,
            "close_avg_px": row.close_avg_px,
            "open_max_pos": row.open_max_pos,
            "close_total_pos": row.close_total_pos,
            "realized_pnl": row.realized_pnl,
            "fee": row.fee,
            "position_ids": row.position_ids,
            "match_status": row.match_status,
            "raw_row": row.raw_row,
        }}
    if order_ids:
        orders = [row for row in orders if any(item in str(row.exchange_order_id or "") for item in order_ids)]
    print(json.dumps({{"symbol": symbol, "positions": [position_row(row) for row in positions], "orders": [order_row(row) for row in orders], "history": [history_row(row) for row in history]}}, ensure_ascii=False, default=str))

asyncio.run(main())
"""
    command = (
        "cd /data/bb/app && runuser -u bb -- /bin/bash -lc "
        + _quote(
            "export DATABASE_URL='postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql'; "
            + ".venv/bin/python -c "
            + _quote(remote_code)
        )
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        print(run_remote_text(ssh, command, timeout=120, check=True, max_output_chars=120_000))
    finally:
        ssh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
