#!/usr/bin/env python3
"""Create online-safe indexes for OKX reconciliation and dashboard audits."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from db.session import get_engine  # noqa: E402

INDEX_DDLS = (
    (
        "idx_positions_closed_scan",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_closed_scan "
        "ON positions (is_open, closed_at DESC, id DESC)",
    ),
    (
        "idx_positions_created_scan",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_created_scan "
        "ON positions (created_at DESC, id DESC)",
    ),
    (
        "idx_positions_open_created_scan",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_open_created_scan "
        "ON positions (is_open, created_at DESC, id DESC)",
    ),
    (
        "idx_orders_filled_exchange_scan",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_filled_exchange_scan "
        "ON orders (status, filled_at DESC, id DESC)",
    ),
    (
        "idx_orders_decision_side_scan",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_decision_side_scan "
        "ON orders (decision_id, side, status, filled_at DESC)",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="actually create the indexes")
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    print({"apply": bool(args.apply), "index_count": len(INDEX_DDLS)})
    for name, ddl in INDEX_DDLS:
        print({"index": name, "ddl": ddl})
    if not args.apply:
        return 0

    engine = await get_engine()
    async with engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        for name, ddl in INDEX_DDLS:
            print({"creating": name})
            await conn.execute(text(ddl))
            print({"created": name})
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
