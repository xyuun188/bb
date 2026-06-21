"""Close historical zero-quantity active order rows.

The live executor can return tracking-only results while waiting for OKX to
fill an existing order. Older versions persisted those tracking events as
active orders with ``quantity <= 0``. They are not executable orders and can
pollute dashboard counts or duplicate-order checks.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from db.session import get_session_ctx  # noqa: E402

ACTIVE_STATUSES = ("open", "pending", "partial")


async def repair_zero_quantity_active_orders(*, dry_run: bool = False) -> dict[str, int | bool]:
    async with get_session_ctx() as session:
        count_stmt = text("""
            SELECT count(*)
            FROM orders
            WHERE quantity <= 0
              AND status IN ('open', 'pending', 'partial')
            """)
        affected = int((await session.execute(count_stmt)).scalar() or 0)
        if dry_run or affected <= 0:
            return {"affected": affected, "dry_run": dry_run, "updated": 0}

        update_stmt = text("""
            UPDATE orders
            SET status = 'cancelled',
                filled_at = COALESCE(filled_at, now())
            WHERE quantity <= 0
              AND status IN ('open', 'pending', 'partial')
            """)
        result = await session.execute(update_stmt)
        updated = int(result.rowcount or 0)
        return {"affected": affected, "dry_run": False, "updated": updated}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark historical zero-quantity active order rows as cancelled."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(repair_zero_quantity_active_orders(dry_run=args.dry_run))
    print(result)


if __name__ == "__main__":
    main()
