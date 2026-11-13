#!/usr/bin/env python3
"""Create or roll back online-safe reconciliation and training indexes."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncConnection  # noqa: E402

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

from db.session import (  # noqa: E402
    SHADOW_CRYPTO_TRAINING_INDEX_DDL,
    SHADOW_CRYPTO_TRAINING_INDEX_NAME,
    SHADOW_EXECUTABLE_TRAINING_INDEX_DDL,
    SHADOW_EXECUTABLE_TRAINING_INDEX_NAME,
    close_db,
    get_engine,
)

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
TRAINING_INDEX_DDLS = (
    (
        SHADOW_EXECUTABLE_TRAINING_INDEX_NAME,
        SHADOW_EXECUTABLE_TRAINING_INDEX_DDL.replace(
            "CREATE INDEX ", "CREATE INDEX CONCURRENTLY ", 1
        ),
    ),
    (
        SHADOW_CRYPTO_TRAINING_INDEX_NAME,
        SHADOW_CRYPTO_TRAINING_INDEX_DDL.replace(
            "CREATE INDEX ", "CREATE INDEX CONCURRENTLY ", 1
        ),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="actually create the indexes")
    parser.add_argument("--scope", choices=("reconciliation", "training"), default="reconciliation")
    parser.add_argument("--rollback", action="store_true", help="drop only the selected indexes")
    return parser.parse_args()


async def _apply_indexes(
    conn: AsyncConnection, indexes: tuple[tuple[str, str], ...], *, rollback: bool
) -> None:
    for name, ddl in indexes:
        if rollback:
            await conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {name}"))
            print({"dropped": name})
            continue
        validity = await conn.execute(
            text(
                "SELECT indisvalid FROM pg_index "
                "WHERE indexrelid = to_regclass(:index_name)"
            ),
            {"index_name": name},
        )
        if validity.scalar() is False:
            # A cancelled concurrent build leaves an unusable index with the
            # same name; IF NOT EXISTS alone would silently keep it forever.
            await conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {name}"))
        print({"creating": name})
        await conn.execute(text(ddl))
        print({"created": name})


async def _main() -> int:
    args = _parse_args()
    indexes = TRAINING_INDEX_DDLS if args.scope == "training" else INDEX_DDLS
    print({
        "apply": bool(args.apply), "rollback": bool(args.rollback),
        "scope": args.scope, "index_count": len(indexes),
    })
    for name, ddl in indexes:
        if args.rollback:
            ddl = f"DROP INDEX CONCURRENTLY IF EXISTS {name}"
        print({"index": name, "ddl": ddl})
    if not args.apply:
        return 0

    engine = await get_engine()
    try:
        if engine.dialect.name != "postgresql":
            raise RuntimeError("online concurrent indexes require PostgreSQL")
        async with engine.connect() as conn:
            await conn.execution_options(isolation_level="AUTOCOMMIT")
            await conn.execute(text("SET lock_timeout = '5s'"))
            await conn.execute(text("SET statement_timeout = '10min'"))
            await conn.execute(text("SET maintenance_work_mem = '64MB'"))
            await conn.execute(text("SET max_parallel_maintenance_workers = 1"))
            await _apply_indexes(conn, indexes, rollback=args.rollback)
    finally:
        await close_db()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
