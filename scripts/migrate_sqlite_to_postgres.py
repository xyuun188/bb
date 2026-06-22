#!/usr/bin/env python3
"""Migrate the local SQLite trading database into PostgreSQL.

The script is intentionally one-way and scoped: it creates the configured
SQLAlchemy tables in the target database, optionally clears those target tables,
and bulk-inserts rows from the SQLite source. It never drops databases or touches
schemas outside SQLAlchemy metadata for this project.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, insert, select, text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import db.session as session_module  # noqa: E402
import models  # noqa: E402, F401 - register all ORM tables
from config.settings import settings  # noqa: E402
from db.session import close_db, get_engine, init_db  # noqa: E402
from models.base import Base  # noqa: E402

DEFAULT_SQLITE_PATH = ROOT / "data" / "trading.db"
JSON_COLUMN_TYPE_NAMES = {"JSON", "JSONB"}
DATETIME_COLUMN_TYPE_NAMES = {"DATETIME", "TIMESTAMP"}


def _table_names() -> list[str]:
    return [table.name for table in Base.metadata.sorted_tables]


def _sqlite_table_identifier(table_name: str) -> str:
    known_tables = set(_table_names())
    if table_name not in known_tables:
        raise ValueError(f"Unknown SQLAlchemy table: {table_name}")
    if not table_name.replace("_", "").isalnum():
        raise ValueError(f"Unsafe SQLite table identifier: {table_name}")
    return f'"{table_name}"'


def _load_json(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str) or value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _load_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        return datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text_value, fmt)
            except ValueError:
                continue
    return None


def _column_type_name(column: Any) -> str:
    return column.type.__class__.__name__.upper()


def _coerce_row(table: Any, row: sqlite3.Row) -> dict[str, Any]:
    result: dict[str, Any] = {}
    row_keys = set(row.keys())
    for column in table.columns:
        if column.name not in row_keys:
            continue
        value = row[column.name]
        type_name = _column_type_name(column)
        if type_name in JSON_COLUMN_TYPE_NAMES:
            value = _load_json(value)
        elif type_name in DATETIME_COLUMN_TYPE_NAMES:
            value = _load_datetime(value)
        result[column.name] = value
    return result


def _chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _sqlite_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(row[0]) for row in rows}


def _count_sqlite(conn: sqlite3.Connection, table_name: str) -> int:
    table_identifier = _sqlite_table_identifier(table_name)
    query = f"SELECT COUNT(*) FROM {table_identifier}"  # noqa: S608
    return int(conn.execute(query).fetchone()[0])


async def _count_target(table: Any) -> int:
    engine = await get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(select(func.count()).select_from(table))
        return int(result.scalar_one())


async def _clear_target_tables() -> None:
    engine = await get_engine()
    async with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(delete(table))


async def _reset_postgres_sequences() -> None:
    if "postgresql" not in settings.database_url:
        return
    engine = await get_engine()
    async with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if "id" not in table.c:
                continue
            max_result = await conn.execute(select(func.max(table.c.id)))
            max_id = int(max_result.scalar_one_or_none() or 0)
            await conn.execute(
                text(
                    "SELECT setval("
                    "pg_get_serial_sequence(:table_name, 'id'), "
                    ":sequence_value, "
                    ":is_called)"
                ),
                {
                    "table_name": table.name,
                    "sequence_value": max(max_id, 1),
                    "is_called": max_id > 0,
                },
            )


async def migrate_sqlite_to_postgres(
    sqlite_path: Path,
    *,
    replace: bool,
    batch_size: int,
    include_tables: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    if not await asyncio.to_thread(sqlite_path.exists):
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
    if "sqlite" in settings.database_url:
        raise ValueError("Target DATABASE_URL must be PostgreSQL, not SQLite.")

    await init_db()
    if replace:
        await _clear_target_tables()

    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    summary: dict[str, dict[str, int]] = {}
    try:
        available_tables = _sqlite_tables(sqlite_conn)
        engine = await get_engine()
        for table in Base.metadata.sorted_tables:
            if include_tables is not None and table.name not in include_tables:
                continue
            if table.name not in available_tables:
                summary[table.name] = {
                    "source": 0,
                    "inserted": 0,
                    "target": await _count_target(table),
                }
                continue

            source_count = _count_sqlite(sqlite_conn, table.name)
            inserted_count = 0
            offset = 0
            table_identifier = _sqlite_table_identifier(table.name)
            while True:
                query = (
                    f"SELECT * FROM {table_identifier} ORDER BY id LIMIT ? OFFSET ?"  # noqa: S608
                )
                rows = sqlite_conn.execute(query, (batch_size, offset)).fetchall()
                if not rows:
                    break
                payload = [_coerce_row(table, row) for row in rows]
                payload = [item for item in payload if item]
                if payload:
                    async with engine.begin() as conn:
                        for chunk in _chunks(payload, batch_size):
                            await conn.execute(insert(table), chunk)
                            inserted_count += len(chunk)
                offset += len(rows)
                print(
                    f"{table.name}: inserted {inserted_count}/{source_count}",
                    flush=True,
                )

            target_count = await _count_target(table)
            summary[table.name] = {
                "source": source_count,
                "inserted": inserted_count,
                "target": target_count,
            }
        await _reset_postgres_sequences()
    finally:
        sqlite_conn.close()
        await close_db()
    return summary


def _set_target_database_url(database_url: str) -> None:
    os.environ["DATABASE_URL"] = database_url
    settings.database_url = database_url
    session_module._engine = None
    session_module._sessionmaker = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--tables",
        default="",
        help="Comma-separated table allowlist. Defaults to every ORM table.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    _set_target_database_url(args.database_url)
    include_tables = {item.strip() for item in args.tables.split(",") if item.strip()} or None
    unknown = include_tables.difference(_table_names()) if include_tables else set()
    if unknown:
        raise ValueError(f"Unknown tables: {', '.join(sorted(unknown))}")

    summary = await migrate_sqlite_to_postgres(
        args.sqlite,
        replace=args.replace,
        batch_size=args.batch_size,
        include_tables=include_tables,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(async_main())
