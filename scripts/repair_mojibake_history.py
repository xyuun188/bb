"""Repair legacy mojibake text already stored in the database.

This is a one-time/backfill tool, not a display-time workaround. It rewrites
known damaged dashboard-facing text fields after passing them through the same
sanitizer used at write time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select, update

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.session import get_session_ctx
from models.decision import AIDecision
from web_dashboard.api.text_sanitize import looks_mojibake, sanitize_payload, sanitize_text

MOJIBAKE_SQL_MARKERS = (
    "\u952b",
    "\u951b",
    "\u9286",
    "\u95ab",
    "\u95b8",
    "\u95b9",
    "\u9207",
    "\u9429",
    "\u6d5a\u72b5",
)


def _visible_mojibake_filter():
    clauses = []
    for marker in MOJIBAKE_SQL_MARKERS:
        pattern = f"%{marker}%"
        clauses.extend(
            [
                AIDecision.reasoning.like(pattern),
                AIDecision.execution_reason.like(pattern),
            ]
        )
    return or_(*clauses)


def _changed_text(value: Any) -> tuple[Any, bool]:
    if not isinstance(value, str):
        return value, False
    cleaned = sanitize_text(value)
    return cleaned, cleaned != value


def _changed_payload(value: Any) -> tuple[Any, bool]:
    if not _payload_has_mojibake(value):
        return value, False
    cleaned = sanitize_payload(value)
    return cleaned, _json_key(cleaned) != _json_key(value)


def _payload_has_mojibake(value: Any) -> bool:
    if isinstance(value, str):
        return looks_mojibake(value)
    if isinstance(value, dict):
        return any(_payload_has_mojibake(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_payload_has_mojibake(item) for item in value)
    return False


def _json_key(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return repr(value)


async def repair_ai_decisions(
    *,
    apply: bool,
    batch_size: int,
    limit: int | None,
    include_raw_json: bool,
    start_id: int,
) -> dict[str, int]:
    scanned = 0
    changed = 0
    last_id = max(int(start_id or 0), 0)
    while True:
        if limit is not None and scanned >= limit:
            break
        current_limit = min(batch_size, limit - scanned) if limit is not None else batch_size
        columns = [AIDecision.id, AIDecision.reasoning, AIDecision.execution_reason]
        if include_raw_json:
            columns.append(AIDecision.raw_llm_response)
        stmt = select(*columns).where(AIDecision.id > last_id)
        if not include_raw_json:
            stmt = stmt.where(_visible_mojibake_filter())

        async with get_session_ctx() as session:
            result = await session.execute(
                stmt.order_by(AIDecision.id.asc()).limit(current_limit)
            )
            rows = list(result.mappings().all())
            if not rows:
                break
            for row in rows:
                scanned += 1
                row_id = int(row["id"])
                last_id = row_id
                updates: dict[str, Any] = {}

                cleaned, did_change = _changed_text(row.get("reasoning"))
                if did_change:
                    updates["reasoning"] = cleaned

                cleaned, did_change = _changed_text(row.get("execution_reason"))
                if did_change:
                    updates["execution_reason"] = cleaned

                if include_raw_json:
                    cleaned, did_change = _changed_payload(row.get("raw_llm_response"))
                    if did_change:
                        updates["raw_llm_response"] = cleaned

                if updates:
                    changed += 1
                    if apply:
                        await session.execute(
                            update(AIDecision).where(AIDecision.id == row_id).values(**updates)
                        )
            if not apply:
                await session.rollback()
    return {
        "scanned": scanned,
        "changed": changed,
        "applied": int(apply),
        "include_raw_json": int(include_raw_json),
        "start_id": max(int(start_id or 0), 0),
        "last_id": last_id,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write repaired values back to DB")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument(
        "--include-raw-json",
        action="store_true",
        help=(
            "also scan raw_llm_response JSON by primary-key batches; use --limit and "
            "--start-id for controlled online batches"
        ),
    )
    args = parser.parse_args()

    summary = await repair_ai_decisions(
        apply=bool(args.apply),
        batch_size=max(int(args.batch_size or 500), 1),
        limit=args.limit,
        include_raw_json=bool(args.include_raw_json),
        start_id=max(int(args.start_id or 0), 0),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
