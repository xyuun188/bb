"""Audit runtime text fields for suspected mojibake without mutating data."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.session import get_session_ctx  # noqa: E402
from models.decision import AIDecision  # noqa: E402
from models.learning import (  # noqa: E402
    ExpertMemory,
    ShadowBacktest,
    StrategyLearningEvent,
    TradeReflection,
)
from services.text_integrity import looks_like_mojibake, repair_mojibake  # noqa: E402

TEXT_AUDIT_FIELDS: dict[str, tuple[str, ...]] = {
    "ai_decisions": ("reasoning", "execution_reason", "feature_snapshot", "raw_llm_response"),
    "strategy_learning_events": (
        "reason",
        "scheduler_reason",
        "strategy_snapshot",
        "market_state",
        "side_weights",
        "expert_integrity",
        "attribution",
    ),
    "expert_memories": ("market_pattern", "lesson", "recommended_action", "extra"),
    "trade_reflections": ("mistake_summary", "improvement_summary", "expert_lessons"),
    "shadow_backtests": ("feature_snapshot", "raw_llm_response", "note"),
}

MODEL_BY_TABLE = {
    "ai_decisions": AIDecision,
    "strategy_learning_events": StrategyLearningEvent,
    "expert_memories": ExpertMemory,
    "trade_reflections": TradeReflection,
    "shadow_backtests": ShadowBacktest,
}


def _record_time(row: Any) -> datetime | None:
    value = getattr(row, "updated_at", None) or getattr(row, "created_at", None)
    return value if isinstance(value, datetime) else None


def _json_safe(value: Any, *, limit: int = 220) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, dict):
        return {str(key): _json_safe(item, limit=limit) for key, item in list(value.items())[:12]}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, limit=limit) for item in list(value)[:12]]
    return str(value)[:limit]


def _iter_text_leaves(value: Any, prefix: str) -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_text_leaves(item, f"{prefix}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_text_leaves(item, f"{prefix}[{index}]")


def _empty_table_stats() -> dict[str, Any]:
    return {
        "scanned_records": 0,
        "suspected_records": 0,
        "suspected_fields": 0,
        "repairable_fields": 0,
        "status": "ok",
    }


def build_runtime_text_integrity_report(
    rows: Iterable[tuple[str, Any]],
    *,
    generated_at: datetime | None = None,
    example_limit: int = 12,
) -> dict[str, Any]:
    generated = generated_at or datetime.now(UTC)
    by_table: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    scanned_records = 0
    suspected_records = 0
    suspected_fields = 0
    repairable_count = 0

    for table, row in rows:
        table_stats = by_table.setdefault(table, _empty_table_stats())
        table_stats["scanned_records"] += 1
        scanned_records += 1
        record_suspected = False
        record_id = getattr(row, "id", None)
        for field in TEXT_AUDIT_FIELDS.get(table, ()):  # unknown tables are counted only.
            value = getattr(row, field, None)
            for field_path, text in _iter_text_leaves(value, field):
                if not looks_like_mojibake(text):
                    continue
                record_suspected = True
                suspected_fields += 1
                table_stats["suspected_fields"] += 1
                repair = repair_mojibake(text)
                if repair.changed and not looks_like_mojibake(repair.text):
                    repairable_count += 1
                    table_stats["repairable_fields"] += 1
                if len(examples) < max(1, int(example_limit)):
                    examples.append(
                        {
                            "table": table,
                            "id": record_id,
                            "field": field_path,
                            "created_at": _json_safe(_record_time(row)),
                            "sample": text[:180],
                            "repairable": bool(repair.changed),
                            "repair_method": repair.method,
                            "repair_reason": repair.reason,
                            "repair_preview": repair.text[:180] if repair.changed else "",
                        }
                    )
        if record_suspected:
            suspected_records += 1
            table_stats["suspected_records"] += 1
            table_stats["status"] = "warning"

    return {
        "status": "warning" if suspected_records else "ok",
        "generated_at": generated.isoformat(),
        "scanned_records": scanned_records,
        "suspected_records": suspected_records,
        "suspected_fields": suspected_fields,
        "repairable_count": repairable_count,
        "by_table": by_table,
        "examples": examples,
        "policy": {
            "dry_run": True,
            "mutates_database": False,
            "preserve_original_text": True,
            "automatic_repair": False,
        },
    }


async def collect_runtime_text_integrity_report(
    *,
    hours: int = 24,
    limit_per_table: int = 200,
    example_limit: int = 12,
) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(hours=max(1, int(hours or 24)))
    collected: list[tuple[str, Any]] = []
    async with get_session_ctx() as session:
        for table, model in MODEL_BY_TABLE.items():
            stmt = (
                select(model).order_by(model.id.desc()).limit(max(1, int(limit_per_table or 200)))
            )
            created_at = getattr(model, "created_at", None)
            if created_at is not None:
                stmt = stmt.where(created_at >= since)
            result = await session.execute(stmt)
            collected.extend((table, row) for row in result.scalars().all())
    return build_runtime_text_integrity_report(collected, example_limit=example_limit)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24, help="Recent hours to inspect.")
    parser.add_argument(
        "--limit-per-table",
        type=int,
        default=200,
        help="Maximum records to inspect per table.",
    )
    parser.add_argument("--examples", type=int, default=12, help="Maximum examples to print.")
    parser.add_argument(
        "--json-indent",
        type=int,
        default=2,
        help="JSON indentation, use 0 for compact output.",
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    try:
        report = await collect_runtime_text_integrity_report(
            hours=args.hours,
            limit_per_table=args.limit_per_table,
            example_limit=args.examples,
        )
        exit_code = 1 if report.get("status") == "warning" else 0
    except Exception as exc:
        report = {
            "status": "warning",
            "generated_at": datetime.now(UTC).isoformat(),
            "scanned_records": 0,
            "suspected_records": 0,
            "suspected_fields": 0,
            "repairable_count": 0,
            "by_table": {},
            "examples": [],
            "error": {"type": type(exc).__name__, "message": str(exc)[:300]},
            "policy": {
                "dry_run": True,
                "mutates_database": False,
                "preserve_original_text": True,
                "automatic_repair": False,
            },
        }
        exit_code = 2
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    return exit_code


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
