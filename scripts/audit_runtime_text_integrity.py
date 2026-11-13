"""Audit runtime text fields for suspected mojibake without mutating data."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sqlalchemy import JSON, Text, cast, select

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.session import get_read_session_ctx  # noqa: E402
from models.decision import AIDecision  # noqa: E402
from models.learning import (  # noqa: E402
    ExpertMemory,
    ShadowBacktest,
    StrategyLearningEvent,
    TradeReflection,
)
from services.text_integrity import (  # noqa: E402
    looks_like_mojibake,
    repair_mojibake,
    serialized_json_may_contain_mojibake,
)

TEXT_AUDIT_FIELDS: dict[str, tuple[str, ...]] = {
    "ai_decisions": (
        "reasoning",
        "execution_reason",
        "feature_snapshot",
        "decision_learning_snapshot",
    ),
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
    "shadow_backtests": ("training_feature_snapshot", "note"),
}

MODEL_BY_TABLE = {
    "ai_decisions": AIDecision,
    "strategy_learning_events": StrategyLearningEvent,
    "expert_memories": ExpertMemory,
    "trade_reflections": TradeReflection,
    "shadow_backtests": ShadowBacktest,
}
# A few strategy rows already contain tens of MiB of nested JSON; keep each
# projection below the shared statement budget even during concurrent audits.
TEXT_PROJECTION_BATCH_SIZE = 5


@dataclass(frozen=True)
class _SerializedJSON:
    text: str


def _recent_record_ids_statement(model: Any, *, since: datetime, limit: int) -> Any:
    return (
        select(model.id)
        .where(model.created_at >= since)
        .order_by(model.created_at.desc(), model.id.desc())
        .limit(max(1, int(limit)))
    )


def _runtime_text_projection_statement(
    table: str,
    model: Any,
    *,
    record_ids: list[int],
) -> Any:
    selected_columns = [model.id, model.created_at, model.updated_at]
    for field in TEXT_AUDIT_FIELDS.get(table, ()):
        column = getattr(model, field)
        selected_columns.append(
            cast(column, Text).label(field) if isinstance(column.type, JSON) else column
        )
    return select(*selected_columns).where(model.id.in_(record_ids))


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
            if isinstance(value, _SerializedJSON):
                if not serialized_json_may_contain_mojibake(value.text):
                    continue
                value = json.loads(value.text)
            for field_path, text in _iter_text_leaves(value, field):
                if text.isascii() or not looks_like_mojibake(text):
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
    example_limit = max(1, int(example_limit))
    report = build_runtime_text_integrity_report([], example_limit=example_limit)
    for table, model in MODEL_BY_TABLE.items():
        json_fields = [
            field for field in TEXT_AUDIT_FIELDS.get(table, ())
            if isinstance(getattr(model, field).type, JSON)
        ]
        async with get_read_session_ctx() as session:
            id_result = await session.execute(
                _recent_record_ids_statement(
                    model,
                    since=since,
                    limit=max(1, int(limit_per_table or 200)),
                )
            )
            record_ids = [int(value) for value in id_result.scalars().all()]
        for offset in range(0, len(record_ids), TEXT_PROJECTION_BATCH_SIZE):
            batch_ids = record_ids[offset : offset + TEXT_PROJECTION_BATCH_SIZE]
            async with get_read_session_ctx() as session:
                result = await session.execute(
                    _runtime_text_projection_statement(
                        table,
                        model,
                        record_ids=batch_ids,
                    )
                )
                rows_by_id = {}
                for row in result.mappings().all():
                    values = dict(row)
                    for field in json_fields:
                        if isinstance(values.get(field), str):
                            values[field] = _SerializedJSON(values[field])
                    rows_by_id[int(row["id"])] = SimpleNamespace(**values)
            batch = [
                (table, rows_by_id[record_id])
                for record_id in batch_ids
                if record_id in rows_by_id
            ]
            # Large JSON snapshots must leave the read transaction before scanning.
            partial = await asyncio.to_thread(
                build_runtime_text_integrity_report, batch, example_limit=example_limit,
            )
            del result, rows_by_id, batch
            for key in ("scanned_records", "suspected_records", "suspected_fields", "repairable_count"):
                report[key] += partial[key]
            for table_name, stats in partial["by_table"].items():
                totals = report["by_table"].setdefault(table_name, _empty_table_stats())
                for key in (
                    "scanned_records", "suspected_records", "suspected_fields", "repairable_fields",
                ):
                    totals[key] += stats[key]
                totals["status"] = "warning" if totals["suspected_records"] else "ok"
            remaining = max(example_limit - len(report["examples"]), 0)
            report["examples"].extend(partial["examples"][:remaining])
    report["status"] = "warning" if report["suspected_records"] else "ok"
    return report


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
