"""Bounded repair of compact decision projections from original decision facts."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from math import isfinite
from typing import Any

from sqlalchemy import select, update

from db.session import get_read_session_ctx, get_session_ctx
from models.decision import AIDecision
from services.training_epoch import load_training_data_start

DECISION_LEARNING_SNAPSHOT_VERSION = 7
EXECUTION_RISK_PROJECTION_KEYS = (
    "current_training_portfolio_stressed_loss_usdt",
    "portfolio_risk_limit_applied",
    "remaining_training_portfolio_risk_budget_usdt",
)


def restore_downside_return(snapshot: Any, raw_lcb: Any) -> dict[str, Any] | None:
    """Restore only the field omitted by v5; never infer a return."""

    if not isinstance(snapshot, dict) or isinstance(raw_lcb, bool):
        return None
    sizing = snapshot.get("profit_risk_sizing")
    if not isinstance(sizing, dict) or sizing.get("return_lcb_pct") is not None:
        return None
    try:
        value = float(raw_lcb)
    except (TypeError, ValueError):
        return None
    if not isfinite(value):
        return None
    result = deepcopy(snapshot)
    result["profit_risk_sizing"]["return_lcb_pct"] = value
    return result


async def backfill_downside_return_projection(
    *,
    apply: bool = False,
    since: datetime | None = None,
    batch_size: int = 50,
    max_rows: int = 10_000,
    before_id: int | None = None,
) -> dict[str, Any]:
    """Clean historical training inputs in small independent transactions."""

    since = since or load_training_data_start()
    batch_size = max(1, min(int(batch_size), 200))
    max_rows = max(1, min(int(max_rows), 100_000))
    last_id = before_id
    scanned = repaired = unavailable = 0
    sample_ids: list[int] = []
    complete = False
    while scanned < max_rows:
        async with get_read_session_ctx(statement_timeout_ms=30_000) as session:
            query = (
                select(
                    AIDecision.id,
                    AIDecision.decision_learning_snapshot,
                )
                .where(
                    AIDecision.created_at >= since,
                    AIDecision.decision_learning_snapshot_version == 5,
                )
                .order_by(AIDecision.id.desc())
                .limit(min(batch_size, max_rows - scanned))
            )
            if last_id is not None:
                query = query.where(AIDecision.id < last_id)
            rows = (await session.execute(query)).mappings().all()
            repair_ids = [
                int(row["id"])
                for row in rows
                if isinstance(row["decision_learning_snapshot"], dict)
                and isinstance(row["decision_learning_snapshot"].get("profit_risk_sizing"), dict)
                and row["decision_learning_snapshot"]["profit_risk_sizing"].get("return_lcb_pct")
                is None
            ]
            # Limit compact rows first. Never filter or sort historic raw JSON
            # across the table just to find the next handful of missing fields.
            raw_values = {}
            if repair_ids:
                facts = (
                    await session.execute(
                        select(
                            AIDecision.id,
                            AIDecision.raw_llm_response["profit_risk_sizing"]["return_lcb_pct"],
                        ).where(AIDecision.id.in_(repair_ids))
                    )
                ).all()
                raw_values = {int(row[0]): row[1] for row in facts}
        if not rows:
            complete = True
            break
        scanned += len(rows)
        last_id = int(rows[-1]["id"])
        updates = []
        for row in rows:
            if int(row["id"]) not in raw_values:
                continue
            snapshot = restore_downside_return(
                row["decision_learning_snapshot"], raw_values[int(row["id"])]
            )
            if snapshot is None:
                unavailable += 1
                continue
            updates.append((int(row["id"]), snapshot))
            if len(sample_ids) < 12:
                sample_ids.append(int(row["id"]))
        if apply and updates:
            async with get_session_ctx() as session:
                for decision_id, snapshot in updates:
                    await session.execute(
                        update(AIDecision)
                        .where(
                            AIDecision.id == decision_id,
                            AIDecision.decision_learning_snapshot_version == 5,
                        )
                        .values(
                            decision_learning_snapshot=snapshot,
                            decision_learning_snapshot_version=6,
                        )
                    )
        repaired += len(updates)
    return {
        "status": "complete" if complete else "partial",
        "apply": apply,
        "since": since.isoformat(),
        "scanned": scanned,
        "repairable": repaired,
        "applied": repaired if apply else 0,
        "original_field_unavailable": unavailable,
        "sample_ids": sample_ids,
        "last_scanned_id": last_id,
        "raw_facts_preserved": True,
    }


def restore_execution_risk_projection(
    snapshot: Any,
    raw_sizing: Any,
) -> dict[str, Any] | None:
    """Restore missing execution-risk fields without rewriting existing facts.

    The compact projection is an audit input, not an authorization source.
    Only fields that are present in the original persisted sizing envelope and
    absent from the compact snapshot are copied.
    """

    if not isinstance(snapshot, dict) or not isinstance(raw_sizing, dict):
        return None
    sizing = snapshot.get("profit_risk_sizing")
    if not isinstance(sizing, dict):
        return None
    source_sizing = dict(raw_sizing)
    audit_inputs = source_sizing.get("audit_inputs")
    if isinstance(audit_inputs, dict):
        for key in EXECUTION_RISK_PROJECTION_KEYS:
            if key not in source_sizing and key in audit_inputs:
                source_sizing[key] = audit_inputs[key]
    repaired = deepcopy(snapshot)
    repaired_sizing = repaired["profit_risk_sizing"]
    changed = False
    for key in EXECUTION_RISK_PROJECTION_KEYS:
        if key not in repaired_sizing and key in source_sizing:
            repaired_sizing[key] = deepcopy(source_sizing[key])
            changed = True
    return repaired if changed else None


async def backfill_execution_risk_projection(
    *,
    apply: bool = False,
    since: datetime | None = None,
    batch_size: int = 50,
    max_rows: int = 10_000,
    before_id: int | None = None,
) -> dict[str, Any]:
    """Repair v6-and-earlier snapshots from raw JSON in bounded batches."""

    since = since or load_training_data_start()
    batch_size = max(1, min(int(batch_size), 200))
    max_rows = max(1, min(int(max_rows), 100_000))
    last_id = before_id
    scanned = repaired = applied = unavailable = 0
    sample_ids: list[int] = []
    complete = False
    while scanned < max_rows:
        async with get_read_session_ctx(statement_timeout_ms=30_000) as session:
            query = (
                select(
                    AIDecision.id,
                    AIDecision.decision_learning_snapshot,
                    AIDecision.raw_llm_response["profit_risk_sizing"].label(
                        "raw_profit_risk_sizing"
                    ),
                    AIDecision.raw_llm_response["profit_risk_sizing"]["return_lcb_pct"].label(
                        "raw_return_lcb_pct"
                    ),
                )
                .where(
                    AIDecision.created_at >= since,
                    AIDecision.decision_learning_snapshot_version >= 1,
                    AIDecision.decision_learning_snapshot_version
                    < DECISION_LEARNING_SNAPSHOT_VERSION,
                )
                .order_by(AIDecision.id.desc())
                .limit(min(batch_size, max_rows - scanned))
            )
            if last_id is not None:
                query = query.where(AIDecision.id < last_id)
            rows = (await session.execute(query)).mappings().all()
        if not rows:
            complete = True
            break
        scanned += len(rows)
        last_id = int(rows[-1]["id"])
        updates: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            existing = row["decision_learning_snapshot"]
            if not isinstance(existing, dict):
                unavailable += 1
                continue
            snapshot = restore_execution_risk_projection(
                existing,
                row.get("raw_profit_risk_sizing"),
            )
            snapshot = snapshot or deepcopy(existing)
            snapshot = (
                restore_downside_return(snapshot, row.get("raw_return_lcb_pct")) or snapshot
            )
            normal_trade = snapshot.get("normal_paper_trade")
            sizing = snapshot.get("profit_risk_sizing")
            training_entry = (
                isinstance(normal_trade, dict)
                and normal_trade.get("selection_reason") == "paper_training_entry"
            )
            if training_entry and (
                not isinstance(sizing, dict)
                or any(key not in sizing for key in EXECUTION_RISK_PROJECTION_KEYS)
            ):
                unavailable += 1
                continue
            updates.append((int(row["id"]), snapshot))
            if len(sample_ids) < 12:
                sample_ids.append(int(row["id"]))
        repaired += len(updates)
        if apply and updates:
            async with get_session_ctx() as session:
                for decision_id, snapshot in updates:
                    result = await session.execute(
                        update(AIDecision)
                        .where(
                            AIDecision.id == decision_id,
                            AIDecision.decision_learning_snapshot_version
                            < DECISION_LEARNING_SNAPSHOT_VERSION,
                        )
                        .values(
                            decision_learning_snapshot=snapshot,
                            decision_learning_snapshot_version=DECISION_LEARNING_SNAPSHOT_VERSION,
                        )
                    )
                    applied += int(getattr(result, "rowcount", 0) or 0)
    return {
        "status": "complete" if complete else "partial",
        "apply": apply,
        "since": since.isoformat(),
        "scanned": scanned,
        "repairable": repaired,
        "applied": applied,
        "original_field_unavailable": unavailable,
        "sample_ids": sample_ids,
        "last_scanned_id": last_id,
        "target_snapshot_version": DECISION_LEARNING_SNAPSHOT_VERSION,
        "raw_facts_preserved": True,
    }
