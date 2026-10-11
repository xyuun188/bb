"""Bounded cleanup of the v5 sizing projection from original decision facts."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from math import isfinite
from typing import Any

from sqlalchemy import select, update

from db.session import get_read_session_ctx, get_session_ctx
from models.decision import AIDecision
from services.training_epoch import load_training_data_start


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
