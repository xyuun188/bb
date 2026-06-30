from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from db.session import get_read_session_ctx
from models.decision import AIDecision

ENTRY_ACTIONS = {"long", "short", "open_long", "open_short", "buy", "sell"}


class HighRiskReviewAuditService:
    """Read-only audit for the independent high-risk review gate."""

    def __init__(self, session_context_factory: Any | None = None) -> None:
        self._session_context_factory = session_context_factory

    async def report(self, *, hours: int = 72, limit: int = 1200) -> dict[str, Any]:
        capped_hours = max(1, min(int(hours or 72), 168))
        capped_limit = max(50, min(int(limit or 1200), 5000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        session_factory = self._session_context_factory or get_read_session_ctx
        async with session_factory() as session:
            result = await session.execute(
                select(AIDecision)
                .where(AIDecision.created_at >= since)
                .order_by(AIDecision.created_at.desc())
                .limit(capped_limit)
            )
        return summarize_high_risk_review(list(result.scalars().all()), hours=capped_hours)


def summarize_high_risk_review(decisions: list[Any], *, hours: int = 72) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()
    approved_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    hard_review_required_count = 0
    blocked_count = 0
    executed_without_required_review = 0
    reviewed_samples: list[dict[str, Any]] = []

    entry_count = 0
    for decision in decisions:
        if not _is_entry(decision):
            continue
        entry_count += 1
        raw = _safe_dict(_row_get(decision, "raw_llm_response"))
        review = _safe_dict(raw.get("high_risk_review"))
        if not review:
            continue
        triggered = bool(review.get("triggered"))
        status = str(review.get("status") or ("triggered" if triggered else "skipped")).lower()
        approved = review.get("approved")
        hard_required = bool(review.get("hard_review_required"))
        status_counts[status] += 1
        trigger_counts["triggered" if triggered else "not_triggered"] += 1
        if approved is True:
            approved_counts["approved_true"] += 1
        elif approved is False:
            approved_counts["approved_false"] += 1
        else:
            approved_counts["approved_null"] += 1
        if hard_required:
            hard_review_required_count += 1
        if approved is False or status in {"skipped_blocked", "error_blocked"}:
            blocked_count += 1
        if hard_required and bool(_row_get(decision, "was_executed")) and approved is not True:
            executed_without_required_review += 1
        for reason in _safe_list(review.get("reasons")) + _safe_list(
            review.get("advisory_reasons")
        ):
            if reason:
                reason_counts[str(reason)] += 1
        if review.get("reason"):
            reason_counts[str(review.get("reason"))[:120]] += 1
        if len(reviewed_samples) < 20:
            reviewed_samples.append(
                {
                    "decision_id": int(_row_get(decision, "id", 0) or 0),
                    "symbol": _row_get(decision, "symbol"),
                    "action": _row_get(decision, "action"),
                    "executed": bool(_row_get(decision, "was_executed")),
                    "triggered": triggered,
                    "status": status,
                    "approved": approved,
                    "hard_review_required": hard_required,
                    "reasons": _safe_list(review.get("reasons"))[:6],
                }
            )

    return {
        "audit_only": True,
        "live_entry_mutation": False,
        "can_bypass_risk_controls": False,
        "window_hours": int(hours),
        "checked_decisions": len(decisions),
        "entry_decision_count": entry_count,
        "review_payload_count": sum(status_counts.values()),
        "hard_review_required_count": hard_review_required_count,
        "blocked_count": blocked_count,
        "executed_without_required_review_count": executed_without_required_review,
        "status_counts": dict(status_counts),
        "trigger_counts": dict(trigger_counts),
        "approved_counts": dict(approved_counts),
        "reason_counts": dict(reason_counts.most_common(12)),
        "samples": reviewed_samples,
        "policy": {
            "high_risk_review_is_independent": True,
            "ordinary_entries_must_not_call_high_risk_review": True,
            "hard_review_must_approve_before_execution": True,
            "failed_required_review_blocks_entry": True,
            "audit_can_bypass_risk_controls": False,
        },
    }


def _is_entry(row: Any) -> bool:
    return str(_row_get(row, "action") or "").lower() in ENTRY_ACTIONS


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)
