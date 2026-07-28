"""Report live return-source health and normal paper-trading continuity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select

from db.session import get_read_session_ctx
from models.decision import AIDecision
from services.normal_paper_trade import normal_paper_trade_contract_reasons


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _raw(row: Any) -> dict[str, Any]:
    return _safe_dict(
        _row_value(row, "raw_llm_response")
        or _row_value(row, "raw_response")
        or _row_value(row, "decision_learning_snapshot")
    )


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _source_count(raw: dict[str, Any]) -> int:
    candidate = _safe_dict(raw.get("authoritative_return_candidate"))
    side = _safe_dict(candidate.get("side_evidence"))
    try:
        direct = int(float(side.get("production_source_count") or 0))
    except (TypeError, ValueError):
        direct = 0
    evidence = _safe_dict(raw.get("entry_candidate_evidence"))
    total = direct
    for name in ("long", "short"):
        try:
            total = max(
                total,
                int(float(_safe_dict(evidence.get(name)).get("production_source_count") or 0)),
            )
        except (TypeError, ValueError):
            continue
    return max(total, 0)


def _is_normal_paper_candidate(raw: dict[str, Any]) -> bool:
    contract = _safe_dict(raw.get("normal_paper_trade"))
    return bool(contract) and not normal_paper_trade_contract_reasons(contract)


def summarize_production_source_health(
    decisions: list[Any],
    *,
    now: datetime | None = None,
    decision_interval_seconds: int = 60,
) -> dict[str, Any]:
    checked_at = now or datetime.now(UTC)
    rows = sorted(
        decisions,
        key=lambda row: _as_utc(_row_value(row, "created_at")) or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )
    market_rows = [
        row for row in rows if str(_row_value(row, "analysis_type") or "market") == "market"
    ]
    source_rows = [row for row in market_rows if _source_count(_raw(row)) > 0]
    last_source_at = _as_utc(_row_value(source_rows[0], "created_at")) if source_rows else None
    oldest_at = _as_utc(_row_value(market_rows[-1], "created_at")) if market_rows else None
    no_source_since = last_source_at or oldest_at
    no_source_seconds = (
        max((checked_at - no_source_since).total_seconds(), 0.0)
        if no_source_since is not None
        else None
    )
    warning_after = max(int(decision_interval_seconds or 60) * 6, 600)
    critical_after = max(int(decision_interval_seconds or 60) * 30, 3600)
    normal_paper_rows = [row for row in market_rows if _is_normal_paper_candidate(_raw(row))]
    normal_paper_executed = sum(bool(_row_value(row, "was_executed")) for row in normal_paper_rows)
    latest_normal_paper_at = (
        _as_utc(_row_value(normal_paper_rows[0], "created_at")) if normal_paper_rows else None
    )
    no_normal_paper_since = latest_normal_paper_at or oldest_at
    no_normal_paper_seconds = (
        max((checked_at - no_normal_paper_since).total_seconds(), 0.0)
        if no_normal_paper_since is not None
        else None
    )
    paper_trade_alert_active = bool(
        market_rows
        and no_normal_paper_seconds is not None
        and no_normal_paper_seconds >= critical_after
    )
    paper_trade_alert_reason = (
        "continuous_no_normal_paper_candidate" if paper_trade_alert_active else None
    )
    if not market_rows:
        status = "warning"
        reason = "market_decision_evidence_unavailable"
    elif source_rows and no_source_seconds is not None and no_source_seconds < warning_after:
        if paper_trade_alert_active:
            status = "warning"
            reason = paper_trade_alert_reason
        else:
            status = "ok"
            reason = "governed_production_return_source_recent"
    elif no_source_seconds is not None and no_source_seconds >= critical_after:
        status = "critical"
        reason = "continuous_no_production_return_source"
    else:
        status = "warning"
        reason = "production_return_source_recovery_window"
    return {
        "status": status,
        "reason": reason,
        "alert_active": status in {"warning", "critical"},
        "production_permission": False,
        "market_decision_count": len(market_rows),
        "production_source_decision_count": len(source_rows),
        "latest_production_source_at": last_source_at.isoformat() if last_source_at else None,
        "continuous_no_source_seconds": (
            round(no_source_seconds, 3) if no_source_seconds is not None else None
        ),
        "warning_after_seconds": warning_after,
        "critical_after_seconds": critical_after,
        "normal_paper_candidate_count": len(normal_paper_rows),
        "normal_paper_executed_count": normal_paper_executed,
        "latest_normal_paper_candidate_at": (
            latest_normal_paper_at.isoformat() if latest_normal_paper_at else None
        ),
        "continuous_no_normal_paper_candidate_seconds": (
            round(no_normal_paper_seconds, 3) if no_normal_paper_seconds is not None else None
        ),
        "continuous_training_after_settlement": bool(normal_paper_rows),
        "paper_trade_status": (
            "active"
            if latest_normal_paper_at is not None
            and no_normal_paper_seconds is not None
            and no_normal_paper_seconds < warning_after
            else "stale"
            if paper_trade_alert_active
            else "waiting"
        ),
        "paper_trade_alert_active": paper_trade_alert_active,
        "paper_trade_alert_reason": paper_trade_alert_reason,
        "recovery_state": (
            "normal_paper_trading" if normal_paper_rows else "normal_paper_candidate_waiting"
        ),
        "checked_at": checked_at.isoformat(),
    }


class ProductionSourceHealthService:
    async def report(
        self,
        *,
        hours: int = 24,
        limit: int = 5000,
        decision_interval_seconds: int = 60,
    ) -> dict[str, Any]:
        capped_hours = max(1, min(int(hours or 24), 168))
        capped_limit = max(100, min(int(limit or 5000), 10000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        async with get_read_session_ctx() as session:
            result = await session.execute(
                select(
                    AIDecision.created_at,
                    AIDecision.analysis_type,
                    AIDecision.was_executed,
                    AIDecision.decision_learning_snapshot.label("raw_llm_response"),
                )
                .where(
                    AIDecision.created_at >= since,
                    AIDecision.analysis_type == "market",
                    AIDecision.decision_learning_snapshot_version >= 1,
                )
                .order_by(AIDecision.created_at.desc())
                .limit(capped_limit)
            )
        report = summarize_production_source_health(
            [SimpleNamespace(**dict(row)) for row in result.mappings().all()],
            decision_interval_seconds=decision_interval_seconds,
        )
        report["window_hours"] = capped_hours
        return report
