"""Fresh, cost-complete paper opportunities for position replacement comparison."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

from sqlalchemy import select

from db.session import get_read_session_ctx
from models.decision import AIDecision

POSITION_REPLACEMENT_OPPORTUNITY_VERSION = (
    "2026-07-25.paper-position-replacement-opportunity.v1"
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _symbol(value: Any) -> str:
    normalized = str(value or "").upper().split(":")[0]
    if "/" in normalized:
        return normalized
    parts = normalized.replace("-SWAP", "").split("-")
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else normalized


def _raw(row: Any) -> dict[str, Any]:
    for value in (
        getattr(row, "raw_llm_response", None),
        getattr(row, "decision_learning_snapshot", None),
    ):
        payload = _dict(value)
        if _dict(payload.get("opportunity_score")):
            return payload
    return {}


def _provenance_complete(
    value: Any,
    *,
    current: datetime,
    cutoff: datetime,
) -> bool:
    provenance = _dict(value)
    generated_at = _utc(provenance.get("generated_at"))
    valid_for_seconds = _float(provenance.get("valid_for_seconds"))
    return bool(
        str(provenance.get("source") or "").strip()
        and str(provenance.get("observation_window") or "").strip()
        and (_float(provenance.get("sample_count")) or 0.0) > 0.0
        and generated_at is not None
        and generated_at >= cutoff
        and generated_at <= current
        and valid_for_seconds is not None
        and valid_for_seconds > 0.0
        and (current - generated_at).total_seconds() <= valid_for_seconds
        and str(provenance.get("strategy_version") or "").strip()
        and not str(provenance.get("fallback_reason") or "").strip()
    )


def select_position_replacement_opportunity(
    rows: list[Any],
    *,
    execution_mode: str,
    open_symbols: set[str],
    now: datetime | None = None,
    max_age_seconds: float,
) -> dict[str, Any]:
    current = _utc(now) or datetime.now(UTC)
    cutoff = current - timedelta(seconds=max(float(max_age_seconds), 1.0))
    if str(execution_mode or "").lower() != "paper":
        return {
            "version": POSITION_REPLACEMENT_OPPORTUNITY_VERSION,
            "available": False,
            "reason": "paper_only",
            "execution_scope": "paper_only",
            "creates_order": False,
            "production_permission": False,
            "can_increase_leverage": False,
        }
    excluded = {_symbol(item) for item in open_symbols if _symbol(item)}
    candidates: list[dict[str, Any]] = []
    inspected_count = 0
    for row in rows:
        inspected_count += 1
        created_at = _utc(getattr(row, "created_at", None))
        symbol = _symbol(getattr(row, "symbol", None))
        action = str(getattr(row, "action", "") or "").lower()
        if (
            created_at is None
            or created_at < cutoff
            or created_at > current
            or not symbol
            or symbol in excluded
            or action not in {"long", "short"}
            or getattr(row, "is_paper", True) is not True
        ):
            continue
        raw = _raw(row)
        opportunity = _dict(raw.get("opportunity_score"))
        execution_cost = _dict(opportunity.get("execution_cost"))
        expected_net = _float(opportunity.get("expected_net_return_pct"))
        lower_bound = _float(opportunity.get("return_lcb_pct"))
        expected_loss = _float(opportunity.get("expected_loss_pct"))
        if not (
            opportunity.get("production_eligible") is True
            and expected_net is not None
            and expected_net > 0.0
            and lower_bound is not None
            and lower_bound > 0.0
            and expected_loss is not None
            and expected_loss >= 0.0
            and execution_cost.get("production_eligible") is True
            and _provenance_complete(
                opportunity.get("policy_provenance"),
                current=current,
                cutoff=cutoff,
            )
        ):
            continue
        candidates.append(
            {
                "version": POSITION_REPLACEMENT_OPPORTUNITY_VERSION,
                "available": True,
                "production_eligible": True,
                "execution_scope": "paper_only",
                "production_permission": False,
                "creates_order": False,
                "can_increase_leverage": False,
                "symbol": symbol,
                "side": action,
                "expected_net_return_pct": round(expected_net, 8),
                "return_lcb_pct": round(lower_bound, 8),
                "expected_loss_pct": round(expected_loss, 8),
                "decision_id": getattr(row, "id", None),
                "observed_at": created_at.isoformat(),
                "evidence_age_seconds": round(
                    max((current - created_at).total_seconds(), 0.0),
                    3,
                ),
                "freshness_limit_seconds": round(max(float(max_age_seconds), 1.0), 3),
                "execution_cost": execution_cost,
                "policy_provenance": _dict(opportunity.get("policy_provenance")),
                "selection_reason": "fresh_cost_complete_positive_return_lcb",
            }
        )
    candidates.sort(
        key=lambda item: (
            float(item["return_lcb_pct"]),
            float(item["expected_net_return_pct"]),
            str(item["observed_at"]),
        ),
        reverse=True,
    )
    if candidates:
        return {
            **candidates[0],
            "inspected_decision_count": inspected_count,
            "eligible_candidate_count": len(candidates),
        }
    return {
        "version": POSITION_REPLACEMENT_OPPORTUNITY_VERSION,
        "available": False,
        "reason": "no_fresh_cost_complete_unheld_opportunity",
        "execution_scope": "paper_only",
        "production_permission": False,
        "creates_order": False,
        "can_increase_leverage": False,
        "inspected_decision_count": inspected_count,
        "eligible_candidate_count": 0,
        "freshness_limit_seconds": round(max(float(max_age_seconds), 1.0), 3),
    }


async def load_position_replacement_opportunity(
    *,
    execution_mode: str,
    open_symbols: set[str],
    max_age_seconds: float,
    limit: int,
) -> dict[str, Any]:
    if str(execution_mode or "").lower() != "paper":
        return select_position_replacement_opportunity(
            [],
            execution_mode=execution_mode,
            open_symbols=open_symbols,
            max_age_seconds=max_age_seconds,
        )
    cutoff = datetime.now(UTC) - timedelta(seconds=max(float(max_age_seconds), 1.0))
    async with get_read_session_ctx() as session:
        result = await session.execute(
            select(AIDecision)
            .where(
                AIDecision.is_paper.is_(True),
                AIDecision.analysis_type == "market",
                AIDecision.created_at >= cutoff,
                AIDecision.action.in_(("long", "short")),
            )
            .order_by(AIDecision.created_at.desc(), AIDecision.id.desc())
            .limit(max(int(limit), 1))
        )
        rows = list(result.scalars().all())
    return select_position_replacement_opportunity(
        rows,
        execution_mode=execution_mode,
        open_symbols=open_symbols,
        max_age_seconds=max_age_seconds,
    )
