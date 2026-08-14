"""Read-only experiment, promotion, configuration, and policy diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter
from sqlalchemy import select

from db.session import get_read_session_ctx
from models.experiment import ExperimentRun
from models.promotion import PromotionReview
from models.runtime_config import RuntimeConfigVersion
from services.protection_chain import ProtectionChain, ProtectionObservation
from web_dashboard.api import dashboard as dashboard_api
from web_dashboard.api.text_sanitize import sanitize_payload

router = APIRouter()


@router.get("/governance/experiments")
async def experiments(limit: int = 50) -> dict[str, Any]:
    rows = await _recent(ExperimentRun, limit)
    return sanitize_payload(
        {
            "read_only": True,
            "count": len(rows),
            "experiments": [
                {
                    "experiment_id": row.experiment_id,
                    "experiment_type": row.experiment_type,
                    "status": row.status,
                    "strategy_id": row.strategy_id,
                    "strategy_version": row.strategy_version,
                    "parameter_set_id": row.parameter_set_id,
                    "dataset_id": row.dataset_id,
                    "git_commit": row.git_commit,
                    "spec_sha256": row.spec_sha256,
                    "result_sha256": row.result_sha256,
                    "metrics": (row.result or {}).get("metrics", {}),
                    "created_at": _iso(row.created_at),
                    "completed_at": _iso(row.completed_at),
                    "failure_reason": row.failure_reason,
                }
                for row in rows
            ],
        }
    )


@router.get("/governance/promotion-reviews")
async def promotion_reviews(limit: int = 50) -> dict[str, Any]:
    rows = await _recent(PromotionReview, limit)
    return sanitize_payload(
        {
            "read_only": True,
            "automatic_live_promotion": False,
            "count": len(rows),
            "reviews": [
                {
                    "review_id": row.review_id,
                    "artifact_id": row.artifact_id,
                    "strategy_id": row.strategy_id,
                    "strategy_version": row.strategy_version,
                    "from_stage": row.from_stage,
                    "to_stage": row.to_stage,
                    "status": row.status,
                    "gate": (row.review or {}).get("gate", {}),
                    "reviewer": row.reviewer,
                    "decision_reason": row.decision_reason,
                    "review_sha256": row.review_sha256,
                    "created_at": _iso(row.created_at),
                }
                for row in rows
            ],
        }
    )


@router.get("/governance/runtime-config-versions")
async def runtime_config_versions(limit: int = 50) -> dict[str, Any]:
    rows = await _recent(RuntimeConfigVersion, limit)
    return sanitize_payload(
        {
            "read_only": True,
            "count": len(rows),
            "versions": [
                {
                    "version_id": row.version_id,
                    "environment": row.environment,
                    "status": row.status,
                    "parent_version": row.parent_version,
                    "config_sha256": row.config_sha256,
                    "changed_by": row.changed_by,
                    "change_reason": row.change_reason,
                    "activation_by": row.activation_by,
                    "activation_reason": row.activation_reason,
                    "created_at": _iso(row.created_at),
                    "activated_at": _iso(row.activated_at),
                    "deactivated_at": _iso(row.deactivated_at),
                }
                for row in rows
            ],
        }
    )


@router.get("/governance/policy-diagnostics")
async def policy_diagnostics() -> dict[str, Any]:
    """Expose existing runtime state through the standard P3 evidence shape."""

    service = getattr(dashboard_api, "_trading_service", None)
    if service is None:
        return {
            "available": False,
            "read_only": True,
            "candidate_pool_funnel": {},
            "protection": {},
            "reason": "trading_service_not_attached",
        }
    risk_engine = getattr(service, "risk_engine", None)
    circuit_breaker = getattr(risk_engine, "circuit_breaker", None)
    risk = circuit_breaker.get_state() if circuit_breaker is not None else {}
    try:
        runtime_stats = service.get_stats()
    except Exception:
        runtime_stats = {}
    authority = runtime_stats.get("okx_authoritative_sync")
    authority = authority if isinstance(authority, dict) else {}
    exchange_healthy = authority.get("can_open_new_entries") is not False
    exchange_reason = str(
        authority.get("reason")
        or authority.get("status")
        or "okx_authoritative_sync_state"
    )
    tripped_at = _parse_utc(risk.get("tripped_at"))
    cooldown = getattr(circuit_breaker, "cooldown", timedelta(0))
    cooldown_until = tripped_at + cooldown if tripped_at else None
    protection = ProtectionChain().evaluate(
        ProtectionObservation(
            scope="portfolio",
            cooldown_until=cooldown_until if risk.get("breaker_state") == "open" else None,
            consecutive_losses=int(risk.get("consecutive_losses") or 0),
            exchange_healthy=exchange_healthy,
            exchange_health_reason=exchange_reason,
            source_event=str(risk.get("tripped_reason") or "circuit_breaker_state"),
            observed_at=tripped_at,
        )
    )
    return sanitize_payload(
        {
            "available": True,
            "read_only": True,
            "candidate_pool_funnel": getattr(service, "_last_market_candidate_funnel", {}) or {},
            "protection": protection,
            "risk_authority": risk,
            "okx_authoritative_sync": authority,
            "boundary": (
                "This endpoint is diagnostic only; the trading risk engine and OKX facts remain authoritative."
            ),
        }
    )


async def _recent(model: Any, limit: int) -> list[Any]:
    bounded = max(1, min(int(limit or 50), 200))
    async with get_read_session_ctx() as session:
        result = await session.execute(select(model).order_by(model.created_at.desc()).limit(bounded))
        return list(result.scalars().all())


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
