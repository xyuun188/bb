"""
Model performance API endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from web_dashboard.api.dashboard import _competition_service, _trading_service

router = APIRouter()


@router.get("/models")
async def list_models():
    """List all AI models with current performance rankings."""
    rankings = []
    if _competition_service:
        rankings = _competition_service.get_rankings()

    return {
        "models": rankings,
        "active_model": (
            _trading_service.models.active_model_name
            if _trading_service and _trading_service.models
            else None
        ),
        "live_model": (
            _trading_service.models.active_model_name
            if _trading_service and _trading_service.models
            else None
        ),
    }


@router.get("/models/{model_name}/performance")
async def get_model_performance(model_name: str):
    """Detailed performance for a specific model."""
    rankings = []
    if _competition_service:
        rankings = _competition_service.get_rankings()

    for r in rankings:
        if r["model_name"] == model_name:
            return r

    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


@router.get("/models/{model_name}/decisions")
async def get_model_decisions(model_name: str, limit: int = 50):
    """Recent decisions from a model."""
    from db.repositories.decision_repo import DecisionRepository
    from db.session import get_session_ctx

    async with get_session_ctx() as session:
        repo = DecisionRepository(session)
        decisions = await repo.get_recent_decisions(model_name=model_name, limit=limit)

    return {
        "model_name": model_name,
        "count": len(decisions),
        "decisions": [
            {
                "id": d.id,
                "symbol": d.symbol,
                "action": d.action,
                "confidence": d.confidence,
                "reasoning": d.reasoning,
                "was_executed": d.was_executed,
                "outcome": d.outcome,
                "outcome_pnl_pct": d.outcome_pnl_pct,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in decisions
        ],
    }
