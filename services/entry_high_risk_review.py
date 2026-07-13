"""Observation-only external high-risk review context."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def entry_side_value(decision: DecisionOutput) -> str:
    if decision.action == Action.LONG:
        return "long"
    if decision.action == Action.SHORT:
        return "short"
    return "hold"


def entry_expert_disagreement(decision: DecisionOutput) -> float:
    opinions = _safe_list(_safe_dict(decision.raw_response).get("opinions"))
    side = entry_side_value(decision)
    directional = [
        _safe_dict(item)
        for item in opinions
        if str(_safe_dict(item).get("action") or "").lower() in {"long", "short"}
    ]
    opposite = "short" if side == "long" else "long"
    opposite_count = sum(
        str(item.get("action") or "").lower() == opposite for item in directional
    )
    return opposite_count / len(directional) if directional else 0.0


def ml_ai_direction_conflict(decision: DecisionOutput) -> bool:
    raw = _safe_dict(decision.raw_response)
    ml_signal = _safe_dict(raw.get("ml_signal"))
    predictions = _safe_list(ml_signal.get("predictions"))
    primary = _safe_dict(predictions[0]) if predictions else {}
    ml_side = str(primary.get("best_side") or "").lower()
    return bool(ml_side in {"long", "short"} and ml_side != entry_side_value(decision))


@dataclass(slots=True)
class EntryHighRiskReviewGatePolicy:
    """Record external reviewer context without allowing it to gate execution."""

    reviewer: Any | None = None
    allocation_state_provider: Callable[[str], Awaitable[dict[str, Any]]] | None = None
    config: Any = field(default_factory=lambda: settings)

    async def evaluate(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None,
    ) -> None:
        del model_mode, open_positions
        if not decision.is_entry:
            return None
        raw = _safe_dict(decision.raw_response)
        raw["high_risk_review"] = {
            "read_only": True,
            "production_permission": False,
            "configured": bool(self.config.high_risk_review_enabled),
            "expert_disagreement": round(entry_expert_disagreement(decision), 8),
            "ml_ai_direction_conflict": ml_ai_direction_conflict(decision),
            "policy": "RiskEngine and dynamic return execution own production risk decisions.",
        }
        decision.raw_response = raw
        return None
