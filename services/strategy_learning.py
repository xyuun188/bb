"""Observation-only attribution of authoritative fee-after returns."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

import structlog
from sqlalchemy import select

from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text
from db.session import get_read_session_ctx, get_session_ctx
from models.learning import StrategyLearningEvent
from models.trade import Position
from services.text_integrity import sanitize_runtime_text
from services.trade_fact_trust import closed_position_trade_fact_trusted

logger = structlog.get_logger(__name__)

UNTRUSTED_EXPERT_STATUSES = {
    "batch_fallback",
    "partial_batch_fallback",
    "circuit_breaker_fallback",
    "failed",
    "invalid",
    "timeout",
    "timeout_fallback",
}
REQUIRED_ENTRY_EXPERTS = {
    "trend_expert",
    "momentum_expert",
    "sentiment_expert",
    "position_expert",
    "risk_expert",
}
DEFAULT_LOOKBACK_HOURS = 168


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return {}


def _lower_hinge(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    lower = ordered[: max((len(ordered) + 1) // 2, 1)]
    middle = len(lower) // 2
    if len(lower) % 2:
        return lower[middle]
    return (lower[middle - 1] + lower[middle]) / 2.0


def _return_summary(values: list[float]) -> dict[str, Any]:
    profit = sum(max(value, 0.0) for value in values)
    loss = abs(sum(min(value, 0.0) for value in values))
    return {
        "sample_count": len(values),
        "realized_net_pnl_usdt": round(sum(values), 8),
        "average_net_pnl_usdt": round(sum(values) / len(values), 8) if values else None,
        "pnl_lower_hinge_usdt": (
            round(float(_lower_hinge(values)), 8) if values else None
        ),
        "gross_profit_usdt": round(profit, 8),
        "gross_loss_usdt": round(loss, 8),
        "profit_factor": round(profit / loss, 8) if loss > 0 else None,
        "negative_sample_count": sum(value < 0 for value in values),
        "positive_sample_count": sum(value > 0 for value in values),
    }


@dataclass(frozen=True, slots=True)
class StrategyProfile:
    profile_id: str
    version: int
    label: str
    status: str
    source: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.profile_id,
            "version": self.version,
            "label": self.label,
            "status": self.status,
            "source": self.source,
            "description": self.description,
            "params": {},
            "promotion": {
                **self.promotion,
                "production_permission": False,
            },
        }


@dataclass(frozen=True, slots=True)
class StrategyFeedback:
    mode: str
    window_hours: int
    generated_at: str
    totals: dict[str, Any]
    side_performance: dict[str, dict[str, Any]]
    open_position_pressure: dict[str, Any]
    decision_quality: dict[str, Any]
    shadow_feedback: dict[str, Any]
    expert_memory: dict[str, Any]
    manual_intervention: dict[str, Any]
    trade_fact_quarantine: dict[str, Any]
    reflection_feedback: dict[str, Any]
    event_feedback: dict[str, Any]
    authoritative_return_observation: dict[str, Any]
    problems: list[dict[str, Any]]
    root_causes: list[str]
    training_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            name: _json_safe(getattr(self, name))
            for name in self.__dataclass_fields__
        }


class StrategyCandidateGenerator:
    """Expose one parameter-free observation profile."""

    @staticmethod
    def baseline(_feedback: StrategyFeedback) -> StrategyProfile:
        return StrategyProfile(
            profile_id="authoritative_return_observation",
            version=1,
            label="Authoritative return observation",
            status="observation_only",
            source="closed_position_fee_after_return",
            description="Read-only attribution; cannot affect production execution.",
            params={},
            promotion={"production_permission": False},
        )

    def generate(self, feedback: StrategyFeedback) -> list[StrategyProfile]:
        return [self.baseline(feedback)]


class StrategyLearningEngine:
    def __init__(self, generator: StrategyCandidateGenerator | None = None, **_: Any) -> None:
        self.generator = generator or StrategyCandidateGenerator()

    def build_from_feedback(
        self,
        feedback: StrategyFeedback,
        *,
        extra_profiles: list[StrategyProfile] | None = None,
    ) -> dict[str, Any]:
        del extra_profiles
        profile = self.generator.generate(feedback)[0]
        profile_payload = profile.to_dict()
        schedule = {
            "active_profile": profile_payload,
            "reason": "Observation only; dynamic return execution owns production decisions.",
            "runtime": {
                "read_only": True,
                "production_permission": False,
                "optimization_target": "realized_fee_after_return",
            },
            "candidates": [profile_payload],
            "backtest": {"rows": []},
            "shadow_validation": {
                "read_only": True,
                "production_permission": False,
                "rows": [],
            },
            "scheduler_mode": "observation_only",
        }
        return {
            "feedback": feedback.to_dict(),
            "schedule": schedule,
            "active_profile": profile_payload,
        }

    def apply_to_context(
        self,
        strategy_context: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        result = dict(strategy_context or {})
        feedback = _safe_dict(payload.get("feedback"))
        observation = _safe_dict(feedback.get("authoritative_return_observation"))
        sample_count = _safe_int(observation.get("sample_count"))
        result["strategy_learning"] = {
            "read_only": True,
            "production_permission": False,
            "optimization_target": "realized_fee_after_return",
            "authoritative_return_observation": observation,
            "policy_provenance": {
                "source": "authoritative_closed_position_return_attribution",
                "observation_window": str(feedback.get("window_hours") or "available"),
                "sample_count": sample_count,
                "generated_at": str(
                    feedback.get("generated_at") or datetime.now(UTC).isoformat()
                ),
                "strategy_version": "2026-07-12.strategy-learning-observation-only.v2",
                "fallback_reason": "" if sample_count > 0 else "no_authoritative_closed_samples",
            },
        }
        return result


class StrategyLearningService:
    def __init__(self, *, engine: StrategyLearningEngine | None = None, **_: Any) -> None:
        self.engine = engine or StrategyLearningEngine()

    async def dashboard_payload(
        self,
        *,
        mode: str,
        hours: int = DEFAULT_LOOKBACK_HOURS,
        limit: int = 500,
        detail: str = "summary",
    ) -> dict[str, Any]:
        del detail
        feedback = await self._feedback(mode=mode, hours=hours, limit=limit)
        payload = self.engine.build_from_feedback(feedback)
        payload.update(
            {
                "mode": mode,
                "window_hours": hours,
                "sample_limit": limit,
                "read_only": True,
                "production_permission": False,
            }
        )
        return payload

    async def apply_to_strategy_context(
        self,
        *,
        mode: str,
        strategy_context: dict[str, Any],
        open_positions: list[dict[str, Any]] | None,
        hours: int = DEFAULT_LOOKBACK_HOURS,
        limit: int = 500,
    ) -> dict[str, Any]:
        del open_positions
        feedback = await self._feedback(mode=mode, hours=hours, limit=limit)
        return self.engine.apply_to_context(
            strategy_context,
            self.engine.build_from_feedback(feedback),
        )

    async def _feedback(self, *, mode: str, hours: int, limit: int) -> StrategyFeedback:
        since = datetime.now(UTC) - timedelta(hours=max(int(hours or 1), 1))
        since_naive = since.replace(tzinfo=None)
        selected_mode = "live" if str(mode).lower() == "live" else "paper"
        async with get_read_session_ctx() as session:
            closed = list(
                (
                    await session.execute(
                        select(Position)
                        .where(
                            Position.execution_mode == selected_mode,
                            Position.is_open.is_(False),
                            Position.closed_at >= since_naive,
                        )
                        .order_by(Position.closed_at.desc())
                        .limit(max(int(limit or 1), 1))
                    )
                )
                .scalars()
                .all()
            )
            open_rows = list(
                (
                    await session.execute(
                        select(Position).where(
                            Position.execution_mode == selected_mode,
                            Position.is_open.is_(True),
                        )
                    )
                )
                .scalars()
                .all()
            )

        raw_closed_count = len(closed)
        closed = [
            row
            for row in closed
            if closed_position_trade_fact_trusted(row) and self._cost_complete(row)
        ]
        net_values = [self._fee_after_pnl(row) for row in closed]
        side_performance = {
            side: _return_summary(
                [
                    self._fee_after_pnl(row)
                    for row in closed
                    if str(getattr(row, "side", "") or "").lower() == side
                ]
            )
            for side in ("long", "short")
        }
        observation = {
            **_return_summary(net_values),
            "cost_complete_sample_count": len(closed),
            "excluded_incomplete_or_untrusted_count": raw_closed_count - len(closed),
            "optimization_target": "realized_fee_after_return",
        }
        generated_at = datetime.now(UTC).isoformat()
        return StrategyFeedback(
            mode=selected_mode,
            window_hours=max(int(hours or 1), 1),
            generated_at=generated_at,
            totals=observation,
            side_performance=side_performance,
            open_position_pressure={"open_position_count": len(open_rows)},
            decision_quality={},
            shadow_feedback={"read_only": True, "production_permission": False},
            expert_memory={"read_only": True, "production_permission": False},
            manual_intervention={},
            trade_fact_quarantine={},
            reflection_feedback={},
            event_feedback={},
            authoritative_return_observation=observation,
            problems=[],
            root_causes=[],
            training_policy={
                "optimization_target": "realized_fee_after_return",
                "production_permission": False,
                "cost_complete_samples_required": True,
            },
        )

    @staticmethod
    def _fee_after_pnl(position: Position) -> float:
        return (
            _safe_float(getattr(position, "realized_pnl", None))
            - max(_safe_float(getattr(position, "entry_fee", None)), 0.0)
            - max(_safe_float(getattr(position, "close_fee", None)), 0.0)
            + _safe_float(getattr(position, "funding_fee", None))
        )

    @staticmethod
    def _cost_complete(position: Position) -> bool:
        return all(
            getattr(position, field, None) is not None
            for field in ("entry_fee", "close_fee", "funding_fee")
        )

    async def record_event(
        self,
        *,
        mode: str,
        model_name: str = ENSEMBLE_TRADER_NAME,
        symbol: str | None = None,
        decision: Any | None = None,
        action: str | None = None,
        event_type: str,
        event_status: str = "recorded",
        reason: str | None = None,
        severity: str = "info",
        decision_id: int | None = None,
        order_id: int | None = None,
        position_id: int | None = None,
        strategy_context: dict[str, Any] | None = None,
        market_state: dict[str, Any] | None = None,
        attribution: dict[str, Any] | None = None,
        exclude_from_training: bool = False,
        raw_response: dict[str, Any] | None = None,
        **_: Any,
    ) -> int | None:
        context = _safe_dict(strategy_context)
        learning = _safe_dict(context.get("strategy_learning"))
        decision_action = action or str(getattr(getattr(decision, "action", None), "value", ""))
        side = "long" if "long" in decision_action else "short" if "short" in decision_action else None
        event = StrategyLearningEvent(
            model_name=model_name,
            execution_mode="live" if str(mode).lower() == "live" else "paper",
            symbol=symbol,
            side=side,
            action=decision_action or None,
            event_type=event_type,
            event_status=event_status,
            severity=severity,
            reason=str(sanitize_runtime_text(reason or "") or "")[:2000],
            decision_id=decision_id,
            order_id=order_id,
            position_id=position_id,
            profile_id=None,
            profile_version=None,
            scheduler_reason=str(
                sanitize_runtime_text(context.get("scheduler_reason") or "") or ""
            )[:2000],
            strategy_snapshot=sanitize_runtime_text(
                _json_safe(
                    {
                        "read_only": True,
                        "production_permission": False,
                        "authoritative_return_observation": learning.get(
                            "authoritative_return_observation", {}
                        ),
                    }
                )
            ),
            market_state=sanitize_runtime_text(_json_safe(market_state or {})),
            side_weights=None,
            expert_integrity=sanitize_runtime_text(
                _json_safe({"observation_only": True, "raw": raw_response or {}})
            ),
            attribution=sanitize_runtime_text(_json_safe(attribution or {})),
            exclude_from_training=bool(exclude_from_training),
        )
        try:
            async with get_session_ctx() as session:
                session.add(event)
                await session.flush()
                return int(event.id)
        except Exception as exc:
            logger.warning("failed to record strategy observation", error=safe_error_text(exc))
            return None
