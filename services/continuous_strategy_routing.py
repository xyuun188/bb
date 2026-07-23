"""Continuous, paper-only strategy weighting and market-state routing."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.learning import StrategyProfileSnapshot
from services.continuous_model_weight import market_regime_name

CONTINUOUS_STRATEGY_ROUTING_VERSION = "2026-07-22.paper-continuous-strategy-routing.v1"
CONTINUOUS_STRATEGY_SOURCE = "continuous_paper_strategy_router"
MIN_STRATEGY_WEIGHT = 0.05
COLD_START_STRATEGY_WEIGHT = 0.15
MAX_STRATEGY_WEIGHT = 1.40
STRATEGY_WEIGHT_SMOOTHING = 0.25
MAX_ROUTED_CANDIDATES = 40
MAX_ROUTE_CHALLENGERS = 8


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any, default: int = 0) -> int:
    try:
        return max(int(float(value)), 0)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("id") or candidate.get("profile_id") or "").strip()


def _candidate_version(candidate: dict[str, Any]) -> int:
    return max(_int(candidate.get("version")), 1)


def _selector(candidate: dict[str, Any]) -> dict[str, Any]:
    return _dict(_dict(candidate.get("params")).get("selector"))


def _metrics(candidate: dict[str, Any], key: str) -> dict[str, Any]:
    return _dict(_dict(candidate.get(key)).get("metrics"))


def _profit_factor_signal(value: Any) -> float:
    factor = _float(value)
    if factor is None:
        return 0.0
    return _clamp((factor - 1.0) / max(factor + 1.0, 1e-12), -1.0, 1.0)


def _metric_quality(metrics: dict[str, Any]) -> float:
    average = _float(metrics.get("average_net_return_pct"))
    lower = _float(metrics.get("return_lcb_pct"))
    tail = _float(metrics.get("tail_loss_pct"))
    drawdown = max(_float(metrics.get("max_drawdown")) or 0.0, 0.0)
    total = _float(metrics.get("realized_net_pnl_usdt"))
    count = max(_int(metrics.get("sample_count")), 1)
    per_sample_total = total / count if total is not None else None
    scale = max(
        abs(average or 0.0),
        abs(lower or 0.0),
        abs(tail or 0.0),
        abs(per_sample_total or 0.0),
        drawdown,
        1e-12,
    )
    return _clamp(
        0.30 * _clamp((average or 0.0) / scale, -1.0, 1.0)
        + 0.25 * _clamp((lower or 0.0) / scale, -1.0, 1.0)
        + 0.20 * _profit_factor_signal(metrics.get("profit_factor"))
        + 0.15 * _clamp((tail or 0.0) / scale, -1.0, 0.0)
        - 0.10 * _clamp(drawdown / scale, 0.0, 1.0),
        -1.0,
        1.0,
    )


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in (
            "sample_count",
            "realized_net_pnl_usdt",
            "average_net_return_pct",
            "return_lcb_pct",
            "profit_factor",
            "max_drawdown",
            "tail_loss_pct",
        )
    }
def _future_stable(development: dict[str, Any], exam: dict[str, Any]) -> bool:
    development_average = _float(development.get("average_net_return_pct"))
    development_lcb = _float(development.get("return_lcb_pct"))
    exam_average = _float(exam.get("average_net_return_pct"))
    exam_lcb = _float(exam.get("return_lcb_pct"))
    if None in {development_average, development_lcb, exam_average, exam_lcb}:
        return False
    average_tolerance = max(abs(float(development_average)) * 0.50, 0.05)
    lcb_tolerance = max(abs(float(development_lcb)) * 0.50, 0.05)
    return bool(
        float(exam_average) >= float(development_average) - average_tolerance
        and float(exam_lcb) >= float(development_lcb) - lcb_tolerance
    )


def _time_separated(candidate: dict[str, Any]) -> bool:
    backtest = _dict(candidate.get("backtest"))
    exam = _dict(candidate.get("shadow_validation"))
    if backtest.get("status") != "complete" or exam.get("status") != "complete":
        return False
    development_partition = str(backtest.get("evidence_partition") or "")
    exam_partition = str(exam.get("evidence_partition") or "")
    if development_partition == "strategy_development":
        return bool(
            exam_partition == "strategy_exam"
            and exam.get("validation_method")
            == "exact_current_model_on_immutable_shadow_snapshot"
        )
    return bool(
        development_partition == "authoritative_closed_positions"
        and exam_partition == "legacy_shadow"
    )


class ContinuousStrategyRoutingPolicy:
    """Keep validated paper strategies weighted without granting order authority."""

    def __init__(self) -> None:
        self._previous_weights: dict[str, float] = {}
        self._previous_primary_by_regime: dict[str, dict[str, Any]] = {}

    def build(
        self,
        *,
        execution_mode: str,
        market_regime: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        update_state: bool = True,
    ) -> dict[str, Any]:
        mode = "live" if str(execution_mode or "").lower() == "live" else "paper"
        regime = market_regime_name(market_regime)
        if mode != "paper":
            return {
                "version": CONTINUOUS_STRATEGY_ROUTING_VERSION,
                "applied": False,
                "execution_scope": "paper_only",
                "live_strategy_unchanged": True,
                "current_regime": regime,
                "candidate_weights": [],
                "current_route": {},
            }

        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            profile_id = _candidate_id(candidate)
            if not profile_id:
                continue
            selector = _selector(candidate)
            development = _metrics(candidate, "backtest")
            exam = _metrics(candidate, "shadow_validation")
            historical = _dict(
                _dict(candidate.get("params")).get("historical_return_distribution")
            )
            validated = _time_separated(candidate)
            development_count = _int(development.get("sample_count"))
            exam_count = _int(exam.get("sample_count"))
            evidence_count = min(development_count, exam_count)
            confidence = 1.0 - math.exp(-max(evidence_count, 0) / 20.0)
            quality = (
                0.25 * _metric_quality(development)
                + 0.55 * _metric_quality(exam)
                + 0.20 * _metric_quality(historical)
            )
            target = (
                _clamp(
                    COLD_START_STRATEGY_WEIGHT
                    + confidence
                    * (
                        _clamp(1.0 + 0.40 * quality, MIN_STRATEGY_WEIGHT, MAX_STRATEGY_WEIGHT)
                        - COLD_START_STRATEGY_WEIGHT
                    ),
                    MIN_STRATEGY_WEIGHT,
                    MAX_STRATEGY_WEIGHT,
                )
                if validated
                else 0.0
            )
            state_key = f"{regime}:{profile_id}:{_candidate_version(candidate)}"
            previous = self._previous_weights.get(state_key)
            effective = (
                target
                if previous is None
                else previous + STRATEGY_WEIGHT_SMOOTHING * (target - previous)
            )
            if update_state:
                self._previous_weights[state_key] = effective
            scope = str(selector.get("scope") or "")
            candidate_regime = str(selector.get("market_regime") or "").lower()
            matching_regime = not candidate_regime or candidate_regime == regime
            stable = _future_stable(development, exam)
            primary_eligible = bool(
                validated
                and stable
                and matching_regime
                and scope not in {"symbol_side", "symbol_side_horizon"}
            )
            rows.append(
                {
                    "profile_id": profile_id,
                    "profile_version": _candidate_version(candidate),
                    "rank": candidate.get("rank"),
                    "selector": selector,
                    "side": str(selector.get("side") or "").lower(),
                    "candidate_regime": candidate_regime or "all",
                    "validated": validated,
                    "future_stable": stable,
                    "primary_eligible": primary_eligible,
                    "historical_prior_context_eligible": _dict(
                        candidate.get("promotion")
                    ).get("historical_prior_context_eligible")
                    is True,
                    "development_metrics": _compact_metrics(development),
                    "exam_metrics": _compact_metrics(exam),
                    "historical_metrics": _compact_metrics(historical),
                    "evidence_count": evidence_count,
                    "combined_return_quality": round(quality, 8),
                    "target_weight": round(target, 8),
                    "previous_weight": round(previous, 8) if previous is not None else None,
                    "effective_weight": round(effective, 8),
                    "prediction_horizon_minutes": _int(
                        _dict(candidate.get("params")).get("prediction_horizon_minutes")
                    )
                    or None,
                    "routing_reason": (
                        "future_time_validation_stable"
                        if primary_eligible
                        else "single_symbol_strategy_is_challenger_only"
                        if validated and scope in {"symbol_side", "symbol_side_horizon"}
                        else "future_time_performance_degraded"
                        if validated and not stable
                        else "time_separated_validation_incomplete"
                    ),
                }
            )

        current_pool = [row for row in rows if row["primary_eligible"]]
        current_pool.sort(
            key=lambda row: (
                float(row["effective_weight"])
                * (1.05 if row["candidate_regime"] == regime else 1.0),
                -_int(row.get("rank"), 999_999),
            ),
            reverse=True,
        )
        training_pool = [
            row
            for row in rows
            if row["validated"]
            and row["candidate_regime"] in {"all", regime}
        ]
        training_pool.sort(
            key=lambda row: (float(row["effective_weight"]), -_int(row.get("rank"), 999_999)),
            reverse=True,
        )
        primary = dict(current_pool[0]) if current_pool else None
        training_primary = dict(training_pool[0]) if training_pool else None
        selected = primary or training_primary
        previous_primary = self._previous_primary_by_regime.get(regime)
        transition = "no_validated_strategy"
        if selected is not None:
            previous_id = str(_dict(previous_primary).get("profile_id") or "")
            transition = (
                "strategy_retained"
                if previous_id == selected["profile_id"]
                else "market_regime_strategy_switched"
                if previous_id
                else "initial_market_regime_strategy_selected"
            )
            if update_state and primary is not None:
                self._previous_primary_by_regime[regime] = dict(primary)

        total = sum(float(row["effective_weight"]) for row in training_pool)
        normalized = {
            row["profile_id"]: (
                float(row["effective_weight"]) / total if total > 0.0 else 0.0
            )
            for row in training_pool
        }
        for row in rows:
            row["normalized_current_regime_weight"] = round(
                normalized.get(row["profile_id"], 0.0),
                8,
            )
            row["route_role"] = (
                "primary"
                if primary and row["profile_id"] == primary["profile_id"]
                else "training_primary"
                if not primary
                and training_primary
                and row["profile_id"] == training_primary["profile_id"]
                else "challenger"
                if row["validated"]
                else "shadow_only"
            )
        if primary is not None:
            primary["normalized_current_regime_weight"] = round(
                normalized.get(primary["profile_id"], 0.0),
                8,
            )
            primary["route_role"] = "primary"
        if training_primary is not None:
            training_primary["normalized_current_regime_weight"] = round(
                normalized.get(training_primary["profile_id"], 0.0),
                8,
            )
            training_primary["route_role"] = (
                "primary"
                if primary
                and training_primary["profile_id"] == primary["profile_id"]
                else "training_primary"
            )

        current_route = {
            "regime": regime,
            "primary": primary,
            "training_primary": training_primary,
            "recommended_side": _dict(selected).get("side") or "neutral",
            "prediction_horizon_minutes": _dict(selected).get(
                "prediction_horizon_minutes"
            ),
            "transition": transition,
            "previous_primary": previous_primary,
            "challengers": [
                row
                for row in rows
                if row["route_role"] == "challenger"
                and row["candidate_regime"] in {"all", regime}
            ][:MAX_ROUTE_CHALLENGERS],
        }
        routed_rows = sorted(
            rows,
            key=lambda row: (
                row["route_role"] in {"primary", "training_primary"},
                float(row["effective_weight"]),
            ),
            reverse=True,
        )[:MAX_ROUTED_CANDIDATES]
        return {
            "version": CONTINUOUS_STRATEGY_ROUTING_VERSION,
            "applied": True,
            "execution_scope": "paper_only",
            "live_strategy_unchanged": True,
            "objective": "fee_after_total_return_drawdown_tail_loss_not_win_rate",
            "current_regime": regime,
            "candidate_weights": routed_rows,
            "candidate_weight_count": len(rows),
            "candidate_weights_truncated": len(rows) > len(routed_rows),
            "current_route": current_route,
            "validated_candidate_count": sum(row["validated"] for row in rows),
            "primary_candidate_count": sum(row["primary_eligible"] for row in rows),
            "single_symbol_strategy_can_be_primary": False,
            "order_creation_permission": False,
            "risk_override_permission": False,
            "rollback": {
                "previous_stable_primary": previous_primary,
                "mode": "paper_training_prior_only",
            },
            "generated_at": datetime.now(UTC).isoformat(),
        }


class ContinuousStrategyRoutingStore:
    """Persist candidate weights without competing with the legacy champion rows."""

    async def persist(
        self,
        *,
        mode: str,
        candidates: list[dict[str, Any]],
        routing: dict[str, Any],
        session: AsyncSession,
    ) -> dict[str, Any]:
        if str(mode or "").lower() == "live" or routing.get("applied") is not True:
            return {"persisted": False, "reason": "paper_only"}
        candidate_by_id = {_candidate_id(row): row for row in candidates}
        weights = {
            str(row.get("profile_id") or ""): row
            for row in _list(routing.get("candidate_weights"))
            if isinstance(row, dict)
        }
        active_ids: set[tuple[str, int]] = set()
        persisted = 0
        for profile_id, route in weights.items():
            candidate = _dict(candidate_by_id.get(profile_id))
            if not candidate:
                continue
            version = _candidate_version(candidate)
            active_ids.add((profile_id, version))
            row = (
                await session.execute(
                    select(StrategyProfileSnapshot).where(
                        StrategyProfileSnapshot.execution_mode == "paper",
                        StrategyProfileSnapshot.source == CONTINUOUS_STRATEGY_SOURCE,
                        StrategyProfileSnapshot.profile_id == profile_id,
                        StrategyProfileSnapshot.version == version,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                row = StrategyProfileSnapshot(
                    execution_mode="paper",
                    profile_id=profile_id,
                    version=version,
                    source=CONTINUOUS_STRATEGY_SOURCE,
                )
                session.add(row)
            role = str(route.get("route_role") or "shadow_only")
            row.label = str(candidate.get("label") or profile_id)
            row.status = role
            row.description = str(candidate.get("description") or "")
            row.params = {
                **_dict(candidate.get("params")),
                "candidate_source": candidate.get("source"),
                "continuous_route": route,
            }
            row.promotion = {
                **_dict(candidate.get("promotion")),
                "continuous_weight": route.get("effective_weight"),
                "route_role": role,
                "paper_training_influence": bool(route.get("validated")),
                "production_permission": False,
                "live_execution_permission": False,
            }
            row.backtest_metrics = _dict(candidate.get("backtest"))
            row.shadow_validation = _dict(candidate.get("shadow_validation"))
            row.probe_state = {
                **_dict(row.probe_state),
                "last_routed_at": routing.get("generated_at"),
                "current_regime": routing.get("current_regime"),
                "transition": _dict(routing.get("current_route")).get("transition"),
            }
            row.scheduler_reason = str(route.get("routing_reason") or "")
            row.is_active = role == "primary"
            row.is_disabled = False
            persisted += 1

        existing = list(
            (
                await session.execute(
                    select(StrategyProfileSnapshot).where(
                        StrategyProfileSnapshot.execution_mode == "paper",
                        StrategyProfileSnapshot.source == CONTINUOUS_STRATEGY_SOURCE,
                        StrategyProfileSnapshot.is_disabled.is_(False),
                    )
                )
            )
            .scalars()
            .all()
        )
        for row in existing:
            if (row.profile_id, row.version) not in active_ids:
                row.is_active = False
                row.status = "stale_rollback_ready"
                row.scheduler_reason = "not_present_in_latest_strategy_window"
        await session.flush()
        return {
            "persisted": True,
            "candidate_count": persisted,
            "source": CONTINUOUS_STRATEGY_SOURCE,
        }
