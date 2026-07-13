"""Dynamic strategy scheduling from authoritative fee-after return evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import ceil, isfinite, sqrt
from typing import Any

import structlog
from sqlalchemy import select

from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text
from db.session import get_read_session_ctx, get_session_ctx
from models.learning import ShadowBacktest, StrategyLearningEvent
from models.trade import Position
from services.shadow_backtest_service import shadow_fee_after_outcome
from services.text_integrity import sanitize_runtime_text
from services.trade_fact_trust import (
    closed_position_trade_fact_trusted,
    closed_position_trade_fact_untrusted_reason,
)

logger = structlog.get_logger(__name__)

DEFAULT_LOOKBACK_HOURS = 168
STRATEGY_SCHEDULER_VERSION = "2026-07-13.dynamic-return-scheduler.v1"
EXECUTION_OWNERS = (
    "return_execution_policy",
    "dynamic_entry_risk_budget",
    "dynamic_position_capacity",
    "dynamic_exit_policy",
)


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


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


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


def _timestamp_text(value: Any) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value or "")


def _regime_label(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    source = _safe_dict(value)
    for key in ("mode", "regime", "market_regime", "state", "label"):
        nested = source.get(key)
        if isinstance(nested, str) and nested.strip():
            return nested.strip().lower()
        if isinstance(nested, dict):
            label = _regime_label(nested)
            if label:
                return label
    return ""


def _dynamic_blocks(samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Partition evidence from its own cardinality instead of a configured gate."""

    if not samples:
        return []
    ordered = sorted(samples, key=lambda item: str(item.get("timestamp") or ""))
    fold_count = ceil(sqrt(len(ordered)))
    block_size = ceil(len(ordered) / fold_count)
    return [ordered[index : index + block_size] for index in range(0, len(ordered), block_size)]


def _max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def _return_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [
        float(value)
        for sample in samples
        if (value := _optional_float(sample.get("net_return_after_cost_pct"))) is not None
    ]
    pnls = [
        float(value)
        for sample in samples
        if (value := _optional_float(sample.get("net_pnl_after_all_costs_usdt"))) is not None
    ]
    gross_profit = sum(max(value, 0.0) for value in pnls)
    gross_loss = abs(sum(min(value, 0.0) for value in pnls))
    if not pnls:
        gross_profit = sum(max(value, 0.0) for value in returns)
        gross_loss = abs(sum(min(value, 0.0) for value in returns))
    blocks = _dynamic_blocks(samples)
    block_means = [
        sum(values) / len(values)
        for block in blocks
        if (
            values := [
                float(value)
                for sample in block
                if (
                    value := _optional_float(sample.get("net_return_after_cost_pct"))
                )
                is not None
            ]
        )
    ]
    drawdown_values = pnls if pnls else returns
    return {
        "sample_count": len(returns),
        "realized_net_pnl_usdt": round(sum(pnls), 8) if pnls else None,
        "average_net_pnl_usdt": round(sum(pnls) / len(pnls), 8) if pnls else None,
        "average_net_return_pct": (
            round(sum(returns) / len(returns), 8) if returns else None
        ),
        "return_lcb_pct": round(min(block_means), 8) if block_means else None,
        "gross_profit_usdt": round(gross_profit, 8),
        "gross_loss_usdt": round(gross_loss, 8),
        "profit_factor": round(gross_profit / gross_loss, 8) if gross_loss else None,
        "profit_factor_above_break_even": bool(
            gross_profit > 0 and (gross_loss == 0 or gross_profit > gross_loss)
        ),
        "max_drawdown": round(_max_drawdown(drawdown_values), 8),
        "tail_loss_pct": round(min(returns), 8) if returns else None,
        "negative_sample_count": sum(value < 0 for value in returns),
        "positive_sample_count": sum(value > 0 for value in returns),
        "fold_count": len(block_means),
    }


def _legacy_observation_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _return_metrics(samples)
    pnls = [
        float(value)
        for sample in samples
        if (value := _optional_float(sample.get("net_pnl_after_all_costs_usdt"))) is not None
    ]
    ordered = sorted(pnls)
    lower = ordered[: ceil(len(ordered) / 2)]
    if not lower:
        lower_hinge = None
    elif len(lower) % 2:
        lower_hinge = lower[len(lower) // 2]
    else:
        middle = len(lower) // 2
        lower_hinge = (lower[middle - 1] + lower[middle]) / 2.0
    return {
        **metrics,
        "pnl_lower_hinge_usdt": round(lower_hinge, 8) if lower_hinge is not None else None,
        "optimization_target": "authoritative_fee_after_return_rate",
    }


def _selector_matches(selector: dict[str, Any], sample: dict[str, Any]) -> bool:
    side = str(selector.get("side") or "").lower()
    symbol = str(selector.get("symbol") or "").upper()
    regime = str(selector.get("market_regime") or "").lower()
    if side and str(sample.get("side") or "").lower() != side:
        return False
    if symbol and str(sample.get("symbol") or "").upper() != symbol:
        return False
    return not regime or str(sample.get("market_regime") or "").lower() == regime


def _profile_identity(selector: dict[str, Any]) -> str:
    canonical = json.dumps(selector, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()[:16]
    return f"fee_after_return_{selector['scope']}_{digest}"


def _profile_label(selector: dict[str, Any]) -> str:
    side = str(selector.get("side") or "").upper()
    if selector.get("scope") == "symbol_side":
        return f"{selector.get('symbol')} {side} return policy"
    if selector.get("scope") == "regime_side":
        return f"{selector.get('market_regime')} {side} return policy"
    return f"Portfolio {side} return policy"


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
            "params": _json_safe(self.params),
            "promotion": {
                **_json_safe(self.promotion),
                "can_authorize_entry": False,
                "can_change_size_or_leverage": False,
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
    authoritative_return_samples: list[dict[str, Any]] = field(default_factory=list)
    shadow_return_samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        payload = {
            name: _json_safe(getattr(self, name))
            for name in self.__dataclass_fields__
            if include_samples or not name.endswith("_samples")
        }
        if include_samples:
            payload["authoritative_return_samples"] = _json_safe(
                self.authoritative_return_samples
            )
            payload["shadow_return_samples"] = _json_safe(self.shadow_return_samples)
        return payload


class StrategyCandidateGenerator:
    """Generate a candidate for every observed return partition."""

    def generate(self, feedback: StrategyFeedback) -> list[StrategyProfile]:
        partitions: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        for sample in feedback.authoritative_return_samples:
            side = str(sample.get("side") or "").lower()
            symbol = str(sample.get("symbol") or "").upper()
            regime = str(sample.get("market_regime") or "").lower()
            selectors = [{"scope": "side", "side": side}]
            if symbol:
                selectors.append(
                    {"scope": "symbol_side", "symbol": symbol, "side": side}
                )
            if regime:
                selectors.append(
                    {
                        "scope": "regime_side",
                        "market_regime": regime,
                        "side": side,
                    }
                )
            for selector in selectors:
                identity = _profile_identity(selector)
                if identity not in partitions:
                    partitions[identity] = (selector, [])
                partitions[identity][1].append(sample)

        profiles: list[StrategyProfile] = []
        for profile_id, (selector, samples) in sorted(partitions.items()):
            metrics = _return_metrics(samples)
            version = max(_safe_int(sample.get("source_id")) for sample in samples)
            provenance = {
                "source": "trusted_cost_complete_closed_positions",
                "observation_window": f"trailing_{feedback.window_hours}_hours",
                "sample_count": len(samples),
                "generated_at": feedback.generated_at,
                "strategy_version": STRATEGY_SCHEDULER_VERSION,
                "fallback_reason": "",
                "position_ids": sorted(
                    _safe_int(sample.get("source_id")) for sample in samples
                ),
            }
            profiles.append(
                StrategyProfile(
                    profile_id=profile_id,
                    version=version,
                    label=_profile_label(selector),
                    status="candidate",
                    source="authoritative_fee_after_return_partition",
                    description=(
                        "Historical return prior; current live return, execution cost, account "
                        "risk, and position contracts remain mandatory."
                    ),
                    params={
                        "selector": selector,
                        "objective": "maximize_authoritative_fee_after_return_rate",
                        "historical_return_distribution": metrics,
                        "current_return_contract_required": True,
                        "execution_owners": list(EXECUTION_OWNERS),
                        "policy_provenance": provenance,
                    },
                    promotion={"evaluation_state": "pending"},
                )
            )
        return profiles


def _walk_forward_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    blocks = _dynamic_blocks(samples)
    if len(blocks) <= 1:
        return {
            "status": "insufficient_chronological_partitions",
            "rows": [],
            "metrics": _return_metrics([]),
            "partition_policy": "sqrt_cardinality_expanding_walk_forward",
        }
    rows: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = list(blocks[0])
    validation_samples: list[dict[str, Any]] = []
    for block in blocks[1:]:
        metrics = _return_metrics(block)
        validation_samples.extend(block)
        rows.append(
            {
                "training_sample_count": len(training),
                "validation_sample_count": len(block),
                "validation_start": block[0].get("timestamp"),
                "validation_end": block[-1].get("timestamp"),
                "metrics": metrics,
            }
        )
        training.extend(block)
    metrics = _return_metrics(validation_samples)
    fold_lcbs = [
        _optional_float(_safe_dict(row.get("metrics")).get("average_net_return_pct"))
        for row in rows
    ]
    fold_lcbs = [value for value in fold_lcbs if value is not None]
    metrics["return_lcb_pct"] = round(min(fold_lcbs), 8) if fold_lcbs else None
    return {
        "status": "complete" if rows else "insufficient_chronological_partitions",
        "rows": rows,
        "metrics": metrics,
        "partition_policy": "sqrt_cardinality_expanding_walk_forward",
    }


def _shadow_report(
    samples: list[dict[str, Any]],
    *,
    include_rows: bool,
) -> dict[str, Any]:
    metrics = _return_metrics(samples)
    return {
        "status": "complete" if samples else "no_cost_complete_shadow_samples",
        "rows": (
            [
                {
                    "source_id": sample.get("source_id"),
                    "symbol": sample.get("symbol"),
                    "side": sample.get("side"),
                    "net_return_after_cost_pct": sample.get(
                        "net_return_after_cost_pct"
                    ),
                    "execution_cost_pct": sample.get("execution_cost_pct"),
                    "timestamp": sample.get("timestamp"),
                }
                for sample in samples
            ]
            if include_rows
            else []
        ),
        "row_detail_included": include_rows,
        "metrics": metrics,
        "cost_contract": "live_fee_spread_depth_imbalance_required",
    }


def _candidate_rejections(
    backtest: dict[str, Any], shadow: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    backtest_metrics = _safe_dict(backtest.get("metrics"))
    shadow_metrics = _safe_dict(shadow.get("metrics"))
    if backtest.get("status") != "complete":
        reasons.append(str(backtest.get("status") or "walk_forward_incomplete"))
    if (_optional_float(backtest_metrics.get("return_lcb_pct")) or 0.0) <= 0:
        reasons.append("walk_forward_fee_after_return_lcb_not_positive")
    if backtest_metrics.get("profit_factor_above_break_even") is not True:
        reasons.append("walk_forward_profit_factor_not_above_break_even")
    if shadow.get("status") != "complete":
        reasons.append(str(shadow.get("status") or "shadow_validation_incomplete"))
    if (_optional_float(shadow_metrics.get("return_lcb_pct")) or 0.0) <= 0:
        reasons.append("shadow_fee_after_return_lcb_not_positive")
    if shadow_metrics.get("profit_factor_above_break_even") is not True:
        reasons.append("shadow_profit_factor_not_above_break_even")
    return list(dict.fromkeys(reasons))


def _rank_value(value: Any, *, descending: bool = True) -> float:
    number = _optional_float(value)
    if number is None:
        return float("-inf") if descending else float("inf")
    return number


def _candidate_rank_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    backtest = _safe_dict(candidate.get("backtest"))
    shadow = _safe_dict(candidate.get("shadow_validation"))
    historical = _safe_dict(_safe_dict(candidate.get("params")).get("historical_return_distribution"))
    backtest_metrics = _safe_dict(backtest.get("metrics"))
    shadow_metrics = _safe_dict(shadow.get("metrics"))
    return (
        bool(_safe_dict(candidate.get("promotion")).get("production_influence_eligible")),
        _rank_value(backtest_metrics.get("return_lcb_pct")),
        _rank_value(shadow_metrics.get("return_lcb_pct")),
        _rank_value(historical.get("realized_net_pnl_usdt")),
        bool(historical.get("profit_factor_above_break_even")),
        -_rank_value(historical.get("max_drawdown"), descending=False),
        _rank_value(historical.get("tail_loss_pct")),
    )


class StrategyLearningEngine:
    def __init__(self, generator: StrategyCandidateGenerator | None = None, **_: Any) -> None:
        self.generator = generator or StrategyCandidateGenerator()

    def build_from_feedback(
        self,
        feedback: StrategyFeedback,
        *,
        extra_profiles: list[StrategyProfile] | None = None,
        current_context: dict[str, Any] | None = None,
        detail: str = "summary",
    ) -> dict[str, Any]:
        del extra_profiles
        candidates: list[dict[str, Any]] = []
        backtest_rows: list[dict[str, Any]] = []
        shadow_rows: list[dict[str, Any]] = []
        include_evidence_rows = detail == "full"
        for profile in self.generator.generate(feedback):
            selector = _safe_dict(profile.params.get("selector"))
            authoritative = [
                sample
                for sample in feedback.authoritative_return_samples
                if _selector_matches(selector, sample)
            ]
            shadows = [
                sample
                for sample in feedback.shadow_return_samples
                if _selector_matches(selector, sample)
            ]
            backtest = _walk_forward_report(authoritative)
            shadow = _shadow_report(shadows, include_rows=include_evidence_rows)
            rejection_reasons = _candidate_rejections(backtest, shadow)
            influence_eligible = not rejection_reasons
            profile_payload = profile.to_dict()
            profile_payload["status"] = "governed" if influence_eligible else "shadow_validation"
            profile_payload["promotion"] = {
                **_safe_dict(profile_payload.get("promotion")),
                "evaluation_state": "governed" if influence_eligible else "blocked",
                "production_influence_eligible": influence_eligible,
                "rejection_reasons": rejection_reasons,
                "can_authorize_entry": False,
                "can_change_size_or_leverage": False,
                "production_permission": False,
            }
            profile_payload["backtest"] = backtest
            profile_payload["shadow_validation"] = shadow
            candidates.append(profile_payload)
            backtest_rows.append(
                {"profile_id": profile.profile_id, "label": profile.label, **backtest}
            )
            shadow_rows.append(
                {"profile_id": profile.profile_id, "label": profile.label, **shadow}
            )

        candidates.sort(key=_candidate_rank_key, reverse=True)
        for rank, candidate in enumerate(candidates, start=1):
            candidate["rank"] = rank
        governed = [
            candidate
            for candidate in candidates
            if _safe_dict(candidate.get("promotion")).get("production_influence_eligible")
            is True
        ]
        context = _safe_dict(current_context)
        current_regime = _regime_label(context.get("market_regime"))
        applicable = [
            candidate
            for candidate in governed
            if _safe_dict(_safe_dict(candidate.get("params")).get("selector")).get("scope")
            != "symbol_side"
            and (
                not _safe_dict(_safe_dict(candidate.get("params")).get("selector")).get(
                    "market_regime"
                )
                or _safe_dict(_safe_dict(candidate.get("params")).get("selector")).get(
                    "market_regime"
                )
                == current_regime
            )
        ]
        leading = (candidates or [None])[0]
        active = (applicable or [None])[0]
        influence_enabled = bool(governed)
        scheduler_mode = (
            "governed_dynamic_return"
            if influence_enabled
            else "shadow_validation"
            if candidates
            else "insufficient_authoritative_evidence"
        )
        reason = (
            "Governed fee-after return candidates are available as matching historical priors; "
            "the current live return and dynamic risk contracts still own execution."
            if influence_enabled
            else "Candidates remain in shadow because walk-forward or cost-complete shadow evidence is incomplete."
            if candidates
            else "No trusted cost-complete return-rate samples are available for candidate generation."
        )
        governed_runtime_profiles = [
            {
                "id": candidate.get("id"),
                "version": candidate.get("version"),
                "rank": candidate.get("rank"),
                "selector": _safe_dict(_safe_dict(candidate.get("params")).get("selector")),
                "historical_return_distribution": _safe_dict(
                    _safe_dict(candidate.get("params")).get("historical_return_distribution")
                ),
                "walk_forward": _safe_dict(candidate.get("backtest")).get("metrics"),
                "shadow_validation": _safe_dict(candidate.get("shadow_validation")).get(
                    "metrics"
                ),
                "policy_provenance": _safe_dict(
                    _safe_dict(candidate.get("params")).get("policy_provenance")
                ),
            }
            for candidate in governed
        ]
        runtime = {
            "optimization_target": "maximize_authoritative_fee_after_return_rate",
            "production_influence_enabled": influence_enabled,
            "can_authorize_entry": False,
            "can_change_size_or_leverage": False,
            "current_return_contract_required": True,
            "execution_owners": list(EXECUTION_OWNERS),
            "governed_profiles": governed_runtime_profiles,
            "current_market_regime": current_regime or None,
            "account_state": {
                "account_equity": context.get("account_equity"),
                "drawdown_pressure": context.get("drawdown_pressure"),
                "position_exposure": context.get("position_exposure"),
            },
            "policy_provenance": {
                "source": "authoritative_walk_forward_and_cost_complete_shadow_scheduler",
                "observation_window": f"trailing_{feedback.window_hours}_hours",
                "sample_count": len(feedback.authoritative_return_samples),
                "generated_at": feedback.generated_at,
                "strategy_version": STRATEGY_SCHEDULER_VERSION,
                "fallback_reason": "" if influence_enabled else scheduler_mode,
            },
        }
        schedule = {
            "active_profile": active,
            "leading_candidate": leading,
            "reason": reason,
            "runtime": runtime,
            "candidates": candidates,
            "candidate_count": len(candidates),
            "governed_candidate_count": len(governed),
            "rejected_candidate_count": len(candidates) - len(governed),
            "backtest": {"rows": backtest_rows},
            "shadow_validation": {
                "cost_complete_required": True,
                "can_authorize_entry": False,
                "rows": shadow_rows,
            },
            "scheduler_mode": scheduler_mode,
        }
        return {
            "feedback": feedback.to_dict(include_samples=detail == "full"),
            "schedule": schedule,
            "active_profile": active,
        }

    def apply_to_context(
        self,
        strategy_context: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        result = dict(strategy_context or {})
        schedule = _safe_dict(payload.get("schedule"))
        runtime = _safe_dict(schedule.get("runtime"))
        active = _safe_dict(schedule.get("active_profile"))
        leading = _safe_dict(schedule.get("leading_candidate"))
        result["strategy_profile_id"] = active.get("id")
        result["strategy_profile_version"] = active.get("version")
        result["scheduler_reason"] = schedule.get("reason")
        result["strategy_learning"] = {
            "scheduler_mode": schedule.get("scheduler_mode"),
            "candidate_count": schedule.get("candidate_count"),
            "governed_candidate_count": schedule.get("governed_candidate_count"),
            "rejected_candidate_count": schedule.get("rejected_candidate_count"),
            "active_profile": active,
            "leading_candidate": leading,
            "runtime": runtime,
            "advisory_prior_only": True,
            "production_permission": False,
            "policy_provenance": runtime.get("policy_provenance"),
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
        feedback = await self._feedback(mode=mode, hours=hours, limit=limit)
        payload = self.engine.build_from_feedback(feedback, detail=detail)
        payload.update(
            {
                "mode": mode,
                "window_hours": hours,
                "sample_limit": limit,
                "optimization_target": "maximize_authoritative_fee_after_return_rate",
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
        feedback = await self._feedback(mode=mode, hours=hours, limit=limit)
        if open_positions is not None:
            feedback.open_position_pressure["runtime_open_position_count"] = len(open_positions)
        payload = self.engine.build_from_feedback(
            feedback,
            current_context=strategy_context,
            detail="summary",
        )
        return self.engine.apply_to_context(strategy_context, payload)

    async def _feedback(self, *, mode: str, hours: int, limit: int) -> StrategyFeedback:
        selected_mode = "live" if str(mode).lower() == "live" else "paper"
        effective_hours = max(int(hours or 1), 1)
        effective_limit = max(int(limit or 1), 1)
        since = datetime.now(UTC) - timedelta(hours=effective_hours)
        since_naive = since.replace(tzinfo=None)
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
                        .limit(effective_limit)
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
            position_ids = [int(row.id) for row in closed if getattr(row, "id", None)]
            events = (
                list(
                    (
                        await session.execute(
                            select(StrategyLearningEvent)
                            .where(
                                StrategyLearningEvent.execution_mode == selected_mode,
                                StrategyLearningEvent.position_id.in_(position_ids),
                            )
                            .order_by(StrategyLearningEvent.created_at.desc())
                        )
                    )
                    .scalars()
                    .all()
                )
                if position_ids
                else []
            )
            shadows = list(
                (
                    await session.execute(
                        select(ShadowBacktest)
                        .where(
                            ShadowBacktest.execution_mode == selected_mode,
                            ShadowBacktest.status == "completed",
                            ShadowBacktest.created_at >= since_naive,
                        )
                        .order_by(ShadowBacktest.created_at.desc())
                        .limit(effective_limit)
                    )
                )
                .scalars()
                .all()
            )

        regime_by_position: dict[int, str] = {}
        for event in events:
            position_id = _safe_int(getattr(event, "position_id", None))
            regime = _regime_label(getattr(event, "market_state", None))
            if position_id and regime:
                regime_by_position.setdefault(position_id, regime)

        quarantine_reasons: dict[str, int] = {}
        authoritative_samples: list[dict[str, Any]] = []
        for row in closed:
            untrusted = closed_position_trade_fact_untrusted_reason(row)
            if untrusted or not closed_position_trade_fact_trusted(row):
                reason = untrusted or "untrusted_trade_fact"
                quarantine_reasons[reason] = quarantine_reasons.get(reason, 0) + 1
                continue
            if not self._cost_complete(row):
                quarantine_reasons["cost_incomplete"] = (
                    quarantine_reasons.get("cost_incomplete", 0) + 1
                )
                continue
            sample = self._position_return_sample(
                row,
                market_regime=regime_by_position.get(_safe_int(getattr(row, "id", None)), ""),
            )
            if sample is None:
                quarantine_reasons["margin_basis_incomplete"] = (
                    quarantine_reasons.get("margin_basis_incomplete", 0) + 1
                )
                continue
            authoritative_samples.append(sample)

        shadow_samples: list[dict[str, Any]] = []
        shadow_excluded: dict[str, int] = {}
        for row in shadows:
            snapshot = _safe_dict(getattr(row, "feature_snapshot", None))
            long_gross = _optional_float(getattr(row, "long_return_pct", None))
            short_gross = _optional_float(getattr(row, "short_return_pct", None))
            if long_gross is None or short_gross is None:
                shadow_excluded["direction_return_missing"] = (
                    shadow_excluded.get("direction_return_missing", 0) + 1
                )
                continue
            outcome = shadow_fee_after_outcome(
                row,
                long_return=long_gross / 100.0,
                short_return=short_gross / 100.0,
            )
            if outcome.get("cost_complete") is not True:
                reasons = _safe_list(outcome.get("incomplete_reasons")) or [
                    "shadow_cost_contract_incomplete"
                ]
                for reason in reasons:
                    reason_text = str(reason)
                    shadow_excluded[reason_text] = shadow_excluded.get(reason_text, 0) + 1
                continue
            execution_cost = _safe_dict(outcome.get("execution_cost"))
            for side in ("long", "short"):
                gross_return = long_gross if side == "long" else short_gross
                net_return = _optional_float(
                    outcome.get(f"{side}_net_return_after_cost_pct")
                )
                shadow_samples.append(
                    {
                        "source": "completed_shadow_with_live_cost_snapshot",
                        "source_id": _safe_int(getattr(row, "id", None)),
                        "symbol": str(getattr(row, "symbol", "") or "").upper(),
                        "side": side,
                        "market_regime": _regime_label(snapshot.get("market_regime")),
                        "net_return_after_cost_pct": (
                            round(net_return, 8) if net_return is not None else None
                        ),
                        "gross_return_pct": round(gross_return, 8),
                        "execution_cost_pct": round(
                            _safe_float(outcome.get("fee_return_pct"))
                            + _safe_float(outcome.get("slippage_return_pct")),
                            8,
                        ),
                        "funding_return_pct": outcome.get(
                            f"funding_return_{side}_pct"
                        ),
                        "timestamp": _timestamp_text(getattr(row, "created_at", None)),
                        "cost_policy_provenance": _safe_dict(
                            execution_cost.get("policy_provenance")
                        ),
                    }
                )

        side_performance = {
            side: _legacy_observation_summary(
                [sample for sample in authoritative_samples if sample.get("side") == side]
            )
            for side in ("long", "short")
        }
        observation = {
            **_legacy_observation_summary(authoritative_samples),
            "cost_complete_sample_count": len(authoritative_samples),
            "excluded_incomplete_or_untrusted_count": len(closed)
            - len(authoritative_samples),
        }
        generated_at = datetime.now(UTC).isoformat()
        problems = [
            {"code": code, "count": count, "kind": "authoritative_sample_excluded"}
            for code, count in sorted(quarantine_reasons.items())
        ] + [
            {"code": code, "count": count, "kind": "shadow_sample_excluded"}
            for code, count in sorted(shadow_excluded.items())
        ]
        return StrategyFeedback(
            mode=selected_mode,
            window_hours=effective_hours,
            generated_at=generated_at,
            totals=observation,
            side_performance=side_performance,
            open_position_pressure={
                "open_position_count": len(open_rows),
                "unrealized_pnl_usdt": round(
                    sum(_safe_float(getattr(row, "unrealized_pnl", None)) for row in open_rows),
                    8,
                ),
            },
            decision_quality={"metric_role": "diagnostic_only"},
            shadow_feedback={
                "completed_row_count": len(shadows),
                "cost_complete_direction_sample_count": len(shadow_samples),
                "excluded_reason_counts": shadow_excluded,
                "can_authorize_entry": False,
            },
            expert_memory={
                "role": "advisory_context_only",
                "can_authorize_entry": False,
            },
            manual_intervention={},
            trade_fact_quarantine={
                "checked_count": len(closed),
                "trusted_cost_complete_count": len(authoritative_samples),
                "reason_counts": quarantine_reasons,
            },
            reflection_feedback={"role": "diagnostic_only"},
            event_feedback={
                "linked_event_count": len(events),
                "regime_linked_position_count": len(regime_by_position),
            },
            authoritative_return_observation=observation,
            problems=problems,
            root_causes=[str(item["code"]) for item in problems],
            training_policy={
                "optimization_target": "maximize_authoritative_fee_after_return_rate",
                "cost_complete_samples_required": True,
                "walk_forward_required": True,
                "cost_complete_shadow_required": True,
                "win_rate_role": "diagnostic_only",
                "fixed_strategy_thresholds_allowed": False,
            },
            authoritative_return_samples=authoritative_samples,
            shadow_return_samples=shadow_samples,
        )

    @classmethod
    def _position_return_sample(
        cls,
        position: Position,
        *,
        market_regime: str,
    ) -> dict[str, Any] | None:
        net_pnl = cls._fee_after_pnl(position)
        quantity = abs(_safe_float(getattr(position, "quantity", None)))
        entry_price = abs(_safe_float(getattr(position, "entry_price", None)))
        leverage = abs(_safe_float(getattr(position, "leverage", None)))
        notional = quantity * entry_price
        margin_basis = notional / leverage if notional > 0 and leverage > 0 else 0.0
        if margin_basis <= 0:
            return None
        return {
            "source": "trusted_cost_complete_closed_position",
            "source_id": _safe_int(getattr(position, "id", None)),
            "symbol": str(getattr(position, "symbol", "") or "").upper(),
            "side": str(getattr(position, "side", "") or "").lower(),
            "market_regime": market_regime,
            "net_pnl_after_all_costs_usdt": round(net_pnl, 8),
            "margin_basis_usdt": round(margin_basis, 8),
            "net_return_after_cost_pct": round(net_pnl / margin_basis * 100.0, 8),
            "timestamp": _timestamp_text(
                getattr(position, "closed_at", None) or getattr(position, "created_at", None)
            ),
            "cost_policy_provenance": {
                "source": "position_realized_pnl_entry_fee_close_fee_funding_fee",
                "observation_window": "full_position_lifecycle",
                "sample_count": 1,
                "generated_at": datetime.now(UTC).isoformat(),
                "strategy_version": STRATEGY_SCHEDULER_VERSION,
                "fallback_reason": "",
            },
        }

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
            getattr(position, field_name, None) is not None
            for field_name in ("entry_fee", "close_fee", "funding_fee")
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
        active = _safe_dict(learning.get("active_profile"))
        runtime = _safe_dict(learning.get("runtime"))
        decision_action = action or str(
            getattr(getattr(decision, "action", None), "value", "")
        )
        side = (
            "long"
            if "long" in decision_action
            else "short"
            if "short" in decision_action
            else None
        )
        resolved_market_state = _safe_dict(
            market_state
            or context.get("market_regime")
            or _safe_dict(raw_response).get("market_regime")
        )
        scheduler_reason = context.get("scheduler_reason") or learning.get("scheduler_reason")
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
            profile_id=str(
                context.get("strategy_profile_id") or active.get("id") or ""
            )
            or None,
            profile_version=(
                _safe_int(context.get("strategy_profile_version") or active.get("version"))
                or None
            ),
            scheduler_reason=str(sanitize_runtime_text(scheduler_reason or "") or "")[:2000],
            strategy_snapshot=sanitize_runtime_text(
                _json_safe(
                    {
                        "scheduler_mode": learning.get("scheduler_mode"),
                        "active_profile": active,
                        "runtime": runtime,
                        "production_permission": False,
                    }
                )
            ),
            market_state=sanitize_runtime_text(_json_safe(resolved_market_state)),
            side_weights=None,
            expert_integrity=sanitize_runtime_text(
                _json_safe(
                    {
                        "advisory_only": True,
                        "can_authorize_entry": False,
                        "raw": raw_response or {},
                    }
                )
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
            logger.warning("failed to record strategy schedule event", error=safe_error_text(exc))
            return None
