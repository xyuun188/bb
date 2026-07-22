"""Dynamic strategy scheduling from authoritative fee-after return evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import ceil, isfinite, sqrt
from types import SimpleNamespace
from typing import Any

import structlog
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import load_only

from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text
from db.session import get_read_session_ctx, get_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest, StrategyLearningEvent
from models.trade import Position
from services.authoritative_trade_outcome import (
    AUTHORITATIVE_TRADE_OUTCOME_VERSION,
    load_authoritative_trade_outcomes,
)
from services.continuous_model_weight import market_regime_name
from services.continuous_strategy_routing import (
    ContinuousStrategyRoutingPolicy,
    ContinuousStrategyRoutingStore,
)
from services.model_strategy_blueprint import paper_strategy_replay_available
from services.paper_strategy_champion import PaperStrategyChampionService
from services.phase3_boundary import PHASE3_CLEAN_START_UTC
from services.shadow_backtest_service import shadow_fee_after_outcome
from services.shadow_training_quarantine import assess_shadow_row
from services.strategy_historical_replay import (
    ModelPredictor,
    build_strategy_historical_replay,
)
from services.text_integrity import sanitize_runtime_text

logger = structlog.get_logger(__name__)

DEFAULT_LOOKBACK_HOURS = 168
STRATEGY_SCHEDULER_VERSION = "2026-07-15.historical-return-prior-scheduler.v2"
PRODUCTION_STRATEGY_ID = "dynamic_fee_after_return_execution"
PRODUCTION_STRATEGY_VERSION = "2026-07-15.dynamic-profit-execution.v1"
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
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            return None
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                default=str,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError):
        return None


def _timestamp_text(value: Any) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value or "")


def _regime_label(value: Any) -> str:
    label = market_regime_name(value)
    return "" if label == "unknown" else label


def _shadow_cost_evidence(
    row: Any,
    *,
    snapshot_override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float | None, float | None, dict[str, Any], list[str]]:
    snapshot = (
        dict(snapshot_override)
        if isinstance(snapshot_override, dict)
        else _safe_dict(getattr(row, "feature_snapshot", None))
    )
    long_gross = _optional_float(getattr(row, "long_return_pct", None))
    short_gross = _optional_float(getattr(row, "short_return_pct", None))
    if long_gross is None or short_gross is None:
        return snapshot, long_gross, short_gross, {}, ["direction_return_missing"]
    evidence_row = (
        SimpleNamespace(
            feature_snapshot=snapshot,
            horizon_minutes=getattr(row, "horizon_minutes", 0),
        )
        if snapshot_override is not None
        else row
    )
    outcome = shadow_fee_after_outcome(
        evidence_row,
        long_return=long_gross / 100.0,
        short_return=short_gross / 100.0,
    )
    if outcome.get("cost_complete") is not True:
        reasons = [
            str(value)
            for value in _safe_list(outcome.get("incomplete_reasons"))
            if value
        ] or ["shadow_cost_contract_incomplete"]
        return snapshot, long_gross, short_gross, outcome, reasons
    return snapshot, long_gross, short_gross, outcome, []


def _historical_replay_observation(row: Any) -> tuple[dict[str, Any] | None, list[str]]:
    replay_snapshot = _safe_dict(
        getattr(row, "training_feature_snapshot", None)
    )
    snapshot, long_gross, short_gross, outcome, reasons = _shadow_cost_evidence(
        row,
        snapshot_override=replay_snapshot,
    )
    if reasons or long_gross is None or short_gross is None:
        return None, reasons
    created_at = getattr(row, "created_at", None)
    if isinstance(created_at, datetime) and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    quality = assess_shadow_row(row)
    training_eligible = bool(
        isinstance(created_at, datetime)
        and created_at >= PHASE3_CLEAN_START_UTC
        and not quality.exclude_from_training
    )
    return {
        "source_id": _safe_int(getattr(row, "id", None)),
        "decision_id": _safe_int(getattr(row, "decision_id", None)) or None,
        "symbol": str(getattr(row, "symbol", "") or "").upper(),
        "market_regime": _regime_label(snapshot),
        "horizon_minutes": _safe_int(getattr(row, "horizon_minutes", None), 10),
        "created_at": _timestamp_text(created_at),
        "completed_at": _timestamp_text(getattr(row, "updated_at", None)),
        "decision_timestamp": _timestamp_text(created_at),
        "label_timestamp": _timestamp_text(getattr(row, "due_at", None) or created_at),
        "feature_snapshot": snapshot,
        "execution_cost_pct": round(
            _safe_float(outcome.get("fee_return_pct"))
            + _safe_float(outcome.get("slippage_return_pct")),
            8,
        ),
        "long_gross_return_pct": round(long_gross, 8),
        "short_gross_return_pct": round(short_gross, 8),
        "long_net_return_after_cost_pct": outcome.get(
            "long_net_return_after_cost_pct"
        ),
        "short_net_return_after_cost_pct": outcome.get(
            "short_net_return_after_cost_pct"
        ),
        "long_funding_return_pct": outcome.get("funding_return_long_pct"),
        "short_funding_return_pct": outcome.get("funding_return_short_pct"),
        "training_eligible": training_eligible,
        "training_quality_status": quality.status,
        "training_quality_reasons": list(quality.reasons),
    }, []


def _runtime_prior_usage(decisions: list[Any]) -> dict[str, Any]:
    """Summarize which governed historical priors recent decisions actually matched."""

    latest_by_symbol_side: dict[tuple[str, str], dict[str, Any]] = {}
    matched_decision_ids: set[int] = set()
    matched_profile_ids: set[str] = set()
    evaluated_side_count = 0
    matched_evaluation_count = 0
    decision_records: list[dict[str, Any]] = []

    for decision in decisions:
        evidence = _safe_dict(getattr(decision, "entry_candidate_evidence", None))
        if not evidence:
            raw = _safe_dict(getattr(decision, "raw_llm_response", None))
            evidence = _safe_dict(raw.get("entry_candidate_evidence"))
        decision_id = _safe_int(getattr(decision, "id", None))
        symbol = str(getattr(decision, "symbol", "") or "").upper()
        side_records: list[dict[str, Any]] = []
        for side in ("long", "short"):
            side_evidence = _safe_dict(evidence.get(side))
            if side_evidence:
                evaluated_side_count += 1
            prior = _safe_dict(side_evidence.get("scheduled_return_prior"))
            matched = prior.get("available") is True
            side_records.append(
                {
                    "side": side,
                    "evaluation_status": (
                        "matched_historical_prior"
                        if matched
                        else "not_matched"
                        if side_evidence
                        else "not_evaluated"
                    ),
                    "profile_id": prior.get("profile_id") if matched else None,
                    "profile_version": prior.get("profile_version") if matched else None,
                    "selector": _safe_dict(prior.get("selector")) if matched else None,
                    "context_fields_influenced": ["scheduled_return_prior"] if matched else [],
                    "can_authorize_entry": False,
                    "can_change_size_or_leverage": False,
                    "missing_reason": (
                        None
                        if matched
                        else str(
                            prior.get("reason")
                            or (
                                "entry_side_evidence_missing"
                                if not side_evidence
                                else "no_governed_historical_prior_matches_context"
                            )
                        )
                    ),
                }
            )
            if prior.get("available") is not True:
                continue

            matched_evaluation_count += 1
            if decision_id:
                matched_decision_ids.add(decision_id)
            profile_id = str(prior.get("profile_id") or "").strip()
            if profile_id:
                matched_profile_ids.add(profile_id)
            route_key = (symbol, side)
            if route_key in latest_by_symbol_side:
                continue
            latest_by_symbol_side[route_key] = {
                "decision_id": decision_id or None,
                "matched_at": _timestamp_text(getattr(decision, "created_at", None)),
                "symbol": symbol,
                "decision_action": str(getattr(decision, "action", "") or ""),
                "evaluated_side": side,
                "profile_id": profile_id or None,
                "profile_version": prior.get("profile_version"),
                "rank": prior.get("rank"),
                "selector": _safe_dict(prior.get("selector")),
                "role": "historical_prior_only",
                "can_authorize_entry": False,
            }

        decision_records.append(
            {
                "decision_id": decision_id or None,
                "created_at": _timestamp_text(getattr(decision, "created_at", None)),
                "symbol": symbol,
                "final_action": str(getattr(decision, "action", "") or ""),
                "was_executed": bool(getattr(decision, "was_executed", False)),
                "final_reason": str(
                    getattr(decision, "execution_reason", "")
                    or (
                        "executed_after_all_production_contracts_passed"
                        if bool(getattr(decision, "was_executed", False))
                        else "not_executed_by_current_production_contracts"
                    )
                ),
                "side_evaluations": side_records,
            }
        )

    latest_matches = list(latest_by_symbol_side.values())
    return {
        "role": "historical_prior_only",
        "inspected_decision_count": len(decisions),
        "evaluated_side_count": evaluated_side_count,
        "matched_decision_count": len(matched_decision_ids),
        "matched_evaluation_count": matched_evaluation_count,
        "matched_profile_count": len(matched_profile_ids),
        "latest_match_at": latest_matches[0]["matched_at"] if latest_matches else None,
        "latest_matches": latest_matches,
        "decision_records": decision_records,
        "can_authorize_entry": False,
    }


def _current_production_strategy(
    feedback: StrategyFeedback,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    authoritative_count = len(feedback.authoritative_return_samples)
    return {
        "id": PRODUCTION_STRATEGY_ID,
        "version": PRODUCTION_STRATEGY_VERSION,
        "name": "Dynamic fee-after return execution",
        "objective": "maximize_authoritative_fee_after_return_rate",
        "owner": "trading_service_production_entry_pipeline",
        "enabled": True,
        "status": "running",
        "scope": "all_symbols_evaluated_independently_per_side_and_decision",
        "entry_permission_owner": "current_return_distribution_plus_dynamic_risk_contracts",
        "historical_prior_role": "context_only",
        "historical_prior_can_authorize_entry": False,
        "historical_prior_can_change_size_or_leverage": False,
        "execution_owners": list(EXECUTION_OWNERS),
        "data_sources": {
            "authoritative_trade_outcome": {
                "status": "available" if authoritative_count else "missing",
                "count": authoritative_count,
                "version": AUTHORITATIVE_TRADE_OUTCOME_VERSION,
            },
            "current_return_distribution": {
                "status": "evaluated_per_decision",
                "owner": "return_execution_policy",
            },
            "live_execution_cost": {
                "status": "evaluated_per_decision",
                "owner": "return_execution_policy",
            },
            "dynamic_risk_budget": {
                "status": "evaluated_per_decision",
                "owner": "dynamic_entry_risk_budget",
            },
            "position_capacity": {
                "status": "evaluated_per_decision",
                "owner": "dynamic_position_capacity",
            },
        },
        "historical_prior_matching_enabled": bool(
            runtime.get("production_influence_enabled")
        ),
    }


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
            gross_loss > 0 and gross_profit > gross_loss
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
    runtime_prior_usage: dict[str, Any] = field(default_factory=dict)
    authoritative_return_samples: list[dict[str, Any]] = field(default_factory=list)
    shadow_return_samples: list[dict[str, Any]] = field(default_factory=list)
    shadow_replay_observations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        payload = {
            name: _json_safe(getattr(self, name))
            for name in self.__dataclass_fields__
            if name != "shadow_replay_observations"
            and (include_samples or not name.endswith("_samples"))
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
        return self.generate_from_samples(
            feedback.authoritative_return_samples,
            window_hours=feedback.window_hours,
            generated_at=feedback.generated_at,
            evidence_source="trusted_cost_complete_closed_positions",
            evidence_mode="authoritative_trade_outcomes",
        )

    def generate_from_samples(
        self,
        samples: list[dict[str, Any]],
        *,
        window_hours: int,
        generated_at: str,
        evidence_source: str,
        evidence_mode: str,
        strategy_id: str | None = None,
    ) -> list[StrategyProfile]:
        partitions: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        for sample in samples:
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
            version = max(_safe_int(sample.get("source_row_id")) for sample in samples)
            horizons = sorted(
                {
                    horizon
                    for sample in samples
                    if (horizon := _safe_int(sample.get("horizon_minutes"))) > 0
                }
            )
            provenance = {
                "source": evidence_source,
                "evidence_mode": evidence_mode,
                "strategy_id": strategy_id,
                "observation_window": f"trailing_{window_hours}_hours",
                "sample_count": len(samples),
                "generated_at": generated_at,
                "strategy_version": STRATEGY_SCHEDULER_VERSION,
                "fallback_reason": "",
                "position_ids": sorted(
                    _safe_int(sample.get("position_id")) for sample in samples
                ),
            }
            profiles.append(
                StrategyProfile(
                    profile_id=profile_id,
                    version=version,
                    label=_profile_label(selector),
                    status="candidate",
                    source=(
                        "trained_model_historical_replay_partition"
                        if evidence_mode == "exact_trained_model_historical_replay"
                        else "authoritative_fee_after_return_partition"
                    ),
                    description=(
                        "Historical return prior; current live return, execution cost, account "
                        "risk, and position contracts remain mandatory."
                    ),
                    params={
                        "selector": selector,
                        "prediction_horizon_minutes": (
                            horizons[0] if len(horizons) == 1 else None
                        ),
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
    if backtest_metrics.get("profit_factor") is None:
        reasons.append("walk_forward_profit_factor_undefined")
    elif backtest_metrics.get("profit_factor_above_break_even") is not True:
        reasons.append("walk_forward_profit_factor_not_above_break_even")
    if shadow.get("status") != "complete":
        reasons.append(str(shadow.get("status") or "shadow_validation_incomplete"))
    if (_optional_float(shadow_metrics.get("return_lcb_pct")) or 0.0) <= 0:
        reasons.append("shadow_fee_after_return_lcb_not_positive")
    if shadow_metrics.get("profit_factor") is None:
        reasons.append("shadow_profit_factor_undefined")
    elif shadow_metrics.get("profit_factor_above_break_even") is not True:
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
    def __init__(
        self,
        generator: StrategyCandidateGenerator | None = None,
        routing_policy: ContinuousStrategyRoutingPolicy | None = None,
        **_: Any,
    ) -> None:
        self.generator = generator or StrategyCandidateGenerator()
        self.routing_policy = routing_policy or ContinuousStrategyRoutingPolicy()

    def build_from_feedback(
        self,
        feedback: StrategyFeedback,
        *,
        current_context: dict[str, Any] | None = None,
        detail: str = "summary",
        model_strategy_blueprint: dict[str, Any] | None = None,
        model_predictor: ModelPredictor | None = None,
        update_strategy_state: bool = True,
    ) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        backtest_rows: list[dict[str, Any]] = []
        shadow_rows: list[dict[str, Any]] = []
        include_evidence_rows = detail == "full"
        blueprint = _safe_dict(model_strategy_blueprint)
        model_replay_required = paper_strategy_replay_available(blueprint)
        replay = build_strategy_historical_replay(
            blueprint=blueprint,
            observations=feedback.shadow_replay_observations,
            predictor=model_predictor,
        )
        exact_replay = bool(model_replay_required and replay.get("status") == "complete")
        replay_development = list(replay.get("development_samples") or [])
        replay_exam = list(replay.get("exam_samples") or [])
        if model_replay_required:
            profiles = (
                self.generator.generate_from_samples(
                    replay_development,
                    window_hours=feedback.window_hours,
                    generated_at=feedback.generated_at,
                    evidence_source="exact_current_model_on_immutable_shadow_snapshot",
                    evidence_mode="exact_trained_model_historical_replay",
                    strategy_id=str(blueprint.get("strategy_id") or "") or None,
                )
                if exact_replay and replay_development
                else []
            )
        else:
            profiles = self.generator.generate(feedback)
        for profile in profiles:
            selector = _safe_dict(profile.params.get("selector"))
            authoritative = [
                sample
                for sample in (
                    replay_development
                    if exact_replay and replay_development
                    else feedback.authoritative_return_samples
                )
                if _selector_matches(selector, sample)
            ]
            shadows = [
                sample
                for sample in (
                    replay_exam
                    if exact_replay
                    else []
                    if model_replay_required
                    else feedback.shadow_return_samples
                )
                if _selector_matches(selector, sample)
            ]
            backtest = _walk_forward_report(authoritative)
            backtest["evidence_mode"] = (
                "exact_trained_model_historical_replay"
                if exact_replay
                else "authoritative_trade_outcomes"
            )
            backtest["evidence_partition"] = (
                "strategy_development"
                if exact_replay
                else "authoritative_closed_positions"
            )
            shadow = _shadow_report(shadows, include_rows=include_evidence_rows)
            shadow["validation_method"] = (
                replay.get("validation_method")
                if exact_replay
                else "model_historical_replay_required"
                if model_replay_required
                else "legacy_selector_matched_shadow"
            )
            shadow["evidence_partition"] = (
                "strategy_exam" if exact_replay else "legacy_shadow"
            )
            rejection_reasons = _candidate_rejections(backtest, shadow)
            if model_replay_required and not exact_replay:
                rejection_reasons.append(
                    f"model_historical_replay_{replay.get('status') or 'incomplete'}"
                )
            provenance = _safe_dict(profile.params.get("policy_provenance"))
            if model_replay_required and provenance.get("evidence_mode") != (
                "exact_trained_model_historical_replay"
            ):
                rejection_reasons.append("exact_model_replay_evidence_missing")
            rejection_reasons = list(dict.fromkeys(rejection_reasons))
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
        continuous_routing = self.routing_policy.build(
            execution_mode=feedback.mode,
            market_regime=_safe_dict(context.get("market_regime")),
            candidates=candidates,
            update_state=update_strategy_state,
        )
        routed_side = str(
            _safe_dict(continuous_routing.get("current_route")).get(
                "recommended_side"
            )
            or ""
        ).lower()
        leading = (candidates or [None])[0]
        influence_enabled = bool(governed)
        scheduler_mode = (
            "governed_dynamic_return"
            if influence_enabled
            else "continuous_paper_strategy_routing"
            if routed_side in {"long", "short"}
            else "model_replay_no_fee_after_entries"
            if model_replay_required
            and replay.get("status") == "complete"
            and not candidates
            else "model_replay_incomplete"
            if model_replay_required and not candidates
            else "shadow_validation"
            if candidates
            else "insufficient_authoritative_evidence"
        )
        reason = (
            "Governed fee-after return candidates are available from exact model replay; "
            "the current live return and dynamic risk contracts still own execution."
            if influence_enabled and exact_replay
            else "Governed fee-after return candidates are available as matching historical priors; "
            "the current live return and dynamic risk contracts still own execution."
            if influence_enabled
            else "Validated strategies remain continuously weighted for paper training; current return and risk contracts still own normal entries."
            if routed_side in {"long", "short"}
            else "Exact trained-model historical replay found no fee-after-positive entries; no model strategy candidate was created."
            if model_replay_required
            and replay.get("status") == "complete"
            and not candidates
            else "The trained-model historical replay is incomplete, so no model strategy candidate can be created."
            if model_replay_required and not candidates
            else "Candidates remain in shadow because exact model replay, walk-forward, or cost-complete exam evidence is incomplete."
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
            "continuous_strategy_routing": continuous_routing,
            "policy_provenance": {
                "source": (
                    "exact_model_historical_replay_scheduler"
                    if exact_replay
                    else "authoritative_walk_forward_and_cost_complete_shadow_scheduler"
                ),
                "observation_window": f"trailing_{feedback.window_hours}_hours",
                "sample_count": (
                    len(replay_development)
                    if exact_replay
                    else len(feedback.authoritative_return_samples)
                ),
                "generated_at": feedback.generated_at,
                "strategy_version": STRATEGY_SCHEDULER_VERSION,
                "fallback_reason": "" if influence_enabled else scheduler_mode,
            },
        }
        production_strategy = _current_production_strategy(feedback, runtime)
        schedule = {
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
                "exact_model_replay_required": model_replay_required,
                "can_authorize_entry": False,
                "rows": shadow_rows,
            },
            "historical_model_replay": {
                key: _json_safe(value)
                for key, value in replay.items()
                if key not in {"development_samples", "exam_samples"}
            },
            "continuous_strategy_routing": continuous_routing,
            "scheduler_mode": scheduler_mode,
            "current_production_strategy": production_strategy,
        }
        return {
            "feedback": feedback.to_dict(include_samples=detail == "full"),
            "schedule": schedule,
            "current_production_strategy": production_strategy,
        }

    def apply_to_context(
        self,
        strategy_context: dict[str, Any],
        payload: dict[str, Any],
        *,
        paper_strategy_champion: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = dict(strategy_context or {})
        schedule = _safe_dict(payload.get("schedule"))
        runtime = _safe_dict(schedule.get("runtime"))
        leading = _safe_dict(schedule.get("leading_candidate"))
        result["scheduler_reason"] = schedule.get("reason")
        result["current_production_strategy"] = _safe_dict(
            schedule.get("current_production_strategy")
        )
        champion = _safe_dict(paper_strategy_champion)
        result["strategy_learning"] = {
            "scheduler_mode": schedule.get("scheduler_mode"),
            "candidate_count": schedule.get("candidate_count"),
            "governed_candidate_count": schedule.get("governed_candidate_count"),
            "rejected_candidate_count": schedule.get("rejected_candidate_count"),
            "leading_candidate": leading,
            "runtime": runtime,
            "advisory_prior_only": True,
            "production_permission": False,
            "policy_provenance": runtime.get("policy_provenance"),
            "current_production_strategy": _safe_dict(
                schedule.get("current_production_strategy")
            ),
            "paper_strategy_champion": champion,
            "continuous_strategy_routing": _safe_dict(
                schedule.get("continuous_strategy_routing")
            ),
        }
        result["paper_strategy_champion"] = champion
        result["continuous_strategy_routing"] = _safe_dict(
            schedule.get("continuous_strategy_routing")
        )
        return result


class StrategyLearningService:
    def __init__(
        self,
        *,
        engine: StrategyLearningEngine | None = None,
        champion_service: PaperStrategyChampionService | None = None,
        replay_model_service: Any | None = None,
        routing_store: ContinuousStrategyRoutingStore | None = None,
        **_: Any,
    ) -> None:
        self.engine = engine or StrategyLearningEngine()
        self.champion_service = champion_service or PaperStrategyChampionService()
        self._replay_model_service = replay_model_service
        self.routing_store = routing_store or ContinuousStrategyRoutingStore()

    def _default_model_replay_context(
        self,
        mode: str,
    ) -> tuple[dict[str, Any], ModelPredictor | None]:
        if str(mode).lower() == "live":
            return {}, None
        try:
            if self._replay_model_service is None:
                from services.ml_signal_service import MLSignalService

                self._replay_model_service = MLSignalService()
            blueprint = self._replay_model_service.strategy_blueprint()
            predictor = getattr(self._replay_model_service, "predict", None)
            return _safe_dict(blueprint), predictor if callable(predictor) else None
        except Exception as exc:
            logger.warning(
                "strategy_historical_replay_model_unavailable",
                error=safe_error_text(exc, limit=160),
            )
            return {}, None

    async def dashboard_payload(
        self,
        *,
        mode: str,
        hours: int = DEFAULT_LOOKBACK_HOURS,
        limit: int = 500,
        detail: str = "summary",
    ) -> dict[str, Any]:
        blueprint, predictor = self._default_model_replay_context(mode)
        feedback = await self._feedback(
            mode=mode,
            hours=hours,
            limit=limit,
            include_historical_replay=bool(
                paper_strategy_replay_available(blueprint)
                and predictor is not None
            ),
        )
        payload = await asyncio.to_thread(
            self.engine.build_from_feedback,
            feedback,
            detail=detail,
            model_strategy_blueprint=blueprint,
            model_predictor=predictor,
            update_strategy_state=False,
        )
        champion = await self.champion_service.current(mode)
        payload.update(
            {
                "mode": mode,
                "window_hours": hours,
                "sample_limit": limit,
                "optimization_target": "maximize_authoritative_fee_after_return_rate",
                "production_permission": False,
                "paper_strategy_champion": champion,
            }
        )
        return payload

    async def apply_to_strategy_context(
        self,
        *,
        mode: str,
        strategy_context: dict[str, Any],
        open_positions: list[dict[str, Any]] | None,
        model_strategy_blueprint: dict[str, Any] | None = None,
        model_predictor: ModelPredictor | None = None,
        hours: int = DEFAULT_LOOKBACK_HOURS,
        limit: int = 500,
    ) -> dict[str, Any]:
        feedback = await self._feedback(
            mode=mode,
            hours=hours,
            limit=limit,
            include_historical_replay=bool(
                paper_strategy_replay_available(
                    _safe_dict(model_strategy_blueprint)
                )
                and model_predictor is not None
            ),
        )
        if open_positions is not None:
            feedback.open_position_pressure["runtime_open_position_count"] = len(open_positions)
        payload = await asyncio.to_thread(
            self.engine.build_from_feedback,
            feedback,
            current_context=strategy_context,
            detail="summary",
            model_strategy_blueprint=model_strategy_blueprint,
            model_predictor=model_predictor,
        )
        schedule = _safe_dict(payload.get("schedule"))
        routing = _safe_dict(schedule.get("continuous_strategy_routing"))
        if routing.get("applied") is True:
            async with get_session_ctx() as session:
                routing["persistence"] = await self.routing_store.persist(
                    mode=mode,
                    candidates=list(schedule.get("candidates") or []),
                    routing=routing,
                    session=session,
                )
            schedule["continuous_strategy_routing"] = routing
            runtime = _safe_dict(schedule.get("runtime"))
            runtime["continuous_strategy_routing"] = routing
            schedule["runtime"] = runtime
            payload["schedule"] = schedule
        champion = await self.champion_service.reconcile(
            mode=mode,
            blueprint=model_strategy_blueprint,
            candidates=list(_safe_dict(payload.get("schedule")).get("candidates") or []),
        )
        result = self.engine.apply_to_context(
            strategy_context,
            payload,
            paper_strategy_champion=champion,
        )
        result["execution_mode"] = "live" if str(mode).lower() == "live" else "paper"
        result["paper_training_mode"] = (
            "bootstrap"
            if str(mode).lower() != "live" and champion.get("active") is not True
            else "normal"
            if str(mode).lower() != "live"
            else "disabled"
        )
        result.setdefault("strategy_learning", {})["paper_training_mode"] = result[
            "paper_training_mode"
        ]
        return result

    async def _feedback(
        self,
        *,
        mode: str,
        hours: int,
        limit: int,
        include_historical_replay: bool = False,
    ) -> StrategyFeedback:
        selected_mode = "live" if str(mode).lower() == "live" else "paper"
        effective_hours = max(int(hours or 1), 1)
        effective_limit = max(int(limit or 1), 1)
        since = datetime.now(UTC) - timedelta(hours=effective_hours)
        since_naive = since.replace(tzinfo=None)
        outcomes = await load_authoritative_trade_outcomes(
            mode=selected_mode,
            since=since,
            limit=effective_limit,
        )
        async with get_read_session_ctx() as session:
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
            position_ids = sorted(
                {
                    int(value)
                    for outcome in outcomes
                    for value in (outcome.get("position_ids") or [outcome.get("position_id")])
                    if str(value or "").isdigit() and int(value) > 0
                }
            )
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
            replay_shadows = (
                list(
                    (
                        await session.execute(
                            select(ShadowBacktest)
                            .where(
                                ShadowBacktest.execution_mode == "paper",
                                ShadowBacktest.status == "completed",
                                ShadowBacktest.created_at
                                >= PHASE3_CLEAN_START_UTC.replace(tzinfo=None),
                                ShadowBacktest.long_return_pct.is_not(None),
                                ShadowBacktest.short_return_pct.is_not(None),
                                or_(
                                    ShadowBacktest.decision_action.in_(["long", "short"]),
                                    and_(
                                        ShadowBacktest.missed_opportunity.is_(True),
                                        ShadowBacktest.best_action.in_(["long", "short"]),
                                    ),
                                ),
                            )
                            .options(
                                load_only(
                                    ShadowBacktest.id,
                                    ShadowBacktest.decision_id,
                                    ShadowBacktest.created_at,
                                    ShadowBacktest.updated_at,
                                    ShadowBacktest.symbol,
                                    ShadowBacktest.analysis_type,
                                    ShadowBacktest.decision_action,
                                    ShadowBacktest.decision_confidence,
                                    ShadowBacktest.training_feature_snapshot,
                                    ShadowBacktest.due_at,
                                    ShadowBacktest.horizon_minutes,
                                    ShadowBacktest.label_version,
                                    ShadowBacktest.long_return_pct,
                                    ShadowBacktest.short_return_pct,
                                    ShadowBacktest.best_action,
                                    ShadowBacktest.missed_opportunity,
                                )
                            )
                            .order_by(
                                ShadowBacktest.created_at.asc(),
                                ShadowBacktest.id.asc(),
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                if selected_mode == "paper" and include_historical_replay
                else []
            )
            decisions = list(
                (
                    await session.execute(
                        select(
                            AIDecision.id,
                            AIDecision.symbol,
                            AIDecision.action,
                            AIDecision.created_at,
                            AIDecision.was_executed,
                            AIDecision.execution_reason,
                            AIDecision.raw_llm_response[
                                "entry_candidate_evidence"
                            ].label("entry_candidate_evidence"),
                        )
                        .where(
                            AIDecision.model_name == ENSEMBLE_TRADER_NAME,
                            AIDecision.is_paper.is_(selected_mode == "paper"),
                            AIDecision.analysis_type == "market",
                            AIDecision.created_at >= since_naive,
                        )
                        .order_by(AIDecision.created_at.desc(), AIDecision.id.desc())
                        .limit(effective_limit)
                    )
                )
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
        for outcome in outcomes:
            if (
                outcome.get("event_type") != "AuthoritativeTradeOutcome"
                or outcome.get("outcome_version") != AUTHORITATIVE_TRADE_OUTCOME_VERSION
                or outcome.get("outcome_complete") is not True
                or outcome.get("trade_fact_trusted") is not True
            ):
                reasons = _safe_list(outcome.get("outcome_evidence_gaps")) or [
                    "authoritative_outcome_incomplete"
                ]
                for reason in reasons:
                    reason_text = str(reason)
                    quarantine_reasons[reason_text] = quarantine_reasons.get(reason_text, 0) + 1
                continue
            position_id = _safe_int(outcome.get("position_id"))
            net_return = _optional_float(outcome.get("authoritative_pnl_ratio_pct"))
            net_pnl = _optional_float(outcome.get("realized_pnl"))
            if net_return is None or net_pnl is None:
                quarantine_reasons["authoritative_return_missing"] = (
                    quarantine_reasons.get("authoritative_return_missing", 0) + 1
                )
                continue
            authoritative_samples.append(
                {
                    "source": "authoritative_trade_outcome",
                    "source_id": outcome.get("outcome_id"),
                    "source_row_id": outcome.get("id"),
                    "outcome_id": outcome.get("outcome_id"),
                    "outcome_version": outcome.get("outcome_version"),
                    "outcome_fingerprint": outcome.get("outcome_fingerprint"),
                    "position_id": position_id,
                    "symbol": str(outcome.get("symbol") or "").upper(),
                    "side": str(outcome.get("side") or "").lower(),
                    "market_regime": regime_by_position.get(position_id, ""),
                    "net_pnl_after_all_costs_usdt": round(net_pnl, 8),
                    "net_return_after_cost_pct": round(net_return, 8),
                    "return_basis_source": "okx_positions_history.pnlRatio",
                    "timestamp": _timestamp_text(outcome.get("label_timestamp")),
                    "cost_policy_provenance": _safe_dict(
                        outcome.get("consumer_provenance")
                    ),
                    "attribution": _safe_dict(outcome.get("attribution")),
                }
            )

        shadow_samples: list[dict[str, Any]] = []
        shadow_replay_observations: list[dict[str, Any]] = []
        shadow_excluded: dict[str, int] = {}
        for row in shadows:
            snapshot, long_gross, short_gross, outcome, reasons = (
                _shadow_cost_evidence(row)
            )
            if reasons or long_gross is None or short_gross is None:
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
                        "market_regime": _regime_label(snapshot),
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

        replay_excluded: dict[str, int] = {}
        for row in replay_shadows:
            replay_observation, reasons = _historical_replay_observation(row)
            if replay_observation is not None:
                shadow_replay_observations.append(replay_observation)
                continue
            for reason in reasons or ["historical_replay_observation_incomplete"]:
                reason_text = str(reason)
                replay_excluded[reason_text] = replay_excluded.get(reason_text, 0) + 1

        side_performance = {
            side: _legacy_observation_summary(
                [sample for sample in authoritative_samples if sample.get("side") == side]
            )
            for side in ("long", "short")
        }
        observation = {
            **_legacy_observation_summary(authoritative_samples),
            "cost_complete_sample_count": len(authoritative_samples),
            "excluded_incomplete_or_untrusted_count": len(outcomes)
            - len(authoritative_samples),
        }
        generated_at = datetime.now(UTC).isoformat()
        problems = [
            {"code": code, "count": count, "kind": "authoritative_sample_excluded"}
            for code, count in sorted(quarantine_reasons.items())
        ] + [
            {"code": code, "count": count, "kind": "shadow_sample_excluded"}
            for code, count in sorted(shadow_excluded.items())
        ] + [
            {"code": code, "count": count, "kind": "historical_replay_excluded"}
            for code, count in sorted(replay_excluded.items())
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
                "historical_replay_observation_count": len(
                    shadow_replay_observations
                ),
                "historical_replay_training_eligible_count": sum(
                    row.get("training_eligible") is True
                    for row in shadow_replay_observations
                ),
                "historical_replay_excluded_reason_counts": replay_excluded,
                "excluded_reason_counts": shadow_excluded,
                "can_authorize_entry": False,
            },
            expert_memory={
                "role": "advisory_context_only",
                "can_authorize_entry": False,
            },
            manual_intervention={},
            trade_fact_quarantine={
                "checked_count": len(outcomes),
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
            runtime_prior_usage=_runtime_prior_usage(decisions),
            authoritative_return_samples=authoritative_samples,
            shadow_return_samples=shadow_samples,
            shadow_replay_observations=shadow_replay_observations,
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
        runtime = _safe_dict(learning.get("runtime"))
        response = _safe_dict(raw_response or getattr(decision, "raw_response", None))
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
        side_evidence = _safe_dict(
            _safe_dict(response.get("entry_candidate_evidence")).get(side)
        )
        matched_prior = _safe_dict(side_evidence.get("scheduled_return_prior"))
        if matched_prior.get("available") is not True:
            matched_prior = {}
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
            profile_id=str(matched_prior.get("profile_id") or "") or None,
            profile_version=_safe_int(matched_prior.get("profile_version")) or None,
            scheduler_reason=str(sanitize_runtime_text(scheduler_reason or "") or "")[:2000],
            strategy_snapshot=sanitize_runtime_text(
                _json_safe(
                    {
                        "scheduler_mode": learning.get("scheduler_mode"),
                        "current_production_strategy": _safe_dict(
                            context.get("current_production_strategy")
                            or learning.get("current_production_strategy")
                        ),
                        "matched_historical_prior": matched_prior,
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
