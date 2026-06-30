"""Profit-First v3 model/strategy/brain training summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from services.profit_first_trade_plan import (
    normalize_no_entry_reason,
    normalize_losing_exit_attribution,
    summarize_model_strategy_realized_pnl,
)


@dataclass(frozen=True, slots=True)
class ProfitFirstBrainTrainingService:
    """Build read-only brain-training signals from Profit-First facts."""

    min_canary_samples: int = 20
    min_live_samples: int = 50
    min_profit_factor: float = 1.12

    def build_dataset(
        self,
        *,
        decisions: list[Any],
        closed_positions: list[Any],
    ) -> dict[str, Any]:
        entries = [_entry_row(row) for row in decisions if _is_entry(row)]
        no_entries = [_no_entry_row(row) for row in decisions if not _was_executed(row)]
        losing_exits = [
            _losing_exit_row(position)
            for position in closed_positions
            if _safe_float(_row_get(position, "realized_pnl"), 0.0) < 0
        ]
        contribution_rows = _model_contribution_rows(closed_positions)
        leaderboard = summarize_model_strategy_realized_pnl(closed_positions)
        recommendations = self._recommendations(
            leaderboard.get("rows", []),
            contribution_rows,
            no_entries=no_entries,
            losing_exits=losing_exits,
        )
        return {
            "audit_only": True,
            "live_mutation": False,
            "policy": "profit_first_brain_shadow_to_canary_to_live",
            "dataset": {
                "entry_plan_count": len(entries),
                "no_entry_count": len(no_entries),
                "losing_exit_count": len(losing_exits),
                "model_contribution_count": len(contribution_rows),
                "entries": entries[:50],
                "no_entries": no_entries[:50],
                "losing_exits": losing_exits[:50],
                "model_contributions": contribution_rows[:100],
            },
            "leaderboard": leaderboard,
            "recommendations": recommendations,
            "training_inputs": [
                "ProfitFirstTradePlan fields",
                "realized_net_pnl",
                "decision_lane",
                "no_entry_reason",
                "losing_exit_attribution",
                "model_contributions",
                "exit_plan_reference",
            ],
            "training_outputs": [
                "source_weights",
                "strategy_weights",
                "lane_threshold_recommendations",
                "size_promotion_demotion",
                "no_entry_threshold_recommendations",
                "exit_policy_adjustments",
                "shadow_canary_live_decisions",
            ],
        }

    def _recommendations(
        self,
        leaderboard_rows: list[dict[str, Any]],
        contribution_rows: list[dict[str, Any]],
        *,
        no_entries: list[dict[str, Any]],
        losing_exits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        strategy_actions: list[dict[str, Any]] = []
        for row in leaderboard_rows:
            count = int(row.get("count") or 0)
            profit_factor = _safe_float(row.get("profit_factor"), 0.0)
            pnl = _safe_float(row.get("realized_net_pnl"), 0.0)
            if count < self.min_canary_samples:
                stage = "shadow"
                reason = "sample_floor_not_met"
            elif pnl > 0 and profit_factor >= self.min_profit_factor:
                stage = "canary" if count < self.min_live_samples else "live_candidate"
                reason = "positive_realized_net_pnl"
            else:
                stage = "shadow"
                reason = "realized_net_pnl_or_profit_factor_weak"
            strategy_actions.append(
                {
                    "model_name": row.get("model_name"),
                    "strategy_profile_id": row.get("strategy_profile_id"),
                    "symbol": row.get("symbol"),
                    "side": row.get("side"),
                    "decision_lane": row.get("decision_lane"),
                    "recommended_stage": stage,
                    "reason": reason,
                    "count": count,
                    "realized_net_pnl": row.get("realized_net_pnl"),
                    "profit_factor": row.get("profit_factor"),
                }
            )

        source_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"source": "", "count": 0, "realized_net_pnl": 0.0}
        )
        for row in contribution_rows:
            source = str(row.get("source") or "unknown")
            bucket = source_stats[source]
            bucket["source"] = source
            bucket["count"] += 1
            bucket["realized_net_pnl"] += _safe_float(row.get("realized_net_pnl"), 0.0)
        source_weights = []
        for bucket in source_stats.values():
            count = int(bucket["count"])
            pnl = float(bucket["realized_net_pnl"])
            state = "promote" if count >= 5 and pnl > 0 else "demote" if count >= 5 and pnl < 0 else "shadow"
            source_weights.append(
                {
                    "source": bucket["source"],
                    "recommended_state": state,
                    "count": count,
                    "realized_net_pnl": round(pnl, 6),
                    "weight_multiplier": 1.12 if state == "promote" else 0.82 if state == "demote" else 1.0,
                }
            )
        source_weights.sort(key=lambda item: item["realized_net_pnl"], reverse=True)
        no_entry_governance = _no_entry_governance(no_entries)
        losing_exit_governance = _losing_exit_governance(losing_exits)
        lane_thresholds = _lane_threshold_recommendations(
            leaderboard_rows,
            no_entry_governance,
            losing_exit_governance,
        )
        size_actions = _size_promotion_demotion(
            strategy_actions,
            losing_exit_governance,
        )
        return {
            "strategy_actions": strategy_actions[:50],
            "source_weights": source_weights[:50],
            "lane_threshold_recommendations": lane_thresholds[:50],
            "size_promotion_demotion": size_actions[:50],
            "no_entry_governance": no_entry_governance,
            "no_entry_threshold_recommendations": _safe_list(
                no_entry_governance.get("recommendations")
            )[:50],
            "losing_exit_governance": losing_exit_governance,
            "exit_policy_adjustments": _safe_list(
                losing_exit_governance.get("exit_policy_adjustments")
            )[:50],
            "brain_output_coverage": {
                "source_weights": True,
                "strategy_weights": True,
                "lane_threshold_recommendations": True,
                "size_promotion_demotion": True,
                "no_entry_threshold_recommendations": True,
                "exit_policy_adjustments": True,
                "shadow_canary_live_decisions": True,
            },
            "live_mutation": False,
            "promotion_flow": "shadow_to_canary_to_live",
            "requires_operator_resume_gate": True,
        }


def _entry_row(row: Any) -> dict[str, Any]:
    raw = _raw(row)
    plan = _safe_dict(raw.get("profit_first_trade_plan"))
    return {
        "decision_id": _row_get(row, "id"),
        "symbol": _row_get(row, "symbol") or plan.get("symbol"),
        "action": _row_get(row, "action") or plan.get("action"),
        "decision_lane": plan.get("decision_lane"),
        "profit_first_score": plan.get("profit_first_score"),
        "expected_net_return_pct": plan.get("expected_net_return_pct"),
        "loss_probability": plan.get("loss_probability"),
        "tail_loss_probability": plan.get("tail_loss_probability"),
        "position_size_pct": plan.get("position_size_pct"),
        "exit_plan_id": plan.get("exit_plan_id"),
        "model_sources": plan.get("model_sources") or [],
    }


def _no_entry_row(row: Any) -> dict[str, Any]:
    raw = _raw(row)
    plan = _safe_dict(raw.get("profit_first_trade_plan"))
    missing_fields = plan.get("missing_required_fields") or []
    reason = normalize_no_entry_reason(
        raw,
        execution_reason=str(
            _row_get(row, "execution_reason")
            or raw.get("reason")
            or raw.get("skip_reason")
            or raw.get("skip_kind")
            or ""
        ),
        plan_missing_fields=missing_fields,
    )
    raw_reason = plan.get("no_entry_reason") or raw.get("no_entry_reason") or raw.get("skip_kind") or ""
    return {
        "decision_id": _row_get(row, "id"),
        "symbol": _row_get(row, "symbol") or plan.get("symbol"),
        "action": _row_get(row, "action") or plan.get("action"),
        "no_entry_reason": reason,
        "raw_no_entry_reason": raw_reason,
        "decision_lane": plan.get("decision_lane"),
        "missing_required_fields": missing_fields,
        "expected_net_return_pct": _safe_float_or_none(plan.get("expected_net_return_pct")),
        "profit_quality_ratio": _safe_float_or_none(plan.get("profit_quality_ratio")),
        "loss_probability": _safe_float_or_none(plan.get("loss_probability")),
        "tail_loss_probability": _safe_float_or_none(plan.get("tail_loss_probability")),
        "profit_first_score": _safe_float_or_none(plan.get("profit_first_score")),
        "missed_opportunity_count": _missed_opportunity_count(raw),
        "shadow_return_pct": _shadow_return_pct(raw),
    }


def _losing_exit_row(position: Any) -> dict[str, Any]:
    raw = _safe_dict(_row_get(position, "entry_raw") or _row_get(position, "raw_llm_response"))
    attribution = normalize_losing_exit_attribution(position, entry_raw=raw)
    return {
        "position_id": _row_get(position, "id"),
        "symbol": _row_get(position, "symbol"),
        "side": _row_get(position, "side"),
        "realized_net_pnl": round(_safe_float(_row_get(position, "realized_pnl"), 0.0), 6),
        "losing_exit_attribution": attribution,
    }


def _model_contribution_rows(closed_positions: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in closed_positions:
        raw = _safe_dict(_row_get(position, "entry_raw") or _row_get(position, "raw_llm_response"))
        plan = _safe_dict(raw.get("profit_first_trade_plan"))
        pnl = _safe_float(_row_get(position, "realized_pnl"), 0.0)
        for contribution in _safe_list(plan.get("model_contributions")):
            source = str(_safe_dict(contribution).get("source") or "")
            if not source:
                continue
            rows.append(
                {
                    "position_id": _row_get(position, "id"),
                    "source": source,
                    "field_path": _safe_dict(contribution).get("field_path") or "",
                    "symbol": _row_get(position, "symbol"),
                    "side": _row_get(position, "side"),
                    "decision_lane": plan.get("decision_lane"),
                    "realized_net_pnl": round(pnl, 6),
                }
            )
    return rows


def _is_entry(row: Any) -> bool:
    return str(_row_get(row, "action") or "").lower() in {"long", "short", "buy", "sell"}


def _was_executed(row: Any) -> bool:
    return bool(_row_get(row, "was_executed"))


def _raw(row: Any) -> dict[str, Any]:
    return _safe_dict(_row_get(row, "raw_llm_response") or _row_get(row, "raw_response"))


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _counter_rows(counter: Counter[str], *, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in counter.most_common(limit)
    ]


def _no_entry_governance(no_entries: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts = Counter(str(row.get("no_entry_reason") or "evidence_insufficient") for row in no_entries)
    missing_field_counts: Counter[str] = Counter()
    missed_positive_shadow_count = 0
    missed_shadow_return_total = 0.0
    for row in no_entries:
        missing_field_counts.update(str(item) for item in _safe_list(row.get("missing_required_fields")))
        shadow_return = _safe_float(row.get("shadow_return_pct"), 0.0)
        missed_count = int(_safe_float(row.get("missed_opportunity_count"), 0.0))
        if shadow_return > 0 or missed_count > 0:
            missed_positive_shadow_count += 1
            missed_shadow_return_total += max(shadow_return, 0.0)
    recommendations = [
        _no_entry_recommendation(reason, count, missing_field_counts)
        for reason, count in reason_counts.most_common()
    ]
    diagnosis = "insufficient_sample"
    if no_entries:
        if missed_positive_shadow_count >= max(2, len(no_entries) // 3):
            diagnosis = "system_over_conservative_review"
        elif reason_counts.get("profit_insufficient", 0) >= len(no_entries) * 0.5:
            diagnosis = "market_unattractive_by_expected_net"
        elif (
            reason_counts.get("okx_unavailable_or_rejected", 0)
            + reason_counts.get("market_data_incomplete", 0)
            + reason_counts.get("phase3_model_unavailable", 0)
        ) >= len(no_entries) * 0.5:
            diagnosis = "external_data_or_model_unavailable"
        else:
            diagnosis = "mixed_blockers_review_top_reasons"
    return {
        "window_policy": "rolling_24h_no_entry_governance",
        "sample_count": len(no_entries),
        "diagnosis": diagnosis,
        "reason_counts": _counter_rows(reason_counts),
        "missing_field_counts": _counter_rows(missing_field_counts),
        "missed_positive_shadow_count": missed_positive_shadow_count,
        "missed_shadow_return_total_pct": round(missed_shadow_return_total, 6),
        "recommendations": recommendations,
        "live_mutation": False,
    }


def _no_entry_recommendation(
    reason: str,
    count: int,
    missing_field_counts: Counter[str],
) -> dict[str, Any]:
    action_by_reason = {
        "profit_insufficient": "keep_expected_net_floor; review only if shadow outcomes later prove missed profit",
        "evidence_insufficient": "collect shadow samples and require stronger independent source alignment",
        "risk_gate_blocked": "keep risk gate; inspect whether risk inputs are stale before threshold changes",
        "model_disagreement": "route to model-cooperation conflict review before live entry",
        "budget_insufficient": "do not open; review capital allocation only after profitable lanes pass",
        "position_capacity_occupied": "release or rotate only with net-benefit evidence",
        "same_side_crowded": "keep side concentration cap unless recent same-side edge is clean and positive",
        "okx_unavailable_or_rejected": "fix OKX/account/execution availability before considering entry",
        "market_data_incomplete": "repair market-data coverage before trusting model evidence",
        "phase3_model_unavailable": "keep candidates shadow until model-server health is verified",
        "shadow_only_missing_plan_fields": "require missing ProfitFirstTradePlan fields before real trading",
        "recent_realized_edge_negative": "keep tiny probes shadow until recent realized edge recovers",
    }
    evidence: dict[str, Any] = {"count": count}
    if reason == "shadow_only_missing_plan_fields":
        evidence["top_missing_fields"] = _counter_rows(missing_field_counts, limit=8)
    return {
        "reason": reason,
        "recommendation": action_by_reason.get(reason, "review_no_entry_reason"),
        "evidence": evidence,
        "live_mutation": False,
    }


def _losing_exit_governance(losing_exits: list[dict[str, Any]]) -> dict[str, Any]:
    attribution_counts = Counter(
        str(row.get("losing_exit_attribution") or "unknown_requires_review")
        for row in losing_exits
    )
    adjustments = [
        _exit_policy_adjustment(attribution, count)
        for attribution, count in attribution_counts.most_common()
    ]
    return {
        "sample_count": len(losing_exits),
        "attribution_counts": _counter_rows(attribution_counts),
        "exit_policy_adjustments": adjustments,
        "live_mutation": False,
    }


def _exit_policy_adjustment(attribution: str, count: int) -> dict[str, Any]:
    action_by_attribution = {
        "entry_wrong_direction": "demote direction source and require stronger opposite-side validation",
        "entry_late": "tighten entry timing and avoid chasing late moves",
        "stop_too_tight": "review stop generation; do not narrow stops further in this regime",
        "position_too_small_fee_drag": "stop tiny probes in this regime unless expected net profit clears fee drag",
        "hold_too_short": "require hard-risk evidence before fast loss exits",
        "trend_reversal": "increase trend-reversal detection weight before holding extensions",
        "model_false_positive": "demote contributing model source until shadow recovery",
        "server_profit_overestimated": "calibrate server profit estimates against realized net PnL",
        "timeseries_false_signal": "demote timeseries source in this regime until clean shadow evidence",
        "sentiment_false_signal": "demote sentiment source for this symbol/regime",
        "okx_slippage_or_execution": "raise execution-cost estimate and inspect OKX fill quality",
        "exit_too_early": "tighten capital-release exits and honor profit drawdown rules",
        "exit_too_late": "tighten max-hold/profit-drawdown exit checks",
        "capital_release_forced_loss": "require stronger replacement opportunity before loss-making release",
        "unknown_requires_review": "keep samples out of promotion until attribution is resolved",
    }
    return {
        "attribution": attribution,
        "count": count,
        "recommendation": action_by_attribution.get(attribution, "review_losing_exit_attribution"),
        "live_mutation": False,
    }


def _lane_threshold_recommendations(
    leaderboard_rows: list[dict[str, Any]],
    no_entry_governance: dict[str, Any],
    losing_exit_governance: dict[str, Any],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for row in leaderboard_rows:
        lane = str(row.get("decision_lane") or "unknown")
        count = int(row.get("count") or 0)
        pnl = _safe_float(row.get("realized_net_pnl"), 0.0)
        profit_factor = _safe_float(row.get("profit_factor"), 0.0)
        if count < 5:
            action = "keep_lane_shadow_until_sample_floor"
            reason = "sample_floor_not_met"
        elif pnl > 0 and profit_factor >= 1.12:
            action = "allow_lane_promotion_review"
            reason = "positive_realized_net_pnl"
        else:
            action = "tighten_or_keep_lane_threshold"
            reason = "weak_realized_net_pnl"
        recommendations.append(
            {
                "lane": lane,
                "recommendation": action,
                "reason": reason,
                "count": count,
                "realized_net_pnl": round(pnl, 6),
                "profit_factor": round(profit_factor, 6),
                "live_mutation": False,
            }
        )
    if no_entry_governance.get("diagnosis") == "system_over_conservative_review":
        recommendations.append(
            {
                "lane": "shadow_only",
                "recommendation": "review_shadow_to_tiny_or_validated_thresholds",
                "reason": "missed_positive_shadow_outcomes",
                "live_mutation": False,
            }
        )
    attribution_values = {
        str(item.get("value") or "")
        for item in _safe_list(losing_exit_governance.get("attribution_counts"))
    }
    if "position_too_small_fee_drag" in attribution_values:
        recommendations.append(
            {
                "lane": "tiny_probe",
                "recommendation": "pause_or_raise_quality_floor_for_tiny_probe",
                "reason": "position_too_small_fee_drag",
                "live_mutation": False,
            }
        )
    return recommendations


def _size_promotion_demotion(
    strategy_actions: list[dict[str, Any]],
    losing_exit_governance: dict[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row in strategy_actions:
        stage = str(row.get("recommended_stage") or "shadow")
        if stage in {"canary", "live_candidate"}:
            recommendation = "eligible_for_budget_increase_after_operator_gate"
        elif stage in {"demote", "disable"}:
            recommendation = "reduce_or_disable_budget"
        else:
            recommendation = "keep_shadow_or_sampling_size"
        actions.append(
            {
                "model_name": row.get("model_name"),
                "strategy_profile_id": row.get("strategy_profile_id"),
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "decision_lane": row.get("decision_lane"),
                "recommended_stage": stage,
                "recommendation": recommendation,
                "live_mutation": False,
            }
        )
    attribution_counts = {
        str(item.get("value") or ""): int(item.get("count") or 0)
        for item in _safe_list(losing_exit_governance.get("attribution_counts"))
    }
    if attribution_counts.get("position_too_small_fee_drag", 0) > 0:
        actions.append(
            {
                "decision_lane": "tiny_probe",
                "recommended_stage": "shadow",
                "recommendation": "do_not_continue_tiny_size_when_fee_drag_losses_repeat",
                "evidence": {
                    "position_too_small_fee_drag": attribution_counts["position_too_small_fee_drag"]
                },
                "live_mutation": False,
            }
        )
    return actions


def _missed_opportunity_count(raw: dict[str, Any]) -> int:
    candidates = [
        _safe_dict(raw.get("review_feedback")),
        _safe_dict(raw.get("memory_feedback")),
        _safe_dict(_safe_dict(raw.get("opportunity_score")).get("review_feedback")),
    ]
    evidence = _safe_dict(_safe_dict(raw.get("opportunity_score")).get("evidence_score"))
    for component in _safe_list(evidence.get("components")):
        row = _safe_dict(component)
        if row.get("source") == "shadow_memory":
            candidates.append(row)
    return max(int(_safe_float(item.get("missed_opportunity_count"), 0.0)) for item in candidates) if candidates else 0


def _shadow_return_pct(raw: dict[str, Any]) -> float | None:
    candidates = [
        raw,
        _safe_dict(raw.get("shadow_outcome")),
        _safe_dict(raw.get("shadow_result")),
        _safe_dict(raw.get("missed_opportunity")),
    ]
    keys = (
        "shadow_return_pct",
        "shadow_realized_return_pct",
        "realized_return_pct",
        "return_pct",
        "missed_opportunity_return_pct",
    )
    for item in candidates:
        for key in keys:
            value = _safe_float_or_none(item.get(key))
            if value is not None:
                return value
    return None
