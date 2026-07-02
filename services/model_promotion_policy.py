from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import settings
from services.profit_first_ranking import ProfitFirstRankingService

PAPER_OBSERVATION_REPORT_REL_PATH = "phase3_paper_resume_observation_reports/latest.json"
MIN_PROMOTION_SHADOW_SAMPLES = 30
MIN_DIRECTION_HIT_RATE = 0.48
MIN_AVG_REALIZED_RETURN_PCT = 0.02
MAX_FALSE_SIGNAL_LOSS_PCT = -0.18
MIN_TIMESERIES_SEQUENCE_LENGTH = 30


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_latest_paper_observation_report(root: Path | None = None) -> dict[str, Any]:
    """Load the latest Phase 3 paper observation report without mutating state."""

    root_candidate = (root or Path.cwd()) / "data" / PAPER_OBSERVATION_REPORT_REL_PATH
    candidates = (
        [root_candidate, settings.data_dir / PAPER_OBSERVATION_REPORT_REL_PATH]
        if root is not None
        else [settings.data_dir / PAPER_OBSERVATION_REPORT_REL_PATH, root_candidate]
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payload.setdefault("available", True)
            payload.setdefault("report_path", str(path))
            return payload
    return {
        "available": False,
        "status": "missing",
        "can_use_for_promotion": False,
        "candidate_paths": [str(path) for path in candidates],
    }


def build_phase3_promotion_recommendation(
    *,
    training_mode: str,
    model_stage: str,
    quality_report: dict[str, Any] | None,
    governance_report: dict[str, Any] | None,
    evaluation_policy: dict[str, Any] | None = None,
    paper_observation_report: dict[str, Any] | None = None,
    completed_shadow_sample_count: int = 0,
    completed_trade_sample_count: int = 0,
    profit_first_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only lifecycle recommendation for a trained model bundle."""

    quality = _safe_dict(quality_report)
    governance = _safe_dict(governance_report)
    policy = _safe_dict(evaluation_policy)
    paper_observation = _safe_dict(paper_observation_report)
    totals = _safe_dict(quality.get("totals"))
    excluded = _safe_int(totals.get("excluded"))
    total = _safe_int(totals.get("total"))
    effective_weight_ratio = _safe_float(totals.get("effective_weight_ratio"))
    trainable = _safe_int(governance.get("trainable_sample_count"))
    contamination_risk = str(governance.get("contamination_risk") or "unknown").lower()
    specialist_models = _safe_dict(quality.get("specialist_shadow_models"))
    profit_gate = _profit_first_promotion_gate(
        quality_report=quality,
        profit_first_report=_safe_dict(profit_first_report),
        completed_trade_sample_count=completed_trade_sample_count,
    )
    mode = str(training_mode or "shadow").lower()
    stage = str(model_stage or "shadow").lower()

    blockers: list[str] = []
    if total <= 0 or trainable <= 0:
        blockers.append("no_trainable_samples")
    if completed_shadow_sample_count < 100:
        blockers.append("shadow_sample_floor_not_met")
    if completed_trade_sample_count < 20:
        blockers.append("trade_sample_floor_not_met")
    if excluded and contamination_risk == "high":
        blockers.append("high_contamination_risk")
    if effective_weight_ratio and effective_weight_ratio < 0.50:
        blockers.append("low_effective_training_weight")
    blockers.extend(_safe_list(profit_gate.get("canary_blocking_reasons")))
    paper_gate_required = bool(policy.get("requires_paper_observation", True))
    paper_status = str(paper_observation.get("status") or "missing").lower()
    paper_gate: dict[str, Any] = {
        "required": paper_gate_required,
        "status": paper_status,
        "paper_active": bool(paper_observation.get("paper_active")),
        "can_use_for_promotion": bool(paper_observation.get("can_use_for_promotion")),
        "checked_at": paper_observation.get("checked_at"),
        "blocker_count": len(paper_observation.get("blockers") or []),
        "warning_count": len(paper_observation.get("warnings") or []),
        "starts_trading_service": bool(paper_observation.get("starts_trading_service")),
        "submits_orders": bool(paper_observation.get("submits_orders")),
        "changes_model_routing": bool(paper_observation.get("changes_model_routing")),
    }
    if paper_gate_required:
        if not paper_observation:
            blockers.append("paper_observation_report_missing")
        elif not bool(paper_observation.get("can_use_for_promotion")):
            blockers.append(f"paper_observation_not_healthy:{paper_status}")
        if bool(paper_observation.get("starts_trading_service")):
            blockers.append("paper_observation_unsafe_starts_trading")
        if bool(paper_observation.get("submits_orders")):
            blockers.append("paper_observation_unsafe_submits_orders")
        if bool(paper_observation.get("changes_model_routing")):
            blockers.append("paper_observation_unsafe_changes_model_routing")
    specialist_gate: dict[str, Any] = {}
    for name, raw_row in specialist_models.items():
        row = _safe_dict(raw_row)
        gate_name = str(row.get("model_key") or name)
        actual_inference_count = _safe_int(row.get("actual_inference_count"))
        direction_count = _safe_int(row.get("direction_count"))
        direction_hit_rate = _safe_float(row.get("direction_hit_rate"))
        avg_realized_return_pct = _safe_float(row.get("avg_realized_return_pct"), None)
        worst_realized_return_pct = _safe_float(row.get("worst_realized_return_pct"), None)
        false_signal_count = _safe_int(row.get("false_signal_count"))
        tail_loss_count = _safe_int(row.get("tail_loss_count"))
        sequence_too_short_count = _safe_int(row.get("sequence_too_short_count"))
        legacy_mixed_shadow_count = _safe_int(row.get("legacy_mixed_shadow_count"))
        legacy_quarantined_count = _safe_int(row.get("legacy_quarantined_count"))
        legacy_sequence_too_short_count = _safe_int(
            row.get("legacy_sequence_too_short_count")
        )
        row_blockers = [
            str(reason)
            for reason in (row.get("promotion_blockers") or row.get("blockers") or [])
            if reason
        ]
        specialist_gate[gate_name] = {
            "tool": row.get("tool"),
            "model": row.get("model"),
            "actual_inference_count": actual_inference_count,
            "direction_count": direction_count,
            "direction_hit_rate": round(direction_hit_rate, 4),
            "avg_realized_return_pct": (
                None
                if avg_realized_return_pct is None
                else round(float(avg_realized_return_pct), 6)
            ),
            "worst_realized_return_pct": worst_realized_return_pct,
            "false_signal_count": false_signal_count,
            "tail_loss_count": tail_loss_count,
            "tail_loss_symbols": row.get("tail_loss_symbols") or [],
            "worst_samples": row.get("worst_samples") or [],
            "sequence_too_short_count": sequence_too_short_count,
            "legacy_mixed_shadow_count": legacy_mixed_shadow_count,
            "legacy_quarantined_count": legacy_quarantined_count,
            "legacy_sequence_too_short_count": legacy_sequence_too_short_count,
            "promotion_blockers": row_blockers,
            "minimum_actual_inference_samples": MIN_PROMOTION_SHADOW_SAMPLES,
            "minimum_direction_hit_rate": MIN_DIRECTION_HIT_RATE,
            "minimum_avg_realized_return_pct": MIN_AVG_REALIZED_RETURN_PCT,
            "max_false_signal_loss_pct": MAX_FALSE_SIGNAL_LOSS_PCT,
            "minimum_timeseries_sequence_length": MIN_TIMESERIES_SEQUENCE_LENGTH,
        }
        if actual_inference_count < MIN_PROMOTION_SHADOW_SAMPLES:
            blockers.append(f"{gate_name}_specialist_shadow_sample_floor_not_met")
        if direction_count >= MIN_PROMOTION_SHADOW_SAMPLES and direction_hit_rate < MIN_DIRECTION_HIT_RATE:
            blockers.append(f"{gate_name}_specialist_direction_hit_rate_low")
        if (
            direction_count >= MIN_PROMOTION_SHADOW_SAMPLES
            and avg_realized_return_pct is not None
            and avg_realized_return_pct < MIN_AVG_REALIZED_RETURN_PCT
        ):
            blockers.append(f"{gate_name}_avg_realized_return_below_floor")
        if tail_loss_count > 0 or (
            worst_realized_return_pct is not None
            and worst_realized_return_pct <= MAX_FALSE_SIGNAL_LOSS_PCT
        ):
            blockers.append(f"{gate_name}_false_signal_loss_exceeds_floor")
        if sequence_too_short_count > 0:
            blockers.append(f"{gate_name}_timeseries_sequence_too_short_for_promotion")
        for reason in row_blockers:
            if reason in {
                "specialist_shadow_sample_floor_not_met",
                "direction_hit_rate_below_floor",
                "avg_realized_return_below_floor",
                "false_signal_loss_exceeds_floor",
                "timeseries_sequence_too_short_for_promotion",
                "legacy_mixed_shadow_result_not_promotable",
            }:
                continue
            blockers.append(f"{gate_name}_{reason}")

    canary_blockers = list(blockers)
    live_blockers = list(blockers)
    live_blockers.extend(_safe_list(profit_gate.get("live_blocking_reasons")))
    if mode != "walk_forward":
        live_blockers.append("walk_forward_required")
    if stage != "live":
        live_blockers.append("model_stage_not_live")
    if not bool(policy.get("live_mutation")):
        live_blockers.append("live_mutation_not_enabled")

    if live_blockers:
        recommended_stage = "canary" if not canary_blockers else "shadow"
    else:
        recommended_stage = "live"
    if stage in {"degraded", "retired"} or "high_contamination_risk" in blockers:
        recommended_stage = "degraded"

    return {
        "policy": "phase3_shadow_to_canary_to_live",
        "current_stage": stage,
        "training_mode": mode,
        "recommended_stage": recommended_stage,
        "canary_ready": not bool(canary_blockers),
        "live_ready": not bool(live_blockers),
        "canary_blocking_reasons": list(dict.fromkeys(canary_blockers)),
        "live_blocking_reasons": list(dict.fromkeys(live_blockers)),
        "sample_floor": {
            "completed_shadow_sample_count": int(completed_shadow_sample_count or 0),
            "completed_trade_sample_count": int(completed_trade_sample_count or 0),
            "minimum_shadow_samples": 100,
            "minimum_trade_samples": 20,
        },
        "quality_gate": {
            "total_samples": total,
            "trainable_sample_count": trainable,
            "excluded_sample_count": excluded,
            "effective_weight_ratio": round(effective_weight_ratio, 4),
            "contamination_risk": contamination_risk,
        },
        "profit_first_gate": profit_gate,
        "runtime_permissions": _safe_dict(profit_gate.get("runtime_permissions")),
        "specialist_shadow_gate": specialist_gate,
        "paper_observation_gate": paper_gate,
        "live_mutation": False,
    }


def build_profit_first_promotion_report(
    *,
    trade_samples: list[dict[str, Any]] | None = None,
    shadow_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a lightweight Profit-First evidence report for promotion gates.

    This is intentionally read-only and operates on already-collected training
    samples so rebuild/preflight/dashboard flows can share the same evidence
    without re-querying the database.
    """

    ranking = ProfitFirstRankingService(min_canary_samples=1, min_live_samples=3).build_report(
        decisions=_promotion_decision_rows(shadow_samples or []),
        closed_positions=_promotion_trade_rows(trade_samples or []),
    )
    ranking.setdefault("evidence_source", "phase3_training_samples")
    ranking.setdefault(
        "policy_context",
        "promotion_gate_runtime_feedback_only_from_collected_training_samples",
    )
    return ranking


def _promotion_decision_rows(shadow_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in shadow_samples:
        item = _safe_dict(sample)
        if not item:
            continue
        action = str(item.get("decision_action") or item.get("action") or "").lower().strip()
        missed = bool(item.get("missed_opportunity"))
        long_return = _safe_float(item.get("long_return_pct"), None)
        short_return = _safe_float(item.get("short_return_pct"), None)
        if not action and not missed and long_return is None and short_return is None:
            continue
        row = dict(item)
        row.setdefault("action", action)
        row.setdefault("symbol", item.get("symbol"))
        row.setdefault("was_executed", bool(action in {"long", "short", "buy", "sell"} and not missed))
        row.setdefault(
            "raw_llm_response",
            _promotion_shadow_raw(
                item,
                action=action,
                missed=missed,
                long_return=long_return,
                short_return=short_return,
            ),
        )
        rows.append(row)
    return rows


def _promotion_trade_rows(trade_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in trade_samples:
        item = _safe_dict(sample)
        if not item:
            continue
        if _safe_float(item.get("realized_pnl"), None) is None:
            continue
        rows.append(dict(item))
    return rows


def _promotion_shadow_raw(
    sample: dict[str, Any],
    *,
    action: str,
    missed: bool,
    long_return: float | None,
    short_return: float | None,
) -> dict[str, Any]:
    raw = _safe_dict(sample.get("raw_llm_response") or sample.get("raw_response"))
    if raw:
        return raw
    best_return = None
    if long_return is not None or short_return is not None:
        candidates = [value for value in (long_return, short_return) if value is not None]
        if candidates:
            best_return = max(candidates)
    return {
        "profit_first_trade_plan": {
            "action": action or sample.get("decision_action"),
            "decision_lane": sample.get("decision_lane"),
            "strategy_profile_id": sample.get("strategy_profile_id"),
        },
        "review_feedback": {
            "missed_opportunity_count": 1 if missed else 0,
        },
        "shadow_outcome": {
            "shadow_return_pct": best_return,
        },
    }


def _profit_first_promotion_gate(
    *,
    quality_report: dict[str, Any],
    profit_first_report: dict[str, Any],
    completed_trade_sample_count: int,
) -> dict[str, Any]:
    runtime = _safe_dict(profit_first_report.get("runtime_feedback"))
    summary = _safe_dict(profit_first_report.get("summary"))
    acceptance = _safe_dict(runtime.get("profit_acceptance"))
    missed = _safe_dict(runtime.get("missed_opportunity_feedback"))
    strategy_rows = _safe_list(profit_first_report.get("strategy_rankings")) or _safe_list(
        runtime.get("strategy_profile_feedback")
    )
    source_rows = _safe_list(profit_first_report.get("source_rankings")) or _safe_list(
        runtime.get("source_weight_feedback")
    )
    size_rows = _safe_list(runtime.get("size_feedback"))
    lane_rows = _safe_list(runtime.get("lane_feedback"))
    exit_rows = _safe_list(runtime.get("exit_feedback"))
    quality_profit = _safe_dict(quality_report.get("profit_learning_summary"))
    trade_profit = _safe_dict(_safe_dict(quality_profit.get("trade")).get("label_counts"))
    shadow_profit = _safe_dict(_safe_dict(quality_profit.get("shadow")).get("label_counts"))

    canary_candidate_count = sum(
        1
        for row in strategy_rows
        if str(_safe_dict(row).get("recommended_stage") or "") in {"canary", "live_candidate"}
    )
    live_candidate_count = sum(
        1
        for row in strategy_rows
        if str(_safe_dict(row).get("recommended_stage") or "") == "live_candidate"
    )
    demote_count = sum(
        1 for row in strategy_rows if str(_safe_dict(row).get("recommended_stage") or "") == "demote"
    )
    disable_count = sum(
        1 for row in strategy_rows if str(_safe_dict(row).get("recommended_stage") or "") == "disable"
    )
    source_promote_count = sum(
        1 for row in source_rows if str(_safe_dict(row).get("recommended_stage") or "") == "promote"
    )
    source_demote_count = sum(
        1 for row in source_rows if str(_safe_dict(row).get("recommended_stage") or "") == "demote"
    )
    size_expand_count = sum(
        1
        for row in size_rows
        if str(_safe_dict(row).get("sizing_bias") or "")
        == "quality_entries_can_expand_after_validation"
    )
    size_reduce_count = sum(
        1
        for row in size_rows
        if str(_safe_dict(row).get("sizing_bias") or "") == "reduce_weak_or_fee_drag_size"
    )
    entry_expand_requested = bool(
        missed.get("entry_bias") == "expand_quality_entries"
        or any(
            str(_safe_dict(row).get("entry_bias") or "") == "expand_quality_entries"
            for row in lane_rows
        )
    )
    fee_drag_loss_count = _label_count(trade_profit, "losing_exit_attribution", "position_too_small_fee_drag")
    fee_drag_loss_count += _label_count(trade_profit, "trade_profit_class", "cost_drag_loss")
    missed_positive_shadow_count = _safe_int(missed.get("missed_positive_shadow_count"))
    if missed_positive_shadow_count <= 0:
        missed_positive_shadow_count = _label_count(
            shadow_profit,
            "missed_opportunity_label",
            "missed_positive_entry",
        )
    exit_hold_count = sum(
        _safe_int(_safe_dict(row).get("count"))
        for row in exit_rows
        if str(_safe_dict(row).get("exit_bias") or "") == "hold_winners_longer"
    )
    exit_cut_count = sum(
        _safe_int(_safe_dict(row).get("count"))
        for row in exit_rows
        if str(_safe_dict(row).get("exit_bias") or "") == "cut_losers_faster"
    )
    window_closed_trade_count = _safe_int(acceptance.get("window_closed_trade_count"))
    net_pnl = _safe_float(acceptance.get("net_pnl"), 0.0)
    profit_factor = _safe_float(acceptance.get("profit_factor"), 0.0)
    canary_blockers: list[str] = []
    live_blockers: list[str] = []
    if profit_first_report:
        if window_closed_trade_count >= max(1, min(int(completed_trade_sample_count or 0), 20)):
            if net_pnl <= 0:
                canary_blockers.append("profit_first_net_pnl_non_positive")
                live_blockers.append("profit_first_net_pnl_non_positive")
            if 0 < profit_factor < 1.0:
                canary_blockers.append("profit_first_profit_factor_below_unity")
                live_blockers.append("profit_first_profit_factor_below_unity")
        if disable_count > 0:
            canary_blockers.append("profit_first_disable_candidate_present")
            live_blockers.append("profit_first_disable_candidate_present")
        if canary_candidate_count <= 0 and (window_closed_trade_count > 0 or demote_count > 0):
            canary_blockers.append("profit_first_no_canary_candidate")
        if live_candidate_count <= 0 and (window_closed_trade_count > 0 or canary_candidate_count > 0):
            live_blockers.append("profit_first_no_live_candidate")
        if source_promote_count <= 0 and source_demote_count > 0:
            canary_blockers.append("profit_first_source_mix_not_ready")
        if size_reduce_count > 0 and size_expand_count <= 0 and fee_drag_loss_count > 0:
            canary_blockers.append("profit_first_size_feedback_reduce_only")
            live_blockers.append("profit_first_size_feedback_reduce_only")

    canary_budget_permission = "shadow_only"
    live_budget_permission = "shadow_only"
    size_permission = "shadow_only"
    leverage_permission = "shadow_only"
    if not canary_blockers and canary_candidate_count > 0:
        canary_budget_permission = "operator_review_canary_expand"
        if size_expand_count > 0 or entry_expand_requested:
            size_permission = "operator_review_canary_expand"
            leverage_permission = "operator_review_canary_expand"
    if not live_blockers and live_candidate_count > 0:
        live_budget_permission = "operator_review_live_expand"
        size_permission = "operator_review_live_expand"
        leverage_permission = "operator_review_live_expand"

    return {
        "available": bool(profit_first_report or quality_profit),
        "summary": {
            "window_closed_trade_count": window_closed_trade_count,
            "net_pnl": round(net_pnl, 6),
            "profit_factor": round(profit_factor, 6),
            "promote_candidate_count": _safe_int(summary.get("promote_candidate_count"), canary_candidate_count),
            "demote_count": _safe_int(summary.get("demote_count"), demote_count),
            "disable_count": _safe_int(summary.get("disable_count"), disable_count),
            "canary_candidate_count": canary_candidate_count,
            "live_candidate_count": live_candidate_count,
            "source_promote_count": source_promote_count,
            "source_demote_count": source_demote_count,
            "size_expand_count": size_expand_count,
            "size_reduce_count": size_reduce_count,
            "fee_drag_loss_count": fee_drag_loss_count,
            "missed_positive_shadow_count": missed_positive_shadow_count,
            "winner_hold_feedback_count": exit_hold_count,
            "loser_cut_feedback_count": exit_cut_count,
        },
        "entry_expansion_candidate": entry_expand_requested or missed_positive_shadow_count > 0,
        "canary_blocking_reasons": list(dict.fromkeys(canary_blockers)),
        "live_blocking_reasons": list(dict.fromkeys(live_blockers)),
        "runtime_permissions": {
            "canary_budget_permission": canary_budget_permission,
            "live_budget_permission": live_budget_permission,
            "size_permission": size_permission,
            "leverage_permission": leverage_permission,
        },
    }


def _label_count(summary: dict[str, Any], key: str, value: str) -> int:
    rows = _safe_list(_safe_dict(summary.get(key)).get("rows")) if isinstance(summary.get(key), dict) else []
    if not rows:
        rows = _safe_list(summary.get(key))
    for row in rows:
        item = _safe_dict(row)
        if str(item.get("value") or "") == value:
            return _safe_int(item.get("count"))
    return 0
