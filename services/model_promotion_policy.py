from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import settings

PAPER_OBSERVATION_REPORT_REL_PATH = "phase3_paper_resume_observation_reports/latest.json"
MIN_PROMOTION_SHADOW_SAMPLES = 30
MIN_DIRECTION_HIT_RATE = 0.48
MIN_AVG_REALIZED_RETURN_PCT = 0.02
MAX_FALSE_SIGNAL_LOSS_PCT = -0.18
MIN_TIMESERIES_SEQUENCE_LENGTH = 30


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
        "specialist_shadow_gate": specialist_gate,
        "paper_observation_gate": paper_gate,
        "live_mutation": False,
    }
