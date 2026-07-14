"""Distribution-driven model promotion for fee-after returns."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from config.settings import settings
from services.profit_supervision import (
    AUTHORITATIVE_REALIZED_RETURN_TASK,
    COUNTERFACTUAL_EXECUTION_COST_TASK,
    MARKET_OPPORTUNITY_TASK,
    PRODUCTION_RETURN_COMBINATION_VERSION,
    PROFIT_SUPERVISION_VERSION,
    weighted_distribution,
)
from services.return_objective import (
    RETURN_LABEL_NAME,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
)

PAPER_OBSERVATION_REPORT_REL_PATH = "phase3_paper_resume_observation_reports/latest.json"
RETURN_PROMOTION_POLICY_VERSION = "2026-07-14.separated-return-promotion.v2"
logger = logging.getLogger(__name__)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def load_latest_paper_observation_report(root: Path | None = None) -> dict[str, Any]:
    root_candidate = (root or Path.cwd()) / "data" / PAPER_OBSERVATION_REPORT_REL_PATH
    candidates = (
        [root_candidate, settings.data_dir / PAPER_OBSERVATION_REPORT_REL_PATH]
        if root is not None
        else [settings.data_dir / PAPER_OBSERVATION_REPORT_REL_PATH, root_candidate]
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("paper observation report unavailable at %s: %s", path, exc)
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


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _lower_half(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return ordered[: max((len(ordered) + 1) // 2, 1)]


def build_return_objective_report(
    *,
    trade_samples: list[dict[str, Any]] | None = None,
    shadow_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    shadow_candidates = [
        _safe_dict(sample) for sample in (shadow_samples or []) if isinstance(sample, dict)
    ]
    trade_candidates = [
        _safe_dict(sample) for sample in (trade_samples or []) if isinstance(sample, dict)
    ]

    def task(sample: dict[str, Any], name: str) -> dict[str, Any]:
        supervision = _safe_dict(sample.get("profit_supervision"))
        if supervision.get("version") != PROFIT_SUPERVISION_VERSION:
            return {}
        return _safe_dict(_safe_dict(supervision.get("tasks")).get(name))

    def weighted_pairs(
        samples: list[dict[str, Any]],
        task_name: str,
        field: str,
    ) -> list[tuple[Any, Any]]:
        return [
            (current.get(field), sample.get("sample_weight", 1.0))
            for sample in samples
            if (current := task(sample, task_name)).get("eligible") is True
        ]

    market_long = weighted_pairs(
        shadow_candidates,
        MARKET_OPPORTUNITY_TASK,
        "long_gross_market_return_pct",
    )
    market_short = weighted_pairs(
        shadow_candidates,
        MARKET_OPPORTUNITY_TASK,
        "short_gross_market_return_pct",
    )
    shadow_cost_long = weighted_pairs(
        shadow_candidates,
        COUNTERFACTUAL_EXECUTION_COST_TASK,
        "long_total_cost_pct",
    )
    shadow_cost_short = weighted_pairs(
        shadow_candidates,
        COUNTERFACTUAL_EXECUTION_COST_TASK,
        "short_total_cost_pct",
    )
    actual_return_pairs = weighted_pairs(
        trade_candidates,
        AUTHORITATIVE_REALIZED_RETURN_TASK,
        "realized_net_return_pct",
    )
    actual_cost_pairs = weighted_pairs(
        trade_candidates,
        COUNTERFACTUAL_EXECUTION_COST_TASK,
        "total_cost_pct",
    )
    actual_slippage_pairs = weighted_pairs(
        trade_candidates,
        COUNTERFACTUAL_EXECUTION_COST_TASK,
        "slippage_pct",
    )
    returns = [
        float(value)
        for value, weight in actual_return_pairs
        if _safe_float(value, None) is not None and (_safe_float(weight, 0.0) or 0.0) > 0
    ]
    lower = _lower_half(returns) if returns else []
    profit = sum(max(value, 0.0) for value in returns)
    loss = abs(sum(min(value, 0.0) for value in returns))
    avg_return = sum(returns) / len(returns) if returns else None
    lower_hinge = _median(lower)
    downside_mean = (
        sum(abs(value) for value in returns if value < 0)
        / sum(value < 0 for value in returns)
        if any(value < 0 for value in returns)
        else 0.0
    )
    profit_factor = profit / loss if loss > 0 else None
    blockers: list[str] = []
    if not market_long or not market_short:
        blockers.append("shadow_market_opportunity_distribution_missing")
    if not shadow_cost_long or not shadow_cost_short:
        blockers.append("counterfactual_execution_cost_distribution_missing")
    if not returns:
        blockers.append("authoritative_realized_return_distribution_missing")
    if not actual_cost_pairs:
        blockers.append("authoritative_execution_cost_distribution_missing")
    if not actual_slippage_pairs:
        blockers.append("authoritative_slippage_distribution_missing")
    if avg_return is None or avg_return <= 0:
        blockers.append("average_fee_after_return_not_positive")
    if lower_hinge is None or lower_hinge <= 0:
        blockers.append("empirical_return_lower_hinge_not_positive")
    if profit <= loss:
        blockers.append("profit_factor_not_above_break_even")
    generated_at = datetime.now(UTC).isoformat()
    return {
        "available": bool(returns),
        "promotion_ready": not blockers,
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "optimization_target": "realized_fee_after_return",
        "sample_count": len(returns),
        "actual_realized_return_sample_count": len(actual_return_pairs),
        "shadow_market_opportunity_sample_count": min(len(market_long), len(market_short)),
        "shadow_counterfactual_cost_sample_count": min(
            len(shadow_cost_long), len(shadow_cost_short)
        ),
        "shadow_samples_are_actual_returns": False,
        "excluded_cost_incomplete_count": len(trade_candidates) - len(actual_cost_pairs),
        "separated_distributions": {
            "market_opportunity": {
                "source_authority": "shadow_native_market_path",
                "long_gross_return_pct": weighted_distribution(market_long),
                "short_gross_return_pct": weighted_distribution(market_short),
            },
            "counterfactual_execution_cost": {
                "source_authority": "shadow_live_microstructure",
                "long_total_cost_pct": weighted_distribution(shadow_cost_long),
                "short_total_cost_pct": weighted_distribution(shadow_cost_short),
            },
            "authoritative_realized_trade": {
                "source_authority": "okx_position_history",
                "net_return_after_cost_pct": weighted_distribution(actual_return_pairs),
                "execution_cost_pct": weighted_distribution(actual_cost_pairs),
                "slippage_pct": weighted_distribution(actual_slippage_pairs),
            },
        },
        "production_combination": {
            "version": PRODUCTION_RETURN_COMBINATION_VERSION,
            "formula": (
                "market_opportunity_distribution-live_execution_cost_distribution-"
                "authoritative_slippage_tail_excess"
            ),
            "ready": not any(
                reason.endswith("distribution_missing") for reason in blockers
            ),
        },
        "average_net_return_after_cost_pct": (
            round(avg_return, 8) if avg_return is not None else None
        ),
        "median_net_return_after_cost_pct": (
            round(float(_median(returns)), 8) if returns else None
        ),
        "empirical_return_lower_hinge_pct": (
            round(float(lower_hinge), 8) if lower_hinge is not None else None
        ),
        "downside_mean_pct": round(downside_mean, 8),
        "profit_factor": round(profit_factor, 8) if profit_factor is not None else None,
        "gross_positive_return_pct": round(profit, 8),
        "gross_negative_return_pct": round(loss, 8),
        "blocking_reasons": blockers,
        "policy_provenance": {
            "source": (
                "shadow_market_and_cost_observation_plus_authoritative_okx_trade_returns"
            ),
            "observation_window": "provided_non_overlapping_training_evaluation_samples",
            "sample_count": len(returns),
            "generated_at": generated_at,
            "strategy_version": RETURN_PROMOTION_POLICY_VERSION,
            "fallback_reason": "" if not blockers else ";".join(blockers),
        },
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
    return_objective_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality = _safe_dict(quality_report)
    governance = _safe_dict(governance_report)
    policy = _safe_dict(evaluation_policy)
    paper = _safe_dict(paper_observation_report)
    return_report = _safe_dict(return_objective_report)
    totals = _safe_dict(quality.get("totals"))
    total = _safe_int(totals.get("total"))
    trainable = _safe_int(governance.get("trainable_sample_count"))
    contamination = str(governance.get("contamination_risk") or "unknown").lower()
    effective_weight = _safe_float(totals.get("effective_weight_ratio"), None)
    blockers: list[str] = []
    if total <= 0 or trainable <= 0:
        blockers.append("no_trainable_samples")
    if contamination == "high":
        blockers.append("high_contamination_risk")
    if effective_weight is not None and effective_weight <= 0:
        blockers.append("effective_training_weight_zero")
    if return_report.get("promotion_ready") is not True:
        blockers.extend(
            str(reason)
            for reason in _safe_list(return_report.get("blocking_reasons"))
            if reason
        )
        if not return_report:
            blockers.append("return_objective_report_missing")

    paper_required = bool(policy.get("requires_paper_observation", True))
    if paper_required and not bool(paper.get("can_use_for_promotion")):
        blockers.append("paper_observation_not_healthy")
    for unsafe_key in ("starts_trading_service", "submits_orders", "changes_model_routing"):
        if bool(paper.get(unsafe_key)):
            blockers.append(f"paper_observation_unsafe:{unsafe_key}")

    canary_blockers = list(dict.fromkeys(blockers))
    live_blockers = list(canary_blockers)
    if str(training_mode or "").lower() != "walk_forward":
        live_blockers.append("walk_forward_required")
    if str(model_stage or "").lower() != "live":
        live_blockers.append("model_stage_not_live")
    if not bool(policy.get("live_mutation")):
        live_blockers.append("live_mutation_not_enabled")
    live_blockers = list(dict.fromkeys(live_blockers))
    recommended_stage = "live" if not live_blockers else "canary" if not canary_blockers else "shadow"
    if contamination == "high" or str(model_stage or "").lower() in {"degraded", "retired"}:
        recommended_stage = "degraded"

    return {
        "policy": RETURN_PROMOTION_POLICY_VERSION,
        "optimization_target": "realized_fee_after_return",
        "current_stage": str(model_stage or "shadow").lower(),
        "training_mode": str(training_mode or "shadow").lower(),
        "recommended_stage": recommended_stage,
        "canary_ready": not canary_blockers,
        "live_ready": not live_blockers,
        "canary_blocking_reasons": canary_blockers,
        "live_blocking_reasons": live_blockers,
        "observed_sample_counts": {
            "completed_shadow_sample_count": int(completed_shadow_sample_count or 0),
            "completed_trade_sample_count": int(completed_trade_sample_count or 0),
            "counts_are_diagnostic_only": True,
        },
        "quality_gate": {
            "total_samples": total,
            "trainable_sample_count": trainable,
            "effective_weight_ratio": effective_weight,
            "contamination_risk": contamination,
        },
        "return_objective_gate": return_report,
        "paper_observation_gate": {
            "required": paper_required,
            "status": paper.get("status") or "missing",
            "can_use_for_promotion": bool(paper.get("can_use_for_promotion")),
        },
        "live_mutation": False,
    }
