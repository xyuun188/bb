from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from services.return_objective import (
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
)
from services.training_data_quality import (
    DATA_QUALITY_VERSION,
    MARKET_FACT_CONTRACT_VERSION,
)


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        result = float(value)
        return (
            result if result == result and result not in {float("inf"), float("-inf")} else default
        )
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _reason(
    code: str,
    message: str,
    *,
    actual: Any = None,
    required: Any = None,
) -> dict[str, Any]:
    payload = {"code": code, "message": message}
    if actual is not None:
        payload["actual"] = actual
    if required is not None:
        payload["required"] = required
    return payload


def _quality_totals(metadata: dict[str, Any]) -> dict[str, Any]:
    quality = _safe_dict(metadata.get("quality_report"))
    return _safe_dict(quality.get("totals"))


def _market_fact_contract_blockers(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    contract = _safe_dict(metadata.get("market_fact_contract"))
    blockers: list[dict[str, Any]] = []
    if contract.get("version") != MARKET_FACT_CONTRACT_VERSION:
        blockers.append(
            _reason(
                "artifact_market_fact_contract_missing_or_stale",
                "Artifact is not bound to the required native market-fact contract.",
                actual=contract.get("version") or "missing",
                required=MARKET_FACT_CONTRACT_VERSION,
            )
        )
        return blockers

    status = str(contract.get("status") or "").strip().lower()
    violation_count = contract.get("violation_count")
    if status != "clean" or violation_count != 0:
        blockers.append(
            _reason(
                "artifact_market_fact_contract_violated",
                "Artifact training data contains unresolved native market-fact violations.",
                actual={"status": status or "missing", "violation_count": violation_count},
                required={"status": "clean", "violation_count": 0},
            )
        )

    assertions = _safe_dict(contract.get("assertions"))
    required_assertions = (
        "native_instrument_identity_verified",
        "same_contract_price_path_verified",
        "executable_market_fact_verified",
    )
    failed_assertions = [name for name in required_assertions if assertions.get(name) is not True]
    if failed_assertions:
        blockers.append(
            _reason(
                "artifact_market_fact_assertions_incomplete",
                "Artifact market-fact assertions are incomplete.",
                actual=failed_assertions,
                required=list(required_assertions),
            )
        )

    provenance = _safe_dict(contract.get("provenance"))
    required_provenance = (
        "source",
        "observation_window",
        "generated_at",
        "strategy_version",
        "fallback_reason",
        "data_fingerprint",
    )
    missing_provenance = [
        name
        for name in required_provenance
        if name not in provenance or (name != "fallback_reason" and not provenance.get(name))
    ]
    if not any(name in provenance for name in ("sample_count", "effective_sample_size")):
        missing_provenance.append("sample_count/effective_sample_size")
    if missing_provenance:
        blockers.append(
            _reason(
                "artifact_market_fact_provenance_incomplete",
                "Artifact market-fact provenance is incomplete.",
                actual=missing_provenance,
                required=list(required_provenance) + ["sample_count/effective_sample_size"],
            )
        )
    return blockers


def _side_metric_blockers(metrics: dict[str, Any], side: str) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    top_return = _safe_float(metrics.get(f"top_{side}_avg_return_pct"), 0.0) or 0.0
    bottom_return = _safe_float(metrics.get(f"bottom_{side}_avg_return_pct"), 0.0) or 0.0
    top_return_lcb = _safe_float(metrics.get(f"top_{side}_return_lcb_pct"), None)
    top_profit_factor = _safe_float(metrics.get(f"top_{side}_profit_factor"), None)
    top_tail_loss = _safe_float(metrics.get(f"top_{side}_tail_loss_rate"), None)
    bottom_tail_loss = _safe_float(metrics.get(f"bottom_{side}_tail_loss_rate"), None)
    if top_return <= bottom_return:
        blockers.append(
            _reason(
                f"{side}_top_return_not_above_bottom",
                f"{side} top-score bucket return is not above the bottom bucket.",
                actual=round(top_return, 4),
                required=round(bottom_return, 4),
            )
        )
    if top_return_lcb is None or top_return_lcb <= 0:
        blockers.append(
            _reason(
                f"{side}_top_return_lcb_not_positive",
                f"{side} top-score return confidence lower bound is not positive.",
                actual=None if top_return_lcb is None else round(top_return_lcb, 4),
                required=0.0,
            )
        )
    if top_profit_factor is None or top_profit_factor <= 1.0:
        blockers.append(
            _reason(
                f"{side}_top_profit_factor_not_above_one",
                f"{side} top-score Profit Factor is not above one.",
                actual=None if top_profit_factor is None else round(top_profit_factor, 4),
                required=1.0,
            )
        )
    if (
        top_tail_loss is None
        or bottom_tail_loss is None
        or top_tail_loss > bottom_tail_loss
    ):
        blockers.append(
            _reason(
                f"{side}_top_tail_loss_not_improved",
                f"{side} top-score tail-loss rate is missing or worse than the bottom bucket.",
                actual=None if top_tail_loss is None else round(top_tail_loss, 4),
                required=None if bottom_tail_loss is None else round(bottom_tail_loss, 4),
            )
        )
    return blockers


def _side_profit_quality_diagnostics(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    score_bucket_diagnostics = _safe_dict(metadata.get("score_bucket_diagnostics"))
    side_buckets = _safe_dict(score_bucket_diagnostics.get(side))
    top_bucket = _safe_dict(side_buckets.get("top"))
    bottom_bucket = _safe_dict(side_buckets.get("bottom"))
    top_return = _safe_float(metrics.get(f"top_{side}_avg_return_pct"), None)
    bottom_return = _safe_float(metrics.get(f"bottom_{side}_avg_return_pct"), None)
    top_return_lcb = _safe_float(metrics.get(f"top_{side}_return_lcb_pct"), None)
    top_profit_factor = _safe_float(metrics.get(f"top_{side}_profit_factor"), None)
    top_win = _safe_float(metrics.get(f"top_{side}_win_rate"), None)
    bottom_win = _safe_float(metrics.get(f"bottom_{side}_win_rate"), None)
    top_tail_loss = _safe_float(metrics.get(f"top_{side}_tail_loss_rate"), None)
    bottom_tail_loss = _safe_float(metrics.get(f"bottom_{side}_tail_loss_rate"), None)
    if top_tail_loss is None:
        top_tail_loss = _safe_float(top_bucket.get("tail_loss_rate"), None)
    if bottom_tail_loss is None:
        bottom_tail_loss = _safe_float(bottom_bucket.get("tail_loss_rate"), None)
    spread = (
        None
        if top_return is None or bottom_return is None
        else round(top_return - bottom_return, 6)
    )
    diagnosis: list[str] = []
    if spread is not None and spread <= 0:
        diagnosis.append("top_score_bucket_not_better_than_bottom")
    if top_return_lcb is None or top_return_lcb <= 0:
        diagnosis.append("top_score_return_lcb_not_positive")
    if top_profit_factor is None or top_profit_factor <= 1.0:
        diagnosis.append("top_score_profit_factor_not_above_one")
    if top_tail_loss is not None and bottom_tail_loss is not None and top_tail_loss > bottom_tail_loss:
        diagnosis.append("top_score_tail_loss_worse_than_bottom")
    return {
        "side": side,
        "training_target": "fee_after_realized_return_quality",
        "top_avg_return_pct": top_return,
        "bottom_avg_return_pct": bottom_return,
        "top_bottom_return_spread_pct": spread,
        "top_return_lcb_pct": top_return_lcb,
        "top_profit_factor": top_profit_factor,
        "top_win_rate": top_win,
        "bottom_win_rate": bottom_win,
        "top_tail_loss_rate": top_tail_loss,
        "bottom_tail_loss_rate": bottom_tail_loss,
        "top_bucket": top_bucket,
        "bottom_bucket": bottom_bucket,
        "diagnosis": diagnosis,
    }


def build_ml_readiness_report(
    metadata: dict[str, Any],
    influence: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    metrics = _safe_dict(metadata.get("metrics"))
    quality = _safe_dict(metadata.get("quality_report"))
    totals = _quality_totals(metadata)
    sample_count = int(metadata.get("sample_count") or 0)
    test_count = int(metadata.get("test_count") or 0)
    total_samples = int(totals.get("total") or sample_count or 0)
    excluded_count = int(totals.get("excluded") or 0)
    downweighted_count = int(totals.get("downweighted") or 0)
    contamination_downweighted_count = totals.get("contamination_downweighted")
    if contamination_downweighted_count is None:
        contamination_downweighted_count = downweighted_count
    contamination_downweighted_count = int(contamination_downweighted_count or 0)
    benign_downweighted_count = int(totals.get("benign_downweighted") or 0)
    dirty_count = excluded_count + contamination_downweighted_count
    dirty_ratio = dirty_count / max(total_samples, 1)
    trained_at = _parse_datetime(metadata.get("trained_at") or metadata.get("version"))
    age_seconds = None
    if trained_at is not None:
        age_seconds = max(((now or datetime.now(UTC)) - trained_at).total_seconds(), 0.0)
    data_quality_version = quality.get("data_quality_version")
    objective_name = metadata.get("objective_name")
    objective_version = metadata.get("objective_version")
    label_version = metadata.get("label_version")
    training_cost_policy = str(metadata.get("training_cost_policy") or "")
    tail_loss_policy = _safe_dict(metadata.get("tail_loss_policy"))
    tail_loss_scales = _safe_dict(metadata.get("tail_loss_scale_pct"))

    global_blockers: list[dict[str, Any]] = _market_fact_contract_blockers(metadata)
    if objective_name != RETURN_OBJECTIVE_NAME or objective_version != RETURN_OBJECTIVE_VERSION:
        global_blockers.append(
            _reason(
                "artifact_objective_version_mismatch",
                "Artifact does not use the required fee-after return objective.",
                actual=f"{objective_name or 'unknown'}@{objective_version or 'unknown'}",
                required=f"{RETURN_OBJECTIVE_NAME}@{RETURN_OBJECTIVE_VERSION}",
            )
        )
    if label_version != RETURN_LABEL_VERSION:
        global_blockers.append(
            _reason(
                "artifact_return_label_version_mismatch",
                "Artifact does not use the required fee-after return label contract.",
                actual=label_version or "unknown",
                required=RETURN_LABEL_VERSION,
            )
        )
    if training_cost_policy != "per_sample_live_spread_fee_and_funding_complete":
        global_blockers.append(
            _reason(
                "artifact_cost_policy_incomplete",
                "Artifact was not trained from per-sample live spread, fee, and funding costs.",
                actual=training_cost_policy or "missing",
                required="per_sample_live_spread_fee_and_funding_complete",
            )
        )
    for side in ("long", "short"):
        side_policy = _safe_dict(tail_loss_policy.get(side))
        scale = _safe_float(tail_loss_scales.get(side), None)
        required_provenance = {
            "source",
            "observation_window",
            "sample_count",
            "generated_at",
            "strategy_version",
            "fallback_reason",
        }
        if not required_provenance.issubset(side_policy) or scale is None or scale <= 0:
            global_blockers.append(
                _reason(
                    f"{side}_dynamic_tail_policy_incomplete",
                    f"{side} dynamic tail-loss policy metadata is incomplete.",
                    actual={"policy": side_policy, "scale_pct": scale},
                    required="complete empirical policy provenance and positive artifact scale",
                )
            )
    if sample_count <= 0:
        global_blockers.append(
            _reason(
                "training_distribution_missing",
                "Training return distribution is missing.",
                actual=sample_count,
            )
        )
    if test_count <= 0:
        global_blockers.append(
            _reason(
                "holdout_distribution_missing",
                "Holdout return distribution is missing.",
                actual=test_count,
            )
        )
    side_blockers = {side: _side_metric_blockers(metrics, side) for side in ("long", "short")}
    profit_quality_diagnostics = {
        side: _side_profit_quality_diagnostics(metadata, metrics, side)
        for side in ("long", "short")
    }
    side_enabled = {
        side: not bool(blockers) and bool(_safe_dict(influence.get(side)).get("enabled", True))
        for side, blockers in side_blockers.items()
    }
    if data_quality_version != DATA_QUALITY_VERSION:
        global_blockers.append(
            _reason(
                "training_data_version_stale",
                "Model was trained with an older data-quality contract.",
                actual=data_quality_version or "unknown",
                required=DATA_QUALITY_VERSION,
            )
        )
    if age_seconds is None:
        global_blockers.append(
            _reason(
                "model_training_timestamp_missing",
                "Model is missing a valid trained_at timestamp.",
            )
        )

    live_enabled_sides = [side for side, enabled in side_enabled.items() if enabled]
    partial_live_influence_allowed = bool(
        not global_blockers and live_enabled_sides and influence.get("enabled")
    )
    blockers = (
        global_blockers
        if partial_live_influence_allowed
        else [
            *global_blockers,
            *side_blockers["long"],
            *side_blockers["short"],
        ]
    )
    maturity_blocked = any(
        item["code"] in {"training_distribution_missing", "holdout_distribution_missing"}
        for item in blockers
    )
    if partial_live_influence_allowed:
        state = "ready" if len(live_enabled_sides) == 2 else "partial_ready"
    elif maturity_blocked:
        state = "learning_only"
    elif blockers:
        state = "degraded"
    elif influence.get("advisory_enabled"):
        state = "shadow_ready"
    else:
        state = "learning_only"

    return {
        "state": state,
        "allow_live_position_influence": partial_live_influence_allowed,
        "live_enabled_sides": live_enabled_sides,
        "side_blocking_reasons": side_blockers,
        "blocking_reasons": blockers,
        "profit_quality_diagnostics": profit_quality_diagnostics,
        "next_training_conditions": {
            "trigger": "new_authoritative_cost_complete_sample_or_data_contract_change",
        },
        "thresholds": {
            "min_top_return_lcb_pct": 0.0,
            "min_top_profit_factor": 1.0,
            "threshold_policy": "profitability_math_boundaries_and_empirical_confidence_intervals",
        },
        "policy_provenance": {
            "source": "artifact_holdout_fee_after_return_distribution",
            "observation_window": "artifact_train_and_holdout_windows",
            "sample_count": sample_count,
            "test_sample_count": test_count,
            "generated_at": (now or datetime.now(UTC)).isoformat(),
            "strategy_version": "2026-07-12.ml-readiness-return-lcb.v1",
            "fallback_reason": "" if sample_count > 0 and test_count > 0 else "distribution_missing",
        },
        "metrics": {
            "sample_count": sample_count,
            "test_count": test_count,
            "quarantined_sample_count": excluded_count,
            "downweighted_sample_count": downweighted_count,
            "benign_downweighted_sample_count": benign_downweighted_count,
            "contamination_downweighted_sample_count": contamination_downweighted_count,
            "dirty_sample_ratio": round(dirty_ratio, 4),
            "long_auc": _safe_float(metrics.get("long_auc"), None),
            "short_auc": _safe_float(metrics.get("short_auc"), None),
            "long_pr_auc": _safe_float(metrics.get("long_pr_auc"), None),
            "short_pr_auc": _safe_float(metrics.get("short_pr_auc"), None),
            "top_long_avg_return_pct": _safe_float(metrics.get("top_long_avg_return_pct"), None),
            "top_long_return_lcb_pct": _safe_float(
                metrics.get("top_long_return_lcb_pct"), None
            ),
            "top_long_profit_factor": _safe_float(
                metrics.get("top_long_profit_factor"), None
            ),
            "bottom_long_avg_return_pct": _safe_float(
                metrics.get("bottom_long_avg_return_pct"), None
            ),
            "top_long_bottom_return_spread_pct": profit_quality_diagnostics["long"].get(
                "top_bottom_return_spread_pct"
            ),
            "top_long_tail_loss_rate": _safe_float(
                metrics.get("top_long_tail_loss_rate"), None
            ),
            "bottom_long_tail_loss_rate": _safe_float(
                metrics.get("bottom_long_tail_loss_rate"), None
            ),
            "top_short_avg_return_pct": _safe_float(metrics.get("top_short_avg_return_pct"), None),
            "top_short_return_lcb_pct": _safe_float(
                metrics.get("top_short_return_lcb_pct"), None
            ),
            "top_short_profit_factor": _safe_float(
                metrics.get("top_short_profit_factor"), None
            ),
            "bottom_short_avg_return_pct": _safe_float(
                metrics.get("bottom_short_avg_return_pct"), None
            ),
            "top_short_bottom_return_spread_pct": profit_quality_diagnostics["short"].get(
                "top_bottom_return_spread_pct"
            ),
            "top_short_tail_loss_rate": _safe_float(
                metrics.get("top_short_tail_loss_rate"), None
            ),
            "bottom_short_tail_loss_rate": _safe_float(
                metrics.get("bottom_short_tail_loss_rate"), None
            ),
            "trained_at": trained_at.isoformat() if trained_at else None,
            "model_age_seconds": None if age_seconds is None else round(age_seconds, 1),
            "training_data_version": data_quality_version,
            "required_training_data_version": DATA_QUALITY_VERSION,
            "objective_name": objective_name,
            "objective_version": objective_version,
            "required_objective_name": RETURN_OBJECTIVE_NAME,
            "required_objective_version": RETURN_OBJECTIVE_VERSION,
            "label_version": label_version,
            "required_label_version": RETURN_LABEL_VERSION,
        },
    }


def disabled_ml_readiness(reason_code: str, message: str) -> dict[str, Any]:
    return {
        "state": "disabled",
        "allow_live_position_influence": False,
        "blocking_reasons": [_reason(reason_code, message)],
        "next_training_conditions": {
            "trigger": "new_authoritative_cost_complete_sample_or_data_contract_change",
        },
        "thresholds": {
            "min_top_return_lcb_pct": 0.0,
            "min_top_profit_factor": 1.0,
            "threshold_policy": "profitability_math_boundaries_and_empirical_confidence_intervals",
        },
        "policy_provenance": {
            "source": "artifact_holdout_fee_after_return_distribution",
            "observation_window": "artifact_train_and_holdout_windows",
            "sample_count": 0,
            "test_sample_count": 0,
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": "2026-07-12.ml-readiness-return-lcb.v1",
            "fallback_reason": reason_code,
        },
        "metrics": {
            "sample_count": 0,
            "test_count": 0,
            "quarantined_sample_count": 0,
            "downweighted_sample_count": 0,
            "dirty_sample_ratio": 0.0,
            "training_data_version": None,
            "required_training_data_version": DATA_QUALITY_VERSION,
        },
    }
