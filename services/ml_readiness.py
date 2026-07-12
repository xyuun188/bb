from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from services.return_objective import (
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
)
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.training_data_quality import DATA_QUALITY_VERSION

_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training


ML_READINESS_MIN_SAMPLE_COUNT = _PARAMS.influence_min_sample_count
ML_READINESS_MIN_TEST_COUNT = _PARAMS.influence_min_test_count
ML_READINESS_MAX_DIRTY_SAMPLE_RATIO = _PARAMS.readiness_max_dirty_sample_ratio
ML_READINESS_MAX_MODEL_AGE_SECONDS = _PARAMS.readiness_max_model_age_seconds
ML_READINESS_MIN_TOP_RETURN_PCT = _PARAMS.positive_net_return_threshold_pct


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


def _side_metric_blockers(metrics: dict[str, Any], side: str) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    top_return = _safe_float(metrics.get(f"top_{side}_avg_return_pct"), 0.0) or 0.0
    bottom_return = _safe_float(metrics.get(f"bottom_{side}_avg_return_pct"), 0.0) or 0.0
    top_return_lcb = _safe_float(metrics.get(f"top_{side}_return_lcb_pct"), None)
    top_profit_factor = _safe_float(metrics.get(f"top_{side}_profit_factor"), None)
    top_tail_loss = _safe_float(metrics.get(f"top_{side}_tail_loss_rate"), None)
    bottom_tail_loss = _safe_float(metrics.get(f"bottom_{side}_tail_loss_rate"), None)
    if top_return <= ML_READINESS_MIN_TOP_RETURN_PCT:
        blockers.append(
            _reason(
                f"{side}_top_return_below_threshold",
                f"{side} top-score bucket return is not strong enough.",
                actual=round(top_return, 4),
                required=ML_READINESS_MIN_TOP_RETURN_PCT,
            )
        )
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
    if top_return is not None and top_return <= ML_READINESS_MIN_TOP_RETURN_PCT:
        diagnosis.append("top_score_bucket_not_fee_after_profitable")
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

    global_blockers: list[dict[str, Any]] = []
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
    if sample_count < ML_READINESS_MIN_SAMPLE_COUNT:
        global_blockers.append(
            _reason(
                "sample_count_below_threshold",
                "Training sample count is below the influence threshold.",
                actual=sample_count,
                required=ML_READINESS_MIN_SAMPLE_COUNT,
            )
        )
    if test_count < ML_READINESS_MIN_TEST_COUNT:
        global_blockers.append(
            _reason(
                "test_count_below_threshold",
                "Holdout test sample count is below the influence threshold.",
                actual=test_count,
                required=ML_READINESS_MIN_TEST_COUNT,
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
    if dirty_ratio > ML_READINESS_MAX_DIRTY_SAMPLE_RATIO:
        global_blockers.append(
            _reason(
                "dirty_sample_ratio_high",
                "Excluded/downweighted sample ratio is too high for live influence.",
                actual=round(dirty_ratio, 4),
                required=ML_READINESS_MAX_DIRTY_SAMPLE_RATIO,
            )
        )
    if data_quality_version != DATA_QUALITY_VERSION:
        global_blockers.append(
            _reason(
                "training_data_version_stale",
                "Model was trained with an older data-quality contract.",
                actual=data_quality_version or "unknown",
                required=DATA_QUALITY_VERSION,
            )
        )
    if age_seconds is None or age_seconds > ML_READINESS_MAX_MODEL_AGE_SECONDS:
        global_blockers.append(
            _reason(
                "model_stale",
                "Model is too old or missing a valid trained_at timestamp.",
                actual=None if age_seconds is None else round(age_seconds, 1),
                required=ML_READINESS_MAX_MODEL_AGE_SECONDS,
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
        item["code"]
        in {
            "sample_count_below_threshold",
            "test_count_below_threshold",
        }
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

    min_new_samples = (
        _PARAMS.auto_train_min_new_samples
        if state == "ready"
        else _PARAMS.auto_train_learning_only_min_new_samples
    )
    min_interval_seconds = (
        _PARAMS.auto_train_min_interval_seconds
        if state == "ready"
        else _PARAMS.auto_train_learning_only_interval_seconds
    )
    return {
        "state": state,
        "allow_live_position_influence": partial_live_influence_allowed,
        "live_enabled_sides": live_enabled_sides,
        "side_blocking_reasons": side_blockers,
        "blocking_reasons": blockers,
        "profit_quality_diagnostics": profit_quality_diagnostics,
        "next_training_conditions": {
            "min_interval_seconds": min_interval_seconds,
            "min_new_samples": min_new_samples,
            "min_training_samples": _PARAMS.min_training_samples,
            "min_influence_samples": ML_READINESS_MIN_SAMPLE_COUNT,
            "min_test_samples": ML_READINESS_MIN_TEST_COUNT,
        },
        "thresholds": {
            "min_top_return_pct": ML_READINESS_MIN_TOP_RETURN_PCT,
            "min_top_return_lcb_pct": 0.0,
            "min_top_profit_factor": 1.0,
            "max_dirty_sample_ratio": ML_READINESS_MAX_DIRTY_SAMPLE_RATIO,
            "max_model_age_seconds": ML_READINESS_MAX_MODEL_AGE_SECONDS,
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
            "min_interval_seconds": _PARAMS.auto_train_learning_only_interval_seconds,
            "min_new_samples": _PARAMS.auto_train_learning_only_min_new_samples,
            "min_training_samples": _PARAMS.min_training_samples,
            "min_influence_samples": ML_READINESS_MIN_SAMPLE_COUNT,
            "min_test_samples": ML_READINESS_MIN_TEST_COUNT,
        },
        "thresholds": {
            "min_top_return_pct": ML_READINESS_MIN_TOP_RETURN_PCT,
            "min_top_return_lcb_pct": 0.0,
            "min_top_profit_factor": 1.0,
            "max_dirty_sample_ratio": ML_READINESS_MAX_DIRTY_SAMPLE_RATIO,
            "max_model_age_seconds": ML_READINESS_MAX_MODEL_AGE_SECONDS,
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
