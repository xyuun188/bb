from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from services.trading_params import DEFAULT_TRADING_PARAMS
from services.training_data_quality import DATA_QUALITY_VERSION

_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training


ML_READINESS_MIN_SAMPLE_COUNT = _PARAMS.influence_min_sample_count
ML_READINESS_MIN_TEST_COUNT = _PARAMS.influence_min_test_count
ML_READINESS_MIN_AUC = _PARAMS.influence_min_auc
ML_READINESS_MIN_PR_AUC = _PARAMS.influence_min_pr_auc
ML_READINESS_MIN_ACCURACY = _PARAMS.influence_min_accuracy
ML_READINESS_MAX_DIRTY_SAMPLE_RATIO = _PARAMS.readiness_max_dirty_sample_ratio
ML_READINESS_MAX_MODEL_AGE_SECONDS = _PARAMS.readiness_max_model_age_seconds
ML_READINESS_MIN_TOP_RETURN_PCT = _PARAMS.win_return_threshold_pct


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
    auc = _safe_float(metrics.get(f"{side}_auc"), 0.0) or 0.0
    pr_auc = _safe_float(metrics.get(f"{side}_pr_auc"), None)
    accuracy = _safe_float(metrics.get(f"{side}_accuracy"), 0.0) or 0.0
    top_return = _safe_float(metrics.get(f"top_{side}_avg_return_pct"), 0.0) or 0.0
    bottom_return = _safe_float(metrics.get(f"bottom_{side}_avg_return_pct"), 0.0) or 0.0
    top_win = _safe_float(metrics.get(f"top_{side}_win_rate"), 0.0) or 0.0
    bottom_win = _safe_float(metrics.get(f"bottom_{side}_win_rate"), 0.0) or 0.0

    if auc < ML_READINESS_MIN_AUC:
        blockers.append(
            _reason(
                f"{side}_auc_below_threshold",
                f"{side} AUC is below the configured threshold.",
                actual=round(auc, 4),
                required=ML_READINESS_MIN_AUC,
            )
        )
    if pr_auc is None:
        blockers.append(
            _reason(
                f"{side}_pr_auc_missing",
                f"{side} PR-AUC is missing; retrain with the current trainer.",
                required=ML_READINESS_MIN_PR_AUC,
            )
        )
    elif pr_auc < ML_READINESS_MIN_PR_AUC:
        blockers.append(
            _reason(
                f"{side}_pr_auc_below_threshold",
                f"{side} PR-AUC is below the configured threshold.",
                actual=round(pr_auc, 4),
                required=ML_READINESS_MIN_PR_AUC,
            )
        )
    if accuracy < ML_READINESS_MIN_ACCURACY:
        blockers.append(
            _reason(
                f"{side}_accuracy_below_threshold",
                f"{side} accuracy is below the configured threshold.",
                actual=round(accuracy, 4),
                required=ML_READINESS_MIN_ACCURACY,
            )
        )
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
    if top_win <= bottom_win:
        blockers.append(
            _reason(
                f"{side}_top_win_not_above_bottom",
                f"{side} top-score win rate is not above the bottom bucket.",
                actual=round(top_win, 4),
                required=round(bottom_win, 4),
            )
        )
    return blockers


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

    blockers: list[dict[str, Any]] = []
    if sample_count < ML_READINESS_MIN_SAMPLE_COUNT:
        blockers.append(
            _reason(
                "sample_count_below_threshold",
                "Training sample count is below the influence threshold.",
                actual=sample_count,
                required=ML_READINESS_MIN_SAMPLE_COUNT,
            )
        )
    if test_count < ML_READINESS_MIN_TEST_COUNT:
        blockers.append(
            _reason(
                "test_count_below_threshold",
                "Holdout test sample count is below the influence threshold.",
                actual=test_count,
                required=ML_READINESS_MIN_TEST_COUNT,
            )
        )
    for side in ("long", "short"):
        blockers.extend(_side_metric_blockers(metrics, side))
    if dirty_ratio > ML_READINESS_MAX_DIRTY_SAMPLE_RATIO:
        blockers.append(
            _reason(
                "dirty_sample_ratio_high",
                "Excluded/downweighted sample ratio is too high for live influence.",
                actual=round(dirty_ratio, 4),
                required=ML_READINESS_MAX_DIRTY_SAMPLE_RATIO,
            )
        )
    if data_quality_version != DATA_QUALITY_VERSION:
        blockers.append(
            _reason(
                "training_data_version_stale",
                "Model was trained with an older data-quality contract.",
                actual=data_quality_version or "unknown",
                required=DATA_QUALITY_VERSION,
            )
        )
    if age_seconds is None or age_seconds > ML_READINESS_MAX_MODEL_AGE_SECONDS:
        blockers.append(
            _reason(
                "model_stale",
                "Model is too old or missing a valid trained_at timestamp.",
                actual=None if age_seconds is None else round(age_seconds, 1),
                required=ML_READINESS_MAX_MODEL_AGE_SECONDS,
            )
        )

    maturity_blocked = any(
        item["code"]
        in {
            "sample_count_below_threshold",
            "test_count_below_threshold",
            "long_pr_auc_missing",
            "short_pr_auc_missing",
        }
        for item in blockers
    )
    if not blockers and influence.get("enabled"):
        state = "ready"
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
        "allow_live_position_influence": state == "ready",
        "blocking_reasons": blockers,
        "next_training_conditions": {
            "min_interval_seconds": min_interval_seconds,
            "min_new_samples": min_new_samples,
            "min_training_samples": _PARAMS.min_training_samples,
            "min_influence_samples": ML_READINESS_MIN_SAMPLE_COUNT,
            "min_test_samples": ML_READINESS_MIN_TEST_COUNT,
        },
        "thresholds": {
            "min_auc": ML_READINESS_MIN_AUC,
            "min_pr_auc": ML_READINESS_MIN_PR_AUC,
            "min_accuracy": ML_READINESS_MIN_ACCURACY,
            "min_top_return_pct": ML_READINESS_MIN_TOP_RETURN_PCT,
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
            "bottom_long_avg_return_pct": _safe_float(
                metrics.get("bottom_long_avg_return_pct"), None
            ),
            "top_long_tail_loss_rate": _safe_float(
                metrics.get("top_long_tail_loss_rate"), None
            ),
            "bottom_long_tail_loss_rate": _safe_float(
                metrics.get("bottom_long_tail_loss_rate"), None
            ),
            "top_short_avg_return_pct": _safe_float(metrics.get("top_short_avg_return_pct"), None),
            "bottom_short_avg_return_pct": _safe_float(
                metrics.get("bottom_short_avg_return_pct"), None
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
            "min_auc": ML_READINESS_MIN_AUC,
            "min_pr_auc": ML_READINESS_MIN_PR_AUC,
            "min_accuracy": ML_READINESS_MIN_ACCURACY,
            "min_top_return_pct": ML_READINESS_MIN_TOP_RETURN_PCT,
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
