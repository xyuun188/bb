"""Leakage-controlled historical replay for trained paper strategies."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from math import isfinite
from threading import Lock
from typing import Any

STRATEGY_HISTORICAL_REPLAY_VERSION = "2026-07-21.model-selected-shadow-replay.v1"

ModelPredictor = Callable[..., dict[str, Any]]

_PREDICTION_CACHE_MAX_SIZE = 4_096
_prediction_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_prediction_cache_lock = Lock()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _int(value: Any) -> int:
    try:
        return max(int(float(value)), 0)
    except (TypeError, ValueError):
        return 0


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _timestamp_text(value: Any) -> str:
    parsed = _timestamp(value)
    return parsed.isoformat() if parsed is not None else ""


def _group_key(row: dict[str, Any]) -> str:
    decision_id = _int(row.get("decision_id"))
    return f"decision:{decision_id}" if decision_id else f"shadow:{_int(row.get('source_id'))}"


def _group_bounds(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)
    bounds: dict[str, dict[str, Any]] = {}
    for group, group_rows in grouped.items():
        labels = [
            parsed
            for row in group_rows
            if (parsed := _timestamp(row.get("label_timestamp") or row.get("timestamp")))
            is not None
        ]
        decisions = [
            parsed
            for row in group_rows
            if (parsed := _timestamp(row.get("decision_timestamp") or row.get("created_at")))
            is not None
        ]
        if not decisions:
            decisions = [
                label - timedelta(minutes=max(_int(row.get("horizon_minutes")), 1))
                for row, label in zip(group_rows, labels, strict=False)
            ]
        if not labels or not decisions:
            continue
        bounds[group] = {
            "rows": group_rows,
            "decision_start": min(decisions),
            "decision_end": max(decisions),
            "label_start": min(labels),
            "label_end": max(labels),
        }
    return bounds


def _ordered_groups(bounds: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        bounds,
        key=lambda group: (
            bounds[group]["decision_start"],
            bounds[group]["decision_end"],
            group,
        ),
    )


def _expand_integer_ranges(values: Any) -> set[int]:
    result: set[int] = set()
    for value in _list(values):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        start, end = _int(value[0]), _int(value[1])
        if start <= 0 or end < start:
            continue
        result.update(range(start, end + 1))
    return result


def _predictor_identity(predictor: ModelPredictor) -> str:
    owner = getattr(predictor, "__self__", None)
    target = type(owner) if owner is not None else predictor
    return f"{getattr(target, '__module__', '')}:{getattr(target, '__qualname__', '')}"


def _compact_cached_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for value in _list(prediction.get("predictions")):
        row = _dict(value)
        rows.append(
            {
                "horizon_minutes": row.get("horizon_minutes"),
                "best_side": row.get("best_side"),
                "actual_trade_calibration_ready": row.get(
                    "actual_trade_calibration_ready"
                ),
                "return_distribution_contract": row.get(
                    "return_distribution_contract"
                ),
                "counterfactual_execution_cost_distribution": row.get(
                    "counterfactual_execution_cost_distribution"
                ),
            }
        )
    return {
        "available": prediction.get("available"),
        "model_version": prediction.get("model_version"),
        "predictions": rows,
    }


def _prediction_cache_context(
    predictor: ModelPredictor,
    *,
    model_version: str,
    observation: dict[str, Any],
    horizon_minutes: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    source_id = _int(observation.get("source_id"))
    features = dict(_dict(observation.get("feature_snapshot")))
    features.setdefault("symbol", str(observation.get("symbol") or ""))
    feature_fingerprint = hashlib.sha256(
        json.dumps(
            features,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return (
        (
            _predictor_identity(predictor),
            model_version,
            source_id,
            horizon_minutes,
            feature_fingerprint,
        ),
        features,
    )


def _cache_prediction(cache_key: tuple[Any, ...], prediction: dict[str, Any]) -> None:
    with _prediction_cache_lock:
        _prediction_cache[cache_key] = _compact_cached_prediction(prediction)
        _prediction_cache.move_to_end(cache_key)
        while len(_prediction_cache) > _PREDICTION_CACHE_MAX_SIZE:
            _prediction_cache.popitem(last=False)


def _prime_prediction_cache(
    predictor: ModelPredictor,
    *,
    model_version: str,
    observations: list[dict[str, Any]],
    horizon_minutes: int,
) -> None:
    owner = getattr(predictor, "__self__", None)
    batch_predictor = getattr(owner, "predict_strategy_replay_batch", None)
    if not callable(batch_predictor):
        return
    missing: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for observation in observations:
        cache_key, features = _prediction_cache_context(
            predictor,
            model_version=model_version,
            observation=observation,
            horizon_minutes=horizon_minutes,
        )
        with _prediction_cache_lock:
            cached = cache_key in _prediction_cache
        if not cached:
            missing.append((cache_key, features))
    if not missing:
        return
    predictions = batch_predictor(
        [features for _cache_key, features in missing],
        horizon_minutes=horizon_minutes,
    )
    if not isinstance(predictions, list) or len(predictions) != len(missing):
        raise ValueError("strategy replay batch prediction count mismatch")
    for (cache_key, _features), prediction in zip(missing, predictions, strict=True):
        _cache_prediction(cache_key, _dict(prediction))


def _cached_prediction(
    predictor: ModelPredictor,
    *,
    model_version: str,
    observation: dict[str, Any],
    horizon_minutes: int,
) -> dict[str, Any]:
    cache_key, features = _prediction_cache_context(
        predictor,
        model_version=model_version,
        observation=observation,
        horizon_minutes=horizon_minutes,
    )
    with _prediction_cache_lock:
        cached = _prediction_cache.get(cache_key)
        if cached is not None:
            _prediction_cache.move_to_end(cache_key)
            return dict(cached)
    prediction = predictor(features, horizons=(horizon_minutes,))
    result = _compact_cached_prediction(
        dict(prediction) if isinstance(prediction, dict) else {}
    )
    _cache_prediction(cache_key, result)
    return dict(result)


def _compact_prediction_fingerprint(
    prediction: dict[str, Any],
    primary: dict[str, Any],
) -> str:
    payload = {
        "model_version": prediction.get("model_version"),
        "horizon_minutes": primary.get("horizon_minutes"),
        "best_side": primary.get("best_side"),
        "return_distribution_contract": primary.get("return_distribution_contract"),
        "counterfactual_execution_cost_distribution": primary.get(
            "counterfactual_execution_cost_distribution"
        ),
        "actual_trade_calibration_ready": primary.get(
            "actual_trade_calibration_ready"
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _replay_entry(
    predictor: ModelPredictor,
    *,
    blueprint: dict[str, Any],
    observation: dict[str, Any],
    partition: str,
    horizon_minutes: int,
) -> tuple[dict[str, Any] | None, str]:
    model_version = str(blueprint.get("model_version") or "")
    try:
        prediction = _cached_prediction(
            predictor,
            model_version=model_version,
            observation=observation,
            horizon_minutes=horizon_minutes,
        )
    except Exception:
        return None, "model_replay_prediction_failed"
    if prediction.get("available") is not True:
        return None, "model_replay_prediction_unavailable"
    if str(prediction.get("model_version") or "") != model_version:
        return None, "model_replay_version_mismatch"
    primary = next(
        (
            _dict(row)
            for row in _list(prediction.get("predictions"))
            if _int(_dict(row).get("horizon_minutes")) == horizon_minutes
        ),
        {},
    )
    if not primary:
        return None, "model_replay_horizon_missing"
    side = str(primary.get("best_side") or "").lower()
    eligible_sides = {
        str(value).lower() for value in _list(blueprint.get("eligible_sides"))
    }
    if side not in {"long", "short"} or side not in eligible_sides:
        return None, "model_replay_side_not_governed"
    return_contract = _dict(
        _dict(primary.get("return_distribution_contract")).get(side)
    )
    if return_contract.get("production_eligible") is not True:
        return None, "model_replay_return_distribution_incomplete"
    cost_distribution = _dict(
        _dict(primary.get("counterfactual_execution_cost_distribution")).get(side)
    )
    if cost_distribution.get("distribution_ready") is not True:
        return None, "model_replay_cost_distribution_incomplete"
    if primary.get("actual_trade_calibration_ready") is not True:
        return None, "model_replay_actual_trade_calibration_incomplete"
    expected = _float(return_contract.get("objective_expected_return_pct"))
    lower = _float(return_contract.get("lower_quantile_return_pct"))
    execution_cost = _float(observation.get("execution_cost_pct"))
    funding_return = _float(observation.get(f"{side}_funding_return_pct"))
    realized = _float(observation.get(f"{side}_net_return_after_cost_pct"))
    if None in {expected, lower, execution_cost, funding_return, realized}:
        return None, "model_replay_fee_after_values_incomplete"
    expected_after_cost = float(expected) - float(execution_cost) + float(funding_return)
    lower_after_cost = float(lower) - float(execution_cost) + float(funding_return)
    if expected_after_cost <= 0:
        return None, "model_replay_expected_return_not_positive_after_cost"
    if lower_after_cost <= 0:
        return None, "model_replay_return_lcb_not_positive_after_cost"
    return (
        {
            "source": "trained_model_historical_shadow_replay",
            "source_id": _int(observation.get("source_id")),
            "source_row_id": _int(observation.get("source_id")),
            "decision_id": _int(observation.get("decision_id")) or None,
            "symbol": str(observation.get("symbol") or "").upper(),
            "side": side,
            "market_regime": str(observation.get("market_regime") or "").lower(),
            "horizon_minutes": horizon_minutes,
            "net_return_after_cost_pct": round(float(realized), 8),
            "gross_return_pct": observation.get(f"{side}_gross_return_pct"),
            "execution_cost_pct": round(float(execution_cost), 8),
            "funding_return_pct": round(float(funding_return), 8),
            "model_expected_return_after_cost_pct": round(expected_after_cost, 8),
            "model_return_lcb_after_cost_pct": round(lower_after_cost, 8),
            "timestamp": _timestamp_text(
                observation.get("label_timestamp") or observation.get("timestamp")
            ),
            "strategy_replay_partition": partition,
            "strategy_replay_version": STRATEGY_HISTORICAL_REPLAY_VERSION,
            "model_version": model_version,
            "prediction_fingerprint": _compact_prediction_fingerprint(
                prediction,
                primary,
            ),
        },
        "selected",
    )


def _partition_observations(
    blueprint: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    trained_at = _timestamp(blueprint.get("trained_at"))
    expected_test_count = _int(
        _dict(blueprint.get("training_evidence")).get("holdout_sample_count")
    )
    eligible = [row for row in observations if row.get("training_eligible") is True]
    if trained_at is None or expected_test_count <= 0:
        return {
            "status": "artifact_holdout_contract_missing",
            "development": [],
            "exam": [],
            "diagnostics": {
                "eligible_observation_count": len(eligible),
                "expected_holdout_sample_count": expected_test_count,
            },
        }
    historical = [
        row
        for row in eligible
        if (
            completed := _timestamp(
                row.get("completed_at") or row.get("label_timestamp") or row.get("created_at")
            )
        )
        is not None
        and completed <= trained_at
    ]
    post_training = [
        row
        for row in eligible
        if (
            completed := _timestamp(
                row.get("completed_at") or row.get("label_timestamp") or row.get("created_at")
            )
        )
        is not None
        and completed > trained_at
    ]
    historical_bounds = _group_bounds(historical)
    historical_groups = _ordered_groups(historical_bounds)
    holdout_groups: list[str] = []
    holdout_identity_method = "artifact_holdout_source_ids"
    holdout_contract = _dict(
        _dict(blueprint.get("training_evidence")).get("strategy_replay_holdout")
    )
    explicit_holdout_ids = _expand_integer_ranges(
        holdout_contract.get("shadow_source_id_ranges")
    )
    if explicit_holdout_ids:
        explicit_rows = [
            row for row in historical if _int(row.get("source_id")) in explicit_holdout_ids
        ]
        explicit_bounds = _group_bounds(explicit_rows)
        if (
            len(explicit_rows) == expected_test_count
            and len(explicit_holdout_ids) == expected_test_count
        ):
            historical_bounds = explicit_bounds
            holdout_groups = _ordered_groups(explicit_bounds)
    else:
        holdout_identity_method = "legacy_artifact_holdout_count_reconstruction"
        for start in range(len(historical_groups)):
            suffix = historical_groups[start:]
            suffix_count = sum(
                len(historical_bounds[group]["rows"]) for group in suffix
            )
            if suffix_count == expected_test_count:
                holdout_groups = suffix
                break
            if suffix_count < expected_test_count:
                break
    if not holdout_groups:
        return {
            "status": "artifact_holdout_rows_not_reconstructable",
            "development": [],
            "exam": [],
            "diagnostics": {
                "eligible_observation_count": len(eligible),
                "historical_observation_count": len(historical),
                "post_training_observation_count": len(post_training),
                "expected_holdout_sample_count": expected_test_count,
                "holdout_identity_method": holdout_identity_method,
            },
        }
    split_at = max(len(holdout_groups) // 2, 1)
    raw_development_groups = holdout_groups[:split_at]
    exam_groups = holdout_groups[split_at:]
    if not exam_groups:
        return {
            "status": "strategy_exam_partition_unavailable",
            "development": [],
            "exam": [],
            "diagnostics": {
                "artifact_holdout_group_count": len(holdout_groups),
                "expected_holdout_sample_count": expected_test_count,
            },
        }
    exam_decision_start = min(
        historical_bounds[group]["decision_start"] for group in exam_groups
    )
    development_groups = [
        group
        for group in raw_development_groups
        if historical_bounds[group]["label_end"] < exam_decision_start
    ]
    development = [
        row
        for group in development_groups
        for row in historical_bounds[group]["rows"]
    ]
    historical_exam = [
        row for group in exam_groups for row in historical_bounds[group]["rows"]
    ]
    post_bounds = _group_bounds(post_training)
    exam = [*historical_exam, *post_training]
    development_ids = {_int(row.get("source_id")) for row in development}
    exam_ids = {_int(row.get("source_id")) for row in exam}
    overlap = development_ids & exam_ids
    status = "complete" if development and exam and not overlap else "partition_incomplete"
    return {
        "status": status,
        "development": development,
        "exam": exam,
        "diagnostics": {
            "eligible_observation_count": len(eligible),
            "historical_observation_count": len(historical),
            "post_training_observation_count": len(post_training),
            "expected_holdout_sample_count": expected_test_count,
            "holdout_identity_method": holdout_identity_method,
            "artifact_holdout_group_count": len(holdout_groups),
            "strategy_development_group_count": len(development_groups),
            "strategy_exam_historical_group_count": len(exam_groups),
            "strategy_exam_post_training_group_count": len(post_bounds),
            "purged_development_group_count": len(raw_development_groups)
            - len(development_groups),
            "development_exam_overlap_count": len(overlap),
            "chronological_partition_disjoint": not overlap,
            "partition_policy": (
                "artifact_holdout_suffix_then_purged_chronological_strategy_split"
            ),
        },
    }


def build_strategy_historical_replay(
    *,
    blueprint: dict[str, Any] | None,
    observations: list[dict[str, Any]],
    predictor: ModelPredictor | None,
) -> dict[str, Any]:
    """Replay the exact trained model on immutable shadow snapshots."""

    strategy = _dict(blueprint)
    base = {
        "version": STRATEGY_HISTORICAL_REPLAY_VERSION,
        "strategy_id": strategy.get("strategy_id"),
        "model_version": strategy.get("model_version"),
        "development_samples": [],
        "exam_samples": [],
        "selected_entry_count": 0,
        "excluded_reason_counts": {},
        "can_authorize_live": False,
    }
    if (
        strategy.get("paper_execution_eligible") is not True
        or strategy.get("execution_scope") != "paper_only"
        or strategy.get("live_execution_permission") is not False
    ):
        return {**base, "status": "trained_paper_strategy_unavailable"}
    if predictor is None:
        return {**base, "status": "model_replay_predictor_unavailable"}
    horizon_minutes = _int(
        _dict(strategy.get("exit_policy")).get("historical_replay_horizon_minutes")
    )
    if horizon_minutes <= 0:
        return {**base, "status": "historical_replay_horizon_missing"}
    partitions = _partition_observations(strategy, observations)
    if partitions["status"] != "complete":
        return {
            **base,
            "status": partitions["status"],
            "horizon_minutes": horizon_minutes,
            "partition": partitions["diagnostics"],
        }
    excluded: Counter[str] = Counter()
    replay_observations: dict[str, list[dict[str, Any]]] = {
        "strategy_development": [],
        "strategy_exam": [],
    }
    for partition_name, source_key in (
        ("strategy_development", "development"),
        ("strategy_exam", "exam"),
    ):
        rows_by_group: dict[str, list[dict[str, Any]]] = {}
        for row in partitions[source_key]:
            rows_by_group.setdefault(_group_key(row), []).append(row)
        for group_rows in rows_by_group.values():
            observation = next(
                (
                    row
                    for row in group_rows
                    if _int(row.get("horizon_minutes")) == horizon_minutes
                ),
                None,
            )
            if observation is None:
                excluded["historical_replay_horizon_row_missing"] += 1
                continue
            replay_observations[partition_name].append(observation)
    try:
        _prime_prediction_cache(
            predictor,
            model_version=str(strategy.get("model_version") or ""),
            observations=[
                row for rows in replay_observations.values() for row in rows
            ],
            horizon_minutes=horizon_minutes,
        )
    except Exception:
        excluded["model_replay_batch_prediction_failed"] += 1
    replayed: dict[str, list[dict[str, Any]]] = {
        "strategy_development": [],
        "strategy_exam": [],
    }
    replayed_source_ids: set[int] = set()
    for partition_name, observations_for_partition in replay_observations.items():
        for observation in observations_for_partition:
            source_id = _int(observation.get("source_id"))
            if source_id in replayed_source_ids:
                excluded["historical_replay_duplicate_source"] += 1
                continue
            replayed_source_ids.add(source_id)
            entry, reason = _replay_entry(
                predictor,
                blueprint=strategy,
                observation=observation,
                partition=partition_name,
                horizon_minutes=horizon_minutes,
            )
            if entry is None:
                excluded[reason] += 1
                continue
            replayed[partition_name].append(entry)
    development_ids = {row["source_id"] for row in replayed["strategy_development"]}
    exam_ids = {row["source_id"] for row in replayed["strategy_exam"]}
    selected_overlap = development_ids & exam_ids
    if selected_overlap:
        return {
            **base,
            "status": "selected_replay_partition_overlap",
            "horizon_minutes": horizon_minutes,
            "partition": {
                **partitions["diagnostics"],
                "selected_overlap_count": len(selected_overlap),
            },
            "excluded_reason_counts": dict(excluded),
        }
    development = replayed["strategy_development"]
    exam = replayed["strategy_exam"]
    return {
        **base,
        "status": "complete",
        "horizon_minutes": horizon_minutes,
        "validation_method": "exact_current_model_on_immutable_shadow_snapshot",
        "partition": {
            **partitions["diagnostics"],
            "selected_overlap_count": 0,
        },
        "development_samples": development,
        "exam_samples": exam,
        "selected_entry_count": len(development) + len(exam),
        "development_selected_entry_count": len(development),
        "exam_selected_entry_count": len(exam),
        "excluded_reason_counts": dict(excluded),
    }
