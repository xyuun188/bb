"""Offline audits for future-data leakage and recursive indicator bias."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from backtest.reproducibility import normalize_ohlcv_dataframe
from core.experiment_contracts import ExperimentContractError

BIAS_ANALYSIS_VERSION = "bb.backtest-bias-analysis.v1"
OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}


def run_lookahead_analysis(
    frame: pd.DataFrame,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    feature_columns: Sequence[str] | None = None,
    checkpoints: Sequence[int] | None = None,
    atol: float = 1e-12,
    rtol: float = 1e-9,
) -> dict[str, Any]:
    """Recompute features on every historical prefix and compare the last row."""

    data = normalize_ohlcv_dataframe(frame)
    full_features = _feature_frame(feature_builder, data)
    columns = list(feature_columns or [column for column in full_features.columns if column not in OHLCV_COLUMNS])
    positions = list(checkpoints) if checkpoints is not None else list(range(len(data)))
    violations: list[dict[str, Any]] = []
    checked = 0
    for position in positions:
        if position < 0 or position >= len(data):
            raise ExperimentContractError("lookahead checkpoint is outside dataset")
        prefix = data.iloc[: position + 1].copy()
        prefix_features = _feature_frame(feature_builder, prefix)
        timestamp = data.index[position]
        full_row = _row_at(full_features, timestamp)
        prefix_row = _row_at(prefix_features, timestamp)
        for column in columns:
            if column not in full_row.index or column not in prefix_row.index:
                violations.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "feature": column,
                        "reason": "feature_missing_after_truncated_recompute",
                    }
                )
                continue
            if not _same_value(full_row[column], prefix_row[column], atol=atol, rtol=rtol):
                violations.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "feature": column,
                        "reason": "future_data_dependent",
                        "full_value": _json_value(full_row[column]),
                        "truncated_value": _json_value(prefix_row[column]),
                    }
                )
        checked += 1
    return {
        "analysis_version": BIAS_ANALYSIS_VERSION,
        "analysis": "lookahead",
        "status": "pass" if not violations else "fail",
        "checked_rows": checked,
        "feature_columns": columns,
        "violation_count": len(violations),
        "violations": violations,
        "tolerances": {"atol": atol, "rtol": rtol},
    }


def run_recursive_warmup_analysis(
    frame: pd.DataFrame,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    warmup_lengths: Sequence[int] = (20, 50, 100),
    feature_columns: Sequence[str] | None = None,
    comparison_points: int = 1,
    required_warmup_length: int | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Compare the same tail after varying historical warm-up lengths."""

    if comparison_points <= 0:
        raise ExperimentContractError("comparison_points must be positive")
    normalized_warmups = [int(item) for item in warmup_lengths]
    if not normalized_warmups:
        raise ExperimentContractError("warmup_lengths must not be empty")
    data = normalize_ohlcv_dataframe(frame)
    full_features = _feature_frame(feature_builder, data)
    columns = list(feature_columns or [column for column in full_features.columns if column not in OHLCV_COLUMNS])
    rows: list[dict[str, Any]] = []
    for warmup in normalized_warmups:
        if warmup <= 0:
            raise ExperimentContractError("warmup lengths must be positive")
        if warmup + comparison_points > len(data):
            rows.append(
                {
                    "warmup_length": warmup,
                    "status": "insufficient_data",
                    "comparison_points": 0,
                }
            )
            continue
        start = len(data) - warmup - comparison_points
        tail = data.iloc[start:].copy()
        tail_features = _feature_frame(feature_builder, tail)
        max_abs = 0.0
        max_relative = 0.0
        differences = 0
        for timestamp in data.index[-comparison_points:]:
            full_row = _row_at(full_features, timestamp)
            tail_row = _row_at(tail_features, timestamp)
            for column in columns:
                if column not in full_row.index or column not in tail_row.index:
                    differences += 1
                    continue
                if _numeric_pair(full_row[column], tail_row[column]):
                    left = float(full_row[column])
                    right = float(tail_row[column])
                    if math.isfinite(left) and math.isfinite(right):
                        delta = abs(left - right)
                        max_abs = max(max_abs, delta)
                        max_relative = max(max_relative, delta / max(abs(left), 1e-12))
                if not _same_value(full_row[column], tail_row[column], atol=atol, rtol=rtol):
                    differences += 1
        rows.append(
            {
                "warmup_length": warmup,
                "status": "pass" if differences == 0 else "fail",
                "comparison_points": comparison_points,
                "difference_count": differences,
                "max_abs_delta": max_abs,
                "max_relative_delta": max_relative,
            }
        )
    required_warmup = int(required_warmup_length or max(normalized_warmups))
    required_rows = [row for row in rows if int(row["warmup_length"]) >= required_warmup]
    failed = not required_rows or any(row["status"] != "pass" for row in required_rows)
    return {
        "analysis_version": BIAS_ANALYSIS_VERSION,
        "analysis": "recursive_warmup",
        "status": "fail" if failed else "pass",
        "feature_columns": columns,
        "comparison_points": comparison_points,
        "required_warmup_length": required_warmup,
        "warmups": rows,
        "tolerances": {"atol": atol, "rtol": rtol},
    }


def validate_feature_availability(
    records: Sequence[Mapping[str, Any]],
    *,
    decision_time_key: str = "decision_time",
    feature_time_keys: Sequence[str] = ("feature_available_at", "feature_timestamp", "observed_at"),
) -> dict[str, Any]:
    """Ensure every feature snapshot was available no later than its decision."""

    violations: list[dict[str, Any]] = []
    checked = 0
    for index, record in enumerate(records):
        decision_at = _parse_timestamp(record.get(decision_time_key))
        feature_value = next(
            (record.get(key) for key in feature_time_keys if record.get(key) is not None),
            None,
        )
        feature_at = _parse_timestamp(feature_value)
        if decision_at is None or feature_at is None:
            violations.append(
                {
                    "index": index,
                    "reason": "feature_or_decision_timestamp_missing_or_naive",
                }
            )
            continue
        checked += 1
        if feature_at > decision_at:
            violations.append(
                {
                    "index": index,
                    "reason": "feature_available_after_decision",
                    "feature_available_at": feature_at.isoformat(),
                    "decision_time": decision_at.isoformat(),
                }
            )
    return {
        "analysis_version": BIAS_ANALYSIS_VERSION,
        "analysis": "feature_availability",
        "status": "pass" if not violations else "fail",
        "checked_records": checked,
        "violation_count": len(violations),
        "violations": violations,
    }


def _feature_frame(
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    data: pd.DataFrame,
) -> pd.DataFrame:
    try:
        result = feature_builder(data.copy())
    except Exception as exc:
        raise ExperimentContractError("feature builder failed during bias analysis") from exc
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise ExperimentContractError("feature builder must return a non-empty DataFrame")
    if not result.index.is_unique:
        raise ExperimentContractError("feature builder returned duplicate timestamps")
    return result


def _row_at(frame: pd.DataFrame, timestamp: Any) -> pd.Series:
    if timestamp not in frame.index:
        raise ExperimentContractError("feature builder dropped a checked timestamp")
    row = frame.loc[timestamp]
    if isinstance(row, pd.DataFrame):
        raise ExperimentContractError("feature builder returned duplicate checked timestamps")
    return row


def _numeric_pair(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number))


def _same_value(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if _numeric_pair(left, right):
        if pd.isna(left) and pd.isna(right):
            return True
        if pd.isna(left) or pd.isna(right):
            return False
        return bool(np.isclose(float(left), float(right), atol=atol, rtol=rtol))
    if pd.isna(left) and pd.isna(right):
        return True
    return left == right


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = pd.Timestamp(value).to_pydatetime()
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)
