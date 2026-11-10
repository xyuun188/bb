"""Immutable, cutoff-bound development inputs for repeatable research."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.symbols import normalize_trading_symbol
from services.ml_development_baselines import _development_input_sha256
from services.ml_signal_service import FEATURE_KEYS, _decision_group_availability
from services.profit_supervision import PROFIT_SUPERVISION_VERSION

SNAPSHOT_VERSION = "2026-11-06.symbol-bound-frozen-development.v2"
REPLAY_SNAPSHOT_VERSION = "2026-11-10.replay-metadata-frozen-development.v3"
REPLAY_METADATA_COLUMNS = ("decision_action", "decision_confidence", "best_action", "market_regime")
SNAPSHOT_FILENAME = "frozen_development_v2.json"
SNAPSHOT_COLUMNS = list(dict.fromkeys([
    "id", "decision_group", "symbol", "decision_timestamp", "label_timestamp",
    "horizon_minutes", *FEATURE_KEYS,
    "long_return_pct", "short_return_pct",
    "long_execution_cost_pct", "short_execution_cost_pct", "sample_weight",
]))


class InsufficientDevelopmentDataError(ValueError):
    """The frozen source window cannot supply usable development observations."""


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _normalize(frame: pd.DataFrame, *, include_replay_metadata: bool = False) -> pd.DataFrame:
    source = frame.copy()
    if "symbol" not in source:
        source["symbol"] = ""
    columns = SNAPSHOT_COLUMNS.copy()
    if include_replay_metadata:
        missing = set(REPLAY_METADATA_COLUMNS) - set(source.columns)
        if missing:
            raise ValueError("replay snapshot requires training metadata: " + ", ".join(sorted(missing)))
        columns.extend(REPLAY_METADATA_COLUMNS)
    result = source[columns].copy()
    string_columns = {"decision_group", "symbol", "decision_timestamp", "label_timestamp"}
    if include_replay_metadata:
        for column in ("decision_action", "best_action", "market_regime"):
            if result[column].isna().any():
                raise ValueError(f"missing replay metadata: {column}")
            result[column] = result[column].astype(str)
            string_columns.add(column)
    for column in ("decision_timestamp", "label_timestamp"):
        result[column] = pd.to_datetime(
            result[column].map(pd.Timestamp), utc=True, errors="raise"
        )
        if result[column].isna().any():
            raise ValueError("snapshot timestamps must be present")
    if (result["label_timestamp"] <= result["decision_timestamp"]).any():
        raise ValueError("snapshot labels must follow decisions")
    result["decision_group"] = result["decision_group"].astype(str)
    if result["decision_group"].isin(("", "None", "nan")).any():
        raise ValueError("snapshot decision group must be present")
    result["symbol"] = result["symbol"].map(
        lambda value: normalize_trading_symbol("" if pd.isna(value) else value)
    )
    for column in columns:
        if column in string_columns:
            continue
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        if not np.isfinite(result[column]).all():
            raise ValueError(f"non-finite snapshot input: {column}")
    for column in ("id", "horizon_minutes"):
        if (result[column] <= 0).any() or (result[column] % 1 != 0).any():
            raise ValueError(f"positive integer snapshot input required: {column}")
        result[column] = result[column].astype("int64")
    if result["id"].duplicated().any() or result.duplicated(
        ["decision_group", "horizon_minutes"]
    ).any():
        raise ValueError("duplicate snapshot observation")
    # Net costs include signed funding: income can exceed fees and slippage.
    if (result["sample_weight"] < 0).any():
        raise ValueError("invalid snapshot sample weights")
    if not (result["sample_weight"] > 0).any():
        raise InsufficientDevelopmentDataError(
            "snapshot needs at least one positive sample weight"
        )
    return result.sort_values([
        "decision_timestamp", "decision_group", "horizon_minutes", "id",
    ]).reset_index(drop=True)


def make_development_snapshot(
    frame: pd.DataFrame, *, cutoff: str, include_replay_metadata: bool = False,
) -> dict[str, Any]:
    boundary = pd.Timestamp(cutoff)
    if boundary.tzinfo is None:
        raise ValueError("development cutoff must have a timezone")
    boundary = boundary.tz_convert("UTC")
    if frame.empty:
        raise InsufficientDevelopmentDataError("no development rows before frozen cutoff")
    groups, bounds = _decision_group_availability(frame)
    allowed = {
        group for group in groups
        if bounds[group]["end"] < boundary
        and bounds[group]["decision_end"] < boundary
    }
    eligible = frame[
        frame["decision_group"].astype(str).isin(allowed)
    ]
    if eligible.empty:
        raise InsufficientDevelopmentDataError("no development rows before frozen cutoff")
    development = _normalize(eligible, include_replay_metadata=include_replay_metadata)
    serializable = development.copy()
    for column in ("decision_timestamp", "label_timestamp"):
        serializable[column] = serializable[column].map(lambda value: value.isoformat())
    payload = {
        "version": REPLAY_SNAPSHOT_VERSION if include_replay_metadata else SNAPSHOT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "development_cutoff": boundary.isoformat(),
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        "cutoff_policy": "all_group_labels_strictly_before_earliest_exam_start",
        "source_frame_row_count": len(frame),
        "row_count": len(development),
        "decision_group_count": development["decision_group"].nunique(),
        "excluded_decision_group_count": len(groups) - len(allowed),
        "development_input_sha256": _development_input_sha256(development),
        "records": serializable.to_dict("records"),
    }
    return {**payload, "snapshot_sha256": _digest(payload)}


def snapshot_frame(
    snapshot: dict[str, Any], *, cutoff: str,
) -> pd.DataFrame:
    payload = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    if snapshot.get("version") not in {SNAPSHOT_VERSION, REPLAY_SNAPSHOT_VERSION} or _digest(payload) != snapshot.get(
        "snapshot_sha256"
    ):
        raise ValueError("development snapshot integrity failure")
    if pd.Timestamp(snapshot["development_cutoff"]) != pd.Timestamp(cutoff):
        raise ValueError("frozen development cutoff cannot change")
    development = _normalize(
        pd.DataFrame(snapshot["records"]),
        include_replay_metadata=snapshot["version"] == REPLAY_SNAPSHOT_VERSION,
    )
    if _development_input_sha256(development) != snapshot["development_input_sha256"]:
        raise ValueError("development frame fingerprint mismatch")
    if (
        development["label_timestamp"].max() >= pd.Timestamp(cutoff)
        or len(development) != snapshot["row_count"]
        or development["decision_group"].nunique() != snapshot["decision_group_count"]
    ):
        raise ValueError("development snapshot boundary or accounting mismatch")
    return development


def load_development_snapshot(path: Path, *, cutoff: str) -> tuple[pd.DataFrame, dict]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    return snapshot_frame(snapshot, cutoff=cutoff), snapshot


def save_development_snapshot(
    snapshot: dict[str, Any], *, path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Publish a complete file atomically; an existing snapshot is never replaced."""

    snapshot_frame(snapshot, cutoff=snapshot["development_cutoff"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(snapshot, handle, allow_nan=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return load_development_snapshot(path, cutoff=snapshot["development_cutoff"])
