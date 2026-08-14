"""Dataset snapshots and immutable artifact bundles for BB backtests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.experiment_contracts import (
    ExperimentContractError,
    build_dataset_manifest,
    canonical_json_bytes,
    verify_experiment_result,
    verify_experiment_spec,
)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
EXPERIMENT_BUNDLE_VERSION = "bb.experiment-bundle.v1"


def normalize_ohlcv_dataframe(value: pd.DataFrame) -> pd.DataFrame:
    """Return chronological, UTC-indexed OHLCV data suitable for replay."""

    if not isinstance(value, pd.DataFrame) or value.empty:
        raise ExperimentContractError("OHLCV dataset must contain at least one row")
    frame = value.copy()
    if "timestamp" in frame.columns:
        frame = frame.set_index("timestamp")
    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ExperimentContractError("OHLCV dataset is missing columns: " + ", ".join(missing))
    try:
        index = pd.to_datetime(frame.index, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError("OHLCV timestamps are invalid") from exc
    frame = frame.loc[:, list(OHLCV_COLUMNS)]
    frame.index = index
    frame.index.name = "timestamp"
    frame = frame.sort_index(kind="stable")
    if frame.index.has_duplicates:
        raise ExperimentContractError("OHLCV timestamps must be unique")
    for column in OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    values = frame.loc[:, list(OHLCV_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ExperimentContractError("OHLCV dataset contains missing or non-finite values")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ExperimentContractError("OHLC prices must be positive")
    if (frame["volume"] < 0).any():
        raise ExperimentContractError("OHLCV volume must be non-negative")
    if (
        (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any()
        or (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any()
    ):
        raise ExperimentContractError("OHLCV high/low bounds are inconsistent")
    return frame


def ohlcv_snapshot_bytes(frame: pd.DataFrame) -> bytes:
    normalized = normalize_ohlcv_dataframe(frame)
    text = normalized.to_csv(
        index=True,
        date_format="%Y-%m-%dT%H:%M:%S.%fZ",
        float_format="%.12g",
        lineterminator="\n",
    )
    return text.encode("utf-8")


def build_ohlcv_dataset_manifest(
    frame: pd.DataFrame,
    *,
    source: str,
    symbol: str,
    timeframe: str,
) -> tuple[pd.DataFrame, bytes, dict[str, Any]]:
    normalized = normalize_ohlcv_dataframe(frame)
    snapshot = ohlcv_snapshot_bytes(normalized)
    manifest = build_dataset_manifest(
        source=source,
        symbols=[symbol],
        timeframe=timeframe,
        started_at=normalized.index[0].to_pydatetime(),
        ended_at=normalized.index[-1].to_pydatetime(),
        timezone="UTC",
        row_count=len(normalized),
        data_sha256=hashlib.sha256(snapshot).hexdigest(),
        columns=["timestamp", *OHLCV_COLUMNS],
        quality={
            "chronological": True,
            "duplicate_timestamp_count": 0,
            "missing_value_count": 0,
            "ohlc_bounds_valid": True,
        },
    )
    return normalized, snapshot, manifest


def write_experiment_bundle(
    root: Path,
    *,
    spec: dict[str, Any],
    result: dict[str, Any],
    dataset_snapshot: bytes,
) -> Path:
    """Write one content-addressed bundle and reject any later mutation."""

    verify_experiment_spec(spec)
    verify_experiment_result(result, spec)
    expected_dataset_hash = str(spec["dataset"]["data_sha256"])
    actual_dataset_hash = hashlib.sha256(dataset_snapshot).hexdigest()
    if actual_dataset_hash != expected_dataset_hash:
        raise ExperimentContractError("dataset snapshot does not match experiment manifest")

    experiment_dir = Path(root) / str(spec["experiment_id"])
    experiment_dir.mkdir(parents=True, exist_ok=True)
    spec_bytes = canonical_json_bytes(spec) + b"\n"
    result_bytes = canonical_json_bytes(result) + b"\n"
    files = {
        "dataset.csv": dataset_snapshot,
        "spec.json": spec_bytes,
        "result.json": result_bytes,
    }
    file_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    bundle_manifest = {
        "bundle_version": EXPERIMENT_BUNDLE_VERSION,
        "immutable": True,
        "experiment_id": spec["experiment_id"],
        "spec_sha256": spec["spec_sha256"],
        "result_sha256": result["result_sha256"],
        "files": file_hashes,
    }
    files["bundle.json"] = canonical_json_bytes(bundle_manifest) + b"\n"
    for name, content in files.items():
        _write_immutable(experiment_dir / name, content)
    return experiment_dir


def load_experiment_bundle(path: Path) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    root = Path(path)
    bundle = _read_json(root / "bundle.json")
    if bundle.get("bundle_version") != EXPERIMENT_BUNDLE_VERSION:
        raise ExperimentContractError("unsupported experiment bundle version")
    if bundle.get("immutable") is not True:
        raise ExperimentContractError("experiment bundle must be immutable")
    files = bundle.get("files")
    if not isinstance(files, dict):
        raise ExperimentContractError("experiment bundle file manifest is missing")
    loaded: dict[str, bytes] = {}
    for name in ("dataset.csv", "spec.json", "result.json"):
        expected_hash = str(files.get(name) or "")
        content = (root / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ExperimentContractError(f"experiment bundle file hash mismatch: {name}")
        loaded[name] = content
    spec = json.loads(loaded["spec.json"])
    result = json.loads(loaded["result.json"])
    verify_experiment_spec(spec)
    verify_experiment_result(result, spec)
    if bundle.get("experiment_id") != spec["experiment_id"]:
        raise ExperimentContractError("bundle experiment_id mismatch")
    if bundle.get("spec_sha256") != spec["spec_sha256"]:
        raise ExperimentContractError("bundle spec_sha256 mismatch")
    if bundle.get("result_sha256") != result["result_sha256"]:
        raise ExperimentContractError("bundle result_sha256 mismatch")
    return spec, result, loaded["dataset.csv"]


def dataframe_from_snapshot(snapshot: bytes) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        handle.write(snapshot)
        temporary_name = handle.name
    try:
        frame = pd.read_csv(temporary_name, parse_dates=["timestamp"])
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return normalize_ohlcv_dataframe(frame)


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ExperimentContractError(f"refusing to mutate experiment artifact {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            temporary.replace(path)
        except OSError:
            if not path.exists() or path.read_bytes() != content:
                raise
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentContractError(f"cannot read experiment artifact {path.name}") from exc
    if not isinstance(value, dict):
        raise ExperimentContractError(f"experiment artifact {path.name} must contain an object")
    return value
