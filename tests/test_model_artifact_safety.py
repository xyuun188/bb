from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.model_artifact_safety import dump_trusted_joblib, load_trusted_joblib
from services.ml_signal_service import MLSignalService, _parse_json, _safe_auc


def test_trusted_joblib_round_trip_inside_model_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_path = model_dir / "bundle.joblib"
    bundle = {"metadata": {"sample_count": 3}, "feature_keys": ["rsi_14"]}

    dump_trusted_joblib(bundle, model_path, trusted_root=model_dir)
    loaded = load_trusted_joblib(model_path, trusted_root=model_dir, expected_type=dict)

    assert loaded == bundle


def test_trusted_joblib_rejects_path_outside_model_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    outside_path = tmp_path / "outside.joblib"

    with pytest.raises(ValueError, match="outside trusted directory"):
        dump_trusted_joblib({"metadata": {}}, outside_path, trusted_root=model_dir)


def test_trusted_joblib_rejects_wrong_suffix(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_path = model_dir / "bundle.pkl"

    with pytest.raises(ValueError, match="unsupported model artifact suffix"):
        dump_trusted_joblib({"metadata": {}}, model_path, trusted_root=model_dir)


def test_trusted_joblib_rejects_unexpected_type(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_path = model_dir / "bundle.joblib"
    dump_trusted_joblib(["not", "a", "dict"], model_path, trusted_root=model_dir)

    with pytest.raises(TypeError, match="unexpected model artifact type"):
        load_trusted_joblib(model_path, trusted_root=model_dir, expected_type=dict)


def test_ml_signal_service_does_not_load_joblib_outside_model_dir(tmp_path: Path) -> None:
    outside_path = tmp_path / "outside.joblib"
    dump_trusted_joblib(
        {"metadata": {"sample_count": 999}, "feature_keys": []},
        outside_path,
        trusted_root=tmp_path,
    )
    service = MLSignalService(model_path=outside_path)

    status = service.status()

    assert status["available"] is False
    assert status["status"] == "no_model"
    assert service._bundle is None


def test_ml_signal_parsing_helpers_degrade_for_invalid_data() -> None:
    service = MLSignalService()

    assert _parse_json("{invalid-json") == {}
    assert _parse_json(["not", "json", "text"]) == {}
    assert _safe_auc(pd.Series([1, 1, 1]), np.array([0.1, 0.2, 0.3])) is None
    assert service._parse_datetime("not-a-date") is None
