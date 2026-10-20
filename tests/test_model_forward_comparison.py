from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.model_artifact_safety import dump_trusted_joblib
from services.ml_signal_service import MLSignalService
from services.model_artifact_registry import ResolvedModelArtifact
from services.model_forward_comparison import (
    common_forward_observations,
    run_model_forward_comparison,
    summarize_forward_predictions,
)


def _artifact(tmp_path, role="challenger"):
    model_path = tmp_path / role / "model.joblib"
    model_path.parent.mkdir()
    dump_trusted_joblib(
        {"metadata": {"artifact_version": role}},
        model_path,
        trusted_root=model_path.parent,
    )
    return ResolvedModelArtifact(
        model_id="local_ml_profit_quality",
        version=role,
        model_path=model_path,
        metadata_path=model_path.with_name("metadata.json"),
        manifest_path=model_path.with_name("manifest.json"),
        sha256=sha256(model_path.read_bytes()).hexdigest(),
        manifest={},
        pointer_role=role,
        pointer_path=tmp_path / f"{role}.json",
    )


def test_artifact_bound_loader_uses_challenger_and_never_resolves_current(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    registry = SimpleNamespace(
        resolve_challenger=Mock(return_value=artifact),
        resolve_current=Mock(side_effect=AssertionError("must not load current")),
    )
    monkeypatch.setattr(
        "services.ml_signal_service.local_ml_artifact_compatibility_errors",
        lambda *args, **kwargs: [],
    )
    service = MLSignalService(artifact_registry=registry, resolved_artifact=artifact)
    service._ensure_loaded()
    assert service._bundle["metadata"]["artifact_version"] == "challenger"
    registry.resolve_current.assert_not_called()


def test_artifact_bound_loader_rejects_unverified_or_tampered_artifact(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    registry = SimpleNamespace(
        resolve_challenger=Mock(return_value=artifact),
        resolve_current=Mock(),
    )
    with pytest.raises(ValueError, match="verified registry"):
        MLSignalService(
            artifact_registry=registry, resolved_artifact=replace(artifact, sha256="0" * 64)
        )
    service = MLSignalService(artifact_registry=registry, resolved_artifact=artifact)
    with artifact.model_path.open("ab") as handle:
        handle.write(b"tamper")
    service._ensure_loaded()
    assert service._bundle is None
    assert service._load_diagnostic["code"] == "artifact_load_failed"


@pytest.mark.asyncio
async def test_artifact_bound_service_cannot_train_or_rollback(tmp_path):
    artifact = _artifact(tmp_path)
    registry = SimpleNamespace(resolve_challenger=lambda: artifact, resolve_current=Mock())
    service = MLSignalService(artifact_registry=registry, resolved_artifact=artifact)
    with pytest.raises(RuntimeError, match="cannot train"):
        await service.maybe_auto_train(force=True)
    with pytest.raises(RuntimeError, match="cannot change"):
        service.rollback_to_strategy_model("old")


def _observation(source_id=1):
    return {
        "source_id": source_id,
        "decision_id": source_id,
        "symbol": "BTC",
        "decision_timestamp": "2026-10-21T00:01:00+00:00",
        "label_timestamp": "2026-10-21T00:06:00+00:00",
        "horizon_minutes": 5,
        "training_eligible": True,
        "execution_cost_pct": 0.1,
        "long_funding_return_pct": 0.0,
        "short_funding_return_pct": 0.0,
        "long_net_return_after_all_cost_pct": -0.2,
        "short_net_return_after_all_cost_pct": 0.1,
    }


def _prediction(version="challenger", side="long"):
    return {
        "available": True,
        "model_version": version,
        "predictions": [
            {
                "horizon_minutes": 5,
                "best_side": side,
                "return_distribution_contract": {
                    side: {
                        "production_eligible": True,
                        "objective_expected_return_pct": 0.3,
                        "lower_quantile_return_pct": 0.2,
                    }
                },
                "counterfactual_execution_cost_distribution": {side: {"distribution_ready": True}},
            }
        ],
    }


def test_common_window_rejects_old_decisions_late_labels_and_quality_failures():
    row = _observation()
    selected, excluded = common_forward_observations(
        [
            row,
            row,
            {**row, "source_id": 2, "decision_timestamp": "2026-10-20T23:59:00Z"},
            {**row, "source_id": 3, "label_timestamp": "2026-10-21T02:00:00Z"},
            {**row, "source_id": 4, "training_eligible": False},
        ],
        cutoff=datetime(2026, 10, 21, tzinfo=UTC),
        evaluated_at=datetime(2026, 10, 21, 1, tzinfo=UTC),
        horizon_minutes=5,
    )
    assert selected == [row]
    assert excluded == {
        "duplicate_source": 1,
        "not_forward": 1,
        "label_not_mature": 1,
        "quality_ineligible": 1,
    }


def test_forward_report_preserves_losing_signals_and_compares_same_rows():
    rows = [_observation()]
    losing = summarize_forward_predictions(
        rows,
        [_prediction()],
        expected_version="challenger",
        horizon_minutes=5,
    )
    winning = summarize_forward_predictions(
        rows,
        [_prediction("current", "short")],
        expected_version="current",
        horizon_minutes=5,
    )
    assert losing["all_model_selected_directions"]["average_net_return_pct"] == -0.2
    assert winning["all_model_selected_directions"]["average_net_return_pct"] == 0.1
    assert losing["fee_after_return_gate"]["sample_count"] == 1
    assert losing["gate_pass_count"] == 1
    assert losing["observed_prediction_bias_pct"] == pytest.approx(0.4)
    assert losing["prediction_count"] == winning["prediction_count"] == 1


def test_forward_report_refuses_silent_model_fallback():
    with pytest.raises(ValueError, match="version mismatch"):
        summarize_forward_predictions(
            [_observation()],
            [_prediction("current")],
            expected_version="challenger",
            horizon_minutes=5,
        )
    with pytest.raises(ValueError, match="count mismatch"):
        summarize_forward_predictions(
            [_observation()],
            [],
            expected_version="challenger",
            horizon_minutes=5,
        )


@pytest.mark.asyncio
async def test_forward_comparison_uses_same_projected_rows_and_serializable_inventory(monkeypatch):
    import json

    import services.model_forward_comparison as comparison

    registry = SimpleNamespace(
        resolve_current=lambda **kwargs: SimpleNamespace(
            version="current",
            sha256="1" * 64,
            manifest={"trained_at": "2026-10-20T00:00:00Z", "created_at": "2026-10-20T00:00:01Z"},
        ),
        resolve_challenger=lambda **kwargs: SimpleNamespace(
            version="challenger",
            sha256="2" * 64,
            manifest={"trained_at": "2026-10-20T00:00:02Z", "created_at": "2026-10-20T00:00:03Z"},
        ),
    )
    statements = []
    inventory = [
        {
            "execution_mode": "paper",
            "status": "completed",
            "horizon_minutes": 5,
            "sample_count": 1,
            "latest_decision_at": datetime(2026, 10, 20, 0, 1),
        }
    ]
    source = {"id": 1, "training_feature_snapshot": {"price": 100.0}}

    class Session:
        async def execute(self, statement):
            statements.append(statement)
            return SimpleNamespace(mappings=lambda: inventory if len(statements) == 1 else [source])

    @asynccontextmanager
    async def session():
        yield Session()

    class Service:
        def __init__(self, *, resolved_artifact, **kwargs):
            self.version = resolved_artifact.version

        def predict_strategy_replay_batch(self, rows, **kwargs):
            assert rows == [{"price": 100.0, "symbol": "BTC"}]
            return [_prediction(self.version)]

    def observations(rows, **kwargs):
        assert rows[0].training_feature_snapshot == {"price": 100.0}
        return [
            {
                **_observation(),
                "feature_snapshot": {"price": 100.0},
                "decision_timestamp": "2026-10-20T00:01:00Z",
                "label_timestamp": "2026-10-20T00:06:00Z",
            }
        ], {}

    monkeypatch.setattr("services.ml_signal_service.ML_SIGNAL_ARTIFACT_REGISTRY", registry)
    monkeypatch.setattr("services.ml_signal_service.MLSignalService", Service)
    monkeypatch.setattr(comparison, "get_read_session_ctx", session)
    monkeypatch.setattr(comparison, "_build_historical_replay_observations", observations)
    report = await run_model_forward_comparison()
    assert report["status"] == "complete"
    assert report["source_ids"] == [1]
    assert report["models"]["current"]["prediction_count"] == 1
    assert report["models"]["challenger"]["prediction_count"] == 1
    where = str(statements[1]).split("WHERE", 1)[1]
    assert "best_action" not in where
    assert "missed_opportunity" not in where
    json.dumps(report, allow_nan=False)
