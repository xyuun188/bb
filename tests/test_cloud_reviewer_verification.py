from __future__ import annotations

from services import cloud_reviewer_verification as verification
from web_dashboard.api import model_training_status


def test_cloud_reviewer_verification_is_durable_and_route_scoped(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(verification, "_verification_path", lambda: tmp_path / "state.json")
    saved = verification.save_cloud_reviewer_verification(
        api_base="https://review.example/v1",
        model="review-model",
        revision="",
        api_key="secret",
        latency_ms=12.34,
        provider="review.example",
        identity_source="chat_probe",
    )

    current = verification.load_cloud_reviewer_verification(
        api_base="https://review.example/v1",
        model="review-model",
        revision="",
        api_key="secret",
    )
    changed = verification.load_cloud_reviewer_verification(
        api_base="https://review.example/v1",
        model="another-model",
        revision="",
        api_key="secret",
    )

    assert saved["connection_verified"] is True
    assert current["connection_verified"] is True
    assert current["latency_ms"] == 12.3
    assert changed["connection_verified"] is False
    assert changed["verified_at"] is None


def test_training_registry_uses_current_verified_cloud_route(monkeypatch) -> None:
    monkeypatch.setattr(model_training_status, "load_model_training_report", lambda _path: {})
    monkeypatch.setattr(
        model_training_status,
        "validate_cloud_reviewer_route",
        lambda *_args: (True, None),
    )
    monkeypatch.setattr(
        model_training_status,
        "load_cloud_reviewer_verification",
        lambda **_kwargs: {
            "connection_verified": True,
            "verified_at": "2026-09-16T00:00:00+00:00",
            "latency_ms": 18.2,
            "identity_source": "chat_probe",
        },
    )
    monkeypatch.setattr(model_training_status.settings, "high_risk_review_enabled", True)
    monkeypatch.setattr(
        model_training_status.settings,
        "high_risk_review_api_base",
        "https://review.example/v1",
    )
    monkeypatch.setattr(model_training_status.settings, "high_risk_review_model", "review-model")
    monkeypatch.setattr(model_training_status.settings, "high_risk_review_model_revision", "")
    monkeypatch.setattr(model_training_status.settings, "high_risk_review_api_key", "secret")

    cloud = model_training_status._model_server_report_with_runtime_configuration()[
        "cloud_reviewer"
    ]

    assert cloud["configured"] is True
    assert cloud["connection_verified"] is True
    assert cloud["identity_verified"] is True
    assert cloud["runtime_available"] is True
    assert cloud["verified_at"] == "2026-09-16T00:00:00+00:00"


def test_registry_cache_clear_removes_persisted_snapshot(monkeypatch, tmp_path) -> None:
    snapshot = tmp_path / "registry.json"
    snapshot.write_text('{"models":[{"model_id":"stale"}]}', encoding="utf-8")
    monkeypatch.setattr(model_training_status, "_REGISTRY_SNAPSHOT_PATH", snapshot)
    monkeypatch.setattr(model_training_status, "_registry_cache", (1.0, {"models": []}))

    model_training_status.clear_model_training_registry_cache()

    assert model_training_status._registry_cache is None
    assert snapshot.exists() is False
