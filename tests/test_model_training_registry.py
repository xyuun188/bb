from __future__ import annotations

from pathlib import Path

from services.model_training_registry import build_model_training_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHA256 = "a" * 64


def _by_id(payload: dict) -> dict[str, dict]:
    return {row["model_id"]: row for row in payload["models"]}


def test_registry_never_treats_old_takeover_alias_as_trained_finquant() -> None:
    payload = build_model_training_registry(
        local_ml_status={"available": True, "status": "degraded"},
        local_tools_status={
            "available": True,
            "service_available": True,
            "model_bundle_available": True,
            "models": {
                key: key
                for key in (
                    "profit",
                    "loss_filter",
                    "timeseries",
                    "deep_timeseries",
                    "deep_sentiment",
                    "exit",
                )
            },
        },
        model_server_report={
            "deployment_contract": "old_one_gpu_timesfm_takeover",
            "old_takeover_runtime": {
                "required_endpoints": [
                    {"served_model_name": "BB-FinQuant-Expert-14B", "ready": True}
                ]
            },
            "required_slots": [
                {"slot": "llm_expert_pool", "base_model_carrier": "qwen3-14b-trade"}
            ],
        },
    )

    finquant = _by_id(payload)["bb_finquant_expert_14b"]

    assert finquant["runtime_available"] is True
    assert finquant["artifact_available"] is False
    assert finquant["alias_only"] is True
    assert finquant["lifecycle"] == "promotion_blocked"
    assert payload["summary"]["alias_only_models"] == ["bb_finquant_expert_14b"]


def test_registry_marks_verified_finquant_specialization_as_trained() -> None:
    payload = build_model_training_registry(
        model_server_report={
            "old_takeover_runtime": {
                "required_endpoints": [
                    {"served_model_name": "BB-FinQuant-Expert-14B", "ready": True}
                ]
            },
            "required_slots": [
                {
                    "slot": "llm_expert_pool",
                    "base_model_carrier": "Qwen3-14B-AWQ",
                    "specialization_evidence_verified": True,
                    "specialization_evidence": {
                        "verification_status": "verified",
                        "identity_verified": True,
                        "legacy_read_only": False,
                        "adapter_version": "20260712T010203Z-aaaaaaaaaaaa",
                        "adapter_path": "/data/BB/models/bb-finquant/adapter",
                        "specialization_manifest": "/data/BB/models/bb-finquant/manifest.json",
                        "specialization_id": "BB-FinQuant-Expert-14B-20260712T010203Z-aaaaaaaaaaaa",
                        "dataset_version": "bb-finquant-sft-v2-aaaaaaaaaaaa-bbbbbbbb",
                        "source_code_version": "commit-sha",
                        "base_model_repo": "Qwen/Qwen3-14B",
                        "trained_at": "2026-07-11T00:00:00+00:00",
                        "sample_count": 128,
                        "adapter_sha256": SHA256,
                        "manifest_sha256": SHA256,
                        "dataset_sha256": SHA256,
                        "dataset_lineage_sha256": SHA256,
                        "dataset_manifest_sha256": SHA256,
                        "source_script_sha256": SHA256,
                        "trainer_code_sha256": SHA256,
                        "base_model_config_sha256": SHA256,
                        "inference_base_model_config_sha256": SHA256,
                    },
                }
            ],
        }
    )

    finquant = _by_id(payload)["bb_finquant_expert_14b"]

    assert finquant["artifact_available"] is True
    assert finquant["identity_verified"] is True
    assert finquant["alias_only"] is False
    assert finquant["lifecycle"] == "trained"


def test_registry_separates_pretrained_specialists_from_project_training() -> None:
    payload = build_model_training_registry(
        local_ml_status={
            "available": True,
            "status": "degraded",
            "trained_at": "2026-07-11T00:00:00+00:00",
        },
        local_tools_status={
            "available": True,
            "service_available": True,
            "model_bundle_available": True,
            "models": {
                key: key
                for key in (
                    "profit",
                    "loss_filter",
                    "timeseries",
                    "deep_timeseries",
                    "deep_sentiment",
                    "exit",
                )
            },
            "transformers_sentiment_backend": {"available": True},
        },
        specialist_report={
            "generated_at": "2026-07-11T00:00:00+00:00",
            "models": [
                {
                    "model": "timesfm-2.5-primary",
                    "actual_inference_count": 31,
                    "promotion_ready": False,
                    "promotion_blockers": ["tail_loss"],
                },
                {
                    "model": "chronos-2-shadow-challenger",
                    "actual_inference_count": 31,
                    "promotion_ready": False,
                },
            ],
        },
    )

    rows = _by_id(payload)

    assert rows["local_ml_profit_quality"]["trainable"] is True
    assert rows["local_ml_profit_quality"]["lifecycle"] == "promotion_blocked"
    assert rows["timesfm_2_5"]["trainable"] is False
    assert rows["timesfm_2_5"]["lifecycle"] == "shadow_evaluating"
    assert rows["chronos_2"]["trainable"] is False
    assert rows["finbert"]["lifecycle"] == "inference_only"


def test_dashboard_renders_model_cards_from_canonical_registry() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    render_block = script[
        script.index("function renderTrainableModels()") : script.index(
            "// ========== Profit Attribution =========="
        )
    ]

    assert "fetchJSON('/api/model-training/registry')" in script
    assert "state.modelTrainingRegistry = registryData || null" in script
    assert "const registryModels = Array.isArray(registry.models)" in render_block
    assert "registryModels.map(model =>" in render_block
    assert "alias_only" in render_block
    assert "const models = [" not in render_block
