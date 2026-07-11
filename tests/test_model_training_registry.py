from __future__ import annotations

from services.model_training_registry import build_model_training_registry


def _by_id(payload: dict) -> dict[str, dict]:
    return {row["model_id"]: row for row in payload["models"]}


def test_registry_never_treats_old_takeover_alias_as_trained_finquant() -> None:
    payload = build_model_training_registry(
        local_ml_status={"available": True, "status": "degraded"},
        local_tools_status={
            "available": True,
            "service_available": True,
            "model_bundle_available": True,
            "models": {key: key for key in ("profit", "loss_filter", "timeseries", "deep_timeseries", "deep_sentiment", "exit")},
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
                    "specialization_evidence": {
                        "adapter_path": "/data/BB/models/bb-finquant/adapter",
                        "trained_at": "2026-07-11T00:00:00+00:00",
                        "sample_count": 128,
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
            "models": {key: key for key in ("profit", "loss_filter", "timeseries", "deep_timeseries", "deep_sentiment", "exit")},
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
