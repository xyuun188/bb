from __future__ import annotations

from pathlib import Path

from services.model_training_registry import build_model_training_registry
from services.profit_training_contract import PROFIT_TRAINING_TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHA256 = "a" * 64


def _by_id(payload: dict) -> dict[str, dict]:
    return {row["model_id"]: row for row in payload["models"]}


def test_registry_ignores_legacy_endpoint_alias_without_current_slot_runtime() -> None:
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

    assert finquant["runtime_available"] is False
    assert finquant["artifact_available"] is False
    assert finquant["alias_only"] is False
    assert finquant["lifecycle"] == "promotion_blocked"
    assert payload["summary"]["alias_only_models"] == []


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
                        "objective_name": "maximize_expected_realized_net_return_after_cost",
                        "objective_version": "2026-07-12.v1",
                        "preference_contract_version": "bb_finquant_return_preference.v1",
                        "preference_selection_accuracy": 1.0,
                        "training_stages": [
                            "sft_format_domain",
                            "trl_dpo_return_preference",
                        ],
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
            "manifest_services": [
                {
                    "slot": "llm_expert_pool",
                    "service_active": True,
                    "endpoint_ready": True,
                }
            ],
        }
    )

    finquant = _by_id(payload)["bb_finquant_expert_14b"]

    assert finquant["artifact_available"] is True
    assert finquant["runtime_available"] is True
    assert finquant["identity_verified"] is True
    assert finquant["alias_only"] is False
    assert finquant["lifecycle"] == "trained"


def test_registry_joins_llm_artifact_and_runtime_rows_by_slot() -> None:
    payload = build_model_training_registry(
        model_server_report={
            "required_slots": [
                {"slot": "llm_decision_maker", "served_model_name": "qwen3-14b-trade"},
                {
                    "slot": "llm_high_risk_review",
                    "served_model_name": "deepseek-r1-14b-risk",
                },
            ],
            "manifest_services": [
                {
                    "slot": "llm_decision_maker",
                    "service_active": True,
                    "endpoint_ready": True,
                },
                {
                    "slot": "llm_high_risk_review",
                    "service_active": True,
                    "endpoint_ready": True,
                },
            ],
        }
    )

    rows = _by_id(payload)
    assert rows["qwen3_14b_trade"]["runtime_available"] is True
    assert rows["qwen3_14b_trade"]["lifecycle"] == "inference_only"
    assert rows["deepseek_r1_14b_risk"]["runtime_available"] is True
    assert rows["deepseek_r1_14b_risk"]["lifecycle"] == "inference_only"


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
                    "model": "google/timesfm-2.5-200m-pytorch",
                    "actual_inference_count": 31,
                    "promotion_ready": False,
                    "avg_shadow_return_after_cost_pct": -0.2,
                    "authoritative_avg_return_after_cost_pct": 0.35,
                    "promotion_blockers": ["tail_loss"],
                },
                {
                    "model": "amazon/chronos-2",
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
    assert rows["timesfm_2_5"]["lifecycle"] == "inference_only"
    assert rows["timesfm_2_5"]["training_mode"] == "inference_only"
    assert rows["timesfm_2_5"]["evaluation_mode"] == "shadow_evaluating"
    assert rows["timesfm_2_5"][PROFIT_TRAINING_TARGET] == 0.35
    assert "net_return_after_cost_pct" not in rows["timesfm_2_5"]
    assert "avg_net_return_after_cost_pct" not in rows["timesfm_2_5"]
    assert rows["timesfm_2_5"]["fine_tune_available"] is False
    assert rows["chronos_2"]["trainable"] is False
    assert rows["chronos_2"]["model_family"] == "amazon/chronos-2"


def test_registry_preserves_zero_authoritative_profit_target() -> None:
    payload = build_model_training_registry(
        specialist_report={
            "generated_at": "2026-07-23T00:00:00+00:00",
            "models": [
                {
                    "model": "google/timesfm-2.5-200m-pytorch",
                    "authoritative_avg_return_after_cost_pct": 0.0,
                    "avg_shadow_return_after_cost_pct": 0.8,
                },
            ],
        }
    )

    rows = _by_id(payload)
    assert rows["timesfm_2_5"][PROFIT_TRAINING_TARGET] == 0.0


def test_registry_keeps_finbert_identity_evidence_separate_from_runtime_probe() -> None:
    payload = build_model_training_registry(
        local_ml_status={},
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
            "transformers_sentiment_backend": {"available": False},
        },
        specialist_report={
            "generated_at": "2026-07-11T23:37:16+00:00",
            "models": [
                {
                    "model": "ProsusAI/finbert",
                    "actual_inference_count": 24,
                    "fallback_count": 1840,
                    "promotion_ready": False,
                    "promotion_blockers": ["specialist_shadow_sample_floor_not_met"],
                },
                {
                    "model": "yiyanghkust/finbert-tone",
                    "actual_inference_count": 24,
                    "fallback_count": 1840,
                    "promotion_ready": False,
                    "promotion_blockers": ["specialist_shadow_sample_floor_not_met"],
                },
            ],
        },
    )

    rows = _by_id(payload)
    for model_id in ("finbert", "finbert_tone"):
        assert rows[model_id]["runtime_available"] is False
        assert rows[model_id]["lifecycle"] == "service_unavailable"
        assert rows[model_id]["identity_verified"] is True
        assert rows[model_id]["training_mode"] == "inference_only"
        assert rows[model_id]["evaluation_mode"] == "shadow_evaluating"
        assert rows[model_id]["quality_state"] == "promotion_blocked"
        assert rows[model_id]["live_influence"] is False
        assert rows[model_id]["fine_tune_available"] is False
    assert rows["finbert_tone"]["model_family"] == "yiyanghkust/finbert-tone"
    assert payload["summary"]["identity_failure_count"] == 0


def test_registry_evaluates_inference_only_llms_by_fee_after_contribution() -> None:
    payload = build_model_training_registry(
        model_server_report={
            "old_takeover_runtime": {
                "required_endpoints": [
                    {"served_model_name": "qwen3-14b-trade", "ready": True},
                    {"served_model_name": "deepseek-r1-14b-risk", "ready": True},
                ]
            }
        },
        contribution_performance={
            "decision_llm": {
                "count": 71,
                "pnl": -12.47,
                "avg_pnl": -0.175,
                "profit_factor": 0.77,
                "state": "degrade",
            },
            "high_risk_review": {
                "count": 67,
                "pnl": -11.18,
                "avg_pnl": -0.167,
                "profit_factor": 0.79,
                "state": "degrade",
            },
        },
    )

    rows = _by_id(payload)
    qwen = rows["qwen3_14b_trade"]
    risk = rows["deepseek_r1_14b_risk"]
    decision = rows["deepseek_online_decision"]

    assert qwen["evaluation_mode"] == "not_evaluated"
    assert qwen["blocking_reasons"] == ["model_specific_fee_after_attribution_missing"]
    assert risk["evaluation_sample_count"] == 67
    assert risk["profit_factor"] == 0.79
    assert "realized_net_pnl_non_positive" in risk["blocking_reasons"]
    assert decision["evaluation_sample_count"] == 71
    assert decision["quality_state"] == "promotion_blocked"


def test_registry_blocks_positive_pnl_when_profit_factor_is_undefined() -> None:
    payload = build_model_training_registry(
        contribution_performance={
            "decision_llm": {
                "count": 4,
                "pnl": 9.5,
                "avg_pnl": 2.375,
                "profit_factor": None,
            }
        }
    )

    decision = _by_id(payload)["deepseek_online_decision"]
    assert decision["realized_net_pnl_usdt"] == 9.5
    assert decision["profit_factor"] is None
    assert decision["quality_state"] == "promotion_blocked"
    assert decision["blocking_reasons"] == ["profit_factor_undefined"]


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
