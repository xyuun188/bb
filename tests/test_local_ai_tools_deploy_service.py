from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

from scripts import deploy_local_ai_tools_service as deploy
from scripts.deploy_local_ai_tools_service import SERVICE_CODE

ROOT = Path(__file__).resolve().parents[1]


def _configure_local_ai_registry(module: ModuleType, root: Path) -> None:
    module.MODEL_DIR = root
    module.VERSIONS_ROOT = root / "versions"
    module.CANDIDATE_POINTER_PATH = root / "candidate.json"
    module.CHALLENGER_POINTER_PATH = root / "challenger.json"
    module.CURRENT_POINTER_PATH = root / "current.json"
    module.ROLLBACK_POINTER_PATH = root / "rollback.json"


def _persist_test_shadow_artifact(
    module: ModuleType,
    root: Path,
    **metadata_overrides: object,
) -> dict[str, object]:
    _configure_local_ai_registry(module, root)
    metadata = _test_artifact_metadata(module, **metadata_overrides)
    bundle = _test_artifact_bundle()
    module.persist_candidate_bundle(bundle, metadata)
    return module.activate_candidate_shadow({"test": "shadow"})


def _test_artifact_metadata(
    module: ModuleType,
    **metadata_overrides: object,
) -> dict[str, object]:
    ready_evidence = {
        "count": 4,
        "avg_return_pct": 0.4,
        "median_return_pct": 0.3,
        "return_lcb_pct": 0.1,
        "profit_factor": 2.0,
        "cvar_10_pct": -0.1,
        "max_drawdown_pct": 0.1,
        "promotion_math_ready": True,
    }
    ready_loso = {
        "version": "2026-07-15.leave-one-symbol-out.v1",
        "evaluated_symbol_count": 2,
        "rows": [],
        "stable": True,
    }
    walk_forward_report = {
        "version": "2026-07-15.expanding-decision-group-walk-forward.v1",
        "status": "complete",
        "decision_group_disjoint": True,
        "chronological_label_disjoint": True,
        "model_refit_per_fold": True,
        "folds": [
            {
                "decision_group_overlap_count": 0,
                "sides": {
                    side: dict(ready_evidence) for side in ("long", "short")
                },
            },
            {
                "decision_group_overlap_count": 0,
                "sides": {
                    side: dict(ready_evidence) for side in ("long", "short")
                },
            },
        ],
        "sides": {
            side: {
                **ready_evidence,
                "leave_one_symbol_out": dict(ready_loso),
                "market_regime_stability": {"stable": True},
            }
            for side in ("long", "short")
        },
    }
    metadata: dict[str, object] = {
        "trained_at": "2026-07-14T00:00:00+00:00",
        "shadow_sample_count": 222,
        "trade_sample_count": 33,
        "profile_count": 7,
        "artifact_persisted": True,
        "training_data_sha256": "a" * 64,
        "source_code_sha256": "b" * 64,
        "objective_name": module.RETURN_OBJECTIVE_NAME,
        "objective_version": module.RETURN_OBJECTIVE_VERSION,
        "label_name": module.RETURN_LABEL_NAME,
        "label_version": module.RETURN_LABEL_VERSION,
        "cost_model_version": module.COST_MODEL_VERSION,
        "profit_supervision_version": module.PROFIT_SUPERVISION_VERSION,
        "time_split_policy": "chronological_disjoint_decision_groups",
        "quality_report": {
            "market_fact_contract": {
                "status": "clean",
                "violation_count": 0,
                "assertions": {
                    "native_instrument_identity_verified": True,
                    "same_contract_price_path_verified": True,
                    "executable_market_fact_verified": True,
                },
                "provenance": {"data_fingerprint": "e" * 64},
            }
        },
        "market_fact_contract": {
            "status": "clean",
            "violation_count": 0,
            "assertions": {
                "native_instrument_identity_verified": True,
                "same_contract_price_path_verified": True,
                "executable_market_fact_verified": True,
            },
            "provenance": {"data_fingerprint": "e" * 64},
        },
        "governance_report": {
            "quality_fingerprint": "quality-fingerprint",
            "artifact_quality_fingerprint": "quality-fingerprint",
            "artifact_matches_quality": True,
            "requires_artifact_refresh": False,
        },
        "walk_forward_report": walk_forward_report,
        "leave_one_symbol_out_report": {
            side: dict(ready_loso) for side in ("long", "short")
        },
        "oos_return_evaluation": {
            side: dict(ready_evidence) for side in ("long", "short")
        },
        "authoritative_trade_return_evidence": {
            "version": "2026-07-15.authoritative-trade-return-evidence.v1",
            "data_fingerprint": "d" * 64,
            "sides": {
                side: dict(ready_evidence) for side in ("long", "short")
            },
        },
    }
    metadata.update(metadata_overrides)
    metadata["evaluation_report_hashes"] = module._evaluation_report_hashes(metadata)
    metadata["artifact_return_evidence_sha256"] = module.canonical_sha256(
        metadata["evaluation_report_hashes"]
    )
    return metadata


def _test_artifact_bundle() -> dict[str, object]:
    return {
        "long_return_model": "long-return",
        "short_return_model": "short-return",
        "long_cost_model": "long-cost",
        "short_cost_model": "short-cost",
    }


def test_local_ai_tools_generated_service_requires_api_key_or_loopback() -> None:
    assert "LOCAL_AI_TOOLS_API_KEY = os.environ.get" in SERVICE_CODE
    assert "def require_api_key(" in SERVICE_CODE
    assert "dependencies=[Depends(require_api_key)]" in SERVICE_CODE
    assert "LOCAL_AI_TOOLS_API_KEY is required for non-loopback access" in SERVICE_CODE
    assert "Bearer {LOCAL_AI_TOOLS_API_KEY}" in SERVICE_CODE


def test_local_ai_tools_training_uses_per_sample_costs_and_empirical_tail_policy() -> None:
    assert "LOCAL_AI_TOOLS_ROUND_TRIP_COST_PCT" not in SERVICE_CODE
    assert "LOCAL_AI_TOOLS_TAIL_LOSS_THRESHOLD_PCT" not in SERVICE_CODE
    assert "def cost_complete_net_returns(" in SERVICE_CODE
    assert "def empirical_lower_hinge(" in SERVICE_CODE
    assert "legacy_fixed_training_thresholds_enabled" not in SERVICE_CODE
    assert "def _dynamic_min_samples_leaf(sample_count: int)" in SERVICE_CODE
    assert "min_samples_leaf=8" not in SERVICE_CODE
    assert "min_samples_leaf=10" not in SERVICE_CODE
    assert "n_jobs=-1" not in SERVICE_CODE
    assert "def _adaptive_training_worker_count()" in SERVICE_CODE
    assert "_configure_bundle_for_inference(candidate)" in SERVICE_CODE
    assert "sentiment_leaf_size = _dynamic_min_samples_leaf(len(sentiment_samples))" in (
        SERVICE_CODE
    )
    assert "if len(rows) < 80:" not in SERVICE_CODE


def test_local_ai_tools_training_parallelism_leaves_inference_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("local_ai_tools_parallelism_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    monkeypatch.setattr(
        module.os,
        "sched_getaffinity",
        lambda _pid: set(range(16)),
        raising=False,
    )

    regressor = module._make_regressor(100)
    classifier = module._make_classifier([0, 1, 0, 1])

    assert module._available_cpu_count() == 16
    assert module._adaptive_training_worker_count() == 4
    assert regressor.named_steps["model"].n_jobs == 4
    assert classifier.named_steps["model"].n_jobs == 4


def test_local_ai_tools_loaded_bundle_forces_single_worker_inference(
    tmp_path: Path,
) -> None:
    from sklearn.ensemble import ExtraTreesRegressor

    module = ModuleType("local_ai_tools_inference_parallelism_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    _configure_local_ai_registry(module, tmp_path)
    estimator_names = (
        "long_return_model",
        "short_return_model",
        "long_cost_model",
        "short_cost_model",
    )
    bundle = {
        name: ExtraTreesRegressor(n_estimators=2, n_jobs=-1)
        for name in estimator_names
    }
    bundle["horizon_models"] = {
        10: {"long_model": ExtraTreesRegressor(n_estimators=2, n_jobs=-1)}
    }
    module.persist_candidate_bundle(bundle, _test_artifact_metadata(module))
    module.activate_candidate_shadow({"test": "shadow"})

    loaded = module.load_bundle()

    assert loaded is not None
    assert all(loaded[name].n_jobs == 1 for name in estimator_names)
    assert loaded["horizon_models"][10]["long_model"].n_jobs == 1


def test_text_sentiment_training_uses_available_distribution_without_fixed_sample_gate() -> None:
    module = ModuleType("local_ai_tools_text_training_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)

    result = module._train_text_sentiment_model(
        [
            {"text": "fee after return improves", "sentiment_score": 0.4},
            {"text": "cost pressure increases", "sentiment_score": -0.3},
        ]
    )

    assert result is not None
    assert result["samples"] == 2


def test_compact_native_sequence_transport_expands_all_windows_lazily() -> None:
    module = ModuleType("local_ai_tools_compact_sequence_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    closes = [100.0 + index for index in range(33)]
    sample = {
        "symbol": "BTC/USDT",
        "timeframe": "1m",
        "sequence_format": module.COMPACT_SEQUENCE_SERIES_FORMAT,
        "close_sequence": closes,
        "volume_sequence": [10.0 + index for index in range(33)],
        "observation_count": 2,
        "label_name": "gross_market_move_pct",
        "label_version": "test-observation-label",
    }

    windows = list(module._iter_sequence_training_windows([sample]))

    assert len(windows) == 2
    assert windows[0]["close_sequence"] == closes[:31]
    assert windows[1]["close_sequence"] == closes[:32]
    assert windows[0]["long_return_pct"] == pytest.approx((131.0 - 130.0) / 130.0 * 100.0)
    assert windows[0]["short_return_pct"] == pytest.approx(
        -windows[0]["long_return_pct"]
    )
    assert list(
        module._iter_sequence_training_windows(
            [{**sample, "observation_count": 1}]
        )
    ) == []


def test_training_upload_uses_scheduler_deadline_instead_of_http_write_timeout() -> None:
    source = (ROOT / "scripts" / "train_local_ai_tools_models.py").read_text(
        encoding="utf-8"
    )

    assert "write=None" in source
    assert "read=None" in source


def test_remote_smoke_probe_rejects_oversized_responses_without_truncating_json() -> None:
    command = deploy._remote_smoke_command()

    assert "response.read(256000)" not in command
    assert "response.read(4 * 1024 * 1024 + 1)" in command
    assert "phase3_quant_api_response_exceeds_4mb" in command


def test_local_ai_tools_generated_service_disables_local_high_risk_review() -> None:
    assert "openai-compatible-risk-review" not in SERVICE_CODE
    assert "SERVED_REVIEW_MODEL" not in SERVICE_CODE
    assert "MAIN_LLM_BASE" not in SERVICE_CODE
    assert "MAIN_LLM_MODEL" not in SERVICE_CODE
    assert "import httpx" not in SERVICE_CODE
    assert '"data": []' in SERVICE_CODE
    assert "status_code=410" in SERVICE_CODE
    assert "Configure HIGH_RISK_REVIEW_*" in SERVICE_CODE


def test_local_ai_tools_health_returns_service_status_without_trained_bundle() -> None:
    assert "def _model_artifact_status()" in SERVICE_CODE
    assert "def _status_metadata()" in SERVICE_CODE
    assert '"ok": True' in SERVICE_CODE
    assert '"service": "phase3_quant_api"' in SERVICE_CODE
    assert '"root": PHASE3_ROOT.as_posix()' in SERVICE_CODE
    assert '"port": PHASE3_API_PORT' in SERVICE_CODE
    assert '"live_mutation": False' in SERVICE_CODE
    assert '"status_endpoint_uses_metadata_only": True' in SERVICE_CODE
    assert '"review_backend": "disabled_use_trading_app_online_model"' in SERVICE_CODE


def test_local_ai_tools_generated_service_adds_phase3_model_metadata() -> None:
    assert "def with_model_metadata(" in SERVICE_CODE
    assert "def _shadow_payload(" in SERVICE_CODE
    assert '"primary_model"' in SERVICE_CODE
    assert '"challenger_model"' in SERVICE_CODE
    assert '"model_version"' in SERVICE_CODE
    assert '"route_mode"' in SERVICE_CODE
    assert '"feature_coverage"' in SERVICE_CODE
    assert '"shadow_payload"' in SERVICE_CODE
    assert '"promotion_flow"' in SERVICE_CODE
    assert '"live_mutation"' in SERVICE_CODE
    assert '_attach_baseline_only_shadow("profit_prediction"' in SERVICE_CODE
    assert 'with_model_metadata("time_series_prediction"' in SERVICE_CODE
    assert 'with_model_metadata("sentiment_analysis"' in SERVICE_CODE
    assert 'with_model_metadata("exit_advice"' in SERVICE_CODE
    assert "def regression_prediction_distribution(" in SERVICE_CODE
    assert '"lower_quantile_return_pct"' in SERVICE_CODE
    assert '"return_distribution_inputs"' in SERVICE_CODE
    assert '"prediction_quality"' in SERVICE_CODE
    assert '"training_cost_policy"' in SERVICE_CODE


def test_local_ai_tools_generated_service_metadata_helpers_are_callable() -> None:
    module = ModuleType("local_ai_tools_api_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)

    coverage = module._feature_coverage({"returns_1": 0.1, "rsi_14": 55})
    health = module.health()

    assert coverage["present"] >= 2
    assert 0 < coverage["ratio"] <= 1
    assert health["ok"] is True
    assert health["service"] == "phase3_quant_api"
    assert health["root"] == "/data/BB"
    assert health["port"] == 8101
    assert health["live_mutation"] is False
    assert health["trained_models_available"] is False

    payload = module.with_model_metadata(
        "profit_prediction",
        {
            "available": True,
            "trained": False,
            "model": "local-profit-artifact-required-v3",
            "best_side": "hold",
            "return_semantics": "gross_market_opportunity_before_execution",
            "return_distribution_input_version": (
                module.RETURN_DISTRIBUTION_INPUT_VERSION
            ),
            "return_distribution_inputs": {"long": {"production_eligible": False}},
            "loss_probability": 0.18,
            "profit_quality_score": 0.25,
        },
        features={"returns_1": 0.1, "rsi_14": 55},
    )

    assert payload["promotion_flow"] == "candidate_to_shadow_to_canary_to_active"
    assert payload["live_mutation"] is False
    assert payload["shadow_payload"]["tool"] == "profit_prediction"
    assert payload["shadow_payload"]["live_mutation"] is False
    assert payload["shadow_payload"]["return_distribution_input_version"] == (
        module.RETURN_DISTRIBUTION_INPUT_VERSION
    )
    assert payload["shadow_payload"]["return_distribution_inputs"] == {
        "long": {"production_eligible": False}
    }
    assert "expected_return_pct" not in payload["shadow_payload"]
    assert "adjusted_expected_return_pct" not in payload["shadow_payload"]
    assert payload["shadow_payload"]["loss_probability"] == 0.18


def test_generated_service_without_artifact_fails_closed_without_heuristic_returns() -> None:
    module = ModuleType("local_ai_tools_api_no_artifact_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module.FeatureRequest.model_rebuild()
    module.load_bundle = lambda: None
    request = module.FeatureRequest(
        symbol="BTC/USDT",
        features={"symbol": "BTC/USDT", "current_price": 100.0},
    )

    for payload in (module.profit_predict(request), module.timeseries_predict(request)):
        assert payload["available"] is False
        assert payload["trained"] is False
        assert payload["best_side"] == "hold"
        assert payload["prediction_quality"]["production_eligible"] is False
        assert payload["return_distribution_input_version"] == (
            module.RETURN_DISTRIBUTION_INPUT_VERSION
        )
        assert set(payload["return_distribution_inputs"]) == {"long", "short"}
        assert all(
            item["production_eligible"] is False
            for item in payload["return_distribution_inputs"].values()
        )
        for legacy_field in (
            "expected_return_pct",
            "adjusted_expected_return_pct",
            "long_expected_return_pct",
            "short_expected_return_pct",
        ):
            assert legacy_field not in payload


def test_generated_service_status_does_not_advertise_heuristic_fallback(
    tmp_path: Path,
) -> None:
    module = ModuleType("local_ai_tools_api_no_artifact_status_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    _configure_local_ai_registry(module, tmp_path)

    status = module._model_artifact_status()
    endpoint = module.local_models_status()

    assert status["available"] is False
    assert status["status"] == "artifact_unavailable"
    assert status["return_distribution_input_version"] == (
        module.RETURN_DISTRIBUTION_INPUT_VERSION
    )
    assert endpoint["status"] == "artifact_unavailable"
    assert "heuristic" not in endpoint["message"].lower()


def test_health_metadata_exposes_separated_supervision_contract(monkeypatch) -> None:
    module = ModuleType("local_ai_tools_health_supervision_contract_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    supervision = {
        "version": module.PROFIT_SUPERVISION_VERSION,
        "shadow_market_sample_count": 14,
        "shadow_counterfactual_cost_sample_count": 14,
        "actual_execution_cost_sample_count": 1,
        "actual_realized_return_sample_count": 61,
    }
    monkeypatch.setattr(
        module,
        "_read_metadata_file",
        lambda: {
            "objective_name": module.RETURN_OBJECTIVE_NAME,
            "objective_version": module.RETURN_OBJECTIVE_VERSION,
            "label_name": module.RETURN_LABEL_NAME,
            "label_version": module.RETURN_LABEL_VERSION,
            "cost_model_version": module.COST_MODEL_VERSION,
            "training_cost_policy": (
                "separated_market_opportunity_and_execution_cost_tasks"
            ),
            "profit_supervision_version": module.PROFIT_SUPERVISION_VERSION,
            "profit_supervision_report": supervision,
            "train_shadow_sample_count": 7,
            "holdout_shadow_sample_count": 7,
            "train_decision_group_count": 7,
            "holdout_decision_group_count": 7,
        },
    )

    metadata = module._status_metadata()

    assert metadata["objective_version"] == module.RETURN_OBJECTIVE_VERSION
    assert metadata["label_version"] == module.RETURN_LABEL_VERSION
    assert metadata["cost_model_version"] == module.COST_MODEL_VERSION
    assert metadata["profit_supervision_report"] == supervision
    assert metadata["train_decision_group_count"] == 7
    assert metadata["holdout_decision_group_count"] == 7
    assert "round_trip_cost_pct" not in module._STATUS_METADATA_KEYS
    assert "tail_loss_threshold_pct" not in module._STATUS_METADATA_KEYS


def test_shadow_model_metadata_blocks_runtime_production_eligibility(monkeypatch) -> None:
    module = ModuleType("local_ai_tools_api_return_contract_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    monkeypatch.setattr(
        module,
        "load_bundle",
        lambda: {
            "metadata": {
                "objective_name": module.RETURN_OBJECTIVE_NAME,
                "objective_version": module.RETURN_OBJECTIVE_VERSION,
                "label_name": module.RETURN_LABEL_NAME,
                "label_version": module.RETURN_LABEL_VERSION,
                "training_cost_policy": (
                    "separated_market_opportunity_and_execution_cost_tasks"
                ),
                "profit_supervision_version": module.PROFIT_SUPERVISION_VERSION,
                "artifact_persisted": True,
                "model_stage": "shadow",
                "training_mode": "shadow",
            }
        },
    )

    payload = module.with_model_metadata(
        "profit_prediction",
        {
            "available": True,
            "trained": True,
            "model": "trained-test-model",
            "return_semantics": "gross_market_opportunity_before_execution",
            "return_distribution_input_version": (
                module.RETURN_DISTRIBUTION_INPUT_VERSION
            ),
            "return_distribution_inputs": {
                side: {"production_eligible": True}
                for side in ("long", "short")
            },
            "prediction_quality": {
                "production_eligible": True,
                "anomalous": False,
                "reason": "current_tree_prediction_distribution_ready",
            },
        },
    )

    assert payload["artifact_persisted"] is True
    assert payload["objective_name"] == module.RETURN_OBJECTIVE_NAME
    assert payload["label_name"] == module.RETURN_LABEL_NAME
    assert payload["training_cost_policy"] == (
        "separated_market_opportunity_and_execution_cost_tasks"
    )
    assert payload["profit_supervision_version"] == module.PROFIT_SUPERVISION_VERSION
    assert payload["model_stage"] == "shadow"
    assert payload["route_mode"] == "shadow_observation"
    assert payload["production_permission"] is False
    assert payload["live_ml_ready"] is False
    assert payload["prediction_quality"]["production_eligible"] is False
    assert all(
        item["production_eligible"] is False
        and "artifact_activation_not_production_authorized" in item["blockers"]
        for item in payload["return_distribution_inputs"].values()
    )


def test_local_ai_tools_status_endpoints_do_not_load_joblib_bundle(tmp_path: Path) -> None:
    module = ModuleType("local_ai_tools_api_metadata_status_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    _persist_test_shadow_artifact(module, tmp_path)

    def fail_load_bundle() -> None:
        raise AssertionError("status endpoints must not deserialize model bundles")

    module.load_bundle = fail_load_bundle

    health = module.health()
    status = module.local_models_status()

    assert health["ok"] is True
    assert health["trained_models_available"] is True
    assert health["shadow_sample_count"] == 222
    assert health["trade_sample_count"] == 33
    assert health["status_endpoint_uses_metadata_only"] is True
    assert status["available"] is True
    assert status["profile_count"] == 7
    assert status["metadata_loaded"] is True
    assert status["status_endpoint_uses_metadata_only"] is True
    assert status["artifact_activation_manifest"]["activation_stage"] == "shadow"
    assert status["artifact_activation_manifest"][
        "live_ml_ready"
    ] is False


def test_local_ai_tools_status_payload_is_bounded_and_exposes_child_contracts(
    tmp_path: Path,
) -> None:
    module = ModuleType("local_ai_tools_api_compact_status_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    current = _persist_test_shadow_artifact(
        module,
        tmp_path,
        profit_supervision_report={
            "version": module.PROFIT_SUPERVISION_VERSION,
            "shadow_market_sample_count": 100,
            "shadow_events": [
                {"symbol": f"TOKEN-{index}", "payload": "x" * 4096}
                for index in range(300)
            ],
            "authoritative_evidence": [
                {"symbol": f"TOKEN-{index}", "payload": "y" * 4096}
                for index in range(300)
            ],
        },
    )
    activation_path = current["version_root"] / "activation-shadow.json"
    activation = module.read_json_object(activation_path)
    activation["return_evidence_report"] = {
        "rows": [
            {"symbol": f"TOKEN-{index}", "payload": "z" * 4096}
            for index in range(300)
        ]
    }
    module.write_json_atomic(activation_path, activation)
    pointer = module.read_json_object(module.CURRENT_POINTER_PATH)
    pointer["activation_manifest_sha256"] = module.sha256_file(activation_path)
    module.write_json_atomic(module.CURRENT_POINTER_PATH, pointer)
    module._STATUS_ARTIFACT_CACHE.clear()

    health = module.health()
    status = module.local_models_status()
    encoded = json.dumps(status, ensure_ascii=False).encode("utf-8")

    assert len(encoded) < 512 * 1024
    assert "shadow_events" not in encoded.decode("utf-8")
    assert "authoritative_evidence" not in encoded.decode("utf-8")
    assert "return_evidence_report" not in encoded.decode("utf-8")
    assert status["status_payload_compacted"] is True
    assert health["child_endpoints"]["profit_prediction"]["available"] is True
    assert status["child_endpoints"]["time_series_prediction"]["probe_mode"] == (
        "metadata_contract"
    )
    assert status["child_endpoints"]["sentiment_analysis"][
        "actual_inference_probe"
    ] is False


def test_load_bundle_reuses_verified_unchanged_artifact(tmp_path: Path) -> None:
    module = ModuleType("local_ai_tools_api_verified_bundle_cache_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    _persist_test_shadow_artifact(module, tmp_path)
    module._BUNDLE_CACHE = None
    module._CURRENT_POINTER_MTIME_NS = None
    module._CURRENT_MODEL_MTIME_NS = None
    module._CURRENT_MODEL_PATH = None
    module._STATUS_ARTIFACT_CACHE.clear()
    resolve_count = 0
    original_resolve = module._resolve_artifact_pointer

    def counted_resolve(*args: object, **kwargs: object) -> dict | None:
        nonlocal resolve_count
        resolve_count += 1
        return original_resolve(*args, **kwargs)

    module._resolve_artifact_pointer = counted_resolve

    first = module.load_bundle()
    second = module.load_bundle()

    assert first is not None
    assert second is first
    assert resolve_count == 1


def test_load_bundle_serializes_concurrent_cold_loads(tmp_path: Path) -> None:
    module = ModuleType("local_ai_tools_api_concurrent_bundle_cache_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    _persist_test_shadow_artifact(module, tmp_path)
    module._BUNDLE_CACHE = None
    module._CURRENT_POINTER_MTIME_NS = None
    module._CURRENT_MODEL_MTIME_NS = None
    module._CURRENT_MODEL_PATH = None
    module._STATUS_ARTIFACT_CACHE.clear()
    load_count = 0
    original_load = module.load_trusted_joblib_bundle

    def counted_load(path: Path) -> dict:
        nonlocal load_count
        load_count += 1
        time.sleep(0.05)
        return original_load(path)

    module.load_trusted_joblib_bundle = counted_load

    with ThreadPoolExecutor(max_workers=3) as pool:
        bundles = list(pool.map(lambda _index: module.load_bundle(), range(3)))

    assert all(bundle is bundles[0] for bundle in bundles)
    assert load_count == 1


def test_rejected_current_bundle_is_not_reloaded_until_pointer_changes(tmp_path: Path) -> None:
    module = ModuleType("local_ai_tools_api_rejected_bundle_cache_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    current = _persist_test_shadow_artifact(module, tmp_path)
    module._BUNDLE_CACHE = None
    module._CURRENT_POINTER_MTIME_NS = None
    module._CURRENT_MODEL_MTIME_NS = None
    module._CURRENT_MODEL_PATH = None
    module._STATUS_ARTIFACT_CACHE.clear()
    load_count = 0

    def load_legacy_bundle(_path: Path) -> dict:
        nonlocal load_count
        load_count += 1
        return {"metadata": {}}

    module.load_trusted_joblib_bundle = load_legacy_bundle

    assert module.load_bundle() is None
    assert module.load_bundle() is None
    assert load_count == 1
    assert module._CURRENT_MODEL_MTIME_NS == current["model_path"].stat().st_mtime_ns


def test_local_ai_candidate_activation_is_atomic_and_rollbackable(tmp_path: Path) -> None:
    module = ModuleType("local_ai_tools_api_registry_lifecycle_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    first = _persist_test_shadow_artifact(module, tmp_path, training_data_sha256="a" * 64)
    second_candidate = module.persist_candidate_bundle(
        _test_artifact_bundle(),
        _test_artifact_metadata(module, training_data_sha256="c" * 64),
    )

    assert module._resolve_artifact_pointer(
        module.CURRENT_POINTER_PATH,
        role="current",
    )["version"] == first["version"]
    assert second_candidate["version"] != first["version"]

    second = module.activate_candidate_shadow({"test": "second-shadow"})

    assert second["version"] == second_candidate["version"]
    module._BUNDLE_CACHE = None
    module._CURRENT_POINTER_MTIME_NS = None
    module._CURRENT_MODEL_MTIME_NS = None
    runtime_bundle = module.load_bundle()
    assert runtime_bundle["metadata"]["artifact_lifecycle"] == "shadow"
    assert runtime_bundle["metadata"]["model_stage"] == "shadow"
    assert runtime_bundle["metadata"]["live_ml_ready"] is False
    assert runtime_bundle["metadata"]["artifact_version"] == second["version"]
    assert module._resolve_artifact_pointer(
        module.ROLLBACK_POINTER_PATH,
        role="rollback",
    )["version"] == first["version"]
    restored = module.rollback_current_artifact()
    assert restored["version"] == first["version"]
    assert module._resolve_artifact_pointer(
        module.ROLLBACK_POINTER_PATH,
        role="rollback",
    )["version"] == second["version"]


def test_local_ai_registry_rejects_tampered_candidate_manifest(tmp_path: Path) -> None:
    module = ModuleType("local_ai_tools_api_registry_tamper_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    _configure_local_ai_registry(module, tmp_path)
    candidate = module.persist_candidate_bundle(
        _test_artifact_bundle(),
        _test_artifact_metadata(module),
    )
    candidate["manifest_path"].write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest hash verification failed"):
        module._resolve_artifact_pointer(
            module.CANDIDATE_POINTER_PATH,
            role="candidate",
        )
    assert not module.CURRENT_POINTER_PATH.exists()


def test_local_ai_registry_rejects_stale_evaluation_report_hashes(
    tmp_path: Path,
) -> None:
    module = ModuleType("local_ai_tools_api_registry_evidence_hash_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    _configure_local_ai_registry(module, tmp_path)
    metadata = _test_artifact_metadata(module)
    metadata["oos_return_evaluation"]["long"]["return_lcb_pct"] = -5.0

    with pytest.raises(ValueError, match="evaluation report hashes are invalid"):
        module.persist_candidate_bundle(_test_artifact_bundle(), metadata)

    assert not module.CANDIDATE_POINTER_PATH.exists()
    assert not module.CURRENT_POINTER_PATH.exists()


def test_local_ai_registry_rejects_live_activation_without_return_readiness(
    tmp_path: Path,
) -> None:
    module = ModuleType("local_ai_tools_api_registry_live_evidence_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    _configure_local_ai_registry(module, tmp_path)
    metadata = _test_artifact_metadata(module)
    metadata["oos_return_evaluation"]["long"]["profit_factor"] = None
    metadata["oos_return_evaluation"]["long"]["promotion_math_ready"] = False
    metadata["evaluation_report_hashes"] = module._evaluation_report_hashes(metadata)
    metadata["artifact_return_evidence_sha256"] = module.canonical_sha256(
        metadata["evaluation_report_hashes"]
    )
    module.persist_candidate_bundle(_test_artifact_bundle(), metadata)
    current = module.activate_candidate_shadow({"test": "shadow"})
    activation_path = current["version_root"] / "activation-shadow.json"
    activation = module.read_json_object(activation_path)
    activation.update(
        {
            "activation_stage": "active",
            "live_ml_ready": True,
            "return_evidence_ready": True,
            "execution_scope": "production",
            "production_permission": True,
            "promotion_recommendation": {
                "recommended_stage": "active",
                "live_ml_ready": True,
            },
        }
    )
    module.write_json_atomic(activation_path, activation)
    pointer = module.read_json_object(module.CURRENT_POINTER_PATH)
    pointer["activation_manifest_sha256"] = module.sha256_file(activation_path)
    module.write_json_atomic(module.CURRENT_POINTER_PATH, pointer)

    with pytest.raises(ValueError, match="return evidence is not ready"):
        module._resolve_artifact_pointer(
            module.CURRENT_POINTER_PATH,
            role="current",
        )


def test_local_ai_registry_activates_governed_paper_canary(tmp_path: Path) -> None:
    module = ModuleType("local_ai_tools_api_registry_canary_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    _configure_local_ai_registry(module, tmp_path)
    metadata = _test_artifact_metadata(module)
    metadata["oos_return_evaluation"]["long"]["return_lcb_pct"] = -0.5
    metadata["oos_return_evaluation"]["long"]["promotion_math_ready"] = False
    metadata["evaluation_report_hashes"] = module._evaluation_report_hashes(metadata)
    metadata["artifact_return_evidence_sha256"] = module.canonical_sha256(
        metadata["evaluation_report_hashes"]
    )
    module.persist_candidate_bundle(_test_artifact_bundle(), metadata)
    recommendation = {
        "recommended_stage": "canary",
        "canary_ready": True,
        "canary_execution_scope": "paper_only",
        "canary_production_permission": False,
        "live_ml_ready": False,
    }

    evidence = {"promotion_recommendation": recommendation}
    module.activate_candidate_shadow(evidence)
    current = module.transition_current_artifact(
        evidence,
        activation_stage="canary",
    )

    activation = current["activation_manifest"]
    assert activation["activation_stage"] == "canary"
    assert activation["execution_scope"] == "paper_only"
    assert activation["production_permission"] is False
    assert activation["live_ml_ready"] is False
    assert activation["canary_authorized"] is True
    assert activation["return_evidence_ready"] is False


def test_local_ai_registry_activates_active_only_with_complete_evidence(
    tmp_path: Path,
) -> None:
    module = ModuleType("local_ai_tools_api_registry_governed_live_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    _configure_local_ai_registry(module, tmp_path)
    metadata = _test_artifact_metadata(module)
    assert module._production_return_evidence_blockers(metadata) == []
    module.persist_candidate_bundle(_test_artifact_bundle(), metadata)
    recommendation = {
        "recommended_stage": "active",
        "canary_ready": True,
        "canary_execution_scope": "paper_only",
        "canary_production_permission": False,
        "live_ml_ready": True,
    }

    evidence = {"promotion_recommendation": recommendation}
    module.activate_candidate_shadow(evidence)
    module.transition_current_artifact(
        evidence,
        activation_stage="canary",
    )
    current = module.transition_current_artifact(
        evidence,
        activation_stage="active",
    )

    activation = current["activation_manifest"]
    assert activation["activation_stage"] == "active"
    assert activation["execution_scope"] == "production"
    assert activation["production_permission"] is True
    assert activation["live_ml_ready"] is True
    assert activation["return_evidence_ready"] is True


def test_local_ai_registry_rejects_regressive_active_challenger(
    tmp_path: Path,
) -> None:
    module = ModuleType("local_ai_tools_api_registry_challenger_rejection_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    champion = _persist_test_shadow_artifact(module, tmp_path)
    recommendation = {
        "recommended_stage": "active",
        "canary_ready": True,
        "canary_execution_scope": "paper_only",
        "canary_production_permission": False,
        "live_ml_ready": True,
    }
    evidence = {"promotion_recommendation": recommendation}
    module.transition_current_artifact(evidence, activation_stage="canary")
    champion = module.transition_current_artifact(evidence, activation_stage="active")
    champion_pointer_before = module.read_json_object(module.CURRENT_POINTER_PATH)

    challenger_metadata = _test_artifact_metadata(module)
    for side in ("long", "short"):
        challenger_metadata["oos_return_evaluation"][side].update(
            {
                "avg_return_pct": 0.2,
                "return_lcb_pct": 0.05,
                "profit_factor": 1.5,
                "cvar_10_pct": -0.2,
                "max_drawdown_pct": 0.2,
            }
        )
    challenger_metadata["evaluation_report_hashes"] = module._evaluation_report_hashes(
        challenger_metadata
    )
    challenger_metadata["artifact_return_evidence_sha256"] = module.canonical_sha256(
        challenger_metadata["evaluation_report_hashes"]
    )
    candidate = module.persist_candidate_bundle(
        _test_artifact_bundle(),
        challenger_metadata,
    )

    comparison = module._compare_candidate_to_current(
        challenger_metadata,
        candidate_stage="active",
    )
    rejected = module.reject_candidate_artifact(comparison)
    champion_pointer_after = module.read_json_object(module.CURRENT_POINTER_PATH)

    assert comparison["accepted"] is False
    assert comparison["reason"] == "champion_retained"
    assert rejected["version"] == candidate["version"]
    assert rejected["rejection_manifest"]["comparison_report"] == comparison
    assert not module.CANDIDATE_POINTER_PATH.exists()
    assert module.CHALLENGER_POINTER_PATH.exists()
    assert champion_pointer_after["version"] == champion["version"]
    assert champion_pointer_after["artifact_sha256"] == champion_pointer_before[
        "artifact_sha256"
    ]


def test_local_ai_registry_accepts_strictly_improved_active_challenger(
    tmp_path: Path,
) -> None:
    module = ModuleType("local_ai_tools_api_registry_challenger_acceptance_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    champion = _persist_test_shadow_artifact(module, tmp_path)
    recommendation = {
        "recommended_stage": "active",
        "canary_ready": True,
        "canary_execution_scope": "paper_only",
        "canary_production_permission": False,
        "live_ml_ready": True,
    }
    evidence = {"promotion_recommendation": recommendation}
    module.transition_current_artifact(evidence, activation_stage="canary")
    champion = module.transition_current_artifact(evidence, activation_stage="active")

    challenger_metadata = _test_artifact_metadata(module)
    for side in ("long", "short"):
        challenger_metadata["oos_return_evaluation"][side].update(
            {
                "avg_return_pct": 0.6,
                "return_lcb_pct": 0.2,
                "profit_factor": 2.5,
                "cvar_10_pct": -0.05,
                "max_drawdown_pct": 0.05,
            }
        )
    challenger_metadata["evaluation_report_hashes"] = module._evaluation_report_hashes(
        challenger_metadata
    )
    challenger_metadata["artifact_return_evidence_sha256"] = module.canonical_sha256(
        challenger_metadata["evaluation_report_hashes"]
    )
    candidate = module.persist_candidate_bundle(
        _test_artifact_bundle(),
        challenger_metadata,
    )

    comparison = module._compare_candidate_to_current(
        challenger_metadata,
        candidate_stage="active",
    )
    assert comparison["accepted"] is True
    assert comparison["reason"] == "challenger_quality_improved"

    module.activate_candidate_shadow(
        {**evidence, "champion_comparison": comparison}
    )
    module.transition_current_artifact(evidence, activation_stage="canary")
    current = module.transition_current_artifact(evidence, activation_stage="active")

    assert current["version"] == candidate["version"]
    assert current["version"] != champion["version"]
    assert current["activation_manifest"]["activation_stage"] == "active"


def test_local_ai_tools_generated_service_reports_specialist_model_chains() -> None:
    module = ModuleType("local_ai_tools_api_specialist_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)

    module._phase3_inventory_status = lambda: {
        "downloaded_model_count": 5,
        "validated_model_count": 5,
        "validation_all_ok": True,
        "model_status": [
            {"slot": "timeseries_primary", "repo_id": "amazon/chronos-2", "status": "ok"},
            {
                "slot": "timeseries_challenger",
                "repo_id": "google/timesfm-2.5-200m-transformers",
                "status": "ok",
            },
            {
                "slot": "timeseries_fallback",
                "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
                "status": "ok",
            },
            {"slot": "sentiment_primary", "repo_id": "ProsusAI/finbert", "status": "ok"},
            {
                "slot": "sentiment_challenger",
                "repo_id": "yiyanghkust/finbert-tone",
                "status": "ok",
            },
        ],
    }

    health = module.health()
    timeseries_chain = health["specialist_model_chains"]["timeseries"]
    sentiment_chain = health["specialist_model_chains"]["sentiment"]

    assert timeseries_chain["primary_model"] == "google/timesfm-2.5-200m-pytorch"
    assert timeseries_chain["challenger_model"] == "amazon/chronos-2"
    assert timeseries_chain["artifacts_ready"] is True
    assert timeseries_chain["actual_inference"] is False
    assert sentiment_chain["primary_model"] == "ProsusAI/finbert"
    assert sentiment_chain["challenger_model"] == "yiyanghkust/finbert-tone"
    assert sentiment_chain["live_mutation"] is False


def test_local_ai_tools_generated_deep_endpoints_expose_professional_shadow_chain() -> None:
    module = ModuleType("local_ai_tools_api_deep_specialist_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module.FeatureRequest.model_rebuild()
    module._phase3_inventory_status = lambda: {
        "downloaded_model_count": 5,
        "validated_model_count": 5,
        "validation_all_ok": True,
        "model_status": [
            {"slot": "timeseries_primary", "repo_id": "amazon/chronos-2", "status": "ok"},
            {
                "slot": "timeseries_challenger",
                "repo_id": "google/timesfm-2.5-200m-transformers",
                "status": "ok",
            },
            {
                "slot": "timeseries_fallback",
                "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
                "status": "ok",
            },
            {"slot": "sentiment_primary", "repo_id": "ProsusAI/finbert", "status": "ok"},
            {
                "slot": "sentiment_challenger",
                "repo_id": "yiyanghkust/finbert-tone",
                "status": "ok",
            },
        ],
    }
    req = module.FeatureRequest(
        symbol="BTC/USDT",
        features={
            "current_price": 100.0,
            "close": 100.0,
            "returns_1": 0.01,
            "returns_5": 0.02,
            "returns_20": 0.03,
            "rsi_14": 55.0,
            "volume_ratio": 1.1,
            "recent_headlines": ["crypto market risk improves"],
        },
    )

    timeseries = module.deep_timeseries_predict(req)
    sentiment = module.deep_sentiment_analyze(req)

    assert timeseries["specialist_primary_model"] == "google/timesfm-2.5-200m-pytorch"
    assert timeseries["specialist_challenger_model"] == "amazon/chronos-2"
    assert timeseries["specialist_artifacts_ready"] is True
    assert timeseries["specialist_inference_active"] is False
    assert timeseries["professional_model_shadow"]["baseline_response"] is True
    assert timeseries["shadow_payload"]["specialist_primary_model"] == "google/timesfm-2.5-200m-pytorch"
    assert (
        timeseries["fallback_reason"]
        == "specialist_timeseries_adapter_not_promoted"
    )

    assert sentiment["specialist_primary_model"] == "ProsusAI/finbert"
    assert sentiment["specialist_challenger_model"] == "yiyanghkust/finbert-tone"
    assert sentiment["specialist_artifacts_ready"] is True
    assert sentiment["specialist_inference_active"] is False
    assert sentiment["professional_model_shadow"]["actual_inference"] is False
    assert sentiment["shadow_payload"]["specialist_challenger_model"] == "yiyanghkust/finbert-tone"
    assert sentiment["fallback_reason"] == "specialist_sentiment_adapter_not_promoted"


def test_local_ai_tools_specialist_preflight_blocks_until_adapters_exist() -> None:
    module = ModuleType("local_ai_tools_api_specialist_preflight_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module._phase3_inventory_status = lambda: {
        "downloaded_model_count": 5,
        "validated_model_count": 5,
        "validation_all_ok": True,
        "model_status": [
            {"slot": "timeseries_primary", "repo_id": "amazon/chronos-2", "status": "ok"},
            {
                "slot": "timeseries_challenger",
                "repo_id": "google/timesfm-2.5-200m-transformers",
                "status": "ok",
            },
            {
                "slot": "timeseries_fallback",
                "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
                "status": "ok",
            },
            {"slot": "sentiment_primary", "repo_id": "ProsusAI/finbert", "status": "ok"},
            {
                "slot": "sentiment_challenger",
                "repo_id": "yiyanghkust/finbert-tone",
                "status": "ok",
            },
        ],
    }
    module._import_state = lambda name: {
        "module": name,
        "available": True,
        "version": "test",
    }

    report = module.specialist_preflight()

    assert report["policy"] == "phase3_specialist_adapter_preflight"
    assert report["stage"] == "preflight_only"
    assert report["live_mutation"] is False
    assert report["all_artifacts_ready"] is True
    assert report["all_required_imports_ready"] is True
    assert report["any_shadow_inference_ready"] is True
    assert "specialist_adapter_not_implemented" in report["blocked_reasons"]
    assert "walk_forward_required" in report["blocked_reasons"]
    by_slot = {row["slot"]: row for row in report["adapters"]}
    assert by_slot["sentiment_primary"]["adapter_code_ready"] is True
    assert by_slot["sentiment_challenger"]["adapter_code_ready"] is True
    assert by_slot["timeseries_primary"]["adapter_code_ready"] is True
    assert by_slot["timeseries_challenger"]["adapter_code_ready"] is True
    assert by_slot["timeseries_primary"]["shadow_inference_ready"] is True
    assert by_slot["timeseries_challenger"]["shadow_inference_ready"] is True
    assert by_slot["sentiment_primary"]["shadow_inference_ready"] is True
    assert by_slot["sentiment_challenger"]["shadow_inference_ready"] is True
    assert by_slot["timeseries_fallback"]["adapter_code_ready"] is False


def test_local_ai_tools_timesfm_shadow_adapter_marks_timeseries_shadow_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("local_ai_tools_api_timesfm_shadow_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module.FeatureRequest.model_rebuild()
    module._phase3_inventory_status = lambda: {
        "model_status": [
            {"slot": "timeseries_primary", "repo_id": "amazon/chronos-2", "status": "ok"},
            {
                "slot": "timeseries_challenger",
                "repo_id": "google/timesfm-2.5-200m-transformers",
                "status": "ok",
            },
            {
                "slot": "timeseries_fallback",
                "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
                "status": "ok",
            },
        ],
    }

    class FakeTensor:
        def __init__(self, values: object) -> None:
            self.values = values

        def reshape(self, *_args: object) -> FakeTensor:
            return self

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def float(self) -> FakeTensor:
            return self

        def tolist(self) -> object:
            return self.values

    class FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeTorch:
        float32 = "float32"

        @staticmethod
        def tensor(values: object, dtype: object | None = None) -> FakeTensor:
            return FakeTensor(values)

        @staticmethod
        def no_grad() -> FakeNoGrad:
            return FakeNoGrad()

    class FakeOutput:
        mean_predictions = FakeTensor([[101.5, 102.0, 103.0]])

    class FakeModel:
        def eval(self) -> None:
            return None

        def __call__(self, **_kwargs: object) -> FakeOutput:
            return FakeOutput()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    module._load_timesfm_model = lambda _model_dir: FakeModel()

    result = module.deep_timeseries_predict(
        module.FeatureRequest(
            symbol="BTC/USDT",
            features={
                "recent_closes": [
                    99.0 + index * ((101.2 - 99.0) / 59) for index in range(60)
                ],
                "returns_1": 0.01,
                "returns_5": 0.02,
                "returns_20": 0.03,
                "volatility_20": 0.01,
                "horizon_steps": 2,
            },
        )
    )

    assert result["specialist_inference_active"] is True
    assert result["fallback_reason"] == "specialist_timeseries_shadow_only"
    assert result["professional_model_shadow"]["actual_inference"] is True
    assert result["professional_model_shadow"]["baseline_response"] is True
    assert result["professional_model_shadow"]["shadow_result"]["model"] == (
        "timesfm-2.5-primary"
    )
    assert result["specialist_response_applied"] is False
    assert result["specialist_applied_model"] is None
    assert result["timesfm_shadow_expected_return_pct"] == pytest.approx(0.790514)
    assert result["timesfm_shadow_side"] == "long"
    assert result["model"] != "timesfm-2.5-primary"
    assert result["live_mutation"] is False
    assert result["shadow_payload"]["specialist_inference_active"] is True


def test_timeseries_forecast_quality_uses_rolling_distribution_not_fixed_move() -> None:
    module = ModuleType("local_ai_tools_api_dynamic_forecast_quality_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)

    calm_closes = [100.0 + index * 0.01 for index in range(60)]
    volatile_closes = [100.0 + ((-1) ** index) * index * 0.2 for index in range(60)]
    calm = module._rolling_forecast_quality(calm_closes, 10.0, 2)
    volatile = module._rolling_forecast_quality(volatile_closes, 10.0, 2)

    assert calm["anomalous"] is True
    assert calm["production_eligible"] is False
    assert calm["reason"] == "outside_dynamic_rolling_forecast_interval"
    assert calm["threshold_source"] == "rolling_horizon_empirical_order_statistics"
    assert calm["sample_count"] == len(calm_closes) - 2
    assert volatile["lower_return_bound_pct"] != calm["lower_return_bound_pct"]
    assert volatile["upper_return_bound_pct"] != calm["upper_return_bound_pct"]


def test_local_ai_tools_chronos_shadow_adapter_records_primary_and_challenger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("local_ai_tools_api_chronos_shadow_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module.FeatureRequest.model_rebuild()
    module._phase3_inventory_status = lambda: {
        "model_status": [
            {"slot": "timeseries_primary", "repo_id": "amazon/chronos-2", "status": "ok"},
            {
                "slot": "timeseries_challenger",
                "repo_id": "google/timesfm-2.5-200m-transformers",
                "status": "ok",
            },
        ],
    }

    class FakeTimestamp:
        @staticmethod
        def utcnow() -> object:
            return object()

    class FakeDateRange:
        def __call__(self, **kwargs: object) -> list[int]:
            return list(range(int(kwargs.get("periods") or 0)))

    class FakeFrame:
        def __init__(self, data: dict[str, object]) -> None:
            self.data = data
            self.columns = ["median"]

        def to_dict(self, orient: str) -> list[dict[str, float]]:
            assert orient == "records"
            return [{"median": 101.0}, {"median": 102.5}, {"median": 104.0}]

    class FakePandas:
        Timestamp = FakeTimestamp
        date_range = FakeDateRange()

        @staticmethod
        def DataFrame(data: dict[str, object]) -> FakeFrame:
            return FakeFrame(data)

    class FakeChronosPipeline:
        @staticmethod
        def from_pretrained(model_dir: str) -> FakeChronosPipeline:
            assert model_dir.endswith("amazon--chronos-2")
            return FakeChronosPipeline()

        def predict_df(self, _df: object, **kwargs: object) -> FakeFrame:
            assert kwargs["prediction_length"] == 2
            assert kwargs["target"] == "target"
            assert kwargs["validate_inputs"] is False
            assert kwargs["freq"] == "min"
            return FakeFrame({})

    class FakeChronosModule:
        Chronos2Pipeline = FakeChronosPipeline

    monkeypatch.setitem(sys.modules, "pandas", FakePandas)
    monkeypatch.setitem(sys.modules, "chronos", FakeChronosModule)
    module._run_timesfm_shadow = lambda _features: {
        "available": True,
        "kind": "timeseries",
        "model": "timesfm-2.5-primary",
        "primary_model": "google/timesfm-2.5-200m-pytorch",
        "challenger_model": "amazon/chronos-2",
        "artifacts_ready": True,
        "actual_inference": True,
        "expected_move_pct": 0.5,
        "expected_return_pct": 0.5,
        "best_side": "long",
        "confidence": 0.5,
        "horizon_step": 2,
        "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        "live_mutation": False,
    }

    result = module.deep_timeseries_predict(
        module.FeatureRequest(
            symbol="BTC/USDT",
            features={
                "recent_closes": [
                    99.0 + index * ((101.0 - 99.0) / 59) for index in range(60)
                ],
                "returns_1": 0.01,
                "returns_5": 0.02,
                "returns_20": 0.03,
                "volatility_20": 0.01,
                "horizon_steps": 2,
            },
        )
    )

    shadow = result["professional_model_shadow"]
    assert result["specialist_inference_active"] is True
    assert result["chronos_shadow_expected_return_pct"] == pytest.approx(1.485149)
    assert result["chronos_shadow_side"] == "long"
    assert result["timesfm_shadow_expected_return_pct"] == pytest.approx(0.5)
    assert shadow["actual_inference"] is True
    assert shadow["shadow_result"]["model"] == "timesfm-2.5-primary"
    assert shadow["primary_shadow_result"]["model"] == "timesfm-2.5-primary"
    assert shadow["challenger_shadow_result"]["model"] == "chronos-2-shadow-challenger"
    assert shadow["challenger_shadow_result"]["actual_inference"] is True
    assert shadow["activation_blocker"] == "walk_forward_required"
    assert result["fallback_reason"] == "specialist_timeseries_shadow_only"
    assert result["live_mutation"] is False


def test_local_ai_tools_chronos_shadow_adapter_falls_back_to_direct_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("local_ai_tools_api_chronos_direct_predict_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module.FeatureRequest.model_rebuild()
    module._phase3_inventory_status = lambda: {
        "model_status": [
            {"slot": "timeseries_primary", "repo_id": "amazon/chronos-2", "status": "ok"},
            {
                "slot": "timeseries_challenger",
                "repo_id": "google/timesfm-2.5-200m-transformers",
                "status": "ok",
            },
        ],
    }

    class FakeTimestamp:
        @staticmethod
        def utcnow() -> object:
            return object()

    class FakeDateRange:
        def __call__(self, **kwargs: object) -> list[int]:
            return list(range(int(kwargs.get("periods") or 0)))

    class FakeFrame:
        def __init__(self, data: dict[str, object]) -> None:
            self.data = data
            self.columns: list[str] = []

        def to_dict(self, orient: str) -> list[dict[str, float]]:
            assert orient == "records"
            return []

    class FakePandas:
        Timestamp = FakeTimestamp
        date_range = FakeDateRange()

        @staticmethod
        def DataFrame(data: dict[str, object]) -> FakeFrame:
            return FakeFrame(data)

    class FakeChronosTensor:
        def __init__(self, values: object) -> None:
            self.values = values

        def detach(self) -> FakeChronosTensor:
            return self

        def cpu(self) -> FakeChronosTensor:
            return self

        def float(self) -> FakeChronosTensor:
            return self

        def numpy(self) -> object:
            return self.values

    class FakeChronosPipeline:
        @staticmethod
        def from_pretrained(_model_dir: str) -> FakeChronosPipeline:
            return FakeChronosPipeline()

        def predict_df(self, _df: object, **_kwargs: object) -> object:
            raise TypeError("Cannot change data-type for array of references.")

        def predict(self, inputs: object, **kwargs: object) -> list[FakeChronosTensor]:
            assert kwargs["prediction_length"] == 2
            assert kwargs["limit_prediction_length"] is False
            assert len(inputs) == 1
            return [FakeChronosTensor([[[99.0, 98.0], [101.5, 102.0], [104.0, 105.0]]])]

    class FakeChronosModule:
        Chronos2Pipeline = FakeChronosPipeline

    monkeypatch.setitem(sys.modules, "pandas", FakePandas)
    monkeypatch.setitem(sys.modules, "chronos", FakeChronosModule)
    module._run_timesfm_shadow = lambda _features: {
        "available": False,
        "kind": "timeseries",
        "actual_inference": False,
        "reason": "disabled_for_test",
        "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        "live_mutation": False,
    }

    result = module.deep_timeseries_predict(
        module.FeatureRequest(
            symbol="BTC/USDT",
            features={
                "recent_closes": [
                    99.0 + index * ((101.0 - 99.0) / 59) for index in range(60)
                ],
                "returns_1": 0.01,
                "returns_5": 0.02,
                "returns_20": 0.03,
                "volatility_20": 0.01,
                "horizon_steps": 2,
            },
        )
    )

    shadow = result["professional_model_shadow"]["challenger_shadow_result"]
    assert shadow["actual_inference"] is True
    assert shadow["forecast_price"] == pytest.approx(102.0)
    assert shadow["expected_return_pct"] == pytest.approx(0.990099)
    assert shadow["adapter"] == "chronos_2_pipeline_adapter"
    assert result["chronos_shadow_expected_return_pct"] == pytest.approx(0.990099)
    assert result["live_mutation"] is False


def test_local_ai_tools_timeseries_shadow_rejects_short_synthetic_sequence() -> None:
    module = ModuleType("local_ai_tools_api_short_sequence_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module.FeatureRequest.model_rebuild()
    module._phase3_inventory_status = lambda: {
        "model_status": [
            {"slot": "timeseries_primary", "repo_id": "amazon/chronos-2", "status": "ok"},
            {
                "slot": "timeseries_challenger",
                "repo_id": "google/timesfm-2.5-200m-transformers",
                "status": "ok",
            },
        ],
    }

    result = module.deep_timeseries_predict(
        module.FeatureRequest(
            symbol="BTC/USDT",
            features={
                "close_sequence": [99.0, 100.0, 100.5, 101.0],
                "returns_1": 0.01,
                "returns_5": 0.02,
                "returns_20": 0.03,
                "volatility_20": 0.01,
            },
        )
    )

    shadow = result["professional_model_shadow"]

    assert result["specialist_inference_active"] is False
    assert shadow["actual_inference"] is False
    assert shadow["primary_shadow_result"]["reason"] == "not_enough_real_close_sequence"
    assert shadow["primary_shadow_result"]["sequence_length"] == 4
    assert shadow["primary_shadow_result"]["model_input_rows"] == 30
    assert shadow["challenger_shadow_result"]["reason"] == "not_enough_real_close_sequence"
    assert result["sequence_input_status"] == "not_enough_real_close_sequence"
    assert result["live_mutation"] is False


def test_local_ai_tools_specialist_preflight_surfaces_missing_imports() -> None:
    module = ModuleType("local_ai_tools_api_specialist_imports_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module._phase3_inventory_status = lambda: {
        "model_status": [
            {"slot": "timeseries_primary", "repo_id": "amazon/chronos-2", "status": "ok"},
        ],
    }
    module._import_state = lambda name: {
        "module": name,
        "available": name != "transformers",
        "error": "missing" if name == "transformers" else "",
    }

    report = module.specialist_preflight("timeseries")

    assert report["all_required_imports_ready"] is False
    assert "specialist_required_import_missing" in report["blocked_reasons"]
    primary = next(row for row in report["adapters"] if row["slot"] == "timeseries_primary")
    assert primary["required_imports_ready"] is False
    assert any(
        item["module"] == "transformers" and item["available"] is False
        for item in primary["required_imports"]
    )


def test_local_ai_tools_finbert_shadow_adapter_marks_sentiment_shadow_only() -> None:
    module = ModuleType("local_ai_tools_api_finbert_shadow_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module.FeatureRequest.model_rebuild()
    module._phase3_inventory_status = lambda: {
        "model_status": [
            {"slot": "sentiment_primary", "repo_id": "ProsusAI/finbert", "status": "ok"},
            {
                "slot": "sentiment_challenger",
                "repo_id": "yiyanghkust/finbert-tone",
                "status": "ok",
            },
        ],
    }
    module._run_finbert_shadow = lambda _features: {
        "available": True,
        "kind": "sentiment",
        "text_count": 1,
        "primary_model": "ProsusAI/finbert",
        "challenger_model": "yiyanghkust/finbert-tone",
        "artifacts_ready": True,
        "actual_inference": True,
        "score": 0.72,
        "label": "positive",
        "disagreement": 0.04,
        "predictions": {},
        "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        "live_mutation": False,
    }

    result = module.deep_sentiment_analyze(
        module.FeatureRequest(
            symbol="BTC/USDT",
            features={"recent_headlines": ["earnings and liquidity improve"]},
        )
    )

    assert result["model"] == "local-sentiment-light-v1"
    assert result["best_side"] == "hold"
    assert result["side"] == "hold"
    assert result["score"] == 0.0
    assert result["status"] == "specialist_shadow_inference"
    assert result["specialist_inference_active"] is True
    assert result["professional_model_shadow"]["actual_inference"] is True
    assert result["professional_model_shadow"]["baseline_response"] is True
    assert result["professional_model_shadow"]["baseline_model"] == "local-sentiment-light-v1"
    assert result["professional_model_shadow"]["score"] == 0.72
    assert result["fallback_reason"] == "specialist_sentiment_shadow_only"
    assert result["live_mutation"] is False
    assert result["shadow_payload"]["specialist_inference_active"] is True


def test_local_ai_tools_transformer_classifier_uses_bert_tokenizer_when_auto_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("local_ai_tools_api_bert_tokenizer_fallback_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    (tmp_path / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    calls: list[str] = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            calls.append("auto")
            raise ValueError("auto tokenizer failed")

    class FakeBertTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            calls.append("bert")
            return object()

    class FakeModel:
        def eval(self) -> None:
            calls.append("eval")

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> FakeModel:
            calls.append("model")
            return FakeModel()

    class FakeTransformers:
        AutoTokenizer = FakeAutoTokenizer
        BertTokenizer = FakeBertTokenizer
        BertConfig = object
        BertForSequenceClassification = object
        AutoModelForSequenceClassification = FakeAutoModel

    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers)
    module._TRANSFORMER_MODEL_CACHE.clear()

    tokenizer, model = module._load_transformer_classifier(tmp_path.as_posix())

    assert tokenizer is not None
    assert model is not None
    assert calls == ["auto", "bert", "model", "eval"]


def test_local_ai_tools_transformer_classifier_uses_bert_config_when_model_type_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("local_ai_tools_api_bert_config_fallback_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    (tmp_path / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    calls: list[str] = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            calls.append("auto_tokenizer")
            return object()

    class FakeBertTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            calls.append("bert_tokenizer")
            return object()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            calls.append("auto_model")
            raise ValueError("Should have a `model_type` key in its config.json")

    class FakeConfig:
        model_type = ""

    class FakeBertConfig:
        @staticmethod
        def from_json_file(*_args: object, **_kwargs: object) -> FakeConfig:
            calls.append("bert_config")
            return FakeConfig()

    class FakeModel:
        def eval(self) -> None:
            calls.append("eval")

    class FakeBertModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> FakeModel:
            calls.append("bert_model")
            return FakeModel()

    class FakeTransformers:
        AutoTokenizer = FakeAutoTokenizer
        BertTokenizer = FakeBertTokenizer
        AutoModelForSequenceClassification = FakeAutoModel
        BertConfig = FakeBertConfig
        BertForSequenceClassification = FakeBertModel

    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers)
    module._TRANSFORMER_MODEL_CACHE.clear()

    tokenizer, model = module._load_transformer_classifier(tmp_path.as_posix())

    assert tokenizer is not None
    assert model is not None
    assert calls == ["auto_tokenizer", "auto_model", "bert_config", "bert_model", "eval"]


def test_local_ai_tools_generated_profit_contract_has_phase3_targets() -> None:
    assert '"loss_probability": round(loss_prob, 4)' in SERVICE_CODE
    assert '"profit_quality_score": round(quality, 4)' in SERVICE_CODE
    assert '"long_loss_probability"' in SERVICE_CODE
    assert '"short_loss_probability"' in SERVICE_CODE
    assert "long_loss_prob * 0.22" not in SERVICE_CODE
    assert "short_loss_prob * 0.22" not in SERVICE_CODE
    assert "long_profile_penalty" not in SERVICE_CODE
    assert "short_profile_penalty" not in SERVICE_CODE
    assert "quality = best_lower_bound" in SERVICE_CODE


def test_local_ai_tools_generated_model_distribution_inputs_are_complete() -> None:
    assert "RETURN_DISTRIBUTION_INPUT_VERSION" in SERVICE_CODE
    assert "def model_return_distribution_input(" in SERVICE_CODE
    for field in (
        "raw_expected_return_pct",
        "median_return_pct",
        "lower_quantile_return_pct",
        "upper_quantile_return_pct",
        "dispersion_pct",
        "tail_loss_probability",
        "tail_loss_scale_pct",
        "distribution_member_count",
    ):
        assert f'"{field}"' in SERVICE_CODE


def test_trained_profit_prediction_does_not_apply_fixed_loss_or_profile_penalties() -> None:
    module = ModuleType("local_ai_tools_api_direct_return_distribution_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module.FeatureRequest.model_rebuild()
    long_model = object()
    short_model = object()
    long_loss_model = object()
    short_loss_model = object()
    long_cost_model = object()
    short_cost_model = object()
    bundle = {
        "metadata": {
            "objective_name": module.RETURN_OBJECTIVE_NAME,
            "objective_version": module.RETURN_OBJECTIVE_VERSION,
            "label_name": module.RETURN_LABEL_NAME,
            "label_version": module.RETURN_LABEL_VERSION,
            "cost_model_version": module.COST_MODEL_VERSION,
            "training_cost_policy": (
                "separated_market_opportunity_and_execution_cost_tasks"
            ),
            "profit_supervision_version": module.PROFIT_SUPERVISION_VERSION,
            "tail_loss_scale_pct": {"long": 0.4, "short": 0.3},
            "artifact_persisted": True,
        },
        "long_return_model": long_model,
        "short_return_model": short_model,
        "long_cost_model": long_cost_model,
        "short_cost_model": short_cost_model,
        "long_loss_model": long_loss_model,
        "short_loss_model": short_loss_model,
        "profiles": {
            f"BTC/USDT|{side}": {
                "source_authority": "okx_position_history",
                "symbol": "BTC/USDT",
                "side": side,
                module.PROFIT_TRAINING_TARGET: {
                    "count": 2,
                    "expected": 0.3,
                    "lower_hinge": 0.1,
                },
                "slippage_pct": {
                    "count": 2,
                    "expected": 0.02,
                    "upper_hinge": 0.04,
                },
            }
            for side in ("long", "short")
        },
    }
    module.load_bundle = lambda: bundle
    module.regression_prediction_distribution = lambda model, _x: {
        "expected": 0.4 if model is long_model else 0.2,
        "median": 0.4 if model is long_model else 0.2,
        "lower_bound": 0.1 if model is long_model else 0.05,
        "upper_bound": 0.5 if model is long_model else 0.3,
        "std": 0.03,
        "spread": 0.08,
        "sample_count": 260,
        "distribution_ready": True,
        "source_authority": "extra_trees_empirical_distribution",
    }
    module.predict_proba_positive = lambda model, _x: (
        0.95 if model is long_loss_model else 0.85
    )

    payload = module.profit_predict(
        module.FeatureRequest(
            symbol="BTC/USDT",
            features={"symbol": "BTC/USDT", "current_price": 100.0},
        )
    )

    assert payload["profit_quality_score"] == 0.1
    assert payload["long_loss_probability"] == 0.95
    assert payload["return_distribution_input_version"] == (
        module.RETURN_DISTRIBUTION_INPUT_VERSION
    )
    long_input = payload["return_distribution_inputs"]["long"]
    assert long_input["raw_expected_return_pct"] == 0.4
    assert long_input["median_return_pct"] == 0.4
    assert long_input["lower_quantile_return_pct"] == 0.1
    assert long_input["upper_quantile_return_pct"] == 0.5
    assert long_input["dispersion_pct"] == 0.03
    assert long_input["tail_loss_probability"] == 0.95
    assert long_input["tail_loss_scale_pct"] == 0.4
    assert long_input["distribution_member_count"] == 260

    module.regression_prediction_distribution = lambda model, _x: {
        "expected": 0.4 if model is long_model else 0.2,
        "median": 0.4 if model is long_model else 0.2,
        "lower_bound": 0.4 if model is long_model else 0.2,
        "upper_bound": 0.4 if model is long_model else 0.2,
        "std": 0.0,
        "spread": 0.0,
        "sample_count": 260,
        "distribution_ready": False,
        "source_authority": "extra_trees_empirical_distribution",
    }
    degenerate = module.profit_predict(
        module.FeatureRequest(
            symbol="BTC/USDT",
            features={"symbol": "BTC/USDT", "current_price": 100.0},
        )
    )

    assert degenerate["prediction_quality"]["production_eligible"] is False
    assert degenerate["prediction_quality"]["anomalous"] is True
    assert degenerate["prediction_quality"]["reason"] == (
        "current_tree_prediction_distribution_degenerate"
    )

    def lower_above_expected(model: object, _x: object) -> dict[str, object]:
        expected = 0.46 if model is long_model else 0.2
        lower = 0.496 if model is long_model else 0.05
        return {
            "expected": expected,
            "median": expected,
            "lower_bound": lower,
            "upper_bound": 0.6 if model is long_model else 0.3,
            "std": 0.03,
            "spread": 0.08,
            "sample_count": 260,
            "distribution_ready": True,
            "source_authority": "extra_trees_empirical_distribution",
        }

    module.regression_prediction_distribution = lower_above_expected
    invalid_ordering = module.profit_predict(
        module.FeatureRequest(
            symbol="ICP/USDT",
            features={"symbol": "ICP/USDT", "current_price": 2.2},
        )
    )

    invalid_long = invalid_ordering["return_distribution_inputs"]["long"]
    assert invalid_long["raw_expected_return_pct"] == 0.46
    assert invalid_long["lower_quantile_return_pct"] == 0.496
    assert invalid_long["production_eligible"] is False
    assert "lower_quantile_above_raw_expected" in invalid_long["blockers"]
    assert invalid_ordering["prediction_quality"]["production_eligible"] is False
    assert invalid_ordering["prediction_quality"]["reason"] == (
        "lower_quantile_above_raw_expected"
    )

    module.regression_prediction_distribution = lambda model, _x: {
        "expected": 0.4 if model is long_model else 0.2,
        "median": 0.35 if model is long_model else 0.15,
        "lower_bound": 0.1 if model is long_model else 0.05,
        "upper_bound": 0.5 if model is long_model else 0.3,
        "std": 0.03,
        "spread": 0.08,
        "sample_count": 260,
        "distribution_ready": True,
        "source_authority": "extra_trees_empirical_distribution",
    }
    bundle["profiles"] = {}
    missing_calibration = module.profit_predict(
        module.FeatureRequest(
            symbol="DOGE/USDT",
            features={"symbol": "DOGE/USDT", "current_price": 0.2},
        )
    )

    assert missing_calibration["prediction_quality"]["production_eligible"] is False
    assert missing_calibration["prediction_quality"]["reason"] == (
        "actual_trade_calibration_not_ready"
    )
    assert "actual_trade_calibration_not_ready" in missing_calibration[
        "prediction_quality"
    ]["blockers"]
    assert "artifact_activation_not_production_authorized" in missing_calibration[
        "prediction_quality"
    ]["blockers"]


def test_local_ai_tools_generated_exit_contract_uses_only_phase3_actions() -> None:
    assert '"action": "no_position"' not in SERVICE_CODE
    assert 'action = "close_if_ai_agrees"' not in SERVICE_CODE
    assert '"action": "hold"' in SERVICE_CODE
    assert 'action = "reduce_or_close"' not in SERVICE_CODE
    assert 'action = "protect_profit"' not in SERVICE_CODE
    assert 'action = "trail_profit"' not in SERVICE_CODE
    assert '"production_permission": False' in SERVICE_CODE
    assert "dynamic_exit_policy_owns_production_exit" in SERVICE_CODE


def test_local_ai_tools_generated_service_does_not_use_wildcard_cors() -> None:
    assert 'allow_origins=["*"]' not in SERVICE_CODE
    assert "LOCAL_AI_TOOLS_CORS_ORIGINS" in SERVICE_CODE
    assert "allow_origins=LOCAL_AI_TOOLS_CORS_ORIGINS" in SERVICE_CODE
    assert "allow_credentials=bool(LOCAL_AI_TOOLS_API_KEY)" in SERVICE_CODE


def test_local_ai_tools_generated_service_uses_trusted_model_artifact_boundary() -> None:
    assert "def _trusted_model_artifact_path(path: Path) -> Path:" in SERVICE_CODE
    assert 'target.suffix != ".joblib"' in SERVICE_CODE
    assert "target.is_relative_to(root)" in SERVICE_CODE
    assert "def load_trusted_joblib_bundle(path: Path) -> dict[str, Any]:" in SERVICE_CODE
    assert "not isinstance(value, dict)" in SERVICE_CODE
    assert (
        "def dump_trusted_joblib_bundle(bundle: dict[str, Any], path: Path) -> Path:"
        in SERVICE_CODE
    )
    assert "tempfile.NamedTemporaryFile" in SERVICE_CODE
    assert "os.replace(tmp_path, target)" in SERVICE_CODE
    assert "local_quant_models.joblib" not in SERVICE_CODE
    assert "local_quant_models_metadata.json" not in SERVICE_CODE
    assert "def persist_candidate_bundle(" in SERVICE_CODE
    assert "def activate_candidate_shadow(" in SERVICE_CODE
    assert "def transition_current_artifact(" in SERVICE_CODE
    assert "def _governed_candidate_activation_stage(" in SERVICE_CODE
    assert "def rollback_current_artifact(" in SERVICE_CODE


def test_local_ai_tools_generated_service_persists_training_cursors() -> None:
    assert "completed_shadow_sample_count: int | None = None" in SERVICE_CODE
    assert (
        '"completed_shadow_sample_count": int(req.completed_shadow_sample_count or len(rows))'
        in SERVICE_CODE
    )
    assert '"last_trained_completed_shadow_sample_count": int(' in SERVICE_CODE
    assert '"completed_trade_sample_count": int(' in SERVICE_CODE
    assert '"last_trained_completed_trade_sample_count": int(' in SERVICE_CODE
    assert 'training_mode: str = "shadow"' in SERVICE_CODE
    assert 'model_stage: str = "shadow"' not in SERVICE_CODE
    assert "promotion_recommendation: dict[str, Any] = {}" in SERVICE_CODE
    assert '"training_mode": str(req.training_mode or "shadow")' in SERVICE_CODE
    assert "requested_model_stage" not in SERVICE_CODE
    assert '"model_stage": "candidate"' in SERVICE_CODE
    assert '"promotion_recommendation": req.promotion_recommendation or {}' in SERVICE_CODE
    assert (
        'PHASE3_REQUIRED_PROMOTION_FLOW = "candidate_to_shadow_to_canary_to_active"'
        in SERVICE_CODE
    )
    assert '"promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW' in SERVICE_CODE
    assert '"live_mutation": live_authorized' in SERVICE_CODE


def test_local_ai_tools_generated_service_persists_phase3_artifact_policy() -> None:
    assert 'PHASE3_ARTIFACT_POLICY_ID = "phase3_clean_training_artifact_v1"' in SERVICE_CODE
    assert 'PHASE3_REQUIRED_TRAINING_POLICY = "clean_training_view_only"' in SERVICE_CODE
    assert '"artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID' in SERVICE_CODE
    assert '"phase": "phase3_model_factory"' in SERVICE_CODE
    assert '"training_policy": PHASE3_REQUIRED_TRAINING_POLICY' in SERVICE_CODE
    assert '"trade_sample_cursor_policy": PHASE3_REQUIRED_TRAINING_POLICY' in SERVICE_CODE
    assert '"artifact_persisted": bool(req.persist_artifact and req.confirm_phase3_rebuild)' in SERVICE_CODE
    assert '"preflight_only": not bool(req.persist_artifact and req.confirm_phase3_rebuild)' in SERVICE_CODE
    assert '"persist_artifact_requested": bool(req.persist_artifact)' in SERVICE_CODE
    assert '"confirm_phase3_rebuild": bool(req.confirm_phase3_rebuild)' in SERVICE_CODE
    assert 'if not req.persist_artifact:' in SERVICE_CODE
    assert '"reason": "phase3_preflight_no_artifact_write"' in SERVICE_CODE
    assert 'if not req.confirm_phase3_rebuild:' in SERVICE_CODE
    assert '"reason": "phase3_rebuild_confirmation_required"' in SERVICE_CODE
    assert "evaluation_policy" not in SERVICE_CODE


def test_local_ai_tools_generated_service_uses_side_specific_fee_after_return_targets() -> None:
    assert 'RETURN_OBJECTIVE_NAME = "maximize_expected_realized_net_return_after_cost"' in SERVICE_CODE
    assert '"objective_name": RETURN_OBJECTIVE_NAME' in SERVICE_CODE
    assert '"label_version": RETURN_LABEL_VERSION' in SERVICE_CODE
    assert 'metadata.get("objective_version") != RETURN_OBJECTIVE_VERSION' in SERVICE_CODE
    assert 'max(r["long_return"], r["short_return"], key=abs)' not in SERVICE_CODE
    assert (
        'max(net_return_pct(f(sample, "long_return_pct")), '
        'net_return_pct(f(sample, "short_return_pct")))'
        not in SERVICE_CODE
    )
    assert '"long_model": long_model' in SERVICE_CODE
    assert '"short_model": short_model' in SERVICE_CODE
    assert '"raw_expected_return_pct": finite_or_none(distribution.get("expected"))' in SERVICE_CODE
    assert '"return_distribution_inputs": return_distribution_inputs' in SERVICE_CODE
    assert '"training_data_sha256": training_data_sha256' in SERVICE_CODE
    assert '"source_code_sha256": source_code_sha256' in SERVICE_CODE
    assert '"time_split_policy": "chronological_disjoint_decision_groups"' in SERVICE_CODE
    assert "persist_candidate_bundle(bundle, metadata)" in SERVICE_CODE
    assert "activate_candidate_shadow(" in SERVICE_CODE
    assert "local_quant_models.joblib" not in SERVICE_CODE
    assert "local_quant_models_metadata.json" not in SERVICE_CODE


def _local_ai_tools_training_module(tmp_path: Path) -> ModuleType:
    module = ModuleType("local_ai_tools_api_training_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module.FeatureRequest.model_rebuild()
    module.TrainRequest.model_rebuild()

    class DummyModel:
        def fit(self, *_args: object, **_kwargs: object) -> DummyModel:
            return self

        def predict(self, rows: list[list[float]]) -> list[float]:
            return [0.0] * len(rows)

        def predict_proba(self, rows: list[list[float]]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in rows]

        @property
        def classes_(self) -> list[int]:
            return [0, 1]

    module._make_regressor = lambda _sample_count: DummyModel()
    module._make_classifier = lambda _y: DummyModel()
    module._train_sequence_model = lambda _samples: None
    module._train_torch_patch_model = lambda _samples: {"available": False, "reason": "test"}
    module._train_text_sentiment_model = lambda _samples: None
    module._probe_transformers_sentiment_backend = lambda: {
        "available": False,
        "reason": "test",
    }
    _configure_local_ai_registry(module, tmp_path)
    return module


def _training_shadow_samples(count: int = 200) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    first_label_at = datetime(2026, 7, 14, tzinfo=UTC)
    for index in range(count):
        samples.append(
            {
                "id": index + 1,
                "decision_id": index + 1,
                "symbol": "BTC/USDT",
                "horizon_minutes": 10,
                "label_timestamp": (
                    first_label_at + timedelta(minutes=index * 11)
                ).isoformat(),
                "features": {
                    "symbol": "BTC/USDT",
                    "current_price": 100.0 + index * 0.01,
                    "close": 100.0 + index * 0.01,
                    "returns_1": 0.01,
                    "returns_5": 0.02,
                    "returns_20": 0.03,
                    "rsi_14": 55.0,
                    "volume_ratio": 1.1,
                    "spread_pct": 0.02,
                    "funding_rate": 0.0001,
                    "funding_interval_hours": 8.0,
                    "round_trip_fee_pct": 0.1,
                },
                "long_return_pct": 0.45,
                "short_return_pct": -0.15,
                "sample_weight": 1.0,
                "correlation_weight": {
                    "correlation_group": f"shadow_decision:{index + 1}",
                },
                "profit_supervision": {
                    "version": "2026-07-14.separated-profit-supervision.v1",
                    "tasks": {
                        "market_opportunity_distribution": {
                            "eligible": True,
                            "long_gross_market_return_pct": 0.45,
                            "short_gross_market_return_pct": -0.15,
                        },
                        "execution_cost_and_slippage_distribution": {
                            "eligible": True,
                            "long_total_cost_pct": 0.12,
                            "short_total_cost_pct": 0.08,
                        },
                        "authoritative_realized_return_distribution": {
                            "eligible": False,
                        },
                    },
                },
            }
        )
    return samples


def test_local_ai_tools_generated_service_train_defaults_to_preflight_only(
    tmp_path: Path,
) -> None:
    module = _local_ai_tools_training_module(tmp_path)
    module.dump_trusted_joblib_bundle = lambda *_args, **_kwargs: pytest.fail(
        "preflight must not write a model bundle"
    )

    result = module.train(module.TrainRequest(shadow_samples=_training_shadow_samples()))

    assert result["trained"] is False
    assert result["reason"] == "phase3_preflight_no_artifact_write"
    assert result["artifact_persisted"] is False
    assert result["preflight_only"] is True
    assert result["persist_artifact_requested"] is False
    assert result["confirm_phase3_rebuild"] is False
    assert not module.CANDIDATE_POINTER_PATH.exists()
    assert not module.CURRENT_POINTER_PATH.exists()


def test_local_ai_tools_training_marks_positive_gross_but_negative_net_as_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _local_ai_tools_training_module(tmp_path)
    samples = _training_shadow_samples(count=2)
    captured: list[dict[str, object]] = []

    def chronological_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        captured.extend(rows)
        return rows

    monkeypatch.setattr(module, "_chronological_rows", chronological_rows)
    for sample in samples:
        tasks = sample["profit_supervision"]["tasks"]
        tasks[module.MARKET_OPPORTUNITY_TASK]["long_gross_market_return_pct"] = 0.05
        tasks[module.MARKET_OPPORTUNITY_TASK]["short_gross_market_return_pct"] = -0.05
        tasks[module.EXECUTION_COST_TASK]["long_total_cost_pct"] = 0.12
        tasks[module.EXECUTION_COST_TASK]["short_total_cost_pct"] = 0.08

    module.train(module.TrainRequest(shadow_samples=samples))

    assert captured
    assert all(row["long_net_return"] < 0.0 for row in captured)
    assert all(row["short_net_return"] < 0.0 for row in captured)
    assert all(row["best_side"] == "hold" for row in captured)


def test_training_builds_each_observed_horizon_without_a_fixed_sample_gate(
    tmp_path: Path,
) -> None:
    module = _local_ai_tools_training_module(tmp_path)

    result = module.train(
        module.TrainRequest(shadow_samples=_training_shadow_samples(count=2))
    )

    assert result["horizons"] == [10]


def test_local_ai_walk_forward_refits_disjoint_chronological_decision_groups(
    tmp_path: Path,
) -> None:
    module = _local_ai_tools_training_module(tmp_path)

    result = module.train(
        module.TrainRequest(shadow_samples=_training_shadow_samples(count=20))
    )

    report = result["walk_forward_report"]
    assert report["status"] == "complete"
    assert report["model_refit_per_fold"] is True
    assert report["decision_group_disjoint"] is True
    assert report["chronological_label_disjoint"] is True
    assert report["chronological"] is True
    assert report["folds"]
    assert all(
        fold["decision_group_overlap_count"] == 0 for fold in report["folds"]
    )
    assert all(
        fold["training_label_end"] < fold["validation_decision_start"]
        and fold["label_timestamp_overlap_count"] == 0
        for fold in report["folds"]
    )
    assert all(
        side_report["training_tail_loss_policy"]["observation_window"]
        == "walk_forward_training_groups_only"
        for fold in report["folds"]
        for side_report in fold["sides"].values()
    )
    assert result["evaluation_report_hashes"] == module._evaluation_report_hashes(
        result
    )
    assert result["artifact_return_evidence_sha256"] == module.canonical_sha256(
        result["evaluation_report_hashes"]
    )


def test_local_ai_walk_forward_purges_unavailable_multi_horizon_groups(
    tmp_path: Path,
) -> None:
    module = _local_ai_tools_training_module(tmp_path)
    decision_start = datetime(2026, 7, 14, tzinfo=UTC)
    samples: list[dict[str, object]] = []
    sample_id = 0
    for decision_index in range(30):
        decision_at = decision_start + timedelta(minutes=decision_index * 5)
        for horizon in (10, 60):
            sample_id += 1
            sample = _training_shadow_samples(count=1)[0]
            sample.update(
                {
                    "id": sample_id,
                    "decision_id": decision_index + 1,
                    "horizon_minutes": horizon,
                    "decision_timestamp": decision_at.isoformat(),
                    "label_timestamp": (
                        decision_at + timedelta(minutes=horizon)
                    ).isoformat(),
                    "correlation_weight": {
                        "correlation_group": f"shadow_decision:{decision_index + 1}"
                    },
                }
            )
            samples.append(sample)

    result = module.train(module.TrainRequest(shadow_samples=samples))
    report = result["walk_forward_report"]

    assert report["status"] == "complete"
    assert report["chronological_label_disjoint"] is True
    assert any(
        fold["purged_training_decision_group_count"] > 0
        for fold in report["folds"]
    )
    assert all(
        fold["training_label_end"] < fold["validation_decision_start"]
        and fold["decision_group_overlap_count"] == 0
        for fold in report["folds"]
    )


def test_local_ai_training_rejects_missing_label_or_decision_identity(
    tmp_path: Path,
) -> None:
    module = _local_ai_tools_training_module(tmp_path)
    missing_label = _training_shadow_samples(count=2)
    missing_label[0]["label_timestamp"] = None
    missing_decision = _training_shadow_samples(count=2)
    missing_decision[0]["id"] = 0
    missing_decision[0]["decision_id"] = None
    missing_decision[0]["correlation_weight"] = {}

    for samples in (missing_label, missing_decision):
        result = module.train(module.TrainRequest(shadow_samples=samples))
        assert result["trained"] is False
        assert result["reason"] == "chronological_training_identity_incomplete"


def test_local_ai_tools_generated_service_uses_profit_target_distribution(
    tmp_path: Path,
) -> None:
    module = _local_ai_tools_training_module(tmp_path)
    assert module.f({module.PROFIT_TRAINING_TARGET: 0.0}, module.PROFIT_TRAINING_TARGET, float("nan")) == 0.0

    sample = _training_shadow_samples(count=1)[0]
    sample["profit_supervision"]["tasks"][module.AUTHORITATIVE_REALIZED_RETURN_TASK] = {
        "eligible": True,
        module.PROFIT_TRAINING_TARGET: 1.25,
        "realized_net_return_pct": 9.99,
        "stop_loss_slippage_pct": 0.03,
        "hold_minutes": 42,
        "side": "long",
    }

    profiles = module._train_profiles([sample])
    profile = profiles["BTC/USDT|long"]

    assert profile[module.PROFIT_TRAINING_TARGET]["count"] == 1
    assert profile[module.PROFIT_TRAINING_TARGET]["expected"] == pytest.approx(1.25)
    assert "net_return_after_cost_pct" not in profile

    old_only = deepcopy(sample)
    realized = old_only["profit_supervision"]["tasks"][
        module.AUTHORITATIVE_REALIZED_RETURN_TASK
    ]
    del realized[module.PROFIT_TRAINING_TARGET]
    realized["realized_net_return_pct"] = 9.99

    assert module._train_profiles([old_only]) == {}


def test_local_ai_walk_forward_tail_policy_ignores_future_extreme_returns(
    tmp_path: Path,
) -> None:
    module = _local_ai_tools_training_module(tmp_path)
    baseline = _training_shadow_samples(count=18)
    for index, sample in enumerate(baseline):
        market_task = sample["profit_supervision"]["tasks"][
            module.MARKET_OPPORTUNITY_TASK
        ]
        market_task["long_gross_market_return_pct"] = -0.4 if index < 6 else 0.5
    changed = [
        {
            **sample,
            "features": dict(sample["features"]),
            "profit_supervision": {
                **sample["profit_supervision"],
                "tasks": {
                    key: dict(value)
                    for key, value in sample["profit_supervision"]["tasks"].items()
                },
            },
        }
        for sample in baseline
    ]
    for sample in changed[12:]:
        sample["profit_supervision"]["tasks"][module.MARKET_OPPORTUNITY_TASK][
            "long_gross_market_return_pct"
        ] = -999.0

    baseline_report = module.train(
        module.TrainRequest(shadow_samples=baseline)
    )["walk_forward_report"]
    changed_report = module.train(
        module.TrainRequest(shadow_samples=changed)
    )["walk_forward_report"]

    assert (
        baseline_report["folds"][0]["sides"]["long"][
            "training_tail_loss_policy"
        ]
        == changed_report["folds"][0]["sides"]["long"][
            "training_tail_loss_policy"
        ]
    )


def test_local_ai_training_data_hash_binds_features_and_text_inputs(
    tmp_path: Path,
) -> None:
    module = _local_ai_tools_training_module(tmp_path)
    baseline = _training_shadow_samples(count=2)
    changed_features = _training_shadow_samples(count=2)
    changed_features[0]["features"]["returns_1"] = 9.5

    baseline_hash = module.train(
        module.TrainRequest(
            shadow_samples=baseline,
            text_sentiment_samples=[
                {"id": 1, "text": "cost is controlled", "sentiment_score": 0.2}
            ],
        )
    )["training_data_sha256"]
    feature_hash = module.train(
        module.TrainRequest(
            shadow_samples=changed_features,
            text_sentiment_samples=[
                {"id": 1, "text": "cost is controlled", "sentiment_score": 0.2}
            ],
        )
    )["training_data_sha256"]
    text_hash = module.train(
        module.TrainRequest(
            shadow_samples=baseline,
            text_sentiment_samples=[
                {"id": 1, "text": "tail loss expanded", "sentiment_score": -0.2}
            ],
        )
    )["training_data_sha256"]

    assert feature_hash != baseline_hash
    assert text_hash != baseline_hash


def test_local_ai_profit_factor_is_undefined_without_loss_denominator() -> None:
    module = ModuleType("local_ai_tools_api_profit_factor_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)

    evidence = module._return_evidence(
        [
            {
                "decision_group": f"decision:{index}",
                "label_timestamp": f"2026-07-14T00:0{index}:00+00:00",
                "return_pct": value,
            }
            for index, value in enumerate((0.2, 0.4, 0.1))
        ]
    )

    assert module._profit_factor([0.2, 0.4, 0.1]) is None
    assert evidence["profit_factor"] is None
    assert evidence["promotion_math_ready"] is False


def test_local_ai_leave_one_symbol_out_blocks_single_symbol_profit_support() -> None:
    module = ModuleType("local_ai_tools_api_loso_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    rows = [
        {
            "symbol": "ROBO/USDT",
            "decision_group": f"robo:{index}",
            "label_timestamp": f"2026-07-14T00:{index:02d}:00+00:00",
            "return_pct": -0.1 if index == 0 else 2.0,
            "score": 100.0 + index,
        }
        for index in range(10)
    ] + [
        {
            "symbol": "BTC/USDT",
            "decision_group": f"btc:{index}",
            "label_timestamp": f"2026-07-14T01:{index % 60:02d}:00+00:00",
            "return_pct": -1.0,
            "score": float(index),
        }
        for index in range(90)
    ]

    report = module._leave_one_symbol_out_stability(rows)
    robo_removed = next(
        row for row in report["rows"] if row["excluded_symbol"] == "ROBO/USDT"
    )

    assert report["stable"] is False
    assert robo_removed["evidence"]["promotion_math_ready"] is False
    assert robo_removed["evidence"]["return_lcb_pct"] < 0.0


def test_local_ai_tools_generated_service_requires_confirmed_phase3_rebuild(
    tmp_path: Path,
) -> None:
    module = _local_ai_tools_training_module(tmp_path)
    module.dump_trusted_joblib_bundle = lambda *_args, **_kwargs: pytest.fail(
        "unconfirmed rebuild must not write a model bundle"
    )

    result = module.train(
        module.TrainRequest(
            shadow_samples=_training_shadow_samples(),
            persist_artifact=True,
        )
    )

    assert result["trained"] is False
    assert result["reason"] == "phase3_rebuild_confirmation_required"
    assert result["artifact_persisted"] is False
    assert result["preflight_only"] is True
    assert result["persist_artifact_requested"] is True
    assert result["confirm_phase3_rebuild"] is False
    assert not module.CANDIDATE_POINTER_PATH.exists()
    assert not module.CURRENT_POINTER_PATH.exists()


def test_local_ai_tools_generated_service_confirmed_rebuild_persists_metadata(
    tmp_path: Path,
) -> None:
    module = _local_ai_tools_training_module(tmp_path)

    stored: dict[str, object] = {}

    def dump_bundle(bundle: dict[str, object], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-bundle")
        assert bundle["metadata"]["artifact_persisted"] is True
        stored["bundle"] = bundle
        return path

    module.dump_trusted_joblib_bundle = dump_bundle
    module.load_trusted_joblib_bundle = lambda _path: stored["bundle"]

    result = module.train(
        module.TrainRequest(
            shadow_samples=_training_shadow_samples(),
            persist_artifact=True,
            confirm_phase3_rebuild=True,
        )
    )

    assert result["trained"] is True
    assert result["artifact_persisted"] is True
    assert result["preflight_only"] is False
    assert result["persist_artifact_requested"] is True
    assert result["confirm_phase3_rebuild"] is True
    current = module._resolve_artifact_pointer(
        module.CURRENT_POINTER_PATH,
        role="current",
    )
    metadata = current["metadata"]
    assert metadata["artifact_policy_id"] == "phase3_clean_training_artifact_v1"
    assert metadata["artifact_persisted"] is True
    assert metadata["preflight_only"] is False
    assert metadata["walk_forward_report"]["status"] == "complete"
    assert metadata["walk_forward_report"]["model_refit_per_fold"] is True
    assert metadata["walk_forward_report"]["decision_group_disjoint"] is True
    assert set(metadata["leave_one_symbol_out_report"]) == {"long", "short"}
    assert set(metadata["oos_return_evaluation"]) == {"long", "short"}
    assert metadata["authoritative_trade_return_evidence"]["sample_count"] == 0
    assert metadata["evaluation_report_hashes"] == module._evaluation_report_hashes(
        metadata
    )
    assert current["activation_manifest"]["activation_stage"] == "shadow"
    assert current["activation_manifest"]["live_ml_ready"] is False
    assert current["activation_manifest"]["return_evidence_ready"] is False
    assert current["activation_manifest"]["return_evidence_blockers"]
    assert not module.CANDIDATE_POINTER_PATH.exists()


def test_local_ai_tools_generated_service_uses_quality_weights() -> None:
    assert 'sample_weight = max(0.0, f(sample, "sample_weight", 1.0))' in SERVICE_CODE
    assert "if sample_weight <= 0.0:" in SERVICE_CODE
    assert "model__sample_weight=sample_weights" in SERVICE_CODE
    assert '"quality_report": req.quality_report or {}' in SERVICE_CODE
    assert "trainable_trade_samples = [" in SERVICE_CODE
    assert 'if not bool(sample.get("exclude_from_training"))' in SERVICE_CODE
    assert 'PROFIT_TRAINING_TARGET = "net_return_after_all_cost_pct"' in SERVICE_CODE


def test_local_ai_tools_systemd_uses_env_file_for_secrets() -> None:
    source = (ROOT / "scripts" / "deploy_local_ai_tools_service.py").read_text(encoding="utf-8")

    assert "EnvironmentFile=-{PHASE3_ENV_FILE}" in source
    assert "chmod 600 {sh(PHASE3_ENV_FILE)}" in source
    assert "LOCAL_AI_TOOLS_ALLOW_UNAUTHENTICATED_LOOPBACK=true" in source
    assert (
        "LOCAL_AI_TOOLS_CORS_ORIGINS=http://127.0.0.1:8002,http://localhost:8002,"
        "http://127.0.0.1:18001"
    ) in source
    assert "LOCAL_AI_TOOLS_ROUND_TRIP_COST_PCT" not in source
    assert "LOCAL_AI_TOOLS_TAIL_LOSS_THRESHOLD_PCT" not in source
    assert "LimitNOFILE=65535" in source
    assert "--timeout-keep-alive 30" in source


def test_phase3_quant_api_deploy_contract_uses_data_bb_and_8101() -> None:
    source = (ROOT / "scripts" / "deploy_local_ai_tools_service.py").read_text(encoding="utf-8")
    service = deploy.render_phase3_quant_api_service()
    plan = deploy.render_phase3_deploy_plan()

    assert plan["phase3_root"] == "/data/BB"
    assert plan["service_name"] == "bb-phase3-quant-api.service"
    assert plan["port"] == 8101
    assert plan["health_url"] == "http://127.0.0.1:8101/health"
    assert plan["shadow_only"] is True
    assert plan["live_mutation"] is False
    assert plan["legacy_root_used"] is False

    assert "WorkingDirectory=/data/BB/services/phase3_quant_api" in service
    assert "Environment=BB_PHASE3_ROOT=/data/BB" in service
    assert "Environment=PHASE3_QUANT_API_PORT=8101" in service
    assert "Environment=LOCAL_AI_TOOLS_MODEL_DIR=/data/BB/models/local_ai_tools" in service
    assert "EnvironmentFile=-/data/BB/env/phase3.env" in service
    assert "--host 127.0.0.1 --port 8101" in service
    assert "KillMode=mixed" in service
    assert "TimeoutStopSec=20" in service
    assert "bb-phase3-quant-api.service" in source
    assert "/data/trade_ai" not in source
    assert "local-ai-tools.service" not in source


def test_phase3_quant_api_remote_smoke_checks_governed_activation_contract() -> None:
    command = deploy._remote_smoke_command()

    assert "http://127.0.0.1:8101" in command
    assert "ENV_FILE = '/data/BB/env/phase3.env'" in command
    assert "Authorization" in command
    assert "health.get('service') == 'phase3_quant_api'" in command
    assert "health.get('root') == '/data/BB'" in command
    assert "health.get('live_mutation') is live" in command
    assert (
        "health.get('artifact_lifecycle') in {'shadow', 'canary', 'active'}"
        in command
    )
    assert "health.get('live_ml_ready') is live" in command
    assert "profit.get('trained') is True" in command
    assert "profit.get('shadow_payload', {}).get('tool') == 'profit_prediction'" in command
    assert "profit.get('return_distribution_input_version')" in command
    assert "profit.get('return_distribution_inputs')" in command
    assert "profit.get('production_permission') is live" in command
    assert "item.get('production_eligible') is live" in command
    assert "'loss_probability' in profit" in command
