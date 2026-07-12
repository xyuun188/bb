from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scripts import deploy_local_ai_tools_service as deploy
from scripts.deploy_local_ai_tools_service import SERVICE_CODE
from scripts.fix_local_ai_tools_service_path import (
    normalize_remote_python_path,
    render_local_ai_tools_service,
)

ROOT = Path(__file__).resolve().parents[1]


def test_local_ai_tools_generated_service_requires_api_key_or_loopback() -> None:
    assert "LOCAL_AI_TOOLS_API_KEY = os.environ.get" in SERVICE_CODE
    assert "def require_api_key(" in SERVICE_CODE
    assert "dependencies=[Depends(require_api_key)]" in SERVICE_CODE
    assert "LOCAL_AI_TOOLS_API_KEY is required for non-loopback access" in SERVICE_CODE
    assert "Bearer {LOCAL_AI_TOOLS_API_KEY}" in SERVICE_CODE


def test_local_ai_tools_training_costs_are_runtime_configurable() -> None:
    assert "LOCAL_AI_TOOLS_ROUND_TRIP_COST_PCT" in SERVICE_CODE
    assert "LOCAL_AI_TOOLS_TAIL_LOSS_THRESHOLD_PCT" in SERVICE_CODE
    assert "ROUND_TRIP_COST_PCT = 0.12" not in SERVICE_CODE
    assert "TAIL_LOSS_THRESHOLD_PCT = 0.18" not in SERVICE_CODE


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
            "model": "local-profit-heuristic-v1",
            "best_side": "long",
            "expected_return_pct": 0.42,
            "adjusted_expected_return_pct": 0.31,
            "loss_probability": 0.18,
            "profit_quality_score": 0.25,
        },
        features={"returns_1": 0.1, "rsi_14": 55},
    )

    assert payload["promotion_flow"] == "shadow_to_canary_to_live"
    assert payload["live_mutation"] is False
    assert payload["shadow_payload"]["tool"] == "profit_prediction"
    assert payload["shadow_payload"]["live_mutation"] is False
    assert payload["shadow_payload"]["expected_return_pct"] == 0.42
    assert payload["shadow_payload"]["adjusted_expected_return_pct"] == 0.31
    assert payload["shadow_payload"]["loss_probability"] == 0.18


def test_local_ai_tools_status_endpoints_do_not_load_joblib_bundle(tmp_path: Path) -> None:
    module = ModuleType("local_ai_tools_api_metadata_status_test")
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module.MODEL_DIR = tmp_path
    module.BUNDLE_PATH = tmp_path / "local_quant_models.joblib"
    module.METADATA_PATH = tmp_path / "local_quant_models_metadata.json"
    module.BUNDLE_PATH.write_bytes(b"not-a-joblib-bundle")
    module.METADATA_PATH.write_text(
        json.dumps(
            {
                "trained_at": "2026-06-30T08:00:00+00:00",
                "shadow_sample_count": 222,
                "trade_sample_count": 33,
                "profile_count": 7,
                "artifact_persisted": True,
            }
        ),
        encoding="utf-8",
    )

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

        def reshape(self, *_args: object) -> "FakeTensor":
            return self

        def detach(self) -> "FakeTensor":
            return self

        def cpu(self) -> "FakeTensor":
            return self

        def float(self) -> "FakeTensor":
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
        def from_pretrained(model_dir: str) -> "FakeChronosPipeline":
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
        "promotion_flow": "shadow_to_canary_to_live",
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

        def detach(self) -> "FakeChronosTensor":
            return self

        def cpu(self) -> "FakeChronosTensor":
            return self

        def float(self) -> "FakeChronosTensor":
            return self

        def numpy(self) -> object:
            return self.values

    class FakeChronosPipeline:
        @staticmethod
        def from_pretrained(_model_dir: str) -> "FakeChronosPipeline":
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
        "promotion_flow": "shadow_to_canary_to_live",
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
    assert shadow["primary_shadow_result"]["minimum_sequence_length"] == 30
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
        "promotion_flow": "shadow_to_canary_to_live",
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
    assert '"adjusted_expected_return_pct": round(best_expected, 4)' in SERVICE_CODE
    assert '"loss_probability": round(loss_prob, 4)' in SERVICE_CODE
    assert '"profit_quality_score": round(quality, 4)' in SERVICE_CODE
    assert '"long_loss_probability"' in SERVICE_CODE
    assert '"short_loss_probability"' in SERVICE_CODE
    assert '"expected_return_pct": round(best_expected, 4)' in SERVICE_CODE


def test_local_ai_tools_generated_exit_contract_uses_only_phase3_actions() -> None:
    assert '"action": "no_position"' not in SERVICE_CODE
    assert 'action = "close_if_ai_agrees"' not in SERVICE_CODE
    assert '"action": "hold"' in SERVICE_CODE
    assert 'action = "reduce_or_close"' in SERVICE_CODE
    assert 'action = "protect_profit"' in SERVICE_CODE
    assert 'action = "trail_profit"' in SERVICE_CODE


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
    assert "joblib.load(BUNDLE_PATH)" not in SERVICE_CODE
    assert "joblib.dump(bundle, BUNDLE_PATH)" not in SERVICE_CODE


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
    assert 'model_stage: str = "shadow"' in SERVICE_CODE
    assert "promotion_recommendation: dict[str, Any] = {}" in SERVICE_CODE
    assert '"training_mode": str(req.training_mode or "shadow")' in SERVICE_CODE
    assert '"model_stage": str(req.model_stage or "shadow")' in SERVICE_CODE
    assert '"promotion_recommendation": req.promotion_recommendation or {}' in SERVICE_CODE
    assert 'PHASE3_REQUIRED_PROMOTION_FLOW = "shadow_to_canary_to_live"' in SERVICE_CODE
    assert 'evaluation_policy.setdefault("promotion_flow", PHASE3_REQUIRED_PROMOTION_FLOW)' in SERVICE_CODE


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
    assert 'evaluation_policy.setdefault("promotion_flow", PHASE3_REQUIRED_PROMOTION_FLOW)' in SERVICE_CODE
    assert 'evaluation_policy.setdefault("live_mutation", False)' in SERVICE_CODE
    assert 'evaluation_policy.setdefault("phase", "phase3_model_factory")' in SERVICE_CODE


def _local_ai_tools_training_module(tmp_path: Path) -> ModuleType:
    module = ModuleType("local_ai_tools_api_training_test")
    sys.modules[module.__name__] = module
    exec(compile(SERVICE_CODE, "local_ai_tools_api.py", "exec"), module.__dict__)
    module.FeatureRequest.model_rebuild()
    module.TrainRequest.model_rebuild()

    class DummyModel:
        def fit(self, *_args: object, **_kwargs: object) -> "DummyModel":
            return self

    module._make_regressor = DummyModel
    module._make_classifier = lambda _y: DummyModel()
    module._train_sequence_model = lambda _samples: None
    module._train_torch_patch_model = lambda _samples: {"available": False, "reason": "test"}
    module._train_text_sentiment_model = lambda _samples: None
    module._probe_transformers_sentiment_backend = lambda: {
        "available": False,
        "reason": "test",
    }
    module.MODEL_DIR = tmp_path
    module.BUNDLE_PATH = tmp_path / "local_quant_models.joblib"
    module.METADATA_PATH = tmp_path / "local_quant_models_metadata.json"
    return module


def _training_shadow_samples(count: int = 200) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for index in range(count):
        samples.append(
            {
                "id": index + 1,
                "symbol": "BTC/USDT",
                "horizon_minutes": 10,
                "features": {
                    "symbol": "BTC/USDT",
                    "current_price": 100.0 + index * 0.01,
                    "close": 100.0 + index * 0.01,
                    "returns_1": 0.01,
                    "returns_5": 0.02,
                    "returns_20": 0.03,
                    "rsi_14": 55.0,
                    "volume_ratio": 1.1,
                },
                "long_return_pct": 0.45,
                "short_return_pct": -0.15,
                "sample_weight": 1.0,
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
    assert not module.BUNDLE_PATH.exists()
    assert not module.METADATA_PATH.exists()


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
    assert not module.BUNDLE_PATH.exists()
    assert not module.METADATA_PATH.exists()


def test_local_ai_tools_generated_service_confirmed_rebuild_persists_metadata(
    tmp_path: Path,
) -> None:
    module = _local_ai_tools_training_module(tmp_path)

    def dump_bundle(bundle: dict[str, object], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-bundle")
        assert bundle["metadata"]["artifact_persisted"] is True
        return path

    module.dump_trusted_joblib_bundle = dump_bundle

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
    metadata = json.loads(module.METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["artifact_policy_id"] == "phase3_clean_training_artifact_v1"
    assert metadata["artifact_persisted"] is True
    assert metadata["preflight_only"] is False


def test_local_ai_tools_generated_service_uses_quality_weights() -> None:
    assert '"sample_weight": max(0.0, min(f(sample, "sample_weight", 1.0), 1.0))' in SERVICE_CODE
    assert "model__sample_weight=sample_weights" in SERVICE_CODE
    assert '"quality_report": req.quality_report or {}' in SERVICE_CODE
    assert "trainable_trade_samples = [" in SERVICE_CODE
    assert 'if not bool(sample.get("exclude_from_training"))' in SERVICE_CODE


def test_local_ai_tools_systemd_uses_env_file_for_secrets() -> None:
    source = (ROOT / "scripts" / "deploy_local_ai_tools_service.py").read_text(encoding="utf-8")

    assert "EnvironmentFile=-{PHASE3_ENV_FILE}" in source
    assert "chmod 600 {sh(PHASE3_ENV_FILE)}" in source
    assert "LOCAL_AI_TOOLS_ALLOW_UNAUTHENTICATED_LOOPBACK=true" in source
    assert (
        "LOCAL_AI_TOOLS_CORS_ORIGINS=http://127.0.0.1:8002,http://localhost:8002,"
        "http://127.0.0.1:18001"
    ) in source
    assert "LOCAL_AI_TOOLS_ROUND_TRIP_COST_PCT=0.12" in source
    assert "LOCAL_AI_TOOLS_TAIL_LOSS_THRESHOLD_PCT=0.18" in source
    assert "LimitNOFILE=65535" in source
    assert "--timeout-keep-alive 5" in source


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
    assert "bb-phase3-quant-api.service" in source
    assert "/data/trade_ai" not in source
    assert "local-ai-tools.service" not in source


def test_phase3_quant_api_remote_smoke_checks_shadow_contract() -> None:
    command = deploy._remote_smoke_command()

    assert "http://127.0.0.1:8101" in command
    assert "ENV_FILE = '/data/BB/env/phase3.env'" in command
    assert "Authorization" in command
    assert "health.get('service') == 'phase3_quant_api'" in command
    assert "health.get('root') == '/data/BB'" in command
    assert "health.get('live_mutation') is False" in command
    assert "profit.get('shadow_payload', {}).get('tool') == 'profit_prediction'" in command
    assert "'adjusted_expected_return_pct' in profit" in command
    assert "'loss_probability' in profit" in command


def test_local_ai_tools_fix_service_uses_remote_posix_python_path() -> None:
    service = render_local_ai_tools_service("/home/linux/anaconda3/envs/trade_ml/bin/python\n")

    assert "Environment=PATH=/home/linux/anaconda3/envs/trade_ml/bin:" in service
    assert (
        "ExecStart=/home/linux/anaconda3/envs/trade_ml/bin/python -m uvicorn "
        "local_ai_tools_api:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 5"
    ) in service
    assert "LimitNOFILE=65535" in service
    assert "EnvironmentFile=-/data/trade_ai/local_ai_tools.env" in service
    assert "LOCAL_AI_TOOLS_ALLOW_UNAUTHENTICATED_LOOPBACK=true" in service
    assert "qwen3-32b-main.service" in service
    assert "\\home\\linux" not in service


def test_normalize_remote_python_path_rejects_unsafe_values() -> None:
    assert (
        normalize_remote_python_path(
            "\n/home/linux/anaconda3/envs/trade_ml/bin/python\n/usr/bin/python3\n"
        )
        == "/home/linux/anaconda3/envs/trade_ml/bin/python"
    )
    with pytest.raises(ValueError, match="absolute"):
        normalize_remote_python_path("python3")
    with pytest.raises(ValueError, match="unsupported"):
        normalize_remote_python_path("/home/linux/bad path/python")
