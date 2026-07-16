from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from config.settings import settings
from data_feed.feature_vector import FeatureVector
from services.local_ai_tools_client import LocalAIToolsClient
from services.profit_supervision import PROFIT_SUPERVISION_VERSION
from services.return_objective import (
    COST_MODEL_VERSION,
    RETURN_DISTRIBUTION_CONTRACT_VERSION,
    RETURN_DISTRIBUTION_INPUT_VERSION,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_VERSION,
)


def _distribution_input(side: str, expected: float, lower: float) -> dict[str, Any]:
    return {
        "side": side,
        "horizon_minutes": 30,
        "raw_expected_return_pct": expected,
        "median_return_pct": expected,
        "lower_quantile_return_pct": lower,
        "upper_quantile_return_pct": expected + 0.2,
        "dispersion_pct": abs(expected - lower),
        "tail_loss_probability": 0.1,
        "tail_loss_scale_pct": 0.3,
        "distribution_member_count": 64,
        "return_semantics": "gross_market_opportunity_before_execution",
        "source_authority": "extra_trees_empirical_distribution",
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_version": RETURN_LABEL_VERSION,
        "cost_model_version": COST_MODEL_VERSION,
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
    }


@pytest.fixture
def local_tools_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_enabled", True)
    monkeypatch.setattr(settings, "local_ai_tools_api_base", "http://local-ai-tools.test")
    monkeypatch.setattr(settings, "local_ai_tools_api_key", "")
    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 0.5)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_failures", 2)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_cooldown_seconds", 30.0)


@pytest.mark.asyncio
async def test_local_ai_tools_circuit_breaker_opens_after_total_failures(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    calls: list[str] = []

    async def fail(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        calls.append(path)
        raise RuntimeError("local tools unavailable")

    monkeypatch.setattr(client, "_post", fail)

    first = await client.enrich_with_context({"symbol": "BTC/USDT"})
    second = await client.enrich_with_context({"symbol": "ETH/USDT"})
    third = await client.enrich_with_context({"symbol": "SOL/USDT"})

    assert first["status"] == "unavailable"
    assert first["failure_count"] == 1
    assert second["status"] == "unavailable"
    assert second["failure_count"] == 2
    assert second.get("circuit_open_until")
    assert third["status"] == "circuit_open"
    assert third["available"] is False
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_local_ai_tools_circuit_breaker_recovers_after_cooldown(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    client._failure_count = 2
    client._circuit_open_until = datetime.now(UTC) - timedelta(seconds=1)

    async def succeed(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_post", succeed)

    result = await client.enrich_with_context({"symbol": "BTC/USDT"})

    assert result["status"] == "completed"
    assert result["failure_count"] == 0
    assert result["profit_prediction"]["available"] is True
    assert result["profit_prediction"]["status"] == "returned"
    assert result["profit_prediction"]["path"] == "/profit/predict"
    assert result["profit_prediction"]["duration_sec"] > 0
    assert result["time_series_prediction"]["side"] == "long"


@pytest.mark.asyncio
async def test_local_ai_tools_enrich_uses_configured_timeout_without_three_second_cap(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 8.0)
    client = LocalAIToolsClient()
    timeouts: list[float | None] = []

    async def succeed(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        timeouts.append(request_timeout)
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_post", succeed)

    result = await client.enrich_with_context({"symbol": "BTC/USDT"})

    assert result["status"] == "completed"
    assert timeouts == [8.0, 8.0, 8.0]
    assert result["profit_prediction"]["duration_sec"] > 0
    assert result["time_series_prediction"]["duration_sec"] > 0
    assert result["sentiment_analysis"]["duration_sec"] > 0


def test_local_ai_tools_feature_payload_preserves_real_timeseries_sequence() -> None:
    client = LocalAIToolsClient()
    features = FeatureVector(
        symbol="BTC/USDT",
        close_sequence=[float(index) for index in range(120)],
        volume_sequence=[float(index * 10) for index in range(120)],
        sequence_timeframe="1m",
    )

    payload = client._feature_payload(features)
    snapshot = payload["features"]

    assert payload["symbol"] == "BTC/USDT"
    assert snapshot["close_sequence"] == [float(index) for index in range(40, 120)]
    assert snapshot["volume_sequence"] == [float(index * 10) for index in range(40, 120)]
    assert snapshot["sequence_timeframe"] == "1m"
    assert snapshot["sequence_length"] == 80


def _healthy_paper_observation() -> dict[str, object]:
    return {
        "status": "healthy",
        "paper_active": True,
        "can_use_for_promotion": True,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "blockers": [],
        "warnings": [],
    }


@pytest.mark.asyncio
async def test_local_ai_tools_train_sends_training_cursors(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    captured: dict[str, Any] = {}

    async def succeed(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        captured["path"] = path
        captured["payload"] = payload
        captured["request_timeout"] = request_timeout
        return {"trained": True}

    monkeypatch.setattr(client, "_post", succeed)

    result = await client.train(
        [{"id": 1}],
        [{"id": 2}],
        completed_shadow_sample_count=1234,
        completed_trade_sample_count=56,
        raw_trade_sample_count=80,
        trainable_trade_sample_count=56,
        quarantined_trade_sample_count=24,
        trade_sample_cursor_policy="clean_training_view_only",
        promotion_recommendation={
            "policy": "phase3_shadow_to_canary_to_live",
            "recommended_stage": "shadow",
        },
    )

    assert result["trained"] is True
    assert captured["path"] == "/train"
    assert captured["payload"]["completed_shadow_sample_count"] == 1234
    assert captured["payload"]["completed_trade_sample_count"] == 56
    assert captured["payload"]["raw_trade_sample_count"] == 80
    assert captured["payload"]["trainable_trade_sample_count"] == 56
    assert captured["payload"]["quarantined_trade_sample_count"] == 24
    assert captured["payload"]["trade_sample_cursor_policy"] == "clean_training_view_only"
    assert captured["payload"]["training_mode"] == "shadow"
    assert captured["payload"]["model_stage"] == "shadow"
    assert captured["payload"]["evaluation_policy"]["promotion_flow"] == "shadow_to_canary_to_live"
    assert captured["payload"]["evaluation_policy"]["live_mutation"] is False
    assert captured["payload"]["evaluation_policy"]["phase"] == "phase3_model_factory"
    assert captured["payload"]["persist_artifact"] is False
    assert captured["payload"]["confirm_phase3_rebuild"] is False
    assert captured["payload"]["return_objective_report"]["objective_name"] == (
        "maximize_expected_realized_net_return_after_cost"
    )
    assert captured["payload"]["promotion_recommendation"]["recommended_stage"] == "shadow"
    assert captured["request_timeout"] == 180.0


@pytest.mark.asyncio
async def test_local_ai_tools_train_can_explicitly_request_confirmed_rebuild(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    captured: dict[str, Any] = {}

    async def succeed(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        captured["path"] = path
        captured["payload"] = payload
        return {"trained": True, "artifact_persisted": True}

    monkeypatch.setattr(client, "_post", succeed)

    result = await client.train(
        [{"id": 1}],
        [{"id": 2}],
        persist_artifact=True,
        confirm_phase3_rebuild=True,
    )

    assert result == {"trained": True, "artifact_persisted": True}
    assert captured["path"] == "/train"
    assert captured["payload"]["persist_artifact"] is True
    assert captured["payload"]["confirm_phase3_rebuild"] is True
    assert captured["payload"]["evaluation_policy"]["phase"] == "phase3_model_factory"
    assert captured["payload"]["evaluation_policy"]["live_mutation"] is False


@pytest.mark.asyncio
async def test_local_ai_tools_train_builds_default_promotion_recommendation(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    captured: dict[str, Any] = {}

    async def succeed(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        captured["payload"] = payload
        return {"trained": True}

    monkeypatch.setattr(client, "_post", succeed)

    await client.train(
        [{"id": 1}],
        [{"id": 2}],
        completed_shadow_sample_count=150,
        completed_trade_sample_count=30,
        quality_report={"totals": {"total": 180, "excluded": 0, "effective_weight_ratio": 0.9}},
        governance_report={"trainable_sample_count": 180, "contamination_risk": "low"},
        paper_observation_report=_healthy_paper_observation(),
    )

    recommendation = captured["payload"]["promotion_recommendation"]
    return_objective_report = captured["payload"]["return_objective_report"]
    assert recommendation["policy"] == "2026-07-14.separated-return-promotion.v2"
    assert recommendation["canary_ready"] is False
    assert "authoritative_realized_return_distribution_missing" in recommendation[
        "canary_blocking_reasons"
    ]
    assert recommendation["live_ready"] is False
    assert "walk_forward_required" in recommendation["live_blocking_reasons"]
    assert captured["payload"]["paper_observation_report"]["status"] == "healthy"
    assert return_objective_report["objective_name"] == (
        "maximize_expected_realized_net_return_after_cost"
    )


@pytest.mark.asyncio
async def test_local_ai_tools_readtimeout_does_not_open_circuit(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 8.0)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_failures", 2)
    client = LocalAIToolsClient()
    calls: list[str] = []

    async def timeout(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        calls.append(path)
        raise RuntimeError("local AI tools request could not reach the service: ReadTimeout")

    monkeypatch.setattr(client, "_post", timeout)

    first = await client.enrich_with_context({"symbol": "BTC/USDT"})
    second = await client.enrich_with_context({"symbol": "ETH/USDT"})
    third = await client.enrich_with_context({"symbol": "SOL/USDT"})

    assert first["status"] == "unavailable"
    assert second["status"] == "unavailable"
    assert third["status"] == "unavailable"
    assert third["profit_prediction"]["status"] == "error"
    assert third["profit_prediction"]["path"] == "/profit/predict"
    assert third["profit_prediction"]["duration_sec"] > 0
    assert "circuit_open_until" not in third
    assert len(calls) == 9


def test_local_ai_tools_localizes_request_timeouts_and_keeps_them_soft() -> None:
    client = LocalAIToolsClient()

    message = client._request_error_message(httpx.ReadTimeout("read timed out"))

    assert message == "服务器量化工具读取响应超时"
    assert client._is_soft_timeout_failure(message) is True


def test_local_ai_tools_client_refreshes_runtime_settings(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()

    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 1.25)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_failures", 5)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_cooldown_seconds", 90.0)

    assert client.enabled() is True
    assert client._timeout == 1.25
    assert client._failure_threshold == 5
    assert client._cooldown_seconds == 90.0


def test_local_ai_tools_client_clamps_runtime_settings(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()

    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 999.0)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_failures", 999)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_cooldown_seconds", 99999.0)

    assert client.enabled() is True
    assert client._timeout == 15.0
    assert client._failure_threshold == 20
    assert client._cooldown_seconds == 3600.0

    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 0.01)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_failures", -5)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_cooldown_seconds", 0.01)

    assert client.enabled() is True
    assert client._timeout == 0.2
    assert client._failure_threshold == 1
    assert client._cooldown_seconds == 0.2


def test_local_ai_tools_normalizes_wrapped_prediction_payloads(
    local_tools_settings: None,
) -> None:
    client = LocalAIToolsClient()

    profit = client._normalize_signal(
        "profit_prediction",
        {
            "ok": True,
            "data": {
                "prediction": {
                    "predicted_side": "short",
                    "expected_short_return_pct": 0.42,
                    "expected_long_return_pct": -0.18,
                }
            },
        },
    )
    timeseries = client._normalize_signal(
        "time_series_prediction",
        {"status": "ok", "result": {"forecast_direction": "up", "expected_move_pct": 0.16}},
    )
    sentiment = client._normalize_signal(
        "sentiment_analysis",
        {"available": True, "payload": {"sentiment": "bearish", "sentiment_score": -0.31}},
    )

    assert profit["available"] is True
    assert profit["best_side"] == "short"
    assert "expected_return_pct" not in profit
    assert profit["prediction_quality"]["production_eligible"] is False
    assert timeseries["side"] == "long"
    assert "expected_return_pct" not in timeseries
    assert timeseries["prediction_quality"]["production_eligible"] is False
    assert sentiment["side"] == "short"
    assert sentiment["available"] is True
    assert profit["primary_model"] == "profit_v1_baseline"
    assert profit["model_version"] == "local_ai_tools.v1"
    assert profit["route_mode"] == "shadow_observation"
    assert profit["feature_coverage"] == {"ratio": None, "status": "not_reported"}
    assert timeseries["primary_model"] == "timeseries_v1_baseline"
    assert sentiment["primary_model"] == "sentiment_v1_baseline"


def test_local_ai_tools_builds_standard_contract_from_remote_distribution_inputs(
    local_tools_settings: None,
) -> None:
    client = LocalAIToolsClient()

    profit = client._normalize_signal(
        "profit_prediction",
        {
            "available": True,
            "best_side": "long",
            "return_distribution_input_version": RETURN_DISTRIBUTION_INPUT_VERSION,
            "return_distribution_inputs": {
                "long": _distribution_input("long", 0.8, 0.5),
                "short": _distribution_input("short", 0.2, 0.1),
            },
            "prediction_quality": {
                "production_eligible": True,
                "anomalous": False,
            },
        },
    )

    contract = profit["return_distribution_contract"]["long"]
    assert profit["return_distribution_contract_version"] == (
        RETURN_DISTRIBUTION_CONTRACT_VERSION
    )
    assert contract["raw_expected_return_pct"] == pytest.approx(0.8)
    assert contract["lower_quantile_return_pct"] == pytest.approx(0.5)
    assert contract["objective_expected_return_pct"] == pytest.approx(0.47)
    assert contract["production_eligible"] is True
    assert profit["prediction_quality"]["production_eligible"] is True


def test_local_ai_tools_blocks_remote_lower_above_expected_without_clamping(
    local_tools_settings: None,
) -> None:
    client = LocalAIToolsClient()

    profit = client._normalize_signal(
        "profit_prediction",
        {
            "available": True,
            "best_side": "long",
            "return_distribution_input_version": RETURN_DISTRIBUTION_INPUT_VERSION,
            "return_distribution_inputs": {
                "long": _distribution_input("long", 0.46, 0.496),
                "short": _distribution_input("short", 0.2, 0.1),
            },
            "prediction_quality": {
                "production_eligible": True,
                "anomalous": False,
            },
        },
    )

    contract = profit["return_distribution_contract"]["long"]
    assert contract["raw_expected_return_pct"] == pytest.approx(0.46)
    assert contract["lower_quantile_return_pct"] == pytest.approx(0.496)
    assert "lower_quantile_above_raw_expected" in contract["blockers"]
    assert contract["production_eligible"] is False
    assert profit["prediction_quality"]["production_eligible"] is False


def test_local_ai_tools_preserves_remote_distribution_blockers(
    local_tools_settings: None,
) -> None:
    client = LocalAIToolsClient()

    profit = client._normalize_signal(
        "profit_prediction",
        {
            "available": True,
            "best_side": "long",
            "return_distribution_input_version": RETURN_DISTRIBUTION_INPUT_VERSION,
            "return_distribution_inputs": {
                "long": _distribution_input("long", 0.8, 0.5),
                "short": _distribution_input("short", 0.2, 0.1),
            },
            "prediction_quality": {
                "production_eligible": False,
                "anomalous": True,
                "reason": "actual_trade_calibration_not_ready",
                "blockers": ["actual_trade_calibration_not_ready"],
            },
        },
    )

    assert profit["prediction_quality"]["production_eligible"] is False
    assert profit["prediction_quality"]["reason"] == (
        "actual_trade_calibration_not_ready"
    )
    assert profit["prediction_quality"]["blockers"] == [
        "actual_trade_calibration_not_ready"
    ]


def test_local_ai_tools_blocks_obsolete_distribution_provenance(
    local_tools_settings: None,
) -> None:
    client = LocalAIToolsClient()
    long_input = _distribution_input("long", 0.8, 0.5)
    long_input["cost_model_version"] = "obsolete-cost-model"

    profit = client._normalize_signal(
        "profit_prediction",
        {
            "available": True,
            "best_side": "long",
            "return_distribution_input_version": RETURN_DISTRIBUTION_INPUT_VERSION,
            "return_distribution_inputs": {
                "long": long_input,
                "short": _distribution_input("short", 0.2, 0.1),
            },
            "prediction_quality": {
                "production_eligible": True,
                "anomalous": False,
            },
        },
    )

    assert profit["prediction_quality"]["production_eligible"] is False
    assert "return_distribution_cost_model_version_mismatch" in profit[
        "prediction_quality"
    ]["blockers"]


def test_local_ai_tools_preserves_server_reported_model_metadata(
    local_tools_settings: None,
) -> None:
    client = LocalAIToolsClient()

    profit = client._normalize_signal(
        "profit_prediction",
        {
            "best_side": "long",
            "expected_long_return_pct": 0.5,
            "primary_model": "catboost_lgbm_profit_v2",
            "challenger_model": "xgboost_profit_shadow",
            "model_version": "profit-v2.20260626",
            "route_mode": "shadow",
            "fallback_reason": "baseline_live_only",
            "feature_coverage": 0.75,
        },
    )

    assert profit["primary_model"] == "catboost_lgbm_profit_v2"
    assert profit["challenger_model"] == "xgboost_profit_shadow"
    assert profit["model_version"] == "profit-v2.20260626"
    assert profit["route_mode"] == "shadow"
    assert profit["fallback_reason"] == "baseline_live_only"
    assert profit["feature_coverage"] == {"ratio": 0.75, "status": "reported"}


def test_local_ai_tools_exit_advice_uses_clean_chinese_labels(
    local_tools_settings: None,
) -> None:
    client = LocalAIToolsClient()

    hold = client._normalize_signal(
        "exit_advice",
        {"action": "hold", "reason": "no trained exit pressure"},
    )
    reduce = client._normalize_signal(
        "exit_advice",
        {
            "action": "reduce",
            "reason": "profit exists but historical giveback/loss pressure is elevated",
        },
    )
    unknown = client._normalize_signal(
        "exit_advice",
        {
            "recommendation": "unexpected_model_token",
            "note": "no matching open position was supplied",
        },
    )

    assert hold["action"] == "hold"
    assert hold["reported_action"] == "hold"
    assert hold["production_permission"] is False
    assert reduce["action"] == "hold"
    assert reduce["reported_action"] == "reduce"
    assert reduce["production_permission"] is False
    assert unknown["action_label"] == "继续观察"
    assert unknown["reported_action"] == "unexpected_model_token"
    assert unknown["production_permission"] is False


@pytest.mark.asyncio
async def test_local_ai_tools_train_returns_structured_failure(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()

    async def fail(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError("service refused training request")

    monkeypatch.setattr(client, "_post", fail)

    result = await client.train([{"symbol": "BTC/USDT"}], [{"symbol": "BTC/USDT"}])

    assert result["trained"] is False
    assert result["reason"] == "request_failed"
    assert result["error"] == "service refused training request"
    assert result["failure_count"] == 1


@pytest.mark.asyncio
async def test_local_ai_tools_enrich_failure_fields_are_redacted(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    client = LocalAIToolsClient()

    async def fail(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"Authorization: Bearer {leaked_value} failed")

    monkeypatch.setattr(client, "_post", fail)

    result = await client.enrich_with_context({"symbol": "BTC/USDT"})

    assert result["status"] == "unavailable"
    assert leaked_value not in str(result)
    assert result["errors"]["profit_prediction"] == "Authorization: *** failed"
    assert result["profit_prediction"]["error"] == "Authorization: *** failed"
    assert client._last_failure == (
        "Authorization: *** failed; Authorization: *** failed; " "Authorization: *** failed"
    )


@pytest.mark.asyncio
async def test_local_ai_tools_status_failure_is_redacted(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    client = LocalAIToolsClient()

    async def fail(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        raise RuntimeError(f"Authorization: Bearer {leaked_value} failed")

    monkeypatch.setattr(client, "_get", fail)

    result = await client.status()

    assert result["status"] == "error"
    assert leaked_value not in str(result)
    assert result["error"] == "Authorization: *** failed"
    assert result["api_base"] == "http://local-ai-tools.test"
    assert client._last_failure == "Authorization: *** failed"
    assert client._circuit_open_until is None


@pytest.mark.asyncio
async def test_local_ai_tools_status_uses_child_endpoint_health_when_bundle_missing(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    get_calls: list[str] = []
    post_calls: list[str] = []

    async def get_status(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        get_calls.append(path)
        assert request_timeout == 0.5
        if path == "/health":
            return {"ok": True, "service": "phase3_quant_api", "trained_models_available": False}
        return {
            "available": False,
            "message": "No trained local quant bundle found",
            "child_endpoints": {
                "profit_prediction": {
                    "available": False,
                    "path": "/profit/predict",
                    "probe_mode": "metadata_contract",
                    "actual_inference_probe": False,
                }
            },
        }

    async def post_probe(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        post_calls.append(path)
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_get", get_status)
    monkeypatch.setattr(client, "_post", post_probe)

    result = await client.status()

    assert get_calls == ["/models/status", "/health"]
    assert post_calls == []
    assert result["available"] is True
    assert result["model_bundle_available"] is False
    assert result["service_available"] is True
    assert result["api_base"] == "http://local-ai-tools.test"
    assert result["enabled_for_trading"] is True
    assert result["status"] == "artifact_unavailable"
    assert result["child_endpoints"]["profit_prediction"]["available"] is False
    assert result["child_endpoints"]["profit_prediction"]["probe_mode"] == (
        "metadata_contract"
    )
    assert result["child_endpoints"]["profit_prediction"]["actual_inference_probe"] is False


@pytest.mark.asyncio
async def test_local_ai_tools_status_probes_service_when_trading_influence_disabled(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_enabled", False)
    client = LocalAIToolsClient()

    async def get_status(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        if path == "/health":
            return {"ok": True, "service": "phase3_quant_api"}
        return {"available": False, "message": "No trained local quant bundle found"}

    async def post_probe(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        return {"available": True, "path": path}

    monkeypatch.setattr(client, "_get", get_status)
    monkeypatch.setattr(client, "_post", post_probe)

    result = await client.status()

    assert result["available"] is True
    assert result["service_available"] is True
    assert result["enabled_for_trading"] is False
    assert result["status"] == "connected_trading_disabled"
    assert client.enabled() is False


@pytest.mark.asyncio
async def test_local_ai_tools_status_defaults_ready_when_bundle_is_available(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()

    async def get_status(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        if path == "/health":
            return {"ok": True, "service": "phase3_quant_api"}
        assert path == "/models/status"
        return {
            "available": True,
            "trained_at": "2026-06-23T16:58:10+00:00",
            "models": {"profit": "trained"},
        }

    async def post_probe(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        return {"available": True, "path": path}

    monkeypatch.setattr(client, "_get", get_status)
    monkeypatch.setattr(client, "_post", post_probe)

    result = await client.status()

    assert result["available"] is True
    assert result["model_bundle_available"] is True
    assert result["service_available"] is True
    assert result["status"] == "ready"
    assert result["trained_at"] == "2026-06-23T16:58:10+00:00"
    assert result["health_available"] is True


@pytest.mark.asyncio
async def test_local_ai_tools_status_preserves_health_supervision_and_route_contract(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    supervision = {
        "shadow_market_sample_count": 14,
        "shadow_counterfactual_cost_sample_count": 14,
        "actual_execution_cost_sample_count": 1,
        "actual_realized_return_sample_count": 61,
    }

    async def get_status(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        if path == "/models/status":
            return {"available": True, "status": "ready"}
        return {
            "ok": True,
            "service": "phase3_quant_api",
            "objective_version": "separated-objective-v2",
            "label_version": "separated-label-v2",
            "cost_model_version": "authoritative-cost-v2",
            "profit_supervision_version": "separated-supervision-v1",
            "profit_supervision_report": supervision,
            "route_mode": "shadow_observation",
            "live_mutation": False,
            "artifact_persisted": True,
        }

    async def post_probe(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        return {"available": True, "path": path}

    monkeypatch.setattr(client, "_get", get_status)
    monkeypatch.setattr(client, "_post", post_probe)

    result = await client.status()

    assert result["profit_supervision_report"] == supervision
    assert result["objective_version"] == "separated-objective-v2"
    assert result["label_version"] == "separated-label-v2"
    assert result["cost_model_version"] == "authoritative-cost-v2"
    assert result["route_mode"] == "shadow_observation"
    assert result["live_mutation"] is False
    assert result["artifact_persisted"] is True


@pytest.mark.asyncio
async def test_local_ai_tools_status_uses_child_endpoint_health_when_status_fails(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()

    async def fail_status(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        assert request_timeout == 0.5
        if path == "/health":
            return {
                "ok": True,
                "service": "phase3_quant_api",
                "child_endpoints": {
                    "profit_prediction": {
                        "available": True,
                        "path": "/profit/predict",
                        "probe_mode": "metadata_contract",
                        "actual_inference_probe": False,
                    }
                },
            }
        raise RuntimeError("models status endpoint unavailable")

    async def post_probe(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        if path == "/profit/predict":
            return {"available": True, "best_side": "long"}
        raise RuntimeError(f"{path} unavailable")

    monkeypatch.setattr(client, "_get", fail_status)
    monkeypatch.setattr(client, "_post", post_probe)

    result = await client.status()

    assert result["available"] is True
    assert result["service_available"] is True
    assert result["model_bundle_available"] is False
    assert result["status"] == "artifact_unavailable"
    assert result["status_error"] == "models status endpoint unavailable"
    assert result["health_available"] is True
    assert result["child_endpoints"]["profit_prediction"]["available"] is True
    assert result["failure_count"] == 0


@pytest.mark.asyncio
async def test_local_ai_tools_status_uses_health_when_status_and_bundle_missing(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    get_calls: list[str] = []

    async def get_status(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        get_calls.append(path)
        if path == "/models/status":
            raise RuntimeError("models status endpoint unavailable")
        return {
            "ok": True,
            "service": "phase3_quant_api",
            "trained_models_available": False,
            "shadow_sample_count": 0,
            "completed_shadow_sample_count": 0,
        }

    async def post_probe(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"{path} unavailable")

    monkeypatch.setattr(client, "_get", get_status)
    monkeypatch.setattr(client, "_post", post_probe)

    result = await client.status()

    assert get_calls == ["/models/status", "/health"]
    assert result["available"] is True
    assert result["service_available"] is True
    assert result["model_bundle_available"] is False
    assert result["status"] == "artifact_unavailable"
    assert result["health_available"] is True
    assert result["failure_count"] == 0


@pytest.mark.asyncio
async def test_local_ai_tools_status_uses_short_cache(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    get_calls: list[str] = []
    post_calls: list[str] = []

    async def get_status(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        get_calls.append(path)
        assert request_timeout == 0.5
        if path == "/health":
            return {"ok": True, "service": "phase3_quant_api"}
        return {
            "available": False,
            "message": "No trained local quant bundle found",
            "child_endpoints": {
                "profit_prediction": {
                    "available": True,
                    "path": "/profit/predict",
                    "probe_mode": "metadata_contract",
                    "actual_inference_probe": False,
                }
            },
        }

    async def post_probe(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        post_calls.append(path)
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_get", get_status)
    monkeypatch.setattr(client, "_post", post_probe)

    first = await client.status()
    first["child_endpoints"]["profit_prediction"]["available"] = False
    second = await client.status()

    assert get_calls == ["/models/status", "/health"]
    assert post_calls == []
    assert second["status_cache"]["hit"] is True
    assert second["child_endpoints"]["profit_prediction"]["available"] is True


def test_local_ai_tools_auth_headers_close_connections(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_api_key", "  local-secret-token  ")
    headers = LocalAIToolsClient()._auth_headers()

    assert headers == {
        "Authorization": "Bearer local-secret-token",
        "Connection": "close",
    }


@pytest.mark.asyncio
async def test_local_ai_tools_train_failure_is_redacted(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    client = LocalAIToolsClient()

    async def fail(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"Authorization: Bearer {leaked_value} failed")

    monkeypatch.setattr(client, "_post", fail)

    result = await client.train([{"symbol": "BTC/USDT"}], [{"symbol": "BTC/USDT"}])

    assert result["trained"] is False
    assert result["reason"] == "request_failed"
    assert leaked_value not in str(result)
    assert result["error"] == "Authorization: *** failed"
    assert client._last_failure == "Authorization: *** failed"


def test_local_ai_tools_client_rejects_credentials_in_base_url(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "local_ai_tools_api_base",
        "http://user:password@127.0.0.1:8001",
    )

    with pytest.raises(RuntimeError, match="must not include credentials"):
        LocalAIToolsClient()._api_base()


@pytest.mark.asyncio
async def test_local_ai_tools_public_payload_does_not_leak_credentials_in_base_url(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "local_ai_tools_api_base",
        "http://user:password@127.0.0.1:8001",
    )
    client = LocalAIToolsClient()

    async def succeed(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_post", succeed)

    result = await client.enrich_with_context({"symbol": "BTC/USDT"})

    assert result["status"] == "completed"
    assert result["api_base"] == "invalid_config"
    assert "password" not in str(result)
    assert "user:password" not in str(result)


def test_local_ai_tools_circuit_payload_does_not_leak_credentials_in_base_url(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "local_ai_tools_api_base",
        "http://user:password@127.0.0.1:8001",
    )
    client = LocalAIToolsClient()
    client._circuit_open_until = datetime.now(UTC) + timedelta(seconds=30)
    client._last_failure = "Authorization: *** failed"

    result = client._circuit_open_payload()

    assert result is not None
    assert result["api_base"] == "invalid_config"
    assert "password" not in str(result)
    assert "user:password" not in str(result)


def test_local_ai_tools_client_auth_failure_is_redacted(local_tools_settings: None) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    response = httpx.Response(
        401,
        json={"detail": f"Authorization: Bearer {leaked_value} is invalid"},
        request=httpx.Request("POST", "http://local-ai-tools.test/train"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        LocalAIToolsClient()._parse_response(response, "/train")

    message = str(exc_info.value)
    assert "HTTP 401" in message
    assert "LOCAL_AI_TOOLS_API_KEY" in message
    assert leaked_value not in message
    assert "Authorization: ***" in message
