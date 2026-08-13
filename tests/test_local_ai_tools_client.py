from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from threading import get_ident
from time import monotonic, sleep
from typing import Any

import httpx
import pytest

from config.settings import settings
from data_feed.feature_vector import FeatureVector
from services import local_ai_tools_client as local_ai_tools_client_module
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
async def test_local_ai_tools_enrich_shares_batch_deadline_across_concurrent_routes(
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
    assert len(timeouts) == 3
    assert all(timeout is not None and 0 < timeout <= 8.0 for timeout in timeouts)
    assert len(set(timeouts)) == 1
    assert 7.5 < timeouts[0] <= 8.0
    assert result["execution_mode"] == "concurrent_routes_serial_batches"
    assert result["batch_budget_policy"] == (
        "shared_batch_deadline_concurrent_routes"
    )
    assert result["batch_budget_seconds"] == 8.0
    assert result["profit_prediction"]["duration_sec"] > 0
    assert result["time_series_prediction"]["duration_sec"] > 0
    assert result["sentiment_analysis"]["duration_sec"] > 0


@pytest.mark.asyncio
async def test_local_ai_tools_serializes_batches_but_concurrently_runs_routes(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    active_calls = 0
    max_active_calls = 0
    calls: list[tuple[str, str]] = []

    async def succeed(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        nonlocal active_calls, max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        calls.append((str(payload.get("symbol") or ""), path))
        await asyncio.sleep(0.01)
        active_calls -= 1
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_post", succeed)

    first, second = await asyncio.gather(
        client.enrich_with_context({"symbol": "BTC/USDT"}),
        client.enrich_with_context({"symbol": "ETH/USDT"}),
    )

    assert first["status"] == "completed"
    assert second["status"] == "analysis_budget_deferred"
    assert max_active_calls == 3
    assert len(calls) == 3
    assert {symbol for symbol, _path in calls} == {"BTC/USDT"}


@pytest.mark.asyncio
async def test_local_ai_tools_keeps_completed_results_when_later_route_exhausts_budget(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 0.2)
    client = LocalAIToolsClient()
    calls: list[str] = []

    async def slow_timeseries(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        del payload
        calls.append(path)
        if path == "/timeseries/predict":
            await asyncio.sleep(1.0)
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_post", slow_timeseries)

    result = await client.enrich_with_context({"symbol": "BTC/USDT"})

    assert result["status"] == "partial"
    assert result["profit_prediction"]["available"] is True
    assert result["sentiment_analysis"]["available"] is True
    assert result["time_series_prediction"]["status"] == "timeout"
    assert calls == [
        "/profit/predict",
        "/sentiment/deep/analyze",
        "/timeseries/predict",
    ]


@pytest.mark.asyncio
async def test_local_ai_tools_slow_first_route_cannot_starve_later_tools(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 0.6)
    client = LocalAIToolsClient()
    calls: list[str] = []

    class StaleClient:
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    stale_client = StaleClient()
    client._http_client = stale_client  # type: ignore[assignment]
    client._http_client_base = "http://local-ai-tools.test"

    async def slow_profit(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        del payload, request_timeout
        calls.append(path)
        if path == "/profit/predict":
            await asyncio.sleep(1.0)
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_post", slow_profit)

    result = await client.enrich_with_context({"symbol": "BTC/USDT"})

    assert result["status"] == "partial"
    assert result["profit_prediction"]["status"] == "timeout"
    assert result["sentiment_analysis"]["available"] is True
    assert result["time_series_prediction"]["available"] is True
    assert result["soft_timeout_tools"] == ["profit_prediction"]
    assert result["http_connection_reset"] is True
    await asyncio.sleep(0.05)
    assert stale_client.closed is True
    assert client._http_client is None
    assert calls == [
        "/profit/predict",
        "/sentiment/deep/analyze",
        "/timeseries/predict",
    ]


@pytest.mark.asyncio
async def test_local_ai_tools_timeout_does_not_wait_for_stubborn_request_cleanup(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 0.6)
    client = LocalAIToolsClient()

    class StaleClient:
        async def aclose(self) -> None:
            return None

    client._http_client = StaleClient()  # type: ignore[assignment]
    client._http_client_base = "http://local-ai-tools.test"

    async def stubborn_post(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        del payload, request_timeout
        if path == "/profit/predict":
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                await asyncio.sleep(0.15)
                return {"available": True, "path": path}
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_post", stubborn_post)

    result = await client.enrich_with_context({"symbol": "BTC/USDT"})

    assert result["profit_prediction"]["status"] == "timeout"
    assert result["profit_prediction"]["duration_sec"] < 0.8
    assert result["http_connection_reset"] is True
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_local_ai_tools_native_post_cancellation_stops_inference_request(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class FakeClient:
        async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
            del args, kwargs
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    monkeypatch.setattr(client, "_shared_http_client", lambda _base: _fake_client())

    async def _fake_client() -> FakeClient:
        return FakeClient()

    task = asyncio.create_task(
        client._post("/profit/predict", {"symbol": "BTC/USDT"}, request_timeout=5.0)
    )
    await asyncio.wait_for(started.wait(), timeout=0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_local_ai_tools_client_reset_does_not_wait_for_stubborn_close_cleanup(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_ai_tools_client_module,
        "_HTTP_CLIENT_CLOSE_TIMEOUT_SECONDS",
        0.05,
    )
    client = LocalAIToolsClient()

    class StubbornClient:
        async def aclose(self) -> None:
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                await asyncio.sleep(0.2)

    client._http_client = StubbornClient()  # type: ignore[assignment]
    client._http_client_base = "http://local-ai-tools.test"

    started = monotonic()
    reset = await client._reset_http_client()
    elapsed = monotonic() - started

    assert reset is True
    assert elapsed < 0.15
    assert client._http_client is None
    await asyncio.sleep(0.25)


@pytest.mark.asyncio
async def test_local_ai_tools_quant_post_isolated_from_trading_event_loop(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 0.6)
    client = LocalAIToolsClient()
    caller_thread = get_ident()
    worker_threads: set[int] = set()

    async def post(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        del payload, request_timeout
        worker_threads.add(get_ident())
        await asyncio.sleep(0.05)
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_post", post)
    task = asyncio.create_task(client.enrich_with_context({"symbol": "BTC/USDT"}))
    await asyncio.sleep(0.01)
    sleep(0.3)  # noqa: ASYNC251 - deliberately starve the trading event loop.
    result = await task

    assert result["status"] == "completed"
    assert result["profit_prediction"]["available"] is True
    assert result["sentiment_analysis"]["available"] is True
    assert result["time_series_prediction"]["available"] is True
    assert worker_threads == {caller_thread}


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


@pytest.mark.asyncio
async def test_local_ai_tools_requests_keep_native_horizon_selection(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    requests: list[tuple[str, dict[str, Any]]] = []

    async def capture(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        del request_timeout
        requests.append((path, dict(payload)))
        return {"available": True, "path": path, "best_side": "hold"}

    monkeypatch.setattr(client, "_post", capture)
    result = await client.enrich_with_context(
        {"symbol": "BTC/USDT", "horizon_minutes": 15},
        ml_signal={"primary_horizon_minutes": 5, "predictions": []},
    )

    assert len(requests) == 3
    assert len(requests) == 3
    assert all(payload["features"]["horizon_minutes"] == 15 for _, payload in requests)
    assert all("shared_prediction_horizon" not in payload for _, payload in requests)
    assert "shared_prediction_horizon" not in result


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
        trade_sample_cursor_policy="current_training_epoch_only",
        promotion_recommendation={
            "policy": "phase3_candidate_to_shadow_to_canary_to_active",
            "recommended_stage": "shadow",
        },
    )

    assert result["trained"] is True
    assert captured["path"] == "/train"
    assert captured["payload"]["completed_shadow_sample_count"] == 1234
    assert captured["payload"]["completed_trade_sample_count"] == 56
    assert captured["payload"]["trade_sample_cursor_policy"] == "current_training_epoch_only"
    assert captured["payload"]["training_mode"] == "shadow"
    assert "model_stage" not in captured["payload"]
    assert "evaluation_policy" not in captured["payload"]
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
    assert "evaluation_policy" not in captured["payload"]


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
    assert recommendation["policy"] == "2026-07-24.model-owned-return-promotion.v3"
    assert recommendation["canary_ready"] is True
    assert recommendation["canary_execution_scope"] == "paper_only"
    assert recommendation["canary_production_permission"] is False
    assert recommendation["live_ml_ready"] is False
    assert "walk_forward_required" in recommendation["live_blocking_reasons"]
    assert "paper_observation_report" not in captured["payload"]
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
    assert third["failure_count"] == 0
    assert len(calls) == 9


def test_local_ai_tools_localizes_request_timeouts_and_keeps_them_soft() -> None:
    client = LocalAIToolsClient()

    message = client._request_error_message(httpx.ReadTimeout("read timed out"))

    assert message == "服务器量化工具读取响应超时"
    assert client._is_soft_timeout_failure(message) is True
    assert client._is_soft_timeout_failure(
        "server quant tool exceeded the remaining batch budget"
    ) is True
    assert client._is_soft_timeout_failure(
        "local AI tools inference queue exhausted the batch budget"
    ) is True


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
            "production_permission": True,
            "live_ml_ready": True,
            "return_distribution_input_version": RETURN_DISTRIBUTION_INPUT_VERSION,
            "return_distribution_inputs": {
                "long": _distribution_input("long", 0.8, 0.5),
                "short": _distribution_input("short", 0.2, 0.1),
            },
            "prediction_quality": {
                "contract_complete": True,
                "paper_eligible": True,
                "production_eligible": True,
                "anomalous": False,
            },
        },
    )

    contract = profit["return_distribution_contract"]["long"]
    assert profit["return_distribution_contract_version"] == (RETURN_DISTRIBUTION_CONTRACT_VERSION)
    assert contract["raw_expected_return_pct"] == pytest.approx(0.8)
    assert contract["lower_quantile_return_pct"] == pytest.approx(0.5)
    assert contract["objective_expected_return_pct"] == pytest.approx(0.47)
    assert contract["production_eligible"] is True
    assert profit["prediction_quality"]["production_eligible"] is True


def test_local_ai_tools_keeps_valid_shadow_prediction_paper_eligible(
    local_tools_settings: None,
) -> None:
    client = LocalAIToolsClient()

    profit = client._normalize_signal(
        "profit_prediction",
        {
            "available": True,
            "best_side": "long",
            "route_mode": "paper_analysis",
            "production_permission": False,
            "live_ml_ready": False,
            "return_distribution_input_version": RETURN_DISTRIBUTION_INPUT_VERSION,
            "return_distribution_inputs": {
                "long": _distribution_input("long", 0.8, 0.5),
                "short": _distribution_input("short", 0.2, 0.1),
            },
            "prediction_quality": {
                "contract_complete": True,
                "paper_eligible": True,
                "production_eligible": False,
                "anomalous": False,
                "blockers": [],
                "production_blockers": ["artifact_activation_not_production_authorized"],
            },
        },
    )

    quality = profit["prediction_quality"]
    assert quality["contract_complete"] is True
    assert quality["paper_eligible"] is True
    assert quality["production_eligible"] is False
    assert quality["anomalous"] is False
    assert quality["blockers"] == []
    assert quality["production_blockers"] == ["artifact_activation_not_production_authorized"]


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
    assert profit["prediction_quality"]["reason"] == ("actual_trade_calibration_not_ready")
    assert profit["prediction_quality"]["blockers"] == ["actual_trade_calibration_not_ready"]


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
    assert (
        "return_distribution_cost_model_version_mismatch"
        in profit["prediction_quality"]["blockers"]
    )


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
        "Authorization: *** failed; Authorization: *** failed; Authorization: *** failed"
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
    assert result["child_endpoints"]["profit_prediction"]["probe_mode"] == ("metadata_contract")
    assert result["child_endpoints"]["profit_prediction"]["actual_inference_probe"] is False


@pytest.mark.asyncio
async def test_local_ai_tools_status_uses_override_timeout_for_parallel_health_reads(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    calls: list[tuple[str, float | None]] = []
    both_started = asyncio.Event()

    async def get_status(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        calls.append((path, request_timeout))
        if len(calls) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.2)
        return {"available": path == "/models/status", "ok": path == "/health"}

    monkeypatch.setattr(client, "_get", get_status)

    result = await client.status(request_timeout=1.25)

    assert calls == [("/models/status", 1.25), ("/health", 1.25)]
    assert result["service_available"] is True


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
            "live_ml_ready": False,
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
    assert result["live_ml_ready"] is False
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


def test_local_ai_tools_auth_headers_allow_keepalive(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_api_key", "  local-secret-token  ")
    headers = LocalAIToolsClient()._auth_headers()

    assert headers == {"Authorization": "Bearer local-secret-token"}


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
