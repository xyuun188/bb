from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.learning import ShadowBacktest
from models.market_data import Kline
from web_dashboard.api import data_collection as data_collection_module
from web_dashboard.app import create_app


async def _use_temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    await close_db()
    db_path = tmp_path / "data-collection-api.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db()


@pytest.mark.asyncio
async def test_data_collection_status_exposes_sources_and_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "external_event_scraper_enabled", False)
    monkeypatch.setattr(settings, "external_event_scraper_sources", [])

    class FakeCryptoFeatureCoverageService:
        async def report(self, *, hours: int = 24, limit: int = 1000) -> dict[str, Any]:
            return {
                "audit_only": True,
                "live_signal_mutation": False,
                "can_missing_features_drive_live_entry": False,
                "feature_defaults_are_neutral": True,
                "status": "warning",
                "missing_features": ["funding_rate"],
                "stale_features": [],
                "neutralized_features": ["funding_rate"],
                "features": [{"key": "funding_rate", "status": "missing"}],
            }

    monkeypatch.setattr(
        data_collection_module,
        "CryptoFeatureCoverageService",
        lambda: FakeCryptoFeatureCoverageService(),
    )

    try:
        app = create_app()
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/data-collection/status")
    finally:
        await close_db()

    assert response.status_code == 200
    body = response.json()
    assert body["config"]["external_event_scraper_enabled"] is False
    assert body["config"]["external_event_scraper_uses_default_sources"] is True
    recommended = body["config"]["recommended_external_event_sources"]
    recommended_names = {source["name"] for source in recommended}
    assert len(recommended) >= 20
    assert {
        "binance_announcements",
        "okx_latest_announcements",
        "ethereum_blog",
        "kucoin_announcements",
        "kraken_asset_listings",
        "polygon_blog",
        "near_blog",
        "certik_blog",
        "slowmist_medium",
    }.issubset(recommended_names)
    assert all(source["url"].startswith("https://") for source in recommended)
    assert {"exchange", "project", "security"}.issubset(
        {source["category"] for source in recommended}
    )
    assert all("description" in source for source in recommended)
    sources_by_key = {source["key"]: source for source in body["sources"]}
    assert sources_by_key["rss"]["group"] == "system"
    assert sources_by_key["cryptopanic"]["group"] == "api"
    assert sources_by_key["scrapling"]["group"] == "scrapling"
    assert body["config"]["api_channels"]["cryptopanic"]["configured"] is False
    assert "news" in body["stats"]
    assert "text_sentiment_quality_sample" in body["training"]
    assert "top_sources" in body["training"]["text_sentiment_quality_sample"]
    assert "local_ai_tools" in body["training"]
    assert "governance" in body["training"]
    assert body["training"]["governance"]["status"] in {"ok", "error"}
    assert body["feature_coverage"]["audit_only"] is True
    assert body["feature_coverage"]["live_signal_mutation"] is False
    assert body["feature_coverage"]["can_missing_features_drive_live_entry"] is False
    assert body["feature_coverage"]["missing_features"] == ["funding_rate"]
    kline_timeframes = {row["timeframe"] for row in body["stats"]["market"]["klines"]}
    assert kline_timeframes == {"1m", "5m", "15m", "1h"}


@pytest.mark.asyncio
async def test_data_collection_kline_coverage_reports_missing_timeframes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    async with get_session_ctx() as session:
        session.add(
            Kline(
                symbol="BTC/USDT",
                timeframe="1m",
                open_time=datetime(2026, 6, 21, 8, 30, tzinfo=UTC),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=123.0,
            )
        )
        await session.flush()

    try:
        stats = await data_collection_module._source_breakdown()
    finally:
        await close_db()

    rows = {row["timeframe"]: row for row in stats["market"]["klines"]}
    assert set(rows) == {"1m", "5m", "15m", "1h"}
    assert rows["1m"]["rows"] == 1
    assert rows["1m"]["missing"] is False
    assert rows["5m"]["rows"] == 0
    assert rows["5m"]["missing"] is True


@pytest.mark.asyncio
async def test_data_collection_normalizes_unknown_local_ai_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLocalAIToolsClient:
        async def status(self) -> dict[str, Any]:
            return {
                "available": True,
                "status": "unknown",
                "shadow_sample_count": 12,
                "trade_sample_count": 3,
                "text_sentiment_sample_count": 8,
            }

    monkeypatch.setattr(
        data_collection_module._dash,
        "_dashboard_local_ai_tools_client",
        lambda: FakeLocalAIToolsClient(),
    )

    status = await data_collection_module._local_ai_training_status()

    assert status["status"] == "learning_only"
    assert status["raw_status"] == "unknown"
    assert status["available"] is True


@pytest.mark.asyncio
async def test_data_collection_unknown_local_ai_without_samples_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLocalAIToolsClient:
        async def status(self) -> dict[str, Any]:
            return {
                "available": True,
                "status": "unknown",
                "shadow_sample_count": 0,
                "trade_sample_count": 0,
                "text_sentiment_sample_count": 0,
            }

    monkeypatch.setattr(
        data_collection_module._dash,
        "_dashboard_local_ai_tools_client",
        lambda: FakeLocalAIToolsClient(),
    )

    status = await data_collection_module._local_ai_training_status()

    assert status["status"] == "ready"
    assert status["raw_status"] == "unknown"
    assert status["available"] is True


def test_data_collection_marks_scrapling_invalid_without_valid_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "external_event_scraper_enabled", True)
    monkeypatch.setattr(
        settings,
        "external_event_scraper_sources",
        [{"name": "broken", "url": "https://example.com/" + ("x" * 520)}],
    )
    monkeypatch.setattr(settings, "external_event_scraper_max_sources", 4)
    monkeypatch.setattr(data_collection_module, "_scrapling_installed", lambda: True)

    sources = data_collection_module._collection_sources_summary()
    scrapling = next(source for source in sources if source["key"] == "scrapling")

    assert scrapling["status"] == "invalid_config"
    assert "没有有效 HTTPS" in scrapling["detail"]


@pytest.mark.asyncio
async def test_data_collection_settings_rejects_private_scrapling_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_updates: list[dict[str, Any]] = []
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")

    def capture_update_env_file(self: object, updates: dict[str, Any]) -> None:
        captured_updates.append(updates)

    monkeypatch.setattr(settings.__class__, "update_env_file", capture_update_env_file)
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/data-collection/settings",
            json={
                "external_event_scraper_sources": [
                    {"name": "bad", "url": "https://127.0.0.1/internal"}
                ]
            },
        )

    assert response.status_code == 400
    assert "public" in response.json()["detail"] or "globally routable" in response.json()["detail"]
    assert captured_updates == []


@pytest.mark.asyncio
async def test_data_collection_settings_persists_safe_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    captured_updates: list[dict[str, Any]] = []
    captured_secrets: list[tuple[str, str]] = []
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "external_event_scraper_enabled", False)
    monkeypatch.setattr(settings, "external_event_scraper_sources", [])

    def capture_update_env_file(self: object, updates: dict[str, Any]) -> None:
        captured_updates.append(updates)

    async def capture_runtime_secret(key: str, value: str, *, actor: str = "dashboard") -> None:
        captured_secrets.append((key, value))

    monkeypatch.setattr(settings.__class__, "update_env_file", capture_update_env_file)
    monkeypatch.setattr(data_collection_module, "set_runtime_secret", capture_runtime_secret)

    try:
        app = create_app()
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/data-collection/settings",
                json={
                    "external_event_scraper_enabled": True,
                    "external_event_scraper_interval_seconds": 600,
                    "external_event_scraper_timeout_seconds": 5,
                    "external_event_scraper_max_sources": 2,
                    "external_event_scraper_max_items_per_source": 4,
                    "external_event_scraper_sources": [
                        {
                            "name": "ethereum_blog",
                            "url": "https://blog.ethereum.org/",
                            "symbols": ["ETH"],
                            "weight": 0.72,
                        }
                    ],
                    "cryptopanic_api_key": "cryptopanic-secret",
                    "coinmarketcal_api_key": "coinmarketcal-secret",
                    "newsapi_api_key": "newsapi-secret",
                },
            )
    finally:
        await close_db()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert captured_updates
    persisted = captured_updates[-1]
    assert persisted["EXTERNAL_EVENT_SCRAPER_ENABLED"] == "true"
    assert persisted["EXTERNAL_EVENT_SCRAPER_INTERVAL_SECONDS"] == "600"
    assert "ethereum_blog" in persisted["EXTERNAL_EVENT_SCRAPER_SOURCES"]
    assert "CRYPTOPANIC_API_KEY" not in persisted
    assert "COINMARKETCAL_API_KEY" not in persisted
    assert "NEWSAPI_API_KEY" not in persisted
    assert ("data_collection.cryptopanic_api_key", "cryptopanic-secret") in captured_secrets
    assert ("data_collection.coinmarketcal_api_key", "coinmarketcal-secret") in captured_secrets
    assert ("data_collection.newsapi_api_key", "newsapi-secret") in captured_secrets
    assert "api_key" not in str(persisted).lower()


@pytest.mark.asyncio
async def test_training_governance_refresh_triggers_clean_artifact_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")

    class FakeMLSignalService:
        async def maybe_auto_train(self, *, force: bool = False) -> dict[str, Any]:
            calls.append(f"ml:{force}")
            return {"trained": True, "force": force}

    class FakeTradingService:
        async def _maybe_train_local_ai_tools(self, *, force: bool = False) -> dict[str, Any]:
            calls.append(f"local_ai:{force}")
            return {"trained": True, "force": force}

    class FakeVectorMemoryService:
        async def reindex_recent(self) -> dict[str, Any]:
            calls.append("vector")
            return {"status": "ok", "indexed": 2}

    async def fake_status() -> dict[str, Any]:
        return {
            "checked_at": "2026-06-20T00:00:00+00:00",
            "config": {},
            "sources": [],
            "stats": {},
            "training": {"governance": {"status": "ok"}},
        }

    monkeypatch.setattr(data_collection_module._dash, "_trading_service", FakeTradingService())
    monkeypatch.setattr(
        data_collection_module._dash,
        "_dashboard_ml_signal_service",
        lambda: FakeMLSignalService(),
    )
    monkeypatch.setattr(
        data_collection_module,
        "get_vector_memory_service",
        lambda: FakeVectorMemoryService(),
    )
    monkeypatch.setattr(data_collection_module, "get_data_collection_status", fake_status)

    async def fake_quarantine(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("quarantine")
        return {"status": "clean", "quarantined_count": 0}

    monkeypatch.setattr(
        "services.shadow_training_quarantine.quarantine_dirty_shadow_samples",
        fake_quarantine,
    )

    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/data-collection/training-governance/refresh")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert calls == ["quarantine", "ml:True", "local_ai:True", "vector"]
    assert body["refresh_result"]["training_quarantine"]["status"] == "clean"
    assert body["refresh_result"]["vector_memory"]["indexed"] == 2


@pytest.mark.asyncio
async def test_training_governance_refresh_trains_local_tools_without_trading_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(data_collection_module._dash, "_trading_service", None)

    class FakeLocalAIToolsClient:
        def enabled(self) -> bool:
            return True

        async def train(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {"trained": True}

    monkeypatch.setattr(
        data_collection_module._dash,
        "_dashboard_local_ai_tools_client",
        lambda: FakeLocalAIToolsClient(),
    )

    async def fake_shadow(_limit: int) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "BTC/USDT",
                "decision_action": "hold",
                "decision_confidence": 0.01,
                "horizon_minutes": 30,
                "features": {"current_price": 100.0, "spread_pct": 0.01},
                "long_return_pct": 0.1,
                "short_return_pct": -0.2,
            }
        ]

    monkeypatch.setattr(
        "scripts.train_local_ai_tools_models._load_shadow_samples",
        fake_shadow,
    )
    monkeypatch.setattr(
        "scripts.train_local_ai_tools_models._load_trade_reflection_samples",
        lambda _limit: _async_value([]),
    )
    monkeypatch.setattr(
        "scripts.train_local_ai_tools_models._load_closed_position_samples",
        lambda _limit: _async_value([]),
    )
    monkeypatch.setattr(
        "scripts.train_local_ai_tools_models._load_sequence_samples",
        lambda _limit: _async_value([]),
    )
    monkeypatch.setattr(
        "scripts.train_local_ai_tools_models._load_text_sentiment_samples",
        lambda _limit: _async_value([]),
    )
    monkeypatch.setattr(
        "scripts.train_local_ai_tools_models._completed_shadow_sample_count",
        lambda: _async_value(9),
    )
    monkeypatch.setattr(
        "scripts.train_local_ai_tools_models._completed_trade_sample_count",
        lambda: _async_value(5),
    )

    result = await data_collection_module._train_local_ai_tools_from_dashboard()

    assert result["trained"] is True
    assert captured["kwargs"]["source"] == "dashboard_training_governance_refresh"
    assert captured["kwargs"]["governance_report"]["cleanup_mode"] == "quarantine_not_delete"


@pytest.mark.asyncio
async def test_training_governance_snapshot_exposes_sample_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    base_time = datetime(2026, 6, 23, 1, 0, tzinfo=UTC)
    async with get_session_ctx() as session:
        session.add_all(
            [
                ShadowBacktest(
                    model_name="ensemble",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    decision_action="long",
                    status="completed",
                    due_at=base_time - timedelta(hours=2),
                    horizon_minutes=30,
                    long_return_pct=0.4,
                    short_return_pct=-0.2,
                    best_action="long",
                ),
                ShadowBacktest(
                    model_name="ensemble",
                    execution_mode="paper",
                    symbol="ETH/USDT",
                    decision_action="hold",
                    status="completed",
                    due_at=base_time - timedelta(hours=1),
                    horizon_minutes=30,
                    long_return_pct=0.1,
                    short_return_pct=-0.1,
                    best_action="hold",
                ),
                ShadowBacktest(
                    model_name="ensemble",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    decision_action="short",
                    status="quarantined",
                    due_at=base_time,
                    horizon_minutes=30,
                    long_return_pct=-0.3,
                    short_return_pct=0.2,
                    best_action="short",
                ),
            ]
        )
        await session.flush()

    try:
        snapshot = await data_collection_module._training_governance_snapshot()
    finally:
        await close_db()

    report = snapshot["local_ai_tools"]
    assert snapshot["status"] == "ok"
    assert report["sample_total_count"] == 3
    assert report["trainable_sample_count"] == 2
    assert report["quarantined_sample_count"] == 1
    assert report["action_counts"] == {"hold": 1, "long": 1, "short": 1}
    assert report["trainable_action_counts"] == {"hold": 1, "long": 1}
    assert report["symbol_count"] == 2
    assert report["symbol_counts"] == {"BTC/USDT": 2, "ETH/USDT": 1}
    assert report["time_span"]["oldest_sample_at"].startswith("2026-06-22T23:00:00")
    assert report["time_span"]["latest_sample_at"].startswith("2026-06-23T01:00:00")
    assert report["data_freshness_minutes"] is not None


@pytest.mark.asyncio
async def test_data_collection_status_is_read_only_for_training_quarantine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    quarantine_calls: list[str] = []
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "external_event_scraper_enabled", False)
    monkeypatch.setattr(settings, "external_event_scraper_sources", [])

    async def fake_quarantine(*args: Any, **kwargs: Any) -> dict[str, Any]:
        quarantine_calls.append("called")
        return {"status": "clean"}

    monkeypatch.setattr(
        "services.shadow_training_quarantine.quarantine_dirty_shadow_samples",
        fake_quarantine,
    )
    monkeypatch.setattr(
        "scripts.train_local_ai_tools_models._load_shadow_samples",
        lambda _limit: _async_value([]),
    )
    monkeypatch.setattr(
        "scripts.train_local_ai_tools_models._load_trade_reflection_samples",
        lambda _limit: _async_value([]),
    )
    monkeypatch.setattr(
        "scripts.train_local_ai_tools_models._load_closed_position_samples",
        lambda _limit: _async_value([]),
    )
    monkeypatch.setattr(
        "scripts.train_local_ai_tools_models._load_sequence_samples",
        lambda _limit: _async_value([]),
    )
    monkeypatch.setattr(
        "scripts.train_local_ai_tools_models._load_text_sentiment_samples",
        lambda _limit: _async_value([]),
    )
    monkeypatch.setattr(
        "services.ml_signal_service.load_shadow_training_rows",
        lambda *, limit: _async_value([]),
    )

    try:
        app = create_app()
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/data-collection/status")
    finally:
        await close_db()

    assert response.status_code == 200
    body = response.json()
    assert quarantine_calls == []
    assert body["training"]["governance"]["training_quarantine"]["status"] == "not_run"
    assert (
        body["training"]["governance"]["local_ai_tools"]["deep_quality_evaluation"]
        == "deferred_to_training_refresh"
    )
    assert body["training"]["governance"]["local_ai_tools"]["raw_records_preserved"] is True
    assert "external_event_scraper_interval_seconds" in body["config"]
    assert "external_event_scraper_sources" in body["config"]


@pytest.mark.asyncio
async def test_data_collection_status_does_not_run_deep_shadow_assessment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "external_event_scraper_enabled", False)
    monkeypatch.setattr(settings, "external_event_scraper_sources", [])

    def forbidden_assess_shadow_row(_row: Any) -> None:
        raise AssertionError("status endpoint must not run deep shadow assessment")

    monkeypatch.setattr(
        "services.shadow_training_quarantine.assess_shadow_row",
        forbidden_assess_shadow_row,
    )

    try:
        app = create_app()
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/data-collection/status")
    finally:
        await close_db()

    assert response.status_code == 200
    body = response.json()
    assert (
        body["training"]["governance"]["local_ai_tools"]["deep_quality_evaluation"]
        == "deferred_to_training_refresh"
    )


@pytest.mark.asyncio
async def test_data_collection_status_keeps_config_when_governance_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeCryptoFeatureCoverageService:
        async def report(self, *, hours: int = 24, limit: int = 1000) -> dict[str, Any]:
            return {"status": "ok", "audit_only": True, "features": []}

    monkeypatch.setattr(
        data_collection_module,
        "CryptoFeatureCoverageService",
        lambda: FakeCryptoFeatureCoverageService(),
    )
    await _use_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "external_event_scraper_enabled", True)
    monkeypatch.setattr(settings, "external_event_scraper_sources", [])

    async def failing_governance() -> dict[str, Any]:
        raise RuntimeError("governance exploded")

    monkeypatch.setattr(
        data_collection_module,
        "_training_governance_snapshot",
        failing_governance,
    )

    try:
        app = create_app()
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/data-collection/status")
    finally:
        await close_db()

    assert response.status_code == 200
    body = response.json()
    assert body["config"]["external_event_scraper_enabled"] is True
    assert "api_channels" in body["config"]
    assert "cryptopanic" in body["config"]["api_channels"]
    assert "configured" in body["config"]["api_channels"]["cryptopanic"]
    assert body["training"]["governance"]["status"] == "error"
    assert "governance exploded" in body["training"]["governance"]["error"]


@pytest.mark.asyncio
async def test_data_collection_status_keeps_config_when_governance_times_out(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeCryptoFeatureCoverageService:
        async def report(self, *, hours: int = 24, limit: int = 1000) -> dict[str, Any]:
            return {"status": "ok", "audit_only": True, "features": []}

    monkeypatch.setattr(
        data_collection_module,
        "CryptoFeatureCoverageService",
        lambda: FakeCryptoFeatureCoverageService(),
    )
    await _use_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "external_event_scraper_enabled", True)
    monkeypatch.setattr(settings, "external_event_scraper_interval_seconds", 600)
    monkeypatch.setattr(data_collection_module, "STATUS_SECTION_TIMEOUT_SECONDS", 0.01)

    async def slow_governance() -> dict[str, Any]:
        await asyncio.sleep(1)
        return {"status": "ok"}

    monkeypatch.setattr(
        data_collection_module,
        "_training_governance_snapshot",
        slow_governance,
    )

    try:
        app = create_app()
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/data-collection/status")
    finally:
        await close_db()

    assert response.status_code == 200
    body = response.json()
    assert body["config"]["external_event_scraper_enabled"] is True
    assert body["config"]["external_event_scraper_interval_seconds"] == 600
    assert body["training"]["governance"]["status"] == "error"
    assert body["training"]["governance"]["section"] == "training_governance"


async def _async_value(value: Any) -> Any:
    return value
