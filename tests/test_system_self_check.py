from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from config.settings import settings
from web_dashboard.api import dashboard, system_health


def test_self_check_endpoint_contract_flags_wrong_model_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "ai_models",
        [
            {
                "name": "decision_maker",
                "api_base": "http://127.0.0.1:8003/v1",
                "api_key": "unit-key",
                "model": "qwen3-32b-trade",
                "enabled": True,
            },
            {
                "name": "trend_expert",
                "api_base": "http://127.0.0.1:18003/v1",
                "api_key": "unit-key",
                "model": "BB-FinQuant-Expert-14B",
                "enabled": True,
            },
            {
                "name": "sentiment_expert",
                "api_base": "http://127.0.0.1:18003/v1",
                "api_key": "unit-key",
                "model": "BB-FinQuant-Expert-14B",
                "enabled": True,
            },
        ],
    )
    monkeypatch.setattr(settings, "local_ai_tools_api_base", "http://127.0.0.1:18001")
    monkeypatch.setattr(settings, "high_risk_review_model", "deepseek-r1-14b-risk")
    monkeypatch.setattr(settings, "high_risk_review_api_base", "http://127.0.0.1:18002/v1")

    items = system_health._configured_endpoint_items()
    by_key = {item["key"]: item for item in items}

    assert by_key["endpoint_qwen3-32b-trade"]["status"] == "critical"
    assert by_key["endpoint_qwen3-32b-trade"]["details"]["expected_platform_endpoint"] == (
        "http://127.0.0.1:18000/v1"
    )
    assert by_key["endpoint_phase3_quant_api"]["status"] == "ok"
    assert by_key["endpoint_deepseek-r1-14b-risk"]["status"] == "ok"
    assert by_key["endpoint_BB-FinQuant-Expert-14B"]["status"] == "ok"


def test_self_check_endpoint_contract_uses_split_runtime_before_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_models", [])
    monkeypatch.setattr(settings, "local_ai_tools_api_base", "")
    monkeypatch.setattr(settings, "high_risk_review_api_base", "")
    monitor_status = {
        "platform_runtime": {
            "ai_models": [
                {
                    "name": "decision_maker",
                    "model": "qwen3-32b-trade",
                    "api_base": "http://127.0.0.1:18000/v1",
                },
                {
                    "model": "deepseek-r1-14b-risk",
                    "api_base": "http://127.0.0.1:18002/v1",
                },
                {
                    "model": "BB-FinQuant-Expert-14B",
                    "api_base": "http://127.0.0.1:18003/v1",
                },
            ],
            "local_ai_tools": {"api_base": "http://127.0.0.1:18001"},
        }
    }

    items = system_health._configured_endpoint_items(monitor_status)
    by_key = {item["key"]: item for item in items}

    assert by_key["endpoint_qwen3-32b-trade"]["status"] == "ok"
    assert by_key["endpoint_phase3_quant_api"]["status"] == "ok"
    assert by_key["endpoint_deepseek-r1-14b-risk"]["status"] == "ok"
    assert by_key["endpoint_BB-FinQuant-Expert-14B"]["status"] == "ok"


def test_self_check_accepts_external_deepseek_final_decision_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "ai_models",
        [
            {
                "name": "decision_maker",
                "api_base": "https://api.deepseek.com/v1",
                "api_key": "unit-key",
                "model": "deepseek-v4-pro",
                "route_mode": "online_slow_brain",
                "enabled": True,
            },
            {
                "name": "trend_expert",
                "api_base": "http://127.0.0.1:18003/v1",
                "model": "BB-FinQuant-Expert-14B",
                "enabled": True,
            },
        ],
    )
    monkeypatch.setattr(settings, "local_ai_tools_api_base", "http://127.0.0.1:18001")
    monkeypatch.setattr(settings, "high_risk_review_model", "deepseek-r1-14b-risk")
    monkeypatch.setattr(settings, "high_risk_review_api_base", "http://127.0.0.1:18002/v1")

    items = system_health._configured_endpoint_items()
    by_key = {item["key"]: item for item in items}

    assert "endpoint_qwen3-32b-trade" not in by_key
    assert by_key["endpoint_deepseek-v4-pro"]["status"] == "ok"
    assert by_key["endpoint_deepseek-v4-pro"]["details"]["slot_name"] == (
        "decision_maker"
    )


def test_server_monitor_items_keep_runtime_models_when_remote_monitor_unavailable() -> None:
    items = system_health._server_monitor_items(
        {
            "available": True,
            "remote_monitor_available": False,
            "status": "model_server_config_error",
            "message": "BB_SECURE_SETTINGS_KEY is required",
            "checked_at": "2026-06-20T00:00:00+00:00",
            "platform_runtime": {
                "ai_models": [
                    {
                        "model": "deepseek-r1-14b-risk",
                        "api_base": "http://127.0.0.1:18002/v1",
                        "available": True,
                        "endpoint_ok": True,
                        "model_available": True,
                    }
                ],
                "local_ai_tools": {
                    "configured": True,
                    "api_base": "http://127.0.0.1:18001",
                    "available": True,
                },
            },
        }
    )

    by_key = {item["key"]: item for item in items}
    assert by_key["server_monitor"]["status"] == "info"
    assert by_key["runtime_model_deepseek-r1-14b-risk"]["status"] == "ok"
    assert by_key["runtime_local_ai_tools"]["status"] == "ok"


def test_server_monitor_items_do_not_mark_extra_legacy_model_critical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "ai_models",
        [
            {
                "name": "trend_expert",
                "api_base": "http://127.0.0.1:18000/v1",
                "api_key": "unit-key",
                "model": "qwen3-32b-trade",
                "enabled": True,
            }
        ],
    )
    monkeypatch.setattr(settings, "high_risk_review_model", "deepseek-r1-14b-risk")

    items = system_health._server_monitor_items(
        {
            "available": True,
            "remote_monitor_available": False,
            "status": "model_server_config_error",
            "platform_runtime": {
                "ai_models": [
                    {
                        "model": "deepseek-v4-pro",
                        "api_base": "https://api.deepseek.com/v1",
                        "available": False,
                        "endpoint_ok": False,
                        "model_available": False,
                    },
                    {
                        "model": "qwen3-32b-trade",
                        "api_base": "http://127.0.0.1:18000/v1",
                        "available": False,
                        "endpoint_ok": False,
                        "model_available": False,
                    },
                ],
                "local_ai_tools": {"configured": True, "available": True},
            },
        }
    )

    by_key = {item["key"]: item for item in items}
    assert by_key["runtime_model_deepseek-v4-pro"]["status"] == "info"
    assert by_key["runtime_model_deepseek-v4-pro"]["details"]["required"] is False
    assert by_key["runtime_model_qwen3-32b-trade"]["status"] == "critical"
    assert by_key["runtime_model_qwen3-32b-trade"]["details"]["required"] is True


def test_expert_model_diversity_flags_shared_expert_provider() -> None:
    monitor_status = {
        "platform_runtime": {
            "ai_models": [
                {
                    "name": name,
                    "label": name,
                    "api_base": "http://127.0.0.1:18003/v1",
                    "model": "BB-FinQuant-Expert-14B",
                }
                for name in (
                    "trend_expert",
                    "momentum_expert",
                    "sentiment_expert",
                    "position_expert",
                    "risk_expert",
                )
            ]
        }
    }

    item = system_health._expert_model_diversity_item(monitor_status)

    assert item["status"] == "warning"
    assert item["key"] == "expert_model_diversity"
    assert item["details"]["configured_expert_count"] == 5
    assert item["details"]["unique_provider_count"] == 1
    assert item["details"]["largest_shared_provider_count"] == 5
    assert item["details"]["same_provider_risk"] is True


@pytest.mark.asyncio
async def test_self_check_repair_only_resets_low_risk_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions: list[str] = []
    local_client = SimpleNamespace(
        _status_cache=("cached", {}),
        _failure_count=3,
        _circuit_open_until="later",
    )

    def clear_cache() -> None:
        actions.append("clear_monitor_cache")

    monkeypatch.setattr(system_health, "clear_server_monitor_cache", clear_cache)
    monkeypatch.setattr(dashboard, "_dashboard_local_ai_tools_client", lambda: local_client)

    payload = await system_health.system_self_check_repair()

    assert payload["status"] == "ok"
    assert "\u8d44\u91d1" in payload["safety_note"]
    assert actions == ["clear_monitor_cache"]
    assert local_client._status_cache is None
    assert local_client._failure_count == 0
    assert local_client._circuit_open_until is None
    assert {item["action"] for item in payload["actions"]} == {
        "clear_monitor_cache",
        "reset_local_ai_tools_breaker",
    }


@pytest.mark.asyncio
async def test_self_check_api_returns_recent_execution_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def monitor_status() -> dict[str, Any]:
        return {
            "available": True,
            "checked_at": "2026-06-17T00:00:00+00:00",
            "platform_runtime": {
                "ai_models": [],
                "local_ai_tools": {"available": True, "api_base": "http://127.0.0.1:18001"},
            },
        }

    async def recent_items() -> list[dict[str, Any]]:
        return [
            system_health._check_item(
                "recent_failed_orders",
                "recent failed orders",
                "warning",
                "1 recent failed order",
            )
        ]

    async def data_source_items() -> list[dict[str, Any]]:
        return [
            system_health._check_item(
                "market_ticker_freshness",
                "ticker",
                "ok",
                "fresh",
            )
        ]

    monkeypatch.setattr(system_health, "get_server_monitor_status_async", monitor_status)
    monkeypatch.setattr(system_health, "_recent_execution_items", recent_items)
    monkeypatch.setattr(system_health, "_data_source_items", data_source_items)

    async def running_item() -> dict[str, Any]:
        return system_health._check_item("trading_service", "trading service", "ok", "running")

    monkeypatch.setattr(system_health, "_trading_service_running_item", running_item)
    monkeypatch.setattr(
        system_health,
        "_okx_config_item",
        lambda mode: system_health._check_item(f"okx_{mode}", f"{mode} OKX", "ok", "configured"),
    )
    monkeypatch.setattr(system_health, "_configured_endpoint_items", lambda: [])

    payload = await system_health.system_self_check()

    assert payload["status"] == "warning"
    assert payload["summary"]["warning"] >= 1
    assert payload["summary"]["info"] == 0
    assert any(item["key"] == "recent_failed_orders" for item in payload["items"])


@pytest.mark.asyncio
async def test_data_source_self_check_reports_market_news_and_social_freshness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = system_health.datetime.now(system_health.UTC)

    class FakeResult:
        def __init__(
            self, *, one_row: tuple[Any, ...] | None = None, rows: list[Any] | None = None
        ) -> None:
            self._one_row = one_row
            self._rows = rows or []

        def one(self) -> tuple[Any, ...]:
            assert self._one_row is not None
            return self._one_row

        def all(self) -> list[Any]:
            return self._rows

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _stmt: Any) -> FakeResult:
            self.calls += 1
            if self.calls == 1:
                return FakeResult(one_row=(12, now))
            if self.calls == 2:
                return FakeResult(
                    rows=[
                        ("1m", 1200, 20, now),
                        ("5m", 1200, 20, now),
                        ("15m", 1200, 20, now),
                        ("1h", 1000, 20, now),
                    ]
                )
            if self.calls == 3:
                return FakeResult(one_row=(200, 3, now))
            if self.calls == 4:
                return FakeResult(one_row=(40, 8, now))
            return FakeResult(one_row=(0, 0, None))

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    monkeypatch.setattr(system_health, "get_session_ctx", fake_session_ctx)

    items = await system_health._data_source_items()
    by_key = {item["key"]: item for item in items}

    assert by_key["market_ticker_freshness"]["status"] == "ok"
    assert by_key["market_kline_coverage"]["status"] == "ok"
    assert by_key["news_source_freshness"]["status"] == "ok"
    assert by_key["external_event_source_freshness"]["status"] == "ok"
    assert by_key["social_source_freshness"]["status"] == "warning"
    assert by_key["market_kline_coverage"]["details"]["missing_timeframes"] == []
    assert by_key["external_event_source_freshness"]["details"]["source_count"] == 8
    assert by_key["social_source_freshness"]["details"]["platform_count"] == 0


@pytest.mark.asyncio
async def test_recent_execution_self_check_flags_entry_without_opportunity_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        def scalars(self) -> FakeResult:
            return self

        def all(self) -> list[Any]:
            return self._rows

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _stmt: Any) -> FakeResult:
            self.calls += 1
            if self.calls == 1:
                return FakeResult([])
            return FakeResult(
                [
                    SimpleNamespace(
                        id=101,
                        action="short",
                        raw_llm_response={"decision_state_machine": {"stages": []}},
                    )
                ]
            )

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    monkeypatch.setattr(system_health, "get_session_ctx", fake_session_ctx)

    items = await system_health._recent_execution_items()
    by_key = {item["key"]: item for item in items}

    assert by_key["entry_opportunity_score_coverage"]["status"] == "critical"
    assert by_key["entry_opportunity_score_coverage"]["details"]["sample_decision_ids"] == [101]
    assert by_key["entry_opportunity_score_coverage"]["details"]["latest_scored_entry_at"] is None


@pytest.mark.asyncio
async def test_recent_execution_self_check_ignores_position_review_without_opportunity_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        def scalars(self) -> FakeResult:
            return self

        def all(self) -> list[Any]:
            return self._rows

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _stmt: Any) -> FakeResult:
            self.calls += 1
            if self.calls == 1:
                return FakeResult([])
            return FakeResult(
                [
                    SimpleNamespace(
                        id=9230,
                        action="short",
                        analysis_type="position",
                        raw_llm_response={
                            "analysis_type": "position_review",
                            "position_review_policy": {"result": "hold"},
                            "decision_state_machine": {"stages": [{"status": "passed"}]},
                        },
                    )
                ]
            )

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    monkeypatch.setattr(system_health, "get_session_ctx", fake_session_ctx)

    items = await system_health._recent_execution_items()
    by_key = {item["key"]: item for item in items}

    assert "entry_opportunity_score_coverage" not in by_key


@pytest.mark.asyncio
async def test_trading_service_split_process_uses_recent_activity_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard, "_trading_service", None)

    async def recent_activity() -> dict[str, Any]:
        return {
            "decision_count": 5,
            "latest_decision_at": "2026-06-17T04:10:00+00:00",
            "order_count": 1,
            "latest_order_at": "2026-06-17T04:09:00+00:00",
            "window_hours": 2,
        }

    monkeypatch.setattr(system_health, "_recent_trading_activity_snapshot", recent_activity)

    item = await system_health._trading_service_running_item()

    assert item["status"] == "ok"
    assert "\u72ec\u7acb\u8fdb\u7a0b" in item["message"]
    assert item["details"]["decision_count"] == 5


@pytest.mark.asyncio
async def test_dashboard_summary_uses_split_process_runtime_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    heartbeat = {
        "running": True,
        "mode": "paper",
        "paused": False,
        "heartbeat_at": datetime.now(UTC).isoformat(),
        "uptime_seconds": 3661,
        "decision_interval": 30,
        "market_loop_interval_seconds": 10.5,
        "position_loop_interval_seconds": 19.5,
        "market_round_time_budget_seconds": 27.0,
        "market_analysis_watchdog_seconds": 240,
        "position_analysis_watchdog_seconds": 300,
        "current_stage": "idle",
        "round_active": True,
        "last_round_started_at": (datetime.now(UTC) - timedelta(seconds=45)).isoformat(),
        "last_round_finished_at": (datetime.now(UTC) - timedelta(seconds=10)).isoformat(),
    }
    (data_dir / "trading_runtime_status.json").write_text(
        json.dumps(heartbeat),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings.__class__, "data_dir", property(lambda _self: data_dir))

    async def recent_activity(_hours: int = 6) -> dict[str, Any]:
        return {
            "decision_count": 2,
            "order_count": 0,
            "heartbeat_age_seconds": 10.0,
            "latest_activity_at": heartbeat["last_round_finished_at"],
            "window_hours": _hours,
        }

    monkeypatch.setattr(dashboard, "_recent_trading_activity_stats", recent_activity)

    stats = await dashboard._split_process_trading_stats("paper")

    assert stats["running"] is True
    assert stats["uptime_seconds"] == 3661
    assert stats["decision_interval"] == 30
    assert stats["market_loop_interval_seconds"] == 10.5
    assert stats["position_loop_interval_seconds"] == 19.5
    assert stats["market_round_time_budget_seconds"] == 27.0
    assert stats["market_analysis_watchdog_seconds"] == 240
    assert stats["position_analysis_watchdog_seconds"] == 300
    assert stats["uptime_source"] == "split_process_heartbeat"
    assert stats["round_active"] is True
    assert stats["round_running_seconds"] >= 40


@pytest.mark.asyncio
async def test_split_process_stats_use_shared_pause_state_over_stale_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "trading_runtime_status.json").write_text(
        json.dumps(
            {
                "running": True,
                "mode": "paper",
                "paused": False,
                "heartbeat_at": datetime.now(UTC).isoformat(),
                "decision_interval": 30,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings.__class__, "data_dir", property(lambda _self: data_dir))
    await dashboard.mode_manager.pause()

    async def recent_activity(_hours: int = 6) -> dict[str, Any]:
        return {"decision_count": 0, "order_count": 0, "heartbeat_age_seconds": 5.0}

    monkeypatch.setattr(dashboard, "_recent_trading_activity_stats", recent_activity)

    stats = await dashboard._split_process_trading_stats("paper")

    assert stats["paused"] is True


@pytest.mark.asyncio
async def test_dashboard_stats_backfills_runtime_fields_when_service_is_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    heartbeat = {
        "running": True,
        "mode": "paper",
        "paused": False,
        "heartbeat_at": datetime.now(UTC).isoformat(),
        "decision_interval": 30,
        "market_loop_interval_seconds": 10.5,
        "position_loop_interval_seconds": 19.5,
        "market_round_time_budget_seconds": 27.0,
    }
    (data_dir / "trading_runtime_status.json").write_text(
        json.dumps(heartbeat),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings.__class__, "data_dir", property(lambda _self: data_dir))

    class PartialTradingService:
        def get_stats(self, mode_filter=None):
            return {"running": True, "decision_interval": None}

    async def recent_activity(_hours: int = 6) -> dict[str, Any]:
        return {"decision_count": 1, "order_count": 0, "heartbeat_age_seconds": 5.0}

    monkeypatch.setattr(dashboard, "_trading_service", PartialTradingService())
    monkeypatch.setattr(dashboard, "_recent_trading_activity_stats", recent_activity)

    stats = await dashboard._trading_stats_with_runtime_heartbeat("paper")

    assert stats["decision_interval"] == 30
    assert stats["market_loop_interval_seconds"] == 10.5
    assert stats["position_loop_interval_seconds"] == 19.5
    assert stats["market_round_time_budget_seconds"] == 27.0


@pytest.mark.asyncio
async def test_dashboard_summary_computes_uptime_from_started_at(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    started_at = datetime.now(UTC) - timedelta(minutes=12)
    heartbeat = {
        "running": True,
        "mode": "paper",
        "paused": False,
        "heartbeat_at": datetime.now(UTC).isoformat(),
        "started_at": started_at.isoformat(),
        "uptime_seconds": 0,
        "decision_interval": 30,
        "current_stage": "fetch_features",
        "round_active": True,
        "last_round_started_at": (datetime.now(UTC) - timedelta(seconds=8)).isoformat(),
        "last_round_finished_at": None,
    }
    (data_dir / "trading_runtime_status.json").write_text(
        json.dumps(heartbeat),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings.__class__, "data_dir", property(lambda _self: data_dir))

    async def recent_activity(_hours: int = 6) -> dict[str, Any]:
        return {
            "decision_count": 0,
            "order_count": 0,
            "window_hours": _hours,
        }

    monkeypatch.setattr(dashboard, "_recent_trading_activity_stats", recent_activity)

    stats = await dashboard._split_process_trading_stats("paper")

    assert stats["running"] is True
    assert stats["decision_interval"] == 30
    assert stats["uptime_seconds"] >= 700
    assert stats["started_at"] == heartbeat["started_at"]
    assert stats["last_heartbeat_at"] == heartbeat["heartbeat_at"]


@pytest.mark.asyncio
async def test_self_check_uses_split_process_runtime_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    heartbeat = {
        "running": True,
        "mode": "paper",
        "paused": False,
        "heartbeat_at": datetime.now(UTC).isoformat(),
        "uptime_seconds": 7200,
        "decision_interval": 30,
        "market_loop_interval_seconds": 10.5,
        "position_loop_interval_seconds": 19.5,
        "market_round_time_budget_seconds": 27.0,
        "current_stage": "review_open_positions",
        "round_active": True,
        "last_round_started_at": datetime.now(UTC).isoformat(),
        "last_round_finished_at": None,
    }
    (data_dir / "trading_runtime_status.json").write_text(
        json.dumps(heartbeat),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings.__class__, "data_dir", property(lambda _self: data_dir))
    monkeypatch.setattr(system_health._dash, "_trading_service", None)

    async def recent_activity(_hours: int = 2) -> dict[str, Any]:
        return {
            "decision_count": 0,
            "order_count": 0,
            "window_hours": _hours,
        }

    monkeypatch.setattr(system_health, "_recent_trading_activity_snapshot", recent_activity)

    item = await system_health._trading_service_running_item()

    assert item["status"] == "ok"
    assert item["details"]["source"] == "runtime_heartbeat"
    assert item["details"]["decision_interval"] == 30
    assert item["details"]["market_loop_interval_seconds"] == 10.5
    assert item["details"]["position_loop_interval_seconds"] == 19.5
    assert item["details"]["market_round_time_budget_seconds"] == 27.0
    assert "心跳正常" in item["message"]


@pytest.mark.asyncio
async def test_self_check_warns_when_split_process_round_is_stuck(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    heartbeat = {
        "running": True,
        "mode": "paper",
        "paused": False,
        "heartbeat_at": datetime.now(UTC).isoformat(),
        "uptime_seconds": 7200,
        "decision_interval": 30,
        "current_stage": "fetch_features",
        "round_active": True,
        "last_round_started_at": (datetime.now(UTC) - timedelta(seconds=180)).isoformat(),
        "last_round_finished_at": (datetime.now(UTC) - timedelta(seconds=240)).isoformat(),
    }
    (data_dir / "trading_runtime_status.json").write_text(
        json.dumps(heartbeat),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings.__class__, "data_dir", property(lambda _self: data_dir))
    monkeypatch.setattr(system_health._dash, "_trading_service", None)

    async def recent_activity(_hours: int = 2) -> dict[str, Any]:
        return {
            "decision_count": 0,
            "order_count": 0,
            "window_hours": _hours,
        }

    monkeypatch.setattr(system_health, "_recent_trading_activity_snapshot", recent_activity)

    item = await system_health._trading_service_running_item()

    assert item["status"] == "warning"
    assert item["details"]["source"] == "runtime_heartbeat"
    assert item["details"]["round_stuck"] is True
    assert item["details"]["current_stage"] == "fetch_features"
    assert "耗时过长" in item["message"]


@pytest.mark.asyncio
async def test_self_check_warns_when_market_round_is_stuck(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    heartbeat = {
        "running": True,
        "mode": "paper",
        "paused": False,
        "heartbeat_at": datetime.now(UTC).isoformat(),
        "uptime_seconds": 7200,
        "decision_interval": 30,
        "current_stage": "analyze:BTC/USDT",
        "round_active": True,
        "last_round_started_at": (datetime.now(UTC) - timedelta(seconds=60)).isoformat(),
        "last_round_finished_at": None,
        "last_market_round_started_at": (datetime.now(UTC) - timedelta(seconds=240)).isoformat(),
        "last_market_round_finished_at": None,
    }
    (data_dir / "trading_runtime_status.json").write_text(
        json.dumps(heartbeat),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings.__class__, "data_dir", property(lambda _self: data_dir))
    monkeypatch.setattr(system_health._dash, "_trading_service", None)

    async def recent_activity(_hours: int = 2) -> dict[str, Any]:
        return {
            "decision_count": 0,
            "order_count": 0,
            "window_hours": _hours,
        }

    monkeypatch.setattr(system_health, "_recent_trading_activity_snapshot", recent_activity)

    item = await system_health._trading_service_running_item()

    assert item["status"] == "warning"
    assert item["details"]["market_round_stuck"] is True
    assert item["details"]["market_round_active"] is True
    assert item["details"]["current_stage"] == "analyze:BTC/USDT"
    assert "市场分析轮次耗时过长" in item["message"]


@pytest.mark.asyncio
async def test_self_check_uses_position_watchdog_for_position_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    heartbeat = {
        "running": True,
        "mode": "paper",
        "paused": False,
        "heartbeat_at": datetime.now(UTC).isoformat(),
        "uptime_seconds": 7200,
        "decision_interval": 30,
        "current_stage": "review_open_positions",
        "position_current_stage": "review_open_positions",
        "round_active": True,
        "last_round_started_at": (datetime.now(UTC) - timedelta(seconds=105)).isoformat(),
        "last_round_finished_at": None,
        "last_position_round_started_at": (datetime.now(UTC) - timedelta(seconds=105)).isoformat(),
        "last_position_round_finished_at": None,
        "position_analysis_watchdog_seconds": 180,
    }
    (data_dir / "trading_runtime_status.json").write_text(
        json.dumps(heartbeat),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings.__class__, "data_dir", property(lambda _self: data_dir))
    monkeypatch.setattr(system_health._dash, "_trading_service", None)

    async def recent_activity(_hours: int = 2) -> dict[str, Any]:
        return {
            "decision_count": 1,
            "order_count": 0,
            "window_hours": _hours,
        }

    monkeypatch.setattr(system_health, "_recent_trading_activity_snapshot", recent_activity)

    item = await system_health._trading_service_running_item()

    assert item["status"] == "ok"
    assert item["details"]["position_round_stuck"] is False
    assert item["details"]["position_round_stuck_limit_seconds"] == 180
    assert item["details"]["position_analysis_watchdog_seconds"] == 180


@pytest.mark.asyncio
async def test_recent_execution_old_missing_score_downgrades_after_new_scored_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        def scalars(self) -> FakeResult:
            return self

        def all(self) -> list[Any]:
            return self._rows

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _stmt: Any) -> FakeResult:
            self.calls += 1
            if self.calls == 1:
                return FakeResult([])
            return FakeResult(
                [
                    SimpleNamespace(
                        id=202,
                        action="short",
                        created_at=system_health.datetime(2026, 6, 17, 4, 10),
                        raw_llm_response={
                            "opportunity_score": {"score": 1.25},
                            "decision_state_machine": {"stages": [{"status": "passed"}]},
                        },
                    ),
                    SimpleNamespace(
                        id=101,
                        action="short",
                        created_at=system_health.datetime(2026, 6, 17, 4, 0),
                        raw_llm_response={"decision_state_machine": {"stages": []}},
                    ),
                ]
            )

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    monkeypatch.setattr(system_health, "get_session_ctx", fake_session_ctx)

    items = await system_health._recent_execution_items()
    by_key = {item["key"]: item for item in items}

    assert by_key["entry_opportunity_score_coverage"]["status"] == "info"
    assert by_key["entry_opportunity_score_coverage"]["details"]["sample_decision_ids"] == [101]


@pytest.mark.asyncio
async def test_recent_execution_no_orders_is_info_not_system_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        def scalars(self) -> FakeResult:
            return self

        def all(self) -> list[Any]:
            return self._rows

    class FakeSession:
        async def execute(self, _stmt: Any) -> FakeResult:
            return FakeResult([])

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    monkeypatch.setattr(system_health, "get_session_ctx", fake_session_ctx)

    items = await system_health._recent_execution_items()
    by_key = {item["key"]: item for item in items}

    assert by_key["recent_execution"]["status"] == "info"
    assert by_key["recent_execution"]["details"]["filled_orders"] == 0
    assert by_key["recent_execution"]["details"]["is_system_failure"] is False


@pytest.mark.asyncio
async def test_recent_execution_step_trace_is_info_when_new_records_have_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        def scalars(self) -> FakeResult:
            return self

        def all(self) -> list[Any]:
            return self._rows

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _stmt: Any) -> FakeResult:
            self.calls += 1
            if self.calls == 1:
                return FakeResult([])
            return FakeResult(
                [
                    SimpleNamespace(
                        id=301,
                        action="short",
                        created_at=system_health.datetime(2026, 6, 17, 4, 10),
                        raw_llm_response={
                            "opportunity_score": {"score": 1.25},
                            "decision_state_machine": {"stages": [{"status": "passed"}]},
                        },
                    ),
                    SimpleNamespace(
                        id=302,
                        action="short",
                        created_at=system_health.datetime(2026, 6, 17, 4, 0),
                        raw_llm_response={"decision_state_machine": {"stages": []}},
                    ),
                ]
            )

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    monkeypatch.setattr(system_health, "get_session_ctx", fake_session_ctx)

    items = await system_health._recent_execution_items()
    by_key = {item["key"]: item for item in items}

    assert by_key["decision_trace_coverage"]["status"] == "info"
    assert by_key["decision_trace_coverage"]["details"]["traced_decisions"] == 1


@pytest.mark.asyncio
async def test_recent_blocked_decisions_are_info_not_system_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        def scalars(self) -> FakeResult:
            return self

        def all(self) -> list[Any]:
            return self._rows

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _stmt: Any) -> FakeResult:
            self.calls += 1
            if self.calls == 1:
                return FakeResult(
                    [
                        SimpleNamespace(
                            id=201,
                            status="filled",
                            created_at=system_health.datetime(
                                2026,
                                6,
                                17,
                                4,
                                30,
                                tzinfo=system_health.UTC,
                            ),
                        )
                    ]
                )
            return FakeResult(
                [
                    SimpleNamespace(
                        id=301,
                        action="long",
                        created_at=system_health.datetime(
                            2026,
                            6,
                            17,
                            4,
                            20,
                            tzinfo=system_health.UTC,
                        ),
                        raw_llm_response={
                            "decision_state_machine": {
                                "stages": [{"status": "blocked"}],
                            }
                        },
                    )
                ]
            )

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    monkeypatch.setattr(system_health, "get_session_ctx", fake_session_ctx)

    items = await system_health._recent_execution_items()
    by_key = {item["key"]: item for item in items}

    assert by_key["recent_execution"]["status"] == "ok"
    assert by_key["recent_blocked_decisions"]["status"] == "info"
    assert by_key["recent_blocked_decisions"]["details"]["sample_decision_ids"] == [301]


@pytest.mark.asyncio
async def test_recent_failed_orders_become_info_after_new_successful_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        def scalars(self) -> FakeResult:
            return self

        def all(self) -> list[Any]:
            return self._rows

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _stmt: Any) -> FakeResult:
            self.calls += 1
            if self.calls == 1:
                return FakeResult(
                    [
                        SimpleNamespace(
                            id=2346,
                            status="rejected",
                            created_at=system_health.datetime(
                                2026,
                                6,
                                17,
                                5,
                                31,
                                tzinfo=system_health.UTC,
                            ),
                        ),
                        SimpleNamespace(
                            id=2401,
                            status="filled",
                            created_at=system_health.datetime(
                                2026,
                                6,
                                17,
                                6,
                                40,
                                tzinfo=system_health.UTC,
                            ),
                        ),
                    ]
                )
            return FakeResult(
                [
                    SimpleNamespace(
                        id=301,
                        action="short",
                        created_at=system_health.datetime(
                            2026,
                            6,
                            17,
                            6,
                            40,
                            tzinfo=system_health.UTC,
                        ),
                        raw_llm_response={
                            "opportunity_score": {"score": 2.0},
                            "decision_state_machine": {"stages": [{"status": "passed"}]},
                        },
                    )
                ]
            )

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    monkeypatch.setattr(system_health, "get_session_ctx", fake_session_ctx)

    items = await system_health._recent_execution_items()
    by_key = {item["key"]: item for item in items}

    assert by_key["recent_failed_orders"]["status"] == "info"
    assert by_key["recent_failed_orders"]["details"]["has_unresolved_order"] is False
    assert by_key["recent_failed_orders"]["details"]["sample_order_ids"] == [2346]
    assert by_key["recent_execution"]["status"] == "ok"


@pytest.mark.asyncio
async def test_recent_failed_orders_known_untradable_reject_is_info_without_later_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        def scalars(self) -> FakeResult:
            return self

        def all(self) -> list[Any]:
            return self._rows

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _stmt: Any) -> FakeResult:
            self.calls += 1
            if self.calls == 1:
                return FakeResult(
                    [
                        SimpleNamespace(
                            id=2561,
                            status="rejected",
                            created_at=system_health.datetime(
                                2026,
                                6,
                                23,
                                7,
                                45,
                                tzinfo=system_health.UTC,
                            ),
                            decision_id=77,
                            exchange_order_id="rejected",
                        )
                    ]
                )
            return FakeResult(
                [
                    SimpleNamespace(
                        id=77,
                        action="long",
                        created_at=system_health.datetime(
                            2026,
                            6,
                            23,
                            7,
                            44,
                            tzinfo=system_health.UTC,
                        ),
                        execution_reason=(
                            "OKX 提示该交易对当前不可交易，可能受账户地区/合规限制影响；"
                            "系统已暂时跳过该交易对，避免重复分析和下单。51155"
                        ),
                        reasoning="entry rejected by exchange",
                        raw_llm_response={
                            "opportunity_score": {"score": 2.0},
                            "decision_state_machine": {"stages": [{"status": "failed"}]},
                        },
                    )
                ]
            )

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    monkeypatch.setattr(system_health, "get_session_ctx", fake_session_ctx)

    items = await system_health._recent_execution_items()
    by_key = {item["key"]: item for item in items}

    assert by_key["recent_failed_orders"]["status"] == "info"
    assert by_key["recent_failed_orders"]["details"]["has_unresolved_order"] is False
    assert by_key["recent_failed_orders"]["details"]["handled_terminal_failure_count"] == 1
    assert by_key["recent_failed_orders"]["details"]["unhandled_terminal_failure_count"] == 0
    assert by_key["recent_failed_orders"]["details"]["sample_handled_order_ids"] == [2561]


@pytest.mark.asyncio
async def test_recent_failed_orders_loads_linked_decision_outside_recent_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        def scalars(self) -> FakeResult:
            return self

        def all(self) -> list[Any]:
            return self._rows

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _stmt: Any) -> FakeResult:
            self.calls += 1
            if self.calls == 1:
                return FakeResult(
                    [
                        SimpleNamespace(
                            id=2561,
                            status="rejected",
                            created_at=system_health.datetime(
                                2026,
                                6,
                                23,
                                7,
                                45,
                                tzinfo=system_health.UTC,
                            ),
                            decision_id=177,
                            exchange_order_id="rejected",
                        )
                    ]
                )
            if self.calls == 2:
                return FakeResult([])
            return FakeResult(
                [
                    SimpleNamespace(
                        id=177,
                        action="long",
                        created_at=system_health.datetime(
                            2026,
                            6,
                            23,
                            1,
                            0,
                            tzinfo=system_health.UTC,
                        ),
                        execution_reason="OKX 51155 local compliance restrictions",
                        raw_llm_response={},
                    )
                ]
            )

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    monkeypatch.setattr(system_health, "get_session_ctx", fake_session_ctx)

    items = await system_health._recent_execution_items()
    by_key = {item["key"]: item for item in items}

    assert by_key["recent_failed_orders"]["status"] == "info"
    assert by_key["recent_failed_orders"]["details"]["handled_terminal_failure_count"] == 1


@pytest.mark.asyncio
async def test_recent_failed_orders_unknown_terminal_reject_stays_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        def scalars(self) -> FakeResult:
            return self

        def all(self) -> list[Any]:
            return self._rows

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, _stmt: Any) -> FakeResult:
            self.calls += 1
            if self.calls == 1:
                return FakeResult(
                    [
                        SimpleNamespace(
                            id=2601,
                            status="rejected",
                            created_at=system_health.datetime(
                                2026,
                                6,
                                23,
                                8,
                                0,
                                tzinfo=system_health.UTC,
                            ),
                            decision_id=88,
                            exchange_order_id="rejected",
                        )
                    ]
                )
            return FakeResult(
                [
                    SimpleNamespace(
                        id=88,
                        action="short",
                        created_at=system_health.datetime(
                            2026,
                            6,
                            23,
                            7,
                            59,
                            tzinfo=system_health.UTC,
                        ),
                        execution_reason="Failed to place order",
                        raw_llm_response={
                            "opportunity_score": {"score": 2.0},
                            "decision_state_machine": {"stages": [{"status": "failed"}]},
                        },
                    )
                ]
            )

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    monkeypatch.setattr(system_health, "get_session_ctx", fake_session_ctx)

    items = await system_health._recent_execution_items()
    by_key = {item["key"]: item for item in items}

    assert by_key["recent_failed_orders"]["status"] == "warning"
    assert by_key["recent_failed_orders"]["details"]["has_unresolved_order"] is False
    assert by_key["recent_failed_orders"]["details"]["handled_terminal_failure_count"] == 0
    assert by_key["recent_failed_orders"]["details"]["unhandled_terminal_failure_count"] == 1
