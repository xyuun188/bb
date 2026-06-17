from __future__ import annotations

from contextlib import asynccontextmanager
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
                "name": "trend_expert",
                "api_base": "http://127.0.0.1:8003/v1",
                "api_key": "unit-key",
                "model": "qwen3-14b-trade",
                "enabled": True,
            },
            {
                "name": "sentiment_expert",
                "api_base": "http://127.0.0.1:18002/v1",
                "api_key": "unit-key",
                "model": "deepseek-r1-14b-risk",
                "enabled": True,
            },
        ],
    )
    monkeypatch.setattr(settings, "local_ai_tools_api_base", "http://127.0.0.1:18001")
    monkeypatch.setattr(settings, "high_risk_review_model", "deepseek-r1-14b-risk")
    monkeypatch.setattr(settings, "high_risk_review_api_base", "http://127.0.0.1:18002/v1")

    items = system_health._configured_endpoint_items()
    by_key = {item["key"]: item for item in items}

    assert by_key["endpoint_qwen3-14b-trade"]["status"] == "critical"
    assert by_key["endpoint_qwen3-14b-trade"]["details"]["expected_platform_endpoint"] == (
        "http://127.0.0.1:18000/v1"
    )
    assert by_key["endpoint_local_ai_tools"]["status"] == "ok"
    assert by_key["endpoint_deepseek-r1-14b-risk"]["status"] == "ok"


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

    monkeypatch.setattr(system_health, "get_server_monitor_status_async", monitor_status)
    monkeypatch.setattr(system_health, "_recent_execution_items", recent_items)

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

    assert by_key["entry_opportunity_score_coverage"]["status"] == "warning"
    assert by_key["entry_opportunity_score_coverage"]["details"]["sample_decision_ids"] == [101]


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
