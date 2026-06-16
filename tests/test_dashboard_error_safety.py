from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import httpx

from core.safe_output import safe_error_text
from core.server_monitor_probe import SERVER_MONITOR_REMOTE_COMMAND_TIMEOUT_SECONDS
from services import server_monitor_status
from web_dashboard.api import dashboard, symbols


class _FakeServerInfo:
    host = "203.0.113.17"


class _FakeSSH:
    def __init__(self) -> None:
        self.closed = False
        self.exec_calls: list[dict[str, Any]] = []

    def close(self) -> None:
        self.closed = True


def _build_server_monitor_service(
    *,
    status: int = 0,
    stdout: str = "{}",
    stderr: str = "",
    raise_exc: Exception | None = None,
    clock=None,
) -> tuple[server_monitor_status.ServerMonitorStatusService, _FakeSSH]:
    ssh = _FakeSSH()

    def fake_exec_remote_command(*args: Any, **kwargs: Any) -> SimpleNamespace:
        ssh.exec_calls.append({"args": args, **kwargs})
        if raise_exc is not None:
            raise raise_exc
        return SimpleNamespace(status=status, stdout=stdout, stderr=stderr)

    service = server_monitor_status.ServerMonitorStatusService(
        model_id_provider=lambda: "qwen3-32b-trade",
        info_loader=lambda _root: _FakeServerInfo(),
        ssh_connector=lambda *args, **kwargs: ssh,
        command_executor=fake_exec_remote_command,
        clock=clock or server_monitor_status.monotonic,
    )
    return service, ssh


def test_safe_error_text_redacts_and_truncates_secret_bearing_text() -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    text = f"Authorization: Bearer {leaked_value} failed. " "x" * 120

    result = safe_error_text(text, limit=40)

    assert leaked_value not in result
    assert "Authorization: ***" in result
    assert len(result) == 43
    assert result.endswith("...")


def test_dashboard_execution_account_payload_separates_allocation_from_okx_equity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dashboard.settings,
        "execution_account_balances",
        {**dashboard.settings.execution_account_balances, "paper": 123.45},
    )
    monkeypatch.setattr(dashboard, "_trading_service", None)

    payload = dashboard._build_execution_account_status(
        "paper",
        paper_summary={"available_balance": 100.0, "positions": []},
        okx_account={"free": 200.0, "used": 10.0, "total": 250.0, "equity": 260.0},
        pnl_summary={},
    )

    assert payload["allocated_balance"] == 123.45
    assert payload["account_balance_source_value"] == 260.0
    assert payload["account_equity"] == 260.0


def test_dashboard_fallback_logger_redacts_exception_text(
    monkeypatch,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    events: list[dict[str, Any]] = []

    class FakeLogger:
        def debug(self, event: str, **fields: Any) -> None:
            events.append({"event": event, **fields})

    monkeypatch.setattr(dashboard, "logger", FakeLogger())

    dashboard._log_dashboard_fallback(
        "unit fallback",
        RuntimeError(f"Authorization: Bearer {leaked_value} failed"),
        mode="paper",
    )

    assert events == [
        {
            "event": "unit fallback",
            "error": "Authorization: *** failed",
            "mode": "paper",
        }
    ]


def test_server_monitor_uses_probe_timeout_budget() -> None:
    service, ssh = _build_server_monitor_service()

    result = service.collect_sync()

    assert result["available"] is True
    assert ssh.closed is True
    assert ssh.exec_calls
    assert ssh.exec_calls[0]["timeout"] == SERVER_MONITOR_REMOTE_COMMAND_TIMEOUT_SECONDS
    assert ssh.exec_calls[0]["timeout"] > 12


def test_server_monitor_status_can_use_async_loaded_info_override() -> None:
    def unexpected_loader(_root: object) -> object:
        raise AssertionError("sync info loader should not run")

    service, ssh = _build_server_monitor_service(stdout='{"hostname": "model-host"}')
    service.info_loader = unexpected_loader

    result = service.get_status_sync(info_override=_FakeServerInfo())

    assert ssh.exec_calls
    assert result["available"] is True
    assert result["hostname"] == "model-host"


def test_server_monitor_defaults_to_model_server_info_loader() -> None:
    service = server_monitor_status.ServerMonitorStatusService(
        model_id_provider=lambda: "qwen3-32b-trade",
    )

    assert service.info_loader is server_monitor_status.load_model_server_info_for_monitor


def test_server_monitor_command_timeout_is_classified_and_redacted() -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    service, ssh = _build_server_monitor_service(
        raise_exc=TimeoutError(f"Authorization: Bearer {leaked_value} timed out"),
    )

    result = service.collect_sync()

    assert ssh.closed is True
    assert result["available"] is False
    assert result["status"] == "remote_command_timeout"
    assert leaked_value not in result["message"]
    assert "Authorization: ***" in result["message"]


def test_server_monitor_status_uses_short_cache() -> None:
    clock = [100.0]
    service, ssh = _build_server_monitor_service(
        stdout='{"hostname": "model-host"}',
        clock=lambda: clock[0],
    )

    first = service.get_status_sync()
    first["hostname"] = "mutated-by-caller"
    second = service.get_status_sync()

    assert ssh.exec_calls
    assert len(ssh.exec_calls) == 1
    assert second["hostname"] == "model-host"
    assert second["available"] is True
    assert second["cache"]["status"] == "fresh"


def test_server_monitor_status_cache_expires() -> None:
    clock = [200.0]
    service, ssh = _build_server_monitor_service(
        stdout='{"hostname": "model-host"}',
        clock=lambda: clock[0],
    )

    service.get_status_sync()
    clock[0] += service.cache_ttl_seconds + 0.1
    service.get_status_sync()

    assert len(ssh.exec_calls) == 2


def test_server_monitor_status_keeps_stale_cache_when_refresh_fails() -> None:
    clock = [250.0]
    service, good_ssh = _build_server_monitor_service(
        stdout='{"hostname": "model-host"}',
        clock=lambda: clock[0],
    )

    service.get_status_sync()
    clock[0] += service.cache_ttl_seconds + 0.1
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    bad_ssh = _FakeSSH()

    def failing_exec_remote_command(*args: Any, **kwargs: Any) -> SimpleNamespace:
        bad_ssh.exec_calls.append({"args": args, **kwargs})
        return SimpleNamespace(
            status=1,
            stdout="",
            stderr=f"Authorization: Bearer {leaked_value} failed",
        )

    service.ssh_connector = lambda *args, **kwargs: bad_ssh
    service.command_executor = failing_exec_remote_command

    result = service.get_status_sync()

    assert len(good_ssh.exec_calls) == 1
    assert len(bad_ssh.exec_calls) == 1
    assert result["available"] is True
    assert result["hostname"] == "model-host"
    assert result["cache"]["status"] == "stale_refresh_failed"
    assert result["refresh_error"]["status"] == "remote_command_failed"
    assert leaked_value not in result["refresh_error"]["message"]
    assert "Authorization: ***" in result["refresh_error"]["message"]


def test_server_monitor_status_returns_stale_cache_while_refreshing() -> None:
    clock = [300.0]
    service, ssh = _build_server_monitor_service(
        stdout='{"hostname": "model-host"}',
        clock=lambda: clock[0],
    )

    service.get_status_sync()
    clock[0] += service.cache_ttl_seconds + 0.1

    assert service._refresh_lock.acquire(blocking=False) is True
    try:
        result = service.get_status_sync()
    finally:
        service._refresh_lock.release()

    assert len(ssh.exec_calls) == 1
    assert result["hostname"] == "model-host"
    assert result["cache"]["status"] == "stale_refreshing"
    assert result["cache"]["age_seconds"] > service.cache_ttl_seconds


def test_server_monitor_status_returns_refreshing_without_cache() -> None:
    service, ssh = _build_server_monitor_service(stdout='{"hostname": "model-host"}')

    assert service._refresh_lock.acquire(blocking=False) is True
    try:
        result = service.get_status_sync()
    finally:
        service._refresh_lock.release()

    assert len(ssh.exec_calls) == 0
    assert result["available"] is False
    assert result["status"] == "server_monitor_refreshing"
    assert result["cache"]["status"] == "initial_refreshing"


def test_server_monitor_invalid_json_payload_is_not_reported_as_ssh_failed() -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    service, ssh = _build_server_monitor_service(
        stdout=f"boot log Authorization: Bearer {leaked_value}",
    )

    result = service.collect_sync()

    assert ssh.closed is True
    assert result["available"] is False
    assert result["status"] == "remote_payload_invalid"
    assert leaked_value not in result["message"]
    assert "Authorization: ***" in result["message"]


def test_server_monitor_non_object_json_payload_is_classified_without_exception() -> None:
    service, _ssh = _build_server_monitor_service(stdout="[]")

    result = service.collect_sync()

    assert result["available"] is False
    assert result["status"] == "remote_payload_invalid"
    assert "non-object JSON payload" in result["message"]


def test_server_monitor_remote_command_failure_is_redacted() -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    service, _ssh = _build_server_monitor_service(
        status=1,
        stdout='{"token": "stdout-secret-value"}',
        stderr=f"password=stderr-secret Authorization: Bearer {leaked_value}",
    )

    result = service.collect_sync()

    assert result["available"] is False
    assert result["status"] == "remote_command_failed"
    assert leaked_value not in result["message"]
    assert "stderr-secret" not in result["message"]
    assert "Authorization: ***" in result["message"]
    assert "password=***" in result["message"]


async def test_collect_platform_runtime_status_probes_real_local_tool_endpoints(
    monkeypatch,
) -> None:
    requests: list[tuple[str, str, str]] = []

    fake_settings = SimpleNamespace(
        get_fixed_ai_models=lambda include_empty=False: [
            {
                "name": "main",
                "label": "主模型",
                "api_base": "http://llm.test/v1",
                "api_key": "hidden-llm-key",
                "model": "qwen3-14b-trade",
                "enabled": True,
            }
        ],
        local_ai_tools_api_base="http://local-ai.test",
        local_ai_tools_api_key="hidden-tools-key",
        ai_api_key="",
    )
    monkeypatch.setattr(server_monitor_status, "settings", fake_settings)

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def request(
            self,
            method: str,
            url: str,
            headers: dict[str, str] | None = None,
            json: dict[str, Any] | None = None,
        ) -> httpx.Response:
            requests.append((method, url, str((headers or {}).get("Authorization") or "")))
            request = httpx.Request(method, url)
            if url == "http://llm.test/v1/models":
                return httpx.Response(
                    200,
                    json={"data": [{"id": "qwen3-14b-trade"}]},
                    request=request,
                )
            if url == "http://local-ai.test/health":
                return httpx.Response(500, json={"detail": "booting"}, request=request)
            if url == "http://local-ai.test/models/status":
                return httpx.Response(503, json={"available": False}, request=request)
            if url.endswith("/profit/predict"):
                return httpx.Response(200, json={"available": True}, request=request)
            return httpx.Response(500, json={"available": False}, request=request)

    monkeypatch.setattr(server_monitor_status.httpx, "AsyncClient", FakeAsyncClient)

    result = await server_monitor_status.collect_platform_runtime_status()

    assert result["ai_models"][0]["available"] is True
    tools = result["local_ai_tools"]
    assert tools["available"] is True
    assert tools["health"]["ok"] is False
    assert tools["status"]["ok"] is False
    assert tools["model_bundle_available"] is False
    assert tools["child_endpoints"]["profit_prediction"]["available"] is True
    assert tools["child_endpoints"]["exit_advice"]["available"] is False
    assert ("POST", "http://local-ai.test/profit/predict", "Bearer hidden-tools-key") in requests


async def test_symbols_available_error_response_is_redacted(
    monkeypatch,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"

    async def failing_sdk_symbols() -> list[str]:
        raise RuntimeError("SDK unavailable")

    class FakeDataService:
        async def get_available_symbols(self) -> list[str]:
            raise RuntimeError(f"Authorization: Bearer {leaked_value} failed")

    monkeypatch.setitem(
        sys.modules,
        "data_feed.okx_sdk_client",
        SimpleNamespace(get_available_symbols=failing_sdk_symbols),
    )
    monkeypatch.setattr(dashboard, "_data_service", FakeDataService())

    result = await symbols.get_available_symbols()

    assert result["count"] == 0
    assert result["symbols"] == []
    assert leaked_value not in result["error"]
    assert result["error"] == "Authorization: *** failed"
