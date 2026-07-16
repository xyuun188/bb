from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from config.settings import settings
from core.server_monitor_probe import (
    display_provider_model_name,
    render_python_here_doc,
    render_server_monitor_probe,
)
from services import server_monitor_status

ROOT = Path(__file__).resolve().parents[1]


def _probe_prelude(script: str) -> str:
    """Return the import/function section before the remote Linux runtime probe."""
    return script.split("\nload = os.getloadavg()", 1)[0]


def _load_probe_namespace(script: str) -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(_probe_prelude(script), namespace)  # noqa: S102 - executes generated test prelude only.
    return namespace


def _load_safe_error(script: str) -> Callable[[object], str]:
    namespace = _load_probe_namespace(script)
    return cast(Callable[[object], str], namespace["safe_error"])


def test_server_monitor_probe_uses_argv_subprocess_commands() -> None:
    script = render_server_monitor_probe("qwen3-32b-trade", "Qwen3 32B")

    assert "shell=True" not in script
    assert "subprocess.run(\n            args," in script
    assert '["systemctl", "is-active", name]' in script
    assert '["ps", "-p", pid, "-o", "etime="]' in script
    assert "vllm_endpoint_runtime(8000" in script
    assert "vllm_endpoint_runtime(8002" in script
    assert "vllm_endpoint_runtime(8003" in script
    assert "http://127.0.0.1:8101/health" in script
    assert "http://127.0.0.1:8101/models/status" in script
    assert "http://127.0.0.1:8001/models/status" not in script


def test_server_monitor_probe_redacts_remote_error_text() -> None:
    script = render_server_monitor_probe("qwen3-32b-trade", "Qwen3 32B")

    assert "safe_error(exc)" in script
    assert "str(exc)" not in script
    compile(script, "<server-monitor-probe>", "exec")

    safe_error = _load_safe_error(script)
    bearer_value = "abcdefghijklmnopqrstuvwxyz" + "123456"

    redacted = safe_error(
        f"Authorization: Bearer {bearer_value} " "password=plain-secret token=local-token"
    )

    assert "abcdefghijklmnopqrstuvwxyz" not in redacted
    assert "plain-secret" not in redacted
    assert "local-token" not in redacted
    assert "Authorization: Bearer ***" in redacted
    assert "password=***" in redacted
    assert "token=***" in redacted


def test_server_monitor_probe_redacts_json_style_remote_secrets() -> None:
    script = render_server_monitor_probe("qwen3-32b-trade", "Qwen3 32B")
    safe_error = _load_safe_error(script)

    redacted = safe_error(
        '{"token": "json-token-value", "api_key": "json-api-key"} ' "access_token=query-token-value"
    )

    assert "json-token-value" not in redacted
    assert "json-api-key" not in redacted
    assert "query-token-value" not in redacted
    assert '"token": "***"' in redacted
    assert '"api_key": "***"' in redacted
    assert "access_token=***" in redacted


def test_server_monitor_probe_safe_prelude_runs_without_shell() -> None:
    script = render_server_monitor_probe("qwen3-32b-trade", "Qwen3 32B")

    safe_error = _load_safe_error(script)

    assert safe_error("api_key=abc") == "api_key=***"


def test_server_monitor_probe_json_escapes_model_values() -> None:
    script = render_server_monitor_probe('model"\\name', "Label")

    assert 'PRIMARY_MODEL_ID = "model\\"\\\\name"' in script
    assert 'PRIMARY_MODEL_LABEL = "Label"' in script


def test_render_python_here_doc_rejects_delimiter_collision() -> None:
    with pytest.raises(ValueError, match="heredoc"):
        render_python_here_doc("print('x')\nPY\nprint('y')")


def test_display_provider_model_name_uses_stable_labels() -> None:
    assert display_provider_model_name("Qwen/Qwen3-32B-AWQ") == "Qwen3-32B-AWQ"
    assert display_provider_model_name("qwen3-32b-trade") == "Qwen3-32B"
    assert display_provider_model_name("qwen3-14b-trade") == "Qwen3-14B-Instruct"
    assert display_provider_model_name("Qwen/Qwen2.5-32B-Instruct") == "Qwen2.5-32B-Instruct"
    assert display_provider_model_name("deepseek-v3") == "deepseek-v3"
    assert display_provider_model_name("") == "Local Model"


def test_server_monitor_probe_reports_endpoint_and_model_health() -> None:
    script = render_server_monitor_probe("qwen3-32b-trade", "Qwen3 32B")
    namespace = _load_probe_namespace(script)

    def fake_http_json(
        url: str,
        timeout: int = 3,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        if url.endswith("/v1/models"):
            return {
                "ok": True,
                "status_code": 200,
                "latency_ms": 12.4,
                "truncated": False,
                "data": {"data": [{"id": "other-model"}]},
            }
        if url == "http://127.0.0.1:8101/health":
            return {
                "ok": True,
                "status_code": 200,
                "latency_ms": 25.0,
                "truncated": False,
                "data": {
                    "ok": True,
                    "service": "phase3_quant_api",
                    "root": "/data/BB",
                    "validation_all_ok": True,
                },
            }
        if url == "http://127.0.0.1:8101/models/status":
            return {
                "ok": True,
                "status_code": 200,
                "latency_ms": 18.0,
                "truncated": False,
                "data": {
                    "available": False,
                    "message": "No trained local quant bundle found; return inference is unavailable.",
                    "shadow_sample_count": 0,
                    "completed_shadow_sample_count": 0,
                },
            }
        return {
            "ok": True,
            "status_code": 200,
            "latency_ms": 4.0,
            "truncated": False,
            "data": {"ok": True},
        }

    namespace["http_json"] = fake_http_json
    runtime = cast(Callable[[], dict[str, object]], namespace["model_runtime"])()
    vllm = cast(dict[str, object], runtime["vllm"])
    vllm_endpoints = cast(list[dict[str, object]], runtime["vllm_endpoints"])
    tools = cast(dict[str, object], runtime["local_ai_tools"])

    assert vllm["endpoint_available"] is True
    assert vllm["model_available"] is False
    assert vllm["available"] is False
    assert vllm["status"] == "model_mismatch"
    assert vllm["model_mismatch"] is True
    assert vllm["label"] == "Qwen3 32B"
    assert vllm["provider_model"] == "qwen3-32b-trade"
    assert cast(dict[str, object], vllm["health"])["status_code"] == 200
    assert cast(dict[str, object], vllm["health"])["latency_ms"] == 12.4
    assert [item["endpoint"] for item in vllm_endpoints] == [
        "127.0.0.1:8000/v1",
        "127.0.0.1:8002/v1",
        "127.0.0.1:8003/v1",
    ]
    assert vllm_endpoints[1]["provider_model"] == "deepseek-r1-14b-risk"
    assert vllm_endpoints[2]["provider_model"] == "BB-FinQuant-Expert-14B"
    assert tools["available"] is True
    assert tools["endpoint"] == "127.0.0.1:8101"
    assert tools["service_role"] == "phase3_quant_api"
    assert tools["model_bundle_available"] is False
    assert tools["status"] == "artifact_unavailable"
    assert cast(dict[str, object], tools["status_health"])["status_code"] == 200
    assert cast(dict[str, object], tools["health"])["ok"] is True


def test_server_monitor_probe_marks_vllm_port_model_mismatch() -> None:
    script = render_server_monitor_probe("qwen3-32b-trade", "Qwen3 32B")
    namespace = _load_probe_namespace(script)

    def fake_http_json(
        url: str,
        timeout: int = 3,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        return {
            "ok": True,
            "status_code": 200,
            "latency_ms": 4.0,
            "truncated": False,
            "data": {"data": [{"id": "qwen3-32b-trade"}]},
        }

    namespace["http_json"] = fake_http_json
    endpoint_runtime = cast(Callable[..., dict[str, object]], namespace["vllm_endpoint_runtime"])
    result = endpoint_runtime(8003, "BB-FinQuant Expert 14B", "BB-FinQuant-Expert-14B")

    assert result["available"] is False
    assert result["status"] == "model_mismatch"
    assert result["model_mismatch"] is True
    assert "BB-FinQuant-Expert-14B" in str(result["error"])


def test_server_monitor_probe_reports_each_vllm_endpoint_as_service() -> None:
    script = render_server_monitor_probe("qwen3-14b-trade", "Qwen3-14B-Instruct")

    assert 'for item in runtime.get("vllm_endpoints", [])' in script
    assert "vllm_model_service_status(item)" in script
    assert 'runtime.get("vllm", {}).get("models")' not in script
    assert '"provider_model": provider_model' in script
    assert "DeepSeek R1 14B" in script
    assert "deepseek-r1-14b-risk" in script


def test_server_monitor_probe_keeps_primary_vllm_separate_from_available_expert() -> None:
    script = render_server_monitor_probe("qwen3-32b-trade", "Qwen3 32B")
    namespace = _load_probe_namespace(script)

    def fake_http_json(
        url: str,
        timeout: int = 3,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        model_by_port = {
            "8000": "qwen3-32b-trade",
            "8002": "deepseek-r1-14b-risk",
            "8003": "BB-FinQuant-Expert-14B",
        }
        for port, model in model_by_port.items():
            if url == f"http://127.0.0.1:{port}/v1/models":
                return {
                    "ok": True,
                    "status_code": 200,
                    "latency_ms": 3.0,
                    "truncated": False,
                    "data": {"data": [{"id": model}]},
                }
        return {
            "ok": False,
            "status_code": 0,
            "latency_ms": 0.0,
            "truncated": False,
            "data": {},
        }

    namespace["http_json"] = fake_http_json
    runtime = cast(Callable[[], dict[str, object]], namespace["model_runtime"])()
    vllm = cast(dict[str, object], runtime["vllm"])

    assert vllm["endpoint"] == "127.0.0.1:8000/v1"
    assert vllm["provider_model"] == "qwen3-32b-trade"


def test_server_monitor_probe_uses_port_role_when_primary_provider_is_expert() -> None:
    script = render_server_monitor_probe("BB-FinQuant-Expert-14B", "BB-FinQuant-Expert-14B")
    namespace = _load_probe_namespace(script)

    def fake_http_json(
        url: str,
        timeout: int = 3,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        model_by_port = {
            "8000": "qwen3-32b-trade",
            "8002": "deepseek-r1-14b-risk",
            "8003": "BB-FinQuant-Expert-14B",
        }
        for port, model in model_by_port.items():
            if url == f"http://127.0.0.1:{port}/v1/models":
                return {
                    "ok": True,
                    "status_code": 200,
                    "latency_ms": 3.0,
                    "truncated": False,
                    "data": {"data": [{"id": model}]},
                }
        return {
            "ok": False,
            "status_code": 0,
            "latency_ms": 0.0,
            "truncated": False,
            "data": {},
        }

    namespace["http_json"] = fake_http_json
    runtime = cast(Callable[[], dict[str, object]], namespace["model_runtime"])()
    vllm = cast(dict[str, object], runtime["vllm"])
    endpoints = cast(list[dict[str, object]], runtime["vllm_endpoints"])

    assert vllm["endpoint"] == "127.0.0.1:8003/v1"
    assert vllm["provider_model"] == "BB-FinQuant-Expert-14B"
    assert vllm["available"] is True
    assert endpoints[2]["provider_model"] == "BB-FinQuant-Expert-14B"
    assert endpoints[2]["available"] is True


def test_server_monitor_probe_treats_external_primary_as_local_fallback() -> None:
    script = render_server_monitor_probe("deepseek-v4-pro", "deepseek-v4-pro")
    namespace = _load_probe_namespace(script)

    def fake_http_json(
        url: str,
        timeout: int = 3,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        model_by_port = {
            "8000": "qwen3-14b-trade",
            "8002": "deepseek-r1-14b-risk",
            "8003": "BB-FinQuant-Expert-14B",
        }
        for port, model in model_by_port.items():
            if url == f"http://127.0.0.1:{port}/v1/models":
                return {
                    "ok": True,
                    "status_code": 200,
                    "latency_ms": 3.0,
                    "truncated": False,
                    "data": {"data": [{"id": model}]},
                }
        return {
            "ok": False,
            "status_code": 0,
            "latency_ms": 0.0,
            "truncated": False,
            "data": {},
        }

    namespace["http_json"] = fake_http_json
    runtime = cast(Callable[[], dict[str, object]], namespace["model_runtime"])()
    vllm = cast(dict[str, object], runtime["vllm"])
    endpoints = cast(list[dict[str, object]], runtime["vllm_endpoints"])

    assert vllm["endpoint"] == "127.0.0.1:8000/v1"
    assert vllm["label"] == "Local decision fallback"
    assert vllm["provider_model"] == ""
    assert vllm["available"] is True
    assert vllm["model_mismatch"] is False
    assert endpoints[0]["models"] == ["qwen3-14b-trade"]


def test_primary_provider_model_id_prefers_decision_maker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "ai_models",
        [
            {
                "name": "trend_expert",
                "api_base": "http://127.0.0.1:18003/v1",
                "api_key": "unit-key",
                "model": "BB-FinQuant-Expert-14B",
                "enabled": True,
            },
            {
                "name": "decision_maker",
                "api_base": "http://127.0.0.1:18000/v1",
                "api_key": "unit-key",
                "model": "qwen3-32b-trade",
                "enabled": True,
            },
        ],
    )

    assert server_monitor_status.primary_provider_model_id() == "qwen3-32b-trade"


def test_server_monitor_ui_uses_dynamic_provider_label_and_endpoint_health() -> None:
    source = (ROOT / "web_dashboard" / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")

    assert "DeepSeek 14B / vLLM" not in source
    assert "const vllmInstanceCards = vllmRows.map(item =>" in source
    assert "item.label || item.provider_model || 'vLLM'" in source
    assert "qwen3-14b-trade" in source
    assert "deepseek-r1-14b-risk" in source
    assert "BB-FinQuant-Expert-14B" in source
    assert "runtimeEndpointSummary" in source
    assert "配置模型" in source
    assert "独立失败回退" not in source
    assert "独立调用失败，本地兜底" in source
    assert "independent_provider_failed: '独立调用失败'" in source
    assert "21840" not in source
    assert "21841" not in source
    assert "21842" not in source
    assert "platform loopback 18003" in source
    assert "configuredBase.includes('127.0.0.1')" in source
    assert "configuredBase.includes('localhost')" in source


def test_platform_server_status_contract_is_secret_free(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_service_status(name: str) -> dict[str, object]:
        return {"name": name, "active": name == "bb-dashboard.service", "status": "active"}

    monkeypatch.setattr(server_monitor_status, "_platform_service_status", fake_service_status)
    payload = server_monitor_status.collect_platform_server_status()

    assert payload["available"] is True
    assert payload["status"] == "ok"
    assert "cpu" in payload
    assert "memory" in payload
    assert "disks" in payload
    service_names = [item["name"] for item in payload["services"]]
    assert "bb-dashboard.service" in service_names
    assert "bb-paper-trading.service" in service_names
    serialized = str(payload).lower()
    assert "api_key" not in serialized
    assert "password" not in serialized
    assert "secret" not in serialized


def test_phase3_model_server_gpu_status_uses_latest_readiness_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "phase3_model_server_readiness_reports"
    report_dir.mkdir(parents=True)
    (report_dir / "latest.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "checked_at": "2026-06-27T12:00:00+00:00",
                "runtime_ready": True,
                "gpu_count": 8,
                "gpu_rows": [
                    "0, NVIDIA GeForce RTX 5090, 12000, 32607, 71, 62",
                    "1, NVIDIA GeForce RTX 5090, 11800, 32607, 68, 61",
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(type(settings), "data_dir", property(lambda _self: tmp_path))

    payload = server_monitor_status.phase3_model_server_gpu_status_from_latest_report()

    assert payload["available"] is True
    assert payload["source"] == "phase3_model_server_readiness"
    assert payload["gpu_count"] == 8
    assert payload["runtime_ready"] is True
    assert len(payload["gpus"]) == 2
    assert payload["gpus"][0]["name"] == "NVIDIA GeForce RTX 5090"
    assert payload["gpus"][0]["memory_total_mb"] == 32607.0
    assert payload["gpus"][0]["memory_used_pct"] == pytest.approx(36.8)
