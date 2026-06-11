from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from core.server_monitor_probe import (
    display_provider_model_name,
    render_python_here_doc,
    render_server_monitor_probe,
)

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
    assert "http://127.0.0.1:8000/v1/models" in script
    assert "http://127.0.0.1:8001/models/status" in script


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

    def fake_http_json(url: str, timeout: int = 3) -> dict[str, object]:
        if url.endswith("/v1/models"):
            return {
                "ok": True,
                "status_code": 200,
                "latency_ms": 12.4,
                "truncated": False,
                "data": {"data": [{"id": "other-model"}]},
            }
        if url.endswith("/models/status"):
            return {
                "ok": False,
                "status_code": 503,
                "latency_ms": 25.0,
                "truncated": False,
                "error": "service warming up",
                "data": None,
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
    tools = cast(dict[str, object], runtime["local_ai_tools"])

    assert vllm["endpoint_available"] is True
    assert vllm["model_available"] is False
    assert vllm["available"] is False
    assert vllm["status"] == "model_not_available"
    assert vllm["label"] == "Qwen3 32B"
    assert vllm["provider_model"] == "qwen3-32b-trade"
    assert cast(dict[str, object], vllm["health"])["status_code"] == 200
    assert cast(dict[str, object], vllm["health"])["latency_ms"] == 12.4
    assert tools["available"] is True
    assert cast(dict[str, object], tools["status_health"])["status_code"] == 503
    assert cast(dict[str, object], tools["health"])["ok"] is True


def test_server_monitor_ui_uses_dynamic_provider_label_and_endpoint_health() -> None:
    source = (ROOT / "web_dashboard" / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")

    assert "DeepSeek 14B / vLLM" not in source
    assert "vllm.label || vllm.provider_model || 'vLLM'" in source
    assert "runtimeEndpointSummary" in source
    assert "配置模型" in source
