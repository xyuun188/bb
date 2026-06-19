from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_SERVER_SCRIPTS = [
    "scripts/check_local_ai_tools_server.py",
    "scripts/restart_local_ai_tools_server.py",
    "scripts/check_server_model_status.py",
    "scripts/inspect_server_ai_services.py",
    "scripts/inspect_deepseek_deploy_status.py",
    "scripts/fix_local_ai_tools_service_path.py",
    "scripts/deploy_local_ai_tools_service.py",
    "scripts/deploy_qwen3_32b_main_service.py",
    "scripts/start_qwen3_32b_main_service.py",
    "scripts/deploy_dual_14b_llm_services.py",
    "scripts/install_sentiment_transformer_models.py",
    "scripts/start_dual_14b_llm_tunnel.py",
]


def test_model_server_maintenance_scripts_use_model_server_settings() -> None:
    for rel_path in MODEL_SERVER_SCRIPTS:
        source = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "load_model_server_info_from_platform" in source, rel_path
        assert "info=info" in source, rel_path


def test_sync_to_online_server_syncs_local_ai_tools_key_without_logging_secret() -> None:
    source = (ROOT / "scripts" / "sync_to_online_server.py").read_text(encoding="utf-8")

    assert "load_local_ai_tools_api_key_from_model_server" in source
    assert "Prepared local AI tools API key sync payload." in source
    assert "upload_runtime_secret" in source
    assert "LOCAL_AI_TOOLS_API_KEY" in source
    assert "trap " in source and "rm -f" in source
    assert "safe_print(local_ai_tools_api_key" not in source
    assert "{local_ai_tools_api_key}" not in source


def test_sync_to_online_server_installs_updated_requirements() -> None:
    source = (ROOT / "scripts" / "sync_to_online_server.py").read_text(encoding="utf-8")

    assert "_install_requirements_command" in source
    assert "pip install --disable-pip-version-check -r requirements.txt" in source
    assert 'path.endswith("/requirements.txt")' in source


def test_sync_to_online_server_installs_loopback_model_tunnels() -> None:
    source = (ROOT / "scripts" / "sync_to_online_server.py").read_text(encoding="utf-8")

    assert 'REMOTE_MODEL_TUNNEL_SERVICE_NAME = "bb-model-tunnels.service"' in source
    assert "scripts/start_online_model_tunnels.py" in source
    assert "systemctl restart {_remote_quote(REMOTE_MODEL_TUNNEL_SERVICE_NAME)}" in source
    assert "for port in (18000, 18001, 18002)" in source
    assert "model-tunnels-ok" in source
    assert "systemctl enable {dashboard_service} {model_tunnel_service}" in source


def test_sync_to_online_server_runtime_env_uses_tunnel_ports() -> None:
    source = (ROOT / "scripts" / "sync_to_online_server.py").read_text(encoding="utf-8")

    assert "http://127.0.0.1:18000/v1" in source
    assert "http://127.0.0.1:18001" in source
    assert "http://127.0.0.1:18002/v1" in source
    assert "values['LOCAL_AI_TOOLS_API_BASE'] = 'http://127.0.0.1:18001'" in source
    assert "values['HIGH_RISK_REVIEW_API_BASE'] = 'http://127.0.0.1:18002/v1'" in source
    assert "qwen3-14b-trade" in source
    assert "deepseek-r1-14b-risk" in source
    assert "127.0.0.1:8003" not in source
    assert ":8003/v1" not in source


def test_start_online_model_tunnels_use_approved_internal_ports() -> None:
    source = (ROOT / "scripts" / "start_online_model_tunnels.py").read_text(encoding="utf-8")

    assert "local_port=18_000" in source and "remote_port=8000" in source
    assert "local_port=18_001" in source and "remote_port=8001" in source
    assert "local_port=18_002" in source and "remote_port=8002" in source
    assert "127.0.0.1:8003" not in source
    assert "21840" not in source and "21841" not in source and "21842" not in source


def test_model_server_bridge_reads_only_local_ai_tools_key_payload() -> None:
    source = (ROOT / "core" / "model_server_bridge.py").read_text(encoding="utf-8")

    assert "load_local_ai_tools_api_key_from_model_server" in source
    assert "'/data/trade_ai/local_ai_tools.env'" in source
    assert "values.get('LOCAL_AI_TOOLS_API_KEY', '')" in source
    assert "safe_error_text" in source
