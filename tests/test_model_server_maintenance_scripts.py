from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

MODEL_SERVER_SCRIPTS = [
    "scripts/check_local_ai_tools_server.py",
    "scripts/restart_local_ai_tools_server.py",
    "scripts/check_server_model_status.py",
    "scripts/inspect_server_ai_services.py",
    "scripts/inspect_deepseek_deploy_status.py",
    "scripts/deploy_local_ai_tools_service.py",
    "scripts/deploy_dual_14b_llm_services.py",
    "scripts/install_sentiment_transformer_models.py",
    "scripts/start_dual_14b_llm_tunnel.py",
]


def test_model_server_maintenance_scripts_use_model_server_settings() -> None:
    for rel_path in MODEL_SERVER_SCRIPTS:
        source = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "load_model_server_info_from_platform" in source, rel_path
        assert "info=info" in source, rel_path


def test_sync_to_online_server_does_not_restore_legacy_local_ai_tools_key() -> None:
    source = (ROOT / "scripts" / "sync_to_online_server.py").read_text(encoding="utf-8")

    assert "load_local_ai_tools_api_key_from_model_server" not in source
    assert "upload_runtime_secret" in source
    assert "sync_legacy_local_ai_tools_key" not in source
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
    assert "for port in (18000, 18001, 18002, 18003)" in source
    assert "model-tunnels-ok" in source
    assert "model-tunnels-degraded" in source
    assert "--require-model-tunnels" in source
    assert "systemctl enable {dashboard_service} {model_tunnel_service}" in source
    assert "REMOTE_MODEL_READINESS_SERVICE_NAME" in source
    assert "systemctl start {_remote_quote(REMOTE_MODEL_READINESS_SERVICE_NAME)}" in source


def test_sync_to_online_server_requires_okx_network_route() -> None:
    from scripts.sync_to_online_server import _okx_network_probe_command

    command = _okx_network_probe_command()

    assert "--noproxy '*'" in command
    assert "https://www.okx.com/api/v5/public/time" in command
    assert "okx-network-unavailable" in command
    assert "exit 9" in command


def test_sync_to_online_server_runtime_env_uses_tunnel_ports() -> None:
    source = (ROOT / "scripts" / "sync_to_online_server.py").read_text(encoding="utf-8")

    assert "http://127.0.0.1:18000/v1" in source
    assert "http://127.0.0.1:18001" in source
    assert "http://127.0.0.1:18002/v1" in source
    assert "http://127.0.0.1:18003/v1" in source
    assert "BB-FinQuant-Expert-14B" in source
    assert "values['LOCAL_AI_TOOLS_ENABLED'] = 'true'" in source
    assert "values['LOCAL_AI_TOOLS_API_BASE'] = 'http://127.0.0.1:18001'" in source
    assert "LOCAL_AI_TOOLS_ROUND_TRIP_COST_PCT" not in source
    assert "LOCAL_AI_TOOLS_TAIL_LOSS_THRESHOLD_PCT" not in source
    assert "values['HIGH_RISK_REVIEW_API_BASE'] = 'http://127.0.0.1:18002/v1'" in source
    assert "qwen3-14b-trade" in source
    assert "deepseek-r1-14b-risk" in source


def test_sync_to_online_server_runtime_env_scrubs_stale_app_env_ai_routes(
    monkeypatch,
    tmp_path,
) -> None:
    from scripts import sync_to_online_server as sync

    runtime_env = tmp_path / "bb-runtime.env"
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    monkeypatch.setattr(sync, "REMOTE_RUNTIME_ENV_PATH", str(runtime_env))
    runtime_env.write_text(
        "DATABASE_URL=postgresql+asyncpg://runtime\n"
        "BB_SECURE_SETTINGS_KEY=runtime-key\n",
        encoding="utf-8",
    )
    app_env = app_dir / ".env"
    app_env.write_text(
        "AI_API_BASE=http://stale-decision-route.example.invalid:31840/v1\n"
        "AI_MODEL=qwen3-32b-trade\n"
        "AI_MODELS=[{\"model\":\"qwen3-14b-trade\",\"api_base\":"
        "\"http://stale-model-route.example.invalid:21840/v1\"}]\n"
        "LOCAL_AI_TOOLS_API_BASE=http://old-local-ai.example\n"
        "HIGH_RISK_REVIEW_MODEL=old-risk-model\n"
        "DATABASE_URL=postgresql+asyncpg://app\n"
        "BB_SECURE_SETTINGS_KEY=app-key\n"
        "PROJECT_ONLY=yes\n",
        encoding="utf-8",
    )

    script = sync._runtime_env_update_script(
        remote_app_dir=str(app_dir),
        backup_runtime_env=False,
        emit_summary=False,
    )
    exec(script, {})  # noqa: S102 - the generated maintenance script is the test target.

    cleaned = app_env.read_text(encoding="utf-8")
    assert "AI_API_BASE=" not in cleaned
    assert "AI_MODEL=" not in cleaned
    assert "AI_MODELS=" not in cleaned
    assert "LOCAL_AI_TOOLS_API_BASE=" not in cleaned
    assert "HIGH_RISK_REVIEW_MODEL=" not in cleaned
    assert "DATABASE_URL=postgresql+asyncpg://app" in cleaned
    assert "BB_SECURE_SETTINGS_KEY=app-key" in cleaned
    assert "PROJECT_ONLY=yes" in cleaned

    backups = list(app_dir.glob(".env.ai-route-cleanup.bak.*"))
    assert len(backups) == 1
    runtime_text = runtime_env.read_text(encoding="utf-8")
    assert "AI_MODELS=" in runtime_text
    assert "http://127.0.0.1:18000/v1" in runtime_text
    assert "http://127.0.0.1:18003/v1" in runtime_text
    assert "BB-FinQuant-Expert-14B" in runtime_text


def test_sync_to_online_server_runtime_env_preserves_online_decision_maker(
    monkeypatch,
    tmp_path,
) -> None:
    from scripts import sync_to_online_server as sync

    runtime_env = tmp_path / "bb-runtime.env"
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    monkeypatch.setattr(sync, "REMOTE_RUNTIME_ENV_PATH", str(runtime_env))
    runtime_env.write_text(
        "DATABASE_URL=postgresql+asyncpg://runtime\n"
        "BB_SECURE_SETTINGS_KEY=runtime-key\n"
        "ONLINE_DECISION_MAKER_API_BASE=https://online-llm.example/v1\n"
        "ONLINE_DECISION_MAKER_MODEL=deepseek-reasoner\n"
        "ONLINE_DECISION_MAKER_API_KEY=secret-online-key\n",
        encoding="utf-8",
    )
    (app_dir / ".env").write_text("PROJECT_ONLY=yes\n", encoding="utf-8")

    script = sync._runtime_env_update_script(
        remote_app_dir=str(app_dir),
        backup_runtime_env=False,
        emit_summary=False,
    )
    exec(script, {})  # noqa: S102 - the generated maintenance script is the test target.

    runtime_text = runtime_env.read_text(encoding="utf-8")
    assert "https://online-llm.example/v1" in runtime_text
    assert "deepseek-reasoner" in runtime_text
    assert "route_mode\":\"online_slow_brain" in runtime_text
    assert "http://127.0.0.1:18003/v1" in runtime_text
    assert "BB-FinQuant-Expert-14B" in runtime_text


def test_sync_to_online_server_runtime_env_preserves_existing_external_ai_models_decision(
    monkeypatch,
    tmp_path,
) -> None:
    from scripts import sync_to_online_server as sync

    runtime_env = tmp_path / "bb-runtime.env"
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    monkeypatch.setattr(sync, "REMOTE_RUNTIME_ENV_PATH", str(runtime_env))
    runtime_env.write_text(
        "DATABASE_URL=postgresql+asyncpg://runtime\n"
        "BB_SECURE_SETTINGS_KEY=runtime-key\n"
        "AI_MODELS=["
        '{"name":"decision_maker","api_base":"https://api.deepseek.com/v1",'
        '"api_key":"unit-test-key","model":"deepseek-v4-pro",'
        '"route_mode":"online_slow_brain","enabled":true}'
        "]\n",
        encoding="utf-8",
    )
    (app_dir / ".env").write_text("PROJECT_ONLY=yes\n", encoding="utf-8")

    script = sync._runtime_env_update_script(
        remote_app_dir=str(app_dir),
        backup_runtime_env=False,
        emit_summary=False,
    )
    exec(script, {})  # noqa: S102 - the generated maintenance script is the test target.

    runtime_text = runtime_env.read_text(encoding="utf-8")
    assert "https://api.deepseek.com/v1" in runtime_text
    assert "deepseek-v4-pro" in runtime_text
    assert "unit-test-key" in runtime_text
    assert '"route_mode":"online_slow_brain"' in runtime_text
    assert '"model":"qwen3-32b-trade"' not in runtime_text


def test_sync_to_online_server_ignores_removed_old_profile_route(
    monkeypatch,
    tmp_path,
) -> None:
    from scripts import sync_to_online_server as sync

    runtime_env = tmp_path / "bb-runtime.env"
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    monkeypatch.setattr(sync, "REMOTE_RUNTIME_ENV_PATH", str(runtime_env))
    runtime_env.write_text(
        "DATABASE_URL=postgresql+asyncpg://runtime\n"
        "BB_SECURE_SETTINGS_KEY=runtime-key\n"
        "MODEL_SERVER_ACTIVE_PROFILE=old\n",
        encoding="utf-8",
    )
    (app_dir / ".env").write_text("PROJECT_ONLY=yes\n", encoding="utf-8")

    script = sync._runtime_env_update_script(
        remote_app_dir=str(app_dir),
        backup_runtime_env=False,
        emit_summary=False,
    )
    exec(script, {})  # noqa: S102 - the generated maintenance script is the test target.

    runtime_text = runtime_env.read_text(encoding="utf-8")
    assert "MODEL_SERVER_ACTIVE_PROFILE=old" not in runtime_text
    assert '"name":"decision_maker"' in runtime_text
    assert '"route_mode":"old_model_server_fast_fallback"' not in runtime_text


def test_sync_to_online_server_prunes_only_stale_managed_python_sources() -> None:
    from scripts import sync_to_online_server as sync

    remote_app_dir = "/srv/bb/app"
    current = ROOT / "services" / "model_server_config.py"

    class FakeSftp:
        def __init__(self) -> None:
            self.removed: list[str] = []

        def listdir_attr(self, remote_dir: str) -> list[SimpleNamespace]:
            if remote_dir == f"{remote_app_dir}/services":
                return [
                    SimpleNamespace(filename="model_server_config.py", st_mode=stat.S_IFREG),
                    SimpleNamespace(filename="legacy_fixed_gate.py", st_mode=stat.S_IFREG),
                    SimpleNamespace(filename="runtime_state.json", st_mode=stat.S_IFREG),
                    SimpleNamespace(filename="__pycache__", st_mode=stat.S_IFDIR),
                ]
            return []

        def remove(self, remote_path: str) -> None:
            self.removed.append(remote_path)

    sftp = FakeSftp()
    removed = sync.prune_remote_stale_sources(sftp, [current], remote_app_dir)

    assert removed == [f"{remote_app_dir}/services/legacy_fixed_gate.py"]
    assert sftp.removed == removed


def test_sync_to_online_server_runtime_env_only_does_not_restart_services() -> None:
    source = (ROOT / "scripts" / "sync_to_online_server.py").read_text(encoding="utf-8")

    assert "--runtime-env-only" in source
    assert "_runtime_env_only_command" in source
    assert "Updating runtime env only; no file upload or service restart will run." in source
    assert "'starts_trading_service': False" in source
    assert "'submits_orders': False" in source
    assert "app_env_ai_route_cleanup" in source

    start = source.index("def _runtime_env_only_command")
    end = source.index("def _install_split_service_command")
    env_only_source = source[start:end]
    assert "systemctl" not in env_only_source
    assert "bb-paper-trading.service" not in env_only_source
    assert "backup_runtime_env=True" in env_only_source


def test_sync_to_online_server_only_filter_limits_upload_scope() -> None:
    from scripts import sync_to_online_server as sync

    files = [
        ROOT / "scripts" / "sync_to_online_server.py",
        ROOT / "services" / "profit_first_trade_plan.py",
        ROOT / "web_dashboard" / "api" / "system_audit.py",
    ]

    selected = sync.filter_upload_files(
        files,
        ["services/profit_first_trade_plan.py", "web_dashboard/api"],
    )

    assert [path.relative_to(ROOT).as_posix() for path in selected] == [
        "services/profit_first_trade_plan.py",
        "web_dashboard/api/system_audit.py",
    ]


def test_sync_to_online_server_only_filter_rejects_unsafe_paths() -> None:
    from scripts import sync_to_online_server as sync

    files = [ROOT / "scripts" / "sync_to_online_server.py"]

    for value in ("", "../secret.txt", "/etc/passwd", "scripts/../secret.txt"):
        try:
            sync.filter_upload_files(files, [value])
        except ValueError:
            continue
        raise AssertionError(f"unsafe --only value was accepted: {value!r}")


def test_start_online_model_tunnels_use_approved_internal_ports() -> None:
    source = (ROOT / "scripts" / "start_online_model_tunnels.py").read_text(encoding="utf-8")

    assert "local_port=18_000" in source and "remote_port=8000" in source
    assert "local_port=18_001" in source and "remote_port=8101" in source
    assert "local_port=18_002" in source and "remote_port=8002" in source
    assert "local_port=18_003" in source and "remote_port=8003" in source
    assert "phase3-quant-api" in source
    assert "21840" not in source and "21841" not in source and "21842" not in source


def test_start_online_model_tunnels_swallow_short_client_disconnects() -> None:
    from scripts.start_online_model_tunnels import ForwardHandler

    class ResetSocket:
        def recv(self, _size: int) -> bytes:
            raise ConnectionResetError("peer closed early")

        def sendall(self, _data: bytes) -> None:
            raise BrokenPipeError("peer closed early")

    socket_obj = ResetSocket()

    assert ForwardHandler._recv_or_empty(socket_obj) == b""
    assert ForwardHandler._sendall_or_closed(socket_obj, b"hello") is False


def test_start_online_model_tunnels_isolate_every_endpoint_transport(monkeypatch) -> None:
    from scripts import start_online_model_tunnels as tunnels

    class FakeTransport:
        def __init__(self) -> None:
            self.keepalive: int | None = None

        def is_active(self) -> bool:
            return True

        def set_keepalive(self, seconds: int) -> None:
            self.keepalive = seconds

    class FakeSSHClient:
        def __init__(self) -> None:
            self.transport = FakeTransport()
            self.closed = False

        def get_transport(self) -> FakeTransport:
            return self.transport

        def close(self) -> None:
            self.closed = True

    clients: list[FakeSSHClient] = []

    def connect(*_args, **_kwargs) -> FakeSSHClient:
        client = FakeSSHClient()
        clients.append(client)
        return client

    monkeypatch.setattr(tunnels, "connect_remote_ssh", connect)
    specs = tunnels.build_default_tunnels()

    opened_clients, transports = tunnels.open_dedicated_transports(specs, object())

    assert opened_clients == clients
    assert len(clients) == len(specs) == 4
    assert len({id(transport) for transport in transports}) == len(specs)
    assert all(
        transport.keepalive == tunnels.TRANSPORT_KEEPALIVE_SECONDS
        for transport in transports
    )
    assert transports[1] is not transports[0]
    assert transports[1] is not transports[2]
    assert transports[1] is not transports[3]


def test_model_server_bridge_cannot_read_legacy_remote_api_key() -> None:
    source = (ROOT / "core" / "model_server_bridge.py").read_text(encoding="utf-8")

    assert "load_model_server_info_from_platform" in source
    assert "load_local_ai_tools_api_key_from_model_server" not in source
    assert "_REMOTE_LOCAL_AI_TOOLS_KEY_COMMAND" not in source
    assert "/data/trade_ai/local_ai_tools.env" not in source
    assert "safe_error_text" in source


def test_model_server_status_scripts_use_dual_14b_contract() -> None:
    check_source = (ROOT / "scripts" / "check_server_model_status.py").read_text(encoding="utf-8")
    contract_source = (ROOT / "core" / "phase3_model_contract.py").read_text(encoding="utf-8")
    inspect_source = (ROOT / "scripts" / "inspect_server_ai_services.py").read_text(
        encoding="utf-8"
    )

    for source in (check_source, inspect_source):
        assert "bb-phase3-llm-decision.service" in source or "qwen3-14b-trade.service" in source
        assert "bb-phase3-llm-risk-review.service" in source or "deepseek-r1-14b-risk.service" in source
        assert "qwen3-32b-main.service" in source
        assert "deprecated service" in source.lower()

    assert "VLLM_SERVICES = PHASE3_MODEL_SERVER_SERVICES" in check_source
    assert '("bb-phase3-llm-decision.service", PHASE3_DECISION_MODEL_ID, 8000)' in contract_source
    assert '("bb-phase3-llm-risk-review.service", PHASE3_RISK_MODEL_ID, 8002)' in contract_source
    assert '("bb-phase3-llm-expert.service", PHASE3_EXPERT_MODEL_ID, 8003)' in contract_source
    assert "/data/trade_models/" in contract_source
    assert "PHASE3_QUANT_API_PORT = 8101" in check_source
    assert "http://127.0.0.1:{PHASE3_QUANT_API_PORT}/health" in check_source
    assert "http://127.0.0.1:{port}/v1/models" in check_source
    assert "Qwen3-14B-AWQ" in contract_source
    assert "qwen3_32b_main.log" not in check_source
    assert "start_qwen3_32b_main.sh" not in inspect_source
