"""Bridge model-server SSH credentials through the online platform server."""

from __future__ import annotations

import json
from pathlib import Path

from core.remote_server_info import RemoteServerInfo
from core.remote_ssh import connect_remote_ssh, exec_remote_command
from core.safe_output import safe_error_text

_REMOTE_LOCAL_AI_TOOLS_KEY_COMMAND = "\n".join(
    [
        "set -euo pipefail",
        "python3 - <<'PY'",
        "from pathlib import Path",
        "import json",
        "",
        "env_paths = [",
        "    Path('/data/BB/env/phase3.env'),",
        "    Path('/data/trade_ai/local_ai_tools.env'),",
        "]",
        "values = {}",
        "selected_env_path = ''",
        "for env_path in env_paths:",
        "    if not env_path.exists():",
        "        continue",
        "    for raw_line in env_path.read_text(encoding='utf-8').splitlines():",
        "        line = raw_line.strip()",
        "        if not line or line.startswith('#') or '=' not in line:",
        "            continue",
        "        key, value = line.split('=', 1)",
        "        values[key.strip()] = value.strip().strip(chr(34)).strip(chr(39))",
        "    if values.get('LOCAL_AI_TOOLS_API_KEY'):",
        "        selected_env_path = str(env_path)",
        "        break",
        "print(json.dumps({",
        "    'local_ai_tools_api_key': values.get('LOCAL_AI_TOOLS_API_KEY', ''),",
        "    'source': selected_env_path,",
        "}))",
        "PY",
    ]
)

_REMOTE_MODEL_INFO_COMMAND = "\n".join(
    [
        "set -euo pipefail",
        "key_line=$(grep -m1 '^BB_SECURE_SETTINGS_KEY=' /etc/bb/bb-runtime.env 2>/dev/null || true)",
        'if [ -z "$key_line" ]; then',
        "  echo 'BB_SECURE_SETTINGS_KEY missing on platform server' >&2",
        "  exit 3",
        "fi",
        "cd /data/bb/app",
        "sudo -u bb env BB_SECURE_SETTINGS_KEY=\"${key_line#BB_SECURE_SETTINGS_KEY=}\" PYTHONPATH=/data/bb/app ./.venv/bin/python - <<'PY'",
        "from __future__ import annotations",
        "",
        "import json",
        "import os",
        "import sys",
        "",
        "if not os.environ.get('BB_SECURE_SETTINGS_KEY'):",
        "    raise RuntimeError('BB_SECURE_SETTINGS_KEY not available on platform server')",
        "",
        "sys.path.insert(0, '/data/bb/app')",
        "from services.model_server_config import load_model_server_info_from_secure_settings_sync",
        "",
        "info = load_model_server_info_from_secure_settings_sync()",
        "print(json.dumps(info.connection_kwargs(), ensure_ascii=False))",
        "PY",
    ]
)


def load_model_server_info_from_platform(project_root: Path) -> RemoteServerInfo:
    """Return model-server SSH credentials by querying the platform server."""

    ssh = connect_remote_ssh(project_root, timeout=20)
    try:
        result = exec_remote_command(ssh, _REMOTE_MODEL_INFO_COMMAND, timeout=120)
        if result.status != 0:
            raise RuntimeError(
                safe_error_text(
                    result.stderr or result.stdout or "failed to load model server settings",
                    fallback="failed to load model server settings",
                )
            )
        try:
            payload = json.loads(result.stdout.strip() or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                safe_error_text(result.stdout or result.stderr or "invalid model server payload")
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("model server payload was not an object")
        return RemoteServerInfo(
            host=payload.get("host", ""),
            port=payload.get("port", 0),
            username=payload.get("username", ""),
            password=payload.get("password", ""),
            source_path=Path("<platform-secure-settings>"),
        )
    finally:
        ssh.close()


def load_local_ai_tools_api_key_from_model_server(project_root: Path) -> str:
    """Return the local AI tools API key from the model server env file."""

    info = load_model_server_info_from_platform(project_root)
    ssh = connect_remote_ssh(project_root, timeout=20, info=info)
    try:
        result = exec_remote_command(
            ssh,
            _REMOTE_LOCAL_AI_TOOLS_KEY_COMMAND,
            timeout=60,
            max_output_chars=4096,
        )
        if result.status != 0:
            raise RuntimeError(
                safe_error_text(
                    result.stderr or result.stdout or "failed to load local AI tools key",
                    fallback="failed to load local AI tools key",
                )
            )
        try:
            payload = json.loads(result.stdout.strip() or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                safe_error_text(
                    result.stdout or result.stderr or "invalid local AI tools key payload"
                )
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("local AI tools key payload was not an object")
        return str(payload.get("local_ai_tools_api_key") or "").strip()
    finally:
        ssh.close()
