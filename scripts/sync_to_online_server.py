#!/usr/bin/env python3
"""Sync the local working tree to the online BB server and restart the service.

The script intentionally uploads source files only. Runtime secrets, local data,
logs, virtualenvs, caches, and Git metadata stay on their current machine.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import posixpath
import secrets
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import FIXED_AI_MODEL_SLOTS  # noqa: E402
from core.model_server_bridge import (  # noqa: E402
    load_local_ai_tools_api_key_from_model_server,
)
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from scripts.audit_online_secret_files import (
    _remote_script as secret_file_audit_script,
)  # noqa: E402

REMOTE_APP_DIR = "/data/bb/app"
REMOTE_SERVICE_NAME = "bb-paper-trading.service"
REMOTE_DASHBOARD_SERVICE_NAME = "bb-dashboard.service"
REMOTE_MODEL_TUNNEL_SERVICE_NAME = "bb-model-tunnels.service"
REMOTE_RUNTIME_ENV_PATH = "/etc/bb/bb-runtime.env"
REMOTE_OWNER = "bb:bb"

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "__pycache__",
    "data",
    "logs",
    ".ssh",
    ".codex-memory",
    ".claude",
    ".rtk",
    "build",
    "dist",
}
SKIP_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    "PROJECT_MEMORY.md",
}
SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".log",
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    ".zip",
    ".7z",
    ".rar",
}
SKIP_NAME_PARTS = (
    "\u670d\u52a1\u5668\u4fe1\u606f",  # server info
    "\u670d\u52a1\u5668\u8d44\u6599",  # server data
    "\u8d26\u53f7",  # account
    "\u5bc6\u7801",  # password
    "\u79d8\u94a5",  # secret key
    "\u5bc6\u94a5",  # key
)


def _remote_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _render_dashboard_service(remote_app_dir: str, owner: str) -> str:
    user, _sep, group = owner.partition(":")
    group = group or user
    return f"""[Unit]
Description=BB Dashboard
After=network-online.target postgresql.service redis-server.service redis.service
Wants=network-online.target

[Service]
Type=simple
User={user}
Group={group}
WorkingDirectory={remote_app_dir}
EnvironmentFile=-{remote_app_dir}/.env
EnvironmentFile={REMOTE_RUNTIME_ENV_PATH}
ExecStart=/bin/bash -lc 'cd {remote_app_dir} && if [ -x .venv/bin/python ]; then exec .venv/bin/python scripts/run_dashboard.py; elif [ -x venv/bin/python ]; then exec venv/bin/python scripts/run_dashboard.py; else exec python3 scripts/run_dashboard.py; fi'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""


def _render_model_tunnel_service(remote_app_dir: str, owner: str) -> str:
    user, _sep, group = owner.partition(":")
    group = group or user
    return f"""[Unit]
Description=BB Platform to Model Server Tunnels
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=simple
User={user}
Group={group}
WorkingDirectory={remote_app_dir}
EnvironmentFile=-{remote_app_dir}/.env
EnvironmentFile={REMOTE_RUNTIME_ENV_PATH}
ExecStart=/bin/bash -lc 'cd {remote_app_dir} && if [ -x .venv/bin/python ]; then exec .venv/bin/python scripts/start_online_model_tunnels.py; elif [ -x venv/bin/python ]; then exec venv/bin/python scripts/start_online_model_tunnels.py; else exec python3 scripts/start_online_model_tunnels.py; fi'
Restart=always
RestartSec=3
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
"""


def _online_tunnel_ai_models_json() -> str:
    qwen_names = {"trend_expert", "momentum_expert", "decision_maker"}
    deepseek_names = {"sentiment_expert", "position_expert", "risk_expert"}
    rows = []
    for slot in FIXED_AI_MODEL_SLOTS:
        name = str(slot["name"])
        if name in qwen_names:
            api_base = "http://127.0.0.1:18000/v1"
            model = "qwen3-14b-trade"
        elif name in deepseek_names:
            api_base = "http://127.0.0.1:18002/v1"
            model = "deepseek-r1-14b-risk"
        else:
            raise ValueError(f"No online tunnel assignment for fixed AI slot: {name}")
        rows.append(
            {
                "name": name,
                "role": slot["role"],
                "label": slot["label"],
                "weight": slot["weight"],
                "api_base": api_base,
                "api_key": "",
                "model": model,
                "enabled": True,
            }
        )
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def _install_split_service_command(
    *,
    remote_app_dir: str,
    owner: str,
    trading_service: str,
    dashboard_service: str,
    model_tunnel_service: str,
    local_ai_tools_key_file: str = "",
) -> str:
    dashboard_unit = _render_dashboard_service(remote_app_dir, owner)
    model_tunnel_unit = _render_model_tunnel_service(remote_app_dir, owner)
    local_ai_tools_key_path = local_ai_tools_key_file if local_ai_tools_key_file else ""
    online_ai_models = _online_tunnel_ai_models_json()
    runtime_env_script = f"""from pathlib import Path
import os
import secrets

runtime_path = Path({REMOTE_RUNTIME_ENV_PATH!r})
app_env_path = Path({remote_app_dir!r}) / '.env'
local_ai_tools_key_path = Path({local_ai_tools_key_path!r}) if {bool(local_ai_tools_key_path)!r} else None
online_ai_models = {online_ai_models!r}

def parse_env(path):
    values = {{}}
    if not path.exists():
        return values
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip().strip(chr(34)).strip(chr(39))
    return values


def read_secret_file(path):
    if path is None or not path.exists():
        return ''
    value = path.read_text(encoding='utf-8').strip()
    if chr(10) in value or chr(13) in value:
        raise ValueError('local AI tools key file must contain one line')
    return value


values = parse_env(runtime_path)
local_ai_tools_api_key = read_secret_file(local_ai_tools_key_path)
if local_ai_tools_api_key:
    values['LOCAL_AI_TOOLS_API_KEY'] = local_ai_tools_api_key
if not values.get('BB_SECURE_SETTINGS_KEY'):
    values['BB_SECURE_SETTINGS_KEY'] = parse_env(app_env_path).get('BB_SECURE_SETTINGS_KEY', '')
if not values.get('BB_SECURE_SETTINGS_KEY'):
    values['BB_SECURE_SETTINGS_KEY'] = secrets.token_hex(32)
values['DASHBOARD_AUTH_ENABLED'] = 'true'
values['DASHBOARD_INLINE_ENABLED'] = 'false'
values['USE_FAKEREDIS'] = 'false'
values['REDIS_URL'] = 'redis://127.0.0.1:6379/0'
values['AI_MODELS'] = online_ai_models
values['LOCAL_AI_TOOLS_ENABLED'] = 'true'
values['LOCAL_AI_TOOLS_API_BASE'] = 'http://127.0.0.1:18001'
values['HIGH_RISK_REVIEW_ENABLED'] = 'true'
values['HIGH_RISK_REVIEW_API_BASE'] = 'http://127.0.0.1:18002/v1'
values['HIGH_RISK_REVIEW_MODEL'] = 'deepseek-r1-14b-risk'
try:
    current_tools_timeout = float(values.get('LOCAL_AI_TOOLS_TIMEOUT_SECONDS') or 0)
except ValueError:
    current_tools_timeout = 0.0
if current_tools_timeout < 8.0:
    values['LOCAL_AI_TOOLS_TIMEOUT_SECONDS'] = '8.0'
try:
    current_tools_breaker = int(values.get('LOCAL_AI_TOOLS_CIRCUIT_BREAKER_FAILURES') or 0)
except ValueError:
    current_tools_breaker = 0
if current_tools_breaker < 3:
    values['LOCAL_AI_TOOLS_CIRCUIT_BREAKER_FAILURES'] = '3'
runtime_path.parent.mkdir(parents=True, exist_ok=True)
runtime_path.write_text(''.join(f'{{key}}={{value}}\\n' for key, value in values.items()), encoding='utf-8')
os.chmod(runtime_path, 0o600)
"""
    trading_dropin = f"""[Service]
EnvironmentFile=-{remote_app_dir}/.env
EnvironmentFile={REMOTE_RUNTIME_ENV_PATH}
"""
    cleanup_prefix = (
        f'trap "rm -f {_remote_quote(local_ai_tools_key_file)}" EXIT; '
        if local_ai_tools_key_file
        else ""
    )
    return (
        cleanup_prefix + "set -e; "
        "(systemctl enable --now redis-server.service >/dev/null 2>&1 || "
        " systemctl enable --now redis.service >/dev/null 2>&1 || true); "
        f"python3 - <<'PY'\n{runtime_env_script}\nPY\n"
        f"cat > /tmp/{dashboard_service} <<'UNIT'\n{dashboard_unit}\nUNIT\n"
        f"install -m 0644 /tmp/{dashboard_service} /etc/systemd/system/{dashboard_service}; "
        f"cat > /tmp/{model_tunnel_service} <<'UNIT'\n{model_tunnel_unit}\nUNIT\n"
        f"install -m 0644 /tmp/{model_tunnel_service} /etc/systemd/system/{model_tunnel_service}; "
        f"mkdir -p /etc/systemd/system/{trading_service}.d; "
        f"cat > /etc/systemd/system/{trading_service}.d/20-split-dashboard.conf <<'DROPIN'\n{trading_dropin}\nDROPIN\n"
        f"systemctl daemon-reload; systemctl enable {dashboard_service} {model_tunnel_service} >/dev/null"
    )


def should_upload(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return False
    name = path.name
    if name in SKIP_FILES:
        return False
    if name.startswith(".env."):
        return False
    if any(part in name for part in SKIP_NAME_PARTS):
        return False
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    return path.is_file()


def iter_upload_files(include_tests: bool) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    files: list[Path] = []
    for raw_name in result.stdout.split(b"\0"):
        if not raw_name:
            continue
        rel_name = raw_name.decode("utf-8", errors="replace")
        path = ROOT / rel_name
        if not should_upload(path):
            continue
        rel = path.relative_to(ROOT)
        if not include_tests and rel.parts and rel.parts[0] == "tests":
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.as_posix().lower())


def remote_path_for(local_path: Path, remote_app_dir: str) -> str:
    rel = local_path.relative_to(ROOT).as_posix()
    return str(PurePosixPath(remote_app_dir) / PurePosixPath(rel))


def needs_upload(sftp, local_path: Path, remote_path: str) -> bool:
    try:
        remote_stat = sftp.stat(remote_path)
    except OSError:
        return True
    local_stat = local_path.stat()
    if int(remote_stat.st_size) != int(local_stat.st_size):
        return True
    return abs(float(remote_stat.st_mtime) - float(local_stat.st_mtime)) > 1.0


def ensure_remote_dir(sftp, remote_dir: str) -> None:
    current = PurePosixPath(remote_dir)
    parts = current.parts
    if not parts:
        return
    path = "/" if current.is_absolute() else "."
    start = 1 if current.is_absolute() else 0
    for part in parts[start:]:
        path = posixpath.join(path, part)
        try:
            sftp.stat(path)
        except OSError:
            sftp.mkdir(path)


def upload_files(sftp, files: list[Path], remote_app_dir: str, *, dry_run: bool) -> list[str]:
    uploaded: list[str] = []
    for local_path in files:
        remote_path = remote_path_for(local_path, remote_app_dir)
        rel_name = local_path.relative_to(ROOT).as_posix()
        if dry_run:
            safe_print(f"would consider {rel_name}")
            continue
        if not needs_upload(sftp, local_path, remote_path):
            continue
        ensure_remote_dir(sftp, posixpath.dirname(remote_path))
        sftp.put(str(local_path), remote_path)
        mode = stat.S_IMODE(local_path.stat().st_mode)
        sftp.chmod(remote_path, mode)
        local_mtime = local_path.stat().st_mtime
        sftp.utime(remote_path, (local_mtime, local_mtime))
        uploaded.append(remote_path)
        safe_print(f"uploaded {rel_name}")
    return uploaded


def upload_runtime_secret(sftp, *, value: str, remote_path: str) -> None:
    """Upload one short runtime secret to a temporary 0600 file."""
    if "\n" in value or "\r" in value:
        raise ValueError("runtime secret must be a single line")
    with sftp.file(remote_path, "w") as remote:
        remote.write(value)
    sftp.chmod(remote_path, 0o600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-app-dir", default=REMOTE_APP_DIR)
    parser.add_argument("--service", default=REMOTE_SERVICE_NAME)
    parser.add_argument("--dashboard-service", default=REMOTE_DASHBOARD_SERVICE_NAME)
    parser.add_argument("--owner", default=REMOTE_OWNER)
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument(
        "--split-services",
        action="store_true",
        help="Run trading and Dashboard as separate systemd services on the online server.",
    )
    parser.add_argument(
        "--skip-local-ai-tools-key-sync",
        action="store_true",
        help="Do not copy the model server local AI tools API key into the platform runtime env.",
    )
    parser.add_argument("--skip-restart", action="store_true")
    parser.add_argument("--skip-secret-file-purge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = iter_upload_files(include_tests=args.include_tests)
    safe_print(f"Prepared {len(files)} files for upload to {args.remote_app_dir}.")
    local_ai_tools_api_key = ""
    if args.split_services and not args.skip_restart and not args.skip_local_ai_tools_key_sync:
        try:
            local_ai_tools_api_key = load_local_ai_tools_api_key_from_model_server(ROOT)
        except Exception as exc:
            safe_print(f"Local AI tools key sync skipped: {exc}")
        else:
            if local_ai_tools_api_key:
                safe_print("Prepared local AI tools API key sync payload.")
            else:
                safe_print("Local AI tools API key missing on model server; skipping key sync.")
    if args.dry_run:
        ssh = connect_remote_ssh(ROOT, timeout=20)
        ssh.close()
        upload_files(None, files, args.remote_app_dir, dry_run=True)
        return

    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        run_remote_text(ssh, f"mkdir -p {_remote_quote(args.remote_app_dir)}", timeout=30)
        if not args.skip_secret_file_purge:
            purge_script = secret_file_audit_script(
                remote_app_dir=args.remote_app_dir,
                delete=True,
            )
            safe_print(run_remote_text(ssh, f"python3 - <<'PY'\n{purge_script}\nPY", timeout=60))
        remote_secret_path = (
            f"/run/bb/local-ai-tools-key-{secrets.token_hex(12)}" if local_ai_tools_api_key else ""
        )
        if remote_secret_path:
            run_remote_text(ssh, "install -d -m 0700 /run/bb", timeout=30)
        sftp = ssh.open_sftp()
        try:
            uploaded = upload_files(sftp, files, args.remote_app_dir, dry_run=False)
            if remote_secret_path:
                upload_runtime_secret(
                    sftp,
                    value=local_ai_tools_api_key,
                    remote_path=remote_secret_path,
                )
        finally:
            sftp.close()
        safe_print(f"Uploaded {len(uploaded)} changed files.")
        if uploaded:
            quoted_paths = " ".join(_remote_quote(path) for path in uploaded)
            run_remote_text(
                ssh,
                f"chown {_remote_quote(args.owner)} {quoted_paths}",
                timeout=120,
            )
        if args.skip_restart:
            safe_print("Skipped service restart.")
            return
        if args.split_services:
            run_remote_text(
                ssh,
                _install_split_service_command(
                    remote_app_dir=args.remote_app_dir,
                    owner=args.owner,
                    trading_service=args.service,
                    dashboard_service=args.dashboard_service,
                    model_tunnel_service=REMOTE_MODEL_TUNNEL_SERVICE_NAME,
                    local_ai_tools_key_file=remote_secret_path,
                ),
                timeout=120,
                check=True,
            )
            command = (
                f"systemctl restart {_remote_quote(REMOTE_MODEL_TUNNEL_SERVICE_NAME)} && "
                "python3 -c "
                + _remote_quote(
                    "import socket, time\n"
                    "for port in (18000, 18001, 18002):\n"
                    "    deadline = time.time() + 20\n"
                    "    while True:\n"
                    "        try:\n"
                    "            socket.create_connection(('127.0.0.1', port), timeout=1).close(); break\n"
                    "        except OSError:\n"
                    "            if time.time() >= deadline:\n"
                    "                raise SystemExit(f'tunnel port {port} unavailable')\n"
                    "            time.sleep(1)\n"
                    "print('model-tunnels-ok')"
                )
                + " && "
                f"systemctl restart {_remote_quote(args.service)} && "
                f"systemctl restart {_remote_quote(args.dashboard_service)} && "
                f"systemctl is-active {_remote_quote(REMOTE_MODEL_TUNNEL_SERVICE_NAME)} && "
                f"systemctl is-active {_remote_quote(args.service)} && "
                f"systemctl is-active {_remote_quote(args.dashboard_service)} && "
                "for i in $(seq 1 30); do "
                "code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:8002/ || true); "
                'case "$code" in 200|302|401) echo dashboard-ok:$code; exit 0;; esac; '
                "sleep 2; "
                "done; echo dashboard-timeout; exit 7"
            )
            safe_print(run_remote_text(ssh, command, timeout=120, check=True))
            return
        command = (
            f"systemctl restart {_remote_quote(args.service)} && "
            f"systemctl is-active {_remote_quote(args.service)} && "
            "for i in $(seq 1 30); do "
            "code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:8002/ || true); "
            'case "$code" in 200|302|401) echo dashboard-ok:$code; exit 0;; esac; '
            "sleep 2; "
            "done; echo dashboard-timeout; exit 7"
        )
        safe_print(run_remote_text(ssh, command, timeout=120, check=True))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
