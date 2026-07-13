#!/usr/bin/env python3
"""Sync the local working tree to the online BB server and restart the service.

The script intentionally uploads source files only. Runtime secrets, local data,
logs, virtualenvs, caches, and Git metadata stay on their current machine.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import fnmatch
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

REMOTE_MANAGED_SOURCE_ROOTS = (
    "ai_brain",
    "backtest",
    "config",
    "core",
    "data_feed",
    "db",
    "executor",
    "models",
    "risk_manager",
    "scripts",
    "services",
    "web_dashboard",
    "workers",
)
REMOTE_MANAGED_SOURCE_SUFFIXES = {".py"}

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
SKIP_PATH_PREFIXES = (
    "docs/superpowers/plans/",
)
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
    rows = []
    for slot in FIXED_AI_MODEL_SLOTS:
        name = str(slot["name"])
        if name == "decision_maker":
            api_base = "http://127.0.0.1:18000/v1"
            model = "qwen3-32b-trade"
        else:
            api_base = "http://127.0.0.1:18003/v1"
            model = "BB-FinQuant-Expert-14B"
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


def _runtime_env_update_script(
    *,
    remote_app_dir: str,
    local_ai_tools_key_file: str = "",
    backup_runtime_env: bool = False,
    emit_summary: bool = False,
) -> str:
    local_ai_tools_key_path = local_ai_tools_key_file if local_ai_tools_key_file else ""
    online_ai_models = _online_tunnel_ai_models_json()
    return f"""from pathlib import Path
import json
import os
import secrets
import time
from urllib.parse import urlparse

runtime_path = Path({REMOTE_RUNTIME_ENV_PATH!r})
app_env_path = Path({remote_app_dir!r}) / '.env'
local_ai_tools_key_path = Path({local_ai_tools_key_path!r}) if {bool(local_ai_tools_key_path)!r} else None
online_ai_models = {online_ai_models!r}
backup_runtime_env = {bool(backup_runtime_env)!r}
emit_summary = {bool(emit_summary)!r}
app_env_ai_route_keys = {{
    'AI_MODELS',
    'AI_API_BASE',
    'AI_MODEL',
    'LOCAL_AI_TOOLS_ENABLED',
    'LOCAL_AI_TOOLS_API_BASE',
    'HIGH_RISK_REVIEW_ENABLED',
    'HIGH_RISK_REVIEW_API_BASE',
    'HIGH_RISK_REVIEW_MODEL',
}}
app_env_ai_route_prefixes = ('MODEL_SERVER_',)

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


def scrub_app_env_ai_routes(path, keys, prefixes):
    if not path.exists():
        return {{
            'exists': False,
            'backup': '',
            'removed_keys': [],
        }}
    original_text = path.read_text(encoding='utf-8')
    removed = []
    kept_lines = []
    for raw_line in original_text.splitlines():
        stripped = raw_line.strip()
        if stripped and not stripped.startswith('#') and '=' in stripped:
            key = stripped.split('=', 1)[0].strip()
            normalized_key = key.upper()
            if normalized_key in keys or any(
                normalized_key.startswith(prefix) for prefix in prefixes
            ):
                removed.append(normalized_key)
                continue
        kept_lines.append(raw_line)
    removed_unique = sorted(set(removed))
    if not removed_unique:
        return {{
            'exists': True,
            'backup': '',
            'removed_keys': [],
        }}
    backup_path = path.with_name(path.name + '.ai-route-cleanup.bak.' + time.strftime('%Y%m%d%H%M%S'))
    backup_path.write_text(original_text, encoding='utf-8')
    os.chmod(backup_path, 0o600)
    path.write_text(chr(10).join(kept_lines).rstrip() + chr(10), encoding='utf-8')
    return {{
        'exists': True,
        'backup': str(backup_path),
        'removed_keys': removed_unique,
    }}


rows = json.loads(online_ai_models)
if any(row.get('model') == 'qwen3-14b-expert-pool' for row in rows):
    raise RuntimeError('refusing to write stale qwen3-14b-expert-pool AI_MODELS')
if not any(row.get('model') == 'BB-FinQuant-Expert-14B' for row in rows):
    raise RuntimeError('refusing to write AI_MODELS without BB-FinQuant-Expert-14B')

current_runtime_text = runtime_path.read_text(encoding='utf-8') if runtime_path.exists() else ''
values = parse_env(runtime_path)
for runtime_key in tuple(values):
    if runtime_key.upper().startswith('MODEL_SERVER_'):
        values.pop(runtime_key, None)
local_ai_tools_api_key = read_secret_file(local_ai_tools_key_path)
app_env_values = parse_env(app_env_path)

def first_non_empty(*items):
    for item in items:
        text = str(item or '').strip()
        if text:
            return text
    return ''


def is_loopback_api_base(value):
    try:
        host = (urlparse(str(value or '')).hostname or '').strip().lower()
    except Exception:
        return False
    return host in ('127.0.0.1', 'localhost', '::1')


def current_external_decision_route(raw_ai_models):
    try:
        configured_rows = json.loads(str(raw_ai_models or '[]'))
    except Exception:
        return {{}}
    if not isinstance(configured_rows, list):
        return {{}}
    for configured_row in configured_rows:
        if not isinstance(configured_row, dict):
            continue
        if str(configured_row.get('name') or '').strip() != 'decision_maker':
            continue
        api_base = str(configured_row.get('api_base') or '').strip().rstrip('/')
        model = str(configured_row.get('model') or '').strip()
        if api_base and model and not is_loopback_api_base(api_base):
            return {{
                'api_base': api_base,
                'api_key': str(configured_row.get('api_key') or '').strip(),
                'model': model,
                'route_mode': str(configured_row.get('route_mode') or 'online_slow_brain').strip(),
            }}
    return {{}}


current_decision_route = current_external_decision_route(values.get('AI_MODELS'))
decision_api_base = first_non_empty(
    values.get('ONLINE_DECISION_MAKER_API_BASE'),
    app_env_values.get('ONLINE_DECISION_MAKER_API_BASE'),
    values.get('QWEN32B_ONLINE_API_BASE'),
    app_env_values.get('QWEN32B_ONLINE_API_BASE'),
    values.get('AI_DECISION_MAKER_API_BASE'),
    app_env_values.get('AI_DECISION_MAKER_API_BASE'),
    current_decision_route.get('api_base'),
)
decision_api_key = first_non_empty(
    values.get('ONLINE_DECISION_MAKER_API_KEY'),
    app_env_values.get('ONLINE_DECISION_MAKER_API_KEY'),
    values.get('QWEN32B_ONLINE_API_KEY'),
    app_env_values.get('QWEN32B_ONLINE_API_KEY'),
    values.get('AI_DECISION_MAKER_API_KEY'),
    app_env_values.get('AI_DECISION_MAKER_API_KEY'),
    current_decision_route.get('api_key'),
)
decision_model = first_non_empty(
    values.get('ONLINE_DECISION_MAKER_MODEL'),
    app_env_values.get('ONLINE_DECISION_MAKER_MODEL'),
    values.get('QWEN32B_ONLINE_MODEL'),
    app_env_values.get('QWEN32B_ONLINE_MODEL'),
    values.get('AI_DECISION_MAKER_MODEL'),
    app_env_values.get('AI_DECISION_MAKER_MODEL'),
    current_decision_route.get('model'),
)
if decision_api_base:
    for row in rows:
        if row.get('name') == 'decision_maker':
            row['api_base'] = decision_api_base.rstrip('/')
            if decision_api_key:
                row['api_key'] = decision_api_key
            if decision_model:
                row['model'] = decision_model
            row['route_mode'] = current_decision_route.get('route_mode') or 'online_slow_brain'
            break
    online_ai_models = json.dumps(rows, ensure_ascii=False, separators=(',', ':'))

if local_ai_tools_api_key:
    values['LOCAL_AI_TOOLS_API_KEY'] = local_ai_tools_api_key
if not values.get('BB_SECURE_SETTINGS_KEY'):
    values['BB_SECURE_SETTINGS_KEY'] = app_env_values.get('BB_SECURE_SETTINGS_KEY', '')
if not values.get('BB_SECURE_SETTINGS_KEY'):
    values['BB_SECURE_SETTINGS_KEY'] = secrets.token_hex(32)
database_url = str(values.get('DATABASE_URL') or app_env_values.get('DATABASE_URL') or '').strip()
if (
    not database_url
    or database_url == 'postgresql+asyncpg:///bb_trading'
    or database_url.startswith('postgresql+asyncpg:///')
):
    database_url = 'postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql'
values['DATABASE_URL'] = database_url
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
backup_path = ''
if backup_runtime_env and runtime_path.exists():
    backup_path = str(runtime_path.with_name(runtime_path.name + '.bak.' + time.strftime('%Y%m%d%H%M%S')))
    Path(backup_path).write_text(current_runtime_text, encoding='utf-8')
    os.chmod(backup_path, 0o600)
runtime_path.write_text(''.join(f'{{key}}={{value}}\\n' for key, value in values.items()), encoding='utf-8')
try:
    import grp
    group_id = grp.getgrnam('bb').gr_gid
    os.chown(runtime_path, 0, group_id)
    os.chmod(runtime_path, 0o640)
except Exception:
    os.chmod(runtime_path, 0o600)
app_env_cleanup = scrub_app_env_ai_routes(
    app_env_path,
    app_env_ai_route_keys,
    app_env_ai_route_prefixes,
)
if emit_summary:
    print(json.dumps({{
        'updated': True,
        'backup': backup_path,
        'app_env_ai_route_cleanup': app_env_cleanup,
        'ai_models': [(row.get('name'), row.get('api_base'), row.get('model')) for row in rows],
        'old_name_remaining': 'qwen3-14b-expert-pool' in runtime_path.read_text(encoding='utf-8'),
        'starts_trading_service': False,
        'submits_orders': False,
    }}, ensure_ascii=False))
"""


def _runtime_env_only_command(
    *,
    remote_app_dir: str,
    local_ai_tools_key_file: str = "",
) -> str:
    runtime_env_script = _runtime_env_update_script(
        remote_app_dir=remote_app_dir,
        local_ai_tools_key_file=local_ai_tools_key_file,
        backup_runtime_env=True,
        emit_summary=True,
    )
    cleanup_prefix = (
        f'trap "rm -f {_remote_quote(local_ai_tools_key_file)}" EXIT; '
        if local_ai_tools_key_file
        else ""
    )
    return cleanup_prefix + f"python3 - <<'PY'\n{runtime_env_script}\nPY"


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
    runtime_env_script = _runtime_env_update_script(
        remote_app_dir=remote_app_dir,
        local_ai_tools_key_file=local_ai_tools_key_file,
    )
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
    rel_name = rel.as_posix()
    if any(rel_name.startswith(prefix) for prefix in SKIP_PATH_PREFIXES):
        return False
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


def _normalise_only_filter(value: str) -> str:
    normalised = str(value or "").strip().replace("\\", "/")
    if not normalised:
        raise ValueError("--only filters must not be empty")
    if (
        normalised == ".."
        or normalised.startswith("/")
        or normalised.startswith("../")
        or normalised.endswith("/..")
        or "/../" in normalised
    ):
        raise ValueError(f"unsafe --only filter: {value!r}")
    while normalised.startswith("./"):
        normalised = normalised[2:]
    if not normalised:
        raise ValueError("--only filters must not be empty")
    return normalised.rstrip("/") if normalised != "." else normalised


def _matches_only_filter(rel_name: str, only_filter: str) -> bool:
    if any(marker in only_filter for marker in ("*", "?", "[")):
        return fnmatch.fnmatchcase(rel_name, only_filter)
    return rel_name == only_filter or rel_name.startswith(f"{only_filter}/")


def filter_upload_files(files: list[Path], only_filters: list[str] | None) -> list[Path]:
    if not only_filters:
        return files
    filters = [_normalise_only_filter(value) for value in only_filters]
    selected = [
        path
        for path in files
        if any(
            _matches_only_filter(path.relative_to(ROOT).as_posix(), only_filter)
            for only_filter in filters
        )
    ]
    if not selected:
        raise SystemExit(f"No upload files matched --only filters: {', '.join(filters)}")
    return selected


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


def prune_remote_stale_sources(
    sftp,
    files: list[Path],
    remote_app_dir: str,
) -> list[str]:
    """Delete managed remote Python sources absent from the current local tree."""

    expected = {remote_path_for(path, remote_app_dir) for path in files}
    stale: list[str] = []
    stack = [
        str(PurePosixPath(remote_app_dir) / root)
        for root in REMOTE_MANAGED_SOURCE_ROOTS
    ]
    while stack:
        remote_dir = stack.pop()
        try:
            entries = sftp.listdir_attr(remote_dir)
        except OSError:
            continue
        for entry in entries:
            name = str(entry.filename)
            if name in {".", ".."} or name in SKIP_DIRS:
                continue
            remote_path = str(PurePosixPath(remote_dir) / name)
            if stat.S_ISDIR(entry.st_mode):
                stack.append(remote_path)
                continue
            if PurePosixPath(name).suffix.lower() not in REMOTE_MANAGED_SOURCE_SUFFIXES:
                continue
            if remote_path not in expected:
                stale.append(remote_path)

    for remote_path in sorted(stale):
        sftp.remove(remote_path)
        relative = PurePosixPath(remote_path).relative_to(PurePosixPath(remote_app_dir))
        safe_print(f"removed stale {relative.as_posix()}")
    return sorted(stale)


def upload_runtime_secret(sftp, *, value: str, remote_path: str) -> None:
    """Upload one short runtime secret to a temporary 0600 file."""
    if "\n" in value or "\r" in value:
        raise ValueError("runtime secret must be a single line")
    with sftp.file(remote_path, "w") as remote:
        remote.write(value)
    sftp.chmod(remote_path, 0o600)


def _install_requirements_command(remote_app_dir: str) -> str:
    return (
        f"cd {_remote_quote(remote_app_dir)} && "
        "PYBIN=python3; "
        "if [ -x .venv/bin/python ]; then PYBIN=.venv/bin/python; "
        "elif [ -x venv/bin/python ]; then PYBIN=venv/bin/python; fi; "
        "$PYBIN -m pip install --disable-pip-version-check -r requirements.txt"
    )


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
        "--require-model-tunnels",
        action="store_true",
        help="Fail the sync if loopback model tunnels do not become reachable.",
    )
    parser.add_argument(
        "--runtime-env-only",
        action="store_true",
        help=(
            "Only update /etc/bb/bb-runtime.env from the Phase 3 tunnel contract; "
            "do not upload files or restart any service."
        ),
    )
    parser.add_argument("--skip-restart", action="store_true")
    parser.add_argument("--skip-secret-file-purge", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PATH_OR_PREFIX",
        help=(
            "Limit uploads to a relative file, directory prefix, or glob. "
            "Repeat for multiple paths. Useful with --skip-restart for staged online validation."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = filter_upload_files(
        iter_upload_files(include_tests=args.include_tests),
        list(args.only or []),
    )
    safe_print(f"Prepared {len(files)} files for upload to {args.remote_app_dir}.")
    local_ai_tools_api_key = ""
    if args.dry_run:
        ssh = connect_remote_ssh(ROOT, timeout=20)
        ssh.close()
        upload_files(None, files, args.remote_app_dir, dry_run=True)
        return

    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        if args.runtime_env_only:
            safe_print("Updating runtime env only; no file upload or service restart will run.")
            safe_print(
                run_remote_text(
                    ssh,
                    _runtime_env_only_command(remote_app_dir=args.remote_app_dir),
                    timeout=60,
                    check=True,
                )
            )
            return
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
            removed = (
                []
                if args.only
                else prune_remote_stale_sources(sftp, files, args.remote_app_dir)
            )
            if remote_secret_path:
                upload_runtime_secret(
                    sftp,
                    value=local_ai_tools_api_key,
                    remote_path=remote_secret_path,
                )
        finally:
            sftp.close()
        safe_print(f"Uploaded {len(uploaded)} changed files.")
        safe_print(f"Removed {len(removed)} stale source files.")
        if any(path.endswith("/requirements.txt") for path in uploaded):
            safe_print("Installing updated Python requirements on online server.")
            run_remote_text(
                ssh,
                _install_requirements_command(args.remote_app_dir),
                timeout=300,
                check=True,
            )
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
            model_tunnel_probe = "python3 -c " + _remote_quote(
                "import socket, time\n"
                "for port in (18000, 18001, 18002, 18003):\n"
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
            model_tunnel_failure_action = "exit 8" if args.require_model_tunnels else "true"
            model_tunnel_restart = (
                "set +e; "
                f"systemctl restart {_remote_quote(REMOTE_MODEL_TUNNEL_SERVICE_NAME)}; "
                "model_tunnel_restart_rc=$?; "
                f"{model_tunnel_probe}; "
                "model_tunnel_probe_rc=$?; "
                'if [ "$model_tunnel_restart_rc" -eq 0 ] && '
                '[ "$model_tunnel_probe_rc" -eq 0 ]; then '
                "echo model-tunnels-ok; "
                "else "
                "echo model-tunnels-degraded; "
                f"systemctl status {_remote_quote(REMOTE_MODEL_TUNNEL_SERVICE_NAME)} "
                "--no-pager -l | sed -n '1,60p' || true; "
                f"{model_tunnel_failure_action}; "
                "fi; "
                "set -e; "
            )
            model_tunnel_active_check = (
                f"systemctl is-active {_remote_quote(REMOTE_MODEL_TUNNEL_SERVICE_NAME)} && "
                if args.require_model_tunnels
                else f"(systemctl is-active {_remote_quote(REMOTE_MODEL_TUNNEL_SERVICE_NAME)} || true) && "
            )
            command = (
                model_tunnel_restart
                + (
                f"systemctl restart {_remote_quote(args.service)} && "
                f"systemctl restart {_remote_quote(args.dashboard_service)} && "
                f"{model_tunnel_active_check}"
                f"systemctl is-active {_remote_quote(args.service)} && "
                f"systemctl is-active {_remote_quote(args.dashboard_service)} && "
                "for i in $(seq 1 30); do "
                "code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:8002/ || true); "
                'case "$code" in 200|302|401) echo dashboard-ok:$code; exit 0;; esac; '
                "sleep 2; "
                "done; echo dashboard-timeout; exit 7"
                )
            )
            safe_print(run_remote_text(ssh, command, timeout=120, check=True))
            return
        command = (
            f"systemctl restart {_remote_quote(args.service)} && "
            f"systemctl is-active {_remote_quote(args.service)} && "
            "for i in $(seq 1 30); do "
            "code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:8002/ || true); "
            'case "$code" in 200|302|401) echo dashboard-ok:$code; exit 0;; esac; '
            "sleep 2; "
            "done; echo dashboard-timeout; exit 7"
        )
        safe_print(run_remote_text(ssh, command, timeout=120, check=True))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
