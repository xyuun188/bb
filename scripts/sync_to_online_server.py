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
import os
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
from core.model_candidate_manifest import ModelCandidateManifest  # noqa: E402
from core.model_topology import (  # noqa: E402
    DEFAULT_MODEL_TOPOLOGY_PROFILE,
    TARGET_SINGLE_MODEL_PROFILE,
    ModelTopology,
    model_tunnel_routes,
    normalize_topology_profile,
    target_topology_ready,
    topology_for_profile,
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
REMOTE_MODEL_READINESS_SERVICE_NAME = "bb-phase3-model-server-readiness.service"
REMOTE_DASHBOARD_PROXY_SERVICE_NAME = "nginx.service"
REMOTE_DASHBOARD_PROXY_SITE = "/etc/nginx/sites-available/bb-dashboard"
REMOTE_DASHBOARD_PROXY_LINK = "/etc/nginx/sites-enabled/bb-dashboard"
MODEL_TUNNEL_DEPLOY_READY_TIMEOUT_SECONDS = 75
REMOTE_RUNTIME_ENV_PATH = "/etc/bb/bb-runtime.env"
# The online platform may use a different service account than the model host.
# Resolve the existing application directory owner unless explicitly overridden.
REMOTE_OWNER = "auto"

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
SKIP_PATH_PREFIXES = ("docs/superpowers/plans/",)
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


def _resolve_remote_owner(ssh: object, remote_app_dir: str, configured_owner: str) -> str:
    """Resolve a valid remote service owner without assuming model-host users."""

    owner = str(configured_owner or "").strip()
    if owner and owner.lower() != "auto":
        return owner
    detected = run_remote_text(
        ssh,
        f"stat -c %U:%G {_remote_quote(remote_app_dir)}",
        timeout=30,
        check=True,
    ).strip()
    if not detected or ":" not in detected or any(char.isspace() for char in detected):
        raise RuntimeError(f"could not resolve a valid owner for {remote_app_dir!r}")
    return detected


def _render_dashboard_service(remote_app_dir: str, owner: str) -> str:
    user, _sep, group = owner.partition(":")
    group = group or user
    return f"""[Unit]
Description=BB Dashboard
After=network-online.target postgresql.service redis-server.service redis.service bb-model-tunnels.service
Wants=network-online.target bb-model-tunnels.service

[Service]
Type=simple
User={user}
Group={group}
WorkingDirectory={remote_app_dir}
EnvironmentFile=-{remote_app_dir}/.env
EnvironmentFile={REMOTE_RUNTIME_ENV_PATH}
Environment=MALLOC_ARENA_MAX=2
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=NUMEXPR_NUM_THREADS=1
ExecStart=/bin/bash -lc 'cd {remote_app_dir} && if [ -x .venv/bin/python ]; then exec .venv/bin/python scripts/run_dashboard.py; elif [ -x venv/bin/python ]; then exec venv/bin/python scripts/run_dashboard.py; else exec python3 scripts/run_dashboard.py; fi'
Restart=always
RestartSec=5
MemoryAccounting=true
MemoryHigh=3G
MemoryMax=5G
TasksMax=256

[Install]
WantedBy=multi-user.target
"""


def _render_dashboard_proxy_config() -> str:
    return """server {
    listen 80;
    listen [::]:80;
    server_name _;

    client_max_body_size 20m;
    proxy_connect_timeout 5s;
    proxy_read_timeout 120s;
    proxy_send_timeout 120s;

    location / {
        proxy_pass http://127.0.0.1:8002;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
    }
}
"""


def _install_dashboard_proxy_command() -> str:
    proxy_config = _render_dashboard_proxy_config()
    return (
        "set -e; "
        "if ! command -v nginx >/dev/null 2>&1; then "
        "export DEBIAN_FRONTEND=noninteractive; "
        "apt-get update -qq; apt-get install -y -qq nginx; "
        "fi; "
        "install -d -m 0755 /etc/nginx/sites-available /etc/nginx/sites-enabled; "
        f"cat > /tmp/bb-dashboard.nginx <<'NGINX'\n{proxy_config}\nNGINX\n"
        f"install -m 0644 /tmp/bb-dashboard.nginx {REMOTE_DASHBOARD_PROXY_SITE}; "
        f"ln -sfn {REMOTE_DASHBOARD_PROXY_SITE} {REMOTE_DASHBOARD_PROXY_LINK}; "
        "rm -f /etc/nginx/sites-enabled/default; "
        "nginx -t; "
        f"systemctl enable --now {_remote_quote(REMOTE_DASHBOARD_PROXY_SERVICE_NAME)} "
        ">/dev/null; "
        f"systemctl reload {_remote_quote(REMOTE_DASHBOARD_PROXY_SERVICE_NAME)}; "
        f"systemctl is-active {_remote_quote(REMOTE_DASHBOARD_PROXY_SERVICE_NAME)}"
    )


def _render_model_tunnel_service(
    remote_app_dir: str,
    owner: str,
    *,
    profile: str | None = None,
) -> str:
    user, _sep, group = owner.partition(":")
    group = group or user
    selected_profile = normalize_topology_profile(
        profile if profile is not None else os.environ.get("BB_MODEL_TOPOLOGY_PROFILE")
    )
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
Environment=MALLOC_ARENA_MAX=2
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=NUMEXPR_NUM_THREADS=1
ExecStart=/bin/bash -lc 'cd {remote_app_dir} && if [ -x .venv/bin/python ]; then exec .venv/bin/python scripts/start_online_model_tunnels.py --profile {selected_profile}; elif [ -x venv/bin/python ]; then exec venv/bin/python scripts/start_online_model_tunnels.py --profile {selected_profile}; else exec python3 scripts/start_online_model_tunnels.py --profile {selected_profile}; fi'
Restart=always
RestartSec=3
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
"""


def _model_tunnel_endpoint_pairs(profile: str | None = None) -> tuple[tuple[int, str], ...]:
    """Return health endpoints required by the selected tunnel profile."""

    selected = profile if profile is not None else os.environ.get("BB_MODEL_TOPOLOGY_PROFILE")
    return tuple((route.local_port, route.health_path) for route in model_tunnel_routes(selected))


def _target_topology_from_environment() -> ModelTopology:
    """Load only a model-host-generated verified manifest.

    Identity environment variables are intentionally ignored.  They are useful
    for diagnostics but are not evidence that weights, tokenizer and runtime
    measurements were checked on the model server.
    """

    manifest_path = str(os.environ.get("BB_TARGET_MODEL_MANIFEST") or "").strip()
    if not manifest_path:
        return topology_for_profile(TARGET_SINGLE_MODEL_PROFILE)
    manifest = ModelCandidateManifest.load(manifest_path)
    topology = manifest.to_topology(stage="candidate_validated")
    if not target_topology_ready(topology):
        raise RuntimeError("verified candidate manifest did not produce a ready target topology")
    return topology


def _online_tunnel_ai_models_json(
    profile: str | None = None,
    *,
    topology: ModelTopology | None = None,
) -> str:
    """Render fixed slots from the verified single-model topology."""

    selected_profile = normalize_topology_profile(
        profile if profile is not None else os.environ.get("BB_MODEL_TOPOLOGY_PROFILE")
    )
    resolved = topology or _target_topology_from_environment()
    if not target_topology_ready(resolved):
        raise RuntimeError(
            "target_single_model requires a verified candidate identity and non-live topology"
        )
    if resolved.profile != selected_profile:
        raise RuntimeError(
            f"model topology profile mismatch: requested={selected_profile} actual={resolved.profile}"
        )

    carrier = resolved.by_role("decision_and_expert_carrier")
    rows = []
    for slot in FIXED_AI_MODEL_SLOTS:
        name = str(slot["name"])
        selected_model = carrier
        rows.append(
            {
                "name": name,
                "role": slot["role"],
                "label": slot["label"],
                "weight": slot["weight"],
                "api_base": selected_model.endpoint if selected_model else "",
                "api_key": "",
                "model": selected_model.model_id if selected_model else "",
                "enabled": True,
            }
        )
    if any(not str(row["api_base"]).strip() or not str(row["model"]).strip() for row in rows):
        raise RuntimeError(f"topology {selected_profile} has incomplete AI slot routing")
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def _runtime_env_update_script(
    *,
    remote_app_dir: str,
    local_ai_tools_key_file: str = "",
    backup_runtime_env: bool = False,
    emit_summary: bool = False,
    model_topology_profile: str | None = None,
) -> str:
    local_ai_tools_key_path = local_ai_tools_key_file if local_ai_tools_key_file else ""
    # Internal callers follow the same target-single default as the CLI. Legacy
    # topology rendering must be requested explicitly for audit/rollback only.
    topology_profile = str(
        model_topology_profile
        or os.environ.get("BB_MODEL_TOPOLOGY_PROFILE")
        or DEFAULT_MODEL_TOPOLOGY_PROFILE
    ).strip().lower()
    online_ai_models = _online_tunnel_ai_models_json(topology_profile)
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
topology_profile = {topology_profile!r}
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
    'HIGH_RISK_REVIEW_API_KEY',
    'HIGH_RISK_REVIEW_MODEL',
    'HIGH_RISK_REVIEW_MODEL_REVISION',
}}
app_env_ai_route_prefixes = (
    'MODEL_SERVER_',
    'ONLINE_DECISION_MAKER_',
    'CLOUD_DECISION_MAKER_',
)

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
if not rows or any(not row.get('api_base') or not row.get('model') for row in rows):
    raise RuntimeError('refusing to write AI_MODELS with incomplete topology routes')
if topology_profile == 'target_single_model':
    model_ids = {{str(row.get('model') or '').strip() for row in rows}}
    if len(model_ids) != 1 or any(
        'deepseek' in model_id.lower()
        or '14b' in model_id.lower()
        or 'finquant-expert' in model_id.lower()
        for model_id in model_ids
    ):
        raise RuntimeError('refusing to write legacy model route under target_single_model')

current_runtime_text = runtime_path.read_text(encoding='utf-8') if runtime_path.exists() else ''
values = parse_env(runtime_path)
for runtime_key in tuple(values):
    normalized_runtime_key = runtime_key.upper()
    if (
        normalized_runtime_key.startswith('MODEL_SERVER_')
        or normalized_runtime_key.startswith('ONLINE_DECISION_MAKER_')
        or normalized_runtime_key.startswith('CLOUD_DECISION_MAKER_')
    ):
        values.pop(runtime_key, None)
local_ai_tools_api_key = read_secret_file(local_ai_tools_key_path)
app_env_values = parse_env(app_env_path)

def first_non_empty(*items):
    for item in items:
        text = str(item or '').strip()
        if text:
            return text
    return ''


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
values['BB_MODEL_TOPOLOGY_PROFILE'] = topology_profile
values['LOCAL_AI_TOOLS_ENABLED'] = 'true'
values['LOCAL_AI_TOOLS_API_BASE'] = 'http://127.0.0.1:18001'
cloud_reviewer_api_base = first_non_empty(
    values.get('ONLINE_HIGH_RISK_REVIEW_API_BASE'),
    app_env_values.get('ONLINE_HIGH_RISK_REVIEW_API_BASE'),
    values.get('CLOUD_HIGH_RISK_REVIEW_API_BASE'),
    app_env_values.get('CLOUD_HIGH_RISK_REVIEW_API_BASE'),
)
cloud_reviewer_api_key = first_non_empty(
    values.get('ONLINE_HIGH_RISK_REVIEW_API_KEY'),
    app_env_values.get('ONLINE_HIGH_RISK_REVIEW_API_KEY'),
    values.get('CLOUD_HIGH_RISK_REVIEW_API_KEY'),
    app_env_values.get('CLOUD_HIGH_RISK_REVIEW_API_KEY'),
)
cloud_reviewer_model = first_non_empty(
    values.get('ONLINE_HIGH_RISK_REVIEW_MODEL'),
    app_env_values.get('ONLINE_HIGH_RISK_REVIEW_MODEL'),
    values.get('CLOUD_HIGH_RISK_REVIEW_MODEL'),
    app_env_values.get('CLOUD_HIGH_RISK_REVIEW_MODEL'),
)
cloud_reviewer_revision = first_non_empty(
    values.get('ONLINE_HIGH_RISK_REVIEW_MODEL_REVISION'),
    app_env_values.get('ONLINE_HIGH_RISK_REVIEW_MODEL_REVISION'),
    values.get('CLOUD_HIGH_RISK_REVIEW_MODEL_REVISION'),
    app_env_values.get('CLOUD_HIGH_RISK_REVIEW_MODEL_REVISION'),
)
if cloud_reviewer_api_base and cloud_reviewer_api_key and cloud_reviewer_model:
    values['HIGH_RISK_REVIEW_ENABLED'] = 'true'
    values['HIGH_RISK_REVIEW_API_BASE'] = cloud_reviewer_api_base.rstrip('/')
    values['HIGH_RISK_REVIEW_API_KEY'] = cloud_reviewer_api_key
    values['HIGH_RISK_REVIEW_MODEL'] = cloud_reviewer_model
    values['HIGH_RISK_REVIEW_MODEL_REVISION'] = cloud_reviewer_revision
else:
    # Never resurrect the removed local DeepSeek reviewer.  Keeping the feature
    # enabled with an incomplete route makes the entry gate fail closed.
    values['HIGH_RISK_REVIEW_ENABLED'] = 'true'
    values['HIGH_RISK_REVIEW_API_BASE'] = ''
    values['HIGH_RISK_REVIEW_API_KEY'] = ''
    values['HIGH_RISK_REVIEW_MODEL'] = ''
    values['HIGH_RISK_REVIEW_MODEL_REVISION'] = ''
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
        'target_model_route_present': any(
            str(row.get('model') or '').strip() == 'qwen3.8-27b' for row in rows
        ),
        'starts_trading_service': False,
        'submits_orders': False,
    }}, ensure_ascii=False))
"""


def _runtime_env_only_command(
    *,
    remote_app_dir: str,
    local_ai_tools_key_file: str = "",
    model_topology_profile: str | None = None,
) -> str:
    runtime_env_script = _runtime_env_update_script(
        remote_app_dir=remote_app_dir,
        local_ai_tools_key_file=local_ai_tools_key_file,
        backup_runtime_env=True,
        emit_summary=True,
        model_topology_profile=model_topology_profile,
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
    model_topology_profile: str | None = None,
) -> str:
    dashboard_unit = _render_dashboard_service(remote_app_dir, owner)
    selected_profile = str(
        model_topology_profile
        or os.environ.get("BB_MODEL_TOPOLOGY_PROFILE")
        or DEFAULT_MODEL_TOPOLOGY_PROFILE
    ).strip().lower()
    model_tunnel_unit = _render_model_tunnel_service(
        remote_app_dir,
        owner,
        profile=selected_profile,
    )
    runtime_env_script = _runtime_env_update_script(
        remote_app_dir=remote_app_dir,
        local_ai_tools_key_file=local_ai_tools_key_file,
        model_topology_profile=selected_profile,
    )
    tunnel_checks = " ".join(
        f"{port}:{path}" for port, path in _model_tunnel_endpoint_pairs(selected_profile)
    )
    trading_dropin = f"""[Service]
EnvironmentFile=-{remote_app_dir}/.env
EnvironmentFile={REMOTE_RUNTIME_ENV_PATH}
Environment=MALLOC_ARENA_MAX=2
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=NUMEXPR_NUM_THREADS=1
ExecStartPre=/bin/bash -lc 'for spec in {tunnel_checks}; do port=${{spec%%:*}}; path=${{spec#*:}}; for i in $(seq 1 60); do curl -fsS --max-time 4 http://127.0.0.1:$port$path >/dev/null 2>&1 && break; sleep 1; done; curl -fsS --max-time 4 http://127.0.0.1:$port$path >/dev/null 2>&1 || exit 1; done'
StandardOutput=journal
StandardError=journal
MemoryAccounting=true
MemoryHigh=4G
MemoryMax=6G
CPUAccounting=true
CPUQuota=200%
TasksMax=256
OOMPolicy=stop
TimeoutStartSec=120
TimeoutStopSec=30
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
    """Return source files already tracked by Git.

    Deployment must not pick up untracked scratch files or half-finished local
    modules. Newly created production files are uploaded after they are added
    to the Git index, which keeps the sync set reviewable and reproducible.
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
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
    *,
    managed_roots: tuple[str, ...] = REMOTE_MANAGED_SOURCE_ROOTS,
) -> list[str]:
    """Delete managed remote Python sources absent from the current local tree."""

    expected = {remote_path_for(path, remote_app_dir) for path in files}
    stale: list[str] = []
    stack = [str(PurePosixPath(remote_app_dir) / root) for root in managed_roots]
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


def _okx_network_probe_command() -> str:
    return (
        "okx_code=$(curl --noproxy '*' -sS -o /dev/null -w '%{http_code}' "
        "--connect-timeout 5 --max-time 10 https://www.okx.com/api/v5/public/time || true); "
        'if [ "$okx_code" != "200" ]; then '
        'echo "okx-network-unavailable:$okx_code"; exit 9; fi; '
        'echo "okx-network-ok"; '
    )


def _split_services_restart_command(
    *,
    trading_service: str,
    dashboard_service: str,
    model_tunnel_restart: str,
    model_tunnel_active_check: str,
    model_readiness_refresh: str,
) -> str:
    quoted_trading = _remote_quote(trading_service)
    quoted_dashboard = _remote_quote(dashboard_service)
    maintenance_window = (
        "set -e; "
        "resume_platform_services() { "
        f"systemctl start {quoted_trading} {quoted_dashboard} >/dev/null 2>&1 || true; "
        "}; "
        "trap resume_platform_services EXIT; "
        f"systemctl stop {quoted_trading} {quoted_dashboard}; "
    )
    service_start = (
        f"systemctl start {quoted_trading} {quoted_dashboard} && "
    )
    return maintenance_window + model_tunnel_restart + (
        f"{model_readiness_refresh}"
        f"{model_tunnel_active_check}"
        f"{_okx_network_probe_command()}"
        f"{service_start}"
        f"systemctl is-active {quoted_trading} && "
        f"systemctl is-active {quoted_dashboard} && "
        "for i in $(seq 1 30); do "
        "code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:8002/ || true); "
        'case "$code" in 200|302|401) trap - EXIT; echo dashboard-ok:$code; exit 0;; esac; '
        "sleep 2; "
        "done; echo dashboard-timeout; exit 7"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-app-dir", default=REMOTE_APP_DIR)
    parser.add_argument("--service", default=REMOTE_SERVICE_NAME)
    parser.add_argument("--dashboard-service", default=REMOTE_DASHBOARD_SERVICE_NAME)
    parser.add_argument("--owner", default=REMOTE_OWNER)
    parser.add_argument(
        "--model-topology-profile",
        choices=(TARGET_SINGLE_MODEL_PROFILE,),
        default=os.environ.get("BB_MODEL_TOPOLOGY_PROFILE", DEFAULT_MODEL_TOPOLOGY_PROFILE),
        help="Select legacy audit tunnels or the verified one-local-model target profile.",
    )
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
        resolved_owner = _resolve_remote_owner(ssh, args.remote_app_dir, args.owner)
        if args.runtime_env_only:
            safe_print("Updating runtime env only; no file upload or service restart will run.")
            safe_print(
                run_remote_text(
                    ssh,
                    _runtime_env_only_command(
                        remote_app_dir=args.remote_app_dir,
                        model_topology_profile=args.model_topology_profile,
                    ),
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
            managed_roots = (
                (*REMOTE_MANAGED_SOURCE_ROOTS, "tests")
                if args.include_tests
                else REMOTE_MANAGED_SOURCE_ROOTS
            )
            removed = (
                []
                if args.only
                else prune_remote_stale_sources(
                    sftp,
                    files,
                    args.remote_app_dir,
                    managed_roots=managed_roots,
                )
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
                f"chown {_remote_quote(resolved_owner)} {quoted_paths}",
                timeout=120,
            )
        if args.skip_restart:
            safe_print("Skipped service restart.")
            return
        safe_print("Ensuring the public Dashboard proxy is installed and healthy.")
        safe_print(
            run_remote_text(
                ssh,
                _install_dashboard_proxy_command(),
                timeout=300,
                check=True,
            )
        )
        if args.split_services:
            run_remote_text(
                ssh,
                _install_split_service_command(
                    remote_app_dir=args.remote_app_dir,
                    owner=resolved_owner,
                    trading_service=args.service,
                    dashboard_service=args.dashboard_service,
                    model_tunnel_service=REMOTE_MODEL_TUNNEL_SERVICE_NAME,
                    local_ai_tools_key_file=remote_secret_path,
                    model_topology_profile=args.model_topology_profile,
                ),
                timeout=120,
                check=True,
            )
            model_tunnel_probe = "python3 -c " + _remote_quote(
                "import http.client, time\n"
                f"deadline = time.time() + {MODEL_TUNNEL_DEPLOY_READY_TIMEOUT_SECONDS}\n"
                f"endpoints = {_model_tunnel_endpoint_pairs(args.model_topology_profile)!r}\n"
                "for port, path in endpoints:\n"
                "    while True:\n"
                "        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=8)\n"
                "        try:\n"
                "            connection.request('GET', path, headers={'Connection': 'close'})\n"
                "            response = connection.getresponse()\n"
                "            response.read(1)\n"
                "            if response.status == 200:\n"
                "                break\n"
                "        except OSError:\n"
                "            pass\n"
                "        finally:\n"
                "            connection.close()\n"
                "        if time.time() >= deadline:\n"
                "            raise SystemExit(f'model endpoint {port}{path} unavailable or not HTTP 200')\n"
                "        time.sleep(1)\n"
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
            model_readiness_refresh = (
                f"if systemctl cat {_remote_quote(REMOTE_MODEL_READINESS_SERVICE_NAME)} "
                ">/dev/null 2>&1; then "
                f"systemctl start {_remote_quote(REMOTE_MODEL_READINESS_SERVICE_NAME)}; "
                "fi; "
            )
            command = _split_services_restart_command(
                trading_service=args.service,
                dashboard_service=args.dashboard_service,
                model_tunnel_restart=model_tunnel_restart,
                model_tunnel_active_check=model_tunnel_active_check,
                model_readiness_refresh=model_readiness_refresh,
            )
            safe_print(run_remote_text(ssh, command, timeout=120, check=True))
            return
        command = (
            f"systemctl restart {_remote_quote(args.service)} && "
            f"systemctl is-active {_remote_quote(args.service)} && "
            f"{_okx_network_probe_command()}"
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
