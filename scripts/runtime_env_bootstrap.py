"""Load online runtime environment files for standalone maintenance scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_RUNTIME_ENV_PATH = Path("/etc/bb/bb-runtime.env")
DEFAULT_RUNTIME_USER = "bb"
RUNTIME_USER_DROPPED_ENV = "BB_RUNTIME_USER_DROPPED"
SKIP_RUNTIME_USER_DROP_ENV = "BB_SKIP_RUNTIME_USER_DROP"


def load_runtime_env_files(
    *,
    project_root: Path,
    runtime_env_path: Path = DEFAULT_RUNTIME_ENV_PATH,
) -> dict[str, str]:
    """Load project and systemd runtime env files before importing settings-heavy modules."""

    loaded: dict[str, str] = {}
    for env_path in (project_root / ".env", runtime_env_path):
        loaded.update(_load_env_file(env_path))
    return loaded


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    loaded: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _clean_env_value(raw_value.strip())
        os.environ[key] = value
        loaded[key] = value
    return loaded


def _clean_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def drop_privileges_to_runtime_user_if_needed(
    *,
    project_root: Path,
    user: str = DEFAULT_RUNTIME_USER,
    require_online_app_root: bool = True,
) -> bool:
    """Drop root-run maintenance scripts to the app user before DB access."""

    if not _is_posix():
        return False
    if os.environ.get(SKIP_RUNTIME_USER_DROP_ENV) or os.environ.get(RUNTIME_USER_DROPPED_ENV):
        return False
    if _effective_uid() != 0:
        return False
    if require_online_app_root and project_root.as_posix() != "/data/bb/app":
        return False
    account = _lookup_user(user)
    if account is None:
        return False
    _drop_to_user(account, user=user, project_root=project_root)
    os.environ[RUNTIME_USER_DROPPED_ENV] = user
    return True


def _is_posix() -> bool:
    return os.name == "posix"


def _effective_uid() -> int | None:
    getter = getattr(os, "geteuid", None)
    if getter is None:
        return None
    try:
        return int(getter())
    except Exception:
        return None


def _lookup_user(user: str) -> Any | None:
    try:
        import pwd

        return pwd.getpwnam(user)
    except Exception:
        return None


def _drop_to_user(account: Any, *, user: str, project_root: Path) -> None:
    gid = int(account.pw_gid)
    uid = int(account.pw_uid)
    home = str(getattr(account, "pw_dir", "") or f"/home/{user}")
    if hasattr(os, "initgroups"):
        os.initgroups(user, gid)
    os.setgid(gid)
    os.setuid(uid)
    os.environ["HOME"] = home
    os.environ["USER"] = user
    os.environ["LOGNAME"] = user
    os.chdir(project_root)
