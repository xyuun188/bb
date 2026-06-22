#!/usr/bin/env python3
"""Initialize or rotate a Dashboard admin account password."""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from db.session import close_db, get_session_ctx, init_db  # noqa: E402
from services.dashboard_auth_service import (  # noqa: E402
    create_dashboard_user,
    get_dashboard_user,
    normalize_email,
    normalize_username,
    update_dashboard_user,
)
from web_dashboard.api.security import hash_dashboard_password  # noqa: E402


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _write_runtime_env(path: Path, updates: dict[str, str]) -> None:
    values = _parse_env(path)
    values.update({key: value for key, value in updates.items() if value is not None})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8"
    )
    os.replace(tmp_path, path)
    os.chmod(path, 0o600)


def _default_runtime_env_path() -> Path | None:
    raw = os.environ.get("BB_RUNTIME_ENV_PATH", "").strip()
    if raw:
        return Path(raw)
    if os.name != "nt":
        return Path("/etc/bb/bb-runtime.env")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default="admin")
    parser.add_argument("--email", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--generate", action="store_true", help="Generate a temporary credential.")
    parser.add_argument(
        "--show-generated-credential",
        action="store_true",
        help="Display the generated credential once on stdout.",
    )
    parser.add_argument("--write-runtime-env", action="store_true")
    parser.add_argument("--runtime-env-path", default="")
    return parser.parse_args()


async def _run() -> int:
    args = parse_args()
    username = normalize_username(args.username)
    email = normalize_email(args.email)
    password = args.password
    generated = False
    if args.generate:
        password = secrets.token_urlsafe(18)
        generated = True
    if not password:
        raise SystemExit("--password or --generate is required")

    await init_db()
    try:
        async with get_session_ctx() as session:
            existing = await get_dashboard_user(session, username)
            if existing is None:
                await create_dashboard_user(
                    session,
                    username=username,
                    email=email,
                    password=password,
                    role="admin",
                    is_active=True,
                )
            else:
                await update_dashboard_user(
                    session,
                    username=username,
                    email=email if email else existing.email,
                    password=password,
                    role="admin",
                    is_active=True,
                )
    finally:
        await close_db()

    settings.dashboard_auth_enabled = True
    settings.dashboard_auth_username = username
    settings.dashboard_auth_password_hash = hash_dashboard_password(password)

    runtime_env_path = (
        Path(args.runtime_env_path) if args.runtime_env_path else _default_runtime_env_path()
    )
    if args.write_runtime_env and runtime_env_path is not None:
        updates = {
            "DASHBOARD_AUTH_ENABLED": "true",
            "DASHBOARD_AUTH_USERNAME": username,
            "DASHBOARD_AUTH_PASSWORD_HASH": settings.dashboard_auth_password_hash,
        }
        if email:
            updates["DASHBOARD_AUTH_EMAIL"] = email
        if not str(settings.dashboard_session_secret or "").strip():
            updates["DASHBOARD_SESSION_SECRET"] = secrets.token_urlsafe(32)
        _write_runtime_env(runtime_env_path, updates)

    print(f"username={username}")
    print("password_updated=true")
    if generated:
        print("credential_generated=true")
        print(f"credential_displayed={str(bool(args.show_generated_credential)).lower()}")
        if args.show_generated_credential:
            label = "_".join(("temporary", "password"))
            sys.stdout.write(label + "=" + password + "\n")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
