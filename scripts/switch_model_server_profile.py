#!/usr/bin/env python3
"""Switch the platform's active model-server profile.

The new model server is the preferred target once repaired.  The old model
server can temporarily take over without hardcoding either host into trading
logic.  Secrets are read from the ignored account-info files and sent to the
platform process over SSH stdin, never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_server_info import DEFAULT_ACCOUNT_INFO_DIR, parse_remote_server_info  # noqa: E402
from core.remote_ssh import connect_remote_ssh, exec_remote_command  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

PROFILE_FILES = {
    "old": "大模型服务器信息.txt",
    "new": "新大模型服务器信息.txt",
}


def _load_profile(profile: str, account_dir: Path) -> dict[str, str | int]:
    filename = PROFILE_FILES[profile]
    path = account_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"model-server profile file not found: {path}")
    info = parse_remote_server_info(path.read_text(encoding="utf-8", errors="replace"), source_path=path)
    payload = info.connection_kwargs()
    payload["profile"] = profile
    return payload


def _remote_save_command() -> str:
    code = r"""
import asyncio
import json
import sys

payload = json.loads(sys.stdin.read() or "{}")

from services.model_server_config import save_model_server_settings


async def main() -> None:
    kwargs = {
        "host": payload.get("host", ""),
        "port": payload.get("port", 0),
        "username": payload.get("username", ""),
        "password": payload.get("password", ""),
        "actor": "switch_model_server_profile",
    }
    try:
        public = await save_model_server_settings(
            **kwargs,
            active_profile=str(payload.get("profile") or "custom"),
        )
    except TypeError:
        public = await save_model_server_settings(**kwargs)
    result = public.as_dict()
    result["password_configured"] = bool(result.get("password_configured"))
    result["masked_password"] = "***" if result["password_configured"] else ""
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


asyncio.run(main())
"""
    return "\n".join(
        [
            "set -euo pipefail",
            "key_line=$(grep -m1 '^BB_SECURE_SETTINGS_KEY=' /etc/bb/bb-runtime.env 2>/dev/null || true)",
            'if [ -z "$key_line" ]; then echo "BB_SECURE_SETTINGS_KEY missing" >&2; exit 3; fi',
            "cd /data/bb/app",
            (
                "sudo -u bb env "
                'BB_SECURE_SETTINGS_KEY="${key_line#BB_SECURE_SETTINGS_KEY=}" '
                "PYTHONPATH=/data/bb/app ./.venv/bin/python -c "
                + sh(code)
            ),
        ]
    )


def _remote_runtime_profile_command(profile: str) -> str:
    code = f"""
from pathlib import Path

path = Path('/etc/bb/bb-runtime.env')
profile = {profile!r}
lines = []
found = False
if path.exists():
    for raw in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if raw.startswith('MODEL_SERVER_ACTIVE_PROFILE='):
            lines.append('MODEL_SERVER_ACTIVE_PROFILE=' + profile)
            found = True
        else:
            lines.append(raw)
if not found:
    lines.append('MODEL_SERVER_ACTIVE_PROFILE=' + profile)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text('\\n'.join(lines).rstrip() + '\\n', encoding='utf-8')
try:
    import grp
    group_id = grp.getgrnam('bb').gr_gid
    import os
    os.chown(path, 0, group_id)
    path.chmod(0o640)
except Exception:
    path.chmod(0o600)
print('MODEL_SERVER_ACTIVE_PROFILE=' + profile)
"""
    return "python3 -c " + sh(textwrap.dedent(code))


def sh(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def switch_profile(profile: str, *, account_dir: Path, apply: bool) -> dict[str, object]:
    payload = _load_profile(profile, account_dir)
    redacted = {
        key: ("***" if key == "password" else value)
        for key, value in payload.items()
        if key != "password" or value
    }
    if not apply:
        return {
            "apply": False,
            "would_switch_to": redacted,
            "note": "dry-run only; pass --apply to update platform secure settings",
        }

    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        _stdin, stdout, stderr = ssh.exec_command(_remote_save_command(), timeout=120)
        _stdin.write(json.dumps(payload, ensure_ascii=False))
        _stdin.channel.shutdown_write()
        output = stdout.read().decode("utf-8", "replace")
        error = stderr.read().decode("utf-8", "replace")
        status = stdout.channel.recv_exit_status()
        if status != 0:
            raise RuntimeError(error.strip() or output.strip() or "profile switch failed")
        result = json.loads(output.strip() or "{}")
        profile_result = exec_remote_command(
            ssh,
            _remote_runtime_profile_command(profile),
            timeout=30,
            max_output_chars=2048,
        )
        if profile_result.status != 0:
            raise RuntimeError(
                profile_result.stderr.strip()
                or profile_result.stdout.strip()
                or "runtime profile update failed"
            )
        return {"apply": True, "profile": profile, "platform_settings": result}
    finally:
        ssh.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=sorted(PROFILE_FILES))
    parser.add_argument(
        "--account-dir",
        type=Path,
        default=DEFAULT_ACCOUNT_INFO_DIR,
        help="Directory containing ignored account/server info files.",
    )
    parser.add_argument("--apply", action="store_true", help="Update platform secure settings.")
    args = parser.parse_args(argv)

    result = switch_profile(args.profile, account_dir=args.account_dir, apply=args.apply)
    safe_print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
