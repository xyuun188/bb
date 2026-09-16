"""Safely refresh the online Qwen3.8-27B target runtime without touching trading."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402

REMOTE_SCRIPT = "/data/BB/scripts/start_target_single_model.sh"
REMOTE_RUNTIME = "/data/BB/scripts/target_transformers_api.py"
SERVICE = "bb-phase3-llm-target.service"


def _normalize_script(text: str) -> str:
    """Keep the bounded target contract even when phase3.env is stale."""

    patterns = {
        "BB_TARGET_MAX_NEW_TOKENS": 96,
        "BB_TARGET_GENERATION_TIMEOUT_SECONDS": 18,
        "BB_TARGET_QUEUE_WAIT_SECONDS": 3,
    }
    updated = text
    for name, default in patterns.items():
        pattern = rf"export {name}=\$\{{{name}:-[^}}]+}}"
        replacement = f"export {name}=${{{name}:-{default}}}"
        updated, count = re.subn(pattern, replacement, updated)
        if count == 0:
            updated += f"\n{replacement}\n"
    return updated


def main() -> None:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    stamp = time.strftime("%Y%m%d%H%M%S")
    backup = f"{REMOTE_SCRIPT}.bak.codex.{stamp}"
    try:
        before = run_remote_text(
            ssh,
            (
                f"set -eu; cp -p {REMOTE_SCRIPT} {backup}; "
                f"printf '%s\n' ---env---; "
                "grep -E 'BB_TARGET_(MAX_NEW_TOKENS|GENERATION_TIMEOUT_SECONDS|QUEUE_WAIT_SECONDS)' "
                "/data/BB/env/phase3.env 2>/dev/null || true; "
                f"printf '%s\n' ---service-before---; systemctl is-active {SERVICE} || true"
            ),
            timeout=45,
            check=True,
        )
        sftp = ssh.open_sftp()
        try:
            with sftp.file(REMOTE_SCRIPT, "r") as remote_file:
                remote_start = remote_file.read()
            if isinstance(remote_start, bytes):
                remote_start = remote_start.decode("utf-8")
            runtime = (ROOT / "scripts" / "target_transformers_api.py").read_text(
                encoding="utf-8"
            )
            updated_start = _normalize_script(str(remote_start))
            for path, content, mode in (
                (REMOTE_SCRIPT, updated_start, 0o755),
                (REMOTE_RUNTIME, runtime, 0o755),
            ):
                temporary = f"{path}.tmp.codex.{time.time_ns()}"
                with sftp.file(temporary, "w") as remote_file:
                    remote_file.write(content)
                sftp.chmod(temporary, mode)
                run_remote_text(
                    ssh,
                    f"mv -f {temporary} {path}",
                    timeout=30,
                    check=True,
                )
        finally:
            sftp.close()
        after = run_remote_text(
            ssh,
            (
                f"set -eu; sudo -n systemctl restart {SERVICE}; "
                "ready=0; for i in $(seq 1 180); do "
                f"state=$(sudo -n systemctl is-active {SERVICE} || true); "
                "if [ \"$state\" = active ] && "
                "curl -fsS --max-time 2 http://127.0.0.1:8000/health/ready >/dev/null 2>&1; "
                "then ready=1; break; fi; sleep 2; done; "
                f"printf '%s\n' ---service-after---; sudo -n systemctl is-active {SERVICE} || true; "
                "printf '%s\n' ---ready---; "
                "curl -fsS --max-time 10 http://127.0.0.1:8000/health/ready || true; "
                "if [ \"$ready\" != 1 ]; then "
                f"printf '%s\n' ---journal---; journalctl -u {SERVICE} -n 80 --no-pager || true; "
                "fi; "
                f"printf '%s\n' ---script-after---; grep -E 'BB_TARGET_(MAX_NEW_TOKENS|GENERATION_TIMEOUT_SECONDS|QUEUE_WAIT_SECONDS)' {REMOTE_SCRIPT} || true"
            ),
            timeout=420,
            check=False,
        )
        print(json.dumps({"backup": backup, "before": before[-1200:], "after": after[-6000:]}, ensure_ascii=False))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
