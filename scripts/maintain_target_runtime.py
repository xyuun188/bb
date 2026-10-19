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
from core.phase3_model_contract import (  # noqa: E402
    PHASE3_TARGET_CARRIER_QUEUE_WAIT_SECONDS,
    PHASE3_TARGET_GENERATION_TIMEOUT_SECONDS,
    PHASE3_TARGET_MAX_NEW_TOKENS,
)
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402

REMOTE_SCRIPT = "/data/BB/scripts/start_target_single_model.sh"
REMOTE_RUNTIME = "/data/BB/scripts/target_transformers_api.py"
SERVICE = "bb-phase3-llm-target.service"

# These are platform-side routing/policy settings that were historically
# copied into the model-host environment.  The target carrier does not read
# them; keeping them in phase3.env only makes diagnostics claim that two
# competing runtime contracts are active.
STALE_PLATFORM_ENV_KEYS = frozenset(
    {
        "AI_LLM_CONCURRENCY",
        "AI_LLM_MAX_CALLS_PER_ANALYSIS",
        "AI_EXPERT_TIMEOUT_SECONDS",
        "AI_DECISION_MAKER_TIMEOUT_SECONDS",
        "AI_EXPERT_MAX_COMPLETION_TOKENS",
        "AI_DECISION_MAKER_MAX_COMPLETION_TOKENS",
        "AI_BATCH_EXPERTS_ENABLED",
        "AI_BATCH_EXPERT_MAX_COMPLETION_TOKENS",
        "AI_BATCH_EXPERT_TIMEOUT_SECONDS",
        "AI_BATCH_EXPERT_CIRCUIT_BREAKER_SECONDS",
        "AI_BATCH_EXPERT_FORMAT_FAILURE_CIRCUIT_BREAKER_SECONDS",
        "AI_TARGET_QWEN_TIMEOUT_SECONDS",
        "AI_TARGET_QWEN_QUEUE_WAIT_SECONDS",
        "HIGH_RISK_REVIEW_ENABLED",
        "HIGH_RISK_REVIEW_API_BASE",
        "HIGH_RISK_REVIEW_API_KEY",
        "HIGH_RISK_REVIEW_MODEL",
        "HIGH_RISK_REVIEW_MODEL_REVISION",
        "BB_TARGET_MAX_NEW_TOKENS",
        "BB_TARGET_GENERATION_TIMEOUT_SECONDS",
        "BB_TARGET_QUEUE_WAIT_SECONDS",
        "BB_TARGET_ATTN_IMPLEMENTATION",
        "BB_TARGET_RESPONSE_CACHE_SECONDS",
    }
)


def _normalize_script(text: str) -> str:
    """Enforce the one target-carrier contract even when phase3.env is stale."""

    patterns = {
        "BB_TARGET_MAX_NEW_TOKENS": PHASE3_TARGET_MAX_NEW_TOKENS,
        "BB_TARGET_GENERATION_TIMEOUT_SECONDS": PHASE3_TARGET_GENERATION_TIMEOUT_SECONDS,
        "BB_TARGET_QUEUE_WAIT_SECONDS": PHASE3_TARGET_CARRIER_QUEUE_WAIT_SECONDS,
    }
    updated = text
    for name, default in patterns.items():
        pattern = rf"^export {name}=.*$"
        replacement = f"export {name}={default}"
        updated, count = re.subn(pattern, replacement, updated, flags=re.MULTILINE)
        if count == 0:
            updated += f"\n{replacement}\n"
    return updated


def _scrub_stale_platform_env(text: str) -> tuple[str, list[str]]:
    """Remove old platform controls while preserving model-host settings."""

    kept: list[str] = []
    removed: list[str] = []
    for line in str(text or "").splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if match and match.group(1) in STALE_PLATFORM_ENV_KEYS:
            removed.append(match.group(1))
            continue
        kept.append(line)
    normalized = "\n".join(kept).rstrip() + "\n"
    return normalized, sorted(set(removed))


def main() -> None:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    stamp = time.strftime("%Y%m%d%H%M%S")
    backup = f"{REMOTE_SCRIPT}.bak.codex.{stamp}"
    env_backup = f"/data/BB/env/phase3.env.bak.codex.{stamp}"
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
            with sftp.file("/data/BB/env/phase3.env", "r") as remote_env_file:
                remote_env = remote_env_file.read()
            if isinstance(remote_env, bytes):
                remote_env = remote_env.decode("utf-8")
            updated_env, removed_env_keys = _scrub_stale_platform_env(str(remote_env))
            if removed_env_keys:
                run_remote_text(
                    ssh,
                    f"cp -p /data/BB/env/phase3.env {env_backup}",
                    timeout=30,
                    check=True,
                )
            for path, content, mode in (
                (REMOTE_SCRIPT, updated_start, 0o755),
                (REMOTE_RUNTIME, runtime, 0o755),
                ("/data/BB/env/phase3.env", updated_env, 0o600),
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
        print(
            json.dumps(
                {
                    "backup": backup,
                    "env_backup": env_backup if removed_env_keys else None,
                    "removed_env_keys": removed_env_keys,
                    "before": before[-1200:],
                    "after": after[-6000:],
                },
                ensure_ascii=False,
            )
        )
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
