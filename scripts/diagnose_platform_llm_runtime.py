"""Read-only inspection of the platform-side LLM route and service churn."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402


def _command() -> str:
    app_probe = r'''
cd /data/bb/app
export DATABASE_URL='postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql'
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from scripts.runtime_env_bootstrap import load_runtime_env_files

# Match the systemd/dashboard environment before importing Settings.  A bare
# DATABASE_URL probe otherwise reports every fixed model as missing even when
# the running service has the target loopback routes injected by runtime env.
load_runtime_env_files(project_root=Path('/data/bb/app'))
from config.settings import settings

rows = []
for item in settings.get_fixed_ai_models(include_empty=True):
    rows.append({
        "name": item.get("name"),
        "role": item.get("role"),
        "enabled": item.get("enabled"),
        "configured": item.get("configured"),
        "configuration_type": item.get("configuration_type"),
        "model": item.get("model"),
        "api_base": item.get("api_base"),
    })
print(json.dumps({
    "trading_mode": str(settings.trading_mode),
    "ai_llm_concurrency": settings.ai_llm_concurrency,
    "ai_llm_max_calls_per_analysis": settings.ai_llm_max_calls_per_analysis,
    "ai_batch_experts_enabled": settings.ai_batch_experts_enabled,
    "ai_batch_expert_max_completion_tokens": settings.ai_batch_expert_max_completion_tokens,
    "ai_batch_expert_timeout_seconds": settings.ai_batch_expert_timeout_seconds,
    "ai_target_qwen_max_completion_tokens": settings.ai_target_qwen_max_completion_tokens,
    "ai_target_qwen_timeout_seconds": settings.ai_target_qwen_timeout_seconds,
    "ai_target_qwen_queue_wait_seconds": settings.ai_target_qwen_queue_wait_seconds,
    "ai_decision_maker_timeout_seconds": settings.ai_decision_maker_timeout_seconds,
    "ai_models": rows,
}, ensure_ascii=False))
PY
'''.strip()
    service_probe = r'''
echo ---runtime-env---
for file in /etc/bb/bb-runtime.env /data/bb/env/phase3.env /data/bb/app/.env; do
  echo FILE:$file
  grep -E 'AI_(BATCH|LLM|EXPERT|DECISION)|TARGET|TRADING_MODE|MODEL' "$file" 2>/dev/null | sed -E 's/(KEY|TOKEN|SECRET|PASSWORD)=.*/\1=<REDACTED>/' || true
done
echo ---routes---
for spec in "18000 /health/ready" "18001 /health/live"; do
  set -- $spec
  port=$1
  path=$2
  echo PORT:$port PATH:$path
  code=$(curl -sS -o /tmp/bb-llm-health-$port.json -w '%{http_code}' --max-time 4 "http://127.0.0.1:$port$path" || true)
  echo HTTP:$code
  cat /tmp/bb-llm-health-$port.json 2>/dev/null || true
  echo
done
echo ---services---
for service in bb-dashboard.service bb-paper-trading.service; do
  printf '%s ' "$service"
  systemctl is-active "$service" || true
  systemctl show "$service" -p NRestarts -p MainPID -p ActiveState 2>/dev/null || true
done
echo ---paper-journal---
journalctl -u bb-paper-trading.service -n 120 --no-pager 2>/dev/null | grep -Ei 'llm|qwen|timeout|batch|expert|decision|503|502' | tail -n 80 || true
'''.strip()
    return "set -u\n" + app_probe + "\n" + service_probe


def main() -> int:
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        print(run_remote_text(ssh, _command(), timeout=180, max_output_chars=20_000, check=False))
    finally:
        ssh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
