"""Check remote model download and service status."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402


def main() -> None:
    ssh = connect_remote_ssh(ROOT, timeout=15)
    try:
        cmd = "\n".join(
            [
                "echo '--- current services ---'",
                "systemctl is-active qwen3-32b-main.service || true",
                "systemctl is-active local-ai-tools.service || true",
                "echo '--- deprecated services should be inactive or missing ---'",
                "systemctl is-active qwen3-14b.service || true",
                "systemctl is-active qwen3-32b-review.service || true",
                "systemctl is-active deepseek-14b-main.service || true",
                "systemctl is-active deepseek-32b-main.service || true",
                "echo '--- vllm models endpoint ---'",
                "curl -fsS --max-time 6 http://127.0.0.1:8000/v1/models || true",
                "echo '--- 32b files ---'",
                "du -sh /data/trade_models/Qwen/Qwen3-32B-AWQ 2>/dev/null || true",
                "find /data/trade_models/Qwen/Qwen3-32B-AWQ -maxdepth 1 -type f -printf '%f %s\\n' 2>/dev/null | sort || true",
                "echo '--- download processes ---'",
                "pgrep -af 'huggingface|modelscope|snapshot_download|Qwen3-32B' || true",
                "echo '--- main qwen log ---'",
                "tail -n 120 /data/trade_ai/logs/qwen3_32b_main.log 2>/dev/null || true",
                "echo '--- disk/gpu ---'",
                "df -h /data || true",
                "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true",
            ]
        )
        safe_print(run_remote_text(ssh, cmd, timeout=180, check=False))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
