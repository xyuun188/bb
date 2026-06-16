"""Inspect DeepSeek 32B deployment/download status on the server."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402


def main() -> None:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=15, info=info)
    try:
        cmd = "\n".join(
            [
                "echo '--- services ---'",
                "systemctl is-active deepseek-32b-main.service || true",
                "systemctl status deepseek-32b-main.service --no-pager -l | head -80 || true",
                "echo '--- deepseek dir ---'",
                "du -sh /data/trade_models/DeepSeek/DeepSeek-R1-Distill-Qwen-32B-AWQ 2>/dev/null || true",
                "find /data/trade_models/DeepSeek/DeepSeek-R1-Distill-Qwen-32B-AWQ -maxdepth 1 -type f -printf '%f %s\\n' 2>/dev/null | sort || true",
                "echo '--- download processes ---'",
                "pgrep -af 'download_deepseek|DeepSeek-R1-Distill-Qwen-32B|snapshot_download|huggingface' || true",
                "echo '--- logs ---'",
                "tail -n 120 /data/trade_ai/logs/deepseek_download.log 2>/dev/null || true",
                "tail -n 120 /data/trade_ai/logs/deepseek_32b_main.log 2>/dev/null || true",
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
