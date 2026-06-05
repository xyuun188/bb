"""Check server model download and service status."""

from __future__ import annotations

import re
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parent.parent


def parse_server_info() -> dict[str, str | int]:
    text = (ROOT / "服务器资料.txt").read_text(encoding="utf-8")
    return {
        "host": re.search(r"公网IP：([0-9.]+)", text).group(1),
        "port": int(re.search(r"端口:\s*(\d+)", text).group(1)),
        "username": re.search(r"账号:\s*(\S+)", text).group(1),
        "password": re.search(r"密码:\s*(\S+)", text).group(1),
    }


def main() -> None:
    info = parse_server_info()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        str(info["host"]),
        port=int(info["port"]),
        username=str(info["username"]),
        password=str(info["password"]),
        timeout=15,
    )
    try:
        cmd = "\n".join([
            "echo '--- services ---'",
            "systemctl is-active qwen3-14b.service || true",
            "systemctl is-active qwen3-32b-review.service || true",
            "systemctl is-active local-ai-tools.service || true",
            "echo '--- 32b files ---'",
            "du -sh /data/trade_models/Qwen/Qwen3-32B-AWQ 2>/dev/null || true",
            "find /data/trade_models/Qwen/Qwen3-32B-AWQ -maxdepth 1 -type f -printf '%f %s\\n' 2>/dev/null | sort || true",
            "echo '--- download processes ---'",
            "pgrep -af 'deploy_trade_models_phase2|huggingface|modelscope|snapshot_download|Qwen3-32B' || true",
            "echo '--- phase2 log ---'",
            "tail -n 120 /data/trade_ai/logs/deploy_trade_models_phase2.log 2>/dev/null || true",
            "echo '--- disk/gpu ---'",
            "df -h /data || true",
            "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true",
        ])
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=180)
        print(stdout.read().decode("utf-8", "replace"))
        print(stderr.read().decode("utf-8", "replace"))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
