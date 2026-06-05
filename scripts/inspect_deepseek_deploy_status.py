"""Inspect DeepSeek 32B deployment/download status on the server."""

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
        ])
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=180)
        print(stdout.read().decode("utf-8", "replace"))
        print(stderr.read().decode("utf-8", "replace"))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
