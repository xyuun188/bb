"""Restart remote local AI tools service and print health."""

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
            "sudo systemctl restart local-ai-tools.service",
            "sleep 3",
            "systemctl is-active local-ai-tools.service",
            "curl -s http://127.0.0.1:8001/health",
            "echo",
            "grep -n 'review_backend\\|RISK_REVIEW_BASE\\|local_review_backend' /data/trade_ai/tools/local_ai_tools_api.py | head -20",
        ])
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=90)
        print(stdout.read().decode("utf-8", "replace"))
        print(stderr.read().decode("utf-8", "replace"))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
