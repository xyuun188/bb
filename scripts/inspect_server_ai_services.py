"""Inspect remote AI service scripts and Python environments."""

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
            "echo '--- python envs ---'",
            "find /home /data /opt \\( -path '*/envs/trade_vllm/bin/python' -o -path '*/envs/trade_ml/bin/python' \\) 2>/dev/null || true",
            "echo '--- 14b service ---'",
            "systemctl cat qwen3-14b.service --no-pager || true",
            "echo '--- scripts ---'",
            "ls -lah /data/trade_ai/scripts || true",
            "echo '--- 32b start script ---'",
            "sed -n '1,240p' /data/trade_ai/scripts/start_qwen3_32b_review.sh 2>/dev/null || true",
            "echo '--- 32b setup script ---'",
            "sed -n '1,240p' /data/trade_ai/scripts/setup_qwen3_32b_review_service.sh 2>/dev/null || true",
        ])
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=120)
        print(stdout.read().decode("utf-8", "replace"))
        print(stderr.read().decode("utf-8", "replace"))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
