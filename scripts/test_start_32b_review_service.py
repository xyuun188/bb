"""Try starting the remote 32B review service and report whether it is usable."""

from __future__ import annotations

import re
import time
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


def exec_text(ssh: paramiko.SSHClient, command: str, timeout: int = 180) -> tuple[int, str, str]:
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out, err


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
        print(exec_text(ssh, "sudo systemctl start qwen3-32b-review.service || true")[1])
        for idx in range(12):
            time.sleep(10)
            cmd = "\n".join([
                "echo '--- active ---'",
                "systemctl is-active qwen3-32b-review.service || true",
                "echo '--- models ---'",
                "curl -s --max-time 5 http://127.0.0.1:8003/v1/models || true",
                "echo",
                "echo '--- gpu ---'",
                "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true",
                "echo '--- log ---'",
                "tail -n 60 /data/trade_ai/logs/qwen3_32b_review.log 2>/dev/null || true",
            ])
            status, out, err = exec_text(ssh, cmd, timeout=120)
            print(f"=== check {idx} ===")
            print(out)
            if '"id":"qwen3-32b-risk-review"' in out or '"id": "qwen3-32b-risk-review"' in out:
                break
            if "out of memory" in out.lower() or "cuda out of memory" in out.lower():
                break
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
