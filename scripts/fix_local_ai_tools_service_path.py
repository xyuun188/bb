"""Fix local AI tools systemd service Python path on the server."""

from __future__ import annotations

import re
import textwrap
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


def run(ssh: paramiko.SSHClient, command: str) -> str:
    stdin, stdout, stderr = ssh.exec_command(command, timeout=120)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(f"{command}\n{out}\n{err}")
    return out + err


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
        found = run(
            ssh,
            "find /home /data /opt -path '*/envs/trade_ml/bin/python' 2>/dev/null | head -1",
        ).strip()
        if not found:
            found = run(ssh, "command -v python3").strip()
        env_bin = str(Path(found).parent)
        service = textwrap.dedent(
            f"""
            [Unit]
            Description=Trade Local AI Tools API
            After=network-online.target qwen3-14b.service
            Wants=network-online.target

            [Service]
            User=linux
            WorkingDirectory=/data/trade_ai/tools
            Environment=PATH={env_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
            ExecStart={found} -m uvicorn local_ai_tools_api:app --host 0.0.0.0 --port 8001
            Restart=always
            RestartSec=5
            StandardOutput=append:/data/trade_ai/logs/local_ai_tools_api.log
            StandardError=append:/data/trade_ai/logs/local_ai_tools_api.err.log

            [Install]
            WantedBy=multi-user.target
            """
        ).strip() + "\n"
        sftp = ssh.open_sftp()
        with sftp.file("/tmp/local-ai-tools.service", "w") as remote:
            remote.write(service)
        sftp.close()
        print(f"python={found}")
        print(run(
            ssh,
            "sudo mv /tmp/local-ai-tools.service /etc/systemd/system/local-ai-tools.service && "
            "sudo systemctl daemon-reload && "
            "sudo systemctl restart local-ai-tools.service && "
            "sleep 3 && systemctl is-active local-ai-tools.service && curl -sS http://127.0.0.1:8001/health",
        ))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
