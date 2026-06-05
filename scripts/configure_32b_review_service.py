"""Configure remote 32B high-risk review service on internal port 8003."""

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
    stdin, stdout, stderr = ssh.exec_command(command, timeout=180)
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
        start_script = textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            source ~/anaconda3/etc/profile.d/conda.sh
            conda activate trade_vllm
            export CUDA_VISIBLE_DEVICES=0
            export VLLM_WORKER_MULTIPROC_METHOD=spawn
            export VLLM_USE_V1=1
            MODEL_DIR=/data/trade_models/Qwen/Qwen3-32B-AWQ
            LOG=/data/trade_ai/logs/qwen3_32b_review.log
            exec python -m vllm.entrypoints.openai.api_server \
              --host 127.0.0.1 \
              --port 8003 \
              --model "$MODEL_DIR" \
              --served-model-name qwen3-32b-risk-review \
              --trust-remote-code \
              --max-model-len 4096 \
              --gpu-memory-utilization 0.36 \
              --dtype half \
              --quantization awq \
              --enforce-eager > "$LOG" 2>&1
            """
        )
        service = textwrap.dedent(
            """\
            [Unit]
            Description=Qwen3 32B AWQ High Risk Review vLLM API
            After=network.target qwen3-14b.service

            [Service]
            Type=simple
            User=linux
            WorkingDirectory=/data/trade_ai
            ExecStart=/data/trade_ai/scripts/start_qwen3_32b_review.sh
            Restart=no
            Environment=CUDA_VISIBLE_DEVICES=0
            Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn
            LimitNOFILE=65535

            [Install]
            WantedBy=multi-user.target
            """
        )
        sftp = ssh.open_sftp()
        with sftp.file("/data/trade_ai/scripts/start_qwen3_32b_review.sh", "w") as remote:
            remote.write(start_script)
        with sftp.file("/tmp/qwen3-32b-review.service", "w") as remote:
            remote.write(service)
        sftp.close()
        print(run(
            ssh,
            "chmod +x /data/trade_ai/scripts/start_qwen3_32b_review.sh && "
            "sudo mv /tmp/qwen3-32b-review.service /etc/systemd/system/qwen3-32b-review.service && "
            "sudo systemctl daemon-reload && "
            "sudo systemctl disable qwen3-32b-review.service || true && "
            "sudo systemctl stop qwen3-32b-review.service || true && "
            "systemctl cat qwen3-32b-review.service --no-pager",
        ))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
