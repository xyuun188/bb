"""Deploy DeepSeek-R1-Distill-Qwen-32B-AWQ as the only main LLM service."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ"
MODEL_DIR = "/data/trade_models/DeepSeek/DeepSeek-R1-Distill-Qwen-32B-AWQ"
SERVED_MODEL = "deepseek-r1-distill-qwen-32b-trade"


def parse_server_info() -> dict[str, str | int]:
    text = (ROOT / "服务器资料.txt").read_text(encoding="utf-8")
    return {
        "host": re.search(r"公网IP：([0-9.]+)", text).group(1),
        "port": int(re.search(r"端口:\s*(\d+)", text).group(1)),
        "username": re.search(r"账号:\s*(\S+)", text).group(1),
        "password": re.search(r"密码:\s*(\S+)", text).group(1),
    }


def run(ssh: paramiko.SSHClient, command: str, timeout: int = 180) -> str:
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(f"command failed ({status}): {command}\n{out}\n{err}")
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
        timeout=20,
    )
    try:
        print(run(
            ssh,
            "sudo systemctl stop qwen3-14b.service qwen3-32b-review.service 2>/dev/null || true && "
            "sudo systemctl disable qwen3-14b.service qwen3-32b-review.service 2>/dev/null || true && "
            "sudo rm -f /etc/systemd/system/qwen3-14b.service /etc/systemd/system/qwen3-32b-review.service && "
            "sudo systemctl daemon-reload && "
            "rm -f /data/trade_ai/scripts/start_qwen3_14b.sh /data/trade_ai/scripts/start_qwen3_32b_review.sh && "
            "rm -rf /data/trade_models/Qwen/Qwen3-14B-AWQ && "
            "mkdir -p /data/trade_models/DeepSeek /data/trade_ai/scripts /data/trade_ai/logs && "
            "echo stopped-qwen-services",
        ))

        download_script = textwrap.dedent(
            f"""\
            set -euo pipefail
            source ~/anaconda3/etc/profile.d/conda.sh
            conda activate trade_vllm
            python - <<'PY'
            from pathlib import Path
            target = Path("{MODEL_DIR}")
            complete = target.exists() and any(target.glob("*.safetensors"))
            if complete:
                print("deepseek model already present:", target)
            else:
                target.mkdir(parents=True, exist_ok=True)
                try:
                    from huggingface_hub import snapshot_download
                except Exception:
                    import subprocess, sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
                    from huggingface_hub import snapshot_download
                import os
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                snapshot_download(
                    repo_id="{MODEL_REPO}",
                    local_dir=str(target),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    endpoint=os.environ.get("HF_ENDPOINT"),
                )
                print("deepseek model downloaded:", target)
            PY
            """
        )
        sftp = ssh.open_sftp()
        with sftp.file("/data/trade_ai/scripts/download_deepseek_32b_awq.sh", "w") as remote:
            remote.write(download_script)
        sftp.close()
        print(run(
            ssh,
            "chmod +x /data/trade_ai/scripts/download_deepseek_32b_awq.sh && "
            "/data/trade_ai/scripts/download_deepseek_32b_awq.sh",
            timeout=7200,
        ))

        start_script = textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            source ~/anaconda3/etc/profile.d/conda.sh
            conda activate trade_vllm
            export CUDA_VISIBLE_DEVICES=0
            export VLLM_WORKER_MULTIPROC_METHOD=spawn
            export VLLM_USE_V1=1
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            LOG=/data/trade_ai/logs/deepseek_32b_main.log
            exec python -m vllm.entrypoints.openai.api_server \\
              --host 0.0.0.0 \\
              --port 8000 \\
              --model "{MODEL_DIR}" \\
              --served-model-name {SERVED_MODEL} \\
              --trust-remote-code \\
              --max-model-len 8192 \\
              --gpu-memory-utilization 0.90 \\
              --dtype half \\
              --quantization awq \\
              --enforce-eager > "$LOG" 2>&1
            """
        )
        service = textwrap.dedent(
            """\
            [Unit]
            Description=DeepSeek R1 Distill Qwen 32B AWQ vLLM OpenAI API
            After=network.target

            [Service]
            Type=simple
            User=linux
            WorkingDirectory=/data/trade_ai
            ExecStart=/data/trade_ai/scripts/start_deepseek_32b_main.sh
            Restart=always
            RestartSec=10
            Environment=CUDA_VISIBLE_DEVICES=0
            Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn
            Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            LimitNOFILE=65535

            [Install]
            WantedBy=multi-user.target
            """
        )
        sftp = ssh.open_sftp()
        with sftp.file("/data/trade_ai/scripts/start_deepseek_32b_main.sh", "w") as remote:
            remote.write(start_script)
        with sftp.file("/tmp/deepseek-32b-main.service", "w") as remote:
            remote.write(service)
        sftp.close()
        print(run(
            ssh,
            "chmod +x /data/trade_ai/scripts/start_deepseek_32b_main.sh && "
            "sudo mv /tmp/deepseek-32b-main.service /etc/systemd/system/deepseek-32b-main.service && "
            "sudo systemctl daemon-reload && "
            "sudo systemctl enable deepseek-32b-main.service && "
            "sudo systemctl restart deepseek-32b-main.service && "
            "sleep 5 && systemctl is-active deepseek-32b-main.service || true && "
            "tail -n 80 /data/trade_ai/logs/deepseek_32b_main.log 2>/dev/null || true",
            timeout=300,
        ))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
