"""Continue deployment after DeepSeek 14B weights are already downloaded."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "redeploy",
    ROOT / "scripts" / "redeploy_server_14b_kronos_architecture.py",
)
redeploy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(redeploy)


def main() -> None:
    info = redeploy.parse_server_info()
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
        start_14b = f"""#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate trade_vllm
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG={redeploy.LOG_DIR}/deepseek_14b_main.log
exec python -m vllm.entrypoints.openai.api_server \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --model "{redeploy.DEEPSEEK14_DIR}" \\
  --served-model-name {redeploy.DEEPSEEK14_SERVED} \\
  --trust-remote-code \\
  --max-model-len 8192 \\
  --gpu-memory-utilization 0.78 \\
  --dtype auto \\
  --enforce-eager > "$LOG" 2>&1
"""
        service_14b = """[Unit]
Description=DeepSeek R1 Distill Qwen 14B Trading vLLM OpenAI API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=linux
WorkingDirectory=/data/trade_ai
ExecStart=/data/trade_ai/scripts/start_deepseek_14b_main.sh
Restart=always
RestartSec=10
Environment=CUDA_VISIBLE_DEVICES=0
Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
"""
        service_tools = """[Unit]
Description=Trade Local AI Tools API
After=network-online.target deepseek-14b-main.service
Wants=network-online.target

[Service]
Type=simple
User=linux
WorkingDirectory=/data/trade_ai/tools
Environment=PATH=/home/linux/anaconda3/envs/trade_ml/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/home/linux/anaconda3/envs/trade_ml/bin/python -m uvicorn local_ai_tools_api:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=5
StandardOutput=append:/data/trade_ai/logs/local_ai_tools_api.log
StandardError=append:/data/trade_ai/logs/local_ai_tools_api.err.log

[Install]
WantedBy=multi-user.target
"""
        sftp = ssh.open_sftp()
        with sftp.file("/data/trade_ai/scripts/start_deepseek_14b_main.sh", "w") as remote:
            remote.write(start_14b)
        with sftp.file("/tmp/deepseek-14b-main.service", "w") as remote:
            remote.write(service_14b)
        with sftp.file("/data/trade_ai/tools/local_ai_tools_api.py", "w") as remote:
            remote.write(redeploy.LOCAL_AI_TOOLS_CODE)
        with sftp.file("/tmp/local-ai-tools.service", "w") as remote:
            remote.write(service_tools)
        sftp.close()

        cmd = """set -e
chmod +x /data/trade_ai/scripts/start_deepseek_14b_main.sh
sudo mv /tmp/deepseek-14b-main.service /etc/systemd/system/deepseek-14b-main.service
sudo mv /tmp/local-ai-tools.service /etc/systemd/system/local-ai-tools.service
sudo systemctl daemon-reload
sudo systemctl enable deepseek-14b-main.service local-ai-tools.service
sudo systemctl restart deepseek-14b-main.service
sleep 20
sudo systemctl restart local-ai-tools.service
sleep 5
echo '--- active ---'
systemctl is-active deepseek-14b-main.service || true
systemctl is-active local-ai-tools.service || true
echo '--- models ---'
curl -s --max-time 20 http://127.0.0.1:8000/v1/models || true
echo
echo '--- tools ---'
curl -s --max-time 10 http://127.0.0.1:8001/health || true
echo
echo '--- gpu ---'
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo '--- recent 14b log ---'
tail -n 100 /data/trade_ai/logs/deepseek_14b_main.log 2>/dev/null || true
echo '--- recent tools err ---'
tail -n 100 /data/trade_ai/logs/local_ai_tools_api.err.log 2>/dev/null || true
"""
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=900)
        print(stdout.read().decode("utf-8", "replace"))
        err = stderr.read().decode("utf-8", "replace")
        if err:
            print("STDERR:", err)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
