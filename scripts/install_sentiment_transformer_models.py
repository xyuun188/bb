"""Prepare optional FinBERT/CryptoBERT sentiment models on the model server."""

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


def run(ssh: paramiko.SSHClient, command: str, timeout: int = 900) -> str:
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
        script = textwrap.dedent(
            r"""
            set -euo pipefail
            mkdir -p /data/trade_models/Sentiment
            source ~/anaconda3/etc/profile.d/conda.sh
            conda activate trade_ml
            python - <<'PY'
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

targets = [
    ("ProsusAI/finbert", "/data/trade_models/Sentiment/finbert"),
]

for model_name, target in targets:
    path = Path(target)
    path.mkdir(parents=True, exist_ok=True)
    print(f"downloading {model_name} -> {target}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer.save_pretrained(target)
    model.save_pretrained(target)
    print(f"ready {target}")
PY
            """
        ).strip()
        with ssh.open_sftp().file("/tmp/install_sentiment_models.sh", "w") as remote:
            remote.write(script)
        print(run(
            ssh,
            "chmod +x /tmp/install_sentiment_models.sh && "
            "/tmp/install_sentiment_models.sh",
            timeout=1200,
        ))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
