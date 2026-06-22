"""Prepare optional FinBERT/CryptoBERT sentiment models on the model server."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402


def main() -> None:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        script = textwrap.dedent(r"""
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
            """).strip()
        run_remote_text(ssh, "mkdir -p /data/trade_ai/scripts")
        remote_script_path = "/data/trade_ai/scripts/install_sentiment_models.sh"
        with ssh.open_sftp().file(remote_script_path, "w") as remote:
            remote.write(script)
        safe_print(
            run_remote_text(
                ssh,
                f"chmod +x {remote_script_path} && {remote_script_path}",
                timeout=1200,
            )
        )
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
