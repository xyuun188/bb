"""Install a local scheduled task for server-side quant model training.

The training data lives in the local trading database, so the scheduled job
runs on this machine and pushes fresh samples to the remote local AI tools API.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "TradeLocalAIToolsAutoTrain"


def _task_command() -> str:
    script = (ROOT / "scripts" / "run_local_ai_tools_autotrain.cmd").resolve()
    return f'"{script}"'


def main() -> None:
    if os.name != "nt":
        raise SystemExit("This installer currently supports Windows scheduled tasks only.")

    command = _task_command()
    subprocess.run(
        [
            "schtasks.exe",
            "/Create",
            "/F",
            "/TN",
            TASK_NAME,
            "/SC",
            "HOURLY",
            "/MO",
            "6",
            "/TR",
            command,
        ],
        check=True,
    )
    subprocess.run(["schtasks.exe", "/Query", "/TN", TASK_NAME], check=True)
    print(f"Installed scheduled task: {TASK_NAME}")
    print(f"Command: {command}")


if __name__ == "__main__":
    main()
