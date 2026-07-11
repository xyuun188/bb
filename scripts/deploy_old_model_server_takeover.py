#!/usr/bin/env python3
"""Report the retired old-server model-alias takeover path.

The original takeover installed an 8003 proxy that renamed qwen3-14b-trade to
BB-FinQuant-Expert-14B. That path is intentionally disabled: deployments must
now use ``finquant_expert_lora_training.py`` so 8003 can only expose a verified
LoRA adapter selected through the versioned current/rollback registry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_server_info import DEFAULT_ACCOUNT_INFO_DIR, parse_remote_server_info  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

OLD_PROFILE_FILENAME = "大模型服务器信息.txt"
REPLACEMENT_SCRIPT = ROOT / "scripts" / "finquant_expert_lora_training.py"


def _load_old_server_info(account_dir: Path):
    path = account_dir / OLD_PROFILE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"old model-server file not found: {path}")
    return parse_remote_server_info(
        path.read_text(encoding="utf-8", errors="replace"),
        source_path=path,
    )


def deploy(*, account_dir: Path, apply: bool) -> dict[str, object]:
    info = _load_old_server_info(account_dir)
    result: dict[str, object] = {
        "apply": False,
        "retired": True,
        "target": info.redacted(),
        "reason": "pure_model_alias_is_not_a_valid_finquant_specialization",
        "replacement": str(REPLACEMENT_SCRIPT),
        "required_flags": [
            "--train",
            "--stop-inference-for-training",
            "--switch-service",
        ],
        "rollback": "Use --rollback-service --switch-service on the replacement script.",
    }
    if apply:
        raise RuntimeError(
            "The pure 8003 model-alias takeover is retired. Train or select a verified "
            "BB-FinQuant LoRA adapter with scripts/finquant_expert_lora_training.py."
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-dir", type=Path, default=DEFAULT_ACCOUNT_INFO_DIR)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Fail closed because the alias-only takeover has been retired.",
    )
    args = parser.parse_args(argv)
    result = deploy(account_dir=args.account_dir, apply=args.apply)
    safe_print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
