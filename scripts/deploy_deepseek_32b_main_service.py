"""Deprecated entrypoint for the rejected DeepSeek 32B main service."""

from __future__ import annotations

import sys

DEPRECATED_MESSAGE = (
    "DeepSeek-R1-Distill-Qwen-32B-AWQ is no longer an allowed main LLM service for "
    "this project. Use scripts/migrate_phase3_model_service_identity.py to manage "
    "Qwen3-32B-AWQ with non-thinking runtime controls."
)


def main() -> None:
    """Fail closed so the deprecated reasoning model cannot be redeployed by accident."""
    print(DEPRECATED_MESSAGE, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
