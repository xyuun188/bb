"""Deprecated entrypoint for the rejected DeepSeek 14B remote stack."""

from __future__ import annotations

import sys

DEPRECATED_MESSAGE = (
    "The DeepSeek 14B/Kronos redeploy path is deprecated and must not replace "
    "the approved Qwen3-32B-AWQ main LLM service. Use "
    "scripts/deploy_qwen3_32b_main_service.py and scripts/deploy_local_ai_tools_service.py."
)


def main() -> None:
    """Fail closed so the old 14B stack cannot overwrite the current architecture."""
    print(DEPRECATED_MESSAGE, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
