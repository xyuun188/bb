"""Deprecated entrypoint for local 32B high-risk review service setup."""

from __future__ import annotations

import sys

DEPRECATED_MESSAGE = (
    "Local 32B high-risk review service setup is deprecated. High-risk review "
    "must use the online model configured in the trading app; keep the remote "
    "Qwen3-32B-AWQ service focused on the main local LLM role."
)


def main() -> None:
    """Fail closed so high-risk review is not moved back onto the local server."""
    print(DEPRECATED_MESSAGE, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
