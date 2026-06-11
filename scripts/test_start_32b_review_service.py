"""Deprecated entrypoint for starting the local 32B review service."""

from __future__ import annotations

import sys

DEPRECATED_MESSAGE = (
    "Starting qwen3-32b-review.service is deprecated. High-risk review and deep "
    "review flows must use the online model configured in the trading app."
)


def main() -> None:
    """Fail closed so local review service cannot be started accidentally."""
    print(DEPRECATED_MESSAGE, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
