"""Deprecated entrypoint for the rejected DeepSeek 14B continuation path."""

from __future__ import annotations

import sys

DEPRECATED_MESSAGE = (
    "The DeepSeek 14B continuation deploy path is deprecated and must not be used. "
    "Use scripts/migrate_phase3_model_service_identity.py for the model services and "
    "scripts/deploy_local_ai_tools_service.py for local quant tools."
)


def main() -> None:
    """Fail closed so the old continuation script cannot alter server services."""
    print(DEPRECATED_MESSAGE, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
