"""Deprecated entrypoint for the rejected DeepSeek 14B remote stack."""

from __future__ import annotations

import sys

DEPRECATED_MESSAGE = (
    "The DeepSeek 14B/Kronos redeploy path is deprecated and must not replace "
    "the verified Phase 3 service identities. Use "
    "scripts/migrate_phase3_model_service_identity.py and "
    "scripts/deploy_local_ai_tools_service.py."
)


def main() -> None:
    """Fail closed so the old 14B stack cannot overwrite the current architecture."""
    print(DEPRECATED_MESSAGE, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
