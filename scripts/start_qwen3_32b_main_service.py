"""Retired entry point for the obsolete Qwen3-32B service."""

from __future__ import annotations

RETIRED_MESSAGE = (
    "Qwen3-32B startup is retired for the verified single-A100 Phase 3 runtime; "
    "use migrate_phase3_model_service_identity.py."
)


def main() -> None:
    raise RuntimeError(RETIRED_MESSAGE)


if __name__ == "__main__":
    main()
