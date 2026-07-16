"""Retired entry point for the removed legacy local-ai-tools systemd unit."""

from __future__ import annotations

RETIRED_MESSAGE = (
    "The legacy local-ai-tools.service path repair is retired; deploy and manage "
    "bb-phase3-quant-api.service with deploy_local_ai_tools_service.py."
)


def main() -> None:
    raise RuntimeError(RETIRED_MESSAGE)


if __name__ == "__main__":
    main()
