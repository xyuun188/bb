#!/usr/bin/env python3
"""Retired entry point for the obsolete Phase 3 Qwen3-32B download."""

from __future__ import annotations

import argparse

RETIRED_MESSAGE = (
    "Phase 3 Qwen3-32B download is retired for the verified single-A100 runtime; "
    "the canonical decision carrier is Qwen3-14B managed by "
    "migrate_phase3_model_service_identity.py."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--status", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    raise RuntimeError(RETIRED_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(main())
