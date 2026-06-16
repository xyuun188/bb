#!/usr/bin/env python3
"""Generate a PBKDF2 hash for DASHBOARD_AUTH_PASSWORD_HASH."""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_dashboard.api.security import hash_dashboard_password


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("password", nargs="?", help="Password to hash. Omit to prompt securely.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = args.password
    if password is None:
        password = getpass.getpass("Dashboard password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            raise SystemExit("Passwords do not match.")
    if not password:
        raise SystemExit("Password must not be empty.")
    sys.stdout.write(hash_dashboard_password(password) + "\n")


if __name__ == "__main__":
    main()
