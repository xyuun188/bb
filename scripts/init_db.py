#!/usr/bin/env python3
"""
Initialize the database — create all tables.
Run: python scripts/init_db.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.logging_config import setup_logging
from db.session import init_db

setup_logging()


async def main():
    print("Creating database tables...")
    await init_db()
    print("Database initialized successfully.")
    print(f"SQLite DB at: data/trading.db")


if __name__ == "__main__":
    asyncio.run(main())
