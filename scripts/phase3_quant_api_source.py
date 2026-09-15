"""Load the Phase 3 quant API source uploaded by the deployment script."""

from __future__ import annotations

from pathlib import Path


SERVICE_SOURCE_PATH = Path(__file__).with_name("phase3_quant_api_service.py")
SERVICE_CODE = SERVICE_SOURCE_PATH.read_text(encoding="utf-8")

