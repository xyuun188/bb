"""Runtime bridge between encrypted secure_settings rows and config.settings."""

from __future__ import annotations

import re
from typing import Any

import structlog

from config.settings import settings
from core.safe_output import safe_error_text
from db.session import get_session_ctx
from services.secure_settings import SecureSettingsError, SecureSettingsService

logger = structlog.get_logger(__name__)

STATIC_SECRET_KEYS: dict[str, str] = {
    "okx.paper.api_key": "okx_paper_api_key",
    "okx.paper.api_secret": "okx_paper_api_secret",
    "okx.paper.passphrase": "okx_paper_passphrase",
    "okx.live.api_key": "okx_live_api_key",
    "okx.live.api_secret": "okx_live_api_secret",
    "okx.live.passphrase": "okx_live_passphrase",
    "ai.api_key": "ai_api_key",
    "local_ai_tools.api_key": "local_ai_tools_api_key",
    "high_risk_review.api_key": "high_risk_review_api_key",
}

_MODEL_KEY_RE = re.compile(r"[^a-z0-9_.:-]+")


def secure_ai_model_key(name: str) -> str:
    normalized = _MODEL_KEY_RE.sub("_", str(name or "model").strip().lower()).strip("_.:-")
    return f"ai_model.{normalized or 'model'}.api_key"


async def set_runtime_secret(key: str, value: str, *, actor: str = "dashboard") -> None:
    async with get_session_ctx() as session:
        service = SecureSettingsService(session)
        await service.set_secret(key, value, actor=actor)


async def load_secure_settings_into_runtime() -> dict[str, Any]:
    """Overlay decrypted secure settings onto the in-memory Settings singleton."""

    loaded: dict[str, Any] = {"loaded_keys": [], "skipped": [], "error": ""}
    try:
        async with get_session_ctx() as session:
            service = SecureSettingsService(session)
            for key, attr in STATIC_SECRET_KEYS.items():
                value = await service.get_secret(key)
                if value:
                    setattr(settings, attr, value)
                    loaded["loaded_keys"].append(key)
            await _load_ai_model_keys(service, loaded)
    except SecureSettingsError as exc:
        loaded["error"] = safe_error_text(exc)
        logger.warning("secure runtime settings unavailable", error=loaded["error"])
    except Exception as exc:
        loaded["error"] = safe_error_text(exc)
        logger.warning("failed to load secure runtime settings", error=loaded["error"])
    return loaded


async def _load_ai_model_keys(service: SecureSettingsService, loaded: dict[str, Any]) -> None:
    model_names = {
        str(item.get("name") or "")
        for item in settings.get_fixed_ai_models(include_empty=True)
        if isinstance(item, dict) and item.get("name")
    }
    model_names.update(
        str(item.get("name") or "")
        for item in settings.ai_models
        if isinstance(item, dict) and item.get("name")
    )
    by_name = {str(item.get("name") or ""): item for item in settings.ai_models if isinstance(item, dict)}
    for name in sorted(model_names):
        key = secure_ai_model_key(name)
        value = await service.get_secret(key)
        if not value:
            continue
        row = by_name.get(name)
        if row is not None:
            row["api_key"] = value
        loaded["loaded_keys"].append(key)


def strip_secret_env_updates(updates: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime secrets from .env updates after saving them encrypted."""

    secret_keys = {
        "OKX_PAPER_API_KEY",
        "OKX_PAPER_API_SECRET",
        "OKX_PAPER_PASSPHRASE",
        "OKX_LIVE_API_KEY",
        "OKX_LIVE_API_SECRET",
        "OKX_LIVE_PASSPHRASE",
        "AI_API_KEY",
        "LOCAL_AI_TOOLS_API_KEY",
        "HIGH_RISK_REVIEW_API_KEY",
    }
    return {key: value for key, value in updates.items() if key not in secret_keys}


def scrub_ai_model_env(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return AI model configs safe to persist in .env without plaintext keys."""

    sanitized: list[dict[str, Any]] = []
    for item in models:
        row = dict(item)
        if row.get("api_key"):
            row["api_key"] = ""
        sanitized.append(row)
    return sanitized
