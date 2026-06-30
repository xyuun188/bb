"""Encrypted model-server connection settings for dashboard monitoring."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.remote_server_info import RemoteServerInfo, load_model_server_info
from core.secret_utils import is_masked_secret, mask_secret
from db.session import get_session_ctx
from services.secure_settings import SecureSettingsError, SecureSettingsService

MODEL_SERVER_PREFIX = "model_server"
MODEL_SERVER_HOST_KEY = f"{MODEL_SERVER_PREFIX}.host"
MODEL_SERVER_PORT_KEY = f"{MODEL_SERVER_PREFIX}.port"
MODEL_SERVER_USERNAME_KEY = f"{MODEL_SERVER_PREFIX}.username"
MODEL_SERVER_PASSWORD_KEY = f"{MODEL_SERVER_PREFIX}.password"
MODEL_SERVER_SECURE_SOURCE = Path("<secure_settings>")
MODEL_SERVER_CONFIG_MESSAGE = "请在系统设置 > 模型服务器 中配置服务器连接信息。"


class ModelServerConfigError(RuntimeError):
    """Raised when model-server settings cannot be used."""


class ModelServerConfigNotConfigured(ModelServerConfigError):
    """Raised when encrypted model-server settings are incomplete."""


@dataclass(frozen=True, slots=True)
class ModelServerSettingsPayload:
    """Public model-server settings shape returned to the Dashboard."""

    configured: bool
    host: str
    port: int | None
    username: str
    password_configured: bool
    masked_password: str
    updated_at: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password_configured": self.password_configured,
            "masked_password": self.masked_password,
            "updated_at": self.updated_at,
        }


async def get_model_server_settings_public() -> ModelServerSettingsPayload:
    """Return saved model-server settings without exposing the password."""

    try:
        async with get_session_ctx() as session:
            service = SecureSettingsService(session)
            host = (await service.get_secret(MODEL_SERVER_HOST_KEY) or "").strip()
            port_text = (await service.get_secret(MODEL_SERVER_PORT_KEY) or "").strip()
            username = (await service.get_secret(MODEL_SERVER_USERNAME_KEY) or "").strip()
            password = await service.get_secret(MODEL_SERVER_PASSWORD_KEY)
            public_password = await service.public(MODEL_SERVER_PASSWORD_KEY)
    except SecureSettingsError as exc:
        raise ModelServerConfigError(str(exc)) from exc

    port = _safe_port_or_none(port_text)
    configured = bool(host and port and username and password)
    return ModelServerSettingsPayload(
        configured=configured,
        host=host,
        port=port,
        username=username,
        password_configured=bool(password),
        masked_password=mask_secret(password or "") if password else "",
        updated_at=public_password.updated_at if public_password else None,
    )


async def save_model_server_settings(
    *,
    host: str,
    port: int | str,
    username: str,
    password: str | None,
    actor: str = "dashboard",
) -> ModelServerSettingsPayload:
    """Validate and persist model-server settings in encrypted storage."""

    info = await build_model_server_info_from_update(
        host=host,
        port=port,
        username=username,
        password=password,
    )
    async with get_session_ctx() as session:
        service = SecureSettingsService(session)
        await service.set_secret(MODEL_SERVER_HOST_KEY, info.host, actor=actor)
        await service.set_secret(MODEL_SERVER_PORT_KEY, str(info.port), actor=actor)
        await service.set_secret(MODEL_SERVER_USERNAME_KEY, info.username, actor=actor)
        if str(password or "").strip() and not is_masked_secret(str(password or "")):
            await service.set_secret(MODEL_SERVER_PASSWORD_KEY, info.password, actor=actor)

    return await get_model_server_settings_public()


async def build_model_server_info_from_update(
    *,
    host: str,
    port: int | str,
    username: str,
    password: str | None,
) -> RemoteServerInfo:
    """Build validated info using the existing password when password is omitted."""

    try:
        async with get_session_ctx() as session:
            service = SecureSettingsService(session)
            existing_password = await service.get_secret(MODEL_SERVER_PASSWORD_KEY)
    except SecureSettingsError as exc:
        raise ModelServerConfigError(str(exc)) from exc

    password_text = str(password or "")
    keep_existing_password = not password_text.strip() or is_masked_secret(password_text)
    password_to_validate = existing_password if keep_existing_password else password_text.strip()
    if not password_to_validate:
        raise ModelServerConfigNotConfigured("模型服务器密码不能为空。")

    return build_remote_server_info(
        host=host,
        port=port,
        username=username,
        password=password_to_validate,
    )


def build_remote_server_info(
    *,
    host: str,
    port: int | str,
    username: str,
    password: str,
) -> RemoteServerInfo:
    """Build a validated RemoteServerInfo object from settings values."""

    try:
        return RemoteServerInfo(
            host=str(host or "").strip(),
            port=port,
            username=str(username or "").strip(),
            password=str(password or "").strip(),
            source_path=MODEL_SERVER_SECURE_SOURCE,
        )
    except ValueError as exc:
        raise ModelServerConfigError(f"模型服务器配置无效：{exc}") from exc


async def load_model_server_info_from_secure_settings() -> RemoteServerInfo:
    """Load model-server SSH settings from encrypted storage."""

    try:
        async with get_session_ctx() as session:
            service = SecureSettingsService(session)
            host = await service.get_secret(MODEL_SERVER_HOST_KEY)
            port = await service.get_secret(MODEL_SERVER_PORT_KEY)
            username = await service.get_secret(MODEL_SERVER_USERNAME_KEY)
            password = await service.get_secret(MODEL_SERVER_PASSWORD_KEY)
    except SecureSettingsError as exc:
        raise ModelServerConfigError(str(exc)) from exc

    missing = [
        label
        for label, value in (
            ("host", host),
            ("port", port),
            ("username", username),
            ("password", password),
        )
        if not str(value or "").strip()
    ]
    if missing:
        raise ModelServerConfigNotConfigured(MODEL_SERVER_CONFIG_MESSAGE)

    return build_remote_server_info(
        host=str(host),
        port=str(port),
        username=str(username),
        password=str(password),
    )


def load_model_server_info_from_secure_settings_sync() -> RemoteServerInfo:
    """Synchronous wrapper used by the server-monitor thread."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(load_model_server_info_from_secure_settings())
    raise ModelServerConfigError("model-server settings must be loaded outside the event loop")


def load_model_server_info_for_monitor(project_root: Path) -> RemoteServerInfo:
    """Load model-server info for monitoring.

    Encrypted Dashboard settings are the production source of truth. Ignored
    local txt files remain a development-only fallback when present, but missing
    files no longer instruct operators to upload secrets to the server.
    """

    try:
        return load_model_server_info_from_secure_settings_sync()
    except ModelServerConfigNotConfigured as secure_exc:
        try:
            return load_model_server_info(project_root)
        except FileNotFoundError:
            raise secure_exc from None
    except ModelServerConfigError:
        raise


async def load_model_server_info_for_monitor_async(project_root: Path) -> RemoteServerInfo:
    """Async variant for Dashboard audit paths that already run in an event loop."""

    try:
        return await load_model_server_info_from_secure_settings()
    except ModelServerConfigNotConfigured as secure_exc:
        try:
            return load_model_server_info(project_root)
        except FileNotFoundError:
            raise secure_exc from None
    except ModelServerConfigError:
        raise


def _safe_port_or_none(value: str) -> int | None:
    try:
        port = int(str(value or "").strip())
    except (TypeError, ValueError):
        return None
    if port <= 0 or port > 65535:
        return None
    return port
