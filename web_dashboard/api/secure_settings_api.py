from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.secret_utils import is_masked_secret
from db.session import get_session_ctx
from services.secure_settings import SecureSettingsError, SecureSettingsService, normalize_secure_key

router = APIRouter()


class SecureSettingUpdateRequest(BaseModel):
    key: str
    value: str
    actor: str = "dashboard"


@router.get("/secure-settings")
async def list_secure_settings(prefix: str | None = None) -> dict[str, object]:
    try:
        async with get_session_ctx() as session:
            service = SecureSettingsService(session)
            rows = await service.list_public(prefix=prefix)
    except SecureSettingsError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"count": len(rows), "settings": [row.as_dict() for row in rows]}


@router.get("/secure-settings/{key:path}")
async def get_secure_setting(key: str) -> dict[str, object]:
    try:
        normalized = normalize_secure_key(key)
        async with get_session_ctx() as session:
            service = SecureSettingsService(session)
            row = await service.public(normalized)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SecureSettingsError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if row is None:
        return {"key": normalized, "configured": False, "masked_value": "", "fingerprint": ""}
    return row.as_dict()


@router.put("/secure-settings/{key:path}")
async def update_secure_setting(key: str, req: SecureSettingUpdateRequest) -> dict[str, object]:
    try:
        normalized = normalize_secure_key(key)
        body_key = normalize_secure_key(req.key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if normalized != body_key:
        raise HTTPException(status_code=400, detail="path key and body key must match")
    if is_masked_secret(req.value):
        raise HTTPException(status_code=400, detail="masked placeholders are not persisted")
    if not req.value:
        raise HTTPException(status_code=400, detail="secure setting value must not be empty")
    try:
        async with get_session_ctx() as session:
            service = SecureSettingsService(session)
            row = await service.set_secret(normalized, req.value, actor=req.actor)
    except SecureSettingsError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return row.as_dict()
