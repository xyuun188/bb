"""Dashboard API for optional vector memory retrieval."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel

from config.settings import settings
from services.vector_memory import get_vector_memory_service
from web_dashboard.api.security import require_dashboard_write_access

router = APIRouter()


class VectorMemorySearchRequest(BaseModel):
    query: str
    top_k: int = 8
    symbol: str = ""
    kind: str = ""


class VectorMemorySettingsRequest(BaseModel):
    enabled: bool | None = None
    backend: str | None = None


@router.get("/vector-memory/status")
async def vector_memory_status() -> dict[str, Any]:
    return await get_vector_memory_service().status()


@router.post("/vector-memory/search")
async def vector_memory_search(req: VectorMemorySearchRequest) -> dict[str, Any]:
    return await get_vector_memory_service().search(
        req.query,
        top_k=req.top_k,
        symbol=req.symbol,
        kind=req.kind,
    )


@router.post("/vector-memory/reindex")
async def vector_memory_reindex(
    _access: None = Depends(require_dashboard_write_access),
) -> dict[str, Any]:
    return await get_vector_memory_service().reindex_recent()


@router.post("/vector-memory/clear")
async def vector_memory_clear(
    _access: None = Depends(require_dashboard_write_access),
) -> dict[str, Any]:
    return await get_vector_memory_service().clear_index(reason="phase3_dashboard_clear_old_index")


@router.post("/vector-memory/settings")
async def vector_memory_settings(
    req: VectorMemorySettingsRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_dashboard_admin_key: str | None = Header(default=None),
) -> dict[str, Any]:
    await require_dashboard_write_access(request, authorization, x_dashboard_admin_key)
    updates: dict[str, str] = {}
    if req.enabled is not None:
        settings.vector_memory_enabled = bool(req.enabled)
        updates["VECTOR_MEMORY_ENABLED"] = "true" if settings.vector_memory_enabled else "false"
    if req.backend is not None:
        backend = str(req.backend or "auto").strip().lower()
        if backend not in {"auto", "zvec", "jsonl"}:
            backend = "auto"
        settings.vector_memory_backend = backend
        updates["VECTOR_MEMORY_BACKEND"] = backend
    if updates:
        settings.update_env_file(updates)
        await get_vector_memory_service().reset_store()
    return await get_vector_memory_service().status()
