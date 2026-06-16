"""
FastAPI application for the trading dashboard.
Serves REST API + WebSocket on the configured dashboard port.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from web_dashboard.api.auth import dashboard_login_page_html
from web_dashboard.api.router import api_router
from web_dashboard.api.security import (
    SESSION_COOKIE_NAME,
    dashboard_admin_key_matches,
    dashboard_login_required,
    dashboard_session_from_token,
    ensure_dashboard_login,
    is_dashboard_write_request,
    validate_dashboard_write_access,
)
from web_dashboard.api.text_sanitize import sanitize_payload
from web_dashboard.api.ws_endpoints import WebSocketManager

logger = structlog.get_logger(__name__)
PUBLIC_AUTH_PATHS = {
    "/login",
    "/api/auth/login",
    "/api/auth/status",
    "/api/auth/logout",
}

# Global WebSocket manager
ws_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    yield
    await ws_manager.close_all()


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Crypto Trading System",
        description="Multi-model AI quantitative trading dashboard",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.dashboard_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def sanitize_json_text_middleware(request: Request, call_next):
        """Protect Dashboard access, then sanitize JSON text responses."""
        if _request_needs_dashboard_session(request):
            try:
                _ensure_dashboard_http_access(request)
            except HTTPException as exc:
                if request.url.path.startswith("/api"):
                    return JSONResponse(content={"detail": exc.detail}, status_code=exc.status_code)
                return RedirectResponse(url="/login", status_code=302)

        if is_dashboard_write_request(request):
            if not _is_public_auth_path(request.url.path):
                try:
                    validate_dashboard_write_access(
                        request,
                        authorization=request.headers.get("authorization"),
                        dashboard_admin_key=request.headers.get("x-dashboard-admin-key"),
                    )
                except HTTPException as exc:
                    return JSONResponse(
                        content={"detail": exc.detail},
                        status_code=exc.status_code,
                        headers=getattr(exc, "headers", None),
                    )

        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if not request.url.path.startswith("/api") or "application/json" not in content_type:
            return response
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        if not body:
            return response
        try:
            payload = json.loads(body)
        except Exception as exc:
            logger.debug(
                "dashboard JSON sanitize skipped invalid payload",
                path=request.url.path,
                status_code=response.status_code,
                error=type(exc).__name__,
            )
            return JSONResponse(
                content=body.decode("utf-8", errors="replace"),
                status_code=response.status_code,
            )
        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"content-length", "content-type"}
        }
        return JSONResponse(
            content=sanitize_payload(payload),
            status_code=response.status_code,
            headers=headers,
        )

    app.include_router(api_router, prefix="/api")

    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/login", response_class=HTMLResponse)
    async def login_page():
        return HTMLResponse(content=dashboard_login_page_html())

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        try:
            ensure_dashboard_login(request)
        except HTTPException:
            return HTMLResponse(content=dashboard_login_page_html(), status_code=401)
        index_path = static_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
        return HTMLResponse(content=DEFAULT_INDEX_HTML)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        if dashboard_login_required(ws) and not dashboard_session_from_token(
            ws.cookies.get(SESSION_COOKIE_NAME)
        ):
            await ws.close(code=1008)
            return
        await ws_manager.connect(ws)
        try:
            await ws.send_json({"type": "connected", "message": "Dashboard WebSocket connected"})
            while True:
                data = await ws.receive_text()
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                except json.JSONDecodeError as exc:
                    logger.debug(
                        "dashboard websocket ignored invalid JSON",
                        error=type(exc).__name__,
                    )
        except WebSocketDisconnect:
            pass
        finally:
            await ws_manager.disconnect(ws)

    return app


def _request_needs_dashboard_session(request: Request) -> bool:
    path = request.url.path
    if _is_public_auth_path(path) or path.startswith("/docs") or path.startswith("/openapi"):
        return False
    return path == "/" or path.startswith("/api") or path.startswith("/static")


def _is_public_auth_path(path: str) -> bool:
    return path in PUBLIC_AUTH_PATHS


def _ensure_dashboard_http_access(request: Request) -> None:
    if dashboard_admin_key_matches(
        request.headers.get("authorization"),
        request.headers.get("x-dashboard-admin-key"),
    ):
        return
    ensure_dashboard_login(request)


DEFAULT_INDEX_HTML = (
    """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>苍鸮量化</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #0d1117; color: #c9d1d9; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #58a6ff; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 10px 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .stat { font-size: 24px; font-weight: bold; color: #58a6ff; }
        .label { font-size: 12px; color: #8b949e; text-transform: uppercase; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }
        th { color: #8b949e; font-size: 12px; }
        .badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; }
        .badge-long { background: #1a7f37; color: #7ee787; }
        .badge-short { background: #da3633; color: #ff7b72; }
        .badge-hold { background: #30363d; color: #8b949e; }
        .badge-live { background: #1a7f37; color: #7ee787; }
        .badge-paper { background: #9e6a03; color: #e3b341; }
    </style>
</head>
<body>
    <div class="container">
        <h1>苍鸮量化</h1>
        <div class="card">
            <p>看板正在加载。如果长时间停留在这里，请检查 Dashboard 服务是否正常运行。</p>
            <p>端口："""
    + str(settings.dashboard_port)
    + """</p>
        </div>
    </div>
</body>
</html>
"""
)


app = create_app()
