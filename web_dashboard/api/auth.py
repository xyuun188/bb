from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from config.settings import settings
from db.session import get_session_ctx
from services.dashboard_auth_service import (
    authenticate_dashboard_user,
    change_dashboard_user_password,
    create_dashboard_user,
    deactivate_dashboard_user,
    delete_dashboard_user,
    get_dashboard_user,
    has_dashboard_users,
    list_dashboard_users,
    configured_dashboard_username,
    normalize_username,
    update_current_dashboard_account,
    update_dashboard_user,
)
from web_dashboard.api.security import (
    DashboardAuthContext,
    create_dashboard_session,
    dashboard_admin_key_matches,
    dashboard_login_required,
    ensure_dashboard_login,
    login_response_cookie,
    logout_response_cookie,
    verify_dashboard_password,
)

router = APIRouter()


class DashboardLoginRequest(BaseModel):
    username: str
    password: str


class DashboardAccountUpdateRequest(BaseModel):
    username: str
    email: str = ""


class DashboardPasswordChangeRequest(BaseModel):
    current_password: str = ""
    new_password: str


class DashboardUserCreateRequest(BaseModel):
    username: str
    email: str = ""
    password: str
    role: str = "admin"
    is_active: bool = True


class DashboardUserUpdateRequest(BaseModel):
    email: str | None = None
    password: str | None = None
    role: str | None = None
    is_active: bool | None = None


@router.get("/auth/login")
async def login_page(request: Request):
    if not dashboard_login_required(request):
        return RedirectResponse(url="/", status_code=302)
    return HTMLResponse(content=_LOGIN_HTML, status_code=200)


@router.get("/auth/status")
async def auth_status(request: Request) -> dict[str, object]:
    session = ensure_dashboard_login(request)
    if session is not None and settings.dashboard_auth_enabled:
        async with get_session_ctx() as db_session:
            current = await get_dashboard_user(db_session, session.username)
        if current is None or not current.is_active:
            raise HTTPException(
                status_code=401,
                detail="当前登录账号不存在或已停用，请重新登录。",
            )
    return {
        "authenticated": True,
        "username": session.username if session else "",
        "login_required": dashboard_login_required(request),
        "auth_enabled": bool(settings.dashboard_auth_enabled),
    }


@router.post("/auth/login")
async def login(request: Request, payload: DashboardLoginRequest) -> JSONResponse:
    if not dashboard_login_required(request):
        return JSONResponse({"authenticated": True, "redirect": "/"})
    configured_username = configured_dashboard_username()
    configured_hash = str(settings.dashboard_auth_password_hash or "").strip()
    has_admin_key_fallback = bool(str(settings.dashboard_admin_api_key or "").strip())

    authenticated_username = ""
    db_has_users = False
    try:
        async with get_session_ctx() as session:
            db_has_users = await has_dashboard_users(session)
            user = await authenticate_dashboard_user(session, payload.username, payload.password)
            if user is not None:
                authenticated_username = user.username
    except Exception:
        authenticated_username = ""
    if db_has_users and not authenticated_username:
        raise HTTPException(status_code=401, detail="账号或密码错误。")

    if not authenticated_username:
        if not configured_hash and not has_admin_key_fallback:
            raise HTTPException(
                status_code=503,
                detail="控制台登录密码未配置。",
            )
        if payload.username.strip() != configured_username:
            raise HTTPException(status_code=401, detail="账号或密码错误。")
        password_ok = bool(configured_hash) and verify_dashboard_password(
            payload.password, configured_hash
        )
        if not password_ok and has_admin_key_fallback:
            password_ok = dashboard_admin_key_matches(None, payload.password)
        if not password_ok:
            raise HTTPException(status_code=401, detail="账号或密码错误。")
        authenticated_username = configured_username

    session_token = create_dashboard_session(authenticated_username)
    response = JSONResponse({"authenticated": True, "redirect": "/"})
    login_response_cookie(response, session_token)
    return response


@router.post("/auth/logout")
async def logout() -> JSONResponse:
    response = JSONResponse({"authenticated": False, "redirect": "/login"})
    logout_response_cookie(response)
    return response


@router.get("/auth/account")
async def dashboard_account(
    request: Request,
    authorization: str | None = Header(default=None),
    x_dashboard_admin_key: str | None = Header(default=None),
) -> dict[str, object]:
    context = _auth_context(request, authorization, x_dashboard_admin_key)
    async with get_session_ctx() as session:
        users = await list_dashboard_users(session)
        current = await get_dashboard_user(session, context.username)
    if (current is None or not current.is_active) and settings.dashboard_auth_enabled:
        raise HTTPException(
            status_code=401,
            detail="当前登录账号不存在或已停用，请重新登录。",
        )
    return {
        "auth_enabled": bool(settings.dashboard_auth_enabled),
        "current_user": current.as_dict() if current else {},
        "users": [user.as_dict() for user in users],
        "multi_user_supported": True,
    }


@router.post("/auth/account")
async def update_dashboard_account(
    request: Request,
    payload: DashboardAccountUpdateRequest,
    authorization: str | None = Header(default=None),
    x_dashboard_admin_key: str | None = Header(default=None),
) -> JSONResponse:
    context = _auth_context(request, authorization, x_dashboard_admin_key)
    try:
        async with get_session_ctx() as session:
            user = await update_current_dashboard_account(
                session,
                current_username=context.username,
                username=payload.username,
                email=payload.email,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    settings.dashboard_auth_username = user.username
    settings.dashboard_auth_email = user.email
    response = JSONResponse({"status": "ok", "user": user.as_dict()})
    if user.username != context.username:
        login_response_cookie(response, create_dashboard_session(user.username))
    return response


@router.post("/auth/account/password")
async def change_dashboard_password(
    request: Request,
    payload: DashboardPasswordChangeRequest,
    authorization: str | None = Header(default=None),
    x_dashboard_admin_key: str | None = Header(default=None),
) -> dict[str, object]:
    context = _auth_context(request, authorization, x_dashboard_admin_key)
    has_admin_key = dashboard_admin_key_matches(authorization, x_dashboard_admin_key)
    try:
        async with get_session_ctx() as session:
            user = await change_dashboard_user_password(
                session,
                username=context.username,
                current_password=payload.current_password,
                new_password=payload.new_password,
                require_current_password=not has_admin_key,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "user": user.as_dict()}


@router.get("/auth/users")
async def dashboard_users(
    request: Request,
    authorization: str | None = Header(default=None),
    x_dashboard_admin_key: str | None = Header(default=None),
) -> dict[str, object]:
    _auth_context(request, authorization, x_dashboard_admin_key)
    async with get_session_ctx() as session:
        users = await list_dashboard_users(session)
    return {"users": [user.as_dict() for user in users]}


@router.post("/auth/users")
async def create_dashboard_account_user(
    request: Request,
    payload: DashboardUserCreateRequest,
    authorization: str | None = Header(default=None),
    x_dashboard_admin_key: str | None = Header(default=None),
) -> dict[str, object]:
    _auth_context(request, authorization, x_dashboard_admin_key)
    try:
        async with get_session_ctx() as session:
            user = await create_dashboard_user(
                session,
                username=payload.username,
                email=payload.email,
                password=payload.password,
                role=payload.role,
                is_active=payload.is_active,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "user": user.as_dict()}


@router.put("/auth/users/{username}")
async def update_dashboard_account_user(
    username: str,
    request: Request,
    payload: DashboardUserUpdateRequest,
    authorization: str | None = Header(default=None),
    x_dashboard_admin_key: str | None = Header(default=None),
) -> dict[str, object]:
    context = _auth_context(request, authorization, x_dashboard_admin_key)
    if payload.is_active is False and normalize_username(username) == normalize_username(
        context.username
    ):
        raise HTTPException(status_code=400, detail="当前登录账号不能停用")
    try:
        async with get_session_ctx() as session:
            user = await update_dashboard_user(
                session,
                username=username,
                email=payload.email,
                password=payload.password,
                role=payload.role,
                is_active=payload.is_active,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "user": user.as_dict()}


@router.post("/auth/users/{username}/deactivate")
async def deactivate_dashboard_account_user(
    username: str,
    request: Request,
    authorization: str | None = Header(default=None),
    x_dashboard_admin_key: str | None = Header(default=None),
) -> dict[str, object]:
    context = _auth_context(request, authorization, x_dashboard_admin_key)
    if normalize_username(username) == normalize_username(context.username):
        raise HTTPException(status_code=400, detail="当前登录账号不能停用")
    try:
        async with get_session_ctx() as session:
            user = await deactivate_dashboard_user(session, username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "user": user.as_dict()}


@router.delete("/auth/users/{username}")
async def delete_dashboard_account_user(
    username: str,
    request: Request,
    authorization: str | None = Header(default=None),
    x_dashboard_admin_key: str | None = Header(default=None),
) -> dict[str, object]:
    context = _auth_context(request, authorization, x_dashboard_admin_key)
    if normalize_username(username) == normalize_username(context.username):
        raise HTTPException(status_code=400, detail="当前登录账号不能删除")
    try:
        async with get_session_ctx() as session:
            user = await delete_dashboard_user(session, username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "user": user.as_dict()}


def dashboard_login_page_html() -> str:
    return _LOGIN_HTML


def _auth_context(
    request: Request,
    authorization: str | None,
    dashboard_admin_key: str | None,
) -> DashboardAuthContext:
    fallback_username = configured_dashboard_username()
    if dashboard_admin_key_matches(authorization, dashboard_admin_key):
        return DashboardAuthContext(username=fallback_username, issued_at=0, expires_at=0)
    return ensure_dashboard_login(request) or DashboardAuthContext(
        username=fallback_username,
        issued_at=0,
        expires_at=0,
    )


_LOGIN_HTML = """<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>苍鸮量化 · 登录</title>
    <link rel=\"icon\" type=\"image/svg+xml\" href=\"/static/img/cangxiao-icon.svg\" />
    <style>
        :root { color-scheme: dark; --cyan:#34d6ff; --mint:#6cffb8; --night:#06101f; --panel:rgba(9,21,38,.82); --line:rgba(110,255,198,.22); }
        * { box-sizing: border-box; }
        body { margin: 0; min-height: 100vh; display: grid; place-items: center; overflow: hidden; background: radial-gradient(circle at 22% 18%, rgba(52,214,255,.22), transparent 28%), radial-gradient(circle at 82% 78%, rgba(108,255,184,.18), transparent 34%), linear-gradient(135deg, #06101f 0%, #081a2f 48%, #041b18 100%); color: #e9f7ff; font-family: 'IBM Plex Sans', 'Noto Sans SC', 'Microsoft YaHei', system-ui, sans-serif; }
        body::before { content: ''; position: fixed; inset: -20%; background-image: linear-gradient(rgba(52,214,255,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(108,255,184,.06) 1px, transparent 1px); background-size: 54px 54px; transform: rotate(-8deg); mask-image: radial-gradient(circle at center, #000, transparent 72%); }
        .login-shell { width: min(980px, calc(100vw - 36px)); display: grid; grid-template-columns: 1.08fr .92fr; border: 1px solid var(--line); border-radius: 28px; background: rgba(5,14,26,.68); box-shadow: 0 30px 80px rgba(0,0,0,.46), inset 0 1px 0 rgba(255,255,255,.05); overflow: hidden; backdrop-filter: blur(18px); position: relative; }
        .brand-panel { padding: 46px; min-height: 560px; background: linear-gradient(150deg, rgba(16,49,78,.86), rgba(6,20,34,.6)); position: relative; }
        .brand-panel::after { content: ''; position: absolute; inset: 32px; border: 1px solid rgba(52,214,255,.12); border-radius: 22px; pointer-events: none; }
        .brand-mark { width: 96px; height: 96px; border-radius: 24px; box-shadow: 0 0 48px rgba(52,214,255,.25); }
        h1 { margin: 28px 0 12px; font-size: clamp(34px, 5vw, 62px); line-height: 1; letter-spacing: .08em; }
        .brand-panel p { max-width: 420px; margin: 0; color: #a8bfd0; font-size: 15px; line-height: 1.9; }
        .signal { margin-top: 42px; display: grid; gap: 12px; color: #8fb2c7; font-size: 13px; }
        .signal span { display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,.08); padding-bottom: 10px; }
        form { padding: 56px 46px; display: flex; flex-direction: column; justify-content: center; background: var(--panel); }
        h2 { margin: 0 0 8px; font-size: 26px; }
        .subtitle { margin: 0 0 28px; color: #8da8ba; }
        label { display: block; margin: 16px 0 8px; font-size: 13px; color: #a9c4d6; letter-spacing: .04em; }
        input { width: 100%; border-radius: 14px; border: 1px solid rgba(108,255,184,.2); padding: 14px 15px; background: rgba(4,13,25,.8); color: #f6fdff; font-size: 15px; outline: none; transition: border-color .18s, box-shadow .18s; }
        input:focus { border-color: var(--cyan); box-shadow: 0 0 0 4px rgba(52,214,255,.12); }
        button { margin-top: 22px; width: 100%; border: 0; border-radius: 14px; padding: 14px 16px; background: linear-gradient(135deg, var(--cyan), var(--mint)); color: #03111a; font-size: 15px; font-weight: 800; cursor: pointer; box-shadow: 0 18px 34px rgba(52,214,255,.22); }
        .hint { margin-top: 16px; min-height: 22px; color: #9fb3d8; font-size: 13px; }
        @media (max-width: 760px) { .login-shell { grid-template-columns: 1fr; } .brand-panel { min-height: 280px; padding: 32px; } form { padding: 32px; } }
    </style>
</head>
<body>
    <main class=\"login-shell\">
        <section class=\"brand-panel\" aria-label=\"苍鸮量化品牌\">
            <img class=\"brand-mark\" src=\"/static/img/cangxiao-logo.svg\" alt=\"苍鸮量化\" />
            <h1>苍鸮量化</h1>
            <p>像夜行猫头鹰一样，在噪声里识别真正有价值的价格信号；把模型协作、风险纪律和持续复盘，沉淀成稳定可执行的交易系统。</p>
            <div class=\"signal\"><span><b>市场洞察</b><em>先看结构，再看机会</em></span><span><b>风险纪律</b><em>先保生存，再做扩张</em></span><span><b>策略进化</b><em>用复盘推动持续学习</em></span></div>
        </section>
        <form id=\"login-form\">
            <h2>安全登录</h2>
            <p class=\"subtitle\">进入深色专业交易看板</p>
            <label for=\"username\">账号</label>
            <input id=\"username\" name=\"username\" autocomplete=\"username\" placeholder=\"请输入登录账号\" />
            <label for=\"password\">密码</label>
            <input id=\"password\" name=\"password\" type=\"password\" autocomplete=\"current-password\" />
            <button type=\"submit\">进入苍鸮量化</button>
            <div class=\"hint\" id=\"hint\"></div>
        </form>
    </main>
    <script>
        const form = document.getElementById('login-form');
        const hint = document.getElementById('hint');
        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            hint.textContent = '正在验证登录信息...';
            const payload = {
                username: document.getElementById('username').value,
                password: document.getElementById('password').value,
            };
            try {
                const response = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify(payload),
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || '登录失败，请检查账号与密码。');
                }
                window.location.href = data.redirect || '/';
            } catch (error) {
                hint.textContent = error.message || String(error);
            }
        });
    </script>
</body>
</html>"""
