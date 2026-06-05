"""
Exchange & AI settings API — get/set OKX credentials (paper/live split), AI config, test connections.
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.settings import ENSEMBLE_TRADER_NAME, settings
from web_dashboard.api import dashboard as _dash

router = APIRouter()
_OKX_BALANCE_CACHE: dict[str, tuple[float, dict]] = {}
_OKX_BALANCE_TTL_SECONDS = 10.0


# ── Request models ──

class OKXSettingsRequest(BaseModel):
    mode: str  # "paper" or "live"
    api_key: str | None = None
    api_secret: str | None = None
    passphrase: str | None = None


class OKXTestRequest(BaseModel):
    mode: str = "paper"  # "paper" or "live"


class AIModelRequest(BaseModel):
    name: str
    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None
    balance: float | None = None
    execution_mode: str = "paper"  # "paper" or "live"


class AIModelTestRequest(BaseModel):
    name: str | None = None  # look up by name in settings.ai_models
    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None


class ExecutionAccountRequest(BaseModel):
    mode: str = "paper"
    account_name: str | None = None
    allocated_balance: float | None = None
    max_loss_pct: float | None = None
    max_loss_usdt: float | None = None
    cooldown_loss_pct: float | None = None


# ── Helpers ──

def mask_secret(value: str) -> str:
    """Show only last 4 chars of a secret string."""
    if not value:
        return ""
    if len(value) <= 4:
        return "****"
    return "****" + value[-4:]

def _masked(m: dict) -> dict:
    """Return a copy with api_key masked."""
    mc = dict(m)
    mc["api_key"] = mask_secret(mc.get("api_key", ""))
    return mc

def _is_masked_secret(value: str | None) -> bool:
    return bool(value and value.strip().startswith("****"))

async def _get_okx_usdt_snapshot(mode: str, force: bool = False) -> dict:
    """Fetch the real OKX USDT balance snapshot for paper/demo or live mode."""
    mode = "live" if mode == "live" else "paper"
    mode_label = "OKX 实盘账户" if mode == "live" else "OKX 模拟盘账户"
    result = {
        "available_balance": None,
        "used_balance": None,
        "total_balance": None,
        "cash_balance": None,
        "equity_balance": None,
        "allocatable_balance": None,
        "balance_error": None,
        "balance_source": mode_label,
    }

    creds = settings.get_okx_credentials(mode)
    if not creds.get("api_key"):
        result["balance_error"] = "未配置 OKX API Key"
        return result

    cached = _OKX_BALANCE_CACHE.get(mode)
    if cached and not force and time.time() - cached[0] < _OKX_BALANCE_TTL_SECONDS:
        return dict(cached[1])

    from executor.okx_executor import OKXExecutor

    executor = OKXExecutor(mode=mode)
    try:
        await executor.initialize()
        snapshot = await executor.get_balance_snapshot("USDT")
        if snapshot.get("error"):
            result["balance_error"] = str(snapshot.get("error"))
            return result
        result.update({
            "available_balance": float(snapshot.get("free") or 0.0),
            "used_balance": float(snapshot.get("used") or 0.0),
            "total_balance": float(snapshot.get("total") or 0.0),
            "cash_balance": float(snapshot.get("cash") or snapshot.get("total") or 0.0),
            "equity_balance": float(snapshot.get("equity") or snapshot.get("total") or 0.0),
            "allocatable_balance": float(
                snapshot.get("allocatable")
                or snapshot.get("equity")
                or snapshot.get("total")
                or snapshot.get("free")
                or 0.0
            ),
        })
        _OKX_BALANCE_CACHE[mode] = (time.time(), dict(result))
        return result
    except Exception as exc:
        result["balance_error"] = f"OKX 余额查询失败: {exc}"
        return result
    finally:
        await executor.shutdown()


async def _get_okx_usdt_balance(mode: str, force: bool = False) -> float | None:
    """Fetch OKX USDT account equity/balance for allocation validation."""
    snapshot = await _get_okx_usdt_snapshot(mode, force=force)
    if snapshot.get("balance_error"):
        return None
    return snapshot.get("allocatable_balance")


async def _paper_execution_account_summary() -> dict:
    """Return current paper execution account balance without requiring OKX."""
    if _dash._trading_service and _dash._trading_service.paper_executor:
        try:
            return await _dash._trading_service.paper_executor.get_account_summary(ENSEMBLE_TRADER_NAME)
        except Exception:
            pass

    from db.repositories.account_repo import AccountRepository
    from db.session import get_session_ctx

    allocated = settings.get_execution_account_config("paper")["allocated_balance"]
    async with get_session_ctx() as session:
        repo = AccountRepository(session)
        account = await repo.get_account(ENSEMBLE_TRADER_NAME)
        if account:
            return {
                "available_balance": account.current_balance,
                "current_balance": account.current_balance,
                "wallet_balance": account.current_balance,
                "equity": account.current_balance + account.unrealized_pnl,
                "initial_balance": account.initial_balance,
                "used_margin": 0.0,
                "unrealized_pnl": account.unrealized_pnl,
                "total_pnl": account.current_balance + account.unrealized_pnl - account.initial_balance,
                "total_pnl_pct": account.total_pnl_pct * 100,
            }

    return {
        "available_balance": allocated,
        "current_balance": allocated,
        "wallet_balance": allocated,
        "equity": allocated,
        "initial_balance": allocated,
        "used_margin": 0.0,
        "unrealized_pnl": 0.0,
        "total_pnl": 0.0,
        "total_pnl_pct": 0.0,
    }


async def _execution_account_status(mode: str) -> dict:
    mode = "live" if mode == "live" else "paper"
    cfg = settings.get_execution_account_config(mode)
    legacy_allocated = float(cfg.get("allocated_balance") or 0.0)
    max_loss_pct = float(cfg.get("max_loss_pct") or 0.0)
    pnl_summary = await _dash._get_execution_pnl_summary(mode)
    okx_snapshot = await _get_okx_usdt_snapshot(mode)
    okx_available = okx_snapshot.get("available_balance")
    okx_allocatable = okx_snapshot.get("allocatable_balance")
    account_equity = float(
        okx_snapshot.get("equity_balance")
        or okx_snapshot.get("total_balance")
        or okx_allocatable
        or okx_available
        or legacy_allocated
        or 0.0
    )
    max_loss_usdt = account_equity * max_loss_pct if account_equity > 0 and max_loss_pct > 0 else 0.0
    risk_floor = max(account_equity - max_loss_usdt, 0.0) if account_equity > 0 else 0.0
    pause_reason = None
    if _dash._trading_service and _dash.mode_manager.mode.value == mode:
        pause_reason = getattr(_dash._trading_service, "_new_pair_pause_reasons", {}).get(ENSEMBLE_TRADER_NAME)
    if okx_snapshot.get("balance_error") and not pause_reason:
        pause_reason = f"未同步到 {okx_snapshot.get('balance_source')} 的实际余额，暂停分析新的交易对。"
    total_pnl_for_risk = float(pnl_summary.get("total_pnl") or 0.0)
    if account_equity > 0 and max_loss_usdt > 0 and total_pnl_for_risk <= -max_loss_usdt and not pause_reason:
        pause_reason = (
            f"{okx_snapshot.get('balance_source')} AI 执行账户累计盈亏 {total_pnl_for_risk:.2f} USDT "
            f"已达到最高亏损限制 {max_loss_pct * 100:.1f}%（{max_loss_usdt:.2f} USDT），暂停分析新的交易对。"
        )
    if not pause_reason:
        pause_reason = _dash._cooldown_pause_reason_from_summary(
            pnl_summary,
            {**cfg, "max_loss_usdt": max_loss_usdt},
            okx_snapshot.get("balance_source") or "执行账户",
        )
    status = dict(cfg)
    status.update({
        "allocated_balance": account_equity,
        "account_equity": account_equity,
        "max_loss_usdt": max_loss_usdt,
        "available_balance": okx_available,
        "equity": okx_snapshot.get("total_balance"),
        "used_margin": okx_snapshot.get("used_balance"),
        "unrealized_pnl": pnl_summary.get("unrealized_pnl", 0.0),
        "realized_profit": pnl_summary.get("realized_profit", 0.0),
        "realized_loss": pnl_summary.get("realized_loss", 0.0),
        "realized_pnl": pnl_summary.get("realized_pnl", 0.0),
        "today_realized_profit": pnl_summary.get("today_realized_profit", 0.0),
        "today_realized_loss": pnl_summary.get("today_realized_loss", 0.0),
        "today_realized_pnl": pnl_summary.get("today_realized_pnl", 0.0),
        "today_closed_realized_profit": pnl_summary.get("today_closed_realized_profit", 0.0),
        "today_closed_realized_loss": pnl_summary.get("today_closed_realized_loss", 0.0),
        "today_closed_realized_pnl": pnl_summary.get("today_closed_realized_pnl", 0.0),
        "today_equity_pnl": pnl_summary.get("today_equity_pnl", 0.0),
        "today_equity_baseline": pnl_summary.get("today_equity_baseline"),
        "today_equity_baseline_total_pnl": pnl_summary.get("today_equity_baseline_total_pnl"),
        "today_equity_baseline_at": pnl_summary.get("today_equity_baseline_at"),
        "today_equity_baseline_source": pnl_summary.get("today_equity_baseline_source"),
        "today_snapshot_date": pnl_summary.get("today_snapshot_date"),
        "today_total_pnl": pnl_summary.get("today_total_pnl", 0.0),
        "today_risk_pnl": pnl_summary.get("today_risk_pnl", 0.0),
        "cumulative_profit": pnl_summary.get("realized_profit", 0.0),
        "cumulative_loss": pnl_summary.get("realized_loss", 0.0),
        "total_pnl": pnl_summary.get("total_pnl", 0.0),
        "remaining_allocation": okx_available,
        "balance_error": okx_snapshot.get("balance_error"),
        "balance_source": okx_snapshot.get("balance_source"),
        "okx_available_balance": okx_available,
        "okx_total_balance": okx_snapshot.get("total_balance"),
        "okx_cash_balance": okx_snapshot.get("cash_balance"),
        "okx_equity_balance": okx_snapshot.get("equity_balance"),
        "okx_used_balance": okx_snapshot.get("used_balance"),
        "max_allocatable_balance": okx_allocatable if okx_allocatable is not None else 0.0,
        "allocation_exceeds_balance": False,
        "risk_floor": risk_floor,
        "risk_paused": bool(pause_reason),
        "risk_pause_reason": pause_reason,
    })
    if mode == "paper":
        summary = await _paper_execution_account_summary()
        status.update({
            "paper_execution_available_balance": okx_available,
            "paper_execution_equity": summary.get("equity"),
            "paper_execution_used_margin": summary.get("used_margin"),
            "paper_execution_unrealized_pnl": pnl_summary.get("unrealized_pnl"),
            "initial_balance": account_equity,
        })
        return status

    return status


async def _sync_execution_account_to_paper_account() -> None:
    """Apply paper allocation to the unified paper execution account when safe."""
    from db.repositories.account_repo import AccountRepository
    from db.repositories.trade_repo import TradeRepository
    from db.session import get_session_ctx

    allocated = float(settings.get_execution_account_config("paper")["allocated_balance"])
    settings.model_initial_balances[ENSEMBLE_TRADER_NAME] = allocated
    async with get_session_ctx() as session:
        account_repo = AccountRepository(session)
        trade_repo = TradeRepository(session)
        account = await account_repo.get_or_create_account(ENSEMBLE_TRADER_NAME, allocated)
        open_count = await trade_repo.count_positions(model_name=ENSEMBLE_TRADER_NAME, is_open=True)
        account.initial_balance = allocated
        if open_count == 0 and int(account.total_trades or 0) == 0:
            account.current_balance = allocated
            account.realized_pnl = 0.0
            account.unrealized_pnl = 0.0
        await session.flush()
        if _dash._trading_service and _dash._trading_service.paper_executor:
            _dash._trading_service.paper_executor._balances[ENSEMBLE_TRADER_NAME] = account.current_balance


async def _sync_models_to_running_services() -> None:
    """Rebuild models from settings.ai_models and sync to running trading service."""
    if not _dash._trading_service or not _dash._trading_service.models:
        return

    import structlog
    log = structlog.get_logger(__name__)

    from db.repositories.account_repo import AccountRepository
    from db.session import get_session_ctx

    registry = _dash._trading_service.models
    old_names, new_names = await registry.sync_from_config()
    log.info("models synced from config", old=list(old_names), new=list(new_names))

    # Sync paper executor account. Expert models analyze only; execution and PnL
    # belong to the unified ensemble_trader account.
    executor = _dash._trading_service.paper_executor
    if executor:
        async with get_session_ctx() as session:
            repo = AccountRepository(session)
            initial_bal = settings.get_initial_balance(ENSEMBLE_TRADER_NAME)
            account = await repo.get_or_create_account(ENSEMBLE_TRADER_NAME, initial_bal)
            executor._balances.setdefault(ENSEMBLE_TRADER_NAME, account.current_balance)
            executor._positions.setdefault(ENSEMBLE_TRADER_NAME, [])
        executor._model_names = [ENSEMBLE_TRADER_NAME]

    # Update competition service active models and trigger evaluation
    if _dash._competition_service:
        _dash._competition_service.set_active_models([ENSEMBLE_TRADER_NAME])
        rankings = await _dash._competition_service.evaluate_all_models(force=True)
        log.info("rankings updated", count=len(rankings), models=[r["model_name"] for r in rankings])


# ── OKX Settings (split paper / live) ──

@router.get("/settings/okx")
async def get_okx_settings():
    """Return both paper and live OKX config with secrets masked."""
    return {
        "paper": {
            "api_key": mask_secret(settings.okx_paper_api_key or settings.okx_api_key),
            "has_secret": bool(settings.okx_paper_api_secret or settings.okx_api_secret),
            "has_passphrase": bool(settings.okx_paper_passphrase or settings.okx_passphrase),
        },
        "live": {
            "api_key": mask_secret(settings.okx_live_api_key or settings.okx_api_key),
            "has_secret": bool(settings.okx_live_api_secret or settings.okx_api_secret),
            "has_passphrase": bool(settings.okx_live_passphrase or settings.okx_passphrase),
        },
    }


@router.post("/settings/okx")
async def update_okx_settings(req: OKXSettingsRequest):
    """Update OKX credentials for a specific mode (paper or live)."""
    if req.mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")

    prefix = "OKX_PAPER" if req.mode == "paper" else "OKX_LIVE"
    updates = {}
    if req.api_key is not None and req.api_key.strip() and not _is_masked_secret(req.api_key):
        updates[f"{prefix}_API_KEY"] = req.api_key.strip()
        setattr(settings, f"okx_{req.mode}_api_key", req.api_key.strip())
    if req.api_secret is not None and req.api_secret.strip() and not _is_masked_secret(req.api_secret):
        updates[f"{prefix}_API_SECRET"] = req.api_secret.strip()
        setattr(settings, f"okx_{req.mode}_api_secret", req.api_secret.strip())
    if req.passphrase is not None and not _is_masked_secret(req.passphrase):
        updates[f"{prefix}_PASSPHRASE"] = req.passphrase.strip()
        setattr(settings, f"okx_{req.mode}_passphrase", req.passphrase.strip())

    if updates:
        settings.update_env_file(updates)
        _OKX_BALANCE_CACHE.pop(req.mode, None)

    # Reinitialize connections so displayed OKX balances immediately use the
    # latest credentials. Failures do not block saving the credentials.
    current_mode = settings.trading_mode.value
    if req.mode == current_mode:
        if _dash._data_service:
            try:
                await _dash._data_service.rest_client.reinitialize()
            except Exception:
                pass
    if _dash._trading_service:
        from executor.okx_executor import OKXExecutor

        attr = "_okx_live" if req.mode == "live" else "_okx_paper"
        old_executor = getattr(_dash._trading_service, attr, None)
        if old_executor:
            try:
                await old_executor.shutdown()
            except Exception:
                pass
        new_executor = OKXExecutor(mode=req.mode)
        try:
            await new_executor.initialize()
            setattr(_dash._trading_service, attr, new_executor)
        except Exception:
            setattr(_dash._trading_service, attr, None)

    return {
        "status": "ok",
        "message": f"{req.mode} settings saved.",
        "updated_keys": list(updates.keys()),
    }


@router.post("/settings/okx/test")
async def test_okx_connection(req: OKXTestRequest):
    """Test OKX credentials for paper or live mode by fetching balance."""
    if req.mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")

    creds = settings.get_okx_credentials(req.mode)
    if not creds.get("api_key") or not creds.get("api_secret"):
        return {"success": False, "error": "请先配置 API Key 和 API Secret"}
    if not creds.get("passphrase"):
        return {"success": False, "error": "请填写 Passphrase（创建 OKX API Key 时设置的密码短语）"}

    from executor.okx_executor import OKXExecutor

    executor = OKXExecutor(mode=req.mode)
    try:
        await executor.initialize()
        snapshot = await executor.get_balance_snapshot("USDT")
        if snapshot.get("error"):
            return {
                "success": False,
                "error": f"连接失败: {snapshot.get('error')}",
            }
        usdt = float(snapshot.get("free") or 0.0)
        used = float(snapshot.get("used") or 0.0)
        total = float(snapshot.get("total") or 0.0)
        cash = float(snapshot.get("cash") or total or 0.0)
        equity = float(snapshot.get("equity") or total or 0.0)
        allocatable = float(snapshot.get("allocatable") or equity or total or usdt)
        return {
            "success": True,
            "message": (
                f"连接成功 ({req.mode})。USDT 账户余额: {cash:.2f}，"
                f"权益: {equity:.2f}，可交易: {usdt:.2f}，占用: {used:.2f}"
            ),
            "balance_usdt": allocatable,
            "available_balance": usdt,
            "used_balance": used,
            "total_balance": total,
            "cash_balance": cash,
            "equity_balance": equity,
            "allocatable_balance": allocatable,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"连接失败: {e}",
        }
    finally:
        await executor.shutdown()


@router.get("/settings/okx/balance")
async def get_okx_balances():
    """Fetch USDT balance for both paper and live OKX accounts."""
    result: dict = {"paper": None, "live": None, "paper_error": None, "live_error": None}

    for mode in ("paper", "live"):
        creds = settings.get_okx_credentials(mode)
        if not creds.get("api_key"):
            result[f"{mode}_error"] = "未配置API密钥"
            continue
        bal = await _get_okx_usdt_balance(mode)
        if bal is not None:
            result[mode] = bal
        else:
            result[f"{mode}_error"] = "查询失败"

    return result


@router.get("/settings/execution-account")
async def get_execution_account_settings():
    """Return unified execution-account settings and current balances."""
    return {
        "paper": await _execution_account_status("paper"),
        "live": await _execution_account_status("live"),
    }


@router.post("/settings/execution-account")
async def update_execution_account_settings(req: ExecutionAccountRequest):
    """Update execution-account display name and percent-based risk controls."""
    mode = "live" if req.mode == "live" else "paper"
    updates: dict[str, str] = {}

    if req.account_name is not None:
        account_name = req.account_name.strip() or "多专家执行账户"
        settings.execution_account_name = account_name
        updates["EXECUTION_ACCOUNT_NAME"] = account_name

    if req.max_loss_pct is not None:
        if req.max_loss_pct < 0 or req.max_loss_pct > 1:
            raise HTTPException(status_code=400, detail="最高亏损比例必须在 0 到 1 之间")
        settings.execution_account_max_loss_pct[mode] = float(req.max_loss_pct)
        updates["EXECUTION_ACCOUNT_MAX_LOSS_PCT"] = json.dumps(
            settings.execution_account_max_loss_pct,
            ensure_ascii=False,
        )

    if req.cooldown_loss_pct is not None:
        if req.cooldown_loss_pct < 0 or req.cooldown_loss_pct > 1:
            raise HTTPException(status_code=400, detail="冷静期触发比例必须在 0 到 100% 之间")
        settings.execution_account_cooldown_loss_pct[mode] = float(req.cooldown_loss_pct)
        updates["EXECUTION_ACCOUNT_COOLDOWN_LOSS_PCT"] = json.dumps(
            settings.execution_account_cooldown_loss_pct,
            ensure_ascii=False,
        )

    if updates:
        settings.update_env_file(updates)

    return {
        "status": "ok",
        "message": "执行账户设置已保存。",
        "paper": await _execution_account_status("paper"),
        "live": await _execution_account_status("live"),
    }


# ── AI Model CRUD ──

@router.get("/settings/ai-models")
async def get_ai_models():
    """Return fixed expert model slots with api_key masked.

    Keep this endpoint lightweight for the settings page; OKX balance checks are
    intentionally handled by /settings/okx/balance and test endpoints.
    """
    models = []
    for m in settings.get_fixed_ai_models(include_empty=True):
        mc = dict(m)
        mc["api_key"] = mask_secret(mc.get("api_key", ""))
        mc["execution_mode"] = "analysis"
        models.append(mc)

    execution_accounts = {
        "paper": settings.get_execution_account_config("paper"),
        "live": settings.get_execution_account_config("live"),
    }
    okx_info: dict = {
        "paper_balance": None,
        "paper_error": None,
        "live_balance": None,
        "live_error": None,
        "skipped": True,
    }

    return {
        "models": models,
        "legacy": [],
        "execution_model": ENSEMBLE_TRADER_NAME,
        "execution_account": execution_accounts,
        "okx": okx_info,
    }


@router.post("/settings/ai-models")
async def add_ai_model_fixed(req: AIModelRequest):
    """Fixed expert slots cannot be added from the UI."""
    raise HTTPException(status_code=400, detail="模型槽位已固定，请直接编辑页面中的专家模型。")


@router.put("/settings/ai-models/{name}")
async def update_ai_model_fixed(name: str, req: AIModelRequest):
    """Update one fixed expert model slot."""
    try:
        updates = {
            "api_base": req.api_base,
            "model": req.model,
        }
        if req.api_key is not None and req.api_key.strip():
            updates["api_key"] = req.api_key
        updated = settings.set_fixed_ai_model(name, updates)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    settings.update_env_file({"AI_MODELS": json.dumps(settings.ai_models, ensure_ascii=False)})
    await _sync_models_to_running_services()
    return {"status": "ok", "message": f"Model '{name}' updated.", "model": _masked(updated)}


@router.delete("/settings/ai-models/{name}")
async def delete_ai_model_fixed(name: str):
    """Fixed expert slots cannot be deleted."""
    raise HTTPException(status_code=400, detail="固定专家模型不能删除，只能清空 Key 或修改模型配置。")


@router.post("/settings/ai-models")
async def add_ai_model(req: AIModelRequest):
    """Add a new AI model configuration (paper or live)."""
    if not req.name or not req.name.strip():
        raise HTTPException(status_code=400, detail="Model name is required")

    # Check for duplicate name
    for m in settings.ai_models:
        if m.get("name") == req.name.strip():
            raise HTTPException(status_code=400, detail=f"Model '{req.name}' already exists")

    mode = req.execution_mode or "paper"

    # Validate balance against OKX account
    if req.balance is not None and req.balance > 0:
        okx_balance = await _get_okx_usdt_balance(mode)
        if okx_balance is not None:
            same_mode_models = [
                m for m in settings.ai_models
                if m.get("execution_mode", "paper") == mode
            ]
            current_total = sum(
                float(m.get("balance", 0)) for m in same_mode_models
                if isinstance(m.get("balance"), (int, float))
            )
            if current_total + req.balance > okx_balance:
                mode_label = "模拟盘" if mode == "paper" else "实盘"
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"配额总和 ({current_total + req.balance:.2f} USDT) "
                        f"超过 OKX {mode_label}账户余额 ({okx_balance:.2f} USDT)。"
                        f"当前已分配: {current_total:.2f} USDT"
                    ),
                )

    new_model = {
        "name": req.name.strip(),
        "api_base": (req.api_base or "").strip(),
        "api_key": (req.api_key or "").strip(),
        "model": (req.model or "gpt-4").strip(),
        "execution_mode": mode,
    }
    if req.balance is not None:
        new_model["balance"] = req.balance
        settings.model_initial_balances[req.name.strip()] = req.balance

    settings.ai_models.append(new_model)
    env_updates = {"AI_MODELS": json.dumps(settings.ai_models)}
    if req.balance is not None:
        env_updates["MODEL_INITIAL_BALANCES"] = json.dumps(settings.model_initial_balances)
    settings.update_env_file(env_updates)

    await _sync_models_to_running_services()

    return {"status": "ok", "message": f"Model '{req.name}' added.", "model": _masked(new_model)}


@router.put("/settings/ai-models/{name}")
async def update_ai_model(name: str, req: AIModelRequest):
    """Update an existing AI model configuration."""
    for i, m in enumerate(settings.ai_models):
        if m.get("name") == name:
            updated = dict(m)
            if req.name and req.name.strip() and req.name.strip() != name:
                updated["name"] = req.name.strip()
            if req.api_base is not None:
                updated["api_base"] = req.api_base.strip()
            if req.api_key is not None and req.api_key.strip():
                updated["api_key"] = req.api_key.strip()
            if req.model is not None and req.model.strip():
                updated["model"] = req.model.strip()
            if req.execution_mode:
                updated["execution_mode"] = req.execution_mode
            mode = updated.get("execution_mode", "paper")
            if req.balance is not None:
                if req.balance > 0:
                    okx_balance = await _get_okx_usdt_balance(mode)
                    if okx_balance is not None:
                        others_total = sum(
                            float(m2.get("balance", 0)) for m2 in settings.ai_models
                            if isinstance(m2.get("balance"), (int, float))
                            and m2.get("name") != name
                            and m2.get("execution_mode", "paper") == mode
                        )
                        if others_total + req.balance > okx_balance:
                            mode_label = "模拟盘" if mode == "paper" else "实盘"
                            raise HTTPException(
                                status_code=400,
                                detail=(
                                    f"配额总和 ({others_total + req.balance:.2f} USDT) "
                                    f"超过 OKX {mode_label}账户余额 ({okx_balance:.2f} USDT)。"
                                    f"其他模型已分配: {others_total:.2f} USDT"
                                ),
                            )
                updated["balance"] = req.balance
                target_name = updated.get("name", name)
                settings.model_initial_balances[target_name] = req.balance
            settings.ai_models[i] = updated
            env_updates = {"AI_MODELS": json.dumps(settings.ai_models)}
            if req.balance is not None:
                env_updates["MODEL_INITIAL_BALANCES"] = json.dumps(settings.model_initial_balances)
            settings.update_env_file(env_updates)
            await _sync_models_to_running_services()
            return {"status": "ok", "message": f"Model '{name}' updated.", "model": _masked(updated)}

    raise HTTPException(status_code=404, detail=f"Model '{name}' not found")


@router.delete("/settings/ai-models/{name}")
async def delete_ai_model(name: str):
    """Delete an AI model configuration."""
    # Check configured models first
    for i, m in enumerate(settings.ai_models):
        if m.get("name") == name:
            settings.ai_models.pop(i)
            settings.update_env_file({"AI_MODELS": json.dumps(settings.ai_models)})
            await _sync_models_to_running_services()
            return {"status": "ok", "message": f"Model '{name}' deleted."}

    # Legacy fallback: if no models configured and name matches legacy, clear it
    if not settings.ai_models and name == "llm_agent":
        settings.ai_api_key = ""
        settings.ai_api_base = ""
        settings.ai_model = ""
        settings.update_env_file({
            "AI_MODELS": "[]",
            "AI_API_KEY": "",
            "AI_API_BASE": "",
            "AI_MODEL": "",
        })
        await _sync_models_to_running_services()
        return {"status": "ok", "message": f"Legacy model '{name}' cleared."}

    raise HTTPException(status_code=404, detail=f"Model '{name}' not found")


@router.post("/settings/ai-models/test")
async def test_ai_model_connection(req: AIModelTestRequest):
    """Test an AI model's API connection by making a simple ChatOpenAI call."""
    # Resolve config: by name from settings.ai_models, or direct fields, or fallback
    api_base = req.api_base
    api_key = req.api_key
    model = req.model

    if req.name and (not api_base or not api_key or not model):
        for m in settings.get_fixed_ai_models(include_empty=True):
            if m.get("name") == req.name:
                api_base = api_base or m.get("api_base")
                api_key = api_key or m.get("api_key")
                model = model or m.get("model")
                break

    # Fallback to global settings
    api_base = api_base or settings.ai_api_base
    api_key = api_key or settings.ai_api_key
    model = model or settings.ai_model

    if not api_key:
        return {"success": False, "error": "No API key configured"}

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            base_url=api_base,
            api_key=api_key,
            model=model,
            temperature=0,
            max_tokens=10,
            timeout=15,
            max_retries=0,
        )
        response = await llm.ainvoke([HumanMessage(content="Hi")])
        content = response.content if hasattr(response, "content") else str(response)
        return {"success": True, "message": f"Connection OK. Response: {content[:100]}", "model": model}
    except Exception as e:
        return {"success": False, "error": str(e), "model": model}


class IntervalRequest(BaseModel):
    interval_seconds: int


class ThresholdsRequest(BaseModel):
    decision_interval: int | None = None
    confidence_threshold: float | None = None
    total_margin_limit_pct: float | None = None
    local_ai_tools_enabled: bool | None = None
    local_ai_tools_api_base: str | None = None
    local_ai_tools_timeout_seconds: float | None = None
    high_risk_review_enabled: bool | None = None
    high_risk_review_api_base: str | None = None
    high_risk_review_api_key: str | None = None
    high_risk_review_model: str | None = None


@router.get("/settings/thresholds")
async def get_thresholds():
    """Get current decision interval and confidence threshold."""
    total_margin_limit_pct = (
        float(settings.max_total_margin_pct)
        if float(settings.max_total_margin_pct or 0.0) > 0
        else float(settings.max_position_pct or 0.0) * 3
    )
    return {
        "decision_interval": settings.decision_interval_seconds,
        "confidence_threshold": settings.confidence_threshold,
        "local_ai_tools_enabled": settings.local_ai_tools_enabled,
        "local_ai_tools_api_base": settings.local_ai_tools_api_base,
        "local_ai_tools_timeout_seconds": settings.local_ai_tools_timeout_seconds,
        "high_risk_review_enabled": settings.high_risk_review_enabled,
        "high_risk_review_api_base": settings.high_risk_review_api_base,
        "high_risk_review_api_key": mask_secret(settings.high_risk_review_api_key),
        "high_risk_review_has_api_key": bool(settings.high_risk_review_api_key),
        "high_risk_review_model": settings.high_risk_review_model,
        "total_margin_limit_pct": total_margin_limit_pct,
    }


@router.post("/settings/thresholds")
async def update_thresholds(req: ThresholdsRequest):
    """Update decision interval and/or confidence threshold dynamically."""
    updates = {}

    if req.decision_interval is not None:
        if req.decision_interval < 10:
            raise HTTPException(status_code=400, detail="Interval must be at least 10 seconds")
        if req.decision_interval > 3600:
            raise HTTPException(status_code=400, detail="Interval must be at most 3600 seconds")
        settings.decision_interval_seconds = req.decision_interval
        updates["DECISION_INTERVAL_SECONDS"] = str(req.decision_interval)

    if req.confidence_threshold is not None:
        if req.confidence_threshold < 0.1:
            raise HTTPException(status_code=400, detail="Confidence threshold must be at least 0.1")
        if req.confidence_threshold > 1.0:
            raise HTTPException(status_code=400, detail="Confidence threshold must be at most 1.0")
        settings.confidence_threshold = req.confidence_threshold
        updates["CONFIDENCE_THRESHOLD"] = str(req.confidence_threshold)

    if req.local_ai_tools_enabled is not None:
        settings.local_ai_tools_enabled = bool(req.local_ai_tools_enabled)
        updates["LOCAL_AI_TOOLS_ENABLED"] = "true" if settings.local_ai_tools_enabled else "false"

    if req.local_ai_tools_api_base is not None:
        settings.local_ai_tools_api_base = req.local_ai_tools_api_base.strip()
        updates["LOCAL_AI_TOOLS_API_BASE"] = settings.local_ai_tools_api_base

    if req.local_ai_tools_timeout_seconds is not None:
        if req.local_ai_tools_timeout_seconds < 0.2 or req.local_ai_tools_timeout_seconds > 15:
            raise HTTPException(status_code=400, detail="Local AI tools timeout must be between 0.2 and 15 seconds")
        settings.local_ai_tools_timeout_seconds = float(req.local_ai_tools_timeout_seconds)
        updates["LOCAL_AI_TOOLS_TIMEOUT_SECONDS"] = str(settings.local_ai_tools_timeout_seconds)

    if req.high_risk_review_enabled is not None:
        settings.high_risk_review_enabled = bool(req.high_risk_review_enabled)
        updates["HIGH_RISK_REVIEW_ENABLED"] = "true" if settings.high_risk_review_enabled else "false"

    if req.high_risk_review_api_base is not None:
        settings.high_risk_review_api_base = req.high_risk_review_api_base.strip()
        updates["HIGH_RISK_REVIEW_API_BASE"] = settings.high_risk_review_api_base

    if req.high_risk_review_api_key is not None:
        api_key = req.high_risk_review_api_key.strip()
        if api_key and not _is_masked_secret(api_key):
            settings.high_risk_review_api_key = api_key
            updates["HIGH_RISK_REVIEW_API_KEY"] = settings.high_risk_review_api_key

    if req.high_risk_review_model is not None:
        settings.high_risk_review_model = req.high_risk_review_model.strip()
        updates["HIGH_RISK_REVIEW_MODEL"] = settings.high_risk_review_model

    if req.total_margin_limit_pct is not None:
        if req.total_margin_limit_pct < 0.10:
            raise HTTPException(status_code=400, detail="总保证金占用上限不能低于 10%")
        if req.total_margin_limit_pct > 1.0:
            raise HTTPException(status_code=400, detail="总保证金占用上限不能超过 100%")
        settings.max_total_margin_pct = float(req.total_margin_limit_pct)
        updates["MAX_TOTAL_MARGIN_PCT"] = str(settings.max_total_margin_pct)

    if updates:
        settings.update_env_file(updates)

    total_margin_limit_pct = (
        float(settings.max_total_margin_pct)
        if float(settings.max_total_margin_pct or 0.0) > 0
        else float(settings.max_position_pct or 0.0) * 3
    )
    return {
        "status": "ok",
        "message": "Settings updated.",
        "decision_interval": settings.decision_interval_seconds,
        "confidence_threshold": settings.confidence_threshold,
        "local_ai_tools_enabled": settings.local_ai_tools_enabled,
        "local_ai_tools_api_base": settings.local_ai_tools_api_base,
        "local_ai_tools_timeout_seconds": settings.local_ai_tools_timeout_seconds,
        "high_risk_review_enabled": settings.high_risk_review_enabled,
        "high_risk_review_api_base": settings.high_risk_review_api_base,
        "high_risk_review_api_key": mask_secret(settings.high_risk_review_api_key),
        "high_risk_review_has_api_key": bool(settings.high_risk_review_api_key),
        "high_risk_review_model": settings.high_risk_review_model,
        "total_margin_limit_pct": total_margin_limit_pct,
    }


@router.post("/settings/interval")
async def update_decision_interval(req: IntervalRequest):
    """Update the decision interval dynamically (no restart needed)."""
    if req.interval_seconds < 10:
        raise HTTPException(status_code=400, detail="Interval must be at least 10 seconds")
    if req.interval_seconds > 3600:
        raise HTTPException(status_code=400, detail="Interval must be at most 3600 seconds")

    settings.decision_interval_seconds = req.interval_seconds
    settings.update_env_file({"DECISION_INTERVAL_SECONDS": str(req.interval_seconds)})

    return {
        "status": "ok",
        "message": f"Decision interval updated to {req.interval_seconds}s",
        "decision_interval": req.interval_seconds,
    }
