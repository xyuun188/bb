"""Dashboard self-check and safe auto-repair endpoints."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter
from sqlalchemy import func, select

from config.settings import settings
from core.safe_output import safe_error_text
from core.trading_mode import mode_manager
from db.session import get_session_ctx
from models.decision import AIDecision
from models.market_data import Kline, Ticker
from models.news import NewsArticle, SocialPost
from models.trade import Order
from services.server_monitor_status import (
    clear_server_monitor_cache,
    get_server_monitor_status_async,
)
from web_dashboard.api import dashboard as _dash
from web_dashboard.api.text_sanitize import sanitize_payload

router = APIRouter()

EXPECTED_PLATFORM_ENDPOINTS = {
    "qwen3-14b-trade": "http://127.0.0.1:18000/v1",
    "local_ai_tools": "http://127.0.0.1:18001",
    "deepseek-r1-14b-risk": "http://127.0.0.1:18002/v1",
}
MODEL_ACCESS_ENDPOINTS = {
    "qwen3-14b-trade": "103.85.84.147:21840",
    "local_ai_tools": "103.85.84.147:21841",
    "deepseek-r1-14b-risk": "103.85.84.147:21842",
}
EXPECTED_KLINE_TIMEFRAMES = ("1m", "5m", "15m", "1h")
TICKER_FRESH_SECONDS = 10 * 60
KLINE_FRESH_SECONDS = 2 * 60 * 60
NEWS_FRESH_SECONDS = 24 * 60 * 60
SOCIAL_FRESH_SECONDS = 24 * 60 * 60
ISSUE_ORDER = {"critical": 0, "warning": 1, "ok": 2, "info": 3}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _as_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def _check_item(
    key: str,
    title: str,
    status: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    repairable: bool = False,
    repair_action: str | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "status": status,
        "message": message,
        "details": details or {},
        "repairable": repairable,
        "repair_action": repair_action,
    }


def _overall_status(items: list[dict[str, Any]]) -> str:
    statuses = {str(item.get("status") or "info") for item in items}
    if "critical" in statuses:
        return "critical"
    if "warning" in statuses:
        return "warning"
    return "ok"


def _mask_endpoint(value: Any) -> str:
    return str(value or "").strip()


def _finite_score(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _age_seconds(value: Any) -> float | None:
    dt = _utc_datetime(value)
    if dt is None:
        return None
    return max((datetime.now(UTC) - dt).total_seconds(), 0.0)


def _age_minutes(value: Any) -> float | None:
    age = _age_seconds(value)
    return round(age / 60.0, 1) if age is not None else None


async def _recent_trading_activity_snapshot(hours: int = 2) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(hours=hours)
    async with get_session_ctx() as session:
        decision_stmt = select(
            func.count(AIDecision.id),
            func.max(AIDecision.created_at),
        ).where(AIDecision.created_at >= since)
        order_stmt = select(
            func.count(Order.id),
            func.max(Order.created_at),
        ).where(Order.created_at >= since)
        decision_row = (await session.execute(decision_stmt)).one()
        order_row = (await session.execute(order_stmt)).one()
    return {
        "decision_count": int(decision_row[0] or 0),
        "latest_decision_at": decision_row[1].isoformat() if decision_row[1] else None,
        "order_count": int(order_row[0] or 0),
        "latest_order_at": order_row[1].isoformat() if order_row[1] else None,
        "window_hours": hours,
    }


async def _trading_service_running_item() -> dict[str, Any]:
    running = _dash._trading_service_is_running()
    paused = mode_manager.is_paused
    if not _dash._trading_service:
        settings.refresh_runtime_env(force=True)
        runtime_status = _dash._load_trading_runtime_status()
        runtime_age = runtime_status.get("heartbeat_age_seconds")
        decision_interval = int(
            runtime_status.get("decision_interval") or settings.decision_interval_seconds
        )
        heartbeat_fresh_limit = max(float(decision_interval) * 4, 180.0)
        if runtime_age is not None and float(runtime_age) <= heartbeat_fresh_limit:
            runtime_paused = bool(runtime_status.get("paused", False))
            runtime_running = bool(runtime_status.get("running", True))
            last_round_started = runtime_status.get("last_round_started_at")
            last_round_finished = runtime_status.get("last_round_finished_at")
            round_active = bool(runtime_status.get("round_active", False))
            round_running_seconds = 0.0
            started_at = _as_utc_datetime(last_round_started)
            finished_at = _as_utc_datetime(last_round_finished)
            if started_at is not None and (finished_at is None or finished_at < started_at):
                round_active = True
                round_running_seconds = max((datetime.now(UTC) - started_at).total_seconds(), 0.0)
            market_started = _as_utc_datetime(runtime_status.get("last_market_round_started_at"))
            market_finished = _as_utc_datetime(runtime_status.get("last_market_round_finished_at"))
            market_round_active = bool(runtime_status.get("market_round_active", False)) or (
                market_started is not None
                and (market_finished is None or market_finished < market_started)
            )
            market_round_running_seconds = (
                max((datetime.now(UTC) - market_started).total_seconds(), 0.0)
                if market_round_active and market_started is not None
                else 0.0
            )
            market_stuck_limit = max(
                float(runtime_status.get("market_analysis_watchdog_seconds") or 0.0),
                float(decision_interval) * 3.0,
                60.0,
            )
            market_round_stuck = (
                market_round_active and market_round_running_seconds >= market_stuck_limit
            )
            position_started = _as_utc_datetime(
                runtime_status.get("last_position_round_started_at")
            )
            position_finished = _as_utc_datetime(
                runtime_status.get("last_position_round_finished_at")
            )
            position_round_active = bool(runtime_status.get("position_round_active", False)) or (
                position_started is not None
                and (position_finished is None or position_finished < position_started)
            )
            position_round_running_seconds = (
                max((datetime.now(UTC) - position_started).total_seconds(), 0.0)
                if position_round_active and position_started is not None
                else 0.0
            )
            position_stuck_limit = max(
                float(runtime_status.get("position_analysis_watchdog_seconds") or 0.0),
                float(decision_interval) * 3.0,
                60.0,
            )
            position_round_stuck = (
                position_round_active and position_round_running_seconds >= position_stuck_limit
            )
            active_scope_limits = []
            if market_round_active:
                active_scope_limits.append(market_stuck_limit)
            if position_round_active:
                active_scope_limits.append(position_stuck_limit)
            round_stuck_limit = max(
                *(active_scope_limits or [0.0]),
                float(decision_interval) * 2.5,
                90.0,
            )
            round_stuck = round_active and round_running_seconds >= round_stuck_limit
            last_round_error = str(
                runtime_status.get("last_round_error")
                or runtime_status.get("market_last_error")
                or runtime_status.get("position_last_error")
                or ""
            ).strip()
            runtime_error = bool(last_round_error)
            status = (
                "warning"
                if (
                    runtime_paused
                    or not runtime_running
                    or round_stuck
                    or market_round_stuck
                    or position_round_stuck
                    or runtime_error
                )
                else "ok"
            )
            message = (
                "独立交易进程心跳正常，但当前处于暂停状态；不会分析新的交易对。"
                if runtime_paused
                else (
                    "独立交易进程心跳正常，但市场分析轮次耗时过长；请查看 market_current_stage 定位卡住步骤。"
                    if market_round_stuck
                    else (
                        "独立交易进程心跳正常，但持仓分析轮次耗时过长；请查看 position_current_stage 定位卡住步骤。"
                        if position_round_stuck
                        else (
                            "独立交易进程心跳正常，但本轮分析耗时过长；请查看 current_stage 定位卡住步骤。"
                            if round_stuck
                            else (
                                "独立交易进程心跳正常，但上一轮分析异常结束；请查看 last_round_error。"
                                if runtime_error
                                else "Dashboard 与交易引擎分离运行；独立交易进程心跳正常。"
                            )
                        )
                    )
                )
            )
            return _check_item(
                "trading_service",
                "交易主循环",
                status,
                message,
                details={
                    "source": "runtime_heartbeat",
                    "mode": runtime_status.get("mode"),
                    "running": runtime_running,
                    "paused": runtime_paused,
                    "current_stage": runtime_status.get("current_stage"),
                    "market_current_stage": runtime_status.get("market_current_stage"),
                    "position_current_stage": runtime_status.get("position_current_stage"),
                    "round_active": round_active,
                    "round_running_seconds": round(round_running_seconds, 3),
                    "round_stuck_limit_seconds": round_stuck_limit,
                    "round_stuck": round_stuck,
                    "market_configured_symbol_limit": runtime_status.get(
                        "market_configured_symbol_limit",
                        runtime_status.get("market_batch_symbol_limit"),
                    ),
                    "market_configured_symbol_limit_is_batch_size": runtime_status.get(
                        "market_configured_symbol_limit_is_batch_size", False
                    ),
                    "market_batch_policy": runtime_status.get("market_batch_policy"),
                    "decision_interval": runtime_status.get("decision_interval"),
                    "market_loop_interval_seconds": runtime_status.get(
                        "market_loop_interval_seconds"
                    ),
                    "position_loop_interval_seconds": runtime_status.get(
                        "position_loop_interval_seconds"
                    ),
                    "market_round_time_budget_seconds": runtime_status.get(
                        "market_round_time_budget_seconds"
                    ),
                    "market_analysis_watchdog_seconds": runtime_status.get(
                        "market_analysis_watchdog_seconds"
                    ),
                    "position_analysis_watchdog_seconds": runtime_status.get(
                        "position_analysis_watchdog_seconds"
                    ),
                    "market_round_active": market_round_active,
                    "market_round_running_seconds": round(market_round_running_seconds, 3),
                    "market_round_stuck_limit_seconds": market_stuck_limit,
                    "market_round_stuck": market_round_stuck,
                    "position_round_active": position_round_active,
                    "position_round_running_seconds": round(
                        position_round_running_seconds,
                        3,
                    ),
                    "position_round_stuck_limit_seconds": position_stuck_limit,
                    "position_round_stuck": position_round_stuck,
                    "decision_interval": decision_interval,
                    "heartbeat_age_seconds": round(float(runtime_age), 3),
                    "heartbeat_fresh_limit_seconds": heartbeat_fresh_limit,
                    "last_round_started_at": last_round_started,
                    "last_round_finished_at": last_round_finished,
                    "last_market_round_started_at": runtime_status.get(
                        "last_market_round_started_at"
                    ),
                    "last_market_round_finished_at": runtime_status.get(
                        "last_market_round_finished_at"
                    ),
                    "last_position_round_started_at": runtime_status.get(
                        "last_position_round_started_at"
                    ),
                    "last_position_round_finished_at": runtime_status.get(
                        "last_position_round_finished_at"
                    ),
                    "market_last_error": runtime_status.get("market_last_error"),
                    "position_last_error": runtime_status.get("position_last_error"),
                    "runtime_error": runtime_error,
                    "last_round_error": last_round_error,
                },
                repairable=False,
            )
        try:
            activity = await _recent_trading_activity_snapshot()
        except Exception as exc:
            return _check_item(
                "trading_service",
                "交易主循环",
                "warning",
                "Dashboard 与交易引擎分离运行，且交易心跳查询失败；请结合服务状态继续检查。",
                details={"error": safe_error_text(exc, limit=180)},
                repairable=False,
            )
        if activity.get("decision_count", 0) or activity.get("order_count", 0):
            return _check_item(
                "trading_service",
                "交易主循环",
                "ok",
                "Dashboard 与交易引擎分离运行；最近仍有分析或成交心跳，交易服务在独立进程中工作。",
                details=activity,
                repairable=False,
            )
        return _check_item(
            "trading_service",
            "交易主循环",
            "critical",
            "Dashboard 未直连交易对象，且最近没有分析/成交心跳；交易主循环可能已经停止。",
            details=activity,
            repairable=False,
        )
    if not running:
        activity = {}
        try:
            activity = await _recent_trading_activity_snapshot()
        except Exception:
            activity = {}
        if activity.get("decision_count", 0) or activity.get("order_count", 0):
            return _check_item(
                "trading_service",
                "交易主循环",
                "warning",
                "交易引擎对象未挂载到 Dashboard 进程，但近期仍有分析或成交心跳，说明线上交易服务在独立进程中运行。",
                details=activity,
                repairable=False,
            )
        return _check_item(
            "trading_service",
            "交易主循环",
            "critical",
            "交易服务未运行，系统不会自动分析和开平仓。",
            repairable=False,
        )
    if paused:
        return _check_item(
            "trading_service",
            "交易主循环",
            "warning",
            "交易服务运行中，但当前处于暂停状态；不会分析新的交易对。",
            details={"paused": True, "mode": mode_manager.mode.value},
        )
    return _check_item(
        "trading_service",
        "交易主循环",
        "ok",
        "交易服务运行中，自动扫描处于可工作状态。",
        details={"mode": mode_manager.mode.value, "scan_mode": mode_manager.scan_mode},
    )


def _okx_config_item(mode: str) -> dict[str, Any]:
    creds = settings.get_okx_credentials(mode)
    missing = [
        label
        for key, label in {
            "api_key": "API Key",
            "api_secret": "API Secret",
            "passphrase": "Passphrase",
        }.items()
        if not str(creds.get(key) or "").strip()
    ]
    title = "实盘 OKX 配置" if mode == "live" else "模拟盘 OKX 配置"
    if missing:
        return _check_item(
            f"okx_{mode}",
            title,
            "critical" if mode_manager.mode.value == mode else "warning",
            f"{title}不完整，缺少：{'、'.join(missing)}。",
            details={"mode": mode, "missing_fields": missing, "settings_tab": "okx"},
        )
    return _check_item(
        f"okx_{mode}",
        title,
        "ok",
        f"{title}已配置完整。",
        details={"mode": mode},
    )


def _configured_endpoint_items(
    monitor_status: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    runtime = (
        monitor_status.get("platform_runtime")
        if isinstance(monitor_status, dict)
        and isinstance(monitor_status.get("platform_runtime"), dict)
        else {}
    )
    model_configs = settings.get_fixed_ai_models(include_empty=False)
    configured_by_model = {
        str(item.get("model") or "").strip(): _mask_endpoint(item.get("api_base"))
        for item in model_configs
        if isinstance(item, dict)
    }
    runtime_models = runtime.get("ai_models") if isinstance(runtime.get("ai_models"), list) else []
    for item in runtime_models:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or "").strip()
        api_base = _mask_endpoint(item.get("api_base"))
        if model and api_base:
            configured_by_model[model] = api_base
    configured_by_model["local_ai_tools"] = _mask_endpoint(settings.local_ai_tools_api_base)
    runtime_local_tools = (
        runtime.get("local_ai_tools") if isinstance(runtime.get("local_ai_tools"), dict) else {}
    )
    if runtime_local_tools.get("api_base"):
        configured_by_model["local_ai_tools"] = _mask_endpoint(runtime_local_tools.get("api_base"))
    high_risk_model = str(getattr(settings, "high_risk_review_model", "") or "").strip()
    high_risk_base = _mask_endpoint(getattr(settings, "high_risk_review_api_base", ""))
    if high_risk_model and high_risk_base:
        configured_by_model[high_risk_model] = high_risk_base
    for model, expected in EXPECTED_PLATFORM_ENDPOINTS.items():
        actual = configured_by_model.get(model, "")
        if not actual:
            items.append(
                _check_item(
                    f"endpoint_{model}",
                    f"{model} 平台调用地址",
                    "critical",
                    f"{model} 未配置平台调用地址。",
                    details={
                        "expected_platform_endpoint": expected,
                        "expected_public_endpoint": MODEL_ACCESS_ENDPOINTS.get(model),
                    },
                )
            )
            continue
        if actual.rstrip("/") != expected.rstrip("/"):
            items.append(
                _check_item(
                    f"endpoint_{model}",
                    f"{model} 平台调用地址",
                    "critical",
                    f"{model} 调用地址不符合部署契约，应使用 {expected}。",
                    details={
                        "actual": actual,
                        "expected_platform_endpoint": expected,
                        "expected_public_endpoint": MODEL_ACCESS_ENDPOINTS.get(model),
                    },
                )
            )
        else:
            items.append(
                _check_item(
                    f"endpoint_{model}",
                    f"{model} 平台调用地址",
                    "ok",
                    f"{model} 调用地址符合 18000/18001/18002 隧道契约。",
                    details={
                        "actual": actual,
                        "public_endpoint": MODEL_ACCESS_ENDPOINTS.get(model),
                    },
                )
            )
    return items


async def _data_source_items() -> list[dict[str, Any]]:
    async with get_session_ctx() as session:
        ticker_row = (
            await session.execute(
                select(
                    func.count(Ticker.id),
                    func.max(func.coalesce(Ticker.updated_at, Ticker.created_at)),
                )
            )
        ).one()
        kline_rows = list(
            (
                await session.execute(
                    select(
                        Kline.timeframe,
                        func.count(Kline.id),
                        func.count(func.distinct(Kline.symbol)),
                        func.max(Kline.open_time),
                    )
                    .where(Kline.timeframe.in_(EXPECTED_KLINE_TIMEFRAMES))
                    .group_by(Kline.timeframe)
                )
            ).all()
        )
        news_row = (
            await session.execute(
                select(
                    func.count(NewsArticle.id),
                    func.count(func.distinct(NewsArticle.source)),
                    func.max(func.coalesce(NewsArticle.published_at, NewsArticle.fetched_at)),
                )
            )
        ).one()
        social_row = (
            await session.execute(
                select(
                    func.count(SocialPost.id),
                    func.count(func.distinct(SocialPost.platform)),
                    func.max(SocialPost.posted_at),
                )
            )
        ).one()

    ticker_count = int(ticker_row[0] or 0)
    ticker_latest = _utc_datetime(ticker_row[1])
    ticker_age = _age_seconds(ticker_latest)
    kline_by_timeframe = {
        str(timeframe): {
            "rows": int(row_count or 0),
            "symbols": int(symbol_count or 0),
            "latest_open_time": _utc_datetime(latest_open_time),
            "age_minutes": _age_minutes(latest_open_time),
        }
        for timeframe, row_count, symbol_count, latest_open_time in kline_rows
    }
    missing_timeframes = [
        timeframe
        for timeframe in EXPECTED_KLINE_TIMEFRAMES
        if not kline_by_timeframe.get(timeframe, {}).get("rows")
    ]
    stale_timeframes = [
        timeframe
        for timeframe, row in kline_by_timeframe.items()
        if (_age_seconds(row.get("latest_open_time")) or 0) > KLINE_FRESH_SECONDS
    ]
    news_count = int(news_row[0] or 0)
    news_source_count = int(news_row[1] or 0)
    news_latest = _utc_datetime(news_row[2])
    news_age = _age_seconds(news_latest)
    social_count = int(social_row[0] or 0)
    social_platform_count = int(social_row[1] or 0)
    social_latest = _utc_datetime(social_row[2])
    social_age = _age_seconds(social_latest)

    items: list[dict[str, Any]] = []
    ticker_ok = bool(ticker_count and ticker_age is not None and ticker_age <= TICKER_FRESH_SECONDS)
    items.append(
        _check_item(
            "market_ticker_freshness",
            "实时行情数据源",
            "ok" if ticker_ok else "critical",
            (
                f"已沉淀 {ticker_count} 个 ticker，最新更新时间约 {_age_minutes(ticker_latest)} 分钟前。"
                if ticker_ok
                else "实时行情 ticker 不新鲜或为空，会导致余额、持仓估值和开仓判断失真。"
            ),
            details={
                "ticker_count": ticker_count,
                "latest_at": ticker_latest.isoformat() if ticker_latest else None,
                "age_minutes": _age_minutes(ticker_latest),
                "fresh_limit_minutes": round(TICKER_FRESH_SECONDS / 60, 1),
            },
        )
    )
    kline_ok = not missing_timeframes and not stale_timeframes
    items.append(
        _check_item(
            "market_kline_coverage",
            "分钟级 K 线覆盖",
            "ok" if kline_ok else "warning",
            (
                "1m/5m/15m/1h K 线均有沉淀，短线训练和复盘具备基础行情上下文。"
                if kline_ok
                else "分钟级 K 线存在缺口或过旧，短线开仓/平仓学习可能偏保守或失真。"
            ),
            details={
                "expected_timeframes": list(EXPECTED_KLINE_TIMEFRAMES),
                "missing_timeframes": missing_timeframes,
                "stale_timeframes": stale_timeframes,
                "timeframes": {
                    key: {
                        **{k: v for k, v in value.items() if k != "latest_open_time"},
                        "latest_open_time": (
                            value["latest_open_time"].isoformat()
                            if value.get("latest_open_time")
                            else None
                        ),
                    }
                    for key, value in kline_by_timeframe.items()
                },
            },
        )
    )
    news_ok = bool(news_count and news_age is not None and news_age <= NEWS_FRESH_SECONDS)
    news_diverse = news_source_count >= 2
    items.append(
        _check_item(
            "news_source_freshness",
            "新闻训练数据源",
            "ok" if news_ok and news_diverse else "warning",
            (
                f"新闻源 {news_source_count} 类、样本 {news_count} 条，最新约 {_age_minutes(news_latest)} 分钟前。"
                if news_ok
                else "新闻样本为空或过旧，情绪/事件模型会更依赖行情与历史样本。"
            ),
            details={
                "news_count": news_count,
                "source_count": news_source_count,
                "latest_at": news_latest.isoformat() if news_latest else None,
                "age_minutes": _age_minutes(news_latest),
                "fresh_limit_hours": round(NEWS_FRESH_SECONDS / 3600, 1),
                "source_diversity_ok": news_diverse,
            },
        )
    )
    social_ok = bool(social_count and social_age is not None and social_age <= SOCIAL_FRESH_SECONDS)
    social_diverse = social_platform_count >= 2
    items.append(
        _check_item(
            "social_source_freshness",
            "社媒训练数据源",
            "ok" if social_ok and social_diverse else "warning",
            (
                f"社媒平台 {social_platform_count} 类、样本 {social_count} 条，最新约 {_age_minutes(social_latest)} 分钟前。"
                if social_ok
                else "社媒样本为空、过旧或平台过少，情绪模型存在来源偏置风险。"
            ),
            details={
                "social_count": social_count,
                "platform_count": social_platform_count,
                "latest_at": social_latest.isoformat() if social_latest else None,
                "age_minutes": _age_minutes(social_latest),
                "fresh_limit_hours": round(SOCIAL_FRESH_SECONDS / 3600, 1),
                "platform_diversity_ok": social_diverse,
            },
        )
    )
    return items


def _server_monitor_items(status: dict[str, Any]) -> list[dict[str, Any]]:
    if not status.get("available"):
        monitor_status = str(status.get("status") or "")
        if monitor_status == "server_monitor_refreshing":
            return [
                _check_item(
                    "server_monitor",
                    "模型服务器监控",
                    "info",
                    "模型服务器监控正在刷新，本次不计入异常或需关注。",
                    details={"status": monitor_status},
                )
            ]
        return [
            _check_item(
                "server_monitor",
                "模型服务器监控",
                "warning",
                str(status.get("message") or "模型服务器监控暂不可用。"),
                details={"status": monitor_status},
                repairable=True,
                repair_action="clear_monitor_cache",
            )
        ]
    runtime = (
        status.get("platform_runtime") if isinstance(status.get("platform_runtime"), dict) else {}
    )
    models = runtime.get("ai_models") if isinstance(runtime.get("ai_models"), list) else []
    local_tools = (
        runtime.get("local_ai_tools") if isinstance(runtime.get("local_ai_tools"), dict) else {}
    )
    items: list[dict[str, Any]] = [
        _check_item(
            "server_monitor",
            "模型服务器监控",
            "ok",
            "模型服务器监控已返回状态。",
            details={"checked_at": status.get("checked_at")},
            repairable=True,
            repair_action="clear_monitor_cache",
        )
    ]
    for row in models:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model") or row.get("name") or "模型")
        ok = bool(row.get("available"))
        endpoint_ok = bool(row.get("endpoint_ok"))
        model_ok = bool(row.get("model_available"))
        items.append(
            _check_item(
                f"runtime_model_{model}",
                f"{model} 运行状态",
                "ok" if ok else "critical",
                "端点和模型名均正常。" if ok else "端点或模型名未通过运行时检查。",
                details={
                    "api_base": row.get("api_base"),
                    "endpoint_ok": endpoint_ok,
                    "model_available": model_ok,
                    "status_code": row.get("status_code"),
                    "latency_ms": row.get("latency_ms"),
                    "error": row.get("error"),
                },
            )
        )
    if local_tools:
        available = bool(local_tools.get("available"))
        items.append(
            _check_item(
                "runtime_local_ai_tools",
                "本地量化工具运行状态",
                "ok" if available else "critical",
                "本地量化工具接口可用。" if available else "本地量化工具接口不可用或密钥不一致。",
                details={
                    "api_base": local_tools.get("api_base"),
                    "health": local_tools.get("health"),
                    "status": local_tools.get("status"),
                    "child_endpoints": local_tools.get("child_endpoints"),
                },
                repairable=True,
                repair_action="reset_local_ai_tools_breaker",
            )
        )
    return items


async def _recent_execution_items() -> list[dict[str, Any]]:
    since = datetime.now(UTC) - timedelta(hours=6)
    async with get_session_ctx() as session:
        orders_result = await session.execute(
            select(Order)
            .where(Order.created_at >= since)
            .order_by(Order.created_at.desc())
            .limit(80)
        )
        orders = list(orders_result.scalars().all())
        decisions_result = await session.execute(
            select(AIDecision)
            .where(AIDecision.created_at >= since)
            .order_by(AIDecision.created_at.desc())
            .limit(80)
        )
        decisions = list(decisions_result.scalars().all())

    failed_orders = [row for row in orders if str(row.status or "").lower() != "filled"]
    executed_orders = [row for row in orders if str(row.status or "").lower() == "filled"]
    hard_gate_decisions = []
    missing_opportunity_score_decisions = []
    missing_stage_decisions = 0
    traced_decisions = 0
    latest_missing_score_at: datetime | None = None
    latest_scored_entry_at: datetime | None = None
    for decision in decisions:
        raw = decision.raw_llm_response if isinstance(decision.raw_llm_response, dict) else {}
        action = str(getattr(decision, "action", "") or "").lower()
        opportunity = raw.get("opportunity_score") if isinstance(raw, dict) else {}
        score = opportunity.get("score") if isinstance(opportunity, dict) else None
        if action in {"long", "short", "open_long", "open_short"}:
            created_at = getattr(decision, "created_at", None)
            if _finite_score(score):
                if isinstance(created_at, datetime) and (
                    latest_scored_entry_at is None or created_at > latest_scored_entry_at
                ):
                    latest_scored_entry_at = created_at
            else:
                missing_opportunity_score_decisions.append(decision)
                if isinstance(created_at, datetime) and (
                    latest_missing_score_at is None or created_at > latest_missing_score_at
                ):
                    latest_missing_score_at = created_at
        machine = raw.get("decision_state_machine") if isinstance(raw, dict) else {}
        stages = machine.get("stages") if isinstance(machine, dict) else []
        if not isinstance(stages, list) or not stages:
            missing_stage_decisions += 1
            continue
        traced_decisions += 1
        for stage in stages:
            if isinstance(stage, dict) and stage.get("status") in {"blocked", "failed"}:
                hard_gate_decisions.append(decision)
                break

    execution_status = "ok" if executed_orders else "info"
    execution_message = (
        f"最近 6 小时已有 {len(executed_orders)} 条成交订单。"
        if executed_orders
        else "最近 6 小时没有成交订单；这只表示策略没有提交/成交订单，不代表交易主循环异常。"
    )
    items = [
        _check_item(
            "recent_execution",
            "最近执行结果",
            execution_status,
            execution_message,
            details={
                "window_hours": 6,
                "orders": len(orders),
                "filled_orders": len(executed_orders),
                "failed_or_unfilled_orders": len(failed_orders),
                "is_system_failure": False,
            },
        )
    ]
    if failed_orders:
        unresolved_statuses = {"open", "pending", "partial", "partially_filled"}
        has_unresolved_order = any(
            str(row.status or "").lower() in unresolved_statuses for row in failed_orders
        )
        latest_failed_at = max(
            (row.created_at for row in failed_orders if isinstance(row.created_at, datetime)),
            default=None,
        )
        latest_executed_at = max(
            (row.created_at for row in executed_orders if isinstance(row.created_at, datetime)),
            default=None,
        )
        recovered_after_failure = bool(
            latest_executed_at and latest_failed_at and latest_executed_at >= latest_failed_at
        )
        failed_status = "warning" if has_unresolved_order or not recovered_after_failure else "info"
        failed_message = (
            f"最近 6 小时有 {len(failed_orders)} 条失败或未成交订单，需要打开执行详情查看失败步骤。"
            if failed_status == "warning"
            else f"最近 6 小时有 {len(failed_orders)} 条历史失败订单，但之后已有更新成交，执行链路已恢复；保留详情供复盘，不计入总体异常。"
        )
        items.append(
            _check_item(
                "recent_failed_orders",
                "最近失败/未成交订单",
                failed_status,
                failed_message,
                details={
                    "sample_order_ids": [row.id for row in failed_orders[:5]],
                    "sample_statuses": [row.status for row in failed_orders[:5]],
                    "has_unresolved_order": has_unresolved_order,
                    "latest_failed_at": latest_failed_at.isoformat() if latest_failed_at else None,
                    "latest_executed_at": (
                        latest_executed_at.isoformat() if latest_executed_at else None
                    ),
                },
            )
        )
    if missing_stage_decisions:
        trace_status = "warning" if traced_decisions == 0 else "info"
        items.append(
            _check_item(
                "decision_trace_coverage",
                "执行步骤覆盖率",
                trace_status,
                (
                    f"最近 6 小时有 {missing_stage_decisions} 条旧/异常决策缺少执行步骤链。"
                    if trace_status == "warning"
                    else f"历史旧记录仍有 {missing_stage_decisions} 条缺少执行步骤链，但新记录已包含步骤链。"
                ),
                details={
                    "missing_stage_decisions": missing_stage_decisions,
                    "traced_decisions": traced_decisions,
                },
            )
        )
    if hard_gate_decisions:
        items.append(
            _check_item(
                "recent_blocked_decisions",
                "最近拦截/失败决策",
                "info",
                f"\u6700\u8fd1 6 \u5c0f\u65f6\u6709 {len(hard_gate_decisions)} \u6761\u51b3\u7b56\u5361\u5728\u62e6\u622a\u6216\u5931\u8d25\u72b6\u6001\uff0c\u53ef\u6253\u5f00\u6267\u884c\u8be6\u60c5\u7ee7\u7eed\u6392\u67e5\u539f\u56e0\u3002",
                details={"sample_decision_ids": [row.id for row in hard_gate_decisions[:5]]},
            )
        )
    if missing_opportunity_score_decisions:
        has_newer_scored_entry = bool(
            latest_scored_entry_at
            and latest_missing_score_at
            and latest_scored_entry_at >= latest_missing_score_at
        )
        if has_newer_scored_entry:
            status = "warning"
            message = f"历史旧记录仍有 {len(missing_opportunity_score_decisions)} 条缺评分，但最新开仓决策已补齐评分。"
        else:
            status = "critical"
            message = (
                f"最近 6 小时有 {len(missing_opportunity_score_decisions)} 条开仓决策缺少或无效机会评分，"
                "且尚未看到更新的有效评分开仓记录，说明评分契约或执行入口仍可能断链。"
            )
        items.append(
            _check_item(
                "entry_opportunity_score_coverage",
                "开仓机会评分覆盖率",
                status,
                message,
                details={
                    "sample_decision_ids": [
                        row.id for row in missing_opportunity_score_decisions[:5]
                    ],
                    "latest_missing_score_at": (
                        latest_missing_score_at.isoformat() if latest_missing_score_at else None
                    ),
                    "latest_scored_entry_at": (
                        latest_scored_entry_at.isoformat() if latest_scored_entry_at else None
                    ),
                },
            )
        )
    return items


@router.get("/system/self-check")
async def system_self_check() -> dict[str, Any]:
    items: list[dict[str, Any]] = [await _trading_service_running_item()]
    items.extend([_okx_config_item("paper"), _okx_config_item("live")])
    monitor_status: dict[str, Any] | None = None
    try:
        monitor_status = await get_server_monitor_status_async()
    except Exception:
        monitor_status = None
    try:
        endpoint_items = _configured_endpoint_items(monitor_status)
    except TypeError:
        endpoint_items = _configured_endpoint_items()
    items.extend(endpoint_items)
    try:
        items.extend(await _data_source_items())
    except Exception as exc:
        items.append(
            _check_item(
                "training_data_sources",
                "训练数据源覆盖",
                "warning",
                "训练数据源自检失败，请检查行情、K 线、新闻和社媒数据表。",
                details={"error": safe_error_text(exc, limit=180)},
            )
        )
    try:
        if monitor_status is None:
            monitor_status = await get_server_monitor_status_async()
        items.extend(_server_monitor_items(monitor_status))
    except Exception as exc:
        items.append(
            _check_item(
                "server_monitor",
                "模型服务器监控",
                "warning",
                "模型服务器监控采集失败。",
                details={"error": safe_error_text(exc, limit=180)},
                repairable=True,
                repair_action="clear_monitor_cache",
            )
        )
    try:
        items.extend(await _recent_execution_items())
    except Exception as exc:
        items.append(
            _check_item(
                "recent_execution",
                "最近执行结果",
                "warning",
                "最近执行记录检查失败。",
                details={"error": safe_error_text(exc, limit=180)},
            )
        )

    items = sorted(
        items, key=lambda item: (ISSUE_ORDER.get(str(item.get("status")), 9), item["key"])
    )
    status = _overall_status(items)
    return sanitize_payload(
        {
            "status": status,
            "status_label": {"ok": "正常", "warning": "需关注", "critical": "异常"}.get(
                status, status
            ),
            "checked_at": _now_iso(),
            "summary": {
                "total": len(items),
                "critical": sum(1 for item in items if item.get("status") == "critical"),
                "warning": sum(1 for item in items if item.get("status") == "warning"),
                "ok": sum(1 for item in items if item.get("status") == "ok"),
                "info": sum(1 for item in items if item.get("status") == "info"),
            },
            "items": items,
        }
    )


@router.post("/system/self-check/repair")
async def system_self_check_repair() -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    try:
        clear_server_monitor_cache()
        actions.append(
            {
                "action": "clear_monitor_cache",
                "status": "ok",
                "message": "已清理服务器监控缓存，下一次自检会重新采集模型状态。",
            }
        )
    except Exception as exc:
        actions.append(
            {
                "action": "clear_monitor_cache",
                "status": "failed",
                "message": safe_error_text(exc, limit=180),
            }
        )

    local_client = _dash._dashboard_local_ai_tools_client()
    if local_client is not None:
        try:
            if hasattr(local_client, "_status_cache"):
                local_client._status_cache = None
            if hasattr(local_client, "_failure_count"):
                local_client._failure_count = 0
            if hasattr(local_client, "_circuit_open_until"):
                local_client._circuit_open_until = None
            actions.append(
                {
                    "action": "reset_local_ai_tools_breaker",
                    "status": "ok",
                    "message": "已清理本地量化工具状态缓存和熔断计数。",
                }
            )
        except Exception as exc:
            actions.append(
                {
                    "action": "reset_local_ai_tools_breaker",
                    "status": "failed",
                    "message": safe_error_text(exc, limit=180),
                }
            )
    else:
        actions.append(
            {
                "action": "reset_local_ai_tools_breaker",
                "status": "skipped",
                "message": "当前进程没有可操作的本地量化工具客户端。",
            }
        )

    return sanitize_payload(
        {
            "status": "ok" if all(a["status"] != "failed" for a in actions) else "partial",
            "repaired_at": _now_iso(),
            "actions": actions,
            "safety_note": "自检修复只执行低风险动作；密钥、账户资金、订单和平仓不会自动修改。",
        }
    )
