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
ISSUE_ORDER = {"critical": 0, "warning": 1, "ok": 2, "info": 3}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


def _configured_endpoint_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    model_configs = settings.get_fixed_ai_models(include_empty=False)
    configured_by_model = {
        str(item.get("model") or "").strip(): _mask_endpoint(item.get("api_base"))
        for item in model_configs
        if isinstance(item, dict)
    }
    configured_by_model["local_ai_tools"] = _mask_endpoint(settings.local_ai_tools_api_base)
    configured_by_model[str(getattr(settings, "high_risk_review_model", "") or "").strip()] = (
        _mask_endpoint(getattr(settings, "high_risk_review_api_base", ""))
    )
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


def _server_monitor_items(status: dict[str, Any]) -> list[dict[str, Any]]:
    if not status.get("available"):
        return [
            _check_item(
                "server_monitor",
                "模型服务器监控",
                "warning",
                str(status.get("message") or "模型服务器监控暂不可用。"),
                details={"status": status.get("status")},
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

    items = [
        _check_item(
            "recent_execution",
            "最近执行结果",
            "ok" if executed_orders else "warning",
            (
                f"最近 6 小时已有 {len(executed_orders)} 条成交订单。"
                if executed_orders
                else "最近 6 小时没有成交订单，请结合策略漏斗和模型状态继续排查。"
            ),
            details={
                "window_hours": 6,
                "orders": len(orders),
                "filled_orders": len(executed_orders),
                "failed_or_unfilled_orders": len(failed_orders),
            },
        )
    ]
    if failed_orders:
        items.append(
            _check_item(
                "recent_failed_orders",
                "最近失败/未成交订单",
                "warning",
                f"最近 6 小时有 {len(failed_orders)} 条失败或未成交订单，需要打开执行详情查看失败步骤。",
                details={
                    "sample_order_ids": [row.id for row in failed_orders[:5]],
                    "sample_statuses": [row.status for row in failed_orders[:5]],
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
    items.extend(_configured_endpoint_items())
    try:
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
