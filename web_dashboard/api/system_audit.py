"""Root-cause radar API for online system audits."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter
from sqlalchemy import func, select

from core.safe_output import safe_error_text
from db.session import get_session_ctx
from models.decision import AIDecision
from models.market_data import Kline, Ticker
from models.trade import Order, Position
from scripts.repair_missing_closed_positions_from_orders import (
    collect_missing_closed_position_plans,
)
from web_dashboard.api import data_collection as data_collection_api
from web_dashboard.api.system_health import system_self_check
from web_dashboard.api.text_sanitize import sanitize_payload

router = APIRouter()

AUDIT_WINDOWS = {"fast_minutes": 10, "trade_hours": 2, "strategy_hours": 24}
EXPECTED_KLINE_TIMEFRAMES = ("1m", "5m", "15m", "1h")
KLINE_STALE_LIMIT_SECONDS = {"1m": 120, "5m": 600, "15m": 1800, "1h": 7200}
STATUS_RANK = {"critical": 0, "warning": 1, "ok": 2, "info": 3}


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _age_seconds(value: Any) -> float | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return max((_now() - value.astimezone(UTC)).total_seconds(), 0.0)


def _status_from_counts(*, critical: bool = False, warning: bool = False) -> str:
    if critical:
        return "critical"
    if warning:
        return "warning"
    return "ok"


def _audit_card(
    key: str,
    title: str,
    status: str,
    summary: str,
    *,
    details: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    next_actions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "status": status,
        "summary": summary,
        "details": details or {},
        "evidence": evidence or [],
        "next_actions": next_actions or [],
    }


async def _trade_loop_audit() -> dict[str, Any]:
    now = _now()
    since_10m = now - timedelta(minutes=AUDIT_WINDOWS["fast_minutes"])
    since_2h = now - timedelta(hours=AUDIT_WINDOWS["trade_hours"])
    async with get_session_ctx() as session:
        recent_decisions = (
            await session.execute(
                select(func.count(AIDecision.id), func.max(AIDecision.created_at)).where(
                    AIDecision.created_at >= since_10m
                )
            )
        ).one()
        decisions_2h = (
            await session.execute(
                select(func.count(AIDecision.id), func.max(AIDecision.created_at)).where(
                    AIDecision.created_at >= since_2h
                )
            )
        ).one()
        orders_2h = (
            await session.execute(
                select(func.count(Order.id), func.max(Order.created_at)).where(
                    Order.created_at >= since_2h
                )
            )
        ).one()
        open_positions = (
            await session.execute(select(func.count(Position.id)).where(Position.is_open.is_(True)))
        ).scalar()
    recent_count = int(recent_decisions[0] or 0)
    decisions_count = int(decisions_2h[0] or 0)
    orders_count = int(orders_2h[0] or 0)
    latest_decision_age = _age_seconds(recent_decisions[1])
    stalled = recent_count == 0 or (latest_decision_age is not None and latest_decision_age > 600)
    status = _status_from_counts(
        critical=stalled, warning=orders_count == 0 and decisions_count > 30
    )
    summary = (
        "最近 10 分钟没有新增分析，交易主循环可能卡住。"
        if stalled
        else (
            "最近 2 小时有分析但没有订单，需结合开仓漏斗判断是否策略正常观望。"
            if orders_count == 0 and decisions_count > 30
            else "分析心跳和订单链路有活动。"
        )
    )
    return _audit_card(
        "trade_loop",
        "交易闭环",
        status,
        summary,
        details={
            "last_10m_decisions": recent_count,
            "last_2h_decisions": decisions_count,
            "last_2h_orders": orders_count,
            "open_positions": int(open_positions or 0),
            "latest_decision_at": _iso(recent_decisions[1]),
            "latest_order_at": _iso(orders_2h[1]),
        },
        evidence=[
            {"label": "10分钟决策", "value": recent_count},
            {"label": "2小时订单", "value": orders_count},
        ],
        next_actions=[
            "若 10 分钟决策为 0，先查交易服务心跳和当前 stage。",
            "若有大量分析但无订单，打开开仓漏斗看收益期望/风控/OKX 规则分布。",
        ],
    )


async def _okx_reconciliation_audit() -> dict[str, Any]:
    try:
        plans = await asyncio.wait_for(collect_missing_closed_position_plans(days=14), timeout=8.0)
    except Exception as exc:
        return _audit_card(
            "okx_reconciliation",
            "OKX 历史对账",
            "warning",
            "OKX 本地订单反推历史仓位 dry-run 执行失败。",
            details={"error": safe_error_text(exc, limit=180)},
            next_actions=["先修复 dry-run 失败原因，不能直接补历史仓位。"],
        )
    missing = len(plans)
    status = "critical" if missing else "ok"
    return _audit_card(
        "okx_reconciliation",
        "OKX 历史对账",
        status,
        (
            "存在可由 OKX 成交订单反推的缺失历史仓位。"
            if missing
            else "14 天历史仓位 dry-run 无缺失。"
        ),
        details={
            "window_days": 14,
            "missing_closed_positions": missing,
            "sample_plans": [
                {
                    "symbol": plan.symbol,
                    "side": plan.side,
                    "quantity": plan.quantity,
                    "realized_pnl": round(float(plan.realized_pnl), 8),
                    "close_order_id": plan.close_order_id,
                    "closed_at": _iso(plan.closed_at),
                }
                for plan in plans[:5]
            ],
        },
        evidence=[{"label": "缺失闭仓", "value": missing}],
        next_actions=[
            "只允许先 dry-run 人工核对，再按 symbol/order-id 精确 apply。",
            "如果缺失不为 0，先不要做策略收益判断，避免训练和盈亏被脏账影响。",
        ],
    )


async def _market_data_audit() -> dict[str, Any]:
    async with get_session_ctx() as session:
        kline_rows = (
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
        ticker_row = (
            await session.execute(
                select(
                    func.count(Ticker.id),
                    func.max(func.coalesce(Ticker.updated_at, Ticker.created_at)),
                )
            )
        ).one()
    by_timeframe = {str(row[0]): row for row in kline_rows}
    rows: list[dict[str, Any]] = []
    stale_timeframes: list[str] = []
    missing_timeframes: list[str] = []
    for timeframe in EXPECTED_KLINE_TIMEFRAMES:
        row = by_timeframe.get(timeframe)
        count = int(row[1] or 0) if row else 0
        symbols = int(row[2] or 0) if row else 0
        latest = row[3] if row else None
        age = _age_seconds(latest)
        missing = count <= 0
        stale = bool(age is None or age > KLINE_STALE_LIMIT_SECONDS[timeframe])
        if missing:
            missing_timeframes.append(timeframe)
        elif stale:
            stale_timeframes.append(timeframe)
        rows.append(
            {
                "timeframe": timeframe,
                "rows": count,
                "symbols": symbols,
                "latest_at": _iso(latest),
                "age_seconds": round(age, 3) if age is not None else None,
                "missing": missing,
                "stale": stale,
            }
        )
    ticker_age = _age_seconds(ticker_row[1])
    ticker_stale = ticker_age is None or ticker_age > 600
    status = _status_from_counts(
        critical=bool(missing_timeframes),
        warning=bool(stale_timeframes) or ticker_stale,
    )
    return _audit_card(
        "market_data",
        "行情与 K线",
        status,
        "行情/K线覆盖正常。" if status == "ok" else "行情或 K线覆盖存在缺失/过期。",
        details={
            "ticker_count": int(ticker_row[0] or 0),
            "ticker_latest_at": _iso(ticker_row[1]),
            "ticker_age_seconds": round(ticker_age, 3) if ticker_age is not None else None,
            "ticker_stale": ticker_stale,
            "klines": rows,
            "missing_timeframes": missing_timeframes,
            "stale_timeframes": stale_timeframes,
        },
        evidence=[{"label": f"{row['timeframe']} 币种", "value": row["symbols"]} for row in rows],
        next_actions=[
            "先查 DataService K线覆盖刷新任务和 OKX REST 错误。",
            "K线异常时不要先调整策略参数。",
        ],
    )


async def _strategy_quality_audit() -> dict[str, Any]:
    since = _now() - timedelta(hours=AUDIT_WINDOWS["strategy_hours"])
    async with get_session_ctx() as session:
        decisions = list(
            (
                await session.execute(
                    select(AIDecision)
                    .where(AIDecision.created_at >= since)
                    .order_by(AIDecision.created_at.desc())
                    .limit(500)
                )
            )
            .scalars()
            .all()
        )
        closed_positions = list(
            (
                await session.execute(
                    select(Position)
                    .where(Position.is_open.is_(False), Position.closed_at >= since)
                    .order_by(Position.closed_at.desc())
                    .limit(200)
                )
            )
            .scalars()
            .all()
        )
    actions = Counter(str(row.action or "unknown").lower() for row in decisions)
    entry_decisions = [
        row for row in decisions if str(row.action or "").lower() in {"long", "short"}
    ]
    blocked_reasons: Counter[str] = Counter()
    zero_expected = 0
    negative_expected = 0
    for row in entry_decisions:
        raw = row.raw_llm_response if isinstance(row.raw_llm_response, dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw, dict) else {}
        if isinstance(opportunity, dict):
            net = opportunity.get("expected_net_return_pct")
            try:
                net_value = float(net)
                if abs(net_value) < 1e-9:
                    zero_expected += 1
                elif net_value < 0:
                    negative_expected += 1
            except (TypeError, ValueError):
                pass
        reason = str(getattr(row, "execution_reason", "") or "").strip()
        if reason:
            blocked_reasons[reason[:80]] += 1
    fast_loss_positions = []
    for pos in closed_positions:
        created = pos.created_at
        closed = pos.closed_at
        if not isinstance(created, datetime) or not isinstance(closed, datetime):
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        if closed.tzinfo is None:
            closed = closed.replace(tzinfo=UTC)
        hold_minutes = max((closed - created).total_seconds() / 60.0, 0.0)
        pnl = float(pos.realized_pnl or 0.0)
        if hold_minutes <= 10 and pnl < 0:
            fast_loss_positions.append(
                {
                    "id": pos.id,
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "hold_minutes": round(hold_minutes, 3),
                    "realized_pnl": round(pnl, 8),
                    "closed_at": _iso(closed),
                }
            )
    warning = bool(
        fast_loss_positions or (entry_decisions and negative_expected >= len(entry_decisions) * 0.7)
    )
    return _audit_card(
        "strategy_quality",
        "策略质量",
        "warning" if warning else "ok",
        "存在快亏平或多数开仓候选净收益为负。" if warning else "最近策略质量未发现硬异常。",
        details={
            "window_hours": AUDIT_WINDOWS["strategy_hours"],
            "decision_count": len(decisions),
            "action_counts": dict(actions),
            "entry_decision_count": len(entry_decisions),
            "zero_expected_net_count": zero_expected,
            "negative_expected_net_count": negative_expected,
            "fast_loss_positions": fast_loss_positions[:10],
            "top_blocked_reasons": [
                {"reason": reason, "count": count}
                for reason, count in blocked_reasons.most_common(8)
            ],
        },
        evidence=[
            {"label": "开仓候选", "value": len(entry_decisions)},
            {"label": "负净收益", "value": negative_expected},
            {"label": "快亏平", "value": len(fast_loss_positions)},
        ],
        next_actions=[
            "负净收益占比高时先查成本/滑点/点差，不直接放宽开仓。",
            "快亏平出现时先看执行详情的风控步骤和 OKX 平仓来源。",
        ],
    )


async def _model_training_audit() -> dict[str, Any]:
    data_status, self_check = await asyncio.gather(
        data_collection_api.get_data_collection_status(),
        system_self_check(),
        return_exceptions=True,
    )
    if isinstance(data_status, Exception):
        return _audit_card(
            "model_training",
            "模型与训练",
            "warning",
            "数据采集/训练状态读取失败。",
            details={"error": safe_error_text(data_status, limit=180)},
        )
    training = data_status.get("training") if isinstance(data_status, dict) else {}
    local_tools = training.get("local_ai_tools") if isinstance(training, dict) else {}
    governance = training.get("governance") if isinstance(training, dict) else {}
    sources = data_status.get("sources") if isinstance(data_status, dict) else []
    source_warnings = [row for row in sources or [] if row.get("status") not in {"active", "ok"}]
    self_items = []
    if isinstance(self_check, dict):
        self_items = self_check.get("items") if isinstance(self_check.get("items"), list) else []
    model_critical = [
        row
        for row in self_items
        if str(row.get("key", "")).startswith("runtime_") and row.get("status") == "critical"
    ]
    status = _status_from_counts(
        critical=bool(model_critical),
        warning=bool(source_warnings) or not bool(local_tools.get("available")),
    )
    return _audit_card(
        "model_training",
        "模型与训练",
        status,
        "模型和训练数据状态正常。" if status == "ok" else "模型服务或训练数据源需要关注。",
        details={
            "local_ai_tools": {
                "available": bool(local_tools.get("available")),
                "status": local_tools.get("status"),
                "shadow_sample_count": local_tools.get("shadow_sample_count"),
                "trade_sample_count": local_tools.get("trade_sample_count"),
                "text_sentiment_sample_count": local_tools.get("text_sentiment_sample_count"),
            },
            "governance_status": governance.get("status") if isinstance(governance, dict) else None,
            "source_warnings": source_warnings[:8],
            "model_critical_items": model_critical[:8],
        },
        evidence=[
            {"label": "影子样本", "value": local_tools.get("shadow_sample_count") or 0},
            {"label": "交易样本", "value": local_tools.get("trade_sample_count") or 0},
            {"label": "文本样本", "value": local_tools.get("text_sentiment_sample_count") or 0},
        ],
        next_actions=[
            "模型 critical 时优先查端口契约 18000/18001/18002。",
            "训练数据源 warning 时先看数据采集页来源新鲜度。",
        ],
    )


def _root_cause_findings(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for card in cards:
        status = str(card.get("status") or "info")
        if status == "ok":
            continue
        findings.append(
            {
                "key": card.get("key"),
                "title": card.get("title"),
                "severity": status,
                "summary": card.get("summary"),
                "evidence": card.get("evidence") or [],
                "next_actions": card.get("next_actions") or [],
            }
        )
    return sorted(findings, key=lambda row: STATUS_RANK.get(str(row.get("severity")), 9))[:10]


@router.get("/system-audit/status")
async def system_audit_status() -> dict[str, Any]:
    results = await asyncio.gather(
        _trade_loop_audit(),
        _okx_reconciliation_audit(),
        _market_data_audit(),
        _strategy_quality_audit(),
        _model_training_audit(),
        return_exceptions=True,
    )
    cards: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        if isinstance(result, Exception):
            cards.append(
                _audit_card(
                    f"audit_section_{index}",
                    "巡检模块",
                    "warning",
                    "巡检模块执行失败。",
                    details={"error": safe_error_text(result, limit=180)},
                )
            )
        else:
            cards.append(result)
    cards = sorted(cards, key=lambda item: STATUS_RANK.get(str(item.get("status")), 9))
    findings = _root_cause_findings(cards)
    status = "ok"
    if any(card.get("status") == "critical" for card in cards):
        status = "critical"
    elif any(card.get("status") == "warning" for card in cards):
        status = "warning"
    return sanitize_payload(
        {
            "status": status,
            "status_label": {"ok": "正常", "warning": "需关注", "critical": "异常"}.get(
                status, status
            ),
            "checked_at": _now().isoformat(),
            "windows": AUDIT_WINDOWS,
            "summary": {
                "cards": len(cards),
                "critical": sum(1 for card in cards if card.get("status") == "critical"),
                "warning": sum(1 for card in cards if card.get("status") == "warning"),
                "ok": sum(1 for card in cards if card.get("status") == "ok"),
                "findings": len(findings),
            },
            "root_causes": findings,
            "cards": cards,
            "safety_note": "根因雷达当前只读巡检；补历史仓位、重启服务、批量训练等动作必须人工确认。",
        }
    )
