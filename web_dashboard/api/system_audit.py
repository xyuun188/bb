"""Root-cause radar API for online system audits."""

from __future__ import annotations

import ast
import asyncio
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from sqlalchemy import func, select

from config.settings import settings
from core.safe_output import safe_error_text
from core.symbols import normalize_trading_symbol
from db.session import get_session_ctx
from models.decision import AIDecision
from models.market_data import Kline, Ticker
from models.trade import Order, Position
from services.exchange_position_state import (
    exchange_position_display_valuation,
    parse_exchange_position_snapshot,
)
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
SYSTEM_AUDIT_HISTORY_FILE = "system_audit_history.jsonl"
POSITION_PRICE_SPLIT_WARN_PCT = 0.03
POSITION_PNL_SPLIT_WARN_USDT = 0.5


def _u(escaped: str) -> str:
    return escaped.encode("ascii").decode("unicode_escape")


SOURCE_MOJIBAKE_SCAN_TARGETS = (
    ("ai_brain", "*.py"),
    ("config", "*.py"),
    ("core", "*.py"),
    ("db", "*.py"),
    ("models", "*.py"),
    ("services", "*.py"),
    ("web_dashboard/api", "*.py"),
    ("web_dashboard/static/js", "*.js"),
    ("web_dashboard/static/css", "*.css"),
    ("web_dashboard/static", "*.html"),
    ("scripts", "*.py"),
)
SOURCE_MOJIBAKE_MARKERS = (
    _u("\\u951f"),
    _u("\\u951b"),
    _u("\\u9286"),
    _u("\\u95ab"),
    _u("\\u95b8"),
    _u("\\u95b9"),
    _u("\\u9227"),
    _u("\\u9225"),
    _u("\\u93c8"),
    _u("\\u93c3"),
    _u("\\u7487"),
    _u("\\u9352"),
    _u("\\u9359"),
    _u("\\u7459"),
    _u("\\u93b4"),
    _u("\\u93c1"),
    _u("\\u7edb"),
    _u("\\u6d5c\\u5fd4\\u5d2f"),
    _u("\\u9429"),
    _u("\\u7ee0\\u20ac"),
    _u("\\ufffd"),
)
STRATEGY_GATE_FORBIDDEN_PATTERNS = (
    "settings.min_entry_volume_ratio",
    "settings.min_entry_adx",
    "if False and",
)
STRATEGY_GATE_ALLOWED_PATHS = {"services/runtime_entry_filters.py"}


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _relative_gap(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denominator


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


async def _position_price_integrity_audit() -> dict[str, Any]:
    from web_dashboard.api import dashboard as dashboard_api

    split_rows: list[dict[str, Any]] = []
    checked_modes: list[str] = []
    unavailable_modes: list[dict[str, str]] = []
    local_open_count = 0
    exchange_open_count = 0

    for mode in ("paper", "live"):
        executor = dashboard_api._dashboard_okx_executor_for_mode(mode)
        if not executor:
            continue
        checked_modes.append(mode)
        try:
            exchange_positions = await asyncio.wait_for(executor.get_positions(), timeout=1.8)
        except Exception as exc:
            unavailable_modes.append({"mode": mode, "error": safe_error_text(exc, limit=120)})
            continue

        exchange_snapshots: dict[tuple[str, str], dict[str, Any]] = {}
        for raw_position in exchange_positions or []:
            snapshot = parse_exchange_position_snapshot(
                raw_position,
                symbol_normalizer=normalize_trading_symbol,
            )
            if not snapshot:
                continue
            exchange_snapshots[(str(snapshot["symbol"]), str(snapshot["side"]))] = snapshot
        exchange_open_count += len(exchange_snapshots)

        async with get_session_ctx() as session:
            local_positions = list(
                (
                    await session.execute(
                        select(Position).where(
                            Position.execution_mode == mode,
                            Position.is_open.is_(True),
                        )
                    )
                )
                .scalars()
                .all()
            )
        local_open_count += len(local_positions)

        for position in local_positions:
            key = (
                normalize_trading_symbol(position.symbol),
                str(position.side or "").lower(),
            )
            snapshot = exchange_snapshots.get(key)
            if not snapshot:
                continue
            valuation = exchange_position_display_valuation(
                snapshot,
                key[1],
                fallback_current_price=position.current_price,
                fallback_unrealized_pnl=position.unrealized_pnl,
                fallback_entry_price=position.entry_price,
                fallback_quantity=position.quantity,
            )
            local_price = _safe_float(position.current_price)
            okx_price = _safe_float(valuation.get("current_price"))
            local_pnl = _safe_float(position.unrealized_pnl)
            okx_pnl = _safe_float(valuation.get("unrealized_pnl"))
            price_gap = (
                _relative_gap(local_price, okx_price) if local_price > 0 and okx_price > 0 else 0.0
            )
            pnl_gap = abs(local_pnl - okx_pnl)
            if price_gap < POSITION_PRICE_SPLIT_WARN_PCT and pnl_gap < POSITION_PNL_SPLIT_WARN_USDT:
                continue
            split_rows.append(
                {
                    "mode": mode,
                    "symbol": key[0],
                    "side": key[1],
                    "local_price": round(local_price, 8),
                    "okx_price": round(okx_price, 8),
                    "price_gap_pct": round(price_gap * 100, 4),
                    "local_unrealized_pnl": round(local_pnl, 8),
                    "okx_unrealized_pnl": round(okx_pnl, 8),
                    "pnl_gap_usdt": round(pnl_gap, 8),
                    "pnl_source": valuation.get("pnl_source"),
                }
            )

    status = _status_from_counts(critical=bool(split_rows), warning=bool(unavailable_modes))
    return _audit_card(
        "position_price_integrity",
        "持仓价格一致性",
        status,
        (
            "发现平台持仓价/浮盈与 OKX 持仓快照不一致，可能影响持仓分析、平仓和训练标签。"
            if split_rows
            else (
                "部分模式暂时无法读取 OKX 持仓快照。"
                if unavailable_modes
                else "平台持仓价格与 OKX 持仓快照一致。"
            )
        ),
        details={
            "checked_modes": checked_modes,
            "unavailable_modes": unavailable_modes,
            "local_open_positions": local_open_count,
            "exchange_open_positions": exchange_open_count,
            "split_count": len(split_rows),
            "price_gap_warn_pct": POSITION_PRICE_SPLIT_WARN_PCT * 100,
            "pnl_gap_warn_usdt": POSITION_PNL_SPLIT_WARN_USDT,
            "splits": split_rows[:12],
        },
        evidence=[
            {"label": "价格/浮盈分裂", "value": len(split_rows)},
            {"label": "本地开仓", "value": local_open_count},
            {"label": "OKX持仓", "value": exchange_open_count},
        ],
        next_actions=[
            "若出现分裂，先运行 OKX 同步并复查持仓页；不要基于分裂数据调整策略参数。",
            "若同一币种反复分裂，检查 OKX 字段解析、合约面值 ctVal、行情缓存和持仓同步任务。",
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


def _source_scan_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iter_source_scan_files() -> list[Path]:
    root = _source_scan_root()
    files: list[Path] = []
    for dirname, pattern in SOURCE_MOJIBAKE_SCAN_TARGETS:
        base = root / dirname
        if not base.exists():
            continue
        files.extend(path for path in base.rglob(pattern) if path.is_file())
    return sorted(set(files), key=lambda item: item.as_posix())


def _relative_source_path(path: Path) -> str:
    try:
        return path.relative_to(_source_scan_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _source_visible_text_audit() -> dict[str, Any]:
    offenders: list[dict[str, Any]] = []
    scanned = 0
    for path in _iter_source_scan_files():
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = sorted({marker for marker in SOURCE_MOJIBAKE_MARKERS if marker in text})
        if hits:
            offenders.append(
                {
                    "path": _relative_source_path(path),
                    "markers": [
                        marker.encode("unicode_escape").decode("ascii") for marker in hits[:8]
                    ],
                }
            )
    status = "critical" if offenders else "ok"
    return _audit_card(
        "visible_text_encoding",
        "中文显示与乱码回归",
        status,
        (
            "源码和前端静态资源未发现裸乱码。"
            if not offenders
            else "发现源码/前端静态资源重新出现裸乱码。"
        ),
        details={
            "scanned_files": scanned,
            "offender_count": len(offenders),
            "offenders": offenders[:20],
            "scope": [f"{dirname}/{pattern}" for dirname, pattern in SOURCE_MOJIBAKE_SCAN_TARGETS],
        },
        evidence=[
            {"label": "扫描文件", "value": scanned},
            {"label": "乱码文件", "value": len(offenders)},
        ],
        next_actions=[
            "若乱码文件不为 0，先定位来源是源码文案、模型返回还是历史数据，不要只在前端替换显示。",
            "历史数据修复样本必须使用 Unicode 转义，不允许裸乱码写入源码。",
        ],
    )


def _strategy_gate_contract_audit() -> dict[str, Any]:
    root = _source_scan_root()
    scan_paths = (
        root / "ai_brain",
        root / "services",
        root / "risk_manager",
        root / "web_dashboard/api/dashboard.py",
    )
    files: list[Path] = []
    for path in scan_paths:
        if path.is_file():
            files.append(path)
        elif path.exists():
            files.extend(candidate for candidate in path.rglob("*.py") if candidate.is_file())
    offenders: list[dict[str, Any]] = []
    for path in sorted(set(files), key=lambda item: item.as_posix()):
        rel_path = _relative_source_path(path)
        if rel_path in STRATEGY_GATE_ALLOWED_PATHS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [pattern for pattern in STRATEGY_GATE_FORBIDDEN_PATTERNS if pattern in text]
        if hits:
            offenders.append({"path": rel_path, "patterns": hits})
    try:
        runtime_source = ast.parse(
            (root / "services/runtime_entry_filters.py").read_text(encoding="utf-8")
        )
        runtime_contract_available = any(
            isinstance(node, ast.ClassDef) and node.name == "RuntimeEntryFilters"
            for node in ast.walk(runtime_source)
        )
    except Exception:
        runtime_contract_available = False
    status = "critical" if offenders or not runtime_contract_available else "ok"
    return _audit_card(
        "strategy_gate_contract",
        "策略门槛契约",
        status,
        (
            "策略运行时门槛保持解释/排序/仓位参考，不是固定硬开仓门槛。"
            if status == "ok"
            else "发现固定门槛或死分支残留，可能重新把策略卡死。"
        ),
        details={
            "runtime_contract_available": runtime_contract_available,
            "forbidden_patterns": list(STRATEGY_GATE_FORBIDDEN_PATTERNS),
            "offender_count": len(offenders),
            "offenders": offenders[:20],
        },
        evidence=[
            {"label": "固定门槛残留", "value": len(offenders)},
            {"label": "运行时契约", "value": "存在" if runtime_contract_available else "缺失"},
        ],
        next_actions=[
            "如发现 settings.min_entry_* 直接参与运行链路，必须改为 RuntimeEntryFilters 动态参考。",
            "低量比、ADX、置信度只能影响排序、仓位、杠杆和解释；不能作为硬开仓门槛。",
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


def _worst_status(*statuses: Any) -> str:
    normalized = [str(status or "info") for status in statuses]
    return min(normalized or ["info"], key=lambda item: STATUS_RANK.get(item, 9))


def _card_map(cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(card.get("key") or ""): card for card in cards if card.get("key")}


def _node_from_cards(
    key: str,
    title: str,
    layer: str,
    cards_by_key: dict[str, dict[str, Any]],
    card_keys: list[str],
    *,
    impact: str,
    upstream: list[str] | None = None,
    downstream: list[str] | None = None,
    checks: list[str] | None = None,
) -> dict[str, Any]:
    related_cards = [cards_by_key[item] for item in card_keys if item in cards_by_key]
    status = _worst_status(*(card.get("status") for card in related_cards))
    summaries = [str(card.get("summary") or "") for card in related_cards if card.get("summary")]
    evidence: list[dict[str, Any]] = []
    next_actions: list[str] = []
    for card in related_cards:
        evidence.extend(card.get("evidence") or [])
        next_actions.extend(card.get("next_actions") or [])
    return {
        "key": key,
        "title": title,
        "layer": layer,
        "status": status,
        "summary": "；".join(summaries[:2]) or "节点暂无异常。",
        "impact": impact,
        "upstream": upstream or [],
        "downstream": downstream or [],
        "checks": checks or [],
        "card_keys": [card.get("key") for card in related_cards],
        "evidence": evidence[:6],
        "next_actions": list(dict.fromkeys(next_actions))[:6],
    }


def _build_audit_nodes(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards_by_key = _card_map(cards)
    return [
        _node_from_cards(
            "runtime_loop",
            "调度与心跳",
            "运行层",
            cards_by_key,
            ["trade_loop"],
            impact="决定系统是否持续分析、是否卡在某个阶段。",
            downstream=["market_data", "position_sync", "strategy_decision"],
            checks=["最近10分钟分析", "最近2小时订单", "当前持仓数量"],
        ),
        _node_from_cards(
            "market_data",
            "行情与K线",
            "数据层",
            cards_by_key,
            ["market_data"],
            impact="影响开仓候选、预期收益、止盈止损和训练特征。",
            upstream=["runtime_loop"],
            downstream=["strategy_decision", "risk_guard", "training_data"],
            checks=["Ticker新鲜度", "1m/5m/15m/1h K线覆盖", "币种覆盖"],
        ),
        _node_from_cards(
            "model_training",
            "模型与训练数据",
            "模型层",
            cards_by_key,
            ["model_training"],
            impact="影响盈利预测、时序预测、情绪预测、本地ML过滤和样本学习。",
            upstream=["market_data"],
            downstream=["strategy_decision", "training_data"],
            checks=["本地量化工具", "影子样本", "交易样本", "外部采集源"],
        ),
        _node_from_cards(
            "strategy_decision",
            "策略决策质量",
            "策略层",
            cards_by_key,
            ["strategy_quality"],
            impact="影响是否开仓、仓位大小、重复亏损复开和快进快出。",
            upstream=["market_data", "model_training", "position_sync"],
            downstream=["risk_guard", "okx_execution"],
            checks=["负净收益候选", "零净收益候选", "快亏平样本", "拦截原因"],
        ),
        _node_from_cards(
            "strategy_gate_contract",
            "策略门槛契约",
            "策略层",
            cards_by_key,
            ["strategy_gate_contract"],
            impact="防止旧固定阈值、死分支、伪硬门槛重新卡住开仓。",
            upstream=["model_training", "strategy_decision"],
            downstream=["risk_guard", "okx_execution"],
            checks=["RuntimeEntryFilters", "settings.min_entry_*残留", "if False死分支"],
        ),
        _node_from_cards(
            "risk_guard",
            "风控与守门",
            "风控层",
            cards_by_key,
            ["strategy_quality", "position_price_integrity"],
            impact="影响动态证据、低质量释放、快速平仓和下单前校验。",
            upstream=["strategy_decision", "position_sync"],
            downstream=["okx_execution", "position_sync"],
            checks=["持仓价一致性", "快亏平", "风险证据", "执行原因"],
        ),
        _node_from_cards(
            "okx_execution",
            "OKX执行与历史对账",
            "执行层",
            cards_by_key,
            ["okx_reconciliation", "position_price_integrity"],
            impact="影响下单、平仓、历史仓位、账户余额和盈亏记录。",
            upstream=["risk_guard"],
            downstream=["position_sync", "training_data"],
            checks=["缺失历史仓", "OKX持仓快照", "价格/PnL对齐"],
        ),
        _node_from_cards(
            "position_sync",
            "持仓同步与PnL",
            "同步层",
            cards_by_key,
            ["position_price_integrity", "okx_reconciliation"],
            impact="影响主面板余额、持仓分析、平仓判断和训练标签。",
            upstream=["okx_execution"],
            downstream=["strategy_decision", "training_data", "dashboard_observability"],
            checks=["平台价 vs OKX标记价", "平台浮盈 vs OKX upl", "合约面值ctVal"],
        ),
        _node_from_cards(
            "training_data",
            "训练标签与样本治理",
            "学习层",
            cards_by_key,
            ["model_training", "strategy_quality", "position_price_integrity"],
            impact="影响模型是否越学越聪明，避免错误价格/错误盈亏污染训练。",
            upstream=["market_data", "okx_execution", "position_sync"],
            downstream=["model_training", "strategy_decision"],
            checks=["样本数量", "数据源状态", "脏样本风险", "收益标签可信度"],
        ),
        _node_from_cards(
            "dashboard_observability",
            "页面与可观测性",
            "展示层",
            cards_by_key,
            ["trade_loop", "position_price_integrity", "model_training"],
            impact="影响你能否从页面直接定位问题，而不是只看到泛化提示。",
            upstream=["position_sync", "model_training"],
            checks=["节点状态", "根因列表", "执行证据", "历史记录"],
        ),
        _node_from_cards(
            "visible_text_encoding",
            "中文显示与乱码",
            "展示层",
            cards_by_key,
            ["visible_text_encoding"],
            impact="防止源码、页面或修复脚本重新出现裸乱码，影响排查和用户判断。",
            upstream=["dashboard_observability"],
            checks=["源码扫描", "前端静态资源", "脚本样本转义"],
        ),
    ]


def _node_summary(nodes: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "nodes": len(nodes),
        "critical": sum(1 for node in nodes if node.get("status") == "critical"),
        "warning": sum(1 for node in nodes if node.get("status") == "warning"),
        "ok": sum(1 for node in nodes if node.get("status") == "ok"),
    }


def _history_path() -> Path:
    return settings.data_dir / SYSTEM_AUDIT_HISTORY_FILE


def _history_record(payload: dict[str, Any], *, source: str) -> dict[str, Any]:
    root_causes = payload.get("root_causes") if isinstance(payload.get("root_causes"), list) else []
    return {
        "checked_at": payload.get("checked_at"),
        "source": source,
        "status": payload.get("status"),
        "status_label": payload.get("status_label"),
        "summary": payload.get("summary") or {},
        "node_summary": payload.get("node_summary") or {},
        "root_causes": root_causes[:8],
    }


def _read_history_records(limit: int = 50) -> list[dict[str, Any]]:
    path = _history_path()
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records[-max(1, int(limit)) :][::-1]


def _append_history_record(payload: dict[str, Any], *, source: str) -> None:
    if not settings.system_audit_history_enabled:
        return
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    max_records = max(50, min(int(settings.system_audit_history_max_records or 500), 5000))
    existing = list(reversed(_read_history_records(limit=max_records - 1)))
    existing.append(_history_record(payload, source=source))
    text = "\n".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        for item in existing[-max_records:]
    )
    path.write_text(text + "\n", encoding="utf-8")


async def collect_system_audit_status(
    *, record_history: bool = True, source: str = "api"
) -> dict[str, Any]:
    results = await asyncio.gather(
        _trade_loop_audit(),
        _okx_reconciliation_audit(),
        _position_price_integrity_audit(),
        _market_data_audit(),
        _strategy_quality_audit(),
        _model_training_audit(),
        asyncio.to_thread(_strategy_gate_contract_audit),
        asyncio.to_thread(_source_visible_text_audit),
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
    nodes = _build_audit_nodes(cards)
    findings = _root_cause_findings(cards)
    status = "ok"
    if any(card.get("status") == "critical" for card in cards):
        status = "critical"
    elif any(card.get("status") == "warning" for card in cards):
        status = "warning"
    payload = sanitize_payload(
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
                "nodes": len(nodes),
            },
            "root_causes": findings,
            "nodes": nodes,
            "node_summary": _node_summary(nodes),
            "cards": cards,
            "history": {
                "enabled": bool(settings.system_audit_history_enabled),
                "interval_seconds": int(settings.system_audit_history_interval_seconds or 300),
                "max_records": int(settings.system_audit_history_max_records or 500),
            },
            "safety_note": "根因雷达当前只读巡检；补历史仓位、重启服务、批量训练等动作必须人工确认。",
        }
    )
    if record_history:
        _append_history_record(payload, source=source)
    return payload


@router.get("/system-audit/status")
async def system_audit_status() -> dict[str, Any]:
    return await collect_system_audit_status(record_history=True, source="api")


@router.get("/system-audit/history")
async def system_audit_history(limit: int = 50) -> dict[str, Any]:
    safe_limit = max(1, min(int(limit or 50), 200))
    records = _read_history_records(limit=safe_limit)
    return sanitize_payload(
        {
            "enabled": bool(settings.system_audit_history_enabled),
            "interval_seconds": int(settings.system_audit_history_interval_seconds or 300),
            "records": records,
            "count": len(records),
        }
    )
