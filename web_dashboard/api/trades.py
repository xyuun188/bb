"""
Trade history API endpoints.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from core.symbols import normalize_trading_symbol, symbol_query_variants
from db.repositories.risk_repo import RiskRepository
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from models.account import OkxAccountBill
from models.decision import AIDecision
from models.trade import Order, Position
from services.decision_execution_trace import build_execution_trace
from services.execution_reason_localizer import localize_execution_reason
from services.execution_result_classifier import ExecutionResultClassifier
from services.manual_close_marker import MANUAL_CLOSE_LABEL, is_manual_close_order
from services.okx_error_classifier import is_okx_temporary_service_error
from services.okx_order_fact_sync import (
    OKX_SYNC_CONFIRMED,
    OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
    OKX_SYNC_OKX_ONLY,
)
from services.okx_position_ledger_view import build_okx_position_ledger_groups
from services.position_open_time import parse_position_time
from services.position_settlement import is_final_settlement_status
from web_dashboard.api import dashboard as dashboard_api
from web_dashboard.api.security import require_destructive_dashboard_confirmation
from web_dashboard.api.text_sanitize import looks_mojibake, sanitize_payload, sanitize_text

router = APIRouter()
EXECUTION_REASON_CLASSIFIER = ExecutionResultClassifier()
CLOSE_ORDER_POSITION_MATCH_WINDOW_SECONDS = 240
EXCHANGE_SYNC_DECISION_SOURCES = {
    "okx_order_pair_repair",
    "okx_position_reconcile",
    "okx_tp_sl_backfill",
}
SYSTEM_PROTECTION_RECONCILE_ORIGINS = {"system_protection"}
SYSTEM_PROTECTION_LABEL = "系统保护单"


def _normalize_display_symbol(symbol: str | None) -> str:
    return normalize_trading_symbol(symbol)


def _symbol_query_variants(symbols: set[str]) -> set[str]:
    return symbol_query_variants(symbols)


def _aware_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _closed_position_matches_order_side(order_side: str | None, position_side: str | None) -> bool:
    side = str(position_side or "").lower()
    close_side = "buy" if side == "short" else "sell" if side == "long" else ""
    return bool(close_side and str(order_side or "").lower() == close_side)


def _close_price_matches_position_protection(order: Any, position: Any | None) -> bool:
    if position is None:
        return False
    close_price = _safe_float(getattr(order, "price", None), 0.0) or _safe_float(
        getattr(position, "current_price", None),
        0.0,
    )
    if close_price <= 0:
        return False
    side = str(getattr(position, "side", "") or "").lower()
    tolerance = max(abs(close_price) * 0.015, 1e-12)
    stop_loss_price = _safe_float(getattr(position, "stop_loss_price", None), 0.0)
    take_profit_price = _safe_float(getattr(position, "take_profit_price", None), 0.0)
    if stop_loss_price > 0 and (
        abs(close_price - stop_loss_price) <= tolerance
        or (side == "long" and close_price <= stop_loss_price)
        or (side == "short" and close_price >= stop_loss_price)
    ):
        return True
    if take_profit_price > 0 and (
        abs(close_price - take_profit_price) <= tolerance
        or (side == "long" and close_price >= take_profit_price)
        or (side == "short" and close_price <= take_profit_price)
    ):
        return True
    return False


def _matching_closed_positions_for_order(
    order: Any,
    positions: list[Any],
    *,
    window_seconds: int = CLOSE_ORDER_POSITION_MATCH_WINDOW_SECONDS,
) -> list[Any]:
    if getattr(order, "status", None) != "filled":
        return []
    filled_at = _aware_datetime(getattr(order, "filled_at", None))
    if filled_at is None:
        return []
    order_symbol = _normalize_display_symbol(getattr(order, "symbol", None))
    order_qty = _safe_float(getattr(order, "quantity", None), 0.0) or 0.0
    candidates: list[tuple[float, float, float, Any]] = []
    for position in positions:
        if getattr(position, "is_open", False):
            continue
        if getattr(position, "model_name", None) != getattr(order, "model_name", None):
            continue
        if getattr(position, "execution_mode", None) != getattr(order, "execution_mode", None):
            continue
        if _normalize_display_symbol(getattr(position, "symbol", None)) != order_symbol:
            continue
        if not _closed_position_matches_order_side(
            getattr(order, "side", None), getattr(position, "side", None)
        ):
            continue
        closed_at = _aware_datetime(getattr(position, "closed_at", None))
        if closed_at is None:
            continue
        time_delta = abs((closed_at - filled_at).total_seconds())
        if time_delta > window_seconds:
            continue
        position_qty = _safe_float(getattr(position, "quantity", None), 0.0) or 0.0
        if order_qty > 0 and position_qty > order_qty * 1.05:
            continue
        price_delta = abs(
            (_safe_float(getattr(order, "price", None), 0.0) or 0.0)
            - (_safe_float(getattr(position, "current_price", None), 0.0) or 0.0)
        )
        qty_delta = abs(order_qty - position_qty)
        candidates.append((time_delta, price_delta, qty_delta, position))
    if not candidates:
        return []
    ordered = [item[3] for item in sorted(candidates, key=lambda item: (item[0], item[1], item[2]))]
    if order_qty <= 0:
        return ordered[:1]
    matched: list[Any] = []
    total_qty = 0.0
    for position in ordered:
        matched.append(position)
        total_qty += _safe_float(getattr(position, "quantity", None), 0.0) or 0.0
        if total_qty >= order_qty * 0.98:
            break
    return matched


def _weighted_entry_price(positions: list[Any]) -> float:
    total_qty = sum(_safe_float(getattr(p, "quantity", None), 0.0) or 0.0 for p in positions)
    if total_qty <= 0:
        return 0.0
    return (
        sum(
            (_safe_float(getattr(p, "entry_price", None), 0.0) or 0.0)
            * (_safe_float(getattr(p, "quantity", None), 0.0) or 0.0)
            for p in positions
        )
        / total_qty
    )


def _execution_source_from_decision(decision: Any | None, order: Any) -> tuple[str, str]:
    if is_manual_close_order(order):
        return "manual", MANUAL_CLOSE_LABEL
    if decision is None:
        return "system", "系统执行"
    meta = {
        "raw_llm_response": getattr(decision, "raw_llm_response", None),
        "feature_snapshot": getattr(decision, "feature_snapshot", None),
    }
    raw, snapshot = _decision_raw_and_snapshot(meta)
    if _is_system_protection_reconcile(raw, snapshot):
        return "system", SYSTEM_PROTECTION_LABEL
    if _is_exchange_sync_decision(raw, snapshot):
        return "okx", "OKX同步"
    return "system", "系统执行"


def _close_order_position_reason(
    order: Any,
    positions: list[Any],
    *,
    execution_source: str | None = None,
) -> str | None:
    if not positions:
        return None
    symbol = _normalize_display_symbol(getattr(order, "symbol", None))
    quantity = sum(_safe_float(getattr(p, "quantity", None), 0.0) or 0.0 for p in positions)
    entry_price = _weighted_entry_price(positions)
    close_price = _safe_float(getattr(order, "price", None), 0.0) or 0.0
    pnl = sum(_safe_float(getattr(p, "realized_pnl", None), 0.0) or 0.0 for p in positions)
    side = str(getattr(positions[0], "side", "") or "").lower()
    action_label = "买入平空" if side == "short" else "卖出平多" if side == "long" else "平仓"
    if execution_source == "okx":
        source = "OKX 平仓成交已同步"
    elif execution_source == "manual":
        source = "手动平仓成交已确认"
    else:
        source = "系统平仓成交已确认"
    return (
        f"{source}：{symbol} {action_label} {quantity:g}，成交价 {close_price:g}，"
        f"平掉本地 {len(positions)} 段仓位，开仓均价 {entry_price:g}，"
        f"实现盈亏 {pnl:.4f} USDT。"
    )


def _looks_mojibake(text: str) -> bool:
    return looks_mojibake(text)


def _extract_okx_error(text: str) -> tuple[str | None, str | None]:
    match = re.search(r"okx\s+(\{.*\})", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None, None
    try:
        payload = json.loads(match.group(1))
    except Exception:
        return None, None

    code = str(payload.get("code") or "").strip() or None
    msg = str(payload.get("msg") or "").strip() or None
    for item in payload.get("data") or []:
        if isinstance(item, dict):
            code = str(item.get("sCode") or code or "").strip() or code
            msg = str(item.get("sMsg") or msg or "").strip() or msg
            break
    return code, msg


def _translate_execution_text(message: str | None) -> str:
    text = str(localize_execution_reason(message) or "").strip()
    if not text:
        return "交易接口未返回具体原因。"

    translated = EXECUTION_REASON_CLASSIFIER.translate_execution_error_text(text)
    if translated:
        return translated

    lower_text = text.lower()
    if (
        "open interest" in lower_text and "platform" in lower_text and "limit" in lower_text
    ) or "has reached the platform's limit" in lower_text:
        return (
            "OKX 拒绝开仓：该合约当前平台总持仓量已经达到 OKX 上限，"
            "交易所暂时不允许继续增加这个合约的新仓。"
            "这不是 AI 方向或下单数量计算错误；系统会临时跳过该币种，稍后等 OKX 限制解除再重新分析。"
        )

    okx_code, okx_msg = _extract_okx_error(text)
    if okx_code:
        if okx_code == "51008":
            return (
                "OKX 返回错误码 51008：账户 USDT 保证金不足，订单没有成交。"
                "请检查可用余额、已有仓位保证金占用、挂单和杠杆设置。"
                f"OKX 原文：{okx_msg or 'Order failed. Insufficient USDT margin in account'}"
            )
        if okx_code == "51004":
            return (
                "OKX 返回错误码 51004：账户可用保证金不足或下单金额超过账户可用额度。"
                f"OKX 原文：{okx_msg or text}"
            )
        if okx_code == "59670":
            return (
                "OKX 返回错误码 59670：该交易对当前挂单超过 5 个，OKX 不允许在这种状态下调整杠杆。"
                "系统需要先撤掉该交易对多余挂单，再重新设置杠杆和下单。"
                f"OKX 原文：{okx_msg or text}"
            )
        if okx_code == "51155":
            return (
                "OKX 返回错误码 51155：订单数量、价格或合约面值不符合交易所限制。"
                f"OKX 原文：{okx_msg or text}"
            )
        if okx_code == "51169":
            return (
                "OKX 返回错误码 51169：订单不符合当前合约交易规则，通常与最小张数、价格精度或仓位模式有关。"
                f"OKX 原文：{okx_msg or text}"
            )
        return f"OKX 返回错误码 {okx_code}：{okx_msg or text}"

    checks = [
        ("59670", "OKX 当前该交易对挂单超过 5 个，需要先撤掉多余挂单后才能调整杠杆。"),
        (
            "more than 5 open orders",
            "OKX 当前该交易对挂单超过 5 个，需要先撤掉多余挂单后才能调整杠杆。",
        ),
        ("51008", "OKX 返回错误码 51008：账户 USDT 保证金不足，订单没有成交。"),
        ("Insufficient USDT margin", "OKX 返回错误码 51008：账户 USDT 保证金不足，订单没有成交。"),
        ("51004", "OKX 提示账户可用保证金不足或订单金额超过可用额度。"),
        ("51155", "OKX 提示订单数量、价格或合约面值不符合交易所限制。"),
        ("51169", "OKX 提示订单不符合当前合约交易规则。"),
        ("local compliance restrictions", "OKX 提示该交易受地区或账户合规限制。"),
        ("don't have any positions in this direction", "OKX 当前没有这个方向的可平仓位。"),
        ("No current price available", "系统没有拿到当前价格，未提交订单。"),
        (
            "Order size is below OKX minimum contract size",
            "订单数量低于 OKX 最小合约张数，未提交订单。",
        ),
        ("No matching position to close", "系统没有找到匹配的可平仓位，未提交平仓单。"),
        ("Insufficient balance", "账户可用余额不足，未提交订单。"),
        ("Invalid OK-ACCESS-KEY", "OKX API Key 无效，请检查系统设置中的 OKX Key。"),
        ("Invalid Sign", "OKX 签名校验失败，请检查 API Secret 和 Passphrase。"),
        ("OKX leverage set failed", "OKX 杠杆设置失败。"),
        ("Failed to place order", "OKX 下单失败。"),
    ]
    for needle, translated in checks:
        if needle in text:
            if needle in {"OKX leverage set failed", "Failed to place order"}:
                return sanitize_text(f"{translated} 原始返回：{text[:260]}")
            return translated
    if _looks_mojibake(text):
        return sanitize_text(text)
    return sanitize_text(text)


def _execution_failure_kind(message: str | None) -> str | None:
    if is_okx_temporary_service_error(message):
        return "transient_exchange_error"
    return None


def _execution_status_label(status: str | None, reason: str | None) -> str:
    normalized_status = str(status or "").lower()
    if normalized_status == "filled":
        return "执行成功"
    if _execution_failure_kind(reason) == "transient_exchange_error":
        return "交易所临时不可用"
    return "执行失败"


CORRUPTED_HISTORY_REASON = "该笔历史记录的原始说明已损坏，无法准确还原"


def _order_okx_confirmed(order: Any) -> bool:
    sync_status = str(getattr(order, "okx_sync_status", "") or "").lower().strip()
    return sync_status in {
        OKX_SYNC_CONFIRMED,
        OKX_SYNC_OKX_ONLY,
        OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
    }


def _order_success(order: Any) -> bool:
    return str(getattr(order, "status", "") or "").lower() == "filled" and _order_okx_confirmed(
        order
    )


def _text_is_unusable(text: str | None) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    if re.fullmatch(r"\d{12,}", value):
        return True
    if CORRUPTED_HISTORY_REASON in value:
        return True
    if value.count("?") >= max(6, int(len(value) * 0.25)):
        return True
    return _looks_mojibake(value)


def _readable_execution_reason(
    *,
    execution_reason: str | None,
    reasoning: str | None,
    exchange_order_id: str | None,
    status: str | None,
) -> str:
    """Choose a human-readable reason without treating order ids as reasons."""

    for candidate in (execution_reason, reasoning):
        text = sanitize_text(candidate)
        if text and not _text_is_unusable(text):
            return _translate_execution_text(text)
    exchange_text = sanitize_text(exchange_order_id)
    if (
        exchange_text
        and exchange_text.lower() != "rejected"
        and not re.fullmatch(r"\d{12,}", exchange_text)
    ):
        return _translate_execution_text(exchange_text)
    if str(status or "").lower() == "filled":
        return "订单已成交，本地订单与持仓记录已同步；具体交易原因请查看 AI 裁决依据和执行步骤。"
    return "订单未成交或执行失败，交易接口未返回更详细原因。"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _decision_raw_and_snapshot(
    meta: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = meta or {}
    raw = data.get("raw_llm_response") or {}
    snapshot = data.get("feature_snapshot") or {}
    return _safe_dict(raw), _safe_dict(snapshot)


def _is_exchange_sync_decision(raw: dict[str, Any], snapshot: dict[str, Any]) -> bool:
    source = str(raw.get("source") or snapshot.get("source") or "").lower().strip()
    if source in EXCHANGE_SYNC_DECISION_SOURCES:
        return True
    if raw.get("system_sync"):
        # system_sync is only an exchange-side execution source when it is tied to
        # an explicit sync/reconciliation origin. Normal system orders also settle
        # through OKX and must remain "system" initiated in the execution ledger.
        return source in EXCHANGE_SYNC_DECISION_SOURCES
    return False


def _reconcile_origin(raw: dict[str, Any], snapshot: dict[str, Any]) -> str:
    close_fill = _safe_dict(raw.get("close_fill"))
    return (
        str(
            raw.get("reconcile_origin")
            or close_fill.get("reconcile_origin")
            or snapshot.get("reconcile_origin")
            or ""
        )
        .lower()
        .strip()
    )


def _is_system_protection_reconcile(raw: dict[str, Any], snapshot: dict[str, Any]) -> bool:
    if _reconcile_origin(raw, snapshot) not in SYSTEM_PROTECTION_RECONCILE_ORIGINS:
        return False
    return _is_exchange_sync_decision(raw, snapshot)


def _pct_text(value, digits: int = 2) -> str | None:
    number = _safe_float(value, 0.0)
    if number == 0:
        return None
    return f"{number * 100:+.{digits}f}%"


def _repair_position_reason_hold_hours(reason: str | None, hold_minutes: float | None) -> str:
    """Replace stale stored zero hold-hour text with matched position duration."""

    text = str(reason or "").strip()
    if not text or hold_minutes is None or hold_minutes <= 0:
        return text
    hold_hours = hold_minutes / 60.0
    formatted = f"{hold_hours:.4f}".rstrip("0").rstrip(".")
    replacements = (
        (r"(持仓小时=)0(?:\.0+)?(?=。|，|；|,|;|\s|$)", rf"\g<1>{formatted}"),
        (r"(hold_hours=)0(?:\.0+)?(?=\.|,|;|\s|$)", rf"\g<1>{formatted}"),
    )
    repaired = text
    for pattern, replacement in replacements:
        repaired = re.sub(pattern, replacement, repaired)
    return repaired


def _fast_risk_reason_from_raw(symbol: str | None, raw: dict[str, Any]) -> str | None:
    """Build a readable reason for old fast-risk close records whose reasoning was mojibake."""
    if not isinstance(raw, dict):
        return None
    trigger = str(raw.get("fast_risk_trigger") or "").strip()
    if not (raw.get("fast_risk_exit") or trigger or raw.get("predictive_reversal")):
        return None

    plan = _safe_dict(raw.get("fast_exit_plan"))
    fraction = _safe_float(raw.get("close_fraction") or plan.get("fraction"), 1.0)
    close_label = "全部平仓" if fraction >= 0.999 else f"部分平仓 {fraction:.0%}"
    symbol_text = str(symbol or "").strip() or "该币种"

    note = str(plan.get("note") or "").strip()
    note_text = note if note and not _text_is_unusable(note) else ""

    reversal = _safe_dict(raw.get("predictive_reversal"))
    reversal_score = _safe_float(raw.get("predictive_reversal_score") or reversal.get("score"), 0.0)
    reversal_text = f"短周期反向风险评分 {reversal_score:.0f}。" if reversal_score > 0 else ""

    returns = [
        label
        for label in (
            f"1分钟 {_pct_text(raw.get('returns_1'))}" if _pct_text(raw.get("returns_1")) else None,
            f"5分钟 {_pct_text(raw.get('returns_5'))}" if _pct_text(raw.get("returns_5")) else None,
            (
                f"20分钟 {_pct_text(raw.get('returns_20'))}"
                if _pct_text(raw.get("returns_20"))
                else None
            ),
        )
        if label
    ]
    returns_text = f"短线涨跌：{'，'.join(returns)}。" if returns else ""

    if trigger in {"profit_drawdown_reduce", "profit_drawdown_close"}:
        return (
            f"盈利保护触发：{symbol_text} 持仓曾经达到可保护浮盈，但利润开始明显回撤，"
            f"系统已执行{close_label}来锁定收益。{note_text}{reversal_text}{returns_text}"
        )
    if trigger in {"hard_adverse_move", "fast_adverse_move", "fast_adverse_reduce"}:
        return (
            f"快速风控触发：{symbol_text} 价格短线明显向当前持仓反方向移动，"
            f"为避免亏损继续扩大，系统已执行{close_label}。{note_text}{reversal_text}{returns_text}"
        )
    if trigger == "take_profit":
        return f"OKX 止盈触发：{symbol_text} 已执行{close_label}。{note_text}{returns_text}"
    if trigger:
        return f"快速风控触发：{symbol_text} 触发 {trigger}，系统已执行{close_label}。{note_text}{reversal_text}{returns_text}"
    return None


@router.get("/trades")
async def get_trades(
    model_name: str | None = Query(None),
    symbol: str | None = Query(None),
    mode: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    page: int = Query(1, ge=1),
):
    """Get recent trades with optional filters."""
    page_size = max(1, min(int(limit or 50), 500))
    offset = (max(int(page or 1), 1) - 1) * page_size
    async with get_session_ctx() as session:
        repo = TradeRepository(session)
        orders = await repo.get_recent_orders(
            model_name=model_name,
            symbol=symbol,
            execution_mode=mode,
            limit=page_size,
            offset=offset,
        )
        total = await repo.count_orders(model_name=model_name, symbol=symbol, execution_mode=mode)
        filled_total = await repo.count_orders(
            model_name=model_name,
            symbol=symbol,
            execution_mode=mode,
            statuses=["filled"],
            require_exchange_order_id=True,
        )
        decision_ids = [o.decision_id for o in orders if o.decision_id]
        decision_meta: dict[int, dict[str, Any]] = {}
        if decision_ids:
            result = await session.execute(
                select(
                    AIDecision.id,
                    AIDecision.action,
                    AIDecision.reasoning,
                    AIDecision.execution_reason,
                    AIDecision.position_size_pct,
                    AIDecision.suggested_leverage,
                    AIDecision.feature_snapshot,
                    AIDecision.raw_llm_response,
                ).where(AIDecision.id.in_(decision_ids))
            )
            decision_meta = {
                int(row.id): {
                    "action": row.action,
                    "reasoning": row.reasoning,
                    "execution_reason": localize_execution_reason(row.execution_reason),
                    "position_size_pct": row.position_size_pct,
                    "suggested_leverage": row.suggested_leverage,
                    "feature_snapshot": row.feature_snapshot,
                    "raw_llm_response": row.raw_llm_response,
                }
                for row in result.all()
            }
        order_symbols = {o.symbol for o in orders if o.symbol}
        position_symbol_variants = _symbol_query_variants(order_symbols)
        position_stmt = select(
            Position.id,
            Position.model_name,
            Position.execution_mode,
            Position.symbol,
            Position.side,
            Position.quantity,
            Position.entry_price,
            Position.current_price,
            Position.leverage,
            Position.stop_loss_price,
            Position.take_profit_price,
            Position.realized_pnl,
            Position.is_open,
            Position.created_at,
            Position.closed_at,
        )
        if mode:
            position_stmt = position_stmt.where(Position.execution_mode == mode)
        if position_symbol_variants:
            position_stmt = position_stmt.where(Position.symbol.in_(position_symbol_variants))
        else:
            position_stmt = position_stmt.where(Position.id == -1)
        position_rows = await session.execute(position_stmt.limit(1000))
        all_positions = list(position_rows.all())
        closed_positions = [p for p in all_positions if not p.is_open]

    def normalize_symbol(symbol: str | None) -> str:
        text = str(symbol or "").replace("-", "/").replace("_", "/").upper()
        if ":" in text:
            text = text.split(":", 1)[0]
        if text.endswith("/USDT/SWAP"):
            text = text[: -len("/SWAP")]
        return text

    def display_symbol(symbol: str | None) -> str:
        normalized = normalize_symbol(symbol)
        return normalized or str(symbol or "")

    def aware_dt(value: datetime | None) -> datetime | None:
        parsed = parse_position_time(value)
        if parsed is not None:
            return parsed
        if value and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    def hold_minutes_for_position(position) -> float | None:
        opened = aware_dt(position.created_at)
        closed = aware_dt(position.closed_at)
        if not opened or not closed:
            return None
        return max((closed - opened).total_seconds() / 60.0, 0.0)

    def matching_closed_position(order):
        if order.status != "filled" or not order.filled_at:
            return None
        candidates = []
        for p in closed_positions:
            if (
                p.model_name == order.model_name
                and p.execution_mode == order.execution_mode
                and normalize_symbol(p.symbol) == normalize_symbol(order.symbol)
                and p.closed_at
            ):
                close_side = "buy" if p.side == "short" else "sell"
                if order.side in {"buy", "sell"} and order.side != close_side:
                    continue
                time_delta = abs((p.closed_at - order.filled_at).total_seconds())
                if time_delta > CLOSE_ORDER_POSITION_MATCH_WINDOW_SECONDS:
                    continue
                price_delta = abs(float(order.price or 0) - float(p.current_price or 0))
                qty_delta = abs(float(order.quantity or 0) - float(p.quantity or 0))
                candidates.append((time_delta, price_delta, qty_delta, p))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[0][3]

    def matching_entry_position(order):
        if order.status != "filled" or not order.filled_at:
            return None
        candidates = []
        for p in all_positions:
            if (
                p.model_name != order.model_name
                or p.execution_mode != order.execution_mode
                or normalize_symbol(p.symbol) != normalize_symbol(order.symbol)
                or not p.created_at
            ):
                continue
            entry_side = "buy" if p.side == "long" else "sell"
            if order.side in {"buy", "sell"} and order.side != entry_side:
                continue
            created_at = aware_dt(p.created_at)
            filled_at = aware_dt(order.filled_at)
            if not created_at or not filled_at:
                continue
            time_delta = abs((created_at - filled_at).total_seconds())
            if time_delta > 10:
                continue
            price_delta = abs(float(order.price or 0) - float(p.entry_price or 0))
            qty_delta = abs(float(order.quantity or 0) - float(p.quantity or 0))
            candidates.append((time_delta, price_delta, qty_delta, p))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[0][3]

    def display_action(order) -> str:
        if order.side in {"long", "short", "close_long", "close_short"}:
            return order.side

        action = (decision_meta.get(order.decision_id) or {}).get("action")
        if action and action != "hold":
            return action

        p = matching_closed_position(order)
        if p:
            return "close_long" if p.side == "long" else "close_short"

        if order.decision_id is None:
            p = matching_entry_position(order)
            if p:
                return "long" if p.side == "long" else "short"
            return "short" if order.side == "sell" else "long"

        if order.side in {"buy", "sell"}:
            failure_text = str(order.exchange_order_id or "").lower()
            if "no matching position to close" in failure_text:
                return "close_long" if order.side == "sell" else "close_short"
            return "long" if order.side == "buy" else "short"

        if action:
            return action

        return "long" if order.side == "buy" else "short"

    def close_status_from_pct(value) -> tuple[str | None, str | None]:
        try:
            pct = float(value or 0)
        except (TypeError, ValueError):
            return None, None
        if pct <= 0:
            return None, None
        if pct < 0.999:
            return "partial", "部分平仓"
        return "full", "全部平仓"

    def closed_position_looks_partial(position) -> bool:
        closed_at = aware_dt(position.closed_at)
        created_at = aware_dt(position.created_at)
        if not closed_at or not created_at:
            return False
        if abs((closed_at - created_at).total_seconds()) > 5:
            return False

        symbol_key = normalize_symbol(position.symbol)
        side_key = str(position.side or "").lower()
        entry = float(position.entry_price or 0.0)
        tolerance = max(abs(entry) * 1e-6, 1e-8)
        for sibling in all_positions:
            if sibling.id == position.id:
                continue
            if (
                sibling.model_name != position.model_name
                or sibling.execution_mode != position.execution_mode
            ):
                continue
            if normalize_symbol(sibling.symbol) != symbol_key:
                continue
            if str(sibling.side or "").lower() != side_key:
                continue
            sibling_entry = float(sibling.entry_price or 0.0)
            if entry > 0 and abs(sibling_entry - entry) > tolerance:
                continue
            sibling_created_at = aware_dt(sibling.created_at)
            if sibling_created_at and sibling_created_at < created_at:
                return True
        return False

    def close_status_for_order(order) -> tuple[str | None, str | None]:
        if is_manual_close_order(order):
            return "manual", MANUAL_CLOSE_LABEL
        action = display_action(order)
        if action not in {"close_long", "close_short"}:
            return None, None

        meta = decision_meta.get(order.decision_id) or {}
        by_pct = close_status_from_pct(meta.get("position_size_pct"))
        if by_pct[0]:
            return by_pct

        p = matching_closed_position(order)
        if not p:
            return None, None
        if closed_position_looks_partial(p):
            return "partial", "部分平仓"
        return "full", "全部平仓"

    def display_execution_source(order) -> tuple[str, str]:
        meta = decision_meta.get(order.decision_id) or {}
        raw, snapshot = _decision_raw_and_snapshot(meta)
        if is_manual_close_order(order):
            return "manual", MANUAL_CLOSE_LABEL
        matched_closed = matching_closed_position(order)
        if _is_system_protection_reconcile(raw, snapshot) or (
            _is_exchange_sync_decision(raw, snapshot)
            and _close_price_matches_position_protection(order, matched_closed)
        ):
            return "system", SYSTEM_PROTECTION_LABEL
        if _is_exchange_sync_decision(raw, snapshot):
            return "okx", "OKX同步"
        return "system", "系统执行"

    def meta_for_order(order) -> dict[str, Any]:
        return _safe_dict(decision_meta.get(order.decision_id))

    def raw_for_order(order) -> dict[str, Any]:
        return _safe_dict(meta_for_order(order).get("raw_llm_response"))

    def execution_leverage_for_order(order) -> dict[str, Any]:
        return _safe_dict(raw_for_order(order).get("execution_leverage"))

    def display_leverage_for_order(
        order,
        matched_position,
        execution_leverage: dict[str, Any],
        meta: dict[str, Any],
    ) -> float:
        action = display_action(order)
        if action in {"close_long", "close_short"} and matched_position is not None:
            position_leverage = _safe_float(getattr(matched_position, "leverage", None), 0.0)
            if position_leverage > 0:
                return position_leverage
        return _safe_float(
            execution_leverage.get("actual_leverage") or meta.get("suggested_leverage"),
            1.0,
        )

    def display_reason(order) -> str:
        meta = decision_meta.get(order.decision_id) or {}
        raw, snapshot = _decision_raw_and_snapshot(meta)
        if order.status != "filled":
            return _readable_execution_reason(
                execution_reason=meta.get("execution_reason"),
                reasoning=meta.get("reasoning"),
                exchange_order_id=order.exchange_order_id,
                status=order.status,
            )

        p = matching_closed_position(order)
        if p:
            close_price = order.price or p.current_price or 0
            if is_manual_close_order(order):
                return f"用户手动平仓，成交价 {close_price:g}，实现盈亏 {float(p.realized_pnl or 0):.4f}。"
            system_protection_reconcile = _is_system_protection_reconcile(
                raw,
                snapshot,
            ) or (
                _is_exchange_sync_decision(raw, snapshot)
                and _close_price_matches_position_protection(order, p)
            )
            sl = p.stop_loss_price or 0
            tp = p.take_profit_price or 0
            tolerance = max(abs(close_price) * 0.002, 1e-12)
            if sl and (
                abs(close_price - sl) <= tolerance
                or (p.side == "long" and close_price <= sl)
                or (p.side == "short" and close_price >= sl)
            ):
                prefix = "系统保护单止损触发" if system_protection_reconcile else "OKX 止损触发"
                return f"{prefix}平仓，触发价约 {sl:g}，成交价 {close_price:g}。"
            if tp and (
                abs(close_price - tp) <= tolerance
                or (p.side == "long" and close_price >= tp)
                or (p.side == "short" and close_price <= tp)
            ):
                prefix = "系统保护单止盈触发" if system_protection_reconcile else "OKX 止盈触发"
                return f"{prefix}平仓，触发价约 {tp:g}，成交价 {close_price:g}。"
            if system_protection_reconcile:
                return (
                    f"系统保护单触发平仓，成交价 {close_price:g}，"
                    f"实现盈亏 {float(p.realized_pnl or 0):.4f}。"
                )
            if _is_exchange_sync_decision(raw, snapshot):
                return f"OKX 侧平仓同步，成交价 {close_price:g}，实现盈亏 {float(p.realized_pnl or 0):.4f}。"

        if _is_system_protection_reconcile(raw, snapshot):
            close_fill = _safe_dict(raw.get("close_fill"))
            close_price = order.price or close_fill.get("price") or snapshot.get("exit_price") or 0
            pnl = close_fill.get("pnl") or snapshot.get("realized_pnl")
            pnl_text = (
                f"，实现盈亏 {_safe_float(pnl):.4f}"
                if isinstance(pnl, (int, float))
                or str(pnl or "").replace(".", "", 1).replace("-", "", 1).isdigit()
                else ""
            )
            return f"系统保护单触发平仓，成交价 {_safe_float(close_price):g}{pnl_text}。"

        if _is_exchange_sync_decision(raw, snapshot):
            close_fill = _safe_dict(raw.get("close_fill"))
            close_price = order.price or close_fill.get("price") or snapshot.get("exit_price") or 0
            pnl = close_fill.get("pnl") or snapshot.get("realized_pnl")
            pnl_text = (
                f"，OKX 返回盈亏 {_safe_float(pnl):.4f}"
                if isinstance(pnl, (int, float))
                or str(pnl or "").replace(".", "", 1).replace("-", "", 1).isdigit()
                else ""
            )
            return (
                "OKX 已触发止盈/止损或交易所侧平仓，系统已同步为平仓记录，"
                f"成交价 {_safe_float(close_price):g}{pnl_text}。"
            )

        fast_reason = _fast_risk_reason_from_raw(order.symbol, raw)
        if fast_reason:
            return fast_reason

        hold_minutes = hold_minutes_for_position(p) if p is not None else None
        reasoning = meta.get("reasoning")
        if reasoning and not _text_is_unusable(str(reasoning)):
            return _repair_position_reason_hold_hours(reasoning, hold_minutes)
        if meta.get("execution_reason") and not _text_is_unusable(
            str(meta.get("execution_reason"))
        ):
            return _repair_position_reason_hold_hours(
                _translate_execution_text(meta["execution_reason"]),
                hold_minutes,
            )
        if not p:
            return "系统同步的交易记录。"
        return "系统执行该交易。"

    def matched_position_for_order(order):
        return matching_closed_position(order) or matching_entry_position(order)

    def serialize_trade(order) -> dict[str, Any]:
        source = display_execution_source(order)
        close_status = close_status_for_order(order)
        matched_position = matched_position_for_order(order)
        hold_minutes = (
            hold_minutes_for_position(matched_position) if matched_position is not None else None
        )
        execution_leverage = execution_leverage_for_order(order)
        meta = meta_for_order(order)
        actual_leverage = display_leverage_for_order(
            order,
            matched_position,
            execution_leverage,
            meta,
        )
        ai_suggested_leverage = _safe_float(
            execution_leverage.get("ai_suggested_leverage") or meta.get("suggested_leverage"),
            1.0,
        )
        reason = sanitize_text(display_reason(order))
        failure_kind = _execution_failure_kind(reason)
        execution_status_label = _execution_status_label(order.status, reason)
        return {
            "id": order.id,
            "decision_id": order.decision_id,
            "model_name": order.model_name,
            "mode": order.execution_mode,
            "symbol": order.symbol,
            "display_symbol": display_symbol(order.symbol),
            "side": order.side,
            "action": display_action(order),
            "order_side": order.side,
            "order_type": order.order_type,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status,
            "fee": order.fee,
            "leverage": actual_leverage,
            "ai_suggested_leverage": ai_suggested_leverage,
            "okx_max_leverage": execution_leverage.get("okx_max_leverage"),
            "actual_leverage": actual_leverage,
            "close_status": close_status[0],
            "close_status_label": close_status[1],
            "hold_minutes": round(hold_minutes, 4) if hold_minutes is not None else None,
            "hold_hours": round(hold_minutes / 60.0, 4) if hold_minutes is not None else None,
            "execution_source": source[0],
            "execution_source_label": source[1],
            "reason": reason,
            "detail": reason,
            "execution_failure_kind": failure_kind,
            "execution_status_label": execution_status_label,
            "success": _order_success(order),
            "okx_confirmed": _order_okx_confirmed(order),
            "exchange_order_id": sanitize_text(order.exchange_order_id),
            "okx_inst_id": sanitize_text(getattr(order, "okx_inst_id", None)),
            "okx_trade_ids": sanitize_text(getattr(order, "okx_trade_ids", None)),
            "okx_fill_contracts": getattr(order, "okx_fill_contracts", None),
            "okx_fill_pnl": getattr(order, "okx_fill_pnl", None),
            "okx_state": sanitize_text(getattr(order, "okx_state", None)),
            "okx_sync_status": sanitize_text(getattr(order, "okx_sync_status", None)),
            "okx_synced_at": (
                order.okx_synced_at.isoformat()
                if getattr(order, "okx_synced_at", None)
                else None
            ),
            "okx_last_error": sanitize_text(getattr(order, "okx_last_error", None)),
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
            "created_at": order.created_at.isoformat() if order.created_at else None,
        }

    return {
        "count": len(orders),
        "total": total,
        "filled_total": filled_total,
        "trades": [serialize_trade(o) for o in orders[:limit]],
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size) if total else 1,
    }


@router.get("/trades/{trade_id}")
async def get_trade_detail(trade_id: int):
    """Get a single trade by ID."""
    async with get_session_ctx() as session:
        repo = TradeRepository(session)
        order = await repo.get(trade_id)
        decision = None
        if order and order.decision_id:
            result = await session.execute(
                select(AIDecision).where(AIDecision.id == order.decision_id)
            )
            decision = result.scalar_one_or_none()
        closed_positions: list[Any] = []
        if order:
            position_stmt = select(Position).where(
                Position.model_name == order.model_name,
                Position.execution_mode == order.execution_mode,
                Position.is_open.is_(False),
                Position.symbol.in_(_symbol_query_variants({order.symbol})),
            )
            if order.filled_at:
                filled_at = order.filled_at
                if filled_at.tzinfo is None:
                    filled_at = filled_at.replace(tzinfo=UTC)
                position_stmt = position_stmt.where(
                    Position.closed_at
                    >= filled_at - timedelta(seconds=CLOSE_ORDER_POSITION_MATCH_WINDOW_SECONDS),
                    Position.closed_at
                    <= filled_at + timedelta(seconds=CLOSE_ORDER_POSITION_MATCH_WINDOW_SECONDS),
                )
            position_result = await session.execute(position_stmt.limit(50))
            closed_positions = list(position_result.scalars().all())

    if not order:
        return {"error": "Trade not found"}

    raw_response = _safe_dict(getattr(decision, "raw_llm_response", None))
    execution_reason = sanitize_text(
        localize_execution_reason(getattr(decision, "execution_reason", None))
    )
    fallback_reason = _readable_execution_reason(
        execution_reason=execution_reason,
        reasoning=sanitize_text(getattr(decision, "reasoning", None)),
        exchange_order_id=order.exchange_order_id,
        status=order.status,
    )
    execution_source = _execution_source_from_decision(decision, order)
    matched_closed_positions = _matching_closed_positions_for_order(order, closed_positions)
    if (
        execution_source[0] == "okx"
        and matched_closed_positions
        and _close_price_matches_position_protection(order, matched_closed_positions[0])
    ):
        execution_source = ("system", SYSTEM_PROTECTION_LABEL)
    position_reason = _close_order_position_reason(
        order,
        matched_closed_positions,
        execution_source=execution_source[0],
    )
    if position_reason:
        fallback_reason = sanitize_text(position_reason)
    trace = build_execution_trace(
        raw_response,
        order_status=order.status,
        order_created_at=order.created_at,
        order_filled_at=order.filled_at,
        fallback_reason=fallback_reason,
    )
    failure_kind = _execution_failure_kind(fallback_reason)
    execution_status_label = _execution_status_label(order.status, fallback_reason)

    return sanitize_payload(
        {
            "id": order.id,
            "decision_id": order.decision_id,
            "model_name": order.model_name,
            "mode": order.execution_mode,
            "symbol": order.symbol,
            "side": order.side,
            "action": getattr(decision, "action", None),
            "order_type": order.order_type,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status,
            "fee": order.fee,
            "exchange_order_id": sanitize_text(order.exchange_order_id),
            "execution_source": execution_source[0],
            "execution_source_label": execution_source[1],
            "reason": fallback_reason,
            "detail": fallback_reason,
            "display_reason": fallback_reason,
            "matched_positions": [
                {
                    "id": position.id,
                    "symbol": _normalize_display_symbol(position.symbol),
                    "side": position.side,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "close_price": position.current_price,
                    "leverage": position.leverage,
                    "realized_pnl": position.realized_pnl,
                    "opened_at": position.created_at.isoformat() if position.created_at else None,
                    "closed_at": position.closed_at.isoformat() if position.closed_at else None,
                }
                for position in matched_closed_positions
            ],
            "execution_failure_kind": failure_kind,
            "execution_status_label": execution_status_label,
            "success": _order_success(order),
            "okx_confirmed": _order_okx_confirmed(order),
            "okx_inst_id": sanitize_text(getattr(order, "okx_inst_id", None)),
            "okx_trade_ids": sanitize_text(getattr(order, "okx_trade_ids", None)),
            "okx_fill_contracts": getattr(order, "okx_fill_contracts", None),
            "okx_fill_pnl": getattr(order, "okx_fill_pnl", None),
            "okx_state": sanitize_text(getattr(order, "okx_state", None)),
            "okx_sync_status": sanitize_text(getattr(order, "okx_sync_status", None)),
            "okx_synced_at": (
                order.okx_synced_at.isoformat()
                if getattr(order, "okx_synced_at", None)
                else None
            ),
            "okx_last_error": sanitize_text(getattr(order, "okx_last_error", None)),
            "decision": (
                {
                    "action": getattr(decision, "action", None),
                    "confidence": getattr(decision, "confidence", None),
                    "position_size_pct": getattr(decision, "position_size_pct", None),
                    "suggested_leverage": getattr(decision, "suggested_leverage", None),
                    "reasoning": sanitize_text(getattr(decision, "reasoning", None)),
                    "execution_reason": execution_reason,
                }
                if decision is not None
                else None
            ),
            "execution_steps": trace["execution_steps"],
            "stage_events": trace["stage_events"],
            "final_result": trace["final_result"],
            "failed_step": trace["failed_step"],
            "repair_suggestions": trace["repair_suggestions"],
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
            "created_at": order.created_at.isoformat() if order.created_at else None,
        }
    )


@router.get("/positions")
async def get_positions(mode: str | None = None):
    """Get positions with OKX-native grouped history for closed lifecycles."""
    selected_mode = mode or None
    async with get_session_ctx() as session:
        repo = TradeRepository(session)
        positions = await repo.get_position_records(execution_mode=selected_mode, limit=500)
        closed_positions = [
            p
            for p in positions
            if not p.is_open and is_final_settlement_status(getattr(p, "settlement_status", None))
        ]
        linked_order_ids = {
            token
            for position in closed_positions
            for value in (
                getattr(position, "entry_exchange_order_id", None),
                getattr(position, "close_exchange_order_id", None),
            )
            for token in _split_exchange_order_ids(value)
        }
        order_stmt = select(Order).where(Order.status == "filled")
        if mode:
            order_stmt = order_stmt.where(Order.execution_mode == mode)
        if linked_order_ids:
            order_stmt = order_stmt.where(Order.exchange_order_id.in_(sorted(linked_order_ids)))
        else:
            order_stmt = order_stmt.where(Order.id == -1)
        order_rows = list((await session.execute(order_stmt.limit(10000))).scalars().all())
        account_bill_rows = await _okx_account_bill_rows_for_closed_positions(
            session,
            closed_positions=closed_positions,
            mode=selected_mode,
        )
        position_history_rows = await dashboard_api._dashboard_okx_position_history_rows(
            mode=selected_mode,
            closed_rows=closed_positions,
        )

    closed_ledger_rows = [
        group.as_dict(include_fills=True)
        for group in build_okx_position_ledger_groups(
            closed_positions,
            order_rows,
            account_bills=account_bill_rows,
            position_history_rows=position_history_rows,
        )
    ]
    try:
        open_rows = await dashboard_api._get_display_open_positions_snapshot(selected_mode)
    except Exception:
        open_rows = [_serialize_open_position_row(p) for p in positions if p.is_open]

    return {
        "count": len(open_rows) + len(closed_ledger_rows),
        "open_count": len(open_rows),
        "closed_count": len(closed_ledger_rows),
        "positions": [*open_rows, *closed_ledger_rows],
        "ledger_source": "okx_current_positions_plus_grouped_closed_cache",
    }


def _serialize_open_position_row(p: Position) -> dict[str, Any]:
    return {
        "id": p.id,
        "model_name": p.model_name,
        "mode": p.execution_mode,
        "symbol": p.symbol,
        "side": p.side,
        "quantity": p.quantity,
        "entry_price": p.entry_price,
        "current_price": p.current_price,
        "unrealized_pnl": p.unrealized_pnl,
        "realized_pnl": p.realized_pnl,
        "leverage": p.leverage,
        "stop_loss": p.stop_loss_price,
        "take_profit": p.take_profit_price,
        "is_open": True,
        "close_status": "open",
        "close_status_label": "持有中",
        "trade_fact_trusted": True,
        "trade_fact_untrusted_reason": None,
        "okx_inst_id": p.okx_inst_id,
        "okx_pos_id": p.okx_pos_id,
        "entry_exchange_order_id": p.entry_exchange_order_id,
        "close_exchange_order_id": p.close_exchange_order_id,
        "opened_at": p.created_at.isoformat() if p.created_at else None,
        "closed_at": None,
        "ledger_source": "okx_current_position_cache",
    }


def _split_exchange_order_ids(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    tokens = {text}
    for separator in (",", ";", "|", "\n", "\t", " "):
        pieces: set[str] = set()
        for token in tokens:
            pieces.update(part.strip() for part in token.split(separator) if part.strip())
        tokens = pieces
    return {token for token in tokens if token}


async def _okx_account_bill_rows_for_closed_positions(
    session: Any,
    *,
    closed_positions: list[Position],
    mode: str | None,
) -> list[OkxAccountBill]:
    if not closed_positions:
        return []
    inst_ids = {
        str(getattr(position, "okx_inst_id", "") or "").strip().upper()
        for position in closed_positions
        if str(getattr(position, "okx_inst_id", "") or "").strip()
    }
    opened_values = [
        value
        for position in closed_positions
        if (value := _as_utc_datetime(getattr(position, "created_at", None))) is not None
    ]
    closed_values = [
        value
        for position in closed_positions
        if (value := _as_utc_datetime(getattr(position, "closed_at", None))) is not None
    ]
    if not inst_ids or not opened_values or not closed_values:
        return []
    start = min(opened_values) - timedelta(hours=1)
    end = max(closed_values) + timedelta(hours=1)
    stmt = select(OkxAccountBill).where(
        OkxAccountBill.inst_id.in_(sorted(inst_ids)),
        OkxAccountBill.bill_ts >= start.replace(tzinfo=None),
        OkxAccountBill.bill_ts <= end.replace(tzinfo=None),
    )
    if mode:
        stmt = stmt.where(OkxAccountBill.mode == ("live" if mode == "live" else "paper"))
    result = await session.execute(
        stmt.order_by(OkxAccountBill.bill_ts.asc(), OkxAccountBill.id.asc()).limit(10000)
    )
    return list(result.scalars().all())


@router.delete(
    "/trades",
    dependencies=[Depends(require_destructive_dashboard_confirmation)],
)
async def clear_trades():
    """Delete all trade/order records."""
    from db.repositories.trade_repo import TradeRepository
    from db.session import get_session_ctx

    async with get_session_ctx() as session:
        repo = TradeRepository(session)
        count = await repo.delete_all()
        await session.commit()

    return {"status": "ok", "message": f"Deleted {count} trade records", "deleted": count}


@router.get("/risk/events")
async def get_risk_events(limit: int = Query(50, ge=1, le=500)):
    """Get recent risk events."""
    async with get_session_ctx() as session:
        repo = RiskRepository(session)
        events = await repo.get_recent_events(limit=limit)

    return {
        "count": len(events),
        "events": [
            {
                "id": e.id,
                "event_type": e.event_type,
                "severity": e.severity,
                "symbol": e.symbol,
                "details": e.details,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ],
    }
