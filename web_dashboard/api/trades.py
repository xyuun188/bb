"""
Trade history API endpoints.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from db.repositories.risk_repo import RiskRepository
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from models.decision import AIDecision
from models.trade import Position
from services.decision_execution_trace import build_execution_trace
from services.execution_result_classifier import ExecutionResultClassifier
from services.manual_close_marker import MANUAL_CLOSE_LABEL, is_manual_close_order
from services.okx_error_classifier import is_okx_temporary_service_error
from services.position_open_time import parse_position_time
from web_dashboard.api.security import require_destructive_dashboard_confirmation
from web_dashboard.api.text_sanitize import looks_mojibake, sanitize_payload, sanitize_text

router = APIRouter()
EXECUTION_REASON_CLASSIFIER = ExecutionResultClassifier()


def _normalize_display_symbol(symbol: str | None) -> str:
    text = str(symbol or "").replace("-", "/").replace("_", "/").upper()
    if ":" in text:
        text = text.split(":", 1)[0]
    if text.endswith("/USDT/SWAP"):
        text = text[: -len("/SWAP")]
    return text


def _symbol_query_variants(symbols: set[str]) -> set[str]:
    variants: set[str] = set()
    for symbol in symbols:
        normalized = _normalize_display_symbol(symbol)
        if not normalized:
            continue
        variants.update(
            {
                symbol,
                normalized,
                normalized.replace("/", "-"),
                f"{normalized}:USDT",
                f"{normalized}-SWAP",
                f"{normalized.replace('/', '-')}-SWAP",
            }
        )
    return variants


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
    text = str(message or "").strip()
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


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
                    "execution_reason": row.execution_reason,
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
                if time_delta > 10:
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
        if is_manual_close_order(order):
            return "manual", MANUAL_CLOSE_LABEL
        meta = decision_meta.get(order.decision_id) or {}
        raw = meta.get("raw_llm_response") or {}
        snapshot = meta.get("feature_snapshot") or {}
        if not isinstance(raw, dict):
            raw = {}
        if not isinstance(snapshot, dict):
            snapshot = {}

        source = str(raw.get("source") or snapshot.get("source") or "").lower()
        if raw.get("system_sync") or source in {"okx_position_reconcile", "okx_tp_sl_backfill"}:
            return "okx", "OKX执行"

        if order.decision_id is None and matching_closed_position(order):
            return "okx", "OKX执行"

        return "system", "系统执行"

    def meta_for_order(order) -> dict[str, Any]:
        return _safe_dict(decision_meta.get(order.decision_id))

    def raw_for_order(order) -> dict[str, Any]:
        return _safe_dict(meta_for_order(order).get("raw_llm_response"))

    def execution_leverage_for_order(order) -> dict[str, Any]:
        return _safe_dict(raw_for_order(order).get("execution_leverage"))

    def display_reason(order) -> str:
        meta = decision_meta.get(order.decision_id) or {}
        raw = meta.get("raw_llm_response") or {}
        snapshot = meta.get("feature_snapshot") or {}
        if not isinstance(raw, dict):
            raw = {}
        if not isinstance(snapshot, dict):
            snapshot = {}
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
            sl = p.stop_loss_price or 0
            tp = p.take_profit_price or 0
            tolerance = max(abs(close_price) * 0.002, 1e-12)
            if sl and (
                abs(close_price - sl) <= tolerance
                or (p.side == "long" and close_price <= sl)
                or (p.side == "short" and close_price >= sl)
            ):
                return f"OKX 止损触发平仓，触发价约 {sl:g}，成交价 {close_price:g}。"
            if tp and (
                abs(close_price - tp) <= tolerance
                or (p.side == "long" and close_price >= tp)
                or (p.side == "short" and close_price <= tp)
            ):
                return f"OKX 止盈触发平仓，触发价约 {tp:g}，成交价 {close_price:g}。"
            if order.decision_id is None or raw.get("system_sync"):
                return f"OKX 侧平仓同步，成交价 {close_price:g}，实现盈亏 {float(p.realized_pnl or 0):.4f}。"

        if (
            raw.get("system_sync")
            or str(raw.get("source") or snapshot.get("source") or "").lower()
            == "okx_position_reconcile"
        ):
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
        actual_leverage = _safe_float(
            execution_leverage.get("actual_leverage") or meta.get("suggested_leverage"),
            1.0,
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
            "success": order.status == "filled",
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

    if not order:
        return {"error": "Trade not found"}

    raw_response = _safe_dict(getattr(decision, "raw_llm_response", None))
    execution_reason = sanitize_text(getattr(decision, "execution_reason", None))
    fallback_reason = _readable_execution_reason(
        execution_reason=execution_reason,
        reasoning=sanitize_text(getattr(decision, "reasoning", None)),
        exchange_order_id=order.exchange_order_id,
        status=order.status,
    )
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
            "reason": fallback_reason,
            "detail": fallback_reason,
            "display_reason": fallback_reason,
            "execution_failure_kind": failure_kind,
            "execution_status_label": execution_status_label,
            "success": order.status == "filled",
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
    """Get persisted position records, including closed positions."""
    async with get_session_ctx() as session:
        repo = TradeRepository(session)
        positions = await repo.get_position_records(execution_mode=mode, limit=500)

    return {
        "count": len(positions),
        "positions": [
            {
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
                "is_open": p.is_open,
                "opened_at": p.created_at.isoformat() if p.created_at else None,
                "closed_at": p.closed_at.isoformat() if p.closed_at else None,
            }
            for p in positions
        ],
    }


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
