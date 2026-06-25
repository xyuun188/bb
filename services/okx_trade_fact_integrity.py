"""Read-only OKX/local trade fact integrity audit.

The local order table stores filled base quantity, while OKX execution payloads
often expose contract counts plus contract size.  This audit keeps that
conversion explicit so symbol aliases or quantity scale issues cannot silently
pollute position history, server-profit learning, or dashboard PnL.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select

from core.symbols import (
    normalize_trading_symbol,
    symbol_from_okx_inst_id,
    trading_symbol_variants,
)
from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.trade import Order, Position

DEFAULT_LOOKBACK_HOURS = 72
DEFAULT_LIMIT = 500
SYMBOL_MISMATCH_SEVERITY = "critical"
POSITION_SYMBOL_MISMATCH_SEVERITY = "critical"
QUANTITY_MISMATCH_SEVERITY = "critical"
PRICE_MISMATCH_SEVERITY = "warning"
NOTIONAL_MISMATCH_SEVERITY = "warning"
ORDER_POSITION_MISSING_SEVERITY = "warning"
QUANTITY_TOLERANCE_RATIO = 0.02
PRICE_TOLERANCE_RATIO = 0.01
NOTIONAL_TOLERANCE_RATIO = 0.05
POSITION_MATCH_WINDOW = timedelta(minutes=10)


@dataclass(frozen=True, slots=True)
class TradeFactIssue:
    kind: str
    severity: str
    order_id: int | None = None
    decision_id: int | None = None
    position_id: int | None = None
    symbol: str = ""
    expected_symbol: str = ""
    order_quantity: float | None = None
    raw_contracts: float | None = None
    contract_size: float | None = None
    expected_base_quantity: float | None = None
    order_price: float | None = None
    raw_price: float | None = None
    order_notional: float | None = None
    expected_notional: float | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "order_id": self.order_id,
            "decision_id": self.decision_id,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "expected_symbol": self.expected_symbol,
            "order_quantity": _round_optional(self.order_quantity),
            "raw_contracts": _round_optional(self.raw_contracts),
            "contract_size": _round_optional(self.contract_size),
            "expected_base_quantity": _round_optional(self.expected_base_quantity),
            "order_price": _round_optional(self.order_price),
            "raw_price": _round_optional(self.raw_price),
            "order_notional": _round_optional(self.order_notional),
            "expected_notional": _round_optional(self.expected_notional),
            "reason": self.reason,
        }


class OkxTradeFactIntegrityService:
    """Compare local order/position rows with authoritative OKX execution facts."""

    def __init__(
        self,
        *,
        lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
        limit: int = DEFAULT_LIMIT,
    ) -> None:
        self.lookback_hours = max(int(lookback_hours or DEFAULT_LOOKBACK_HOURS), 1)
        self.limit = max(1, min(int(limit or DEFAULT_LIMIT), 5000))

    async def audit(self) -> dict[str, Any]:
        since = datetime.now(UTC) - timedelta(hours=self.lookback_hours)
        since_naive = since.replace(tzinfo=None)
        async with get_read_session_ctx() as session:
            order_rows = await session.execute(
                select(Order)
                .where(
                    or_(Order.created_at >= since_naive, Order.filled_at >= since_naive),
                    Order.status == "filled",
                )
                .order_by(Order.created_at.desc())
                .limit(self.limit)
            )
            orders = list(order_rows.scalars().all())
            decision_ids = {int(order.decision_id) for order in orders if order.decision_id}
            decisions: dict[int, AIDecision] = {}
            if decision_ids:
                decision_rows = await session.execute(
                    select(AIDecision).where(AIDecision.id.in_(decision_ids))
                )
                decisions = {int(decision.id): decision for decision in decision_rows.scalars()}
            position_rows = await session.execute(
                select(Position)
                .where(or_(Position.created_at >= since_naive, Position.closed_at >= since_naive))
                .order_by(Position.created_at.desc())
                .limit(self.limit)
            )
            positions = list(position_rows.scalars().all())

        issues: list[TradeFactIssue] = []
        for order in orders:
            decision = decisions.get(int(order.decision_id or 0))
            execution_result = _execution_result_payload(decision)
            raw = _execution_raw_response(execution_result)
            if raw:
                issues.extend(
                    self._audit_order_against_raw(
                        order,
                        decision,
                        raw,
                        execution_result,
                    )
                )
            issues.extend(
                self._audit_order_position_alignment(
                    order,
                    decision,
                    raw,
                    positions,
                )
            )

        return _summary(
            issues,
            checked_orders=len(orders),
            checked_positions=len(positions),
            lookback_hours=self.lookback_hours,
        )

    def _audit_order_against_raw(
        self,
        order: Order,
        decision: AIDecision | None,
        raw: dict[str, Any],
        execution_result: dict[str, Any] | None = None,
    ) -> list[TradeFactIssue]:
        issues: list[TradeFactIssue] = []
        local_symbol = normalize_trading_symbol(order.symbol)
        raw_symbol = _raw_exchange_symbol(raw, fallback=local_symbol)
        if raw_symbol and local_symbol and raw_symbol != local_symbol:
            issues.append(
                TradeFactIssue(
                    kind="symbol_alias_mismatch",
                    severity=SYMBOL_MISMATCH_SEVERITY,
                    order_id=int(order.id),
                    decision_id=int(order.decision_id or 0) or None,
                    symbol=local_symbol,
                    expected_symbol=raw_symbol,
                    reason="OKX instId/raw payload symbol differs from local order symbol.",
                )
            )

        contract_size = _first_positive(
            raw.get("contract_size"),
            raw.get("contractSize"),
            _nested(raw, "info", "ctVal"),
            default=1.0,
        )
        raw_contracts = _first_positive(
            raw.get("filled_contracts"),
            raw.get("order_contracts"),
            raw.get("filled"),
            raw.get("amount"),
            _nested(raw, "info", "accFillSz"),
            _nested(raw, "info", "fillSz"),
            _nested(raw, "info", "sz"),
            default=0.0,
        )
        local_quantity = _safe_float(order.quantity)
        expected_base_quantity = raw_contracts * contract_size if raw_contracts > 0 else 0.0
        if (
            local_quantity > 0
            and expected_base_quantity > 0
            and not _relative_close_enough(
                local_quantity,
                expected_base_quantity,
                QUANTITY_TOLERANCE_RATIO,
            )
        ):
            issues.append(
                TradeFactIssue(
                    kind="contract_base_quantity_mismatch",
                    severity=QUANTITY_MISMATCH_SEVERITY,
                    order_id=int(order.id),
                    decision_id=int(order.decision_id or 0) or None,
                    symbol=local_symbol,
                    expected_symbol=raw_symbol or local_symbol,
                    order_quantity=local_quantity,
                    raw_contracts=raw_contracts,
                    contract_size=contract_size,
                    expected_base_quantity=expected_base_quantity,
                    reason="Local order quantity does not equal OKX filled contracts converted by contract size.",
                )
            )

        local_price = _safe_float(order.price)
        raw_price = _execution_fact_price(raw, execution_result)
        if (
            local_price > 0
            and raw_price > 0
            and not _relative_close_enough(local_price, raw_price, PRICE_TOLERANCE_RATIO)
        ):
            issues.append(
                TradeFactIssue(
                    kind="execution_price_mismatch",
                    severity=PRICE_MISMATCH_SEVERITY,
                    order_id=int(order.id),
                    decision_id=int(order.decision_id or 0) or None,
                    symbol=local_symbol,
                    expected_symbol=raw_symbol or local_symbol,
                    order_price=local_price,
                    raw_price=raw_price,
                    reason="Local order price differs from OKX average/fill price.",
                )
            )

        local_notional = local_quantity * local_price
        expected_notional = expected_base_quantity * (raw_price or local_price)
        if (
            local_notional > 0
            and expected_notional > 0
            and not _relative_close_enough(
                local_notional,
                expected_notional,
                NOTIONAL_TOLERANCE_RATIO,
            )
        ):
            issues.append(
                TradeFactIssue(
                    kind="notional_mismatch",
                    severity=NOTIONAL_MISMATCH_SEVERITY,
                    order_id=int(order.id),
                    decision_id=int(order.decision_id or 0) or None,
                    symbol=local_symbol,
                    expected_symbol=raw_symbol or local_symbol,
                    order_quantity=local_quantity,
                    raw_contracts=raw_contracts,
                    contract_size=contract_size,
                    expected_base_quantity=expected_base_quantity,
                    order_price=local_price,
                    raw_price=raw_price or None,
                    order_notional=local_notional,
                    expected_notional=expected_notional,
                    reason="Local order notional differs from OKX contracts * contract size * fill price.",
                )
            )
        return issues

    def _audit_order_position_alignment(
        self,
        order: Order,
        decision: AIDecision | None,
        raw: dict[str, Any],
        positions: list[Position],
    ) -> list[TradeFactIssue]:
        if decision is None or not order.decision_id:
            return []
        action = str(decision.action or "").lower()
        side = _position_side_for_action(action)
        if side is None:
            return []
        local_symbol = normalize_trading_symbol(order.symbol)
        related_positions = _related_positions_for_order(
            order,
            decision,
            raw,
            positions,
            action=action,
            side=side,
        )
        issues: list[TradeFactIssue] = []
        if not related_positions:
            issues.append(
                TradeFactIssue(
                    kind="order_position_missing",
                    severity=ORDER_POSITION_MISSING_SEVERITY,
                    order_id=int(order.id),
                    decision_id=int(order.decision_id),
                    symbol=local_symbol,
                    expected_symbol=local_symbol,
                    reason=(
                        "Filled entry/exit order has no matching local position in the "
                        "model/mode/side/time window. Check whether position persistence "
                        "or historical repair skipped this exchange-confirmed order."
                    ),
                )
            )
            return issues

        for position in related_positions:
            position_symbol = normalize_trading_symbol(position.symbol)
            if position_symbol and local_symbol and position_symbol != local_symbol:
                issues.append(
                    TradeFactIssue(
                        kind="order_position_symbol_mismatch",
                        severity=POSITION_SYMBOL_MISMATCH_SEVERITY,
                        order_id=int(order.id),
                        decision_id=int(order.decision_id),
                        position_id=int(position.id),
                        symbol=position_symbol,
                        expected_symbol=local_symbol,
                        reason="Position created/closed by the decision uses a different symbol than the filled order.",
                    )
                )
        return issues


def _execution_result_payload(decision: AIDecision | None) -> dict[str, Any]:
    raw = getattr(decision, "raw_llm_response", None)
    raw = raw if isinstance(raw, dict) else {}
    execution_result = raw.get("execution_result")
    return execution_result if isinstance(execution_result, dict) else {}


def _execution_raw_response(execution_result: dict[str, Any]) -> dict[str, Any]:
    raw_response = execution_result.get("raw_response")
    if isinstance(raw_response, dict):
        return raw_response
    return {}


def _execution_fact_price(
    raw: dict[str, Any],
    execution_result: dict[str, Any] | None = None,
) -> float:
    """Return the order-level execution price for this raw OKX fact.

    Split close orders store the last child order at the top level because the
    executor merges ``last_order`` into raw_response.  The authoritative parent
    fill price is the weighted average of ``split_chunks`` by closed contracts.
    """
    if isinstance(raw, dict) and raw.get("split_exit_order"):
        split_price = _weighted_split_exit_price(raw.get("split_chunks"))
        if split_price > 0:
            return split_price
        result_price = _first_positive(
            (execution_result or {}).get("price"),
            default=0.0,
        )
        if result_price > 0:
            return result_price
    return _first_positive(
        raw.get("average"),
        raw.get("avgPx"),
        raw.get("price"),
        raw.get("px"),
        _nested(raw, "info", "avgPx"),
        _nested(raw, "info", "fillPx"),
        default=0.0,
    )


def _weighted_split_exit_price(chunks: Any) -> float:
    if not isinstance(chunks, list):
        return 0.0
    total = 0.0
    contracts = 0.0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        closed_contracts = _safe_float(chunk.get("closed_contracts"))
        price = _safe_float(chunk.get("price"))
        if closed_contracts <= 0 or price <= 0:
            continue
        total += closed_contracts * price
        contracts += closed_contracts
    return total / contracts if contracts > 0 else 0.0


def _raw_exchange_symbol(raw: dict[str, Any], *, fallback: Any = "") -> str:
    data = raw if isinstance(raw, dict) else {}
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    for candidate in (
        info.get("instId"),
        data.get("instId"),
        data.get("okx_inst_id"),
        data.get("okx_symbol"),
    ):
        symbol = symbol_from_okx_inst_id(candidate)
        if symbol:
            return symbol
    explicit = normalize_trading_symbol(data.get("canonical_exchange_symbol"))
    if explicit:
        return explicit
    symbol = normalize_trading_symbol(data.get("symbol"))
    if symbol:
        return symbol
    return normalize_trading_symbol(fallback)


def _related_positions_for_order(
    order: Order,
    decision: AIDecision,
    raw: dict[str, Any],
    positions: list[Position],
    *,
    action: str,
    side: str,
) -> list[Position]:
    order_time = _order_time(order)
    if order_time is None:
        return []
    entry_action = action in {"long", "short"}
    local_symbol = normalize_trading_symbol(order.symbol)
    expected_symbols = set()
    for symbol in (
        local_symbol,
        normalize_trading_symbol(getattr(decision, "symbol", "")),
        _raw_exchange_symbol(raw, fallback=local_symbol),
    ):
        expected_symbols.update(trading_symbol_variants(symbol))
    expected_symbols = {normalize_trading_symbol(symbol) for symbol in expected_symbols if symbol}

    matches: list[tuple[float, Position]] = []
    for position in positions:
        if str(position.model_name or "") != str(order.model_name or ""):
            continue
        if str(position.execution_mode or "") != str(order.execution_mode or ""):
            continue
        if str(position.side or "").lower() != side:
            continue
        position_time = _position_match_time(position, entry_action=entry_action)
        if position_time is None:
            continue
        time_delta = abs((position_time - order_time).total_seconds())
        if time_delta > POSITION_MATCH_WINDOW.total_seconds():
            continue

        position_symbol = normalize_trading_symbol(position.symbol)
        symbol_matches = bool(position_symbol and position_symbol in expected_symbols)
        price_matches = _position_price_matches_order(position, order, entry_action=entry_action)
        quantity_matches = _relative_close_enough(
            abs(_safe_float(position.quantity)),
            abs(_safe_float(order.quantity)),
            QUANTITY_TOLERANCE_RATIO,
        )
        if not (symbol_matches or price_matches or quantity_matches):
            continue
        score = time_delta
        if not symbol_matches:
            score += 60.0
        if not price_matches:
            score += 30.0
        if not quantity_matches:
            score += 15.0
        matches.append((score, position))
    matches.sort(key=lambda item: item[0])
    return [position for _score, position in matches[:5]]


def _order_time(order: Order) -> datetime | None:
    return _ensure_aware(getattr(order, "filled_at", None) or getattr(order, "created_at", None))


def _position_match_time(position: Position, *, entry_action: bool) -> datetime | None:
    if entry_action:
        return _ensure_aware(getattr(position, "created_at", None))
    return _ensure_aware(getattr(position, "closed_at", None))


def _position_price_matches_order(position: Position, order: Order, *, entry_action: bool) -> bool:
    order_price = _safe_float(order.price)
    if order_price <= 0:
        return False
    position_price = _safe_float(position.entry_price if entry_action else position.current_price)
    if position_price <= 0:
        return False
    return _relative_close_enough(position_price, order_price, PRICE_TOLERANCE_RATIO)


def _ensure_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _position_side_for_action(action: str) -> str | None:
    if action in {"long", "close_long"}:
        return "long"
    if action in {"short", "close_short"}:
        return "short"
    return None


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_positive(*values: Any, default: float = 0.0) -> float:
    for value in values:
        number = _safe_float(value, 0.0)
        if number > 0:
            return number
    return default


def _close_enough(left: float, right: float, tolerance_ratio: float) -> bool:
    tolerance = max(abs(left), abs(right), 1.0) * max(tolerance_ratio, 0.0)
    return abs(left - right) <= tolerance


def _relative_close_enough(left: float, right: float, tolerance_ratio: float) -> bool:
    tolerance = max(abs(left), abs(right), 1e-12) * max(tolerance_ratio, 0.0)
    return abs(left - right) <= tolerance


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 8)


def _summary(
    issues: list[TradeFactIssue],
    *,
    checked_orders: int,
    checked_positions: int,
    lookback_hours: int,
) -> dict[str, Any]:
    severity_counts = Counter(issue.severity for issue in issues)
    kind_counts = Counter(issue.kind for issue in issues)
    critical_count = int(severity_counts.get("critical", 0))
    warning_count = int(severity_counts.get("warning", 0))
    status = "critical" if critical_count else "warning" if warning_count else "ok"
    return {
        "read_only": True,
        "status": status,
        "lookback_hours": lookback_hours,
        "checked_orders": int(checked_orders),
        "checked_positions": int(checked_positions),
        "issue_count": len(issues),
        "critical_count": critical_count,
        "warning_count": warning_count,
        "severity_counts": dict(severity_counts.most_common()),
        "kind_counts": dict(kind_counts.most_common()),
        "issues": [issue.as_dict() for issue in issues[:20]],
        "diagnostic_boundary": (
            "Read-only trade fact integrity audit. Local order quantity is base quantity; "
            "OKX filled_contracts must be converted by contract_size before comparison. "
            "Do not apply historical repairs from this report without a separate backup and dry-run."
        ),
    }
