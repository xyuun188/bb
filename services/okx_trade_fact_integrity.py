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

from sqlalchemy import or_, select, text

from core.symbols import (
    normalize_trading_symbol,
    symbol_from_okx_inst_id,
    trading_symbol_variants,
)
from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.trade import Order, Position
from services.manual_close_marker import (
    ORPHAN_QUARANTINE_EXCHANGE_ID_PREFIX,
    is_local_non_exchange_close_marker,
    is_manual_close_order,
)
from services.training_epoch import load_training_epoch_start

DEFAULT_LOOKBACK_HOURS = 72
DEFAULT_LIMIT = 500
OKX_AUTHORITATIVE_POSITION_MODEL = "okx_authoritative_sync"
SYMBOL_MISMATCH_SEVERITY = "critical"
POSITION_SYMBOL_MISMATCH_SEVERITY = "critical"
QUANTITY_MISMATCH_SEVERITY = "critical"
PRICE_MISMATCH_SEVERITY = "warning"
NOTIONAL_MISMATCH_SEVERITY = "warning"
CONTRACT_SPECIFICATION_EVIDENCE_MISSING_SEVERITY = "warning"
ORDER_POSITION_MISSING_SEVERITY = "warning"
POSITION_LINK_MISSING_SEVERITY = "critical"
POSITION_LINK_MISMATCH_SEVERITY = "critical"
ORPHAN_QUARANTINE_SEVERITY = "info"
POSITION_LINK_ORDER_MISSING_SEVERITY = "warning"
HISTORICAL_LINK_ORDER_MISSING_OBSERVATION_DAYS = 3
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
        checked_at = datetime.now(UTC)
        nominal_since = checked_at - timedelta(hours=self.lookback_hours)
        epoch_started_at = load_training_epoch_start()
        since = max(nominal_since, epoch_started_at)
        since_naive = since.replace(tzinfo=None)
        async with get_read_session_ctx() as session:
            await _start_consistent_read_snapshot(session)
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
                .where(
                    or_(
                        Position.created_at >= since_naive,
                        Position.closed_at >= since_naive,
                        Position.is_open.is_(True),
                    )
                )
                .order_by(Position.created_at.desc())
                .limit(self.limit)
            )
            positions = list(position_rows.scalars().all())
            linked_order_ids = _position_linked_order_ids(positions)
            linked_orders: list[Order] = []
            if linked_order_ids:
                linked_order_rows = await session.execute(
                    select(Order).where(
                        Order.exchange_order_id.in_(sorted(linked_order_ids)),
                        Order.status == "filled",
                    )
                )
                linked_orders = list(linked_order_rows.scalars().all())

        issues: list[TradeFactIssue] = []
        for order in orders:
            decision = decisions.get(int(order.decision_id or 0))
            execution_result = _execution_result_payload(decision)
            raw = _order_execution_raw(order, execution_result)
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
        position_link_orders = _dedupe_orders_by_id([*orders, *linked_orders])
        issues.extend(
            self._audit_position_authority_links(
                positions,
                position_link_orders,
                since=since,
            )
        )

        return _summary(
            issues,
            checked_orders=len(orders),
            checked_positions=len(positions),
            lookback_hours=self.lookback_hours,
            nominal_since=nominal_since,
            since=since,
            epoch_started_at=epoch_started_at,
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

        contract_size_verified = bool(
            raw.get("contract_size_verified") is True
            and str(raw.get("contract_size_source") or "").strip()
            == "okx_public_instruments"
        )
        contract_size = (
            _first_positive(raw.get("contract_size"), default=0.0)
            if contract_size_verified
            else 0.0
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
        raw_base_quantity = _first_positive(
            raw.get("base_quantity"),
            raw.get("baseQuantity"),
            raw.get("filled_base_quantity"),
            raw.get("filledBaseQuantity"),
            default=0.0,
        )
        expected_base_quantity = (
            raw_contracts * contract_size
            if raw_contracts > 0 and contract_size > 0
            else 0.0
        )
        if local_quantity > 0 and raw_contracts > 0 and expected_base_quantity <= 0:
            issues.append(
                TradeFactIssue(
                    kind="contract_specification_evidence_missing",
                    severity=CONTRACT_SPECIFICATION_EVIDENCE_MISSING_SEVERITY,
                    order_id=int(order.id),
                    decision_id=int(order.decision_id or 0) or None,
                    symbol=local_symbol,
                    expected_symbol=raw_symbol or local_symbol,
                    order_quantity=local_quantity,
                    raw_contracts=raw_contracts,
                    reason=(
                        "OKX execution exposes contract counts but has no verified "
                        "okx_public_instruments contract size; quantity and notional "
                        "comparisons are disabled."
                    ),
                )
            )
        if (
            local_quantity > 0
            and expected_base_quantity > 0
            and (
                not _relative_close_enough(
                    local_quantity,
                    expected_base_quantity,
                    QUANTITY_TOLERANCE_RATIO,
                )
                or (
                    raw_base_quantity > 0
                    and not _relative_close_enough(
                        raw_base_quantity,
                        expected_base_quantity,
                        QUANTITY_TOLERANCE_RATIO,
                    )
                )
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
        local_symbol = _order_authoritative_symbol(order, raw)
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
            position_symbol = _position_authoritative_symbol(position)
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
                        reason=(
                            "Position created/closed by the decision uses a different "
                            "OKX-native instrument than the filled order."
                        ),
                    )
                )
        return issues

    def _audit_position_authority_links(
        self,
        positions: list[Position],
        orders: list[Order],
        *,
        since: datetime,
    ) -> list[TradeFactIssue]:
        issues: list[TradeFactIssue] = []
        exchange_orders_by_id: dict[str, list[Order]] = {}
        for order in orders:
            if is_manual_close_order(order):
                continue
            for exchange_order_id in _split_exchange_order_ids(order.exchange_order_id):
                exchange_orders_by_id.setdefault(exchange_order_id, []).append(order)

        for position in positions:
            position_symbol = normalize_trading_symbol(position.symbol)
            okx_inst_id = str(getattr(position, "okx_inst_id", "") or "").strip().upper()
            if okx_inst_id:
                expected_symbol = symbol_from_okx_inst_id(okx_inst_id)
                if expected_symbol and position_symbol and expected_symbol != position_symbol:
                    issues.append(
                        TradeFactIssue(
                            kind="position_okx_inst_id_symbol_mismatch",
                            severity=POSITION_LINK_MISMATCH_SEVERITY,
                            position_id=int(position.id),
                            symbol=position_symbol,
                            expected_symbol=expected_symbol,
                            reason=(
                                "Position okx_inst_id points to a different OKX instrument "
                                "than the local position symbol."
                            ),
                        )
                    )

            raw_entry_ids = _split_exchange_order_ids(
                getattr(position, "entry_exchange_order_id", None)
            )
            raw_close_ids = _split_exchange_order_ids(
                getattr(position, "close_exchange_order_id", None)
            )
            local_marker_ids = {
                item for item in raw_close_ids if is_local_non_exchange_close_marker(item)
            }
            entry_ids = {
                item for item in raw_entry_ids if not is_local_non_exchange_close_marker(item)
            }
            close_ids = raw_close_ids - local_marker_ids
            if (
                str(getattr(position, "settlement_status", "") or "")
                == "superseded_position_residual"
            ):
                issues.append(
                    TradeFactIssue(
                        kind="superseded_position_residual",
                        severity="info",
                        position_id=int(position.id),
                        symbol=position_symbol,
                        expected_symbol=position_symbol,
                        reason=(
                            "This preserved local row was retired after another row became "
                            "the canonical projection for the same OKX position lifecycle."
                        ),
                    )
                )
                continue
            if _is_superseded_position_residual(position, positions):
                if not entry_ids or (
                    not bool(position.is_open)
                    and _safe_float(getattr(position, "realized_pnl", None), 0.0) != 0.0
                    and not close_ids
                    and not local_marker_ids
                ):
                    issues.append(
                        TradeFactIssue(
                            kind="superseded_position_residual",
                            severity="info",
                            position_id=int(position.id),
                            symbol=position_symbol,
                            expected_symbol=position_symbol,
                            reason=(
                                "A more complete OKX-authoritative lifecycle row covers this "
                                "legacy split/residual position; keep it excluded from training "
                                "and dashboard truth instead of treating it as an active blocker."
                            ),
                        )
                    )
                    continue
            if (
                not bool(position.is_open)
                and str(getattr(position, "model_name", "") or "")
                == "okx_authoritative_sync"
                and str(getattr(position, "settlement_status", "") or "")
                == "okx_position_history"
                and (not entry_ids or not close_ids)
            ):
                issues.append(
                    TradeFactIssue(
                        kind="legacy_position_history_projection_gap",
                        severity="info",
                        position_id=int(position.id),
                        symbol=position_symbol,
                        expected_symbol=position_symbol,
                        reason=(
                            "Legacy Position projection lacks complete order links; the "
                            "authoritative OKX position-history mirror remains preserved, "
                            "while this projection stays excluded from training and dashboard truth."
                        ),
                    )
                )
                continue
            for local_marker_id in sorted(local_marker_ids):
                is_orphan = local_marker_id.startswith(ORPHAN_QUARANTINE_EXCHANGE_ID_PREFIX)
                issues.append(
                    TradeFactIssue(
                        kind=(
                            "orphan_position_quarantine_not_exchange_backed"
                            if is_orphan
                            else "manual_close_position_fact_not_exchange_backed"
                        ),
                        severity=(
                            ORPHAN_QUARANTINE_SEVERITY
                            if is_orphan
                            else POSITION_LINK_MISMATCH_SEVERITY
                        ),
                        position_id=int(position.id),
                        symbol=position_symbol,
                        expected_symbol=position_symbol,
                        reason=(
                            "Position close_exchange_order_id uses local synthetic marker "
                            f"{local_marker_id}; it is valid for audit display but not an "
                            "OKX-backed training or reconciliation fact."
                        ),
                    )
                )
            if _position_has_exchange_order_match(position, orders, entry=True) and not entry_ids:
                issues.append(
                    TradeFactIssue(
                        kind="position_missing_entry_order_link",
                        severity=POSITION_LINK_MISSING_SEVERITY,
                        position_id=int(position.id),
                        symbol=position_symbol,
                        expected_symbol=position_symbol,
                        reason=(
                            "Exchange-backed local position has no entry_exchange_order_id; "
                            "future reconciliation would have to infer the entry by symbol/time."
                        ),
                    )
                )
            if (
                not bool(position.is_open)
                and _safe_float(getattr(position, "realized_pnl", None), 0.0) != 0.0
                and not close_ids
                and not local_marker_ids
            ):
                issues.append(
                    TradeFactIssue(
                        kind="closed_position_missing_close_order_link",
                        severity=POSITION_LINK_MISSING_SEVERITY,
                        position_id=int(position.id),
                        symbol=position_symbol,
                        expected_symbol=position_symbol,
                        reason=(
                            "Closed position has realized PnL but no close_exchange_order_id; "
                            "profit, replay, and training cannot prove the OKX close fill."
                        ),
                    )
                )
            recent_entry = _is_recent(position.created_at, since)
            recent_close = _is_recent(position.closed_at, since)
            for linked_order_id in (entry_ids if recent_entry else ()):
                if linked_order_id not in exchange_orders_by_id:
                    issues.append(
                        _linked_order_missing_issue(
                            position,
                            position_symbol,
                            linked_order_id,
                            since=since,
                        )
                    )
            for linked_order_id in (close_ids if recent_close else ()):
                if linked_order_id not in exchange_orders_by_id:
                    issues.append(
                        _linked_order_missing_issue(
                            position,
                            position_symbol,
                            linked_order_id,
                            since=since,
                        )
                    )
        return issues


async def _start_consistent_read_snapshot(session: Any) -> None:
    """Keep a PostgreSQL integrity audit on one immutable read view.

    The order-fact synchronizer can atomically update an order while a long
    system audit is running. PostgreSQL's default READ COMMITTED mode would
    otherwise let one audit combine pre-update orders with post-update linked
    positions, producing a transient false critical. SQLite test databases do
    not support this transaction command, so they keep their existing behavior.
    """

    try:
        bind = session.get_bind()
        dialect_name = str(getattr(getattr(bind, "dialect", None), "name", "") or "")
    except Exception:
        return
    if dialect_name == "postgresql":
        await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))


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


def _order_execution_raw(
    order: Order,
    execution_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the most authoritative OKX execution fact for a local order."""
    okx_raw_fills = getattr(order, "okx_raw_fills", None)
    if (
        isinstance(okx_raw_fills, dict)
        and okx_raw_fills.get("position_snapshot_confirmed") is True
        and okx_raw_fills.get("fills_history_confirmed") is False
    ):
        return {}
    if isinstance(okx_raw_fills, dict) and okx_raw_fills:
        return _raw_from_order_fills(order, okx_raw_fills)
    return _execution_raw_response(execution_result or {})


def _raw_from_order_fills(order: Order, okx_raw_fills: dict[str, Any]) -> dict[str, Any]:
    raw = dict(okx_raw_fills)
    rows = raw.get("rows") if isinstance(raw.get("rows"), list) else []
    first_row = next((row for row in rows if isinstance(row, dict)), {})
    inst_id = (
        raw.get("inst_id")
        or raw.get("instId")
        or getattr(order, "okx_inst_id", None)
        or first_row.get("instId")
    )
    info = dict(raw.get("info") if isinstance(raw.get("info"), dict) else {})
    if inst_id and not info.get("instId"):
        info["instId"] = inst_id
    if not info.get("avgPx"):
        avg_price = raw.get("avg_price") or raw.get("avgPx") or first_row.get("fillPx")
        if avg_price is not None:
            info["avgPx"] = avg_price
    if not info.get("fillPx") and first_row.get("fillPx") is not None:
        info["fillPx"] = first_row.get("fillPx")
    if not info.get("accFillSz"):
        contracts = raw.get("contracts") or raw.get("filled_contracts") or first_row.get("fillSz")
        if contracts is not None:
            info["accFillSz"] = contracts
    raw["info"] = info
    raw.setdefault("okx_inst_id", inst_id)
    raw.setdefault("filled_contracts", raw.get("contracts") or raw.get("filled") or raw.get("amount"))
    raw.setdefault("average", raw.get("avg_price") or raw.get("avgPx"))
    raw.setdefault("avgPx", raw.get("avg_price") or raw.get("average"))
    raw.setdefault("price", raw.get("avg_price") or raw.get("average") or first_row.get("fillPx"))
    raw.setdefault("base_quantity", raw.get("filled_base_quantity") or raw.get("quantity"))
    return raw


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
    native_symbol = _native_exchange_symbol(data)
    if native_symbol:
        return native_symbol
    explicit = normalize_trading_symbol(data.get("canonical_exchange_symbol"))
    if explicit:
        return explicit
    symbol = normalize_trading_symbol(data.get("symbol"))
    if symbol:
        return symbol
    return normalize_trading_symbol(fallback)


def _native_exchange_symbol(raw: dict[str, Any]) -> str:
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
    return ""


def _order_authoritative_symbol(order: Order, raw: dict[str, Any]) -> str:
    return _native_exchange_symbol(raw) or normalize_trading_symbol(order.symbol)


def _position_authoritative_symbol(position: Position) -> str:
    okx_symbol = symbol_from_okx_inst_id(getattr(position, "okx_inst_id", None))
    if okx_symbol:
        return okx_symbol
    return normalize_trading_symbol(position.symbol)


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
    authoritative_order_symbol = _order_authoritative_symbol(order, raw)
    expected_symbols = set()
    for symbol in (
        authoritative_order_symbol,
        local_symbol,
        normalize_trading_symbol(getattr(decision, "symbol", "")),
        _raw_exchange_symbol(raw, fallback=local_symbol),
    ):
        expected_symbols.update(trading_symbol_variants(symbol))
    expected_symbols = {normalize_trading_symbol(symbol) for symbol in expected_symbols if symbol}

    direct_matches = _directly_linked_positions_for_order(
        order,
        positions,
        entry_action=entry_action,
    )
    if direct_matches:
        return direct_matches

    matches: list[tuple[float, Position]] = []
    for position in positions:
        if not _position_model_matches_order(position, order):
            continue
        if str(position.execution_mode or "") != str(order.execution_mode or ""):
            continue
        if str(position.side or "").lower() != side:
            continue
        position_symbol = _position_authoritative_symbol(position)
        symbol_matches = bool(position_symbol and position_symbol in expected_symbols)
        lifecycle_match = (
            entry_action
            and symbol_matches
            and _entry_position_lifecycle_contains_order(position, order_time)
        )
        position_time = _position_match_time(position, entry_action=entry_action)
        if lifecycle_match:
            time_delta = 0.0
        else:
            if position_time is None:
                continue
            time_delta = abs((position_time - order_time).total_seconds())
        if not lifecycle_match and time_delta > POSITION_MATCH_WINDOW.total_seconds():
            continue

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


def _directly_linked_positions_for_order(
    order: Order,
    positions: list[Position],
    *,
    entry_action: bool,
) -> list[Position]:
    order_ids = _split_exchange_order_ids(getattr(order, "exchange_order_id", None))
    if not order_ids:
        return []
    field_name = "entry_exchange_order_id" if entry_action else "close_exchange_order_id"
    matches = [
        position
        for position in positions
        if _position_model_matches_order(position, order)
        and str(position.execution_mode or "") == str(order.execution_mode or "")
        and order_ids.intersection(_split_exchange_order_ids(getattr(position, field_name, None)))
    ]
    return sorted(matches, key=lambda position: getattr(position, "id", 0) or 0)[:5]


def _position_model_matches_order(position: Position, order: Order) -> bool:
    position_model = str(getattr(position, "model_name", "") or "")
    order_model = str(getattr(order, "model_name", "") or "")
    return position_model == order_model or position_model == OKX_AUTHORITATIVE_POSITION_MODEL


def _is_superseded_position_residual(position: Position, positions: list[Position]) -> bool:
    if _has_explicit_superseded_position_metadata(position):
        return True
    return any(
        other is not position and _position_supersedes_for_integrity(other, position)
        for other in positions
    )


def _position_supersedes_for_integrity(candidate: Position, other: Position) -> bool:
    if _position_integrity_base_key(candidate) != _position_integrity_base_key(other):
        return False
    candidate_pos_id = str(getattr(candidate, "okx_pos_id", "") or "").strip()
    other_pos_id = str(getattr(other, "okx_pos_id", "") or "").strip()
    if candidate_pos_id and other_pos_id and candidate_pos_id != other_pos_id:
        return False
    candidate_entry_ids = _split_exchange_order_ids(getattr(candidate, "entry_exchange_order_id", None))
    candidate_close_ids = _split_exchange_order_ids(getattr(candidate, "close_exchange_order_id", None))
    other_entry_ids = _split_exchange_order_ids(getattr(other, "entry_exchange_order_id", None))
    other_close_ids = _split_exchange_order_ids(getattr(other, "close_exchange_order_id", None))
    if not (candidate_entry_ids or candidate_close_ids):
        return False
    if not (candidate_entry_ids and candidate_close_ids):
        return False
    if (
        candidate_pos_id
        and other_pos_id
        and candidate_pos_id == other_pos_id
        and _position_open_times_align(candidate, other)
        and (not other_entry_ids or candidate_entry_ids.issuperset(other_entry_ids))
        and (not other_close_ids or candidate_close_ids.issuperset(other_close_ids))
        and (
            candidate_entry_ids != other_entry_ids
            or candidate_close_ids != other_close_ids
        )
    ):
        return True
    if other_close_ids and candidate_close_ids and not candidate_close_ids.issuperset(other_close_ids):
        return False
    if other_entry_ids and candidate_entry_ids and not candidate_entry_ids.issuperset(other_entry_ids):
        return False
    if _is_zero_quantity_unlinked_residual(other):
        return _position_open_times_align(candidate, other)
    if not _position_times_align(candidate, other):
        return False
    return _position_integrity_score(candidate) > _position_integrity_score(other)


def _has_explicit_superseded_position_metadata(position: Position) -> bool:
    if str(getattr(position, "settlement_status", "") or "") == (
        "superseded_position_residual"
    ):
        return True
    raw = getattr(position, "settlement_raw", None)
    raw = raw if isinstance(raw, dict) else {}
    return bool(
        str(raw.get("reason") or "")
        == "duplicate_local_open_position_for_same_okx_pos_id"
        and _safe_float(raw.get("canonical_position_id"), 0.0) > 0
    )


def _position_integrity_base_key(position: Position) -> tuple[str, str, str]:
    return (
        str(getattr(position, "execution_mode", "") or ""),
        _position_authoritative_symbol(position),
        str(getattr(position, "side", "") or "").lower(),
    )


def _position_times_align(left: Position, right: Position) -> bool:
    if not _position_open_times_align(left, right):
        return False
    left_closed = _ensure_aware(getattr(left, "closed_at", None))
    right_closed = _ensure_aware(getattr(right, "closed_at", None))
    if left_closed and right_closed and abs((left_closed - right_closed).total_seconds()) > 3:
        return False
    return True


def _position_open_times_align(left: Position, right: Position) -> bool:
    left_opened = _ensure_aware(getattr(left, "created_at", None))
    right_opened = _ensure_aware(getattr(right, "created_at", None))
    if left_opened and right_opened and abs((left_opened - right_opened).total_seconds()) > 3:
        return False
    return True


def _is_zero_quantity_unlinked_residual(position: Position) -> bool:
    return (
        not bool(getattr(position, "is_open", False))
        and abs(_safe_float(getattr(position, "quantity", None))) <= 1e-12
        and abs(_safe_float(getattr(position, "realized_pnl", None))) <= 1e-12
        and not _split_exchange_order_ids(getattr(position, "entry_exchange_order_id", None))
        and not _split_exchange_order_ids(getattr(position, "close_exchange_order_id", None))
    )


def _position_integrity_score(position: Position) -> tuple[int, int, int, int, int]:
    entry_ids = _split_exchange_order_ids(getattr(position, "entry_exchange_order_id", None))
    close_ids = _split_exchange_order_ids(getattr(position, "close_exchange_order_id", None))
    quantity = abs(_safe_float(getattr(position, "quantity", None)))
    realized = abs(_safe_float(getattr(position, "realized_pnl", None)))
    return (
        len(entry_ids),
        len(close_ids),
        1 if quantity > 1e-12 else 0,
        1 if realized > 1e-12 else 0,
        int(getattr(position, "id", 0) or 0),
    )


def _position_has_exchange_order_match(
    position: Position,
    orders: list[Order],
    *,
    entry: bool,
) -> bool:
    position_symbol = normalize_trading_symbol(position.symbol)
    side = str(position.side or "").lower()
    expected_order_side = (
        "buy" if (entry and side == "long") or (not entry and side == "short") else "sell"
    )
    position_time = _ensure_aware(position.created_at if entry else position.closed_at)
    if position_time is None:
        return False
    for order in orders:
        if is_manual_close_order(order):
            continue
        if not str(getattr(order, "exchange_order_id", "") or "").strip():
            continue
        if str(order.execution_mode or "") != str(position.execution_mode or ""):
            continue
        if normalize_trading_symbol(order.symbol) != position_symbol:
            continue
        if str(order.side or "").lower() != expected_order_side:
            continue
        order_time = _order_time(order)
        if order_time is None:
            continue
        if (
            abs((order_time - position_time).total_seconds())
            <= POSITION_MATCH_WINDOW.total_seconds()
        ):
            return True
    return False


def _linked_order_missing_issue(
    position: Position,
    position_symbol: str,
    linked_order_id: str,
    *,
    since: datetime,
) -> TradeFactIssue:
    reference_time = _ensure_aware(getattr(position, "closed_at", None)) or _ensure_aware(
        getattr(position, "created_at", None)
    )
    age_days = (
        max((datetime.now(UTC) - reference_time).total_seconds() / 86400.0, 0.0)
        if reference_time is not None
        else 0.0
    )
    severity = (
        "info"
        if age_days > HISTORICAL_LINK_ORDER_MISSING_OBSERVATION_DAYS
        else POSITION_LINK_ORDER_MISSING_SEVERITY
    )
    return TradeFactIssue(
        kind="position_order_link_missing_local_order",
        severity=severity,
        position_id=int(position.id),
        symbol=position_symbol,
        expected_symbol=position_symbol,
        reason=(
            "Position references OKX order id "
            f"{linked_order_id}, but that order is not present in recent local filled orders."
        ),
    )


def _is_recent(value: datetime | None, since: datetime) -> bool:
    aware = _ensure_aware(value)
    return aware is not None and aware >= since


def _split_exchange_order_ids(value: Any) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def _position_linked_order_ids(positions: list[Position]) -> set[str]:
    order_ids: set[str] = set()
    for position in positions:
        order_ids.update(_split_exchange_order_ids(getattr(position, "entry_exchange_order_id", None)))
        order_ids.update(_split_exchange_order_ids(getattr(position, "close_exchange_order_id", None)))
    return order_ids


def _dedupe_orders_by_id(orders: list[Order]) -> list[Order]:
    deduped: dict[int, Order] = {}
    for order in orders:
        order_id = int(getattr(order, "id", 0) or 0)
        if not order_id:
            continue
        deduped[order_id] = order
    return list(deduped.values())


def _order_time(order: Order) -> datetime | None:
    return _ensure_aware(getattr(order, "filled_at", None) or getattr(order, "created_at", None))


def _position_match_time(position: Position, *, entry_action: bool) -> datetime | None:
    if entry_action:
        return _ensure_aware(getattr(position, "created_at", None))
    return _ensure_aware(getattr(position, "closed_at", None))


def _entry_position_lifecycle_contains_order(position: Position, order_time: datetime) -> bool:
    created_at = _ensure_aware(getattr(position, "created_at", None))
    if created_at is None or created_at > order_time:
        return False
    closed_at = _ensure_aware(getattr(position, "closed_at", None))
    return closed_at is None or closed_at >= order_time


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
    nominal_since: datetime,
    since: datetime,
    epoch_started_at: datetime,
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
        "nominal_audit_window_start": nominal_since.isoformat(),
        "audit_window_start": since.isoformat(),
        "training_epoch_started_at": epoch_started_at.isoformat(),
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
