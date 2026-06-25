from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

WEAK_EVIDENCE_TIERS = {"weak_conflict_probe", "degraded_missing_probe"}
HIGH_QUALITY_EVIDENCE_TIERS = {"exploration", "small", "medium", "normal"}
STRONG_EXIT_INTENTS = {"hard_risk", "trend_failure", "predictive_downside", "profit_drawdown"}
FAST_LOSS_MINUTES = 15.0
SMALL_SIZE_REASON_THRESHOLD = 0.015
FRESH_LOSS_REENTRY_HOURS = 2.0
DEFAULT_REPORT_WINDOW_HOURS = 24
DEFAULT_REPORT_LIMIT = 500
MAX_SUPPLEMENTAL_FAST_LOSS_LOOKUPS = 100
SUPPLEMENTAL_EXIT_LOOKUP_MINUTES = 30


class TradeExecutionContractService:
    def __init__(self, session_context_factory: Any | None = None) -> None:
        self._session_context_factory = session_context_factory

    async def report(
        self,
        *,
        hours: int = DEFAULT_REPORT_WINDOW_HOURS,
        limit: int = DEFAULT_REPORT_LIMIT,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        from sqlalchemy import and_, or_, select

        from db.session import get_read_session_ctx
        from models.decision import AIDecision
        from models.trade import Order, Position

        capped_hours = max(1, min(int(hours or DEFAULT_REPORT_WINDOW_HOURS), 168))
        capped_limit = max(50, min(int(limit or DEFAULT_REPORT_LIMIT), 1000))
        since_utc = _normalize_since(since)
        session_factory = self._session_context_factory or get_read_session_ctx
        async with session_factory() as session:
            decision_result = await session.execute(
                _apply_since_filter(
                    select(AIDecision),
                    AIDecision,
                    since_utc=since_utc,
                )
                .order_by(AIDecision.id.desc())
                .limit(capped_limit)
            )
            order_result = await session.execute(
                _apply_since_filter(
                    select(Order),
                    Order,
                    since_utc=since_utc,
                )
                .order_by(Order.id.desc())
                .limit(capped_limit)
            )
            position_result = await session.execute(
                _apply_position_since_filter(
                    select(Position),
                    Position,
                    since_utc=since_utc,
                    or_=or_,
                )
                .order_by(Position.id.desc())
                .limit(capped_limit)
            )
            if since_utc is not None:
                decisions = [
                    row
                    for row in decision_result.scalars().all()
                    if _row_at_or_after(row, since_utc)
                ]
                orders = [
                    row for row in order_result.scalars().all() if _row_at_or_after(row, since_utc)
                ]
                positions = [
                    row
                    for row in position_result.scalars().all()
                    if _row_at_or_after(row, since_utc) or _closed_at_or_after(row, since_utc)
                ]
            else:
                decisions = [
                    row for row in decision_result.scalars().all() if _row_recent(row, capped_hours)
                ]
                orders = [
                    row for row in order_result.scalars().all() if _row_recent(row, capped_hours)
                ]
                positions = [
                    row
                    for row in position_result.scalars().all()
                    if _row_recent(row, capped_hours) or _closed_recent(row, capped_hours)
                ]
            supplemental_order_decisions = await _load_supplemental_order_decisions(
                session=session,
                decision_model=AIDecision,
                orders=orders,
                known_decision_ids={_safe_int(_row_get(row, "id"), 0) for row in decisions},
                supplemental_limit=capped_limit,
                select=select,
            )
            fast_loss_positions = [row for row in positions if _fast_loss_summary(row) is not None][
                :MAX_SUPPLEMENTAL_FAST_LOSS_LOOKUPS
            ]
            supplemental_exit_decisions = await _load_supplemental_exit_decisions(
                session=session,
                decision_model=AIDecision,
                fast_loss_positions=fast_loss_positions,
                supplemental_limit=capped_limit,
                and_=and_,
                or_=or_,
                select=select,
            )

        all_decisions = _dedupe_rows_by_id(
            [*decisions, *supplemental_order_decisions, *supplemental_exit_decisions]
        )
        report = summarize_trade_execution_contract(
            all_decisions,
            orders=orders,
            positions=positions,
        )
        report["window_hours"] = capped_hours
        report["query_policy"] = {
            "online_safe": True,
            "ordered_by_primary_key": True,
            "db_time_filter": since_utc is not None,
            "row_limit": capped_limit,
            "supplemental_order_decision_lookup": bool(supplemental_order_decisions),
            "supplemental_order_decision_count": len(supplemental_order_decisions),
            "supplemental_exit_lookup": bool(fast_loss_positions),
            "supplemental_exit_lookup_minutes": SUPPLEMENTAL_EXIT_LOOKUP_MINUTES,
            "supplemental_exit_decision_count": len(supplemental_exit_decisions),
            "supplemental_fast_loss_position_count": len(fast_loss_positions),
        }
        if since_utc is not None:
            report["query_policy"]["since_utc"] = since_utc.isoformat()
        return report


async def _load_supplemental_order_decisions(
    *,
    session: Any,
    decision_model: Any,
    orders: Sequence[Any],
    known_decision_ids: set[int],
    supplemental_limit: int,
    select: Any,
) -> list[Any]:
    decision_ids = [
        decision_id
        for decision_id in sorted(
            {
                _safe_int(_row_get(order, "decision_id"), 0)
                for order in orders
                if str(_row_get(order, "status") or "").lower() == "filled"
            },
            reverse=True,
        )
        if decision_id and decision_id not in known_decision_ids
    ][:supplemental_limit]
    if not decision_ids:
        return []
    result = await session.execute(
        select(decision_model)
        .where(decision_model.id.in_(decision_ids))
        .order_by(decision_model.id.desc())
        .limit(len(decision_ids))
    )
    return list(result.scalars().all())


async def _load_supplemental_exit_decisions(
    *,
    session: Any,
    decision_model: Any,
    fast_loss_positions: Sequence[Any],
    supplemental_limit: int,
    and_: Any,
    or_: Any,
    select: Any,
) -> list[Any]:
    clauses = []
    lookup_delta = timedelta(minutes=SUPPLEMENTAL_EXIT_LOOKUP_MINUTES)
    for position in fast_loss_positions:
        closed = _parse_datetime(_row_get(position, "closed_at"))
        symbol = str(_row_get(position, "symbol") or "")
        side = str(_row_get(position, "side") or "").lower()
        action = "close_long" if side == "long" else "close_short" if side == "short" else ""
        if closed is None or not symbol or not action:
            continue
        clauses.append(
            and_(
                decision_model.symbol == symbol,
                decision_model.action == action,
                decision_model.created_at >= closed - lookup_delta,
                decision_model.created_at <= closed + lookup_delta,
            )
        )
    if not clauses:
        return []
    result = await session.execute(
        select(decision_model)
        .where(or_(*clauses))
        .order_by(decision_model.id.desc())
        .limit(max(supplemental_limit, len(clauses) * 4))
    )
    return list(result.scalars().all())


def _dedupe_rows_by_id(rows: Sequence[Any]) -> list[Any]:
    seen: set[tuple[str, Any]] = set()
    deduped: list[Any] = []
    for row in rows:
        row_id = _row_get(row, "id")
        key = (type(row).__name__, row_id) if row_id is not None else ("object", id(row))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def summarize_trade_execution_contract(
    decisions: Sequence[Any],
    *,
    orders: Sequence[Any] | None = None,
    positions: Sequence[Any] | None = None,
) -> dict[str, Any]:
    orders_by_decision = _orders_by_decision(orders or [])
    exit_decisions = [row for row in decisions if _is_exit(row)]
    entry_rows = [row for row in decisions if _is_entry(row)]
    executed_entries = [row for row in entry_rows if _entry_executed(row, orders_by_decision)]

    entry_explanations: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for row in executed_entries:
        explanation = _entry_explanation(row)
        entry_explanations.append(explanation)
        for reason in explanation["violations"]:
            reason_counts[reason] += 1
            violations.append(_violation(row, reason, explanation))

    fast_loss_rows: list[dict[str, Any]] = []
    exchange_sync_estimated_reductions: list[dict[str, Any]] = []
    for position in positions or []:
        fast_loss = _fast_loss_summary(position)
        if fast_loss is None:
            continue
        matching_exit = _matching_exit_decision(exit_decisions, position, fast_loss["closed_at"])
        if _is_estimated_exchange_quantity_reduction(matching_exit):
            exchange_sync_estimated_reductions.append(
                {
                    **fast_loss,
                    "decision_id": _row_get(matching_exit, "id") if matching_exit else None,
                    "reason": "estimated_exchange_quantity_reduction",
                }
            )
            continue
        fast_loss_rows.append(fast_loss)
        if not _has_strong_exit_evidence(matching_exit):
            reason_counts["fast_loss_without_strong_exit"] += 1
            violations.append(
                {
                    "reason": "fast_loss_without_strong_exit",
                    "decision_id": _row_get(matching_exit, "id") if matching_exit else None,
                    "symbol": fast_loss["symbol"],
                    "side": fast_loss["side"],
                    "details": fast_loss,
                }
            )

    summary = {
        "decision_count": len(decisions),
        "executed_entry_count": len(executed_entries),
        "missing_entry_explanation_count": reason_counts["missing_entry_execution_reason"],
        "missing_sizing_explanation_count": reason_counts["missing_profit_risk_sizing"],
        "small_size_without_reason_count": reason_counts["small_size_without_reason"],
        "weak_evidence_executed_count": reason_counts["weak_evidence_executed"],
        "negative_expected_executed_count": reason_counts["non_positive_expected_net_executed"],
        "fast_loss_count": len(fast_loss_rows),
        "exchange_sync_estimated_reduction_count": len(exchange_sync_estimated_reductions),
        "fast_loss_without_strong_exit_count": reason_counts["fast_loss_without_strong_exit"],
        "reentry_without_strong_unlock_count": reason_counts["reentry_without_strong_unlock"],
        "contract_violation_count": sum(reason_counts.values()),
    }
    return {
        "audit_only": True,
        "live_entry_mutation": False,
        "live_exit_mutation": False,
        "can_bypass_risk_controls": False,
        "summary": summary,
        "violation_reason_counts": dict(reason_counts),
        "entry_explanations": entry_explanations[:20],
        "fast_loss_samples": fast_loss_rows[:20],
        "exchange_sync_estimated_reductions": exchange_sync_estimated_reductions[:20],
        "violations": violations[:30],
        "policy": {
            "entry_requires_positive_expected_net": True,
            "entry_requires_structured_evidence": True,
            "position_size_requires_profit_risk_sizing": True,
            "fast_loss_exit_requires_strong_exit_evidence": True,
            "recent_loss_reentry_requires_strong_unlock": True,
        },
    }


def _entry_explanation(row: Any) -> dict[str, Any]:
    raw = _safe_dict(_row_get(row, "raw_llm_response"))
    opportunity = _safe_dict(raw.get("opportunity_score"))
    sizing = _safe_dict(raw.get("profit_risk_sizing"))
    evidence = _safe_dict(opportunity.get("evidence_score"))
    expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
    evidence_tier = str(evidence.get("tier") or "missing")
    execution_reason, execution_reason_source = _entry_execution_reason(
        row,
        raw=raw,
        opportunity=opportunity,
    )
    size = _safe_float(_row_get(row, "position_size_pct"), 0.0)
    violations: list[str] = []

    if not execution_reason:
        violations.append("missing_entry_execution_reason")
    if not sizing:
        violations.append("missing_profit_risk_sizing")
    if evidence_tier in WEAK_EVIDENCE_TIERS:
        violations.append("weak_evidence_executed")
    if expected_net <= 0:
        violations.append("non_positive_expected_net_executed")
    if size <= SMALL_SIZE_REASON_THRESHOLD and not _has_small_size_reason(sizing):
        violations.append("small_size_without_reason")
    if _fresh_loss_reentry_active(opportunity) and not _has_reentry_unlock(raw, opportunity):
        violations.append("reentry_without_strong_unlock")

    return {
        "decision_id": _row_get(row, "id"),
        "symbol": _row_get(row, "symbol"),
        "action": _side(_row_get(row, "action")),
        "expected_net_return_pct": round(expected_net, 6),
        "profit_quality_ratio": round(_safe_float(opportunity.get("profit_quality_ratio")), 6),
        "loss_probability": round(
            _safe_float(opportunity.get("server_profit_loss_probability"), 1.0), 6
        ),
        "tail_risk_score": round(_safe_float(opportunity.get("tail_risk_score")), 6),
        "evidence_tier": evidence_tier,
        "evidence_effective_score": evidence.get("effective_score"),
        "position_size_pct": round(size, 8),
        "suggested_leverage": round(_safe_float(_row_get(row, "suggested_leverage"), 1.0), 6),
        "has_execution_reason": bool(execution_reason),
        "execution_reason_source": execution_reason_source,
        "has_profit_risk_sizing": bool(sizing),
        "sizing_quality_tier": sizing.get("quality_tier"),
        "sizing_reason": sizing.get("meaningful_size_reason") or sizing.get("reason") or "",
        "loss_cooldown_unlock": _safe_dict(
            raw.get("loss_cooldown_override") or opportunity.get("loss_cooldown_override")
        ),
        "violations": violations,
    }


def _entry_execution_reason(
    row: Any,
    *,
    raw: dict[str, Any],
    opportunity: dict[str, Any],
) -> tuple[str, str]:
    candidates = (
        ("execution_reason", _row_get(row, "execution_reason")),
        ("selection_reason", opportunity.get("selection_reason")),
        ("raw_reason", raw.get("execution_reason") or raw.get("reason")),
    )
    for source, value in candidates:
        text = str(value or "").strip()
        if text:
            return text, source
    return "", "missing"


def _orders_by_decision(orders: Sequence[Any]) -> dict[int, list[Any]]:
    result: dict[int, list[Any]] = {}
    for order in orders:
        decision_id = _safe_int(_row_get(order, "decision_id"), 0)
        if decision_id:
            result.setdefault(decision_id, []).append(order)
    return result


def _entry_executed(row: Any, orders_by_decision: dict[int, list[Any]]) -> bool:
    if bool(_row_get(row, "was_executed")):
        return True
    decision_id = _safe_int(_row_get(row, "id"), 0)
    return any(
        str(_row_get(order, "status") or "").lower() == "filled"
        for order in orders_by_decision.get(decision_id, [])
    )


def _is_entry(row: Any) -> bool:
    return _side(_row_get(row, "action")) in {"long", "short"}


def _is_exit(row: Any) -> bool:
    return _side(_row_get(row, "action")) in {"close_long", "close_short"}


def _side(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"long", "buy"}:
        return "long"
    if normalized in {"short", "sell"}:
        return "short"
    if normalized in {"close_long", "sell_long"}:
        return "close_long"
    if normalized in {"close_short", "buy_short"}:
        return "close_short"
    return normalized


def _has_small_size_reason(sizing: dict[str, Any]) -> bool:
    if not sizing:
        return False
    return bool(
        sizing.get("reason")
        or sizing.get("meaningful_size_reason")
        or sizing.get("notional_floor_blocked")
        or sizing.get("probe_budget_guard")
        or sizing.get("strategy_learning_sizing")
        or sizing.get("quality_caps")
    )


def _fresh_loss_reentry_active(opportunity: dict[str, Any]) -> bool:
    profile = _safe_dict(opportunity.get("symbol_side_profile"))
    if not profile:
        profile = _safe_dict(opportunity.get("symbol_profile"))
    losses = _safe_int(profile.get("losses"), 0)
    if losses <= 0:
        return False
    last_loss_age = _safe_float(profile.get("last_loss_age_hours"), 9999.0)
    pnl = _safe_float(profile.get("pnl"), 0.0)
    today_pnl = _safe_float(profile.get("today_pnl"), 0.0)
    return bool(last_loss_age < FRESH_LOSS_REENTRY_HOURS and (pnl < 0 or today_pnl < 0))


def _has_reentry_unlock(raw: dict[str, Any], opportunity: dict[str, Any]) -> bool:
    override = _safe_dict(raw.get("loss_cooldown_override")) or _safe_dict(
        opportunity.get("loss_cooldown_override")
    )
    if not bool(override.get("allowed")):
        return False
    metrics = _safe_dict(override.get("metrics"))
    aligned = (
        metrics.get("aligned_sources") if isinstance(metrics.get("aligned_sources"), list) else []
    )
    if metrics and metrics.get("fresh_loss"):
        return bool(
            _safe_float(metrics.get("expected_net_return_pct"), 0.0) >= 1.2
            and _safe_float(metrics.get("profit_quality_ratio"), 0.0) >= 1.1
            and _safe_float(metrics.get("server_profit_loss_probability"), 1.0) <= 0.42
            and len(aligned) >= 3
        )
    return True


def _fast_loss_summary(position: Any) -> dict[str, Any] | None:
    opened = _parse_datetime(_row_get(position, "created_at"))
    closed = _parse_datetime(_row_get(position, "closed_at"))
    if opened is None or closed is None:
        return None
    hold_minutes = max((closed - opened).total_seconds() / 60.0, 0.0)
    pnl = _safe_float(_row_get(position, "realized_pnl"), 0.0)
    if hold_minutes > FAST_LOSS_MINUTES or pnl >= 0:
        return None
    quantity = abs(_safe_float(_row_get(position, "quantity"), 0.0))
    entry_price = _safe_float(_row_get(position, "entry_price"), 0.0)
    return {
        "id": _row_get(position, "id"),
        "symbol": _row_get(position, "symbol"),
        "side": _row_get(position, "side"),
        "hold_minutes": round(hold_minutes, 3),
        "realized_pnl": round(pnl, 8),
        "notional_usdt": round(quantity * entry_price, 6),
        "closed_at": closed,
        "closed_at_iso": closed.isoformat(),
    }


def _matching_exit_decision(
    exit_decisions: Sequence[Any],
    position: Any,
    closed_at: datetime,
) -> Any | None:
    symbol = str(_row_get(position, "symbol") or "")
    side = str(_row_get(position, "side") or "").lower()
    expected_action = "close_long" if side == "long" else "close_short" if side == "short" else ""
    best: Any | None = None
    best_delta = 999999.0
    for decision in exit_decisions:
        if symbol and str(_row_get(decision, "symbol") or "") != symbol:
            continue
        if expected_action and _side(_row_get(decision, "action")) != expected_action:
            continue
        created = _parse_datetime(_row_get(decision, "created_at"))
        if created is None:
            continue
        delta = abs((closed_at - created).total_seconds())
        if delta <= 30 * 60 and delta < best_delta:
            best = decision
            best_delta = delta
    return best


def _has_strong_exit_evidence(decision: Any | None) -> bool:
    if decision is None:
        return False
    raw = _safe_dict(_row_get(decision, "raw_llm_response"))
    close_evidence = _safe_dict(raw.get("close_evidence"))
    arbitration = _safe_dict(raw.get("exit_arbitration"))
    intent = str(
        raw.get("exit_intent")
        or close_evidence.get("exit_intent")
        or arbitration.get("intent")
        or ""
    )
    if intent in STRONG_EXIT_INTENTS:
        return True
    if raw.get("forced_exit") or raw.get("fast_risk_exit"):
        return True
    if _has_exchange_confirmed_close_fill(raw):
        return True
    return bool(
        close_evidence.get("hard_risk")
        or close_evidence.get("trend_failure")
        or close_evidence.get("predictive_reversal_exit")
        or close_evidence.get("profit_retrace_protection")
    )


def _has_exchange_confirmed_close_fill(raw: dict[str, Any]) -> bool:
    close_fill = _safe_dict(raw.get("close_fill"))
    if not raw.get("system_sync") or str(raw.get("source") or "") != "okx_position_reconcile":
        return False
    if bool(close_fill.get("estimated")):
        return False
    if not str(close_fill.get("order_id") or "").strip():
        return False
    return bool(
        _safe_float(close_fill.get("price"), 0.0) > 0
        and _safe_float(close_fill.get("quantity"), 0.0) > 0
    )


def _is_estimated_exchange_quantity_reduction(decision: Any | None) -> bool:
    if decision is None:
        return False
    raw = _safe_dict(_row_get(decision, "raw_llm_response"))
    close_fill = _safe_dict(raw.get("close_fill"))
    if not raw.get("system_sync") or str(raw.get("source") or "") != "okx_position_reconcile":
        return False
    return bool(close_fill.get("estimated") and close_fill.get("partial_reduction"))


def _violation(row: Any, reason: str, explanation: dict[str, Any]) -> dict[str, Any]:
    return {
        "reason": reason,
        "decision_id": _row_get(row, "id"),
        "symbol": _row_get(row, "symbol"),
        "action": _row_get(row, "action"),
        "details": explanation,
    }


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _row_recent(row: Any, hours: int) -> bool:
    created = _parse_datetime(_row_get(row, "created_at"))
    if created is None:
        return True
    return (_now_utc() - created).total_seconds() <= hours * 3600


def _closed_recent(row: Any, hours: int) -> bool:
    closed = _parse_datetime(_row_get(row, "closed_at"))
    if closed is None:
        return False
    return (_now_utc() - closed).total_seconds() <= hours * 3600


def _normalize_since(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _row_at_or_after(row: Any, since: datetime) -> bool:
    created = _parse_datetime(_row_get(row, "created_at"))
    return created is not None and created >= since


def _closed_at_or_after(row: Any, since: datetime) -> bool:
    closed = _parse_datetime(_row_get(row, "closed_at"))
    return closed is not None and closed >= since


def _apply_since_filter(statement: Any, model: Any, *, since_utc: datetime | None) -> Any:
    if since_utc is None:
        return statement
    return statement.where(model.created_at >= since_utc)


def _apply_position_since_filter(
    statement: Any,
    model: Any,
    *,
    since_utc: datetime | None,
    or_: Any,
) -> Any:
    if since_utc is None:
        return statement
    return statement.where(or_(model.created_at >= since_utc, model.closed_at >= since_utc))


def _now_utc() -> datetime:
    return datetime.now(UTC)
