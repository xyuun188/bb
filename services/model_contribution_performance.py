"""Closed-loop performance by decision evidence source.

Entry scoring uses several evidence sources: local ML, server profit model,
time-series prediction, sentiment, shadow memory, and expert agreement.  This
service measures which sources actually led to realized profit recently, then
turns that feedback into bounded score and size adjustments.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from math import sqrt
from typing import Any

import structlog
from sqlalchemy import and_, or_, select

from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text
from core.symbols import normalize_trading_symbol, symbol_query_variants
from db.session import get_session_ctx
from models.decision import AIDecision
from models.trade import Order, Position
from services.manual_close_marker import position_has_manual_close_order
from services.trade_fact_trust import (
    closed_position_trade_fact_trusted,
    split_exchange_order_ids,
)

SessionFactory = Callable[[], Any]

DEFAULT_CONTRIBUTION_LOOKBACK_DAYS = 7.0
DEFAULT_POSITION_LIMIT = 800
DEFAULT_ORDER_LIMIT = 3000
MANUAL_CLOSE_LOOKUP_GRACE_SECONDS = 15.0
OKX_AUTHORITATIVE_LEDGER_MODEL = "okx_authoritative_sync"

logger = structlog.get_logger(__name__)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _empty_bucket(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "count": 0,
        "wins": 0,
        "losses": 0,
        "pnl": 0.0,
        "profit": 0.0,
        "loss": 0.0,
        "avg_pnl": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "pnl_lcb_usdt": 0.0,
        "production_permission": False,
        "state": "not_observed",
        "reason": "尚无可归因的权威平仓结果。",
        "_pnl_samples": [],
    }


def _default_stats() -> dict[str, dict[str, Any]]:
    return {
        "decision_llm": _empty_bucket("决策大模型"),
        "ml_profit_model": _empty_bucket("本地 ML 盈利模型"),
        "server_profit_model": _empty_bucket("服务器盈利模型"),
        "timeseries_model": _empty_bucket("时序预测模型"),
        "sentiment_model": _empty_bucket("情绪模型"),
        "shadow_memory": _empty_bucket("影子/交易记忆"),
        "expert_alignment": _empty_bucket("专家一致信号"),
        "high_risk_review": _empty_bucket("高风险复核模型"),
        "ai_only_without_quant": _empty_bucket("AI 单独支持但量化未同向"),
    }


class ModelContributionPerformanceService:
    """Measure recent realized PnL by evidence source and score its reliability."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory = get_session_ctx,
        model_name: str = ENSEMBLE_TRADER_NAME,
        ledger_model_names: tuple[str, ...] | None = None,
        lookback_days: float = DEFAULT_CONTRIBUTION_LOOKBACK_DAYS,
        position_limit: int = DEFAULT_POSITION_LIMIT,
        order_limit: int = DEFAULT_ORDER_LIMIT,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._model_name = model_name
        self._ledger_model_names = tuple(
            dict.fromkeys(
                ledger_model_names
                or (model_name, OKX_AUTHORITATIVE_LEDGER_MODEL)
            )
        )
        self._lookback_days = float(lookback_days)
        self._position_limit = int(position_limit)
        self._order_limit = int(order_limit)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._cache_by_mode: dict[str, dict[str, Any]] = {}

    async def recent(self, mode: str) -> dict[str, dict[str, Any]]:
        """Return cached recent contribution performance for the selected mode."""

        selected_mode = "live" if mode == "live" else "paper"
        now = _aware(self._clock()) or datetime.now(UTC)
        cache_entry = self._cache_by_mode.get(selected_mode, {})
        expires_at = cache_entry.get("expires_at")
        cached_stats = cache_entry.get("stats")
        if isinstance(expires_at, datetime) and expires_at > now and isinstance(cached_stats, dict):
            return cached_stats

        start_utc = now - timedelta(days=self._lookback_days)
        stats = _default_stats()
        try:
            async with self._session_factory() as session:
                positions_result = await session.execute(
                    select(Position)
                    .where(
                        Position.model_name.in_(self._ledger_model_names),
                        Position.execution_mode == selected_mode,
                        Position.is_open.is_(False),
                        Position.closed_at.is_not(None),
                        Position.closed_at >= start_utc,
                    )
                    .order_by(Position.closed_at.desc())
                    .limit(self._position_limit)
                )
                positions = list(positions_result.scalars().all())
                if not positions:
                    self._cache_by_mode[selected_mode] = {
                        "expires_at": now + timedelta(minutes=15),
                        "stats": stats,
                    }
                    return stats

                positions = [
                    pos for pos in positions if closed_position_trade_fact_trusted(pos)
                ]
                symbols = {p.symbol for p in positions if p.symbol}
                symbol_variants = symbol_query_variants(symbols)
                close_times = [
                    closed_at
                    for pos in positions
                    if (closed_at := _aware(getattr(pos, "closed_at", None))) is not None
                ]
                manual_close_orders = []
                if symbol_variants and close_times:
                    grace = timedelta(seconds=MANUAL_CLOSE_LOOKUP_GRACE_SECONDS)
                    close_window_start = min(close_times) - grace
                    close_window_end = max(close_times) + grace
                    manual_close_result = await session.execute(
                        select(Order).where(
                            Order.model_name.in_(self._ledger_model_names),
                            Order.execution_mode == selected_mode,
                            Order.status == "filled",
                            Order.symbol.in_(symbol_variants),
                            Order.exchange_order_id.like("manual_close:%"),
                            or_(
                                Order.filled_at.between(close_window_start, close_window_end),
                                and_(
                                    Order.filled_at.is_(None),
                                    Order.created_at.between(close_window_start, close_window_end),
                                ),
                            ),
                        )
                    )
                    manual_close_orders = list(manual_close_result.scalars().all())
                positions = [
                    pos
                    for pos in positions
                    if not position_has_manual_close_order(pos, manual_close_orders)
                ]
                if not positions:
                    self._cache_by_mode[selected_mode] = {
                        "expires_at": now + timedelta(minutes=15),
                        "stats": stats,
                    }
                    return stats
                entry_order_ids = {
                    order_id
                    for pos in positions
                    for order_id in split_exchange_order_ids(
                        getattr(pos, "entry_exchange_order_id", None)
                    )
                }
                if not entry_order_ids:
                    self._cache_by_mode[selected_mode] = {
                        "expires_at": now + timedelta(minutes=10),
                        "stats": stats,
                    }
                    return stats
                orders_result = await session.execute(
                    select(Order)
                    .where(
                        Order.model_name.in_(self._ledger_model_names),
                        Order.execution_mode == selected_mode,
                        Order.status == "filled",
                        Order.decision_id.is_not(None),
                        Order.exchange_order_id.in_(entry_order_ids),
                    )
                    .order_by(Order.filled_at.desc(), Order.created_at.desc())
                )
                orders = list(orders_result.scalars().all())
                decision_ids = [o.decision_id for o in orders if o.decision_id]
                decisions: dict[int, AIDecision] = {}
                if decision_ids:
                    decisions_result = await session.execute(
                        select(AIDecision).where(AIDecision.id.in_(decision_ids))
                    )
                    decisions = {d.id: d for d in decisions_result.scalars().all()}
        except Exception as exc:
            logger.warning(
                "failed to calculate model contribution performance",
                error=safe_error_text(exc),
            )
            return {}

        stats = self.build_stats(positions, orders, decisions)
        self._cache_by_mode[selected_mode] = {
            "expires_at": now + timedelta(minutes=10),
            "stats": stats,
        }
        return stats

    def build_stats(
        self,
        positions: Iterable[Any],
        orders: Iterable[Any],
        decisions: dict[int, Any],
    ) -> dict[str, dict[str, Any]]:
        """Build contribution stats from loaded position/order/decision records."""

        stats = _default_stats()
        order_list = list(orders)
        for pos in positions:
            if not closed_position_trade_fact_trusted(pos):
                continue
            matched_decision = self._match_entry_decision(pos, order_list, decisions)
            if matched_decision is None:
                continue
            raw = _safe_dict(getattr(matched_decision, "raw_llm_response", None))
            opportunity = _safe_dict(raw.get("opportunity_score"))
            pnl = float(getattr(pos, "realized_pnl", 0.0) or 0.0)
            for source in self.contribution_sources(opportunity, raw, self._position_side(pos)):
                self._add_sample(stats[source], pnl)

        for bucket in stats.values():
            self._finalize_bucket(bucket)
        return stats

    def build_lineage_diagnostics(
        self,
        positions: Iterable[Any],
        orders: Iterable[Any],
        decisions: dict[int, Any],
    ) -> dict[str, Any]:
        """Explain whether realized positions can be linked back to entry decisions."""

        position_list = [
            pos for pos in positions if closed_position_trade_fact_trusted(pos)
        ]
        order_list = list(orders)
        linked_orders = [order for order in order_list if getattr(order, "decision_id", None)]
        orders_with_loaded_decisions = [
            order
            for order in linked_orders
            if _safe_int(getattr(order, "decision_id", None)) in decisions
        ]
        matched_positions = [
            pos
            for pos in position_list
            if self._match_entry_decision(pos, order_list, decisions) is not None
        ]
        total_positions = len(position_list)
        matched_count = len(matched_positions)
        match_rate = matched_count / total_positions if total_positions else 0.0
        reason = "ok"
        if total_positions <= 0:
            reason = "no_closed_positions"
        elif not order_list:
            reason = "no_filled_orders_for_symbols"
        elif not linked_orders:
            reason = "filled_orders_missing_decision_id"
        elif not orders_with_loaded_decisions:
            reason = "linked_decisions_missing"
        elif matched_count <= 0:
            reason = "position_order_time_or_side_mismatch"
        elif matched_count < total_positions:
            reason = "partial_lineage"
        return {
            "total_closed_positions": total_positions,
            "filled_order_count": len(order_list),
            "orders_with_decision_id": len(linked_orders),
            "orders_with_loaded_decision": len(orders_with_loaded_decisions),
            "matched_position_count": matched_count,
            "unmatched_position_count": max(total_positions - matched_count, 0),
            "match_rate": round(match_rate, 6),
            "reason": reason,
            "ready_for_profit_learning": bool(total_positions and matched_count),
        }

    def contribution_sources(
        self,
        opportunity: dict[str, Any],
        raw: dict[str, Any],
        side: str,
    ) -> list[str]:
        """Infer which evidence sources supported an entry decision."""

        del raw, side
        breakdown = _safe_dict(opportunity.get("expected_net_breakdown"))
        source_map = {
            "local_ml": "ml_profit_model",
            "server_profit": "server_profit_model",
            "timeseries": "timeseries_model",
        }
        return list(
            dict.fromkeys(
                source_map[str(item.get("key"))]
                for item in _safe_list(breakdown.get("components"))
                if isinstance(item, dict)
                and item.get("production_eligible") is True
                and str(item.get("key")) in source_map
            )
        )

    def score_adjustment(
        self,
        sources: list[str],
        performance: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Expose attribution without changing production score or size."""

        return {
            "active": False,
            "sources": sources,
            "score_adjustment": 0.0,
            "evidence": [performance[source] for source in sources if source in performance],
            "production_permission": False,
            "reason": "模型贡献历史仅用于费后收益归因，不改变生产评分或仓位。",
        }

    def _match_entry_decision(
        self,
        position: Any,
        orders: list[Any],
        decisions: dict[int, Any],
    ) -> Any | None:
        pos_created = _aware(getattr(position, "created_at", None))
        pos_side = self._position_side(position)
        pos_symbol = normalize_trading_symbol(getattr(position, "symbol", ""))
        matched_decision = None
        best_delta = None
        for order in orders:
            order_symbol = normalize_trading_symbol(getattr(order, "symbol", ""))
            if order_symbol != pos_symbol or order.decision_id not in decisions:
                continue
            decision = decisions[order.decision_id]
            action = str(getattr(decision, "action", "") or "").lower()
            if action not in {"long", "short"} or action != pos_side:
                continue
            order_time = _aware(
                getattr(order, "filled_at", None) or getattr(order, "created_at", None)
            )
            if pos_created and order_time:
                delta = abs((order_time - pos_created).total_seconds())
                if delta > 300:
                    continue
            else:
                delta = 0.0
            if best_delta is None or delta < best_delta:
                best_delta = delta
                matched_decision = decision
        return matched_decision

    @staticmethod
    def _position_side(position: Any) -> str:
        return "short" if str(getattr(position, "side", "") or "").lower() == "short" else "long"

    @staticmethod
    def _add_sample(bucket: dict[str, Any], pnl: float) -> None:
        bucket["count"] = int(bucket.get("count") or 0) + 1
        bucket.setdefault("_pnl_samples", []).append(float(pnl))
        bucket["pnl"] = float(bucket.get("pnl") or 0.0) + pnl
        if pnl >= 0:
            bucket["wins"] = int(bucket.get("wins") or 0) + 1
            bucket["profit"] = float(bucket.get("profit") or 0.0) + pnl
        else:
            bucket["losses"] = int(bucket.get("losses") or 0) + 1
            bucket["loss"] = float(bucket.get("loss") or 0.0) + abs(pnl)

    @staticmethod
    def _finalize_bucket(bucket: dict[str, Any]) -> None:
        count = int(bucket.get("count") or 0)
        profit = float(bucket.get("profit") or 0.0)
        loss = float(bucket.get("loss") or 0.0)
        pnl = float(bucket.get("pnl") or 0.0)
        samples = [float(value) for value in bucket.pop("_pnl_samples", [])]
        if count <= 0:
            return
        win_rate = int(bucket.get("wins") or 0) / count
        profit_factor = profit / loss if loss > 0 else None
        avg_pnl = pnl / count
        variance = (
            sum((value - avg_pnl) ** 2 for value in samples) / (count - 1)
            if count > 1
            else avg_pnl**2
        )
        lcb = avg_pnl - sqrt(max(variance, 0.0) / count)
        state = (
            "positive_return_observation"
            if lcb > 0 and profit_factor is not None and profit_factor > 1
            else "negative_return_observation"
            if lcb < 0
            else "uncertain_return_observation"
        )
        bucket.update(
            {
                "pnl": round(pnl, 6),
                "profit": round(profit, 6),
                "loss": round(loss, 6),
                "avg_pnl": round(avg_pnl, 6),
                "pnl_lcb_usdt": round(lcb, 6),
                "win_rate": round(win_rate, 6),
                "profit_factor": round(profit_factor, 6) if profit_factor is not None else None,
                "state": state,
                "production_permission": False,
                "reason": "只读费后贡献归因，不参与生产评分、仓位或路由。",
            }
        )
