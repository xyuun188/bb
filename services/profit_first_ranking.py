"""Read-only Profit-First v3 model and strategy ranking."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from services.profit_first_brain_training import ProfitFirstBrainTrainingService
from services.profit_first_trade_plan import summarize_model_strategy_realized_pnl
from services.trade_fact_trust import (
    closed_position_trade_fact_untrusted_reason_with_orders,
    orders_by_exchange_id,
    split_exchange_order_ids,
)

DEFAULT_RANKING_HOURS = 72
DEFAULT_RANKING_LIMIT = 800


class ProfitFirstRankingService:
    """Rank model/strategy/lane combinations by realized net PnL.

    This service is deliberately read-only.  It creates promotion/demotion
    evidence for the master control, but never mutates live routing, sizing, or
    strategy state.
    """

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] | None = None,
        min_canary_samples: int = 20,
        min_live_samples: int = 50,
        min_profit_factor: float = 1.12,
        disable_consecutive_losses: int = 3,
        max_tail_loss_usdt: float = 8.0,
        max_fee_drag_ratio: float = 0.45,
    ) -> None:
        self._session_factory = session_factory
        self.min_canary_samples = int(min_canary_samples)
        self.min_live_samples = int(min_live_samples)
        self.min_profit_factor = float(min_profit_factor)
        self.disable_consecutive_losses = int(disable_consecutive_losses)
        self.max_tail_loss_usdt = float(max_tail_loss_usdt)
        self.max_fee_drag_ratio = float(max_fee_drag_ratio)

    async def report(
        self,
        *,
        hours: int = DEFAULT_RANKING_HOURS,
        limit: int = DEFAULT_RANKING_LIMIT,
    ) -> dict[str, Any]:
        decisions, closed_positions, trade_fact_report = await self._load_recent(
            hours=hours,
            limit=limit,
        )
        return self.build_report(
            decisions=decisions,
            closed_positions=closed_positions,
            trade_fact_report=trade_fact_report,
        )

    def build_report(
        self,
        *,
        decisions: Sequence[Any],
        closed_positions: Sequence[Any],
        trade_fact_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        decisions_list = list(decisions)
        positions_list = list(closed_positions)
        fact_report = _trade_fact_report_or_default(
            trade_fact_report,
            checked=len(positions_list),
            trusted=len(positions_list),
        )
        leaderboard = summarize_model_strategy_realized_pnl(positions_list)
        brain = ProfitFirstBrainTrainingService(
            min_canary_samples=self.min_canary_samples,
            min_live_samples=self.min_live_samples,
            min_profit_factor=self.min_profit_factor,
        ).build_dataset(decisions=decisions_list, closed_positions=positions_list)
        strategy_rows = self._strategy_rows(
            leaderboard_rows=_safe_list(leaderboard.get("rows")),
            closed_positions=positions_list,
        )
        source_rows = self._source_rows(
            _safe_list(brain.get("recommendations", {}).get("source_weights"))
        )
        blockers = self._blockers(strategy_rows, source_rows)
        summary = {
            "decision_count": len(decisions_list),
            "closed_position_count": len(positions_list),
            "trusted_closed_position_count": len(positions_list),
            "checked_closed_position_count": _safe_int(fact_report.get("checked")),
            "quarantined_closed_position_count": _safe_int(fact_report.get("quarantined")),
            "leaderboard_row_count": len(strategy_rows),
            "source_row_count": len(source_rows),
            "promote_candidate_count": sum(
                1
                for row in strategy_rows
                if row.get("recommended_stage") in {"canary", "live_candidate"}
            ),
            "demote_count": sum(
                1 for row in strategy_rows if row.get("recommended_stage") == "demote"
            ),
            "disable_count": sum(
                1 for row in strategy_rows if row.get("recommended_stage") == "disable"
            ),
            "negative_source_count": sum(
                1 for row in source_rows if row.get("recommended_stage") in {"demote", "shadow"}
            ),
            "blocker_count": len(blockers),
        }
        ranking_ready = bool(strategy_rows or source_rows)
        return {
            "status": "ready" if ranking_ready else "collecting_evidence",
            "audit_only": True,
            "read_only": True,
            "live_mutation": False,
            "live_weight_mutation": False,
            "live_sizing_mutation": False,
            "can_change_model_routing": False,
            "can_change_strategy_weight": False,
            "can_increase_live_size": False,
            "requires_operator_resume_gate": True,
            "ranking_ready": ranking_ready,
            "summary": summary,
            "leaderboard": leaderboard,
            "strategy_rankings": strategy_rows[:80],
            "source_rankings": source_rows[:40],
            "blockers": blockers[:40],
            "policy": {
                "promotion_flow": "shadow_to_canary_to_live",
                "min_canary_samples": self.min_canary_samples,
                "min_live_samples": self.min_live_samples,
                "min_profit_factor": self.min_profit_factor,
                "disable_consecutive_losses": self.disable_consecutive_losses,
                "max_tail_loss_usdt": self.max_tail_loss_usdt,
                "max_fee_drag_ratio": self.max_fee_drag_ratio,
                "losing_profiles_cannot_keep_live_size": True,
                "profitable_profiles_need_clean_sample_floor": True,
                "ranking_changes_are_auditable": True,
                "trade_fact_policy": "okx_confirmed_closed_positions_only",
            },
            "trade_fact_report": fact_report,
            "brain_recommendations": brain.get("recommendations") or {},
        }

    async def _load_recent(
        self,
        *,
        hours: int,
        limit: int,
    ) -> tuple[list[Any], list[Any], dict[str, Any]]:
        from sqlalchemy import select

        from db.session import get_read_session_ctx
        from models.decision import AIDecision
        from models.trade import Order, Position

        session_factory = self._session_factory or get_read_session_ctx
        capped_hours = max(1, min(int(hours or DEFAULT_RANKING_HOURS), 24 * 14))
        capped_limit = max(50, min(int(limit or DEFAULT_RANKING_LIMIT), 3000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        async with session_factory() as session:
            decisions_result = await session.execute(
                select(AIDecision)
                .where(AIDecision.created_at >= since)
                .order_by(AIDecision.id.desc())
                .limit(capped_limit)
            )
            positions_result = await session.execute(
                select(Position)
                .where(
                    Position.is_open.is_(False),
                    Position.closed_at.is_not(None),
                    Position.closed_at >= since,
                )
                .order_by(Position.closed_at.desc(), Position.id.desc())
                .limit(capped_limit)
            )
            decisions = list(decisions_result.scalars().all())
            positions = list(positions_result.scalars().all())
            symbols = {str(_row_get(row, "symbol") or "") for row in positions}
            symbols.discard("")
            linked_order_ids = _linked_exchange_order_ids(positions)
            orders: list[Any] = []
            if symbols:
                order_result = await session.execute(
                    select(Order)
                    .where(
                        Order.status == "filled",
                        Order.symbol.in_(symbols),
                        Order.created_at >= since - timedelta(days=7),
                    )
                    .order_by(Order.filled_at.desc(), Order.created_at.desc(), Order.id.desc())
                    .limit(capped_limit * 3)
                )
                orders = list(order_result.scalars().all())
            loaded_order_ids = {
                order_id
                for order in orders
                for order_id in split_exchange_order_ids(_row_get(order, "exchange_order_id"))
            }
            missing_linked_order_ids = sorted(linked_order_ids - loaded_order_ids)
            if missing_linked_order_ids:
                linked_order_result = await session.execute(
                    select(Order)
                    .where(Order.exchange_order_id.in_(missing_linked_order_ids))
                    .order_by(Order.filled_at.desc(), Order.created_at.desc(), Order.id.desc())
                    .limit(min(len(missing_linked_order_ids), capped_limit * 3))
                )
                orders.extend(list(linked_order_result.scalars().all()))
            known_decision_ids = {_safe_int(_row_get(row, "id")) for row in decisions}
            order_decision_ids = {
                _safe_int(_row_get(order, "decision_id"))
                for order in orders
                if _safe_int(_row_get(order, "decision_id")) > 0
            }
            missing_decision_ids = sorted(order_decision_ids - known_decision_ids, reverse=True)[
                :capped_limit
            ]
            if missing_decision_ids:
                linked_result = await session.execute(
                    select(AIDecision)
                    .where(AIDecision.id.in_(missing_decision_ids))
                    .order_by(AIDecision.id.desc())
                    .limit(len(missing_decision_ids))
                )
                decisions.extend(list(linked_result.scalars().all()))
        decisions_by_id = {_safe_int(_row_get(row, "id")): row for row in decisions}
        linked_orders_by_id = orders_by_exchange_id(orders)
        trusted_positions, trade_fact_report = _filter_trusted_closed_positions_with_orders(
            positions,
            linked_orders_by_id,
        )
        _attach_entry_decisions(trusted_positions, orders, decisions_by_id)
        return decisions, trusted_positions, trade_fact_report

    def _strategy_rows(
        self,
        *,
        leaderboard_rows: list[Any],
        closed_positions: Sequence[Any],
    ) -> list[dict[str, Any]]:
        metrics = _closed_position_metrics(closed_positions)
        rows: list[dict[str, Any]] = []
        for item in leaderboard_rows:
            row = _safe_dict(item)
            if not row:
                continue
            key = _strategy_key(row)
            metric = metrics.get(key, {})
            count = _safe_int(row.get("count"))
            pnl = _safe_float(row.get("realized_net_pnl"))
            profit_factor = _safe_float(row.get("profit_factor"))
            tail_loss_usdt = _safe_float(metric.get("tail_loss_usdt"))
            consecutive_losses = _safe_int(metric.get("consecutive_losses"))
            fee_drag_ratio = _safe_float(metric.get("fee_drag_ratio"))
            fast_loss_rate = _safe_float(metric.get("fast_loss_rate"))
            stage, reasons = self._strategy_stage(
                count=count,
                pnl=pnl,
                profit_factor=profit_factor,
                tail_loss_usdt=tail_loss_usdt,
                consecutive_losses=consecutive_losses,
                fee_drag_ratio=fee_drag_ratio,
            )
            rows.append(
                {
                    **row,
                    "recommended_stage": stage,
                    "ranking_reasons": reasons,
                    "tail_loss_usdt": round(tail_loss_usdt, 6),
                    "tail_loss_count": _safe_int(metric.get("tail_loss_count")),
                    "consecutive_losses": consecutive_losses,
                    "fast_loss_rate": round(fast_loss_rate, 6),
                    "fee_drag_ratio": round(fee_drag_ratio, 6),
                    "can_increase_budget": stage in {"canary", "live_candidate"},
                    "can_keep_live_size": stage not in {"demote", "disable"},
                    "live_mutation": False,
                }
            )
        rows.sort(
            key=lambda item: (
                _stage_rank(str(item.get("recommended_stage") or "")),
                _safe_float(item.get("realized_net_pnl")),
                _safe_float(item.get("profit_factor")),
                _safe_int(item.get("count")),
            ),
            reverse=True,
        )
        return rows

    def _strategy_stage(
        self,
        *,
        count: int,
        pnl: float,
        profit_factor: float,
        tail_loss_usdt: float,
        consecutive_losses: int,
        fee_drag_ratio: float,
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []
        if consecutive_losses >= self.disable_consecutive_losses:
            reasons.append("consecutive_losses")
        if tail_loss_usdt >= self.max_tail_loss_usdt:
            reasons.append("tail_loss")
        if pnl < 0 and profit_factor < 0.85:
            reasons.append("negative_realized_net_pnl")
        if fee_drag_ratio > self.max_fee_drag_ratio and pnl <= 0:
            reasons.append("fee_drag_churn")
        if "consecutive_losses" in reasons:
            return "disable", reasons
        if "tail_loss" in reasons and count >= self.disable_consecutive_losses:
            return "disable", reasons
        if reasons:
            return "demote", reasons
        if count < self.min_canary_samples:
            return "shadow", ["sample_floor_not_met"]
        if pnl > 0 and profit_factor >= self.min_profit_factor:
            if count >= self.min_live_samples:
                return "live_candidate", ["positive_realized_net_pnl", "live_sample_floor_met"]
            return "canary", ["positive_realized_net_pnl", "canary_sample_floor_met"]
        return "shadow", ["realized_net_pnl_or_profit_factor_weak"]

    def _source_rows(self, source_weights: list[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in source_weights:
            row = _safe_dict(item)
            if not row:
                continue
            pnl = _safe_float(row.get("realized_net_pnl"))
            count = _safe_int(row.get("count"))
            if count >= 5 and pnl < 0:
                stage = "demote"
                weight = min(_safe_float(row.get("weight_multiplier"), 0.82), 0.82)
                reasons = ["negative_realized_net_pnl"]
            elif count >= 5 and pnl > 0:
                stage = "promote"
                weight = max(_safe_float(row.get("weight_multiplier"), 1.12), 1.0)
                reasons = ["positive_realized_net_pnl"]
            else:
                stage = "shadow"
                weight = 1.0
                reasons = ["sample_floor_not_met"]
            rows.append(
                {
                    **row,
                    "recommended_stage": stage,
                    "weight_multiplier": round(weight, 6),
                    "ranking_reasons": reasons,
                    "live_weight_mutation": False,
                }
            )
        rows.sort(
            key=lambda item: (
                _stage_rank(str(item.get("recommended_stage") or "")),
                _safe_float(item.get("realized_net_pnl")),
            ),
            reverse=True,
        )
        return rows

    @staticmethod
    def _blockers(
        strategy_rows: list[dict[str, Any]],
        source_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        blockers: list[dict[str, Any]] = []
        for row in strategy_rows:
            stage = str(row.get("recommended_stage") or "")
            if stage not in {"demote", "disable"}:
                continue
            blockers.append(
                {
                    "code": f"strategy_{stage}",
                    "severity": "blocking" if stage == "disable" else "warning",
                    "message": "Strategy/model/lane combination must not receive live budget increase.",
                    "evidence": {
                        "model_name": row.get("model_name"),
                        "strategy_profile_id": row.get("strategy_profile_id"),
                        "symbol": row.get("symbol"),
                        "side": row.get("side"),
                        "decision_lane": row.get("decision_lane"),
                        "realized_net_pnl": row.get("realized_net_pnl"),
                        "profit_factor": row.get("profit_factor"),
                        "ranking_reasons": row.get("ranking_reasons") or [],
                    },
                }
            )
        for row in source_rows:
            if str(row.get("recommended_stage") or "") != "demote":
                continue
            blockers.append(
                {
                    "code": "model_source_demote",
                    "severity": "warning",
                    "message": "Model source has negative realized net PnL and should stay reduced or shadow-only.",
                    "evidence": {
                        "source": row.get("source"),
                        "count": row.get("count"),
                        "realized_net_pnl": row.get("realized_net_pnl"),
                        "weight_multiplier": row.get("weight_multiplier"),
                    },
                }
        )
        blockers.sort(key=lambda item: str(item.get("severity") or "") != "blocking")
        return blockers


def _linked_exchange_order_ids(positions: Sequence[Any]) -> set[str]:
    ids: set[str] = set()
    for position in positions:
        ids.update(split_exchange_order_ids(_row_get(position, "entry_exchange_order_id")))
        ids.update(split_exchange_order_ids(_row_get(position, "close_exchange_order_id")))
    return ids


def _filter_trusted_closed_positions_with_orders(
    positions: Sequence[Any],
    linked_orders_by_id: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    position_list = list(positions)
    trusted: list[Any] = []
    reason_counts: dict[str, int] = {}
    quarantined_ids: list[int] = []
    for row in position_list:
        reason = closed_position_trade_fact_untrusted_reason_with_orders(
            row,
            linked_orders_by_id,
        )
        if reason is None:
            trusted.append(row)
            continue
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        row_id = _safe_int(_row_get(row, "id"))
        if row_id > 0:
            quarantined_ids.append(row_id)
    return trusted, {
        "policy": "okx_confirmed_closed_positions_only",
        "checked": len(position_list),
        "trusted": len(trusted),
        "quarantined": len(position_list) - len(trusted),
        "reason_counts": reason_counts,
        "position_ids": quarantined_ids[:50],
    }


def _trade_fact_report_or_default(
    value: dict[str, Any] | None,
    *,
    checked: int,
    trusted: int,
) -> dict[str, Any]:
    report = _safe_dict(value)
    if not report:
        report = {}
    checked_count = _safe_int(report.get("checked"), checked)
    trusted_count = _safe_int(report.get("trusted"), trusted)
    quarantined_count = _safe_int(report.get("quarantined"), max(checked_count - trusted_count, 0))
    return {
        "policy": str(report.get("policy") or "okx_confirmed_closed_positions_only"),
        "checked": checked_count,
        "trusted": trusted_count,
        "quarantined": quarantined_count,
        "reason_counts": _safe_dict(report.get("reason_counts")),
        "position_ids": _safe_list(report.get("position_ids"))[:50],
    }


def _closed_position_metrics(
    closed_positions: Sequence[Any],
) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "tail_loss_usdt": 0.0,
            "tail_loss_count": 0,
            "consecutive_losses": 0,
            "fast_loss_count": 0,
            "count": 0,
            "fee_total": 0.0,
            "abs_pnl_total": 0.0,
        }
    )
    ordered = sorted(
        list(closed_positions),
        key=lambda row: _parse_datetime(_row_get(row, "closed_at"))
        or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )
    loss_streak_done: set[tuple[str, str, str, str, str]] = set()
    for row in ordered:
        key = _position_key(row)
        bucket = buckets[key]
        pnl = _safe_float(_row_get(row, "realized_pnl"))
        fee = abs(_safe_float(_row_get(row, "fee")))
        bucket["count"] += 1
        bucket["fee_total"] += fee
        bucket["abs_pnl_total"] += abs(pnl)
        if pnl < 0:
            loss_abs = abs(pnl)
            if loss_abs > bucket["tail_loss_usdt"]:
                bucket["tail_loss_usdt"] = loss_abs
            if loss_abs >= 3.0:
                bucket["tail_loss_count"] += 1
            if key not in loss_streak_done:
                bucket["consecutive_losses"] += 1
        else:
            loss_streak_done.add(key)
        hold_minutes = _hold_minutes(row)
        if pnl < 0 and hold_minutes is not None and hold_minutes <= 15.0:
            bucket["fast_loss_count"] += 1
    for bucket in buckets.values():
        count = max(_safe_int(bucket.get("count")), 1)
        abs_pnl = _safe_float(bucket.get("abs_pnl_total"))
        bucket["fast_loss_rate"] = _safe_float(bucket.get("fast_loss_count")) / count
        bucket["fee_drag_ratio"] = (
            _safe_float(bucket.get("fee_total")) / abs_pnl if abs_pnl > 0 else 0.0
        )
    return dict(buckets)


def _attach_entry_decisions(
    positions: Sequence[Any],
    orders: Sequence[Any],
    decisions_by_id: dict[int, Any],
) -> None:
    order_list = list(orders)
    for position in positions:
        decision = _match_entry_decision(position, order_list, decisions_by_id)
        if decision is None:
            continue
        raw = _safe_dict(_row_get(decision, "raw_llm_response"))
        if not raw:
            continue
        try:
            setattr(position, "entry_raw", raw)
            setattr(position, "entry_decision_id", _row_get(decision, "id"))
        except Exception:
            continue


def _match_entry_decision(
    position: Any,
    orders: Sequence[Any],
    decisions_by_id: dict[int, Any],
) -> Any | None:
    position_symbol = str(_row_get(position, "symbol") or "")
    position_side = str(_row_get(position, "side") or "").lower()
    position_created = _parse_datetime(_row_get(position, "created_at"))
    entry_exchange_order_id = str(_row_get(position, "entry_exchange_order_id") or "").strip()
    best: Any | None = None
    best_delta: float | None = None
    for order in orders:
        if position_symbol and str(_row_get(order, "symbol") or "") != position_symbol:
            continue
        decision = decisions_by_id.get(_safe_int(_row_get(order, "decision_id")))
        if decision is None:
            continue
        action = str(_row_get(decision, "action") or "").lower()
        if action not in {"long", "short"} or action != position_side:
            continue
        order_exchange_id = str(_row_get(order, "exchange_order_id") or "").strip()
        if entry_exchange_order_id and order_exchange_id == entry_exchange_order_id:
            return decision
        order_time = _parse_datetime(_row_get(order, "filled_at") or _row_get(order, "created_at"))
        if position_created is not None and order_time is not None:
            delta = abs((position_created - order_time).total_seconds())
            if delta > 15 * 60:
                continue
        else:
            delta = 0.0
        if best_delta is None or delta < best_delta:
            best = decision
            best_delta = delta
    return best


def _strategy_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("model_name") or "unknown"),
        str(row.get("strategy_profile_id") or "unknown"),
        str(row.get("symbol") or "unknown"),
        str(row.get("side") or "unknown"),
        str(row.get("decision_lane") or "unknown"),
    )


def _position_key(row: Any) -> tuple[str, str, str, str, str]:
    raw = _safe_dict(_row_get(row, "entry_raw") or _row_get(row, "raw_llm_response"))
    plan = _safe_dict(raw.get("profit_first_trade_plan"))
    return (
        str(_row_get(row, "model_name") or "unknown"),
        str(plan.get("strategy_profile_id") or "unknown"),
        str(_row_get(row, "symbol") or "unknown"),
        str(_row_get(row, "side") or "unknown"),
        str(plan.get("decision_lane") or "unknown"),
    )


def _stage_rank(stage: str) -> int:
    return {
        "live_candidate": 5,
        "canary": 4,
        "promote": 3,
        "shadow": 2,
        "demote": 1,
        "disable": 0,
    }.get(stage, 0)


def _hold_minutes(row: Any) -> float | None:
    opened = _parse_datetime(_row_get(row, "created_at") or _row_get(row, "entry_at"))
    closed = _parse_datetime(_row_get(row, "closed_at"))
    if opened is None or closed is None:
        return None
    return max((closed - opened).total_seconds() / 60.0, 0.0)


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


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
