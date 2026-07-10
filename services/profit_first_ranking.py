"""Read-only Profit-First v3 model and strategy ranking."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
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
_RANKING_DECISION_RAW_KEYS = (
    "profit_first_trade_plan",
    "reason",
    "skip_reason",
    "skip_kind",
    "rejection_reason",
    "opportunity_score",
    "entry_filters",
    "review_feedback",
    "memory_feedback",
    "shadow_outcome",
    "shadow_result",
    "missed_opportunity",
    "shadow_return_pct",
    "shadow_realized_return_pct",
    "realized_return_pct",
    "return_pct",
    "missed_opportunity_return_pct",
)
_RANKING_DECISION_RAW_COLUMN_PREFIX = "ranking_raw__"


def _ranking_decision_columns(AIDecision: Any) -> tuple[Any, ...]:
    return (
        AIDecision.id,
        AIDecision.model_name,
        AIDecision.symbol,
        AIDecision.action,
        AIDecision.was_executed,
        AIDecision.execution_reason,
        *(
            AIDecision.decision_learning_snapshot[key].label(
                f"{_RANKING_DECISION_RAW_COLUMN_PREFIX}{key}"
            )
            for key in _RANKING_DECISION_RAW_KEYS
        ),
    )


def _ranking_decision_from_mapping(mapping: Any) -> SimpleNamespace:
    raw = {
        key: mapping.get(f"{_RANKING_DECISION_RAW_COLUMN_PREFIX}{key}")
        for key in _RANKING_DECISION_RAW_KEYS
        if mapping.get(f"{_RANKING_DECISION_RAW_COLUMN_PREFIX}{key}") is not None
    }
    return SimpleNamespace(
        id=mapping.get("id"),
        model_name=mapping.get("model_name"),
        symbol=mapping.get("symbol"),
        action=mapping.get("action"),
        was_executed=bool(mapping.get("was_executed")),
        execution_reason=mapping.get("execution_reason"),
        raw_llm_response=raw,
    )


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
        runtime_feedback = self._runtime_feedback(
            decisions=decisions_list,
            strategy_rows=strategy_rows,
            source_rows=source_rows,
            closed_positions=positions_list,
            trade_fact_report=fact_report,
            brain_recommendations=_safe_dict(brain.get("recommendations")),
        )
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
            "exit_plan_reference_missing_count": _safe_int(
                _safe_dict(runtime_feedback.get("exit_plan_reference")).get("missing_count")
            ),
            "generated_strategy_count": _safe_int(
                _safe_dict(runtime_feedback.get("strategy_lifecycle")).get("generated_count")
            ),
            "strategy_lifecycle_stage_counts": _safe_dict(
                _safe_dict(runtime_feedback.get("strategy_lifecycle")).get("stage_counts")
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
                "runtime_feedback_policy": (
                    "bounded_strategy_context_feedback_only; no order submission, no direct "
                    "routing mutation, and no direct live-size increase"
                ),
            },
            "trade_fact_report": fact_report,
            "brain_recommendations": brain.get("recommendations") or {},
            "runtime_feedback": runtime_feedback,
        }

    def _runtime_feedback(
        self,
        *,
        decisions: Sequence[Any],
        strategy_rows: list[dict[str, Any]],
        source_rows: list[dict[str, Any]],
        closed_positions: Sequence[Any],
        trade_fact_report: dict[str, Any],
        brain_recommendations: dict[str, Any],
    ) -> dict[str, Any]:
        side_feedback = _side_runtime_feedback(closed_positions)
        exit_plan_reference = _exit_plan_reference_report(closed_positions)
        strategy_feedback = _strategy_runtime_feedback(strategy_rows)
        strategy_lifecycle = _strategy_lifecycle_report(decisions, strategy_rows)
        source_feedback = _source_runtime_feedback(source_rows)
        lane_feedback = _lane_runtime_feedback(
            _safe_list(brain_recommendations.get("lane_threshold_recommendations"))
        )
        size_feedback = _size_runtime_feedback(
            _safe_list(brain_recommendations.get("size_promotion_demotion"))
        )
        missed_opportunity_feedback = _missed_opportunity_runtime_feedback(
            _safe_dict(brain_recommendations.get("no_entry_governance"))
        )
        exit_feedback = _exit_runtime_feedback(
            _safe_dict(brain_recommendations.get("losing_exit_governance"))
        )
        acceptance = _profit_acceptance_report(
            closed_positions=closed_positions,
            strategy_feedback=strategy_feedback,
            source_feedback=source_feedback,
            side_feedback=side_feedback,
        )
        return {
            "status": "ready" if closed_positions or strategy_rows or source_rows else "collecting_evidence",
            "objective": "maximize_realized_net_pnl",
            "audit_only": True,
            "read_only": True,
            "live_mutation": False,
            "live_weight_mutation": False,
            "live_sizing_mutation": False,
            "can_submit_orders": False,
            "can_change_model_routing": False,
            "can_change_strategy_weight": False,
            "can_increase_live_size": False,
            "can_influence_strategy_context": True,
            "objective_basis": {
                "metric": "closed_position_realized_pnl",
                "cost_policy": "optimize_realized_net_pnl_after_recorded_costs",
                "window_policy": "rolling_closed_position_window",
            },
            "side_weights": {
                side: row["weight_multiplier"]
                for side, row in side_feedback.items()
                if row.get("weight_multiplier") is not None
            },
            "side_feedback": side_feedback,
            "strategy_profile_feedback": strategy_feedback[:40],
            "strategy_lifecycle": strategy_lifecycle,
            "source_weight_feedback": source_feedback[:40],
            "lane_feedback": lane_feedback[:24],
            "size_feedback": size_feedback[:24],
            "missed_opportunity_feedback": missed_opportunity_feedback,
            "exit_feedback": exit_feedback[:24],
            "local_ml_live_influence": _local_ml_live_influence(source_feedback),
            "exit_plan_reference": exit_plan_reference,
            "profit_acceptance": acceptance,
            "trade_fact_report": {
                "policy": trade_fact_report.get("policy"),
                "checked": _safe_int(trade_fact_report.get("checked")),
                "trusted": _safe_int(trade_fact_report.get("trusted")),
                "quarantined": _safe_int(trade_fact_report.get("quarantined")),
                "reason_counts": _safe_dict(trade_fact_report.get("reason_counts")),
            },
            "policy": {
                "feedback_type": "bounded_strategy_context_feedback",
                "optimization_target": "realized_net_pnl",
                "side_weight_policy": "relative_window_realized_pnl_not_fixed_usdt_thresholds",
                "strategy_flow": "shadow_to_canary_to_live",
                "strategy_lifecycle_policy": (
                    "generated_to_shadow_to_canary_to_live_or_demote_by_rolling_realized_pnl"
                ),
                "low_sample_policy": "observe_or_shadow_until_window_evidence_improves",
                "direct_order_mutation_allowed": False,
                "direct_live_size_increase_allowed": False,
                "hard_direction_ban_allowed": False,
                "exit_plan_reference_required_for_clean_attribution": True,
            },
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
        from services.position_settlement import final_settlement_status_values

        session_factory = self._session_factory or get_read_session_ctx
        capped_hours = max(1, min(int(hours or DEFAULT_RANKING_HOURS), 24 * 14))
        capped_limit = max(50, min(int(limit or DEFAULT_RANKING_LIMIT), 3000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        async with session_factory() as session:
            decisions_result = await session.execute(
                select(*_ranking_decision_columns(AIDecision))
                .where(AIDecision.created_at >= since)
                .order_by(AIDecision.id.desc())
                .limit(capped_limit)
            )
            positions_result = await session.execute(
                select(Position)
                .where(
                    Position.is_open.is_(False),
                    Position.settlement_status.in_(final_settlement_status_values()),
                    Position.closed_at.is_not(None),
                    Position.closed_at >= since,
                )
                .order_by(Position.closed_at.desc(), Position.id.desc())
                .limit(capped_limit)
            )
            decisions = [
                _ranking_decision_from_mapping(row) for row in decisions_result.mappings().all()
            ]
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
                    select(*_ranking_decision_columns(AIDecision))
                    .where(AIDecision.id.in_(missing_decision_ids))
                    .order_by(AIDecision.id.desc())
                    .limit(len(missing_decision_ids))
                )
                decisions.extend(
                    _ranking_decision_from_mapping(row) for row in linked_result.mappings().all()
                )
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
            lifecycle = _strategy_lifecycle_record(
                row,
                recommended_stage=stage,
                ranking_reasons=reasons,
                count=count,
                realized_net_pnl=pnl,
                profit_factor=profit_factor,
            )
            rows.append(
                {
                    **row,
                    "recommended_stage": stage,
                    "lifecycle_stage": lifecycle["lifecycle_stage"],
                    "strategy_lifecycle": lifecycle,
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


def _side_runtime_feedback(closed_positions: Sequence[Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {
        "long": _empty_side_runtime_bucket("long"),
        "short": _empty_side_runtime_bucket("short"),
    }
    for row in closed_positions:
        side = str(_row_get(row, "side") or "").lower().strip()
        if side not in stats:
            continue
        pnl = _safe_float(_row_get(row, "realized_pnl"), 0.0)
        bucket = stats[side]
        bucket["count"] += 1
        bucket["realized_net_pnl"] += pnl
        bucket["wins"] += 1 if pnl > 0 else 0
        bucket["losses"] += 1 if pnl < 0 else 0
        bucket["profit"] += max(pnl, 0.0)
        bucket["loss"] += abs(min(pnl, 0.0))

    total_count = sum(_safe_int(row.get("count")) for row in stats.values())
    sample_floor = max(1, min(8, (total_count + 3) // 4)) if total_count else 1
    total_abs_pnl = sum(abs(_safe_float(row.get("realized_net_pnl"))) for row in stats.values())
    total_abs_pnl = max(total_abs_pnl, 1e-9)
    for side, bucket in stats.items():
        other_side = "short" if side == "long" else "long"
        other = stats[other_side]
        count = _safe_int(bucket.get("count"))
        pnl = _safe_float(bucket.get("realized_net_pnl"))
        profit = _safe_float(bucket.get("profit"))
        loss = _safe_float(bucket.get("loss"))
        other_pnl = _safe_float(other.get("realized_net_pnl"))
        profit_factor = profit / loss if loss > 0 else (999.0 if profit > 0 else 0.0)
        avg_pnl = pnl / count if count else 0.0
        reasons: list[str] = []
        stage = "observe"
        weight = 1.0
        if count < sample_floor:
            stage = "shadow"
            reasons.append("sample_floor_not_met")
        elif pnl < 0:
            loss_share = abs(pnl) / total_abs_pnl
            underperformance_share = max(other_pnl - pnl, 0.0) / total_abs_pnl
            profit_factor_pressure = max(0.0, 1.0 - min(profit_factor, 1.0))
            reduction = min(
                0.55,
                0.18
                + loss_share * 0.24
                + underperformance_share * 0.18
                + profit_factor_pressure * 0.16,
            )
            weight = max(0.45, 1.0 - reduction)
            stage = "demote"
            reasons.extend(["negative_realized_net_pnl", "relative_window_underperformance"])
        elif pnl > 0:
            advantage_share = max(pnl - other_pnl, 0.0) / total_abs_pnl
            if advantage_share > 0:
                weight = min(1.12, 1.0 + advantage_share * 0.10)
                stage = "promote"
                reasons.append("positive_relative_realized_net_pnl")
        bucket.update(
            {
                "sample_floor": sample_floor,
                "avg_pnl": round(avg_pnl, 6),
                "win_rate": round(_safe_int(bucket.get("wins")) / count, 6)
                if count
                else 0.0,
                "profit_factor": round(profit_factor, 6),
                "relative_edge_vs_opposite": round(pnl - other_pnl, 6),
                "recommended_stage": stage,
                "weight_multiplier": round(weight, 6),
                "ranking_reasons": reasons or ["observe_realized_net_pnl"],
                "hard_ban": False,
                "live_weight_mutation": False,
                "policy": "relative_window_realized_pnl_not_fixed_usdt_thresholds",
            }
        )
        for key in ("realized_net_pnl", "profit", "loss"):
            bucket[key] = round(_safe_float(bucket.get(key)), 6)
    return stats


def _empty_side_runtime_bucket(side: str) -> dict[str, Any]:
    return {
        "side": side,
        "count": 0,
        "wins": 0,
        "losses": 0,
        "realized_net_pnl": 0.0,
        "profit": 0.0,
        "loss": 0.0,
    }


def _lifecycle_stage(recommended_stage: str, *, count: int) -> str:
    stage = str(recommended_stage or "shadow")
    if stage == "disable":
        return "quarantine"
    if stage == "demote":
        return "demote"
    if stage == "live_candidate":
        return "live_candidate"
    if stage == "canary":
        return "canary_candidate"
    if count > 0:
        return "shadow_observation"
    return "generated"


def _strategy_lifecycle_record(
    row: dict[str, Any],
    *,
    recommended_stage: str,
    ranking_reasons: list[str],
    count: int,
    realized_net_pnl: float,
    profit_factor: float,
) -> dict[str, Any]:
    lifecycle_stage = _lifecycle_stage(recommended_stage, count=count)
    next_action_by_stage = {
        "generated": "collect_shadow_or_execution_evidence",
        "shadow_observation": "keep_shadow_until_rolling_evidence_improves",
        "canary_candidate": "operator_review_for_small_budget_canary",
        "live_candidate": "operator_review_for_live_budget_after_clean_window",
        "demote": "reduce_or_disable_budget_until_shadow_recovery",
        "quarantine": "stop_budget_until_root_cause_review",
    }
    return {
        "policy": "generated_to_shadow_to_canary_to_live_or_demote",
        "lifecycle_stage": lifecycle_stage,
        "recommended_stage": recommended_stage,
        "strategy_profile_id": row.get("strategy_profile_id"),
        "model_name": row.get("model_name"),
        "symbol": row.get("symbol"),
        "side": row.get("side"),
        "decision_lane": row.get("decision_lane"),
        "closed_trade_count": count,
        "realized_net_pnl": round(realized_net_pnl, 6),
        "profit_factor": round(profit_factor, 6),
        "ranking_reasons": list(ranking_reasons),
        "transition_path": ["generated", "shadow", "canary", "live"],
        "next_action": next_action_by_stage.get(
            lifecycle_stage,
            "keep_shadow_until_rolling_evidence_improves",
        ),
        "operator_gate_required": lifecycle_stage in {"canary_candidate", "live_candidate"},
        "live_mutation": False,
    }


def _decision_profit_first_plan(row: Any) -> dict[str, Any]:
    raw = _safe_dict(_row_get(row, "raw_llm_response") or _row_get(row, "raw_response"))
    return _safe_dict(raw.get("profit_first_trade_plan"))


def _decision_lifecycle_key(row: Any, plan: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(_row_get(row, "model_name") or "decision_llm"),
        str(plan.get("strategy_profile_id") or "unknown"),
        str(_row_get(row, "symbol") or plan.get("symbol") or "unknown"),
        str(_row_get(row, "action") or plan.get("side") or plan.get("action") or "unknown"),
        str(plan.get("decision_lane") or "unknown"),
    )


def _strategy_generation_rows(decisions: Sequence[Any]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for decision in decisions:
        plan = _decision_profit_first_plan(decision)
        if not plan:
            continue
        key = _decision_lifecycle_key(decision, plan)
        row = rows.setdefault(
            key,
            {
                "model_name": key[0],
                "strategy_profile_id": key[1],
                "symbol": key[2],
                "side": key[3],
                "decision_lane": key[4],
                "generated_count": 0,
                "executed_entry_count": 0,
                "skipped_count": 0,
                "expected_net_return_sum_pct": 0.0,
                "profit_quality_sum": 0.0,
                "profit_quality_count": 0,
            },
        )
        row["generated_count"] += 1
        if bool(_row_get(decision, "was_executed")):
            row["executed_entry_count"] += 1
        else:
            row["skipped_count"] += 1
        row["expected_net_return_sum_pct"] += _safe_float(
            plan.get("expected_net_return_pct"),
            0.0,
        )
        quality = plan.get("profit_quality_ratio")
        if quality is not None:
            row["profit_quality_sum"] += _safe_float(quality, 0.0)
            row["profit_quality_count"] += 1
    result = []
    for row in rows.values():
        generated = _safe_int(row.get("generated_count"))
        quality_count = _safe_int(row.get("profit_quality_count"))
        row["avg_expected_net_return_pct"] = round(
            _safe_float(row.pop("expected_net_return_sum_pct"), 0.0) / max(generated, 1),
            6,
        )
        row["avg_profit_quality_ratio"] = (
            round(_safe_float(row.pop("profit_quality_sum"), 0.0) / quality_count, 6)
            if quality_count
            else None
        )
        row.pop("profit_quality_count", None)
        row["lifecycle_stage"] = "generated"
        row["next_action"] = "collect_shadow_or_execution_evidence"
        row["live_mutation"] = False
        result.append(row)
    result.sort(
        key=lambda item: (
            _safe_int(item.get("generated_count")),
            _safe_float(item.get("avg_expected_net_return_pct")),
        ),
        reverse=True,
    )
    return result


def _strategy_lifecycle_report(
    decisions: Sequence[Any],
    strategy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_rows = _strategy_generation_rows(decisions)
    stage_counts = Counter(
        str(_safe_dict(row.get("strategy_lifecycle")).get("lifecycle_stage") or "shadow")
        for row in strategy_rows
    )
    if generated_rows:
        stage_counts["generated"] += sum(
            _safe_int(row.get("generated_count")) for row in generated_rows
        )
    return {
        "policy": "generated_to_shadow_to_canary_to_live_or_demote_by_rolling_realized_pnl",
        "read_only": True,
        "live_mutation": False,
        "generated_count": sum(_safe_int(row.get("generated_count")) for row in generated_rows),
        "executed_entry_count": sum(
            _safe_int(row.get("executed_entry_count")) for row in generated_rows
        ),
        "closed_strategy_count": len(strategy_rows),
        "stage_counts": dict(stage_counts),
        "generated_profiles": generated_rows[:40],
        "ranked_profiles": [
            _safe_dict(row.get("strategy_lifecycle")) for row in strategy_rows[:40]
        ],
    }


def _strategy_runtime_feedback(strategy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in strategy_rows:
        stage = str(row.get("recommended_stage") or "shadow")
        if stage == "disable":
            multiplier = 0.35
        elif stage == "demote":
            multiplier = 0.62
        elif stage == "canary":
            multiplier = 1.02
        elif stage == "live_candidate":
            multiplier = 1.05
        else:
            multiplier = 1.0
        result.append(
            {
                "model_name": row.get("model_name"),
                "strategy_profile_id": row.get("strategy_profile_id"),
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "decision_lane": row.get("decision_lane"),
                "recommended_stage": stage,
                "lifecycle_stage": row.get("lifecycle_stage"),
                "strategy_lifecycle": _safe_dict(row.get("strategy_lifecycle")),
                "weight_multiplier": round(multiplier, 6),
                "count": row.get("count"),
                "realized_net_pnl": row.get("realized_net_pnl"),
                "profit_factor": row.get("profit_factor"),
                "ranking_reasons": _safe_list(row.get("ranking_reasons")),
                "can_increase_budget": False,
                "can_keep_live_size": stage not in {"demote", "disable"},
                "live_mutation": False,
            }
        )
    return result


def _source_runtime_feedback(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in source_rows:
        stage = str(row.get("recommended_stage") or "shadow")
        weight = _safe_float(row.get("weight_multiplier"), 1.0)
        result.append(
            {
                "source": row.get("source"),
                "recommended_stage": stage,
                "weight_multiplier": round(min(max(weight, 0.50), 1.15), 6),
                "count": row.get("count"),
                "realized_net_pnl": row.get("realized_net_pnl"),
                "ranking_reasons": _safe_list(row.get("ranking_reasons")),
                "can_change_model_routing": False,
                "live_weight_mutation": False,
            }
        )
    return result


def _local_ml_live_influence(source_feedback: list[dict[str, Any]]) -> dict[str, Any]:
    row = next(
        (
            item
            for item in source_feedback
            if str(item.get("source") or "").lower() in {"local_ml", "ml", "local_ml_model"}
        ),
        {},
    )
    if not row:
        return {
            "source": "local_ml",
            "allow_live_entry_influence": False,
            "reason": "no_realized_net_pnl_source_evidence",
            "requires_top_bucket_positive_confirmation": True,
        }
    stage = str(row.get("recommended_stage") or "shadow")
    pnl = _safe_float(row.get("realized_net_pnl"), 0.0)
    return {
        "source": row.get("source") or "local_ml",
        "allow_live_entry_influence": False,
        "eligible_for_shadow_to_canary_review": bool(stage == "promote" and pnl > 0),
        "reason": (
            "requires_top_bucket_positive_confirmation"
            if stage == "promote" and pnl > 0
            else "degraded_or_negative_realized_net_pnl_source"
        ),
        "recommended_stage": stage,
        "realized_net_pnl": row.get("realized_net_pnl"),
        "weight_multiplier": min(_safe_float(row.get("weight_multiplier"), 1.0), 1.0),
        "requires_top_bucket_positive_confirmation": True,
        "can_change_model_routing": False,
    }


def _lane_runtime_feedback(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        item = _safe_dict(row)
        recommendation = str(item.get("recommendation") or "")
        bias = "observe"
        if recommendation in {
            "allow_lane_promotion_review",
            "review_shadow_to_tiny_or_validated_thresholds",
        }:
            bias = "expand_quality_entries"
        elif recommendation in {
            "tighten_or_keep_lane_threshold",
            "pause_or_raise_quality_floor_for_tiny_probe",
        }:
            bias = "tighten_or_limit_weak_entries"
        result.append(
            {
                "lane": item.get("lane"),
                "recommendation": recommendation,
                "reason": item.get("reason"),
                "count": item.get("count"),
                "realized_net_pnl": item.get("realized_net_pnl"),
                "profit_factor": item.get("profit_factor"),
                "entry_bias": bias,
                "live_mutation": False,
            }
        )
    return result


def _size_runtime_feedback(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        item = _safe_dict(row)
        recommendation = str(item.get("recommendation") or "")
        bias = "observe"
        if recommendation == "eligible_for_budget_increase_after_operator_gate":
            bias = "quality_entries_can_expand_after_validation"
        elif recommendation in {
            "reduce_or_disable_budget",
            "do_not_continue_tiny_size_when_fee_drag_losses_repeat",
        }:
            bias = "reduce_weak_or_fee_drag_size"
        elif recommendation == "keep_shadow_or_sampling_size":
            bias = "keep_sampling_size"
        result.append(
            {
                "model_name": item.get("model_name"),
                "strategy_profile_id": item.get("strategy_profile_id"),
                "symbol": item.get("symbol"),
                "side": item.get("side"),
                "decision_lane": item.get("decision_lane"),
                "recommended_stage": item.get("recommended_stage"),
                "recommendation": recommendation,
                "sizing_bias": bias,
                "evidence": _safe_dict(item.get("evidence")),
                "live_mutation": False,
            }
        )
    return result


def _missed_opportunity_runtime_feedback(no_entry_governance: dict[str, Any]) -> dict[str, Any]:
    governance = _safe_dict(no_entry_governance)
    return {
        "sample_count": _safe_int(governance.get("sample_count")),
        "diagnosis": governance.get("diagnosis"),
        "missed_positive_shadow_count": _safe_int(
            governance.get("missed_positive_shadow_count")
        ),
        "missed_shadow_return_total_pct": _safe_float(
            governance.get("missed_shadow_return_total_pct")
        ),
        "reason_counts": _safe_list(governance.get("reason_counts"))[:12],
        "recommendations": _safe_list(governance.get("recommendations"))[:12],
        "entry_bias": (
            "expand_quality_entries"
            if governance.get("diagnosis") == "system_over_conservative_review"
            else "observe"
        ),
        "live_mutation": False,
    }


def _exit_runtime_feedback(losing_exit_governance: dict[str, Any]) -> list[dict[str, Any]]:
    governance = _safe_dict(losing_exit_governance)
    counts = {
        str(_safe_dict(item).get("value") or ""): _safe_int(_safe_dict(item).get("count"))
        for item in _safe_list(governance.get("attribution_counts"))
    }
    result: list[dict[str, Any]] = []
    for item in _safe_list(governance.get("exit_policy_adjustments")):
        row = _safe_dict(item)
        attribution = str(row.get("attribution") or "")
        recommendation = str(row.get("recommendation") or "")
        exit_bias = "observe"
        if attribution in {"exit_too_early", "hold_too_short"}:
            exit_bias = "hold_winners_longer"
        elif attribution in {"exit_too_late", "capital_release_forced_loss"}:
            exit_bias = "cut_losers_faster"
        elif attribution == "position_too_small_fee_drag":
            exit_bias = "keep_tiny_entries_shadow_only"
        elif attribution in {
            "entry_wrong_direction",
            "model_false_positive",
            "timeseries_false_signal",
            "sentiment_false_signal",
        }:
            exit_bias = "demote_false_positive_inputs"
        result.append(
            {
                "attribution": attribution,
                "recommendation": recommendation,
                "count": counts.get(attribution, _safe_int(row.get("count"))),
                "exit_bias": exit_bias,
                "live_mutation": False,
            }
        )
    return result


def _exit_plan_reference_report(closed_positions: Sequence[Any]) -> dict[str, Any]:
    missing_ids: list[int] = []
    present = 0
    for row in closed_positions:
        if _position_exit_plan_id(row):
            present += 1
            continue
        row_id = _safe_int(_row_get(row, "id"))
        if row_id > 0:
            missing_ids.append(row_id)
    total = len(list(closed_positions))
    missing = max(total - present, 0)
    return {
        "checked_count": total,
        "present_count": present,
        "missing_count": missing,
        "coverage_ratio": round(present / total, 6) if total else 0.0,
        "missing_position_ids": missing_ids[:50],
        "clean_attribution_ready": bool(total > 0 and missing == 0),
        "training_attribution_blocker": bool(missing > 0),
        "policy": "closed_trade_training_requires_profit_first_exit_plan_reference",
    }


def _position_exit_plan_id(row: Any) -> str:
    raw = _safe_dict(_row_get(row, "entry_raw") or _row_get(row, "raw_llm_response"))
    plan = _safe_dict(raw.get("profit_first_trade_plan"))
    exit_plan = _safe_dict(raw.get("profit_first_exit_plan"))
    reference = _safe_dict(raw.get("profit_first_exit_reference"))
    close_evidence = _safe_dict(raw.get("close_evidence"))
    for value in (
        _row_get(row, "profit_first_exit_plan_id"),
        reference.get("exit_plan_id"),
        close_evidence.get("profit_first_exit_plan_id"),
        exit_plan.get("exit_plan_id"),
        plan.get("exit_plan_id"),
        raw.get("profit_first_exit_plan_id"),
        raw.get("exit_plan_id"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _profit_acceptance_report(
    *,
    closed_positions: Sequence[Any],
    strategy_feedback: list[dict[str, Any]],
    source_feedback: list[dict[str, Any]],
    side_feedback: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pnls = [_safe_float(_row_get(row, "realized_pnl"), 0.0) for row in closed_positions]
    wins = [value for value in pnls if value > 0]
    losses = [abs(value) for value in pnls if value < 0]
    profit = sum(wins)
    loss = sum(losses)
    return {
        "window_closed_trade_count": len(pnls),
        "net_pnl": round(sum(pnls), 6),
        "profit_factor": round(profit / loss, 6) if loss > 0 else (999.0 if profit > 0 else 0.0),
        "avg_win": round(profit / len(wins), 6) if wins else 0.0,
        "avg_loss": round(loss / len(losses), 6) if losses else 0.0,
        "side_contribution": {
            side: {
                "realized_net_pnl": row.get("realized_net_pnl"),
                "weight_multiplier": row.get("weight_multiplier"),
                "recommended_stage": row.get("recommended_stage"),
            }
            for side, row in side_feedback.items()
        },
        "promoted_strategies": [
            row
            for row in strategy_feedback
            if row.get("recommended_stage") in {"canary", "live_candidate"}
        ][:12],
        "demoted_strategies": [
            row
            for row in strategy_feedback
            if row.get("recommended_stage") in {"demote", "disable"}
        ][:12],
        "promoted_sources": [
            row for row in source_feedback if row.get("recommended_stage") == "promote"
        ][:12],
        "demoted_sources": [
            row for row in source_feedback if row.get("recommended_stage") == "demote"
        ][:12],
        "before_after_available": False,
        "before_after_note": "current rolling window only; compare with previous reports for lift",
    }


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
        with suppress(Exception):
            position.entry_raw = raw
            position.entry_decision_id = _row_get(decision, "id")


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
