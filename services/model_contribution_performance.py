"""Closed-loop performance by decision evidence source.

Entry scoring uses several evidence sources: local ML, server profit model,
time-series prediction, sentiment, shadow memory, and expert agreement.  This
service measures which sources actually led to realized profit recently, then
turns that feedback into bounded score and size adjustments.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select

from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text
from db.session import get_session_ctx
from models.decision import AIDecision
from models.trade import Order, Position
from services.manual_close_marker import position_has_manual_close_order

SessionFactory = Callable[[], Any]

DEFAULT_CONTRIBUTION_LOOKBACK_DAYS = 7.0
DEFAULT_POSITION_LIMIT = 800
DEFAULT_ORDER_LIMIT = 3000

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
        "score_multiplier": 1.0,
        "size_multiplier": 1.0,
        "state": "learning",
        "reason": "样本不足，先学习不强干预。",
    }


def _default_stats() -> dict[str, dict[str, Any]]:
    return {
        "ml_profit_model": _empty_bucket("本地 ML 盈利模型"),
        "server_profit_model": _empty_bucket("服务器盈利模型"),
        "timeseries_model": _empty_bucket("时序预测模型"),
        "sentiment_model": _empty_bucket("情绪模型"),
        "shadow_memory": _empty_bucket("影子/交易记忆"),
        "expert_alignment": _empty_bucket("专家一致信号"),
        "ai_only_without_quant": _empty_bucket("AI 单独支持但量化未同向"),
    }


class ModelContributionPerformanceService:
    """Measure recent realized PnL by evidence source and score its reliability."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory = get_session_ctx,
        model_name: str = ENSEMBLE_TRADER_NAME,
        lookback_days: float = DEFAULT_CONTRIBUTION_LOOKBACK_DAYS,
        position_limit: int = DEFAULT_POSITION_LIMIT,
        order_limit: int = DEFAULT_ORDER_LIMIT,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._model_name = model_name
        self._lookback_days = float(lookback_days)
        self._position_limit = int(position_limit)
        self._order_limit = int(order_limit)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._cache: dict[str, Any] = {"expires_at": None, "stats": {}}

    async def recent(self, mode: str) -> dict[str, dict[str, Any]]:
        """Return cached recent contribution performance for the selected mode."""

        selected_mode = "live" if mode == "live" else "paper"
        now = _aware(self._clock()) or datetime.now(UTC)
        expires_at = self._cache.get("expires_at")
        cached_stats = self._cache.get("stats")
        if isinstance(expires_at, datetime) and expires_at > now and isinstance(cached_stats, dict):
            return cached_stats

        start_utc = now - timedelta(days=self._lookback_days)
        stats = _default_stats()
        try:
            async with self._session_factory() as session:
                positions_result = await session.execute(
                    select(Position)
                    .where(
                        Position.model_name == self._model_name,
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
                    self._cache = {
                        "expires_at": now + timedelta(minutes=15),
                        "stats": stats,
                    }
                    return stats

                symbols = {p.symbol for p in positions if p.symbol}
                manual_close_orders = []
                if symbols:
                    manual_close_result = await session.execute(
                        select(Order).where(
                            Order.model_name == self._model_name,
                            Order.execution_mode == selected_mode,
                            Order.status == "filled",
                            Order.symbol.in_(symbols),
                            Order.exchange_order_id.like("manual_close:%"),
                        )
                    )
                    manual_close_orders = list(manual_close_result.scalars().all())
                positions = [
                    pos
                    for pos in positions
                    if not position_has_manual_close_order(pos, manual_close_orders)
                ]
                if not positions:
                    self._cache = {
                        "expires_at": now + timedelta(minutes=15),
                        "stats": stats,
                    }
                    return stats
                symbols = {p.symbol for p in positions if p.symbol}
                symbol_filter = Order.symbol.in_(symbols) if symbols else Order.id == -1
                orders_result = await session.execute(
                    select(Order)
                    .where(
                        Order.model_name == self._model_name,
                        Order.execution_mode == selected_mode,
                        Order.status == "filled",
                        Order.decision_id.is_not(None),
                        symbol_filter,
                    )
                    .order_by(Order.filled_at.desc(), Order.created_at.desc())
                    .limit(self._order_limit)
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
        self._cache = {
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

    def contribution_sources(
        self,
        opportunity: dict[str, Any],
        raw: dict[str, Any],
        side: str,
    ) -> list[str]:
        """Infer which evidence sources supported an entry decision."""

        sources: list[str] = []
        if bool(opportunity.get("ml_aligned")):
            sources.append("ml_profit_model")
        if bool(opportunity.get("local_profit_aligned")):
            sources.append("server_profit_model")
        if bool(opportunity.get("timeseries_aligned")):
            sources.append("timeseries_model")
        evidence_score = _safe_dict(opportunity.get("evidence_score"))
        components = _safe_list(evidence_score.get("components"))
        for item in components:
            if not isinstance(item, dict) or item.get("status") != "aligned":
                continue
            if item.get("source") == "sentiment":
                sources.append("sentiment_model")
            elif item.get("source") == "shadow_memory":
                sources.append("shadow_memory")
        if bool(opportunity.get("expert_aligned")):
            sources.append("expert_alignment")
        has_quant = any(
            source in sources
            for source in ("ml_profit_model", "server_profit_model", "timeseries_model")
        )
        if not has_quant and side in {"long", "short"}:
            sources.append("ai_only_without_quant")
        return sources

    def score_adjustment(
        self,
        sources: list[str],
        performance: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Convert source performance into bounded score and size adjustment."""

        if not sources or not isinstance(performance, dict):
            return {
                "active": False,
                "sources": sources or [],
                "score_multiplier": 1.0,
                "size_multiplier": 1.0,
                "score_adjustment": 0.0,
                "reason": "暂无模型贡献统计，使用基础机会评分。",
            }

        weighted_score_multiplier = 0.0
        weighted_size_multiplier = 0.0
        total_weight = 0.0
        evidence: list[dict[str, Any]] = []
        for source in sources:
            bucket = performance.get(source)
            if not isinstance(bucket, dict):
                continue
            count = int(bucket.get("count") or 0)
            if count <= 0:
                continue
            sample_weight = min(max(count, 1), 25)
            score_multiplier = _safe_float(bucket.get("score_multiplier"), 1.0)
            size_multiplier = _safe_float(bucket.get("size_multiplier"), 1.0)
            weighted_score_multiplier += score_multiplier * sample_weight
            weighted_size_multiplier += size_multiplier * sample_weight
            total_weight += sample_weight
            evidence.append(
                {
                    "source": source,
                    "label": bucket.get("label") or source,
                    "count": count,
                    "pnl": bucket.get("pnl", 0.0),
                    "profit_factor": bucket.get("profit_factor", 0.0),
                    "state": bucket.get("state", "learning"),
                    "score_multiplier": round(score_multiplier, 6),
                    "size_multiplier": round(size_multiplier, 6),
                    "reason": bucket.get("reason", ""),
                }
            )

        if total_weight <= 0:
            return {
                "active": False,
                "sources": sources,
                "score_multiplier": 1.0,
                "size_multiplier": 1.0,
                "score_adjustment": 0.0,
                "evidence": evidence,
                "reason": "贡献样本不足，先学习不强干预。",
            }

        score_multiplier = min(max(weighted_score_multiplier / total_weight, 0.60), 1.38)
        size_multiplier = min(max(weighted_size_multiplier / total_weight, 0.65), 1.25)
        score_adjustment = (score_multiplier - 1.0) * 2.25
        state = (
            "promote"
            if score_multiplier > 1.04
            else "degrade" if score_multiplier < 0.96 else "neutral"
        )
        negative_sources = [
            item
            for item in evidence
            if _safe_float(item.get("pnl"), 0.0) < -8.0
            and _safe_float(item.get("profit_factor"), 1.0) < 0.75
            and int(item.get("count") or 0) >= 5
        ]
        hard_caution = bool(negative_sources)
        if state == "promote":
            reason = "这些证据来源最近真实平仓贡献为正，本轮提高机会评分和仓位倾向。"
        elif state == "degrade":
            reason = "这些证据来源最近真实平仓贡献偏弱，本轮降低机会评分并缩小仓位。"
        else:
            reason = "这些证据来源最近真实贡献接近中性，本轮保持基础评分。"
        if hard_caution:
            reason = (
                f"{reason} 其中 {len(negative_sources)} 个证据来源最近真实净亏且盈利因子偏低，"
                "本轮进入闭环强审查。"
            )

        return {
            "active": True,
            "sources": sources,
            "state": state,
            "hard_caution": hard_caution,
            "negative_sources": negative_sources,
            "score_multiplier": round(score_multiplier, 6),
            "size_multiplier": round(size_multiplier, 6),
            "score_adjustment": round(score_adjustment, 6),
            "evidence": evidence,
            "reason": reason,
        }

    def _match_entry_decision(
        self,
        position: Any,
        orders: list[Any],
        decisions: dict[int, Any],
    ) -> Any | None:
        pos_created = _aware(getattr(position, "created_at", None))
        pos_side = self._position_side(position)
        matched_decision = None
        best_delta = None
        for order in orders:
            if order.symbol != position.symbol or order.decision_id not in decisions:
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
        if count <= 0:
            return
        win_rate = int(bucket.get("wins") or 0) / count
        profit_factor = profit / loss if loss > 0 else (3.0 if profit > 0 else 0.0)
        avg_pnl = pnl / count
        edge = max(min(avg_pnl / 5.0, 0.28), -0.34)
        factor_edge = max(min((profit_factor - 1.0) * 0.14, 0.22), -0.26)
        win_edge = max(min((win_rate - 0.5) * 0.10, 0.05), -0.05)
        multiplier = min(max(1.0 + edge + factor_edge + win_edge, 0.60), 1.38)
        state = "learning"
        reason = "样本不足，先学习不强干预。"
        if count >= 5:
            if pnl > 0 and profit_factor >= 1.15:
                state = "promote"
                reason = (
                    f"最近 {count} 笔贡献净盈利 {pnl:.2f}U，盈利因子 {profit_factor:.2f}，"
                    "下轮提高权重。"
                )
            elif pnl < 0 or profit_factor < 0.85:
                state = "degrade"
                reason = (
                    f"最近 {count} 笔贡献净亏损 {pnl:.2f}U，盈利因子 {profit_factor:.2f}，"
                    "下轮降低权重。"
                )
            else:
                state = "neutral"
                reason = f"最近 {count} 笔贡献接近中性，保持基础权重。"
        bucket.update(
            {
                "pnl": round(pnl, 6),
                "profit": round(profit, 6),
                "loss": round(loss, 6),
                "avg_pnl": round(avg_pnl, 6),
                "win_rate": round(win_rate, 6),
                "profit_factor": round(profit_factor, 6),
                "score_multiplier": round(multiplier, 6),
                "size_multiplier": round(min(max(multiplier, 0.65), 1.25), 6),
                "state": state,
                "reason": reason,
            }
        )
