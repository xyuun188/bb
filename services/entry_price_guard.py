"""Pre-execution entry price drift guard."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from services.entry_direction_metrics import selected_entry_metrics
from services.trading_params import DEFAULT_TRADING_PARAMS

logger = structlog.get_logger(__name__)

_PRICE_GUARD_PARAMS = DEFAULT_TRADING_PARAMS.entry_price_guard
ENTRY_PRICE_RECHECK_RESCUE_MAX_MOVE_PCT = _PRICE_GUARD_PARAMS.recheck_rescue_max_move_pct
ENTRY_PRICE_RECHECK_EXCEPTIONAL_MAX_MOVE_PCT = _PRICE_GUARD_PARAMS.recheck_exceptional_max_move_pct
ENTRY_PRICE_RECHECK_EXPECTED_BUFFER_MULTIPLE = _PRICE_GUARD_PARAMS.recheck_expected_buffer_multiple
PRICE_GUARD_ENTRY_BLOCK_MINUTES = _PRICE_GUARD_PARAMS.entry_block_minutes


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _feature_snapshot(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is not None and hasattr(value, "to_dict"):
        snapshot = value.to_dict()
        return snapshot if isinstance(snapshot, dict) else {}
    return {}


def _noop_temporary_block(symbol: str, reason: str, minutes: float) -> None:
    return None


@dataclass(slots=True)
class EntryPriceGuardPolicy:
    """Block stale or badly drifted entry decisions immediately before order submit."""

    latest_price_provider: Callable[[str], Awaitable[float]]
    fresh_feature_provider: Callable[[str], Awaitable[Any]]
    market_data_quality_reason_provider: Callable[..., str | None]
    decision_age_seconds_provider: Callable[[DecisionOutput], float]
    temporary_entry_block_recorder: Callable[[str, str, float], None] = _noop_temporary_block
    temporary_block_minutes: float = PRICE_GUARD_ENTRY_BLOCK_MINUTES
    config: Any = field(default_factory=lambda: settings)
    params: Any = _PRICE_GUARD_PARAMS

    async def guard_reason(self, decision: DecisionOutput) -> str | None:
        """Return a blocker reason when the latest market state invalidates an entry."""

        if not decision.is_entry:
            return None

        snapshot = _safe_dict(decision.feature_snapshot)
        raw = _safe_dict(decision.raw_response)
        snapshot_quality_reason = self.market_data_quality_reason_provider(
            snapshot,
            stage_label="下单前分析快照",
        )
        if snapshot_quality_reason:
            fresh_snapshot = await self._fresh_snapshot(decision.symbol)
            fresh_quality_reason = (
                self.market_data_quality_reason_provider(
                    fresh_snapshot,
                    stage_label="下单前刷新行情",
                )
                if fresh_snapshot
                else "下单前刷新行情失败，无法确认盘口和短周期特征。"
            )
            raw["pre_execution_data_quality_recheck"] = {
                "original_reason": snapshot_quality_reason,
                "fresh_recheck_available": bool(fresh_snapshot),
                "fresh_reason": fresh_quality_reason,
                "original_snapshot_timestamp": snapshot.get("timestamp"),
                "fresh_snapshot_timestamp": fresh_snapshot.get("timestamp"),
            }
            decision.raw_response = raw
            if fresh_snapshot and not fresh_quality_reason:
                snapshot = fresh_snapshot
                decision.feature_snapshot = fresh_snapshot
            else:
                return (
                    f"下单前行情质量复核未通过：{fresh_quality_reason or snapshot_quality_reason}"
                    "系统已即时刷新该币种行情，但数据仍不足以安全下单，本次不执行。"
                )

        snapshot_current_price = _safe_float(snapshot.get("current_price"), 0.0)
        snapshot_close_price = _safe_float(snapshot.get("close"), 0.0)
        snapshot_price = snapshot_current_price or snapshot_close_price
        if snapshot_price <= 0:
            return None

        latest_price = await self.latest_price_provider(decision.symbol)
        if latest_price <= 0:
            return "下单前没有重新拿到最新价格，系统不使用过期行情盲目下单，本次跳过。"

        snapshot_price_source = "current_price" if snapshot_current_price > 0 else "close"
        if snapshot_current_price > 0 and snapshot_close_price > 0:
            current_gap = abs(latest_price - snapshot_current_price) / max(
                snapshot_current_price, 1e-12
            )
            close_gap = abs(latest_price - snapshot_close_price) / max(snapshot_close_price, 1e-12)
            if (
                abs(snapshot_current_price - snapshot_close_price)
                / max(snapshot_close_price, 1e-12)
                > 0.03
            ):
                if close_gap <= current_gap:
                    snapshot_price = snapshot_close_price
                    snapshot_price_source = "close_reconciled"
                else:
                    snapshot_price = snapshot_current_price
                    snapshot_price_source = "current_price_reconciled"

        move = (latest_price - snapshot_price) / snapshot_price
        allowed = min(
            max(
                float(self.config.max_slippage_pct or self.params.min_allowed_slippage_pct),
                self.params.min_allowed_slippage_pct,
            ),
            self.params.max_allowed_slippage_pct,
        )
        opportunity = _safe_dict(raw.get("opportunity_score"))
        quant_probe = _safe_dict(raw.get("quant_profit_probe"))
        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = _safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        selected_metrics = selected_entry_metrics(decision)
        if selected_metrics.has_selected_side:
            expected_net = selected_metrics.expected_net_return_pct
            profit_quality = selected_metrics.profit_quality_ratio
        if quant_probe.get("triggered") and expected_net > 0:
            if (
                quant_probe.get("strong_probe")
                and expected_net >= self.params.strong_probe_expected_net
                and profit_quality >= self.params.strong_probe_profit_quality
            ):
                allowed = max(allowed, self.params.strong_probe_allowed_move_pct)
            elif (
                expected_net >= self.params.normal_probe_expected_net
                and profit_quality >= self.params.normal_probe_profit_quality
            ):
                allowed = max(allowed, self.params.normal_probe_allowed_move_pct)
        if (
            expected_net >= self.params.exceptional_expected_net
            and profit_quality >= self.params.exceptional_profit_quality
        ):
            allowed = max(allowed, self.params.exceptional_allowed_move_pct)
        elif (
            expected_net >= self.params.strong_expected_net
            and profit_quality >= self.params.strong_profit_quality
        ):
            allowed = max(allowed, self.params.strong_allowed_move_pct)
        elif (
            expected_net >= self.params.normal_expected_net
            and profit_quality >= self.params.normal_profit_quality
        ):
            allowed = max(allowed, self.params.normal_allowed_move_pct)
        raw["pre_execution_price_check"] = {
            "snapshot_price": snapshot_price,
            "snapshot_price_source": snapshot_price_source,
            "snapshot_current_price": snapshot_current_price,
            "snapshot_close_price": snapshot_close_price,
            "snapshot_timestamp": snapshot.get("timestamp"),
            "snapshot_age_seconds": round(self.decision_age_seconds_provider(decision), 3),
            "latest_price": latest_price,
            "move_pct": round(move * 100, 4),
            "allowed_pct": round(allowed * 100, 4),
            "expected_net_return_pct": round(expected_net, 6),
            "profit_quality_ratio": round(profit_quality, 6),
            "selected_side": selected_metrics.side,
            "selected_metrics_source": selected_metrics.source,
        }
        decision.raw_response = raw

        if self._adverse_directional_move(decision.action, move, allowed):
            rescue_allowed = await self._try_rescue_with_fresh_recheck(
                decision,
                raw,
                latest_price=latest_price,
                move=move,
                allowed=allowed,
                expected_net=expected_net,
                profit_quality=profit_quality,
            )
            if rescue_allowed:
                return None

        return self._price_drift_reason(
            decision,
            move=move,
            allowed=allowed,
        )

    async def _fresh_snapshot(self, symbol: str) -> dict[str, Any]:
        return _feature_snapshot(await self.fresh_feature_provider(symbol))

    @staticmethod
    def _adverse_directional_move(action: Action, move: float, allowed: float) -> bool:
        return (
            (action == Action.LONG and move > allowed)
            or (action == Action.SHORT and move < -allowed)
            or abs(move) > allowed * 2
        )

    async def _try_rescue_with_fresh_recheck(
        self,
        decision: DecisionOutput,
        raw: dict[str, Any],
        *,
        latest_price: float,
        move: float,
        allowed: float,
        expected_net: float,
        profit_quality: float,
    ) -> bool:
        fresh_snapshot = await self._fresh_snapshot(decision.symbol)
        fresh_quality_reason = (
            self.market_data_quality_reason_provider(
                fresh_snapshot,
                stage_label="偏移后刷新行情",
            )
            if fresh_snapshot
            else "偏移后刷新行情失败，无法确认最新盘口和短周期特征。"
        )
        fresh_price = _safe_float(
            fresh_snapshot.get("current_price") or fresh_snapshot.get("close"),
            0.0,
        )
        fresh_latest_gap = (
            abs(latest_price - fresh_price) / max(fresh_price, 1e-12) if fresh_price > 0 else 0.0
        )
        move_abs = abs(move)
        rescue_cap = (
            self.params.recheck_exceptional_max_move_pct
            if expected_net >= self.params.exceptional_expected_net
            and profit_quality >= self.params.exceptional_profit_quality
            else self.params.recheck_rescue_max_move_pct
        )
        expected_buffer = (
            self.params.recheck_expected_buffer_multiple
            if expected_net < self.params.normal_expected_net
            else (
                self.params.medium_expected_buffer_multiple
                if expected_net < self.params.strong_expected_net
                else self.params.strong_expected_buffer_multiple
            )
        )
        expected_covers_chase = expected_net >= move_abs * 100 * expected_buffer
        fresh_returns_1 = _safe_float(fresh_snapshot.get("returns_1"), 0.0)
        fresh_returns_5 = _safe_float(fresh_snapshot.get("returns_5"), 0.0)
        fresh_momentum_ok = (
            decision.action == Action.LONG
            and fresh_returns_1 >= self.params.fresh_long_returns_1_floor
            and fresh_returns_5 >= self.params.fresh_long_returns_5_floor
        ) or (
            decision.action == Action.SHORT
            and fresh_returns_1 <= self.params.fresh_short_returns_1_ceiling
            and fresh_returns_5 <= self.params.fresh_short_returns_5_ceiling
        )
        rescue_allowed = bool(
            fresh_snapshot
            and not fresh_quality_reason
            and fresh_price > 0
            and fresh_latest_gap <= max(allowed, self.params.fresh_latest_gap_floor_pct)
            and move_abs <= rescue_cap
            and expected_covers_chase
            and fresh_momentum_ok
        )
        raw["pre_execution_price_recheck"] = {
            "triggered": True,
            "fresh_recheck_available": bool(fresh_snapshot),
            "fresh_reason": fresh_quality_reason,
            "fresh_price": fresh_price,
            "fresh_latest_gap_pct": round(fresh_latest_gap * 100, 4),
            "original_move_pct": round(move * 100, 4),
            "rescue_cap_pct": round(rescue_cap * 100, 4),
            "expected_buffer_multiple": round(expected_buffer, 4),
            "expected_covers_chase": bool(expected_covers_chase),
            "fresh_momentum_ok": bool(fresh_momentum_ok),
            "rescued": rescue_allowed,
        }
        decision.raw_response = raw
        if rescue_allowed:
            decision.feature_snapshot = fresh_snapshot
            logger.info(
                "pre-order price guard rescued by fresh recheck",
                symbol=decision.symbol,
                move_pct=round(move * 100, 4),
                fresh_price=fresh_price,
                expected_net=round(expected_net, 4),
            )
        return rescue_allowed

    def _price_drift_reason(
        self,
        decision: DecisionOutput,
        *,
        move: float,
        allowed: float,
    ) -> str | None:
        if decision.action == Action.LONG and move > allowed:
            reason = (
                f"下单前价格已比分时分析上涨 {move * 100:.2f}%，"
                f"超过允许偏移 {allowed * 100:.2f}%。系统已即时刷新该币种行情复核，"
                "但偏移仍过大或盘口/动量未通过复核；为避免追高，本次不执行。"
            )
            self.temporary_entry_block_recorder(
                decision.symbol,
                reason,
                self.temporary_block_minutes,
            )
            return reason
        if decision.action == Action.SHORT and move < -allowed:
            reason = (
                f"下单前价格已比分时分析下跌 {abs(move) * 100:.2f}%，"
                f"超过允许偏移 {allowed * 100:.2f}%。系统已即时刷新该币种行情复核，"
                "但偏移仍过大或盘口/动量未通过复核；为避免追空，本次不执行。"
            )
            self.temporary_entry_block_recorder(
                decision.symbol,
                reason,
                self.temporary_block_minutes,
            )
            return reason
        if abs(move) > allowed * 2:
            reason = (
                f"下单前价格较分析时波动 {abs(move) * 100:.2f}%，行情变化太快，"
                "系统已即时刷新该币种行情复核，但仍不适合沿用旧信号，本次不执行。"
            )
            self.temporary_entry_block_recorder(
                decision.symbol,
                reason,
                self.temporary_block_minutes,
            )
            return reason
        return None
