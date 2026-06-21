"""Risk management engine for final pre-execution validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from risk_manager.black_swan import BlackSwanDetector, BlackSwanResult
from risk_manager.circuit_breaker import CircuitBreaker
from risk_manager.position_limits import PositionLimitChecker
from risk_manager.stop_loss import StopLossManager, StopLossResult
from services.runtime_entry_filters import entry_filters_from_decision

logger = structlog.get_logger(__name__)

MIN_ENTRY_CONFIDENCE_AFTER_FEES = 0.62
MIN_TAKE_PROFIT_AFTER_COSTS = 0.015
MIN_REWARD_RISK_RATIO = 1.8


@dataclass
class RiskAssessment:
    """Result of the full risk evaluation pipeline."""

    approved: bool
    decision: DecisionOutput | None
    stop_loss_result: StopLossResult | None = None
    black_swan_result: BlackSwanResult | None = None
    rejection_reason: str = ""
    warnings: list[str] = field(default_factory=list)


class RiskEngine:
    """Validate decisions against hard safety controls and advisory risk context."""

    def __init__(
        self,
        max_open_positions_provider=None,
        *,
        position_checker: PositionLimitChecker | None = None,
        stop_loss_manager: StopLossManager | None = None,
        black_swan_detector: BlackSwanDetector | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self.position_checker = position_checker or PositionLimitChecker()
        self.stop_loss_manager = stop_loss_manager or StopLossManager()
        self.black_swan_detector = black_swan_detector or BlackSwanDetector()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.max_open_positions_provider = max_open_positions_provider or (
            lambda: settings.max_open_positions_per_model
        )

    def assess(
        self,
        decision: DecisionOutput,
        current_positions: list[dict],
        account_balance: float,
        headlines: list[str] | None = None,
        sentiment_scores: list[float] | None = None,
        price_change_1m: float = 0.0,
        volume_ratio: float = 1.0,
        adx_14: float | None = None,
    ) -> RiskAssessment:
        """Evaluate a trading decision before it reaches the executor."""

        warnings: list[str] = []

        if self.circuit_breaker.is_open and decision.is_entry:
            return RiskAssessment(
                approved=False,
                decision=decision,
                rejection_reason="Circuit breaker is open; no new entries are allowed.",
            )

        black_swan_result = self._assess_black_swan(
            decision=decision,
            current_positions=current_positions,
            headlines=headlines,
            sentiment_scores=sentiment_scores,
            price_change_1m=price_change_1m,
            volume_ratio=volume_ratio,
            warnings=warnings,
        )
        if black_swan_result is not None:
            return black_swan_result

        stop_result = self._assess_stop_loss(decision, current_positions)
        if stop_result is not None:
            return stop_result

        if decision.is_entry:
            entry_result = self._assess_entry(
                decision=decision,
                current_positions=current_positions,
                account_balance=account_balance,
                volume_ratio=volume_ratio,
                adx_14=adx_14,
                warnings=warnings,
            )
            if entry_result is not None:
                return entry_result

        self.circuit_breaker.evaluate_daily_loss(account_balance)
        if self.circuit_breaker.is_open and decision.is_entry:
            return RiskAssessment(
                approved=False,
                decision=decision,
                rejection_reason=(
                    f"Daily loss limit reached: {self.circuit_breaker._state.tripped_reason}"
                ),
            )

        return RiskAssessment(
            approved=True,
            decision=decision,
            stop_loss_result=None,
            warnings=warnings,
        )

    def _assess_black_swan(
        self,
        *,
        decision: DecisionOutput,
        current_positions: list[dict],
        headlines: list[str] | None,
        sentiment_scores: list[float] | None,
        price_change_1m: float,
        volume_ratio: float,
        warnings: list[str],
    ) -> RiskAssessment | None:
        if not headlines:
            return None

        result = self.black_swan_detector.check_combined(
            headlines or [],
            sentiment_scores or [],
            price_change_1m,
            volume_ratio,
        )
        if result.triggered and result.severity == "critical":
            matching_positions = [
                pos
                for pos in current_positions
                if pos.get("symbol") == decision.symbol and pos.get("is_open", True)
            ]
            if not matching_positions:
                if decision.is_entry:
                    return RiskAssessment(
                        approved=False,
                        decision=decision,
                        black_swan_result=result,
                        rejection_reason=(
                            "风控硬拦截：检测到重大行情风险，当前没有可平仓位，"
                            f"因此禁止新开仓。原因：{result.reason}"
                        ),
                        warnings=[result.reason],
                    )
                return RiskAssessment(
                    approved=True,
                    decision=decision,
                    black_swan_result=result,
                    warnings=[result.reason],
                )

            target_side = str(matching_positions[0].get("side") or "long").lower()
            close_action = Action.CLOSE_LONG if target_side == "long" else Action.CLOSE_SHORT
            return RiskAssessment(
                approved=True,
                decision=DecisionOutput(
                    model_name="risk_engine",
                    symbol=decision.symbol,
                    action=close_action,
                    confidence=1.0,
                    reasoning=f"BLACK SWAN CRITICAL: {result.reason}",
                    position_size_pct=1.0,
                ),
                black_swan_result=result,
                warnings=[result.reason],
            )

        if result.severity == "warn":
            warnings.append(
                "黑天鹅预警：检测到潜在异常新闻或快速波动；当前仅记录风险提示，"
                f"不作为固定开仓门槛。原因：{result.reason}"
            )
        return None

    def _assess_stop_loss(
        self,
        decision: DecisionOutput,
        current_positions: list[dict],
    ) -> RiskAssessment | None:
        for position in current_positions:
            if position.get("symbol") != decision.symbol or not position.get("is_open"):
                continue
            current_price = 0.0
            if decision.feature_snapshot:
                current_price = float(decision.feature_snapshot.get("close", 0) or 0)
            result = self.stop_loss_manager.evaluate(
                symbol=position["symbol"],
                side=position["side"],
                entry_price=position["entry_price"],
                current_price=current_price,
            )
            if not result.triggered:
                continue
            close_action = Action.CLOSE_LONG if position["side"] == "long" else Action.CLOSE_SHORT
            return RiskAssessment(
                approved=True,
                decision=DecisionOutput(
                    model_name="risk_engine",
                    symbol=decision.symbol,
                    action=close_action,
                    confidence=1.0,
                    reasoning=f"STOP LOSS ({result.stop_type.value}): {result.reason}",
                    position_size_pct=1.0,
                ),
                stop_loss_result=result,
                warnings=[result.reason],
            )
        return None

    def _assess_entry(
        self,
        *,
        decision: DecisionOutput,
        current_positions: list[dict],
        account_balance: float,
        volume_ratio: float,
        adx_14: float | None,
        warnings: list[str],
    ) -> RiskAssessment | None:
        model_open_positions = [
            position
            for position in current_positions
            if position.get("model_name") == decision.model_name
        ]
        decision_side = "long" if decision.action == Action.LONG else "short"
        same_symbol_positions = [
            position
            for position in model_open_positions
            if position.get("side") == decision_side and position.get("symbol") == decision.symbol
        ]
        is_same_symbol_add = bool(same_symbol_positions)

        capacity = self._max_open_positions_context()
        max_open_positions = int(
            capacity.get("entry_limit") or capacity.get("effective_limit") or 0
        )
        model_open_group_count = self._model_open_group_count(model_open_positions)
        if (
            not is_same_symbol_add
            and max_open_positions > 0
            and model_open_group_count >= max_open_positions
        ):
            return RiskAssessment(
                approved=False,
                decision=decision,
                rejection_reason=(
                    "当前持仓组数已达到动态容量上限，暂停新开不同币种/方向仓位。"
                    f"当前 {model_open_group_count} 组，限制 {max_open_positions} 组。"
                    f"{self._capacity_suffix(capacity)}"
                ),
            )

        entry_filters = entry_filters_from_decision(decision)
        min_confidence = max(
            float(settings.confidence_threshold or 0.0),
            MIN_ENTRY_CONFIDENCE_AFTER_FEES,
        )
        if float(decision.confidence or 0.0) < min_confidence:
            warnings.append(
                "入场信心低于手续费后参考线；此项只影响排序、仓位和解释，不是硬开仓门槛。"
            )

        min_take_profit = max(
            MIN_TAKE_PROFIT_AFTER_COSTS,
            float(decision.stop_loss_pct or 0.0) * MIN_REWARD_RISK_RATIO,
        )
        if float(decision.take_profit_pct or 0.0) < min_take_profit:
            warnings.append(
                "止盈空间低于成本/止损参考线；此项只影响排序、仓位和解释，不是硬开仓门槛。"
            )

        trend_adx = self._get_adx(decision, adx_14)
        entry_confirmations = [
            trend_adx >= entry_filters.min_entry_adx,
            volume_ratio >= entry_filters.min_entry_volume_ratio,
            self._trend_aligned(decision),
        ]
        if sum(1 for passed in entry_confirmations if passed) < 2:
            warnings.append(
                "运行时入场参考项不足 2 项；动态策略会降低优先级或仓位，但不在风控层硬拦截。"
            )

        size_check = self.position_checker.check_contract_entry_limits(
            proposed_margin_pct=decision.position_size_pct,
            proposed_leverage=decision.suggested_leverage,
            proposed_stop_loss_pct=decision.stop_loss_pct,
            current_positions=current_positions,
            account_balance=account_balance,
            symbol=decision.symbol,
        )
        if not size_check.passed:
            return RiskAssessment(
                approved=False,
                decision=decision,
                rejection_reason=size_check.reason,
            )
        if size_check.adjusted_size_pct is not None:
            decision.position_size_pct = size_check.adjusted_size_pct
            warnings.append(size_check.reason)

        leverage_cap = self._max_allowed_leverage(decision, volume_ratio, trend_adx)
        if decision.suggested_leverage > leverage_cap:
            decision.suggested_leverage = leverage_cap
            warnings.append(f"杠杆已按动态运行时参考质量限制为 {leverage_cap:.1f}x。")

        leverage_check = self.position_checker.check_leverage(decision.suggested_leverage)
        if leverage_check.adjusted_size_pct:
            decision.suggested_leverage = leverage_check.adjusted_size_pct
        return None

    def _max_open_positions_context(self) -> dict[str, Any]:
        raw = self.max_open_positions_provider()
        if isinstance(raw, dict):
            return dict(raw)
        as_dict = getattr(raw, "as_dict", None)
        if callable(as_dict):
            value = as_dict()
            if isinstance(value, dict):
                return value
        effective = int(raw or 0)
        return {
            "entry_limit": effective,
            "effective_limit": effective,
            "base_limit": effective,
            "reason": "",
        }

    @staticmethod
    def _model_open_group_count(positions: list[dict]) -> int:
        groups: set[tuple[str, str]] = set()
        for position in positions or []:
            if position.get("is_open", True) is False:
                continue
            symbol = str(position.get("symbol") or "").strip().upper()
            side = str(position.get("side") or "unknown").strip().lower() or "unknown"
            groups.add((symbol, side))
        return len(groups)

    @staticmethod
    def _capacity_suffix(capacity: dict[str, Any]) -> str:
        base_limit = capacity.get("base_limit")
        effective_limit = capacity.get("effective_limit")
        entry_limit = capacity.get("entry_limit")
        reason = str(capacity.get("reason") or "").strip()
        parts: list[str] = []
        if base_limit and effective_limit and int(base_limit) != int(effective_limit):
            parts.append(f"基础上限 {base_limit}，运行上限 {effective_limit}")
        if entry_limit and int(entry_limit) != int(effective_limit or 0):
            parts.append(f"开仓上限 {entry_limit}")
        readable_reason = RiskEngine._capacity_reason_text(capacity, reason)
        if readable_reason:
            parts.append(readable_reason[:160])
        return " " + "；".join(parts) if parts else ""

    @staticmethod
    def _capacity_reason_text(capacity: dict[str, Any], reason: str) -> str:
        factors = capacity.get("factors") if isinstance(capacity.get("factors"), dict) else {}
        codes = factors.get("reason_codes") if isinstance(factors, dict) else None
        if isinstance(codes, list) and codes:
            labels = {
                "strategy_rotation_slots": "策略学习已为轮换释放预留开仓槽",
                "release_rotation_slots": "低质量持仓释放中，系统预留轮换槽",
                "rotation_entry_expansion": "开仓上限已按轮换释放策略上调",
                "low_quality_pressure": "低质量持仓压力较高，优先复盘释放旧仓",
                "low_quality_warn": "低质量持仓偏高，降低扩仓节奏",
                "drawdown": "当日回撤达到收缩区间",
                "drawdown_watch": "当日回撤进入观察区间",
            }
            return "；".join(labels.get(str(code), str(code)) for code in codes[:4])
        if "=" in reason:
            return "容量由策略学习、持仓质量和账户风险动态计算。"
        return reason

    @staticmethod
    def _get_adx(decision: DecisionOutput, adx_14: float | None) -> float:
        if adx_14 is not None:
            return float(adx_14)
        snapshot = decision.feature_snapshot or {}
        try:
            return float(snapshot.get("adx_14", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _max_allowed_leverage(
        self,
        decision: DecisionOutput,
        volume_ratio: float,
        trend_adx: float,
    ) -> float:
        base_cap = min(5.0, settings.max_leverage)
        if decision.confidence < 0.58:
            return base_cap

        entry_filters = entry_filters_from_decision(decision)
        filters_pass = (
            sum(
                1
                for passed in (
                    volume_ratio >= entry_filters.min_entry_volume_ratio,
                    trend_adx >= entry_filters.min_entry_adx,
                    self._trend_aligned(decision),
                )
                if passed
            )
            >= 2
        )
        if not filters_pass:
            return base_cap
        if decision.confidence < 0.72:
            return min(10.0, settings.max_leverage)
        return min(20.0, settings.max_leverage)

    def _trend_aligned(self, decision: DecisionOutput) -> bool:
        snapshot = decision.feature_snapshot or {}
        try:
            price_vs_sma20 = float(snapshot.get("price_vs_sma20", 0.0) or 0.0)
            price_vs_sma50 = float(snapshot.get("price_vs_sma50", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False

        if decision.action == Action.LONG:
            return price_vs_sma20 > 0 and price_vs_sma50 > 0
        if decision.action == Action.SHORT:
            return price_vs_sma20 < 0 and price_vs_sma50 < 0
        return False
