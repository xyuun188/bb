"""
Risk management engine — the central pipeline that validates all trading decisions
before they reach the executor.

Flow:
  1. Position limits check
  2. Stop-loss evaluation (for existing positions)
  3. Black swan detection
  4. Circuit breaker check
  5. Approval/rejection of decision
"""

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

logger = structlog.get_logger(__name__)

MIN_ENTRY_CONFIDENCE_AFTER_FEES = 0.62
MIN_TAKE_PROFIT_AFTER_COSTS = 0.015
MIN_REWARD_RISK_RATIO = 1.8


@dataclass
class RiskAssessment:
    """Result of the full risk evaluation pipeline."""

    approved: bool
    decision: DecisionOutput | None  # May be modified (e.g., reduced size)
    stop_loss_result: StopLossResult | None = None
    black_swan_result: BlackSwanResult | None = None
    rejection_reason: str = ""
    warnings: list[str] = field(default_factory=list)


class RiskEngine:
    """Orchestrates all risk checks and produces a final RiskAssessment."""

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
        """Evaluate a trading decision against all risk controls.

        Args:
            decision: The AI model's proposed trade.
            current_positions: Open positions (list of dicts with side, symbol, quantity, entry_price).
            account_balance: Current account balance in quote currency.
            headlines: Recent news headlines for black swan detection.
            sentiment_scores: Corresponding sentiment scores.
            price_change_1m: Recent 1-minute price change (for flash crash detection).
            volume_ratio: Current volume / average volume.
            adx_14: ADX trend-strength indicator for entry filtering.

        Returns:
            RiskAssessment with approval status and any modifications.
        """
        warnings: list[str] = []

        # === 0. Circuit breaker check ===
        if self.circuit_breaker.is_open and decision.is_entry:
            return RiskAssessment(
                approved=False,
                decision=decision,
                rejection_reason="Circuit breaker is OPEN — no new positions allowed.",
            )

        # === 1. Black swan check ===
        if headlines:
            bs_result = self.black_swan_detector.check_combined(
                headlines or [],
                sentiment_scores or [],
                price_change_1m,
                volume_ratio,
            )
            if bs_result.triggered and bs_result.severity == "critical":
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
                            black_swan_result=bs_result,
                            rejection_reason=(
                                "风控拦截：检测到重大行情风险，当前没有可平仓仓位，"
                                f"因此禁止追涨/追空开新仓。原因：{bs_result.reason}"
                            ),
                            warnings=[bs_result.reason],
                        )
                    return RiskAssessment(
                        approved=True,
                        decision=decision,
                        black_swan_result=bs_result,
                        warnings=[bs_result.reason],
                    )

                target_side = str(matching_positions[0].get("side") or "long").lower()
                close_action = Action.CLOSE_LONG if target_side == "long" else Action.CLOSE_SHORT
                # Force-close the matching open side only when a position exists.
                return RiskAssessment(
                    approved=True,  # Action must be taken
                    decision=DecisionOutput(
                        model_name="risk_engine",
                        symbol=decision.symbol,
                        action=close_action,
                        confidence=1.0,
                        reasoning=f"BLACK SWAN CRITICAL: {bs_result.reason}",
                        position_size_pct=1.0,
                    ),
                    black_swan_result=bs_result,
                    warnings=[bs_result.reason],
                )
            if bs_result.severity == "warn":
                warning_text = (
                    "黑天鹅预警：检测到潜在异常新闻或快速下跌，但未达到重大风险级别。"
                    "系统继续允许交易，同时把新开仓限制为小仓位、1x 杠杆。"
                    f" 原因：{bs_result.reason}"
                )
                warnings.append(warning_text)
                # Warn mode should alert and reduce risk, not freeze trading.
                # Critical mode above is the only black-swan branch that blocks
                # fresh entries or forces a close.
                if False and decision.is_entry:
                    decision.position_size_pct = min(float(decision.position_size_pct or 0.0), 0.03)
                    decision.suggested_leverage = min(
                        float(decision.suggested_leverage or 1.0), 1.0
                    )
                    decision.reasoning = (
                        f"{decision.reasoning} [风控预警：检测到 warning 级别黑天鹅线索，"
                        "已自动降为小仓位 1x 试单。]"
                    )

        # === 2. Stop-loss evaluation for existing positions ===
        stop_result = None
        for pos in current_positions:
            if pos.get("symbol") == decision.symbol and pos.get("is_open"):
                sl_result = self.stop_loss_manager.evaluate(
                    symbol=pos["symbol"],
                    side=pos["side"],
                    entry_price=pos["entry_price"],
                    current_price=(
                        decision.feature_snapshot.get("close", 0)
                        if decision.feature_snapshot
                        else 0
                    ),
                )
                if sl_result.triggered:
                    stop_result = sl_result
                    # Override the AI decision with a forced close
                    close_action = (
                        Action.CLOSE_LONG if pos["side"] == "long" else Action.CLOSE_SHORT
                    )
                    return RiskAssessment(
                        approved=True,
                        decision=DecisionOutput(
                            model_name="risk_engine",
                            symbol=decision.symbol,
                            action=close_action,
                            confidence=1.0,
                            reasoning=f"STOP LOSS ({sl_result.stop_type.value}): {sl_result.reason}",
                            position_size_pct=1.0,
                        ),
                        stop_loss_result=sl_result,
                        warnings=[sl_result.reason],
                    )

        # === 3. Position size checks (only for entries) ===
        if decision.is_entry:
            model_open_positions = [
                p for p in current_positions if p.get("model_name") == decision.model_name
            ]
            decision_side = "long" if decision.action == Action.LONG else "short"
            same_symbol_positions = [
                p
                for p in model_open_positions
                if p.get("side") == decision_side and p.get("symbol") == decision.symbol
            ]
            is_same_symbol_add = bool(same_symbol_positions)

            capacity = self._max_open_positions_context()
            max_open_positions = int(
                capacity.get("entry_limit")
                or capacity.get("effective_limit")
                or 0
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

            min_confidence = max(
                float(settings.confidence_threshold or 0.0), MIN_ENTRY_CONFIDENCE_AFTER_FEES
            )
            if False and float(decision.confidence or 0.0) < min_confidence:
                return RiskAssessment(
                    approved=False,
                    decision=decision,
                    rejection_reason=(
                        "入场信心度未达到手续费修正后的执行门槛，暂不下单。"
                        f"当前信心度={decision.confidence:.2f}，要求>={min_confidence:.2f}。"
                    ),
                )

            min_take_profit = max(
                MIN_TAKE_PROFIT_AFTER_COSTS,
                float(decision.stop_loss_pct or 0.0) * MIN_REWARD_RISK_RATIO,
            )
            if False and float(decision.take_profit_pct or 0.0) < min_take_profit:
                return RiskAssessment(
                    approved=False,
                    decision=decision,
                    rejection_reason=(
                        "止盈空间不足以覆盖手续费、滑点和止损风险，暂不下单。"
                        f"当前止盈={decision.take_profit_pct:.2%}，要求>={min_take_profit:.2%}。"
                    ),
                )

            if False and len(same_symbol_positions) >= settings.max_same_symbol_positions_per_side:
                return RiskAssessment(
                    approved=False,
                    decision=decision,
                    rejection_reason=(
                        f"同币种同方向持仓已达上限，暂停加仓。"
                        f"{decision.symbol} {decision_side} 当前 {len(same_symbol_positions)} 笔，"
                        f"限制 {settings.max_same_symbol_positions_per_side} 笔。"
                    ),
                )

            trend_adx = self._get_adx(decision, adx_14)
            entry_confirmations = [
                trend_adx >= settings.min_entry_adx,
                volume_ratio >= settings.min_entry_volume_ratio,
                self._trend_aligned(decision),
            ]
            if False and sum(1 for ok in entry_confirmations if ok) < 2:
                return RiskAssessment(
                    approved=False,
                    decision=decision,
                    rejection_reason=(
                        "入场确认不足，暂不下单。"
                        f"当前 ADX={trend_adx:.1f}（要求 {settings.min_entry_adx:.1f}），"
                        f"成交量倍数={volume_ratio:.2f}（要求 {settings.min_entry_volume_ratio:.2f}），"
                        "且均线趋势需与方向配合；三项至少满足两项。"
                    ),
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
            if False and decision.suggested_leverage > leverage_cap:
                decision.suggested_leverage = leverage_cap
                warnings.append(f"杠杆已按置信度和过滤条件限制为 {leverage_cap:.1f}x。")

            # Leverage check
            lev_check = self.position_checker.check_leverage(decision.suggested_leverage)
            if lev_check.adjusted_size_pct:
                decision.suggested_leverage = lev_check.adjusted_size_pct

        # === 4. Daily loss check ===
        self.circuit_breaker.evaluate_daily_loss(account_balance)
        if self.circuit_breaker.is_open:
            if decision.is_entry:
                return RiskAssessment(
                    approved=False,
                    decision=decision,
                    rejection_reason=f"Daily loss limit reached: {self.circuit_breaker._state.tripped_reason}",
                )

        # === 5. Approve ===
        return RiskAssessment(
            approved=True,
            decision=decision,
            stop_loss_result=stop_result,
            warnings=warnings,
        )

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
                "release_rotation_slots": "低质量持仓释放中，系统预留了小仓轮换槽",
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

        filters_pass = (
            sum(
                1
                for ok in (
                    volume_ratio >= settings.min_entry_volume_ratio,
                    trend_adx >= settings.min_entry_adx,
                    self._trend_aligned(decision),
                )
                if ok
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
