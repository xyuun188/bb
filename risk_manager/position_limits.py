"""
Position and leverage limit enforcement.
Ensures no single position exceeds configured thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)


@dataclass
class LimitCheckResult:
    passed: bool
    reason: str = ""
    adjusted_size_pct: float | None = None


@dataclass
class ContractExposureSnapshot:
    account_equity_usdt: float
    current_margin_usdt: float
    proposed_margin_usdt: float
    current_margin_pct: float
    proposed_margin_pct: float
    margin_after_pct: float
    margin_limit_pct: float
    current_stop_risk_pct: float
    proposed_stop_risk_pct: float
    stop_risk_after_pct: float
    stop_risk_limit_pct: float
    current_notional_pct: float
    proposed_notional_pct: float
    notional_after_pct: float
    notional_limit_pct: float

    @property
    def has_entry_capacity(self) -> bool:
        return (
            self.margin_after_pct <= self.margin_limit_pct
            and self.notional_after_pct <= self.notional_limit_pct
        )


class PositionLimitChecker:
    """Validates that a proposed trade respects position and leverage limits."""

    def __init__(self) -> None:
        self.max_position_pct = settings.max_position_pct
        self.max_leverage = settings.max_leverage

    def check_position_size(
        self, proposed_size_pct: float, current_exposure_pct: float, symbol: str
    ) -> LimitCheckResult:
        """Check if adding this position would exceed position limits.

        Args:
            proposed_size_pct: Fraction of total account to allocate (0.0-1.0).
            current_exposure_pct: Current total exposure across all positions.
            symbol: Trading symbol for logging.
        """
        new_total = current_exposure_pct + proposed_size_pct

        if proposed_size_pct > self.max_position_pct:
            # Cap at max
            adjusted = self.max_position_pct
            return LimitCheckResult(
                passed=True,
                reason=(
                    f"Position size {proposed_size_pct:.1%} exceeds max {self.max_position_pct:.1%}. "
                    f"Capped to {adjusted:.1%}."
                ),
                adjusted_size_pct=adjusted,
            )

        if new_total > self.max_position_pct * 3:  # allow up to 3x for multi-symbol
            adjusted = max(0, self.max_position_pct * 3 - current_exposure_pct)
            if adjusted <= 0:
                return LimitCheckResult(
                    passed=False,
                    reason=(
                        f"Total exposure would be {new_total:.1%} — "
                        f"already at limit ({current_exposure_pct:.1%})."
                    ),
                )
            return LimitCheckResult(
                passed=True,
                reason=f"Size adjusted from {proposed_size_pct:.1%} to {adjusted:.1%} due to exposure limits.",
                adjusted_size_pct=adjusted,
            )

        return LimitCheckResult(passed=True, reason="Position size within limits.")

    def check_contract_entry_limits(
        self,
        proposed_margin_pct: float,
        proposed_leverage: float,
        proposed_stop_loss_pct: float,
        current_positions: list[dict],
        account_balance: float,
        symbol: str,
    ) -> LimitCheckResult:
        """Check futures capacity using margin, stop-risk, and notional exposure.

        `position_size_pct` is treated as margin allocation because executors
        size entries as: balance * position_size_pct * leverage.
        """
        snapshot = self.contract_exposure_snapshot(
            proposed_margin_pct=proposed_margin_pct,
            proposed_leverage=proposed_leverage,
            proposed_stop_loss_pct=proposed_stop_loss_pct,
            current_positions=current_positions,
            account_balance=account_balance,
        )

        if proposed_margin_pct > self.max_position_pct:
            adjusted = self.max_position_pct
            return LimitCheckResult(
                passed=True,
                reason=(
                    f"单笔保证金占用 {proposed_margin_pct:.1%} 超过上限 "
                    f"{self.max_position_pct:.1%}，已降到 {adjusted:.1%}。"
                ),
                adjusted_size_pct=adjusted,
            )

        if snapshot.margin_after_pct > snapshot.margin_limit_pct:
            remaining = max(snapshot.margin_limit_pct - snapshot.current_margin_pct, 0.0)
            if remaining >= 0.01:
                return LimitCheckResult(
                    passed=True,
                    reason=(
                        f"保证金占用接近上限：当前 {snapshot.current_margin_pct:.1%}，"
                        f"计划新增 {snapshot.proposed_margin_pct:.1%}，执行后 "
                        f"{snapshot.margin_after_pct:.1%}，上限 {snapshot.margin_limit_pct:.1%}。"
                        f"已把本次仓位降到 {remaining:.1%}。"
                    ),
                    adjusted_size_pct=remaining,
                )
            return LimitCheckResult(
                passed=False,
                reason=(
                    f"保证金占用过高：当前已占用 {snapshot.current_margin_pct:.1%}"
                    f"（约 {snapshot.current_margin_usdt:.2f} USDT，按账户权益 "
                    f"{snapshot.account_equity_usdt:.2f} USDT 计算），"
                    f"计划新增 {snapshot.proposed_margin_pct:.1%}"
                    f"（约 {snapshot.proposed_margin_usdt:.2f} USDT），"
                    f"执行后将达到 {snapshot.margin_after_pct:.1%}，"
                    f"超过上限 {snapshot.margin_limit_pct:.1%}。本次不执行新开仓。"
                ),
            )

        if snapshot.notional_after_pct > snapshot.notional_limit_pct:
            return LimitCheckResult(
                passed=False,
                reason=(
                    f"名义敞口过高：当前 {snapshot.current_notional_pct:.1%}，"
                    f"本次新增 {snapshot.proposed_notional_pct:.1%}，执行后 "
                    f"{snapshot.notional_after_pct:.1%}，超过上限 "
                    f"{snapshot.notional_limit_pct:.1%}。本次不执行新开仓。"
                ),
            )

        return LimitCheckResult(
            passed=True,
            reason=(
                f"合约仓位容量通过：保证金 {snapshot.margin_after_pct:.1%}/"
                f"{snapshot.margin_limit_pct:.1%}，名义敞口 {snapshot.notional_after_pct:.1%}/"
                f"{snapshot.notional_limit_pct:.1%}。"
            ),
        )

    def contract_exposure_snapshot(
        self,
        proposed_margin_pct: float,
        proposed_leverage: float,
        proposed_stop_loss_pct: float,
        current_positions: list[dict],
        account_balance: float,
    ) -> ContractExposureSnapshot:
        base = max(float(account_balance or 0.0), 1.0)
        current_notional = 0.0
        current_margin = 0.0
        current_stop_risk = 0.0

        for pos in current_positions or []:
            if not pos.get("is_open", True):
                continue
            quantity = self._safe_float(pos.get("quantity"), 0.0)
            entry_price = self._safe_float(pos.get("entry_price"), 0.0)
            if quantity <= 0 or entry_price <= 0:
                continue
            leverage = max(self._safe_float(pos.get("leverage"), 1.0), 1.0)
            notional = self._position_notional(pos, quantity, entry_price)
            margin = self._position_margin(pos, notional, leverage)
            current_notional += notional
            current_margin += margin
            current_stop_risk += self._position_stop_risk(pos, notional)

        leverage = max(float(proposed_leverage or 1.0), 1.0)
        margin_pct = max(float(proposed_margin_pct or 0.0), 0.0)
        stop_pct = max(float(proposed_stop_loss_pct or settings.hard_stop_loss_pct), 0.0)
        proposed_margin = base * margin_pct
        proposed_notional = proposed_margin * leverage
        proposed_stop_risk = proposed_notional * stop_pct

        current_margin_pct = current_margin / base
        current_notional_pct = current_notional / base
        current_stop_risk_pct = current_stop_risk / base
        proposed_notional_pct = proposed_notional / base
        proposed_stop_risk_pct = proposed_stop_risk / base

        return ContractExposureSnapshot(
            account_equity_usdt=base,
            current_margin_usdt=current_margin,
            proposed_margin_usdt=proposed_margin,
            current_margin_pct=current_margin_pct,
            proposed_margin_pct=margin_pct,
            margin_after_pct=current_margin_pct + margin_pct,
            margin_limit_pct=self.total_margin_limit_pct,
            current_stop_risk_pct=current_stop_risk_pct,
            proposed_stop_risk_pct=proposed_stop_risk_pct,
            stop_risk_after_pct=current_stop_risk_pct + proposed_stop_risk_pct,
            stop_risk_limit_pct=self.stop_risk_limit_pct,
            current_notional_pct=current_notional_pct,
            proposed_notional_pct=proposed_notional_pct,
            notional_after_pct=current_notional_pct + proposed_notional_pct,
            notional_limit_pct=self.total_notional_limit_pct,
        )

    @property
    def total_margin_limit_pct(self) -> float:
        configured = float(settings.max_total_margin_pct or 0.0)
        if configured > 0:
            return configured
        return self.max_position_pct * 3

    @property
    def stop_risk_limit_pct(self) -> float:
        return max(float(settings.max_daily_loss_pct or 0.05) * 1.5, self.max_position_pct * 0.5)

    @property
    def total_notional_limit_pct(self) -> float:
        leverage_cap = max(min(float(settings.max_leverage or 1.0), 20.0), 1.0)
        return max(self.max_position_pct * leverage_cap, self.total_margin_limit_pct)

    def entry_capacity_reason(
        self,
        current_positions: list[dict],
        account_balance: float,
        min_new_margin_pct: float = 0.02,
        default_leverage: float = 5.0,
        default_stop_loss_pct: float = 0.05,
    ) -> str | None:
        snapshot = self.contract_exposure_snapshot(
            proposed_margin_pct=min_new_margin_pct,
            proposed_leverage=default_leverage,
            proposed_stop_loss_pct=default_stop_loss_pct,
            current_positions=current_positions,
            account_balance=account_balance,
        )
        if snapshot.has_entry_capacity:
            return None
        if snapshot.margin_after_pct > snapshot.margin_limit_pct:
            return (
                "新开仓分析已暂停：保证金占用接近上限，为节省 Token 暂不分析新的交易对。"
                f"当前保证金占用 {snapshot.current_margin_pct:.1%}，预留最小新仓 "
                f"{snapshot.proposed_margin_pct:.1%} 后将达到 {snapshot.margin_after_pct:.1%}，"
                f"上限 {snapshot.margin_limit_pct:.1%}。已有持仓仍会继续复盘。"
            )
        return (
            "新开仓分析已暂停：当前名义敞口接近上限，为节省 Token 暂不分析新的交易对。"
            f"当前名义敞口 {snapshot.current_notional_pct:.1%}，预留最小新仓后 "
            f"{snapshot.notional_after_pct:.1%}，上限 {snapshot.notional_limit_pct:.1%}。"
            "已有持仓仍会继续复盘。"
        )

    def _position_stop_risk(self, pos: dict, notional: float) -> float:
        side = str(pos.get("side") or "").lower()
        entry = self._safe_float(pos.get("entry_price"), 0.0)
        stop = self._safe_float(pos.get("stop_loss") or pos.get("stop_loss_price"), 0.0)
        if entry <= 0 or notional <= 0:
            return notional * float(settings.hard_stop_loss_pct or 0.05)
        if stop > 0:
            if side == "short":
                adverse_move_pct = max(stop - entry, 0.0) / entry
            else:
                adverse_move_pct = max(entry - stop, 0.0) / entry
            if adverse_move_pct > 0:
                return notional * adverse_move_pct
        return notional * float(settings.hard_stop_loss_pct or 0.05)

    def _position_notional(self, pos: dict, quantity: float, entry_price: float) -> float:
        info_raw = pos.get("info")
        info = info_raw if isinstance(info_raw, dict) else {}
        direct = self._safe_float(
            pos.get("notional")
            or pos.get("notional_usd")
            or pos.get("notionalUsd")
            or pos.get("position_value")
            or info.get("notional")
            or info.get("notionalUsd")
            or info.get("posValue"),
            0.0,
        )
        if direct > 0:
            return abs(direct)
        contract_size = self._safe_float(
            pos.get("contract_size") or pos.get("contractSize") or info.get("ctVal"),
            1.0,
        )
        if (
            contract_size > 0
            and contract_size != 1.0
            and (
                pos.get("contracts") is not None
                or pos.get("contractSize") is not None
                or info.get("ctVal") is not None
            )
        ):
            return abs(quantity * contract_size * entry_price)
        return abs(quantity * entry_price)

    def _position_margin(self, pos: dict, notional: float, leverage: float) -> float:
        info_raw = pos.get("info")
        info = info_raw if isinstance(info_raw, dict) else {}
        computed = abs(notional) / max(leverage, 1.0)
        direct = self._safe_float(
            pos.get("margin")
            or pos.get("initial_margin")
            or pos.get("initialMargin")
            or pos.get("margin_used")
            or info.get("margin")
            or info.get("imr"),
            0.0,
        )
        if direct > 0:
            # Some OKX/CCXT swap payloads expose fields such as imr/initialMargin
            # as notional-like values instead of actual occupied margin.  Treat a
            # direct margin that is far above notional/leverage as suspicious;
            # otherwise a few positions can be misread as >200% margin usage and
            # block all new entries.
            if computed > 0 and direct > max(computed * 3.0, computed + 5.0):
                logger.warning(
                    "ignored suspicious direct position margin",
                    symbol=pos.get("symbol"),
                    direct_margin=direct,
                    computed_margin=computed,
                    notional=notional,
                    leverage=leverage,
                )
                return computed
            return abs(direct)
        return computed

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def check_leverage(self, proposed_leverage: float) -> LimitCheckResult:
        if proposed_leverage > self.max_leverage:
            return LimitCheckResult(
                passed=True,
                reason=f"Leverage {proposed_leverage:.1f}x capped to {self.max_leverage:.1f}x.",
                adjusted_size_pct=self.max_leverage,
            )
        return LimitCheckResult(passed=True, reason="Leverage within limits.")

    def check_single_symbol_exposure(
        self, symbol: str, proposed_size_pct: float, current_symbol_exposure: float
    ) -> LimitCheckResult:
        """Ensure a single symbol does not exceed position limit."""
        new_exposure = current_symbol_exposure + proposed_size_pct
        if new_exposure > self.max_position_pct * 1.5:
            return LimitCheckResult(
                passed=False,
                reason=(
                    f"Symbol {symbol} exposure would be {new_exposure:.1%}, "
                    f"exceeding limit of {self.max_position_pct * 1.5:.1%}."
                ),
            )
        return LimitCheckResult(passed=True, reason="Symbol exposure within limits.")
