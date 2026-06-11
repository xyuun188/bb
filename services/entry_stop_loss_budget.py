"""Entry stop-loss budget policy for risk sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ENTRY_MAX_STOP_LOSS_NORMAL_USDT = 16.0
ENTRY_MAX_STOP_LOSS_DRAWDOWN_USDT = 8.0
ENTRY_MAX_STOP_LOSS_DEFENSIVE_USDT = 4.0
ENTRY_HIGH_QUALITY_STOP_LOSS_DRAWDOWN_USDT = 12.0
ENTRY_HIGH_QUALITY_STOP_LOSS_DEFENSIVE_USDT = 8.0
ENTRY_MAX_STOP_LOSS_PCT_OF_EQUITY = 0.008
ENTRY_MAX_STOP_LOSS_CAP_USDT = 36.0

_DRAWDOWN_RISK_MODES = frozenset({"drawdown_recovery"})
_DEFENSIVE_RISK_MODES = frozenset({"defensive_recovery", "hard_recovery"})
_HIGH_QUALITY_BOOST_RISK_MODES = _DRAWDOWN_RISK_MODES | _DEFENSIVE_RISK_MODES


@dataclass(frozen=True, slots=True)
class EntryStopLossBudget:
    """Resolved stop-loss budget used by entry risk sizing."""

    max_loss_usdt: float
    configured_max_loss_usdt: float
    dynamic_hard_cap_usdt: float
    risk_budget_boost: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class EntryStopLossBudgetPolicy:
    """Resolve the planned-loss budget for a new entry candidate."""

    normal_budget_usdt: float = ENTRY_MAX_STOP_LOSS_NORMAL_USDT
    drawdown_budget_usdt: float = ENTRY_MAX_STOP_LOSS_DRAWDOWN_USDT
    defensive_budget_usdt: float = ENTRY_MAX_STOP_LOSS_DEFENSIVE_USDT
    high_quality_drawdown_budget_usdt: float = ENTRY_HIGH_QUALITY_STOP_LOSS_DRAWDOWN_USDT
    high_quality_defensive_budget_usdt: float = ENTRY_HIGH_QUALITY_STOP_LOSS_DEFENSIVE_USDT
    equity_cap_pct: float = ENTRY_MAX_STOP_LOSS_PCT_OF_EQUITY
    min_dynamic_cap_usdt: float = 6.0
    max_dynamic_cap_usdt: float = ENTRY_MAX_STOP_LOSS_CAP_USDT
    min_final_budget_usdt: float = 1.0

    def default_budget_usdt(self, risk_mode: str) -> float:
        """Return the default stop-loss budget for a risk mode."""

        if risk_mode == "defensive_recovery":
            return self.defensive_budget_usdt
        if risk_mode == "drawdown_recovery":
            return self.drawdown_budget_usdt
        return self.normal_budget_usdt

    def dynamic_hard_cap_usdt(self, balance: float) -> float:
        """Return the account-equity based hard cap."""

        return min(
            max(max(float(balance or 0.0), 0.0) * self.equity_cap_pct, self.min_dynamic_cap_usdt),
            self.max_dynamic_cap_usdt,
        )

    def resolve(
        self,
        *,
        risk_mode: str,
        configured_max_loss_usdt: float,
        balance: float,
        high_quality_entry: bool,
        low_payoff_quality: bool,
    ) -> EntryStopLossBudget:
        """Resolve final stop-loss budget and any high-quality recovery boost."""

        configured = float(configured_max_loss_usdt or 0.0)
        if configured <= 0:
            configured = self.default_budget_usdt(risk_mode)

        dynamic_hard_cap = self.dynamic_hard_cap_usdt(balance)
        risk_budget_boost: dict[str, Any] | None = None

        if (
            high_quality_entry
            and not low_payoff_quality
            and risk_mode in _HIGH_QUALITY_BOOST_RISK_MODES
        ):
            boost_budget = (
                self.high_quality_defensive_budget_usdt
                if risk_mode in _DEFENSIVE_RISK_MODES
                else self.high_quality_drawdown_budget_usdt
            )
            boosted_max_loss = min(max(configured, boost_budget), dynamic_hard_cap)
            if boosted_max_loss > configured:
                risk_budget_boost = {
                    "applied": True,
                    "from_usdt": round(configured, 6),
                    "to_usdt": round(boosted_max_loss, 6),
                    "reason": ("当前为高质量同向机会，亏损恢复模式下允许更合理的单笔风险预算。"),
                }
                configured = boosted_max_loss

        if low_payoff_quality:
            configured = min(configured, self.defensive_budget_usdt)

        configured = min(configured, dynamic_hard_cap)
        max_loss = max(configured, self.min_final_budget_usdt)
        return EntryStopLossBudget(
            max_loss_usdt=max_loss,
            configured_max_loss_usdt=configured,
            dynamic_hard_cap_usdt=dynamic_hard_cap,
            risk_budget_boost=risk_budget_boost,
        )
