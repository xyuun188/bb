"""Risk management engine for final pre-execution validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from ai_brain.base_model import Action, DecisionOutput
from core.symbols import normalize_trading_symbol
from risk_manager.circuit_breaker import CircuitBreaker
from risk_manager.position_limits import PositionLimitChecker
from risk_manager.stop_loss import StopLossResult

logger = structlog.get_logger(__name__)


@dataclass
class RiskAssessment:
    """Result of the full risk evaluation pipeline."""

    approved: bool
    decision: DecisionOutput | None
    stop_loss_result: StopLossResult | None = None
    rejection_reason: str = ""
    warnings: list[str] = field(default_factory=list)


class RiskEngine:
    """Validate decisions against hard safety controls and advisory risk context."""

    def __init__(
        self,
        *,
        position_checker: PositionLimitChecker | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self.position_checker = position_checker or PositionLimitChecker()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

    def assess(
        self,
        decision: DecisionOutput,
        current_positions: list[dict],
        account_balance: float,
    ) -> RiskAssessment:
        """Evaluate a trading decision before it reaches the executor."""

        warnings: list[str] = []

        if self.circuit_breaker.is_open and decision.is_entry:
            return RiskAssessment(
                approved=False,
                decision=decision,
                rejection_reason="Circuit breaker is open; no new entries are allowed.",
            )

        if decision.is_entry:
            entry_result = self._assess_entry(
                decision=decision,
                current_positions=current_positions,
                account_balance=account_balance,
                warnings=warnings,
            )
            if entry_result is not None:
                return entry_result

        return RiskAssessment(
            approved=True,
            decision=decision,
            stop_loss_result=None,
            warnings=warnings,
        )

    def _assess_entry(
        self,
        *,
        decision: DecisionOutput,
        current_positions: list[dict],
        account_balance: float,
        warnings: list[str],
    ) -> RiskAssessment | None:
        contract_reason = self._dynamic_risk_contract_reason(decision)
        if contract_reason:
            return RiskAssessment(
                approved=False,
                decision=decision,
                rejection_reason=contract_reason,
            )
        model_open_positions = [
            position
            for position in current_positions
            if position.get("model_name") == decision.model_name
        ]
        decision_side = "long" if decision.action == Action.LONG else "short"
        decision_symbol = normalize_trading_symbol(decision.symbol)
        opposite_side = "short" if decision_side == "long" else "long"
        opposite_symbol_positions = [
            position
            for position in model_open_positions
            if self._is_effective_open_position(position)
            if position.get("side") == opposite_side
            and normalize_trading_symbol(position.get("symbol")) == decision_symbol
        ]
        if opposite_symbol_positions:
            return RiskAssessment(
                approved=False,
                decision=decision,
                rejection_reason=(
                    "OKX 净持仓模式下，同币种反向 entry 会先抵消/平掉已有仓位，"
                    f"禁止把 {decision_side} 当作普通新开仓提交；请先平掉或反转已有 "
                    f"{opposite_side} 仓位后再重新评估。"
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

        return None

    @staticmethod
    def _dynamic_risk_contract_reason(decision: DecisionOutput) -> str | None:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        sizing = raw.get("profit_risk_sizing")
        sizing = sizing if isinstance(sizing, dict) else {}
        provenance = sizing.get("policy_provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        required = (
            "source",
            "observation_window",
            "sample_count",
            "generated_at",
            "strategy_version",
            "fallback_reason",
        )
        provenance_complete = bool(
            all(key in provenance for key in required)
            and str(provenance.get("source") or "").strip()
            and str(provenance.get("observation_window") or "").strip()
            and RiskEngine._safe_positive_float(provenance.get("sample_count")) > 0
            and str(provenance.get("generated_at") or "").strip()
            and str(provenance.get("strategy_version") or "").strip()
            and not str(provenance.get("fallback_reason") or "").strip()
        )
        planned_loss = RiskEngine._safe_positive_float(sizing.get("planned_stop_loss_usdt"))
        max_loss = RiskEngine._safe_positive_float(sizing.get("max_stop_loss_usdt"))
        stress_stop = RiskEngine._safe_positive_float(sizing.get("stress_stop_loss_pct"))
        if sizing.get("production_eligible") is not True:
            return "Dynamic account risk budget is not production eligible."
        if not provenance_complete:
            return "Dynamic account risk budget provenance is incomplete."
        if planned_loss <= 0 or max_loss <= 0 or planned_loss > max_loss + 1e-9:
            return "Dynamic planned loss exceeds or is missing from the account risk budget."
        if stress_stop <= 0 or float(decision.position_size_pct or 0.0) <= 0:
            return "Dynamic stop distance or position size is missing."
        return None

    @staticmethod
    def _safe_positive_float(value: Any) -> float:
        try:
            return max(float(value or 0.0), 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _is_effective_open_position(position: dict[str, Any]) -> bool:
        if position.get("is_open", True) is False:
            return False
        if "quantity" not in position:
            return True
        try:
            return float(position.get("quantity") or 0.0) > 1e-12
        except (TypeError, ValueError):
            return True
