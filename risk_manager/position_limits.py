"""Validate authoritative entry contracts against the physical OKX portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any


@dataclass
class LimitCheckResult:
    passed: bool
    reason: str = ""


@dataclass
class ContractExposureSnapshot:
    contract_algebra_valid: bool
    account_equity_usdt: float
    available_margin_usdt: float
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
            self.contract_algebra_valid
            and self.margin_after_pct <= self.margin_limit_pct + 1e-12
            and self.stop_risk_after_pct <= self.stop_risk_limit_pct + 1e-12
            and self.notional_after_pct <= self.notional_limit_pct + 1e-12
        )


class PositionLimitChecker:
    """Validate one authoritative risk contract without resizing it."""

    def check_contract_entry_limits(
        self,
        *,
        risk_contract: dict[str, Any],
        current_positions: list[dict[str, Any]],
        account_balance: float,
        symbol: str,
    ) -> LimitCheckResult:
        snapshot = self.contract_exposure_snapshot(
            risk_contract=risk_contract,
            current_positions=current_positions,
            account_balance=account_balance,
        )
        if not snapshot.contract_algebra_valid:
            return LimitCheckResult(
                passed=False,
                reason=f"{symbol} authoritative notional and stressed loss are inconsistent.",
            )
        if snapshot.proposed_margin_usdt > snapshot.available_margin_usdt + 1e-8:
            return LimitCheckResult(
                passed=False,
                reason=f"{symbol} authoritative margin exceeds current OKX available margin.",
            )
        if snapshot.stop_risk_after_pct > snapshot.stop_risk_limit_pct + 1e-12:
            return LimitCheckResult(
                passed=False,
                reason=f"{symbol} would exceed the dynamic portfolio stressed-loss budget.",
            )
        if snapshot.notional_after_pct > snapshot.notional_limit_pct + 1e-12:
            return LimitCheckResult(
                passed=False,
                reason=f"{symbol} would exceed notional capacity implied by stressed loss.",
            )
        if not snapshot.has_entry_capacity:
            return LimitCheckResult(
                passed=False,
                reason=f"{symbol} has no remaining authoritative portfolio risk capacity.",
            )
        return LimitCheckResult(
            passed=True,
            reason="Authoritative margin and portfolio stressed-loss boundaries passed.",
        )

    def contract_exposure_snapshot(
        self,
        *,
        risk_contract: dict[str, Any],
        current_positions: list[dict[str, Any]],
        account_balance: float,
    ) -> ContractExposureSnapshot:
        contract = risk_contract if isinstance(risk_contract, dict) else {}
        equity = self._safe_float(contract.get("account_equity_usdt"), 0.0)
        if equity <= 0:
            equity = max(float(account_balance or 0.0), 0.0)
        base = max(equity, 1e-12)
        available_margin = max(
            self._safe_float(contract.get("available_margin_usdt"), 0.0),
            0.0,
        )
        contract_specs = (
            contract.get("exchange_contract_specs")
            if isinstance(contract.get("exchange_contract_specs"), dict)
            else {}
        )
        current_notional = 0.0
        current_margin = 0.0
        current_stop_risk = 0.0
        for position in current_positions or []:
            if not position.get("is_open", True):
                continue
            notional = self._position_notional(position, contract_specs)
            leverage = max(
                self._safe_float(
                    position.get("leverage") or self._info(position).get("lever"),
                    1.0,
                ),
                1.0,
            )
            current_notional += notional
            current_margin += self._position_margin(position, notional, leverage)
            current_stop_risk += self._position_stop_risk(position, notional)

        proposed_margin = max(
            self._safe_float(contract.get("final_margin_usdt"), 0.0),
            0.0,
        )
        proposed_notional = max(
            self._safe_float(contract.get("final_notional_usdt"), 0.0),
            0.0,
        )
        proposed_stop_risk = max(
            self._safe_float(contract.get("planned_stressed_loss_usdt"), 0.0),
            0.0,
        )
        stress_fraction = max(
            self._safe_float(contract.get("stressed_loss_fraction"), 0.0),
            0.0,
        )
        portfolio_budget = max(
            self._safe_float(contract.get("portfolio_risk_budget_usdt"), 0.0),
            0.0,
        )
        remaining_stop_capacity = max(portfolio_budget - current_stop_risk, 0.0)
        proposed_notional_capacity = (
            remaining_stop_capacity / stress_fraction if stress_fraction > 0 else 0.0
        )
        margin_limit = current_margin + available_margin
        notional_limit = current_notional + proposed_notional_capacity

        contract_algebra_valid = isclose(
            proposed_stop_risk,
            proposed_notional * stress_fraction,
            rel_tol=1e-9,
            abs_tol=1e-8,
        )

        return ContractExposureSnapshot(
            contract_algebra_valid=contract_algebra_valid,
            account_equity_usdt=base,
            available_margin_usdt=available_margin,
            current_margin_usdt=current_margin,
            proposed_margin_usdt=proposed_margin,
            current_margin_pct=current_margin / base,
            proposed_margin_pct=proposed_margin / base,
            margin_after_pct=(current_margin + proposed_margin) / base,
            margin_limit_pct=margin_limit / base,
            current_stop_risk_pct=current_stop_risk / base,
            proposed_stop_risk_pct=proposed_stop_risk / base,
            stop_risk_after_pct=(current_stop_risk + proposed_stop_risk) / base,
            stop_risk_limit_pct=portfolio_budget / base,
            current_notional_pct=current_notional / base,
            proposed_notional_pct=proposed_notional / base,
            notional_after_pct=(current_notional + proposed_notional) / base,
            notional_limit_pct=notional_limit / base,
        )

    def _position_stop_risk(self, position: dict[str, Any], notional: float) -> float:
        side = str(position.get("side") or "").lower()
        info = self._info(position)
        mark = self._safe_float(
            position.get("current_price")
            or position.get("markPrice")
            or info.get("markPx"),
            0.0,
        )
        stop = self._safe_float(
            position.get("stop_loss") or position.get("stop_loss_price"),
            0.0,
        )
        if mark <= 0 or stop <= 0 or notional <= 0:
            return 0.0
        adverse_move = (
            max(stop - mark, 0.0) / mark
            if side == "short"
            else max(mark - stop, 0.0) / mark
        )
        return notional * adverse_move

    def _position_notional(
        self,
        position: dict[str, Any],
        contract_specs: dict[str, Any],
    ) -> float:
        info = self._info(position)
        direct = abs(
            self._safe_float(
                position.get("notional")
                or position.get("notional_usd")
                or position.get("notionalUsd")
                or info.get("notionalUsd")
                or info.get("notional")
                or info.get("posValue"),
                0.0,
            )
        )
        if direct > 0:
            return direct
        inst_id = str(info.get("instId") or position.get("okx_inst_id") or "").upper()
        spec = contract_specs.get(inst_id) if isinstance(contract_specs.get(inst_id), dict) else {}
        contracts = abs(
            self._safe_float(
                position.get("contracts")
                or position.get("sz")
                or info.get("pos")
                or info.get("qty"),
                0.0,
            )
        )
        quantity = abs(self._safe_float(position.get("quantity"), 0.0))
        ct_val = max(
            self._safe_float(spec.get("ctVal"), 0.0),
            self._safe_float(position.get("contract_size") or position.get("contractSize"), 0.0),
            self._safe_float(info.get("ctVal"), 0.0),
        )
        ct_mult = max(self._safe_float(spec.get("ctMult") or info.get("ctMult"), 0.0), 0.0)
        mark = max(
            self._safe_float(position.get("current_price") or position.get("markPrice"), 0.0),
            self._safe_float(info.get("markPx"), 0.0),
            self._safe_float(position.get("entry_price") or position.get("entryPrice"), 0.0),
        )
        if contracts > 0:
            return contracts * ct_val * ct_mult * mark
        return quantity * mark

    def _position_margin(self, position: dict[str, Any], notional: float, leverage: float) -> float:
        info = self._info(position)
        direct = self._safe_float(
            position.get("margin")
            or position.get("initial_margin")
            or position.get("initialMargin")
            or position.get("margin_used")
            or info.get("margin")
            or info.get("imr"),
            0.0,
        )
        return abs(direct) if direct > 0 else abs(notional) / max(leverage, 1.0)

    @staticmethod
    def _info(position: dict[str, Any]) -> dict[str, Any]:
        return position.get("info") if isinstance(position.get("info"), dict) else {}

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
