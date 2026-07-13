"""Physical account-bound enforcement after dynamic position sizing."""

from __future__ import annotations

from dataclasses import dataclass


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
    """Enforce account-equity math; exchange leverage is checked by OKX."""

    def check_contract_entry_limits(
        self,
        proposed_margin_pct: float,
        proposed_leverage: float,
        proposed_stop_loss_pct: float,
        current_positions: list[dict],
        account_balance: float,
        symbol: str,
    ) -> LimitCheckResult:
        snapshot = self.contract_exposure_snapshot(
            proposed_margin_pct=proposed_margin_pct,
            proposed_leverage=proposed_leverage,
            proposed_stop_loss_pct=proposed_stop_loss_pct,
            current_positions=current_positions,
            account_balance=account_balance,
        )

        if snapshot.margin_after_pct > snapshot.margin_limit_pct:
            remaining = max(snapshot.margin_limit_pct - snapshot.current_margin_pct, 0.0)
            if remaining > 0.0:
                return LimitCheckResult(
                    passed=True,
                    reason=(
                        f"{symbol} margin capped to current account capacity "
                        f"({snapshot.margin_limit_pct:.1%})."
                    ),
                    adjusted_size_pct=remaining,
                )
            return LimitCheckResult(
                passed=False,
                reason=f"{symbol} has no remaining account margin capacity.",
            )

        return LimitCheckResult(passed=True, reason="Physical account margin boundary passed.")

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

        for position in current_positions or []:
            if not position.get("is_open", True):
                continue
            quantity = self._safe_float(position.get("quantity"), 0.0)
            entry_price = self._safe_float(position.get("entry_price"), 0.0)
            if quantity <= 0.0 or entry_price <= 0.0:
                continue
            leverage = max(self._safe_float(position.get("leverage"), 1.0), 1.0)
            notional = self._position_notional(position, quantity, entry_price)
            margin = self._position_margin(position, notional, leverage)
            current_notional += notional
            current_margin += margin
            current_stop_risk += self._position_stop_risk(position, notional)

        leverage = max(float(proposed_leverage or 1.0), 1.0)
        margin_pct = max(float(proposed_margin_pct or 0.0), 0.0)
        stop_pct = max(float(proposed_stop_loss_pct or 0.0), 0.0)
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
            stop_risk_limit_pct=current_stop_risk_pct + proposed_stop_risk_pct,
            current_notional_pct=current_notional_pct,
            proposed_notional_pct=proposed_notional_pct,
            notional_after_pct=current_notional_pct + proposed_notional_pct,
            notional_limit_pct=float("inf"),
        )

    @property
    def total_margin_limit_pct(self) -> float:
        return 1.0

    def _position_stop_risk(self, position: dict, notional: float) -> float:
        side = str(position.get("side") or "").lower()
        entry = self._safe_float(position.get("entry_price"), 0.0)
        stop = self._safe_float(
            position.get("stop_loss") or position.get("stop_loss_price"), 0.0
        )
        if entry <= 0.0 or notional <= 0.0:
            return 0.0
        if stop > 0.0:
            adverse_move_pct = (
                max(stop - entry, 0.0) / entry
                if side == "short"
                else max(entry - stop, 0.0) / entry
            )
            if adverse_move_pct > 0.0:
                return notional * adverse_move_pct
        return 0.0

    def _position_notional(self, position: dict, quantity: float, entry_price: float) -> float:
        info = position.get("info") if isinstance(position.get("info"), dict) else {}
        direct = self._safe_float(
            position.get("notional")
            or position.get("notional_usd")
            or position.get("notionalUsd")
            or position.get("position_value")
            or info.get("notional")
            or info.get("notionalUsd")
            or info.get("posValue"),
            0.0,
        )
        if direct > 0.0:
            return abs(direct)
        contract_size = self._safe_float(
            position.get("contract_size")
            or position.get("contractSize")
            or info.get("ctVal"),
            1.0,
        )
        has_contract_shape = (
            position.get("contracts") is not None
            or position.get("contractSize") is not None
            or info.get("ctVal") is not None
        )
        multiplier = contract_size if contract_size > 0.0 and has_contract_shape else 1.0
        return abs(quantity * multiplier * entry_price)

    def _position_margin(self, position: dict, notional: float, leverage: float) -> float:
        info = position.get("info") if isinstance(position.get("info"), dict) else {}
        direct = self._safe_float(
            position.get("margin")
            or position.get("initial_margin")
            or position.get("initialMargin")
            or position.get("margin_used")
            or info.get("margin")
            or info.get("imr"),
            0.0,
        )
        return abs(direct) if direct > 0.0 else abs(notional) / max(leverage, 1.0)

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
