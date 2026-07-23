"""Authoritative trading mode gate for the profit-loop refactor.

This module is intentionally small and side-effect free.  It centralizes the
question "what may trade now?" without submitting orders, mutating models, or
reading external services directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Literal

from services.profit_training_contract import PROFIT_TRAINING_TARGET

TradeGateMode = Literal["observe", "live_rules_canary", "live_ml", "blocked"]
DecisionAuthority = Literal["none", "rules", "model"]

PRODUCTION_TRADE_GATE_VERSION = "2026-07-23.profit-loop-trade-gate.v1"


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    number = _safe_float(value, None)
    if number is None:
        return default
    return int(number)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


@dataclass(frozen=True, slots=True)
class TradeGateRiskLimits:
    max_notional_usdt: float = 10.0
    max_open_positions: int = 1
    max_daily_loss_usdt: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProductionTradeGateResult:
    can_trade: bool
    mode: TradeGateMode
    decision_authority: DecisionAuthority
    model_can_influence: bool
    reason: str
    risk: TradeGateRiskLimits = field(default_factory=TradeGateRiskLimits)
    evidence: dict[str, Any] = field(default_factory=dict)
    version: str = PRODUCTION_TRADE_GATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk"] = self.risk.to_dict()
        return payload


def _risk_limits(settings: dict[str, Any], risk: dict[str, Any]) -> TradeGateRiskLimits:
    source = _safe_dict(settings.get("rules_canary_risk")) or _safe_dict(
        risk.get("rules_canary_risk")
    )
    max_notional = _safe_float(source.get("max_notional_usdt"), None)
    max_open_positions = _safe_int(source.get("max_open_positions"), 1)
    max_daily_loss = _safe_float(source.get("max_daily_loss_usdt"), None)
    return TradeGateRiskLimits(
        max_notional_usdt=max(10.0 if max_notional is None else max_notional, 0.0),
        max_open_positions=max(max_open_positions, 0),
        max_daily_loss_usdt=max(3.0 if max_daily_loss is None else max_daily_loss, 0.0),
    )


def _okx_healthy(okx: dict[str, Any]) -> bool:
    if okx.get("healthy") is False or okx.get("ok") is False:
        return False
    if okx.get("can_open_new_entries") is False:
        return False
    if str(okx.get("status") or "").lower() in {"blocked", "unhealthy", "error"}:
        return False
    return True


def _risk_blocker(risk: dict[str, Any], limits: TradeGateRiskLimits) -> str | None:
    if risk.get("blocked") is True or risk.get("risk_blocked") is True:
        return str(risk.get("reason") or "risk_blocked")
    open_positions = _safe_int(risk.get("open_position_count"), 0)
    if open_positions >= limits.max_open_positions:
        return "max_open_positions_reached"
    daily_loss = _safe_float(risk.get("daily_loss_usdt"), 0.0) or 0.0
    if daily_loss >= limits.max_daily_loss_usdt > 0:
        return "daily_loss_limit_reached"
    return None


def _model_profit_ready(model: dict[str, Any]) -> tuple[bool, list[str]]:
    metrics = _safe_dict(model.get("metrics"))
    if not metrics:
        metrics = model
    blockers: list[str] = []
    return_lcb = _safe_float(
        _first_present(
            metrics.get("return_lcb_pct"),
            metrics.get("top_return_lcb_pct"),
            metrics.get("top_long_return_lcb_pct"),
            metrics.get("top_short_return_lcb_pct"),
        ),
        None,
    )
    profit_factor = _safe_float(
        _first_present(
            metrics.get("profit_factor"),
            metrics.get("top_profit_factor"),
            metrics.get("top_long_profit_factor"),
            metrics.get("top_short_profit_factor"),
        ),
        None,
    )
    expected_net = _safe_float(
        _first_present(
            metrics.get(PROFIT_TRAINING_TARGET),
            metrics.get("expected_net_return_pct"),
            metrics.get("avg_return_pct"),
            metrics.get("top_avg_return_pct"),
        ),
        None,
    )
    sample_count = _safe_int(
        _first_present(
            metrics.get("production_sample_count"),
            metrics.get("closed_trade_sample_count"),
            metrics.get("sample_count"),
        ),
        0,
    )
    min_samples = _safe_int(model.get("min_live_ml_samples"), 30)

    if sample_count < min_samples:
        blockers.append("model_profit_sample_count_insufficient")
    if expected_net is None or expected_net <= 0:
        blockers.append("model_expected_net_return_not_positive")
    if return_lcb is None or return_lcb <= 0:
        blockers.append("model_return_lcb_not_positive")
    if profit_factor is None or profit_factor <= 1.0:
        blockers.append("model_profit_factor_not_above_one")
    return not blockers, blockers


def _model_live_ready(model: dict[str, Any]) -> tuple[bool, list[str]]:
    lifecycle = str(model.get("artifact_lifecycle") or model.get("stage") or "").lower()
    influence_allowed = bool(
        model.get("live_ml_ready")
        or model.get("production_influence_authorized")
        or model.get("allow_live_position_influence")
    )
    blockers: list[str] = []
    if lifecycle not in {"active", "live"} and not bool(model.get("live_ml_ready")):
        blockers.append("model_not_promoted_to_live")
    if not influence_allowed:
        blockers.append("model_live_influence_not_authorized")
    profit_ready, profit_blockers = _model_profit_ready(model)
    if not profit_ready:
        blockers.extend(profit_blockers)
    return not blockers, blockers


def evaluate_production_trade_gate(
    *,
    okx: dict[str, Any] | None = None,
    risk: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
) -> ProductionTradeGateResult:
    """Return the single authoritative trade mode for the current cycle."""

    okx = _safe_dict(okx)
    risk = _safe_dict(risk)
    model = _safe_dict(model)
    training = _safe_dict(training)
    settings = _safe_dict(settings)
    limits = _risk_limits(settings, risk)
    evidence = {
        "okx": okx,
        "risk": risk,
        "model": model,
        "training": training,
    }

    if not _okx_healthy(okx):
        return ProductionTradeGateResult(
            can_trade=False,
            mode="blocked",
            decision_authority="none",
            model_can_influence=False,
            reason="okx_unhealthy",
            risk=limits,
            evidence=evidence,
        )

    risk_reason = _risk_blocker(risk, limits)
    if risk_reason:
        return ProductionTradeGateResult(
            can_trade=False,
            mode="blocked",
            decision_authority="none",
            model_can_influence=False,
            reason=risk_reason,
            risk=limits,
            evidence=evidence,
        )

    model_ready, model_blockers = _model_live_ready(model)
    if model_ready:
        return ProductionTradeGateResult(
            can_trade=True,
            mode="live_ml",
            decision_authority="model",
            model_can_influence=True,
            reason="ok",
            risk=limits,
            evidence=evidence,
        )

    rules_canary_enabled = settings.get("rules_canary_enabled") is not False
    if rules_canary_enabled:
        evidence["live_ml_blockers"] = model_blockers
        if limits.max_notional_usdt <= 0:
            return ProductionTradeGateResult(
                can_trade=False,
                mode="blocked",
                decision_authority="none",
                model_can_influence=False,
                reason="rules_canary_notional_limit_closed",
                risk=limits,
                evidence=evidence,
            )
        return ProductionTradeGateResult(
            can_trade=True,
            mode="live_rules_canary",
            decision_authority="rules",
            model_can_influence=False,
            reason="collecting_authoritative_profit_samples",
            risk=limits,
            evidence=evidence,
        )

    reason = "data_insufficient" if model_blockers else "observe_only"
    evidence["live_ml_blockers"] = model_blockers
    return ProductionTradeGateResult(
        can_trade=False,
        mode="observe",
        decision_authority="none",
        model_can_influence=False,
        reason=reason,
        risk=limits,
        evidence=evidence,
    )
