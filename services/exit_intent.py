"""Exit-intent labels derived only from the governed dynamic exit contract."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from ai_brain.base_model import DecisionOutput


class ExitIntent(StrEnum):
    HARD_RISK = "hard_risk"
    TREND_FAILURE = "trend_failure"
    PREDICTIVE_DOWNSIDE = "predictive_downside"
    PROFIT_DRAWDOWN = "profit_drawdown"
    PROFIT_PROTECTION = "profit_protection"
    CAPITAL_ROTATION = "capital_rotation"
    LOSS_REPAIR = "loss_repair"
    ORDINARY = "ordinary"
    HOLD = "hold"


PROTECTIVE_DOWNSIDE_INTENTS = frozenset(
    {
        ExitIntent.HARD_RISK,
        ExitIntent.TREND_FAILURE,
        ExitIntent.PREDICTIVE_DOWNSIDE,
    }
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dynamic_exit_contract(raw: dict[str, Any]) -> dict[str, Any]:
    policy = _safe_dict(raw.get("dynamic_exit_policy"))
    if not policy:
        policy = _safe_dict(_safe_dict(raw.get("close_evidence")).get("dynamic_exit_policy"))
    provenance = _safe_dict(policy.get("policy_provenance"))
    if policy.get("eligible") is not True or provenance.get("source") not in {
        "current_position_fee_after_pnl_peak_planned_stop_and_market_returns",
        "current_position_takeover_fee_after_pnl_peak_planned_stop_market_and_portfolio_facts",
    }:
        return {}
    return policy


def classify_exit_intent(
    decision: DecisionOutput,
    *,
    update_raw: bool = True,
) -> ExitIntent:
    """Label an exit without granting any execution permission."""

    if not decision.is_exit:
        return ExitIntent.HOLD

    raw = _safe_dict(decision.raw_response)
    policy = _dynamic_exit_contract(raw)
    if not policy:
        intent = ExitIntent.ORDINARY
    elif policy.get("hard_risk") is True and policy.get("planned_stop_crossed") is True:
        intent = ExitIntent.HARD_RISK
    else:
        net_pnl = _safe_float(policy.get("fee_after_unrealized_pnl_usdt"))
        retrace = _safe_float(policy.get("profit_retrace_ratio"))
        stop_usage = _safe_float(policy.get("stop_risk_usage"))
        continuation = _safe_float(policy.get("continuation_deterioration"))
        opposite = _safe_float(policy.get("opposite_pressure"))
        if net_pnl > 0.0 and retrace > 0.0:
            intent = ExitIntent.PROFIT_DRAWDOWN
        elif net_pnl < 0.0 and stop_usage > 0.0:
            intent = ExitIntent.LOSS_REPAIR
        elif continuation > 0.0 or opposite > 0.0:
            intent = ExitIntent.PREDICTIVE_DOWNSIDE
        elif net_pnl > 0.0 and _safe_float(policy.get("close_fraction")) > 0.0:
            intent = ExitIntent.PROFIT_PROTECTION
        else:
            intent = ExitIntent.ORDINARY

    if update_raw:
        raw["exit_intent"] = intent.value
        raw["exit_intent_policy"] = {
            "intent": intent.value,
            "structured": True,
            "source": "governed_dynamic_exit_contract",
            "production_permission": False,
        }
        decision.raw_response = raw
    return intent
