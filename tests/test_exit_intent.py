from __future__ import annotations

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.exit_intent import ExitIntent, classify_exit_intent


def _decision(raw_response: dict | None = None, action: Action = Action.CLOSE_LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="exit",
        position_size_pct=1.0,
        suggested_leverage=3.0,
        raw_response=raw_response or {},
        feature_snapshot={"current_price": 100.0},
    )


def _policy(**overrides: object) -> dict:
    policy = {
        "eligible": True,
        "hard_risk": False,
        "planned_stop_crossed": False,
        "fee_after_unrealized_pnl_usdt": 0.0,
        "profit_retrace_ratio": 0.0,
        "stop_risk_usage": 0.0,
        "continuation_deterioration": 0.0,
        "opposite_pressure": 0.0,
        "close_fraction": 0.0,
        "policy_provenance": {
            "source": "current_position_fee_after_pnl_peak_planned_stop_and_market_returns"
        },
    }
    policy.update(overrides)
    return policy


@pytest.mark.parametrize(
    "legacy",
    [
        {"exit_intent": "trend_failure"},
        {"fast_risk_exit": True, "fast_risk_trigger": "fast_adverse_move"},
        {"forced_exit": True, "close_evidence": {"hard_risk": True}},
        {"close_evidence": {"moderate_opposite_pressure": True}},
        {"position_release_policy": {"forced": True}},
    ],
)
def test_exit_intent_rejects_legacy_authorization(legacy: dict) -> None:
    decision = _decision(legacy)

    assert classify_exit_intent(decision) == ExitIntent.ORDINARY
    assert decision.raw_response["exit_intent_policy"]["production_permission"] is False


def test_exit_intent_uses_governed_planned_stop() -> None:
    decision = _decision(
        {"dynamic_exit_policy": _policy(hard_risk=True, planned_stop_crossed=True)}
    )

    assert classify_exit_intent(decision) == ExitIntent.HARD_RISK


def test_exit_intent_classifies_governed_profit_drawdown() -> None:
    decision = _decision(
        {
            "close_evidence": {
                "dynamic_exit_policy": _policy(
                    fee_after_unrealized_pnl_usdt=3.0,
                    profit_retrace_ratio=0.4,
                    close_fraction=0.4,
                )
            }
        }
    )

    assert classify_exit_intent(decision) == ExitIntent.PROFIT_DRAWDOWN


def test_exit_intent_classifies_governed_loss_repair() -> None:
    decision = _decision(
        {
            "dynamic_exit_policy": _policy(
                fee_after_unrealized_pnl_usdt=-2.0,
                stop_risk_usage=0.3,
                close_fraction=0.3,
            )
        }
    )

    assert classify_exit_intent(decision) == ExitIntent.LOSS_REPAIR


def test_exit_intent_classifies_continuous_market_deterioration() -> None:
    decision = _decision(
        {
            "dynamic_exit_policy": _policy(
                continuation_deterioration=0.6,
                close_fraction=0.6,
            )
        }
    )

    assert classify_exit_intent(decision) == ExitIntent.PREDICTIVE_DOWNSIDE


def test_exit_intent_requires_authoritative_provenance() -> None:
    policy = _policy(hard_risk=True, planned_stop_crossed=True)
    policy["policy_provenance"] = {"source": "legacy_exit_rule"}

    assert classify_exit_intent(_decision({"dynamic_exit_policy": policy})) == ExitIntent.ORDINARY


def test_non_exit_action_is_hold() -> None:
    assert classify_exit_intent(_decision(action=Action.HOLD)) == ExitIntent.HOLD
