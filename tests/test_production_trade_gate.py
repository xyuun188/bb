from __future__ import annotations

from services.production_trade_gate import evaluate_production_trade_gate
from services.profit_training_contract import PROFIT_TRAINING_TARGET


def test_gate_allows_rules_canary_before_model_promotion() -> None:
    gate = evaluate_production_trade_gate(
        okx={"healthy": True, "can_open_new_entries": True},
        risk={"open_position_count": 0},
        model={
            "artifact_lifecycle": "shadow",
            "live_ml_ready": False,
            "metrics": {"sample_count": 5},
        },
        settings={"rules_canary_enabled": True},
    )

    assert gate.can_trade is True
    assert gate.mode == "live_rules_canary"
    assert gate.decision_authority == "rules"
    assert gate.model_can_influence is False
    assert gate.reason == "collecting_authoritative_profit_samples"


def test_gate_allows_live_ml_only_after_profit_and_authorization() -> None:
    gate = evaluate_production_trade_gate(
        okx={"healthy": True, "can_open_new_entries": True},
        risk={"open_position_count": 0},
        model={
            "artifact_lifecycle": "active",
            "live_ml_ready": True,
            "metrics": {
                "sample_count": 50,
                PROFIT_TRAINING_TARGET: 0.18,
                "return_lcb_pct": 0.04,
                "profit_factor": 1.35,
            },
        },
    )

    assert gate.can_trade is True
    assert gate.mode == "live_ml"
    assert gate.decision_authority == "model"
    assert gate.model_can_influence is True


def test_gate_uses_profit_training_target_for_live_ml() -> None:
    gate = evaluate_production_trade_gate(
        okx={"healthy": True, "can_open_new_entries": True},
        risk={"open_position_count": 0},
        model={
            "artifact_lifecycle": "active",
            "live_ml_ready": True,
            "metrics": {
                "sample_count": 50,
                PROFIT_TRAINING_TARGET: 0.18,
                "return_lcb_pct": 0.04,
                "profit_factor": 1.35,
            },
        },
    )

    assert gate.can_trade is True
    assert gate.mode == "live_ml"


def test_gate_keeps_zero_rules_canary_notional_limit_closed() -> None:
    gate = evaluate_production_trade_gate(
        okx={"healthy": True, "can_open_new_entries": True},
        risk={"open_position_count": 0},
        model={"artifact_lifecycle": "shadow"},
        settings={
            "rules_canary_enabled": True,
            "rules_canary_risk": {"max_notional_usdt": 0},
        },
    )

    assert gate.can_trade is False
    assert gate.mode == "blocked"
    assert gate.reason == "rules_canary_notional_limit_closed"
    assert gate.risk.max_notional_usdt == 0.0


def test_gate_blocks_unhealthy_okx_before_any_trading_mode() -> None:
    gate = evaluate_production_trade_gate(
        okx={"healthy": False, "can_open_new_entries": True},
        risk={"open_position_count": 0},
        model={"live_ml_ready": True},
    )

    assert gate.can_trade is False
    assert gate.mode == "blocked"
    assert gate.reason == "okx_unhealthy"


def test_gate_blocks_when_rules_canary_position_limit_reached() -> None:
    gate = evaluate_production_trade_gate(
        okx={"healthy": True, "can_open_new_entries": True},
        risk={"open_position_count": 1},
        settings={"rules_canary_risk": {"max_open_positions": 1}},
    )

    assert gate.can_trade is False
    assert gate.mode == "blocked"
    assert gate.reason == "max_open_positions_reached"
