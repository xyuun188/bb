from __future__ import annotations

from ai_brain.base_model import Action, DecisionOutput
from services.normal_paper_trade import (
    NORMAL_PAPER_TRADE_MAX_COVERAGE_RISK_FRACTION,
    NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION,
    build_normal_paper_trade_contract,
    ensure_normal_paper_trade_contract,
    normal_paper_trade_contract_reasons,
    select_normal_paper_trade_side,
)


def _support(side: str, *, expected_net: float) -> dict:
    return {
        "eligible": True,
        "selected_side": side,
        "prediction_horizon_minutes": 30.0,
        "expected_net_return_pct": expected_net,
        "objective_net_return_pct": expected_net - 0.1,
        "loss_probability": 0.4,
        "quant_evidence_families": ["local_ml"],
        "strong_expert_opposition": False,
    }


def _decision(raw: dict, *, action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.5,
        reasoning="test",
        position_size_pct=0.0,
        suggested_leverage=2.0,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        raw_response=raw,
        feature_snapshot={"current_price": 100.0},
    )


def test_positive_net_direction_builds_normal_policy_trade() -> None:
    selection = select_normal_paper_trade_side(
        {"long": _support("long", expected_net=0.2)}
    )
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side=selection["selected_side"],
        selection_reason=selection["selection_reason"],
        direction_support=selection["selected_support"],
    )

    assert normal_paper_trade_contract_reasons(contract) == []
    assert contract["entry_type"] == "normal_strategy_trade"
    assert contract["trade_kind"] == "normal_strategy_trade"
    assert contract["selection_reason"] == "policy_exploitation"
    assert (
        contract["single_trade_risk_fraction_cap"]
        == NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION
    )
    assert contract["production_permission"] is False


def test_negative_net_direction_uses_bounded_coverage_in_same_order_pipeline() -> None:
    selection = select_normal_paper_trade_side(
        {"short": _support("short", expected_net=-0.05)}
    )
    contract = build_normal_paper_trade_contract(
        symbol="ETH/USDT",
        side=selection["selected_side"],
        selection_reason=selection["selection_reason"],
        direction_support=selection["selected_support"],
    )

    assert normal_paper_trade_contract_reasons(contract) == []
    assert contract["selection_reason"] == "coverage_sampling"
    assert (
        contract["single_trade_risk_fraction_cap"]
        == NORMAL_PAPER_TRADE_MAX_COVERAGE_RISK_FRACTION
    )
    assert contract["uses_shared_order_pipeline"] is True
    assert contract["uses_shared_position_ledger"] is True
    assert contract["separate_sampling_order"] is False


def test_existing_signed_contract_can_be_attached_to_paper_decision() -> None:
    support = _support("long", expected_net=0.2)
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="policy_exploitation",
        direction_support=support,
    )
    decision = _decision(
        {
            "paper_trade_selection": {
                "selection_reason": "policy_exploitation",
                "decision_authority": "ensemble",
            },
            "independent_direction_support": support,
            "normal_paper_trade": contract,
        }
    )

    assert ensure_normal_paper_trade_contract(decision, "paper") == contract


def test_legacy_paper_identities_cannot_authorize_a_new_trade() -> None:
    for raw in (
        {"paper_training": {"authorized": True}},
        {"paper_exploration": {"authorized": True}},
        {"paper_bootstrap_canary": {"authorized": True}},
    ):
        decision = _decision(raw)
        assert ensure_normal_paper_trade_contract(decision, "paper") == {}
        assert "normal_paper_trade" not in decision.raw_response


def test_normal_paper_contract_never_attaches_to_live() -> None:
    decision = _decision(
        {
            "paper_trade_selection": {"selection_reason": "policy_exploitation"},
            "independent_direction_support": _support("long", expected_net=0.2),
        }
    )

    assert ensure_normal_paper_trade_contract(decision, "live") == {}
    assert "normal_paper_trade" not in decision.raw_response
