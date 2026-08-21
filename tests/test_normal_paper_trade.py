from __future__ import annotations

from ai_brain.base_model import Action, DecisionOutput
from services.normal_paper_trade import (
    NORMAL_PAPER_ORDER_IDENTITY_VERSION,
    NORMAL_PAPER_TRADE_MAX_QUALITY_OBSERVATION_RISK_FRACTION,
    NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION,
    attach_normal_paper_order_identity,
    build_normal_paper_trade_contract,
    ensure_normal_paper_trade_contract,
    legacy_normal_paper_v4_trade_contract_reasons,
    normal_paper_decision_id_from_client_order_id,
    normal_paper_order_identity_reasons,
    normal_paper_settlement_contract_reasons,
    normal_paper_trade_contract_reasons,
    select_normal_paper_trade_side,
)
from tests.legacy_paper_contract_fixtures import (
    build_legacy_normal_paper_v4_trade_contract,
)


def _support(
    side: str,
    *,
    expected_net: float,
    objective_net: float | None = None,
) -> dict:
    return {
        "eligible": True,
        "selected_side": side,
        "prediction_horizon_minutes": 30.0,
        "expected_net_return_pct": expected_net,
        "objective_net_return_pct": (
            expected_net - 0.1 if objective_net is None else objective_net
        ),
        "loss_probability": 0.4,
        "quant_evidence_families": ["local_ml"],
        "quant_quality_permissions": {
            "local_ml": {
                "paper_execution_permission": True,
                "paper_execution_reason": (
                    "authoritative_fee_after_quality_above_break_even"
                ),
                "paper_execution_evidence_source": "test_authoritative_trade",
                "paper_execution_evidence": {
                    "sample_count": 20,
                    "average_return": 0.2,
                    "return_lcb": 0.1,
                    "profit_factor": 1.5,
                    "profit_factor_above_break_even": True,
                },
                "break_even_contract": {
                    "average_return_above_zero": True,
                    "return_lcb_above_zero": True,
                    "profit_factor_above_one": True,
                },
            }
        },
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
    assert contract["selection_reason"] == "strategy_edge_selected"
    assert (
        contract["single_trade_risk_fraction_cap"]
        == NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION
    )
    assert contract["production_permission"] is False
    assert "portfolio_risk_fraction_cap" not in contract
    assert contract["leverage_policy"] == "dynamic_risk_and_okx_tier"
    assert contract["model_leverage_role"] == "upper_bound_when_explicit"
    assert "leverage_cap" not in contract


def test_negative_net_direction_cannot_authorize_a_paper_order() -> None:
    selection = select_normal_paper_trade_side(
        {"short": _support("short", expected_net=-0.05)}
    )
    contract = build_normal_paper_trade_contract(
        symbol="ETH/USDT",
        side=selection["selected_side"],
        selection_reason=selection["selection_reason"],
        direction_support=selection["selected_support"],
    )

    assert selection["selected"] is False
    assert selection["selected_side"] == "neutral"
    assert contract == {}


def test_positive_direction_without_quality_permission_cannot_authorize_order() -> None:
    support = _support("long", expected_net=0.2)
    support.pop("quant_quality_permissions")

    assert build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=support,
    ) == {}


def test_unpromoted_quality_model_builds_lower_risk_observation_contract() -> None:
    support = _support("short", expected_net=0.4)
    permission = support["quant_quality_permissions"]["local_ml"]
    permission.update(
        {
            "paper_execution_permission": False,
            "paper_execution_reason": "fee_after_return_lcb_not_positive",
            "paper_execution_blockers": ["fee_after_return_lcb_not_positive"],
            "paper_execution_evidence": {"sample_count": 0},
        }
    )
    support["paper_quality_observation_only"] = True
    support["paper_quality_observation_reasons"] = [
        "fee_after_return_lcb_not_positive"
    ]

    selection = select_normal_paper_trade_side({"short": support})
    contract = build_normal_paper_trade_contract(
        symbol="LTC/USDT",
        side=selection["selected_side"],
        selection_reason=selection["selection_reason"],
        direction_support=selection["selected_support"],
    )

    assert selection["selection_reason"] == "paper_quality_observation"
    assert normal_paper_trade_contract_reasons(contract) == []
    assert contract["paper_quality_mode"] == "quality_observation"
    assert contract["paper_quality_observation_only"] is True
    assert contract["production_permission"] is False
    assert contract["single_trade_risk_fraction_cap"] == (
        NORMAL_PAPER_TRADE_MAX_QUALITY_OBSERVATION_RISK_FRACTION
    )


def test_positive_expected_net_with_non_positive_objective_cannot_authorize_order() -> None:
    support = _support("long", expected_net=0.2, objective_net=-0.01)
    selection = select_normal_paper_trade_side({"long": support})

    assert selection["selected"] is False
    assert selection["selected_side"] == "neutral"
    assert (
        build_normal_paper_trade_contract(
            symbol="BTC/USDT",
            side="long",
            selection_reason="strategy_edge_selected",
            direction_support=support,
        )
        == {}
    )


def test_existing_signed_contract_can_be_attached_to_paper_decision() -> None:
    support = _support("long", expected_net=0.2)
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=support,
    )
    decision = _decision(
        {
            "paper_trade_selection": {
                "selection_reason": "strategy_edge_selected",
                "decision_authority": "ensemble",
            },
            "independent_direction_support": support,
            "normal_paper_trade": contract,
        }
    )

    assert ensure_normal_paper_trade_contract(decision, "paper") == contract


def test_legacy_v4_contract_is_removed_before_new_entry_authorization() -> None:
    support = _support("long", expected_net=0.2, objective_net=-0.01)
    decision = _decision(
        {
            "paper_trade_selection": {
                "selection_reason": "strategy_edge_selected",
                "decision_authority": "ensemble",
            },
            "independent_direction_support": support,
            "normal_paper_trade": build_legacy_normal_paper_v4_trade_contract(
                symbol="BTC/USDT",
                side="long",
                objective_net_return_pct=-0.01,
            ),
        }
    )

    assert ensure_normal_paper_trade_contract(decision, "paper") == {}
    assert "normal_paper_trade" not in decision.raw_response


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
            "paper_trade_selection": {"selection_reason": "strategy_edge_selected"},
            "independent_direction_support": _support("long", expected_net=0.2),
        }
    )

    assert ensure_normal_paper_trade_contract(decision, "live") == {}
    assert "normal_paper_trade" not in decision.raw_response


def test_normal_paper_order_identity_round_trips_exact_decision() -> None:
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=_support("long", expected_net=0.2),
    )
    decision = _decision({"normal_paper_trade": contract})

    identity = attach_normal_paper_order_identity(
        decision,
        model_mode="paper",
        decision_id=137947,
    )

    assert identity["client_order_id"] == "BBNP137947"
    assert normal_paper_decision_id_from_client_order_id("BBNP137947") == 137947
    assert normal_paper_order_identity_reasons(
        identity,
        decision_id=137947,
        contract=contract,
    ) == []


def test_normal_paper_order_identity_rejects_cross_decision_reuse() -> None:
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=_support("long", expected_net=0.2),
    )
    decision = _decision({"normal_paper_trade": contract})
    identity = attach_normal_paper_order_identity(
        decision,
        model_mode="paper",
        decision_id=11,
    )

    assert "normal_paper_order_identity_decision_mismatch" in normal_paper_order_identity_reasons(
        identity,
        decision_id=12,
        contract=contract,
    )


def test_v4_contract_is_recovery_eligible_but_cannot_authorize_new_entry() -> None:
    contract = build_legacy_normal_paper_v4_trade_contract(
        symbol="BTC/USDT",
        side="long",
        objective_net_return_pct=-0.2,
    )
    identity = {
        "version": NORMAL_PAPER_ORDER_IDENTITY_VERSION,
        "decision_id": 23,
        "client_order_id": "BBNP23",
        "execution_scope": "paper_only",
        "entry_type": "normal_strategy_trade",
        "production_permission": False,
        "normal_trade_contract_fingerprint": contract["contract_fingerprint"],
    }

    assert legacy_normal_paper_v4_trade_contract_reasons(contract) == []
    assert normal_paper_settlement_contract_reasons(contract) == []
    assert "normal_paper_trade_version_invalid" in normal_paper_trade_contract_reasons(contract)
    assert normal_paper_order_identity_reasons(
        identity,
        decision_id=23,
        contract=contract,
    ) == []
