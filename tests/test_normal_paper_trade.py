from __future__ import annotations

import pytest

from ai_brain.base_model import Action, DecisionOutput
from models.decision import _compact_decision_learning_snapshot
from services.normal_paper_trade import (
    HISTORICAL_NORMAL_PAPER_TRADE_VERSIONS,
    NORMAL_PAPER_ORDER_IDENTITY_VERSION,
    NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION,
    NORMAL_PAPER_TRADE_TRAINING_PORTFOLIO_RISK_FRACTION,
    NORMAL_PAPER_TRADE_TRAINING_SINGLE_TRADE_RISK_FRACTION,
    NORMAL_PAPER_TRADE_VERSION,
    _contract_fingerprint_payload,
    _fingerprint,
    attach_normal_paper_order_identity,
    build_normal_paper_trade_contract,
    ensure_normal_paper_trade_contract,
    normal_paper_decision_id_from_client_order_id,
    normal_paper_order_identity_reasons,
    normal_paper_settlement_contract_reasons,
    normal_paper_trade_contract_reasons,
    normal_paper_trade_observation_contract_reasons,
    normalize_normal_paper_contract,
    reconcile_normal_paper_trade_contract,
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
        "raw_expected_return_pct": expected_net + 0.1,
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
        "current_edge_validated": True,
    }


def _quality_observation_support(
    side: str,
    *,
    expected_net: float = 0.4,
    objective_net: float = 0.2,
    loss_probability: float = 0.4,
) -> dict:
    support = _support(
        side,
        expected_net=expected_net,
        objective_net=objective_net,
    )
    permission = support["quant_quality_permissions"]["local_ml"]
    permission.update(
        {
            "paper_execution_permission": False,
            "paper_execution_reason": "fee_after_return_lcb_not_positive",
            "paper_execution_blockers": ["fee_after_return_lcb_not_positive"],
            "paper_execution_evidence": {"sample_count": 0},
        }
    )
    support["loss_probability"] = loss_probability
    support["paper_quality_observation_only"] = True
    support["current_edge_validated"] = False
    support["paper_quality_observation_reasons"] = [
        "fee_after_return_lcb_not_positive"
    ]
    return support


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


def test_reconcile_normal_trade_uses_current_execution_return_and_blocks_stale_positive_edge() -> None:
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=_support("long", expected_net=0.2, objective_net=0.1),
    )
    reconciled = reconcile_normal_paper_trade_contract(
        contract,
        {
            "expected_net_return_pct": -0.12,
            "return_lcb_pct": -0.55,
            "execution_cost": {"total_pct": 0.31},
            "return_distribution_contract": {
                "gross_market_distribution": {"raw_expected_return_pct": 0.19}
            },
        },
    )

    assert reconciled["authorized"] is False
    assert reconciled["expected_net_return_pct"] == pytest.approx(-0.12)
    assert reconciled["current_raw_expected_return_pct"] == pytest.approx(0.19)
    assert reconciled["reconciliation"]["status"] == (
        "blocked_current_execution_return"
    )
    assert "normal_paper_trade_expected_net_not_positive" in (
        normal_paper_trade_contract_reasons(reconciled)
    )


def test_reconcile_normal_trade_keeps_positive_current_execution_edge_authorized() -> None:
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=_support("long", expected_net=0.2, objective_net=0.1),
    )
    reconciled = reconcile_normal_paper_trade_contract(
        contract,
        {
            "expected_net_return_pct": 0.08,
            "return_lcb_pct": -0.22,
            "execution_cost": {"total_pct": 0.19},
            "return_distribution_contract": {
                "gross_market_distribution": {"raw_expected_return_pct": 0.27}
            },
        },
    )

    assert reconciled["authorized"] is True
    assert reconciled["current_edge_validated"] is True
    assert reconciled["expected_net_return_pct"] == pytest.approx(0.08)
    assert reconciled["current_raw_expected_return_pct"] == pytest.approx(0.27)
    assert normal_paper_trade_contract_reasons(reconciled) == []


def test_reconcile_cannot_restore_entry_after_historical_quality_permission_is_denied() -> None:
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=_support("long", expected_net=0.2, objective_net=0.1),
    )
    contract["quant_quality_permissions"]["local_ml"][
        "paper_execution_permission"
    ] = False
    contract["contract_fingerprint"] = _fingerprint(
        _contract_fingerprint_payload(contract)
    )

    reconciled = reconcile_normal_paper_trade_contract(
        contract,
        {
            "expected_net_return_pct": 0.08,
            "return_lcb_pct": -0.22,
            "execution_cost": {"total_pct": 0.19},
        },
    )

    assert reconciled["authorized"] is False
    assert "normal_paper_trade_quality_permission_not_profit_validated" in (
        normal_paper_trade_contract_reasons(reconciled)
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


def test_normal_paper_contract_keeps_an_immutable_quality_evidence_snapshot() -> None:
    support = _support("long", expected_net=0.2)
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=support,
    )

    support_permission = support["quant_quality_permissions"]["local_ml"]
    support_permission["paper_execution_reason"] = "mutated_after_contract_build"
    support_permission["paper_execution_evidence"]["sample_count"] = 999
    support_permission["paper_execution_blockers"] = ["late_mutation"]

    contract_permission = contract["quant_quality_permissions"]["local_ml"]
    assert contract_permission["paper_execution_reason"] == (
        "authoritative_fee_after_quality_above_break_even"
    )
    assert contract_permission["paper_execution_evidence"]["sample_count"] == 20
    assert "paper_execution_blockers" not in contract_permission
    assert normal_paper_trade_contract_reasons(contract) == []


def test_learning_snapshot_preserves_exact_fingerprinted_normal_paper_contract() -> None:
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="short",
        selection_reason="paper_quality_observation",
        direction_support=_quality_observation_support(
            "short",
            expected_net=0.2,
            objective_net=-0.3,
        ),
    )

    compact = _compact_decision_learning_snapshot(
        {"normal_paper_trade": contract}
    )

    assert compact["normal_paper_trade"] == contract
    assert normal_paper_trade_observation_contract_reasons(
        compact["normal_paper_trade"]
    ) == []


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
    assert selection["diagnostic_status"] == "evidence_blocked"
    assert "direction_support_expected_net_not_positive" in selection["blocking_reasons"]
    assert selection["blocking_reasons_by_side"]["short"]
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


def test_current_positive_edge_with_denied_historical_quality_is_training_only() -> None:
    support = _support("long", expected_net=0.2)
    support["quant_quality_permissions"]["local_ml"]["paper_execution_permission"] = False

    selection = select_normal_paper_trade_side({"long": support})
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason=selection["selection_reason"],
        direction_support=selection["selected_support"],
    )

    assert selection["selection_reason"] == "paper_training_entry"
    assert contract["authorized"] is True
    assert contract["paper_training_only"] is True
    assert contract["production_permission"] is False
    assert contract["single_trade_risk_fraction_cap"] == (
        NORMAL_PAPER_TRADE_TRAINING_SINGLE_TRADE_RISK_FRACTION
    )
    assert contract["portfolio_risk_fraction_cap"] == (
        NORMAL_PAPER_TRADE_TRAINING_PORTFOLIO_RISK_FRACTION
    )
    assert normal_paper_trade_contract_reasons(contract) == []
    assert normal_paper_trade_observation_contract_reasons(contract) == []


def test_strategy_entry_contract_cannot_bypass_denied_historical_quality() -> None:
    support = _support("long", expected_net=0.2)
    support["quant_quality_permissions"]["local_ml"]["paper_execution_permission"] = False

    assert build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=support,
    ) == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_count", 0),
        ("sample_count", float("nan")),
        ("average_return", 0.0),
        ("average_return", -0.01),
        ("average_return", None),
        ("return_lcb", 0.0),
        ("return_lcb", -0.1),
        ("return_lcb", None),
        ("profit_factor", 1.0),
        ("profit_factor", 0.9),
        ("profit_factor", float("inf")),
        ("profit_factor", "invalid"),
        ("profit_factor_above_break_even", False),
        ("profit_factor_above_break_even", None),
    ],
)
def test_positive_permission_cannot_override_invalid_fee_after_evidence(
    field: str,
    value: object,
) -> None:
    support = _support("long", expected_net=0.2)
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=support,
    )
    permission = support["quant_quality_permissions"]["local_ml"]
    permission["paper_execution_evidence"][field] = value

    selection = select_normal_paper_trade_side({"long": support})
    assert selection["selection_reason"] == "paper_training_entry"
    assert build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=support,
    ) == {}

    contract["quant_quality_permissions"]["local_ml"] = permission
    contract["contract_fingerprint"] = _fingerprint(
        _contract_fingerprint_payload(contract)
    )
    assert "normal_paper_trade_quality_permission_not_profit_validated" in (
        normal_paper_trade_contract_reasons(contract)
    )
    reconciled = reconcile_normal_paper_trade_contract(
        contract,
        {"expected_net_return_pct": 0.2, "return_lcb_pct": -0.1},
    )
    assert reconciled["authorized"] is False


def test_all_winning_historical_evidence_does_not_require_a_finite_profit_factor() -> None:
    support = _support("long", expected_net=0.2)
    permission = support["quant_quality_permissions"]["local_ml"]
    permission["paper_execution_evidence"]["profit_factor"] = None
    permission["break_even_contract"]["additional_diagnostic"] = True

    selection = select_normal_paper_trade_side({"long": support})
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason=selection["selection_reason"],
        direction_support=selection["selected_support"],
    )

    assert selection["selection_reason"] == "strategy_edge_selected"
    assert contract["authorized"] is True
    assert normal_paper_trade_contract_reasons(contract) == []


def test_malformed_quality_source_cannot_be_filtered_into_entry_permission() -> None:
    support = _support("long", expected_net=0.2)
    support["quant_quality_permissions"]["unverified_source"] = None

    assert build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=support,
    ) == {}


def test_historical_quality_permission_cannot_replace_invalid_current_edge() -> None:
    support = _quality_observation_support(
        "long",
        expected_net=0.2,
        objective_net=0.1,
    )
    support["current_edge_validated"] = False

    assert build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=support,
    ) == {}


def test_unpromoted_quality_model_stays_shadow_only() -> None:
    support = _quality_observation_support("short")

    selection = select_normal_paper_trade_side({"short": support})
    assert selection["selected"] is False
    assert selection["selected_side"] == "neutral"
    assert selection["selection_reason"] == "no_direction"


def test_current_positive_fee_after_mean_with_negative_downside_stays_training_only() -> None:
    support = _quality_observation_support(
        "short",
        expected_net=0.08,
        objective_net=-0.45,
    )
    support["current_edge_validated"] = True
    support["raw_expected_return_pct"] = 0.22

    selection = select_normal_paper_trade_side({"short": support})

    assert selection["selected"] is True
    assert selection["selection_reason"] == "paper_training_entry"

    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="short",
        selection_reason=selection["selection_reason"],
        direction_support=selection["selected_support"],
    )
    assert contract["authorized"] is True
    assert contract["current_edge_validated"] is True
    assert contract["objective_net_return_pct"] == pytest.approx(-0.45)
    assert normal_paper_trade_contract_reasons(contract) == []
    assert normal_paper_trade_observation_contract_reasons(contract) == []


def test_current_positive_signal_with_negative_downside_and_denied_history_is_shadow_only() -> None:
    support = _support(
        "short",
        expected_net=0.08,
        objective_net=-0.45,
    )
    support["raw_expected_return_pct"] = 0.22
    permission = support["quant_quality_permissions"]["local_ml"]
    permission["paper_execution_permission"] = False
    permission["paper_execution_reason"] = "fee_after_return_lcb_not_positive"
    permission["paper_execution_blockers"] = ["fee_after_return_lcb_not_positive"]

    selection = select_normal_paper_trade_side({"short": support})

    assert selection["selected"] is True
    assert selection["selection_reason"] == "paper_training_entry"
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side=selection["selected_side"],
        selection_reason=selection["selection_reason"],
        direction_support=selection["selected_support"],
    )
    assert contract["authorized"] is True
    assert contract["objective_net_return_pct"] == pytest.approx(-0.45)
    assert normal_paper_trade_contract_reasons(contract) == []
    assert normal_paper_trade_observation_contract_reasons(contract) == []


def test_current_positive_edge_does_not_override_denied_quality_permission() -> None:
    support = _quality_observation_support(
        "long",
        expected_net=0.12,
        objective_net=0.04,
    )
    support["current_edge_validated"] = True
    support["raw_expected_return_pct"] = 0.28
    support["aligned_expert_count"] = 3
    support["opposition_expert_count"] = 0

    selection = select_normal_paper_trade_side({"long": support})

    assert selection["selected"] is True
    assert selection["selection_reason"] == "paper_training_entry"

    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason=selection["selection_reason"],
        direction_support=selection["selected_support"],
    )
    assert contract["authorized"] is True
    assert normal_paper_trade_contract_reasons(contract) == []
    assert normal_paper_trade_observation_contract_reasons(contract) == []


def test_negative_lcb_quality_observation_cannot_authorize_a_paper_entry() -> None:
    support = _quality_observation_support(
        "long",
        expected_net=0.35,
        objective_net=-3.2,
    )

    selection = select_normal_paper_trade_side({"long": support})
    assert selection["selected"] is False
    assert selection["selected_side"] == "neutral"
    assert selection["selection_reason"] == "no_direction"


def test_unauthorized_shadow_observation_cannot_be_reused_as_entry_permission() -> None:
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="paper_quality_observation",
        direction_support=_quality_observation_support("long", objective_net=-0.3),
    )
    contract["authorized"] = False
    contract["contract_fingerprint"] = _fingerprint(_contract_fingerprint_payload(contract))

    assert normal_paper_trade_observation_contract_reasons(contract) == []
    assert "normal_paper_trade_not_authorized" in normal_paper_trade_contract_reasons(contract)


def test_negative_downside_does_not_veto_positive_mean_normal_paper_entry() -> None:
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=_support("long", expected_net=0.2, objective_net=-0.3),
    )
    assert contract["authorized"] is True
    assert contract["objective_net_return_pct"] == -0.3
    assert normal_paper_trade_contract_reasons(contract) == []


def test_current_positive_edge_does_not_inherit_historical_quality_observation_cap() -> None:
    support = _quality_observation_support(
        "long",
        expected_net=0.8,
        objective_net=0.3,
    )
    support["current_edge_validated"] = True
    support["paper_quality_observation_only"] = False
    support["quant_quality_permissions"]["local_ml"].update(
        {
            "paper_execution_permission": True,
            "paper_execution_reason": (
                "authoritative_fee_after_quality_above_break_even"
            ),
            "paper_execution_blockers": [],
            "paper_execution_evidence": {
                "sample_count": 20,
                "average_return": 0.2,
                "return_lcb": 0.1,
                "profit_factor": 1.5,
                "profit_factor_above_break_even": True,
            },
        }
    )

    selection = select_normal_paper_trade_side({"long": support})
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side=selection["selected_side"],
        selection_reason=selection["selection_reason"],
        direction_support=support,
    )

    assert selection["selection_reason"] == "strategy_edge_selected"
    assert contract["paper_quality_observation_only"] is False
    assert (
        contract["single_trade_risk_fraction_cap"]
        == NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION
    )
    assert normal_paper_trade_contract_reasons(contract) == []


@pytest.mark.parametrize("legacy_version", sorted(HISTORICAL_NORMAL_PAPER_TRADE_VERSIONS))
def test_every_historical_contract_uses_one_read_only_normalization_path(
    legacy_version: str,
) -> None:
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="short",
        selection_reason="paper_quality_observation",
        direction_support=_quality_observation_support(
            "short",
            expected_net=0.8,
            objective_net=0.3,
        ),
    )
    source_cap = 0.00015
    contract["version"] = legacy_version
    contract["single_trade_risk_fraction_cap"] = source_cap
    contract["contract_fingerprint"] = _fingerprint(
        _contract_fingerprint_payload(contract)
    )

    normalized = normalize_normal_paper_contract(contract)

    assert normalized["protocol_state"] == "historical_normalized"
    assert normalized["source_version"] == legacy_version
    assert normalized["version"] == NORMAL_PAPER_TRADE_VERSION
    assert normalized["single_trade_risk_fraction_cap"] == (
        NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION
    )
    assert normalized["source_contract"]["single_trade_risk_fraction_cap"] == source_cap
    assert normal_paper_settlement_contract_reasons(contract) == []
    assert "normal_paper_trade_version_invalid" in normal_paper_trade_contract_reasons(
        contract
    )


def test_unknown_contract_version_is_not_migrated_or_authorized() -> None:
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support=_support("long", expected_net=0.2),
    )
    contract["version"] = "unregistered.normal-paper-contract"

    assert normalize_normal_paper_contract(contract) == {}
    assert normal_paper_settlement_contract_reasons(contract)
    assert normal_paper_trade_contract_reasons(contract)


def test_validated_strategy_candidate_precedes_quality_observation_candidate() -> None:
    selection = select_normal_paper_trade_side(
        {
            "long": _support("long", expected_net=0.15, objective_net=0.05),
            "short": _quality_observation_support(
                "short",
                expected_net=1.5,
                objective_net=-0.1,
            ),
        }
    )

    assert selection["selected_side"] == "long"
    assert selection["selection_reason"] == "strategy_edge_selected"


def test_quality_observation_rejects_excessive_loss_probability() -> None:
    support = _quality_observation_support("long", loss_probability=0.61)
    selection = select_normal_paper_trade_side({"long": support})

    assert selection["selected"] is False
    assert selection["selected_side"] == "neutral"
    assert build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="paper_quality_observation",
        direction_support=support,
    ) == {}


def test_positive_expected_net_with_non_positive_objective_stays_shadow_only() -> None:
    support = _quality_observation_support(
        "long",
        expected_net=0.2,
        objective_net=-0.01,
    )
    selection = select_normal_paper_trade_side({"long": support})

    assert selection["selected"] is False
    assert selection["selected_side"] == "neutral"
    assert selection["selection_reason"] == "no_direction"


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


def test_v19_entry_can_settle_but_cannot_be_retried_as_a_new_order() -> None:
    support = _support("short", expected_net=0.2, objective_net=-0.3)
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="short",
        selection_reason="strategy_edge_selected",
        direction_support=support,
    )
    contract["version"] = "2026-10-21.normal-paper-strategy-trade.v19"
    contract["quant_quality_permissions"]["local_ml"]["paper_execution_permission"] = False
    contract["contract_fingerprint"] = _fingerprint(
        _contract_fingerprint_payload(contract)
    )
    decision = _decision(
        {
            "normal_paper_trade": contract,
            "paper_trade_selection": {"selection_reason": "strategy_edge_selected"},
            "independent_direction_support": support,
        },
        action=Action.SHORT,
    )

    assert normal_paper_settlement_contract_reasons(contract) == []
    assert normal_paper_trade_contract_reasons(contract)
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

    assert normal_paper_settlement_contract_reasons(contract) == []
    assert "normal_paper_trade_version_invalid" in normal_paper_trade_contract_reasons(contract)
    assert normal_paper_order_identity_reasons(
        identity,
        decision_id=23,
        contract=contract,
    ) == []
