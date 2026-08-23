from copy import deepcopy
from math import isinf

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.entry_opportunity_scoring import EntryOpportunityScoringPolicy
from services.profit_supervision import (
    PRODUCTION_RETURN_COMBINATION_VERSION,
    PROFIT_SUPERVISION_VERSION,
)
from services.return_objective import (
    COST_MODEL_VERSION,
    PAPER_RETURN_COMBINATION_VERSION,
    RETURN_DISTRIBUTION_CONTRACT_VERSION,
    RETURN_LABEL_NAME,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
    standardized_return_distribution,
)
from tests.paper_canary_fixtures import complete_paper_canary_raw


def _scorer() -> EntryOpportunityScoringPolicy:
    return EntryOpportunityScoringPolicy(
        normalize_symbol=lambda value: str(value or ""),
        annotate_decision_source=lambda _decision: None,
    )


def _return_distribution(
    side: str,
    expected: float,
    *,
    horizon_minutes: int = 30,
) -> dict:
    return standardized_return_distribution(
        side=side,
        horizon_minutes=horizon_minutes,
        raw_expected_return_pct=expected,
        median_return_pct=expected,
        lower_quantile_return_pct=expected - 0.1,
        upper_quantile_return_pct=expected + 0.1,
        dispersion_pct=0.1,
        tail_loss_probability=0.2 if side == "long" else 0.3,
        tail_loss_scale_pct=0.1,
        distribution_member_count=32,
        return_semantics="gross_market_opportunity_before_execution",
        source_authority="test_tree_empirical_distribution",
        objective_version=RETURN_OBJECTIVE_VERSION,
        label_version=RETURN_LABEL_VERSION,
        cost_model_version=COST_MODEL_VERSION,
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
    )


def _live_payload(*, side: str, long_return: float, short_return: float) -> dict:
    return {
        "available": True,
        "route_mode": "live",
        "live_ml_ready": True,
        "promotion_ready": True,
        "horizon_minutes": 30,
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "training_cost_policy": "separated_market_opportunity_and_execution_cost_tasks",
        "prediction_quality": {
            "production_eligible": True,
            "anomalous": False,
        },
        "best_side": side,
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        "return_semantics": "gross_market_opportunity_before_execution",
        "return_distribution_contract_version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
        "return_distribution_contract": {
            "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
            "long": _return_distribution("long", long_return),
            "short": _return_distribution("short", short_return),
        },
        "long_market_expected_return_pct": long_return,
        "short_market_expected_return_pct": short_return,
        "long_expected_return_pct": long_return,
        "short_expected_return_pct": short_return,
        "long_lower_bound_return_pct": long_return - 0.1,
        "short_lower_bound_return_pct": short_return - 0.1,
        "counterfactual_execution_cost_distribution": {
            "long": _cost_distribution(),
            "short": _cost_distribution(),
            "source_authority": "shadow_counterfactual_live_microstructure",
        },
        "actual_trade_calibration": {
            "long": _trade_calibration("long"),
            "short": _trade_calibration("short"),
            "source_authority": "okx_position_history",
        },
        "long_loss_probability": 0.2,
        "short_loss_probability": 0.3,
    }


def _cost_distribution() -> dict:
    return {
        "expected_pct": 0.09,
        "upper_tail_pct": 0.10,
        "uncertainty_pct": 0.01,
        "distribution_ready": True,
        "source_authority": "shadow_counterfactual_live_microstructure",
    }


def _trade_calibration(side: str) -> dict:
    return {
        "source_authority": "okx_position_history",
        "symbol": "BTC/USDT",
        "side": side,
        "profile_source": "symbol_side",
        "net_return_after_all_cost_pct": {
            "count": 12,
            "expected": 0.7,
            "lower_hinge": 0.6,
        },
        "slippage_pct": {
            "count": 12,
            "expected": 0.012,
            "upper_hinge": 0.02,
        },
    }


def _live_ml() -> dict:
    return {
        "available": True,
        "route_mode": "live",
        "live_ml_ready": True,
        "promotion_ready": True,
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "training_cost_policy": "separated_market_opportunity_and_execution_cost_tasks",
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        "return_semantics": "gross_market_opportunity_before_execution",
        "prediction_quality": {
            "production_eligible": True,
            "anomalous": False,
        },
        "return_distribution_contract_version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
        "return_distribution_contract": {
            "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
            "long": _return_distribution("long", 0.8),
            "short": _return_distribution("short", -0.2),
        },
        "readiness": {"live_ml_ready": True},
        "influence_policy": {"long": {"enabled": True}, "short": {"enabled": True}},
        "predictions": [
            {
                "best_side": "long",
                "return_distribution_contract_version": (
                    RETURN_DISTRIBUTION_CONTRACT_VERSION
                ),
                "return_distribution_contract": {
                    "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
                    "long": _return_distribution("long", 0.8),
                    "short": _return_distribution("short", -0.2),
                },
                "long_market_expected_return_pct": 0.8,
                "short_market_expected_return_pct": -0.2,
                "long_expected_return_pct": 0.8,
                "short_expected_return_pct": -0.2,
                "long_market_lower_hinge_return_pct": 0.7,
                "short_market_lower_hinge_return_pct": -0.3,
                "long_market_distribution_ready": True,
                "short_market_distribution_ready": True,
                "long_tail_loss_probability": 0.1,
                "short_tail_loss_probability": 0.4,
                "horizon_minutes": 30,
                "long_win_rate": 0.9,
                "short_win_rate": 0.1,
                "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
                "return_semantics": "gross_market_opportunity_before_execution",
                "counterfactual_execution_cost_distribution": {
                    "long": _cost_distribution(),
                    "short": _cost_distribution(),
                    "source_authority": (
                        "shadow_counterfactual_live_microstructure"
                    ),
                },
                "actual_trade_calibration": {
                    "long": _trade_calibration("long"),
                    "short": _trade_calibration("short"),
                    "source_authority": "okx_position_history",
                },
            }
        ],
    }


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="test",
        position_size_pct=0.05,
        suggested_leverage=4.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.08,
        feature_snapshot={
            "current_price": 100.0,
            "bid": 99.99,
            "ask": 100.01,
            "orderbook_bid_depth": 10_000.0,
            "orderbook_ask_depth": 9_000.0,
            "taker_fee_rate": 0.0004,
            "funding_rate": 0.0,
            "funding_data_available": True,
            "funding_interval_minutes": 480.0,
            "funding_rate_observed_at": "2026-08-16T08:00:00+00:00",
        },
        raw_response={
            "ml_signal": _live_ml(),
            "local_ai_tools": {
                "profit_prediction": _live_payload(
                    side="long",
                    long_return=1.0,
                    short_return=-0.4,
                ),
                "time_series_prediction": _live_payload(
                    side="long",
                    long_return=0.6,
                    short_return=-0.1,
                ),
            },
        },
    )


def test_live_models_use_equal_empirical_observations_and_live_cost() -> None:
    decision = _decision()

    score = _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    components = opportunity["expected_net_breakdown"]["components"]
    assert all(component["production_eligible"] for component in components)
    assert opportunity["expected_gross_return_pct"] == pytest.approx(0.8)
    assert opportunity["expected_net_return_pct"] < 0.8
    assert score == pytest.approx(opportunity["return_lcb_pct"])
    assert opportunity["score_policy"] == "standardized_objective_expected_return"
    assert opportunity["return_combination_version"] == PRODUCTION_RETURN_COMBINATION_VERSION
    breakdown = opportunity["expected_net_breakdown"]
    assert breakdown["cost_deduction_count"] == 1
    assert opportunity["expected_net_return_pct"] == pytest.approx(
        opportunity["expected_gross_return_pct"]
        - breakdown["live_execution_cost_pct"]
        - breakdown["current_adverse_funding_cost_pct"]
        - breakdown["authoritative_slippage_tail_excess_pct"]
    )
    assert opportunity["policy_provenance"]["valid_for_seconds"] > 0
    assert opportunity["policy_provenance"]["fallback_reason"] == ""


def test_extreme_payer_funding_is_deducted_without_crediting_receiver_alpha() -> None:
    payer = _decision()
    payer.feature_snapshot.update(
        {
            "funding_rate": 0.15,
            "funding_interval_minutes": 60.0,
        }
    )
    receiver = _decision()
    receiver.feature_snapshot.update(
        {
            "funding_rate": -0.15,
            "funding_interval_minutes": 60.0,
        }
    )

    payer_score = _scorer().score_candidate(payer)
    receiver_score = _scorer().score_candidate(receiver)

    payer_opportunity = payer.raw_response["opportunity_score"]
    receiver_opportunity = receiver.raw_response["opportunity_score"]
    assert payer_opportunity["funding_cost"]["adverse_cost_pct"] == pytest.approx(7.5)
    assert payer_opportunity["expected_net_return_pct"] < 0.0
    assert payer_score < 0.0
    assert receiver_opportunity["funding_cost"]["adverse_cost_pct"] == 0.0
    assert receiver_score == pytest.approx(receiver_opportunity["return_lcb_pct"])


def test_missing_current_funding_evidence_blocks_entry_distribution() -> None:
    decision = _decision()
    decision.feature_snapshot["funding_data_available"] = False

    score = _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    assert score == float("-inf")
    assert opportunity["decision_eligible"] is False
    assert opportunity["funding_cost"]["reason"] == "funding_data_unavailable"
    assert any(
        blocker.startswith("current_funding_cost_incomplete:")
        for blocker in opportunity["return_distribution_contract"]["blockers"]
    )


def test_legacy_paper_canary_cannot_override_current_opportunity_score() -> None:
    decision = _decision()
    decision.raw_response = complete_paper_canary_raw()

    score = _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    assert score == float("-inf")
    assert opportunity["score"] is None
    assert "contract_lifecycle" not in opportunity
    assert opportunity["production_eligible"] is False


def test_diagnostic_win_rate_cannot_change_expected_return_or_score() -> None:
    first = _decision()
    second = deepcopy(first)
    second.raw_response["ml_signal"]["predictions"][0]["long_win_rate"] = 0.01

    first_score = _scorer().score_candidate(first)
    second_score = _scorer().score_candidate(second)

    assert first_score == pytest.approx(second_score)
    assert first.raw_response["opportunity_score"]["expected_net_return_pct"] == pytest.approx(
        second.raw_response["opportunity_score"]["expected_net_return_pct"]
    )


def test_memory_experts_and_ai_confidence_are_observation_only() -> None:
    first = _decision()
    second = deepcopy(first)
    second.confidence = 0.01
    second.raw_response["memory_feedback"] = {
        "probe_permission": True,
        "expected_return_hint_pct": 99.0,
    }
    second.raw_response["experts"] = [
        {"action": "short", "confidence": 1.0} for _ in range(20)
    ]

    first_score = _scorer().score_candidate(first)
    second_score = _scorer().score_candidate(second)

    assert first_score == pytest.approx(second_score)
    observed = second.raw_response["opportunity_score"]["expected_net_breakdown"][
        "observed_not_in_formula"
    ]
    assert observed["memory_feedback"]["expected_return_hint_pct"] == 99.0
    assert len(observed["experts"]) == 20


def test_shadow_timeseries_is_visible_but_cannot_enter_return_distribution() -> None:
    decision = _decision()
    timeseries = decision.raw_response["local_ai_tools"]["time_series_prediction"]
    timeseries["route_mode"] = "shadow"
    timeseries["long_expected_return_pct"] = 1000.0

    _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    component = next(
        item
        for item in opportunity["expected_net_breakdown"]["components"]
        if item["key"] == "timeseries"
    )
    assert component["available"] is True
    assert component["production_eligible"] is False
    assert opportunity["expected_gross_return_pct"] == pytest.approx(0.9)


def test_advisory_ml_cannot_enter_production_return_distribution() -> None:
    decision = _decision()
    decision.raw_response["ml_signal"].update(
        {
            "live_ml_ready": False,
            "advisory_enabled": True,
        }
    )
    decision.raw_response["ml_signal"]["predictions"][0][
        "long_market_expected_return_pct"
    ] = 1000.0

    _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    component = next(
        item
        for item in opportunity["expected_net_breakdown"]["components"]
        if item["key"] == "local_ml"
    )
    assert component["production_eligible"] is False
    assert opportunity["expected_gross_return_pct"] == pytest.approx(0.8)


def test_active_paper_strategy_uses_model_distribution_in_normal_entry_path() -> None:
    decision = _decision()
    decision.raw_response["local_ai_tools"] = {}
    signal = decision.raw_response["ml_signal"]
    signal.update(
        {
            "route_mode": "shadow_observation",
            "live_ml_ready": False,
            "artifact_lifecycle": "canary",
            "model_version": "model-v1",
            "paper_canary_authorized": True,
            "paper_canary": {
                "authorized": True,
                "execution_scope": "paper_only",
                "eligible_sides": ["long"],
            },
        }
    )
    strategy = {
        "execution_mode": "paper",
        "paper_strategy_champion": {
            "active": True,
            "execution_scope": "paper_only",
            "paper_execution_permission": True,
            "live_execution_permission": False,
            "model_version": "model-v1",
            "selector": {"side": "long"},
        },
    }

    score = _scorer().score_candidate(decision, strategy)

    opportunity = decision.raw_response["opportunity_score"]
    component = opportunity["expected_net_breakdown"]["components"][0]
    assert score == pytest.approx(opportunity["return_lcb_pct"])
    assert opportunity["decision_eligible"] is True
    assert opportunity["paper_eligible"] is True
    assert opportunity["production_eligible"] is False
    assert opportunity["return_combination_version"] == PAPER_RETURN_COMBINATION_VERSION
    assert component["decision_eligible"] is True
    assert component["paper_eligible"] is True
    assert component["production_eligible"] is False
    assert component["execution_scope"] == "paper"
    assert component["paper_strategy_authorization"]["eligible"] is True
    assert "paper_bootstrap_canary" not in decision.raw_response


def test_paper_canary_blueprint_allows_side_without_champion() -> None:
    decision = _decision()
    decision.raw_response["local_ai_tools"] = {}
    signal = decision.raw_response["ml_signal"]
    signal.update(
        {
            "route_mode": "shadow_observation",
            "live_ml_ready": False,
            "artifact_lifecycle": "canary",
            "model_version": "model-v1",
            "paper_canary_authorized": True,
            "paper_canary": {
                "authorized": True,
                "execution_scope": "paper_only",
                "eligible_sides": ["long"],
            },
            "prediction_quality": {
                "contract_complete": False,
                "paper_eligible": False,
                "production_eligible": False,
                "anomalous": False,
            },
            "strategy_blueprint": {
                "model_version": "model-v1",
                "paper_execution_eligible": True,
                "paper_bootstrap_sides": ["long"],
                "paper_execution_sides": ["long"],
                "production_eligible_sides": [],
                "live_execution_permission": False,
            },
        }
    )

    score = _scorer().score_candidate(decision, {"execution_mode": "paper"})

    component = decision.raw_response["opportunity_score"]["expected_net_breakdown"][
        "components"
    ][0]
    assert score == pytest.approx(decision.raw_response["opportunity_score"]["return_lcb_pct"])
    assert component["decision_eligible"] is True
    assert component["paper_eligible"] is True
    assert component["model_strategy_side_authorization"]["reason"] == (
        "paper_bootstrap_direction_authorized"
    )


def test_model_blueprint_blocks_owned_short_but_not_independent_server_candidate() -> None:
    decision = _decision()
    decision.action = Action.SHORT
    decision.raw_response["ml_signal"] = _paper_payload(
        side="short",
        long_return=0.2,
        short_return=1.2,
    )
    decision.raw_response["ml_signal"]["strategy_blueprint"] = {
        "version": "blueprint-v1",
        "model_version": "model-v1",
        "paper_execution_eligible": True,
        "live_execution_permission": False,
        "eligible_sides": ["long"],
    }
    decision.raw_response["local_ai_tools"] = {
        "profit_prediction": _paper_payload(
            side="short",
            long_return=0.1,
            short_return=2.0,
        )
    }

    score = _scorer().score_candidate(decision, {"execution_mode": "paper"})

    opportunity = decision.raw_response["opportunity_score"]
    assert not isinf(score)
    assert opportunity["decision_eligible"] is True
    assert opportunity["paper_eligible"] is True
    components = {
        item["key"]: item
        for item in opportunity["expected_net_breakdown"]["components"]
    }
    assert components["local_ml"]["decision_eligible"] is False
    assert components["local_ml"]["eligibility_reason"] == (
        "direction_not_authorized_by_model_blueprint"
    )
    assert components["server_profit"]["decision_eligible"] is True
    assert components["server_profit"]["model_strategy_side_authorization"][
        "enforced"
    ] is False


def _paper_payload(*, side: str, long_return: float, short_return: float) -> dict:
    payload = _live_payload(
        side=side,
        long_return=long_return,
        short_return=short_return,
    )
    payload.update(
        {
            "route_mode": "paper_analysis",
            "live_ml_ready": False,
            "production_permission": False,
            "prediction_quality": {
                "contract_complete": True,
                "paper_eligible": True,
                "production_eligible": False,
                "anomalous": False,
                "production_blockers": [
                    "artifact_activation_not_production_authorized"
                ],
            },
        }
    )
    payload.pop("counterfactual_execution_cost_distribution", None)
    payload.pop("actual_trade_calibration", None)
    return payload


def test_shadow_server_models_participate_in_normal_paper_scoring() -> None:
    decision = _decision()
    decision.raw_response["ml_signal"] = {}
    decision.raw_response["local_ai_tools"] = {
        "profit_prediction": _paper_payload(
            side="long",
            long_return=1.0,
            short_return=-0.4,
        ),
        "time_series_prediction": _paper_payload(
            side="long",
            long_return=0.6,
            short_return=-0.1,
        ),
    }

    score = _scorer().score_candidate(decision, {"execution_mode": "paper"})

    opportunity = decision.raw_response["opportunity_score"]
    assert score == pytest.approx(opportunity["return_lcb_pct"])
    assert opportunity["decision_eligible"] is True
    assert opportunity["paper_eligible"] is True
    assert opportunity["production_eligible"] is False
    assert opportunity["expected_gross_return_pct"] == pytest.approx(0.8)
    assert opportunity["expected_net_breakdown"][
        "counterfactual_cost_distribution_count"
    ] == 0
    assert opportunity["expected_net_breakdown"][
        "authoritative_trade_calibration_count"
    ] == 0
    components = opportunity["expected_net_breakdown"]["components"]
    assert all(component["paper_eligible"] is True for component in components[1:])
    assert all(component["included_in_return_distribution"] is True for component in components[1:])


def test_paper_scoring_uses_only_the_authorized_prediction_horizon() -> None:
    decision = _decision()
    decision.raw_response["ml_signal"] = {}
    profit = _paper_payload(side="long", long_return=0.5, short_return=-0.2)
    timeseries = _paper_payload(side="long", long_return=1.2, short_return=-0.4)
    for side in ("long", "short"):
        timeseries["return_distribution_contract"][side]["horizon_minutes"] = 60
    timeseries["horizon_minutes"] = 60
    decision.raw_response["local_ai_tools"] = {
        "profit_prediction": profit,
        "time_series_prediction": timeseries,
    }
    decision.raw_response["normal_paper_trade"] = {
        "prediction_horizon_minutes": 30,
    }

    score = _scorer().score_candidate(decision, {"execution_mode": "paper"})

    opportunity = decision.raw_response["opportunity_score"]
    components = {
        item["key"]: item
        for item in opportunity["expected_net_breakdown"]["components"]
    }
    assert score == pytest.approx(opportunity["return_lcb_pct"])
    assert opportunity["paper_eligible"] is True
    assert opportunity["return_distribution_contract"]["gross_market_distribution"][
        "model_count"
    ] == 1
    assert components["server_profit"]["included_in_return_distribution"] is True
    assert components["timeseries"]["included_in_return_distribution"] is False
    assert components["timeseries"]["observation_only"] is True
    assert components["timeseries"]["eligibility_reason"] == (
        "paper_prediction_horizon_not_selected"
    )


def test_paper_scoring_auto_selects_coherent_horizon_before_trade_contract() -> None:
    decision = _decision()
    decision.raw_response["ml_signal"] = {}
    profit = _paper_payload(side="long", long_return=0.2, short_return=-0.1)
    timeseries = _paper_payload(side="long", long_return=0.7, short_return=-0.3)
    for side in ("long", "short"):
        profit["return_distribution_contract"][side]["horizon_minutes"] = 10
        timeseries["return_distribution_contract"][side]["horizon_minutes"] = 5
    decision.raw_response["local_ai_tools"] = {
        "profit_prediction": profit,
        "time_series_prediction": timeseries,
    }

    score = _scorer().score_candidate(decision, {"execution_mode": "paper"})

    opportunity = decision.raw_response["opportunity_score"]
    components = {
        item["key"]: item
        for item in opportunity["expected_net_breakdown"]["components"]
    }
    assert score == pytest.approx(opportunity["return_lcb_pct"])
    assert opportunity["paper_eligible"] is True
    assert opportunity["selected_horizon_minutes"] == 5
    assert opportunity["horizon_cohort_selection"]["selected_sources"] == ["timeseries"]
    assert components["timeseries"]["included_in_return_distribution"] is True
    assert components["server_profit"]["included_in_return_distribution"] is False
    assert components["server_profit"]["decision_eligible"] is False
    assert components["server_profit"]["eligibility_reason"] == (
        "paper_prediction_horizon_not_selected"
    )


def test_paper_scoring_horizon_selection_uses_continuous_model_weights() -> None:
    decision = _decision()
    decision.raw_response["ml_signal"] = {}
    profit = _paper_payload(side="long", long_return=0.2, short_return=-0.1)
    timeseries = _paper_payload(side="long", long_return=0.7, short_return=-0.3)
    for side in ("long", "short"):
        profit["return_distribution_contract"][side]["horizon_minutes"] = 10
        timeseries["return_distribution_contract"][side]["horizon_minutes"] = 5
    decision.raw_response["local_ai_tools"] = {
        "profit_prediction": profit,
        "time_series_prediction": timeseries,
    }
    strategy = {
        "execution_mode": "paper",
        "continuous_model_weights": {
            "applied": True,
            "quant_source_weights": {
                "server_profit": {"effective_multiplier": 1.4},
                "timeseries": {"effective_multiplier": 0.1},
            },
        },
    }

    score = _scorer().score_candidate(decision, strategy)

    opportunity = decision.raw_response["opportunity_score"]
    assert score == pytest.approx(opportunity["return_lcb_pct"])
    assert opportunity["paper_eligible"] is True
    assert opportunity["selected_horizon_minutes"] == 10
    assert opportunity["horizon_cohort_selection"]["selected_sources"] == [
        "server_profit"
    ]


def test_paper_scoring_keeps_direction_competition_selected_native_horizon() -> None:
    decision = _decision()
    decision.raw_response["ml_signal"] = {}
    profit = _paper_payload(side="short", long_return=-0.2, short_return=1.2)
    timeseries = _paper_payload(side="long", long_return=0.5, short_return=-0.2)
    for side in ("long", "short"):
        profit["return_distribution_contract"][side]["horizon_minutes"] = 240
        timeseries["return_distribution_contract"][side]["horizon_minutes"] = 5
    decision.raw_response["local_ai_tools"] = {
        "profit_prediction": profit,
        "time_series_prediction": timeseries,
    }
    decision.raw_response["direction_competition"] = {
        "authorized_prediction_horizon_minutes": 5,
        "selected_horizon_minutes": 240,
    }
    strategy = {
        "execution_mode": "paper",
        "continuous_model_weights": {
            "applied": True,
            "quant_source_weights": {
                "server_profit": {"effective_multiplier": 1.4},
                "timeseries": {"effective_multiplier": 0.1},
            },
        },
    }

    _scorer().score_candidate(decision, strategy)

    opportunity = decision.raw_response["opportunity_score"]
    assert opportunity["selected_horizon_minutes"] == 240
    assert opportunity["horizon_cohort_selection"]["selected_sources"] == [
        "server_profit"
    ]


def test_shadow_server_models_remain_blocked_in_live_scoring() -> None:
    decision = _decision()
    decision.raw_response["ml_signal"] = {}
    decision.raw_response["local_ai_tools"] = {
        "profit_prediction": _paper_payload(
            side="long",
            long_return=1.0,
            short_return=-0.4,
        ),
        "time_series_prediction": _paper_payload(
            side="long",
            long_return=0.6,
            short_return=-0.1,
        ),
    }

    score = _scorer().score_candidate(decision, {"execution_mode": "live"})

    opportunity = decision.raw_response["opportunity_score"]
    assert isinf(score) and score < 0
    assert opportunity["decision_eligible"] is False
    assert opportunity["production_eligible"] is False
    assert all(
        component["production_eligible"] is False
        for component in opportunity["expected_net_breakdown"]["components"]
    )


def test_paper_scoring_rejects_structurally_invalid_shadow_distribution() -> None:
    decision = _decision()
    decision.raw_response["ml_signal"] = {}
    payload = _paper_payload(side="long", long_return=1.0, short_return=-0.4)
    payload["return_distribution_contract"]["long"][
        "lower_quantile_return_pct"
    ] = 2.0
    decision.raw_response["local_ai_tools"] = {"profit_prediction": payload}

    score = _scorer().score_candidate(decision, {"execution_mode": "paper"})

    opportunity = decision.raw_response["opportunity_score"]
    assert isinf(score) and score < 0
    assert opportunity["paper_eligible"] is False
    assert "lower_quantile_above_raw_expected" in opportunity[
        "return_distribution_contract"
    ]["blockers"]


def test_paper_permission_does_not_depend_on_historical_profitability() -> None:
    first = _decision()
    first.raw_response["ml_signal"] = {}
    payload = _paper_payload(side="long", long_return=1.0, short_return=-0.4)
    payload["actual_trade_calibration"] = {
        "long": {
            **_trade_calibration("long"),
            "net_return_after_all_cost_pct": {
                "count": 200,
                "expected": -5.0,
                "lower_hinge": -8.0,
            },
        }
    }
    first.raw_response["local_ai_tools"] = {"profit_prediction": payload}
    second = deepcopy(first)
    second.raw_response["local_ai_tools"]["profit_prediction"].pop(
        "actual_trade_calibration"
    )

    first_score = _scorer().score_candidate(first, {"execution_mode": "paper"})
    second_score = _scorer().score_candidate(second, {"execution_mode": "paper"})

    assert first_score == pytest.approx(second_score)
    assert first.raw_response["opportunity_score"]["paper_eligible"] is True
    assert second.raw_response["opportunity_score"]["paper_eligible"] is True


def test_paper_strategy_cannot_authorize_same_model_in_live_mode() -> None:
    decision = _decision()
    decision.raw_response["local_ai_tools"] = {}
    signal = decision.raw_response["ml_signal"]
    signal.update(
        {
            "route_mode": "shadow_observation",
            "live_ml_ready": False,
            "artifact_lifecycle": "canary",
            "model_version": "model-v1",
            "paper_canary_authorized": True,
            "paper_canary": {
                "authorized": True,
                "execution_scope": "paper_only",
                "eligible_sides": ["long"],
            },
        }
    )
    strategy = {
        "execution_mode": "live",
        "paper_strategy_champion": {
            "active": True,
            "execution_scope": "paper_only",
            "paper_execution_permission": True,
            "live_execution_permission": False,
            "model_version": "model-v1",
            "selector": {"side": "long"},
        },
    }

    score = _scorer().score_candidate(decision, strategy)

    opportunity = decision.raw_response["opportunity_score"]
    component = opportunity["expected_net_breakdown"]["components"][0]
    assert isinf(score) and score < 0
    assert opportunity["production_eligible"] is False
    assert component["production_eligible"] is False
    assert component["paper_strategy_authorization"]["eligible"] is False


def test_runtime_recovery_predictions_have_zero_production_weight() -> None:
    decision = _decision()
    decision.raw_response["ml_signal"].update(
        {
            "live_ml_ready": False,
            "trained_sample_count": 100,
            "model_version": "2026-07-13T12:00:00+00:00",
            "readiness": {
                "live_ml_ready": False,
                "blocking_reasons": [
                    {"code": "long_top_return_lcb_not_positive"},
                ],
                "policy_provenance": {
                    "sample_count": 100,
                    "test_sample_count": 20,
                    "fallback_reason": "",
                },
            },
        }
    )
    for payload in decision.raw_response["local_ai_tools"].values():
        payload.update(
            {
                "route_mode": "shadow_candidate",
                "trained": True,
                "artifact_persisted": True,
                "training_cost_policy": "per_sample_live_spread_fee_and_funding_complete",
                "label_name": "net_return_after_all_cost_pct",
                "label_version": RETURN_OBJECTIVE_VERSION,
            }
        )

    score = _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    components = opportunity["expected_net_breakdown"]["components"]
    assert opportunity["return_distribution_mode"] == "unavailable"
    assert opportunity["production_eligible"] is False
    assert opportunity["return_lcb_pct"] is None
    assert isinf(score) and score < 0
    assert all(component["production_eligible"] is False for component in components)
    assert all(component["included_in_return_distribution"] is False for component in components)
    assert all(component["production_weight"] == 0.0 for component in components)
    assert opportunity["policy_provenance"]["fallback_reason"]


def test_observation_only_anomalous_server_prediction_cannot_enter_distribution() -> None:
    decision = _decision()
    decision.raw_response["ml_signal"] = {}
    for payload in decision.raw_response["local_ai_tools"].values():
        payload.update(
            {
                "route_mode": "shadow_candidate",
                "promotion_ready": False,
                "trained": True,
                "prediction_quality": {
                    "production_eligible": False,
                    "anomalous": True,
                    "reason": "outside_dynamic_rolling_forecast_interval",
                },
            }
        )

    _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    assert opportunity["return_distribution_mode"] == "unavailable"
    assert opportunity["production_eligible"] is False
    assert all(
        component["included_in_return_distribution"] is False
        for component in opportunity["expected_net_breakdown"]["components"]
    )


def test_missing_authoritative_slippage_distribution_blocks_production() -> None:
    decision = _decision()
    for payload in decision.raw_response["local_ai_tools"].values():
        payload["actual_trade_calibration"]["long"]["slippage_pct"] = {
            "count": 0,
            "expected": None,
            "upper_hinge": None,
        }
    decision.raw_response["ml_signal"]["predictions"][0][
        "actual_trade_calibration"
    ]["long"]["slippage_pct"] = {
        "count": 0,
        "expected": None,
        "upper_hinge": None,
    }

    score = _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    assert opportunity["production_eligible"] is False
    assert opportunity["expected_net_return_pct"] is None
    assert "authoritative_realized_return_or_slippage_distribution_missing" in (
        opportunity["policy_provenance"]["fallback_reason"]
    )
    assert isinf(score) and score < 0


def test_mismatched_symbol_trade_calibration_blocks_production() -> None:
    decision = _decision()
    for payload in decision.raw_response["local_ai_tools"].values():
        payload["actual_trade_calibration"]["long"]["symbol"] = "ETH/USDT"
    decision.raw_response["ml_signal"]["predictions"][0][
        "actual_trade_calibration"
    ]["long"]["symbol"] = "ETH/USDT"

    score = _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    assert opportunity["production_eligible"] is False
    assert opportunity["expected_net_breakdown"][
        "authoritative_trade_calibration_count"
    ] == 0
    assert isinf(score) and score < 0


def test_degenerate_counterfactual_cost_distribution_blocks_production() -> None:
    decision = _decision()
    for payload in decision.raw_response["local_ai_tools"].values():
        payload["counterfactual_execution_cost_distribution"]["long"][
            "distribution_ready"
        ] = False
    decision.raw_response["ml_signal"]["predictions"][0][
        "counterfactual_execution_cost_distribution"
    ]["long"]["distribution_ready"] = False

    score = _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    assert opportunity["production_eligible"] is False
    assert opportunity["expected_net_breakdown"][
        "counterfactual_cost_distribution_count"
    ] == 0
    assert isinf(score) and score < 0


def test_missing_live_spread_fails_closed_without_cost_fallback() -> None:
    decision = _decision()
    decision.feature_snapshot = {"current_price": 100.0}

    _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    assert opportunity["production_eligible"] is False
    assert opportunity["execution_cost"]["production_eligible"] is False
    assert opportunity["policy_provenance"]["fallback_reason"]


def test_missing_production_return_models_fails_closed() -> None:
    decision = _decision()
    decision.raw_response = {
        "ml_signal": {"available": True, "advisory_enabled": True},
        "memory_feedback": {"expected_return_hint_pct": 50.0},
    }

    score = _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    assert score <= 0.0
    assert opportunity["production_eligible"] is False
    assert opportunity["expected_net_return_pct"] is None
    assert opportunity["policy_provenance"]["sample_count"] == 0


def test_icp_lower_quantile_above_point_blocks_entire_production_distribution() -> None:
    decision = _decision()
    decision.symbol = "ICP/USDT"
    decision.raw_response["ml_signal"] = {}
    profit = decision.raw_response["local_ai_tools"]["profit_prediction"]
    timeseries = decision.raw_response["local_ai_tools"]["time_series_prediction"]
    timeseries["route_mode"] = "shadow_observation"
    contract = profit["return_distribution_contract"]["long"]
    contract["raw_expected_return_pct"] = 0.46
    contract["lower_quantile_return_pct"] = 0.496
    contract["production_eligible"] = True
    contract["blockers"] = []

    score = _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    assert isinf(score) and score < 0
    assert opportunity["production_eligible"] is False
    assert "lower_quantile_above_raw_expected" in opportunity[
        "return_distribution_contract"
    ]["blockers"]
    assert all(
        component["production_weight"] == 0.0
        for component in opportunity["expected_net_breakdown"]["components"]
    )


def test_doge_single_governed_source_keeps_model_distribution_uncertainty() -> None:
    decision = _decision()
    decision.symbol = "DOGE/USDT"
    decision.raw_response["ml_signal"] = {}
    decision.raw_response["local_ai_tools"]["time_series_prediction"][
        "route_mode"
    ] = "shadow_observation"
    profit = decision.raw_response["local_ai_tools"]["profit_prediction"]
    for side in ("long", "short"):
        profit["actual_trade_calibration"][side]["symbol"] = "DOGE/USDT"

    score = _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    contract = opportunity["return_distribution_contract"]
    assert score == pytest.approx(opportunity["return_lcb_pct"])
    assert opportunity["production_eligible"] is True
    assert contract["gross_market_distribution"]["model_count"] == 1
    assert contract["gross_market_distribution"]["dispersion_pct"] > 0
    assert opportunity["return_uncertainty_pct"] > 0


def test_mismatched_model_horizons_make_all_sources_observation_only() -> None:
    decision = _decision()
    timeseries = decision.raw_response["local_ai_tools"]["time_series_prediction"]
    timeseries["return_distribution_contract"]["long"]["horizon_minutes"] = 60

    score = _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    assert isinf(score) and score < 0
    assert opportunity["production_eligible"] is False
    assert "model_distribution_horizon_minutes_mismatch" in opportunity[
        "return_distribution_contract"
    ]["blockers"]
    assert all(
        component["aggregate_eligible"] is False
        for component in opportunity["expected_net_breakdown"]["components"]
    )
    assert all(
        component["included_in_return_distribution"] is False
        for component in opportunity["expected_net_breakdown"]["components"]
    )
