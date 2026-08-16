from copy import deepcopy
from types import SimpleNamespace

import pytest

from services.entry_direction_competition import EntryDirectionCompetitionPolicy
from services.entry_direction_support import assess_paper_model_trade_support
from services.profit_supervision import PROFIT_SUPERVISION_VERSION
from services.return_objective import (
    COST_MODEL_VERSION,
    RETURN_DISTRIBUTION_CONTRACT_VERSION,
    RETURN_LABEL_NAME,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
    standardized_return_distribution,
)


def _distribution(side: str, expected: float, *, horizon_minutes: int = 30) -> dict:
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
        cost_model_version=COST_MODEL_VERSION,
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
    )


def _governed_payload(long_return: float, short_return: float) -> dict:
    return {
        "available": True,
        "route_mode": "live",
        "live_ml_ready": True,
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
        "best_side": "long" if long_return >= short_return else "short",
        "return_distribution_contract_version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
        "return_distribution_contract": {
            "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
            "long": _distribution("long", long_return),
            "short": _distribution("short", short_return),
        },
        "long_market_expected_return_pct": long_return,
        "short_market_expected_return_pct": short_return,
    }


def _paper_payload(long_return: float, short_return: float) -> dict:
    payload = _governed_payload(long_return, short_return)
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
    return payload


def _governed_ml(long_return: float, short_return: float) -> dict:
    return {
        **_governed_payload(long_return, short_return),
        "primary_horizon_minutes": 30,
        "live_ml_ready": True,
        "influence_policy": {
            "long": {"enabled": True},
            "short": {"enabled": True},
        },
        "predictions": [
            {
                "best_side": "long" if long_return >= short_return else "short",
                "return_distribution_contract_version": (
                    RETURN_DISTRIBUTION_CONTRACT_VERSION
                ),
                "return_distribution_contract": {
                    "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
                    "long": _distribution("long", long_return),
                    "short": _distribution("short", short_return),
                },
                "long_market_expected_return_pct": long_return,
                "short_market_expected_return_pct": short_return,
            }
        ],
    }


def _context(*, ml=None, tools=None, feature=None, market=None, strategy=None) -> dict:
    return EntryDirectionCompetitionPolicy().context(
        feature or SimpleNamespace(),
        ml,
        tools,
        market,
        strategy,
    )


def test_only_governed_return_models_choose_observed_side() -> None:
    context = _context(
        ml=_governed_ml(0.8, -0.2),
        tools={"profit_prediction": _governed_payload(0.6, -0.1)},
    )

    assert context["preferred_side"] == "long"
    assert context["long"]["raw_expected_return_pct"] == 0.7
    assert context["long"]["objective_expected_return_pct"] == pytest.approx(0.58)
    assert context["short"]["raw_expected_return_pct"] == pytest.approx(-0.15)
    assert context["short"]["objective_expected_return_pct"] == pytest.approx(-0.28)
    assert context["production_source_count"] == 4
    assert context["production_permission"] is False
    assert context["policy"] == (
        "execution_scoped_gross_market_observation_only_no_fixed_gap"
    )


def test_missing_governance_cannot_enter_direction_scores() -> None:
    context = _context(
        ml={
            "predictions": [
                {
                    "long_expected_return_pct": 1000.0,
                    "short_expected_return_pct": -1000.0,
                }
            ]
        },
        tools={"profit_prediction": {"long_expected_return_pct": 1000.0}},
    )

    assert context["preferred_side"] == "neutral"
    assert context["production_source_count"] == 0
    assert context["policy_provenance"]["fallback_reason"]


def test_shadow_model_cannot_enter_direction_scores() -> None:
    payload = _governed_payload(1000.0, -1000.0)
    payload["route_mode"] = "shadow"

    context = _context(tools={"profit_prediction": payload})

    assert context["preferred_side"] == "neutral"
    assert context["production_source_count"] == 0


def test_shadow_models_choose_direction_in_paper_scope_without_live_permission() -> None:
    context = _context(
        tools={"profit_prediction": _paper_payload(0.7, -0.2)},
        strategy={"execution_mode": "paper"},
    )

    assert context["execution_scope"] == "paper"
    assert context["preferred_side"] == "long"
    assert context["decision_source_count"] == 2
    assert context["paper_source_count"] == 2
    assert context["production_source_count"] == 0
    assert all(
        item["paper_eligible"] is True
        and item["production_eligible"] is False
        for side in ("long", "short")
        for item in context[side]["evidence"]
        if item["source"] == "server_profit"
    )


def test_structurally_invalid_shadow_model_cannot_choose_paper_direction() -> None:
    payload = _paper_payload(0.7, -0.2)
    payload["return_distribution_contract"]["long"][
        "lower_quantile_return_pct"
    ] = 1.0

    context = _context(
        tools={"profit_prediction": payload},
        strategy={"execution_mode": "paper"},
    )

    assert context["preferred_side"] == "short"
    assert context["long"]["decision_source_count"] == 0
    assert context["short"]["decision_source_count"] == 1


def test_negative_shadow_scores_do_not_create_an_intervention_direction() -> None:
    payload = _governed_ml(-0.4, -0.1)
    payload.update(
        {
            "route_mode": "shadow_observation",
            "live_ml_ready": False,
            "promotion_ready": False,
        }
    )

    context = _context(ml=payload)

    assert context["preferred_side"] == "neutral"
    assert context["production_source_count"] == 0
    assert context["training_preferred_side"] == "neutral"
    assert context["training_short"]["observation_count"] == 1
    assert context["training_short"]["horizon_minutes"] == 30
    assert context["training_short"]["horizon_source_count"] == 1
    assert context["training_permission"] is False


def test_diagnostic_win_rate_cannot_change_direction_scores() -> None:
    first_ml = _governed_ml(0.5, 0.2)
    second_ml = deepcopy(first_ml)
    first_ml["predictions"][0]["long_win_rate"] = 0.99
    second_ml["predictions"][0]["long_win_rate"] = 0.01

    first = _context(ml=first_ml)
    second = _context(ml=second_ml)

    assert first["long"]["score"] == second["long"]["score"]
    assert first["preferred_side"] == second["preferred_side"]


def test_features_regime_and_strategy_weights_are_observation_excluded() -> None:
    ml = _governed_ml(0.4, 0.6)
    first = _context(
        ml=ml,
        feature=SimpleNamespace(adx_14=99.0, returns_5=10.0),
        market={"mode": "uptrend"},
        strategy={"side_weights": {"long": 999.0}, "blocked_directions": ["short"]},
    )
    second = _context(
        ml=ml,
        feature=SimpleNamespace(adx_14=0.0, returns_5=-10.0),
        market={"mode": "downtrend"},
        strategy={"side_weights": {"short": 999.0}, "blocked_directions": ["long"]},
    )

    assert first["preferred_side"] == "short"
    assert first["long"]["score"] == second["long"]["score"]
    assert first["short"]["score"] == second["short"]["score"]


def test_paper_continuous_weights_can_change_direction_without_affecting_live() -> None:
    ml = _governed_ml(0.6, 0.0)
    tools = {"profit_prediction": _governed_payload(0.0, 0.5)}
    equal = _context(ml=ml, tools=tools)
    weighted = _context(
        ml=ml,
        tools=tools,
        strategy={
            "execution_mode": "paper",
            "continuous_model_weights": {
                "applied": True,
                "quant_source_weights": {
                    "local_ml": {"effective_multiplier": 0.1},
                    "server_profit": {"effective_multiplier": 1.4},
                },
            },
        },
    )
    live = _context(
        ml=ml,
        tools=tools,
        strategy={
            "execution_mode": "live",
            "continuous_model_weights": {
                "applied": True,
                "quant_source_weights": {
                    "local_ml": {"effective_multiplier": 0.1},
                    "server_profit": {"effective_multiplier": 1.4},
                },
            },
        },
    )

    assert equal["preferred_side"] == "long"
    assert weighted["preferred_side"] == "short"
    assert weighted["continuous_model_weighting"]["applied"] is True
    assert live == equal


def test_paper_mixed_horizons_select_highest_weight_native_cohort() -> None:
    server = _paper_payload(0.1, 0.7)
    for side in ("long", "short"):
        server["return_distribution_contract"][side]["horizon_minutes"] = 10

    context = _context(
        ml=_governed_ml(0.8, -0.2),
        tools={"profit_prediction": server},
        strategy={"execution_mode": "paper"},
    )

    assert context["enabled"] is True
    assert context["preferred_side"] == "short"
    assert context["selected_horizon_minutes"] == 10
    assert context["decision_source_count"] == 2
    assert context["aggregate_blockers"] == []
    assert context["horizon_cohort_selection"]["selected_sources"] == ["server_profit"]
    server_rows = [
        item
        for side in ("long", "short")
        for item in context[side]["evidence"]
        if item["source"] == "local_ml"
    ]
    assert all(item["decision_eligible"] is False for item in server_rows)
    assert all(
        item["eligibility_reason"] == "paper_prediction_horizon_not_selected"
        for item in server_rows
    )


def test_high_weight_long_horizon_selects_native_model_cohort() -> None:
    server = _paper_payload(0.0, 0.7)
    for side in ("long", "short"):
        server["return_distribution_contract"][side]["horizon_minutes"] = 60

    context = _context(
        ml=_governed_ml(0.8, -0.2),
        tools={"profit_prediction": server},
        strategy={
            "execution_mode": "paper",
            "continuous_model_weights": {
                "applied": True,
                "quant_source_weights": {
                    "local_ml": {"effective_multiplier": 0.1},
                    "server_profit": {"effective_multiplier": 1.4},
                },
            },
        },
    )

    assert context["enabled"] is True
    assert context["preferred_side"] == "short"
    assert context["selected_horizon_minutes"] == 60
    assert context["training_long"]["horizon_minutes"] == 60
    assert context["training_short"]["horizon_minutes"] == 60
    assert context["horizon_cohort_selection"]["selected_sources"] == ["server_profit"]
    assert context["authorized_prediction_horizon_minutes"] == 30
    assert context["authorized_prediction_horizon_source"] == "local_ml_primary_horizon"


def test_unavailable_local_ml_horizon_does_not_hide_native_tool_cohort() -> None:
    ml = _governed_ml(0.8, -0.2)
    ml["primary_horizon_minutes"] = 5
    server = _paper_payload(0.0, 0.7)
    for side in ("long", "short"):
        server["return_distribution_contract"][side]["horizon_minutes"] = 240

    context = _context(
        ml=ml,
        tools={"profit_prediction": server},
        strategy={
            "execution_mode": "paper",
            "continuous_model_weights": {
                "applied": True,
                "quant_source_weights": {
                    "local_ml": {"effective_multiplier": 0.1},
                    "server_profit": {"effective_multiplier": 1.0},
                },
            },
        },
    )

    assert context["preferred_side"] == "short"
    assert context["decision_source_count"] == 2
    assert context["selected_horizon_minutes"] == 240
    assert context["aggregate_blockers"] == []


def test_model_blueprint_authority_blocks_unauthorized_short_observation() -> None:
    signal = _paper_payload(0.1, 0.2)
    signal["strategy_blueprint"] = {
        "version": "blueprint-v1",
        "model_version": "model-v1",
        "paper_execution_eligible": True,
        "live_execution_permission": False,
        "eligible_sides": ["long"],
    }
    tools = {
        "profit_prediction": _paper_payload(0.1, 1.5),
    }

    context = _context(
        ml=signal,
        tools=tools,
        strategy={"execution_mode": "paper"},
    )

    assert context["preferred_side"] == "long"
    assert context["model_strategy_direction_authorization"]["short"][
        "reason"
    ] == "direction_not_authorized_by_model_blueprint"
    short_evidence = context["short"]["evidence"]
    assert short_evidence
    assert all(item["decision_eligible"] is False for item in short_evidence)
    assert all(item["paper_eligible"] is False for item in short_evidence)
    assert all(item["observation_only"] is True for item in short_evidence)


def test_authorized_long_can_compare_against_unauthorized_short_observation() -> None:
    signal = _paper_payload(1.2, 0.3)
    signal["strategy_blueprint"] = {
        "version": "blueprint-v1",
        "model_version": "model-v1",
        "paper_execution_eligible": True,
        "live_execution_permission": False,
        "eligible_sides": ["long"],
    }

    context = _context(ml=signal, strategy={"execution_mode": "paper"})
    support = assess_paper_model_trade_support(
        context,
        [],
        "long",
        execution_cost_pct=0.1,
    )

    assert context["preferred_side"] == "long"
    assert support["eligible"] is True
    assert support["quant_evidence_families"] == ["local_ml"]
    assert context["short"]["evidence"][0]["decision_eligible"] is False
    assert context["short"]["evidence"][0]["direction_comparison_eligible"] is True


def test_mismatched_horizons_cannot_enter_direction_aggregation() -> None:
    payload = _governed_payload(0.6, -0.1)
    for side in ("long", "short"):
        payload["return_distribution_contract"][side]["horizon_minutes"] = 60

    context = _context(
        ml=_governed_ml(0.8, -0.2),
        tools={"profit_prediction": payload},
    )

    assert context["preferred_side"] == "neutral"
    assert context["production_source_count"] == 0
    assert "direction_competition_horizon_minutes_mismatch" in context[
        "aggregate_blockers"
    ]
