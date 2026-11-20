from copy import deepcopy
from types import SimpleNamespace

import pytest

from services.entry_direction_competition import (
    EntryDirectionCompetitionPolicy,
    _enforce_aggregate_contract_consistency,
    _paper_quant_quality_permissions,
)
from services.entry_direction_support import assess_paper_model_trade_support
from services.ml_prediction_contract import ENTRY_COHORT_VERSION, PREDICTION_SCORE_POLICY_VERSION
from services.model_strategy_blueprint import build_model_strategy_blueprint
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


@pytest.mark.parametrize("current_version", ["current", "unseen", ""])
def test_quality_permission_uses_only_the_current_model_version(current_version):
    permissions = _paper_quant_quality_permissions(
        {
            "execution_mode": "paper",
            "continuous_model_weights": {
                "applied": True,
                "quant_source_weights": {
                    "local_ml": {
                        "paper_execution_permission": True,
                        "model_quality_permissions": {
                            "replaced": {"paper_execution_permission": True},
                            "current": {
                                "paper_execution_permission": False,
                                "paper_execution_reason": "current_model_fee_after_evidence_missing",
                                "paper_execution_model_version": "current",
                            },
                        },
                    },
                },
            },
        },
        {"model_version": current_version},
    )
    assert permissions["local_ml"]["paper_execution_permission"] is False
    assert permissions["local_ml"]["paper_execution_model_version"] == (current_version or None)


@pytest.mark.parametrize("cohort_version", [None, "legacy-cohort", ENTRY_COHORT_VERSION])
@pytest.mark.parametrize("score_policy", [None, "legacy-score-policy", PREDICTION_SCORE_POLICY_VERSION])
def test_quality_permission_rejects_cached_model_permission_from_old_sample_cohort(cohort_version, score_policy):
    permissions = _paper_quant_quality_permissions(
        {
            "execution_mode": "paper",
            "continuous_model_weights": {
                "applied": True,
                "quant_source_weights": {"local_ml": {
                    "paper_execution_permission": True,
                    "model_quality_permissions": {"current": {
                        "paper_execution_permission": True,
                        "paper_execution_model_version": "current",
                        "entry_cohort_version": cohort_version,
                        "prediction_score_policy_version": score_policy,
                    }},
                }},
            },
        },
        {"model_version": "current"},
    )
    assert permissions["local_ml"]["paper_execution_permission"] is (
        cohort_version == ENTRY_COHORT_VERSION and score_policy == PREDICTION_SCORE_POLICY_VERSION
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
        "long_loss_probability": 0.2,
        "short_loss_probability": 0.3,
    }


def _paper_payload(long_return: float, short_return: float) -> dict:
    payload = _governed_payload(long_return, short_return)
    payload["predictions"] = [{
        "horizon_minutes": 30,
        "fee_after_side_selection": {
            "return_semantics": "risk_adjusted_net_after_counterfactual_execution_cost",
            "long": {"objective_net_return_pct": long_return},
            "short": {"objective_net_return_pct": short_return},
        },
    }]
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
                "horizon_minutes": 30,
                "fee_after_side_selection": {
                    "return_semantics": "risk_adjusted_net_after_counterfactual_execution_cost",
                    "long": {"objective_net_return_pct": long_return},
                    "short": {"objective_net_return_pct": short_return},
                },
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


@pytest.mark.parametrize("selected", ["long", "short"])
@pytest.mark.parametrize("score", [-0.8, 0.0, None, float("nan"), 0.15])
def test_paper_ml_requires_a_positive_frozen_fee_after_score(selected, score):
    signal = _governed_ml(0.8, 0.6)
    primary = signal["predictions"][0]
    primary["horizon_minutes"] = 30
    primary["fee_after_side_selection"] = {
        "return_semantics": "risk_adjusted_net_after_counterfactual_execution_cost",
        **{side: {"objective_net_return_pct": score if side == selected else -1.2}
           for side in ("long", "short")},
    }
    context = _context(ml=signal, strategy={"execution_mode": "paper"})
    for side in ("long", "short"):
        row = next(i for i in context[side]["evidence"] if i["source"] == "local_ml")
        allowed = side == selected and score == 0.15
        assert row["paper_eligible"] is allowed
        if not allowed:
            assert row["decision_eligible"] is False
            assert row["direction_comparison_eligible"] is True
            assert row["eligibility_reason"].startswith("model_fee_after_objective_")


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


def test_signed_funding_cashflow_participates_in_long_short_comparison() -> None:
    context = _context(
        ml=_governed_ml(0.6, 0.5),
        feature={
            "timestamp": "2026-08-17T00:00:00+00:00",
            "funding_data_available": True,
            "funding_rate": 0.01,
            "funding_interval_minutes": 480,
            "next_funding_time": "2026-08-17T00:15:00+00:00",
            "funding_rate_observed_at": "2026-08-17T00:00:00+00:00",
        },
    )

    assert context["preferred_side"] == "short"
    assert context["funding_projection"]["evidence_complete"] is True
    assert context["funding_projection"]["long"]["signed_cashflow_pct"] == -1.0
    assert context["funding_projection"]["short"]["signed_cashflow_pct"] == 1.0
    assert context["long"]["score"] < context["short"]["score"]


def test_funding_projection_uses_model_horizon_when_quality_gate_blocks_entries() -> None:
    strategy = {
        "execution_mode": "paper",
        "continuous_model_weights": {
            "applied": True,
            "quant_source_weights": {
                source: {
                    "effective_multiplier": 0.35,
                    "paper_execution_permission": False,
                    "paper_execution_reason": "fee_after_return_quality_evidence_missing",
                }
                for source in ("local_ml", "server_profit")
            },
        },
    }
    context = _context(
        ml=_governed_ml(0.6, 0.5),
        tools={"profit_prediction": _paper_payload(0.4, 0.3)},
        feature={
            "timestamp": "2026-08-20T05:00:00+00:00",
            "funding_data_available": True,
            "funding_rate": 0.0001,
            "funding_interval_minutes": 480,
            "next_funding_time": "2026-08-20T08:00:00+00:00",
            "funding_rate_observed_at": "2026-08-20T05:00:00+00:00",
        },
        strategy=strategy,
    )

    assert context["enabled"] is True
    assert context["selected_horizon_minutes"] == 30
    assert context["authorized_prediction_horizon_minutes"] == 30
    assert context["funding_projection"]["selected_horizon_minutes"] == 30
    assert context["funding_projection"]["horizon_source"] == (
        "selected_prediction_cohort"
    )
    assert context["funding_projection"]["evidence_complete"] is True
    assert all(
        item["funding_evidence_status"] == "complete"
        for side in ("long", "short")
        for item in context[side]["evidence"]
        if item["available"] is True
    )


@pytest.mark.parametrize(
    ("missing_field", "expected_reason"),
    [
        ("funding_rate", "funding_rate_missing"),
        ("next_funding_time", "next_funding_time_missing"),
        ("funding_rate_observed_at", "funding_rate_observed_at_missing"),
    ],
)
def test_missing_funding_fact_keeps_market_projection_incomplete(
    missing_field: str,
    expected_reason: str,
) -> None:
    feature = {
        "timestamp": "2026-08-20T05:00:00+00:00",
        "funding_data_available": True,
        "funding_rate": 0.0001,
        "funding_interval_minutes": 480,
        "next_funding_time": "2026-08-20T08:00:00+00:00",
        "funding_rate_observed_at": "2026-08-20T05:00:00+00:00",
    }
    feature.pop(missing_field)

    context = _context(
        ml=_governed_ml(0.6, 0.5),
        feature=feature,
        strategy={"execution_mode": "paper"},
    )

    assert context["funding_projection"]["evidence_complete"] is False
    assert context["funding_projection"]["long"]["reason"] == expected_reason
    assert context["funding_projection"]["short"]["reason"] == expected_reason


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


def test_negative_fee_after_source_quality_enters_paper_observation_mode() -> None:
    context = _context(
        tools={"time_series_prediction": _paper_payload(-0.5, 0.8)},
        strategy={
            "execution_mode": "paper",
            "continuous_model_weights": {
                "applied": True,
                "quant_source_weights": {
                    "timeseries": {
                        "effective_multiplier": 0.8,
                        "paper_execution_permission": False,
                        "paper_execution_reason": "fee_after_return_lcb_not_positive",
                        "paper_execution_blockers": [
                            "fee_after_return_lcb_not_positive"
                        ],
                    }
                },
            },
        },
    )

    assert context["preferred_side"] == "short"
    assert context["decision_source_count"] == 2
    short = next(
        item
        for item in context["short"]["evidence"]
        if item["source"] == "timeseries"
    )
    assert short["paper_quality_observation_only"] is True
    assert short["decision_eligible"] is True
    assert short["paper_quality_governance_reason"] == "fee_after_return_lcb_not_positive"
    assert context["continuous_model_weighting"]["quality_permissions"][
        "timeseries"
    ]["paper_execution_permission"] is False


def test_paper_mixed_horizons_select_highest_weight_native_cohort() -> None:
    server = _paper_payload(0.1, 0.7)
    for side in ("long", "short"):
        server["return_distribution_contract"][side]["horizon_minutes"] = 10

    context = _context(
        ml=_governed_ml(0.8, -0.2),
        tools={"profit_prediction": server},
        strategy={
            "execution_mode": "paper",
            "continuous_model_weights": {
                "applied": True,
                "quant_source_weights": {
                    "server_profit": {"effective_multiplier": 1.5},
                    "local_ml": {"effective_multiplier": 1.0},
                },
            },
        },
    )

    assert context["enabled"] is True
    assert context["preferred_side"] == "long"
    assert context["selected_horizon_minutes"] == 30
    assert context["decision_source_count"] == 1
    assert context["aggregate_blockers"] == []
    assert context["horizon_cohort_selection"]["selection_reason"] == (
        "authorized_prediction_horizon"
    )
    assert context["horizon_cohort_selection"]["selected_sources"] == ["local_ml"]
    server_rows = [
        item
        for side in ("long", "short")
        for item in context[side]["evidence"]
        if item["source"] == "server_profit"
    ]
    assert all(item["decision_eligible"] is False for item in server_rows)
    assert all(
        item["eligibility_reason"] == "paper_prediction_horizon_not_selected"
        for item in server_rows
    )


def test_high_weight_tool_horizon_cannot_displace_authorized_model_horizon() -> None:
    server = _paper_payload(0.0, 0.7)
    for side in ("long", "short"):
        server["return_distribution_contract"][side]["horizon_minutes"] = 60

    context = _context(
        ml=_governed_ml(0.8, 0.2),
        tools={"profit_prediction": server},
        feature={
            "timestamp": "2026-08-17T00:00:00+00:00",
            "funding_data_available": True,
            "funding_rate": 0.01,
            "funding_interval_minutes": 480,
            "next_funding_time": "2026-08-17T00:15:00+00:00",
            "funding_rate_observed_at": "2026-08-17T00:00:00+00:00",
        },
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
    assert context["selected_horizon_minutes"] == 30
    assert context["funding_projection"]["selected_horizon_minutes"] == 30
    assert context["funding_projection"]["long"]["horizon_minutes"] == 30
    assert context["funding_projection"]["short"]["horizon_minutes"] == 30
    assert context["training_long"]["horizon_minutes"] == 30
    assert context["training_short"]["horizon_minutes"] == 30
    assert context["horizon_cohort_selection"]["selected_sources"] == ["local_ml"]
    assert context["authorized_prediction_horizon_minutes"] == 30
    assert context["authorized_prediction_horizon_source"] == "local_ml_primary_horizon"


def test_unavailable_authorized_horizon_fails_closed_for_paper_execution() -> None:
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

    assert context["preferred_side"] == "neutral"
    assert context["decision_source_count"] == 0
    assert context["selected_horizon_minutes"] is None
    assert "paper_prediction_horizon_unavailable" in context["aggregate_blockers"]


def test_model_blueprint_only_blocks_its_owned_local_ml_short() -> None:
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

    assert context["preferred_side"] == "short"
    assert context["model_strategy_direction_authorization"]["short"][
        "reason"
    ] == "direction_not_authorized_by_model_blueprint"
    short_evidence = context["short"]["evidence"]
    assert short_evidence
    local_ml = next(item for item in short_evidence if item["source"] == "local_ml")
    server_profit = next(
        item for item in short_evidence if item["source"] == "server_profit"
    )
    assert local_ml["decision_eligible"] is False
    assert local_ml["paper_eligible"] is False
    assert local_ml["observation_only"] is True
    assert server_profit["decision_eligible"] is True
    assert server_profit["paper_eligible"] is True
    assert server_profit["model_strategy_side_authorization"]["enforced"] is False


def test_model_blueprint_does_not_block_independent_timeseries_when_local_ml_is_unavailable() -> None:
    signal = {
        "available": False,
        "status": "analysis_budget_deferred",
        "strategy_blueprint": {
            "version": "blueprint-v1",
            "model_version": "champion-v1",
            "paper_execution_eligible": True,
            "live_execution_permission": False,
            "eligible_sides": ["long"],
        },
    }
    tools = {"time_series_prediction": _paper_payload(0.5, 1.5)}

    context = _context(
        ml=signal,
        tools=tools,
        strategy={"execution_mode": "paper"},
    )

    assert context["preferred_side"] == "short"
    assert context["model_strategy_direction_authorization"]["short"][
        "reason"
    ] == "direction_not_authorized_by_model_blueprint"
    short_evidence = context["short"]["evidence"]
    assert short_evidence
    timeseries = next(
        item for item in short_evidence if item["source"] == "timeseries"
    )
    assert timeseries["decision_eligible"] is True
    assert timeseries["paper_eligible"] is True
    assert timeseries["model_strategy_side_authorization"]["enforced"] is False


def test_unavailable_blueprint_authority_does_not_claim_independent_tool_ownership() -> None:
    signal = {
        "available": False,
        "strategy_blueprint": {
            "version": "runtime-authority-fail-closed-v1",
            "authority_available": False,
            "paper_execution_eligible": False,
            "live_execution_permission": False,
            "eligible_sides": [],
        },
    }

    context = _context(
        ml=signal,
        tools={"time_series_prediction": _paper_payload(0.5, 1.5)},
        strategy={"execution_mode": "paper"},
    )

    assert context["preferred_side"] == "short"
    for side in ("long", "short"):
        assert context["model_strategy_direction_authorization"][side][
            "reason"
        ] == "model_strategy_blueprint_authority_unavailable"
        timeseries = next(
            item
            for item in context[side]["evidence"]
            if item["source"] == "timeseries"
        )
        assert timeseries["decision_eligible"] is True
        assert timeseries["model_strategy_side_authorization"]["enforced"] is False


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


@pytest.mark.parametrize("selected_side", ["long", "short"])
@pytest.mark.parametrize("source", ["local_ml", "server_profit"])
@pytest.mark.parametrize("blueprint_blocks_opposite", [False, True])
def test_missing_opposite_loss_probability_keeps_only_return_comparison(
    selected_side, source, blueprint_blocks_opposite,
) -> None:
    opposite = "short" if selected_side == "long" else "long"
    payload = _paper_payload(
        1.2 if selected_side == "long" else -0.3,
        1.2 if selected_side == "short" else -0.3,
    )
    payload.pop(f"{opposite}_loss_probability")
    if source == "local_ml":
        payload["predictions"][0]["return_distribution_contract"] = deepcopy(payload["return_distribution_contract"])
    signal = payload if source == "local_ml" else None
    if blueprint_blocks_opposite:
        signal = signal or {}
        signal["strategy_blueprint"] = {
            "version": "blueprint-v1",
            "model_version": "model-v1",
            "paper_execution_eligible": True,
            "live_execution_permission": False,
            "eligible_sides": [selected_side],
        }
    context = _context(
        ml=signal,
        tools={"profit_prediction": payload} if source == "server_profit" else None,
        strategy={"execution_mode": "paper"},
    )
    row = next(item for item in context[opposite]["evidence"] if item["source"] == source)
    assert row["loss_probability"] is None
    assert row["direction_comparison_eligible"] is True
    assert row["decision_eligible"] is False
    assert row["paper_eligible"] is False
    assert row["production_eligible"] is False
    if blueprint_blocks_opposite and source == "local_ml":
        assert row["model_strategy_side_authorization"]["eligible"] is False
    assert context[opposite]["score"] is None
    assert context["preferred_side"] == selected_side
    support = assess_paper_model_trade_support(context, [], selected_side, execution_cost_pct=0.1)
    assert support["eligible"] is True
    assert assess_paper_model_trade_support(
        context, [], opposite, execution_cost_pct=0.1,
    )["eligible"] is False


@pytest.mark.parametrize("selected_side", ["long", "short"])
def test_better_opposite_baseline_still_blocks_weaker_selected_direction(selected_side) -> None:
    opposite = "short" if selected_side == "long" else "long"
    payload = _paper_payload(
        1.2 if selected_side == "long" else 1.8,
        1.2 if selected_side == "short" else 1.8,
    )
    payload.pop(f"{opposite}_loss_probability")
    context = _context(
        tools={"profit_prediction": payload}, strategy={"execution_mode": "paper"},
    )
    support = assess_paper_model_trade_support(context, [], selected_side, execution_cost_pct=0.1)
    assert support["eligible"] is False
    assert "direction_support_quant_family_conflict" in support["blocking_reasons"]


def test_contract_mismatch_revokes_comparison_only_baseline() -> None:
    evidence = {
        "long": [{"decision_eligible": True, "cost_model_version": "current"}],
        "short": [{"direction_comparison_eligible": True, "cost_model_version": "old"}],
    }
    assert _enforce_aggregate_contract_consistency(evidence, include_horizon=False) == [
        "direction_competition_cost_model_version_mismatch",
    ]
    for rows in evidence.values():
        assert rows[0]["decision_eligible"] is False
        assert rows[0]["direction_comparison_eligible"] is False


@pytest.mark.parametrize("invalid_contract", ["missing", "wrong_side", "wrong_version"])
def test_missing_opposite_probability_cannot_rescue_invalid_return_contract(invalid_contract) -> None:
    payload = _paper_payload(1.2, -0.3)
    payload.pop("short_loss_probability")
    if invalid_contract == "missing":
        payload["return_distribution_contract"].pop("short")
    elif invalid_contract == "wrong_side":
        payload["return_distribution_contract"]["short"]["side"] = "long"
    else:
        payload["return_distribution_contract"]["short"]["objective_version"] = "invalid"
    context = _context(
        tools={"profit_prediction": payload}, strategy={"execution_mode": "paper"},
    )
    row = next(item for item in context["short"]["evidence"] if item["source"] == "server_profit")
    assert row.get("direction_comparison_eligible") is not True
    assert row["decision_eligible"] is False
    assert assess_paper_model_trade_support(
        context, [], "long", execution_cost_pct=0.1,
    )["eligible"] is False


@pytest.mark.parametrize("training_side", ["long", "short"])
def test_side_scoped_canary_keeps_opposite_complete_paper_training_direction(training_side) -> None:
    compared_side = "short" if training_side == "long" else "long"
    signal = _paper_payload(
        0.8 if training_side == "long" else -0.8,
        0.8 if training_side == "short" else -0.8,
    )
    signal["model_version"] = "current-model"
    signal["strategy_blueprint"] = build_model_strategy_blueprint(
        metadata={
            "artifact_version": "current-model",
            "trained_at": "2026-11-15T00:00:00+00:00",
            "test_count": 120,
            "oos_return_evaluation": {
                "long": {"return_lcb_pct": -0.2},
                "short": {"return_lcb_pct": -0.1},
            },
        },
        readiness={},
        activation={
            "activation_stage": "canary",
            "champion_comparison": {
                "accepted": True, "candidate_stage": "canary", "comparison_sides": [compared_side],
            },
        },
    )
    context = _context(ml=signal, strategy={"execution_mode": "paper"})
    evidence = context[training_side]["evidence"][0]
    assert evidence["decision_eligible"] is True
    assert evidence["paper_training_only"] is True
    assert context["production_source_count"] == 0
    support = assess_paper_model_trade_support(context, [], training_side, execution_cost_pct=0.1)
    assert support["eligible"] is True
    assert support["paper_training_only"] is True
    assert support["quant_evidence_families"] == ["local_ml"]

    signal["prediction_quality"]["anomalous"] = True
    invalid = _context(ml=signal, strategy={"execution_mode": "paper"})
    assert invalid[training_side]["evidence"][0]["decision_eligible"] is False


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
