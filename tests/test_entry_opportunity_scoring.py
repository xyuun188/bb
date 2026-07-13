from copy import deepcopy

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.entry_opportunity_scoring import EntryOpportunityScoringPolicy
from services.return_objective import RETURN_OBJECTIVE_NAME, RETURN_OBJECTIVE_VERSION


def _scorer() -> EntryOpportunityScoringPolicy:
    return EntryOpportunityScoringPolicy(
        normalize_symbol=lambda value: str(value or ""),
        annotate_decision_source=lambda _decision: None,
    )


def _live_payload(*, side: str, long_return: float, short_return: float) -> dict:
    return {
        "available": True,
        "route_mode": "live",
        "live_influence": True,
        "promotion_ready": True,
        "horizon_minutes": 30,
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "prediction_quality": {
            "production_eligible": True,
            "anomalous": False,
        },
        "best_side": side,
        "long_expected_return_pct": long_return,
        "short_expected_return_pct": short_return,
        "long_loss_probability": 0.2,
        "short_loss_probability": 0.3,
    }


def _live_ml() -> dict:
    return {
        "available": True,
        "allow_live_position_influence": True,
        "influence_enabled": True,
        "readiness": {"allow_live_position_influence": True},
        "influence_policy": {"long": {"enabled": True}, "short": {"enabled": True}},
        "predictions": [
            {
                "long_expected_return_pct": 0.8,
                "short_expected_return_pct": -0.2,
                "long_lower_quantile_return_pct": 0.4,
                "short_lower_quantile_return_pct": -0.5,
                "long_tail_loss_probability": 0.1,
                "short_tail_loss_probability": 0.4,
                "horizon_minutes": 30,
                "long_win_rate": 0.9,
                "short_win_rate": 0.1,
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
    assert score == pytest.approx(
        opportunity["return_lcb_pct"] - opportunity["expected_loss_pct"]
    )
    assert opportunity["score_policy"] == "fee_after_return_lcb_minus_expected_downside"
    assert opportunity["policy_provenance"]["valid_for_seconds"] > 0
    assert opportunity["policy_provenance"]["fallback_reason"] == ""


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
            "allow_live_position_influence": False,
            "influence_enabled": False,
            "advisory_enabled": True,
        }
    )
    decision.raw_response["ml_signal"]["predictions"][0][
        "long_expected_return_pct"
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


def test_trained_runtime_predictions_form_recovery_distribution_when_live_sources_absent() -> None:
    decision = _decision()
    decision.raw_response["ml_signal"].update(
        {
            "allow_live_position_influence": False,
            "influence_enabled": False,
            "trained_sample_count": 100,
            "model_version": "2026-07-13T12:00:00+00:00",
            "readiness": {
                "allow_live_position_influence": False,
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
                "live_influence": False,
                "trained": True,
                "artifact_persisted": True,
                "training_cost_policy": "per_sample_live_spread_fee_and_funding_complete",
                "label_name": "net_return_after_cost_pct",
                "label_version": RETURN_OBJECTIVE_VERSION,
            }
        )

    _scorer().score_candidate(decision)

    opportunity = decision.raw_response["opportunity_score"]
    components = opportunity["expected_net_breakdown"]["components"]
    assert opportunity["return_distribution_mode"] == "runtime_recovery"
    assert opportunity["production_eligible"] is True
    assert opportunity["return_lcb_pct"] > 0
    assert all(component["production_eligible"] is False for component in components)
    assert all(component["included_in_return_distribution"] is True for component in components)
    assert opportunity["policy_provenance"]["fallback_reason"] == ""


def test_runtime_recovery_excludes_anomalous_trained_server_prediction() -> None:
    decision = _decision()
    decision.raw_response["ml_signal"] = {}
    for payload in decision.raw_response["local_ai_tools"].values():
        payload.update(
            {
                "route_mode": "shadow_candidate",
                "live_influence": False,
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
    assert opportunity["expected_net_return_pct"] == 0.0
    assert opportunity["policy_provenance"]["sample_count"] == 0
