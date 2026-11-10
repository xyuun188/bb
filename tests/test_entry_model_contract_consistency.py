from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from services import ml_signal_service as ml
from services.entry_direction_support import assess_paper_model_trade_support
from services.entry_signal_extraction import signal_loss_probability
from services.ml_prediction_contract import risk_adjusted_expected_scores
from services.normal_paper_trade import select_normal_paper_trade_side
from services.profit_supervision import PROFIT_SUPERVISION_VERSION
from services.return_objective import (
    combine_paper_return_distribution,
    standardized_return_distribution,
)


@pytest.mark.parametrize("size", [100, 1000])
def test_probability_classifier_preserves_a_rare_event_prior(size: int) -> None:
    x = pd.DataFrame({"uninformative": np.zeros(size)})
    y = pd.Series([0] * (size * 9 // 10) + [1] * (size // 10))
    classifier = ml._make_classifier(y)
    classifier.fit(x, y)
    assert ml._positive_proba(classifier, x.iloc[:1])[0] == pytest.approx(0.1, abs=0.015)


def test_training_side_selection_and_execution_use_identical_model_dispersion() -> None:
    market = {
        "expected": np.array([10.269162924167457]),
        "median": np.array([12.430190701158566]),
        "lower_quantile": np.array([-1.9621961668115095]),
        "upper_quantile": np.array([17.75715036222544]),
        "std": np.array([6.053513037697171]),
    }
    cost = {
        "expected": np.array([0.12148]),
        "lower_quantile": np.array([0.12148]),
        "upper_quantile": np.array([0.12148]),
    }
    probability, scale = 0.43875631752805705, 2.4493965759155643
    gross = standardized_return_distribution(
        side="short",
        horizon_minutes=240,
        raw_expected_return_pct=market["expected"][0],
        median_return_pct=market["median"][0],
        lower_quantile_return_pct=market["lower_quantile"][0],
        upper_quantile_return_pct=market["upper_quantile"][0],
        dispersion_pct=market["std"][0],
        tail_loss_probability=probability,
        tail_loss_scale_pct=scale,
        distribution_member_count=220,
        return_semantics="gross_market_opportunity_before_execution",
        source_authority="random_forest_tree_empirical_distribution",
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
    )
    training_score = risk_adjusted_expected_scores(
        market, cost, np.array([probability]), tail_loss_scale_pct=scale,
    )[0]
    prediction = ml._multitask_side_prediction(
        side="short",
        horizon_minutes=240,
        model_version="test",
        calibration_version="test",
        return_contract=gross,
        cost_distribution=cost,
        win_probability=0.58,
        tail_loss_probability=probability,
        expected_mfe_pct=1.0,
        expected_mae_pct=1.0,
    )
    side_score = ml._fee_after_prediction_objective(
        prediction, tail_loss_scale_pct=scale,
    )["objective_net_return_pct"]
    execution = combine_paper_return_distribution(
        side="short",
        model_contracts=[gross],
        current_execution_cost_pct=0.12148,
        current_slippage_pct=0.02148,
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
        source_authority="test",
    )
    assert execution["paper_eligible"] is True
    assert execution["lower_quantile_return_pct"] <= execution["median_return_pct"]
    assert training_score == pytest.approx(3.0194816646557405)
    assert side_score == pytest.approx(training_score)
    assert execution["objective_expected_return_pct"] == pytest.approx(training_score)
    assert execution["dispersion_pct"] == pytest.approx(gross["dispersion_pct"])


def test_ordinary_loss_probability_never_falls_back_to_tail_probability() -> None:
    payload = {
        "return_distribution_contract": {"long": {"tail_loss_probability": 0.05}},
        "predictions": [{
            "multitask_prediction": {
                "long": {"loss_probability": 0.55, "tail_loss_probability": 0.05}
            }
        }],
    }
    assert signal_loss_probability(payload, "long") == 0.55
    del payload["predictions"]
    assert signal_loss_probability(payload, "long") is None
    payload["long_loss_probability"] = 0.4
    assert signal_loss_probability(payload, "long") == 0.4
    payload["model"] = "local-profit-trained-v2"
    assert signal_loss_probability(payload, "long") is None


def test_opportunity_keeps_ordinary_and_tail_probabilities_separate() -> None:
    from tests.test_entry_opportunity_scoring import _decision, _scorer

    decision = _decision()
    decision.raw_response["local_ai_tools"] = {}
    primary = decision.raw_response["ml_signal"]["predictions"][0]
    primary["long_win_rate"] = 0.45
    _scorer().score_candidate(decision, {"execution_mode": "paper"})
    opportunity = decision.raw_response["opportunity_score"]
    assert opportunity["loss_probability"] == pytest.approx(0.55)
    assert opportunity["server_profit_loss_probability"] == pytest.approx(0.55)
    assert opportunity["tail_risk_score"] == pytest.approx(0.2)


@pytest.mark.parametrize("scope", ["paper", "live"])
def test_tail_only_tools_do_not_dilute_a_complete_execution_prediction(scope: str) -> None:
    from tests.test_entry_direction_competition import _context, _governed_ml, _governed_payload
    from tests.test_entry_opportunity_scoring import _decision, _scorer

    tail_only = _governed_payload(100.0, -100.0)
    tail_only.pop("long_loss_probability")
    tail_only.pop("short_loss_probability")
    context = _context(
        ml=_governed_ml(0.6, -0.4),
        tools={"time_series_prediction": tail_only},
        strategy={"execution_mode": scope},
    )
    assert context["preferred_side"] == "long"
    assert context["long"]["raw_expected_return_pct"] == pytest.approx(0.6)
    observation = next(row for row in context["long"]["evidence"] if row["source"] == "timeseries")
    assert observation["observation_only"] is True
    assert observation["decision_eligible"] is False
    assert observation["eligibility_reason"] == "ordinary_loss_probability_missing"

    decision = _decision()
    decision.raw_response["local_ai_tools"] = {}
    scorer = _scorer()
    scorer.score_candidate(decision, {"execution_mode": scope})
    baseline = deepcopy(decision.raw_response["opportunity_score"])
    decision.raw_response["local_ai_tools"] = {"time_series_prediction": tail_only}
    scorer.score_candidate(decision, {"execution_mode": scope})
    scored = decision.raw_response["opportunity_score"]
    for field in ("expected_net_return_pct", "return_lcb_pct", "loss_probability"):
        assert scored[field] == pytest.approx(baseline[field])
    observation = next(
        row for row in scored["expected_net_breakdown"]["components"] if row["key"] == "timeseries"
    )
    assert observation["decision_eligible"] is False
    assert observation["observation_only"] is True


@pytest.mark.parametrize("scope", ["paper", "live"])
def test_tail_only_tool_cannot_authorize_entry_on_its_own(scope: str) -> None:
    from tests.test_entry_direction_competition import _context, _governed_payload
    from tests.test_entry_opportunity_scoring import _decision, _scorer

    tail_only = _governed_payload(100.0, -100.0)
    tail_only.pop("long_loss_probability")
    tail_only.pop("short_loss_probability")
    context = _context(
        tools={"time_series_prediction": tail_only}, strategy={"execution_mode": scope},
    )
    assert context["preferred_side"] == "neutral"
    assert context["decision_source_count"] == 0
    decision = _decision()
    decision.raw_response["ml_signal"] = {}
    decision.raw_response["local_ai_tools"] = {"time_series_prediction": tail_only}
    _scorer().score_candidate(decision, {"execution_mode": scope})
    assert decision.raw_response["opportunity_score"]["expected_net_return_pct"] is None


def test_reconciliation_rejects_high_ordinary_loss_despite_low_tail_probability() -> None:
    from services.normal_paper_trade import (
        build_normal_paper_trade_contract,
        reconcile_normal_paper_trade_contract,
    )
    from tests.test_normal_paper_trade import _support

    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT", side="long", selection_reason="strategy_edge_selected",
        direction_support=_support("long", expected_net=0.8, objective_net=0.5),
    )
    scored = {
        "expected_net_return_pct": 0.8,
        "return_lcb_pct": 0.5,
        "loss_probability": 0.75,
        "server_profit_loss_probability": 0.05,
        "return_distribution_contract": {"tail_loss_probability": 0.05},
    }
    result = reconcile_normal_paper_trade_contract(contract, scored)
    assert result["authorized"] is False
    assert result["loss_probability"] == 0.75
    assert result["reconciliation"]["reason"] == "current_execution_loss_probability_invalid"


def test_negative_evidence_is_classified_as_profit_blocked_without_granting_entry() -> None:
    row = {
        "source": "local_ml", "decision_eligible": True,
        "raw_expected_return_pct": -0.2, "objective_expected_return_pct": -0.4,
        "horizon_minutes": 240, "loss_probability": 0.4,
    }
    competition = {
        "long": {"evidence": [row]},
        "short": {"evidence": [{**row, "raw_expected_return_pct": -0.1}]},
    }
    supports = {
        side: assess_paper_model_trade_support(
            competition, [], side, execution_cost_pct=0.1,
        )
        for side in ("long", "short")
    }
    selection = select_normal_paper_trade_side(supports)
    assert selection["selected"] is False
    assert selection["diagnostic_status"] == "profit_gate_blocked"
    assert "direction_support_expected_net_not_positive" in selection["blocking_reasons"]
    assert "direction_support_quant_evidence_missing" not in selection["blocking_reasons"]
    assert supports["long"]["quant_family_summaries"]


def test_probability_replay_weights_are_independent_of_future_error_emphasis() -> None:
    from tests.test_ml_signal_training_quality import _training_frame

    frame = _training_frame(40)
    frame.index = np.arange(100, 140)
    original, _ = ml._apply_training_replay_weights(frame)
    emphasized = frame.copy()
    emphasized["long_tail_loss"] = 1
    emphasized["decision_confidence"] = 0.9
    emphasized["decision_action"] = "long"
    emphasized["best_action"] = "short"
    changed, _ = ml._apply_training_replay_weights(emphasized)
    assert changed["sample_weight"].notna().all()
    assert np.any(changed["sample_weight"] > original["sample_weight"])
    np.testing.assert_allclose(
        changed["probability_sample_weight"], original["probability_sample_weight"],
    )


def test_walk_forward_rebuilds_fold_local_targets_before_replay_weighting() -> None:
    from tests.test_ml_signal_training_quality import _training_frame

    frame = _training_frame(40)
    train, validation = frame.iloc[:30].copy(), frame.iloc[30:].copy()
    first, report = ml._walk_forward_side_scores(train, validation, side="long")
    contaminated = deepcopy(train)
    contaminated["long_tail_loss"] = 1
    contaminated["short_tail_loss"] = 1
    second, _ = ml._walk_forward_side_scores(contaminated, validation, side="long")
    np.testing.assert_allclose(first, second)
    assert report["training_replay_weights"]["sample_count"] == len(train)
    assert report["training_replay_weights"]["version"] == ml.REPLAY_WEIGHT_POLICY_VERSION


def test_final_decision_confidence_is_metadata_and_never_a_prediction_feature() -> None:
    assert "decision_confidence" not in ml.FEATURE_KEYS
    snapshot = {"symbol": "BTC/USDT", "current_price": 100.0, "rsi_14": 45.0}
    training = ml._feature_row_from_snapshot(
        snapshot, decision_confidence=0.95, horizon_minutes=240,
    )
    serving = ml._feature_row_from_feature_vector(snapshot, horizon_minutes=240)
    assert training == serving
