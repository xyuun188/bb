from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from services.model_strategy_blueprint import strategy_evidence_contract
from services.strategy_historical_replay import build_strategy_historical_replay
from services.strategy_learning import StrategyFeedback, StrategyLearningEngine


def _blueprint(*, holdout_sample_count: int = 12) -> dict:
    return {
        **strategy_evidence_contract(),
        "strategy_id": "trained-model-v2",
        "model_version": "model-v2",
        "trained_at": "2026-07-20T20:00:00+00:00",
        "execution_scope": "paper_only",
        "eligible_sides": ["long", "short"],
        "paper_execution_eligible": True,
        "live_execution_permission": False,
        "training_evidence": {
            "holdout_sample_count": holdout_sample_count,
            "strategy_replay_holdout": {
                "shadow_source_id_ranges": [[13, 24]],
            },
        },
        "exit_policy": {
            "historical_replay_horizon_minutes": 10,
        },
    }


def _observation(
    source_id: int,
    *,
    decision_id: int,
    created_at: datetime,
    horizon: int,
    side: str = "long",
    execution_cost: float = 0.1,
    realized_long: float = 0.6,
    realized_short: float = -0.8,
) -> dict:
    return {
        "source_id": source_id,
        "decision_id": decision_id,
        "symbol": "BTC/USDT",
        "market_regime": "trend",
        "horizon_minutes": horizon,
        "created_at": created_at.isoformat(),
        "decision_timestamp": created_at.isoformat(),
        "label_timestamp": (created_at + timedelta(minutes=horizon)).isoformat(),
        "feature_snapshot": {"replay_side": side, "current_price": 100.0},
        "execution_cost_pct": execution_cost,
        "long_gross_return_pct": realized_long + execution_cost,
        "short_gross_return_pct": realized_short + execution_cost,
        "long_net_return_after_all_cost_pct": realized_long,
        "short_net_return_after_all_cost_pct": realized_short,
        "long_funding_return_pct": 0.0,
        "short_funding_return_pct": 0.0,
        "training_eligible": True,
    }


def _observations() -> list[dict]:
    base = datetime(2026, 7, 20, 0, 0, tzinfo=UTC)
    rows: list[dict] = []
    source_id = 1
    for decision_id in range(1, 9):
        created_at = base + timedelta(hours=decision_id * 2)
        for horizon in (10, 30, 60):
            rows.append(
                _observation(
                    source_id,
                    decision_id=decision_id,
                    created_at=created_at,
                    horizon=horizon,
                )
            )
            source_id += 1
    rows.append(
        _observation(
            source_id,
            decision_id=9,
            created_at=datetime(2026, 7, 20, 22, 0, tzinfo=UTC),
            horizon=10,
        )
    )
    return rows


def _predictor(features: dict, *, horizons: tuple[int, ...]) -> dict:
    side = str(features.get("replay_side") or "long")
    horizon = horizons[0]
    predicted_cost = float(features.get("replay_predicted_cost", 0.1))
    return {
        **strategy_evidence_contract(),
        "available": True,
        "model_version": "model-v2",
        "predictions": [
            {
                "horizon_minutes": horizon,
                "best_side": side,
                "fee_after_side_selection": {
                    "return_semantics": "risk_adjusted_net_after_counterfactual_execution_cost",
                    side: {
                        "objective_net_return_pct": 1.2 - predicted_cost,
                        "lower_quantile_net_return_pct": 0.8 - predicted_cost,
                    },
                },
                "actual_trade_calibration_ready": True,
                "return_distribution_contract": {
                    side: {
                        "production_eligible": True,
                        "objective_expected_return_pct": 1.2,
                        "lower_quantile_return_pct": 0.8,
                    }
                },
                "counterfactual_execution_cost_distribution": {
                    side: {"distribution_ready": True}
                },
            }
        ],
    }


class _BatchPredictor:
    def __init__(self) -> None:
        self.batch_calls = 0
        self.single_calls = 0

    def predict(self, features: dict, *, horizons: tuple[int, ...]) -> dict:
        self.single_calls += 1
        return _predictor(features, horizons=horizons)

    def predict_strategy_replay_batch(
        self,
        feature_rows: list[dict],
        *,
        horizon_minutes: int,
    ) -> list[dict]:
        self.batch_calls += 1
        return [
            _predictor(features, horizons=(horizon_minutes,))
            for features in feature_rows
        ]


def test_replay_reuses_historical_holdout_and_keeps_exam_disjoint() -> None:
    report = build_strategy_historical_replay(
        blueprint=_blueprint(),
        observations=_observations(),
        predictor=_predictor,
    )

    assert report["status"] == "complete"
    assert report["development_selected_entry_count"] == 2
    assert report["exam_selected_entry_count"] == 3
    development_ids = {row["source_id"] for row in report["development_samples"]}
    exam_ids = {row["source_id"] for row in report["exam_samples"]}
    assert development_ids.isdisjoint(exam_ids)
    assert report["partition"]["development_exam_overlap_count"] == 0
    assert report["partition"]["chronological_partition_disjoint"] is True
    assert report["partition"]["holdout_identity_method"] == (
        "artifact_holdout_source_ids"
    )
    assert all(
        row["strategy_replay_partition"] == "strategy_development"
        for row in report["development_samples"]
    )
    assert all(
        row["strategy_replay_partition"] == "strategy_exam"
        for row in report["exam_samples"]
    )


def test_replay_batches_uncached_model_inference() -> None:
    model = _BatchPredictor()

    report = build_strategy_historical_replay(
        blueprint={**_blueprint(), "model_version": "batch-model-v2"},
        observations=_observations(),
        predictor=model.predict,
    )

    assert report["status"] == "complete"
    assert model.batch_calls == 1
    assert model.single_calls == 0


def test_replay_uses_model_selected_side_instead_of_scoring_both_directions() -> None:
    observations = _observations()
    for row in observations:
        row["feature_snapshot"]["replay_side"] = "short"
    blueprint = _blueprint()
    blueprint["eligible_sides"] = ["long"]

    report = build_strategy_historical_replay(
        blueprint=blueprint,
        observations=observations,
        predictor=_predictor,
    )

    assert report["selected_entry_count"] == 0
    assert report["excluded_reason_counts"]["model_replay_side_not_governed"] == 5


def test_replay_keeps_unprofitable_prediction_for_continuous_paper_evaluation() -> None:
    observations = _observations()
    for row in observations:
        row["feature_snapshot"]["replay_predicted_cost"] = 1.0

    report = build_strategy_historical_replay(
        blueprint=_blueprint(),
        observations=observations,
        predictor=_predictor,
    )

    assert report["selected_entry_count"] == 0
    assert report["diagnostic_observation_count"] == 5
    assert report["development_samples"] == report["exam_samples"] == []
    assert all(
        row["paper_continuous_evaluation"] is True
        and row["normal_entry_return_gate_passed"] is False
        for row in [
            *report["diagnostic_development_samples"],
            *report["diagnostic_exam_samples"],
        ]
    )


@pytest.mark.parametrize("score_policy", [None, "obsolete-score-policy"])
@pytest.mark.parametrize("batch", [False, True])
def test_replay_rejects_predictions_from_another_scoring_policy(score_policy, batch) -> None:
    from services import strategy_historical_replay as replay_module

    replay_module._prediction_cache.clear()

    class WrongPolicyPredictor(_BatchPredictor):
        def predict(self, features, *, horizons):
            result = _predictor(features, horizons=horizons)
            result["prediction_score_policy_version"] = score_policy
            return result

        def predict_strategy_replay_batch(self, feature_rows, *, horizon_minutes):
            return [self.predict(row, horizons=(horizon_minutes,)) for row in feature_rows]

    owner = WrongPolicyPredictor()
    predictor = owner.predict if batch else lambda features, **kw: owner.predict(features, **kw)
    report = build_strategy_historical_replay(
        blueprint=_blueprint(), observations=_observations(), predictor=predictor,
    )
    assert report["status"] == "complete"
    assert report["selected_entry_count"] == 0
    assert report["excluded_reason_counts"]["model_replay_score_policy_mismatch"] == 5
    assert not replay_module._prediction_cache


@pytest.mark.parametrize("objective,lower", [(0.0, 0.0), (-0.1, -0.2), (0.5, 0.0)])
def test_observation_returns_cannot_become_entry_profit_evidence(objective, lower) -> None:
    def blocked_prediction(features, *, horizons):
        result = _predictor(features, horizons=horizons)
        result["predictions"][0]["fee_after_side_selection"]["long"].update({
            "objective_net_return_pct": objective, "lower_quantile_net_return_pct": lower,
        })
        return result

    observations = _observations()
    # Different source features isolate each parameter case from the replay cache.
    for row in observations:
        row["feature_snapshot"]["case"] = [objective, lower]
        row["long_net_return_after_all_cost_pct"] = 100.0
    report = build_strategy_historical_replay(
        blueprint=_blueprint(), observations=observations, predictor=blocked_prediction,
    )
    assert report["selected_entry_count"] == 0
    assert report["diagnostic_observation_count"] == 5
    assert report["development_samples"] == report["exam_samples"] == []


@pytest.mark.parametrize("field", list(strategy_evidence_contract()))
def test_legacy_blueprint_cannot_relabel_replay_evidence(field) -> None:
    blueprint = _blueprint()
    blueprint.pop(field)
    report = build_strategy_historical_replay(
        blueprint=blueprint, observations=_observations(), predictor=_predictor,
    )
    assert report["status"] == "trained_paper_strategy_unavailable"
    assert report["selected_entry_count"] == 0


def test_exit_observations_cannot_change_prediction_or_entry_gate() -> None:
    observations = _observations()
    for row in observations:
        row["feature_snapshot"]["shadow_settlement_cost_observation"] = {"future": 999}
        row["feature_snapshot"]["training_label_contract"] = {"best_action": "short"}
        row["side_realized_execution_cost_pct"] = {"long": 9.0}
        row["label_cost_basis"] = "entry_and_observed_exit_microstructure"
        row["execution_replay_complete"] = False

    def predictor(features, *, horizons):
        assert "shadow_settlement_cost_observation" not in features
        assert "training_label_contract" not in features
        return _predictor(features, horizons=horizons)

    report = build_strategy_historical_replay(
        blueprint=_blueprint(), observations=observations, predictor=predictor,
    )
    selected = [*report["development_samples"], *report["exam_samples"]]

    assert len(selected) == 5
    assert all(row["normal_entry_return_gate_passed"] is True for row in selected)
    assert all(row["realized_execution_cost_pct"] == 9.0 for row in selected)
    assert all(row["execution_replay_complete"] is False for row in selected)


@pytest.mark.parametrize("funding_return", [-10.0, 10.0])
def test_future_funding_cannot_change_prediction_or_entry_gate(funding_return: float) -> None:
    observations = _observations()
    baseline = build_strategy_historical_replay(
        blueprint=_blueprint(), observations=observations, predictor=_predictor,
    )
    for row in observations:
        row["long_funding_return_pct"] = funding_return
        row["long_net_return_after_all_cost_pct"] += funding_return
    replay = build_strategy_historical_replay(
        blueprint=_blueprint(), observations=observations, predictor=_predictor,
    )
    before = [*baseline["development_samples"], *baseline["exam_samples"]]
    after = [*replay["development_samples"], *replay["exam_samples"]]

    assert len(after) == len(before) == 5
    for original, changed in zip(before, after, strict=True):
        for key in (
            "side", "prediction_fingerprint", "model_expected_return_after_all_cost_pct",
            "model_return_lcb_after_cost_pct", "normal_entry_return_gate_passed",
        ):
            assert changed[key] == original[key]
        assert changed["net_return_after_all_cost_pct"] == pytest.approx(
            original["net_return_after_all_cost_pct"] + funding_return
        )


def test_delayed_labels_are_purged_before_the_exam_decision() -> None:
    observations = _observations()
    for row in observations:
        if row["decision_id"] == 6:
            row["label_timestamp"] = "2026-07-20T15:00:00+00:00"
    report = build_strategy_historical_replay(
        blueprint=_blueprint(), observations=observations, predictor=_predictor,
    )

    assert report["status"] == "complete"
    assert report["partition"]["purged_development_group_count"] == 1
    assert {row["decision_id"] for row in report["development_samples"]} == {5}


@pytest.mark.parametrize("forward_available", [False, True])
def test_dense_holdout_uses_only_independent_forward_decisions_for_exam(
    forward_available: bool,
) -> None:
    blueprint = _blueprint()
    blueprint["trained_at"] = "2026-07-20T05:00:00+00:00"
    blueprint["exit_policy"]["historical_replay_horizon_minutes"] = 240
    blueprint["training_evidence"]["strategy_replay_holdout"] = {
        "shadow_source_id_ranges": [[1, 12]],
    }
    base = datetime(2026, 7, 20, tzinfo=UTC)
    observations = [
        _observation(
            source_id, decision_id=source_id,
            created_at=base + timedelta(minutes=source_id), horizon=240,
        )
        for source_id in range(1, 13)
    ]
    if forward_available:
        observations.append(_observation(
            13, decision_id=13, created_at=base + timedelta(hours=6), horizon=240,
        ))

    report = build_strategy_historical_replay(
        blueprint=blueprint, observations=observations, predictor=_predictor,
    )

    if not forward_available:
        assert report["status"] == "partition_incomplete"
        assert report["selected_entry_count"] == 0
        return
    assert report["status"] == "complete"
    assert report["partition"]["partition_policy"] == (
        "artifact_holdout_development_then_purged_forward_exam"
    )
    assert report["partition"]["strategy_exam_historical_group_count"] == 0
    assert report["partition"]["chronological_partition_disjoint"] is True
    assert {row["source_id"] for row in report["development_samples"]} == set(range(1, 13))
    assert {row["source_id"] for row in report["exam_samples"]} == {13}
    assert report["forward_exam_samples"] == report["exam_samples"]


def test_dense_holdout_forward_exam_waits_for_all_development_labels() -> None:
    blueprint = _blueprint()
    blueprint["trained_at"] = "2026-07-20T05:00:00+00:00"
    blueprint["exit_policy"]["historical_replay_horizon_minutes"] = 240
    blueprint["training_evidence"]["strategy_replay_holdout"] = {
        "shadow_source_id_ranges": [[1, 12]],
    }
    base = datetime(2026, 7, 20, tzinfo=UTC)
    observations = [
        _observation(
            source_id, decision_id=source_id,
            created_at=base + timedelta(minutes=source_id), horizon=240,
        )
        for source_id in range(1, 13)
    ]
    observations[-1]["label_timestamp"] = "2026-07-20T06:30:00+00:00"
    observations.extend(
        _observation(
            source_id, decision_id=source_id,
            created_at=base + timedelta(hours=hour), horizon=240,
        )
        for source_id, hour in ((13, 6), (14, 7))
    )
    report = build_strategy_historical_replay(
        blueprint=blueprint, observations=observations, predictor=_predictor,
    )

    assert report["status"] == "complete"
    assert {row["source_id"] for row in report["exam_samples"]} == {14}
    assert report["partition"]["purged_forward_exam_group_count"] == 1


@pytest.mark.parametrize("missing_field", ["fee_after_side_selection", "objective_net_return_pct"])
def test_replay_requires_decision_time_fee_after_prediction(missing_field: str) -> None:
    def incomplete_predictor(features: dict, *, horizons: tuple[int, ...]) -> dict:
        prediction = _predictor(features, horizons=horizons)
        primary = prediction["predictions"][0]
        if missing_field == "fee_after_side_selection":
            primary.pop(missing_field)
        else:
            primary["fee_after_side_selection"][primary["best_side"]].pop(missing_field)
        return prediction

    observations = _observations()
    for row in observations:
        row["feature_snapshot"]["missing_prediction_field"] = missing_field
    report = build_strategy_historical_replay(
        blueprint=_blueprint(), observations=observations, predictor=incomplete_predictor,
    )

    assert report["selected_entry_count"] == 0
    assert sum(report["excluded_reason_counts"].values()) == 5


def test_paper_bootstrap_replay_does_not_require_existing_trade_calibration() -> None:
    blueprint = _blueprint()
    blueprint.update(
        {
            "eligible_sides": [],
            "production_eligible_sides": [],
            "paper_bootstrap_sides": ["long"],
            "paper_execution_sides": ["long"],
        }
    )

    def paper_predictor(features: dict, *, horizons: tuple[int, ...]) -> dict:
        prediction = _predictor(features, horizons=horizons)
        prediction["predictions"][0]["actual_trade_calibration_ready"] = False
        return prediction

    report = build_strategy_historical_replay(
        blueprint=blueprint,
        observations=_observations(),
        predictor=paper_predictor,
    )

    assert report["status"] == "complete"
    assert report["selected_entry_count"] == 5


def test_replay_does_not_wait_for_model_promotion() -> None:
    blueprint = _blueprint()
    blueprint["paper_execution_eligible"] = False

    report = build_strategy_historical_replay(
        blueprint=blueprint,
        observations=_observations(),
        predictor=_predictor,
    )

    assert report["status"] == "complete"
    assert report["selected_entry_count"] == 5
    assert report["can_authorize_live"] is False


def test_replay_fails_closed_when_artifact_holdout_cannot_be_reconstructed() -> None:
    report = build_strategy_historical_replay(
        blueprint=_blueprint(holdout_sample_count=999),
        observations=_observations(),
        predictor=_predictor,
    )

    assert report["status"] == "artifact_holdout_rows_not_reconstructable"
    assert report["development_samples"] == []
    assert report["exam_samples"] == []


def test_replay_reconstructs_explicit_holdout_crossing_training_timestamp() -> None:
    observations = _observations()
    blueprint = _blueprint()
    blueprint["training_evidence"]["strategy_replay_holdout"] = {
        "shadow_source_id_ranges": [[13, 25]],
        "sample_count": 13,
    }
    blueprint["trained_at"] = "2026-07-20T20:00:00+00:00"

    report = build_strategy_historical_replay(
        blueprint=blueprint,
        observations=observations,
        predictor=_predictor,
    )

    assert report["status"] == "complete"
    assert report["partition"]["post_training_observation_count"] == 1
    assert report["partition"]["strategy_exam_post_training_added_count"] == 0


def test_late_history_update_is_not_new_post_training_evidence() -> None:
    observations = _observations()
    late_row = _observation(
        99,
        decision_id=99,
        created_at=datetime(2026, 7, 20, 18, 0, tzinfo=UTC),
        horizon=10,
    )
    late_row["completed_at"] = "2026-07-21T00:00:00+00:00"
    observations.append(late_row)
    report = build_strategy_historical_replay(
        blueprint=_blueprint(), observations=observations, predictor=_predictor,
    )

    assert report["status"] == "complete"
    assert {row["source_id"] for row in report["forward_exam_samples"]} == {25}
    assert report["forward_exam_selected_entry_count"] == 1


def test_profitable_history_cannot_hide_a_losing_independent_forward_exam() -> None:
    blueprint = _blueprint(holdout_sample_count=42)
    blueprint["training_evidence"]["strategy_replay_holdout"] = {
        "shadow_source_id_ranges": [[1, 42]], "sample_count": 42,
    }
    base = datetime(2026, 7, 20, tzinfo=UTC)
    observations = [
        _observation(
            source_id, decision_id=source_id,
            created_at=base + timedelta(minutes=source_id * 20),
            horizon=10, realized_long=-0.01 if source_id % 5 == 0 else 0.5,
        )
        for source_id in range(1, 43)
    ]
    observations.extend(
        _observation(
            source_id, decision_id=source_id,
            created_at=base + timedelta(hours=22, minutes=(source_id - 43) * 20),
            horizon=10, realized_long=-0.1,
        )
        for source_id in (43, 44)
    )
    feedback = StrategyFeedback(
        mode="paper", window_hours=168, generated_at="2026-07-21T00:00:00+00:00",
        totals={}, side_performance={}, open_position_pressure={}, decision_quality={},
        shadow_feedback={}, expert_memory={}, manual_intervention={}, trade_fact_quarantine={},
        reflection_feedback={}, event_feedback={}, authoritative_return_observation={},
        problems=[], root_causes=[], training_policy={},
        shadow_replay_observations=observations,
    )
    schedule = StrategyLearningEngine().build_from_feedback(
        feedback, model_strategy_blueprint=blueprint, model_predictor=_predictor,
    )["schedule"]
    candidate = next(
        row for row in schedule["candidates"]
        if row["params"]["selector"]["scope"] == "symbol_side"
    )

    assert candidate["backtest"]["metrics"]["return_lcb_pct"] > 0
    assert candidate["shadow_validation"]["metrics"]["return_lcb_pct"] > 0
    assert candidate["forward_validation"]["metrics"]["sample_count"] == 2
    for key, value in strategy_evidence_contract().items():
        assert candidate["forward_validation"][key] == value
    assert candidate["forward_validation"]["metrics"]["return_lcb_pct"] < 0
    assert candidate["promotion"]["historical_prior_context_eligible"] is False
    assert "forward_exam_fee_after_return_lcb_not_positive" in (
        candidate["promotion"]["rejection_reasons"]
    )


def test_strategy_candidates_are_built_from_exact_replay_not_legacy_shadow_matching() -> None:
    feedback = StrategyFeedback(
        mode="paper",
        window_hours=168,
        generated_at="2026-07-21T00:00:00+00:00",
        totals={},
        side_performance={},
        open_position_pressure={},
        decision_quality={},
        shadow_feedback={},
        expert_memory={},
        manual_intervention={},
        trade_fact_quarantine={},
        reflection_feedback={},
        event_feedback={},
        authoritative_return_observation={},
        problems=[],
        root_causes=[],
        training_policy={},
        shadow_replay_observations=_observations(),
    )

    payload = StrategyLearningEngine().build_from_feedback(
        feedback,
        model_strategy_blueprint=_blueprint(),
        model_predictor=_predictor,
    )
    schedule = payload["schedule"]

    assert schedule["historical_model_replay"]["status"] == "complete"
    assert schedule["candidate_count"] > 0
    assert schedule["shadow_validation"]["exact_model_replay_required"] is True
    assert all(
        candidate["params"]["policy_provenance"]["evidence_mode"]
        == "exact_trained_model_historical_replay"
        for candidate in schedule["candidates"]
    )
    assert all(
        candidate["shadow_validation"]["validation_method"]
        == "exact_current_model_on_immutable_shadow_snapshot"
        for candidate in schedule["candidates"]
    )


@pytest.mark.parametrize("blocked_by", ["side", "current_return", "legacy_policy"])
def test_completed_replay_without_selected_entries_creates_no_legacy_candidates(blocked_by) -> None:
    observations = _observations()
    for row in observations:
        row["feature_snapshot"]["replay_side"] = "short" if blocked_by == "side" else "long"
        if blocked_by == "current_return":
            row["feature_snapshot"]["replay_predicted_cost"] = 1.0
    blueprint = _blueprint()
    blueprint["eligible_sides"] = ["long"]
    if blocked_by == "legacy_policy":
        blueprint.pop("prediction_score_policy_version")
    feedback = StrategyFeedback(
        mode="paper",
        window_hours=168,
        generated_at="2026-07-21T00:00:00+00:00",
        totals={},
        side_performance={},
        open_position_pressure={},
        decision_quality={},
        shadow_feedback={},
        expert_memory={},
        manual_intervention={},
        trade_fact_quarantine={},
        reflection_feedback={},
        event_feedback={},
        authoritative_return_observation={},
        problems=[],
        root_causes=[],
        training_policy={},
        authoritative_return_samples=[
            {
                "source_id": 999,
                "source_row_id": 999,
                "side": "long",
                "symbol": "BTC/USDT",
                "market_regime": "trend",
                "net_return_after_all_cost_pct": 5.0,
                "timestamp": "2026-07-20T00:00:00+00:00",
            }
        ],
        shadow_replay_observations=observations,
    )

    schedule = StrategyLearningEngine().build_from_feedback(
        feedback,
        model_strategy_blueprint=blueprint,
        model_predictor=_predictor,
    )["schedule"]

    assert schedule["historical_model_replay"]["status"] == (
        "trained_paper_strategy_unavailable" if blocked_by == "legacy_policy" else "complete"
    )
    assert schedule["candidate_count"] == 0
    if blocked_by != "legacy_policy":
        assert schedule["scheduler_mode"] == "model_replay_no_fee_after_entries"
