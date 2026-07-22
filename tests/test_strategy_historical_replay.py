from __future__ import annotations

from datetime import UTC, datetime, timedelta

from services.strategy_historical_replay import build_strategy_historical_replay
from services.strategy_learning import StrategyFeedback, StrategyLearningEngine


def _blueprint(*, holdout_sample_count: int = 12) -> dict:
    return {
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
        "long_net_return_after_cost_pct": realized_long,
        "short_net_return_after_cost_pct": realized_short,
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
    return {
        "available": True,
        "model_version": "model-v2",
        "predictions": [
            {
                "horizon_minutes": horizon,
                "best_side": side,
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
        row["execution_cost_pct"] = 1.0

    report = build_strategy_historical_replay(
        blueprint=_blueprint(),
        observations=observations,
        predictor=_predictor,
    )

    assert report["selected_entry_count"] == 5
    assert all(
        row["paper_continuous_evaluation"] is True
        and row["normal_entry_return_gate_passed"] is False
        for row in [
            *report["development_samples"],
            *report["exam_samples"],
        ]
    )


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


def test_completed_replay_without_selected_entries_creates_no_legacy_candidates() -> None:
    observations = _observations()
    for row in observations:
        row["feature_snapshot"]["replay_side"] = "short"
    blueprint = _blueprint()
    blueprint["eligible_sides"] = ["long"]
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
                "net_return_after_cost_pct": 5.0,
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

    assert schedule["historical_model_replay"]["status"] == "complete"
    assert schedule["candidate_count"] == 0
    assert schedule["scheduler_mode"] == "model_replay_no_fee_after_entries"
