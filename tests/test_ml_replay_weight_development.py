from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from services import ml_replay_weight_development as research
from services.ml_development_baselines import _development_input_sha256
from services.ml_development_snapshot import (
    REPLAY_METADATA_COLUMNS,
    REPLAY_SNAPSHOT_VERSION,
    SNAPSHOT_VERSION,
    make_development_snapshot,
    snapshot_frame,
)
from services.ml_signal_service import FEATURE_KEYS
from tests.test_ml_development_baselines import _frame

CUTOFF = "2026-08-10T00:00:00+00:00"


def _source():
    frame = _frame()
    frame["symbol"] = "BTC/USDT"
    frame["decision_action"] = "hold"
    frame["decision_confidence"] = 0.8
    frame["best_action"] = "short"
    frame["market_regime"] = "ranging"
    return frame


def _snapshot(frame):
    snapshot = make_development_snapshot(frame, cutoff=CUTOFF, include_replay_metadata=True)
    return snapshot_frame(snapshot, cutoff=CUTOFF), snapshot


def test_replay_snapshot_seals_all_training_metadata_without_changing_legacy_schema():
    frame = _source()
    legacy = make_development_snapshot(frame, cutoff=CUTOFF)
    assert legacy["version"] == SNAPSHOT_VERSION
    assert not set(REPLAY_METADATA_COLUMNS).intersection(legacy["records"][0])
    development, snapshot = _snapshot(frame)
    assert snapshot["version"] == REPLAY_SNAPSHOT_VERSION
    assert development["decision_confidence"].eq(0.8).all()
    assert not set(REPLAY_METADATA_COLUMNS).intersection(FEATURE_KEYS)
    assert _development_input_sha256(
        development.drop(columns=list(REPLAY_METADATA_COLUMNS))
    ) == legacy["development_input_sha256"]
    for column, value in (("decision_action", "long"), ("decision_confidence", 0.1),
                          ("best_action", "long"), ("market_regime", "trending")):
        changed = frame.copy()
        changed.loc[0, column] = value
        assert _snapshot(changed)[1]["development_input_sha256"] != snapshot["development_input_sha256"]


def test_replay_snapshot_cannot_silently_drop_training_metadata():
    with pytest.raises(ValueError, match="requires training metadata"):
        _snapshot(_source().drop(columns=["decision_confidence"]))


def test_frozen_cutoff_excludes_future_training_and_validation_inputs():
    frame = _source()
    future = frame.tail(2).copy()
    future["id"] += 10000
    future["decision_group"] = "future"
    future["decision_timestamp"] = pd.Timestamp(CUTOFF)
    future["label_timestamp"] = pd.Timestamp(CUTOFF) + pd.Timedelta(days=1)
    original, snapshot = _snapshot(pd.concat([frame, future]))
    future[FEATURE_KEYS] = 999999.0
    future["long_return_pct"] = 999999.0
    future["sample_weight"] = 999999.0
    changed, changed_snapshot = _snapshot(pd.concat([frame, future]))
    pd.testing.assert_frame_equal(original, changed)
    assert snapshot["development_input_sha256"] == changed_snapshot["development_input_sha256"]


def test_paired_fit_measures_outcome_weight_bias_and_never_reads_validation_outcomes():
    frame = _source().query("horizon_minutes == 240").head(50).copy()
    frame[FEATURE_KEYS] = 0.0
    frame["long_return_pct"] = [1.0, -1.0] * 25
    frame["short_return_pct"] = 0.0
    frame["long_execution_cost_pct"] = -0.1
    frame["short_execution_cost_pct"] = 0.1
    frame["decision_action"] = "long"
    frame["best_action"] = ["long", "short"] * 25
    train, validation = frame.iloc[:40].copy(), frame.iloc[40:].copy()
    paired, manifests = research._paired_predictions(train, validation, feature_policy="existing")
    assert manifests["outcome_emphasis"]["hard_error_sample_count"] == 20
    assert manifests["natural"]["outcome_emphasis"] is False
    assert np.mean(paired["natural"]["net"]["long"]) > np.mean(
        paired["outcome_emphasis"]["net"]["long"]
    )
    # Constant signed costs are subtracted exactly once in both arms.
    expected = np.average(
        train["long_return_pct"],
        weights=research._apply_training_replay_weights(train, outcome_emphasis=False)[0]["sample_weight"],
    ) + 0.1
    np.testing.assert_allclose(paired["natural"]["net"]["long"], expected)
    changed = validation.copy()
    changed[["long_return_pct", "short_return_pct", "sample_weight"]] = 1e9
    repeated, _ = research._paired_predictions(train, changed, feature_policy="existing")
    for policy in research.WEIGHT_POLICIES:
        for task in ("net", "score"):
            for side in ("long", "short"):
                np.testing.assert_allclose(paired[policy][task][side], repeated[policy][task][side])


def test_report_keeps_negative_scores_as_hold_and_purges_training_labels(monkeypatch):
    calls = []

    def predict(train, validation, *, feature_policy):
        calls.append((train.copy(), validation.copy(), feature_policy))
        prediction = {
            "net": {side: np.full(len(validation), 1.0) for side in ("long", "short")},
            "score": {side: np.full(len(validation), -1.0) for side in ("long", "short")},
        }
        return {policy: deepcopy(prediction) for policy in research.WEIGHT_POLICIES}, {}

    monkeypatch.setattr(research, "_paired_predictions", predict)
    development, snapshot = _snapshot(_source())
    report = research.build_replay_weight_development_report(development, snapshot=snapshot)
    assert report["status"] == "rejected"
    assert len(report["candidates"]) == 4
    assert len(calls) == 6
    assert report["qualified_candidate_ids"] == []
    assert report["promotion_authorized"] is False
    assert report["exam_labels_evaluated"] is False
    assert report["artifact_persisted"] is False
    assert report["orders_submitted"] is False
    for train, validation, _ in calls:
        assert train["label_timestamp"].max() < validation["decision_timestamp"].min()
        assert not set(train["decision_group"]).intersection(validation["decision_group"])
    for candidate in report["candidates"]:
        assert candidate["aggregate_evidence"]["capacity"]["selected_count"] == 0
        assert candidate["rank_only_top_20_percent"]["selected_count"] > 0
        assert candidate["capacity_attribution"]["symbol"]["BTC/USDT"]["selected_count"] == 0


def test_report_rejects_changed_frozen_metadata():
    development, snapshot = _snapshot(_source())
    development.loc[0, "decision_confidence"] = 0.1
    with pytest.raises(ValueError, match="exact frozen"):
        research.build_replay_weight_development_report(development, snapshot=snapshot)


def test_alternative_horizon_fits_only_its_own_mature_labels(monkeypatch):
    source = _source()
    alternative = source[source["horizon_minutes"] == 240].copy()
    alternative["id"] += 10000
    alternative["horizon_minutes"] = 60
    alternative["label_timestamp"] = alternative["decision_timestamp"] + pd.Timedelta(minutes=60)
    alternative["long_return_pct"] = 0.7
    calls = []

    def predict(train, validation, *, feature_policy):
        assert train["horizon_minutes"].eq(60).all()
        assert validation["horizon_minutes"].eq(60).all()
        assert train["long_return_pct"].eq(0.7).all()
        calls.append(feature_policy)
        value = {task: {side: np.full(len(validation), -1.0) for side in ("long", "short")}
                 for task in ("net", "score")}
        return {policy: deepcopy(value) for policy in research.WEIGHT_POLICIES}, {}

    monkeypatch.setattr(research, "_paired_predictions", predict)
    development, snapshot = _snapshot(pd.concat([source, alternative]))
    report = research.build_replay_weight_development_report(
        development, snapshot=snapshot, horizon_minutes=60,
    )
    assert len(calls) == 6
    assert report["protocol"]["horizon_minutes"] == 60
    assert report["primary_sample_count"] == len(alternative)
    assert report["promotion_authorized"] is False
    with pytest.raises(ValueError, match="supports only"):
        research.protocol(horizon_minutes=15)


def test_standardized_experiment_fits_scale_only_on_training_features(monkeypatch):
    from sklearn.preprocessing import StandardScaler

    frame = _source().query("horizon_minutes == 240").head(50).copy()
    train, validation = frame.iloc[:40].copy(), frame.iloc[40:].copy()
    validation.loc[:, "returns_20"] = 1e6
    observed = []
    original = StandardScaler.fit

    def fit(scaler, values, *args, **kwargs):
        observed.append(values.copy())
        return original(scaler, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", fit)
    predictions, _ = research._paired_predictions(
        train, validation, feature_policy="market_only", preprocessing="standardized_linear",
    )
    assert len(observed) == 1
    assert observed[0].index.tolist() == train.index.tolist()
    assert observed[0]["returns_20"].max() < 1e6
    validation[["long_return_pct", "short_return_pct", "sample_weight"]] = 1e9
    repeated, _ = research._paired_predictions(
        train, validation, feature_policy="market_only", preprocessing="standardized_linear",
    )
    for policy in research.WEIGHT_POLICIES:
        for task in ("net", "score"):
            for side in ("long", "short"):
                np.testing.assert_allclose(predictions[policy][task][side], repeated[policy][task][side])
    registered = research.protocol(preprocessing="standardized_linear")
    assert registered["preprocessing"] == "standardized_linear"
    assert registered["promotion_authorized"] is False


@pytest.mark.parametrize("sample_count", (40, 300))
def test_linear_research_keeps_estimator_family_across_sample_counts(sample_count):
    from sklearn.linear_model import LogisticRegression, Ridge

    y = pd.Series(np.arange(sample_count) % 2)
    classifier = research._research_model(
        y, classification=True, preprocessing="standardized_linear",
    )
    regressor = research._research_model(
        y, classification=False, preprocessing="standardized_linear",
    )
    assert isinstance(classifier.named_steps["model"], LogisticRegression)
    assert isinstance(regressor.named_steps["model"], Ridge)
    assert regressor.named_steps["model"].alpha == 1.0
