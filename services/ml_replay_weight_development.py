"""Paired training-recipe diagnostics on immutable development data only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from services.dynamic_policy_values import empirical_policy_value
from services.liquidity_regime_development import _rejections, capacity_sides
from services.ml_blocked_return_evidence import blocked_return_evidence
from services.ml_development_baselines import (
    _development_input_sha256,
    _evidence,
    _net_returns,
    _prediction_diagnostics,
    _rank_only_evidence,
    _selected_sides,
)
from services.ml_development_snapshot import REPLAY_METADATA_COLUMNS, REPLAY_SNAPSHOT_VERSION
from services.ml_horizon_development import FEATURE_POLICIES, _folds
from services.ml_prediction_contract import (
    regression_prediction_distribution,
    risk_adjusted_expected_scores,
)
from services.ml_signal_service import (
    MIN_TRAINING_SAMPLE_COUNT,
    PRIMARY_PREDICTION_HORIZON_MINUTES,
    REPLAY_WEIGHT_POLICY_VERSION,
    _apply_training_replay_weights,
    _make_classifier,
    _make_regressor,
    _positive_proba,
)

POLICY_VERSION = "2026-11-11.explicit-linear-replay-development.v2"
WEIGHT_POLICIES = ("outcome_emphasis", "natural")
PREPROCESSING_POLICIES = ("existing", "standardized_linear")


def protocol(
    *, horizon_minutes: int = PRIMARY_PREDICTION_HORIZON_MINUTES,
    preprocessing: str = "existing",
) -> dict[str, Any]:
    if horizon_minutes not in (60, PRIMARY_PREDICTION_HORIZON_MINUTES):
        raise ValueError("paired recipe research supports only 60m and the production horizon")
    if preprocessing not in PREPROCESSING_POLICIES:
        raise ValueError("unknown registered preprocessing policy")
    return {
        "version": POLICY_VERSION,
        "replay_weight_version": REPLAY_WEIGHT_POLICY_VERSION,
        "horizon_minutes": horizon_minutes,
        "preprocessing": preprocessing,
        "preprocessing_policy": (
            "standard scaler fitted only on each purged training fold; "
            "validation features transformed without refit; production unchanged"
            if preprocessing == "standardized_linear" else "existing production inputs"
        ),
        "estimator_policy": (
            "Ridge alpha=1 and LogisticRegression C=1 max_iter=1000 seed=42; "
            "constant-label dummy only; no sample-count model switching"
            if preprocessing == "standardized_linear" else
            "existing production sample-count estimator selection"
        ),
        "candidates": [
            {"id": f"{features}-{weights}", "feature_policy": features, "weight_policy": weights}
            for features in FEATURE_POLICIES for weights in WEIGHT_POLICIES
        ],
        "feature_keys": {key: list(value) for key, value in FEATURE_POLICIES.items()},
        "fold_policy": "three expanding folds; purge labels unavailable at validation start",
        "weight_policy": (
            "paired fits differ only in regression outcome emphasis; both retain quality "
            "and recency; tail probabilities use natural weights in both arms"
        ),
        "score_policy": "deployed gross-minus-cost-minus-risk objective; positive score else hold",
        "capacity_policy": "three concurrent positions, one per symbol; carry across folds",
        "qualification_policy": (
            "at least 16 selected, fee-after mean>0, naive LCB>0 and PF>1 "
            "in both opportunity and capacity views, aggregate and each fold"
        ),
        "rank_policy": "fixed top 20 percent diagnostic only; cannot authorize entry",
        "uncertainty_policy": (
            "report moving two-day UTC block-bootstrap 95% lower bound; all symbols "
            "joint; minimum 16 selected calendar days; development only"
        ),
        "exam_policy": "development only; prospective decisions after registration remain unexamined",
        "promotion_authorized": False,
    }


def _research_model(y: pd.Series, *, classification: bool, preprocessing: str) -> Pipeline:
    if preprocessing != "standardized_linear":
        return _make_classifier(y) if classification else _make_regressor(y)
    if int(y.nunique()) < 2:
        estimator = DummyClassifier(strategy="prior") if classification else DummyRegressor()
    else:
        estimator = (
            LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            if classification else Ridge(alpha=1.0)
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def _paired_predictions(
    train: pd.DataFrame, validation: pd.DataFrame, *, feature_policy: str,
    preprocessing: str = "existing",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    train = train.copy()
    boundaries = {}
    for side in ("long", "short"):
        net = _net_returns(train, side)
        policy = empirical_policy_value(
            f"{side}_tail_loss_boundary_pct", net[net < 0].tolist(),
            selector="lower_hinge", observation_window="development_fold_training_only",
        )
        boundary = float(policy.value) if policy.value is not None else 0.0
        boundaries[side] = boundary
        train[f"{side}_tail_loss"] = (net < boundary).astype(int)
    frames, manifests = {}, {}
    for policy in WEIGHT_POLICIES:
        frames[policy], manifests[policy] = _apply_training_replay_weights(
            train, outcome_emphasis=policy == "outcome_emphasis",
        )
    features = list(FEATURE_POLICIES[feature_policy])
    x_train, x_validation = train[features], validation[features]
    if preprocessing == "standardized_linear":
        scaler = StandardScaler().fit(x_train)
        x_train = pd.DataFrame(scaler.transform(x_train), columns=features, index=train.index)
        x_validation = pd.DataFrame(
            scaler.transform(x_validation), columns=features, index=validation.index,
        )
    elif preprocessing != "existing":
        raise ValueError("unknown registered preprocessing policy")
    predictions: dict[str, dict[str, Any]] = {
        policy: {"net": {}, "score": {}} for policy in WEIGHT_POLICIES
    }
    for side in ("long", "short"):
        tail = _research_model(
            train[f"{side}_tail_loss"], classification=True, preprocessing=preprocessing,
        )
        tail.fit(
            x_train, train[f"{side}_tail_loss"],
            model__sample_weight=frames["natural"]["probability_sample_weight"],
        )
        tail_probabilities = _positive_proba(tail, x_validation)
        for policy in WEIGHT_POLICIES:
            distributions = []
            for column in (f"{side}_return_pct", f"{side}_execution_cost_pct"):
                model = _research_model(
                    train[column], classification=False, preprocessing=preprocessing,
                )
                model.fit(x_train, train[column], model__sample_weight=frames[policy]["sample_weight"])
                distributions.append(regression_prediction_distribution(model, x_validation))
            market, cost = distributions
            net_prediction = np.asarray(market["expected"] - cost["expected"], dtype=float)
            score = risk_adjusted_expected_scores(
                market, cost, tail_probabilities,
                tail_loss_scale_pct=max(abs(boundaries[side]), float(np.finfo(float).eps)),
            )
            if not np.isfinite(net_prediction).all() or not np.isfinite(score).all():
                raise ValueError("non-finite training-recipe diagnostic prediction")
            predictions[policy]["net"][side] = net_prediction
            predictions[policy]["score"][side] = score
    return predictions, manifests


def _views(frame: pd.DataFrame, sides: np.ndarray, executable: np.ndarray) -> dict[str, Any]:
    return {
        "all_opportunities": _evidence(frame, sides),
        "capacity": _evidence(frame, executable),
    }


def _attribution(frame: pd.DataFrame, sides: np.ndarray) -> dict[str, Any]:
    result = {}
    for column in ("market_regime", "symbol", "liquidity_regime_high"):
        result[column] = {
            str(key): _evidence(group, sides[frame.index.get_indexer(group.index)])
            for key, group in frame.groupby(column, sort=True)
        }
    result["direction"] = {
        side: _evidence(frame, np.where(sides == side, side, "hold"))
        for side in ("long", "short")
    }
    return result


def build_replay_weight_development_report(
    development: pd.DataFrame, *, snapshot: dict[str, Any],
    horizon_minutes: int = PRIMARY_PREDICTION_HORIZON_MINUTES,
    preprocessing: str = "existing",
) -> dict[str, Any]:
    registered = protocol(horizon_minutes=horizon_minutes, preprocessing=preprocessing)
    if snapshot.get("version") != REPLAY_SNAPSHOT_VERSION:
        raise ValueError("training-recipe comparison requires replay metadata snapshot")
    if _development_input_sha256(development) != snapshot["development_input_sha256"]:
        raise ValueError("research must use the exact frozen development frame")
    if not set(REPLAY_METADATA_COLUMNS).issubset(development.columns):
        raise ValueError("training-recipe comparison requires replay metadata")
    primary = development[
        development["horizon_minutes"] == horizon_minutes
    ].copy()
    report: dict[str, Any] = {
        "status": "insufficient_development_data",
        "reason": "insufficient_purged_chronological_folds",
        "protocol": registered,
        "diagnostic_only": True,
        "exam_labels_evaluated": False,
        "artifact_persisted": False,
        "promotion_authorized": False,
        "orders_submitted": False,
        "runtime_policy_changed": False,
        "development_input_sha256": snapshot["development_input_sha256"],
        "base_development_input_sha256": _development_input_sha256(
            development.drop(columns=list(REPLAY_METADATA_COLUMNS))
        ),
        "primary_sample_count": len(primary),
        "zero_weight_sample_count": int(primary["sample_weight"].eq(0).sum()),
        "metric_limitations": (
            "development counterfactuals, not executed portfolio PnL; native cost labels "
            "do not simulate maker fill probability; correlated trades make naive LCB "
            "insufficient for independent profitability validation"
        ),
        "folds": [], "candidates": [], "qualified_candidate_ids": [],
    }
    if primary.empty:
        return report
    folds = _folds(primary, 3)
    if len(folds) != 3 or any(
        boundary["training_positive_weight_sample_count"] < MIN_TRAINING_SAMPLE_COUNT
        for _, _, boundary in folds
    ):
        return report
    predictions: dict[str, list[dict[str, Any]]] = {
        candidate["id"]: [] for candidate in registered["candidates"]
    }
    for train, validation, boundary in folds:
        fold = {**boundary, "replay_manifests": {}}
        for features in FEATURE_POLICIES:
            kwargs = {"preprocessing": preprocessing} if preprocessing != "existing" else {}
            paired, manifests = _paired_predictions(
                train, validation, feature_policy=features, **kwargs,
            )
            fold["replay_manifests"][features] = manifests
            for policy in WEIGHT_POLICIES:
                predictions[f"{features}-{policy}"].append(paired[policy])
        report["folds"].append(fold)
    evaluation = pd.concat([validation for _, validation, _ in folds], ignore_index=True)
    report["evaluated_opportunity_count"] = len(evaluation)
    offsets = np.cumsum([0, *[len(validation) for _, validation, _ in folds]])
    for candidate in registered["candidates"]:
        chunks = predictions[candidate["id"]]
        combined = {
            task: {side: np.concatenate([chunk[task][side] for chunk in chunks])
                   for side in ("long", "short")}
            for task in ("net", "score")
        }
        sides = _selected_sides(combined["score"])
        executable = capacity_sides(evaluation, sides)
        aggregate = _views(evaluation, sides, executable)
        rejections = [f"aggregate:{reason}" for reason in _rejections(aggregate)]
        uncertainty = blocked_return_evidence(evaluation, executable)
        if not uncertainty["ready"]:
            rejections.append("capacity_calendar_block_evidence_insufficient")
        elif uncertainty["return_lcb_pct"] <= 0:
            rejections.append("capacity_calendar_block_lcb_nonpositive")
        per_fold = []
        for index, (_, validation, _) in enumerate(folds):
            selected = slice(offsets[index], offsets[index + 1])
            views = _views(validation, sides[selected], executable[selected])
            reasons = _rejections(views)
            rejections.extend(f"fold_{index}:{reason}" for reason in reasons)
            per_fold.append({"fold_index": index, "evidence": views, "rejection_reasons": reasons})
        report["candidates"].append({
            "policy": candidate,
            "development_qualified": not rejections,
            "rejection_reasons": rejections,
            "aggregate_evidence": aggregate,
            "capacity_block_uncertainty": uncertainty,
            "per_fold": per_fold,
            "prediction_diagnostics": {
                side: _prediction_diagnostics(
                    _net_returns(evaluation, side).to_numpy(), combined["net"][side],
                ) for side in ("long", "short")
            },
            "rank_only_top_20_percent": _rank_only_evidence(evaluation, combined["score"]),
            "capacity_attribution": _attribution(evaluation, executable),
        })
        if not rejections:
            report["qualified_candidate_ids"].append(candidate["id"])
    report["status"] = "development_candidate_qualified" if report["qualified_candidate_ids"] else "rejected"
    report["reason"] = "paired_comparison_complete"
    return report
