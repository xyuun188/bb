import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from services import ml_prediction_contract, ml_signal_service


def test_prediction_contract_owns_extracted_functions() -> None:
    extracted_functions = {
        "regression_prediction_distribution",
        "standardized_model_return_distribution",
        "risk_adjusted_expected_scores",
        "profit_quality_score",
    }
    private_legacy_names = {f"_{name}" for name in extracted_functions}

    assert private_legacy_names.isdisjoint(ml_signal_service.__dict__)
    assert extracted_functions <= ml_prediction_contract.__dict__.keys()


def test_agreeing_trees_preserve_outcome_risk_and_reject_false_positive_edge() -> None:
    x = pd.DataFrame({"feature": [0.0] * 20})
    model = Pipeline([
        ("imputer", SimpleImputer()),
        ("model", RandomForestRegressor(n_estimators=4, bootstrap=False, random_state=42)),
    ]).fit(x, [-2.0, 4.0] * 10)

    distribution = ml_prediction_contract.regression_prediction_distribution(model, x.iloc[:1])

    assert distribution["expected"][0] == pytest.approx(1.0)
    assert distribution["between_tree_std"][0] == pytest.approx(0.0)
    assert distribution["within_leaf_std"][0] == pytest.approx(3.0)
    assert distribution["std"][0] == pytest.approx(3.0)
    assert distribution["dispersion_basis"] == "tree_leaf_total_variance"
    scores = ml_prediction_contract.risk_adjusted_expected_scores(
        distribution,
        {"expected": np.array([0.1]), "upper_quantile": np.array([0.1])},
        np.array([0.0]),
        tail_loss_scale_pct=0.0,
    )
    assert scores[0] == pytest.approx(-2.1)


def test_leaf_risk_is_conditional_and_batch_matches_single_predictions() -> None:
    x = pd.DataFrame({"feature": [0.0] * 20 + [1.0] * 20})
    y = [1.0] * 20 + [-2.0, 6.0] * 10
    model = Pipeline([
        ("imputer", SimpleImputer()),
        ("model", RandomForestRegressor(n_estimators=4, bootstrap=False, random_state=42)),
    ]).fit(x, y)
    probes = x.iloc[[0, -1]]
    batch = ml_prediction_contract.regression_prediction_distribution(model, probes)
    assert batch["std"] == pytest.approx([0.0, 4.0])
    for index in range(len(probes)):
        single = ml_prediction_contract.regression_prediction_distribution(model, probes.iloc[[index]])
        assert single["std"][0] == pytest.approx(batch["std"][index])
