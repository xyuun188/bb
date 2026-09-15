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
