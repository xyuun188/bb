from services import ml_signal_service, ml_training_dataset


def test_ml_training_dataset_owns_database_access_functions() -> None:
    owned = {
        "load_shadow_training_rows",
        "count_shadow_training_rows",
        "count_shadow_training_decision_groups",
        "load_authoritative_trade_training_samples",
        "select_shadow_training_rows",
    }

    assert owned <= ml_training_dataset.__dict__.keys()
    assert owned.isdisjoint(ml_signal_service.__dict__)
