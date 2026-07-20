from services.trading_params import (
    DEFAULT_TRADING_PARAMS,
    default_trading_parameter_snapshot,
)


def test_parameter_snapshot_contains_operations_and_data_quality_only() -> None:
    snapshot = default_trading_parameter_snapshot()

    assert snapshot["version"] == DEFAULT_TRADING_PARAMS.version
    assert "entry_feature_ranker" not in snapshot
    assert "entry_tiers" not in snapshot
    assert "entry_evidence" not in snapshot
    assert "entry_risk_sizing" not in snapshot
    assert "entry_quant_profit_probe" not in snapshot
    assert "ensemble_ml_probe" not in snapshot
    assert "execution_cost" not in snapshot
    assert "fast_loss_exit_minutes" not in snapshot["training_data_quality"]
    assert "fee_dominated_multiple" not in snapshot["training_data_quality"]
    assert "training_shadow_sample_limit" not in snapshot["local_ml_training"]
    assert snapshot["entry_market_data_quality"]["feature_snapshot_timeout_seconds"] > 0


def test_training_data_quality_uses_the_current_snapshot() -> None:
    from services import training_data_quality

    assert training_data_quality._QUALITY_PARAMS == DEFAULT_TRADING_PARAMS.training_data_quality
    assert (
        training_data_quality.DATA_QUALITY_VERSION
        == "2026-07-14.separated-profit-supervision.v4"
    )


def test_auto_scan_snapshot_is_operational_not_entry_permission() -> None:
    from services import trading_service

    params = DEFAULT_TRADING_PARAMS.auto_scan
    assert trading_service.AUTO_SCAN_PARAMS == params
    assert trading_service.AUTO_SCAN_FEATURE_FETCH_TIMEOUT_SECONDS == (
        params.feature_fetch_timeout_seconds
    )
    assert "BTC/USDT" in params.major_symbols
    selection = DEFAULT_TRADING_PARAMS.market_analysis_selection
    assert selection.candidate_pool_multiplier >= 2
    assert selection.discovery_slots == 1
    assert selection.cooldown_seconds > 0
    assert 0 < selection.unchanged_repeat_penalty_ratio < 1
