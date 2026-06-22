from services.trading_params import (
    DEFAULT_TRADING_PARAMS,
    ESTIMATED_TAKER_FEE_PCT,
    default_trading_parameter_snapshot,
)


def test_default_trading_parameter_snapshot_is_serializable() -> None:
    snapshot = default_trading_parameter_snapshot()

    assert snapshot["fee"]["estimated_taker_fee_pct"] == ESTIMATED_TAKER_FEE_PCT
    assert snapshot["execution_cost"]["default_max_slippage_pct"] == 0.005
    assert snapshot["execution_cost"]["default_paper_slippage_pct"] == 0.05
    assert snapshot["execution_cost"]["local_ml_round_trip_cost_pct"] == 0.12
    assert snapshot["execution_cost"]["local_ml_tail_loss_threshold_pct"] == 0.18
    assert snapshot["entry_tiers"]["normal_score"] == 80.0
    assert snapshot["entry_tiers"]["weak_probe_min_aligned_sources"] == 3
    assert snapshot["entry_evidence"]["positive_net_probe_min_expected_pct"] == 0.35
    assert snapshot["entry_evidence"]["short_size_multiplier"] == 0.60
    assert snapshot["entry_opportunity_gate"]["selected_side_positive_net_hard_gate"] is True
    assert snapshot["entry_opportunity_gate"]["advisory_risk_size_cap"] == 0.020
    assert (
        snapshot["entry_opportunity_gate"]["advisory_strong_quality_min_expected_net_pct"] == 0.90
    )
    assert snapshot["entry_opportunity_scoring"]["ai_expected_return_weight"] == 0.25
    assert snapshot["entry_opportunity_scoring"]["local_ml_expected_return_weight"] == 0.40
    assert snapshot["entry_opportunity_scoring"]["server_profit_expected_return_weight"] == 0.08
    assert snapshot["entry_opportunity_scoring"]["timeseries_expected_return_weight"] == 0.22
    assert snapshot["entry_opportunity_scoring"]["quant_profit_probe_min_score"] == 0.35
    assert snapshot["entry_execution_priority"]["min_entry_opportunity_score"] == 0.95
    assert snapshot["entry_execution_priority"]["exceptional_score_floor"] == 4.20
    assert snapshot["entry_execution_priority"]["positive_expectancy_min_expected_net"] == 0.35
    assert snapshot["entry_payoff_quality"]["min_expected_net_return_pct"] == 0.45
    assert snapshot["entry_market_data_quality"]["short_return_feature_timeframes"][0] == "1m"
    assert snapshot["entry_market_data_quality"]["trend_feature_timeframes"][0] == "1h"
    assert snapshot["entry_market_data_quality"]["feature_snapshot_timeout_seconds"] >= 5.0
    assert (
        snapshot["entry_market_data_quality"]["feature_snapshot_timeout_seconds"]
        > snapshot["entry_market_data_quality"]["kline_remote_fetch_timeout_seconds"]
    )
    assert snapshot["entry_feature_ranker"]["tradable_alt_min_notional_usdt"] == 1200000.0
    assert snapshot["entry_feature_ranker"]["analysis_alt_min_notional_usdt"] == 700000.0
    assert snapshot["entry_feature_ranker"]["tradable_max_day_change_pct"] == 22.0
    assert snapshot["training_data_quality"]["hold_observation_penalty"] == 0.55
    assert snapshot["training_data_quality"]["fast_loss_exit_minutes"] == 3.0
    assert snapshot["training_data_quality"]["abnormal_indicator_price_gap_pct"] == 20.0
    assert snapshot["training_data_quality"]["training_price_24h_range_tolerance_pct"] == 0.03
    assert snapshot["training_data_quality"]["allowed_sequence_timeframes"] == (
        "1m",
        "5m",
        "15m",
        "1h",
    )
    assert snapshot["local_ml_training"]["training_shadow_sample_limit"] == 20000
    assert snapshot["local_ml_training"]["auto_quarantine_batch_size"] == 1000
    assert snapshot["local_ml_training"]["auto_quarantine_max_batches"] == 5
    assert snapshot["local_ml_training"]["training_trade_sample_limit"] == 8000
    assert snapshot["local_ml_training"]["training_sequence_sample_limit"] == 12000
    assert snapshot["local_ml_training"]["local_tools_min_new_trade_samples"] == 50
    assert snapshot["local_ml_training"]["influence_min_auc"] == 0.53
    assert snapshot["local_ml_training"]["influence_min_pr_auc"] == 0.52
    assert snapshot["local_ml_training"]["readiness_max_dirty_sample_ratio"] == 0.08
    assert snapshot["local_ml_training"]["readiness_max_model_age_seconds"] == 259200
    assert snapshot["auto_scan"]["rotation_pool_multiplier"] == 20
    assert snapshot["auto_scan"]["rotation_pool_min"] == 240
    assert snapshot["auto_scan"]["feature_fetch_pool_multiplier"] == 1
    assert snapshot["auto_scan"]["feature_fetch_pool_min"] == 12
    assert snapshot["auto_scan"]["feature_fetch_timeout_seconds"] == 8.0
    assert "BTC/USDT" in snapshot["auto_scan"]["major_symbols"]
    assert snapshot["strategy_learning"]["default_lookback_hours"] == 168
    assert snapshot["strategy_learning"]["dashboard_default_limit"] == 3000
    assert snapshot["strategy_learning"]["dashboard_summary_limit"] == 1500
    assert snapshot["strategy_learning"]["dashboard_full_limit"] == 20000
    assert snapshot["strategy_learning"]["min_trade_count_target_baseline"] == 8
    assert snapshot["strategy_learning"]["market_scan_target_ratio"] == 0.08
    assert snapshot["entry_winner"]["half_life_days"] == 3.0
    assert snapshot["entry_winner"]["max_age_days"] == 14.0
    assert snapshot["entry_risk_sizing"]["low_quality_max_size"] == 0.018
    assert snapshot["entry_risk_sizing"]["ensemble_recovery_probe_size_cap"] == 0.018
    assert snapshot["entry_risk_sizing"]["balanced_probe_max_position_size_pct"] == 0.018
    assert snapshot["entry_risk_sizing"]["memory_feedback_strong_probe_size_pct"] == 0.025
    assert snapshot["entry_risk_sizing"]["recovery_probe_max_cap_pct"] == 0.06
    assert snapshot["entry_risk_sizing"]["recovery_learning_probe_max_cap_pct"] == 0.012
    assert snapshot["entry_risk_sizing"]["recovery_health_probe_max_cap_pct"] == 0.024
    assert snapshot["entry_risk_sizing"]["high_profit_min_expected_net"] == 1.60
    assert snapshot["entry_risk_sizing"]["good_probe_min_expected_net"] == 0.35
    assert snapshot["entry_quant_profit_probe"]["min_profit_quality_ratio"] == 0.12
    assert snapshot["entry_quant_profit_probe"]["strong_probe_size_pct"] == 0.06
    assert snapshot["entry_quant_profit_probe"]["stop_loss_pct"] == 0.012
    assert snapshot["entry_price_guard"]["normal_probe_allowed_move_pct"] == 0.008
    assert snapshot["entry_price_guard"]["fresh_latest_gap_floor_pct"] == 0.006
    assert snapshot["entry_account_guard"]["min_available_balance_usdt"] == 10.0
    assert snapshot["entry_account_guard"]["capacity_min_new_margin_divisor"] == 6.0
    assert snapshot["entry_crowded_side_cap"]["hard_max_side_count"] == 14
    assert snapshot["ensemble_entry_decision"]["normal_entry_score_threshold"] == 0.42
    assert snapshot["ensemble_entry_decision"]["min_executable_entry_confidence"] == 0.58
    assert snapshot["ensemble_entry_decision"]["market_direction_excluded_experts"] == (
        "position_expert",
        "risk_expert",
    )
    assert snapshot["ensemble_entry_decision"]["no_position_trend_expert_weight"] == 0.33
    assert snapshot["ensemble_entry_decision"]["add_position_max_size"] == 0.06
    assert snapshot["ensemble_entry_decision"]["winner_expand_score_threshold"] == 0.18
    assert snapshot["ensemble_exit_decision"]["fast_profit_min_hold_minutes"] == 6.0
    assert snapshot["ensemble_exit_decision"]["loss_compress_full_usdt"] == 8.0
    assert snapshot["ensemble_exit_decision"]["profit_lock_fee_multiple"] == 4.0
    assert snapshot["ensemble_exit_decision"]["loss_repair_reduce_support_count"] == 2
    assert snapshot["ensemble_ml_probe"]["ml_profit_first_min_expected_return_pct"] == 0.12
    assert snapshot["ensemble_ml_probe"]["quant_validation_min_local_expected_return_pct"] == 0.03
    assert snapshot["entry_stop_loss_budget"]["normal_budget_usdt"] == 16.0
    assert snapshot["exit_fast_risk"]["fast_risk_fresh_review_min_hold_minutes"] == 12.0
    assert snapshot["exit_cooldown"]["ordinary_seconds"] == 600.0
    assert snapshot["exit_arbitration"]["hard_risk_priority"] == 100
    assert snapshot["version"] == DEFAULT_TRADING_PARAMS.version


def test_entry_evidence_tiers_use_central_trading_params() -> None:
    from services import entry_evidence

    params = DEFAULT_TRADING_PARAMS.entry_tiers

    assert entry_evidence.ENTRY_EVIDENCE_SCORE_NORMAL == params.normal_score
    assert entry_evidence.ENTRY_EVIDENCE_SCORE_MEDIUM == params.medium_score
    assert entry_evidence.ENTRY_EVIDENCE_SCORE_SMALL == params.small_score
    assert entry_evidence.ENTRY_EVIDENCE_SCORE_PROBE == params.exploration_score
    assert entry_evidence.ENTRY_EVIDENCE_SCORE_WEAK_PROBE == params.weak_probe_score
    assert entry_evidence.ENTRY_EVIDENCE_SCORE_HARD_BLOCK == params.weak_probe_score
    assert (
        entry_evidence.ENTRY_EVIDENCE_WEAK_PROBE_MIN_ALIGNED_SOURCES
        == params.weak_probe_min_aligned_sources
    )
    assert entry_evidence.ENTRY_EVIDENCE_EXPLORATION_SIZE_CAP == params.exploration_size_cap
    assert entry_evidence.ENTRY_EVIDENCE_WEAK_CONFLICT_SIZE_CAP == params.weak_probe_size_cap


def test_entry_priority_uses_central_trading_params() -> None:
    from services import entry_priority

    params = DEFAULT_TRADING_PARAMS.entry_execution_priority

    assert entry_priority.MIN_ENTRY_OPPORTUNITY_SCORE == params.min_entry_opportunity_score


def test_entry_opportunity_scoring_uses_central_trading_params() -> None:
    from services import entry_opportunity_scoring

    params = DEFAULT_TRADING_PARAMS.entry_opportunity_scoring

    assert entry_opportunity_scoring.ENTRY_NET_WEIGHT_AI == params.ai_expected_return_weight
    assert (
        entry_opportunity_scoring.ENTRY_NET_WEIGHT_LOCAL_ML
        == params.local_ml_expected_return_weight
    )
    assert (
        entry_opportunity_scoring.ENTRY_NET_WEIGHT_SERVER_PROFIT
        == params.server_profit_expected_return_weight
    )
    assert (
        entry_opportunity_scoring.ENTRY_NET_WEIGHT_TIMESERIES
        == params.timeseries_expected_return_weight
    )
    assert (
        entry_opportunity_scoring.QUANT_PROFIT_PROBE_MIN_SCORE
        == params.quant_profit_probe_min_score
    )


def test_training_data_quality_uses_central_trading_params() -> None:
    from services import training_data_quality

    params = DEFAULT_TRADING_PARAMS.training_data_quality

    assert training_data_quality._QUALITY_PARAMS == params
    assert training_data_quality.DATA_QUALITY_VERSION.endswith(".v3")


def test_local_ml_training_uses_central_trading_params() -> None:
    from services import ml_signal_service

    params = DEFAULT_TRADING_PARAMS.local_ml_training

    assert ml_signal_service.AUTO_TRAIN_MIN_NEW_SAMPLES == params.auto_train_min_new_samples
    assert ml_signal_service.MIN_TRAINING_SAMPLES == params.min_training_samples
    assert ml_signal_service.TRAINING_SHADOW_SAMPLE_LIMIT == params.training_shadow_sample_limit
    assert ml_signal_service.ML_INFLUENCE_MIN_AUC == params.influence_min_auc
    assert ml_signal_service.ML_INFLUENCE_MIN_PR_AUC == params.influence_min_pr_auc


def test_trading_service_local_tools_training_uses_central_params() -> None:
    from services import trading_service

    params = DEFAULT_TRADING_PARAMS.local_ml_training

    assert trading_service.LOCAL_ML_TRAINING_PARAMS == params


def test_trading_service_auto_scan_uses_central_params() -> None:
    from services import trading_service

    params = DEFAULT_TRADING_PARAMS.auto_scan

    assert trading_service.AUTO_SCAN_PARAMS == params
    assert trading_service.AUTO_SCAN_ROTATION_POOL_MULTIPLIER == params.rotation_pool_multiplier
    assert trading_service.AUTO_SCAN_ROTATION_POOL_MIN == params.rotation_pool_min
    assert (
        trading_service.AUTO_SCAN_FEATURE_FETCH_POOL_MULTIPLIER
        == params.feature_fetch_pool_multiplier
    )
    assert trading_service.AUTO_SCAN_FEATURE_FETCH_POOL_MIN == params.feature_fetch_pool_min
    assert (
        trading_service.AUTO_SCAN_FEATURE_FETCH_TIMEOUT_SECONDS
        == params.feature_fetch_timeout_seconds
    )
    assert trading_service.AUTO_SCAN_FEATURE_FETCH_CONCURRENCY == params.feature_fetch_concurrency
    assert trading_service.ALT_LONG_ALLOWED_SYMBOLS == set(params.major_symbols)


def test_entry_feature_ranker_uses_central_params() -> None:
    from services.entry_feature_ranker import EntryFeatureRankerPolicy

    ranker = EntryFeatureRankerPolicy(
        suspicious_symbol_reason=lambda _symbol: None,
        min_entry_volume_ratio_provider=lambda: 0.3,
        min_entry_adx_provider=lambda: 14.0,
        major_symbols=frozenset({"BTC/USDT"}),
    )

    assert ranker.params == DEFAULT_TRADING_PARAMS.entry_feature_ranker


def test_entry_price_guard_uses_central_params() -> None:
    from services import entry_price_guard, entry_symbol_blocklist, trading_service

    params = DEFAULT_TRADING_PARAMS.entry_price_guard

    assert entry_price_guard._PRICE_GUARD_PARAMS == params
    assert entry_price_guard.PRICE_GUARD_ENTRY_BLOCK_MINUTES == params.entry_block_minutes
    assert entry_symbol_blocklist.PRICE_GUARD_ENTRY_BLOCK_MINUTES == params.entry_block_minutes
    assert trading_service.ENTRY_PRICE_RECHECK_TIMEOUT_SECONDS == params.recheck_timeout_seconds
    assert (
        entry_price_guard.ENTRY_PRICE_RECHECK_RESCUE_MAX_MOVE_PCT
        == params.recheck_rescue_max_move_pct
    )


def test_entry_crowded_side_cap_uses_central_params() -> None:
    from services.entry_crowded_side_cap import EntryCrowdedSideCapPolicy

    params = DEFAULT_TRADING_PARAMS.entry_crowded_side_cap
    policy = EntryCrowdedSideCapPolicy()

    assert policy.params == params
    assert policy.hard_max_side_count == params.hard_max_side_count
    assert policy.strong_min_expected_net_pct == params.strong_min_expected_net_pct


def test_strategy_learning_uses_central_params() -> None:
    from services import strategy_learning

    params = DEFAULT_TRADING_PARAMS.strategy_learning
    risk_params = DEFAULT_TRADING_PARAMS.entry_risk_sizing

    assert strategy_learning.STRATEGY_LEARNING_PARAMS == params
    assert strategy_learning.ENTRY_RISK_SIZING_PARAMS == risk_params
    assert strategy_learning.DEFAULT_LOOKBACK_HOURS == params.default_lookback_hours
    assert (
        strategy_learning.DEFAULT_MIN_TRADE_TARGET_FALLBACK
        == params.min_trade_count_target_baseline
    )


def test_sizing_and_probe_modules_use_central_params() -> None:
    from ai_brain import ensemble_coordinator
    from services import market_decision_risk_assessment, memory_feedback

    risk_params = DEFAULT_TRADING_PARAMS.entry_risk_sizing
    entry_params = DEFAULT_TRADING_PARAMS.ensemble_entry_decision
    exit_params = DEFAULT_TRADING_PARAMS.ensemble_exit_decision
    ml_probe_params = DEFAULT_TRADING_PARAMS.ensemble_ml_probe

    assert (
        ensemble_coordinator.NORMAL_ENTRY_SCORE_THRESHOLD
        == entry_params.normal_entry_score_threshold
    )
    assert (
        ensemble_coordinator.PROBE_ENTRY_SCORE_THRESHOLD == entry_params.probe_entry_score_threshold
    )
    assert ensemble_coordinator.MAX_ENTRY_DISAGREEMENT == entry_params.max_entry_disagreement
    assert ensemble_coordinator.MARKET_DIRECTION_EXCLUDED_EXPERTS == set(
        entry_params.market_direction_excluded_experts
    )
    assert (
        ensemble_coordinator.NO_POSITION_ENTRY_BASE_WEIGHTS["trend_expert"]
        == entry_params.no_position_trend_expert_weight
    )
    assert (
        ensemble_coordinator.MIN_EXECUTABLE_ENTRY_CONFIDENCE
        == entry_params.min_executable_entry_confidence
    )
    assert ensemble_coordinator.ADD_POSITION_MAX_SIZE == entry_params.add_position_max_size
    assert (
        ensemble_coordinator.WINNER_EXPAND_SCORE_THRESHOLD
        == entry_params.winner_expand_score_threshold
    )
    assert ensemble_coordinator.MAX_PROBE_ENTRY_SIZE == risk_params.ensemble_max_probe_entry_size
    assert (
        ensemble_coordinator.ML_SOFT_CAUTION_MAX_ENTRY_SIZE
        == risk_params.ensemble_ml_soft_caution_max_entry_size
    )
    assert (
        ensemble_coordinator.QUANT_VALIDATION_PROBE_SIZE
        == risk_params.ensemble_quant_validation_probe_size
    )
    assert (
        market_decision_risk_assessment.BALANCED_PROBE_MAX_POSITION_SIZE_PCT
        == risk_params.balanced_probe_max_position_size_pct
    )
    assert memory_feedback.ENTRY_RISK_SIZING_PARAMS == risk_params
    assert (
        ensemble_coordinator.PROFIT_PROTECT_REDUCE_PNL_RATIO
        == exit_params.profit_protect_reduce_pnl_ratio
    )
    assert (
        ensemble_coordinator.FAST_PROFIT_MIN_HOLD_MINUTES
        == exit_params.fast_profit_min_hold_minutes
    )
    assert ensemble_coordinator.LOSS_COMPRESS_FULL_USDT == exit_params.loss_compress_full_usdt
    assert ensemble_coordinator.PROFIT_LOCK_FEE_MULTIPLE == exit_params.profit_lock_fee_multiple
    assert (
        ensemble_coordinator.LOSS_REPAIR_REDUCE_SUPPORT_COUNT
        == exit_params.loss_repair_reduce_support_count
    )
    assert (
        ensemble_coordinator.PREDICTIVE_REVERSAL_EXIT_SCORE
        == exit_params.predictive_reversal_exit_score
    )
    assert (
        ensemble_coordinator.ML_PROFIT_FIRST_MIN_EXPECTED_RETURN_PCT
        == ml_probe_params.ml_profit_first_min_expected_return_pct
    )
    assert (
        ensemble_coordinator.QUANT_VALIDATION_MIN_LOCAL_EXPECTED_RETURN_PCT
        == ml_probe_params.quant_validation_min_local_expected_return_pct
    )


def test_entry_opportunity_gate_uses_central_advisory_params() -> None:
    from services import entry_opportunity_gate

    params = DEFAULT_TRADING_PARAMS.entry_opportunity_gate

    assert entry_opportunity_gate.ADVISORY_RISK_SIZE_CAP == params.advisory_risk_size_cap
    assert (
        entry_opportunity_gate.ADVISORY_DIRECTION_CONFLICT_SIZE_CAP
        == params.advisory_direction_conflict_size_cap
    )
    assert (
        entry_opportunity_gate.ADVISORY_STRONG_QUALITY_MIN_EXPECTED_NET_PCT
        == params.advisory_strong_quality_min_expected_net_pct
    )
