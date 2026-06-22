"""Central trading policy parameters and runtime snapshots.

This module is the first step toward moving tunable strategy numbers out of
the orchestration layer.  Keep runtime defaults here, while environment
settings remain in config.settings and backtests can persist the snapshot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ai_brain.base_model import DecisionOutput

ESTIMATED_TAKER_FEE_PCT = 0.0005


@dataclass(frozen=True, slots=True)
class FeeParams:
    estimated_taker_fee_pct: float = ESTIMATED_TAKER_FEE_PCT


@dataclass(frozen=True, slots=True)
class ExecutionCostParams:
    default_max_slippage_pct: float = 0.005
    default_paper_slippage_pct: float = 0.05
    min_execution_slippage_pct: float = 0.02
    liquidity_depth_reference_usdt: float = 50_000.0
    local_ml_round_trip_cost_pct: float = 0.12
    local_ml_tail_loss_threshold_pct: float = 0.18


@dataclass(frozen=True, slots=True)
class EntryTierParams:
    normal_score: float = 80.0
    medium_score: float = 70.0
    small_score: float = 60.0
    exploration_score: float = 45.0
    weak_probe_score: float = 35.0
    weak_probe_min_aligned_sources: int = 3
    exploration_size_cap: float = 0.015
    weak_probe_size_cap: float = 0.010


@dataclass(frozen=True, slots=True)
class EntryEvidenceParams:
    short_score_offset: float = 10.0
    short_size_multiplier: float = 0.60
    major_conflict_size_cap: float = 0.025
    missing_key_size_cap: float = 0.018
    weak_opposite_return_pct: float = 0.15
    strong_opposite_return_pct: float = 0.35
    weak_opposite_penalty_ratio: float = 0.20
    normal_opposite_penalty_ratio: float = 0.70
    short_probe_relief_min_base_score: float = 35.0
    short_probe_relief_min_effective_score: float = 30.0
    short_probe_relief_min_direction_gap: float = 0.08
    short_probe_relief_max_loss_probability: float = 0.58
    positive_net_probe_min_expected_pct: float = 0.35
    positive_net_probe_min_confidence: float = 0.62
    positive_net_probe_min_profit_quality: float = 0.20
    positive_net_probe_max_loss_probability: float = 0.62
    positive_net_probe_max_tail_risk: float = 0.95
    strong_positive_relief_min_expected_pct: float = 1.20
    strong_positive_relief_min_confidence: float = 0.78
    strong_positive_relief_min_profit_quality: float = 1.20
    strong_positive_relief_max_loss_probability: float = 0.45
    strong_positive_relief_max_tail_risk: float = 0.75
    strong_positive_relief_min_opportunity_score: float = 2.80
    strong_positive_relief_min_aligned_sources: int = 2
    elite_positive_relief_min_expected_pct: float = 1.80
    elite_positive_relief_min_confidence: float = 0.86
    elite_positive_relief_min_profit_quality: float = 2.00
    elite_positive_relief_max_loss_probability: float = 0.42
    elite_positive_relief_max_tail_risk: float = 0.65
    elite_positive_relief_min_opportunity_score: float = 4.20


@dataclass(frozen=True, slots=True)
class EntryOpportunityGateParams:
    direction_hard_conflict_gap: float = 0.12
    direction_min_support_score: float = 0.02
    min_net_profit_quality_ratio: float = 1.50
    advisory_direction_conflict_size_cap: float = 0.025
    advisory_risk_size_cap: float = 0.020
    advisory_low_quality_size_cap: float = 0.015
    advisory_strong_quality_min_expected_net_pct: float = 0.90
    advisory_strong_quality_min_profit_quality_ratio: float = 0.85
    advisory_strong_quality_max_loss_probability: float = 0.48
    advisory_strong_quality_max_tail_risk: float = 0.88
    quant_profit_probe_min_expected_pct: float = 0.18
    portfolio_roster_fill_min_expected_pct: float = 0.08
    portfolio_roster_fill_max_loss_probability: float = 0.66
    portfolio_roster_fill_min_net_pct: float = 0.20
    portfolio_roster_fill_min_profit_quality_ratio: float = 0.25
    selected_side_positive_net_hard_gate: bool = True


@dataclass(frozen=True, slots=True)
class EntryOpportunityScoringParams:
    """Expected-net-return scoring weights and historical risk penalties."""

    ml_expected_return_score_cap_pct: float = 3.0
    ai_expected_return_weight: float = 0.25
    local_ml_expected_return_weight: float = 0.40
    server_profit_expected_return_weight: float = 0.08
    timeseries_expected_return_weight: float = 0.22
    small_win_big_loss_penalty_cap: float = 0.90
    realized_edge_bonus_cap: float = 0.85
    realized_edge_penalty_cap: float = 1.15
    min_net_profit_quality_ratio: float = 1.50
    weak_history_min_profit_quality_ratio: float = 2.00
    strong_aligned_min_profit_quality_ratio: float = 0.85
    weak_history_strong_aligned_min_profit_quality_ratio: float = 1.05
    weak_history_min_score: float = 3.20
    dynamic_entry_score_ml_aligned_strong: float = 0.75
    dynamic_entry_score_ml_aligned: float = 0.85
    dynamic_entry_score_expert_aligned: float = 0.90
    quant_profit_probe_min_expected_pct: float = 0.18
    quant_profit_probe_min_score: float = 0.35
    abnormal_wick_tail_risk_max_pct: float = 60.0
    shadow_memory_expected_return_max_pct: float = 0.35
    shadow_memory_expected_return_weight: float = 1.0
    shadow_memory_min_missed_count: int = 2
    shadow_memory_max_risk_evidence_ratio: float = 0.60


@dataclass(frozen=True, slots=True)
class EntryExecutionPriorityParams:
    min_entry_opportunity_score: float = 0.95
    exceptional_score_delta: float = 2.0
    exceptional_score_floor: float = 4.20
    exceptional_min_confidence: float = 0.86
    exceptional_min_expected_net: float = 1.20
    exceptional_min_profit_quality: float = 1.50
    strong_score_delta: float = 1.00
    strong_score_floor: float = 2.80
    strong_min_confidence: float = 0.88
    strong_min_expected_net: float = 0.80
    strong_min_profit_quality: float = 1.20
    strong_min_entry_votes: int = 2
    strong_quant_min_confidence: float = 0.78
    strong_quant_min_expected_net: float = 0.75
    strong_quant_min_profit_quality: float = 0.85
    strong_quant_max_loss_probability: float = 0.42
    medium_quant_score_floor: float = 0.35
    medium_quant_min_confidence: float = 0.66
    medium_quant_min_expected_net: float = 0.25
    medium_quant_min_profit_quality: float = 0.20
    medium_quant_max_loss_probability: float = 0.52
    medium_quant_max_position_size: float = 0.03
    positive_expectancy_score_delta: float = -0.15
    positive_expectancy_score_floor: float = 0.55
    positive_expectancy_min_confidence: float = 0.68
    positive_expectancy_min_expected_net: float = 0.35
    positive_expectancy_min_profit_quality: float = 0.30
    positive_expectancy_max_loss_probability: float = 0.58
    positive_expectancy_max_tail_risk: float = 0.88
    roster_fill_score_floor: float = 0.25
    roster_fill_min_confidence: float = 0.66
    roster_fill_min_expected_net: float = 0.30
    roster_fill_min_profit_quality: float = 0.30
    roster_fill_max_loss_probability: float = 0.62
    roster_fill_max_position_size: float = 0.025


@dataclass(frozen=True, slots=True)
class EntryPayoffQualityParams:
    min_expected_net_return_pct: float = 0.45
    min_profit_quality_ratio: float = 0.75
    max_small_win_big_loss_penalty: float = 0.65


@dataclass(frozen=True, slots=True)
class EntryMarketDataQualityParams:
    price_field_split_block_pct: float = 0.08
    price_24h_range_tolerance_pct: float = 0.03
    stale_zero_returns_min_24h_change: float = 0.003
    default_max_slippage_pct: float = 0.005
    min_indicator_rows: int = 21
    kline_cache_max_age_multiplier: float = 3.0
    kline_cache_min_max_age_seconds: float = 180.0
    feature_snapshot_timeout_seconds: float = 5.0
    kline_remote_fetch_timeout_seconds: float = 3.5
    indicator_snapshot_cache_ttl_seconds: float = 30.0
    kline_background_refresh_min_interval_seconds: float = 45.0
    kline_coverage_refresh_interval_seconds: float = 60.0
    kline_coverage_refresh_batch_size: int = 4
    kline_coverage_refresh_symbol_cap: int = 80
    kline_coverage_initial_delay_seconds: float = 5.0
    indicator_remote_refresh_concurrency: int = 3
    derivatives_stale_max_age_seconds: float = 180.0
    short_return_feature_timeframes: tuple[str, ...] = ("1m", "5m", "15m", "1h")
    trend_feature_timeframes: tuple[str, ...] = ("1h", "15m", "5m", "1m")


@dataclass(frozen=True, slots=True)
class EntryFeatureRankerParams:
    """Auto-scan liquidity and volatility filters.

    These values decide which symbols are worth spending model tokens on.
    Keeping them in the strategy snapshot prevents hidden fixed thresholds
    from silently starving the scheduler.
    """

    liquidity_log_weight: float = 10.0
    participation_weight: float = 10.0
    volume_ratio_score_cap: float = 5.0
    adx_score_cap: float = 50.0
    adx_weight: float = 0.8
    momentum_returns_1_weight: float = 1_200.0
    momentum_returns_5_weight: float = 700.0
    momentum_returns_20_weight: float = 350.0
    momentum_score_cap: float = 45.0
    day_move_cap_pct: float = 12.0
    day_move_weight: float = 1.6
    volatility_weight: float = 900.0
    volatility_score_cap: float = 30.0
    trend_distance_weight: float = 600.0
    trend_distance_cap: float = 25.0
    bollinger_extreme_low: float = 0.18
    bollinger_extreme_high: float = 0.82
    bollinger_extreme_bonus: float = 8.0
    low_activity_penalty: float = 80.0
    extreme_volatility_penalty: float = 45.0
    elevated_volatility_penalty: float = 18.0
    elevated_volatility_threshold: float = 0.08
    extreme_volatility_threshold: float = 0.12
    extreme_volatility_day_move_pct: float = 8.0
    tradable_major_min_notional_usdt: float = 800_000.0
    tradable_alt_min_notional_usdt: float = 1_200_000.0
    tradable_volume_provider_floor: float = 0.16
    tradable_volume_multiplier: float = 0.55
    tradable_volume_cap: float = 0.42
    tradable_volume_floor: float = 0.18
    tradable_adx_provider_offset: float = 6.0
    tradable_adx_provider_floor: float = 8.0
    tradable_adx_cap: float = 16.0
    tradable_adx_floor: float = 10.0
    tradable_max_volatility: float = 0.12
    tradable_max_day_change_pct: float = 22.0
    analysis_major_min_notional_usdt: float = 500_000.0
    analysis_alt_min_notional_usdt: float = 700_000.0
    analysis_volume_provider_floor: float = 0.12
    analysis_volume_multiplier: float = 0.25
    analysis_volume_cap: float = 0.24
    analysis_volume_floor: float = 0.05
    analysis_adx_provider_offset: float = 9.0
    analysis_adx_provider_floor: float = 6.0
    analysis_adx_cap: float = 14.0
    analysis_adx_floor: float = 8.0
    analysis_max_volatility: float = 0.18
    analysis_max_day_change_pct: float = 32.0


@dataclass(frozen=True, slots=True)
class TrainingDataQualityParams:
    """Quality policy for samples used by local ML and server local tools.

    These values are deliberately kept in the auditable strategy snapshot so
    old hidden training gates cannot silently teach the system to over-hold,
    over-probe, or reuse manually contaminated trades.
    """

    include_score_threshold: float = 0.75
    downweighted_min_weight: float = 0.20
    downweighted_max_weight: float = 0.85
    excluded_score_cap: float = 0.20
    abnormal_shadow_return_abs_pct: float = 50.0
    hold_missed_opportunity_penalty: float = 0.18
    hold_observation_penalty: float = 0.55
    very_low_confidence_threshold: float = 0.05
    very_low_confidence_penalty: float = 0.22
    max_horizon_minutes: int = 1440
    invalid_horizon_penalty: float = 0.35
    invalid_price_penalty: float = 0.25
    abnormal_indicator_price_gap_pct: float = 20.0
    training_price_24h_range_tolerance_pct: float = 0.03
    abnormal_spread_pct: float = 2.0
    wide_spread_pct: float = 0.50
    wide_spread_penalty: float = 0.20
    invalid_trade_price_penalty: float = 0.35
    fast_loss_exit_minutes: float = 3.0
    fast_loss_exit_penalty: float = 0.45
    missing_hold_duration_penalty: float = 0.20
    fee_dominated_multiple: float = 1.20
    fee_dominated_penalty: float = 0.18
    unknown_outcome_penalty: float = 0.12
    min_sequence_length: int = 30
    abnormal_future_return_abs_pct: float = 30.0
    unknown_timeframe_penalty: float = 0.10
    min_text_length: int = 12
    unknown_text_source_penalty: float = 0.15
    missing_sentiment_penalty: float = 0.25
    allowed_sequence_timeframes: tuple[str, ...] = ("1m", "5m", "15m", "1h")
    manual_trade_sources: tuple[str, ...] = ("manual", "manual_trade", "test")


@dataclass(frozen=True, slots=True)
class LocalMLTrainingParams:
    """Local ML training cadence and influence gates."""

    auto_train_check_interval_seconds: int = 30 * 60
    auto_train_min_interval_seconds: int = 6 * 60 * 60
    auto_train_min_new_samples: int = 500
    auto_train_learning_only_interval_seconds: int = 2 * 60 * 60
    auto_train_learning_only_min_new_samples: int = 120
    auto_quarantine_batch_size: int = 1000
    auto_quarantine_max_batches: int = 5
    min_training_samples: int = 200
    training_shadow_sample_limit: int = 20_000
    training_trade_sample_limit: int = 8_000
    training_sequence_sample_limit: int = 12_000
    training_text_sample_limit: int = 8_000
    local_tools_learning_only_min_new_trade_samples: int = 15
    local_tools_min_new_trade_samples: int = 50
    win_return_threshold_pct: float = 0.05
    min_profit_edge_pct: float = 0.02
    min_profit_signal_win_rate: float = 0.0
    influence_min_sample_count: int = 1_000
    influence_min_test_count: int = 200
    influence_min_auc: float = 0.53
    influence_min_accuracy: float = 0.52
    train_split_ratio: float = 0.75
    min_test_rows: int = 40


@dataclass(frozen=True, slots=True)
class AutoScanParams:
    """Market scan breadth and major-symbol policy.

    These numbers affect how many symbols the scheduler can inspect per round.
    They are not entry gates, but keeping them in the snapshot makes scan
    breadth auditable when diagnosing "no open positions" symptoms.
    """

    rotation_pool_multiplier: int = 20
    rotation_pool_min: int = 240
    feature_fetch_pool_multiplier: int = 1
    feature_fetch_pool_min: int = 12
    feature_fetch_timeout_seconds: float = 8.0
    feature_fetch_concurrency: int = 8
    major_symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")


@dataclass(frozen=True, slots=True)
class StrategyLearningParams:
    """Strategy-learning windows and advisory sample-target policy.

    These values describe how much evidence the scheduler reviews and how it
    computes learning confidence.  They are not direct entry gates.
    """

    default_lookback_hours: int = 168
    max_lookback_hours: int = 24 * 90
    dashboard_default_limit: int = 3_000
    dashboard_summary_limit: int = 1_500
    dashboard_full_limit: int = 20_000
    min_dashboard_limit: int = 100
    runtime_context_timeout_seconds: float = 3.0
    runtime_context_row_limit: int = 800
    runtime_context_cache_ttl_seconds: float = 600.0
    runtime_perf_timeout_seconds: float = 1.2
    runtime_account_timeout_seconds: float = 1.0
    market_event_limit_multiplier: int = 3
    expert_memory_limit: int = 2_000
    expert_memory_lookback_days: int = 30
    order_extra_lookback_hours: int = 2
    min_trade_count_target_baseline: int = 8
    min_trade_count_target_min: int = 1
    min_trade_count_target_settings_cap: int = 80
    dynamic_trade_target_min: int = 3
    dynamic_trade_target_max: int = 40
    dynamic_trade_target_reference_hours: int = 168
    dynamic_trade_target_min_window_factor: float = 0.50
    dynamic_trade_target_max_window_factor: float = 2.00
    entry_signal_target_ratio: float = 0.60
    shadow_opportunity_target_ratio: float = 0.35
    reflection_target_ratio: float = 0.50
    market_scan_target_ratio: float = 0.08
    entry_volume_ratio_min: float = 0.05
    entry_volume_ratio_default: float = 0.20
    entry_volume_ratio_max: float = 0.55
    entry_adx_min: float = 6.0
    entry_adx_default: float = 15.0
    entry_adx_max: float = 24.0
    entry_filter_profit_tighten_factor: float = 0.85
    entry_filter_loss_relax_factor: float = 0.70
    entry_filter_release_relax_factor: float = 0.80


@dataclass(frozen=True, slots=True)
class EntryWinnerParams:
    min_count: int = 2
    min_pnl_usdt: float = 5.0
    min_profit_factor: float = 1.20
    score_relief: float = 0.12
    score_bonus_cap: float = 0.45
    half_life_days: float = 3.0
    max_age_days: float = 14.0
    missing_timestamp_weight: float = 0.50


@dataclass(frozen=True, slots=True)
class EntryLossCooldownParams:
    side_cooldown_loss_usdt: float = 25.0
    total_cooldown_loss_usdt: float = 60.0
    quarantine_loss_usdt: float = 120.0
    quarantine_min_losses: int = 2
    hard_cooldown_hours: float = 6.0
    override_min_confidence: float = 0.72
    override_min_score: float = 3.20
    override_score_multiple: float = 1.45
    override_min_expected_net: float = 0.90
    override_min_profit_quality: float = 0.90
    override_min_reward_risk: float = 1.30
    override_min_server_expected: float = 0.80
    override_max_loss_probability: float = 0.48
    override_max_tail_risk: float = 0.85
    fresh_loss_reentry_cooldown_hours: float = 2.0
    fresh_loss_reentry_min_confidence: float = 0.76
    fresh_loss_reentry_score_multiple: float = 1.75
    fresh_loss_reentry_min_expected_net: float = 1.20
    fresh_loss_reentry_min_profit_quality: float = 1.10
    fresh_loss_reentry_min_reward_risk: float = 1.45
    fresh_loss_reentry_max_loss_probability: float = 0.42
    fresh_loss_reentry_min_aligned_sources: int = 3


@dataclass(frozen=True, slots=True)
class EntryProbeMarketQualityParams:
    max_price_field_gap: float = 0.03
    strong_contra_20m_pct: float = 0.05
    min_volume_ratio: float = 0.02


@dataclass(frozen=True, slots=True)
class EntryQuantProfitProbeParams:
    min_expected_pct: float = 0.18
    min_edge_pct: float = 0.22
    min_profit_quality_ratio: float = 0.12
    min_concentrated_loss_usdt: float = 5.0
    default_max_loss_probability: float = 0.58
    roster_fill_min_expected_pct: float = 0.08
    roster_fill_min_edge_pct: float = 0.08
    roster_fill_max_loss_probability: float = 0.66
    stop_loss_pct: float = 0.012
    strong_min_reward_risk: float = 3.60
    normal_min_reward_risk: float = 2.80
    strong_take_profit_cap_pct: float = 0.085
    normal_take_profit_cap_pct: float = 0.065
    strong_probe_size_pct: float = 0.060
    roster_fill_probe_size_pct: float = 0.020
    normal_probe_size_pct: float = 0.025
    strong_probe_leverage: float = 5.0
    normal_probe_leverage: float = 3.0


@dataclass(frozen=True, slots=True)
class EntryRiskSizingParams:
    ensemble_max_normal_entry_size: float = 0.12
    ensemble_max_probe_entry_size: float = 0.030
    ensemble_ml_soft_caution_max_entry_size: float = 0.025
    ensemble_profit_first_probe_size: float = 0.025
    ensemble_quant_validation_probe_size: float = 0.040
    ensemble_recovery_probe_size_cap: float = 0.018
    ensemble_crowded_side_size_cap: float = 0.025
    ensemble_profit_expand_min_entry_size: float = 0.055
    ensemble_profit_expand_recovery_max_size: float = 0.090
    ensemble_profit_expand_selective_max_size: float = 0.075
    ensemble_profit_expand_normal_max_size: float = 0.110
    ensemble_profit_expand_crowded_max_size: float = 0.070
    balanced_probe_position_size_multiplier: float = 0.62
    balanced_probe_max_position_size_pct: float = 0.018
    memory_feedback_strong_probe_size_pct: float = 0.025
    memory_feedback_normal_probe_size_pct: float = 0.015
    memory_feedback_light_probe_size_pct: float = 0.012
    min_net_profit_quality_ratio: float = 1.50
    weak_history_max_size: float = 0.025
    weak_history_max_leverage: float = 5.0
    weak_history_strong_aligned_max_size: float = 0.045
    weak_history_strong_aligned_max_leverage: float = 8.0
    negative_local_expected_max_size: float = 0.02
    negative_local_expected_max_leverage: float = 4.0
    low_quality_max_size: float = 0.018
    low_quality_max_leverage: float = 3.0
    symbol_loser_size_multiplier: float = 0.55
    high_quality_min_notional_balance_ratio: float = 0.10
    normal_min_notional_balance_ratio: float = 0.06
    notional_floor_max_size_pct: float = 0.12
    high_profit_min_notional_balance_ratio: float = 0.75
    high_profit_min_leverage: float = 8.0
    high_profit_elite_min_leverage: float = 10.0
    good_probe_min_notional_balance_ratio: float = 0.25
    winner_add_min_notional_balance_ratio: float = 0.35
    strong_probe_min_notional_balance_ratio: float = 0.45
    elite_min_notional_balance_ratio: float = 0.60
    meaningful_size_max_tail_risk: float = 0.82
    meaningful_size_min_profit_usdt: float = 0.75
    meaningful_size_min_profit_ratio: float = 0.003
    portfolio_roster_fill_max_loss_probability: float = 0.66
    portfolio_roster_fill_min_net_pct: float = 0.20
    portfolio_roster_fill_min_profit_quality_ratio: float = 0.25
    portfolio_roster_fill_notional_balance_ratio: float = 0.18
    pnl_structure_min_expected_profit_usdt: float = 1.50
    pnl_structure_low_quality_max_loss_multiple: float = 0.65
    pnl_structure_normal_max_loss_multiple: float = 1.05
    pnl_structure_high_quality_max_loss_multiple: float = 1.35
    balanced_probe_max_loss_usdt: float = 5.0
    strong_probe_max_loss_usdt: float = 9.0
    quality_risk_base_cap_pct: float = 0.008
    quality_risk_max_cap_pct: float = 0.024
    quality_risk_elite_cap_pct: float = 0.030
    recovery_probe_base_cap_pct: float = 0.012
    recovery_probe_max_cap_pct: float = 0.060
    strategy_sizing_min_multiplier: float = 0.10
    strategy_sizing_max_multiplier: float = 1.25
    release_probe_fraction_floor: float = 0.030
    release_probe_default_cap_pct: float = 0.012
    release_probe_min_cap_pct: float = 0.008
    release_probe_max_cap_pct: float = 0.018
    recovery_multiplier_cap: float = 0.35
    recovery_probe_fraction_floor: float = 0.020
    recovery_probe_default_cap_pct: float = 0.010
    recovery_probe_min_cap_pct: float = 0.006
    recovery_learning_probe_max_cap_pct: float = 0.012
    recovery_health_probe_max_cap_pct: float = 0.024
    strategy_probe_fraction_max: float = 0.10
    strategy_probe_cap_max_pct: float = 0.030
    adaptive_quality_expected_component_cap: float = 0.012
    adaptive_quality_profit_quality_component_cap: float = 0.006
    adaptive_quality_probability_component_cap: float = 0.006
    adaptive_quality_tail_discount_cap: float = 0.006
    adaptive_recovery_min_profit_quality: float = 0.75
    adaptive_recovery_max_loss_probability: float = 0.58
    adaptive_recovery_max_tail_risk: float = 0.90
    adaptive_recovery_expected_component_cap: float = 0.018
    adaptive_recovery_profit_quality_component_cap: float = 0.018
    adaptive_recovery_score_component_cap: float = 0.012
    adaptive_recovery_alignment_component_cap: float = 0.012
    adaptive_recovery_loss_discount_anchor: float = 0.35
    adaptive_recovery_tail_discount_anchor: float = 0.55
    strong_probe_min_expected_net: float = 0.75
    strong_probe_min_profit_quality: float = 0.85
    strong_probe_max_loss_probability: float = 0.42
    strong_probe_min_score_floor: float = 2.80
    elite_min_expected_net: float = 1.20
    elite_min_profit_quality: float = 1.20
    elite_max_loss_probability: float = 0.38
    high_profit_min_expected_net: float = 1.60
    high_profit_min_profit_quality: float = 1.45
    high_profit_max_loss_probability: float = 0.34
    high_profit_max_tail_risk: float = 0.72
    high_profit_min_score_floor: float = 1.15
    high_profit_elite_expected_net: float = 2.20
    high_profit_elite_profit_quality: float = 1.80
    high_profit_elite_max_loss_probability: float = 0.30
    quality_override_min_expected_net: float = 0.75
    quality_override_min_profit_quality: float = 0.75
    quality_override_max_loss_probability: float = 0.48
    good_probe_min_expected_net: float = 0.35
    good_probe_min_profit_quality: float = 0.20
    good_probe_max_loss_probability: float = 0.52
    same_side_winner_min_expected_net: float = 0.55
    same_side_winner_min_profit_quality: float = 0.55
    same_side_winner_max_loss_probability: float = 0.48
    winner_notional_lift_multiplier: float = 1.15
    high_quality_floor_min_multiplier: float = 0.75
    high_quality_floor_max_multiplier: float = 1.35
    normal_positive_min_profit_quality: float = 0.65
    normal_positive_floor_min_multiplier: float = 0.75
    normal_positive_floor_max_multiplier: float = 1.25


@dataclass(frozen=True, slots=True)
class EntryPriceGuardParams:
    """Pre-order price drift and rescue thresholds."""

    recheck_timeout_seconds: float = 5.0
    recheck_rescue_max_move_pct: float = 0.012
    recheck_exceptional_max_move_pct: float = 0.020
    recheck_expected_buffer_multiple: float = 2.0
    entry_block_minutes: float = 8.0
    min_allowed_slippage_pct: float = 0.003
    max_allowed_slippage_pct: float = 0.020
    strong_probe_expected_net: float = 1.20
    strong_probe_profit_quality: float = 1.20
    strong_probe_allowed_move_pct: float = 0.012
    normal_probe_expected_net: float = 0.35
    normal_probe_profit_quality: float = 0.20
    normal_probe_allowed_move_pct: float = 0.008
    exceptional_expected_net: float = 4.0
    exceptional_profit_quality: float = 2.0
    exceptional_allowed_move_pct: float = 0.020
    strong_expected_net: float = 2.0
    strong_profit_quality: float = 1.20
    strong_allowed_move_pct: float = 0.016
    normal_expected_net: float = 1.0
    normal_profit_quality: float = 0.80
    normal_allowed_move_pct: float = 0.012
    medium_expected_buffer_multiple: float = 1.35
    strong_expected_buffer_multiple: float = 1.05
    fresh_long_returns_1_floor: float = -0.002
    fresh_long_returns_5_floor: float = -0.004
    fresh_short_returns_1_ceiling: float = 0.002
    fresh_short_returns_5_ceiling: float = 0.004
    fresh_latest_gap_floor_pct: float = 0.006


@dataclass(frozen=True, slots=True)
class EntryAccountGuardParams:
    """Account-level scan guards that must stay advisory and auditable."""

    min_available_balance_usdt: float = 10.0
    min_available_balance_ratio: float = 0.005
    capacity_min_new_margin_divisor: float = 6.0
    capacity_min_new_margin_floor_pct: float = 0.02
    capacity_default_leverage: float = 5.0
    capacity_default_stop_loss_pct: float = 0.05


@dataclass(frozen=True, slots=True)
class EntryCrowdedSideCapParams:
    """Crowded-side exposure policy parameters.

    These values are hard-risk guards for one-sided portfolio concentration.
    They must stay in the auditable strategy snapshot so same-side entry
    blocks cannot silently drift into hidden fixed thresholds.
    """

    min_dominant_count: int = 8
    dominant_count_share: float = 0.72
    dominant_net_ratio: float = 0.55
    hard_max_side_count: int = 14
    strong_min_score_multiple: float = 1.6
    strong_min_score_floor: float = 2.6
    strong_min_expected_net_pct: float = 0.55
    strong_min_profit_quality_ratio: float = 1.4
    strong_max_loss_probability: float = 0.42
    hard_override_score_multiple: float = 2.4
    hard_override_score_floor: float = 4.2
    hard_override_min_expected_net_pct: float = 0.90
    hard_override_min_profit_quality_ratio: float = 1.8
    hard_override_max_loss_probability: float = 0.28
    hard_override_max_probe_fraction: float = 0.05
    hard_override_max_size_pct: float = 0.018


@dataclass(frozen=True, slots=True)
class EnsembleEntryDecisionParams:
    """Ensemble entry voting thresholds.

    These values translate expert/model votes into executable entry intent.
    Keeping them in the snapshot makes no-entry and tiny-probe decisions
    auditable instead of being hidden inside the coordinator.
    """

    normal_entry_score_threshold: float = 0.42
    probe_entry_score_threshold: float = 0.26
    probe_entry_enabled: bool = True
    max_entry_disagreement: float = 0.50
    min_executable_entry_confidence: float = 0.58
    daily_recovery_entry_score_bonus: float = 0.10
    daily_recovery_min_entry_confidence: float = 0.74
    daily_recovery_max_entry_size: float = 0.04
    daily_recovery_max_leverage: float = 5.0
    market_direction_excluded_experts: tuple[str, ...] = (
        "position_expert",
        "risk_expert",
    )
    entry_direction_support_experts: tuple[str, ...] = (
        "trend_expert",
        "sentiment_expert",
    )
    entry_profit_quality_experts: tuple[str, ...] = ("momentum_expert",)
    no_position_trend_expert_weight: float = 0.33
    no_position_momentum_expert_weight: float = 0.33
    no_position_sentiment_expert_weight: float = 0.14
    no_position_position_expert_weight: float = 0.05
    no_position_position_expert_weight_cap: float = 0.05
    risk_entry_score_discount_max: float = 0.30
    risk_entry_size_multiplier_floor: float = 0.45
    min_review_close_support: int = 2
    full_close_support: int = 3
    review_close_min_confidence: float = 0.55
    review_close_strong_confidence: float = 0.65
    review_strong_opposite_score: float = 0.45
    add_position_min_support: int = 2
    add_position_min_confidence: float = 0.60
    add_position_strong_confidence: float = 0.68
    add_position_score_threshold: float = 0.32
    add_position_min_profit_ratio: float = 0.002
    add_position_max_risk_usage: float = 0.20
    add_position_min_size: float = 0.02
    add_position_max_size: float = 0.06
    winner_expand_min_unrealized_usdt: float = 1.2
    winner_expand_min_profit_ratio: float = 0.0012
    winner_expand_score_threshold: float = 0.18
    winner_expand_max_risk_usage: float = 0.25
    winner_run_min_profit_ratio: float = 0.001


@dataclass(frozen=True, slots=True)
class EnsembleExitDecisionParams:
    """Ensemble position-review exit thresholds.

    These thresholds decide profit protection, loss compression, and
    predictive reversal exits, so they must be visible in strategy audits.
    """

    profit_protect_reduce_pnl_ratio: float = 0.004
    profit_protect_strong_pnl_ratio: float = 0.010
    profit_protect_full_pnl_ratio: float = 0.018
    profit_protect_min_lock_usdt: float = 1.50
    profit_protect_strong_min_lock_usdt: float = 2.50
    profit_protect_full_min_lock_usdt: float = 4.00
    profit_protect_moderate_opposite_score: float = 0.10
    profit_protect_reduce_size: float = 0.45
    fast_profit_min_hold_minutes: float = 6.0
    profit_exit_analysis_min_floor_usdt: float = 0.75
    min_discretionary_close_hold_minutes: float = 4.0
    early_close_min_risk_usage: float = 0.70
    loss_reduce_min_risk_usage: float = 0.55
    loss_full_min_risk_usage: float = 0.82
    quick_profit_reduce_pnl_ratio: float = 0.008
    capital_rotation_profit_pnl_ratio: float = 0.012
    quick_profit_full_pnl_ratio: float = 0.020
    profit_lock_fee_multiple: float = 4.0
    profit_lock_notional_ratio: float = 0.0025
    profit_lock_risk_ratio: float = 0.18
    profit_lock_min_floor_usdt: float = 0.25
    profit_lock_max_floor_usdt: float = 8.0
    profit_lock_meaningful_reduce_usdt: float = 3.0
    profit_lock_reduce_fee_multiple: float = 10.0
    profit_lock_reduce_notional_ratio: float = 0.008
    profit_lock_reduce_risk_ratio: float = 0.16
    portfolio_focus_lock_min_usdt: float = 3.0
    portfolio_focus_lock_min_share: float = 0.20
    portfolio_focus_lock_reduce_size: float = 0.35
    profit_retrace_peak_line_multiple: float = 1.15
    profit_retrace_current_line_multiple: float = 0.25
    profit_retrace_base_reduce_ratio: float = 0.24
    profit_retrace_base_full_ratio: float = 0.52
    profit_retrace_min_reduce_ratio: float = 0.16
    profit_retrace_max_reduce_ratio: float = 0.40
    profit_retrace_min_full_ratio: float = 0.42
    profit_retrace_max_full_ratio: float = 0.72
    loss_compress_reduce_usdt: float = 3.0
    loss_compress_full_usdt: float = 8.0
    loss_compress_reduce_ratio: float = 0.006
    loss_compress_full_ratio: float = 0.012
    loss_compress_reduce_risk_ratio: float = 0.35
    loss_compress_full_risk_ratio: float = 0.65
    loss_repair_max_loss_probability: float = 0.54
    loss_expand_min_loss_probability: float = 0.55
    loss_expand_full_loss_probability: float = 0.61
    loss_repair_reduce_support_count: int = 2
    loss_repair_full_support_count: int = 3
    predictive_reversal_review_score: float = 38.0
    predictive_reversal_exit_score: float = 60.0
    predictive_reversal_full_exit_score: float = 78.0
    predictive_reversal_reduce_size: float = 0.60


@dataclass(frozen=True, slots=True)
class EnsembleMLProbeParams:
    """ML and quant-validation probe thresholds used by the coordinator."""

    ml_min_expected_return_pct: float = 0.05
    ml_min_profit_edge_pct: float = 0.02
    ml_min_support_win_rate: float = 0.0
    ml_strong_support_win_rate: float = 0.0
    ml_support_confidence_bonus: float = 0.02
    ml_low_edge_confidence_bonus: float = 0.06
    ml_low_win_confidence_bonus: float = 0.08
    ml_profit_first_score_relief: float = 0.08
    ml_profit_first_min_expected_return_pct: float = 0.12
    ml_profit_first_min_edge_pct: float = 0.08
    ml_quant_only_min_expected_return_pct: float = 0.10
    ml_quant_only_min_edge_pct: float = 0.18
    ml_quant_only_max_loss_probability: float = 0.60
    ml_profit_first_low_win_rate_size_multiplier: float = 0.60
    local_tools_max_loss_probability: float = 0.62
    profit_first_probe_confidence: float = 0.72
    quant_validation_probe_confidence: float = 0.70
    quant_validation_max_loss_probability: float = 0.58
    quant_validation_min_local_expected_return_pct: float = 0.03
    quant_validation_min_profit_quality_score: float = 0.20
    quant_only_short_direction_min_gap: float = 0.08


@dataclass(frozen=True, slots=True)
class EntryStopLossBudgetParams:
    normal_budget_usdt: float = 16.0
    drawdown_budget_usdt: float = 8.0
    defensive_budget_usdt: float = 4.0
    high_quality_drawdown_budget_usdt: float = 12.0
    high_quality_defensive_budget_usdt: float = 8.0
    equity_cap_pct: float = 0.008
    max_dynamic_cap_usdt: float = 36.0


@dataclass(frozen=True, slots=True)
class ExitFastRiskParams:
    profit_drawdown_min_hold_minutes: float = 8.0
    profit_drawdown_min_profit_ratio: float = 0.006
    profit_drawdown_strong_profit_ratio: float = 0.016
    profit_drawdown_partial_retrace: float = 0.38
    profit_drawdown_full_retrace: float = 0.68
    profit_drawdown_partial_close_fraction: float = 0.35
    profit_drawdown_min_net_usdt: float = 5.0
    profit_drawdown_min_fee_multiple: float = 4.0
    profit_drawdown_min_seconds_between_exits: float = 600.0
    profit_drawdown_volume_confirm_ratio: float = 1.05
    profit_drawdown_accelerated_hold_minutes: float = 8.0
    fast_risk_1m_move_pct: float = 0.025
    fast_risk_5m_move_pct: float = 0.04
    fast_risk_min_hold_minutes: float = 4.0
    fast_risk_fresh_review_min_hold_minutes: float = 12.0
    fast_risk_min_loss_pct: float = 0.008
    fast_risk_reduce_loss_pct: float = 0.012
    fast_risk_full_loss_pct: float = 0.018
    fast_risk_near_stop_progress: float = 0.50
    fast_risk_full_stop_progress: float = 0.78
    fast_risk_volume_confirm_ratio: float = 1.05
    fast_risk_reduce_position_pct: float = 0.50
    fast_risk_force_full_loss_usdt: float = 4.0
    fast_risk_force_full_progress: float = 0.50
    fast_risk_max_feature_position_price_gap: float = 0.03
    fast_risk_price_24h_range_tolerance_pct: float = 0.03


@dataclass(frozen=True, slots=True)
class ExitPositionQualityParams:
    """Low-quality position release policy.

    Fresh positions need a short observation window so ordinary adverse noise
    cannot be mislabeled as an old low-quality position. Hard-risk exits still
    bypass this protection when loss evidence is severe.
    """

    fresh_position_min_release_hold_hours: float = 0.25
    fresh_position_score_floor: float = 72.0
    fresh_position_hard_risk_loss_ratio: float = -0.05


@dataclass(frozen=True, slots=True)
class ExitCooldownParams:
    ordinary_seconds: float = 600.0
    volatile_seconds: float = 300.0
    elevated_volatility_seconds: float = 420.0
    stable_seconds: float = 900.0


@dataclass(frozen=True, slots=True)
class ExitArbitrationParams:
    hard_risk_priority: int = 100
    trend_failure_priority: int = 90
    predictive_downside_priority: int = 85
    profit_drawdown_priority: int = 75
    profit_protection_priority: int = 60
    capital_rotation_priority: int = 50
    loss_repair_priority: int = 40
    ordinary_priority: int = 10


@dataclass(frozen=True, slots=True)
class TradingParameterSnapshot:
    fee: FeeParams = field(default_factory=FeeParams)
    execution_cost: ExecutionCostParams = field(default_factory=ExecutionCostParams)
    entry_tiers: EntryTierParams = field(default_factory=EntryTierParams)
    entry_evidence: EntryEvidenceParams = field(default_factory=EntryEvidenceParams)
    entry_opportunity_gate: EntryOpportunityGateParams = field(
        default_factory=EntryOpportunityGateParams
    )
    entry_opportunity_scoring: EntryOpportunityScoringParams = field(
        default_factory=EntryOpportunityScoringParams
    )
    entry_execution_priority: EntryExecutionPriorityParams = field(
        default_factory=EntryExecutionPriorityParams
    )
    entry_payoff_quality: EntryPayoffQualityParams = field(default_factory=EntryPayoffQualityParams)
    entry_market_data_quality: EntryMarketDataQualityParams = field(
        default_factory=EntryMarketDataQualityParams
    )
    entry_feature_ranker: EntryFeatureRankerParams = field(default_factory=EntryFeatureRankerParams)
    training_data_quality: TrainingDataQualityParams = field(
        default_factory=TrainingDataQualityParams
    )
    local_ml_training: LocalMLTrainingParams = field(default_factory=LocalMLTrainingParams)
    auto_scan: AutoScanParams = field(default_factory=AutoScanParams)
    strategy_learning: StrategyLearningParams = field(default_factory=StrategyLearningParams)
    entry_winner: EntryWinnerParams = field(default_factory=EntryWinnerParams)
    entry_loss_cooldown: EntryLossCooldownParams = field(default_factory=EntryLossCooldownParams)
    entry_probe_market_quality: EntryProbeMarketQualityParams = field(
        default_factory=EntryProbeMarketQualityParams
    )
    entry_quant_profit_probe: EntryQuantProfitProbeParams = field(
        default_factory=EntryQuantProfitProbeParams
    )
    entry_risk_sizing: EntryRiskSizingParams = field(default_factory=EntryRiskSizingParams)
    entry_price_guard: EntryPriceGuardParams = field(default_factory=EntryPriceGuardParams)
    entry_account_guard: EntryAccountGuardParams = field(default_factory=EntryAccountGuardParams)
    entry_crowded_side_cap: EntryCrowdedSideCapParams = field(
        default_factory=EntryCrowdedSideCapParams
    )
    ensemble_entry_decision: EnsembleEntryDecisionParams = field(
        default_factory=EnsembleEntryDecisionParams
    )
    ensemble_exit_decision: EnsembleExitDecisionParams = field(
        default_factory=EnsembleExitDecisionParams
    )
    ensemble_ml_probe: EnsembleMLProbeParams = field(default_factory=EnsembleMLProbeParams)
    entry_stop_loss_budget: EntryStopLossBudgetParams = field(
        default_factory=EntryStopLossBudgetParams
    )
    exit_fast_risk: ExitFastRiskParams = field(default_factory=ExitFastRiskParams)
    exit_position_quality: ExitPositionQualityParams = field(
        default_factory=ExitPositionQualityParams
    )
    exit_cooldown: ExitCooldownParams = field(default_factory=ExitCooldownParams)
    exit_arbitration: ExitArbitrationParams = field(default_factory=ExitArbitrationParams)
    version: str = "2026-06-22.strategy-v6-auditable-caps"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_TRADING_PARAMS = TradingParameterSnapshot()


def default_trading_parameter_snapshot() -> dict[str, Any]:
    """Return a serializable snapshot of strategy parameters used by policies."""

    return DEFAULT_TRADING_PARAMS.to_dict()


def trading_parameter_snapshot_payload(scope: str) -> dict[str, Any]:
    """Return the policy parameter payload persisted with a decision."""

    return {
        "scope": scope,
        "snapshot": default_trading_parameter_snapshot(),
    }


def attach_trading_parameter_snapshot(
    decision: DecisionOutput,
    *,
    scope: str,
) -> dict[str, Any]:
    """Attach the active strategy-parameter snapshot to a decision raw payload."""

    raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
    payload = trading_parameter_snapshot_payload(scope)
    raw["strategy_parameters"] = payload
    decision.raw_response = raw
    return payload
