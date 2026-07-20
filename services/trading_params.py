"""Central trading policy parameters and runtime snapshots.

This module is the first step toward moving tunable strategy numbers out of
the orchestration layer.  Keep runtime defaults here, while environment
settings remain in config.settings and backtests can persist the snapshot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ai_brain.base_model import DecisionOutput


@dataclass(frozen=True, slots=True)
class EntryMarketDataQualityParams:
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
class TrainingDataQualityParams:
    """Structural trust contract for local training samples."""

    allowed_sequence_timeframes: tuple[str, ...] = ("1m", "5m", "15m", "1h")
    manual_trade_sources: tuple[str, ...] = ("manual", "manual_trade", "test")


@dataclass(frozen=True, slots=True)
class LocalMLTrainingParams:
    """Local ML collection limits; return distributions own readiness."""

    auto_train_check_interval_seconds: int = 30 * 60
    auto_quarantine_batch_size: int = 1000
    auto_quarantine_max_batches: int = 5


@dataclass(frozen=True, slots=True)
class AutoScanParams:
    """Market scan breadth and major-symbol policy.

    These numbers affect how many symbols the scheduler can inspect per round.
    They are not entry gates, but keeping them in the snapshot makes scan
    breadth auditable when diagnosing "no open positions" symptoms.
    """

    rotation_pool_multiplier: int = 20
    rotation_pool_min: int = 240
    feature_fetch_pool_multiplier: int = 5
    feature_fetch_pool_min: int = 48
    feature_fetch_pool_max: int = 64
    feature_fetch_timeout_seconds: float = 8.0
    feature_fetch_concurrency: int = 8
    major_symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")


@dataclass(frozen=True, slots=True)
class MarketAnalysisSelectionParams:
    """Expert-analysis allocation controls that do not grant trade permission."""

    candidate_pool_multiplier: int = 3
    discovery_slots: int = 1
    cooldown_seconds: int = 5 * 60
    unchanged_repeat_penalty_ratio: float = 0.35
    history_limit: int = 120
    material_price_change_ratio: float = 0.003
    material_volume_ratio_change_ratio: float = 0.35
    material_adx_change: float = 3.0
    material_return_change: float = 0.003
    material_volatility_change_ratio: float = 0.35
    relative_change_floor: float = 1e-6


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
    expert_memory_lookback_days: int = 30
    order_extra_lookback_hours: int = 2










@dataclass(frozen=True, slots=True)
class TradingParameterSnapshot:
    entry_market_data_quality: EntryMarketDataQualityParams = field(
        default_factory=EntryMarketDataQualityParams
    )
    training_data_quality: TrainingDataQualityParams = field(
        default_factory=TrainingDataQualityParams
    )
    local_ml_training: LocalMLTrainingParams = field(default_factory=LocalMLTrainingParams)
    auto_scan: AutoScanParams = field(default_factory=AutoScanParams)
    market_analysis_selection: MarketAnalysisSelectionParams = field(
        default_factory=MarketAnalysisSelectionParams
    )
    strategy_learning: StrategyLearningParams = field(default_factory=StrategyLearningParams)
    version: str = "2026-07-21.market-analysis-value-v2"

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
