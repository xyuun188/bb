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


@dataclass(frozen=True, slots=True)
class EntryProbeMarketQualityParams:
    max_price_field_gap: float = 0.03
    strong_contra_20m_pct: float = 0.05
    min_volume_ratio: float = 0.02


@dataclass(frozen=True, slots=True)
class EntryQuantProfitProbeParams:
    min_expected_pct: float = 0.18
    min_edge_pct: float = 0.22
    min_concentrated_loss_usdt: float = 5.0
    default_max_loss_probability: float = 0.58
    roster_fill_min_expected_pct: float = 0.08
    roster_fill_min_edge_pct: float = 0.08
    roster_fill_max_loss_probability: float = 0.66


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
    entry_tiers: EntryTierParams = field(default_factory=EntryTierParams)
    entry_winner: EntryWinnerParams = field(default_factory=EntryWinnerParams)
    entry_loss_cooldown: EntryLossCooldownParams = field(default_factory=EntryLossCooldownParams)
    entry_probe_market_quality: EntryProbeMarketQualityParams = field(
        default_factory=EntryProbeMarketQualityParams
    )
    entry_quant_profit_probe: EntryQuantProfitProbeParams = field(
        default_factory=EntryQuantProfitProbeParams
    )
    exit_cooldown: ExitCooldownParams = field(default_factory=ExitCooldownParams)
    exit_arbitration: ExitArbitrationParams = field(default_factory=ExitArbitrationParams)
    version: str = "2026-06-09.strategy-v2"

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
