from services.trading_params import (
    DEFAULT_TRADING_PARAMS,
    ESTIMATED_TAKER_FEE_PCT,
    default_trading_parameter_snapshot,
)


def test_default_trading_parameter_snapshot_is_serializable() -> None:
    snapshot = default_trading_parameter_snapshot()

    assert snapshot["fee"]["estimated_taker_fee_pct"] == ESTIMATED_TAKER_FEE_PCT
    assert snapshot["entry_tiers"]["normal_score"] == 80.0
    assert snapshot["entry_tiers"]["weak_probe_min_aligned_sources"] == 3
    assert snapshot["entry_winner"]["half_life_days"] == 3.0
    assert snapshot["entry_winner"]["max_age_days"] == 14.0
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
