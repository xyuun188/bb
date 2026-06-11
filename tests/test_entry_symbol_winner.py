from datetime import UTC, datetime, timedelta

from services.entry_symbol_winner import EntrySymbolWinnerDecayPolicy


def test_entry_symbol_winner_decay_prefers_recent_side_winner() -> None:
    now = datetime(2026, 6, 9, tzinfo=UTC)
    policy = EntrySymbolWinnerDecayPolicy(clock=lambda: now)
    recent = (now - timedelta(days=1)).isoformat()

    result = policy.evaluate(
        side="long",
        side_profile={
            "count": 3,
            "pnl": 16.0,
            "profit_factor": 2.0,
            "last_closed_at": recent,
        },
        symbol_profile={},
        base_min_score_required=0.95,
        current_min_score_required=0.95,
        side_loss=0.0,
        side_profit=16.0,
        side_losses=0,
    )

    assert result.tier == "side_winner"
    assert result.score_adjustment > 0
    assert result.min_score_required < 0.95
    assert result.side_effective_pnl < 16.0
    assert result.side_age_days == 1.0


def test_entry_symbol_winner_decay_ignores_stale_winner() -> None:
    now = datetime(2026, 6, 9, tzinfo=UTC)
    policy = EntrySymbolWinnerDecayPolicy(clock=lambda: now)
    stale = (now - timedelta(days=30)).isoformat()

    result = policy.evaluate(
        side="long",
        side_profile={
            "count": 8,
            "pnl": 100.0,
            "profit_factor": 4.0,
            "last_closed_at": stale,
        },
        symbol_profile={},
        base_min_score_required=0.95,
        current_min_score_required=0.95,
        side_loss=0.0,
        side_profit=100.0,
        side_losses=0,
    )

    assert result.tier == "neutral"
    assert result.score_adjustment == 0.0
    assert result.min_score_required == 0.95
    assert result.side_decay_weight == 0.0
    assert result.side_effective_pnl == 0.0


def test_entry_symbol_winner_decay_keeps_recent_loser_penalty() -> None:
    policy = EntrySymbolWinnerDecayPolicy()

    result = policy.evaluate(
        side="short",
        side_profile={
            "count": 3,
            "pnl": -12.0,
            "profit_factor": 0.5,
        },
        symbol_profile={},
        base_min_score_required=0.95,
        current_min_score_required=0.95,
        side_loss=18.0,
        side_profit=2.0,
        side_losses=3,
    )

    assert result.tier == "side_loser"
    assert result.score_adjustment < 0
    assert result.min_score_required == 0.95
