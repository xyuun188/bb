from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.entry_market_hold_penalty import (
    MARKET_NO_OPPORTUNITY_MAX_PENALTY,
    MARKET_RECENT_ANALYSIS_DEDUPE_SECONDS,
    MARKET_RECENT_ANALYSIS_MAX_PENALTY,
    MARKET_RECENT_HOLD_MAX_PENALTY,
    MARKET_RECENT_LOSS_MAX_PENALTY,
    EntryMarketHoldPenaltyPolicy,
)
from services.trading_service import TradingService


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _policy(clock: Clock | None = None) -> EntryMarketHoldPenaltyPolicy:
    return EntryMarketHoldPenaltyPolicy(
        normalize_symbol=lambda symbol: str(symbol or "").upper() or None,
        feature_opportunity_score=lambda feature: float(
            getattr(feature, "opportunity_score", 10.0)
        ),
        min_entry_volume_ratio_provider=lambda: 0.3,
        min_entry_adx_provider=lambda: 12.0,
        now_provider=clock or Clock(datetime.now(UTC)),
    )


def _feature(**overrides):
    defaults = {
        "volume_ratio": 0.4,
        "adx_14": 18.0,
        "returns_1": 0.0,
        "returns_5": -0.002,
        "returns_20": -0.004,
        "volatility_20": 0.02,
        "price_vs_sma20": -0.003,
        "price_vs_sma50": -0.002,
        "bb_pct": 0.5,
        "change_24h_pct": 0.0,
        "opportunity_score": 10.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_recent_hold_penalty_is_capped_and_decays() -> None:
    clock = Clock(datetime(2026, 6, 10, 10, 0, tzinfo=UTC))
    policy = _policy(clock)
    policy.recent_hold_symbols["BTC/USDT"] = clock.now - timedelta(minutes=10)

    penalty = policy.recent_hold_penalty("BTC/USDT")

    assert 0 < penalty < MARKET_RECENT_HOLD_MAX_PENALTY


def test_recent_loss_profiles_create_front_rank_rotation_penalty() -> None:
    clock = Clock(datetime(2026, 6, 10, 10, 0, tzinfo=UTC))
    policy = _policy(clock)

    policy.sync_recent_loss_profiles(
        {
            "LINK/USDT|all": {
                "cooldown": True,
                "pnl": -3.2,
                "loss": 3.2,
                "today_loss": 1.0,
                "largest_loss": 2.0,
                "cooldown_remaining_hours": 0.5,
                "cooldown_reason": "recent_realized_loss",
            }
        }
    )

    penalty = policy.recent_hold_penalty("link/usdt")

    assert 0 < penalty <= MARKET_RECENT_LOSS_MAX_PENALTY
    assert policy.recent_loss_symbols["LINK/USDT"]["reason"] == "recent_realized_loss"


def test_recent_loss_penalty_decays_and_expires() -> None:
    clock = Clock(datetime(2026, 6, 10, 10, 0, tzinfo=UTC))
    policy = _policy(clock)
    policy.sync_recent_loss_profiles(
        {
            "LINK/USDT|short": {
                "cooldown": True,
                "pnl": -2.0,
                "loss": 2.0,
                "cooldown_remaining_hours": 1.0,
            }
        }
    )
    initial_penalty = policy.recent_loss_penalty("LINK/USDT")

    clock.now += timedelta(minutes=30)
    decayed_penalty = policy.recent_loss_penalty("LINK/USDT")
    clock.now += timedelta(minutes=31)
    expired_penalty = policy.recent_loss_penalty("LINK/USDT")

    assert 0 < decayed_penalty < initial_penalty
    assert expired_penalty == 0.0
    assert "LINK/USDT" not in policy.recent_loss_symbols


def test_recent_analysis_penalty_is_capped_and_decays() -> None:
    clock = Clock(datetime(2026, 6, 10, 10, 0, tzinfo=UTC))
    policy = _policy(clock)
    policy.remember_analyzed_symbol("BTC/USDT")
    clock.now += timedelta(minutes=10)

    penalty = policy.recent_analysis_penalty("BTC/USDT")

    assert 0 < penalty < MARKET_RECENT_ANALYSIS_MAX_PENALTY


def test_recently_analyzed_blocks_same_symbol_inside_short_window() -> None:
    clock = Clock(datetime(2026, 6, 10, 10, 0, tzinfo=UTC))
    policy = _policy(clock)
    policy.remember_analyzed_symbol("BTC/USDT")

    assert policy.recently_analyzed("btc/usdt") is True

    clock.now += timedelta(seconds=MARKET_RECENT_ANALYSIS_DEDUPE_SECONDS + 1)
    assert policy.recently_analyzed("BTC/USDT") is False


def test_no_opportunity_penalty_accumulates_after_repeated_holds() -> None:
    clock = Clock(datetime(2026, 6, 10, 10, 0, tzinfo=UTC))
    policy = _policy(clock)
    policy.remember_hold_symbol("BTC/USDT", _feature(), "AI hold")
    clock.now += timedelta(minutes=2)
    policy.remember_hold_symbol("BTC/USDT", _feature(), "AI hold again")

    penalty = policy.no_opportunity_rotation_penalty("BTC/USDT", _feature())

    assert 0 < penalty <= MARKET_NO_OPPORTUNITY_MAX_PENALTY
    assert policy.no_opportunity_symbols["BTC/USDT"]["hold_count"] == 2


def test_no_opportunity_penalty_clears_when_market_regime_changes() -> None:
    clock = Clock(datetime(2026, 6, 10, 10, 0, tzinfo=UTC))
    policy = _policy(clock)
    policy.recent_hold_symbols["BTC/USDT"] = clock.now - timedelta(minutes=5)
    policy.no_opportunity_symbols["BTC/USDT"] = {
        "first_seen_at": clock.now - timedelta(minutes=8),
        "last_hold_at": clock.now - timedelta(minutes=5),
        "hold_count": 3,
        "last_feature_score": 10.0,
        "last_volume_ratio": 0.4,
        "last_returns_5": -0.002,
        "last_returns_20": -0.004,
        "last_price_vs_sma20": -0.003,
    }
    changed_market = _feature(
        volume_ratio=1.2,
        adx_14=28.0,
        returns_1=0.001,
        returns_5=0.006,
        returns_20=0.012,
        price_vs_sma20=0.004,
        price_vs_sma50=0.003,
        bb_pct=0.55,
        change_24h_pct=2.0,
    )

    assert policy.no_opportunity_rotation_penalty("BTC/USDT", changed_market) == 0.0
    assert "BTC/USDT" not in policy.no_opportunity_symbols
    assert "BTC/USDT" not in policy.recent_hold_symbols


def test_trading_service_market_hold_penalty_delegates_to_policy() -> None:
    service = object.__new__(TradingService)
    service.entry_market_hold_penalty = _policy()

    service._remember_market_hold_symbol("BTC/USDT", _feature(), "AI hold")

    assert service._recent_market_hold_penalty("BTC/USDT") > 0.0
    service._clear_market_no_opportunity_symbol("BTC/USDT")
    assert service._recent_market_hold_penalty("BTC/USDT") == 0.0


def test_clear_symbol_removes_recent_loss_rotation_penalty() -> None:
    clock = Clock(datetime(2026, 6, 10, 10, 0, tzinfo=UTC))
    policy = _policy(clock)
    policy.sync_recent_loss_profiles(
        {
            "LINK/USDT|all": {
                "cooldown": True,
                "pnl": -2.0,
                "loss": 2.0,
                "cooldown_remaining_hours": 1.0,
            }
        }
    )

    policy.clear_symbol("LINK/USDT")

    assert policy.recent_loss_penalty("LINK/USDT") == 0.0
