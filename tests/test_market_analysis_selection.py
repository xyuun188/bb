from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.market_analysis_selection import MarketAnalysisSelectionPolicy
from services.trading_params import MarketAnalysisSelectionParams


def _feature(
    symbol: str,
    score: float,
    *,
    price: float = 100.0,
    volume_ratio: float = 1.5,
    adx: float = 25.0,
    returns_5: float = 0.002,
    volatility: float = 0.001,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        score=score,
        current_price=price,
        entry_activity_volume_ratio=volume_ratio,
        adx_14=adx,
        returns_5=returns_5,
        volatility_20=volatility,
    )


def _policy(**overrides) -> MarketAnalysisSelectionPolicy:
    params = MarketAnalysisSelectionParams(**overrides)
    return MarketAnalysisSelectionPolicy(
        normalize_symbol=lambda value: str(value or "").upper(),
        advantage_scorer=lambda feature: float(feature.score),
        params=params,
    )


def test_fresh_candidates_keep_advantage_order() -> None:
    policy = _policy()
    candidates = {
        "BTC/USDT": _feature("BTC/USDT", 10.0),
        "ETH/USDT": _feature("ETH/USDT", 9.0),
        "SOL/USDT": _feature("SOL/USDT", 8.0),
        "DOGE/USDT": _feature("DOGE/USDT", 7.0),
    }

    result = policy.select(candidates, 3, now=datetime(2026, 7, 21, tzinfo=UTC))

    assert list(result.selected) == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert result.diagnostics["discovery_slots"] == 1
    assert result.diagnostics["discovery_selected_symbols"] == ["SOL/USDT"]
    assert result.diagnostics["recent_unchanged_candidate_count"] == 0
    assert result.diagnostics["is_entry_gate"] is False


def test_unchanged_recent_candidates_are_penalized_without_being_banned() -> None:
    now = datetime(2026, 7, 21, 1, 0, tzinfo=UTC)
    policy = _policy(unchanged_repeat_penalty_ratio=0.35)
    repeated = _feature("BTC/USDT", 20.0)
    policy.remember("BTC/USDT", repeated, observed_at=now - timedelta(seconds=30))
    candidates = {
        "BTC/USDT": repeated,
        "ETH/USDT": _feature("ETH/USDT", 9.0),
        "SOL/USDT": _feature("SOL/USDT", 8.0),
        "DOGE/USDT": _feature("DOGE/USDT", 7.0),
    }

    result = policy.select(candidates, 3, now=now)

    assert list(result.selected) == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    btc = next(row for row in result.diagnostics["selected"] if row["symbol"] == "BTC/USDT")
    assert btc["selection_status"] == "recent_unchanged_penalty"
    assert btc["repeat_penalty"] == 7.0
    assert btc["evaluation_score"] == 13.0


def test_discovery_capacity_replaces_lower_value_unchanged_repeat() -> None:
    now = datetime(2026, 7, 21, 1, 0, tzinfo=UTC)
    policy = _policy(unchanged_repeat_penalty_ratio=0.35)
    repeated = {
        "BTC/USDT": _feature("BTC/USDT", 10.0),
        "ETH/USDT": _feature("ETH/USDT", 9.0),
    }
    for symbol, feature in repeated.items():
        policy.remember(symbol, feature, observed_at=now - timedelta(seconds=30))
    candidates = {
        **repeated,
        "SOL/USDT": _feature("SOL/USDT", 8.0),
        "DOGE/USDT": _feature("DOGE/USDT", 7.0),
    }

    result = policy.select(candidates, 3, now=now)

    assert list(result.selected) == ["SOL/USDT", "DOGE/USDT", "BTC/USDT"]
    assert result.diagnostics["discovery_selected_symbols"] == ["DOGE/USDT"]
    assert result.diagnostics["skipped_symbols"] == ["ETH/USDT"]
    assert result.diagnostics["recent_unchanged_candidate_count"] == 2


def test_material_market_change_bypasses_recent_penalty() -> None:
    now = datetime(2026, 7, 21, 1, 0, tzinfo=UTC)
    policy = _policy(material_price_change_ratio=0.003)
    policy.remember(
        "BTC/USDT",
        _feature("BTC/USDT", 10.0, price=100.0),
        observed_at=now - timedelta(seconds=30),
    )
    changed = _feature("BTC/USDT", 10.0, price=100.5)

    result = policy.select(
        {
            "BTC/USDT": changed,
            "ETH/USDT": _feature("ETH/USDT", 9.0),
        },
        1,
        now=now,
    )

    selected = result.diagnostics["selected"][0]
    assert selected["symbol"] == "BTC/USDT"
    assert selected["selection_status"] == "material_change_bypass"
    assert selected["repeat_penalty"] == 0.0
    assert selected["material_change_reasons"][0]["feature"] == "current_price"


def test_all_recent_candidates_still_produce_a_shortlist() -> None:
    now = datetime(2026, 7, 21, 1, 0, tzinfo=UTC)
    policy = _policy()
    candidates = {
        "BTC/USDT": _feature("BTC/USDT", 10.0),
        "ETH/USDT": _feature("ETH/USDT", 9.0),
    }
    for symbol, feature in candidates.items():
        policy.remember(symbol, feature, observed_at=now - timedelta(seconds=30))

    result = policy.select(candidates, 2, now=now)

    assert list(result.selected) == ["BTC/USDT", "ETH/USDT"]
    assert result.diagnostics["selected_count"] == 2
    assert result.diagnostics["skipped_count"] == 0


def test_candidate_pool_expands_before_final_expert_limit() -> None:
    policy = _policy(candidate_pool_multiplier=3)

    assert policy.candidate_pool_limit(3, 20) == 9
    assert policy.candidate_pool_limit(3, 5) == 5
    assert policy.candidate_pool_limit(0, 20) == 0
