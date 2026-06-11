from types import SimpleNamespace

from services.entry_direction_competition import EntryDirectionCompetitionPolicy
from services.trading_service import TradingService


def _feature(**overrides):
    defaults = {
        "symbol": "BTC/USDT",
        "returns_1": 0.0,
        "returns_5": 0.0,
        "returns_20": 0.0,
        "price_vs_sma20": 0.0,
        "price_vs_sma50": 0.0,
        "adx_14": 14.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _context(
    *,
    feature=None,
    ml=None,
    tools=None,
    market=None,
    strategy=None,
):
    return EntryDirectionCompetitionPolicy().context(
        feature or _feature(),
        ml,
        tools,
        market,
        strategy,
    )


def test_aligned_models_prefer_long() -> None:
    context = _context(
        feature=_feature(
            returns_1=0.002,
            returns_5=0.006,
            returns_20=0.01,
            price_vs_sma20=0.02,
            price_vs_sma50=0.03,
            adx_14=28.0,
        ),
        ml={
            "influence_enabled": True,
            "predictions": [
                {
                    "long_expected_return_pct": 0.8,
                    "long_win_rate": 0.66,
                    "short_expected_return_pct": -0.2,
                    "short_win_rate": 0.43,
                }
            ],
        },
        tools={
            "profit_prediction": {
                "adjusted_long_return_pct": 0.7,
                "long_loss_probability": 0.25,
                "adjusted_short_return_pct": -0.2,
                "short_loss_probability": 0.66,
                "profit_quality_score": 0.5,
            },
            "time_series_prediction": {
                "best_side": "long",
                "expected_return_pct": 0.45,
            },
            "sentiment_analysis": {
                "best_side": "long",
                "expected_return_pct": 0.3,
                "score": 0.6,
            },
        },
        market={"mode": "uptrend_continuation"},
    )

    assert context["preferred_side"] == "long"
    assert context["score_gap"] > 0.08
    assert context["long"]["score"] > context["short"]["score"]
    assert context["long"]["expected_return_pct"] > context["short"]["expected_return_pct"]
    assert any("ML long" in note for note in context["long"]["evidence"])
    assert context["market_regime_mode"] == "uptrend_continuation"


def test_ml_influence_disabled_ignores_ml_signal() -> None:
    context = _context(
        feature=_feature(returns_5=-0.003, returns_20=-0.006, adx_14=24.0),
        ml={
            "influence_enabled": False,
            "predictions": [
                {
                    "long_expected_return_pct": 2.0,
                    "long_win_rate": 0.9,
                    "short_expected_return_pct": -1.0,
                    "short_win_rate": 0.2,
                }
            ],
        },
        tools={
            "profit_prediction": {
                "adjusted_long_return_pct": -0.3,
                "long_loss_probability": 0.75,
                "adjusted_short_return_pct": 0.8,
                "short_loss_probability": 0.25,
                "profit_quality_score": 0.3,
            },
            "time_series_prediction": {
                "best_side": "short",
                "expected_return_pct": 0.5,
            },
        },
    )

    all_evidence = context["long"]["evidence"] + context["short"]["evidence"]
    assert context["preferred_side"] == "short"
    assert not any(note.startswith("ML ") for note in all_evidence)


def test_degraded_source_performance_reduces_model_weight() -> None:
    ml = {
        "predictions": [
            {
                "long_expected_return_pct": 1.0,
                "long_win_rate": 0.7,
                "short_expected_return_pct": 0.0,
                "short_win_rate": 0.5,
            }
        ],
    }

    baseline = _context(ml=ml)
    degraded = _context(
        ml=ml,
        strategy={
            "model_contribution_performance": {
                "ml_profit_model": {
                    "count": 10,
                    "pnl": -12.0,
                    "profit_factor": 0.7,
                    "score_multiplier": 1.0,
                    "state": "degrade",
                }
            }
        },
    )

    assert degraded["long"]["score"] < baseline["long"]["score"] * 0.5
    assert any("weight=0.40" in note for note in degraded["long"]["evidence"])


def test_soft_avoided_direction_is_penalty_not_block() -> None:
    tools = {
        "profit_prediction": {
            "adjusted_long_return_pct": 1.0,
            "long_loss_probability": 0.2,
            "adjusted_short_return_pct": -0.2,
            "short_loss_probability": 0.7,
            "profit_quality_score": 0.4,
        }
    }

    baseline = _context(tools=tools)
    penalized = _context(
        tools=tools,
        strategy={"soft_avoided_directions": ["long"]},
    )

    assert penalized["preferred_side"] == "long"
    assert penalized["long"]["score"] < baseline["long"]["score"]
    assert any("soft penalty" in note for note in penalized["long"]["evidence"])


def test_losing_dominant_exposure_discounts_same_side_and_only_nudges_positive_opposite() -> None:
    tools = {
        "profit_prediction": {
            "adjusted_long_return_pct": 0.2,
            "long_loss_probability": 0.55,
            "adjusted_short_return_pct": 0.35,
            "short_loss_probability": 0.35,
            "profit_quality_score": 0.2,
        }
    }

    baseline = _context(tools=tools)
    exposed = _context(
        tools=tools,
        strategy={
            "position_exposure": {
                "dominant_side": "long",
                "net_ratio": 1.0,
                "long_unrealized_pnl": -10.0,
            }
        },
    )

    assert exposed["long"]["score"] < baseline["long"]["score"]
    assert exposed["short"]["score"] > baseline["short"]["score"]
    assert exposed["short"]["score"] - baseline["short"]["score"] <= 0.04
    assert any("concentrated long" in note for note in exposed["long"]["evidence"])
    assert any("small balance nudge" in note for note in exposed["short"]["evidence"])


def test_trading_service_direction_competition_delegates_to_policy() -> None:
    service = object.__new__(TradingService)

    context = service._direction_competition_context(
        _feature(returns_5=0.005, returns_20=0.008, adx_14=26.0),
        {
            "predictions": [
                {
                    "long_expected_return_pct": 0.6,
                    "long_win_rate": 0.62,
                    "short_expected_return_pct": -0.1,
                    "short_win_rate": 0.45,
                }
            ]
        },
        {},
        {"mode": "mixed"},
        {},
    )

    assert context["enabled"] is True
    assert context["preferred_side"] == "long"
    assert context["market_regime_mode"] == "mixed"
