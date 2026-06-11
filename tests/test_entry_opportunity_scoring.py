from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_opportunity_scoring import EntryOpportunityScoringPolicy
from services.entry_symbol_winner import EntrySymbolWinnerDecayPolicy


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.76,
        reasoning="test entry",
        position_size_pct=0.04,
        suggested_leverage=3.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.06,
        raw_response={
            "analysis_type": "market",
            "ml_signal": {
                "predictions": [
                    {
                        "long_expected_return_pct": 0.6,
                        "short_expected_return_pct": -0.1,
                        "long_win_rate": 0.66,
                        "profit_quality_score": 1.1,
                    }
                ]
            },
            "local_ai_tools": {
                "profit_prediction": {
                    "available": True,
                    "best_side": "long",
                    "adjusted_long_return_pct": 0.7,
                    "long_loss_probability": 0.30,
                    "profit_quality_score": 0.9,
                },
                "time_series_prediction": {
                    "best_side": "long",
                    "expected_return_pct": 0.3,
                },
            },
        },
        feature_snapshot={
            "current_price": 100.0,
            "volatility_20": 0.02,
            "change_24h_pct": 1.0,
        },
    )


def _policy(now: datetime) -> EntryOpportunityScoringPolicy:
    return EntryOpportunityScoringPolicy(
        normalize_symbol=lambda symbol: str(symbol or ""),
        model_contribution_score_adjustment=lambda _sources, _performance: {},
        annotate_decision_source=lambda _decision: {},
        entry_symbol_winner_decay=EntrySymbolWinnerDecayPolicy(clock=lambda: now),
    )


def _strategy(last_closed_at: str) -> dict[str, Any]:
    return {
        "min_opportunity_score": 0.95,
        "symbol_side_performance": {
            "BTC/USDT|long": {
                "count": 4,
                "wins": 3,
                "losses": 1,
                "pnl": 24.0,
                "profit": 28.0,
                "loss": 4.0,
                "avg_pnl": 6.0,
                "profit_factor": 2.2,
                "largest_loss": -2.0,
                "last_closed_at": last_closed_at,
            }
        },
    }


def test_entry_opportunity_scoring_embeds_recent_winner_decay() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    decision = _decision()

    _policy(now).score_candidate(
        decision,
        _strategy((now - timedelta(days=1)).isoformat()),
    )

    winner_decay = decision.raw_response["opportunity_score"]["symbol_winner_decay"]
    assert winner_decay["tier"] == "side_winner"
    assert winner_decay["side_decay_weight"] > 0.0
    assert winner_decay["side_effective_pnl"] < 24.0
    assert decision.raw_response["opportunity_score"]["symbol_tier_score_adjustment"] > 0.0


def test_entry_opportunity_scoring_does_not_relax_stale_winner() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    decision = _decision()

    _policy(now).score_candidate(
        decision,
        _strategy((now - timedelta(days=30)).isoformat()),
    )

    opportunity = decision.raw_response["opportunity_score"]
    winner_decay = opportunity["symbol_winner_decay"]
    assert winner_decay["tier"] == "neutral"
    assert winner_decay["side_decay_weight"] == 0.0
    assert winner_decay["side_effective_pnl"] == 0.0
    assert opportunity["symbol_tier_score_adjustment"] == 0.0


def test_entry_opportunity_scoring_turns_memory_habit_into_probe_cap() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    decision = _decision()
    decision.raw_response["memory_feedback"] = {
        "decision_habit": {
            "by_side": {
                "long": {
                    "stance": "probe_when_ev_ok",
                    "proactive_level": 0.6,
                    "probe_budget_pct": 0.015,
                    "min_expected_net_pct": 0.12,
                    "max_loss_probability": 0.58,
                    "max_tail_risk": 0.98,
                }
            }
        }
    }

    _policy(now).score_candidate(decision, {"min_opportunity_score": 0.95})

    habit = decision.raw_response["opportunity_score"]["memory_habit_adjustment"]
    assert habit["applied"] is True
    assert habit["stance"] == "probe_when_ev_ok"
    assert habit["quality_ok"] is True
    assert habit["score_adjustment"] > 0
    assert habit["adjusted_position_size"] == 0.015
    assert decision.position_size_pct == 0.015


def test_entry_opportunity_scoring_tightens_degraded_side_quality() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    decision = _decision()

    _policy(now).score_candidate(
        decision,
        {
            "min_opportunity_score": 0.95,
            "side_quality": {
                "long": {
                    "state": "degraded",
                    "score_adjustment": -0.25,
                    "min_score_delta": 0.22,
                    "size_multiplier": 0.65,
                    "reason": "long side realized performance is weak",
                }
            },
        },
    )

    opportunity = decision.raw_response["opportunity_score"]
    side_quality = opportunity["side_quality_adjustment"]
    assert side_quality["applied"] is True
    assert side_quality["state"] == "degraded"
    assert side_quality["score_adjustment"] < 0
    assert side_quality["min_score_delta"] > 0
    assert opportunity["min_score_required"] >= 0.85
    assert side_quality["adjusted_position_size"] < side_quality["original_position_size"]
    assert decision.position_size_pct < 0.04
