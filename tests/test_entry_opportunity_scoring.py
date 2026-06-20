from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

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


def test_entry_opportunity_scoring_uses_vector_memory_as_soft_adjustment() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    baseline = _decision()
    decision = _decision()
    decision.raw_response["memory_feedback"] = {
        "vector_memory": {
            "enabled": True,
            "status": "ok",
            "matched_count": 2,
            "hits": [
                {"score": 0.72, "action": "long", "pnl_pct": -0.8},
                {"score": 0.61, "action": "long", "pnl_pct": -0.3},
            ],
            "policy": "相似历史只作为软证据调节和解释，不作为硬拦截。",
        }
    }

    policy = _policy(now)
    baseline_score = policy.score_candidate(baseline, {"min_opportunity_score": 0.95})
    score = policy.score_candidate(decision, {"min_opportunity_score": 0.95})

    adjustment = decision.raw_response["opportunity_score"]["vector_memory_adjustment"]
    assert adjustment["applied"] is True
    assert adjustment["level"] == "negative"
    assert adjustment["score_adjustment"] < 0
    assert adjustment["is_hard_gate"] is False
    assert score < baseline_score


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


def test_entry_opportunity_scoring_reads_wrapped_server_quant_payloads() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    decision = _decision()
    decision.raw_response["local_ai_tools"] = {
        "profit_prediction": {
            "ok": True,
            "data": {
                "prediction": {
                    "best_side": "long",
                    "adjusted_long_return_pct": 0.82,
                    "adjusted_short_return_pct": -0.12,
                    "long_loss_probability": 0.29,
                    "profit_quality_score": 0.88,
                }
            },
        },
        "time_series_prediction": {"result": {"best_side": "long", "expected_return_pct": 0.37}},
    }

    _policy(now).score_candidate(decision, {"min_opportunity_score": 0.95})

    opportunity = decision.raw_response["opportunity_score"]
    assert opportunity["server_profit_best_side"] == "long"
    assert opportunity["server_profit_expected_return_pct"] == 0.82
    assert opportunity["timeseries_expected_return_pct"] == 0.37
    assert opportunity["timeseries_aligned"] is True
    assert "鏈" not in opportunity["dynamic_score_reason"]


def test_entry_opportunity_scoring_caps_ai_only_profit_when_quant_is_not_aligned() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    decision = _decision()
    decision.action = Action.SHORT
    decision.confidence = 0.92
    decision.stop_loss_pct = 0.02
    decision.take_profit_pct = 0.12
    decision.raw_response["ml_signal"] = {"influence_enabled": False, "predictions": []}
    decision.raw_response["local_ai_tools"] = {
        "profit_prediction": {
            "available": True,
            "best_side": "short",
            "adjusted_short_return_pct": -0.28,
            "short_loss_probability": 0.54,
            "profit_quality_score": -0.06,
        },
        "time_series_prediction": {
            "available": True,
            "best_side": "short",
            "direction": "down",
            "expected_move_pct": -0.05,
            "expected_return_pct": -0.05,
        },
    }

    _policy(now).score_candidate(decision, {"min_opportunity_score": 0.95})

    opportunity = decision.raw_response["opportunity_score"]
    assert opportunity["ai_expected_return_pct"] > 10.0
    assert opportunity["ai_expected_return_contribution_pct"] == 0.15
    assert opportunity["timeseries_expected_return_pct"] == 0.05
    assert opportunity["expected_net_return_pct"] < 0.05
    breakdown = opportunity["expected_net_breakdown"]
    assert breakdown["formula"] == "ai + local_ml + server_profit + timeseries - fee - slippage"
    assert breakdown["net_pct"] == opportunity["expected_net_return_pct"]
    components = {row["key"]: row for row in breakdown["components"]}
    assert components["ai"]["contribution_pct"] == 0.15
    assert components["local_ml"]["available"] is False
    assert components["server_profit"]["raw_return_pct"] == -0.28
    assert components["fee"]["contribution_pct"] < 0
    assert components["slippage"]["source"] == "dynamic_microstructure"
    assert components["slippage"]["raw_return_pct"] < 0.5
    assert breakdown["observed_not_in_formula"][0]["key"] == "sentiment"


def test_entry_opportunity_scoring_uses_advisory_ml_with_reduced_weight() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    decision = _decision()
    decision.raw_response["ml_signal"].update(
        {
            "influence_enabled": False,
            "advisory_enabled": True,
            "influence_policy": {
                "advisory_enabled": True,
                "long": {
                    "enabled": False,
                    "advisory_enabled": True,
                    "influence_weight": 0.35,
                },
            },
        }
    )

    _policy(now).score_candidate(decision, {"min_opportunity_score": 0.95})

    opportunity = decision.raw_response["opportunity_score"]
    components = {row["key"]: row for row in opportunity["expected_net_breakdown"]["components"]}

    assert opportunity["ml_influence_enabled"] is True
    assert opportunity["ml_full_influence_enabled"] is False
    assert opportunity["ml_advisory_enabled"] is True
    assert opportunity["ml_influence_weight"] == 0.35
    assert components["local_ml"]["available"] is True
    assert components["local_ml"]["weight"] == pytest.approx(0.14)
    assert components["local_ml"]["contribution_pct"] > 0


def test_entry_opportunity_scoring_does_not_use_max_slippage_as_fixed_cost(monkeypatch) -> None:
    from config.settings import settings

    monkeypatch.setattr(settings, "max_slippage_pct", 0.005)
    now = datetime(2026, 6, 10, tzinfo=UTC)
    decision = _decision()
    decision.feature_snapshot.update(
        {
            "spread_pct": 0.012,
            "orderbook_bid_depth": 150_000.0,
            "orderbook_ask_depth": 150_000.0,
            "orderbook_imbalance": 0.0,
        }
    )

    _policy(now).score_candidate(decision, {"min_opportunity_score": 0.95})

    opportunity = decision.raw_response["opportunity_score"]
    execution_cost = opportunity["execution_cost"]
    assert execution_cost["configured_max_slippage_pct"] == 0.5
    assert opportunity["slippage_pct"] == 0.05
    assert opportunity["slippage_pct"] < execution_cost["configured_max_slippage_pct"]
    assert execution_cost["slippage_source"] == "dynamic_microstructure"
