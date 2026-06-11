from types import SimpleNamespace

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.entry_evidence import build_entry_evidence_score
from services.entry_opportunity_gate import EntryOpportunityGatePolicy
from services.entry_payoff_quality import EntryLowPayoffQualityPolicy
from services.entry_signal_extraction import expected_return_pct
from services.entry_sizing import apply_evidence_sizing_policy, evidence_is_low_payoff_quality
from services.entry_stop_loss_budget import EntryStopLossBudgetPolicy
from services.entry_stress_stop import EntryStressStopPolicy
from services.profit_attribution import extract_signal_sides
from services.trading_service import TradingService


def _service() -> TradingService:
    return object.__new__(TradingService)


def _raw_response(decision: DecisionOutput) -> dict:
    raw = decision.raw_response
    assert isinstance(raw, dict)
    return raw


def _gate_reason(decision: DecisionOutput) -> str | None:
    return EntryOpportunityGatePolicy().gate_reason(decision)


def _decision(action: Action, raw: dict, *, confidence: float = 0.8) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=confidence,
        reasoning="entry",
        position_size_pct=0.08,
        suggested_leverage=6.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.06,
        raw_response=raw,
        feature_snapshot={"current_price": 100.0},
    )


def test_entry_evidence_uses_structured_review_feedback():
    decision = _decision(
        Action.LONG,
        {
            "memory_adjustment": 0.0,
            "memory_summary": {"used": 0, "positive_lessons": 0, "risk_lessons": 0},
            "memory_feedback": {
                "enabled": True,
                "by_side": {
                    "long": {
                        "score_adjustment": 0.10,
                        "allow_probe": True,
                        "action_bias": "prefer_small_probe_when_current_ev_positive",
                        "missed_opportunity_count": 4,
                        "risk_evidence_count": 0,
                    }
                },
            },
        },
    )

    evidence = build_entry_evidence_score(decision, {"ml_influence_enabled": False})
    memory_component = next(
        item for item in evidence["components"] if item["source"] == "shadow_memory"
    )

    assert memory_component["available"] is True
    assert memory_component["status"] == "aligned"
    assert memory_component["review_feedback"]["allow_probe"] is True
    assert memory_component["review_feedback"]["missed_opportunity_count"] == 4


def test_entry_evidence_blocks_ml_and_timeseries_opposite():
    service = _service()
    decision = _decision(
        Action.LONG,
        {
            "ml_signal": {"predictions": [{"best_side": "short", "best_expected_return_pct": 0.8}]},
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.5,
                },
                "profit_prediction": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.2,
                },
            },
        },
    )

    service._candidate_opportunity_score(decision, {"min_opportunity_score": 0.95})
    reason = _gate_reason(decision)

    evidence = _raw_response(decision)["opportunity_score"]["evidence_score"]
    assert evidence["hard_block"] is True
    assert any("ML 和时序" in reason for reason in evidence["hard_block_reasons"])
    assert reason is not None
    assert "动态证据评分硬拦截" in reason


def test_entry_evidence_allows_weak_conflict_probe_when_support_is_aligned():
    service = _service()
    decision = _decision(
        Action.LONG,
        {
            "ml_signal": {
                "predictions": [{"best_side": "short", "best_expected_return_pct": 0.10}]
            },
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": -0.04,
                },
                "sentiment_analysis": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 1.0,
                },
                "profit_prediction": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 1.2,
                },
            },
            "memory_adjustment": 0.15,
            "memory_summary": {"used": 4, "positive_lessons": 3, "risk_lessons": 0},
        },
        confidence=0.86,
    )

    service._candidate_opportunity_score(decision, {"min_opportunity_score": 0.95})
    reason = _gate_reason(decision)

    evidence = _raw_response(decision)["opportunity_score"]["evidence_score"]
    assert reason is None
    assert evidence["hard_block"] is False
    assert evidence["tier"] == "weak_conflict_probe"
    assert evidence["size_multiplier"] == 0.05
    assert "ml" in evidence["weak_opposites"]
    assert "timeseries" in evidence["weak_opposites"]
    assert "ml" not in evidence["major_opposites"]
    assert "server_profit" in evidence["aligned_support_sources"]


def test_entry_evidence_blocks_weak_probe_without_three_aligned_sources():
    decision = _decision(
        Action.LONG,
        {
            "ml_signal": {
                "available": True,
                "influence_enabled": False,
                "predictions": [{"best_side": "short", "best_expected_return_pct": 0.7}],
            },
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.6,
                }
            },
        },
        confidence=0.5,
    )

    evidence = build_entry_evidence_score(decision, {"ml_influence_enabled": False})

    assert 35 <= evidence["effective_score"] < 45
    assert evidence["tier"] == "blocked"
    assert evidence["hard_block"] is True
    assert evidence["size_multiplier"] == 0.0
    assert len(evidence["aligned_support_sources"]) == 2
    assert any("three aligned" in item for item in evidence["hard_block_reasons"])


def test_probe_derived_hold_entry_does_not_get_full_ai_or_server_points():
    decision = _decision(
        Action.LONG,
        {
            "opinions": [
                {"model_name": "trend_expert", "action": "hold", "confidence": 0.82},
                {"model_name": "momentum_expert", "action": "hold", "confidence": 0.80},
                {"model_name": "sentiment_expert", "action": "hold", "confidence": 0.78},
            ],
            "evidence_profit_probe": {
                "triggered": True,
                "ai_original_action": "hold",
                "side": "long",
            },
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.5,
                },
                "profit_prediction": {
                    "available": True,
                    "best_side": "long",
                    "adjusted_long_return_pct": -0.05,
                    "long_expected_return_pct": 1.2,
                },
            },
        },
        confidence=0.84,
    )

    evidence = build_entry_evidence_score(decision, {"ml_influence_enabled": False})

    ai_component = next(item for item in evidence["components"] if item["source"] == "ai")
    server_component = next(
        item for item in evidence["components"] if item["source"] == "server_profit"
    )
    assert ai_component["status"] == "probe_derived_no_expert_support"
    assert ai_component["points"] == 0.0
    assert server_component["status"] == "ignored_negative_expected"
    assert server_component["points"] == 0.0
    assert evidence["tier"] == "blocked"
    assert evidence["hard_block"] is True


def test_server_profit_uses_adjusted_side_return_before_raw_return():
    assert (
        expected_return_pct(
            {
                "best_side": "long",
                "adjusted_long_return_pct": -0.05,
                "long_expected_return_pct": 1.2,
            },
            "long",
        )
        == -0.05
    )

    decision = _decision(
        Action.LONG,
        {
            "local_ai_tools": {
                "profit_prediction": {
                    "available": True,
                    "best_side": "long",
                    "adjusted_long_return_pct": -0.05,
                    "long_expected_return_pct": 1.2,
                }
            }
        },
        confidence=0.70,
    )

    evidence = build_entry_evidence_score(decision, {"ml_influence_enabled": False})
    server_component = next(
        item for item in evidence["components"] if item["source"] == "server_profit"
    )

    assert server_component["expected_return_pct"] == -0.05
    assert server_component["status"] == "ignored_negative_expected"
    assert "server_profit" not in evidence["aligned_support_sources"]


def test_short_probe_relief_allows_tiny_short_when_quant_and_direction_align():
    decision = _decision(
        Action.SHORT,
        {
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.30,
                },
                "sentiment_analysis": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.05,
                },
                "profit_prediction": {
                    "available": True,
                    "best_side": "short",
                    "adjusted_short_return_pct": 0.20,
                    "short_loss_probability": 0.52,
                },
            },
        },
        confidence=0.29,
    )

    evidence = build_entry_evidence_score(
        decision,
        {
            "ml_influence_enabled": False,
            "local_profit_aligned": True,
            "timeseries_aligned": True,
            "server_profit_expected_return_pct": 0.20,
            "server_profit_loss_probability": 0.52,
            "direction_competition": {
                "preferred_side": "short",
                "score_gap": 0.20,
                "short": {"score": 0.50},
                "long": {"score": 0.20},
            },
        },
    )

    assert evidence["short_probe_relief"]["applied"] is True
    assert evidence["effective_score"] == 35.0
    assert evidence["tier"] == "weak_conflict_probe"
    assert evidence["size_multiplier"] == pytest.approx(0.03)


def test_entry_evidence_maps_45_to_exploration_tier():
    decision = _decision(
        Action.LONG,
        {
            "ml_signal": {
                "available": True,
                "influence_enabled": False,
                "predictions": [{"best_side": "short", "best_expected_return_pct": 0.7}],
            },
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.6,
                }
            },
        },
        confidence=0.84,
    )

    evidence = build_entry_evidence_score(decision, {"ml_influence_enabled": False})

    assert 45 <= evidence["effective_score"] < 60
    assert evidence["tier"] == "exploration"
    assert evidence["hard_block"] is False
    assert evidence["size_multiplier"] == 0.1


def test_entry_evidence_allows_strong_aligned_signal():
    service = _service()
    decision = _decision(
        Action.LONG,
        {
            "ml_signal": {"predictions": [{"best_side": "long", "best_expected_return_pct": 1.0}]},
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.6,
                },
                "sentiment_analysis": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.2,
                },
                "profit_prediction": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.4,
                },
            },
            "memory_adjustment": 0.05,
            "memory_summary": {"used": 2, "positive_lessons": 2, "risk_lessons": 0},
        },
        confidence=0.86,
    )

    service._candidate_opportunity_score(decision, {"min_opportunity_score": 0.95})
    reason = _gate_reason(decision)

    evidence = _raw_response(decision)["opportunity_score"]["evidence_score"]
    assert reason is None
    assert evidence["hard_block"] is False
    assert evidence["tier"] == "normal"
    assert evidence["score"] >= 80


def test_learning_only_ml_is_ignored_by_evidence_score():
    service = _service()
    decision = _decision(
        Action.LONG,
        {
            "ml_signal": {
                "available": True,
                "influence_enabled": False,
                "status": "learning_only",
                "predictions": [{"best_side": "short", "best_expected_return_pct": 0.8}],
            },
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.4,
                },
                "profit_prediction": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.2,
                },
            },
        },
    )

    service._candidate_opportunity_score(decision, {"min_opportunity_score": 0.95})

    evidence = _raw_response(decision)["opportunity_score"]["evidence_score"]
    ml_component = next(item for item in evidence["components"] if item["source"] == "ml")
    assert ml_component["status"] == "ignored"
    assert ml_component["points"] == 0.0
    assert "ml" not in evidence["missing_key_sources"]
    assert "ml" not in evidence["major_opposites"]


@pytest.mark.asyncio
async def test_short_evidence_score_reduces_size_even_when_aligned(monkeypatch):
    service = _service()

    async def fake_balance(*args, **kwargs):
        return 1000.0

    service.account_accounting_service = SimpleNamespace(
        allocated_order_balance=fake_balance,
    )

    class FakeExistingWinnerContext:
        def context(self, *args, **kwargs):
            return {"has_winner": False}

    service.entry_existing_winner_context = FakeExistingWinnerContext()
    service.entry_low_payoff_quality = EntryLowPayoffQualityPolicy()
    service.entry_stress_stop = EntryStressStopPolicy()
    service.entry_stop_loss_budget = EntryStopLossBudgetPolicy()
    decision = _decision(
        Action.SHORT,
        {
            "ml_signal": {"predictions": [{"best_side": "short", "best_expected_return_pct": 0.8}]},
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.4,
                },
                "sentiment_analysis": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.2,
                },
                "profit_prediction": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.2,
                },
            },
        },
        confidence=0.80,
    )

    service._candidate_opportunity_score(decision, {"min_opportunity_score": 0.95})
    original_size = decision.position_size_pct
    await service._apply_entry_profit_risk_sizing(decision, "paper", [])

    evidence = _raw_response(decision)["opportunity_score"]["evidence_score"]
    assert evidence["side"] == "short"
    assert evidence["size_multiplier"] < 1.0
    assert decision.position_size_pct < original_size


@pytest.mark.asyncio
async def test_atr_stress_stop_caps_position_by_loss_budget() -> None:
    service = _service()

    async def fake_balance(*args, **kwargs):
        return 1000.0

    service.account_accounting_service = SimpleNamespace(
        allocated_order_balance=fake_balance,
    )

    class FakeExistingWinnerContext:
        def context(self, *args, **kwargs):
            return {"has_winner": False}

    service.entry_existing_winner_context = FakeExistingWinnerContext()
    service.entry_low_payoff_quality = EntryLowPayoffQualityPolicy()
    service.entry_stress_stop = EntryStressStopPolicy()
    service.entry_stop_loss_budget = EntryStopLossBudgetPolicy()
    decision = _decision(
        Action.LONG,
        {
            "opportunity_score": {
                "score": 3.0,
                "min_score_required": 0.95,
                "expected_net_return_pct": 0.8,
                "expected_loss_pct": 1.0,
                "tail_risk_score": 0.15,
                "raw_expected_return_pct": 0.8,
                "profit_quality_ratio": 1.0,
                "server_profit_loss_probability": 0.40,
                "ml_aligned": True,
                "local_profit_aligned": True,
                "timeseries_aligned": False,
                "evidence_score": {
                    "tier": "normal",
                    "effective_score": 82.0,
                    "size_multiplier": 1.0,
                    "max_size_pct": None,
                },
            },
        },
    )
    decision.feature_snapshot = {"current_price": 100.0, "atr_14": 5.0}

    await service._apply_entry_profit_risk_sizing(decision, "paper", [])

    sizing = _raw_response(decision)["profit_risk_sizing"]
    assert sizing["stress_stop_loss_pct"] == pytest.approx(0.08)
    assert sizing["atr_pct"] == pytest.approx(0.05)
    assert sizing["applied"] is True
    assert decision.position_size_pct <= 0.0167


def test_signal_extraction_falls_back_to_evidence_components():
    signals = extract_signal_sides(
        {
            "opportunity_score": {
                "evidence_score": {
                    "components": [
                        {
                            "source": "ml",
                            "available": True,
                            "status": "aligned",
                            "side": "long",
                            "expected_return_pct": 0.7,
                        },
                        {
                            "source": "timeseries",
                            "available": True,
                            "status": "aligned",
                            "side": "long",
                            "expected_return_pct": 0.4,
                        },
                    ],
                },
            },
        }
    )

    assert signals["ml"]["available"] is True
    assert signals["ml"]["side"] == "long"
    assert signals["ml"]["expected_return_pct"] == 0.7
    assert signals["timeseries"]["side"] == "long"


def test_evidence_sizing_policy_is_independent_from_trading_service():
    result = apply_evidence_sizing_policy(
        evidence_score={
            "tier": "small",
            "effective_score": 62.0,
            "size_multiplier": 0.3,
            "max_size_pct": 0.02,
        },
        current_size=0.08,
        leverage=8.0,
    )

    assert result.position_size_pct == 0.02
    assert result.leverage == 4.0
    assert result.effective_score == 62.0
    assert result.caps
    assert evidence_is_low_payoff_quality({"tier": "small"}, 59.0) is True


def test_weak_conflict_probe_sizing_caps_position_and_leverage():
    result = apply_evidence_sizing_policy(
        evidence_score={
            "tier": "weak_conflict_probe",
            "effective_score": 38.0,
            "size_multiplier": 0.05,
            "max_size_pct": 0.01,
        },
        current_size=0.08,
        leverage=8.0,
    )

    assert result.position_size_pct == 0.004
    assert result.leverage == 2.0
    assert result.tier == "weak_conflict_probe"
    assert (
        evidence_is_low_payoff_quality(
            {"tier": "weak_conflict_probe"},
            38.0,
        )
        is True
    )
