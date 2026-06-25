from types import SimpleNamespace

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.entry_evidence import build_entry_evidence_score
from services.entry_opportunity_gate import EntryOpportunityGatePolicy
from services.entry_payoff_quality import EntryLowPayoffQualityPolicy
from services.entry_signal_extraction import directional_expected_return_pct, expected_return_pct
from services.entry_sizing import (
    apply_evidence_sizing_policy,
    evidence_is_low_payoff_quality,
    evidence_is_tradeable_probe,
)
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


def test_directional_expected_return_flips_generic_short_move() -> None:
    payload = {
        "best_side": "short",
        "direction": "down",
        "expected_move_pct": -0.42,
        "expected_return_pct": -0.42,
    }

    assert expected_return_pct(payload, "short") == -0.42
    assert directional_expected_return_pct(payload, "short") == 0.42
    assert directional_expected_return_pct(payload, "long") == -0.42


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
    assert "动态证据强冲突硬拦截" in reason


def test_entry_opportunity_advisory_keeps_strong_quality_size() -> None:
    decision = _decision(
        Action.LONG,
        {
            "opportunity_score": {
                "score": 0.80,
                "min_score_required": 0.95,
                "confidence": 0.82,
                "side": "long",
                "expected_net_return_pct": 1.10,
                "profit_quality_ratio": 1.10,
                "server_profit_loss_probability": 0.35,
                "tail_risk_score": 0.40,
                "success_probability": 0.68,
                "local_profit_aligned": True,
                "ml_aligned": True,
                "timeseries_aligned": True,
                "direction_preferred_side": "short",
                "direction_competition": {
                    "preferred_side": "short",
                    "score_gap": 0.18,
                    "long": {"score": 0.05},
                    "short": {"score": 0.23},
                },
                "server_profit_best_side": "short",
                "server_profit_expected_return_pct": -0.10,
            }
        },
        confidence=0.82,
    )

    reason = _gate_reason(decision)

    assert reason is None
    assert decision.position_size_pct == 0.08
    warnings = decision.raw_response["opportunity_score"]["execution_advisory_warnings"]
    assert warnings
    assert any(item.get("size_cap_skipped") is True for item in warnings)
    override = next(item["size_cap_override"] for item in warnings if item.get("size_cap_skipped"))
    assert override["strong_quality"] is True
    assert override["aligned_sources"] == ["local_profit", "ml", "timeseries"]


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


def test_missing_ml_and_timeseries_degrades_to_tiny_probe():
    service = _service()
    decision = _decision(
        Action.LONG,
        {
            "local_ai_tools": {
                "profit_prediction": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.22,
                },
                "sentiment_analysis": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.18,
                },
            },
        },
        confidence=0.72,
    )

    service._candidate_opportunity_score(decision, {"min_opportunity_score": 0.95})
    reason = _gate_reason(decision)

    evidence = _raw_response(decision)["opportunity_score"]["evidence_score"]
    assert reason is None
    assert evidence["hard_block"] is False
    assert evidence["tier"] == "degraded_missing_probe"
    assert evidence["size_multiplier"] == 0.05
    assert evidence["missing_key_sources"] == ["ml", "timeseries"]
    assert evidence["missing_key_degraded_relief"]["applied"] is True
    assert evidence["missing_key_degraded_relief"]["tradeable_probe"] is False
    assert evidence["missing_key_degraded_relief"]["shadow_only"] is True
    assert evidence["tradeable_probe"] is False
    assert evidence["shadow_only"] is True
    assert "动态证据评分硬拦截" not in str(reason or "")


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
    assert evidence["hard_block"] is False
    assert evidence["size_multiplier"] == 0.0
    assert len(evidence["aligned_support_sources"]) == 2
    assert any("three aligned" in item for item in evidence["advisory_wait_reasons"])
    assert any("观望" in item or "探针" in item for item in evidence["advisory_wait_reasons"])


def test_gate_blocks_selected_side_negative_even_when_aggregate_is_positive() -> None:
    decision = _decision(
        Action.LONG,
        {
            "opportunity_score": {
                "score": 3.2,
                "min_score_required": 0.95,
                "expected_net_return_pct": 2.4,
                "profit_quality_ratio": 1.6,
                "server_profit_loss_probability": 0.25,
                "tail_risk_score": 0.20,
                "ml_aligned": True,
                "local_profit_aligned": True,
            },
            "entry_candidate_evidence": {
                "long": {
                    "expected_net_return_pct": -0.05,
                    "profit_quality_ratio": -0.1,
                    "loss_probability": 0.62,
                    "tail_risk_score": 0.40,
                },
                "short": {
                    "expected_net_return_pct": 2.4,
                    "profit_quality_ratio": 1.6,
                    "loss_probability": 0.25,
                    "tail_risk_score": 0.20,
                },
            },
        },
        confidence=0.88,
    )

    reason = _gate_reason(decision)

    assert reason is not None
    assert "-0.0500%" in reason
    gate = _raw_response(decision)["opportunity_score"]["selected_side_quality_gate"]
    assert gate["blocked"] is True
    assert gate["side"] == "long"
    assert gate["selected_expected_net_return_pct"] == pytest.approx(-0.05)
    assert gate["aggregate_expected_net_return_pct"] == pytest.approx(2.4)


def test_positive_net_return_relieves_blocked_evidence_to_controlled_probe():
    decision = _decision(
        Action.SHORT,
        {
            "ml_signal": {"predictions": [{"best_side": "long", "best_expected_return_pct": 0.8}]},
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.1,
                }
            },
        },
        confidence=0.78,
    )

    evidence = build_entry_evidence_score(
        decision,
        {
            "score": 2.40,
            "min_score_required": 0.95,
            "expected_net_return_pct": 1.60,
            "profit_quality_ratio": 0.70,
            "server_profit_loss_probability": 0.42,
            "tail_risk_score": 0.35,
            "confidence": 0.78,
        },
    )

    assert evidence["tier"] == "weak_conflict_probe"
    assert evidence["hard_block"] is False
    assert evidence["size_multiplier"] > 0.0
    assert evidence["positive_net_probe_relief"]["applied"] is True
    assert evidence["positive_net_probe_relief"]["expected_net_return_pct"] == pytest.approx(1.60)
    assert evidence["positive_net_probe_relief"]["tradeable_probe"] is False
    assert evidence["positive_net_probe_relief"]["shadow_only"] is True
    assert evidence["tradeable_probe"] is False
    assert evidence["shadow_only"] is True
    assert not any("three aligned" in item for item in evidence["advisory_wait_reasons"])


def test_positive_net_weak_conflict_probe_stays_shadow_only_without_score_lift():
    decision = _decision(
        Action.SHORT,
        {
            "ml_signal": {"predictions": [{"best_side": "long", "best_expected_return_pct": 0.8}]},
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.2,
                },
                "profit_prediction": {
                    "available": True,
                    "best_side": "short",
                    "adjusted_short_return_pct": 0.9,
                    "short_loss_probability": 0.40,
                    "profit_quality_score": 0.75,
                },
            },
        },
        confidence=0.82,
    )

    evidence = build_entry_evidence_score(
        decision,
        {
            "score": 2.80,
            "min_score_required": 0.95,
            "expected_net_return_pct": 0.80,
            "profit_quality_ratio": 0.65,
            "server_profit_loss_probability": 0.40,
            "tail_risk_score": 0.45,
            "confidence": 0.82,
        },
    )

    assert evidence["tier"] == "weak_conflict_probe"
    assert evidence["positive_net_probe_relief"]["applied"] is True
    assert evidence["positive_net_probe_relief"]["tradeable_probe"] is False
    assert evidence["positive_net_probe_relief"]["shadow_only"] is True
    assert evidence["tradeable_probe"] is False
    assert evidence["shadow_only"] is True


def test_repeated_missed_opportunity_memory_lifts_positive_weak_probe_to_tradeable():
    decision = _decision(
        Action.LONG,
        {
            "memory_feedback": {
                "by_side": {
                    "long": {
                        "score_adjustment": 0.18,
                        "candidate_score_bonus": 0.24,
                        "allow_probe": True,
                        "action_bias": "prefer_small_probe_when_current_ev_positive",
                        "max_probe_size_pct": 0.025,
                        "missed_opportunity_count": 12,
                        "risk_evidence_count": 0,
                    }
                },
                "decision_habit": {
                    "by_side": {
                        "long": {
                            "stance": "probe_when_ev_ok",
                            "proactive_level": 0.85,
                            "probe_budget_pct": 0.025,
                        }
                    }
                },
            },
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.2,
                }
            },
        },
        confidence=0.74,
    )

    evidence = build_entry_evidence_score(
        decision,
        {
            "score": 1.05,
            "min_score_required": 0.95,
            "expected_net_return_pct": 0.72,
            "profit_quality_ratio": 0.65,
            "server_profit_loss_probability": 0.50,
            "tail_risk_score": 0.60,
            "confidence": 0.74,
        },
    )

    relief = evidence["memory_missed_opportunity_relief"]
    assert relief["applied"] is True
    assert relief["missed_opportunity_count"] == 12
    assert relief["tradeable_probe"] is True
    assert evidence["tier"] == "exploration"
    assert evidence["tradeable_probe"] is True
    assert evidence["shadow_only"] is False
    assert evidence["max_size_pct"] <= 0.015


def test_strong_positive_net_relief_lifts_weak_conflict_out_of_micro_probe():
    decision = _decision(
        Action.SHORT,
        {
            "ml_signal": {"predictions": [{"best_side": "short", "best_expected_return_pct": 1.1}]},
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.8,
                },
                "sentiment_analysis": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.18,
                },
                "profit_prediction": {
                    "available": True,
                    "best_side": "short",
                    "adjusted_short_return_pct": 2.3,
                    "short_loss_probability": 0.35,
                    "profit_quality_score": 4.2,
                },
            },
            "memory_adjustment": -0.10,
            "memory_summary": {"used": 3, "positive_lessons": 0, "risk_lessons": 2},
        },
        confidence=0.92,
    )

    evidence = build_entry_evidence_score(
        decision,
        {
            "score": 9.27,
            "min_score_required": 0.95,
            "expected_net_return_pct": 2.58,
            "profit_quality_ratio": 4.23,
            "server_profit_loss_probability": 0.35,
            "tail_risk_score": 0.55,
            "confidence": 0.92,
        },
    )

    assert evidence["hard_block"] is False
    assert evidence["positive_net_probe_relief"]["applied"] is True
    assert evidence["positive_net_probe_relief"]["tradeable_probe"] is True
    assert evidence["positive_net_probe_relief"]["shadow_only"] is False
    assert evidence["strong_positive_net_relief"]["applied"] is True
    assert evidence["strong_positive_net_relief"]["tier_floor"] == "small"
    assert evidence["tier"] == "small"
    assert evidence["size_multiplier"] == pytest.approx(0.18)
    assert evidence["max_size_pct"] <= 0.025
    assert evidence["tradeable_probe"] is True
    assert evidence["shadow_only"] is False


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
    assert evidence["hard_block"] is False
    assert evidence["advisory_wait_reasons"]
    assert any("观望" in item or "探针" in item for item in evidence["advisory_wait_reasons"])


def test_probe_derived_hold_entry_gets_capped_ai_points_after_independent_support():
    decision = _decision(
        Action.LONG,
        {
            "opinions": [
                {
                    "model_name": "trend_expert",
                    "action": "long",
                    "confidence": 0.72,
                    "independent_expert_retry": True,
                },
                {
                    "model_name": "momentum_expert",
                    "action": "long",
                    "confidence": 0.70,
                    "independent_expert_retry": True,
                },
                {"model_name": "sentiment_expert", "action": "hold", "confidence": 0.52},
            ],
            "evidence_profit_probe": {
                "triggered": True,
                "ai_original_action": "hold",
                "side": "long",
            },
            "ml_signal": {
                "available": True,
                "influence_enabled": True,
                "predictions": [
                    {
                        "best_side": "long",
                        "best_expected_return_pct": 0.55,
                    }
                ],
            },
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.45,
                },
                "sentiment_analysis": {
                    "available": True,
                    "best_side": "long",
                    "expected_return_pct": 0.20,
                },
                "profit_prediction": {
                    "available": True,
                    "best_side": "long",
                    "adjusted_long_return_pct": 0.65,
                },
            },
        },
        confidence=0.84,
    )

    evidence = build_entry_evidence_score(decision, {"ml_influence_enabled": True})

    ai_component = next(item for item in evidence["components"] if item["source"] == "ai")
    assert ai_component["status"] == "probe_derived_independent_expert_support"
    assert ai_component["directional_support_count"] == 2
    assert ai_component["points"] == 18.0
    assert "ai" not in evidence["aligned_support_sources"]
    assert evidence["tier"] in {"small", "medium", "normal"}


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


def test_short_probe_relief_stays_shadow_only_when_quant_and_direction_align():
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
    assert evidence["short_probe_relief"]["tradeable_probe"] is False
    assert evidence["short_probe_relief"]["shadow_only"] is True
    assert evidence["tradeable_probe"] is False
    assert evidence["shadow_only"] is True


def test_strong_aligned_short_signal_uses_dynamic_offset_and_full_size():
    decision = _decision(
        Action.SHORT,
        {
            "ml_signal": {
                "predictions": [{"best_side": "short", "best_expected_return_pct": 1.20}]
            },
            "local_ai_tools": {
                "time_series_prediction": {
                    "available": True,
                    "best_side": "short",
                    "short_expected_return_pct": 0.90,
                },
                "sentiment_analysis": {
                    "available": True,
                    "best_side": "short",
                    "expected_return_pct": 0.30,
                },
                "profit_prediction": {
                    "available": True,
                    "best_side": "short",
                    "adjusted_short_return_pct": 1.40,
                    "short_loss_probability": 0.34,
                },
            },
        },
        confidence=0.90,
    )

    evidence = build_entry_evidence_score(
        decision,
        {
            "score": 3.80,
            "min_score_required": 0.95,
            "expected_net_return_pct": 1.45,
            "profit_quality_ratio": 1.35,
            "server_profit_loss_probability": 0.34,
            "tail_risk_score": 0.40,
            "confidence": 0.90,
        },
    )

    adjustment = evidence["short_evidence_adjustment"]
    assert evidence["side"] == "short"
    assert evidence["tier"] == "normal"
    assert adjustment["score_offset"] == 0.0
    assert adjustment["size_multiplier"] == 1.0
    assert evidence["effective_score"] == evidence["score"]
    assert evidence["size_multiplier"] == 1.0


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
    assert evidence_is_tradeable_probe({"tier": "weak_conflict_probe"}, 38.0) is False
    assert evidence_is_tradeable_probe({"tier": "exploration"}, 45.0) is True


@pytest.mark.asyncio
async def test_probe_budget_caps_loss_to_balanced_probe_limit() -> None:
    service = _service()

    async def fake_balance(*args, **kwargs):
        return 1000.0

    service.account_accounting_service = SimpleNamespace(allocated_order_balance=fake_balance)

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
            "opportunity_score": {
                "score": 1.6,
                "min_score_required": 0.55,
                "expected_net_return_pct": 0.6,
                "expected_loss_pct": 0.4,
                "tail_risk_score": 0.2,
                "raw_expected_return_pct": 0.6,
                "profit_quality_ratio": 1.0,
                "server_profit_loss_probability": 0.45,
                "max_entry_stop_loss_usdt": 16.0,
                "risk_mode": "normal",
            },
            "quant_profit_probe": {
                "triggered": True,
                "strong_probe": False,
                "side": "short",
                "loss_probability": 0.5,
            },
        },
        confidence=0.62,
    )
    decision.position_size_pct = 0.05
    decision.suggested_leverage = 3.0
    decision.stop_loss_pct = 0.012
    decision.feature_snapshot = {"current_price": 100.0}

    await service._apply_entry_profit_risk_sizing(decision, "paper", [])

    sizing = _raw_response(decision)["profit_risk_sizing"]
    guard = sizing["probe_budget_guard"]
    assert guard["applied"] is True
    assert guard["strong_probe"] is False
    assert guard["max_stop_loss_usdt"] == pytest.approx(5.0)
    assert guard["previous_max_stop_loss_usdt"] > 5.0
    # Later structural guards may tighten further, but never above the probe budget.
    assert sizing["max_stop_loss_usdt"] <= 5.0


@pytest.mark.asyncio
async def test_high_quality_entry_escapes_learning_probe_micro_size() -> None:
    service = _service()

    async def fake_balance(*args, **kwargs):
        return 1000.0

    service.account_accounting_service = SimpleNamespace(allocated_order_balance=fake_balance)

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
                "score": 3.4,
                "min_score_required": 0.95,
                "expected_net_return_pct": 1.8,
                "expected_loss_pct": 0.35,
                "tail_risk_score": 0.25,
                "raw_expected_return_pct": 1.9,
                "profit_quality_ratio": 1.6,
                "server_profit_loss_probability": 0.30,
                "risk_mode": "normal",
                "ml_aligned": True,
                "local_profit_aligned": True,
                "timeseries_aligned": True,
                "evidence_score": {
                    "tier": "normal",
                    "effective_score": 88.0,
                    "size_multiplier": 1.0,
                    "max_size_pct": None,
                },
            },
            "strategy_learning_context": {
                "strategy_learning_release_pressure_active": True,
                "strategy_learning_sizing": {
                    "profile_id": "loss_release",
                    "release_pressure_active": True,
                    "position_size_multiplier": 0.25,
                    "probe_fraction": 0.03,
                    "max_probe_size_pct": 0.012,
                },
            },
        },
        confidence=0.82,
    )
    decision.position_size_pct = 0.08
    decision.suggested_leverage = 6.0
    decision.stop_loss_pct = 0.012
    decision.feature_snapshot = {"current_price": 100.0, "atr_14": 0.6}

    await service._apply_entry_profit_risk_sizing(decision, "paper", [])

    sizing = _raw_response(decision)["profit_risk_sizing"]
    assert sizing["high_quality_entry"] is True
    assert sizing["low_payoff_quality"] is False
    assert sizing["strategy_learning_sizing"]["quality_override"] is True
    assert sizing["notional_floor_applied"] is True
    assert decision.position_size_pct >= 0.06
