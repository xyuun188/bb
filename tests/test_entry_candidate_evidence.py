from types import SimpleNamespace
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_candidate_evidence import EntryCandidateEvidencePolicy
from services.trading_service import TradingService


def _feature() -> SimpleNamespace:
    feature = SimpleNamespace(symbol="BTC/USDT", current_price=100.0)
    feature.to_dict = lambda: {"current_price": 100.0}
    return feature


def _policy(
    opportunities: dict[str, dict[str, Any]],
    scores: dict[str, float],
) -> EntryCandidateEvidencePolicy:
    def score_candidate(
        decision: DecisionOutput,
        strategy: dict[str, Any] | None,
    ) -> float:
        side = "long" if decision.action == Action.LONG else "short"
        assert strategy == {"mode": "test"}
        assert decision.raw_response["pre_ai_candidate_evidence"] is True
        assert decision.raw_response["direction_competition"]["preferred_side"] == "long"
        decision.raw_response["opportunity_score"] = dict(opportunities[side])
        return scores[side]

    return EntryCandidateEvidencePolicy(
        model_name="ensemble_trader",
        score_candidate=score_candidate,
        feature_opportunity_score=lambda _feature: 3.14159,
    )


def test_candidate_evidence_builds_high_profit_long_side_and_compacts_history() -> None:
    profile = {
        "count": "3",
        "pnl": 12.34567,
        "today_pnl": 2.0,
        "wins": 2,
        "losses": 1,
        "profit_factor": 2.4,
        "largest_loss": -1.25,
        "first_closed_at": "2026-06-08T10:00:00+00:00",
        "last_closed_at": "2026-06-09T10:00:00+00:00",
        "last_loss_at": "2026-06-08T11:00:00+00:00",
        "last_loss_age_hours": 23.5,
        "lookback_days": 14,
        "cooldown": False,
        "cooldown_reason": "not active",
    }
    opportunities = {
        "long": {
            "expected_net_return_pct": 1.4,
            "tail_risk_score": 0.5,
            "server_profit_loss_probability": 0.3,
            "profit_quality_ratio": 1.3,
            "min_score_required": 0.7,
            "local_profit_aligned": True,
            "reward_risk_ratio": 2.0,
            "symbol_profile": profile,
            "symbol_side_profile": profile,
        },
        "short": {
            "expected_net_return_pct": 0.2,
            "tail_risk_score": 0.75,
            "server_profit_loss_probability": 0.52,
            "profit_quality_ratio": 0.4,
            "min_score_required": 0.7,
        },
    }

    evidence = _policy(opportunities, {"long": 1.35, "short": 0.75}).build(
        _feature(),
        {"mode": "test"},
        {"predictions": []},
        {"profit_prediction": {}},
        {"preferred_side": "long"},
    )

    assert evidence["feature_opportunity_score"] == 3.1416
    assert evidence["preferred_side_by_evidence"] == "long"
    assert evidence["long"]["high_profit_potential"] is True
    assert (
        evidence["long"]["recommendation"] == "high_profit_candidate_allow_larger_size_and_leverage"
    )
    assert evidence["short"]["recommendation"] == "tradable_if_ai_thesis_confirms"
    assert evidence["long"]["symbol_side_profile"]["last_closed_at"] == (
        "2026-06-09T10:00:00+00:00"
    )
    assert evidence["long"]["symbol_side_profile"]["pnl"] == 12.3457


def test_candidate_evidence_marks_low_quality_side_as_tiny_probe_only() -> None:
    opportunities = {
        "long": {
            "expected_net_return_pct": -0.1,
            "tail_risk_score": 0.4,
            "server_profit_loss_probability": 0.5,
            "profit_quality_ratio": 0.5,
            "min_score_required": 0.7,
        },
        "short": {
            "expected_net_return_pct": 0.1,
            "tail_risk_score": 1.2,
            "server_profit_loss_probability": 0.5,
            "profit_quality_ratio": 0.6,
            "min_score_required": 0.7,
        },
    }

    evidence = _policy(opportunities, {"long": 0.2, "short": 0.22}).build(
        _feature(),
        {"mode": "test"},
        {},
        {},
        {"preferred_side": "long"},
    )

    assert evidence["preferred_side_by_evidence"] == "neutral"
    assert evidence["long"]["recommendation"] == "hold_or_tiny_probe_only"
    assert evidence["short"]["recommendation"] == "hold_or_tiny_probe_only"


def test_candidate_evidence_uses_memory_feedback_without_changing_expected_return() -> None:
    opportunities = {
        "long": {
            "expected_net_return_pct": 0.35,
            "tail_risk_score": 0.82,
            "server_profit_loss_probability": 0.48,
            "profit_quality_ratio": 0.35,
            "min_score_required": 0.7,
        },
        "short": {
            "expected_net_return_pct": 0.20,
            "tail_risk_score": 0.8,
            "server_profit_loss_probability": 0.50,
            "profit_quality_ratio": 0.30,
            "min_score_required": 0.7,
        },
    }
    memory_feedback = {
        "enabled": True,
        "preferred_side_by_memory": "long",
        "by_side": {
            "long": {
                "side": "long",
                "candidate_score_bonus": 0.18,
                "score_adjustment": 0.12,
                "allow_probe": True,
                "action_bias": "prefer_small_probe_when_current_ev_positive",
                "missed_opportunity_count": 5,
                "positive_evidence_count": 5,
                "risk_evidence_count": 0,
                "max_probe_size_pct": 0.015,
            },
            "short": {
                "side": "short",
                "candidate_score_bonus": 0.0,
                "score_adjustment": 0.0,
                "allow_probe": False,
                "action_bias": "neutral",
            },
        },
    }

    evidence = _policy(opportunities, {"long": 0.52, "short": 0.51}).build(
        _feature(),
        {"mode": "test"},
        {},
        {},
        {"preferred_side": "long"},
        memory_feedback,
    )

    assert evidence["preferred_side_by_evidence"] == "long"
    assert evidence["memory_feedback"]["preferred_side_by_memory"] == "long"
    assert evidence["long"]["score_before_memory_feedback"] == 0.52
    assert evidence["long"]["score"] == 0.70
    assert evidence["long"]["expected_net_return_pct"] == 0.35
    assert evidence["long"]["recommendation"] == "memory_supported_probe_candidate"
    assert evidence["long"]["review_feedback"]["missed_opportunity_count"] == 5


def test_trading_service_candidate_evidence_delegates_to_policy() -> None:
    service = object.__new__(TradingService)

    class FakeEntryPolicy:
        def score_candidate(
            self,
            decision: DecisionOutput,
            _strategy: dict[str, Any] | None,
        ) -> float:
            side = "long" if decision.action == Action.LONG else "short"
            decision.raw_response["opportunity_score"] = {
                "expected_net_return_pct": 0.8 if side == "long" else 0.1,
                "tail_risk_score": 0.3,
                "server_profit_loss_probability": 0.35,
                "profit_quality_ratio": 1.0,
                "min_score_required": 0.7,
            }
            return 0.9 if side == "long" else 0.3

    service.entry_policy = FakeEntryPolicy()
    evidence = service._ai_entry_candidate_evidence(
        _feature(),
        {},
        {},
        {},
        {},
    )

    assert evidence["enabled"] is True
    assert evidence["preferred_side_by_evidence"] == "long"
    assert evidence["long"]["score"] == 0.9
