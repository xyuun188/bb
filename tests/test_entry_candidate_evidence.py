from dataclasses import dataclass

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.entry_candidate_evidence import EntryCandidateEvidencePolicy


@dataclass
class _Feature:
    symbol: str = "BTC/USDT"

    def to_dict(self) -> dict:
        return {"current_price": 100.0, "bid": 99.99, "ask": 100.01}


def _score(decision: DecisionOutput, _strategy: dict | None) -> float:
    is_long = decision.action == Action.LONG
    expected_net = 0.8 if is_long else 0.3
    return_lcb = 0.5 if is_long else 0.1
    raw = dict(decision.raw_response)
    raw["opportunity_score"] = {
        "score": return_lcb - 0.05,
        "score_policy": "fee_after_return_lcb_minus_expected_downside",
        "expected_net_return_pct": expected_net,
        "return_lcb_pct": return_lcb,
        "return_uncertainty_pct": expected_net - return_lcb,
        "expected_loss_pct": 0.05,
        "profit_quality_ratio": expected_net / 0.05,
        "server_profit_loss_probability": 0.2,
        "tail_risk_score": 0.1,
        "production_eligible": True,
        "execution_cost": {"production_eligible": True, "total_pct": 0.05},
        "policy_provenance": {
            "source": "test_return_distribution",
            "observation_window": "test_window",
            "sample_count": 2,
            "generated_at": "2026-07-12T00:00:00+00:00",
            "strategy_version": "test.v1",
            "fallback_reason": "",
        },
    }
    decision.raw_response = raw
    return return_lcb - 0.05


def _policy(score=_score) -> EntryCandidateEvidencePolicy:
    return EntryCandidateEvidencePolicy(
        model_name="ensemble_trader",
        score_candidate=score,
        feature_opportunity_score=lambda _feature: 3.14159,
    )


def test_candidate_evidence_prefers_highest_positive_fee_after_lcb() -> None:
    evidence = _policy().build(_Feature(), {}, {}, {}, {}, {})

    assert evidence["preferred_side_by_evidence"] == "long"
    assert evidence["long"]["production_eligible"] is True
    assert evidence["long"]["return_lcb_pct"] > evidence["short"]["return_lcb_pct"]
    assert evidence["feature_opportunity_score"] == pytest.approx(3.14159)
    assert evidence["read_only"] is True
    assert evidence["is_entry_gate"] is False


def test_memory_feedback_is_observation_only_and_cannot_change_side_scores() -> None:
    memory = {
        "preferred_side_by_memory": "short",
        "by_side": {"short": {"allow_probe": True, "score_adjustment": 100.0}},
    }

    evidence = _policy().build(_Feature(), {}, {}, {}, {}, memory)

    assert evidence["preferred_side_by_evidence"] == "long"
    assert evidence["memory_feedback_observation"] == memory
    assert evidence["long"]["score"] == pytest.approx(0.45)
    assert evidence["short"]["score"] == pytest.approx(0.05)
    assert "memory_score_adjustment" not in evidence["policy_provenance"]


def test_governed_schedule_is_exposed_as_prior_without_changing_current_score() -> None:
    strategy = {
        "strategy_learning": {
            "runtime": {
                "current_market_regime": "trend",
                "governed_profiles": [
                    {
                        "id": "portfolio_long",
                        "version": 11,
                        "rank": 1,
                        "selector": {"scope": "side", "side": "long"},
                        "historical_return_distribution": {"return_lcb_pct": 0.3},
                    },
                    {
                        "id": "btc_long",
                        "version": 12,
                        "rank": 2,
                        "selector": {
                            "scope": "symbol_side",
                            "symbol": "BTC/USDT",
                            "side": "long",
                        },
                        "historical_return_distribution": {"return_lcb_pct": 0.8},
                        "walk_forward": {"return_lcb_pct": 0.6},
                        "shadow_validation": {"return_lcb_pct": 0.4},
                    },
                ],
            }
        }
    }

    evidence = _policy().build(_Feature(), strategy, {}, {}, {}, {})

    prior = evidence["long"]["scheduled_return_prior"]
    assert prior["available"] is True
    assert prior["profile_id"] == "btc_long"
    assert prior["role"] == "historical_prior_only"
    assert prior["match_status"] == "matched_historical_prior"
    assert prior["context_fields_influenced"] == ["scheduled_return_prior"]
    assert prior["can_authorize_entry"] is False
    assert prior["can_change_size_or_leverage"] is False
    assert evidence["long"]["score"] == pytest.approx(0.45)


def test_icp_does_not_match_another_symbols_historical_prior() -> None:
    strategy = {
        "strategy_learning": {
            "runtime": {
                "governed_profiles": [
                    {
                        "id": "arb_short_prior",
                        "version": 9,
                        "selector": {
                            "scope": "symbol_side",
                            "symbol": "ARB/USDT",
                            "side": "short",
                        },
                    }
                ]
            }
        }
    }

    evidence = _policy().build(_Feature(symbol="ICP/USDT"), strategy, {}, {}, {}, {})
    prior = evidence["short"]["scheduled_return_prior"]

    assert prior["available"] is False
    assert prior["match_status"] == "not_matched"
    assert prior["context_fields_influenced"] == []
    assert prior["reason"] == "no_governed_historical_prior_matches_context"


def test_arb_profitable_prior_is_recorded_but_cannot_authorize_or_resize() -> None:
    strategy = {
        "strategy_learning": {
            "runtime": {
                "governed_profiles": [
                    {
                        "id": "arb_long_prior",
                        "version": 10,
                        "rank": 1,
                        "selector": {
                            "scope": "symbol_side",
                            "symbol": "ARB/USDT",
                            "side": "long",
                        },
                        "historical_return_distribution": {"return_lcb_pct": 2.1},
                    }
                ]
            }
        }
    }

    evidence = _policy().build(_Feature(symbol="ARB/USDT"), strategy, {}, {}, {}, {})
    prior = evidence["long"]["scheduled_return_prior"]

    assert prior["available"] is True
    assert prior["profile_id"] == "arb_long_prior"
    assert prior["profile_version"] == 10
    assert prior["can_authorize_entry"] is False
    assert prior["can_change_size_or_leverage"] is False
    assert evidence["long"]["score"] == pytest.approx(0.45)


def test_governance_blocked_icp_prior_is_absent_from_runtime_matching() -> None:
    strategy = {
        "strategy_learning": {
            "runtime": {
                "governed_profiles": [],
                "blocked_profile_ids": ["icp_short_prior"],
            }
        }
    }

    evidence = _policy().build(_Feature(symbol="ICP/USDT"), strategy, {}, {}, {}, {})

    assert evidence["short"]["scheduled_return_prior"]["available"] is False
    assert evidence["short"]["scheduled_return_prior"]["can_authorize_entry"] is False


def test_candidate_evidence_has_no_legacy_probe_contract() -> None:
    evidence = _policy().build(_Feature(), {}, {}, {}, {}, {})

    for side in ("long", "short"):
        assert "probe_conversion_ready" not in evidence[side]
        assert "probe_conversion_block_reasons" not in evidence[side]
    assert "legacy_probe_permission_enabled" not in evidence["policy_provenance"]


def test_no_positive_production_lcb_returns_neutral() -> None:
    def ineligible_score(decision: DecisionOutput, _strategy: dict | None) -> float:
        raw = dict(decision.raw_response)
        raw["opportunity_score"] = {
            "score": -0.2,
            "expected_net_return_pct": 0.1,
            "return_lcb_pct": -0.1,
            "production_eligible": True,
            "policy_provenance": {"sample_count": 1},
        }
        decision.raw_response = raw
        return -0.2

    evidence = _policy(ineligible_score).build(_Feature(), {}, {}, {}, {}, {})

    assert evidence["preferred_side_by_evidence"] == "neutral"
    assert evidence["long"]["production_eligible"] is False
    assert evidence["short"]["production_eligible"] is False
