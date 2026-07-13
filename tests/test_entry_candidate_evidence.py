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
