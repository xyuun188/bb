from types import SimpleNamespace

from ai_brain.base_model import Action, DecisionOutput
from services.entry_evidence_probe import EntryEvidenceProbePolicy
from services.entry_probe_market_quality import EntryProbeMarketQualityPolicy


def _hold_decision(evidence: dict) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.HOLD,
        confidence=0.55,
        reasoning="hold",
        raw_response={"entry_candidate_evidence": evidence},
        feature_snapshot={"close": 100.0},
    )


def _fv(**values):
    defaults = {
        "current_price": 100.0,
        "close": 100.0,
        "returns_20": 0.0,
        "volume_ratio": 1.0,
        "price_vs_sma20": 0.0,
        "price_vs_sma50": 0.0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def _policy() -> EntryEvidenceProbePolicy:
    return EntryEvidenceProbePolicy(
        "ensemble_trader",
        lambda: 6.0,
        EntryProbeMarketQualityPolicy(),
    )


def test_entry_evidence_probe_creates_controlled_candidate() -> None:
    decision = _hold_decision(
        {
            "long": {
                "expected_net_return_pct": 0.80,
                "profit_quality_ratio": 0.70,
                "loss_probability": 0.42,
                "tail_risk_score": 0.60,
                "score": 1.1,
                "min_score_reference": 0.95,
            }
        }
    )

    candidate = _policy().create(decision, _fv(), {}, {"ml": True}, {"tools": True}, None)

    assert candidate is not None
    assert candidate.action == Action.LONG
    assert candidate.position_size_pct == 0.055
    assert candidate.suggested_leverage == 6.0
    assert candidate.raw_response["analysis_type"] == "entry_candidate"
    assert candidate.raw_response["source_analysis_type"] == "market"
    assert candidate.raw_response["evidence_profit_probe"]["triggered"] is True


def test_entry_evidence_probe_rejects_low_quality_evidence() -> None:
    decision = _hold_decision(
        {
            "long": {
                "expected_net_return_pct": 0.20,
                "profit_quality_ratio": 0.70,
                "loss_probability": 0.42,
                "tail_risk_score": 0.60,
                "score": 1.1,
                "min_score_reference": 0.95,
            }
        }
    )

    assert _policy().create(decision, _fv(), None, None, None, None) is None


def test_entry_evidence_probe_records_market_quality_block() -> None:
    decision = _hold_decision(
        {
            "long": {
                "expected_net_return_pct": 0.80,
                "profit_quality_ratio": 0.70,
                "loss_probability": 0.42,
                "tail_risk_score": 0.60,
                "score": 1.1,
                "min_score_reference": 0.95,
            }
        }
    )

    candidate = _policy().create(
        decision,
        _fv(returns_20=-0.06, price_vs_sma20=-0.2, price_vs_sma50=-0.3),
        None,
        None,
        None,
        None,
    )

    assert candidate is None
    assert decision.raw_response["evidence_profit_probe_blocked"]["blocked"] is True


def test_repeated_missed_opportunity_memory_cannot_create_tradeable_probe() -> None:
    decision = _hold_decision(
        {
            "long": {
                "expected_net_return_pct": 0.42,
                "profit_quality_ratio": 0.38,
                "loss_probability": 0.50,
                "tail_risk_score": 0.82,
                "score": 0.24,
                "min_score_reference": 0.95,
                "recommendation": "memory_supported_probe_candidate",
                "review_feedback": {
                    "allow_probe": True,
                    "missed_opportunity_count": 8,
                    "positive_evidence_count": 7,
                    "risk_evidence_count": 2,
                    "candidate_score_bonus": 0.08,
                    "max_probe_size_pct": 0.04,
                },
            }
        }
    )

    candidate = _policy().create(decision, _fv(), {}, {"ml": True}, {"tools": True}, None)

    assert candidate is None


def test_missed_opportunity_memory_does_not_override_negative_expectancy() -> None:
    decision = _hold_decision(
        {
            "long": {
                "expected_net_return_pct": -0.10,
                "profit_quality_ratio": 0.70,
                "loss_probability": 0.42,
                "tail_risk_score": 0.60,
                "score": 0.40,
                "min_score_reference": 0.95,
                "recommendation": "memory_supported_probe_candidate",
                "review_feedback": {
                    "allow_probe": True,
                    "missed_opportunity_count": 12,
                    "positive_evidence_count": 10,
                    "risk_evidence_count": 1,
                    "candidate_score_bonus": 0.12,
                },
            }
        }
    )

    assert _policy().create(decision, _fv(), None, None, None, None) is None


def test_missed_opportunity_memory_records_subthreshold_expected_net_block() -> None:
    decision = _hold_decision(
        {
            "long": {
                "expected_net_return_pct": 0.34,
                "profit_quality_ratio": 0.70,
                "loss_probability": 0.42,
                "tail_risk_score": 0.60,
                "score": 0.40,
                "min_score_reference": 0.95,
                "recommendation": "memory_watchlist_needs_probe_threshold",
                "review_feedback": {
                    "allow_probe": True,
                    "missed_opportunity_count": 90,
                    "positive_evidence_count": 90,
                    "risk_evidence_count": 0,
                    "candidate_score_bonus": 0.24,
                },
            }
        }
    )

    assert _policy().create(decision, _fv(), None, None, None, None) is None

    block = decision.raw_response["evidence_profit_probe_blocked"]
    assert block["blocked"] is True
    assert block["block_kind"] == "probe_threshold_not_met"
    assert block["block_reasons"] == ["expected_net_below_probe_threshold"]
    assert block["expected_net_return_pct"] == 0.34
    assert block["thresholds"]["min_expected_net_return_pct"] == 0.35


def test_missed_opportunity_memory_does_not_override_market_quality_block() -> None:
    decision = _hold_decision(
        {
            "long": {
                "expected_net_return_pct": 0.42,
                "profit_quality_ratio": 0.38,
                "loss_probability": 0.50,
                "tail_risk_score": 0.82,
                "score": 0.24,
                "min_score_reference": 0.95,
                "recommendation": "memory_supported_probe_candidate",
                "review_feedback": {
                    "allow_probe": True,
                    "missed_opportunity_count": 8,
                    "positive_evidence_count": 7,
                    "risk_evidence_count": 2,
                },
            }
        }
    )

    candidate = _policy().create(
        decision,
        _fv(returns_20=-0.06, price_vs_sma20=-0.2, price_vs_sma50=-0.3),
        None,
        None,
        None,
        None,
    )

    assert candidate is None
    assert decision.raw_response["evidence_profit_probe_blocked"]["blocked"] is True
