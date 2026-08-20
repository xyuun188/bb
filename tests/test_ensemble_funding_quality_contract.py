from ai_brain.ensemble_coordinator import _apply_market_funding_quality_contract


def _complete_quality_contract() -> dict:
    return {
        "analysis_complete": True,
        "decision_eligible": True,
        "result": "complete",
        "reason_code": "complete",
        "reason": "complete",
    }


def test_complete_market_funding_projection_preserves_analysis_quality() -> None:
    quality = _complete_quality_contract()

    _apply_market_funding_quality_contract(
        quality,
        {
            "direction_competition": {
                "enabled": False,
                "funding_projection": {"evidence_complete": True},
            }
        },
    )

    assert quality["funding_evidence_status"] == "complete"
    assert quality["analysis_complete"] is True
    assert quality["decision_eligible"] is True


def test_missing_market_funding_projection_blocks_analysis_quality() -> None:
    quality = _complete_quality_contract()

    _apply_market_funding_quality_contract(quality, {"direction_competition": {}})

    assert quality["funding_evidence_status"] == "funding_evidence_unavailable"
    assert quality["analysis_complete"] is False
    assert quality["decision_eligible"] is False
    assert quality["reason_code"] == "funding_evidence_unavailable"


def test_position_analysis_does_not_use_market_funding_projection_contract() -> None:
    quality = _complete_quality_contract()

    _apply_market_funding_quality_contract(quality, {"review_positions": [{}]})

    assert "funding_evidence_status" not in quality
    assert quality["analysis_complete"] is True
