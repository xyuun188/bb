from __future__ import annotations

import pytest

from ai_brain.analysis_quality import (
    build_expert_call_contract,
    finalize_analysis_quality,
    usable_expert_opinions,
)
from ai_brain.base_model import Action, DecisionOutput
from ai_brain.cross_validator import CrossValidator


def _decision(name: str, *, raw: dict | None = None, reasoning: str = "有效证据") -> DecisionOutput:
    return DecisionOutput(
        model_name=name,
        symbol="BTC/USDT",
        action=Action.HOLD,
        confidence=0.6,
        reasoning=reasoning,
        raw_response=raw,
    )


def test_quality_contract_does_not_count_fallback_as_success() -> None:
    opinions = {
        "trend_expert": _decision("trend_expert"),
        "risk_expert": _decision("risk_expert", raw={"timeout_fallback": True}),
    }
    contract = build_expert_call_contract(
        expected_names=("trend_expert", "risk_expert"),
        attempted_names=("trend_expert", "risk_expert"),
        opinions=opinions,
        timings=[
            {"name": "trend_expert", "status": "completed"},
            {"name": "risk_expert", "status": "timeout_fallback", "reason": "timeout"},
        ],
        failures=[],
    )

    assert contract["successful_expert_count"] == 1
    assert contract["returned_expert_count"] == 2
    assert contract["status_counts"]["timeout"] == 1
    assert contract["expert_complete"] is False
    assert set(usable_expert_opinions(opinions, contract)) == {"trend_expert"}


def test_quality_contract_classifies_invalid_empty_and_unavailable() -> None:
    contract = build_expert_call_contract(
        expected_names=("trend_expert", "momentum_expert", "risk_expert"),
        attempted_names=("trend_expert", "momentum_expert"),
        opinions={"momentum_expert": _decision("momentum_expert", reasoning="")},
        timings=[{"name": "trend_expert", "status": "invalid", "reason": "invalid JSON"}],
        failures=[],
    )

    assert [row["status"] for row in contract["experts"]] == [
        "parse_failed",
        "empty",
        "unavailable",
    ]
    assert contract["reason_code"] == "insufficient_evidence"


@pytest.mark.asyncio
async def test_automatic_pairwise_cross_validation_has_full_coverage_without_llm() -> None:
    opinions = {
        name: _decision(name)
        for name in ("trend_expert", "momentum_expert", "sentiment_expert", "risk_expert")
    }
    timing: dict = {}

    validations, consultation = await CrossValidator().validate_all(opinions, timing)

    assert len(validations) == 6
    assert {row["validation_origin"] for row in validations} == {"automatic_pairwise"}
    assert timing["_cross_validation_timing"]["completed"] == 6
    assert consultation is None

    contract = build_expert_call_contract(
        expected_names=tuple(opinions),
        attempted_names=tuple(opinions),
        opinions=opinions,
        timings=[{"name": name, "status": "completed"} for name in opinions],
        failures=[],
    )
    final = finalize_analysis_quality(contract, validations, final_action="hold")
    assert final["analysis_complete"] is True
    assert final["decision_eligible"] is True
    assert final["cross_validation"]["coverage_ratio"] == 1.0


def test_unresolved_major_conflict_cannot_complete_analysis() -> None:
    opinions = {
        "trend_expert": _decision("trend_expert"),
        "risk_expert": _decision("risk_expert"),
    }
    contract = build_expert_call_contract(
        expected_names=tuple(opinions),
        attempted_names=tuple(opinions),
        opinions=opinions,
        timings=[{"name": name, "status": "completed"} for name in opinions],
        failures=[],
    )
    validations = [
        {
            "expert_pair": ["trend_expert", "risk_expert"],
            "validation_status": "completed",
            "major_conflict": True,
            "needs_resolution": True,
        }
    ]

    final = finalize_analysis_quality(contract, validations, final_action="long")

    assert final["analysis_complete"] is False
    assert final["decision_eligible"] is False
    assert final["result"] == "unclear"
    assert final["reason_code"] == "direction_conflict"
    assert final["cross_validation"]["unresolved_major_conflict_count"] == 1


def test_unresolved_major_conflict_can_be_recorded_for_paper_market_observation() -> None:
    opinions = {
        "trend_expert": _decision("trend_expert"),
        "risk_expert": _decision("risk_expert"),
    }
    contract = build_expert_call_contract(
        expected_names=tuple(opinions),
        attempted_names=tuple(opinions),
        opinions=opinions,
        timings=[{"name": name, "status": "completed"} for name in opinions],
        failures=[],
    )
    validations = [
        {
            "expert_pair": ["trend_expert", "risk_expert"],
            "validation_status": "completed",
            "major_conflict": True,
            "needs_resolution": True,
        }
    ]

    final = finalize_analysis_quality(
        contract,
        validations,
        final_action="short",
        execution_scope="paper",
        allow_paper_conflict_observation=True,
    )

    assert final["analysis_complete"] is False
    assert final["decision_eligible"] is False
    assert final["paper_observation_eligible"] is True
    assert final["paper_observation_mode"] == "model_led_conflict_observation"


def test_major_conflict_requires_matching_explicit_resolution() -> None:
    opinions = {
        "trend_expert": _decision("trend_expert"),
        "risk_expert": _decision("risk_expert"),
    }
    contract = build_expert_call_contract(
        expected_names=tuple(opinions),
        attempted_names=tuple(opinions),
        opinions=opinions,
        timings=[{"name": name, "status": "completed"} for name in opinions],
        failures=[],
    )
    validations = [
        {
            "expert_pair": ["trend_expert", "risk_expert"],
            "validation_status": "completed",
            "major_conflict": True,
        }
    ]
    consultation = {
        "status": "completed",
        "resolution_status": "resolved",
        "resolved_action": "long",
        "resolved_conflict_pairs": [["risk_expert", "trend_expert"]],
    }

    final = finalize_analysis_quality(
        contract,
        validations,
        final_action="long",
        consultation=consultation,
    )

    assert final["analysis_complete"] is True
    assert final["decision_eligible"] is True
    assert final["result"] == "long"
    assert final["cross_validation"]["conflicts_resolved"] is True
