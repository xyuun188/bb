from copy import deepcopy

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.entry_profit_risk_sizing import EntryProfitRiskSizingPolicy


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="dynamic sizing test",
        position_size_pct=0.07,
        suggested_leverage=8.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.08,
        feature_snapshot={"current_price": 100.0, "atr_14": 1.5},
        raw_response={
            "opportunity_score": {
                "expected_net_return_pct": 0.9,
                "expected_loss_pct": 0.2,
                "server_profit_loss_probability": 0.25,
                "tail_risk_score": 0.2,
                "profit_quality_ratio": 1.6,
                "execution_cost": {
                    "production_eligible": True,
                    "total_pct": 0.08,
                    "slippage_pct": 0.02,
                    "spread_pct": 0.02,
                },
                "expected_net_breakdown": {
                    "components": [
                        {
                            "key": "server_profit",
                            "production_eligible": True,
                            "raw_return_pct": 1.1,
                        }
                    ]
                },
            }
        },
    )


async def _balance(_mode: str, _decision: DecisionOutput | None) -> float:
    return 1000.0


@pytest.mark.asyncio
async def test_legacy_probe_and_evidence_payload_cannot_change_dynamic_sizing() -> None:
    first = _decision()
    second = deepcopy(first)
    first.raw_response.update(
        {
            "quant_profit_probe": {"triggered": True, "strong_probe": True},
            "evidence_profit_probe": {"triggered": True},
        }
    )
    first.raw_response["opportunity_score"]["evidence_score"] = {
        "tier": "blocked",
        "shadow_only": True,
    }
    second.raw_response["opportunity_score"]["evidence_score"] = {
        "tier": "elite",
        "tradeable_probe": True,
    }
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(first, "paper", [])
    await policy.apply(second, "paper", [])

    assert first.position_size_pct == pytest.approx(second.position_size_pct)
    assert first.suggested_leverage == second.suggested_leverage
    sizing = first.raw_response["profit_risk_sizing"]
    assert sizing["production_eligible"] is True
    provenance = sizing["policy_provenance"]
    assert "legacy_evidence_tier_enabled" not in provenance
    assert "legacy_probe_sizing_enabled" not in provenance
    assert {
        "source",
        "observation_window",
        "sample_count",
        "generated_at",
        "strategy_version",
        "fallback_reason",
    }.issubset(provenance)


@pytest.mark.asyncio
async def test_missing_production_cost_fails_closed_without_fixed_fallback() -> None:
    decision = _decision()
    decision.raw_response["opportunity_score"]["execution_cost"][
        "production_eligible"
    ] = False
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(decision, "paper", [])

    assert decision.position_size_pct == 0.0
    assert decision.suggested_leverage == 1.0
    sizing = decision.raw_response["profit_risk_sizing"]
    assert sizing["production_eligible"] is False
    assert "production_execution_cost_missing" in sizing["reason"]
