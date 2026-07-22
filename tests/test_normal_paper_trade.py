from __future__ import annotations

from ai_brain.base_model import Action, DecisionOutput
from services.normal_paper_trade import (
    ensure_normal_paper_trade_contract,
    normal_paper_trade_contract_reasons,
)


def _decision(raw: dict, *, action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.5,
        reasoning="test",
        position_size_pct=0.0,
        suggested_leverage=2.0,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        raw_response=raw,
        feature_snapshot={"current_price": 100.0},
    )


def test_all_paper_entry_routes_share_one_normal_trade_lifecycle() -> None:
    cases = (
        ({"paper_training": {"authorized": True, "prediction_horizon_minutes": 10}}, "cold_start_exploration"),
        ({"paper_exploration": {"authorized": True, "prediction_horizon_minutes": 30}}, "bounded_exploration"),
        (
            {
                "authoritative_return_candidate": {"production_eligible": True},
                "opportunity_score": {
                    "return_distribution_contract": {"horizon_minutes": 60}
                },
            },
            "evidence_best",
        ),
    )

    for raw, route_kind in cases:
        decision = _decision(raw)
        contract = ensure_normal_paper_trade_contract(decision, "paper")

        assert normal_paper_trade_contract_reasons(contract) == []
        assert contract["trade_kind"] == "normal_paper_trade"
        assert contract["route_kind"] == route_kind
        assert contract["uses_shared_order_pipeline"] is True
        assert contract["uses_shared_position_ledger"] is True
        assert contract["separate_sampling_order"] is False
        assert contract["continuous_training_after_trusted_settlement"] is True
        assert contract["sample_target"] is None
        assert contract["daily_sample_quota"] is None


def test_unified_paper_trade_contract_never_attaches_to_live() -> None:
    decision = _decision(
        {"opportunity_score": {"return_distribution_contract": {"horizon_minutes": 10}}}
    )

    assert ensure_normal_paper_trade_contract(decision, "live") == {}
    assert "normal_paper_trade" not in decision.raw_response
