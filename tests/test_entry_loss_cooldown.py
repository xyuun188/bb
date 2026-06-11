from ai_brain.base_model import Action, DecisionOutput
from services.entry_loss_cooldown import EntryLossCooldownPolicy
from services.entry_opportunity_gate import EntryOpportunityGatePolicy


def _decision(opportunity_score: dict, *, action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.74,
        reasoning="entry",
        position_size_pct=0.04,
        suggested_leverage=3.0,
        raw_response={"opportunity_score": opportunity_score},
    )


def _cooldown_profile() -> dict:
    return {
        "cooldown": True,
        "pnl": -32.0,
        "today_pnl": -16.0,
        "loss": 38.0,
        "today_loss": 16.0,
        "largest_loss": -14.0,
        "count": 5,
        "losses": 3,
        "wins": 2,
        "profit_factor": 0.45,
        "cooldown_remaining_hours": 2.5,
        "cooldown_reason": "recent losses",
    }


def test_entry_loss_cooldown_blocks_recent_same_side_loss() -> None:
    decision = _decision(
        {
            "score": 1.4,
            "min_score_required": 0.95,
            "expected_net_return_pct": 0.2,
            "profit_quality_ratio": 0.4,
            "reward_risk_ratio": 0.8,
            "server_profit_expected_return_pct": 0.1,
            "server_profit_loss_probability": 0.7,
            "tail_risk_score": 1.0,
            "symbol_side_profile": _cooldown_profile(),
        }
    )

    reason = EntryLossCooldownPolicy(lambda symbol: symbol.replace("/", "-")).reason(decision)

    assert reason is not None
    override = decision.raw_response["loss_cooldown_override"]
    assert override["allowed"] is False
    assert "score" in override["failed"]
    assert decision.raw_response["opportunity_score"]["loss_cooldown_override"]["allowed"] is False


def test_entry_loss_cooldown_allows_high_quality_override() -> None:
    decision = _decision(
        {
            "score": 4.2,
            "min_score_required": 0.95,
            "confidence": 0.82,
            "expected_net_return_pct": 1.1,
            "profit_quality_ratio": 1.1,
            "reward_risk_ratio": 1.6,
            "server_profit_expected_return_pct": 0.9,
            "server_profit_loss_probability": 0.32,
            "tail_risk_score": 0.4,
            "symbol_side_profile": _cooldown_profile(),
        }
    )

    reason = EntryLossCooldownPolicy().reason(decision)

    assert reason is None
    override = decision.raw_response["loss_cooldown_override"]
    assert override["allowed"] is True
    assert override["failed"] == []


def test_entry_opportunity_gate_uses_injected_loss_cooldown_policy() -> None:
    decision = _decision(
        {
            "score": 1.4,
            "min_score_required": 0.95,
            "symbol_side_profile": _cooldown_profile(),
        }
    )
    policy = EntryOpportunityGatePolicy(symbol_loss_cooldown_policy=EntryLossCooldownPolicy())

    reason = policy.gate_reason(decision)

    assert reason is not None
    assert decision.raw_response["loss_cooldown_override"]["allowed"] is False
