from ai_brain.base_model import Action, DecisionOutput
from services.entry_market_regime import EntryMarketRegimePolicy


def test_market_regime_policy_has_no_direction_authority() -> None:
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="ARB/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="entry",
        raw_response={},
    )

    assert EntryMarketRegimePolicy().reason(
        decision,
        {"avoid_long": True, "allow_alt_long": False},
    ) is None
    assert "alt_long_style_filter" not in decision.raw_response
