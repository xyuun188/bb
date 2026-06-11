from ai_brain.base_model import Action, DecisionOutput
from services.pipeline_context import EntryPipelineContext, ExitPipelineContext


def _decision(action: Action) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="test",
        position_size_pct=0.1,
        suggested_leverage=3.0,
        raw_response={},
        feature_snapshot={"current_price": 100.0},
    )


def test_entry_pipeline_context_public_data() -> None:
    context = EntryPipelineContext.from_inputs(
        decision=_decision(Action.LONG),
        model_name="ensemble_trader",
        model_mode="paper",
        open_positions=[{"symbol": "ETH/USDT"}],
    )

    assert context.public_data() == {
        "pipeline": "entry",
        "model_name": "ensemble_trader",
        "model_mode": "paper",
        "symbol": "BTC/USDT",
        "action": "long",
        "open_position_count": 1,
    }


def test_exit_pipeline_context_tracks_refresh_and_arbitration() -> None:
    context = ExitPipelineContext.from_inputs(
        decision=_decision(Action.CLOSE_LONG),
        model_name="ensemble_trader",
        open_positions=[{"symbol": "BTC/USDT"}],
    )
    context = context.with_arbitration({"intent": "ordinary"})
    context = context.with_refreshed_positions([{"symbol": "BTC/USDT"}, {"symbol": "ETH/USDT"}])

    assert context.public_data() == {
        "pipeline": "exit",
        "model_name": "ensemble_trader",
        "symbol": "BTC/USDT",
        "action": "close_long",
        "open_position_count": 1,
        "refreshed_position_count": 2,
        "arbitration": {"intent": "ordinary"},
    }
