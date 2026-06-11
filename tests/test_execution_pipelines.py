import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.execution_pipelines import EntryExecutionPipeline, ExitExecutionPipeline
from services.trading_policies import PolicyGateResult


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


@pytest.mark.asyncio
async def test_entry_execution_pipeline_attaches_parameter_snapshot() -> None:
    calls: list[tuple[str, str, int]] = []

    class FakeEntryPolicy:
        async def evaluate(self, decision, model_name, model_mode, open_positions):
            calls.append((decision.symbol, model_name, len(open_positions or [])))
            return PolicyGateResult.allow({"intent": "entry"})

    decision = _decision(Action.LONG)
    result = await EntryExecutionPipeline(lambda: FakeEntryPolicy()).evaluate(
        decision,
        "ensemble_trader",
        "paper",
        [{"symbol": "ETH/USDT"}],
    )

    assert calls == [("BTC/USDT", "ensemble_trader", 1)]
    assert result.data["strategy_parameters"]["scope"] == "entry_execution"
    assert decision.raw_response["strategy_parameters"]["snapshot"]["version"]


@pytest.mark.asyncio
async def test_exit_execution_pipeline_attaches_parameter_snapshot() -> None:
    calls: list[tuple[str, str, int, bool]] = []

    class FakeExitPolicy:
        async def evaluate(
            self,
            decision,
            model_name,
            open_positions,
            *,
            refresh_positions=True,
        ):
            calls.append(
                (decision.symbol, model_name, len(open_positions or []), refresh_positions)
            )
            return PolicyGateResult.allow({"intent": "exit"})

    decision = _decision(Action.CLOSE_LONG)
    result = await ExitExecutionPipeline(lambda: FakeExitPolicy()).evaluate(
        decision,
        "ensemble_trader",
        [{"symbol": "BTC/USDT"}],
        refresh_positions=False,
    )

    assert calls == [("BTC/USDT", "ensemble_trader", 1, False)]
    assert result.data["strategy_parameters"]["scope"] == "exit_execution"
    assert decision.raw_response["exit_pipeline"] == {
        "intent": "ordinary",
        "stage": "pre_policy",
        "structured": True,
    }
    assert decision.raw_response["strategy_parameters"]["snapshot"]["version"]


@pytest.mark.asyncio
async def test_execution_pipelines_fail_fast_without_policy() -> None:
    with pytest.raises(RuntimeError, match="EntryExecutionPipeline.entry_policy"):
        await EntryExecutionPipeline(lambda: None).evaluate(
            _decision(Action.LONG),
            "ensemble_trader",
            "paper",
            [],
        )

    with pytest.raises(RuntimeError, match="ExitExecutionPipeline.exit_policy"):
        await ExitExecutionPipeline(lambda: None).evaluate(
            _decision(Action.CLOSE_LONG),
            "ensemble_trader",
            [],
        )
