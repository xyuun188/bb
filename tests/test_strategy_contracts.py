from datetime import UTC, datetime

import pandas as pd
import pytest

from ai_brain.base_model import Action, DecisionOutput
from backtest.engine import BacktestEngine
from core.strategy_contracts import (
    ExecutionMode,
    PositionSide,
    StrategyAction,
    StrategyContext,
    StrategyContractError,
    StrategyDecision,
)
from data_feed.feature_vector import FeatureVector
from services.paper_live_consistency import (
    assert_strategy_context_decision_parity,
)
from services.strategy_contract_adapter import (
    ai_output_from_decision,
    context_from_feature_vector,
    decision_from_ai_output,
)

DECISION_TIME = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
STRATEGY = {
    "strategy_id": "contract-test",
    "strategy_version": "1.0.0",
}
PARAMETERS = {
    "parameter_set_id": "params_contract_test",
    "parameter_version": "1.0.0",
    "values": {"entry_threshold": 0.5},
}


def _context(mode: ExecutionMode, **assumptions: object) -> StrategyContext:
    return StrategyContext(
        symbol="BTC/USDT",
        market_snapshot={"close": 100.0, "bid": 99.9, "ask": 100.1},
        feature_snapshot={"rsi_14": 42.0, "close_sequence": [98.0, 100.0]},
        position_snapshot={"side": "none", "exposure": 0.0},
        account_constraints={"max_exposure": 0.3},
        decision_time=DECISION_TIME,
        execution_mode=mode,
        strategy_id=STRATEGY["strategy_id"],
        strategy_version=STRATEGY["strategy_version"],
        parameter_set_id=PARAMETERS["parameter_set_id"],
        parameter_version=PARAMETERS["parameter_version"],
        parameter_values=PARAMETERS["values"],
        execution_assumptions=assumptions,
    )


def _decision(context: StrategyContext, **overrides: object) -> StrategyDecision:
    values = {
        "symbol": context.symbol,
        "action": StrategyAction.ENTER,
        "side": PositionSide.LONG,
        "target_exposure": 0.2,
        "confidence": 0.8,
        "reason_codes": ("SIGNAL_TEST",),
        "protection_hints": {"stop_loss_pct": 0.02},
        "strategy_version": context.strategy_version,
        "parameter_version": context.parameter_version,
        "decision_time": context.decision_time,
        "strategy_input_sha256": context.strategy_input_sha256,
        "source": "test",
    }
    values.update(overrides)
    return StrategyDecision(**values)


def test_execution_mode_and_cost_assumptions_do_not_change_pure_input() -> None:
    paper = _context(ExecutionMode.PAPER, commission_rate=0.001)
    live = _context(ExecutionMode.LIVE, commission_rate=0.002, latency_ms=80)
    assert paper.strategy_input_sha256 == live.strategy_input_sha256

    paper_decision = _decision(paper)
    live_decision = _decision(live)
    report = assert_strategy_context_decision_parity(
        paper, paper_decision, live, live_decision
    )
    assert report["ok"] is True
    assert report["execution_mode_difference_allowed"] is True
    assert "commission_rate" in report["execution_assumption_differences"]


def test_context_is_deeply_immutable_and_requires_aware_time() -> None:
    context = _context(ExecutionMode.PAPER)
    with pytest.raises(TypeError):
        context.feature_snapshot["rsi_14"] = 50.0  # type: ignore[index]
    with pytest.raises(StrategyContractError, match="timezone-aware"):
        _context(ExecutionMode.PAPER).decision_time.replace(tzinfo=None)
        StrategyContext(
            symbol="BTC/USDT",
            market_snapshot={},
            feature_snapshot={},
            position_snapshot={},
            account_constraints={},
            decision_time=datetime(2026, 1, 1),
            execution_mode="paper",
            strategy_id="s",
            strategy_version="1",
            parameter_set_id="p",
            parameter_version="1",
            parameter_values={},
        )


def test_invalid_decision_contract_fails_closed() -> None:
    context = _context(ExecutionMode.PAPER)
    with pytest.raises(StrategyContractError, match="enter requires"):
        _decision(context, target_exposure=0.0)
    with pytest.raises(StrategyContractError, match="between 0 and 1"):
        _decision(context, confidence=1.1)


def test_decision_output_round_trip_preserves_execution_semantics() -> None:
    context = _context(ExecutionMode.PAPER)
    legacy = DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="test",
        position_size_pct=0.2,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        timestamp=DECISION_TIME,
    )
    standard = decision_from_ai_output(legacy, context)
    assert standard.action == StrategyAction.ENTER
    assert standard.side == PositionSide.LONG
    restored = ai_output_from_decision(standard, context)
    assert restored.action == Action.LONG
    assert restored.position_size_pct == pytest.approx(0.2)
    assert restored.raw_response["strategy_contract"]["decision_sha256"] == standard.decision_sha256


def test_realtime_adapter_rejects_naive_feature_time() -> None:
    features = FeatureVector(symbol="BTC/USDT", timestamp=datetime(2026, 8, 13, 12, 0))
    with pytest.raises(StrategyContractError, match="timezone-aware"):
        context_from_feature_vector(
            features,
            execution_mode=ExecutionMode.PAPER,
            strategy=STRATEGY,
            parameters=PARAMETERS,
        )


def test_backtest_engine_can_run_standard_contract_adapter() -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100 + i for i in range(8)],
            "high": [101 + i for i in range(8)],
            "low": [99 + i for i in range(8)],
            "close": [100 + i for i in range(8)],
            "volume": [1000.0] * 8,
        },
        index=index,
    )

    def provider(context: StrategyContext) -> StrategyDecision:
        return StrategyDecision(
            symbol=context.symbol,
            action=StrategyAction.HOLD,
            side=PositionSide.NONE,
            target_exposure=0.0,
            confidence=0.5,
            reason_codes=("SIGNAL_HOLD",),
            protection_hints={},
            strategy_version=context.strategy_version,
            parameter_version=context.parameter_version,
            decision_time=context.decision_time,
            strategy_input_sha256=context.strategy_input_sha256,
            source="test_provider",
        )

    engine = BacktestEngine(initial_cash=1000.0)
    engine.load_data(data)
    engine.add_contract_strategy(
        provider,
        symbol="BTC/USDT",
        strategy=STRATEGY,
        parameters=PARAMETERS,
    )
    result = engine.run()
    assert result["strategy_contract"]["decision_count"] == 8
