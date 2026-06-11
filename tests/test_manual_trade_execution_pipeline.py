from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from config.settings import ENSEMBLE_TRADER_NAME
from executor.base_executor import ExecutionResult, OrderStatus
from services.manual_trade_risk_assessment import ManualTradeRiskAssessmentPolicy
from services.trading_service import TradingService


def _decision(action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name=ENSEMBLE_TRADER_NAME,
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="manual test",
        position_size_pct=0.1,
        suggested_leverage=3.0,
        raw_response={},
        feature_snapshot={"current_price": 100.0},
    )


def _feature_vector() -> SimpleNamespace:
    return SimpleNamespace(
        symbol="BTC/USDT",
        recent_headlines=["headline"],
        returns_1=0.01,
        volume_ratio=1.2,
        adx_14=20.0,
    )


def _execution_result(status: OrderStatus = OrderStatus.FILLED) -> ExecutionResult:
    return ExecutionResult(
        order_id="order-1",
        exchange_order_id="okx-1",
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        quantity=2.0,
        price=100.0,
        status=status,
    )


def _manual_service(
    *,
    decision: DecisionOutput | None = None,
    execution_result: ExecutionResult | None = None,
    execution_block_reason: str | None = None,
) -> tuple[TradingService, list[tuple[str, Any]]]:
    service = TradingService.__new__(TradingService)
    calls: list[tuple[str, Any]] = []
    decision = decision if decision is not None else _decision()

    class FakeDataService:
        async def get_feature_vector(self, symbol: str) -> SimpleNamespace:
            calls.append(("feature", symbol))
            return _feature_vector()

    class FakeOkxSync:
        async def get_open_positions_context(self) -> list[dict[str, Any]]:
            calls.append(("open_positions", None))
            return [
                {"model_name": ENSEMBLE_TRADER_NAME, "symbol": "BTC/USDT"},
                {"model_name": "other", "symbol": "ETH/USDT"},
            ]

    class FakeExpertMemory:
        async def context(self, symbol: str) -> dict[str, Any]:
            calls.append(("memory", symbol))
            return {"memory_context": "ok"}

    class FakeMlSignal:
        def predict(self, fv: Any) -> dict[str, Any]:
            calls.append(("ml", fv.symbol))
            return {"ml_signal": "ok"}

    class FakeEnsemble:
        async def decide(self, fv: Any, context: dict[str, Any]) -> tuple[Any, list[Any]]:
            calls.append(("ensemble_context", context))
            return decision, []

    class FakeRiskEngine:
        def assess(self, decision_arg: DecisionOutput, **kwargs: Any) -> SimpleNamespace:
            calls.append(("risk", kwargs))
            return SimpleNamespace(approved=True, decision=None, warnings=[])

    async def local_ai_tools_context(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(("local_ai_tools", kwargs))
        return {"local_ai_tools": "ok"}

    async def strategy_mode_context(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(("strategy_mode", None))
        return {"strategy_mode": "ok"}

    async def log_decision(decision_arg: DecisionOutput, is_paper: bool) -> int:
        calls.append(("log_decision", decision_arg.action.value, is_paper))
        return 42

    async def execute_candidate(
        symbol: str,
        model_name: str,
        decision_arg: DecisionOutput,
        assessment: Any,
        decision_db_id: int | None,
        results: dict[str, Any],
        *,
        open_positions: list[dict[str, Any]] | None = None,
        refresh_exit_positions: bool = True,
    ) -> ExecutionResult | None:
        calls.append(
            (
                "execute_candidate",
                symbol,
                model_name,
                decision_arg.action.value,
                decision_db_id,
                len(open_positions or []),
                refresh_exit_positions,
            )
        )
        if execution_block_reason:
            results["decisions"].append({"reason": execution_block_reason})
            return None
        return execution_result or _execution_result()

    service.data_service = FakeDataService()
    service.okx_sync_service = FakeOkxSync()
    service.expert_memory_service = FakeExpertMemory()
    service.ml_signal_service = FakeMlSignal()
    service.ensemble = FakeEnsemble()
    service.risk_engine = FakeRiskEngine()
    service.manual_trade_risk_assessment = ManualTradeRiskAssessmentPolicy(service.risk_engine)
    service._local_ai_tools_context = local_ai_tools_context
    service._market_regime_context = lambda feature_vectors: {"market_regime": "ok"}
    service._strategy_mode_context = strategy_mode_context
    service._get_model_execution_mode = lambda model_name: "paper"
    service.get_account_balance = lambda model_name: _async_value(100.0)
    service._log_decision = log_decision
    service._execute_candidate = execute_candidate
    service._decision_count = 0
    return service, calls


@pytest.mark.asyncio
async def test_manual_trade_uses_unified_execution_pipeline() -> None:
    service, calls = _manual_service()

    result = await service.manual_trade("BTC/USDT")

    assert result["approved"] is True
    assert result["model"] == ENSEMBLE_TRADER_NAME
    assert result["execution"] == {
        "order_id": "order-1",
        "status": "filled",
        "quantity": 2.0,
        "price": 100.0,
    }
    assert service._decision_count == 1
    assert (
        "execute_candidate",
        "BTC/USDT",
        ENSEMBLE_TRADER_NAME,
        "long",
        42,
        2,
        True,
    ) in calls
    risk_call = next(call for call in calls if call[0] == "risk")
    assert risk_call[1]["current_positions"] == [
        {"model_name": ENSEMBLE_TRADER_NAME, "symbol": "BTC/USDT"}
    ]


@pytest.mark.asyncio
async def test_manual_trade_returns_pipeline_block_reason() -> None:
    service, calls = _manual_service(execution_block_reason="pipeline blocked")

    result = await service.manual_trade("BTC/USDT")

    assert result["approved"] is False
    assert result["rejection_reason"] == "pipeline blocked"
    assert any(call[0] == "execute_candidate" for call in calls)


@pytest.mark.asyncio
async def test_manual_trade_hold_does_not_enter_execution_pipeline() -> None:
    service, calls = _manual_service(decision=_decision(Action.HOLD))

    result = await service.manual_trade("BTC/USDT")

    assert result["approved"] is True
    assert result["reason"] == "AI 选择观望，未提交订单。"
    assert not any(call[0] == "log_decision" for call in calls)
    assert not any(call[0] == "execute_candidate" for call in calls)


async def _async_value(value: Any) -> Any:
    return value
