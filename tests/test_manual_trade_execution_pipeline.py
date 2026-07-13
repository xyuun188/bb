import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

import services.trading_service as trading_module
from ai_brain.base_model import Action, DecisionOutput
from config.settings import ENSEMBLE_TRADER_NAME
from executor.base_executor import ExecutionResult, OrderStatus
from services.execution_result_classifier import ExecutionResultClassifier
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

    def score_candidate(
        decision_arg: DecisionOutput,
        _strategy_context: dict[str, Any] | None,
    ) -> float:
        calls.append(("opportunity", decision_arg.action.value))
        return 0.2

    async def prepare_entry_risk(
        decision_arg: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]],
        decision_db_id: int | None = None,
    ) -> None:
        calls.append(
            (
                "dynamic_prepare",
                decision_arg.action.value,
                model_mode,
                len(open_positions),
                decision_db_id,
            )
        )

    service.data_service = FakeDataService()
    service.okx_sync_service = FakeOkxSync()
    service.expert_memory_service = FakeExpertMemory()
    service.ml_signal_service = FakeMlSignal()
    service.ensemble = FakeEnsemble()
    service.risk_engine = FakeRiskEngine()
    service.execution_result_classifier = ExecutionResultClassifier()
    service.manual_trade_risk_assessment = ManualTradeRiskAssessmentPolicy(service.risk_engine)
    service._local_ai_tools_context = local_ai_tools_context
    service._market_regime_context = lambda feature_vectors: {"market_regime": "ok"}
    service._strategy_mode_context = strategy_mode_context
    service._get_model_execution_mode = lambda model_name: "paper"
    service.get_account_balance = lambda model_name: _async_value(100.0)
    service._log_decision = log_decision
    service._execute_candidate = execute_candidate
    service._candidate_opportunity_score = score_candidate
    service._prepare_entry_for_hard_risk = prepare_entry_risk
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
    ordered_stages = [
        call[0]
        for call in calls
        if call[0] in {"opportunity", "dynamic_prepare", "risk", "execute_candidate"}
    ]
    assert ordered_stages == [
        "opportunity",
        "dynamic_prepare",
        "risk",
        "execute_candidate",
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


@pytest.mark.asyncio
async def test_manual_close_bypasses_ai_risk_and_persists_manual_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, state = _manual_close_service(monkeypatch, _execution_result_for_manual_close())

    result = await service._execute_manual_close_payload(
        _manual_close_payload(position_id=7, side="long"),
        reason="manual close test",
    )

    assert result["approved"] is True
    assert result["closed"] is True
    assert result["manual_close"] is True
    assert result["exclude_from_training"] is True
    assert state["risk_calls"] == 0
    assert state["log_decision_calls"] == 0
    assert state["execute_candidate_calls"] == 0
    assert state["executor_decisions"][0].action == Action.CLOSE_LONG
    assert state["executor_decisions"][0].raw_response["exclude_from_training"] is True
    assert state["orders"][0]["decision_id"] is None
    assert state["orders"][0]["exchange_order_id"] == "manual_close:okx-manual-1"
    assert state["orders"][0]["side"] == "sell"
    assert state["db_position"].is_open is False
    assert state["db_position"].realized_pnl == pytest.approx(9.7)
    assert state["db_position"].close_fill_pnl == pytest.approx(10.0)
    assert state["db_position"].entry_fee == pytest.approx(0.2)
    assert state["db_position"].close_fee == pytest.approx(0.1)
    assert state["db_position"].funding_fee == pytest.approx(0.0)
    assert state["db_position"].settlement_status == "settling"
    assert state["db_position"].settlement_source == "manual_close_execution"
    assert state["account_updates"] == [pytest.approx(9.7)]
    executed_event = next(
        row for row in state["strategy_events"] if row["event_status"] == "executed"
    )
    assert executed_event["event_type"] == "manual_close"
    assert executed_event["order_id"] == 1
    assert executed_event["position_id"] == 7
    assert executed_event["exclude_from_training"] is True


def _manual_close_payload(position_id: int, side: str) -> dict[str, Any]:
    return {
        "id": position_id,
        "model_name": ENSEMBLE_TRADER_NAME,
        "mode": "paper",
        "symbol": "BTC/USDT",
        "side": side,
        "quantity": 2.0,
        "entry_price": 100.0,
        "current_price": 99.0,
        "leverage": 3.0,
        "unrealized_pnl": -2.0,
    }


def _execution_result_for_manual_close(
    status: OrderStatus = OrderStatus.FILLED,
    quantity: float = 2.0,
) -> ExecutionResult:
    return ExecutionResult(
        order_id="local-manual-1",
        exchange_order_id="okx-manual-1",
        symbol="BTC/USDT",
        side="sell",
        order_type="market",
        quantity=quantity,
        price=105.0,
        status=status,
        fee=0.1,
        timestamp=datetime(2026, 6, 12, 1, 2, 3, tzinfo=UTC),
    )


class _ManualClosePosition:
    id = 7
    model_name = ENSEMBLE_TRADER_NAME
    execution_mode = "paper"
    symbol = "BTC/USDT"
    side = "long"
    quantity = 2.0
    entry_price = 100.0
    current_price = 99.0
    leverage = 3.0
    unrealized_pnl = -2.0
    realized_pnl = 0.0
    stop_loss_price = None
    take_profit_price = None
    is_open = True
    created_at = datetime(2026, 6, 12, 0, 0, 0, tzinfo=UTC)
    closed_at = None


def _manual_close_service(
    monkeypatch: pytest.MonkeyPatch,
    execution_result: ExecutionResult,
) -> tuple[TradingService, dict[str, Any]]:
    service = TradingService.__new__(TradingService)
    db_position = _ManualClosePosition()
    state: dict[str, Any] = {
        "orders": [],
        "executor_decisions": [],
        "account_updates": [],
        "strategy_events": [],
        "risk_calls": 0,
        "log_decision_calls": 0,
        "execute_candidate_calls": 0,
        "db_position": db_position,
    }

    class FakeExecutor:
        async def get_positions_strict(self, symbol: str) -> list[dict[str, Any]]:
            assert symbol == "BTC/USDT"
            return [{"side": "long", "quantity": 2.0}]

        async def place_order(
            self,
            decision_arg: DecisionOutput,
            account_id: str | None = None,
        ) -> ExecutionResult:
            assert account_id == ENSEMBLE_TRADER_NAME
            state["executor_decisions"].append(decision_arg)
            return execution_result

    class FakeRepo:
        def __init__(self, session: Any) -> None:
            self.session = session

        async def create_order(self, payload: dict[str, Any]) -> Any:
            state["orders"].append(payload)
            return SimpleNamespace(id=len(state["orders"]), **payload)

        async def open_position(self, payload: dict[str, Any]) -> Any:
            state.setdefault("split_positions", []).append(payload)
            return SimpleNamespace(id=99, **payload)

    class FakeSession:
        async def get(self, model: Any, key: int) -> Any:
            assert key in {7, 8}
            db_position.id = key
            return db_position

        async def flush(self) -> None:
            state["flushed"] = True

        async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
            return SimpleNamespace(scalar_one_or_none=lambda: None)

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    class FakeEntryFeeProvider:
        @staticmethod
        def proportional_fee(fee: float | None, close_qty: float, total_qty: float) -> float:
            return float(fee or 0.0) * close_qty / total_qty

        async def entry_fee_for_position(
            self,
            _session: Any,
            _position: Any,
            _close_qty: float,
        ) -> float:
            return 0.2

    class FakeAccountAccounting:
        async def persist_account_update(
            self,
            _model_name: str,
            _decision_model_name: str,
            result: ExecutionResult,
        ) -> None:
            state["account_updates"].append(result.pnl)

    class FakeRiskEngine:
        def assess(self, *_args: Any, **_kwargs: Any) -> Any:
            state["risk_calls"] += 1
            raise AssertionError("manual close must bypass risk")

    async def log_decision(*_args: Any, **_kwargs: Any) -> int:
        state["log_decision_calls"] += 1
        raise AssertionError("manual close must not log an AI decision")

    async def execute_candidate(*_args: Any, **_kwargs: Any) -> ExecutionResult:
        state["execute_candidate_calls"] += 1
        raise AssertionError("manual close must bypass AI execution pipeline")

    async def record_strategy_event(**kwargs: Any) -> None:
        state["strategy_events"].append(kwargs)

    monkeypatch.setattr(trading_module, "get_session_ctx", fake_session_ctx)
    monkeypatch.setattr(trading_module, "TradeRepository", FakeRepo)
    service.get_okx_executor_for_mode = lambda _mode: _async_value(FakeExecutor())
    service._execution_lock = asyncio.Lock()
    service.entry_fee_provider = FakeEntryFeeProvider()
    service.account_accounting_service = FakeAccountAccounting()
    service.position_profit_peaks = SimpleNamespace(remove=lambda *_args: None)
    service.execution_result_classifier = ExecutionResultClassifier()
    service.risk_engine = FakeRiskEngine()
    service._log_decision = log_decision
    service._execute_candidate = execute_candidate
    service._record_strategy_learning_event = record_strategy_event  # type: ignore[method-assign]
    service._trade_count = 0
    return service, state


@pytest.mark.asyncio
async def test_manual_close_rejected_execution_does_not_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected = _execution_result_for_manual_close(status=OrderStatus.REJECTED, quantity=0.0)
    service, state = _manual_close_service(monkeypatch, rejected)

    result = await service._execute_manual_close_payload(
        _manual_close_payload(position_id=8, side="short"),
        reason="manual close test",
    )

    assert result["approved"] is False
    assert result["position_id"] == 8
    assert result["execution"]["status"] == "rejected"
    assert state["orders"] == []
    assert state["db_position"].is_open is True
    assert state["risk_calls"] == 0
    assert state["log_decision_calls"] == 0
    assert state["execute_candidate_calls"] == 0
