import asyncio
from types import SimpleNamespace

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.analysis_services import MarketAnalysisService, PositionReviewService
from services.execution_service import ExecutionService
from services.trading_policies import EntryPolicy, ExitPolicy


def _decision(action: Action) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="测试决策",
        position_size_pct=0.05,
        suggested_leverage=3.0,
        raw_response={},
        feature_snapshot={"current_price": 100.0},
    )


@pytest.mark.asyncio
async def test_analysis_services_call_their_own_scope():
    calls = []

    class FakeOrchestrator:
        async def run_once(self, scope):
            calls.append(scope)
            return {"scope": scope}

    market = MarketAnalysisService(FakeOrchestrator())
    position = PositionReviewService(FakeOrchestrator())

    assert await market.run_once() == {"scope": "market"}
    assert await position.run_once() == {"scope": "position"}
    assert calls == ["market", "position"]


@pytest.mark.asyncio
async def test_execution_service_serializes_candidate_execution():
    lock = asyncio.Lock()
    calls = []

    class FakeOrchestrator:
        _execution_lock = lock

        async def _execute_candidate_locked(
            self,
            symbol,
            model_name,
            decision,
            assessment,
            decision_db_id,
            results,
            *,
            open_positions=None,
        ):
            assert self._execution_lock.locked()
            calls.append((symbol, model_name, decision.action.value, decision_db_id))
            return "executed"

    service = ExecutionService(FakeOrchestrator())
    result = await service.execute_candidate(
        "BTC/USDT",
        "ensemble_trader",
        _decision(Action.LONG),
        SimpleNamespace(warnings=[]),
        123,
        {},
        open_positions=[],
    )

    assert result == "executed"
    assert calls == [("BTC/USDT", "ensemble_trader", "long", 123)]


@pytest.mark.asyncio
async def test_entry_policy_blocks_stale_signal_before_okx_submit():
    class FakeOrchestrator:
        def _stale_decision_reason(self, decision):
            return "AI 信号已过有效期，等待下一轮新行情。"

        def _abnormal_wick_entry_guard_reason(self, decision):
            raise AssertionError("stale should stop later entry checks")

    result = await EntryPolicy(FakeOrchestrator()).evaluate(
        _decision(Action.LONG),
        "ensemble_trader",
        "paper",
        [],
    )

    assert result.passed is False
    assert result.blocker == "stale_decision"
    assert result.reason == "AI 信号已过有效期，等待下一轮新行情。"


@pytest.mark.asyncio
async def test_exit_policy_blocks_when_local_and_okx_position_are_missing():
    class FakeSyncService:
        def __init__(self):
            self.reconciled = False

        async def reconcile_positions(self, reason):
            self.reconciled = reason == "exit precheck"

        async def get_open_positions_context(self):
            return []

        async def has_matching_exchange_exit_position(self, model_name, decision):
            return False

    class FakeOrchestrator:
        def __init__(self):
            self.okx_sync_service = FakeSyncService()

        def _has_matching_exit_position(self, positions, model_name, decision):
            return False

        def _no_matching_exit_position_reason(self, decision):
            return "没有找到该币种同方向可平仓位，未向 OKX 重复提交平仓。"

    orchestrator = FakeOrchestrator()
    open_positions = [{"symbol": "BTC/USDT"}]
    result = await ExitPolicy(orchestrator).evaluate(
        _decision(Action.CLOSE_LONG),
        "ensemble_trader",
        open_positions,
    )

    assert orchestrator.okx_sync_service.reconciled is True
    assert open_positions == []
    assert result.passed is False
    assert result.blocker == "no_matching_exit_position"
    assert result.reason == "没有找到该币种同方向可平仓位，未向 OKX 重复提交平仓。"
