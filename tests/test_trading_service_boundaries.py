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


@pytest.mark.asyncio
async def test_position_review_service_runs_sl_tp_and_executes_review_candidates():
    decision = _decision(Action.CLOSE_LONG)
    assessment = SimpleNamespace(warnings=[])
    claimed = []
    round_ids = set()
    executions = []

    class FakeSyncService:
        async def get_open_positions_context(self):
            return [{"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"}]

    class FakeOrchestrator:
        def __init__(self):
            self.okx_sync_service = FakeSyncService()
            self.stages = []

        def _set_loop_stage(self, stage):
            self.stages.append(stage)

        async def _enforce_sl_tp(self, feature_vectors):
            return [{
                "model_name": "ensemble_trader",
                "symbol": "ETH/USDT",
                "trigger": "take_profit",
                "quantity": 1.2,
                "exit_price": 2000.0,
                "status": "filled",
            }]

        async def _review_open_positions(
            self,
            open_positions,
            feature_vectors,
            *,
            results,
            round_decision_ids,
            position_entry_pause_reason,
            max_groups_override,
        ):
            assert max_groups_override == 3
            return [("BTC/USDT", "ensemble_trader", decision, assessment, 456)], set()

        async def _try_claim_analysis_symbol(self, symbol, owner):
            assert owner == "position"
            return True

        def _normalize_position_symbol(self, symbol):
            return symbol

        async def _execute_candidate(
            self,
            symbol,
            model_name,
            decision_arg,
            assessment_arg,
            decision_db_id,
            results,
            *,
            open_positions=None,
        ):
            executions.append((symbol, model_name, decision_arg.action.value, decision_db_id, open_positions))

    service = PositionReviewService(FakeOrchestrator())
    results = {"executions": []}
    open_positions, blocked = await service.review_open_positions(
        feature_vectors={"BTC/USDT": object()},
        results=results,
        round_decision_ids=round_ids,
        open_positions=[],
        position_entry_pause_reason=None,
        max_groups_override=3,
        claimed_analysis_symbols=claimed,
    )

    assert service.orchestrator.stages == ["enforce_sl_tp", "review_open_positions"]
    assert results["executions"][0]["action"] == "auto_close_take_profit"
    assert open_positions == [{"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"}]
    assert blocked == {("ensemble_trader", "BTC/USDT")}
    assert claimed == ["BTC/USDT"]
    assert round_ids == {456}
    assert executions == [("BTC/USDT", "ensemble_trader", "close_long", 456, open_positions)]
