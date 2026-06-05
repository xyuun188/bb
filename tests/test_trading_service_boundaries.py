import asyncio
from types import SimpleNamespace

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.analysis_services import MarketAnalysisService, PositionReviewService
from services.execution_service import ExecutionService
from services.sync_service import OkxSyncService
from services.trading_policies import EntryPolicy, ExitPolicy, PolicyGateResult


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
    stages = []

    class FakeExecutor:
        async def place_order(self, decision, account_id=None, override_balance=None):
            assert lock.locked()
            calls.append(("place_order", account_id, decision.action.value, override_balance))
            return ExecutionResult(
                order_id="order-1",
                exchange_order_id="exchange-1",
                symbol=decision.symbol,
                side=decision.action.value,
                order_type="market",
                quantity=2.0,
                price=100.0,
                status=OrderStatus.FILLED,
                raw_response={},
            )

    class FakePolicy:
        async def evaluate(self, *args, **kwargs):
            return PolicyGateResult.allow({"intent": "entry"})

    class FakeSkills:
        def execution_skills(self, **kwargs):
            return []

        def attach(self, *args, **kwargs):
            raise AssertionError("no skills should be attached in this test")

        def block_reason(self, *args, **kwargs):
            return None

    class FakeBreaker:
        def record_trade(self, amount):
            calls.append(("record_trade", amount))

    class FakeRiskEngine:
        circuit_breaker = FakeBreaker()

    class FakeOrchestrator:
        _execution_lock = lock
        entry_policy = FakePolicy()
        exit_policy = FakePolicy()
        agent_skills = FakeSkills()
        risk_engine = FakeRiskEngine()
        _trade_count = 0

        def _get_model_execution_mode(self, model_name):
            return "paper"

        async def _record_and_persist_decision_stage(self, decision_db_id, decision, stage, status, reason, data=None):
            assert self._execution_lock.locked()
            stages.append((stage, status, reason))

        async def _duplicate_decision_order_reason(self, decision_db_id, decision):
            return None

        async def _mark_decision_raw_response(self, decision_db_id, raw_response):
            calls.append(("raw", decision_db_id))

        async def _get_okx_executor_for_mode(self, mode):
            return FakeExecutor()

        async def _allocated_order_balance(self, model_mode, decision):
            return 123.0

        def _attach_execution_leverage_summary(self, decision, execution_result, ai_requested_leverage):
            calls.append(("leverage", ai_requested_leverage))

        def _is_untradable_exchange_error(self, text):
            return False

        def _is_transient_entry_exchange_error(self, text):
            return False

        async def _log_trade(self, execution_result, model_name, decision, decision_db_id):
            calls.append(("log_trade", execution_result.order_id))

        def _is_exchange_confirmed_execution(self, execution_result):
            return bool(execution_result and execution_result.status == OrderStatus.FILLED)

        def _is_exit_progress_execution(self, execution_result):
            return False

        def _execution_reason_from_result(self, execution_result):
            return execution_result.status.value if execution_result else "missing"

        async def _persist_position_from_execution(self, model_name, decision, execution_result, model_mode):
            calls.append(("persist_position", model_name, model_mode))

        def _apply_execution_to_open_positions(self, open_positions, model_name, decision, execution_result):
            calls.append(("apply_open_positions", len(open_positions)))

        async def _mark_decision_executed(self, decision_db_id, price):
            calls.append(("executed", decision_db_id, price))

        def _clear_market_no_opportunity_symbol(self, symbol):
            calls.append(("clear_symbol", symbol))

        def _position_review_alert_context(self, decision):
            return None

    service = ExecutionService(FakeOrchestrator())
    results = {"warnings": [], "decisions": [], "executions": []}
    result = await service.execute_candidate(
        "BTC/USDT",
        "ensemble_trader",
        _decision(Action.LONG),
        SimpleNamespace(warnings=[]),
        123,
        results,
        open_positions=[],
    )

    assert result.order_id == "order-1"
    assert ("place_order", "ensemble_trader", "long", 123.0) in calls
    assert ("persist_position", "ensemble_trader", "paper") in calls
    assert ("executed", 123, 100.0) in calls
    assert results["executions"][0]["order_id"] == "order-1"
    assert results["decisions"][0]["executed"] is True
    assert [stage for stage, _status, _reason in stages] == [
        "strategy_arbitration",
        "risk_check",
        "risk_check",
        "exchange_submit",
        "exchange_confirm",
        "local_sync",
    ]


@pytest.mark.asyncio
async def test_entry_policy_blocks_stale_signal_before_okx_submit():
    stale_reason = "AI 信号已过有效期，等待下一轮新行情。"

    class FakeOrchestrator:
        def _stale_decision_reason(self, decision):
            return stale_reason

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
    assert result.reason == stale_reason


@pytest.mark.asyncio
async def test_exit_policy_blocks_when_local_and_okx_position_are_missing():
    missing_reason = "没有找到 BTC/USDT 对应的可平多单仓位，未向 OKX 提交平仓单。"

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

        def _normalize_position_symbol(self, symbol):
            return symbol

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
    assert result.reason == missing_reason


@pytest.mark.asyncio
async def test_sync_service_reconcile_positions_owns_lock_boundary():
    lock = asyncio.Lock()
    calls = []

    class FakeOrchestrator:
        _exchange_reconcile_lock = lock
        _last_round_error = None

    service = OkxSyncService(FakeOrchestrator())

    async def fake_reconcile():
        assert lock.locked()
        calls.append("reconciled")
        return [{"symbol": "BTC/USDT", "side": "long"}]

    service.reconcile_exchange_positions = fake_reconcile
    result = await service.reconcile_positions("unit test")

    assert result == [{"symbol": "BTC/USDT", "side": "long"}]
    assert calls == ["reconciled"]


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
