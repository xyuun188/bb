import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

import services.trading_service as trading_service
import services.sync_service as sync_module
from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.account_accounting_service import AccountAccountingService
from services.analysis_services import MarketAnalysisService, PositionReviewService
from services.decision_final_state_ensurer import DecisionFinalStateEnsurer
from services.entry_existing_winner import EntryExistingWinnerContextPolicy
from services.entry_fee_provider import EntryFeeProvider
from services.entry_market_data_quality import EntryMarketDataQualityPolicy, MarketValueReader
from services.entry_opportunity_gate import EntryOpportunityGatePolicy
from services.entry_opportunity_score import EntryOpportunityScorePolicy
from services.entry_payoff_quality import EntryLowPayoffQualityPolicy
from services.entry_profit_risk_sizing import EntryProfitRiskSizingPolicy
from services.entry_stop_loss_budget import EntryStopLossBudgetPolicy
from services.entry_stress_stop import EntryStressStopPolicy
from services.entry_symbol_blocklist import EntrySymbolBlocklistPolicy
from services.exchange_backed_position_provider import ExchangeBackedPositionProvider
from services.exchange_close_fill_finder import ExchangeCloseFillFinder
from services.exchange_position_state import (
    ExchangePositionStatePolicy,
    ExchangeProtectionMapProvider,
)
from services.execution_allocation_service import ExecutionAllocationService
from services.execution_pipelines import EntryExecutionPipeline, ExitExecutionPipeline
from services.execution_service import ExecutionService
from services.expert_memory_service import ExpertMemoryService
from services.memory_position_store import MemoryPositionStore
from services.ml_signal_service import MLSignalService
from services.new_pair_loss_pause import NewPairLossPausePolicy
from services.position_margin import PositionMarginCalculator
from services.position_profit_peaks import PositionProfitPeakTracker
from services.position_protection_fallback import PositionProtectionFallbackPolicy
from services.position_snapshot_syncer import PositionSnapshotSyncer
from services.position_time import PositionTimeParser
from services.shadow_backtest_service import ShadowBacktestService
from services.stale_entry_candidate_expirer import StaleEntryCandidateExpirer
from services.sync_service import OPEN_ORDER_SNAPSHOT_UNKNOWN_KIND, OkxSyncService
from services.trading_policies import EntryPolicy, ExitPolicy, PolicyGateResult
from services.trading_service import TradingService, _AnalysisRuntimeState


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


def test_trading_service_detects_policy_skipped_execution_result() -> None:
    result = ExecutionResult(
        order_id="rejected",
        symbol="BTC/USDT",
        side="long",
        order_type="market",
        quantity=0.0,
        price=0.0,
        status=OrderStatus.REJECTED,
        raw_response={
            "execution_skipped": True,
            "skip_kind": "entry_evidence_shadow_only",
        },
    )

    assert TradingService._is_policy_skipped_execution_result(result) is True


def test_parallel_market_position_runtime_state_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    service = TradingService.__new__(TradingService)
    service._running = True
    service._start_time = datetime.now(UTC) - timedelta(minutes=5)
    service._current_stage = "idle"
    service._last_round_started_at = None
    service._last_round_finished_at = None
    service._last_round_error = None
    service._last_market_round_started_at = None
    service._last_market_round_finished_at = None
    service._last_position_round_started_at = None
    service._last_position_round_finished_at = None
    service._analysis_runtime = {
        "market": _AnalysisRuntimeState(),
        "position": _AnalysisRuntimeState(),
        "full": _AnalysisRuntimeState(),
    }

    from config.settings import settings

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings.__class__, "data_dir", property(lambda _self: data_dir))

    market_start = datetime.now(UTC) - timedelta(seconds=20)
    position_start = datetime.now(UTC) - timedelta(seconds=5)
    service._start_runtime_round("market", market_start)
    service._set_loop_stage("fetch_features", scope="market")
    service._start_runtime_round("position", position_start)
    service._set_loop_stage("review_open_positions", scope="position")
    service._finish_runtime_round("position", datetime.now(UTC), ok=True)
    service._write_runtime_heartbeat()

    payload = json.loads((data_dir / "trading_runtime_status.json").read_text(encoding="utf-8"))

    assert payload["market_round_active"] is True
    assert payload["market_current_stage"] == "fetch_features"
    assert payload["position_round_active"] is False
    assert payload["position_current_stage"] == "idle"
    assert payload["round_active"] is True
    assert payload["current_stage"] == "fetch_features"
    assert TradingService._is_policy_skipped_execution_result(None) is False


async def _async_value(value: Any) -> Any:
    return value


def test_ai_entry_candidate_evidence_exposes_profile_recency_to_model() -> None:
    service = TradingService.__new__(TradingService)
    profile = {
        "count": 3,
        "pnl": 12.5,
        "today_pnl": 2.0,
        "wins": 2,
        "losses": 1,
        "profit_factor": 2.4,
        "largest_loss": -1.2,
        "first_closed_at": "2026-06-08T10:00:00+00:00",
        "last_closed_at": "2026-06-09T10:00:00+00:00",
        "last_loss_at": "2026-06-08T11:00:00+00:00",
        "last_loss_age_hours": 23.5,
        "lookback_days": 14,
        "cooldown": False,
    }

    class FakeEntryPolicy:
        def score_candidate(
            self,
            decision: DecisionOutput,
            _strategy: dict[str, Any] | None,
        ) -> float:
            decision.raw_response["opportunity_score"] = {
                "expected_net_return_pct": 0.8,
                "tail_risk_score": 0.2,
                "server_profit_loss_probability": 0.3,
                "profit_quality_ratio": 1.1,
                "min_score_required": 0.7,
                "symbol_profile": profile,
                "symbol_side_profile": profile,
            }
            return 0.9 if decision.action == Action.LONG else 0.8

    service.entry_policy = FakeEntryPolicy()
    fv = SimpleNamespace(
        symbol="BTC/USDT",
        volume_24h=10_000,
        volume_ratio=1.0,
        adx_14=20.0,
        returns_1=0.001,
        returns_5=0.002,
        returns_20=0.003,
        volatility_20=0.02,
        change_24h_pct=1.0,
        bb_pct=0.5,
        price_vs_sma20=0.01,
        price_vs_sma50=0.02,
        current_price=100.0,
    )
    fv.to_dict = lambda: {"current_price": 100.0}

    evidence = service._ai_entry_candidate_evidence(fv, {}, {}, {}, {})

    side_profile = evidence["long"]["symbol_side_profile"]
    assert side_profile["last_closed_at"] == "2026-06-09T10:00:00+00:00"
    assert side_profile["first_closed_at"] == "2026-06-08T10:00:00+00:00"
    assert side_profile["last_loss_at"] == "2026-06-08T11:00:00+00:00"
    assert side_profile["last_loss_age_hours"] == 23.5
    assert side_profile["lookback_days"] == 14


@pytest.mark.asyncio
async def test_memory_context_merges_vector_memory_soft_feedback(monkeypatch) -> None:
    service = TradingService.__new__(TradingService)

    class FakeExpertMemoryService:
        async def context(self, symbol):
            return {
                "memory_feedback": {
                    "enabled": True,
                    "preferred_side_by_memory": "long",
                }
            }

    class FakeVectorMemoryService:
        async def search(self, query, *, top_k=8, symbol="", kind="", min_score=None):
            assert "BTC/USDT" in query
            assert symbol == "BTC/USDT"
            return {
                "enabled": True,
                "status": "ok",
                "hits": [
                    {
                        "score": 0.71,
                        "action": "long",
                        "outcome": "loss",
                        "pnl_pct": -0.6,
                    }
                ],
            }

    service.expert_memory_service = FakeExpertMemoryService()
    monkeypatch.setattr("services.trading_service.settings.vector_memory_enabled", True)
    monkeypatch.setattr(
        "services.trading_service.get_vector_memory_service",
        lambda: FakeVectorMemoryService(),
    )

    context = await service._memory_context_with_vector_feedback("BTC/USDT")

    vector = context["memory_feedback"]["vector_memory"]
    assert vector["status"] == "ok"
    assert vector["matched_count"] == 1
    assert vector["is_hard_gate"] is False
    assert "硬拦截" in vector["policy"]
    assert context["vector_memory_feedback"] == vector


def _noop_reconcile_close_boundaries() -> dict[str, Any]:
    async def find_exchange_close_fill(_pos):
        return {}

    async def fresh_feature_vector(_symbol):
        return None

    def market_value(source, key):
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    async def entry_fee(_session, _pos, _close_qty):
        return 0.0

    async def log_close_decision(**_kwargs):
        return None

    async def record_reflection(*_args, **_kwargs):
        return None

    return {
        "exchange_close_fill_finder": find_exchange_close_fill,
        "fresh_feature_vector_provider": fresh_feature_vector,
        "market_value_reader": market_value,
        "entry_fee_provider": entry_fee,
        "exchange_sync_close_decision_logger": log_close_decision,
        "trade_reflection_recorder": record_reflection,
        "position_margin_calculator": lambda _notional, _leverage: 0.0,
        "memory_position_remover": lambda _model_name, _symbol, _side: None,
    }


@pytest.mark.asyncio
async def test_analysis_services_call_their_own_scope():
    calls: list[tuple[Any, ...]] = []

    async def run_once(scope):
        calls.append(scope)
        return {"scope": scope}

    market = MarketAnalysisService(run_once_provider=run_once)
    position = PositionReviewService(run_once_provider=run_once)

    assert await market.run_once() == {"scope": "market"}
    assert await position.run_once() == {"scope": "position"}
    assert calls == ["market", "position"]


@pytest.mark.asyncio
async def test_analysis_service_loop_uses_injected_lifecycle_boundary():
    calls: list[str] = []
    running = True

    async def run_once(scope):
        nonlocal running
        calls.append(scope)
        running = False
        return {"scope": scope}

    service = MarketAnalysisService(
        run_once_provider=run_once,
        is_running_provider=lambda: running,
    )
    service.initial_delay_seconds = 0.0

    await service.loop(0.0)

    assert calls == ["market"]


@pytest.mark.asyncio
async def test_analysis_service_loop_sleeps_interval_after_round_finishes(monkeypatch):
    calls: list[str] = []
    running = True
    sleeps: list[float] = []
    now = 100.0

    async def fake_sleep(seconds: float) -> None:
        nonlocal running
        sleeps.append(seconds)
        if len(sleeps) > 1:
            running = False

    async def run_once(scope):
        nonlocal now
        calls.append(scope)
        now += 7.0
        return {"scope": scope}

    service = MarketAnalysisService(
        run_once_provider=run_once,
        is_running_provider=lambda: running,
    )
    service.initial_delay_seconds = 0.0
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await service.loop(lambda: 10.0)

    assert calls == ["market"]
    assert sleeps == [0.0, 10.0]


@pytest.mark.asyncio
async def test_analysis_service_loop_times_out_stuck_round(monkeypatch):
    calls: list[str] = []
    running = True
    sleeps: list[float] = []
    original_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        nonlocal running
        sleeps.append(seconds)
        if len(sleeps) > 1:
            running = False

    async def run_once(scope):
        calls.append(scope)
        await original_sleep(60)
        return {"scope": scope}

    service = MarketAnalysisService(
        run_once_provider=run_once,
        is_running_provider=lambda: running,
        time_budget_provider=lambda: 0.05,
    )
    service.initial_delay_seconds = 0.0
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await service.loop(lambda: 30.0)

    assert calls == ["market"]
    assert sleeps == [0.0, 30.0]


@pytest.mark.asyncio
async def test_position_review_loop_keeps_stage_timeout_separate_from_round_watchdog(
    monkeypatch,
):
    calls: list[str] = []
    running = True
    sleeps: list[float] = []
    original_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        nonlocal running
        sleeps.append(seconds)
        if len(sleeps) > 1:
            running = False

    async def run_once(scope):
        calls.append(scope)
        await original_sleep(60)
        return {"scope": scope}

    service = PositionReviewService(
        run_once_provider=run_once,
        is_running_provider=lambda: running,
        timeout_provider=lambda: 0.05,
        round_watchdog_provider=lambda: 0.2,
    )
    service.initial_delay_seconds = 0.0
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await service.loop(lambda: 30.0)

    assert calls == ["position"]
    assert sleeps == [0.0, 30.0]


@pytest.mark.asyncio
async def test_position_review_loop_without_round_watchdog_does_not_use_stage_timeout(
    monkeypatch,
):
    calls: list[str] = []
    running = True
    sleeps: list[float] = []
    original_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        nonlocal running
        sleeps.append(seconds)
        if len(sleeps) > 1:
            running = False

    async def run_once(scope):
        calls.append(scope)
        await original_sleep(0.02)
        return {"scope": scope}

    service = PositionReviewService(
        run_once_provider=run_once,
        is_running_provider=lambda: running,
        timeout_provider=lambda: 0.001,
    )
    service.initial_delay_seconds = 0.0
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await service.loop(lambda: 30.0)

    assert calls == ["position"]
    assert sleeps == [0.0, 30.0]


@pytest.mark.asyncio
async def test_analysis_service_loop_fails_fast_without_lifecycle_boundary():
    async def run_once(_scope):
        return None

    service = MarketAnalysisService(run_once_provider=run_once)
    service.initial_delay_seconds = 0.0

    with pytest.raises(RuntimeError, match="is_running_provider"):
        await service.loop(0.0)


def test_analysis_services_do_not_keep_legacy_orchestrator_reference():
    market = MarketAnalysisService(run_once_provider=lambda _scope: None)
    position = PositionReviewService(run_once_provider=lambda _scope: None)

    assert not hasattr(market, "orchestrator")
    assert not hasattr(position, "orchestrator")


def test_position_protection_fallback_is_not_a_trading_service_private_rule():
    policy = PositionProtectionFallbackPolicy()

    assert callable(policy.protection_from_decision)
    assert not hasattr(TradingService, "fallback_position_protection_from_decision")
    assert not hasattr(TradingService, "_fallback_position_protection_from_decision")


def test_position_profit_peak_state_is_not_a_trading_service_private_rule(tmp_path):
    tracker = PositionProfitPeakTracker(path=tmp_path / "peaks.json")

    assert callable(tracker.update)
    assert not hasattr(TradingService, "_position_peak_key")
    assert not hasattr(TradingService, "_load_position_profit_peaks")
    assert not hasattr(TradingService, "_save_position_profit_peaks")
    assert not hasattr(TradingService, "_update_position_profit_peak")
    assert not hasattr(TradingService, "_prune_position_profit_peaks")


def test_position_snapshot_syncer_is_not_a_trading_service_private_rule():
    syncer = PositionSnapshotSyncer()

    assert callable(syncer.sync)
    assert not hasattr(TradingService, "sync_local_open_position_snapshot")
    assert not hasattr(TradingService, "_sync_local_open_position_snapshot")


def test_position_review_decision_normalizer_is_not_a_trading_service_private_rule():
    assert not hasattr(TradingService, "_normalize_review_decision_for_positions")


def test_exchange_position_state_is_not_a_trading_service_private_rule():
    state = ExchangePositionStatePolicy()
    provider = ExchangeProtectionMapProvider(
        symbol_normalizer=lambda symbol: str(symbol or ""),
        position_open_checker=state.is_open,
    )

    assert callable(state.is_open)
    assert callable(provider.fetch)
    assert not hasattr(TradingService, "exchange_position_is_open")
    assert not hasattr(TradingService, "_exchange_position_is_open")
    assert not hasattr(TradingService, "fetch_exchange_protection_map")
    assert not hasattr(TradingService, "_fetch_exchange_protection_map")


def test_exchange_close_fill_finder_is_not_a_trading_service_private_rule():
    finder = ExchangeCloseFillFinder(paper_okx_provider=lambda: None)

    assert callable(finder.find)
    assert not hasattr(TradingService, "find_exchange_close_fill")
    assert not hasattr(TradingService, "_find_exchange_close_fill")
    assert not hasattr(TradingService, "_order_fee_cost")


def test_entry_fee_provider_is_not_a_trading_service_private_rule():
    provider = EntryFeeProvider()

    assert callable(provider.entry_fee_for_position)
    assert callable(provider.proportional_fee)
    assert not hasattr(TradingService, "entry_fee_for_position")
    assert not hasattr(TradingService, "_entry_fee_for_position")
    assert not hasattr(TradingService, "_proportional_fee")


def test_position_time_parser_is_not_a_trading_service_private_rule():
    parser = PositionTimeParser()

    assert callable(parser.datetime_from_ms)
    assert callable(parser.position_age_minutes)
    assert not hasattr(TradingService, "datetime_from_ms")
    assert not hasattr(TradingService, "_datetime_from_ms")
    assert not hasattr(TradingService, "position_age_minutes")
    assert not hasattr(TradingService, "_position_age_minutes")


def test_memory_position_store_is_not_a_trading_service_private_rule():
    store = MemoryPositionStore(
        paper_executor_provider=lambda: None,
        symbol_normalizer=lambda symbol: str(symbol or ""),
    )

    assert callable(store.remove_open_position)
    assert not hasattr(TradingService, "remove_memory_position")
    assert not hasattr(TradingService, "_remove_memory_position")


def test_expert_memory_service_is_not_a_trading_service_private_rule():
    service = ExpertMemoryService(memory_enabled_provider=lambda: False)

    assert callable(service.context)
    assert callable(service.record_trade_reflection_in_session)
    assert callable(service.backfill_trade_reflections)
    assert not hasattr(TradingService, "_expert_memory_context")
    assert not hasattr(TradingService, "_realized_expert_weight_adjustments")
    assert not hasattr(TradingService, "_record_trade_reflection_in_session")
    assert not hasattr(TradingService, "_backfill_trade_reflections")
    assert not hasattr(TradingService, "_build_expert_lessons")


def test_account_accounting_service_is_not_a_trading_service_private_rule():
    async def balance_snapshot(_mode):
        return {"free": 123.0}

    async def allocation_state(_mode):
        return {"allocated": 456.0}

    service = AccountAccountingService(
        balance_snapshot_provider=balance_snapshot,
        allocation_state_provider=allocation_state,
        model_execution_mode_provider=lambda _model_name: "paper",
    )

    assert callable(service.allocated_order_balance)
    assert callable(service.persist_account_update)
    assert callable(service.record_unrealized_pnl)
    assert not hasattr(TradingService, "_allocated_order_balance")
    assert not hasattr(TradingService, "_get_account_balance")
    assert not hasattr(TradingService, "_get_account_equity_for_risk")
    assert not hasattr(TradingService, "_persist_account_update")
    assert not hasattr(TradingService, "_persist_paper_balance_delta")
    assert not hasattr(TradingService, "_persist_paper_execution_balance")
    assert not hasattr(TradingService, "_okx_tradeable_balance_from_snapshot")
    assert not hasattr(TradingService, "_okx_allocatable_balance_from_snapshot")


def test_execution_allocation_service_is_not_a_trading_service_private_rule():
    service = ExecutionAllocationService(
        balance_snapshot_provider=lambda _mode: None,
        active_executor_provider=lambda _mode: None,
        exchange_position_open_checker=lambda _payload: True,
        symbol_normalizer=lambda symbol: str(symbol or ""),
    )

    assert callable(service.calculate)
    assert not hasattr(TradingService, "_execution_allocation_state")


def test_exchange_backed_position_provider_is_not_a_trading_service_private_rule():
    provider = ExchangeBackedPositionProvider()

    assert callable(provider.ids)
    assert not hasattr(TradingService, "exchange_backed_position_ids")
    assert not hasattr(TradingService, "_exchange_backed_position_ids")


def test_position_margin_calculator_is_not_a_trading_service_private_rule():
    calculator = PositionMarginCalculator()

    assert callable(calculator.margin)
    assert not hasattr(TradingService, "position_margin")
    assert not hasattr(TradingService, "_position_margin")


def test_entry_market_data_quality_is_not_a_trading_service_private_rule():
    policy = EntryMarketDataQualityPolicy()
    reader = MarketValueReader()

    assert callable(policy.reason)
    assert callable(reader.read)
    assert not hasattr(TradingService, "market_value")
    assert not hasattr(TradingService, "_market_value")
    assert not hasattr(TradingService, "entry_market_data_quality_reason")
    assert not hasattr(TradingService, "_entry_market_data_quality_reason")


def test_new_pair_loss_pause_is_not_a_trading_service_private_rule():
    service = TradingService.__new__(TradingService)

    assert callable(NewPairLossPausePolicy(lambda _mode: None).cooldown_loss_pause_reason)
    assert not hasattr(TradingService, "_cooldown_loss_pause_reason")
    assert not hasattr(TradingService, "_recent_loss_streak_pause_reason")
    assert not hasattr(service, "_cooldown_loss_pause_reason")


@pytest.mark.asyncio
async def test_new_pair_loss_cooldown_is_advisory_not_global_scan_pause():
    service = TradingService.__new__(TradingService)
    service._model_execution_modes = {}
    service.risk_engine = SimpleNamespace(
        circuit_breaker=SimpleNamespace(
            is_open=False,
            get_state=lambda: {},
        ),
        position_checker=SimpleNamespace(
            entry_capacity_reason=lambda **_kwargs: None,
        ),
    )
    service.execution_allocation_state = lambda _mode: _async_value({"total_pnl": -20.0})
    service._get_okx_balance_snapshot_for_mode = lambda _mode: _async_value(
        {
            "free": 1000.0,
            "allocatable": 1000.0,
            "equity": 1000.0,
        }
    )

    class FakeLossPause:
        async def cooldown_loss_pause_reason(self, *_args):
            return "日内亏损冷却，仅降速"

        async def recent_loss_streak_pause_reason(self, *_args):
            return "连续亏损冷却，仅降速"

    service.new_pair_loss_pause = FakeLossPause()

    reason = await service._new_pair_analysis_pause_reason("ensemble_trader", open_positions=[])

    assert reason is None


def test_shadow_backtest_service_is_not_a_trading_service_private_rule():
    assert callable(
        ShadowBacktestService(
            latest_price_provider=lambda _symbol: None,
            symbol_normalizer=lambda symbol: str(symbol or ""),
            float_parser=lambda _value, default=0.0: default,
        ).update_due
    )
    assert not hasattr(TradingService, "_create_shadow_backtests")
    assert not hasattr(TradingService, "_update_due_shadow_backtests")
    assert not hasattr(TradingService, "_record_shadow_memory_in_session")
    assert not hasattr(TradingService, "_shadow_expert_lessons")
    assert not hasattr(TradingService, "_shadow_memory_pattern")
    assert not hasattr(TradingService, "_shadow_feature_bucket")


def test_stale_entry_candidate_expirer_is_not_a_trading_service_private_rule():
    assert callable(StaleEntryCandidateExpirer(lambda _value, default=0.0: default).expire)
    assert not hasattr(TradingService, "_expire_stale_waiting_entry_candidates")
    assert not hasattr(TradingService, "_is_pending_execution_reason")
    assert not hasattr(TradingService, "_pending_execution_failed_reason")
    assert not hasattr(TradingService, "_action_label")


def test_decision_final_state_ensurer_is_not_a_trading_service_private_rule():
    assert callable(
        DecisionFinalStateEnsurer(
            execution_reason_unusable_checker=lambda _reason: False,
            execution_reason_recoverer=lambda _row: None,
            model_execution_mode_provider=lambda _model_name: "paper",
        ).ensure
    )
    assert not hasattr(TradingService, "_ensure_decision_final_state")


@pytest.mark.asyncio
async def test_trading_service_position_review_boundaries_call_internal_owners():
    service = TradingService.__new__(TradingService)
    calls: list[tuple[Any, ...]] = []
    decision = _decision(Action.CLOSE_LONG)
    assessment = SimpleNamespace(warnings=[])

    def set_loop_stage(stage):
        calls.append(("stage", stage))

    async def enforce_sl_tp(feature_vectors):
        calls.append(("sl_tp", sorted(feature_vectors)))
        return [{"symbol": "BTC/USDT"}]

    class FakeSyncService:
        async def get_open_positions_context(self):
            calls.append(("open_positions",))
            return [{"symbol": "BTC/USDT"}]

    async def review_positions(
        open_positions,
        feature_vectors,
        *,
        results,
        round_decision_ids,
        position_entry_pause_reason,
        max_groups_override,
    ):
        calls.append(
            (
                "review",
                len(open_positions),
                sorted(feature_vectors),
                len(results["executions"]),
                len(round_decision_ids),
                position_entry_pause_reason,
                max_groups_override,
            )
        )
        return [("BTC/USDT", "ensemble_trader", decision, assessment, 456)], {
            ("ensemble_trader", "ETH/USDT")
        }

    async def claim_symbol(symbol, scope):
        calls.append(("claim", symbol, scope))
        return True

    def normalize_symbol(symbol):
        calls.append(("normalize", symbol))
        return "BTC/USDT"

    async def execute_candidate(
        symbol,
        model_name,
        decision_arg,
        assessment_arg,
        decision_db_id,
        results,
        *,
        open_positions=None,
    ):
        calls.append(
            (
                "execute",
                symbol,
                model_name,
                decision_arg.action.value,
                assessment_arg is assessment,
                decision_db_id,
                len(open_positions or []),
            )
        )
        return None

    service._set_loop_stage = set_loop_stage  # type: ignore[method-assign]
    service._enforce_sl_tp = enforce_sl_tp  # type: ignore[method-assign]
    service.okx_sync_service = FakeSyncService()
    service._review_open_positions = review_positions  # type: ignore[method-assign]
    service._try_claim_analysis_symbol = claim_symbol  # type: ignore[method-assign]
    service._normalize_position_symbol = normalize_symbol  # type: ignore[method-assign]
    service._execute_candidate = execute_candidate  # type: ignore[method-assign]

    service.set_loop_stage("position")
    assert await service.enforce_sl_tp_for_position_review({"BTC/USDT": object()}) == [
        {"symbol": "BTC/USDT"}
    ]
    assert await service.open_positions_context_for_position_review() == [{"symbol": "BTC/USDT"}]
    review_result = await service.review_open_positions_for_position_service(
        [{"symbol": "BTC/USDT"}],
        {"BTC/USDT": object()},
        results={"executions": []},
        round_decision_ids={123},
        position_entry_pause_reason="paused",
        max_groups_override=3,
    )
    assert review_result[1] == {("ensemble_trader", "ETH/USDT")}
    assert await service.claim_analysis_symbol("BTC/USDT", "position")
    assert service.normalize_position_symbol("BTC/USDT") == "BTC/USDT"
    await service.execute_position_review_candidate(
        "BTC/USDT",
        "ensemble_trader",
        decision,
        assessment,
        456,
        {"executions": []},
        open_positions=[{"symbol": "BTC/USDT"}],
    )

    assert calls == [
        ("stage", "position"),
        ("sl_tp", ["BTC/USDT"]),
        ("open_positions",),
        ("review", 1, ["BTC/USDT"], 0, 1, "paused", 3),
        ("claim", "BTC/USDT", "position"),
        ("normalize", "BTC/USDT"),
        ("execute", "BTC/USDT", "ensemble_trader", "close_long", True, 456, 1),
    ]


@pytest.mark.asyncio
async def test_fast_sl_tp_delegates_execution_to_execution_service_boundary():
    service = TradingService.__new__(TradingService)
    calls: list[tuple[Any, ...]] = []
    service._decision_count = 0

    class FakeSyncService:
        async def get_open_positions_context(self):
            return [
                {
                    "model_name": "ensemble_trader",
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "is_open": True,
                    "entry_price": 100.0,
                    "current_price": 110.0,
                    "stop_loss": 90.0,
                    "take_profit": 105.0,
                    "quantity": 2.0,
                    "leverage": 2.0,
                    "unrealized_pnl": 20.0,
                    "created_at": "2026-06-09T10:00:00+00:00",
                }
            ]

    class FakePositionTime:
        def position_age_minutes(self, created_at):
            calls.append(("position_age", created_at))
            return 120.0

    class FakeProfitPeaks:
        def update(self, **kwargs):
            calls.append(("peak_update", kwargs["symbol"], kwargs["side"]))
            return {"peak_unrealized_pnl": 20.0}

        def remember_profit_exit(self, model_name, symbol, side):
            calls.append(("remember_profit_exit", model_name, symbol, side))

    class FakePredictiveReversal:
        def evidence(self, **_kwargs):
            return {"score": 0.0}

    class FakeFastRisk:
        def profit_drawdown_exit_plan(self, **_kwargs):
            return {"should_exit": False}

    async def log_decision(decision, is_paper):
        calls.append(("log_decision", decision.action.value, is_paper))
        return 321

    async def execute_candidate(
        symbol,
        model_name,
        decision,
        assessment,
        decision_db_id,
        results,
        *,
        open_positions=None,
        refresh_exit_positions=True,
    ):
        calls.append(
            (
                "execute_candidate",
                symbol,
                model_name,
                decision.action.value,
                assessment.warnings,
                decision_db_id,
                len(open_positions or []),
                refresh_exit_positions,
            )
        )
        return ExecutionResult(
            order_id="fast-close-1",
            exchange_order_id="exchange-fast-close-1",
            symbol=symbol,
            side=decision.action.value,
            order_type="market",
            quantity=2.0,
            price=110.0,
            status=OrderStatus.FILLED,
            pnl=20.0,
            raw_response={},
        )

    async def log_risk_event(level, symbol, message, model_name):
        calls.append(("risk_event", level, symbol, model_name, "PnL" in message))

    service.okx_sync_service = FakeSyncService()
    service.position_time = FakePositionTime()
    service.position_profit_peaks = FakeProfitPeaks()
    service.exit_predictive_reversal = FakePredictiveReversal()
    service.exit_fast_risk = FakeFastRisk()
    service._normalize_position_symbol = lambda symbol: str(symbol)
    service._get_model_execution_mode = lambda _model_name: "paper"
    service._log_decision = log_decision
    service._execute_candidate = execute_candidate
    service._is_exchange_confirmed_execution = (
        lambda execution_result: execution_result.status == OrderStatus.FILLED
    )
    service._is_exit_progress_execution = lambda _execution_result: False
    service._log_risk_event = log_risk_event

    feature_vectors = {
        "BTC/USDT": SimpleNamespace(
            current_price=110.0,
            returns_1=0.0,
            returns_5=0.0,
            returns_20=0.0,
            volume_ratio=1.0,
            rsi_14=55.0,
            bb_pct=0.7,
            macd_diff=0.0,
            adx_14=20.0,
        )
    }

    auto_closes = await service._enforce_sl_tp(feature_vectors)

    assert service._decision_count == 1
    assert auto_closes == [
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "side": "long",
            "quantity": 2.0,
            "entry_price": 100.0,
            "exit_price": 110.0,
            "pnl": 20.0,
            "trigger": "take_profit",
            "close_fraction": 1.0,
            "status": "filled",
        }
    ]
    assert (
        "execute_candidate",
        "BTC/USDT",
        "ensemble_trader",
        "close_long",
        [],
        321,
        1,
        False,
    ) in calls
    assert ("risk_event", "info", "BTC/USDT", "ensemble_trader", True) in calls


def test_exit_policy_uses_injected_exit_cooldown_boundary():
    calls: list[tuple[str, str]] = []

    class FakeCooldown:
        def recent_exit_cooldown_reason(self, model_name, decision):
            calls.append((model_name, decision.symbol))
            return "cooldown-blocked"

    policy = ExitPolicy(exit_cooldown=FakeCooldown())

    assert (
        policy.recent_exit_cooldown_reason(
            "ensemble_trader",
            _decision(Action.CLOSE_LONG),
        )
        == "cooldown-blocked"
    )
    assert calls == [("ensemble_trader", "BTC/USDT")]


def test_exit_policy_fails_fast_without_exit_cooldown_dependency():
    policy = ExitPolicy()

    with pytest.raises(RuntimeError, match="exit_cooldown"):
        policy.recent_exit_cooldown_reason(
            "ensemble_trader",
            _decision(Action.CLOSE_LONG),
        )


def test_entry_policy_uses_injected_decision_freshness_boundary():
    calls: list[str] = []

    class FakeFreshness:
        def stale_decision_reason(self, decision):
            calls.append(decision.symbol)
            return "stale-blocked"

    policy = EntryPolicy(decision_freshness=FakeFreshness())

    assert policy.stale_decision_reason(_decision(Action.LONG)) == "stale-blocked"
    assert calls == ["BTC/USDT"]


def test_entry_and_exit_policies_do_not_keep_legacy_orchestrator_reference():
    entry_policy = EntryPolicy()
    exit_policy = ExitPolicy()

    assert not hasattr(entry_policy, "orchestrator")
    assert not hasattr(exit_policy, "orchestrator")


def test_entry_policy_uses_injected_entry_priority_boundary():
    calls: list[tuple[str, int | None, int | None]] = []

    class FakePriority:
        def immediate_execution_reason(self, decision):
            calls.append((decision.symbol, None, None))
            return "immediate"

        def wait_sort_reason(self, decision, *, rank=None, candidate_count=None):
            calls.append((decision.symbol, rank, candidate_count))
            return "wait"

    policy = EntryPolicy(entry_priority=FakePriority())
    decision = _decision(Action.LONG)

    assert policy.immediate_execution_reason(decision) == "immediate"
    assert policy.wait_sort_reason(decision, rank=1, candidate_count=3) == "wait"
    assert calls == [("BTC/USDT", None, None), ("BTC/USDT", 1, 3)]


def test_entry_policy_uses_injected_opportunity_score_boundary():
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def fake_score(decision, strategy):
        calls.append((decision.symbol, strategy))
        return 1.23

    policy = EntryPolicy(
        entry_opportunity_score=EntryOpportunityScorePolicy(fake_score),
    )
    strategy = {"min_opportunity_score": 0.95}

    assert policy.score_candidate(_decision(Action.LONG), strategy) == 1.23
    assert calls == [("BTC/USDT", strategy)]


def test_entry_policy_fails_fast_without_opportunity_score_dependency():
    policy = EntryPolicy()

    with pytest.raises(RuntimeError, match="entry_opportunity_score"):
        policy.score_candidate(_decision(Action.LONG), {})


def test_entry_policy_uses_injected_opportunity_gate_boundary():
    calls: list[str] = []

    class FakeGate:
        def gate_reason(self, decision):
            calls.append(decision.symbol)
            return "gate-blocked"

    policy = EntryPolicy(entry_opportunity_gate=FakeGate())

    assert policy.gate_reason(_decision(Action.LONG)) == "gate-blocked"
    assert calls == ["BTC/USDT"]


def test_entry_policy_fails_fast_without_opportunity_gate_dependency():
    policy = EntryPolicy()

    with pytest.raises(RuntimeError, match="entry_opportunity_gate"):
        policy.gate_reason(_decision(Action.LONG))


def test_entry_opportunity_gate_checks_suspicious_symbol_before_legacy_evaluator():
    calls: list[str] = []

    class FakeSuspicious:
        def reason(self, symbol):
            calls.append(symbol)
            return "suspicious-blocked"

    policy = EntryPolicy(
        entry_opportunity_gate=EntryOpportunityGatePolicy(
            lambda decision: "legacy-blocked",
            FakeSuspicious(),
        ),
    )

    assert policy.gate_reason(_decision(Action.LONG)) == "suspicious-blocked"
    assert calls == ["BTC/USDT"]


@pytest.mark.asyncio
async def test_entry_policy_uses_injected_profit_risk_sizing_boundary():
    calls: list[tuple[str, str, int]] = []

    async def fake_sizing(decision, model_mode, open_positions):
        calls.append((decision.symbol, model_mode, len(open_positions)))

    policy = EntryPolicy(
        entry_profit_risk_sizing=EntryProfitRiskSizingPolicy(fake_sizing),
    )

    await policy.apply_profit_risk_sizing(
        _decision(Action.LONG),
        "paper",
        [{"is_open": True}],
    )

    assert calls == [("BTC/USDT", "paper", 1)]


@pytest.mark.asyncio
async def test_entry_profit_risk_sizing_policy_owns_runtime_sizing_without_private_callback():
    async def allocated_balance(_model_mode, _decision):
        return 1000.0

    decision = _decision(Action.LONG)
    decision.raw_response = {
        "opportunity_score": {
            "score": 3.0,
            "min_score_required": 0.95,
            "expected_net_return_pct": 0.8,
            "expected_loss_pct": 1.0,
            "tail_risk_score": 0.15,
            "raw_expected_return_pct": 0.8,
            "profit_quality_ratio": 1.0,
            "server_profit_loss_probability": 0.40,
            "ml_aligned": True,
            "local_profit_aligned": True,
            "timeseries_aligned": False,
            "evidence_score": {
                "tier": "normal",
                "effective_score": 82.0,
                "size_multiplier": 1.0,
                "max_size_pct": None,
            },
        }
    }
    policy = EntryProfitRiskSizingPolicy(
        allocated_order_balance=allocated_balance,
        entry_low_payoff_quality=EntryLowPayoffQualityPolicy(),
        entry_stop_loss_budget=EntryStopLossBudgetPolicy(),
        entry_stress_stop=EntryStressStopPolicy(),
        entry_existing_winner_context=EntryExistingWinnerContextPolicy(lambda symbol: str(symbol)),
        max_leverage_provider=lambda: 10.0,
    )

    assert policy.evaluator is None

    await policy.apply(decision, "paper", [])

    sizing = decision.raw_response["profit_risk_sizing"]
    assert sizing["risk_mode"] == "normal"
    assert sizing["quality_tier"] == "base"
    assert sizing["planned_stop_loss_usdt"] > 0


def test_entry_opportunity_gate_treats_strategy_learning_pause_as_advisory():
    decision = _decision(Action.LONG)
    decision.raw_response = {
        "strategy_learning_context": {
            "strategy_learning_entry_pause": True,
            "strategy_learning_entry_pause_reason": "策略护栏已触发回滚且持仓压力仍在，暂停新开仓探针。",
        },
        "opportunity_score": {
            "score": 3.0,
            "min_score_required": 0.95,
            "expected_net_return_pct": 0.8,
            "expected_loss_pct": 1.0,
            "tail_risk_score": 0.15,
            "raw_expected_return_pct": 0.8,
            "profit_quality_ratio": 1.0,
            "server_profit_loss_probability": 0.40,
            "ml_aligned": True,
            "local_profit_aligned": True,
            "evidence_score": {
                "tier": "normal",
                "effective_score": 82.0,
                "size_multiplier": 1.0,
                "max_size_pct": None,
            },
        },
    }

    reason = EntryOpportunityGatePolicy().gate_reason(decision)

    assert reason is None
    opportunity = decision.raw_response["opportunity_score"]
    assert opportunity["strategy_learning_pause_is_hard_gate"] is False
    assert opportunity["execution_advisory_warnings"][0]["blocks_entry"] is False


@pytest.mark.asyncio
async def test_entry_profit_risk_sizing_converts_strategy_learning_pause_to_probe():
    async def allocated_balance(_model_mode, _decision):
        return 1000.0

    decision = _decision(Action.LONG)
    decision.position_size_pct = 0.05
    decision.raw_response = {
        "strategy_learning_context": {
            "strategy_learning_entry_pause": True,
            "strategy_learning_entry_pause_reason": "策略护栏暂停新探针",
            "strategy_learning": {
                "runtime": {
                    "profile_id": "candidate_1",
                    "position_size_multiplier": 0.62,
                    "probe_fraction": 0.05,
                    "max_probe_size_pct": 0.018,
                }
            },
        },
        "opportunity_score": {
            "score": 3.0,
            "min_score_required": 0.95,
            "expected_net_return_pct": 4.0,
            "expected_loss_pct": 1.0,
            "tail_risk_score": 0.15,
            "raw_expected_return_pct": 4.0,
            "profit_quality_ratio": 1.0,
            "server_profit_loss_probability": 0.35,
            "ml_aligned": True,
            "local_profit_aligned": True,
            "timeseries_aligned": True,
            "evidence_score": {
                "tier": "normal",
                "effective_score": 82.0,
                "size_multiplier": 1.0,
                "max_size_pct": None,
            },
        },
    }
    policy = EntryProfitRiskSizingPolicy(
        allocated_order_balance=allocated_balance,
        entry_low_payoff_quality=EntryLowPayoffQualityPolicy(),
        entry_stop_loss_budget=EntryStopLossBudgetPolicy(),
        entry_stress_stop=EntryStressStopPolicy(),
        entry_existing_winner_context=EntryExistingWinnerContextPolicy(lambda symbol: str(symbol)),
        max_leverage_provider=lambda: 10.0,
    )

    await policy.apply(decision, "paper", [])

    sizing = decision.raw_response["profit_risk_sizing"]["strategy_learning_sizing"]
    assert sizing["applied"] is True
    assert sizing["entry_paused"] is False
    assert sizing["strategy_learning_pause_is_hard_gate"] is False
    assert sizing["recovery_probe_allowed"] is True
    assert sizing["reason"] == "策略护栏暂停新探针"
    assert sizing["adaptive_recovery_lift_applied"] is True
    assert sizing["adaptive_recovery_cap_pct"] > 0.012
    assert decision.position_size_pct >= 0.018


@pytest.mark.asyncio
async def test_entry_profit_risk_sizing_allows_recovery_probe_when_not_paused():
    async def allocated_balance(_model_mode, _decision):
        return 1000.0

    decision = _decision(Action.LONG)
    decision.position_size_pct = 0.05
    decision.raw_response = {
        "strategy_learning_context": {
            "strategy_learning_entry_pause": False,
            "strategy_learning_health_guard_active": True,
            "strategy_learning_recovery_probe_allowed": True,
            "strategy_learning_recovery_probe_reason": "fallback 依赖偏高，改为极小仓恢复探针",
            "strategy_learning_sizing": {
                "profile_id": "balanced_probe",
                "position_size_multiplier": 0.8,
                "probe_fraction": 0.05,
                "max_probe_size_pct": 0.02,
            },
        },
        "opportunity_score": {
            "score": 3.0,
            "min_score_required": 0.95,
            "expected_net_return_pct": 4.0,
            "expected_loss_pct": 1.0,
            "tail_risk_score": 0.15,
            "raw_expected_return_pct": 4.0,
            "profit_quality_ratio": 1.0,
            "server_profit_loss_probability": 0.35,
            "ml_aligned": True,
            "local_profit_aligned": True,
            "timeseries_aligned": True,
            "evidence_score": {
                "tier": "normal",
                "effective_score": 82.0,
                "size_multiplier": 1.0,
                "max_size_pct": None,
            },
        },
    }
    policy = EntryProfitRiskSizingPolicy(
        allocated_order_balance=allocated_balance,
        entry_low_payoff_quality=EntryLowPayoffQualityPolicy(),
        entry_stop_loss_budget=EntryStopLossBudgetPolicy(),
        entry_stress_stop=EntryStressStopPolicy(),
        entry_existing_winner_context=EntryExistingWinnerContextPolicy(lambda symbol: str(symbol)),
        max_leverage_provider=lambda: 10.0,
    )

    await policy.apply(decision, "paper", [])

    sizing = decision.raw_response["profit_risk_sizing"]["strategy_learning_sizing"]
    assert sizing["applied"] is True
    assert not sizing.get("entry_paused", False)
    assert sizing["health_guard_active"] is True
    assert sizing["recovery_probe_allowed"] is True
    assert sizing["quality_override"] is True
    assert sizing["probe_cap_applied"] is False
    assert sizing["adaptive_recovery_lift_applied"] is True
    assert sizing["adaptive_recovery_cap_pct"] > 0.012
    assert decision.position_size_pct >= 0.018


@pytest.mark.asyncio
async def test_entry_profit_risk_sizing_applies_strategy_learning_probe_cap():
    async def allocated_balance(_model_mode, _decision):
        return 1000.0

    decision = _decision(Action.LONG)
    decision.position_size_pct = 0.05
    decision.raw_response = {
        "strategy_mode": {
            "strategy_learning_sizing": {
                "profile_id": "balanced_probe",
                "position_size_multiplier": 0.8,
                "probe_fraction": 0.08,
                "max_probe_size_pct": 0.018,
                "side_overrides": {"long": {"size_multiplier": 0.9, "reason": "long probe"}},
            }
        },
        "opportunity_score": {
            "score": 3.0,
            "min_score_required": 0.95,
            "expected_net_return_pct": 0.8,
            "expected_loss_pct": 1.0,
            "tail_risk_score": 0.15,
            "raw_expected_return_pct": 0.8,
            "profit_quality_ratio": 1.0,
            "server_profit_loss_probability": 0.40,
            "ml_aligned": True,
            "local_profit_aligned": True,
            "timeseries_aligned": False,
            "evidence_score": {
                "tier": "normal",
                "effective_score": 82.0,
                "size_multiplier": 1.0,
                "max_size_pct": None,
            },
        },
    }
    policy = EntryProfitRiskSizingPolicy(
        allocated_order_balance=allocated_balance,
        entry_low_payoff_quality=EntryLowPayoffQualityPolicy(),
        entry_stop_loss_budget=EntryStopLossBudgetPolicy(),
        entry_stress_stop=EntryStressStopPolicy(),
        entry_existing_winner_context=EntryExistingWinnerContextPolicy(lambda symbol: str(symbol)),
        max_leverage_provider=lambda: 10.0,
    )

    await policy.apply(decision, "paper", [])

    sizing = decision.raw_response["profit_risk_sizing"]["strategy_learning_sizing"]
    assert sizing["applied"] is True
    assert sizing["profile_id"] == "balanced_probe"
    assert sizing["probe_cap_applied"] is True
    assert decision.position_size_pct <= 0.018


@pytest.mark.asyncio
async def test_entry_profit_risk_sizing_unlocks_strong_recovery_probe():
    async def allocated_balance(_model_mode, _decision):
        return 1000.0

    decision = _decision(Action.LONG)
    decision.position_size_pct = 0.05
    decision.raw_response = {
        "strategy_learning_context": {
            "strategy_learning_health_guard_active": True,
            "strategy_learning_recovery_probe_allowed": True,
            "strategy_learning_sizing": {
                "profile_id": "balanced_probe",
                "position_size_multiplier": 0.6,
                "probe_fraction": 0.08,
                "max_probe_size_pct": 0.012,
            },
        },
        "opportunity_score": {
            "score": 2.4,
            "min_score_required": 1.0,
            "expected_net_return_pct": 1.35,
            "expected_loss_pct": 0.8,
            "tail_risk_score": 0.30,
            "raw_expected_return_pct": 1.35,
            "profit_quality_ratio": 1.05,
            "server_profit_loss_probability": 0.38,
            "ml_aligned": True,
            "local_profit_aligned": True,
            "timeseries_aligned": False,
            "evidence_score": {
                "tier": "normal",
                "effective_score": 82.0,
                "size_multiplier": 1.0,
                "max_size_pct": None,
            },
        },
    }
    policy = EntryProfitRiskSizingPolicy(
        allocated_order_balance=allocated_balance,
        entry_low_payoff_quality=EntryLowPayoffQualityPolicy(),
        entry_stop_loss_budget=EntryStopLossBudgetPolicy(),
        entry_stress_stop=EntryStressStopPolicy(),
        entry_existing_winner_context=EntryExistingWinnerContextPolicy(lambda symbol: str(symbol)),
        max_leverage_provider=lambda: 10.0,
    )

    await policy.apply(decision, "paper", [])

    sizing = decision.raw_response["profit_risk_sizing"]["strategy_learning_sizing"]
    assert sizing["quality_override"] is True
    assert sizing["probe_cap_applied"] is False
    assert decision.position_size_pct > 0.012
    assert decision.raw_response["profit_risk_sizing"]["final_notional_usdt"] >= 30.0


@pytest.mark.asyncio
async def test_entry_profit_risk_sizing_reads_strategy_learning_context_probe_cap():
    async def allocated_balance(_model_mode, _decision):
        return 1000.0

    decision = _decision(Action.LONG)
    decision.position_size_pct = 0.06
    decision.raw_response = {
        "strategy_learning_context": {
            "strategy_learning_release_pressure_active": True,
            "strategy_learning_release_pressure_reason": "低质量仓位压力，先释放并只做小仓探针",
            "strategy_learning_sizing": {
                "profile_id": "loss_release",
                "position_size_multiplier": 0.9,
                "probe_fraction": 0.03,
                "max_probe_size_pct": 0.014,
                "side_overrides": {"long": {"size_multiplier": 0.8}},
            },
        },
        "opportunity_score": {
            "score": 3.0,
            "min_score_required": 0.95,
            "expected_net_return_pct": 0.8,
            "expected_loss_pct": 1.0,
            "tail_risk_score": 0.15,
            "raw_expected_return_pct": 0.8,
            "profit_quality_ratio": 1.0,
            "server_profit_loss_probability": 0.40,
            "ml_aligned": True,
            "local_profit_aligned": True,
            "timeseries_aligned": False,
            "evidence_score": {
                "tier": "normal",
                "effective_score": 82.0,
                "size_multiplier": 1.0,
                "max_size_pct": None,
            },
        },
    }
    policy = EntryProfitRiskSizingPolicy(
        allocated_order_balance=allocated_balance,
        entry_low_payoff_quality=EntryLowPayoffQualityPolicy(),
        entry_stop_loss_budget=EntryStopLossBudgetPolicy(),
        entry_stress_stop=EntryStressStopPolicy(),
        entry_existing_winner_context=EntryExistingWinnerContextPolicy(lambda symbol: str(symbol)),
        max_leverage_provider=lambda: 10.0,
    )

    await policy.apply(decision, "paper", [])

    sizing = decision.raw_response["profit_risk_sizing"]["strategy_learning_sizing"]
    assert sizing["applied"] is True
    assert sizing["profile_id"] == "loss_release"
    assert sizing["probe_cap_applied"] is True
    assert sizing["release_pressure_active"] is True
    assert decision.position_size_pct <= 0.014


@pytest.mark.asyncio
async def test_entry_profit_risk_sizing_does_not_trap_strong_quality_in_release_probe():
    async def allocated_balance(_model_mode, _decision):
        return 1000.0

    decision = _decision(Action.LONG)
    decision.position_size_pct = 0.05
    decision.suggested_leverage = 5.0
    decision.raw_response = {
        "strategy_learning_context": {
            "strategy_learning_release_pressure_active": True,
            "strategy_learning_release_pressure_reason": "release old low quality slots",
            "strategy_learning_sizing": {
                "profile_id": "loss_release",
                "position_size_multiplier": 0.45,
                "probe_fraction": 0.08,
                "max_probe_size_pct": 0.012,
                "side_overrides": {"long": {"size_multiplier": 0.62}},
            },
        },
        "opportunity_score": {
            "score": 2.2,
            "min_score_required": 1.0,
            "expected_net_return_pct": 0.95,
            "expected_loss_pct": 0.55,
            "tail_risk_score": 0.32,
            "raw_expected_return_pct": 0.95,
            "profit_quality_ratio": 0.92,
            "server_profit_loss_probability": 0.44,
            "ml_aligned": True,
            "local_profit_aligned": True,
            "timeseries_aligned": False,
            "evidence_score": {
                "tier": "normal",
                "effective_score": 82.0,
                "size_multiplier": 1.0,
                "max_size_pct": None,
            },
        },
    }
    policy = EntryProfitRiskSizingPolicy(
        allocated_order_balance=allocated_balance,
        entry_low_payoff_quality=EntryLowPayoffQualityPolicy(),
        entry_stop_loss_budget=EntryStopLossBudgetPolicy(),
        entry_stress_stop=EntryStressStopPolicy(),
        entry_existing_winner_context=EntryExistingWinnerContextPolicy(lambda symbol: str(symbol)),
        max_leverage_provider=lambda: 10.0,
    )

    await policy.apply(decision, "paper", [])

    sizing = decision.raw_response["profit_risk_sizing"]
    strategy_sizing = sizing["strategy_learning_sizing"]
    assert sizing["strategy_quality_override"] is True
    assert "strong_positive_strategy_signal" in sizing["strategy_quality_override_reasons"]
    assert strategy_sizing["quality_override"] is True
    assert strategy_sizing["probe_cap_applied"] is False
    assert decision.position_size_pct > 0.012
    assert sizing["final_notional_usdt"] > 60.0


@pytest.mark.asyncio
async def test_entry_policy_fails_fast_without_profit_risk_sizing_dependency():
    policy = EntryPolicy()

    with pytest.raises(RuntimeError, match="entry_profit_risk_sizing"):
        await policy.apply_profit_risk_sizing(_decision(Action.LONG), "paper", [])


@pytest.mark.asyncio
async def test_entry_policy_scores_missing_opportunity_before_sizing_and_gate() -> None:
    calls: list[str] = []
    decision = _decision(Action.LONG)
    decision.raw_response = {
        "strategy_mode": {"min_opportunity_score": 0.7},
        "strategy_learning_context": {"strategy_profile_id": "unit-profile"},
    }

    def fake_score(scored_decision, strategy):
        calls.append("score")
        assert strategy["min_opportunity_score"] == 0.7
        assert strategy["strategy_profile_id"] == "unit-profile"
        raw = scored_decision.raw_response
        raw["opportunity_score"] = {
            "score": 1.35,
            "min_score_required": 0.7,
            "expected_net_return_pct": 0.8,
            "profit_quality_ratio": 1.8,
            "tail_risk_score": 0.2,
            "success_probability": 0.62,
        }
        scored_decision.raw_response = raw
        return 1.35

    async def fake_sizing(sized_decision, model_mode, open_positions):
        calls.append("sizing")
        assert model_mode == "paper"
        assert len(open_positions) == 1
        assert sized_decision.raw_response["opportunity_score"]["score"] == 1.35

    class FakeGate:
        def gate_reason(self, gated_decision):
            calls.append("gate")
            assert gated_decision.raw_response["opportunity_score"]["score"] == 1.35
            return None

    policy = EntryPolicy(
        entry_opportunity_score=EntryOpportunityScorePolicy(fake_score),
        entry_profit_risk_sizing=EntryProfitRiskSizingPolicy(fake_sizing),
        entry_opportunity_gate=FakeGate(),
    )

    result = await policy.evaluate(
        decision,
        "ensemble_trader",
        "paper",
        [{"symbol": "ETH/USDT"}],
    )

    assert result.passed is True
    assert calls == ["score", "sizing", "gate"]
    assert decision.raw_response["opportunity_score"]["score"] == 1.35


@pytest.mark.asyncio
async def test_entry_policy_keeps_weak_evidence_shadow_only_even_with_size() -> None:
    calls: list[str] = []
    decision = _decision(Action.LONG)
    decision.position_size_pct = 0.018
    decision.raw_response = {
        "opportunity_score": {
            "score": 0.75,
            "min_score_required": 0.7,
            "expected_net_return_pct": 0.18,
            "profit_quality_ratio": 0.2,
            "tail_risk_score": 0.7,
            "success_probability": 0.47,
            "evidence_score": {
                "tier": "weak_conflict_probe",
                "effective_score": 38.0,
                "size_multiplier": 0.05,
            },
        }
    }

    async def fake_sizing(sized_decision, model_mode, open_positions):
        calls.append("sizing")
        assert model_mode == "paper"
        assert open_positions == []
        assert sized_decision.position_size_pct > 0

    class FakeGate:
        def gate_reason(self, _decision):
            calls.append("gate")
            return None

    policy = EntryPolicy(
        entry_profit_risk_sizing=EntryProfitRiskSizingPolicy(fake_sizing),
        entry_opportunity_gate=FakeGate(),
    )

    result = await policy.evaluate(decision, "ensemble_trader", "paper", [])

    assert result.passed is False
    assert result.blocker == "entry_evidence_shadow_only"
    assert result.data["shadow_only"] is True
    assert result.data["skip_kind"] == "entry_evidence_shadow_only"
    assert result.data["position_size_pct_before_block"] == 0.018
    assert calls == ["sizing"]


@pytest.mark.asyncio
async def test_entry_policy_allows_positive_net_tradeable_probe() -> None:
    calls: list[str] = []
    decision = _decision(Action.LONG)
    decision.position_size_pct = 0.018
    decision.raw_response = {
        "opportunity_score": {
            "score": 1.1,
            "min_score_required": 0.7,
            "expected_net_return_pct": 0.72,
            "profit_quality_ratio": 0.55,
            "tail_risk_score": 0.45,
            "success_probability": 0.58,
            "evidence_score": {
                "tier": "weak_conflict_probe",
                "effective_score": 38.0,
                "size_multiplier": 0.05,
                "tradeable_probe": True,
                "shadow_only": False,
            },
        }
    }

    async def fake_sizing(sized_decision, model_mode, open_positions):
        calls.append("sizing")
        assert model_mode == "paper"
        assert open_positions == []
        assert sized_decision.position_size_pct > 0

    class FakeGate:
        def gate_reason(self, _decision):
            calls.append("gate")
            return None

    policy = EntryPolicy(
        entry_profit_risk_sizing=EntryProfitRiskSizingPolicy(fake_sizing),
        entry_opportunity_gate=FakeGate(),
    )

    result = await policy.evaluate(decision, "ensemble_trader", "paper", [])

    assert result.passed is True
    assert calls == ["sizing", "gate"]


def test_entry_policy_gate_reason_scores_missing_opportunity_for_all_gate_callers() -> None:
    calls: list[str] = []
    decision = _decision(Action.SHORT)
    decision.raw_response = {"strategy_mode": {"min_opportunity_score": 0.8}}

    def fake_score(scored_decision, strategy):
        calls.append("score")
        assert strategy == {"min_opportunity_score": 0.8}
        raw = scored_decision.raw_response
        raw["opportunity_score"] = {
            "score": 1.1,
            "min_score_required": 0.8,
            "expected_net_return_pct": 0.3,
            "profit_quality_ratio": 1.1,
            "tail_risk_score": 0.3,
            "success_probability": 0.55,
        }
        scored_decision.raw_response = raw
        return 1.1

    class FakeGate:
        def gate_reason(self, gated_decision):
            calls.append("gate")
            assert gated_decision.raw_response["opportunity_score"]["score"] == 1.1
            return None

    policy = EntryPolicy(
        entry_opportunity_score=EntryOpportunityScorePolicy(fake_score),
        entry_opportunity_gate=FakeGate(),
    )

    assert policy.gate_reason(decision) is None
    assert calls == ["score", "gate"]


def test_entry_policy_uses_injected_abnormal_wick_guard_boundary():
    calls: list[str] = []

    class FakeGuard:
        def guard_reason(self, decision):
            calls.append(decision.symbol)
            return "wick-blocked"

    policy = EntryPolicy(abnormal_wick_guard=FakeGuard())

    assert policy.abnormal_wick_guard_reason(_decision(Action.LONG)) == "wick-blocked"
    assert calls == ["BTC/USDT"]


@pytest.mark.asyncio
async def test_entry_policy_uses_injected_price_guard_boundary():
    calls: list[str] = []

    class FakePriceGuard:
        async def guard_reason(self, decision):
            calls.append(decision.symbol)
            return "price-blocked"

    policy = EntryPolicy(entry_price_guard=FakePriceGuard())

    reason = await policy.pre_execution_price_guard_reason(_decision(Action.LONG))

    assert reason == "price-blocked"
    assert calls == ["BTC/USDT"]


def test_entry_symbol_blocklist_recognizes_chinese_entry_price_guard_skip_reason():
    policy = EntrySymbolBlocklistPolicy(lambda symbol: str(symbol or ""))

    assert policy.is_entry_price_guard_skip("下单前没有重新拿到最新价格，系统本次跳过。")
    assert policy.is_entry_price_guard_skip("下单前行情质量复核未通过：盘口数据异常。")


def test_trading_service_dashboard_runtime_boundaries_expose_public_state():
    service = TradingService.__new__(TradingService)
    paper_executor = object()
    live_executor = object()
    service._okx_paper = paper_executor
    service._okx_live = live_executor
    service._decision_count = 8
    service._recent_decisions = [{"symbol": "BTC/USDT"}]

    assert service.okx_executor_for_dashboard("paper") is paper_executor
    assert service.okx_executor_for_dashboard("live") is live_executor

    service.reset_decision_runtime_state()

    assert service._decision_count == 0
    assert service._recent_decisions == []


@pytest.mark.asyncio
async def test_trading_service_dashboard_async_boundaries_call_internal_owners():
    service = TradingService.__new__(TradingService)
    calls: list[tuple[str, str]] = []

    async def balance_snapshot(mode):
        calls.append(("balance", mode))
        return {"equity": 123.0}

    async def shadow_total():
        calls.append(("shadow", "completed"))
        return 456

    service._get_okx_balance_snapshot_for_mode = balance_snapshot  # type: ignore[method-assign]
    service._completed_shadow_backtest_total = shadow_total  # type: ignore[method-assign]

    assert await service.get_okx_balance_snapshot_for_mode("live") == {"equity": 123.0}
    assert await service.completed_shadow_backtest_total() == 456
    assert calls == [("balance", "live"), ("shadow", "completed")]


@pytest.mark.asyncio
async def test_paper_balance_snapshot_uses_virtual_account_without_okx() -> None:
    service = TradingService.__new__(TradingService)
    service._safe_float = TradingService._safe_float.__get__(service, TradingService)
    service._okx_paper = None
    service._okx_live = None
    service._okx_balance_snapshot_cache = {}

    class FakePaperExecutor:
        async def get_account_summary(self, model_name: str) -> dict[str, Any]:
            assert model_name == "ensemble_trader"
            return {
                "available_balance": 1234.5,
                "used_margin": 100.0,
                "wallet_balance": 1334.5,
                "equity": 1320.0,
            }

    service.paper_executor = FakePaperExecutor()

    snapshot = await service._get_okx_balance_snapshot_for_mode("paper")

    assert snapshot == {
        "free": 1234.5,
        "used": 100.0,
        "total": 1334.5,
        "cash": 1334.5,
        "equity": 1334.5,
        "allocatable": 1234.5,
        "source": "paper_virtual_account",
        "exchange_required": False,
        "degraded": False,
        "analysis_only_balance": True,
    }


@pytest.mark.asyncio
async def test_paper_new_pair_pause_does_not_depend_on_okx_balance_timeout() -> None:
    service = TradingService.__new__(TradingService)
    service._safe_float = TradingService._safe_float.__get__(service, TradingService)
    service._get_model_execution_mode = lambda _model_name: "paper"
    service.execution_allocation_state = lambda _mode: _async_value({"total_pnl": 0.0})
    service.paper_executor = None
    service._okx_paper = None
    service._okx_live = None
    service._okx_balance_snapshot_cache = {}
    service._get_okx_executor_for_mode = lambda _mode: _async_value(None)
    service.new_pair_loss_pause = NewPairLossPausePolicy(
        balance_snapshot_provider=service._get_okx_balance_snapshot_for_mode
    )
    service.risk_engine = SimpleNamespace(
        circuit_breaker=SimpleNamespace(is_open=False, get_state=lambda: {}),
        position_checker=SimpleNamespace(entry_capacity_reason=lambda **_kwargs: None),
    )

    snapshot = await service._get_okx_balance_snapshot_for_mode("paper")
    reason = await service._new_pair_analysis_pause_reason(
        "ensemble_trader",
        open_positions=[],
    )

    assert snapshot is not None
    assert snapshot["source"] == "paper_configured_budget"
    assert snapshot["analysis_only_balance"] is True
    assert reason is None


@pytest.mark.asyncio
async def test_latest_price_uses_ws_cache_before_okx_rest() -> None:
    service = TradingService.__new__(TradingService)
    service._safe_float = TradingService._safe_float.__get__(service, TradingService)
    service._normalize_position_symbol = lambda symbol: str(symbol or "")

    class FailingRestClient:
        async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
            raise AssertionError(f"REST should not be called for cached {symbol}")

    service.data_service = SimpleNamespace(
        ws_client=SimpleNamespace(
            latest_tickers={"BTC/USDT": {"last_price": 43210.5}},
        ),
        rest_client=FailingRestClient(),
    )

    price = await service._latest_price_for_symbol("BTC/USDT")

    assert price == 43210.5


@pytest.mark.asyncio
async def test_latest_price_falls_back_to_feature_vector_when_okx_rest_fails() -> None:
    service = TradingService.__new__(TradingService)
    service._safe_float = TradingService._safe_float.__get__(service, TradingService)
    service._normalize_position_symbol = lambda symbol: str(symbol or "")

    class FailingRestClient:
        async def fetch_ticker(self, _symbol: str) -> dict[str, Any]:
            raise RuntimeError("okx public instruments timeout")

    async def get_feature_vector(symbol: str) -> SimpleNamespace:
        assert symbol == "ETH/USDT"
        return SimpleNamespace(current_price=3210.75)

    service.data_service = SimpleNamespace(
        ws_client=SimpleNamespace(latest_tickers={}),
        rest_client=FailingRestClient(),
        get_feature_vector=get_feature_vector,
    )

    price = await service._latest_price_for_symbol("ETH/USDT")

    assert price == 3210.75


def test_auto_scan_feature_budget_rotates_market_pool_and_keeps_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    service._normalize_position_symbol = lambda symbol: str(symbol or "")
    service._auto_scan_feature_cursor = 0
    service.entry_symbol_universe = SimpleNamespace(
        dedupe_symbols=lambda symbols: list(dict.fromkeys(symbols))
    )
    monkeypatch.setattr(trading_service, "AUTO_SCAN_FEATURE_FETCH_POOL_MULTIPLIER", 2)
    monkeypatch.setattr(trading_service, "AUTO_SCAN_FEATURE_FETCH_POOL_MIN", 4)
    symbols = [f"S{i}/USDT" for i in range(10)]

    first = service._budget_auto_scan_feature_symbols(
        symbols,
        ["S8/USDT"],
        configured_limit=2,
    )
    second = service._budget_auto_scan_feature_symbols(
        symbols,
        ["S8/USDT"],
        configured_limit=2,
    )

    assert "S8/USDT" in first
    assert "S8/USDT" in second
    assert len(first) == 5
    assert len(second) == 5
    assert first != second
    assert first[1:] == ["S0/USDT", "S1/USDT", "S2/USDT", "S3/USDT"]
    assert second[1:] == ["S4/USDT", "S5/USDT", "S6/USDT", "S7/USDT"]


def test_market_round_time_budget_tracks_runtime_decision_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    monkeypatch.setattr(
        trading_service.settings.__class__,
        "refresh_runtime_env",
        lambda _self, force=False: True,
    )
    monkeypatch.setattr(trading_service.settings, "decision_interval_seconds", 30)

    assert service.market_round_time_budget_seconds() == 27.0
    assert service.market_round_watchdog_seconds() >= 180.0


def test_parallel_loop_intervals_are_not_market_throttles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    monkeypatch.setattr(
        trading_service.settings.__class__,
        "refresh_runtime_env",
        lambda _self, force=False: True,
    )
    monkeypatch.setattr(trading_service.settings, "decision_interval_seconds", 30)

    assert service.market_loop_interval_seconds() == pytest.approx(10.5)
    assert service.position_loop_interval_seconds() == pytest.approx(19.5)
    assert service.market_loop_interval_seconds() < service.position_loop_interval_seconds()
    assert service.market_loop_interval_seconds() < service.market_round_time_budget_seconds()


def test_market_round_budget_is_not_used_as_outer_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    monkeypatch.setattr(
        trading_service.settings.__class__,
        "refresh_runtime_env",
        lambda _self, force=False: True,
    )
    monkeypatch.setattr(trading_service.settings, "decision_interval_seconds", 30)

    assert service.market_round_watchdog_seconds() > service.market_round_time_budget_seconds()
    assert service.market_round_watchdog_seconds() == 180.0


@pytest.mark.asyncio
async def test_feature_batch_wait_cancels_slow_tasks() -> None:
    cancelled = False

    async def slow_task() -> tuple[str, Any]:
        nonlocal cancelled
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled = True
            raise
        return "SLOW/USDT", object()

    async def fast_task() -> tuple[str, Any]:
        return "FAST/USDT", object()

    tasks = [asyncio.create_task(fast_task()), asyncio.create_task(slow_task())]
    done, pending = await asyncio.wait(tasks, timeout=0.01)
    await trading_service.drain_cancelled_tasks(pending)

    assert len(done) == 1
    assert len(pending) == 1
    assert cancelled is True


@pytest.mark.asyncio
async def test_cancelled_feature_tasks_do_not_block_round_drain() -> None:
    cleanup_started = asyncio.Event()

    async def cancellation_resistant_task() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cleanup_started.set()
            await asyncio.sleep(60)

    task = asyncio.create_task(cancellation_resistant_task())
    await asyncio.sleep(0)

    started_at = asyncio.get_running_loop().time()
    await trading_service.drain_cancelled_tasks({task}, timeout_seconds=0.01)
    elapsed = asyncio.get_running_loop().time() - started_at

    assert cleanup_started.is_set()
    assert elapsed < 0.5
    assert task.cancelled() is False
    task.cancel()


@pytest.mark.asyncio
async def test_trading_service_stage_boundary_passes_duration() -> None:
    service = TradingService.__new__(TradingService)
    decision = _decision(Action.LONG)
    captured: dict[str, Any] = {}

    async def record_stage(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"ok": True}

    service._record_and_persist_decision_stage = record_stage  # type: ignore[method-assign]

    await service.record_and_persist_decision_stage(
        9,
        decision,
        "exchange_submit",
        "passed",
        "OKX 已返回",
        {"order_id": "1"},
        duration_sec=1.25,
    )

    assert captured["kwargs"]["duration_sec"] == 1.25
    assert captured["args"][:6] == (
        9,
        decision,
        "exchange_submit",
        "passed",
        "OKX 已返回",
        {"order_id": "1"},
    )


@pytest.mark.asyncio
async def test_trading_service_execution_boundaries_call_internal_owners():
    service = TradingService.__new__(TradingService)
    calls: list[tuple[Any, ...]] = []
    decision = _decision(Action.LONG)

    def model_mode(model_name):
        calls.append(("mode", model_name))
        return "paper"

    async def log_risk(event_type, symbol, details, model_name, severity="warn"):
        calls.append(("risk", event_type, symbol, details, model_name, severity))

    async def record_stage(decision_id, decision_arg, stage, status, reason, data=None):
        calls.append(("stage", decision_id, decision_arg.symbol, stage, status, reason, data))
        return {"stage": stage}

    async def mark_reason(decision_id, reason):
        calls.append(("reason", decision_id, reason))

    async def mark_raw(decision_id, raw_response):
        calls.append(("raw", decision_id, raw_response))

    class FakePositionReviewRiskAlertPolicy:
        def alert_context(self, decision_arg):
            calls.append(("alert_context", decision_arg.symbol))
            return {"message": "risk"}

        def execution_result_text(
            self,
            decision_arg,
            execution_result,
            execution_reason_provider,
        ):
            calls.append(("position_review_result_text", decision_arg.symbol))
            return execution_reason_provider(execution_result)

        def risk_event_detail(self, decision_arg, alert, result_text):
            calls.append(("position_review_detail", decision_arg.symbol, alert, result_text))
            return "position review detail"

    async def duplicate_order_reason(decision_id, decision_arg):
        calls.append(("duplicate", decision_id, decision_arg.symbol))
        return None

    async def okx_executor(mode):
        calls.append(("executor", mode))
        return "okx-executor"

    async def allocated_balance(model_mode, decision_arg):
        calls.append(("allocated", model_mode, decision_arg.symbol))
        return 123.0

    def rejected_result(decision_arg, reason):
        calls.append(("rejected", decision_arg.symbol, reason))
        return ExecutionResult(
            order_id=None,
            exchange_order_id=None,
            symbol=decision_arg.symbol,
            side=decision_arg.action.value,
            order_type="market",
            quantity=0.0,
            price=0.0,
            status=OrderStatus.REJECTED,
            raw_response={"reason": reason},
        )

    def leverage_summary(decision_arg, execution_result, requested):
        calls.append(("leverage", decision_arg.symbol, execution_result.symbol, requested))

    def execution_reason(execution_result):
        calls.append(("execution_reason", execution_result.status.value))
        return execution_result.status.value

    async def mark_pending(decision_id, reason):
        calls.append(("pending", decision_id, reason))

    def is_untradable_error(result_text):
        calls.append(("untradable", result_text))
        return True

    def remember_untradable(symbol, result_text):
        calls.append(("remember_untradable", symbol, result_text))

    def is_transient_error(result_text):
        calls.append(("transient", result_text))
        return True

    def remember_temporary_block(symbol, reason, minutes):
        calls.append(("temporary_block", symbol, reason, minutes))

    def transient_minutes(result_text):
        calls.append(("transient_minutes", result_text))
        return 7.0

    class FakeEntrySymbolBlocklist:
        def is_untradable_exchange_error(self, result_text):
            return is_untradable_error(result_text)

        def remember_untradable_symbol(self, symbol, result_text):
            remember_untradable(symbol, result_text)

        def is_transient_entry_exchange_error(self, result_text):
            return is_transient_error(result_text)

        def remember_temporary_entry_block(self, symbol, reason, minutes):
            remember_temporary_block(symbol, reason, minutes)

        def transient_entry_block_minutes(self, result_text):
            return transient_minutes(result_text)

    async def log_trade(execution_result, model_name, decision_arg, decision_id=None):
        calls.append(
            (
                "trade",
                execution_result.symbol,
                model_name,
                decision_arg.symbol,
                decision_id,
            )
        )

    def is_exchange_confirmed(execution_result):
        calls.append(("confirmed", execution_result.status.value))
        return execution_result.status == OrderStatus.FILLED

    def is_exit_progress(execution_result):
        calls.append(("exit_progress", execution_result.status.value))
        return False

    def result_has_no_position(execution_result):
        calls.append(("no_position", execution_result.status.value))
        return True

    async def persist_position(model_name, decision_arg, execution_result, model_mode):
        calls.append(
            (
                "persist_position",
                model_name,
                decision_arg.symbol,
                execution_result.symbol,
                model_mode,
            )
        )

    def apply_execution(open_positions, model_name, decision_arg, execution_result):
        calls.append(
            (
                "apply_execution",
                len(open_positions),
                model_name,
                decision_arg.symbol,
                execution_result.symbol,
            )
        )

    async def mark_executed(decision_id, price):
        calls.append(("executed", decision_id, price))

    def clear_symbol(symbol):
        calls.append(("clear_symbol", symbol))

    async def persist_account(model_name, decision_model_name, execution_result):
        calls.append(("account_update", model_name, decision_model_name, execution_result.symbol))

    async def account_balance(model_name):
        calls.append(("account_balance", model_name))
        return 456.0

    async def mark_outcome(decision_id, outcome, pnl_pct):
        calls.append(("outcome", decision_id, outcome, pnl_pct))

    class FakeEntryPolicy:
        async def evaluate(self, decision_arg, model_name, model_mode, open_positions):
            calls.append(
                (
                    "entry_policy",
                    decision_arg.symbol,
                    model_name,
                    model_mode,
                    len(open_positions or []),
                )
            )
            return PolicyGateResult.allow({"intent": "entry"})

    class FakeExitPolicy:
        async def evaluate(
            self,
            decision_arg,
            model_name,
            open_positions,
            *,
            refresh_positions=True,
        ):
            calls.append(
                (
                    "exit_policy",
                    decision_arg.symbol,
                    model_name,
                    len(open_positions or []),
                    refresh_positions,
                )
            )
            return PolicyGateResult.allow({"intent": "exit"})

        def has_matching_position(self, positions, model_name, decision_arg):
            calls.append(
                (
                    "local_exit_position",
                    decision_arg.symbol,
                    model_name,
                    len(positions),
                )
            )
            return True

    class FakeAgentSkills:
        def execution_skills(self, **kwargs):
            calls.append(
                (
                    "agent_skills",
                    kwargs["decision"].symbol,
                    kwargs["model_mode"],
                    kwargs["override_balance"],
                )
            )
            return ["guard"]

        def attach(self, decision_arg, *, phase, skills, note):
            calls.append(("attach_agent_skills", decision_arg.symbol, phase, skills, note))

        def block_reason(self, skills, *, for_entry):
            calls.append(("agent_skill_block", skills, for_entry))
            return "skill-block"

    class FakeOkxSyncService:
        async def reconcile_positions(self, reason):
            calls.append(("reconcile", reason))

        async def get_open_positions_context(self):
            calls.append(("open_positions_context",))
            return [{"symbol": "BTC/USDT"}]

        async def has_matching_exchange_exit_position(self, model_name, decision_arg):
            calls.append(("exchange_exit_position", decision_arg.symbol, model_name))
            return True

    class FakeExitCooldown:
        def remember_exit(self, model_name, decision_arg):
            calls.append(("exit_cooldown", decision_arg.symbol, model_name))

    class FakeCircuitBreaker:
        def record_trade(self, amount):
            calls.append(("record_trade", amount))

    class FakeAccounting:
        async def allocated_order_balance(self, model_mode, decision_arg=None):
            return await allocated_balance(model_mode, decision_arg)

        async def persist_account_update(
            self,
            model_name,
            decision_model_name,
            execution_result,
        ):
            await persist_account(model_name, decision_model_name, execution_result)

        async def account_balance(self, model_name):
            return await account_balance(model_name)

    service.entry_policy = FakeEntryPolicy()
    service.exit_policy = FakeExitPolicy()
    service.agent_skills = FakeAgentSkills()
    service.okx_sync_service = FakeOkxSyncService()
    service.exit_cooldown = FakeExitCooldown()
    service.account_accounting_service = FakeAccounting()
    service.risk_engine = SimpleNamespace(circuit_breaker=FakeCircuitBreaker())
    service.position_review_risk_alert_policy = FakePositionReviewRiskAlertPolicy()

    service._get_model_execution_mode = model_mode  # type: ignore[method-assign]
    service._log_risk_event = log_risk  # type: ignore[method-assign]
    service._record_and_persist_decision_stage = record_stage  # type: ignore[method-assign]
    service._mark_decision_reason = mark_reason  # type: ignore[method-assign]
    service._mark_decision_raw_response = mark_raw  # type: ignore[method-assign]
    service._duplicate_decision_order_reason = duplicate_order_reason  # type: ignore[method-assign]
    service._get_okx_executor_for_mode = okx_executor  # type: ignore[method-assign]
    service._rejected_execution_result = rejected_result  # type: ignore[method-assign]
    service._attach_execution_leverage_summary = leverage_summary  # type: ignore[method-assign]
    service._execution_reason_from_result = execution_reason  # type: ignore[method-assign]
    service._mark_decision_pending_execution = mark_pending  # type: ignore[method-assign]
    service.entry_symbol_blocklist = FakeEntrySymbolBlocklist()
    service._log_trade = log_trade  # type: ignore[method-assign]
    service._is_exchange_confirmed_execution = is_exchange_confirmed  # type: ignore[method-assign]
    service._is_exit_progress_execution = is_exit_progress  # type: ignore[method-assign]
    service._result_has_no_exchange_position = result_has_no_position  # type: ignore[method-assign]
    service._trade_count = 3
    service._persist_position_from_execution = persist_position  # type: ignore[method-assign]
    service._apply_execution_to_open_positions = apply_execution  # type: ignore[method-assign]
    service._mark_decision_executed = mark_executed  # type: ignore[method-assign]
    service._clear_market_no_opportunity_symbol = clear_symbol  # type: ignore[method-assign]
    service._mark_decision_outcome = mark_outcome  # type: ignore[method-assign]

    assert service.get_model_execution_mode("ensemble_trader") == "paper"
    await service.log_risk_event("warning", "BTC/USDT", "detail", "ensemble_trader")
    assert await service.record_and_persist_decision_stage(
        12,
        decision,
        "risk_check",
        "passed",
        "ok",
        {"x": 1},
    ) == {"stage": "risk_check"}
    await service.mark_decision_reason(12, "ok")
    await service.mark_decision_raw_response(12, {"a": 1})
    assert service.position_review_alert_context(decision) == {"message": "risk"}
    await service.log_position_review_risk_result(decision, "ensemble_trader", "done")
    assert await service.duplicate_decision_order_reason(12, decision) is None
    assert await service.get_okx_executor_for_mode("paper") == "okx-executor"
    assert await service.allocated_order_balance("paper", decision) == 123.0
    rejected = service.rejected_execution_result(decision, "blocked")
    service.attach_execution_leverage_summary(decision, rejected, 3.0)
    assert service.execution_reason_from_result(rejected) == "rejected"
    await service.mark_decision_pending_execution(12, "pending")
    assert service.is_untradable_exchange_error("bad symbol")
    service.remember_untradable_symbol("BTC/USDT", "bad symbol")
    assert service.is_transient_entry_exchange_error("retry later")
    service.remember_temporary_entry_block("BTC/USDT", "retry later", 7.0)
    assert service.transient_entry_block_minutes("retry later") == 7.0
    await service.log_trade(rejected, "ensemble_trader", decision, 12)
    assert service.is_exchange_confirmed_execution(rejected) is False
    assert service.is_exit_progress_execution(rejected) is False
    assert service.result_has_no_exchange_position(rejected) is True
    service.increment_trade_count()
    assert service._trade_count == 4
    await service.persist_position_from_execution(
        "ensemble_trader",
        decision,
        rejected,
        "paper",
    )
    open_positions: list[dict[str, Any]] = []
    service.apply_execution_to_open_positions(
        open_positions,
        "ensemble_trader",
        decision,
        rejected,
    )
    await service.mark_decision_executed(12, 100.0)
    service.clear_market_no_opportunity_symbol("BTC/USDT")
    await service.persist_account_update("ensemble_trader", decision.model_name, rejected)
    assert await service.get_account_balance("ensemble_trader") == 456.0
    await service.mark_decision_outcome(12, "loss", -0.01)
    service.entry_execution_pipeline = EntryExecutionPipeline(lambda: service.entry_policy)
    service.exit_execution_pipeline = ExitExecutionPipeline(lambda: service.exit_policy)
    entry_result = await service.entry_execution_pipeline.evaluate(
        decision,
        "ensemble_trader",
        "paper",
        [{"symbol": "BTC/USDT"}],
    )
    exit_result = await service.exit_execution_pipeline.evaluate(
        decision,
        "ensemble_trader",
        [{"symbol": "BTC/USDT"}],
    )
    assert entry_result.passed is True
    assert exit_result.passed is True
    assert entry_result.data["strategy_parameters"]["scope"] == "entry_execution"
    assert exit_result.data["strategy_parameters"]["scope"] == "exit_execution"
    assert decision.raw_response["strategy_parameters"]["snapshot"]["version"]
    assert service.execution_agent_skills(
        decision=decision,
        model_mode="paper",
        override_balance=123.0,
    ) == ["guard"]
    service.attach_execution_agent_skills(
        decision,
        phase="execution_precheck",
        skills=["guard"],
        note="note",
    )
    assert service.execution_agent_skill_block_reason(["guard"], for_entry=True) == "skill-block"
    await service.reconcile_positions_for_execution("manual")
    assert await service.open_positions_context_for_execution() == [{"symbol": "BTC/USDT"}]
    assert service.has_matching_local_exit_position(
        [{"symbol": "BTC/USDT"}],
        "ensemble_trader",
        decision,
    )
    assert (
        await service.has_matching_exchange_exit_position_for_execution(
            "ensemble_trader",
            decision,
        )
        is True
    )
    service.remember_exit_cooldown("ensemble_trader", decision)
    service.record_executed_trade_notional(200.0)

    assert calls == [
        ("mode", "ensemble_trader"),
        ("risk", "warning", "BTC/USDT", "detail", "ensemble_trader", "warn"),
        ("stage", 12, "BTC/USDT", "risk_check", "passed", "ok", {"x": 1}),
        ("reason", 12, "ok"),
        ("raw", 12, {"a": 1}),
        ("alert_context", "BTC/USDT"),
        ("alert_context", "BTC/USDT"),
        ("position_review_detail", "BTC/USDT", {"message": "risk"}, "done"),
        (
            "risk",
            "position_review_warning",
            "BTC/USDT",
            "position review detail",
            "ensemble_trader",
            "warn",
        ),
        ("duplicate", 12, "BTC/USDT"),
        ("executor", "paper"),
        ("allocated", "paper", "BTC/USDT"),
        ("rejected", "BTC/USDT", "blocked"),
        ("leverage", "BTC/USDT", "BTC/USDT", 3.0),
        ("execution_reason", "rejected"),
        ("pending", 12, "pending"),
        ("untradable", "bad symbol"),
        ("remember_untradable", "BTC/USDT", "bad symbol"),
        ("transient", "retry later"),
        ("temporary_block", "BTC/USDT", "retry later", 7.0),
        ("transient_minutes", "retry later"),
        ("trade", "BTC/USDT", "ensemble_trader", "BTC/USDT", 12),
        ("confirmed", "rejected"),
        ("exit_progress", "rejected"),
        ("no_position", "rejected"),
        ("persist_position", "ensemble_trader", "BTC/USDT", "BTC/USDT", "paper"),
        ("apply_execution", 0, "ensemble_trader", "BTC/USDT", "BTC/USDT"),
        ("executed", 12, 100.0),
        ("clear_symbol", "BTC/USDT"),
        ("account_update", "ensemble_trader", "ensemble_trader", "BTC/USDT"),
        ("account_balance", "ensemble_trader"),
        ("outcome", 12, "loss", -0.01),
        ("entry_policy", "BTC/USDT", "ensemble_trader", "paper", 1),
        ("exit_policy", "BTC/USDT", "ensemble_trader", 1, True),
        ("agent_skills", "BTC/USDT", "paper", 123.0),
        ("attach_agent_skills", "BTC/USDT", "execution_precheck", ["guard"], "note"),
        ("agent_skill_block", ["guard"], True),
        ("reconcile", "manual"),
        ("open_positions_context",),
        ("local_exit_position", "BTC/USDT", "ensemble_trader", 1),
        ("exchange_exit_position", "BTC/USDT", "ensemble_trader"),
        ("exit_cooldown", "BTC/USDT", "ensemble_trader"),
        ("record_trade", 200.0),
    ]


@pytest.mark.asyncio
async def test_ml_signal_service_completed_shadow_sample_boundary_calls_internal_owner():
    service = MLSignalService.__new__(MLSignalService)
    calls: list[str] = []

    async def completed_shadow_sample_count():
        calls.append("completed")
        return 321

    service._completed_shadow_sample_count = completed_shadow_sample_count  # type: ignore[method-assign]

    assert await service.completed_shadow_sample_count() == 321
    assert calls == ["completed"]


@pytest.mark.asyncio
async def test_ml_signal_auto_train_uses_completed_cursor_for_new_samples() -> None:
    service = MLSignalService()

    async def completed_shadow_sample_count() -> int:
        return 1120

    def current_metadata() -> dict[str, Any]:
        return {
            "sample_count": 1000,
            "last_trained_completed_shadow_sample_count": 1050,
            "trained_at": datetime.now(UTC).isoformat(),
            "test_count": 250,
            "metrics": {
                "long_auc": 0.40,
                "short_auc": 0.41,
                "long_accuracy": 0.48,
                "short_accuracy": 0.49,
                "top_long_avg_return_pct": -0.10,
                "top_short_avg_return_pct": -0.08,
                "top_long_win_rate": 0.40,
                "bottom_long_win_rate": 0.45,
                "top_short_win_rate": 0.42,
                "bottom_short_win_rate": 0.46,
            },
        }

    service._completed_shadow_sample_count = completed_shadow_sample_count  # type: ignore[method-assign]
    service._current_metadata = current_metadata  # type: ignore[method-assign]

    result = await service.maybe_auto_train()

    assert result["reason"] == "not_due"
    assert result["new_sample_count"] == 70
    assert result["last_trained_completed_sample_count"] == 1050
    assert result["training_policy"]["learning_only"] is True
    assert result["training_policy"]["min_new_samples"] == 120


@pytest.mark.asyncio
async def test_entry_policy_uses_injected_high_risk_review_gate_boundary():
    calls: list[tuple[str, str, int]] = []

    class FakeGate:
        async def evaluate(self, decision, model_mode, open_positions):
            calls.append((decision.symbol, model_mode, len(open_positions)))
            return "review-blocked"

    policy = EntryPolicy(high_risk_review_gate=FakeGate())

    reason = await policy.high_risk_review_gate(
        _decision(Action.LONG),
        "paper",
        [{"is_open": True}],
    )

    assert reason == "review-blocked"
    assert calls == [("BTC/USDT", "paper", 1)]


def test_exit_policy_uses_injected_exit_position_matcher_boundary():
    calls: list[tuple[str, str]] = []

    class FakeMatcher:
        def has_matching_position(self, positions, model_name, decision):
            calls.append((model_name, decision.symbol))
            return True

    policy = ExitPolicy(exit_position_matcher=FakeMatcher())

    assert policy.has_matching_position([], "ensemble_trader", _decision(Action.CLOSE_LONG))
    assert calls == [("ensemble_trader", "BTC/USDT")]


def test_exit_policy_fails_fast_without_exit_position_matcher_dependency():
    policy = ExitPolicy()

    with pytest.raises(RuntimeError, match="exit_position_matcher"):
        policy.has_matching_position([], "ensemble_trader", _decision(Action.CLOSE_LONG))


def test_exit_policy_allows_non_exit_without_exit_position_matcher_dependency():
    policy = ExitPolicy()

    assert policy.has_matching_position([], "ensemble_trader", _decision(Action.LONG))


def test_exit_policy_uses_injected_partial_guard_boundary():
    calls: list[tuple[str, str]] = []

    class FakeGuard:
        def guard_reason(self, model_name, decision, open_positions):
            calls.append((model_name, decision.symbol))
            return "partial-blocked"

    policy = ExitPolicy(exit_partial_guard=FakeGuard())
    reason = policy.loss_partial_guard_reason(
        "ensemble_trader",
        _decision(Action.CLOSE_LONG),
        [],
    )

    assert reason == "partial-blocked"
    assert calls == [("ensemble_trader", "BTC/USDT")]


@pytest.mark.asyncio
async def test_exit_policy_uses_injected_profit_precheck_boundary():
    calls: list[tuple[str, int]] = []

    class FakeProfitPrecheck:
        async def guard_reason(self, decision, open_positions):
            calls.append((decision.symbol, len(open_positions or [])))
            return "profit-precheck-blocked"

    policy = ExitPolicy(exit_profit_precheck=FakeProfitPrecheck())

    reason = await policy.pre_execution_profit_guard_reason(
        _decision(Action.CLOSE_LONG),
        [{"symbol": "BTC/USDT"}],
    )

    assert reason == "profit-precheck-blocked"
    assert calls == [("BTC/USDT", 1)]


@pytest.mark.asyncio
async def test_exit_policy_uses_injected_fee_churn_guard_boundary():
    calls: list[tuple[str, str]] = []

    class FakeFeeChurnGuard:
        async def guard_reason(self, model_name, decision):
            calls.append((model_name, decision.symbol))
            return "fee-churn-blocked"

    policy = ExitPolicy(exit_fee_churn_guard=FakeFeeChurnGuard())

    reason = await policy.fee_churn_guard_reason(
        "ensemble_trader",
        _decision(Action.CLOSE_LONG),
    )

    assert reason == "fee-churn-blocked"
    assert calls == [("ensemble_trader", "BTC/USDT")]


@pytest.mark.asyncio
async def test_exit_policy_uses_injected_exit_position_snapshot_boundary():
    calls: list[tuple[str, str]] = []

    class FakeSnapshot:
        async def refresh_positions(self, open_positions):
            calls.append(("refresh", str(len(open_positions or []))))
            if open_positions is not None:
                open_positions[:] = []
            return []

        async def has_matching_exchange_position(self, model_name, decision):
            calls.append((model_name, decision.symbol))
            return False

    class FakeMatcher:
        def has_matching_position(self, positions, model_name, decision):
            return False

    policy = ExitPolicy(
        exit_position_matcher=FakeMatcher(),
        exit_position_snapshot=FakeSnapshot(),
    )
    open_positions = [{"symbol": "BTC/USDT"}]

    result = await policy.evaluate(
        _decision(Action.CLOSE_LONG),
        "ensemble_trader",
        open_positions,
    )

    assert open_positions == []
    assert result.passed is False
    assert result.blocker == "no_matching_exit_position"
    assert calls == [("refresh", "1"), ("ensemble_trader", "BTC/USDT")]


@pytest.mark.asyncio
async def test_execution_service_serializes_candidate_execution():
    lock = asyncio.Lock()
    calls: list[tuple[Any, ...]] = []
    stages: list[tuple[Any, ...]] = []
    raw_updates: list[dict[str, Any] | None] = []

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

    async def log_risk_event(*args, **_kwargs):
        calls.append(("risk", *args))

    def get_model_execution_mode(model_name):
        calls.append(("mode", model_name))
        return "paper"

    async def record_decision_stage(
        decision_db_id,
        decision,
        stage,
        status,
        reason,
        data=None,
    ):
        assert lock.locked()
        stages.append((stage, status, reason))
        return decision.raw_response

    async def mark_decision_reason(decision_db_id, reason):
        calls.append(("reason", decision_db_id, reason))

    async def mark_decision_raw_response(decision_db_id, raw_response):
        calls.append(("raw", decision_db_id))
        raw_updates.append(raw_response)

    async def log_position_review_risk_result(*args, **kwargs):
        calls.append(("position_review_risk", args, kwargs))

    async def duplicate_decision_order_reason(decision_db_id, decision):
        calls.append(("duplicate", decision_db_id, decision.symbol))
        return None

    async def get_okx_executor(mode):
        calls.append(("executor", mode))
        return FakeExecutor()

    async def allocated_order_balance(model_mode, decision):
        calls.append(("balance", model_mode, decision.symbol))
        return 123.0

    def rejected_execution_result(decision, reason):
        calls.append(("rejected", decision.symbol, reason))
        return ExecutionResult(
            order_id=None,
            exchange_order_id=None,
            symbol=decision.symbol,
            side=decision.action.value,
            order_type="market",
            quantity=0.0,
            price=0.0,
            status=OrderStatus.REJECTED,
            raw_response={"reason": reason},
        )

    def attach_leverage_summary(decision, execution_result, ai_requested_leverage):
        calls.append(("leverage", ai_requested_leverage))

    def execution_reason(execution_result):
        return execution_result.status.value if execution_result else "missing"

    async def mark_pending(decision_db_id, reason):
        calls.append(("pending", decision_db_id, reason))

    def is_untradable_exchange_error(text):
        calls.append(("untradable_check", bool(text)))
        return False

    def remember_untradable_symbol(symbol, text):
        calls.append(("remember_untradable", symbol, text))

    def is_transient_entry_exchange_error(text):
        calls.append(("transient_check", bool(text)))
        return False

    def remember_temporary_entry_block(symbol, reason, minutes):
        calls.append(("temporary_block", symbol, reason, minutes))

    def transient_entry_block_minutes(text):
        calls.append(("transient_minutes", text))
        return 5.0

    async def log_trade(execution_result, model_name, decision, decision_db_id):
        calls.append(("log_trade", execution_result.order_id, model_name, decision_db_id))

    def is_exchange_confirmed_execution(execution_result):
        status = execution_result.status if execution_result is not None else None
        calls.append(("confirmed", status.value if status else None))
        return bool(execution_result and execution_result.status == OrderStatus.FILLED)

    def is_exit_progress_execution(execution_result):
        status = execution_result.status if execution_result is not None else None
        calls.append(("exit_progress", status.value if status else None))
        return False

    def result_has_no_exchange_position(execution_result):
        calls.append(("no_position", execution_result.status.value))
        return False

    def increment_trade_count():
        calls.append(("increment_trade_count",))

    async def persist_position_from_execution(
        model_name,
        decision,
        execution_result,
        model_mode,
    ):
        calls.append(("persist_position", model_name, model_mode))

    def apply_execution_to_open_positions(
        open_positions,
        model_name,
        decision,
        execution_result,
    ):
        calls.append(("apply_open_positions", len(open_positions)))

    async def mark_decision_executed(decision_db_id, price):
        calls.append(("executed", decision_db_id, price))

    def clear_market_no_opportunity_symbol(symbol):
        calls.append(("clear_symbol", symbol))

    async def persist_account_update(model_name, decision_model_name, execution_result):
        calls.append(("account_update", model_name, decision_model_name))

    async def get_account_balance(model_name):
        calls.append(("account_balance", model_name))
        return 1000.0

    async def mark_decision_outcome(decision_db_id, outcome, pnl_pct):
        calls.append(("outcome", decision_db_id, outcome, pnl_pct))

    async def evaluate_entry_policy(decision, model_name, model_mode, open_positions):
        calls.append(("entry_policy", model_name, model_mode, len(open_positions or [])))
        decision.position_size_pct = 0.004
        decision.suggested_leverage = 2.0
        return PolicyGateResult.allow({"intent": "entry"})

    async def evaluate_exit_policy(
        decision,
        model_name,
        open_positions,
        *,
        refresh_positions=True,
    ):
        calls.append(("exit_policy", model_name, len(open_positions or []), refresh_positions))
        return PolicyGateResult.allow({"intent": "exit"})

    def execution_skills(**kwargs):
        calls.append(
            (
                "execution_skills",
                kwargs["model_mode"],
                kwargs["override_balance"],
            )
        )
        return []

    def attach_execution_skills(*args, **kwargs):
        calls.append(("attach_skills", args, kwargs))

    def execution_skill_block_reason(skills, *, for_entry):
        calls.append(("skill_block", len(skills), for_entry))
        return None

    async def reconcile_positions(reason):
        calls.append(("reconcile", reason))

    async def open_positions_context():
        calls.append(("open_positions_context",))
        return []

    def has_matching_local_exit_position(positions, model_name, decision):
        calls.append(("local_exit_position", model_name, decision.symbol, len(positions)))
        return True

    async def has_matching_exchange_exit_position(model_name, decision):
        calls.append(("exchange_exit_position", model_name, decision.symbol))
        return True

    def remember_exit_cooldown(model_name, decision):
        calls.append(("exit_cooldown", model_name, decision.symbol))

    def record_trade_notional(amount):
        calls.append(("record_trade", amount))

    service = ExecutionService(
        execution_lock=lock,
        risk_event_logger=log_risk_event,
        model_execution_mode_provider=get_model_execution_mode,
        decision_stage_recorder=record_decision_stage,
        decision_reason_marker=mark_decision_reason,
        decision_raw_response_marker=mark_decision_raw_response,
        position_review_alert_context_provider=lambda _decision_arg: None,
        position_review_risk_result_logger=log_position_review_risk_result,
        duplicate_decision_order_reason_provider=duplicate_decision_order_reason,
        okx_executor_provider=get_okx_executor,
        allocated_order_balance_provider=allocated_order_balance,
        rejected_execution_result_factory=rejected_execution_result,
        execution_leverage_summary_attacher=attach_leverage_summary,
        execution_reason_provider=execution_reason,
        pending_execution_marker=mark_pending,
        untradable_exchange_error_checker=is_untradable_exchange_error,
        untradable_symbol_rememberer=remember_untradable_symbol,
        transient_entry_exchange_error_checker=is_transient_entry_exchange_error,
        temporary_entry_block_rememberer=remember_temporary_entry_block,
        transient_entry_block_minutes_provider=transient_entry_block_minutes,
        trade_logger=log_trade,
        exchange_confirmed_checker=is_exchange_confirmed_execution,
        exit_progress_checker=is_exit_progress_execution,
        no_exchange_position_result_checker=result_has_no_exchange_position,
        trade_count_incrementer=increment_trade_count,
        position_execution_persister=persist_position_from_execution,
        open_positions_execution_applier=apply_execution_to_open_positions,
        decision_executed_marker=mark_decision_executed,
        market_no_opportunity_symbol_clearer=clear_market_no_opportunity_symbol,
        account_update_persister=persist_account_update,
        account_balance_provider=get_account_balance,
        decision_outcome_marker=mark_decision_outcome,
        entry_policy_evaluator=evaluate_entry_policy,
        exit_policy_evaluator=evaluate_exit_policy,
        execution_skills_provider=execution_skills,
        execution_skills_attacher=attach_execution_skills,
        execution_skills_block_reason_provider=execution_skill_block_reason,
        position_reconciler=reconcile_positions,
        open_positions_context_provider=open_positions_context,
        matching_exit_local_position_checker=has_matching_local_exit_position,
        matching_exit_exchange_position_checker=has_matching_exchange_exit_position,
        exit_cooldown_recorder=remember_exit_cooldown,
        trade_notional_recorder=record_trade_notional,
    )
    results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}
    result = await service.execute_candidate(
        "BTC/USDT",
        "ensemble_trader",
        _decision(Action.LONG),
        SimpleNamespace(warnings=[]),
        123,
        results,
        open_positions=[],
    )

    assert result is not None
    assert result.order_id == "order-1"
    assert ("mode", "ensemble_trader") in calls
    assert ("duplicate", 123, "BTC/USDT") in calls
    assert ("entry_policy", "ensemble_trader", "paper", 0) in calls
    assert ("executor", "paper") in calls
    assert ("balance", "paper", "BTC/USDT") in calls
    assert ("execution_skills", "paper", 123.0) in calls
    assert ("skill_block", 0, True) in calls
    assert ("place_order", "ensemble_trader", "long", 123.0) in calls
    assert ("increment_trade_count",) in calls
    assert ("persist_position", "ensemble_trader", "paper") in calls
    assert ("executed", 123, 100.0) in calls
    assert ("clear_symbol", "BTC/USDT") in calls
    assert ("record_trade", 200.0) in calls
    assert raw_updates[-1] is not None
    assert raw_updates[-1]["execution_parameters"]["position_size_pct"] == 0.004
    assert raw_updates[-1]["execution_parameters"]["suggested_leverage"] == 2.0
    assert results["executions"][0]["order_id"] == "order-1"
    assert results["decisions"][0]["executed"] is True
    assert [stage for stage, _status, _reason in stages] == [
        "strategy_arbitration",
        "risk_check",
        "risk_check",
        "exchange_submit",
        "exchange_submit",
        "exchange_confirm",
        "local_sync",
    ]

    calls.clear()
    stages.clear()
    exit_results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}
    exit_result = await service.execute_candidate(
        "BTC/USDT",
        "ensemble_trader",
        _decision(Action.CLOSE_LONG),
        SimpleNamespace(warnings=[]),
        124,
        exit_results,
        open_positions=[{"symbol": "BTC/USDT", "side": "long"}],
        refresh_exit_positions=False,
    )

    assert exit_result is not None
    assert ("exit_policy", "ensemble_trader", 1, False) in calls
    assert ("exit_cooldown", "ensemble_trader", "BTC/USDT") in calls
    assert exit_results["executions"][0]["order_id"] == "order-1"

    calls.clear()
    stages.clear()
    blocked_reason = "动态证据仍处于弱证据学习档，本轮只记录影子样本。"

    async def evaluate_entry_policy_blocked(decision, model_name, model_mode, open_positions):
        calls.append(("entry_policy_blocked", model_name, model_mode, len(open_positions or [])))
        return PolicyGateResult.block(
            "entry_evidence_shadow_only",
            blocked_reason,
            {
                "stage_status": "skipped",
                "skip_kind": "entry_evidence_shadow_only",
                "shadow_only": True,
            },
        )

    service.entry_policy_evaluator = evaluate_entry_policy_blocked
    blocked_results: dict[str, Any] = {"warnings": [], "decisions": [], "executions": []}
    blocked_result = await service.execute_candidate(
        "BTC/USDT",
        "ensemble_trader",
        _decision(Action.LONG),
        SimpleNamespace(warnings=[]),
        125,
        blocked_results,
        open_positions=[],
    )

    assert blocked_result is not None
    assert blocked_result.status == OrderStatus.REJECTED
    assert blocked_result.raw_response["execution_skipped"] is True
    assert blocked_result.raw_response["skip_kind"] == "entry_evidence_shadow_only"
    assert blocked_result.raw_response["opportunity_score"]["selected_for_execution"] is False
    assert blocked_result.raw_response["opportunity_score"]["selection_reason"] == blocked_reason
    assert blocked_result.raw_response["opportunity_score"]["execution_final_state"] == "skipped"
    assert blocked_results["decisions"][0]["execution_status"] == "skipped"
    assert ("entry_policy_blocked", "ensemble_trader", "paper", 0) in calls
    assert not any(call[0] in {"executor", "place_order"} for call in calls)
    assert ("risk_check", "skipped", blocked_reason) in stages


@pytest.mark.asyncio
async def test_execution_service_fails_fast_without_execution_lock_dependency():
    service = ExecutionService()

    with pytest.raises(RuntimeError, match="execution_lock"):
        await service.execute_candidate(
            "BTC/USDT",
            "ensemble_trader",
            _decision(Action.LONG),
            SimpleNamespace(warnings=[]),
            123,
            {"warnings": [], "decisions": [], "executions": []},
            open_positions=[],
        )


@pytest.mark.asyncio
async def test_execution_service_fails_fast_without_runtime_boundaries():
    lock = asyncio.Lock()
    service = ExecutionService(execution_lock=lock)

    with pytest.raises(RuntimeError, match="risk_event_logger"):
        await service.execute_candidate(
            "BTC/USDT",
            "ensemble_trader",
            _decision(Action.LONG),
            SimpleNamespace(warnings=[]),
            123,
            {"warnings": [], "decisions": [], "executions": []},
            open_positions=[],
        )


@pytest.mark.asyncio
async def test_entry_policy_blocks_stale_signal_before_okx_submit():
    stale_reason = "AI 信号已过有效期，等待下一轮新行情。"

    class FakeFreshness:
        def stale_decision_reason(self, decision):
            return stale_reason

    result = await EntryPolicy(decision_freshness=FakeFreshness()).evaluate(
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

    class FakeSnapshot:
        def __init__(self):
            self.reconciled = False

        async def refresh_positions(self, open_positions):
            self.reconciled = True
            if open_positions is not None:
                open_positions[:] = []
            return []

        async def has_matching_exchange_position(self, model_name, decision):
            return False

    class FakeMatcher:
        def has_matching_position(self, positions, model_name, decision):
            return False

    snapshot = FakeSnapshot()
    open_positions = [{"symbol": "BTC/USDT"}]
    result = await ExitPolicy(
        exit_position_matcher=FakeMatcher(),
        exit_position_snapshot=snapshot,
    ).evaluate(
        _decision(Action.CLOSE_LONG),
        "ensemble_trader",
        open_positions,
    )

    assert snapshot.reconciled is True
    assert open_positions == []
    assert result.passed is False
    assert result.blocker == "no_matching_exit_position"
    assert result.reason == missing_reason


@pytest.mark.asyncio
async def test_exit_policy_blocks_unknown_okx_position_snapshot_separately():
    class FakeSnapshot:
        def __init__(self):
            self.reconciled = False

        async def refresh_positions(self, open_positions):
            self.reconciled = True
            return []

        async def has_matching_exchange_position(self, model_name, decision):
            return None

    class FakeMatcher:
        def has_matching_position(self, positions, model_name, decision):
            return False

    snapshot = FakeSnapshot()
    result = await ExitPolicy(
        exit_position_matcher=FakeMatcher(),
        exit_position_snapshot=snapshot,
    ).evaluate(
        _decision(Action.CLOSE_LONG),
        "ensemble_trader",
        [],
    )

    assert snapshot.reconciled is True
    assert result.passed is False
    assert result.blocker == "exchange_position_snapshot_unavailable"
    assert result.reason is not None
    assert "OKX" in result.reason


@pytest.mark.asyncio
async def test_sync_service_reconcile_positions_owns_lock_boundary():
    lock = asyncio.Lock()
    calls: list[str] = []

    service = OkxSyncService(exchange_reconcile_lock=lock)

    async def fake_reconcile():
        assert lock.locked()
        calls.append("reconciled")
        return [{"symbol": "BTC/USDT", "side": "long"}]

    service.reconcile_exchange_positions = fake_reconcile  # type: ignore[method-assign]
    result = await service.reconcile_positions("unit test")

    assert result == [{"symbol": "BTC/USDT", "side": "long"}]
    assert calls == ["reconciled"]


def test_sync_service_does_not_keep_legacy_orchestrator_reference():
    service = OkxSyncService()

    assert not hasattr(service, "orchestrator")


@pytest.mark.asyncio
async def test_sync_service_records_reconcile_timeout_through_injected_boundary():
    lock = asyncio.Lock()
    recorded_errors: list[str] = []

    service = OkxSyncService(
        exchange_reconcile_lock=lock,
        round_error_recorder=recorded_errors.append,
    )

    async def slow_reconcile():
        await asyncio.sleep(1.0)
        return [{"symbol": "BTC/USDT"}]

    service.reconcile_exchange_positions = slow_reconcile  # type: ignore[method-assign]
    result = await service.reconcile_positions("unit test", timeout_seconds=0.01)

    assert result == []
    assert len(recorded_errors) == 1
    assert "unit test" in recorded_errors[0]
    assert "timed out" in recorded_errors[0]


@pytest.mark.asyncio
async def test_sync_service_skips_duplicate_reconcile_without_recording_error():
    lock = asyncio.Lock()
    recorded_errors: list[str] = []
    calls = 0

    service = OkxSyncService(
        exchange_reconcile_lock=lock,
        round_error_recorder=recorded_errors.append,
    )

    async def slow_reconcile():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return [{"symbol": "BTC/USDT"}]

    service.reconcile_exchange_positions = slow_reconcile  # type: ignore[method-assign]

    running = asyncio.create_task(service.reconcile_positions("market", timeout_seconds=1.0))
    await asyncio.sleep(0)
    duplicate = await service.reconcile_positions(
        "position",
        timeout_seconds=1.0,
        lock_wait_seconds=0.001,
    )
    result = await running

    assert result == [{"symbol": "BTC/USDT"}]
    assert duplicate == []
    assert calls == 1
    assert recorded_errors == []


@pytest.mark.asyncio
async def test_sync_service_fails_fast_without_reconcile_lock_dependency():
    service = OkxSyncService()

    with pytest.raises(RuntimeError, match="exchange_reconcile_lock"):
        await service.reconcile_positions("unit test")


@pytest.mark.asyncio
async def test_sync_service_reconcile_exchange_positions_fails_fast_without_boundaries():
    service = OkxSyncService()

    with pytest.raises(RuntimeError, match="symbol_normalizer"):
        await service.reconcile_exchange_positions()


@pytest.mark.asyncio
async def test_sync_service_reconcile_exchange_positions_uses_injected_paper_okx_boundary():
    service = OkxSyncService(
        symbol_normalizer=lambda symbol: symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position: bool(position),
        paper_okx_provider=lambda: None,
    )

    assert await service.reconcile_exchange_positions() == []


@pytest.mark.asyncio
async def test_sync_service_reconcile_exchange_positions_requires_protection_boundary():
    class FakePaperOKX:
        async def get_positions_strict(self):
            return []

    service = OkxSyncService(
        symbol_normalizer=lambda symbol: symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position: bool(position),
        paper_okx_provider=lambda: FakePaperOKX(),
    )

    with pytest.raises(RuntimeError, match="exchange_protection_map_provider"):
        await service.reconcile_exchange_positions()


@pytest.mark.asyncio
async def test_sync_service_reconcile_exchange_positions_requires_snapshot_syncer():
    class FakePaperOKX:
        async def get_positions_strict(self):
            return []

    async def protection_map(_paper_okx, _exchange_positions):
        return {}

    async def fallback_protection(_session, **_kwargs):
        return {}

    service = OkxSyncService(
        symbol_normalizer=lambda symbol: symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position: bool(position),
        paper_okx_provider=lambda: FakePaperOKX(),
        exchange_protection_map_provider=protection_map,
        position_protection_fallback_provider=fallback_protection,
    )

    with pytest.raises(RuntimeError, match="local_position_snapshot_syncer"):
        await service.reconcile_exchange_positions()


@pytest.mark.asyncio
async def test_sync_service_reconcile_exchange_positions_requires_datetime_parser():
    class FakePaperOKX:
        async def get_positions_strict(self):
            return []

    async def protection_map(_paper_okx, _exchange_positions):
        return {}

    async def fallback_protection(_session, **_kwargs):
        return {}

    service = OkxSyncService(
        symbol_normalizer=lambda symbol: symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position: bool(position),
        paper_okx_provider=lambda: FakePaperOKX(),
        exchange_protection_map_provider=protection_map,
        position_protection_fallback_provider=fallback_protection,
        local_position_snapshot_syncer=lambda _positions, **_kwargs: False,
    )

    with pytest.raises(RuntimeError, match="datetime_from_ms_parser"):
        await service.reconcile_exchange_positions()


@pytest.mark.asyncio
async def test_sync_service_reconcile_exchange_positions_requires_close_fill_finder():
    class FakePaperOKX:
        async def get_positions_strict(self):
            return []

    async def protection_map(_paper_okx, _exchange_positions):
        return {}

    async def fallback_protection(_session, **_kwargs):
        return {}

    service = OkxSyncService(
        symbol_normalizer=lambda symbol: symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position: bool(position),
        paper_okx_provider=lambda: FakePaperOKX(),
        exchange_protection_map_provider=protection_map,
        position_protection_fallback_provider=fallback_protection,
        local_position_snapshot_syncer=lambda _positions, **_kwargs: False,
        datetime_from_ms_parser=lambda _timestamp_ms: datetime.now(UTC),
    )

    with pytest.raises(RuntimeError, match="exchange_close_fill_finder"):
        await service.reconcile_exchange_positions()


@pytest.mark.asyncio
async def test_sync_service_reconcile_exchange_positions_uses_injected_snapshot_syncer(
    monkeypatch: pytest.MonkeyPatch,
):
    local_position = SimpleNamespace(
        id=11,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
        is_open=True,
    )
    sync_calls: list[dict[str, Any]] = []

    class FakePaperOKX:
        async def get_positions_strict(self):
            return [
                {
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "contracts": "2",
                    "contractSize": "0.5",
                    "entryPrice": "100",
                    "markPrice": "111",
                    "leverage": "3",
                    "unrealizedPnl": "11",
                }
            ]

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            return [local_position]

    class FakeAccountRepository:
        def __init__(self, _session):
            pass

    @asynccontextmanager
    async def fake_session_ctx():
        yield object()

    async def protection_map(_paper_okx, _exchange_positions):
        return {
            ("BTC/USDT", "long"): {
                "stop_loss_price": 95.0,
                "take_profit_price": 125.0,
            }
        }

    async def fallback_protection(_session, **_kwargs):
        return {}

    def sync_snapshot(positions, **kwargs):
        sync_calls.append({"positions": positions, "kwargs": kwargs})
        return True

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "AccountRepository", FakeAccountRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    result = await OkxSyncService(
        symbol_normalizer=lambda symbol: symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position: bool(position),
        paper_okx_provider=lambda: FakePaperOKX(),
        exchange_protection_map_provider=protection_map,
        position_protection_fallback_provider=fallback_protection,
        local_position_snapshot_syncer=sync_snapshot,
        datetime_from_ms_parser=lambda _timestamp_ms: datetime.now(UTC),
        **_noop_reconcile_close_boundaries(),
    ).reconcile_exchange_positions()

    assert result == [
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "side": "long",
            "quantity": 1.0,
            "current_price": 111.0,
            "note": "OKX 持仓数量或价格已变化，本地持仓快照已同步更新。",
        }
    ]
    assert sync_calls == [
        {
            "positions": [local_position],
            "kwargs": {
                "exchange_quantity": 1.0,
                "current_price": 111.0,
                "entry_price": 100.0,
                "leverage": 3.0,
                "exchange_unrealized": 11.0,
                "stop_loss_price": 95.0,
                "take_profit_price": 125.0,
            },
        }
    ]


@pytest.mark.asyncio
async def test_sync_service_reconcile_exchange_positions_uses_injected_datetime_parser(
    monkeypatch: pytest.MonkeyPatch,
):
    parsed_at = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    parser_calls: list[Any] = []
    opened_positions: list[dict[str, Any]] = []
    order = SimpleNamespace(
        model_name="ensemble_trader",
        exchange_order_id="entry-order-1",
        status=OrderStatus.OPEN.value,
        quantity=None,
        price=None,
        filled_at=None,
    )

    class FakeScalarResult:
        def __init__(self, value):
            self.value = value

        def scalar_one_or_none(self):
            return self.value

    class FakeSession:
        def __init__(self):
            self.results = [None, order]

        async def execute(self, _stmt):
            return FakeScalarResult(self.results.pop(0))

    class FakePaperOKX:
        async def get_positions_strict(self):
            return [
                {
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "contracts": "2",
                    "contractSize": "0.5",
                    "entryPrice": "100",
                    "markPrice": "111",
                    "leverage": "3",
                    "unrealizedPnl": "11",
                    "timestamp": "1770379200000",
                }
            ]

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            return []

        async def open_position(self, payload):
            opened_positions.append(payload)

    class FakeAccountRepository:
        def __init__(self, _session):
            pass

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    async def protection_map(_paper_okx, _exchange_positions):
        return {}

    async def fallback_protection(_session, **_kwargs):
        return {
            "stop_loss_price": 95.0,
            "take_profit_price": 125.0,
        }

    def datetime_from_ms(timestamp_ms):
        parser_calls.append(timestamp_ms)
        return parsed_at

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "AccountRepository", FakeAccountRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    result = await OkxSyncService(
        symbol_normalizer=lambda symbol: symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position: bool(position),
        paper_okx_provider=lambda: FakePaperOKX(),
        exchange_protection_map_provider=protection_map,
        position_protection_fallback_provider=fallback_protection,
        local_position_snapshot_syncer=lambda _positions, **_kwargs: False,
        datetime_from_ms_parser=datetime_from_ms,
        **_noop_reconcile_close_boundaries(),
    ).reconcile_exchange_positions()

    assert parser_calls == ["1770379200000"]
    assert order.status == OrderStatus.FILLED.value
    assert order.quantity == 1.0
    assert order.price == 100.0
    assert order.filled_at == parsed_at
    assert opened_positions == [
        {
            "model_name": "ensemble_trader",
            "execution_mode": "paper",
            "symbol": "BTC/USDT",
            "side": "long",
            "quantity": 1.0,
            "entry_price": 100.0,
            "current_price": 111.0,
            "leverage": 3.0,
            "unrealized_pnl": 11.0,
            "realized_pnl": 0.0,
            "stop_loss_price": 95.0,
            "take_profit_price": 125.0,
        }
    ]
    assert result == [
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 100.0,
            "exchange_order_id": "entry-order-1",
            "note": "OKX 已有持仓但本地缺失，已按执行订单补回持仓记录。",
        }
    ]


@pytest.mark.asyncio
async def test_sync_service_reconcile_exchange_positions_uses_injected_close_boundaries(
    monkeypatch: pytest.MonkeyPatch,
):
    closed_at = datetime(2026, 6, 8, 13, 0, tzinfo=UTC)
    position = SimpleNamespace(
        id=21,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
        is_open=True,
        entry_price=100.0,
        current_price=105.0,
        quantity=2.0,
        leverage=4.0,
        unrealized_pnl=10.0,
        realized_pnl=0.0,
        created_at=datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
        closed_at=None,
    )
    created_orders: list[dict[str, Any]] = []
    balance_updates: list[tuple[str, float, float]] = []
    trade_results: list[tuple[str, bool]] = []
    close_fill_calls: list[Any] = []
    entry_fee_calls: list[tuple[Any, Any, float]] = []
    decision_logs: list[dict[str, Any]] = []
    reflection_calls: list[dict[str, Any]] = []
    margin_calls: list[tuple[float, float | None]] = []
    removed_positions: list[tuple[str, str, str]] = []

    class FakeScalarResult:
        def scalar_one_or_none(self):
            return None

    class FakeSession:
        async def refresh(self, _position):
            return None

        async def execute(self, _stmt):
            return FakeScalarResult()

    class FakePaperOKX:
        async def get_positions_strict(self):
            return []

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            return [position]

        async def create_order(self, payload):
            created_orders.append(payload)

    class FakeAccountRepository:
        def __init__(self, _session):
            pass

        async def update_balance(self, model_name, amount, realized_pnl):
            balance_updates.append((model_name, amount, realized_pnl))

        async def record_trade_result(self, model_name, is_win):
            trade_results.append((model_name, is_win))

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    async def protection_map(_paper_okx, _exchange_positions):
        return {}

    async def fallback_protection(_session, **_kwargs):
        return {}

    async def close_fill(pos):
        close_fill_calls.append(pos)
        return {
            "order_id": "close-order-1",
            "price": 112.0,
            "fee": 0.5,
            "timestamp": closed_at,
        }

    async def entry_fee(session, pos, close_qty):
        entry_fee_calls.append((session, pos, close_qty))
        return 1.5

    async def log_close_decision(**kwargs):
        decision_logs.append(kwargs)
        return 42

    async def record_reflection(_session, pos, **kwargs):
        reflection_calls.append({"pos": pos, "kwargs": kwargs})

    def position_margin(notional, leverage):
        margin_calls.append((notional, leverage))
        return 50.0

    def remove_memory_position(model_name, symbol, side):
        removed_positions.append((model_name, symbol, side))

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "AccountRepository", FakeAccountRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    result = await OkxSyncService(
        symbol_normalizer=lambda symbol: symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position_payload: bool(position_payload),
        paper_okx_provider=lambda: FakePaperOKX(),
        exchange_protection_map_provider=protection_map,
        position_protection_fallback_provider=fallback_protection,
        local_position_snapshot_syncer=lambda _positions, **_kwargs: False,
        datetime_from_ms_parser=lambda _timestamp_ms: datetime.now(UTC),
        exchange_close_fill_finder=close_fill,
        fresh_feature_vector_provider=lambda _symbol: None,
        market_value_reader=lambda source, key: getattr(source, key, None),
        entry_fee_provider=entry_fee,
        exchange_sync_close_decision_logger=log_close_decision,
        trade_reflection_recorder=record_reflection,
        position_margin_calculator=position_margin,
        memory_position_remover=remove_memory_position,
    ).reconcile_exchange_positions()

    assert close_fill_calls == [position]
    assert entry_fee_calls[0][1:] == (position, 2.0)
    assert decision_logs[0]["exit_price"] == 112.0
    assert decision_logs[0]["realized_pnl"] == 22.0
    assert reflection_calls[0]["kwargs"]["source"] == "okx_reconcile"
    assert margin_calls == [(200.0, 4.0)]
    assert balance_updates == [("ensemble_trader", 72.0, 22.0)]
    assert trade_results == [("ensemble_trader", True)]
    assert removed_positions == [("ensemble_trader", "BTC/USDT", "long")]
    assert position.is_open is False
    assert position.current_price == 112.0
    assert position.realized_pnl == 22.0
    assert created_orders[0]["decision_id"] == 42
    assert created_orders[0]["fee"] == 0.5
    assert result == [
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "side": "long",
            "exit_price": 112.0,
            "realized_pnl": 22.0,
            "gross_pnl": 24.0,
            "fees": 2.0,
            "exchange_order_id": "close-order-1",
        }
    ]


@pytest.mark.asyncio
async def test_sync_service_reconcile_exchange_positions_uses_injected_price_recheck(
    monkeypatch: pytest.MonkeyPatch,
):
    position = SimpleNamespace(
        id=22,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
        is_open=True,
        entry_price=100.0,
        current_price=104.0,
        quantity=2.0,
        leverage=4.0,
        unrealized_pnl=8.0,
        realized_pnl=0.0,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
        closed_at=None,
    )
    created_orders: list[dict[str, Any]] = []
    fresh_calls: list[str] = []
    market_value_calls: list[str] = []
    decision_logs: list[dict[str, Any]] = []

    class FakeSession:
        async def refresh(self, _position):
            return None

    class FakePaperOKX:
        async def get_positions_strict(self):
            return []

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            return [position]

        async def create_order(self, payload):
            created_orders.append(payload)

    class FakeAccountRepository:
        def __init__(self, _session):
            pass

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    async def protection_map(_paper_okx, _exchange_positions):
        return {}

    async def fallback_protection(_session, **_kwargs):
        return {}

    async def fresh_feature_vector(symbol):
        fresh_calls.append(symbol)
        return SimpleNamespace(current_price=115.0)

    def market_value(source, key):
        market_value_calls.append(key)
        return getattr(source, key, None)

    async def log_close_decision(**kwargs):
        decision_logs.append(kwargs)
        return 77

    async def no_close_fill(_position):
        return {}

    async def entry_fee(_session, _pos, _close_qty):
        return 1.0

    async def record_reflection(*_args, **_kwargs):
        return None

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "AccountRepository", FakeAccountRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    service = OkxSyncService(
        symbol_normalizer=lambda symbol: symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position_payload: bool(position_payload),
        paper_okx_provider=lambda: FakePaperOKX(),
        exchange_protection_map_provider=protection_map,
        position_protection_fallback_provider=fallback_protection,
        local_position_snapshot_syncer=lambda _positions, **_kwargs: False,
        datetime_from_ms_parser=lambda _timestamp_ms: datetime.now(UTC),
        exchange_close_fill_finder=no_close_fill,
        fresh_feature_vector_provider=fresh_feature_vector,
        market_value_reader=market_value,
        entry_fee_provider=entry_fee,
        exchange_sync_close_decision_logger=log_close_decision,
        trade_reflection_recorder=record_reflection,
        position_margin_calculator=lambda _notional, _leverage: 0.0,
        memory_position_remover=lambda _model_name, _symbol, _side: None,
    )

    async def no_active_order(_position):
        return None

    service.active_exchange_order_for_local_position = no_active_order  # type: ignore[method-assign]
    result = await service.reconcile_exchange_positions()

    assert fresh_calls == ["BTC/USDT"]
    assert market_value_calls == ["current_price"]
    assert decision_logs[0]["exit_price"] == 115.0
    assert decision_logs[0]["realized_pnl"] == 29.0
    assert created_orders[0]["decision_id"] == 77
    assert position.is_open is False
    assert position.current_price == 115.0
    assert position.realized_pnl == 29.0
    assert result[0]["exit_price"] == 115.0
    assert result[0]["realized_pnl"] == 29.0
    assert result[0]["exchange_order_id"] is None


@pytest.mark.asyncio
async def test_refresh_position_prices_fails_fast_without_peak_boundaries():
    service = OkxSyncService()

    with pytest.raises(RuntimeError, match="position_profit_peak_recorder"):
        await service.refresh_position_prices({})


@pytest.mark.asyncio
async def test_refresh_position_prices_uses_injected_profit_peak_boundaries(
    monkeypatch: pytest.MonkeyPatch,
):
    updated_prices: list[tuple[int, float, float]] = []
    account_updates: list[tuple[str, float]] = []
    peak_calls: list[dict[str, Any]] = []
    pruned_contexts: list[list[dict[str, Any]]] = []

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            return [
                SimpleNamespace(
                    id=7,
                    model_name="ensemble_trader",
                    symbol="BTC/USDT",
                    side="long",
                    entry_price=100.0,
                    current_price=100.0,
                    quantity=0.5,
                    created_at="created",
                )
            ]

        async def update_position_price(self, position_id, current_price, unrealized_pnl):
            updated_prices.append((position_id, current_price, unrealized_pnl))

    class FakeAccountRepository:
        def __init__(self, _session):
            pass

        async def update_unrealized_pnl(self, model_name, unrealized_pnl):
            account_updates.append((model_name, unrealized_pnl))

    @asynccontextmanager
    async def fake_session_ctx():
        yield object()

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "AccountRepository", FakeAccountRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    service = OkxSyncService(
        position_profit_peak_recorder=lambda **kwargs: peak_calls.append(kwargs),
        position_age_minutes_provider=lambda created_at: 12.5 if created_at else None,
        position_profit_peak_pruner=lambda open_context: pruned_contexts.append(open_context),
    )

    await service.refresh_position_prices({"BTC/USDT": SimpleNamespace(current_price=110.0)})

    assert updated_prices == [(7, 110.0, 5.0)]
    assert account_updates == [("ensemble_trader", 5.0)]
    assert peak_calls[0]["symbol"] == "BTC/USDT"
    assert peak_calls[0]["unrealized_pnl"] == 5.0
    assert peak_calls[0]["hold_minutes"] == 12.5
    assert pruned_contexts[0][0]["symbol"] == "BTC/USDT"


@pytest.mark.asyncio
async def test_sync_service_treats_open_order_lookup_failure_as_unknown():
    token = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"
    hidden_value = "plain-credential-value"
    error_text = f"Authorization: Bearer {token} failed password={hidden_value}"
    calls: list[str] = []

    class FakeExecutor:
        async def get_open_orders_strict(self, symbol):
            calls.append(symbol)
            raise RuntimeError(error_text)

    async def okx_executor(mode):
        assert mode == "paper"
        return FakeExecutor()

    service = OkxSyncService(
        symbol_normalizer=lambda symbol: symbol,
        okx_executor_provider=okx_executor,
    )
    position = SimpleNamespace(
        id=1,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
    )

    result = await service.active_exchange_order_for_local_position(position)

    assert calls == ["BTC/USDT"]
    assert result is not None
    assert result["kind"] == OPEN_ORDER_SNAPSHOT_UNKNOWN_KIND
    assert result["state"] == "unavailable"
    rendered = str(result)
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in result["error"]
    assert "password=***" in result["error"]


@pytest.mark.asyncio
async def test_sync_service_open_order_lookup_fails_fast_without_boundaries():
    service = OkxSyncService()
    position = SimpleNamespace(
        id=1,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
    )

    with pytest.raises(RuntimeError, match="symbol_normalizer"):
        await service.active_exchange_order_for_local_position(position)


@pytest.mark.asyncio
async def test_sync_service_exit_position_lookup_failure_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
):
    token = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"
    hidden_value = "plain-credential-value"
    error_text = f"Authorization: Bearer {token} failed password={hidden_value}"
    warnings: list[dict[str, Any]] = []

    class FakeLogger:
        def warning(self, _message, **kwargs):
            warnings.append(kwargs)

    class FakeExecutor:
        async def get_positions_strict(self, symbol):
            assert symbol == "BTC/USDT"
            raise RuntimeError(error_text)

    def model_execution_mode(model_name):
        assert model_name == "ensemble_trader"
        return "paper"

    async def okx_executor(mode):
        assert mode == "paper"
        return FakeExecutor()

    def parse_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    monkeypatch.setattr(sync_module, "logger", FakeLogger())

    result = await OkxSyncService(
        symbol_normalizer=lambda symbol: symbol,
        model_execution_mode_provider=model_execution_mode,
        okx_executor_provider=okx_executor,
        float_parser=parse_float,
    ).has_matching_exchange_exit_position(
        "ensemble_trader",
        _decision(Action.CLOSE_LONG),
    )

    assert result is None
    rendered = str(warnings)
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in rendered
    assert "password=***" in rendered


@pytest.mark.asyncio
async def test_sync_service_exit_position_lookup_fails_fast_without_boundaries():
    service = OkxSyncService()

    with pytest.raises(RuntimeError, match="symbol_normalizer"):
        await service.has_matching_exchange_exit_position(
            "ensemble_trader",
            _decision(Action.CLOSE_LONG),
        )


@pytest.mark.asyncio
async def test_open_positions_context_keeps_local_positions_when_okx_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    token = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"
    hidden_value = "plain-credential-value"
    error_text = f"Authorization: Bearer {token} failed password={hidden_value}"
    warnings: list[dict[str, Any]] = []

    class FakeLogger:
        def warning(self, _message, **kwargs):
            warnings.append(kwargs)

    class FakeOKX:
        async def get_positions_strict(self):
            raise RuntimeError(error_text)

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_position_records(self, **_kwargs):
            return [
                SimpleNamespace(
                    model_name="ensemble_trader",
                    symbol="BTC/USDT",
                    side="long",
                    entry_price=100.0,
                    current_price=101.0,
                    quantity=0.2,
                    leverage=3.0,
                    unrealized_pnl=0.2,
                    stop_loss_price=95.0,
                    take_profit_price=110.0,
                    is_open=True,
                    created_at=None,
                )
            ]

    @asynccontextmanager
    async def fake_session_ctx():
        yield object()

    async def paper_positions():
        return []

    def parse_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)
    monkeypatch.setattr(sync_module, "logger", FakeLogger())

    result = await OkxSyncService(
        symbol_normalizer=lambda symbol: symbol,
        float_parser=parse_float,
        paper_positions_provider=paper_positions,
        active_okx_provider=lambda: FakeOKX(),
        exchange_position_open_checker=lambda position: bool(position),
    ).get_open_positions_context()

    assert len(result) == 1
    assert result[0]["symbol"] == "BTC/USDT"
    assert result[0]["side"] == "long"
    rendered = str(warnings)
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in rendered
    assert "password=***" in rendered


@pytest.mark.asyncio
async def test_open_positions_context_fails_fast_without_boundaries():
    service = OkxSyncService()

    with pytest.raises(RuntimeError, match="symbol_normalizer"):
        await service.get_open_positions_context()


@pytest.mark.asyncio
async def test_position_review_service_runs_sl_tp_and_executes_review_candidates():
    decision = _decision(Action.CLOSE_LONG)
    assessment = SimpleNamespace(warnings=[])
    claimed: list[str] = []
    round_ids: set[int] = set()
    executions: list[tuple[Any, ...]] = []
    stages: list[str] = []

    def set_loop_stage(stage):
        stages.append(stage)

    async def enforce_sl_tp(feature_vectors):
        return [
            {
                "model_name": "ensemble_trader",
                "symbol": "ETH/USDT",
                "trigger": "take_profit",
                "quantity": 1.2,
                "exit_price": 2000.0,
                "status": "filled",
            }
        ]

    async def open_positions_context():
        return [{"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"}]

    async def review_open_positions(
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

    async def claim_symbol(symbol, owner):
        assert owner == "position"
        return True

    async def execute_candidate(
        symbol,
        model_name,
        decision_arg,
        assessment_arg,
        decision_db_id,
        results,
        *,
        open_positions=None,
    ):
        executions.append(
            (symbol, model_name, decision_arg.action.value, decision_db_id, open_positions)
        )

    service = PositionReviewService(
        loop_stage_setter=set_loop_stage,
        sl_tp_enforcer=enforce_sl_tp,
        open_positions_context_provider=open_positions_context,
        position_reviewer=review_open_positions,
        analysis_symbol_claimer=claim_symbol,
        symbol_normalizer=lambda symbol: symbol,
        candidate_executor=execute_candidate,
    )
    results: dict[str, Any] = {"executions": []}
    open_positions, blocked = await service.review_open_positions(
        feature_vectors={"BTC/USDT": object()},
        results=results,
        round_decision_ids=round_ids,
        open_positions=[],
        position_entry_pause_reason=None,
        max_groups_override=3,
        claimed_analysis_symbols=claimed,
    )

    assert stages == ["enforce_sl_tp", "review_open_positions"]
    assert results["executions"][0]["action"] == "auto_close_take_profit"
    assert open_positions == [
        {"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"}
    ]
    assert blocked == {("ensemble_trader", "BTC/USDT")}
    assert claimed == ["BTC/USDT"]
    assert round_ids == {456}
    assert executions == [("BTC/USDT", "ensemble_trader", "close_long", 456, open_positions)]


@pytest.mark.asyncio
async def test_position_review_service_times_out_slow_review_without_stalling_round():
    stages: list[str] = []

    def set_loop_stage(stage):
        stages.append(stage)

    async def enforce_sl_tp(feature_vectors):
        return []

    async def open_positions_context():
        return [{"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"}]

    async def review_open_positions(
        open_positions,
        feature_vectors,
        *,
        results,
        round_decision_ids,
        position_entry_pause_reason,
        max_groups_override,
    ):
        await asyncio.sleep(10)
        return [], set()

    service = PositionReviewService(
        loop_stage_setter=set_loop_stage,
        sl_tp_enforcer=enforce_sl_tp,
        open_positions_context_provider=open_positions_context,
        position_reviewer=review_open_positions,
        analysis_symbol_claimer=lambda _symbol, _owner: asyncio.sleep(0, result=True),
        symbol_normalizer=lambda symbol: symbol,
        candidate_executor=lambda *args, **kwargs: asyncio.sleep(0),
        timeout_provider=lambda: 0.01,
    )
    results: dict[str, Any] = {"executions": [], "warnings": []}
    open_positions, blocked = await service.review_open_positions(
        feature_vectors={"BTC/USDT": object()},
        results=results,
        round_decision_ids=set(),
        open_positions=[],
        position_entry_pause_reason=None,
        max_groups_override=3,
        claimed_analysis_symbols=[],
    )

    assert stages == ["enforce_sl_tp", "review_open_positions"]
    assert open_positions == [
        {"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"}
    ]
    assert blocked == set()
    assert results["position_review_diagnostics"][0]["stage"] == "review_positions"
    assert "持仓复盘阶段超时" in results["warnings"][0]["warning"]


@pytest.mark.asyncio
async def test_position_review_service_fails_fast_without_boundaries():
    service = PositionReviewService()

    with pytest.raises(RuntimeError, match="loop_stage_setter"):
        await service.review_open_positions(
            feature_vectors={},
            results={"executions": []},
            round_decision_ids=set(),
            open_positions=[],
            position_entry_pause_reason=None,
            max_groups_override=1,
            claimed_analysis_symbols=[],
        )
