import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

import services.sync_service as sync_module
import services.trading_service as trading_service
from ai_brain.base_model import Action, DecisionOutput
from core.symbols import normalize_trading_symbol
from core.trading_mode import mode_manager
from executor.base_executor import ExecutionResult, OrderStatus
from risk_manager.engine import RiskEngine
from services.account_accounting_service import AccountAccountingService
from services.analysis_services import MarketAnalysisService, PositionReviewService
from services.decision_final_state_ensurer import DecisionFinalStateEnsurer
from services.decision_state import DecisionStage, DecisionStageStatus, append_decision_stage
from services.entry_fee_provider import EntryFeeProvider
from services.entry_market_data_quality import EntryMarketDataQualityPolicy, MarketValueReader
from services.entry_opportunity_score import EntryOpportunityScorePolicy
from services.entry_profit_risk_sizing import EntryProfitRiskSizingPolicy
from services.exchange_backed_position_provider import ExchangeBackedPositionProvider
from services.exchange_close_fill_finder import ExchangeCloseFillFinder
from services.exchange_position_state import (
    ExchangePositionStatePolicy,
    ExchangeProtectionMapProvider,
)
from services.execution_allocation_service import ExecutionAllocationService
from services.execution_service import ExecutionService
from services.expert_memory_service import ExpertMemoryService
from services.memory_position_store import MemoryPositionStore
from services.ml_signal_service import MLSignalService
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
from services.training_data_quality import DATA_QUALITY_VERSION


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
async def test_local_ml_signal_context_offloads_and_serializes_predictions() -> None:
    service = TradingService.__new__(TradingService)
    service._local_ml_inference_lock = asyncio.Lock()
    main_thread_id = threading.get_ident()
    worker_thread_ids: list[int] = []
    active_predictions = 0
    max_active_predictions = 0

    class Predictor:
        def predict(self, features: dict[str, str]) -> dict[str, str]:
            nonlocal active_predictions, max_active_predictions
            worker_thread_ids.append(threading.get_ident())
            active_predictions += 1
            max_active_predictions = max(max_active_predictions, active_predictions)
            time.sleep(0.02)
            active_predictions -= 1
            return features

    service.ml_signal_service = Predictor()

    first, second = await asyncio.gather(
        service._local_ml_signal_context({"symbol": "BTC/USDT"}),
        service._local_ml_signal_context({"symbol": "ETH/USDT"}),
    )

    assert first == {"symbol": "BTC/USDT"}
    assert second == {"symbol": "ETH/USDT"}
    assert max_active_predictions == 1
    assert worker_thread_ids
    assert all(thread_id != main_thread_id for thread_id in worker_thread_ids)


@pytest.mark.asyncio
async def test_round_unresolved_decision_finalizer_fills_reason_and_terminal_state() -> None:
    service = TradingService.__new__(TradingService)
    calls: list[tuple[str, Any]] = []

    class _DecisionPersistence:
        async def fill_missing_reasons(self, decision_ids, reason):
            calls.append(("fill", sorted(decision_ids), reason))

        async def finalize_unresolved_decisions(self, decisions, reason):
            calls.append(("finalize", sorted(decisions), reason))

    service.decision_persistence = _DecisionPersistence()
    decision = _decision(Action.LONG)

    await service._finalize_round_unresolved_decisions({7}, {7: decision}, "轮次被取消")

    assert calls == [("finalize", [7], "轮次被取消")]


@pytest.mark.asyncio
async def test_round_unresolved_finalizer_ignores_hold_decisions() -> None:
    service = TradingService.__new__(TradingService)
    calls: list[tuple[str, Any]] = []

    class _DecisionPersistence:
        async def fill_missing_reasons(self, decision_ids, reason):
            calls.append(("fill", sorted(decision_ids), reason))

        async def finalize_unresolved_decisions(self, decisions, reason):
            calls.append(("finalize", sorted(decisions), reason))

    service.decision_persistence = _DecisionPersistence()

    await service._finalize_round_unresolved_decisions(
        {7, 8},
        {7: _decision(Action.HOLD), 8: _decision(Action.SHORT)},
        "round ended",
    )

    assert calls == [("finalize", [8], "round ended")]


@pytest.mark.asyncio
async def test_round_unresolved_finalizer_preserves_existing_terminal_entry_reason() -> None:
    service = TradingService.__new__(TradingService)
    calls: list[tuple[str, Any]] = []

    class _DecisionPersistence:
        async def finalize_unresolved_decisions(self, decisions, reason):
            calls.append(("finalize", sorted(decisions), reason))

    service.decision_persistence = _DecisionPersistence()
    decision = _decision(Action.LONG)
    decision.raw_response = append_decision_stage(
        decision.raw_response,
        DecisionStage.STRATEGY_ARBITRATION,
        DecisionStageStatus.SKIPPED,
        "该币种正由另一条分析流程处理，本轮跳过重复开仓。",
        {"skip_kind": "analysis_symbol_claimed"},
    )

    await service._finalize_round_unresolved_decisions({7}, {7: decision}, "round ended")

    assert calls == []


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
    service._okx_authoritative_sync_task = None
    service._okx_authoritative_sync_started_at = None
    service._okx_authoritative_sync_last_success_at = None
    service._okx_authoritative_sync_last_failure_at = None
    service._okx_authoritative_sync_last_error = None
    service._okx_authoritative_sync_last_duration_seconds = None
    service._okx_authoritative_sync_last_result_count = None
    service._okx_authoritative_sync_last_result_kinds = {}
    service._okx_authoritative_sync_last_requires_attention_count = 0
    service._okx_authoritative_sync_last_samples = []
    service._okx_authoritative_sync_success_count = 0
    service._okx_authoritative_sync_failure_count = 0
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
    assert payload["market_stage_durations"]
    assert payload["market_stage_durations"][-1]["stage"] == "starting"
    assert payload["position_round_active"] is False
    assert payload["position_current_stage"] == "idle"
    assert payload["position_stage_durations"]
    assert payload["round_active"] is True
    assert payload["current_stage"] == "fetch_features"
    assert payload["okx_authoritative_sync"]["status"] == "pending"
    assert payload["okx_authoritative_sync"]["source"] == "okx_private_api_current_positions"
    assert payload["okx_authoritative_sync"]["last_result_kinds"] == {}
    assert payload["okx_authoritative_sync"]["last_requires_attention_count"] == 0
    assert payload["okx_authoritative_sync"]["last_samples"] == []
    assert TradingService._is_policy_skipped_execution_result(None) is False


@pytest.mark.asyncio
async def test_okx_balance_snapshot_returns_stale_cache_and_refreshes_in_background() -> None:
    service = TradingService.__new__(TradingService)
    service._okx_balance_snapshot_cache = {
        "paper": {
            "snapshot": {"free": 123.0, "equity": 456.0},
            "fetched_at": datetime.now(UTC)
            - timedelta(seconds=trading_service.OKX_BALANCE_SNAPSHOT_FRESH_SECONDS + 1),
        }
    }
    service._okx_balance_snapshot_locks = {}
    service._okx_balance_snapshot_refresh_tasks = {}
    refresh_calls: list[str] = []

    async def fake_refresh(mode: str) -> None:
        refresh_calls.append(mode)
        await asyncio.sleep(0)

    service._refresh_okx_balance_snapshot_for_mode = fake_refresh  # type: ignore[method-assign]

    snapshot = await service._get_okx_balance_snapshot_for_mode("paper")

    assert snapshot is not None
    assert snapshot["free"] == 123.0
    assert snapshot["stale"] is True
    assert snapshot["refresh_in_background"] is True
    task = service._okx_balance_snapshot_refresh_tasks["paper"]
    await task
    assert refresh_calls == ["paper"]


@pytest.mark.asyncio
async def test_stop_writes_inactive_runtime_heartbeat(
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
    service._position_analysis_task = None
    service._market_analysis_task = None
    service._runtime_heartbeat_task = None
    service._okx_authoritative_sync_task = None
    service._ml_auto_train_task = None
    service.paper_executor = None
    service._okx_paper = None
    service._okx_live = None
    service._okx_authoritative_sync_started_at = None
    service._okx_authoritative_sync_last_success_at = None
    service._okx_authoritative_sync_last_failure_at = None
    service._okx_authoritative_sync_last_error = None
    service._okx_authoritative_sync_last_duration_seconds = None
    service._okx_authoritative_sync_last_result_count = None
    service._okx_authoritative_sync_last_result_kinds = {}
    service._okx_authoritative_sync_last_requires_attention_count = 0
    service._okx_authoritative_sync_last_samples = []
    service._okx_authoritative_sync_success_count = 0
    service._okx_authoritative_sync_failure_count = 0

    class FakeModelRegistry:
        async def shutdown_all(self) -> None:
            return None

    service.models = FakeModelRegistry()
    from config.settings import settings

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings.__class__, "data_dir", property(lambda _self: data_dir))

    await service.stop()

    payload = json.loads((data_dir / "trading_runtime_status.json").read_text(encoding="utf-8"))
    assert payload["running"] is False
    assert payload["round_active"] is False
    assert payload["current_stage"] == "idle"
    assert payload["okx_authoritative_sync"]["status"] == "pending"


@pytest.mark.asyncio
async def test_okx_authoritative_sync_loop_reconciles_current_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    service._running = True
    calls: list[dict[str, Any]] = []

    class FakeOkxSyncService:
        async def reconcile_positions(self, reason, timeout_seconds, lock_wait_seconds):
            calls.append(
                {
                    "reason": reason,
                    "timeout_seconds": timeout_seconds,
                    "lock_wait_seconds": lock_wait_seconds,
                }
            )
            return [
                {
                    "kind": "snapshot_update",
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "exchange_order_id": None,
                    "note": "updated from OKX",
                },
                {
                    "kind": "missing_exchange_position_without_close_fill",
                    "symbol": "SPK/USDT",
                    "side": "short",
                    "exchange_order_id": None,
                    "requires_attention": True,
                    "note": "waiting for authoritative close fill",
                },
            ]

    async def fake_sleep(_seconds: float) -> None:
        service._running = False

    monkeypatch.setattr(trading_service.asyncio, "sleep", fake_sleep)
    service.okx_sync_service = FakeOkxSyncService()
    service.okx_authoritative_sync_interval_seconds = lambda: 20.0  # type: ignore[method-assign]
    service.round_start_reconcile_timeout_seconds = lambda: 8.0  # type: ignore[method-assign]
    service._okx_authoritative_sync_task = None
    service._okx_authoritative_sync_started_at = None
    service._okx_authoritative_sync_last_success_at = None
    service._okx_authoritative_sync_last_failure_at = None
    service._okx_authoritative_sync_last_error = None
    service._okx_authoritative_sync_last_duration_seconds = None
    service._okx_authoritative_sync_last_result_count = None
    service._okx_authoritative_sync_last_result_kinds = {}
    service._okx_authoritative_sync_last_requires_attention_count = 0
    service._okx_authoritative_sync_last_samples = []
    service._okx_authoritative_sync_success_count = 0
    service._okx_authoritative_sync_failure_count = 0

    await service._okx_authoritative_sync_loop()

    assert calls == [
        {
            "reason": "auto okx authoritative sync",
            "timeout_seconds": 8.0,
            "lock_wait_seconds": 0.1,
        }
    ]
    status = service._okx_authoritative_sync_status_payload()
    assert status["status"] == "ok"
    assert status["success_count"] == 1
    assert status["failure_count"] == 0
    assert status["last_success_at"]
    assert status["last_error"] is None
    assert status["last_result_count"] == 2
    assert status["last_result_kinds"] == {
        "snapshot_update": 1,
        "missing_exchange_position_without_close_fill": 1,
    }
    assert status["last_requires_attention_count"] == 1
    assert status["last_samples"][1]["symbol"] == "SPK/USDT"
    assert status["last_samples"][1]["requires_attention"] is True


@pytest.mark.asyncio
async def test_okx_authoritative_sync_loop_does_not_wait_for_order_fact_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    service._running = True
    calls: list[dict[str, Any]] = []
    order_fact_started: list[bool] = []
    never_finish = asyncio.Event()

    class FakeOkxSyncService:
        async def reconcile_positions(self, reason, timeout_seconds, lock_wait_seconds):
            calls.append(
                {
                    "reason": reason,
                    "timeout_seconds": timeout_seconds,
                    "lock_wait_seconds": lock_wait_seconds,
                }
            )
            return [
                {
                    "kind": "snapshot_update",
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "note": "updated from OKX",
                }
            ]

    class SlowOrderFactSyncService:
        async def sync(self) -> dict[str, Any]:
            order_fact_started.append(True)
            await never_finish.wait()
            return {"status": "ok", "okx_pull_available": True, "unverified_count": 0}

    def factory(**_kwargs: Any) -> SlowOrderFactSyncService:
        return SlowOrderFactSyncService()

    async def fake_sleep(_seconds: float) -> None:
        service._running = False

    monkeypatch.setattr(trading_service.asyncio, "sleep", fake_sleep)
    service.okx_sync_service = FakeOkxSyncService()
    service.okx_order_fact_sync_factory = factory
    service.okx_authoritative_sync_interval_seconds = lambda: 20.0  # type: ignore[method-assign]
    service.round_start_reconcile_timeout_seconds = lambda: 8.0  # type: ignore[method-assign]
    service._okx_authoritative_sync_task = None
    service._okx_authoritative_sync_started_at = None
    service._okx_authoritative_sync_last_success_at = None
    service._okx_authoritative_sync_last_failure_at = None
    service._okx_authoritative_sync_last_error = None
    service._okx_authoritative_sync_last_duration_seconds = None
    service._okx_authoritative_sync_last_result_count = None
    service._okx_authoritative_sync_last_result_kinds = {}
    service._okx_authoritative_sync_last_requires_attention_count = 0
    service._okx_authoritative_sync_last_degraded_count = 0
    service._okx_authoritative_sync_last_samples = []
    service._okx_authoritative_sync_success_count = 0
    service._okx_authoritative_sync_failure_count = 0
    service._okx_order_fact_sync_task = None
    service._okx_order_fact_sync_last_started_at = None
    service._okx_order_fact_sync_last_finished_at = None
    service._okx_order_fact_sync_last_row = None
    service._okx_order_fact_sync_last_error = None
    service._okx_order_fact_sync_success_count = 0
    service._okx_order_fact_sync_failure_count = 0

    await service._okx_authoritative_sync_loop()

    status = service._okx_authoritative_sync_status_payload()
    assert calls == [
        {
            "reason": "auto okx authoritative sync",
            "timeout_seconds": 8.0,
            "lock_wait_seconds": 0.1,
        }
    ]
    assert status["status"] == "ok"
    assert status["last_result_kinds"] == {"snapshot_update": 1}
    assert status["last_requires_attention_count"] == 0
    assert status["last_duration_seconds"] is not None
    assert status["last_duration_seconds"] < 1.0
    assert status["order_fact_sync"]["task_running"] is True

    task = service._okx_order_fact_sync_task
    assert task is not None
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_okx_order_fact_sync_background_backs_off_after_degraded_result() -> None:
    service = TradingService.__new__(TradingService)
    service.okx_order_fact_sync_factory = object()
    service._okx_order_fact_sync_task = None
    service._okx_order_fact_sync_last_finished_at = datetime.now(UTC) - timedelta(seconds=30)
    service._okx_order_fact_sync_last_row = {
        "kind": "order_fact_sync",
        "status": "warning",
        "okx_pull_available": False,
        "degraded": True,
    }
    service.okx_order_fact_sync_interval_seconds = lambda: 90.0  # type: ignore[method-assign]
    service.okx_order_fact_sync_degraded_interval_seconds = lambda: 240.0  # type: ignore[method-assign]

    service._start_okx_order_fact_sync_background()

    assert service._okx_order_fact_sync_task is None


@pytest.mark.asyncio
async def test_okx_order_fact_sync_position_confirmed_does_not_block_runtime_gate() -> None:
    service = TradingService.__new__(TradingService)
    service.round_start_reconcile_timeout_seconds = lambda: 8.0  # type: ignore[method-assign]

    class FakeOrderFactSyncService:
        async def sync(self) -> dict[str, Any]:
            return {
                "status": "ok",
                "okx_pull_available": True,
                "confirmed_count": 96,
                "position_confirmed_count": 1,
                "unverified_count": 0,
                "backfilled_count": 0,
                "position_history_backfilled_count": 2,
                "position_history_updated_count": 3,
            }

    def factory(**_kwargs: Any) -> FakeOrderFactSyncService:
        return FakeOrderFactSyncService()

    service.okx_order_fact_sync_factory = factory

    row = await service._sync_okx_order_facts_for_loop()

    assert row["kind"] == "order_fact_sync"
    assert row["requires_attention"] is False
    assert "position_confirmed=1" in row["note"]
    assert "position_history=2+3" in row["note"]
    assert row["order_fact_sync"]["unverified_count"] == 0


@pytest.mark.asyncio
async def test_okx_order_fact_sync_pull_degraded_does_not_create_state_difference() -> None:
    service = TradingService.__new__(TradingService)
    service.round_start_reconcile_timeout_seconds = lambda: 8.0  # type: ignore[method-assign]

    class FakeOrderFactSyncService:
        async def sync(self) -> dict[str, Any]:
            return {
                "status": "warning",
                "okx_pull_available": False,
                "local_checked": 231,
                "confirmed_count": 0,
                "position_confirmed_count": 0,
                "unverified_count": 0,
                "backfilled_count": 0,
                "position_history_backfilled_count": 0,
                "position_history_updated_count": 0,
                "error": "TimeoutError",
            }

    def factory(**_kwargs: Any) -> FakeOrderFactSyncService:
        return FakeOrderFactSyncService()

    service.okx_order_fact_sync_factory = factory

    row = await service._sync_okx_order_facts_for_loop()
    summary = TradingService._okx_authoritative_sync_result_summary([row])

    assert row["kind"] == "order_fact_sync"
    assert row["requires_attention"] is False
    assert row["degraded"] is True
    assert row["okx_pull_available"] is False
    assert row["error"] == "TimeoutError"
    assert "OKX 订单事实同步降级" in row["note"]
    assert "不把拉取失败误判为当前状态差异" in row["note"]
    assert summary["requires_attention_count"] == 0
    assert summary["degraded_count"] == 1


@pytest.mark.asyncio
async def test_okx_order_fact_sync_unverified_still_blocks_runtime_gate() -> None:
    service = TradingService.__new__(TradingService)
    service.round_start_reconcile_timeout_seconds = lambda: 8.0  # type: ignore[method-assign]

    class FakeOrderFactSyncService:
        async def sync(self) -> dict[str, Any]:
            return {
                "status": "warning",
                "okx_pull_available": True,
                "local_checked": 3,
                "confirmed_count": 2,
                "position_confirmed_count": 0,
                "unverified_count": 1,
                "backfilled_count": 0,
                "position_history_backfilled_count": 0,
                "position_history_updated_count": 0,
            }

    def factory(**_kwargs: Any) -> FakeOrderFactSyncService:
        return FakeOrderFactSyncService()

    service.okx_order_fact_sync_factory = factory

    row = await service._sync_okx_order_facts_for_loop()
    summary = TradingService._okx_authoritative_sync_result_summary([row])

    assert row["requires_attention"] is True
    assert row["degraded"] is False
    assert "未被 OKX 原生成交确认" in row["note"]
    assert summary["requires_attention_count"] == 1
    assert summary["degraded_count"] == 0


@pytest.mark.asyncio
async def test_okx_authoritative_sync_loop_records_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    service._running = True

    class FakeOkxSyncService:
        async def reconcile_positions(self, reason, timeout_seconds, lock_wait_seconds):
            raise RuntimeError("OKX timeout")

    async def fake_sleep(_seconds: float) -> None:
        service._running = False

    monkeypatch.setattr(trading_service.asyncio, "sleep", fake_sleep)
    service.okx_sync_service = FakeOkxSyncService()
    service.okx_authoritative_sync_interval_seconds = lambda: 20.0  # type: ignore[method-assign]
    service.round_start_reconcile_timeout_seconds = lambda: 8.0  # type: ignore[method-assign]
    service._okx_authoritative_sync_task = None
    service._okx_authoritative_sync_started_at = None
    service._okx_authoritative_sync_last_success_at = None
    service._okx_authoritative_sync_last_failure_at = None
    service._okx_authoritative_sync_last_error = None
    service._okx_authoritative_sync_last_duration_seconds = None
    service._okx_authoritative_sync_last_result_count = None
    service._okx_authoritative_sync_last_result_kinds = {}
    service._okx_authoritative_sync_last_requires_attention_count = 0
    service._okx_authoritative_sync_last_samples = []
    service._okx_authoritative_sync_success_count = 0
    service._okx_authoritative_sync_failure_count = 0

    await service._okx_authoritative_sync_loop()

    status = service._okx_authoritative_sync_status_payload()
    assert status["status"] == "warning"
    assert status["success_count"] == 0
    assert status["failure_count"] == 1
    assert status["last_failure_at"]
    assert "OKX timeout" in status["last_error"]


def test_okx_authoritative_sync_recent_success_downgrades_later_timeout() -> None:
    service = TradingService.__new__(TradingService)
    now = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
    service.okx_authoritative_sync_interval_seconds = lambda: 20.0  # type: ignore[method-assign]
    service._okx_authoritative_sync_started_at = now - timedelta(seconds=6)
    service._okx_authoritative_sync_last_success_at = now - timedelta(seconds=40)
    service._okx_authoritative_sync_last_failure_at = now - timedelta(seconds=5)
    service._okx_authoritative_sync_last_error = "TimeoutError"
    service._okx_authoritative_sync_last_duration_seconds = 10.0
    service._okx_authoritative_sync_last_result_count = 1
    service._okx_authoritative_sync_last_result_kinds = {"snapshot_update": 1}
    service._okx_authoritative_sync_last_requires_attention_count = 0
    service._okx_authoritative_sync_last_degraded_count = 0
    service._okx_authoritative_sync_last_samples = []
    service._okx_authoritative_sync_success_count = 3
    service._okx_authoritative_sync_failure_count = 1
    service._okx_authoritative_sync_task = None
    service._okx_order_fact_sync_status_payload = lambda _now=None: {"status": "ok"}

    status = service._okx_authoritative_sync_status_payload(now)

    assert status["status"] == "degraded"
    assert status["fresh_success_available"] is True
    assert status["last_failure_covered_by_fresh_success"] is True
    assert service._okx_authoritative_sync_entry_block_reason(now) is None


def test_successful_runtime_round_clears_recovered_scope_error(
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
    service._start_runtime_round("market", market_start)
    scope_token = trading_service._analysis_scope_context.set("market")
    try:
        service.record_round_error(
            "exchange position reconciliation timed out during market round start; "
            "continuing with local position state"
        )
    finally:
        trading_service._analysis_scope_context.reset(scope_token)

    payload = json.loads((data_dir / "trading_runtime_status.json").read_text(encoding="utf-8"))
    assert payload["market_last_error"]
    assert payload["last_round_error"]
    assert service._last_round_error is None

    service._finish_runtime_round("market", datetime.now(UTC), ok=True)
    service._write_runtime_heartbeat()

    payload = json.loads((data_dir / "trading_runtime_status.json").read_text(encoding="utf-8"))
    assert payload["market_round_active"] is False
    assert payload["market_current_stage"] == "idle"
    assert payload["market_last_error"] is None
    assert payload["last_round_error"] is None


def test_market_scope_skips_full_reconciliation_at_round_start() -> None:
    assert TradingService._should_run_full_reconciliation_at_round_start("market") is False
    assert TradingService._should_run_full_reconciliation_at_round_start("position") is True
    assert TradingService._should_run_full_reconciliation_at_round_start("full") is True


def test_market_scope_skips_sync_position_price_refresh_before_ai() -> None:
    assert TradingService._should_refresh_position_prices_before_review("market") is False
    assert TradingService._should_refresh_position_prices_before_review("position") is True
    assert TradingService._should_refresh_position_prices_before_review("full") is True


def test_market_scope_skips_pending_exit_recovery_before_ai() -> None:
    assert TradingService._should_recover_pending_exits_for_scope("market") is False
    assert TradingService._should_recover_pending_exits_for_scope("position") is True
    assert TradingService._should_recover_pending_exits_for_scope("full") is True


@pytest.mark.asyncio
async def test_final_market_candidate_refresh_blocks_on_complete_market_sources() -> None:
    service = TradingService.__new__(TradingService)
    received: dict[str, Any] = {}
    vector = SimpleNamespace(current_price=100.0, close=100.0, bid=99.9, ask=100.1)

    async def feature_snapshot(symbol: str, **kwargs: Any) -> Any:
        received["symbol"] = symbol
        received.update(kwargs)
        return vector

    service._get_feature_vector_snapshot = feature_snapshot  # type: ignore[method-assign]

    result = await service._fresh_feature_vector_for_analysis("BTC/USDT")

    assert result is vector
    assert received == {
        "symbol": "BTC/USDT",
        "wait_for_sentiment": False,
        "block_on_remote_indicators": True,
        "block_on_remote_derivatives": True,
        "allow_cached_indicator_build": False,
        "allow_indicator_background_refresh": False,
        "allow_derivatives_background_refresh": False,
    }


def test_auto_scan_feature_fetch_early_quorum_is_market_only() -> None:
    service = TradingService.__new__(TradingService)
    service._safe_dict = TradingService._safe_dict.__get__(service, TradingService)
    service._last_auto_feature_fetch_budget_diagnostics = {
        "selected_market_feature_fetch_count": 48,
    }

    met, diagnostics = service._auto_scan_feature_fetch_early_quorum(
        completed_valid_count=16,
        total_fetch_count=48,
        configured_limit=8,
        run_market_analysis=True,
        run_position_analysis=False,
        auto_scan=True,
    )
    assert met is True
    assert diagnostics["quorum"] == 16
    assert diagnostics["is_entry_gate"] is False
    assert diagnostics["budget_ready_quorum"] == 8
    assert diagnostics["budget_ready_met"] is True

    near_met, near_diagnostics = service._auto_scan_feature_fetch_early_quorum(
        completed_valid_count=15,
        total_fetch_count=48,
        configured_limit=8,
        run_market_analysis=True,
        run_position_analysis=False,
        auto_scan=True,
    )
    assert near_met is True
    assert near_diagnostics["exact_met"] is False
    assert near_diagnostics["near_quorum_met"] is True
    assert near_diagnostics["near_quorum"] == 15
    assert near_diagnostics["budget_ready_met"] is True

    budget_ready_met, budget_ready_diagnostics = service._auto_scan_feature_fetch_early_quorum(
        completed_valid_count=8,
        total_fetch_count=48,
        configured_limit=8,
        run_market_analysis=True,
        run_position_analysis=False,
        auto_scan=True,
    )
    assert budget_ready_met is True
    assert budget_ready_diagnostics["exact_met"] is False
    assert budget_ready_diagnostics["near_quorum_met"] is False
    assert budget_ready_diagnostics["budget_ready_met"] is True

    not_met, _diagnostics = service._auto_scan_feature_fetch_early_quorum(
        completed_valid_count=7,
        total_fetch_count=48,
        configured_limit=8,
        run_market_analysis=True,
        run_position_analysis=False,
        auto_scan=True,
    )
    assert not_met is False

    position_scope_met, position_scope_diagnostics = (
        service._auto_scan_feature_fetch_early_quorum(
            completed_valid_count=48,
            total_fetch_count=48,
            configured_limit=8,
            run_market_analysis=True,
            run_position_analysis=True,
            auto_scan=True,
        )
    )
    assert position_scope_met is False
    assert position_scope_diagnostics["eligible"] is False


@pytest.mark.asyncio
async def test_paused_market_scope_does_not_start_market_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    service = TradingService.__new__(TradingService)
    service._running = True
    service._last_market_round_started_at = None
    service._analysis_runtime = {
        "market": _AnalysisRuntimeState(),
        "position": _AnalysisRuntimeState(),
        "full": _AnalysisRuntimeState(),
    }
    state_path = tmp_path / "trading-control-state.json"
    monkeypatch.setattr(mode_manager, "_state_path", state_path)
    monkeypatch.setattr(mode_manager, "_last_state_mtime", 0.0)
    await mode_manager.pause()

    result = await service.run_once("market")

    assert result["status"] == "paused"
    assert result["market_analysis_paused"] is True
    assert service._last_market_round_started_at is None
    assert service._runtime_state("market").current_stage == "idle"


async def _async_value(value: Any) -> Any:
    return value


@pytest.mark.asyncio
async def test_shadow_backtest_maintenance_runs_in_background_for_market_round() -> None:
    service = TradingService.__new__(TradingService)
    started = asyncio.Event()
    release = asyncio.Event()
    limits: list[int] = []

    class SlowShadowBacktestService:
        async def update_due(self, limit: int = 200) -> int:
            limits.append(limit)
            started.set()
            await release.wait()
            return 7

    service.shadow_backtest_service = SlowShadowBacktestService()
    service._shadow_backtest_update_task = None
    results: dict[str, Any] = {}

    await service._update_shadow_backtests_for_round(
        analysis_scope="market",
        results=results,
    )

    assert results["shadow_backtest_maintenance"]["started_in_background"] is True
    assert results["shadow_backtest_maintenance"]["is_entry_gate"] is False
    assert results["shadow_backtest_maintenance"]["update_limit"] == (
        trading_service.SHADOW_BACKTEST_MARKET_BACKGROUND_UPDATE_LIMIT
    )
    assert service._shadow_backtest_update_task is not None
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert limits == [trading_service.SHADOW_BACKTEST_MARKET_BACKGROUND_UPDATE_LIMIT]

    release.set()
    await asyncio.wait_for(service._shadow_backtest_update_task, timeout=1.0)
    await asyncio.sleep(0)

    status = service._shadow_backtest_maintenance_status()
    assert status["running"] is False
    assert status["last_completed_count"] == 7
    assert status["success_count"] == 1
    assert status["failure_count"] == 0


@pytest.mark.asyncio
async def test_shadow_backtest_maintenance_reuses_running_background_task() -> None:
    service = TradingService.__new__(TradingService)
    release = asyncio.Event()
    calls = 0

    class SlowShadowBacktestService:
        async def update_due(self, limit: int = 200) -> int:
            nonlocal calls
            calls += 1
            await release.wait()
            return 1

    service.shadow_backtest_service = SlowShadowBacktestService()
    service._shadow_backtest_update_task = None
    first_results: dict[str, Any] = {}
    second_results: dict[str, Any] = {}

    await service._update_shadow_backtests_for_round(
        analysis_scope="market",
        results=first_results,
    )
    await asyncio.sleep(0)
    await service._update_shadow_backtests_for_round(
        analysis_scope="market",
        results=second_results,
    )

    assert calls == 1
    assert second_results["shadow_backtest_maintenance"]["running"] is True
    assert "started_in_background" not in second_results["shadow_backtest_maintenance"]

    release.set()
    await asyncio.wait_for(service._shadow_backtest_update_task, timeout=1.0)


@pytest.mark.asyncio
async def test_stale_entry_maintenance_runs_in_background_for_market_round() -> None:
    service = TradingService.__new__(TradingService)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class SlowStaleEntryExpirer:
        async def expire(self) -> int:
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return 5

    service.stale_entry_candidate_expirer = SlowStaleEntryExpirer()
    service._stale_entry_expire_task = None
    results: dict[str, Any] = {}

    await service._update_stale_entry_candidates_for_round(results=results)

    assert results["stale_entry_maintenance"]["started_in_background"] is True
    assert results["stale_entry_maintenance"]["is_entry_gate"] is False
    assert service._stale_entry_expire_task is not None
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert calls == 1

    release.set()
    await asyncio.wait_for(service._stale_entry_expire_task, timeout=1.0)
    await asyncio.sleep(0)

    status = service._stale_entry_candidate_maintenance_status()
    assert status["running"] is False
    assert status["last_expired_count"] == 5
    assert status["success_count"] == 1
    assert status["failure_count"] == 0


@pytest.mark.asyncio
async def test_stale_entry_maintenance_reuses_running_background_task() -> None:
    service = TradingService.__new__(TradingService)
    release = asyncio.Event()
    calls = 0

    class SlowStaleEntryExpirer:
        async def expire(self) -> int:
            nonlocal calls
            calls += 1
            await release.wait()
            return 1

    service.stale_entry_candidate_expirer = SlowStaleEntryExpirer()
    service._stale_entry_expire_task = None
    first_results: dict[str, Any] = {}
    second_results: dict[str, Any] = {}

    await service._update_stale_entry_candidates_for_round(results=first_results)
    await asyncio.sleep(0)
    await service._update_stale_entry_candidates_for_round(results=second_results)

    assert calls == 1
    assert second_results["stale_entry_maintenance"]["running"] is True
    assert "started_in_background" not in second_results["stale_entry_maintenance"]

    release.set()
    await asyncio.wait_for(service._stale_entry_expire_task, timeout=1.0)




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
        async def search(self, query, *, top_k=8, symbol="", kind=""):
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
async def test_market_analysis_loop_keeps_stage_budget_separate_from_outer_watchdog(monkeypatch):
    calls: list[str] = []
    completed: list[str] = []
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
        await original_sleep(0.03)
        completed.append(scope)
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
    assert completed == ["market"]
    assert sleeps == [0.0, 30.0]


@pytest.mark.asyncio
async def test_analysis_service_loop_continues_after_internal_round_cancellation(monkeypatch):
    calls: list[str] = []
    running = True
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        nonlocal running
        sleeps.append(seconds)
        if len(calls) >= 2 and len(sleeps) > 2:
            running = False

    async def run_once(scope):
        calls.append(scope)
        if len(calls) == 1:
            raise asyncio.CancelledError()
        return {"scope": scope}

    service = PositionReviewService(
        run_once_provider=run_once,
        is_running_provider=lambda: running,
        round_watchdog_provider=lambda: 30.0,
    )
    service.initial_delay_seconds = 0.0
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await service.loop(lambda: 30.0)

    assert calls == ["position", "position"]
    assert sleeps == [0.0, 30.0, 30.0]


@pytest.mark.asyncio
async def test_position_review_loop_keeps_stage_timeout_separate_from_round_watchdog(
    monkeypatch,
):
    calls: list[str] = []
    completed: list[str] = []
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
        await original_sleep(0.03)
        completed.append(scope)
        return {"scope": scope}

    service = PositionReviewService(
        run_once_provider=run_once,
        is_running_provider=lambda: running,
        timeout_provider=lambda: 0.001,
        round_watchdog_provider=lambda: 0.001,
    )
    service.initial_delay_seconds = 0.0
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await service.loop(lambda: 30.0)

    assert calls == ["position"]
    assert completed == ["position"]
    assert sleeps == [0.0, 30.0]


def test_position_review_batch_timeout_scales_with_selected_groups() -> None:
    service = PositionReviewService(timeout_provider=lambda: 2.0)

    assert service._review_positions_timeout_seconds(1) == pytest.approx(6.0)
    assert service._review_positions_timeout_seconds(4) == pytest.approx(16.0)
    assert service._review_positions_timeout_seconds(20) == pytest.approx(70.0)


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


def test_opposite_entry_has_no_private_full_close_conversion():
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


@pytest.mark.asyncio
async def test_okx_authoritative_sync_warning_pauses_new_pair_analysis() -> None:
    service = TradingService.__new__(TradingService)
    service.risk_engine = SimpleNamespace(
        circuit_breaker=SimpleNamespace(is_open=False, get_state=lambda: {}),
    )
    service._okx_authoritative_sync_status_payload = lambda _now=None: {
        "status": "warning",
        "last_error": "OKX timeout",
        "last_requires_attention_count": 0,
    }

    reason = await service._new_pair_analysis_pause_reason("ensemble_trader", open_positions=[])

    assert "OKX 自动对账异常" in reason
    assert "OKX timeout" in reason


@pytest.mark.asyncio
async def test_okx_authoritative_sync_attention_pauses_new_pair_analysis() -> None:
    service = TradingService.__new__(TradingService)
    service.risk_engine = SimpleNamespace(
        circuit_breaker=SimpleNamespace(is_open=False, get_state=lambda: {}),
    )
    service._okx_authoritative_sync_status_payload = lambda _now=None: {
        "status": "ok",
        "last_error": None,
        "last_requires_attention_count": 2,
    }

    reason = await service._new_pair_analysis_pause_reason("ensemble_trader", open_positions=[])

    assert "发现 2 个当前状态差异" in reason
    assert "暂停新开仓" in reason


@pytest.mark.asyncio
async def test_local_ai_tools_auto_train_blocks_when_okx_daily_training_gate_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    calls: list[str] = []

    class FakeLocalAITools:
        def enabled(self) -> bool:
            return True

        async def status(self) -> dict[str, Any]:
            calls.append("status")
            return {"available": True}

        async def train(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            calls.append("train")
            return {"trained": True}

    monkeypatch.setattr(
        "services.okx_training_gate.okx_training_refresh_gate",
        lambda: {
            "allowed": False,
            "reason": "okx_daily_reconciliation_training_blocked",
            "can_refresh_training": False,
            "read_only": True,
            "mutates_database": False,
        },
    )
    service.local_ai_tools = FakeLocalAITools()

    result = await service._maybe_train_local_ai_tools(force=True)

    assert result["trained"] is False
    assert result["reason"] == "okx_daily_reconciliation_training_blocked"
    assert result["okx_daily_reconciliation_gate"]["allowed"] is False
    assert calls == []


@pytest.mark.asyncio
async def test_local_ml_auto_train_process_isolated_from_trading_database_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    captured: dict[str, Any] = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'{"trained":false,"reason":"not_due"}', b""

    async def fake_create_subprocess_exec(*args: str, **kwargs: Any) -> FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        trading_service.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    result = await service._run_local_ml_training_subprocess()

    assert result["reason"] == "not_due"
    assert result["training_process_isolated"] is True
    script_path = str(captured["args"][1]).replace("\\", "/")
    assert script_path.endswith("scripts/run_local_ml_auto_train.py")
    assert captured["kwargs"]["cwd"] == str(trading_service.PROJECT_ROOT)


@pytest.mark.asyncio
async def test_local_ai_cursor_probe_isolated_from_trading_database_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    captured: dict[str, Any] = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (
                b'{"reason":"cursor_probe_complete",'
                b'"completed_shadow_sample_count":14,'
                b'"completed_trade_sample_count":61}',
                b"",
            )

    async def fake_create_subprocess_exec(*args: str, **kwargs: Any) -> FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        trading_service.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    result = await service._run_local_ai_tools_training_cursor_subprocess()

    assert result["reason"] == "cursor_probe_complete"
    assert result["completed_shadow_sample_count"] == 14
    assert result["completed_trade_sample_count"] == 61
    assert result["training_process_isolated"] is True
    script_path = str(captured["args"][1]).replace("\\", "/")
    assert script_path.endswith("scripts/run_local_ai_tools_training_cursors.py")
    assert captured["kwargs"]["cwd"] == str(trading_service.PROJECT_ROOT)


@pytest.mark.asyncio
async def test_local_ml_auto_train_failure_uses_retry_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    service._running = True
    delays: list[float] = []
    service._run_local_ml_training_subprocess = lambda: _async_value(  # type: ignore[method-assign]
        {
            "trained": False,
            "reason": "error",
            "error": "QueuePool connection timed out",
            "training_process_isolated": True,
        }
    )
    service._maybe_train_local_ai_tools = lambda: _async_value(  # type: ignore[method-assign]
        {"trained": False, "reason": "not_due"}
    )

    async def stop_after_sleep(delay: float) -> None:
        delays.append(delay)
        service._running = False

    monkeypatch.setattr(trading_service.asyncio, "sleep", stop_after_sleep)

    await service._ml_auto_train_loop()

    assert delays == [trading_service.AUTO_TRAIN_RETRY_INTERVAL_SECONDS]


@pytest.mark.asyncio
async def test_local_ai_cursor_failure_uses_retry_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    service._running = True
    delays: list[float] = []
    service._run_local_ml_training_subprocess = lambda: _async_value(  # type: ignore[method-assign]
        {"trained": False, "reason": "not_due"}
    )
    service._maybe_train_local_ai_tools = lambda: _async_value(  # type: ignore[method-assign]
        {
            "trained": False,
            "reason": "error",
            "error": "QueuePool connection timed out in isolated cursor probe",
            "training_process_isolated": True,
        }
    )

    async def stop_after_sleep(delay: float) -> None:
        delays.append(delay)
        service._running = False

    monkeypatch.setattr(trading_service.asyncio, "sleep", stop_after_sleep)

    await service._ml_auto_train_loop()

    assert delays == [trading_service.AUTO_TRAIN_RETRY_INTERVAL_SECONDS]


@pytest.mark.asyncio
async def test_local_ai_tools_auto_train_persists_artifact_after_status_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import train_local_ai_tools_models as train_script

    service = TradingService.__new__(TradingService)
    service._local_tools_last_completed_shadow_count = 0
    captured: dict[str, Any] = {}

    class FakeLocalAITools:
        def enabled(self) -> bool:
            return True

        async def status(self) -> dict[str, Any]:
            raise TimeoutError("phase3 status endpoint timed out")

        async def train(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {
                "trained": True,
                "shadow_sample_count": len(args[0]),
                "trade_sample_count": len(args[1]),
                "trained_at": "2026-06-30T08:00:00+00:00",
            }

    async def load_shadow_samples(_limit: int | None = None) -> list[dict[str, Any]]:
        assert _limit is None
        return [
            {"id": 1, "features": {"symbol": "BTC/USDT"}},
            {"id": 2, "features": {"symbol": "ETH/USDT"}},
        ]

    async def load_trade_reflections(_limit: int | None = None) -> list[dict[str, Any]]:
        assert _limit is None
        return [{"id": 2, "symbol": "BTC/USDT", "side": "long", "pnl": 1.2}]

    async def load_empty(_limit: int | None = None) -> list[dict[str, Any]]:
        assert _limit is None
        return []

    async def completed_trade_count() -> int:
        return 33

    def annotate_payload(
        *,
        shadow_samples: list[dict[str, Any]],
        trade_samples: list[dict[str, Any]],
        sequence_samples: list[dict[str, Any]],
        text_sentiment_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "shadow_samples": shadow_samples,
            "trade_samples": trade_samples,
            "sequence_samples": sequence_samples,
            "text_sentiment_samples": text_sentiment_samples,
            "quality_report": {
                "totals": {
                    "total": 2,
                    "excluded": 0,
                    "effective_weight_ratio": 1.0,
                }
            },
            "governance_report": {
                "trainable_sample_count": 2,
                "contamination_risk": "low",
            },
        }

    monkeypatch.setattr(
        "services.okx_training_gate.okx_training_refresh_gate",
        lambda: {
            "allowed": True,
            "reason": "okx_daily_reconciliation_allows_training_refresh",
            "can_refresh_training": True,
            "read_only": True,
            "mutates_database": False,
        },
    )
    monkeypatch.setattr(train_script, "_load_shadow_samples", load_shadow_samples)
    monkeypatch.setattr(train_script, "_load_trade_reflection_samples", load_trade_reflections)
    monkeypatch.setattr(train_script, "_load_authoritative_trade_samples", load_empty)
    monkeypatch.setattr(train_script, "_load_sequence_samples", load_empty)
    monkeypatch.setattr(train_script, "_load_text_sentiment_samples", load_empty)
    monkeypatch.setattr(train_script, "_merge_trade_samples", lambda a, b: [*a, *b])
    monkeypatch.setattr(train_script, "_completed_trade_sample_count", completed_trade_count)
    monkeypatch.setattr(
        "services.training_data_quality.annotate_training_payload",
        annotate_payload,
    )
    service.local_ai_tools = FakeLocalAITools()
    service._completed_shadow_backtest_total = lambda: _async_value(9999)  # type: ignore[method-assign]
    service._run_local_ai_tools_training_subprocess = lambda: _async_value(  # type: ignore[method-assign]
        {
            "trained": True,
            "shadow_sample_count": 2,
            "completed_shadow_sample_count": 9999,
            "completed_trade_sample_count": 33,
        }
    )

    result = await service._maybe_train_local_ai_tools(force=True)

    assert result["trained"] is True
    assert result["completed_shadow_sample_count"] == 9999
    assert result["completed_trade_sample_count"] == 33
    assert result["training_process_isolated"] is True
    assert result["training_policy"]["concurrency_policy"] == (
        "exclusive_local_ai_tools_training_process_lock"
    )
    assert result["training_policy"]["status_probe_fallback"] == (
        "train_in_isolated_process"
    )
    assert captured == {}


@pytest.mark.asyncio
async def test_local_ai_tools_auto_train_checks_cursors_before_loading_training_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import train_local_ai_tools_models as train_script

    service = TradingService.__new__(TradingService)
    service._local_tools_last_completed_shadow_count = 0

    class FakeLocalAITools:
        def enabled(self) -> bool:
            return True

        async def status(self) -> dict[str, Any]:
            return {
                "available": True,
                "model_bundle_available": True,
                "last_trained_completed_shadow_sample_count": 10,
                "last_trained_completed_trade_sample_count": 3,
            }

    async def fail_heavy_load(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("not-due checks must not load the full training payload")

    monkeypatch.setattr(
        "services.okx_training_gate.okx_training_refresh_gate",
        lambda: {"allowed": True},
    )
    monkeypatch.setattr(
        train_script,
        "_completed_trade_sample_count",
        lambda: _async_value(3),
    )
    monkeypatch.setattr(train_script, "_load_sequence_samples", fail_heavy_load)
    service.local_ai_tools = FakeLocalAITools()
    service._run_local_ai_tools_training_cursor_subprocess = lambda: _async_value(  # type: ignore[method-assign]
        {
            "reason": "cursor_probe_complete",
            "completed_shadow_sample_count": 10,
            "completed_trade_sample_count": 3,
        }
    )

    result = await service._maybe_train_local_ai_tools_process(force=False)

    assert result["trained"] is False
    assert result["reason"] == "not_due"
    assert result["training_policy"]["process_boundary"] == (
        "dedicated_training_subprocess"
    )


@pytest.mark.asyncio
async def test_local_ai_tools_auto_train_runs_due_training_outside_trading_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import train_local_ai_tools_models as train_script

    service = TradingService.__new__(TradingService)
    service._local_tools_last_completed_shadow_count = 0
    service._local_tools_active_training_run_id = "isolated-training-run"
    started_runs: list[dict[str, Any]] = []

    class FakeTrainingState:
        def start_run(self, **kwargs: Any) -> None:
            started_runs.append(kwargs)

    class FakeLocalAITools:
        def enabled(self) -> bool:
            return True

        async def status(self) -> dict[str, Any]:
            return {
                "available": True,
                "model_bundle_available": True,
                "last_trained_completed_shadow_sample_count": 10,
                "last_trained_completed_trade_sample_count": 3,
            }

    async def fail_heavy_load(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("due auto-training must not load payload in the trading process")

    monkeypatch.setattr(
        "services.okx_training_gate.okx_training_refresh_gate",
        lambda: {"allowed": True},
    )
    monkeypatch.setattr(
        train_script,
        "_completed_trade_sample_count",
        lambda: _async_value(3),
    )
    monkeypatch.setattr(train_script, "_load_sequence_samples", fail_heavy_load)
    service.local_ai_tools = FakeLocalAITools()
    service.model_training_state_store = FakeTrainingState()
    service._run_local_ai_tools_training_cursor_subprocess = lambda: _async_value(  # type: ignore[method-assign]
        {
            "reason": "cursor_probe_complete",
            "completed_shadow_sample_count": 11,
            "completed_trade_sample_count": 3,
        }
    )
    service._run_local_ai_tools_training_subprocess = lambda: _async_value(  # type: ignore[method-assign]
        {
            "trained": True,
            "shadow_sample_count": 12,
            "completed_shadow_sample_count": 12,
            "last_trained_completed_shadow_sample_count": 12,
            "completed_trade_sample_count": 4,
            "last_trained_completed_trade_sample_count": 4,
        }
    )

    result = await service._maybe_train_local_ai_tools_process(force=False)

    assert result["trained"] is True
    assert result["training_process_isolated"] is True
    assert result["new_shadow_sample_count"] == 1
    assert result["last_trained_completed_shadow_sample_count"] == 12
    assert result["last_trained_completed_trade_sample_count"] == 4
    assert service._local_tools_last_completed_shadow_count == 12
    assert started_runs == [
        {
            "scheduler_id": "local_ai_tools_auto_train",
            "model_ids": trading_service.LOCAL_AI_TOOL_MODEL_IDS,
            "run_id": "isolated-training-run",
            "trigger_reason": "training_due",
            "sample_cursor": {"shadow": 11, "trade": 3},
            "timeout_seconds": trading_service.AUTO_TRAIN_LEASE_STALE_SECONDS,
        }
    ]
    assert result["training_policy"]["process_boundary"] == (
        "dedicated_training_subprocess"
    )


@pytest.mark.asyncio
async def test_local_ai_tools_auto_train_rebuilds_when_clean_view_rebases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import train_local_ai_tools_models as train_script

    service = TradingService.__new__(TradingService)
    service._local_tools_last_completed_shadow_count = 0

    class FakeLocalAITools:
        def enabled(self) -> bool:
            return True

        async def status(self) -> dict[str, Any]:
            return {
                "available": True,
                "model_bundle_available": True,
                "last_trained_completed_shadow_sample_count": 11,
                "last_trained_completed_trade_sample_count": 3,
            }

    monkeypatch.setattr(
        "services.okx_training_gate.okx_training_refresh_gate",
        lambda: {"allowed": True},
    )
    monkeypatch.setattr(
        train_script,
        "_completed_trade_sample_count",
        lambda: _async_value(3),
    )
    service.local_ai_tools = FakeLocalAITools()
    service._run_local_ai_tools_training_cursor_subprocess = lambda: _async_value(  # type: ignore[method-assign]
        {
            "reason": "cursor_probe_complete",
            "completed_shadow_sample_count": 10,
            "completed_trade_sample_count": 3,
        }
    )
    service._run_local_ai_tools_training_subprocess = lambda: _async_value(  # type: ignore[method-assign]
        {"trained": True, "shadow_sample_count": 4}
    )

    result = await service._maybe_train_local_ai_tools_process(force=False)

    assert result["trained"] is True
    assert result["new_shadow_sample_count"] == 0
    assert result["training_process_isolated"] is True
    assert result["last_trained_completed_shadow_sample_count"] == 10
    assert result["last_trained_completed_trade_sample_count"] == 3
    assert result["training_policy"]["shadow_training_view_rebased"] is True


@pytest.mark.asyncio
async def test_entry_execution_policy_blocks_entries_when_okx_sync_is_unhealthy() -> None:
    service = TradingService.__new__(TradingService)
    service._refresh_entry_symbol_blocks_if_stale = lambda **_kwargs: _async_value(None)
    service._okx_authoritative_sync_status_payload = lambda _now=None: {
        "status": "stale",
        "last_error": None,
        "last_requires_attention_count": 0,
        "source": "okx_private_api_current_positions",
    }

    result = await service.evaluate_entry_execution_policy(
        _decision(Action.LONG),
        "ensemble_trader",
        "paper",
        [],
    )

    assert result.passed is False
    assert result.blocker == "okx_authoritative_sync_unhealthy"
    assert result.data["stage_status"] == "blocked"
    assert result.data["execution_blocker"] == "okx_authoritative_sync_unhealthy"


@pytest.mark.asyncio
async def test_entry_execution_policy_allows_degraded_okx_sync_with_fresh_snapshot() -> None:
    service = TradingService.__new__(TradingService)
    service._refresh_entry_symbol_blocks_if_stale = lambda **_kwargs: _async_value(None)
    service._okx_authoritative_sync_status_payload = lambda _now=None: {
        "status": "degraded",
        "last_error": "TimeoutError",
        "last_requires_attention_count": 0,
        "fresh_success_available": True,
        "last_failure_covered_by_fresh_success": True,
        "source": "okx_private_api_current_positions",
    }

    class FakeEntryExecutionPipeline:
        async def evaluate(self, decision, model_name, model_mode, open_positions):
            assert decision.is_entry
            return PolicyGateResult.allow({"intent": "entry-passthrough"})

    service.entry_execution_pipeline = FakeEntryExecutionPipeline()

    result = await service.evaluate_entry_execution_policy(
        _decision(Action.LONG),
        "ensemble_trader",
        "paper",
        [],
    )

    assert result.passed is True
    assert result.data["intent"] == "entry-passthrough"


@pytest.mark.asyncio
async def test_entry_execution_policy_does_not_block_exit_on_okx_sync_warning() -> None:
    service = TradingService.__new__(TradingService)
    service._refresh_entry_symbol_blocks_if_stale = lambda **_kwargs: _async_value(None)
    service._okx_authoritative_sync_status_payload = lambda _now=None: {
        "status": "warning",
        "last_error": "OKX timeout",
        "last_requires_attention_count": 0,
    }

    class FakeEntryExecutionPipeline:
        async def evaluate(self, decision, model_name, model_mode, open_positions):
            assert decision.is_exit
            return PolicyGateResult.allow({"intent": "exit-passthrough"})

    service.entry_execution_pipeline = FakeEntryExecutionPipeline()

    result = await service.evaluate_entry_execution_policy(
        _decision(Action.CLOSE_LONG),
        "ensemble_trader",
        "paper",
        [],
    )

    assert result.passed is True
    assert result.data["intent"] == "exit-passthrough"


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

    async def mark_reason(decision_id, reason):
        calls.append(("reason", decision_id, reason))

    async def ensure_final(decision_id, symbol, model_name, decision_arg, results):
        calls.append(("ensure", decision_id, symbol, model_name, decision_arg.action.value))

    service._set_loop_stage = set_loop_stage  # type: ignore[method-assign]
    service._enforce_sl_tp = enforce_sl_tp  # type: ignore[method-assign]
    service.okx_sync_service = FakeSyncService()
    service._review_open_positions = review_positions  # type: ignore[method-assign]
    service._try_claim_analysis_symbol = claim_symbol  # type: ignore[method-assign]
    service._normalize_position_symbol = normalize_symbol  # type: ignore[method-assign]
    service._execute_candidate = execute_candidate  # type: ignore[method-assign]
    service._mark_decision_reason = mark_reason  # type: ignore[method-assign]
    service.decision_final_state_ensurer = SimpleNamespace(ensure=ensure_final)

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
        (
            "reason",
            456,
            "本轮还在分析或排队中：持仓复盘候选已进入执行队列，正在等待执行链路空闲并继续完成风控复核；尚未开始向 OKX 提交订单。",
        ),
        ("execute", "BTC/USDT", "ensemble_trader", "close_long", True, 456, 1),
        ("ensure", 456, "BTC/USDT", "ensemble_trader", "close_long"),
    ]


@pytest.mark.asyncio
async def test_trading_service_sl_tp_boundary_forwards_round_positions_when_supported():
    service = TradingService.__new__(TradingService)
    open_positions = [{"model_name": "ensemble_trader", "symbol": "BTC/USDT"}]
    calls: list[tuple[Any, ...]] = []

    async def enforce_sl_tp(feature_vectors, *, open_positions=None):
        calls.append(("sl_tp", sorted(feature_vectors), open_positions))
        return [{"symbol": "BTC/USDT"}]

    service._enforce_sl_tp = enforce_sl_tp  # type: ignore[method-assign]

    result = await service.enforce_sl_tp_for_position_review(
        {"BTC/USDT": object()},
        open_positions=open_positions,
    )

    assert result == [{"symbol": "BTC/USDT"}]
    assert calls == [("sl_tp", ["BTC/USDT"], open_positions)]


@pytest.mark.asyncio
async def test_fixed_take_profit_crossing_cannot_authorize_dynamic_exit():
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

    assert service._decision_count == 0
    assert auto_closes == []
    assert not any(call[0] == "execute_candidate" for call in calls)






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
async def test_entry_policy_reprices_execution_cost_with_planned_order_notional() -> None:
    events: list[tuple[str, float]] = []

    def fake_score(decision: DecisionOutput, _strategy: dict[str, Any] | None) -> float:
        planned = float(
            (decision.feature_snapshot or {}).get("planned_order_notional_usdt") or 0.0
        )
        events.append(("score", planned))
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["opportunity_score"] = {
            "score": 1.0,
            "execution_cost": {"order_size_complete": planned > 0},
        }
        decision.raw_response = raw
        return 1.0

    async def fake_sizing(
        decision: DecisionOutput,
        _model_mode: str,
        _open_positions: list[dict[str, Any]],
    ) -> None:
        sizing_count = sum(name == "sizing" for name, _value in events)
        final_notional = 100.0 if sizing_count == 0 else 80.0
        events.append(("sizing", final_notional))
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["profit_risk_sizing"] = {
            "production_eligible": True,
            "final_notional_usdt": final_notional,
        }
        decision.raw_response = raw
        decision.position_size_pct = final_notional / 1000.0

    policy = EntryPolicy(
        entry_opportunity_score=EntryOpportunityScorePolicy(fake_score),
        entry_profit_risk_sizing=EntryProfitRiskSizingPolicy(fake_sizing),
    )
    decision = _decision(Action.LONG)
    decision.feature_snapshot = {"current_price": 100.0}

    await policy.prepare_dynamic_risk_contract(decision, "paper", [])

    assert events == [
        ("score", 0.0),
        ("sizing", 100.0),
        ("score", 100.0),
        ("sizing", 80.0),
    ]
    assert decision.raw_response["execution_cost_sizing_pass"] == {
        "impact_basis_notional_usdt": 100.0,
        "final_notional_usdt": 80.0,
        "order_size_complete": True,
    }


@pytest.mark.asyncio
async def test_dynamic_entry_contract_is_ready_before_hard_risk_engine() -> None:
    service = TradingService.__new__(TradingService)
    events: list[str] = []

    async def allocated_balance(_mode: str, _decision: DecisionOutput | None) -> float:
        events.append("sizing")
        return 1000.0

    async def persist_raw(_decision_id: int, _raw: dict[str, Any]) -> None:
        events.append("persist")

    service.entry_policy = EntryPolicy(
        entry_profit_risk_sizing=EntryProfitRiskSizingPolicy(
            allocated_order_balance=allocated_balance,
        ),
    )
    service._mark_decision_raw_response = persist_raw  # type: ignore[method-assign]
    decision = _decision(Action.LONG)
    decision.stop_loss_pct = 0.02
    decision.take_profit_pct = 0.04
    decision.feature_snapshot = {
        "current_price": 100.0,
        "atr_pct": 0.01,
        "orderbook_ask_depth": 500.0,
        "orderbook_bid_depth": 500.0,
    }
    decision.raw_response = {
        "strategy_mode": {"drawdown_pressure": 0.0, "portfolio_correlation": {}},
        "exchange_risk_facts": {
            "production_eligible": True,
            "account_equity_usdt": 1000.0,
            "available_margin_usdt": 1000.0,
            "reported_max_leverage": 20.0,
            "target_inst_id": "BTC-USDT-SWAP",
            "contract_specs": {
                "BTC-USDT-SWAP": {"ctVal": "1", "ctMult": "1"},
            },
            "leverage_tiers": [
                {"tier": "1", "minSz": "0", "maxSz": "1000", "maxLeverage": 20},
            ],
            "policy_provenance": {
                "source": "okx_test_facts",
                "observation_window": "current",
                "sample_count": 1,
                "generated_at": "2026-07-15T00:00:00+00:00",
                "strategy_version": "test",
                "fallback_reason": "",
            },
        },
        "opportunity_score": {
            "score": 0.2,
            "expected_net_return_pct": 0.5,
            "expected_loss_pct": 0.1,
            "server_profit_loss_probability": 0.2,
            "tail_risk_score": 0.2,
            "profit_quality_ratio": 2.0,
            "return_lcb_pct": 0.3,
            "return_distribution_contract": {
                "raw_expected_return_pct": 0.5,
                "objective_expected_return_pct": 0.3,
                "uncertainty_penalty_pct": 0.1,
                "tail_loss_penalty_pct": 0.1,
                "tail_loss_probability": 0.2,
            },
            "execution_cost": {"production_eligible": True, "total_pct": 0.1},
            "expected_net_breakdown": {
                "components": [
                    {
                        "production_eligible": True,
                        "included_in_return_distribution": True,
                        "actual_trade_calibration": {
                            "source_authority": "okx_position_history",
                            "profile_source": "symbol_side",
                            "net_return_after_cost_pct": {
                                "count": 3,
                                "expected": 0.2,
                                "lower_hinge": 0.1,
                            },
                            "slippage_pct": {
                                "count": 3,
                                "expected": 0.01,
                                "upper_hinge": 0.02,
                            },
                        },
                    }
                ]
            },
        }
    }

    await service._prepare_entry_for_hard_risk(
        decision,
        "paper",
        [],
        decision_db_id=17,
    )
    events.append("risk")
    assessment = RiskEngine().assess(
        decision,
        current_positions=[],
        account_balance=1000.0,
    )

    assert events == ["sizing", "persist", "risk"]
    assert decision.raw_response["profit_risk_sizing"]["production_eligible"] is True
    assert decision.position_size_pct > 0
    assert assessment.approved is True




































@pytest.mark.asyncio
async def test_entry_policy_fails_fast_without_profit_risk_sizing_dependency():
    policy = EntryPolicy()

    with pytest.raises(RuntimeError, match="entry_profit_risk_sizing"):
        await policy.apply_profit_risk_sizing(_decision(Action.LONG), "paper", [])




























@pytest.mark.asyncio
async def test_entry_policy_uses_injected_price_guard_boundary():
    calls: list[str] = []

    class FakePriceGuard:
        async def guard_reason(self, decision, model_mode):
            calls.append(decision.symbol)
            assert model_mode == "paper"
            return "price-blocked"

    policy = EntryPolicy(entry_price_guard=FakePriceGuard())

    reason = await policy.pre_execution_price_guard_reason(_decision(Action.LONG), "paper")

    assert reason == "price-blocked"
    assert calls == ["BTC/USDT"]




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


def test_okx_backed_paper_sync_does_not_mutate_virtual_balances() -> None:
    import inspect

    source = inspect.getsource(TradingService._sync_paper_after_okx)

    assert "settings.get_initial_balance" not in source
    assert "pe._balances[" not in source
    assert "persist_balance_delta" not in source


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
async def test_paper_balance_snapshot_refuses_virtual_account_without_okx() -> None:
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

    async def raise_okx_down(_mode: str):
        raise RuntimeError("OKX down")

    service._get_okx_executor_for_mode = raise_okx_down

    snapshot = await service._get_okx_balance_snapshot_for_mode("paper")

    assert snapshot is None


@pytest.mark.asyncio
async def test_okx_balance_snapshot_reuses_fresh_cache_between_calls() -> None:
    service = TradingService.__new__(TradingService)
    service._safe_float = TradingService._safe_float.__get__(service, TradingService)
    service._okx_live = None
    service._okx_balance_snapshot_cache = {}
    service._okx_balance_snapshot_locks = {}

    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def get_balance_snapshot(self, _asset: str) -> dict[str, Any]:
            self.calls += 1
            return {
                "free": 12.0,
                "used": 3.0,
                "total": 15.0,
                "cash": 15.0,
                "equity": 16.0,
                "allocatable": 16.0,
            }

    executor = CountingExecutor()
    service._okx_paper = executor

    first = await service._get_okx_balance_snapshot_for_mode("paper")
    second = await service._get_okx_balance_snapshot_for_mode("paper")

    assert first == second == {
        "free": 12.0,
        "used": 3.0,
        "total": 15.0,
        "cash": 15.0,
        "equity": 16.0,
        "allocatable": 16.0,
    }
    assert executor.calls == 1


@pytest.mark.asyncio
async def test_okx_balance_snapshot_returns_stale_cache_while_refresh_in_progress() -> None:
    service = TradingService.__new__(TradingService)
    service._safe_float = TradingService._safe_float.__get__(service, TradingService)
    service._okx_live = None
    service._okx_paper = None
    service._okx_balance_snapshot_cache = {
        "paper": {
            "snapshot": {
                "free": 12.0,
                "used": 3.0,
                "total": 15.0,
                "cash": 15.0,
                "equity": 16.0,
                "allocatable": 16.0,
            },
            "fetched_at": datetime.now(UTC) - timedelta(seconds=30),
        }
    }
    lock = asyncio.Lock()
    await lock.acquire()
    service._okx_balance_snapshot_locks = {"paper": lock}

    try:
        snapshot = await service._get_okx_balance_snapshot_for_mode("paper")
    finally:
        lock.release()

    assert snapshot is not None
    assert snapshot["free"] == 12.0
    assert snapshot["stale"] is True
    assert snapshot["error"] == "OKX balance refresh already in progress"


@pytest.mark.asyncio
async def test_market_allocated_order_balance_uses_cached_okx_snapshot_without_fresh_pull() -> None:
    service = TradingService.__new__(TradingService)
    service._okx_balance_snapshot_cache = {
        "paper": {
            "snapshot": {"free": 88.0, "allocatable": 100.0, "equity": 100.0},
            "fetched_at": datetime.now(UTC),
        }
    }

    class Accounting:
        async def allocated_order_balance(self, *_args, **_kwargs):
            raise AssertionError("market sizing must not perform a fresh OKX balance pull")

    service.account_accounting_service = Accounting()
    token = trading_service._analysis_scope_context.set("market")
    try:
        balance = await service.allocated_order_balance("paper")
    finally:
        trading_service._analysis_scope_context.reset(token)

    assert balance == 88.0


@pytest.mark.asyncio
async def test_market_allocated_order_balance_schedules_refresh_when_cache_is_cold() -> None:
    service = TradingService.__new__(TradingService)
    service._okx_balance_snapshot_cache = {}
    refresh_calls: list[str] = []
    service._schedule_okx_balance_snapshot_refresh_for_new_pair_pause = (  # type: ignore[method-assign]
        lambda mode: refresh_calls.append(mode)
    )

    class Accounting:
        async def allocated_order_balance(self, *_args, **_kwargs):
            raise AssertionError("cold market sizing must not block on OKX balance")

    service.account_accounting_service = Accounting()
    token = trading_service._analysis_scope_context.set("market")
    try:
        balance = await service.allocated_order_balance("paper")
    finally:
        trading_service._analysis_scope_context.reset(token)

    assert balance == 0.0
    assert refresh_calls == ["paper"]


@pytest.mark.asyncio
async def test_paper_new_pair_pause_treats_missing_okx_balance_snapshot_as_advisory() -> None:
    service = TradingService.__new__(TradingService)
    service._safe_float = TradingService._safe_float.__get__(service, TradingService)
    service._get_model_execution_mode = lambda _model_name: "paper"
    service._normalize_position_symbol = lambda symbol: str(symbol or "")
    service.execution_allocation_state = lambda _mode: _async_value({"total_pnl": 0.0})
    service.paper_executor = None
    service._okx_paper = None
    service._okx_live = None
    service._okx_balance_snapshot_cache = {}
    service._get_okx_executor_for_mode = lambda _mode: _async_value(None)
    service.risk_engine = SimpleNamespace(
        circuit_breaker=SimpleNamespace(is_open=False, get_state=lambda: {}),
        position_checker=SimpleNamespace(entry_capacity_reason=lambda **_kwargs: None),
    )
    refresh_calls: list[str] = []
    service._schedule_okx_balance_snapshot_refresh_for_new_pair_pause = (  # type: ignore[method-assign]
        lambda mode: refresh_calls.append(mode)
    )

    snapshot = await service._get_okx_balance_snapshot_for_mode("paper")
    reason = await service._new_pair_analysis_pause_reason(
        "ensemble_trader",
        open_positions=[],
    )

    assert snapshot is None
    assert reason is None
    assert refresh_calls == ["paper"]


@pytest.mark.asyncio
async def test_new_pair_pause_context_reuses_short_lived_cached_balance_checks() -> None:
    service = TradingService.__new__(TradingService)
    service._safe_float = TradingService._safe_float.__get__(service, TradingService)
    service._normalize_position_symbol = lambda symbol: str(symbol or "")
    service._get_model_execution_mode = lambda _model_name: "paper"
    service._okx_authoritative_sync_entry_block_reason = lambda: None
    service._new_pair_pause_context_cache = {}
    allocation_calls = 0
    service._okx_balance_snapshot_cache = {
        "paper": {
            "snapshot": {"free": 1000.0, "allocatable": 1000.0, "equity": 1000.0},
            "fetched_at": datetime.now(UTC),
        }
    }

    async def allocation_state(_mode: str):
        nonlocal allocation_calls
        allocation_calls += 1
        return {"total_pnl": 0.0}

    service.execution_allocation_state = allocation_state  # type: ignore[method-assign]
    service.risk_engine = SimpleNamespace(
        circuit_breaker=SimpleNamespace(is_open=False, get_state=lambda: {}),
        position_checker=SimpleNamespace(entry_capacity_reason=lambda **_kwargs: None),
    )

    first = await service._new_pair_analysis_pause_reason(
        "ensemble_trader",
        open_positions=[{"symbol": "BTC/USDT", "side": "long", "is_open": True}],
    )
    second = await service._new_pair_analysis_pause_reason(
        "ensemble_trader",
        open_positions=[{"symbol": "BTC/USDT", "side": "long", "is_open": True}],
    )

    assert first is None
    assert second == ""
    assert allocation_calls == 0


@pytest.mark.asyncio
async def test_new_pair_pause_does_not_cache_circuit_breaker_block() -> None:
    service = TradingService.__new__(TradingService)
    service._new_pair_pause_context_cache = {
        ("ensemble_trader", "paper", ()): {
            "created_at": datetime.now(UTC),
            "reason": "",
        }
    }
    service.risk_engine = SimpleNamespace(
        circuit_breaker=SimpleNamespace(
            is_open=True,
            get_state=lambda: {"tripped_reason": "daily loss"},
        )
    )

    reason = await service._new_pair_analysis_pause_reason("ensemble_trader", open_positions=[])

    assert "风险熔断已开启" in str(reason)
    assert "daily loss" in str(reason)


@pytest.mark.asyncio
async def test_market_new_pair_pause_does_not_start_slow_context_refresh() -> None:
    service = TradingService.__new__(TradingService)
    service._normalize_position_symbol = lambda symbol: str(symbol or "")
    service._get_model_execution_mode = lambda _model_name: "paper"
    service._okx_authoritative_sync_entry_block_reason = lambda: None
    service._new_pair_pause_context_cache = {}
    refresh_calls: list[dict[str, Any]] = []
    service._schedule_new_pair_pause_context_refresh = (  # type: ignore[method-assign]
        lambda **kwargs: refresh_calls.append(kwargs)
    )
    service.risk_engine = SimpleNamespace(
        circuit_breaker=SimpleNamespace(is_open=False, get_state=lambda: {}),
    )

    reason = await service._new_pair_analysis_pause_reason(
        "ensemble_trader",
        open_positions=[{"symbol": "BTC/USDT", "side": "long", "is_open": True}],
        allow_background_refresh=True,
    )

    assert reason is None
    assert refresh_calls == []


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
    monkeypatch.setattr(trading_service, "AUTO_SCAN_FEATURE_FETCH_POOL_MAX", 20)
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
    assert service._last_auto_feature_fetch_budget_diagnostics["read_only"] is True
    assert service._last_auto_feature_fetch_budget_diagnostics["is_entry_gate"] is False
    assert (
        service._last_auto_feature_fetch_budget_diagnostics["selected_market_feature_fetch_count"]
        == 4
    )


def test_auto_scan_feature_budget_expands_discovery_without_lowering_entry_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    service._normalize_position_symbol = lambda symbol: str(symbol or "")
    service._auto_scan_feature_cursor = 0
    service.entry_symbol_universe = SimpleNamespace(
        dedupe_symbols=lambda symbols: list(dict.fromkeys(symbols))
    )
    monkeypatch.setattr(trading_service, "AUTO_SCAN_FEATURE_FETCH_POOL_MULTIPLIER", 5)
    monkeypatch.setattr(trading_service, "AUTO_SCAN_FEATURE_FETCH_POOL_MIN", 48)
    monkeypatch.setattr(trading_service, "AUTO_SCAN_FEATURE_FETCH_POOL_MAX", 64)
    symbols = [f"S{i}/USDT" for i in range(120)]

    selected = service._budget_auto_scan_feature_symbols(
        symbols,
        [],
        configured_limit=8,
    )

    assert len(selected) == 48
    diagnostics = service._last_auto_feature_fetch_budget_diagnostics
    assert diagnostics["selected_market_feature_fetch_count"] == 48
    assert diagnostics["configured_market_symbol_limit"] == 8
    assert diagnostics["pool_min"] == 48
    assert diagnostics["pool_max"] == 64
    assert diagnostics["is_entry_gate"] is False
    assert "not entry permission" in diagnostics["diagnostic_boundary"]


def test_market_candidate_funnel_snapshot_is_read_only_and_exposes_rank_dedupe_counts() -> None:
    service = TradingService.__new__(TradingService)
    service._safe_dict = TradingService._safe_dict.__get__(service, TradingService)
    service._last_auto_feature_fetch_budget_diagnostics = {
        "read_only": True,
        "is_entry_gate": False,
        "selected_market_feature_fetch_count": 48,
    }
    service._last_auto_feature_rank_diagnostics = {
        "candidates": 4,
        "tradable_candidates": 2,
        "secondary_candidates": 1,
        "filtered_out_candidates": 1,
        "rank_underfilled": False,
        "rank_underfill_reason": "",
        "fallback_filtered_fill_count": 1,
        "fallback_filtered_fill_policy": {
            "read_only": True,
            "is_entry_gate": False,
            "applied": True,
            "symbols": ["NEAR/USDT"],
        },
        "filtered_out_reason_counts": [{"reason": "analysis_volume_ratio_below_floor", "count": 1}],
        "symbols": [{"symbol": "BTC/USDT", "score": 80.0}],
        "ranked_symbol_sample": [
            {
                "symbol": "BTC/USDT",
                "selected": True,
                "non_selected_reason": "selected_for_market_analysis",
            },
            {
                "symbol": "ETH/USDT",
                "selected": False,
                "non_selected_reason": "outside_market_symbol_budget",
            },
        ],
        "filtered_symbol_sample": [
            {
                "symbol": "THIN/USDT",
                "selected": False,
                "non_selected_reason": "feature_filter_rejected",
            },
        ],
    }
    skipped = SimpleNamespace(skipped=[SimpleNamespace(symbol="OLD/USDT")])
    empty_filter = SimpleNamespace(skipped=[])
    analysis_budget = {
        "risk_level": "low",
        "market_symbol_limit": 2,
        "position_max_groups": 3,
        "budget_source": "config",
        "market_limit_policy": "position_first_low_risk",
        "configured_market_symbol_limit": 12,
        "position_group_count": 1,
        "target_position_groups": 3,
        "roster_underfilled": True,
        "market_limit_diagnostics": {
            "read_only": True,
            "selected_market_symbol_limit": 2,
        },
        "reason": "position_first_low_risk",
        "recent_market_analysis_dedupe": {
            "skipped_count": 1,
            "skipped_symbols": ["ETH/USDT"],
        },
        "market_budget_rotation": {
            "read_only": True,
            "is_entry_gate": False,
            "applied": True,
            "start_symbol": "ETH/USDT",
        },
    }

    funnel = service._market_candidate_funnel_snapshot(
        scan_symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        open_position_filter=empty_filter,
        unclaimed_filter=skipped,
        fetch_symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        feature_fetch_budget_diagnostics={
            "read_only": True,
            "is_entry_gate": False,
            "selected_market_feature_fetch_count": 48,
        },
        feature_vectors={"BTC/USDT": object(), "ETH/USDT": object()},
        invalid_symbols=["SOL/USDT"],
        market_feature_vectors_before_rank={
            "BTC/USDT": object(),
            "ETH/USDT": object(),
            "SOL/USDT": object(),
        },
        market_feature_vectors_after_rank={"BTC/USDT": object(), "ETH/USDT": object()},
        market_feature_vectors_after_dedupe={"BTC/USDT": object()},
        rank_diagnostics=service._last_auto_feature_rank_diagnostics,
        analysis_budget_context=analysis_budget,
        market_symbol_budget=2,
        run_market_analysis=True,
        mode_is_auto_scan=True,
        analysis_scope="market",
    )
    decision = _decision(Action.HOLD)
    service._attach_market_candidate_funnel(decision, funnel)

    assert funnel["read_only"] is True
    assert funnel["is_entry_gate"] is False
    assert funnel["analysis_scope"] == "market"
    assert funnel["scan_symbol_count"] == 3
    assert funnel["feature_fetch_budget"]["selected_market_feature_fetch_count"] == 48
    assert funnel["feature_valid_count"] == 2
    assert funnel["feature_fetch_budget"]["selected_market_feature_fetch_count"] == 48
    assert funnel["feature_fetch_budget"]["is_entry_gate"] is False
    assert funnel["feature_invalid_count"] == 1
    assert funnel["market_feature_before_rank_count"] == 3
    assert funnel["rank_selected_count"] == 2
    assert funnel["rank_tradable_candidates"] == 2
    assert funnel["rank_filtered_out_candidates"] == 1
    assert funnel["rank_filtered_out_reason_counts"][0]["reason"] == (
        "analysis_volume_ratio_below_floor"
    )
    assert funnel["rank_underfilled"] is False
    assert funnel["rank_underfill_reason"] == ""
    assert "rank_fallback_filtered_fill_count" not in funnel
    assert "rank_fallback_filtered_fill_policy" not in funnel
    assert funnel["ranked_symbol_sample"][1]["non_selected_reason"] == (
        "outside_market_symbol_budget"
    )
    assert funnel["filtered_symbol_sample"][0]["non_selected_reason"] == ("feature_filter_rejected")
    assert funnel["recent_analysis_dedupe_count"] == 1
    assert funnel["market_budget_rotation"]["read_only"] is True
    assert funnel["market_budget_rotation"]["is_entry_gate"] is False
    assert funnel["market_budget_rotation"]["applied"] is True
    assert funnel["market_budget_rotation"]["start_symbol"] == "ETH/USDT"
    assert funnel["market_feature_after_dedupe_count"] == 1
    assert funnel["analysis_budget"]["market_limit_policy"] == "position_first_low_risk"
    assert funnel["analysis_budget"]["configured_market_symbol_limit"] == 12
    assert funnel["analysis_budget"]["position_group_count"] == 1
    assert funnel["analysis_budget"]["target_position_groups"] == 3
    assert funnel["analysis_budget"]["roster_underfilled"] is True
    assert funnel["analysis_budget"]["market_limit_diagnostics"]["read_only"] is True
    assert "threshold" in funnel["diagnostic_boundary"]
    assert decision.raw_response["market_candidate_funnel"] == funnel


def test_market_budget_deferred_rotation_starts_from_skipped_symbol() -> None:
    service = TradingService.__new__(TradingService)
    service._normalize_position_symbol = TradingService._normalize_position_symbol.__get__(
        service,
        TradingService,
    )
    service._market_budget_deferred_symbols = ["SOL/USDT", "XRP/USDT"]
    analysis_budget = {}

    rotated = service._rotate_market_feature_vectors_for_budget_coverage(
        {
            "BTC/USDT": "btc",
            "ETH/USDT": "eth",
            "SOL/USDT": "sol",
            "XRP/USDT": "xrp",
        },
        analysis_budget_context=analysis_budget,
    )

    assert list(rotated) == ["SOL/USDT", "XRP/USDT", "BTC/USDT", "ETH/USDT"]
    rotation = analysis_budget["market_budget_rotation"]
    assert rotation["read_only"] is True
    assert rotation["is_entry_gate"] is False
    assert rotation["applied"] is True
    assert rotation["start_symbol"] == "SOL/USDT"
    assert "thresholds" in rotation["reason"]
    assert "risk gates" in rotation["reason"]


def test_market_budget_deferred_rotation_keeps_order_when_no_match() -> None:
    service = TradingService.__new__(TradingService)
    service._normalize_position_symbol = TradingService._normalize_position_symbol.__get__(
        service,
        TradingService,
    )
    service._market_budget_deferred_symbols = ["DOGE/USDT"]
    analysis_budget = {}

    rotated = service._rotate_market_feature_vectors_for_budget_coverage(
        {
            "BTC/USDT": "btc",
            "ETH/USDT": "eth",
        },
        analysis_budget_context=analysis_budget,
    )

    assert list(rotated) == ["BTC/USDT", "ETH/USDT"]
    rotation = analysis_budget["market_budget_rotation"]
    assert rotation["read_only"] is True
    assert rotation["is_entry_gate"] is False
    assert rotation["applied"] is False
    assert rotation["reason"] == "deferred symbols no longer match current shortlist"




def test_market_budget_deferred_symbols_are_deduped_and_clearable() -> None:
    service = TradingService.__new__(TradingService)
    service._normalize_position_symbol = TradingService._normalize_position_symbol.__get__(
        service,
        TradingService,
    )

    service._remember_market_budget_deferred_symbols(
        ["BTC/USDT", "BTC/USDT", "", "ETH/USDT"],
    )
    assert service._market_budget_deferred_symbols == ["BTC/USDT", "ETH/USDT"]

    service._remember_market_budget_deferred_symbols([])
    assert service._market_budget_deferred_symbols == []




def test_market_ai_budget_clock_ignores_pre_ai_round_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    monkeypatch.setattr(
        trading_service.settings.__class__,
        "refresh_runtime_env",
        lambda _self, force=False: True,
    )
    monkeypatch.setattr(trading_service.settings, "decision_interval_seconds", 30)

    full_round_started_at = datetime.now(UTC) - timedelta(seconds=60)
    market_ai_started_at = datetime.now(UTC) - timedelta(seconds=2)

    assert service._round_budget_exhausted(full_round_started_at) is True
    assert service._market_ai_budget_exhausted(market_ai_started_at) is False

    progress = service._market_analysis_progress_snapshot(
        symbol="ETH/USDT",
        market_index=0,
        market_total=8,
        round_start=full_round_started_at,
        market_ai_started_at=market_ai_started_at,
    )

    assert progress["budget_clock_scope"] == "market_ai_phase"
    assert progress["full_round_elapsed_seconds_before_ai"] >= 60.0
    assert (
        progress["round_elapsed_seconds_before_ai"]
        == progress["full_round_elapsed_seconds_before_ai"]
    )
    assert progress["market_ai_elapsed_seconds_before_symbol"] < 3.0
    assert progress["budget_used_ratio_before_ai"] < 0.12


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






def test_market_round_time_budget_expands_for_profit_first_quality_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    monkeypatch.setattr(
        trading_service.settings.__class__,
        "refresh_runtime_env",
        lambda _self, force=False: True,
    )
    monkeypatch.setattr(trading_service.settings, "decision_interval_seconds", 30)
    strategy_context = {
        "risk_mode": "normal",
        "portfolio_roster": {
            "underfilled": False,
            "gap": 0,
            "current_position_groups": 5,
            "target_position_groups": 5,
            "market_symbol_min": 6,
        },
        "dynamic_position_capacity": {
            "entry_limit": 6,
            "open_group_count": 5,
            "factors": {"rotation_slots": 1},
        },
        "strategy_learning_release_pressure_active": True,
        "profit_first_runtime_feedback": {
            "missed_opportunity_feedback": {
                "entry_bias": "expand_quality_entries",
            },
            "profit_acceptance": {
                "net_pnl": 3.2,
                "profit_factor": 1.28,
            },
        },
    }

    assert (
        service.market_round_time_budget_seconds(
            strategy_context=strategy_context,
            market_symbol_count=8,
        )
        > 27.0
    )

    progress = service._market_analysis_progress_snapshot(
        symbol="BTC/USDT",
        market_index=0,
        market_total=8,
        round_start=datetime.now(UTC) - timedelta(seconds=3),
        market_ai_started_at=datetime.now(UTC) - timedelta(seconds=1),
        strategy_context=strategy_context,
    )

    assert progress["market_round_time_budget_policy"] == "portfolio_roster_underfilled_extension"
    assert progress["market_round_time_budget_seconds"] > progress[
        "base_market_round_time_budget_seconds"
    ]


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


@pytest.mark.asyncio
async def test_market_strategy_mode_uses_cached_learning_without_waiting() -> None:
    service = TradingService.__new__(TradingService)
    service._strategy_learning_context_cache = {
        "paper": {
            "created_at": datetime.now(UTC),
            "context": {
                "strategy": "cached_strategy",
                "current_production_strategy": {"id": "cached_production_strategy"},
                "account_equity": 200.0,
            },
        }
    }
    service._strategy_learning_context_refresh_tasks = {}
    service._current_strategy_mode_context = {"account_equity": 200.0}
    service._last_strategy_context_account_equity_source = ""
    service._okx_balance_snapshot_cache = {}
    service._safe_float = TradingService._safe_float.__get__(service, TradingService)
    service._json_safe_payload = lambda value: value
    service._safe_set_strategy_context_stage = lambda _stage: None
    service.strategy_learning_perf_timeout_seconds = lambda: 0.01  # type: ignore[method-assign]
    service.strategy_learning_account_timeout_seconds = lambda: 0.01  # type: ignore[method-assign]
    service.strategy_learning_context_wait_timeout_seconds = lambda _scope=None: 0.01  # type: ignore[method-assign]
    service._entry_strategy_mode_context_policy = (  # type: ignore[method-assign]
        lambda: SimpleNamespace(
            build=lambda **_kwargs: {
                "strategy": "baseline",
                "current_production_strategy": {"id": "baseline_production_strategy"},
            }
        )
    )
    service.entry_position_exposure = SimpleNamespace(context=lambda _positions: {})
    service.entry_symbol_universe = SimpleNamespace(open_position_group_count=lambda _positions: 0)
    service.daily_performance_service = SimpleNamespace(state=lambda _mode: _async_value({}))
    service._today_side_performance = lambda _mode: _async_value({})  # type: ignore[method-assign]
    service._multiday_side_performance = lambda _mode: _async_value({})  # type: ignore[method-assign]
    service._recent_symbol_side_performance = lambda _mode: _async_value({})  # type: ignore[method-assign]
    service._recent_model_contribution_performance = lambda _mode: _async_value({})  # type: ignore[method-assign]
    service._strategy_context_account_equity = lambda _mode: _async_value(200.0)  # type: ignore[method-assign]

    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()

    class SlowLearning:
        async def apply_to_strategy_context(self, **kwargs):
            refresh_started.set()
            await release_refresh.wait()
            return {
                **kwargs["strategy_context"],
                "current_production_strategy": {"id": "fresh_production_strategy"},
            }

    service.strategy_learning_service = SlowLearning()
    service._refresh_dynamic_capacity = (  # type: ignore[method-assign]
        lambda open_positions, strategy_context, market_regime, account_equity: {
            **strategy_context,
            "account_equity": account_equity,
        }
    )

    token = trading_service._analysis_scope_context.set("market")
    try:
        result = await service._strategy_mode_context("paper", {}, [])
    finally:
        trading_service._analysis_scope_context.reset(token)

    assert result["current_production_strategy"]["id"] == "cached_production_strategy"
    assert result["strategy_learning_cache_status"] == "stale_background_refresh"
    task = service._strategy_learning_context_refresh_tasks.get("paper")
    assert task is not None and not task.done()
    await asyncio.sleep(0)
    assert refresh_started.is_set()
    assert not task.done()
    release_refresh.set()
    await task


def test_decision_snapshot_keeps_governed_strategy_attribution_without_permission() -> None:
    service = TradingService.__new__(TradingService)
    decision = _decision(Action.LONG)
    strategy_context = {
        "scheduler_reason": "highest governed fee-after return LCB",
        "market_regime": {"mode": "trend"},
        "current_production_strategy": {
            "id": "dynamic_fee_after_return_execution",
            "version": "2026-07-15.dynamic-profit-execution.v1",
        },
        "strategy_learning": {
            "scheduler_mode": "governed_dynamic_return",
            "runtime": {"production_influence_enabled": True},
        },
    }

    service._attach_strategy_learning_context(decision, strategy_context)

    snapshot = decision.raw_response["strategy_learning_context"]
    assert snapshot["current_production_strategy"]["id"] == (
        "dynamic_fee_after_return_execution"
    )
    assert "strategy_profile_id" not in snapshot
    assert "strategy_profile_version" not in snapshot
    assert snapshot["scheduler_reason"] == "highest governed fee-after return LCB"
    assert snapshot["production_influence_enabled"] is True
    assert snapshot["production_permission"] is False
    assert snapshot["advisory_prior_only"] is True


@pytest.mark.asyncio
async def test_strategy_context_io_gate_bounds_shared_history_query_fanout() -> None:
    service = TradingService.__new__(TradingService)
    running = 0
    peak_running = 0
    release = asyncio.Event()
    gate_filled = asyncio.Event()

    async def load_value(value: int) -> int:
        nonlocal running, peak_running
        running += 1
        peak_running = max(peak_running, running)
        if running >= trading_service.STRATEGY_CONTEXT_IO_CONCURRENCY:
            gate_filled.set()
        try:
            await release.wait()
            return value
        finally:
            running -= 1

    tasks = [
        asyncio.create_task(
            service._bounded_strategy_context_value(
                f"context_{index}",
                load_value(index),
                None,
                1.0,
            )
        )
        for index in range(3)
    ]
    await asyncio.wait_for(gate_filled.wait(), timeout=0.2)

    assert peak_running == trading_service.STRATEGY_CONTEXT_IO_CONCURRENCY
    release.set()
    assert await asyncio.gather(*tasks) == [0, 1, 2]


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


def test_position_round_watchdog_follows_position_review_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TradingService.__new__(TradingService)
    monkeypatch.setattr(
        trading_service.settings.__class__,
        "refresh_runtime_env",
        lambda _self, force=False: True,
    )
    monkeypatch.setattr(trading_service.settings, "decision_interval_seconds", 30)
    monkeypatch.setattr(trading_service.settings, "position_analysis_watchdog_seconds", 180)
    monkeypatch.setattr(trading_service.settings, "market_analysis_watchdog_seconds", 180)
    monkeypatch.setattr(trading_service.settings, "ai_batch_expert_timeout_seconds", 35.0)
    monkeypatch.setattr(trading_service.settings, "ai_decision_maker_timeout_seconds", 20.0)
    monkeypatch.setattr(trading_service.settings, "local_ai_tools_timeout_seconds", 8.0)

    assert service.position_review_stage_timeout_seconds() == 63.0
    assert service.position_loop_interval_seconds() == pytest.approx(19.5)
    assert service.position_round_watchdog_seconds() == pytest.approx(180.0)




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
async def test_ml_signal_auto_train_skips_when_authoritative_cursor_has_no_new_samples() -> None:
    service = MLSignalService()

    async def completed_shadow_sample_count() -> int:
        return 1050

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
    assert result["new_sample_count"] == 0
    assert result["last_trained_completed_sample_count"] == 1050
    assert result["training_policy"]["learning_only"] is True




@pytest.mark.asyncio
async def test_ml_signal_auto_train_quarantines_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MLSignalService()
    counts = [320, 318]
    calls: list[str] = []

    async def completed_shadow_sample_count() -> int:
        return counts.pop(0) if counts else 318

    async def quarantine_dirty_training_samples(**_kwargs: Any) -> dict[str, Any]:
        calls.append("quarantine")
        return {"scanned": 320, "quarantined": 2}

    def current_metadata() -> dict[str, Any]:
        return {"sample_count": 200, "last_trained_completed_shadow_sample_count": 200}

    async def load_rows(limit: int | None = None) -> list[Any]:
        assert limit is None
        calls.append("load_rows")
        return [object()]

    def quality_report(_rows: list[Any]) -> dict[str, Any]:
        calls.append("quality_report")
        return {"quality_report": {"totals": {"total": 1}}}

    def build_frame(_rows: list[Any]) -> list[Any]:
        calls.append("build_frame")
        return [object(), object()]

    def train_frame(_frame: list[Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(f"train_frame:{bool(kwargs['persist_artifact'])}")
        assert kwargs["completed_sample_count"] == 318
        now = datetime.now(UTC).isoformat()
        return {
            "version": now,
            "trained_at": now,
            "sample_count": 1200,
            "test_count": 240,
            "last_trained_completed_shadow_sample_count": 318,
            "training_run_mode": "persist" if kwargs["persist_artifact"] else "dry_run",
            "artifact_persisted": bool(kwargs["persist_artifact"]),
            "quality_report": {
                "data_quality_version": DATA_QUALITY_VERSION,
                "totals": {"total": 1200, "included": 1200, "downweighted": 0, "excluded": 0},
            },
            "metrics": {
                "long_auc": 0.64,
                "short_auc": 0.63,
                "long_pr_auc": 0.60,
                "short_pr_auc": 0.59,
                "long_accuracy": 0.61,
                "short_accuracy": 0.60,
                "top_long_avg_return_pct": 0.16,
                "bottom_long_avg_return_pct": -0.03,
                "top_short_avg_return_pct": 0.15,
                "bottom_short_avg_return_pct": -0.02,
                "top_long_win_rate": 0.72,
                "bottom_long_win_rate": 0.41,
                "top_short_win_rate": 0.71,
                "bottom_short_win_rate": 0.40,
            },
        }

    service._completed_shadow_sample_count = completed_shadow_sample_count  # type: ignore[method-assign]
    service._current_metadata = current_metadata  # type: ignore[method-assign]
    service._quarantine_dirty_training_samples = quarantine_dirty_training_samples  # type: ignore[method-assign]
    service._ensure_loaded = lambda: None  # type: ignore[method-assign]
    service.artifact_registry = SimpleNamespace(
        promote_candidate=lambda _evidence: SimpleNamespace(version="candidate-v1")
    )
    monkeypatch.setattr(trading_service, "datetime", datetime)
    monkeypatch.setattr("services.ml_signal_service.load_shadow_training_rows", load_rows)
    monkeypatch.setattr("services.ml_signal_service.shadow_training_quality_report", quality_report)
    monkeypatch.setattr("services.ml_signal_service.build_training_frame", build_frame)
    monkeypatch.setattr("services.ml_signal_service.train_from_frame", train_frame)

    result = await service.maybe_auto_train(force=True)

    assert result["trained"] is True
    assert result["completed_sample_count"] == 318
    assert result["training_quarantine"]["quarantined"] == 2
    assert calls[:4] == ["quarantine", "load_rows", "quality_report", "build_frame"]
    assert calls[-2:] == ["train_frame:False", "train_frame:True"]


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

    decision = _decision(Action.LONG)
    decision.raw_response = {"opportunity_score": {"score": 0.0}}
    result = await EntryPolicy(decision_freshness=FakeFreshness()).evaluate(
        decision,
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
async def test_sync_service_close_fill_timeout_is_degraded_not_round_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_errors: list[str] = []
    monkeypatch.setattr(sync_module, "EXCHANGE_CLOSE_FILL_LOOKUP_TIMEOUT_SECONDS", 0.001)
    service = OkxSyncService(round_error_recorder=recorded_errors.append)

    async def slow_close_fill(_position):
        await asyncio.sleep(0.05)
        return {"order_id": "should-not-finish"}

    result = await service._find_exchange_close_fill_with_timeout(
        slow_close_fill,
        SimpleNamespace(id=21, symbol="BTC/USDT", side="long"),
        context="unit test",
    )

    assert result == {"lookup_unavailable": True, "error": "timeout"}
    assert recorded_errors == []
    degraded_rows = service._with_reconcile_degraded_rows([])
    assert degraded_rows == [
        {
            "kind": "close_fill_lookup_unavailable",
            "source": "okx_authoritative_current_position",
            "requires_attention": False,
            "degraded": True,
            "note": (
                "exchange close-fill lookup timed out during unit test; "
                "skipping local close reconciliation for this position this round"
            ),
            "symbol": "BTC/USDT",
            "side": "long",
            "error": "timeout",
        }
    ]


@pytest.mark.asyncio
async def test_sync_service_close_fill_lookup_defers_when_reconcile_budget_is_low() -> None:
    recorded_errors: list[str] = []
    service = OkxSyncService(round_error_recorder=recorded_errors.append)
    service._reconcile_deadline_monotonic = asyncio.get_running_loop().time() + 0.1
    called = False

    async def should_not_call(_position):
        nonlocal called
        called = True
        return {"order_id": "unexpected"}

    result = await service._find_exchange_close_fill_with_timeout(
        should_not_call,
        SimpleNamespace(id=22, symbol="ETH/USDT", side="short"),
        context="missing exchange position",
    )

    assert result == {"lookup_unavailable": True, "error": "deadline_budget_exhausted"}
    assert called is False
    assert recorded_errors == []
    degraded_rows = service._with_reconcile_degraded_rows([])
    assert degraded_rows[0]["kind"] == "close_fill_lookup_deferred"
    assert degraded_rows[0]["requires_attention"] is False
    assert degraded_rows[0]["degraded"] is True
    assert degraded_rows[0]["symbol"] == "ETH/USDT"
    assert degraded_rows[0]["side"] == "short"


def test_okx_authoritative_degraded_optional_rows_do_not_block_entries() -> None:
    service = TradingService.__new__(TradingService)
    service._okx_authoritative_sync_status_payload = lambda _now=None: {  # type: ignore[method-assign]
        "status": "ok",
        "last_error": None,
        "last_requires_attention_count": 0,
        "last_samples": [
            {
                "kind": "close_fill_lookup_unavailable",
                "requires_attention": False,
                "degraded": True,
                "symbol": "BTC/USDT",
            }
        ],
    }

    assert service._okx_authoritative_sync_entry_block_reason() is None


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
async def test_sync_service_close_fill_lookup_timeout_is_phase_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []
    service = OkxSyncService(round_error_recorder=errors.append)
    monkeypatch.setattr(sync_module, "EXCHANGE_CLOSE_FILL_LOOKUP_TIMEOUT_SECONDS", 0.001)

    async def slow_lookup(_position):
        await asyncio.sleep(0.05)
        return {"order_id": "late-close"}

    result = await service._find_exchange_close_fill_with_timeout(
        slow_lookup,
        SimpleNamespace(id=7, symbol="LINK/USDT", side="short"),
        context="missing exchange position",
    )

    assert result == {"lookup_unavailable": True, "error": "timeout"}
    assert errors == []
    degraded_rows = service._with_reconcile_degraded_rows([])
    assert degraded_rows[0]["kind"] == "close_fill_lookup_unavailable"
    assert degraded_rows[0]["requires_attention"] is False
    assert degraded_rows[0]["degraded"] is True
    assert degraded_rows[0]["symbol"] == "LINK/USDT"
    assert degraded_rows[0]["side"] == "short"


@pytest.mark.asyncio
async def test_sync_service_missing_market_symbol_close_fill_returns_empty_without_round_error() -> (
    None
):
    errors: list[str] = []
    service = OkxSyncService(round_error_recorder=errors.append)

    async def missing_market_lookup(_position):
        raise RuntimeError("okx does not have market symbol NG/USDT:USDT")

    result = await service._find_exchange_close_fill_with_timeout(
        missing_market_lookup,
        SimpleNamespace(id=1599, symbol="NG/USDT", side="short"),
        context="missing exchange position",
    )

    assert result == {}
    assert errors == []


@pytest.mark.asyncio
async def test_sync_service_missing_market_symbol_active_order_returns_no_active_order():
    class FakeExecutor:
        async def get_open_orders_strict(self, _symbol):
            raise RuntimeError("okx does not have market symbol NG/USDT:USDT")

    async def okx_executor(_mode):
        return FakeExecutor()

    service = OkxSyncService(
        symbol_normalizer=lambda symbol: str(symbol or ""),
        okx_executor_provider=okx_executor,
    )

    active = await service.active_exchange_order_for_local_position(
        SimpleNamespace(
            id=1599,
            execution_mode="paper",
            symbol="NG/USDT",
            side="short",
        )
    )

    assert active is None


@pytest.mark.asyncio
async def test_sync_service_quarantines_orphan_local_position_without_close_fill(
    monkeypatch: pytest.MonkeyPatch,
):
    created_at = datetime.now(UTC) - timedelta(days=2)
    local_position = SimpleNamespace(
        id=1599,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="NG/USDT",
        side="short",
        is_open=True,
        entry_price=3.339,
        current_price=3.333,
        quantity=1.0,
        leverage=2.0,
        unrealized_pnl=0.006,
        realized_pnl=0.0,
        stop_loss_price=3.39,
        take_profit_price=3.094,
        created_at=created_at,
        closed_at=None,
    )
    created_orders: list[dict[str, Any]] = []
    decision_logs: list[dict[str, Any]] = []
    quarantine_reflections: list[Any] = []

    class FakeSession:
        async def refresh(self, _pos):
            return None

        def add(self, row):
            quarantine_reflections.append(row)

    class FakePaperOKX:
        async def get_positions_strict(self):
            return []

    class FakeExecutor:
        async def get_open_orders_strict(self, _symbol):
            raise RuntimeError("okx does not have market symbol NG/USDT:USDT")

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            return [local_position]

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

    async def close_fill(_pos):
        raise RuntimeError("okx does not have market symbol NG/USDT:USDT")

    async def okx_executor(_mode):
        return FakeExecutor()

    async def entry_fee(_session, _pos, close_qty):
        assert close_qty == 1.0
        return 0.0016695

    async def log_close_decision(**kwargs):
        decision_logs.append(kwargs)
        return 8801

    async def fresh_feature_vector(_symbol):
        return SimpleNamespace(current_price=3.333)

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    result = await OkxSyncService(
        symbol_normalizer=lambda symbol: str(symbol or ""),
        okx_executor_provider=okx_executor,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position: bool(position),
        paper_okx_provider=lambda: FakePaperOKX(),
        exchange_protection_map_provider=protection_map,
        position_protection_fallback_provider=fallback_protection,
        local_position_snapshot_syncer=lambda _positions, **_kwargs: False,
        datetime_from_ms_parser=lambda _timestamp_ms: datetime.now(UTC),
        exchange_close_fill_finder=close_fill,
        fresh_feature_vector_provider=fresh_feature_vector,
        market_value_reader=lambda source, key: getattr(source, key, None),
        entry_fee_provider=entry_fee,
        exchange_sync_close_decision_logger=log_close_decision,
        trade_reflection_recorder=lambda *_args, **_kwargs: None,
        position_margin_calculator=lambda notional, leverage: notional / float(leverage or 1.0),
        memory_position_remover=lambda _model_name, _symbol, _side: None,
    ).reconcile_exchange_positions()

    assert local_position.is_open is False
    assert local_position.realized_pnl == 0.0
    assert local_position.unrealized_pnl == 0.0
    assert local_position.closed_at is not None
    assert local_position.close_exchange_order_id == "okx_orphan_quarantine:1599"
    assert created_orders == []
    assert decision_logs == []
    assert len(quarantine_reflections) == 1
    assert quarantine_reflections[0].source == sync_module.ORPHAN_QUARANTINE_REFLECTION_SOURCE
    assert quarantine_reflections[0].expert_lessons["training_policy"] == "exclude_until_manual_trust"
    assert result[0]["kind"] == "orphan_local_position_quarantined"
    assert result[0]["source"] == "okx_authoritative_current_position"
    assert result[0]["requires_attention"] is False
    assert result[0]["training_policy"] == "exclude_until_manual_trust"
    assert "quarantined" in result[0]["note"]


@pytest.mark.asyncio
async def test_sync_service_quarantines_position_created_from_entry_close_fill(
    monkeypatch: pytest.MonkeyPatch,
):
    created_at = datetime.now(UTC)
    local_position = SimpleNamespace(
        id=4174,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="LINK/USDT",
        side="long",
        is_open=True,
        entry_price=7.9,
        current_price=7.895,
        quantity=3.8,
        leverage=1.0,
        unrealized_pnl=-0.019,
        realized_pnl=0.0,
        stop_loss_price=None,
        take_profit_price=None,
        entry_exchange_order_id="3730002078048428032",
        close_exchange_order_id=None,
        created_at=created_at,
        closed_at=None,
    )
    entry_order = SimpleNamespace(
        execution_mode="paper",
        exchange_order_id="3730002078048428032",
        okx_sync_status="okx_confirmed",
        okx_fill_contracts=3.8,
        okx_fill_pnl=-0.5472,
        okx_raw_fills={
            "fills_history_confirmed": True,
            "order_id": "3730002078048428032",
            "inst_id": "LINK-USDT-SWAP",
            "contracts": 3.8,
            "fill_pnl": -0.5472,
        },
    )
    quarantine_reflections: list[Any] = []
    close_fill_calls: list[Any] = []

    class FakeScalarResult:
        def all(self):
            return [entry_order]

    class FakeExecuteResult:
        def scalars(self):
            return FakeScalarResult()

    class FakeSession:
        async def refresh(self, _pos):
            return None

        async def execute(self, _statement):
            return FakeExecuteResult()

        def add(self, row):
            quarantine_reflections.append(row)

    class FakePaperOKX:
        async def get_positions_strict(self):
            return []

    class FakeExecutor:
        async def get_open_orders_strict(self, _symbol):
            return []

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            return [local_position]

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    async def close_fill(pos):
        close_fill_calls.append(pos)
        return {}

    async def okx_executor(_mode):
        return FakeExecutor()

    async def protection_map(_paper_okx, _exchange_positions):
        return {}

    async def fallback_protection(_session, **_kwargs):
        return {}

    async def fresh_feature_vector(_symbol):
        return SimpleNamespace(current_price=7.895)

    async def entry_fee(_session, _pos, _close_qty):
        return 0.0

    async def log_close_decision(**_kwargs):
        return None

    async def record_reflection(*_args, **_kwargs):
        return None

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    result = await OkxSyncService(
        symbol_normalizer=lambda symbol: str(symbol or ""),
        okx_executor_provider=okx_executor,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position: bool(position),
        paper_okx_provider=lambda: FakePaperOKX(),
        exchange_protection_map_provider=protection_map,
        position_protection_fallback_provider=fallback_protection,
        local_position_snapshot_syncer=lambda _positions, **_kwargs: False,
        datetime_from_ms_parser=lambda _timestamp_ms: datetime.now(UTC),
        exchange_close_fill_finder=close_fill,
        fresh_feature_vector_provider=fresh_feature_vector,
        market_value_reader=lambda source, key: getattr(source, key, None),
        entry_fee_provider=entry_fee,
        exchange_sync_close_decision_logger=log_close_decision,
        trade_reflection_recorder=record_reflection,
        position_margin_calculator=lambda notional, leverage: notional / float(leverage or 1.0),
        memory_position_remover=lambda _model_name, _symbol, _side: None,
    ).reconcile_exchange_positions()

    assert close_fill_calls == []
    assert local_position.is_open is False
    assert local_position.close_exchange_order_id == "okx_orphan_quarantine:4174"
    assert quarantine_reflections[0].expert_lessons["repair_plan"]["reason"].startswith(
        "The local entry order is an OKX-confirmed close/reduce fill"
    )
    assert result[0]["kind"] == "entry_order_close_fill_position_quarantined"
    assert result[0]["requires_attention"] is False
    assert result[0]["training_policy"] == "exclude_until_manual_trust"


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
            "kind": "snapshot_update",
            "source": "okx_authoritative_current_position",
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
async def test_sync_service_reconcile_exchange_positions_matches_okx_net_mode_position(
    monkeypatch: pytest.MonkeyPatch,
):
    local_position = SimpleNamespace(
        id=1706,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="SPK/USDT",
        side="short",
        is_open=True,
        quantity=100.0,
        entry_price=0.0179,
        current_price=0.01769,
        unrealized_pnl=0.0,
        stop_loss_price=None,
        take_profit_price=None,
    )
    sync_calls: list[dict[str, Any]] = []
    close_fill_calls: list[Any] = []

    class FakePaperOKX:
        async def get_positions_strict(self):
            return [
                {
                    "info": {
                        "instId": "SPK-USDT-SWAP",
                        "posSide": "net",
                        "pos": "-200",
                        "ctVal": "1",
                        "avgPx": "0.01785",
                        "markPx": "0.0177",
                        "last": "0.01762",
                        "upl": "0.0300000000000002",
                        "posId": "3688338318498172929",
                    }
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
        return {}

    async def fallback_protection(_session, **_kwargs):
        return {}

    def sync_snapshot(positions, **kwargs):
        sync_calls.append({"positions": positions, "kwargs": kwargs})
        positions[0].quantity = kwargs["exchange_quantity"]
        return True

    async def close_fill(pos):
        close_fill_calls.append(pos)
        return {}

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    result = await OkxSyncService(
        symbol_normalizer=normalize_trading_symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position: bool(
            (position.get("info") or {}).get("pos")
        ),
        paper_okx_provider=lambda: FakePaperOKX(),
        exchange_protection_map_provider=protection_map,
        position_protection_fallback_provider=fallback_protection,
        local_position_snapshot_syncer=sync_snapshot,
        datetime_from_ms_parser=lambda _timestamp_ms: datetime.now(UTC),
        exchange_close_fill_finder=close_fill,
        fresh_feature_vector_provider=lambda _symbol: None,
        market_value_reader=lambda source, key: getattr(source, key, None),
        entry_fee_provider=lambda *_args: 0.0,
        exchange_sync_close_decision_logger=lambda **_kwargs: None,
        trade_reflection_recorder=lambda *_args, **_kwargs: None,
        position_margin_calculator=lambda notional, leverage: notional / float(leverage or 1.0),
        memory_position_remover=lambda _model_name, _symbol, _side: None,
    ).reconcile_exchange_positions()

    assert close_fill_calls == []
    assert local_position.quantity == pytest.approx(200.0)
    assert sync_calls[0]["kwargs"]["exchange_quantity"] == pytest.approx(200.0)
    assert sync_calls[0]["kwargs"]["current_price"] == pytest.approx(0.0177)
    assert sync_calls[0]["kwargs"]["entry_price"] == pytest.approx(0.01785)
    assert result[0]["kind"] == "snapshot_update"
    assert result[0]["symbol"] == "SPK/USDT"
    assert result[0]["side"] == "short"
    assert result[0]["quantity"] == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_sync_service_reconcile_exchange_positions_records_exchange_quantity_reduction(
    monkeypatch: pytest.MonkeyPatch,
):
    created_at = datetime(2026, 6, 24, 5, 20, tzinfo=UTC)
    closed_at = datetime(2026, 6, 24, 5, 26, tzinfo=UTC)
    local_position = SimpleNamespace(
        id=31,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="USAR/USDT",
        side="long",
        is_open=True,
        entry_price=2.31,
        current_price=2.44,
        quantity=16.0,
        leverage=6.0,
        unrealized_pnl=2.08,
        realized_pnl=0.0,
        stop_loss_price=2.0,
        take_profit_price=4.2,
        created_at=created_at,
        closed_at=None,
    )
    closed_positions: list[Any] = []
    created_orders: list[dict[str, Any]] = []
    balance_updates: list[tuple[str, float, float]] = []
    trade_results: list[tuple[str, bool]] = []
    decision_logs: list[dict[str, Any]] = []
    reflection_calls: list[dict[str, Any]] = []
    close_fill_probes: list[Any] = []

    class FakeScalarResult:
        def scalar_one_or_none(self):
            return None

    class FakeSession:
        async def execute(self, _statement):
            return FakeScalarResult()

    class FakePaperOKX:
        async def get_positions_strict(self):
            return [
                {
                    "symbol": "USAR/USDT",
                    "side": "long",
                    "contracts": "6",
                    "contractSize": "1",
                    "entryPrice": "2.31",
                    "markPrice": "4.05",
                    "leverage": "6",
                    "unrealizedPnl": "10.44",
                }
            ]

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            return [local_position]

        async def open_position(self, payload):
            closed_position = SimpleNamespace(id=99, **payload)
            closed_positions.append(closed_position)
            return closed_position

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

    def sync_snapshot(positions, **_kwargs):
        positions[0].quantity = 6.0
        return True

    async def close_fill(pos):
        close_fill_probes.append(pos)
        return {
            "order_id": "usar-close-10",
            "price": 3.85,
            "fee": 0.02,
            "quantity": 10.0,
            "timestamp": closed_at,
        }

    async def entry_fee(_session, _pos, close_qty):
        assert close_qty == 10.0
        return 0.01

    async def log_close_decision(**kwargs):
        decision_logs.append(kwargs)
        return 909

    async def record_reflection(_session, pos, **kwargs):
        reflection_calls.append({"pos": pos, "kwargs": kwargs})

    def position_margin(notional, leverage):
        return notional / leverage

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
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
        exchange_close_fill_finder=close_fill,
        fresh_feature_vector_provider=lambda _symbol: None,
        market_value_reader=lambda source, key: getattr(source, key, None),
        entry_fee_provider=entry_fee,
        exchange_sync_close_decision_logger=log_close_decision,
        trade_reflection_recorder=record_reflection,
        position_margin_calculator=position_margin,
        memory_position_remover=lambda _model_name, _symbol, _side: None,
    ).reconcile_exchange_positions()

    assert close_fill_probes[0].symbol == "USAR/USDT"
    assert close_fill_probes[0].side == "long"
    assert close_fill_probes[0].quantity == 10.0
    assert local_position.quantity == 6.0
    assert len(closed_positions) == 1
    closed = closed_positions[0]
    assert closed.quantity == 10.0
    assert closed.entry_price == 2.31
    assert closed.current_price == 3.85
    assert closed.realized_pnl == pytest.approx(15.37)
    assert closed.closed_at == closed_at
    assert closed.okx_inst_id == "USAR-USDT-SWAP"
    assert closed.close_exchange_order_id == "usar-close-10"
    assert decision_logs[0]["position_size_pct"] == pytest.approx(10.0 / 16.0)
    assert decision_logs[0]["close_fill"]["partial_reduction"] is True
    assert decision_logs[0]["close_fill"]["order_id"] == "usar-close-10"
    assert reflection_calls[0]["kwargs"]["source"] == "okx_reconcile"
    assert len(created_orders) == 1
    assert created_orders[0]["model_name"] == "ensemble_trader"
    assert created_orders[0]["execution_mode"] == "paper"
    assert created_orders[0]["symbol"] == "USAR/USDT"
    assert created_orders[0]["side"] == "sell"
    assert created_orders[0]["order_type"] == "market"
    assert created_orders[0]["quantity"] == 10.0
    assert created_orders[0]["price"] == 3.85
    assert created_orders[0]["status"] == OrderStatus.FILLED.value
    assert created_orders[0]["fee"] == 0.02
    assert created_orders[0]["decision_id"] == 909
    assert created_orders[0]["exchange_order_id"] == "usar-close-10"
    assert created_orders[0]["filled_at"] == closed_at
    assert created_orders[0]["okx_inst_id"] == "USAR-USDT-SWAP"
    assert created_orders[0]["okx_state"] == "filled"
    assert created_orders[0]["okx_sync_status"] == "okx_confirmed"
    assert isinstance(created_orders[0]["okx_synced_at"], datetime)
    assert balance_updates == []
    assert trade_results == []
    assert result[0]["kind"] == "quantity_reduction_closed_slice"
    assert result[0]["source"] == "okx_authoritative_current_position"
    assert result[0]["quantity"] == 10.0
    assert result[0]["remaining_quantity"] == 6.0
    assert result[0]["exchange_order_id"] == "usar-close-10"


@pytest.mark.asyncio
async def test_sync_service_quantity_reduction_uses_close_fill_inst_id_over_local_alias():
    created_at = datetime(2026, 6, 24, 5, 20, tzinfo=UTC)
    local_position = SimpleNamespace(
        id=41,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="SAHARA/USDT",
        side="long",
        is_open=True,
        entry_price=0.012,
        current_price=0.013,
        quantity=6.0,
        leverage=3.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        stop_loss_price=0.01,
        take_profit_price=0.02,
        created_at=created_at,
        closed_at=None,
    )
    closed_positions: list[Any] = []
    created_orders: list[dict[str, Any]] = []

    class FakeScalarResult:
        def scalar_one_or_none(self):
            return None

    class FakeSession:
        async def execute(self, _statement):
            return FakeScalarResult()

    class FakeTradeRepository:
        async def open_position(self, payload):
            closed_position = SimpleNamespace(id=199, **payload)
            closed_positions.append(closed_position)
            return closed_position

        async def create_order(self, payload):
            created_orders.append(payload)

    async def log_close_decision(**_kwargs):
        return 991

    async def record_reflection(*_args, **_kwargs):
        return None

    async def entry_fee(*_args):
        return 0.0

    result = await OkxSyncService()._record_exchange_quantity_reduction(
        session=FakeSession(),
        trade_repo=FakeTradeRepository(),
        positions=[local_position],
        quantity_before_by_id={41: 10.0},
        exchange_quantity=6.0,
        exit_price=0.013,
        close_fill={
            "order_id": "spk-close-4",
            "price": 0.013,
            "fee": 0.001,
            "quantity": 4.0,
            "timestamp": datetime(2026, 6, 24, 5, 26, tzinfo=UTC),
            "order_info": {"instId": "SPK-USDT-SWAP", "ordId": "spk-close-4"},
        },
        entry_fee_for_position=entry_fee,
        log_exchange_sync_close_decision=log_close_decision,
        record_trade_reflection=record_reflection,
        calculate_position_margin=lambda notional, leverage: notional / float(leverage or 1.0),
    )

    assert closed_positions[0].symbol == "SPK/USDT"
    assert closed_positions[0].okx_inst_id == "SPK-USDT-SWAP"
    assert created_orders[0]["symbol"] == "SPK/USDT"
    assert created_orders[0]["exchange_order_id"] == "spk-close-4"
    assert result[0]["symbol"] == "SPK/USDT"


@pytest.mark.asyncio
async def test_sync_service_does_not_record_quantity_reduction_without_close_fill():
    pos = SimpleNamespace(
        id=1700,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="LAB/USDT",
        side="long",
        entry_price=16.865555555555556,
        current_price=18.0,
        quantity=0.9,
        leverage=3.0,
        realized_pnl=0.0,
        stop_loss_price=14.3,
        take_profit_price=22.463,
        created_at=datetime(2026, 6, 25, 9, 50, tzinfo=UTC),
        closed_at=None,
    )
    opened_positions: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []

    class FakeTradeRepository:
        async def open_position(self, payload):
            opened_positions.append(payload)
            return SimpleNamespace(id=1701, **payload)

        async def create_order(self, payload):
            orders.append(payload)

    result = await OkxSyncService()._record_exchange_quantity_reduction(
        session=object(),
        trade_repo=FakeTradeRepository(),
        positions=[pos],
        quantity_before_by_id={1700: 9.0},
        exchange_quantity=0.9,
        exit_price=18.0,
        close_fill=None,
        entry_fee_for_position=lambda *_args: 0.0,
        log_exchange_sync_close_decision=lambda **_kwargs: None,
        record_trade_reflection=lambda *_args, **_kwargs: None,
        calculate_position_margin=lambda notional, leverage: notional / float(leverage or 1.0),
    )

    assert result == []
    assert opened_positions == []
    assert orders == []


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
            "okx_inst_id": "BTC-USDT-SWAP",
            "entry_exchange_order_id": "entry-order-1",
        }
    ]
    assert result == [
        {
            "kind": "created_missing_local_position",
            "source": "okx_authoritative_current_position",
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 100.0,
            "exchange_order_id": "entry-order-1",
            "note": "OKX 已有持仓但本地缺失，已按执行订单补回持仓记录。",
        }
    ]


@pytest.mark.asyncio
async def test_sync_service_created_missing_position_uses_okx_inst_id_over_alias(
    monkeypatch: pytest.MonkeyPatch,
):
    opened_positions: list[dict[str, Any]] = []
    order = SimpleNamespace(
        model_name="ensemble_trader",
        exchange_order_id="spk-entry-1",
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
                    "symbol": "SAHARA/USDT:USDT",
                    "info": {
                        "instId": "SPK-USDT-SWAP",
                        "posSide": "long",
                        "pos": "10",
                        "ctVal": "1",
                        "avgPx": "0.012",
                        "markPx": "0.013",
                        "upl": "0.01",
                        "cTime": "1770379200000",
                    },
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

    async def protection_map(_paper_okx, _positions):
        return {}

    async def fallback_protection(_session, **_kwargs):
        return {}

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    result = await OkxSyncService(
        symbol_normalizer=normalize_trading_symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        exchange_position_open_checker=lambda position: bool((position.get("info") or {}).get("pos")),
        paper_okx_provider=lambda: FakePaperOKX(),
        exchange_protection_map_provider=protection_map,
        position_protection_fallback_provider=fallback_protection,
        local_position_snapshot_syncer=lambda _positions, **_kwargs: False,
        datetime_from_ms_parser=lambda _timestamp_ms: datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
        **_noop_reconcile_close_boundaries(),
    ).reconcile_exchange_positions()

    assert opened_positions[0]["symbol"] == "SPK/USDT"
    assert opened_positions[0]["okx_inst_id"] == "SPK-USDT-SWAP"
    assert result[0]["symbol"] == "SPK/USDT"


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
    assert balance_updates == []
    assert trade_results == []
    assert removed_positions == [("ensemble_trader", "BTC/USDT", "long")]
    assert position.is_open is False
    assert position.current_price == 112.0
    assert position.realized_pnl == 22.0
    assert created_orders[0]["decision_id"] == 42
    assert created_orders[0]["fee"] == 0.5
    assert result == [
        {
            "kind": "closed_from_okx_close_fill",
            "source": "okx_authoritative_current_position",
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
async def test_sync_service_reconcile_exchange_positions_does_not_estimate_missing_close_fill(
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

    assert fresh_calls == []
    assert market_value_calls == []
    assert decision_logs == []
    assert created_orders == []
    assert position.is_open is False
    assert position.current_price == 104.0
    assert position.realized_pnl == 0.0
    assert position.unrealized_pnl == 0.0
    assert position.close_exchange_order_id == "okx_orphan_quarantine:22"
    assert result[0]["kind"] == "orphan_local_position_quarantined"
    assert result[0]["source"] == "okx_authoritative_current_position"
    assert result[0]["requires_attention"] is False
    assert result[0]["training_policy"] == "exclude_until_manual_trust"
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
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    service = OkxSyncService(
        position_profit_peak_recorder=lambda **kwargs: peak_calls.append(kwargs),
        position_age_minutes_provider=lambda created_at: 12.5 if created_at else None,
        position_profit_peak_pruner=lambda open_context: pruned_contexts.append(open_context),
    )

    await service.refresh_position_prices({"BTC/USDT": SimpleNamespace(current_price=110.0)})

    assert updated_prices == [(7, 110.0, 5.0)]
    assert account_updates == []
    assert peak_calls[0]["symbol"] == "BTC/USDT"
    assert peak_calls[0]["unrealized_pnl"] == 5.0
    assert peak_calls[0]["hold_minutes"] == 12.5
    assert pruned_contexts[0][0]["symbol"] == "BTC/USDT"


@pytest.mark.asyncio
async def test_refresh_position_prices_batches_loaded_position_updates(
    monkeypatch: pytest.MonkeyPatch,
):
    batch_updates: list[list[tuple[int, float, float]]] = []

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            return [
                SimpleNamespace(
                    id=71,
                    model_name="ensemble_trader",
                    symbol="BTC/USDT",
                    side="long",
                    entry_price=100.0,
                    current_price=99.0,
                    quantity=0.5,
                    created_at="created",
                    is_open=True,
                ),
                SimpleNamespace(
                    id=72,
                    model_name="ensemble_trader",
                    symbol="ETH/USDT",
                    side="short",
                    entry_price=200.0,
                    current_price=201.0,
                    quantity=0.25,
                    created_at="created",
                    is_open=True,
                ),
            ]

        async def update_open_position_prices(self, updates):
            batch_updates.append(
                [(pos.id, current_price, unrealized_pnl) for pos, current_price, unrealized_pnl in updates]
            )

        async def update_position_price(self, *_args):
            raise AssertionError("single-position update path should not be used")

    @asynccontextmanager
    async def fake_session_ctx():
        yield object()

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    service = OkxSyncService(
        position_profit_peak_recorder=lambda **_kwargs: None,
        position_age_minutes_provider=lambda _created_at: 12.5,
        position_profit_peak_pruner=lambda _open_context: None,
    )

    result = await service.refresh_position_prices(
        {
            "BTC/USDT": SimpleNamespace(current_price=110.0),
            "ETH/USDT": SimpleNamespace(current_price=190.0),
        }
    )

    assert batch_updates == [[(71, 110.0, 5.0), (72, 190.0, 2.5)]]
    assert result["updated_position_count"] == 2
    assert result["persist_updates_ms"] >= 0


@pytest.mark.asyncio
async def test_refresh_position_prices_fetches_active_and_paper_snapshots_concurrently(
    monkeypatch: pytest.MonkeyPatch,
):
    active_started = asyncio.Event()
    paper_started = asyncio.Event()
    active_saw_paper = False

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            return []

    class ActiveOkx:
        async def get_positions_strict(self):
            nonlocal active_saw_paper
            active_started.set()
            await asyncio.wait_for(paper_started.wait(), timeout=0.1)
            active_saw_paper = True
            return []

    class PaperOkx:
        async def get_positions_strict(self):
            paper_started.set()
            await asyncio.wait_for(active_started.wait(), timeout=0.1)
            return []

    @asynccontextmanager
    async def fake_session_ctx():
        yield object()

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)
    service = OkxSyncService(
        active_okx_provider=ActiveOkx,
        paper_okx_provider=PaperOkx,
        position_profit_peak_recorder=lambda **_kwargs: None,
        position_age_minutes_provider=lambda _created_at: 12.5,
        position_profit_peak_pruner=lambda _open_context: None,
    )

    await service.refresh_position_prices({})

    assert active_saw_paper is True


@pytest.mark.asyncio
async def test_refresh_position_prices_defers_database_write_when_pool_is_busy(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            await asyncio.sleep(0.05)
            return []

    @asynccontextmanager
    async def fake_session_ctx():
        yield object()

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)
    monkeypatch.setattr(sync_module, "POSITION_PRICE_REFRESH_DB_LOAD_TIMEOUT_SECONDS", 0.01)
    service = OkxSyncService(
        position_profit_peak_recorder=lambda **_kwargs: None,
        position_age_minutes_provider=lambda _created_at: 12.5,
        position_profit_peak_pruner=lambda _open_context: None,
    )

    result = await service.refresh_position_prices({})

    assert result["database_deferred"]["reason"] == (
        "connection_pool_or_open_position_query_timeout"
    )
    assert result["database_deferred"]["timeout_seconds"] == 0.01


@pytest.mark.asyncio
async def test_refresh_position_prices_prefers_okx_position_mark_and_upl(
    monkeypatch: pytest.MonkeyPatch,
):
    updated_prices: list[tuple[int, float, float]] = []
    peak_calls: list[dict[str, Any]] = []

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            return [
                SimpleNamespace(
                    id=17,
                    model_name="ensemble_trader",
                    symbol="LAB/USDT",
                    side="long",
                    entry_price=16.865555555555556,
                    current_price=17.32,
                    quantity=0.9,
                    created_at="created",
                )
            ]

        async def update_position_price(self, position_id, current_price, unrealized_pnl):
            updated_prices.append((position_id, current_price, unrealized_pnl))

    class FakeAccountRepository:
        def __init__(self, _session):
            pass

        async def update_unrealized_pnl(self, _model_name, _unrealized_pnl):
            pass

    class FakePaperOkx:
        async def get_positions_strict(self):
            return [
                {
                    "symbol": "LAB-USDT-SWAP",
                    "side": "long",
                    "contracts": 9.0,
                    "markPrice": 18.192,
                    "entryPrice": 16.865555555555556,
                    "unrealizedPnl": 1.1937999999999998,
                    "info": {
                        "instId": "LAB-USDT-SWAP",
                        "pos": "9",
                        "avgPx": "16.8655555555555556",
                        "markPx": "18.192",
                        "last": "17.32",
                        "upl": "1.1937999999999998",
                    },
                }
            ]

    @asynccontextmanager
    async def fake_session_ctx():
        yield object()

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    service = OkxSyncService(
        paper_okx_provider=lambda: FakePaperOkx(),
        symbol_normalizer=normalize_trading_symbol,
        position_profit_peak_recorder=lambda **kwargs: peak_calls.append(kwargs),
        position_age_minutes_provider=lambda created_at: 12.5 if created_at else None,
        position_profit_peak_pruner=lambda _open_context: None,
    )

    await service.refresh_position_prices({"LAB/USDT": SimpleNamespace(current_price=17.32)})

    assert updated_prices == [(17, pytest.approx(18.192), pytest.approx(1.1938))]
    assert peak_calls[0]["current_price"] == pytest.approx(18.192)
    assert peak_calls[0]["unrealized_pnl"] == pytest.approx(1.1938)


@pytest.mark.asyncio
async def test_refresh_position_prices_uses_current_mode_okx_without_cross_mode_pollution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    updated_prices: list[tuple[int, float, float]] = []

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_open_positions(self):
            return [
                SimpleNamespace(
                    id=21,
                    model_name="ensemble_trader",
                    execution_mode="live",
                    symbol="ETH/USDT",
                    side="short",
                    entry_price=100.0,
                    current_price=99.0,
                    quantity=2.0,
                    created_at="created-live",
                ),
                SimpleNamespace(
                    id=22,
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ETH/USDT",
                    side="short",
                    entry_price=100.0,
                    current_price=99.0,
                    quantity=2.0,
                    created_at="created-paper",
                ),
            ]

        async def update_position_price(self, position_id, current_price, unrealized_pnl):
            updated_prices.append((position_id, current_price, unrealized_pnl))

    class FakeAccountRepository:
        def __init__(self, _session):
            pass

        async def update_unrealized_pnl(self, _model_name, _unrealized_pnl):
            pass

    class FakeLiveOkx:
        async def get_positions_strict(self):
            return [
                {
                    "symbol": "ETH-USDT-SWAP",
                    "side": "short",
                    "contracts": 2.0,
                    "info": {
                        "instId": "ETH-USDT-SWAP",
                        "posSide": "short",
                        "pos": "-2",
                        "ctVal": "1",
                        "avgPx": "100",
                        "markPx": "90",
                        "upl": "20",
                    },
                }
            ]

    class FakePaperOkx:
        async def get_positions_strict(self):
            return [
                {
                    "symbol": "ETH-USDT-SWAP",
                    "side": "short",
                    "contracts": 2.0,
                    "info": {
                        "instId": "ETH-USDT-SWAP",
                        "posSide": "short",
                        "pos": "-2",
                        "ctVal": "1",
                        "avgPx": "100",
                        "markPx": "95",
                        "upl": "10",
                    },
                }
            ]

    @asynccontextmanager
    async def fake_session_ctx():
        yield object()

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)
    monkeypatch.setattr(mode_manager, "_state_path", tmp_path / "trading-control-state.json")
    monkeypatch.setattr(mode_manager, "_last_state_mtime", 0.0)
    monkeypatch.setattr(mode_manager, "_last_state_size", -1)
    await mode_manager.switch_to_live("ensemble_trader")

    service = OkxSyncService(
        active_okx_provider=lambda: FakeLiveOkx(),
        paper_okx_provider=lambda: FakePaperOkx(),
        symbol_normalizer=normalize_trading_symbol,
        position_profit_peak_recorder=lambda **_kwargs: None,
        position_age_minutes_provider=lambda _created_at: 12.5,
        position_profit_peak_pruner=lambda _open_context: None,
    )

    await service.refresh_position_prices({"ETH/USDT": SimpleNamespace(current_price=80.0)})

    assert updated_prices == [
        (21, pytest.approx(90.0), pytest.approx(20.0)),
        (22, pytest.approx(95.0), pytest.approx(10.0)),
    ]


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
async def test_sync_service_exit_position_lookup_uses_okx_inst_id_when_symbol_is_missing():
    class FakeExecutor:
        async def get_positions_strict(self, symbol):
            assert symbol == "SPK/USDT"
            return [
                {
                    "symbol": "",
                    "side": "",
                    "contracts": None,
                    "info": {
                        "instId": "SPK-USDT-SWAP",
                        "posSide": "long",
                        "pos": "18",
                        "ctVal": "1",
                        "avgPx": "0.012",
                        "markPx": "0.013",
                    },
                }
            ]

    def parse_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    async def okx_executor(mode):
        assert mode == "paper"
        return FakeExecutor()

    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="SPK/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning="close spk",
        position_size_pct=1.0,
        suggested_leverage=3.0,
        raw_response={},
        feature_snapshot={"current_price": 0.013},
    )

    result = await OkxSyncService(
        symbol_normalizer=normalize_trading_symbol,
        model_execution_mode_provider=lambda _model_name: "paper",
        okx_executor_provider=okx_executor,
        float_parser=parse_float,
    ).has_matching_exchange_exit_position("ensemble_trader", decision)

    assert result is True


@pytest.mark.asyncio
async def test_sync_service_exit_position_lookup_fails_fast_without_boundaries():
    service = OkxSyncService()

    with pytest.raises(RuntimeError, match="symbol_normalizer"):
        await service.has_matching_exchange_exit_position(
            "ensemble_trader",
            _decision(Action.CLOSE_LONG),
        )


@pytest.mark.asyncio
async def test_open_positions_context_returns_empty_when_okx_lookup_fails(
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
        active_okx_provider=lambda: FakeOKX(),
        exchange_position_open_checker=lambda position: bool(position),
    ).get_open_positions_context()

    assert result == []
    rendered = str(warnings)
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in rendered
    assert "password=***" in rendered


@pytest.mark.asyncio
async def test_open_positions_context_matches_okx_net_mode_position(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeOKX:
        async def get_positions_strict(self):
            return [
                {
                    "info": {
                        "instId": "SPK-USDT-SWAP",
                        "posSide": "net",
                        "pos": "-200",
                        "ctVal": "1",
                        "avgPx": "0.01785",
                        "markPx": "0.0177",
                        "upl": "0.03",
                    }
                }
            ]

    class FakeTradeRepository:
        def __init__(self, _session):
            pass

        async def get_position_records(self, **_kwargs):
            return [
                SimpleNamespace(
                    model_name="ensemble_trader",
                    symbol="SPK/USDT",
                    side="short",
                    entry_price=0.0179,
                    current_price=0.01769,
                    quantity=100.0,
                    leverage=2.0,
                    unrealized_pnl=0.021,
                    stop_loss_price=None,
                    take_profit_price=None,
                    is_open=True,
                    created_at=None,
                    okx_inst_id="SPK-USDT-SWAP",
                    okx_pos_id="3688338318498172929",
                )
            ]

    @asynccontextmanager
    async def fake_session_ctx():
        yield object()

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    result = await OkxSyncService(
        symbol_normalizer=normalize_trading_symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
        paper_positions_provider=lambda: [],
        active_okx_provider=lambda: FakeOKX(),
        exchange_position_open_checker=lambda position: bool((position.get("info") or {}).get("pos")),
    ).get_open_positions_context()

    assert len(result) == 1
    assert result[0]["symbol"] == "SPK/USDT"
    assert result[0]["side"] == "short"
    assert result[0]["quantity"] == pytest.approx(200.0)
    assert result[0]["entry_price"] == pytest.approx(0.01785)
    assert result[0]["current_price"] == pytest.approx(0.0177)
    assert result[0]["unrealized_pnl"] == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_open_positions_context_fails_fast_without_boundaries():
    service = OkxSyncService()

    with pytest.raises(RuntimeError, match="symbol_normalizer"):
        await service.get_open_positions_context()


@pytest.mark.asyncio
async def test_local_open_positions_context_returns_db_positions_without_okx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                    current_price=101.5,
                    quantity=0.25,
                    leverage=2.0,
                    unrealized_pnl=0.375,
                    stop_loss_price=95.0,
                    take_profit_price=110.0,
                    is_open=True,
                    created_at=None,
                    okx_inst_id="BTC-USDT-SWAP",
                    okx_pos_id="pos-1",
                    entry_exchange_order_id="entry-1",
                )
            ]

    class FakeScalarResult:
        def scalars(self):
            return self

        def all(self):
            return []

    execute_calls: list[str] = []

    class FakeSession:
        async def execute(self, _statement):
            execute_calls.append("execute")
            return FakeScalarResult()

    @asynccontextmanager
    async def fake_session_ctx():
        yield FakeSession()

    monkeypatch.setattr(sync_module, "TradeRepository", FakeTradeRepository)
    monkeypatch.setattr(sync_module, "get_session_ctx", fake_session_ctx)

    result = await OkxSyncService(
        symbol_normalizer=normalize_trading_symbol,
        float_parser=lambda value, default=0.0: default if value is None else float(value),
    ).get_local_open_positions_context(
        strict=True,
    )

    assert len(result) == 1
    position = result[0]
    assert position["model_name"] == "ensemble_trader"
    assert position["symbol"] == "BTC/USDT"
    assert position["side"] == "long"
    assert position["entry_price"] == pytest.approx(100.0)
    assert position["current_price"] == pytest.approx(101.5)
    assert position["quantity"] == pytest.approx(0.25)
    assert position["leverage"] == pytest.approx(2.0)
    assert position["unrealized_pnl"] == pytest.approx(0.375)
    assert position["stop_loss"] == pytest.approx(95.0)
    assert position["take_profit"] == pytest.approx(110.0)
    assert position["okx_inst_id"] == "BTC-USDT-SWAP"
    assert position["okx_pos_id"] == "pos-1"
    assert position["entry_exchange_order_id"] == "entry-1"
    assert execute_calls == []


@pytest.mark.asyncio
async def test_market_round_reuses_short_lived_open_positions_context_cache() -> None:
    service = TradingService.__new__(TradingService)
    service._open_positions_context_cache = {}
    service._open_positions_context_refresh_task = None
    calls = 0

    async def load_positions():
        nonlocal calls
        calls += 1
        return [{"symbol": "BTC/USDT", "side": "long", "is_open": True}]

    service._get_open_positions_context = load_positions  # type: ignore[method-assign]

    first = await service._open_positions_context_for_round("market")
    second = await service._open_positions_context_for_round("market")

    assert first == second == [{"symbol": "BTC/USDT", "side": "long", "is_open": True}]
    assert calls == 1
    task = service._open_positions_context_refresh_task
    if task is not None and not task.done():
        await task


@pytest.mark.asyncio
async def test_position_round_refreshes_open_positions_context_authoritatively() -> None:
    service = TradingService.__new__(TradingService)
    service._open_positions_context_cache = {
        "created_at": datetime.now(UTC),
        "positions": [{"symbol": "OLD/USDT", "side": "long", "is_open": True}],
    }
    service._open_positions_context_refresh_task = None
    calls = 0

    async def load_positions():
        nonlocal calls
        calls += 1
        return [{"symbol": "ETH/USDT", "side": "short", "is_open": True}]

    service._get_open_positions_context = load_positions  # type: ignore[method-assign]

    result = await service._open_positions_context_for_round("position")

    assert result == [{"symbol": "ETH/USDT", "side": "short", "is_open": True}]
    assert calls == 1
    assert service._open_positions_context_cache["positions"] == result


@pytest.mark.asyncio
async def test_market_round_uses_local_positions_after_fresh_okx_sync() -> None:
    service = TradingService.__new__(TradingService)
    service._open_positions_context_cache = {}
    service._open_positions_context_refresh_task = None
    service.okx_authoritative_sync_interval_seconds = lambda: 20.0  # type: ignore[method-assign]
    service._okx_authoritative_sync_task = None
    service._okx_authoritative_sync_last_success_at = datetime.now(UTC)
    service._okx_authoritative_sync_last_failure_at = None
    service._okx_authoritative_sync_last_error = None
    service._okx_authoritative_sync_last_requires_attention_count = 0
    service._okx_authoritative_sync_last_degraded_count = 0
    service._okx_authoritative_sync_last_samples = []
    authoritative_calls = 0
    local_calls = 0

    async def load_authoritative():
        nonlocal authoritative_calls
        authoritative_calls += 1
        return [{"symbol": "AUTH/USDT", "side": "long", "is_open": True}]

    async def load_local(
        *,
        strict: bool = False,
    ):
        nonlocal local_calls
        assert strict is True
        local_calls += 1
        return [{"symbol": "LOCAL/USDT", "side": "short", "is_open": True}]

    service._get_open_positions_context = load_authoritative  # type: ignore[method-assign]
    service._get_local_open_positions_context = load_local  # type: ignore[method-assign]

    result = await service._open_positions_context_for_round("market")

    assert result == [{"symbol": "LOCAL/USDT", "side": "short", "is_open": True}]
    assert local_calls == 1
    assert authoritative_calls == 0
    task = service._open_positions_context_refresh_task
    if task is not None and not task.done():
        await task
    assert authoritative_calls == 1


@pytest.mark.asyncio
async def test_market_round_falls_back_to_authoritative_when_local_positions_fail() -> None:
    service = TradingService.__new__(TradingService)
    service._open_positions_context_cache = {}
    service._open_positions_context_refresh_task = None
    service.okx_authoritative_sync_interval_seconds = lambda: 20.0  # type: ignore[method-assign]
    service._okx_authoritative_sync_task = None
    service._okx_authoritative_sync_last_success_at = datetime.now(UTC)
    service._okx_authoritative_sync_last_failure_at = None
    service._okx_authoritative_sync_last_error = None
    service._okx_authoritative_sync_last_requires_attention_count = 0
    service._okx_authoritative_sync_last_degraded_count = 0
    service._okx_authoritative_sync_last_samples = []
    authoritative_calls = 0

    async def load_authoritative():
        nonlocal authoritative_calls
        authoritative_calls += 1
        return [{"symbol": "AUTH/USDT", "side": "long", "is_open": True}]

    async def load_local(*, strict: bool = False, include_profit_first_metadata: bool = True):
        assert strict is True
        assert include_profit_first_metadata is False
        raise RuntimeError("db unavailable")

    service._get_open_positions_context = load_authoritative  # type: ignore[method-assign]
    service._get_local_open_positions_context = load_local  # type: ignore[method-assign]

    result = await service._open_positions_context_for_round("market")

    assert result == [{"symbol": "AUTH/USDT", "side": "long", "is_open": True}]
    assert authoritative_calls == 1


@pytest.mark.asyncio
async def test_market_strategy_context_account_equity_uses_cached_balance() -> None:
    service = TradingService.__new__(TradingService)
    service._okx_balance_snapshot_cache = {
        "paper": {
            "snapshot": {"free": 234.5, "equity": 250.0, "allocatable": 250.0},
            "fetched_at": datetime.now(UTC) - timedelta(seconds=30),
        }
    }
    service._okx_balance_snapshot_refresh_tasks = {}
    allocated_calls = 0
    refresh_calls: list[str] = []

    async def allocated(_mode: str):
        nonlocal allocated_calls
        allocated_calls += 1
        return 0.0

    service.allocated_order_balance = allocated  # type: ignore[method-assign]
    service._schedule_okx_balance_snapshot_refresh = refresh_calls.append  # type: ignore[method-assign]
    token = trading_service._analysis_scope_context.set("market")
    try:
        result = await service._strategy_context_account_equity("paper")
    finally:
        trading_service._analysis_scope_context.reset(token)

    assert result == 234.5
    assert allocated_calls == 0
    assert refresh_calls == []


@pytest.mark.asyncio
async def test_market_strategy_context_account_equity_uses_previous_context_without_cache() -> None:
    service = TradingService.__new__(TradingService)
    service._okx_balance_snapshot_cache = {}
    service._current_strategy_mode_context = {"account_equity": 321.0}
    allocated_calls = 0

    async def allocated(_mode: str):
        nonlocal allocated_calls
        allocated_calls += 1
        return 0.0

    service.allocated_order_balance = allocated  # type: ignore[method-assign]
    token = trading_service._analysis_scope_context.set("market")
    try:
        result = await service._strategy_context_account_equity("paper")
    finally:
        trading_service._analysis_scope_context.reset(token)

    assert result == 321.0
    assert allocated_calls == 0
    assert service._last_strategy_context_account_equity_source == "previous_strategy_context"


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

    async def record_stage(*args, **kwargs):
        return {}

    async def mark_reason(decision_id, reason):
        return None

    service = PositionReviewService(
        loop_stage_setter=set_loop_stage,
        sl_tp_enforcer=enforce_sl_tp,
        open_positions_context_provider=open_positions_context,
        position_reviewer=review_open_positions,
        analysis_symbol_claimer=claim_symbol,
        symbol_normalizer=lambda symbol: symbol,
        candidate_executor=execute_candidate,
        decision_stage_recorder=record_stage,
        decision_reason_marker=mark_reason,
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
async def test_position_review_service_records_claim_skip_for_review_candidates():
    decision = _decision(Action.LONG)
    assessment = SimpleNamespace(warnings=[])
    stages: list[tuple[Any, ...]] = []
    reasons: list[tuple[int, str]] = []

    async def enforce_sl_tp(_feature_vectors):
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
        return [("BTC/USDT", "ensemble_trader", decision, assessment, 789)], set()

    async def claim_symbol(_symbol, _owner):
        return False

    async def execute_candidate(*_args, **_kwargs):
        raise AssertionError("claim-failed position candidate must not execute")

    async def record_stage(
        decision_id,
        decision_arg,
        stage,
        status,
        reason,
        data=None,
        **_kwargs,
    ):
        stages.append((decision_id, decision_arg, stage, status, reason, data))
        return {}

    async def mark_reason(decision_id, reason):
        reasons.append((decision_id, reason))

    service = PositionReviewService(
        loop_stage_setter=lambda _stage: None,
        sl_tp_enforcer=enforce_sl_tp,
        open_positions_context_provider=open_positions_context,
        position_reviewer=review_open_positions,
        analysis_symbol_claimer=claim_symbol,
        symbol_normalizer=lambda symbol: symbol,
        candidate_executor=execute_candidate,
        decision_stage_recorder=record_stage,
        decision_reason_marker=mark_reason,
    )
    results: dict[str, Any] = {"executions": [], "decisions": []}

    _open_positions, blocked = await service.review_open_positions(
        feature_vectors={"BTC/USDT": object()},
        results=results,
        round_decision_ids=set(),
        open_positions=[],
        position_entry_pause_reason=None,
        max_groups_override=3,
        claimed_analysis_symbols=[],
    )

    assert blocked == set()
    assert stages[0][0] == 789
    assert stages[0][2] == DecisionStage.STRATEGY_ARBITRATION
    assert stages[0][3] == DecisionStageStatus.SKIPPED
    assert stages[0][5]["skip_kind"] == "position_analysis_symbol_claimed"
    assert reasons == [(789, stages[0][4])]
    assert results["decisions"][0]["execution_status"] == "skipped"
    assert "另一条分析流程" in results["decisions"][0]["reason"]


@pytest.mark.asyncio
async def test_position_review_reuses_round_positions_when_fast_sl_tp_has_no_action():
    decision = _decision(Action.CLOSE_LONG)
    assessment = SimpleNamespace(warnings=[])
    calls: list[tuple[Any, ...]] = []
    initial_positions = [
        {"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"}
    ]

    async def enforce_sl_tp(feature_vectors, *, open_positions=None):
        calls.append(("sl_tp_positions", open_positions is initial_positions))
        return []

    async def open_positions_context():
        calls.append(("open_positions_refresh",))
        return [{"model_name": "ensemble_trader", "symbol": "ETH/USDT", "side": "short"}]

    async def review_open_positions(
        open_positions,
        feature_vectors,
        *,
        results,
        round_decision_ids,
        position_entry_pause_reason,
        max_groups_override,
    ):
        calls.append(("review_positions", open_positions is initial_positions))
        return [("BTC/USDT", "ensemble_trader", decision, assessment, 777)], set()

    async def claim_symbol(_symbol, _owner):
        return False

    async def record_stage(*args, **kwargs):
        return {}

    async def mark_reason(*args, **kwargs):
        return None

    service = PositionReviewService(
        loop_stage_setter=lambda _stage: None,
        sl_tp_enforcer=enforce_sl_tp,
        open_positions_context_provider=open_positions_context,
        position_reviewer=review_open_positions,
        analysis_symbol_claimer=claim_symbol,
        symbol_normalizer=lambda symbol: symbol,
        candidate_executor=lambda *args, **kwargs: asyncio.sleep(0),
        decision_stage_recorder=record_stage,
        decision_reason_marker=mark_reason,
    )
    results: dict[str, Any] = {"executions": [], "decisions": []}

    open_positions, _blocked = await service.review_open_positions(
        feature_vectors={"BTC/USDT": object()},
        results=results,
        round_decision_ids=set(),
        open_positions=initial_positions,
        position_entry_pause_reason=None,
        max_groups_override=1,
        claimed_analysis_symbols=[],
    )

    assert open_positions is initial_positions
    assert calls == [
        ("sl_tp_positions", True),
        ("review_positions", True),
    ]
    assert results["position_review_diagnostics"][0]["kind"] == "reused_round_open_positions"


@pytest.mark.asyncio
async def test_execute_position_review_candidate_waits_for_handoff_after_outer_timeout():
    service = TradingService.__new__(TradingService)
    calls: list[tuple[Any, ...]] = []
    decision = _decision(Action.LONG)
    assessment = SimpleNamespace(warnings=[])

    async def mark_reason(decision_id, reason):
        calls.append(("reason", decision_id, reason))

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
        calls.append(("execute_start", symbol, model_name, decision_arg.action.value))
        await asyncio.sleep(0.03)
        calls.append(("execute_done", symbol, model_name, decision_arg.action.value))
        return ExecutionResult(
            order_id="local-1",
            symbol=symbol,
            side=decision_arg.action.value,
            order_type="market",
            quantity=1.0,
            price=100.0,
            status=OrderStatus.FILLED,
            exchange_order_id="okx-1",
        )

    async def ensure_final(decision_id, symbol, model_name, decision_arg, results):
        calls.append(("ensure", decision_id, symbol, model_name, decision_arg.action.value))

    service._mark_decision_reason = mark_reason  # type: ignore[method-assign]
    service._execute_candidate = execute_candidate  # type: ignore[method-assign]
    service.decision_final_state_ensurer = SimpleNamespace(ensure=ensure_final)

    result = await asyncio.wait_for(
        service.execute_position_review_candidate(
            "BTC/USDT",
            "ensemble_trader",
            decision,
            assessment,
            901,
            {"executions": []},
            open_positions=[{"symbol": "BTC/USDT"}],
        ),
        timeout=0.01,
    )

    assert result is not None
    assert result.exchange_order_id == "okx-1"
    assert ("execute_start", "BTC/USDT", "ensemble_trader", "long") in calls
    assert ("execute_done", "BTC/USDT", "ensemble_trader", "long") in calls
    assert ("ensure", 901, "BTC/USDT", "ensemble_trader", "long") in calls


@pytest.mark.asyncio
async def test_position_review_post_decision_handoff_finishes_after_stage_timeout():
    service = TradingService.__new__(TradingService)
    calls: list[tuple[Any, ...]] = []
    decision = _decision(Action.LONG)
    candidate = ("BTC/USDT", "ensemble_trader", decision, SimpleNamespace(warnings=[]), 902)

    async def record_stage(
        decision_id,
        decision_arg,
        stage,
        status,
        reason,
        data,
    ):
        calls.append(("stage", decision_id, decision_arg.action.value, stage, status, reason, data))
        return {}

    async def slow_post_process():
        calls.append(("post_start", decision.action.value))
        await asyncio.sleep(0.03)
        calls.append(("post_done", decision.action.value))
        return SimpleNamespace(handled=False, candidate=candidate)

    service._record_and_persist_decision_stage = record_stage  # type: ignore[method-assign]

    result = await asyncio.wait_for(
        service._await_position_review_post_decision_handoff(
            slow_post_process(),
            symbol="BTC/USDT",
            model_name="ensemble_trader",
            decision=decision,
            decision_db_id=902,
        ),
        timeout=0.01,
    )

    assert result.handled is False
    assert result.candidate == candidate
    assert ("post_start", "long") in calls
    assert ("post_done", "long") in calls
    stage_calls = [call for call in calls if call[0] == "stage"]
    assert stage_calls
    assert stage_calls[0][3] == DecisionStage.RISK_CHECK
    assert stage_calls[0][4] == DecisionStageStatus.PENDING
    assert stage_calls[0][6]["source"] == "position_review_post_decision_handoff"


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
        decision_stage_recorder=lambda *args, **kwargs: asyncio.sleep(0, result={}),
        decision_reason_marker=lambda *args, **kwargs: asyncio.sleep(0),
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
