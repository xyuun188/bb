"""
Trading service: the central orchestrator.
Wires together data feed, AI brain, risk manager, and executor
into the main trading loop.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Awaitable
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import func, select

from ai_brain.base_model import Action, DecisionOutput
from ai_brain.ensemble_coordinator import EnsembleCoordinator
from ai_brain.model_registry import ModelRegistry
from config.settings import ENSEMBLE_TRADER_NAME, settings
from core.safe_output import safe_error_text
from core.trading_mode import mode_manager
from db.repositories.decision_repo import DecisionRepository
from db.repositories.risk_repo import RiskRepository
from db.repositories.trade_repo import TradeRepository
from db.session import get_read_session_ctx, get_session_ctx
from executor.base_executor import ExecutionResult, OrderStatus
from executor.okx_executor import OKXExecutor
from executor.paper_executor import PaperExecutor
from models.decision import AIDecision
from models.learning import ShadowBacktest
from models.trade import Order, Position
from risk_manager.engine import RiskEngine
from services.account_accounting_service import (
    AccountAccountingService,
    allocatable_balance_from_snapshot,
    balance_from_snapshot,
    tradeable_balance_from_snapshot,
)
from services.analysis_budget import POSITION_REVIEW_MAX_GROUPS_PER_ROUND, AnalysisBudgetPolicy
from services.analysis_services import MarketAnalysisService, PositionReviewService
from services.daily_performance_service import DailyPerformanceService
from services.daily_side_performance import DailySidePerformanceService
from services.data_service import DataService
from services.decision_final_state_ensurer import DecisionFinalStateEnsurer
from services.decision_freshness import DecisionFreshnessPolicy
from services.decision_persistence_service import DecisionPersistenceService
from services.decision_reason_recovery import DecisionReasonRecoveryPolicy
from services.decision_state import (
    DecisionStage,
    DecisionStageStatus,
    decision_state_from_raw,
    is_decision_terminal_state,
)
from services.dynamic_exit_policy import apply_dynamic_exit
from services.dynamic_position_capacity import DynamicPositionCapacityPolicy
from services.entry_candidate_evidence import EntryCandidateEvidencePolicy
from services.entry_candidate_filter import EntryCandidateFilterPolicy
from services.entry_candidate_queue import EntryCandidateQueuePolicy
from services.entry_capacity import EntryCapacityPolicy
from services.entry_direction_competition import EntryDirectionCompetitionPolicy
from services.entry_execution_handoff import await_entry_execution_handoff
from services.entry_feature_ranker import EntryFeatureRankerPolicy
from services.entry_fee_provider import EntryFeeProvider
from services.entry_immediate_execution import EntryImmediateExecutionPlanner
from services.entry_market_data_quality import (
    EntryMarketDataQualityPolicy,
    MarketValueReader,
)
from services.entry_market_prefilter import EntryMarketLLMPrefilterPolicy
from services.entry_market_regime import EntryMarketRegimeContextPolicy, EntryMarketRegimePolicy
from services.entry_opportunity_gate import EntryOpportunityGatePolicy
from services.entry_opportunity_scoring import EntryOpportunityScoringPolicy
from services.entry_position_exposure import EntryPositionExposurePolicy
from services.entry_price_guard import EntryPriceGuardPolicy
from services.entry_priority import EntryExecutionPriorityPolicy
from services.entry_profit_risk_sizing import (
    EntryProfitRiskSizingPolicy,
    build_portfolio_correlation_context,
)
from services.entry_strategy_mode import EntryStrategyModeContextPolicy
from services.entry_suspicious_symbol import EntrySuspiciousSymbolPolicy
from services.entry_symbol_universe import EntrySymbolUniversePolicy
from services.exchange_backed_position_provider import ExchangeBackedPositionProvider
from services.exchange_close_fill_finder import ExchangeCloseFillFinder
from services.exchange_position_state import (
    ExchangePositionStatePolicy,
    ExchangeProtectionMapProvider,
)
from services.execution_allocation_service import ExecutionAllocationService
from services.execution_cost_model import attach_execution_cost_facts
from services.execution_pipelines import EntryExecutionPipeline, ExitExecutionPipeline
from services.execution_result_classifier import ExecutionResultClassifier
from services.execution_result_factory import ExecutionResultFactory
from services.execution_service import ExecutionService
from services.exit_position_matcher import ExitPositionMatcher
from services.exit_position_snapshot import ExitPositionSnapshotPolicy
from services.expert_memory_service import ExpertMemoryService
from services.fast_risk_exit_execution import FastRiskExitExecutionProcessor
from services.forced_exit import ForcedExitPolicy
from services.high_risk_review_service import HighRiskReviewService
from services.local_ai_tools_client import LocalAIToolsClient
from services.manual_trade_execution import ManualTradeExecutionProcessor
from services.manual_trade_risk_assessment import ManualTradeRiskAssessmentPolicy
from services.market_auto_entry_processor import MarketAutoEntryProcessor
from services.market_decision_result_recorder import MarketDecisionResultRecorder
from services.market_decision_risk_assessment import MarketDecisionRiskAssessmentPolicy
from services.market_direct_entry_processor import MarketDirectEntryProcessor
from services.market_queued_entry_processor import MarketQueuedEntryProcessor
from services.memory_position_store import MemoryPositionStore
from services.ml_signal_service import (
    AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
    AUTO_TRAIN_LEASE_STALE_SECONDS,
    AUTO_TRAIN_RETRY_INTERVAL_SECONDS,
    MODEL_TRAINING_STATE_STORE,
    MLSignalService,
)
from services.model_contribution_performance import ModelContributionPerformanceService
from services.model_training_state import (
    ALL_TRAINABLE_MODEL_IDS,
    LOCAL_AI_TOOL_MODEL_IDS,
    ModelTrainingStateStore,
)
from services.okx_order_fact_sync import OkxOrderFactSyncService
from services.okx_position_history_sync import OkxPositionHistoryMirrorSyncService
from services.okx_position_settlement_sync import OkxPositionSettlementSyncService
from services.open_positions_execution_applier import OpenPositionsExecutionApplier
from services.pending_exit_recovery import PendingExitDecisionRecoveryProcessor
from services.portfolio_profit_protection import PortfolioProfitProtectionPolicy
from services.position_execution_persistence import PositionExecutionPersistenceService
from services.position_group_aggregator import PositionGroupAggregator
from services.position_margin import PositionMarginCalculator
from services.position_open_time import position_open_time
from services.position_profit_peak_context import PositionProfitPeakContextPolicy
from services.position_profit_peaks import PositionProfitPeakTracker
from services.position_protection_fallback import PositionProtectionFallbackPolicy
from services.position_review_batch import PositionReviewBatchPolicy
from services.position_review_decision_processor import PositionReviewDecisionProcessor
from services.position_review_decision_service import (
    PositionReviewDecisionRequest,
    PositionReviewDecisionService,
)
from services.position_review_defer_tracker import PositionReviewDeferTracker
from services.position_review_entry_guard import PositionReviewEntryGuardPolicy
from services.position_review_fast_scan_hold import PositionReviewFastScanHoldPolicy
from services.position_review_fast_scan_recorder import PositionReviewFastScanRecorder
from services.position_review_grouping import PositionReviewGroupingPolicy
from services.position_review_outcome import (
    PositionReviewOutcomePolicy,
)
from services.position_review_priority import PositionReviewPriorityPolicy
from services.position_review_result_recorder import PositionReviewResultRecorder
from services.position_review_risk_alert import PositionReviewRiskAlertPolicy
from services.position_review_risk_assessment import PositionReviewRiskAssessmentPolicy
from services.position_settlement import (
    SETTLEMENT_STATUS_SETTLING,
    apply_position_settlement_snapshot,
    build_position_settlement_snapshot,
    funding_fee_from_payload,
    proportional_signed_value,
    settlement_payload_fields,
)
from services.position_snapshot_syncer import PositionSnapshotSyncer
from services.position_time import PositionTimeParser
from services.shadow_backtest_service import ShadowBacktestService
from services.stale_entry_candidate_expirer import StaleEntryCandidateExpirer
from services.strategy_context_performance import StrategyContextPerformanceService
from services.strategy_learning import StrategyLearningService
from services.symbol_side_performance import SymbolSidePerformanceService
from services.sync_service import OkxSyncService
from services.trade_order_log_service import TradeOrderLogService
from services.trading_agent_skills import TradingAgentSkillBook
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.trading_policies import EntryPolicy, ExitPolicy, PolicyGateResult
from services.vector_memory import get_vector_memory_service
from web_dashboard.api.text_sanitize import sanitize_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class _AnalysisRuntimeState:
    """Runtime status for one independent analysis loop."""

    current_stage: str = "idle"
    last_started_at: datetime | None = None
    last_finished_at: datetime | None = None
    last_error: str | None = None
    current_stage_started_at: datetime | None = None
    recent_stage_durations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.last_started_at is not None and (
            self.last_finished_at is None or self.last_finished_at < self.last_started_at
        )


PENDING_FEATURE_CANCEL_DRAIN_SECONDS = 0.25


async def drain_cancelled_tasks(
    tasks: set[asyncio.Task] | list[asyncio.Task],
    *,
    timeout_seconds: float = PENDING_FEATURE_CANCEL_DRAIN_SECONDS,
) -> None:
    """Cancel slow tasks without letting cancellation cleanup block the round watchdog."""

    pending = [task for task in tasks if not task.done()]
    if not pending:
        return
    for task in pending:
        task.cancel()

    done, still_pending = await asyncio.wait(
        pending,
        timeout=max(float(timeout_seconds), 0.0),
    )
    for task in done:
        try:
            task.result()
        except asyncio.CancelledError:
            continue
        except Exception as exc:
            logger.debug("cancelled task cleanup raised", error=safe_error_text(exc))
            continue
    if still_pending:
        for task in still_pending:
            task.add_done_callback(_consume_task_result)
        logger.warning(
            "cancelled tasks did not drain before timeout; continuing round",
            pending=len(still_pending),
            timeout_seconds=round(max(float(timeout_seconds), 0.0), 3),
        )


def _consume_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        return


_analysis_scope_context: ContextVar[str | None] = ContextVar(
    "analysis_scope_context",
    default=None,
)

PRE_AGENT_SKILLS_ROLLBACK_MODE = False
AGENT_SKILLS_TRADING_EFFECTS_ENABLED = True
LOCAL_QUANT_PROMPT_ENABLED = True
LOCAL_QUANT_MARKET_PREFILTER_ENABLED = True
OKX_BALANCE_SNAPSHOT_FRESH_SECONDS = 15.0
OKX_BALANCE_SNAPSHOT_STALE_SECONDS = 120.0
MARKET_OPEN_POSITIONS_CONTEXT_TTL_SECONDS = 5.0
NEW_PAIR_PAUSE_CONTEXT_TTL_SECONDS = 5.0
SHADOW_BACKTEST_FOREGROUND_UPDATE_LIMIT = 200
SHADOW_BACKTEST_MARKET_BACKGROUND_UPDATE_LIMIT = 25
STRATEGY_CONTEXT_IO_CONCURRENCY = 2
STRATEGY_CONTEXT_PERFORMANCE_SNAPSHOT_FRESH_SECONDS = 20.0
STRATEGY_CONTEXT_PERFORMANCE_SNAPSHOT_MAX_STALE_SECONDS = 300.0
STRATEGY_CONTEXT_PERFORMANCE_REFRESH_TIMEOUT_SECONDS = 8.0
STALE_ENTRY_EXPIRE_BACKGROUND_LIMIT_DESCRIPTION = "single_flight_batched_maintenance"
UNCONFIRMED_EXCHANGE_CLOSE_GRACE_SECONDS = 180.0
ENTRY_PRICE_RECHECK_TIMEOUT_SECONDS = 5.0
SYMBOL_SIDE_PROFILE_LOOKBACK = 2000
SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS = 7.0
LOCAL_ML_TRAINING_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training
AUTO_SCAN_PARAMS = DEFAULT_TRADING_PARAMS.auto_scan
AUTO_SCAN_ROTATION_POOL_MULTIPLIER = AUTO_SCAN_PARAMS.rotation_pool_multiplier
AUTO_SCAN_ROTATION_POOL_MIN = AUTO_SCAN_PARAMS.rotation_pool_min
AUTO_SCAN_FEATURE_FETCH_POOL_MULTIPLIER = AUTO_SCAN_PARAMS.feature_fetch_pool_multiplier
AUTO_SCAN_FEATURE_FETCH_POOL_MIN = AUTO_SCAN_PARAMS.feature_fetch_pool_min
AUTO_SCAN_FEATURE_FETCH_POOL_MAX = AUTO_SCAN_PARAMS.feature_fetch_pool_max
AUTO_SCAN_FEATURE_FETCH_TIMEOUT_SECONDS = AUTO_SCAN_PARAMS.feature_fetch_timeout_seconds
AUTO_SCAN_FEATURE_FETCH_CONCURRENCY = AUTO_SCAN_PARAMS.feature_fetch_concurrency
ALT_LONG_ALLOWED_SYMBOLS = set(AUTO_SCAN_PARAMS.major_symbols)


class TradingService:
    """Central trading orchestrator.

    Main loop:
    1. Fetch latest FeatureVectors for all symbols
    2. Run all AI models against each symbol
    3. Validate decisions through risk engine
    4. Execute approved decisions
    5. Log everything to database
    6. Push updates to dashboard via Redis/pubsub
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        data_service: DataService,
        redis_client=None,
        model_training_state_store: ModelTrainingStateStore | None = None,
    ) -> None:
        self.models = model_registry
        self.ensemble = EnsembleCoordinator(model_registry)
        self.data_service = data_service
        self.dynamic_capacity = DynamicPositionCapacityPolicy()
        self._current_capacity_context = self.dynamic_capacity.evaluate(open_positions=[]).as_dict()
        self._current_strategy_mode_context: dict[str, Any] = {}
        self.risk_engine = RiskEngine()
        self.market_decision_risk_assessment = MarketDecisionRiskAssessmentPolicy(
            risk_engine=self.risk_engine,
            account_balance_provider=self.get_account_balance,
        )
        self.manual_trade_risk_assessment = ManualTradeRiskAssessmentPolicy(self.risk_engine)
        self.manual_trade_execution_processor = ManualTradeExecutionProcessor(
            decision_logger=self._log_decision,
            decision_count_incrementer=self.increment_decision_count,
            candidate_executor=self._execute_candidate,
            is_paper_provider=lambda: mode_manager.is_paper,
        )
        self.model_training_state_store = (
            model_training_state_store or MODEL_TRAINING_STATE_STORE
        )
        self.ml_signal_service = MLSignalService(
            training_state_store=self.model_training_state_store
        )
        self.local_ai_tools = LocalAIToolsClient()
        self.high_risk_review_service = HighRiskReviewService()
        self.agent_skills = TradingAgentSkillBook()
        self.decision_persistence = DecisionPersistenceService(
            normalize_symbol=self._normalize_position_symbol
        )
        self.decision_reason_recovery = DecisionReasonRecoveryPolicy()
        self.trade_order_log_service = TradeOrderLogService(
            execution_mode_provider=self._get_model_execution_mode
        )
        self.market_decision_result_recorder = MarketDecisionResultRecorder()
        self.redis = redis_client
        self.execution_result_classifier = ExecutionResultClassifier()
        self.execution_result_factory = ExecutionResultFactory()
        self.open_positions_execution_applier = OpenPositionsExecutionApplier(
            normalize_symbol=self._normalize_position_symbol,
            is_exit_progress_execution=self.is_exit_progress_execution,
        )
        self.entry_symbol_universe = EntrySymbolUniversePolicy(self._normalize_position_symbol)
        self.market_analysis_service = MarketAnalysisService(
            run_once_provider=self.run_once,
            is_running_provider=self.is_running,
            time_budget_provider=self.market_round_watchdog_seconds,
        )
        self.position_review_service = PositionReviewService(
            run_once_provider=self.run_once,
            is_running_provider=self.is_running,
            loop_stage_setter=self.set_loop_stage,
            sl_tp_enforcer=self.enforce_sl_tp_for_position_review,
            open_positions_context_provider=self.open_positions_context_for_position_review,
            position_reviewer=self.review_open_positions_for_position_service,
            analysis_symbol_claimer=self.claim_analysis_symbol,
            symbol_normalizer=self.normalize_position_symbol,
            candidate_executor=self.execute_position_review_candidate,
            decision_stage_recorder=self.record_and_persist_decision_stage,
            decision_reason_marker=self.mark_decision_reason,
            timeout_provider=self.position_review_stage_timeout_seconds,
            round_watchdog_provider=self.position_round_watchdog_seconds,
        )
        self._execution_lock = asyncio.Lock()
        self.entry_execution_pipeline = EntryExecutionPipeline(lambda: self.entry_policy)
        self.exit_execution_pipeline = ExitExecutionPipeline(lambda: self.exit_policy)
        self.execution_service = ExecutionService(
            execution_lock=self._execution_lock,
            risk_event_logger=self.log_risk_event,
            model_execution_mode_provider=self.get_model_execution_mode,
            decision_stage_recorder=self.record_and_persist_decision_stage,
            decision_reason_marker=self.mark_decision_reason,
            decision_raw_response_marker=self.mark_decision_raw_response,
            position_review_alert_context_provider=self.position_review_alert_context,
            position_review_risk_result_logger=self.log_position_review_risk_result,
            duplicate_decision_order_reason_provider=self.duplicate_decision_order_reason,
            okx_executor_provider=self.get_okx_executor_for_mode,
            allocated_order_balance_provider=self.allocated_order_balance,
            rejected_execution_result_factory=self.rejected_execution_result,
            execution_leverage_summary_attacher=self.attach_execution_leverage_summary,
            execution_reason_provider=self.execution_reason_from_result,
            pending_execution_marker=self.mark_decision_pending_execution,
            trade_logger=self.log_trade,
            exchange_confirmed_checker=self.is_exchange_confirmed_execution,
            exit_progress_checker=self.is_exit_progress_execution,
            no_exchange_position_result_checker=self.result_has_no_exchange_position,
            trade_count_incrementer=self.increment_trade_count,
            position_execution_persister=self.persist_position_from_execution,
            order_fact_recovery_trigger=self.request_okx_order_fact_recovery,
            open_positions_execution_applier=self.apply_execution_to_open_positions,
            decision_executed_marker=self.mark_decision_executed,
            account_update_persister=self.persist_account_update,
            account_balance_provider=self.get_account_balance,
            decision_outcome_marker=self.mark_decision_outcome,
            entry_policy_evaluator=self.evaluate_entry_execution_policy,
            exit_policy_evaluator=self.exit_execution_pipeline.evaluate,
            execution_skills_provider=self.execution_agent_skills,
            execution_skills_attacher=self.attach_execution_agent_skills,
            execution_skills_block_reason_provider=self.execution_agent_skill_block_reason,
            position_reconciler=self.reconcile_positions_for_execution,
            open_positions_context_provider=self.open_positions_context_for_execution,
            matching_exit_local_position_checker=self.has_matching_local_exit_position,
            matching_exit_exchange_position_checker=(
                self.has_matching_exchange_exit_position_for_execution
            ),
            trade_notional_recorder=self.record_executed_trade_notional,
        )
        self._exchange_reconcile_lock = asyncio.Lock()
        self.position_protection_fallback = PositionProtectionFallbackPolicy(self._safe_float)
        self.position_group_aggregator = PositionGroupAggregator(
            normalize_symbol=self._normalize_position_symbol,
            float_parser=self._safe_float,
            default_model_name=ENSEMBLE_TRADER_NAME,
        )
        self.position_profit_peaks = PositionProfitPeakTracker(
            symbol_normalizer=self._normalize_position_symbol,
            float_parser=self._safe_float,
        )
        self.position_profit_peak_context = PositionProfitPeakContextPolicy(
            normalize_symbol=self._normalize_position_symbol,
            aggregate_position_group=self.position_group_aggregator.aggregate,
            position_peak_key=self.position_profit_peaks.key,
            position_peaks_provider=lambda: self.position_profit_peaks.peaks,
            default_model_name=ENSEMBLE_TRADER_NAME,
        )
        self.position_snapshot_syncer = PositionSnapshotSyncer(self._safe_float)
        self.position_time = PositionTimeParser()
        self.position_review_defer_tracker = PositionReviewDeferTracker()
        self.position_review_entry_guard = PositionReviewEntryGuardPolicy()
        self.position_review_fast_scan_hold = PositionReviewFastScanHoldPolicy()
        self.position_review_grouping = PositionReviewGroupingPolicy()
        self.position_review_outcome = PositionReviewOutcomePolicy()
        self.position_review_result_recorder = PositionReviewResultRecorder(
            outcome_policy=self.position_review_outcome,
            decision_reason_marker=self._mark_decision_reason,
            decision_raw_response_marker=self._mark_decision_raw_response,
            risk_result_logger=self.log_position_review_risk_result,
        )
        self.position_review_fast_scan_recorder = PositionReviewFastScanRecorder(
            default_model_name=ENSEMBLE_TRADER_NAME,
            normalize_symbol=self._normalize_position_symbol,
            urgent_exit_checker=self._is_urgent_position_exit_scan,
            portfolio_symbol_context_provider=self._portfolio_profit_protection_symbol_context,
            position_skills_provider=self.agent_skills.position_skills,
            agent_skills_summary_provider=self.agent_skills.summary,
            defer_count_provider=self.position_review_defer_tracker.count,
            defer_count_applier=self.position_review_defer_tracker.apply_plan_count,
            model_execution_mode_provider=self._get_model_execution_mode,
            decision_logger=self._log_decision,
            decision_reason_marker=self._mark_decision_reason,
            result_recorder=self.position_review_result_recorder,
            hold_policy=self.position_review_fast_scan_hold,
        )
        self.position_review_risk_assessment = PositionReviewRiskAssessmentPolicy(self.risk_engine)
        self.position_review_risk_alert_policy = PositionReviewRiskAlertPolicy(
            float_parser=self._safe_float,
            text_shortener=self._short_text,
            action_labeler=self._action_label_text,
        )
        self.market_value_reader = MarketValueReader()
        self.entry_market_data_quality = EntryMarketDataQualityPolicy(
            market_value_reader=self.market_value_reader.read,
        )
        self.entry_market_llm_prefilter = EntryMarketLLMPrefilterPolicy(
            self.entry_market_data_quality.reason
        )
        self.memory_position_store = MemoryPositionStore(
            paper_executor_provider=lambda: self.paper_executor,
            symbol_normalizer=self._normalize_position_symbol,
        )
        self.expert_memory_service = ExpertMemoryService()
        self.position_review_decision_service = PositionReviewDecisionService(
            default_model_name=ENSEMBLE_TRADER_NAME,
            expert_memory_context_provider=self.expert_memory_service.context,
            ml_signal_predictor=self.ml_signal_service.predict,
            local_ai_tools_context_provider=self._local_ai_tools_context,
            position_skills_provider=self.agent_skills.position_skills,
            agent_skills_attacher=self.agent_skills.attach,
            ensemble_decider=self.ensemble.decide,
            model_provider=self.models.get,
            pre_agent_skills_rollback=PRE_AGENT_SKILLS_ROLLBACK_MODE,
            local_quant_prompt_enabled=LOCAL_QUANT_PROMPT_ENABLED,
        )
        self.daily_performance_service = DailyPerformanceService()
        self.daily_side_performance_service = DailySidePerformanceService()
        self.symbol_side_performance_service = SymbolSidePerformanceService(
            normalize_symbol=self._normalize_position_symbol,
            lookback_limit=SYMBOL_SIDE_PROFILE_LOOKBACK,
            lookback_days=SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS,
        )
        self.strategy_context_performance_service = StrategyContextPerformanceService(
            daily_performance=self.daily_performance_service,
            daily_side_performance=self.daily_side_performance_service,
            symbol_side_performance=self.symbol_side_performance_service,
        )
        self.position_margin_calculator = PositionMarginCalculator()
        self.exchange_backed_position_provider = ExchangeBackedPositionProvider()
        self.exchange_position_state = ExchangePositionStatePolicy()
        self.execution_allocation_service = ExecutionAllocationService(
            balance_snapshot_provider=self._get_okx_balance_snapshot_for_mode,
            active_executor_provider=self.active_okx_for_mode,
            exchange_position_open_checker=self.exchange_position_state.is_open,
            symbol_normalizer=self._normalize_position_symbol,
        )
        self.account_accounting_service = AccountAccountingService(
            balance_snapshot_provider=self._get_okx_balance_snapshot_for_mode,
            allocation_state_provider=self.execution_allocation_state,
            model_execution_mode_provider=self._get_model_execution_mode,
        )
        self.exchange_protection_map_provider = ExchangeProtectionMapProvider(
            symbol_normalizer=self._normalize_position_symbol,
            position_open_checker=self.exchange_position_state.is_open,
            float_parser=self._safe_float,
        )
        self.exchange_close_fill_finder = ExchangeCloseFillFinder(
            paper_okx_provider=self.paper_okx_for_reconciliation,
            float_parser=self._safe_float,
            datetime_from_ms_parser=self.position_time.datetime_from_ms,
        )
        self.entry_fee_provider = EntryFeeProvider()
        self.position_execution_persistence = PositionExecutionPersistenceService(
            exchange_confirmed_checker=self.is_exchange_confirmed_execution,
            exit_progress_checker=self.is_exit_progress_execution,
            exchange_backed_id_provider=self.exchange_backed_position_provider.ids,
            entry_fee_provider=self.entry_fee_provider.entry_fee_for_position,
            proportional_fee=self.entry_fee_provider.proportional_fee,
            trade_reflection_recorder=self.record_trade_reflection_in_session,
            position_peak_remover=self.position_profit_peaks.remove,
        )
        self.okx_sync_service = OkxSyncService(
            exchange_reconcile_lock=self._exchange_reconcile_lock,
            round_error_recorder=self.record_round_error,
            symbol_normalizer=self._normalize_position_symbol,
            model_execution_mode_provider=self._get_model_execution_mode,
            okx_executor_provider=self._get_okx_executor_for_mode,
            float_parser=self._safe_float,
            active_okx_provider=self.active_okx_for_current_mode,
            paper_okx_provider=self.paper_okx_for_reconciliation,
            exchange_position_open_checker=self.exchange_position_state.is_open,
            exchange_protection_map_provider=self.exchange_protection_map_provider.fetch,
            position_protection_fallback_provider=(
                self.position_protection_fallback.protection_from_decision
            ),
            position_profit_peak_recorder=self.record_position_profit_peak,
            position_age_minutes_provider=self.position_time.position_age_minutes,
            position_profit_peak_pruner=self.prune_position_profit_peaks,
            local_position_snapshot_syncer=self.position_snapshot_syncer.sync,
            datetime_from_ms_parser=self.position_time.datetime_from_ms,
            exchange_close_fill_finder=self.exchange_close_fill_finder.find,
            fresh_feature_vector_provider=self.fresh_feature_vector_for_price_recheck,
            market_value_reader=self.market_value_reader.read,
            entry_fee_provider=self.entry_fee_provider.entry_fee_for_position,
            exchange_sync_close_decision_logger=self.log_exchange_sync_close_decision,
            trade_reflection_recorder=self.record_trade_reflection_in_session,
            position_margin_calculator=self.position_margin_calculator.margin,
            memory_position_remover=self.memory_position_store.remove_open_position,
        )
        self.okx_order_fact_sync_factory = OkxOrderFactSyncService
        self.okx_position_history_mirror_sync_factory = OkxPositionHistoryMirrorSyncService
        self.okx_position_settlement_sync_factory = OkxPositionSettlementSyncService
        self.exit_position_matcher = ExitPositionMatcher(self._normalize_position_symbol)
        self.exit_position_snapshot = ExitPositionSnapshotPolicy(self.okx_sync_service)
        self.forced_exit_policy = ForcedExitPolicy()
        self.portfolio_profit_protection = PortfolioProfitProtectionPolicy(
            normalize_symbol=self._normalize_position_symbol,
            default_model_name=ENSEMBLE_TRADER_NAME,
        )
        self.position_review_priority = PositionReviewPriorityPolicy(
            normalize_symbol=self._normalize_position_symbol,
            position_peak_key=self.position_profit_peaks.key,
            position_peaks_provider=lambda: self.position_profit_peaks.peaks,
        )
        self.position_review_batch = PositionReviewBatchPolicy(
            urgent_exit_checker=self._is_urgent_position_exit_scan,
        )
        self.analysis_budget = AnalysisBudgetPolicy(
            normalize_symbol=self._normalize_position_symbol,
            open_position_group_counter=self.entry_symbol_universe.open_position_group_count,
            portfolio_profit_context_provider=self._portfolio_profit_protection_context,
            position_review_scanner=self._scan_position_review_groups,
            urgent_exit_checker=self._is_urgent_position_exit_scan,
            default_model_name=ENSEMBLE_TRADER_NAME,
        )
        self.fast_risk_exit_execution_processor = FastRiskExitExecutionProcessor(
            model_execution_mode_provider=self._get_model_execution_mode,
            decision_logger=self._log_decision,
            decision_count_incrementer=self.increment_decision_count,
            candidate_executor=self._execute_candidate,
            exchange_confirmed_checker=self._is_exchange_confirmed_execution,
            exit_progress_checker=self._is_exit_progress_execution,
            profit_exit_recorder=self.position_profit_peaks.remember_profit_exit,
            risk_event_logger=self._log_risk_event,
            rejected_execution_factory=self._rejected_execution_result,
            trade_logger=self._log_trade,
            decision_reason_marker=self._mark_decision_reason,
            execution_reason_provider=self._execution_reason_from_result,
        )
        self.decision_freshness = DecisionFreshnessPolicy(self.forced_exit_policy.is_forced_exit)
        self.entry_priority = EntryExecutionPriorityPolicy()
        self.model_contribution_performance_service = ModelContributionPerformanceService(
            lookback_days=SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS,
        )
        self.entry_opportunity_score = EntryOpportunityScoringPolicy(
            normalize_symbol=self._normalize_position_symbol,
            annotate_decision_source=self._annotate_decision_source,
        )
        self.entry_capacity = EntryCapacityPolicy(self._normalize_position_symbol)
        self.entry_position_exposure = EntryPositionExposurePolicy()
        self.entry_market_regime = EntryMarketRegimePolicy()
        self.entry_market_regime_context = EntryMarketRegimeContextPolicy(
            self._is_valid_feature_vector,
        )
        self.entry_direction_competition = EntryDirectionCompetitionPolicy()
        self.entry_strategy_mode_context = EntryStrategyModeContextPolicy()
        self.strategy_learning_service = StrategyLearningService()
        self.entry_suspicious_symbol = EntrySuspiciousSymbolPolicy(self._normalize_position_symbol)
        self.entry_feature_ranker = EntryFeatureRankerPolicy(
            suspicious_symbol_reason=self.entry_suspicious_symbol.reason,
            major_symbols=frozenset(ALT_LONG_ALLOWED_SYMBOLS),
        )
        self._market_budget_deferred_symbols: list[str] = []
        self.entry_opportunity_gate = EntryOpportunityGatePolicy(
            suspicious_symbol_policy=self.entry_suspicious_symbol,
        )
        self.entry_profit_risk_sizing = EntryProfitRiskSizingPolicy(
            allocated_order_balance=self.allocated_order_balance,
            exchange_risk_facts=self.entry_exchange_risk_facts,
        )
        self._open_positions_context_cache: dict[str, Any] = {}
        self._open_positions_context_refresh_task: asyncio.Task | None = None
        self._new_pair_pause_context_cache: dict[str, Any] = {}
        self._new_pair_pause_context_refresh_task: asyncio.Task | None = None
        self.shadow_backtest_service = ShadowBacktestService(
            latest_price_provider=self._latest_price_for_symbol,
            symbol_normalizer=self._normalize_position_symbol,
            float_parser=self._safe_float,
            execution_cost_facts_provider=self._shadow_execution_cost_facts,
            latest_market_fact_provider=self.data_service.get_latest_market_fact,
            price_path_provider=self.data_service.verify_market_fact_path,
        )
        self.stale_entry_candidate_expirer = StaleEntryCandidateExpirer(self._safe_float)
        self.decision_final_state_ensurer = DecisionFinalStateEnsurer(
            execution_reason_unusable_checker=self._execution_reason_is_unusable,
            execution_reason_recoverer=self._recover_execution_reason_from_decision_row,
            model_execution_mode_provider=self._get_model_execution_mode,
        )
        self.entry_price_guard = EntryPriceGuardPolicy(
            fresh_feature_provider=self._fresh_feature_vector_for_price_recheck,
            market_data_quality_reason_provider=self.entry_market_data_quality.reason,
            decision_age_seconds_provider=self.decision_freshness.decision_age_seconds,
            pre_order_execution_facts_provider=self.pre_order_execution_facts,
        )
        self.entry_policy = EntryPolicy(
            decision_freshness=self.decision_freshness,
            entry_priority=self.entry_priority,
            entry_opportunity_score=self.entry_opportunity_score,
            entry_profit_risk_sizing=self.entry_profit_risk_sizing,
            entry_price_guard=self.entry_price_guard,
            entry_opportunity_gate=self.entry_opportunity_gate,
        )
        self.entry_candidate_queue = EntryCandidateQueuePolicy(
            score_candidate=self.entry_policy.score_candidate,
            wait_sort_reason=self.entry_policy.wait_sort_reason,
        )
        self.entry_candidate_filter = EntryCandidateFilterPolicy(
            gate_reason=self.entry_policy.gate_reason,
            market_regime_reason=self.entry_market_regime.reason,
            capacity_reason=self.entry_capacity.reason,
            reserve_capacity=self.entry_capacity.reserve_slot,
        )
        self.entry_immediate_execution = EntryImmediateExecutionPlanner(
            immediate_reason_provider=self.entry_policy.immediate_execution_reason,
            capacity_reason_provider=self.entry_capacity.reason,
            capacity_reserver=self.entry_capacity.reserve_slot,
        )
        self.market_auto_entry_processor = MarketAutoEntryProcessor(
            score_candidate=self.entry_policy.score_candidate,
            gate_reason=self.entry_policy.gate_reason,
            immediate_execution=self.entry_immediate_execution,
            annotate_candidate_selection=self._annotate_candidate_selection,
            mark_decision_raw_response=self._mark_decision_raw_response,
            mark_decision_reason=self._mark_decision_reason,
            mark_decision_pending_execution=self._mark_decision_pending_execution,
            result_recorder=self.market_decision_result_recorder,
            set_loop_stage=self._set_loop_stage,
            candidate_executor=self._execute_candidate,
            final_state_ensurer=self.decision_final_state_ensurer.ensure,
            capacity_releaser=self.entry_capacity.release_slot,
            pre_execution_capacity_reason=self.entry_capacity.reason,
            execution_confirmed_checker=self._is_exchange_confirmed_execution,
        )
        self.market_direct_entry_processor = MarketDirectEntryProcessor(
            capacity_reason_provider=self.entry_capacity.reason,
            capacity_reserver=self.entry_capacity.reserve_slot,
            annotate_candidate_selection=self._annotate_candidate_selection,
            mark_decision_raw_response=self._mark_decision_raw_response,
            mark_decision_reason=self._mark_decision_reason,
            result_recorder=self.market_decision_result_recorder,
            candidate_executor=self._execute_candidate,
            capacity_releaser=self.entry_capacity.release_slot,
            execution_confirmed_checker=self._is_exchange_confirmed_execution,
        )
        self.market_queued_entry_processor = MarketQueuedEntryProcessor(
            normalize_symbol=self._normalize_position_symbol,
            analysis_symbol_claimer=self._try_claim_analysis_symbol,
            annotate_candidate_selection=self._annotate_candidate_selection,
            mark_decision_raw_response=self._mark_decision_raw_response,
            mark_decision_reason=self._mark_decision_reason,
            mark_decision_pending_execution=self._mark_decision_pending_execution,
            result_recorder=self.market_decision_result_recorder,
            model_execution_mode_provider=self._get_model_execution_mode,
            set_loop_stage=self._set_loop_stage,
            candidate_executor=self._execute_candidate,
            final_state_ensurer=self.decision_final_state_ensurer.ensure,
            capacity_releaser=self.entry_capacity.release_slot,
            execution_confirmed_checker=self._is_exchange_confirmed_execution,
        )
        self.pending_exit_recovery_processor = PendingExitDecisionRecoveryProcessor(
            set_loop_stage=self._set_loop_stage,
            candidate_executor=self._execute_candidate,
        )
        self.entry_candidate_evidence = EntryCandidateEvidencePolicy(
            model_name=ENSEMBLE_TRADER_NAME,
            score_candidate=self.entry_policy.score_candidate,
            feature_opportunity_score=self._feature_opportunity_score,
        )
        self.exit_policy = ExitPolicy(
            exit_position_matcher=self.exit_position_matcher,
            exit_position_snapshot=self.exit_position_snapshot,
        )
        self.position_review_decision_processor = PositionReviewDecisionProcessor(
            entry_guard=self.position_review_entry_guard,
            entry_capacity=self.entry_capacity,
            risk_assessment=self.position_review_risk_assessment,
            result_recorder=self.position_review_result_recorder,
            candidate_executor=self._execute_candidate,
            final_state_ensurer=self.decision_final_state_ensurer.ensure,
            account_balance_provider=self.get_account_balance,
            entry_risk_contract_preparer=self._prepare_entry_for_hard_risk,
        )

        # Executors: paper routes to OKX demo, live routes to OKX real.
        self.paper_executor: PaperExecutor | None = None
        self._okx_paper: OKXExecutor | None = None
        self._okx_live: OKXExecutor | None = None
        self._model_execution_modes: dict[str, str] = {}  # model_name -> "paper"/"live"

        self._running = False
        self._decision_count = 0
        self._trade_count = 0
        self._recent_decisions: list[dict] = []
        self._recent_executions: list[dict] = []
        self._start_time: datetime | None = None
        self._current_stage = "idle"
        self._last_round_started_at: datetime | None = None
        self._last_round_finished_at: datetime | None = None
        self._last_market_round_started_at: datetime | None = None
        self._last_market_round_finished_at: datetime | None = None
        self._last_position_round_started_at: datetime | None = None
        self._last_position_round_finished_at: datetime | None = None
        self._last_round_error: str | None = None
        self._analysis_runtime: dict[str, _AnalysisRuntimeState] = {
            "market": _AnalysisRuntimeState(),
            "position": _AnalysisRuntimeState(),
            "full": _AnalysisRuntimeState(),
        }
        self._pnl_history: dict[str, list[dict]] = {}  # model_name -> [{time, equity}, ...]
        self._new_pair_pause_reasons: dict[str, str] = {}
        self._okx_balance_snapshot_cache: dict[str, dict[str, Any]] = {}
        self._okx_balance_snapshot_locks: dict[str, asyncio.Lock] = {}
        self._okx_balance_snapshot_refresh_tasks: dict[str, asyncio.Task] = {}
        self._shadow_backtest_update_task: asyncio.Task | None = None
        self._shadow_backtest_update_last_started_at: datetime | None = None
        self._shadow_backtest_update_last_finished_at: datetime | None = None
        self._shadow_backtest_update_last_count: int | None = None
        self._shadow_backtest_update_last_error: str | None = None
        self._shadow_backtest_update_success_count = 0
        self._shadow_backtest_update_failure_count = 0
        self._stale_entry_expire_task: asyncio.Task | None = None
        self._stale_entry_expire_last_started_at: datetime | None = None
        self._stale_entry_expire_last_finished_at: datetime | None = None
        self._stale_entry_expire_last_count: int | None = None
        self._stale_entry_expire_last_error: str | None = None
        self._stale_entry_expire_success_count = 0
        self._stale_entry_expire_failure_count = 0
        self._position_review_cursor = 0
        self._position_review_priority_cursor = 0
        self._auto_scan_feature_cursor = 0
        self._active_analysis_symbols: set[str] = set()
        self._analysis_symbol_lock = asyncio.Lock()
        self._market_analysis_task: asyncio.Task | None = None
        self._position_analysis_task: asyncio.Task | None = None
        self._runtime_heartbeat_task: asyncio.Task | None = None
        self._okx_authoritative_sync_task: asyncio.Task | None = None
        self._okx_order_fact_sync_task: asyncio.Task | None = None
        self._okx_order_fact_sync_last_started_at: datetime | None = None
        self._okx_order_fact_sync_last_finished_at: datetime | None = None
        self._okx_order_fact_sync_last_row: dict[str, Any] | None = None
        self._okx_order_fact_sync_last_error: str | None = None
        self._okx_order_fact_sync_success_count = 0
        self._okx_order_fact_sync_failure_count = 0
        self._okx_position_history_mirror_sync_task: asyncio.Task | None = None
        self._okx_position_history_mirror_sync_last_started_at: datetime | None = None
        self._okx_position_history_mirror_sync_last_finished_at: datetime | None = None
        self._okx_position_history_mirror_sync_last_row: dict[str, Any] | None = None
        self._okx_position_history_mirror_sync_last_error: str | None = None
        self._okx_position_history_mirror_sync_success_count = 0
        self._okx_position_history_mirror_sync_failure_count = 0
        self._okx_position_settlement_sync_task: asyncio.Task | None = None
        self._okx_position_settlement_sync_last_started_at: datetime | None = None
        self._okx_position_settlement_sync_last_finished_at: datetime | None = None
        self._okx_position_settlement_sync_last_row: dict[str, Any] | None = None
        self._okx_position_settlement_sync_last_error: str | None = None
        self._okx_position_settlement_sync_success_count = 0
        self._okx_position_settlement_sync_failure_count = 0
        self._okx_authoritative_sync_started_at: datetime | None = None
        self._okx_authoritative_sync_last_success_at: datetime | None = None
        self._okx_authoritative_sync_last_failure_at: datetime | None = None
        self._okx_authoritative_sync_last_error: str | None = None
        self._okx_authoritative_sync_last_duration_seconds: float | None = None
        self._okx_authoritative_sync_last_result_count: int | None = None
        self._okx_authoritative_sync_last_result_kinds: dict[str, int] = {}
        self._okx_authoritative_sync_last_requires_attention_count: int = 0
        self._okx_authoritative_sync_last_degraded_count: int = 0
        self._okx_authoritative_sync_last_samples: list[dict[str, Any]] = []
        self._okx_authoritative_sync_success_count = 0
        self._okx_authoritative_sync_failure_count = 0
        self._ml_auto_train_task: asyncio.Task | None = None
        self._model_training_heartbeat_task: asyncio.Task | None = None
        self._local_tools_active_training_run_id: str | None = None
        self._local_tools_last_train_started_at: datetime | None = None
        self._local_tools_last_completed_shadow_count: int = 0
        self._strategy_learning_context_cache: dict[str, Any] = {}
        self._strategy_learning_context_refresh_tasks: dict[str, asyncio.Task] = {}

    def is_running(self) -> bool:
        """Expose lifecycle state without coupling loop services to private fields."""

        return bool(getattr(self, "_running", False))

    def _model_training_state(self) -> ModelTrainingStateStore:
        store = getattr(self, "model_training_state_store", None)
        if store is None:
            store = MODEL_TRAINING_STATE_STORE
            self.model_training_state_store = store
        return store

    def set_loop_stage(self, stage: str) -> None:
        """Set loop stage through an explicit analysis-service boundary."""

        self._set_loop_stage(stage)

    def position_review_stage_timeout_seconds(self) -> float:
        """Return the configured hard boundary for one position-review stage."""

        return max(
            10.0,
            float(settings.ai_batch_expert_timeout_seconds or 0.0)
            + float(settings.ai_decision_maker_timeout_seconds or 0.0)
            + float(settings.local_ai_tools_timeout_seconds or 0.0),
        )

    def market_round_time_budget_seconds(
        self,
        strategy_context: dict[str, Any] | None = None,
        market_symbol_count: int | None = None,
    ) -> float:
        """Return the soft per-round scan budget used inside market analysis."""

        settings.refresh_runtime_env(force=True)
        interval = max(10.0, float(settings.decision_interval_seconds or 60))
        base_budget = max(8.0, interval * 0.90)
        del strategy_context
        requested_symbols = max(self._safe_int(market_symbol_count, 0), 0)
        target_symbols = min(max(requested_symbols, 1), 8)
        per_symbol_floor = max(8.0, min(14.0, interval * 0.35))
        market_budget = max(base_budget, target_symbols * per_symbol_floor)
        watchdog_ceiling = max(base_budget, self.market_round_watchdog_seconds() * 0.75)
        return min(market_budget, watchdog_ceiling)

    def market_symbol_start_reserve_seconds(
        self,
        strategy_context: dict[str, Any] | None = None,
        market_symbol_count: int | None = None,
    ) -> float:
        """Return remaining time needed before starting another market AI symbol."""

        settings.refresh_runtime_env(force=True)
        interval = max(10.0, float(settings.decision_interval_seconds or 60))
        batch_timeout = max(8.0, float(settings.ai_batch_expert_timeout_seconds or 18.0))
        decision_timeout = max(0.0, float(settings.ai_decision_maker_timeout_seconds or 0.0))
        local_tools_timeout = max(0.0, float(settings.local_ai_tools_timeout_seconds or 0.0))
        model_reserve = max(
            8.0,
            batch_timeout * 0.60
            + min(decision_timeout, 10.0) * 0.15
            + min(local_tools_timeout, 6.0) * 0.25,
        )
        budget_seconds = self.market_round_time_budget_seconds(
            strategy_context=strategy_context,
            market_symbol_count=market_symbol_count,
        )
        return round(
            min(max(model_reserve, interval * 0.20), max(6.0, budget_seconds * 0.45)),
            3,
        )

    def position_loop_interval_seconds(self) -> float:
        """Return the sleep interval between independent position-review rounds."""

        settings.refresh_runtime_env(force=True)
        interval = max(10.0, float(settings.decision_interval_seconds or 60))
        return max(5.0, interval * 0.65)

    def market_loop_interval_seconds(self) -> float:
        """Return the sleep interval between independent market-scan rounds."""

        settings.refresh_runtime_env(force=True)
        interval = max(10.0, float(settings.decision_interval_seconds or 60))
        return max(8.0, min(14.0, interval * 0.35))

    def market_round_watchdog_seconds(self) -> float:
        """Return the hard watchdog for a genuinely stuck market-analysis round."""

        settings.refresh_runtime_env(force=True)
        interval = max(10.0, float(settings.decision_interval_seconds or 60))
        expert_budget = (
            float(settings.ai_batch_expert_timeout_seconds or 0.0)
            + float(settings.ai_decision_maker_timeout_seconds or 0.0)
            + float(settings.local_ai_tools_timeout_seconds or 0.0)
        )
        configured_watchdog = float(settings.market_analysis_watchdog_seconds or 180)
        return max(configured_watchdog, interval * 4.0, expert_budget * 2.0)

    def position_round_watchdog_seconds(self) -> float:
        """Return the stuck-round watchdog for one full position-review round.

        Position review also has softer per-stage deadlines.  The round watchdog
        must be larger than those stage budgets; otherwise a normal slow review
        gets cancelled before the stage code can skip one group and persist the
        real diagnostic.
        """

        settings.refresh_runtime_env(force=True)
        interval = max(10.0, float(settings.decision_interval_seconds or 60))
        stage_budget = self.position_review_stage_timeout_seconds()
        configured_watchdog = float(
            settings.position_analysis_watchdog_seconds
            or settings.market_analysis_watchdog_seconds
            or 180
        )
        return max(configured_watchdog, interval * 4.0, stage_budget * 2.0)

    def round_start_reconcile_timeout_seconds(self) -> float:
        """Return the short OKX sync boundary used at analysis round start."""

        settings.refresh_runtime_env(force=True)
        interval = max(10.0, float(settings.decision_interval_seconds or 60))
        return max(8.0, min(14.0, interval * 0.35))

    def okx_authoritative_sync_interval_seconds(self) -> float:
        """Return the background cadence for current OKX position sync."""

        settings.refresh_runtime_env(force=True)
        interval = max(10.0, float(settings.decision_interval_seconds or 60))
        return max(20.0, min(60.0, interval * 0.5))

    def okx_order_fact_sync_interval_seconds(self) -> float:
        """Return the normal cadence for optional OKX order-fact repair."""

        return max(90.0, self.okx_authoritative_sync_interval_seconds() * 4.0)

    def okx_order_fact_sync_degraded_interval_seconds(self) -> float:
        """Back off optional order-fact repair after OKX native fact timeouts."""

        return max(240.0, self.okx_authoritative_sync_interval_seconds() * 8.0)

    def okx_position_history_mirror_sync_interval_seconds(self) -> float:
        """Return the cadence for account-level OKX history mirror refresh."""

        return max(60.0, self.okx_authoritative_sync_interval_seconds() * 2.0)

    def okx_position_settlement_sync_interval_seconds(self) -> float:
        """Return the retry cadence for official closed-position settlement."""

        return 10.0

    def _okx_authoritative_sync_status_payload(
        self,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return runtime diagnostics for the current-position OKX sync loop."""

        now = now or datetime.now(UTC)
        interval_seconds = self.okx_authoritative_sync_interval_seconds()
        stale_after_seconds = max(interval_seconds * 3.0, 180.0)
        started_at = getattr(self, "_okx_authoritative_sync_started_at", None)
        last_success_at = getattr(self, "_okx_authoritative_sync_last_success_at", None)
        last_failure_at = getattr(self, "_okx_authoritative_sync_last_failure_at", None)
        task = getattr(self, "_okx_authoritative_sync_task", None)
        last_success_age_seconds = (
            max((now - last_success_at).total_seconds(), 0.0)
            if isinstance(last_success_at, datetime)
            else None
        )
        last_failure_age_seconds = (
            max((now - last_failure_at).total_seconds(), 0.0)
            if isinstance(last_failure_at, datetime)
            else None
        )
        fresh_success_available = (
            isinstance(last_success_age_seconds, (int, float))
            and last_success_age_seconds <= stale_after_seconds
        )
        failure_after_success = isinstance(last_failure_at, datetime) and (
            not isinstance(last_success_at, datetime) or last_failure_at > last_success_at
        )
        status = "pending"
        if failure_after_success:
            status = "degraded" if fresh_success_available else "warning"
        elif isinstance(last_success_at, datetime):
            status = (
                "stale"
                if last_success_age_seconds is not None
                and last_success_age_seconds > stale_after_seconds
                else "ok"
            )
        return {
            "enabled": True,
            "status": status,
            "task_running": bool(task is not None and not task.done()),
            "interval_seconds": round(interval_seconds, 3),
            "stale_after_seconds": round(stale_after_seconds, 3),
            "last_started_at": started_at.isoformat() if isinstance(started_at, datetime) else None,
            "last_success_at": (
                last_success_at.isoformat() if isinstance(last_success_at, datetime) else None
            ),
            "last_success_age_seconds": (
                round(last_success_age_seconds, 3)
                if last_success_age_seconds is not None
                else None
            ),
            "last_failure_at": (
                last_failure_at.isoformat() if isinstance(last_failure_at, datetime) else None
            ),
            "last_failure_age_seconds": (
                round(last_failure_age_seconds, 3)
                if last_failure_age_seconds is not None
                else None
            ),
            "fresh_success_available": bool(fresh_success_available),
            "last_failure_covered_by_fresh_success": bool(
                failure_after_success and fresh_success_available
            ),
            "last_error": getattr(self, "_okx_authoritative_sync_last_error", None),
            "last_duration_seconds": getattr(
                self,
                "_okx_authoritative_sync_last_duration_seconds",
                None,
            ),
            "last_result_count": getattr(
                self,
                "_okx_authoritative_sync_last_result_count",
                None,
            ),
            "last_result_kinds": dict(
                getattr(self, "_okx_authoritative_sync_last_result_kinds", {}) or {}
            ),
            "last_requires_attention_count": int(
                getattr(
                    self,
                    "_okx_authoritative_sync_last_requires_attention_count",
                    0,
                )
                or 0
            ),
            "last_degraded_count": int(
                getattr(self, "_okx_authoritative_sync_last_degraded_count", 0) or 0
            ),
            "last_samples": list(
                getattr(self, "_okx_authoritative_sync_last_samples", []) or []
            ),
            "success_count": int(getattr(self, "_okx_authoritative_sync_success_count", 0) or 0),
            "failure_count": int(getattr(self, "_okx_authoritative_sync_failure_count", 0) or 0),
            "order_fact_sync": self._okx_order_fact_sync_status_payload(now),
            "position_history_mirror_sync": self._okx_position_history_mirror_sync_status_payload(
                now
            ),
            "position_settlement_sync": self._okx_position_settlement_sync_status_payload(now),
            "source": "okx_private_api_current_positions",
        }

    def _okx_position_history_mirror_sync_status_payload(
        self,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return diagnostics for the account-level OKX history mirror sync."""

        now = now or datetime.now(UTC)
        task = getattr(self, "_okx_position_history_mirror_sync_task", None)
        started_at = getattr(self, "_okx_position_history_mirror_sync_last_started_at", None)
        finished_at = getattr(self, "_okx_position_history_mirror_sync_last_finished_at", None)
        last_row = getattr(self, "_okx_position_history_mirror_sync_last_row", None)
        last_finished_age_seconds = (
            max((now - finished_at).total_seconds(), 0.0)
            if isinstance(finished_at, datetime)
            else None
        )
        interval_seconds = self.okx_position_history_mirror_sync_interval_seconds()
        status = "pending"
        if task is not None and not task.done():
            status = "running"
        elif isinstance(last_row, dict):
            status = str(last_row.get("status") or "ok").lower() or "ok"
        elif getattr(self, "_okx_position_history_mirror_sync_last_error", None):
            status = "degraded"
        return {
            "status": status,
            "task_running": bool(task is not None and not task.done()),
            "last_started_at": started_at.isoformat() if isinstance(started_at, datetime) else None,
            "last_finished_at": (
                finished_at.isoformat() if isinstance(finished_at, datetime) else None
            ),
            "last_finished_age_seconds": (
                round(last_finished_age_seconds, 3)
                if last_finished_age_seconds is not None
                else None
            ),
            "last_error": getattr(self, "_okx_position_history_mirror_sync_last_error", None),
            "last_row": last_row,
            "success_count": int(
                getattr(self, "_okx_position_history_mirror_sync_success_count", 0) or 0
            ),
            "failure_count": int(
                getattr(self, "_okx_position_history_mirror_sync_failure_count", 0) or 0
            ),
            "interval_seconds": round(interval_seconds, 3),
        }

    def _okx_position_settlement_sync_status_payload(
        self,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return diagnostics for the non-blocking OKX official settlement sync."""

        now = now or datetime.now(UTC)
        task = getattr(self, "_okx_position_settlement_sync_task", None)
        started_at = getattr(self, "_okx_position_settlement_sync_last_started_at", None)
        finished_at = getattr(self, "_okx_position_settlement_sync_last_finished_at", None)
        last_row = getattr(self, "_okx_position_settlement_sync_last_row", None)
        last_finished_age_seconds = (
            max((now - finished_at).total_seconds(), 0.0)
            if isinstance(finished_at, datetime)
            else None
        )
        status = "pending"
        if task is not None and not task.done():
            status = "running"
        elif isinstance(last_row, dict):
            status = str(last_row.get("status") or "ok").lower() or "ok"
        elif getattr(self, "_okx_position_settlement_sync_last_error", None):
            status = "degraded"
        return {
            "status": status,
            "task_running": bool(task is not None and not task.done()),
            "last_started_at": started_at.isoformat() if isinstance(started_at, datetime) else None,
            "last_finished_at": (
                finished_at.isoformat() if isinstance(finished_at, datetime) else None
            ),
            "last_finished_age_seconds": (
                round(last_finished_age_seconds, 3)
                if last_finished_age_seconds is not None
                else None
            ),
            "last_error": getattr(self, "_okx_position_settlement_sync_last_error", None),
            "last_row": last_row,
            "success_count": int(
                getattr(self, "_okx_position_settlement_sync_success_count", 0) or 0
            ),
            "failure_count": int(
                getattr(self, "_okx_position_settlement_sync_failure_count", 0) or 0
            ),
            "retry_interval_seconds": round(
                self.okx_position_settlement_sync_interval_seconds(),
                3,
            ),
        }

    def _okx_order_fact_sync_status_payload(
        self,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return diagnostics for the non-blocking OKX order-fact sync."""

        now = now or datetime.now(UTC)
        task = getattr(self, "_okx_order_fact_sync_task", None)
        started_at = getattr(self, "_okx_order_fact_sync_last_started_at", None)
        finished_at = getattr(self, "_okx_order_fact_sync_last_finished_at", None)
        last_row = getattr(self, "_okx_order_fact_sync_last_row", None)
        last_finished_age_seconds = (
            max((now - finished_at).total_seconds(), 0.0)
            if isinstance(finished_at, datetime)
            else None
        )
        status = "pending"
        if task is not None and not task.done():
            status = "running"
        elif isinstance(last_row, dict):
            status = str(last_row.get("status") or "ok").lower() or "ok"
        elif getattr(self, "_okx_order_fact_sync_last_error", None):
            status = "degraded"
        return {
            "status": status,
            "task_running": bool(task is not None and not task.done()),
            "last_started_at": started_at.isoformat() if isinstance(started_at, datetime) else None,
            "last_finished_at": (
                finished_at.isoformat() if isinstance(finished_at, datetime) else None
            ),
            "last_finished_age_seconds": (
                round(last_finished_age_seconds, 3)
                if last_finished_age_seconds is not None
                else None
            ),
            "last_error": getattr(self, "_okx_order_fact_sync_last_error", None),
            "success_count": int(getattr(self, "_okx_order_fact_sync_success_count", 0) or 0),
            "failure_count": int(getattr(self, "_okx_order_fact_sync_failure_count", 0) or 0),
            "normal_interval_seconds": round(self.okx_order_fact_sync_interval_seconds(), 3),
            "degraded_interval_seconds": round(
                self.okx_order_fact_sync_degraded_interval_seconds(),
                3,
            ),
        }

    def _okx_authoritative_sync_entry_block_reason(
        self,
        now: datetime | None = None,
    ) -> str | None:
        """Block new entries when OKX/local current-state truth is not healthy."""

        payload = self._okx_authoritative_sync_status_payload(now)
        status = str(payload.get("status") or "").lower()
        requires_attention = int(payload.get("last_requires_attention_count") or 0)
        last_error = str(payload.get("last_error") or "").strip()
        if status in {"warning", "stale"}:
            reason = "OKX 自动对账异常"
            if status == "stale":
                reason = "OKX 自动对账已过期"
            if last_error:
                reason = f"{reason}：{last_error}"
            return f"{reason}；暂停新开仓，等待 OKX 与本地后台状态恢复一致。"
        if requires_attention > 0:
            detail = self._okx_authoritative_sync_attention_detail(
                payload.get("last_samples"),
            )
            detail_text = f" 具体差异：{detail}；" if detail else ""
            return (
                f"OKX 自动对账发现 {requires_attention} 个当前状态差异需要复核；"
                f"{detail_text}暂停新开仓，等待状态对齐后再恢复。"
            )
        return None

    @staticmethod
    def _okx_authoritative_sync_attention_detail(samples: Any) -> str | None:
        """Format the attention sample that made OKX/local current state unsafe."""

        if not isinstance(samples, list):
            return None
        details: list[str] = []
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("requires_attention") is not True:
                continue
            kind = str(sample.get("kind") or "unknown").strip() or "unknown"
            pieces = [kind]
            symbol = str(sample.get("symbol") or "").strip()
            side = str(sample.get("side") or "").strip()
            exchange_order_id = str(sample.get("exchange_order_id") or "").strip()
            note = safe_error_text(sample.get("note"), limit=160) if sample.get("note") else None
            error = safe_error_text(sample.get("error"), limit=120) if sample.get("error") else None
            if symbol:
                pieces.append(symbol)
            if side:
                pieces.append(side)
            if exchange_order_id:
                pieces.append(f"订单 {exchange_order_id}")
            if note:
                pieces.append(note)
            if error:
                pieces.append(f"错误：{error}")
            details.append(" / ".join(pieces))
            if len(details) >= 3:
                break
        return "；".join(details) if details else None

    @staticmethod
    def _okx_authoritative_sync_result_summary(
        result: Any,
        *,
        sample_limit: int = 8,
    ) -> dict[str, Any]:
        """Summarize one OKX sync result without exposing large raw payloads."""

        rows = result if isinstance(result, list) else []
        kind_counts: Counter[str] = Counter()
        samples: list[dict[str, Any]] = []
        requires_attention_count = 0
        degraded_count = 0
        for row in rows:
            if not isinstance(row, dict):
                kind_counts["unknown"] += 1
                continue
            kind = str(row.get("kind") or "legacy_reconciled").strip() or "legacy_reconciled"
            kind_counts[kind] += 1
            if row.get("requires_attention") is True:
                requires_attention_count += 1
            if row.get("degraded") is True:
                degraded_count += 1
            if len(samples) >= sample_limit:
                continue
            samples.append(
                {
                    "kind": kind,
                    "symbol": row.get("symbol"),
                    "side": row.get("side"),
                    "exchange_order_id": row.get("exchange_order_id"),
                    "requires_attention": bool(row.get("requires_attention") is True),
                    "degraded": bool(row.get("degraded") is True),
                    "status": row.get("status"),
                    "okx_pull_available": row.get("okx_pull_available"),
                    "error": safe_error_text(row.get("error"), limit=180)
                    if row.get("error") is not None
                    else None,
                    "note": safe_error_text(row.get("note"), limit=180)
                    if row.get("note") is not None
                    else None,
                }
            )
        return {
            "count": len(rows),
            "kinds": dict(kind_counts),
            "requires_attention_count": requires_attention_count,
            "degraded_count": degraded_count,
            "samples": samples,
        }

    async def _sync_okx_order_facts_for_loop(self) -> dict[str, Any]:
        factory = getattr(self, "okx_order_fact_sync_factory", None)
        if factory is None:
            return {
                "kind": "order_fact_sync",
                "requires_attention": False,
                "note": "OKX order fact sync factory is not configured in this test runtime",
                "order_fact_sync": {"status": "skipped", "reason": "factory_not_configured"},
            }
        mode = "live" if mode_manager.mode.value == "live" else "paper"
        report = await factory(
            mode=mode,
            lookback_hours=24,
            timeout_seconds=max(3.0, min(6.0, self.round_start_reconcile_timeout_seconds() * 0.45)),
        ).sync()
        status = str(report.get("status") or "unknown").lower()
        unverified_count = int(report.get("unverified_count") or 0)
        position_confirmed_count = int(report.get("position_confirmed_count") or 0)
        okx_pull_available = bool(report.get("okx_pull_available") is not False)
        local_checked = int(report.get("local_checked") or 0)
        pull_error = safe_error_text(report.get("error"), limit=180) if report.get("error") else None
        degraded = not okx_pull_available
        requires_attention = unverified_count > 0 or (
            okx_pull_available and status in {"critical", "error", "unavailable"}
        )
        if degraded:
            note = (
                "OKX 订单事实同步降级：本轮未能从 OKX 拉取原生订单/成交数据，"
                f"status={status}, local_checked={local_checked}, "
                f"confirmed={int(report.get('confirmed_count') or 0)}, "
                f"unverified={unverified_count}"
            )
            if pull_error:
                note = f"{note}, error={pull_error}"
            note = f"{note}；当前持仓对账已单独执行，本轮不把拉取失败误判为当前状态差异。"
        elif requires_attention:
            note = (
                "OKX 订单事实同步发现本地已成交订单未被 OKX 原生成交确认："
                f"status={status}, local_checked={local_checked}, "
                f"confirmed={int(report.get('confirmed_count') or 0)}, "
                f"position_confirmed={position_confirmed_count}, "
                f"unverified={unverified_count}；需要复核后再允许新开仓。"
            )
        else:
            note = (
                f"OKX 订单事实同步正常：status={status}, "
                f"local_checked={local_checked}, "
                f"confirmed={int(report.get('confirmed_count') or 0)}, "
                f"position_confirmed={position_confirmed_count}, "
                f"unverified={unverified_count}, "
                f"backfilled={int(report.get('backfilled_count') or 0)}, "
                "position_history="
                f"{int(report.get('position_history_backfilled_count') or 0)}+"
                f"{int(report.get('position_history_updated_count') or 0)}"
            )
        return {
            "kind": "order_fact_sync",
            "symbol": None,
            "side": None,
            "exchange_order_id": None,
            "requires_attention": requires_attention,
            "degraded": degraded,
            "status": status,
            "okx_pull_available": okx_pull_available,
            "error": pull_error,
            "note": note,
            "order_fact_sync": report,
        }

    def request_okx_order_fact_recovery(self, _execution_mode: str) -> None:
        """Force a background fill sync after an exchange-confirmed local write failure."""

        self._start_okx_order_fact_sync_background(force=True)

    def _start_okx_order_fact_sync_background(self, *, force: bool = False) -> None:
        """Start optional OKX order-fact sync without delaying current-position sync."""

        if getattr(self, "okx_order_fact_sync_factory", None) is None:
            return
        task = getattr(self, "_okx_order_fact_sync_task", None)
        if task is not None and not task.done():
            return
        now = datetime.now(UTC)
        last_finished_at = getattr(self, "_okx_order_fact_sync_last_finished_at", None)
        if not force and isinstance(last_finished_at, datetime):
            last_row = getattr(self, "_okx_order_fact_sync_last_row", None)
            degraded = bool(
                isinstance(last_row, dict)
                and (
                    last_row.get("degraded") is True
                    or last_row.get("okx_pull_available") is False
                    or str(last_row.get("status") or "").lower() in {"warning", "error", "critical"}
                )
            )
            min_interval = (
                self.okx_order_fact_sync_degraded_interval_seconds()
                if degraded
                else self.okx_order_fact_sync_interval_seconds()
            )
            if (now - last_finished_at).total_seconds() < min_interval:
                return
        self._okx_order_fact_sync_last_started_at = now
        task = asyncio.create_task(self._sync_okx_order_facts_for_loop())
        self._okx_order_fact_sync_task = task
        task.add_done_callback(self._consume_okx_order_fact_sync_result)

    def _consume_okx_order_fact_sync_result(self, task: asyncio.Task) -> None:
        """Persist optional order-fact sync outcome as diagnostics only."""

        if task.cancelled():
            return
        finished_at = datetime.now(UTC)
        self._okx_order_fact_sync_last_finished_at = finished_at
        try:
            row = task.result()
        except Exception as exc:
            error = safe_error_text(exc, limit=180)
            self._okx_order_fact_sync_last_error = error
            self._okx_order_fact_sync_failure_count = (
                int(getattr(self, "_okx_order_fact_sync_failure_count", 0) or 0) + 1
            )
            self._okx_order_fact_sync_last_row = {
                "kind": "order_fact_sync",
                "symbol": None,
                "side": None,
                "exchange_order_id": None,
                "requires_attention": False,
                "degraded": True,
                "status": "degraded",
                "okx_pull_available": False,
                "error": error,
                "note": (
                    "OKX 订单事实后台同步降级：本轮未能拉取原生订单/成交数据；"
                    "当前持仓对账已独立完成，本次降级不作为当前状态差异阻断新开仓。"
                ),
            }
        else:
            self._okx_order_fact_sync_last_error = None
            self._okx_order_fact_sync_success_count = (
                int(getattr(self, "_okx_order_fact_sync_success_count", 0) or 0) + 1
            )
            self._okx_order_fact_sync_last_row = row if isinstance(row, dict) else {
                "kind": "order_fact_sync",
                "requires_attention": False,
                "degraded": True,
                "status": "degraded",
                "okx_pull_available": False,
                "error": "invalid_order_fact_sync_result",
                "note": "OKX 订单事实后台同步返回了无效结果；当前持仓对账不受影响。",
            }
        if getattr(self, "_okx_order_fact_sync_task", None) is task:
            self._okx_order_fact_sync_task = None

    async def _okx_position_history_mirror_sync_loop(self) -> None:
        """Keep the local OKX positions-history mirror current account-wide."""

        while self._running:
            factory = getattr(self, "okx_position_history_mirror_sync_factory", None)
            if factory is None:
                await asyncio.sleep(self.okx_position_history_mirror_sync_interval_seconds())
                continue
            started_at = datetime.now(UTC)
            self._okx_position_history_mirror_sync_last_started_at = started_at
            try:
                mode = "live" if mode_manager.mode.value == "live" else "paper"
                row = await factory(
                    mode=mode,
                    lookback_hours=72,
                    limit=100,
                    max_pages=5,
                    timeout_seconds=8.0,
                ).sync_once()
                if not isinstance(row, dict):
                    row = {
                        "status": "degraded",
                        "mode": mode,
                        "source": "okx_position_history_account_sync",
                        "okx_pull_available": False,
                        "error": "invalid_position_history_mirror_sync_result",
                    }
                self._okx_position_history_mirror_sync_last_row = row
                error = safe_error_text(row.get("error"), limit=220) if row.get("error") else None
                self._okx_position_history_mirror_sync_last_error = error
                if str(row.get("status") or "").lower() in {"degraded", "error", "critical"}:
                    self._okx_position_history_mirror_sync_failure_count = (
                        int(
                            getattr(
                                self,
                                "_okx_position_history_mirror_sync_failure_count",
                                0,
                            )
                            or 0
                        )
                        + 1
                    )
                else:
                    self._okx_position_history_mirror_sync_success_count = (
                        int(
                            getattr(
                                self,
                                "_okx_position_history_mirror_sync_success_count",
                                0,
                            )
                            or 0
                        )
                        + 1
                    )
                    changed_count = int(row.get("inserted_count") or 0) + int(
                        row.get("updated_count") or 0
                    )
                    if changed_count > 0:
                        row["authoritative_outcome_feedback"] = (
                            await self.expert_memory_service.backfill_trade_reflections(mode)
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = safe_error_text(exc, limit=220)
                self._okx_position_history_mirror_sync_last_error = error
                self._okx_position_history_mirror_sync_last_row = {
                    "status": "degraded",
                    "mode": mode_manager.mode.value,
                    "source": "okx_position_history_account_sync",
                    "okx_pull_available": False,
                    "live_count": 0,
                    "upserted_count": 0,
                    "inserted_count": 0,
                    "updated_count": 0,
                    "skipped_count": 0,
                    "error": error,
                }
                self._okx_position_history_mirror_sync_failure_count = (
                    int(getattr(self, "_okx_position_history_mirror_sync_failure_count", 0) or 0)
                    + 1
                )
                logger.warning(
                    "OKX position history mirror background sync failed",
                    error=error,
                )
            finally:
                self._okx_position_history_mirror_sync_last_finished_at = datetime.now(UTC)
            await asyncio.sleep(self.okx_position_history_mirror_sync_interval_seconds())

    async def _okx_position_settlement_sync_loop(self) -> None:
        """Retry official OKX settlement for recently closed local positions."""

        while self._running:
            factory = getattr(self, "okx_position_settlement_sync_factory", None)
            if factory is None:
                await asyncio.sleep(self.okx_position_settlement_sync_interval_seconds())
                continue
            started_at = datetime.now(UTC)
            self._okx_position_settlement_sync_last_started_at = started_at
            try:
                mode = "live" if mode_manager.mode.value == "live" else "paper"
                row = await factory(
                    mode=mode,
                    retry_seconds=self.okx_position_settlement_sync_interval_seconds(),
                    timeout_seconds=6.0,
                    limit=10,
                ).sync_once()
                self._okx_position_settlement_sync_last_row = row
                self._okx_position_settlement_sync_last_error = None
                self._okx_position_settlement_sync_success_count = (
                    int(getattr(self, "_okx_position_settlement_sync_success_count", 0) or 0) + 1
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = safe_error_text(exc, limit=220)
                self._okx_position_settlement_sync_last_error = error
                self._okx_position_settlement_sync_last_row = {
                    "status": "degraded",
                    "mode": mode_manager.mode.value,
                    "error": error,
                    "checked_count": 0,
                    "reconciled_count": 0,
                    "exception_count": 0,
                    "skipped_count": 0,
                    "samples": [],
                }
                self._okx_position_settlement_sync_failure_count = (
                    int(getattr(self, "_okx_position_settlement_sync_failure_count", 0) or 0) + 1
                )
                logger.warning(
                    "OKX position settlement background sync failed",
                    error=error,
                )
            finally:
                self._okx_position_settlement_sync_last_finished_at = datetime.now(UTC)
            await asyncio.sleep(self.okx_position_settlement_sync_interval_seconds())

    async def _run_shadow_backtest_update(self, *, limit: int) -> int:
        self._shadow_backtest_update_last_started_at = datetime.now(UTC)
        return await self.shadow_backtest_service.update_due(limit=limit)

    def _shadow_backtest_maintenance_status(self) -> dict[str, Any]:
        task = getattr(self, "_shadow_backtest_update_task", None)
        started_at = getattr(self, "_shadow_backtest_update_last_started_at", None)
        finished_at = getattr(self, "_shadow_backtest_update_last_finished_at", None)
        running_seconds = None
        if isinstance(started_at, datetime) and (not isinstance(finished_at, datetime) or finished_at < started_at):
            running_seconds = round(max((datetime.now(UTC) - started_at).total_seconds(), 0.0), 3)
        return {
            "read_only": True,
            "is_entry_gate": False,
            "running": bool(task is not None and not task.done()),
            "running_seconds": running_seconds,
            "last_started_at": started_at.isoformat() if isinstance(started_at, datetime) else None,
            "last_finished_at": (
                finished_at.isoformat() if isinstance(finished_at, datetime) else None
            ),
            "last_completed_count": getattr(self, "_shadow_backtest_update_last_count", None),
            "last_error": getattr(self, "_shadow_backtest_update_last_error", None),
            "success_count": int(getattr(self, "_shadow_backtest_update_success_count", 0) or 0),
            "failure_count": int(getattr(self, "_shadow_backtest_update_failure_count", 0) or 0),
            "diagnostic_boundary": (
                "影子复盘维护只负责训练/记忆回灌，不是开仓门槛；market-only 轮次只触发后台刷新，"
                "不能同步拖慢 market AI 启动。"
            ),
        }

    def _consume_shadow_backtest_update_result(self, task: asyncio.Task) -> None:
        finished_at = datetime.now(UTC)
        self._shadow_backtest_update_last_finished_at = finished_at
        try:
            completed_count = int(task.result() or 0)
        except asyncio.CancelledError:
            if getattr(self, "_shadow_backtest_update_task", None) is task:
                self._shadow_backtest_update_task = None
            return
        except Exception as exc:
            self._shadow_backtest_update_last_error = safe_error_text(exc, limit=180)
            self._shadow_backtest_update_failure_count = (
                int(getattr(self, "_shadow_backtest_update_failure_count", 0) or 0) + 1
            )
        else:
            self._shadow_backtest_update_last_error = None
            self._shadow_backtest_update_last_count = completed_count
            self._shadow_backtest_update_success_count = (
                int(getattr(self, "_shadow_backtest_update_success_count", 0) or 0) + 1
            )
        if getattr(self, "_shadow_backtest_update_task", None) is task:
            self._shadow_backtest_update_task = None

    async def _update_shadow_backtests_for_round(
        self,
        *,
        analysis_scope: str,
        results: dict[str, Any],
    ) -> None:
        """Run shadow-backtest maintenance without blocking market entry discovery."""

        task = getattr(self, "_shadow_backtest_update_task", None)
        if task is not None and not task.done():
            results["shadow_backtest_maintenance"] = self._shadow_backtest_maintenance_status()
            return

        update_limit = (
            SHADOW_BACKTEST_MARKET_BACKGROUND_UPDATE_LIMIT
            if analysis_scope == "market"
            else SHADOW_BACKTEST_FOREGROUND_UPDATE_LIMIT
        )
        task = asyncio.create_task(
            self._run_shadow_backtest_update(
                limit=update_limit,
            )
        )
        self._shadow_backtest_update_task = task
        task.add_done_callback(self._consume_shadow_backtest_update_result)
        results["shadow_backtest_maintenance"] = {
            **self._shadow_backtest_maintenance_status(),
            "started_in_background": True,
            "update_limit": update_limit,
        }

    async def _run_stale_entry_candidate_expire(self) -> int:
        self._stale_entry_expire_last_started_at = datetime.now(UTC)
        return await self.stale_entry_candidate_expirer.expire()

    def _stale_entry_candidate_maintenance_status(self) -> dict[str, Any]:
        task = getattr(self, "_stale_entry_expire_task", None)
        started_at = getattr(self, "_stale_entry_expire_last_started_at", None)
        finished_at = getattr(self, "_stale_entry_expire_last_finished_at", None)
        running_seconds = None
        if isinstance(started_at, datetime) and (
            not isinstance(finished_at, datetime) or finished_at < started_at
        ):
            running_seconds = round(max((datetime.now(UTC) - started_at).total_seconds(), 0.0), 3)
        return {
            "read_only": True,
            "is_entry_gate": False,
            "running": bool(task is not None and not task.done()),
            "running_seconds": running_seconds,
            "last_started_at": started_at.isoformat() if isinstance(started_at, datetime) else None,
            "last_finished_at": (
                finished_at.isoformat() if isinstance(finished_at, datetime) else None
            ),
            "last_expired_count": getattr(self, "_stale_entry_expire_last_count", None),
            "last_error": getattr(self, "_stale_entry_expire_last_error", None),
            "success_count": int(getattr(self, "_stale_entry_expire_success_count", 0) or 0),
            "failure_count": int(getattr(self, "_stale_entry_expire_failure_count", 0) or 0),
            "maintenance_scope": STALE_ENTRY_EXPIRE_BACKGROUND_LIMIT_DESCRIPTION,
            "diagnostic_boundary": (
                "过期开仓候选维护只负责补齐旧候选终态和清理旧等待状态，不是开仓门槛；"
                "交易主轮次只触发后台单飞维护，不能同步拖慢 market AI 或持仓复盘。"
            ),
        }

    def _consume_stale_entry_candidate_expire_result(self, task: asyncio.Task) -> None:
        finished_at = datetime.now(UTC)
        self._stale_entry_expire_last_finished_at = finished_at
        try:
            expired_count = int(task.result() or 0)
        except asyncio.CancelledError:
            if getattr(self, "_stale_entry_expire_task", None) is task:
                self._stale_entry_expire_task = None
            return
        except Exception as exc:
            self._stale_entry_expire_last_error = safe_error_text(exc, limit=180)
            self._stale_entry_expire_failure_count = (
                int(getattr(self, "_stale_entry_expire_failure_count", 0) or 0) + 1
            )
        else:
            self._stale_entry_expire_last_error = None
            self._stale_entry_expire_last_count = expired_count
            self._stale_entry_expire_success_count = (
                int(getattr(self, "_stale_entry_expire_success_count", 0) or 0) + 1
            )
        if getattr(self, "_stale_entry_expire_task", None) is task:
            self._stale_entry_expire_task = None

    async def _update_stale_entry_candidates_for_round(
        self,
        *,
        results: dict[str, Any],
    ) -> None:
        """Trigger stale-entry cleanup without spending the trading round budget."""

        task = getattr(self, "_stale_entry_expire_task", None)
        if task is not None and not task.done():
            results["stale_entry_maintenance"] = (
                self._stale_entry_candidate_maintenance_status()
            )
            return

        task = asyncio.create_task(self._run_stale_entry_candidate_expire())
        self._stale_entry_expire_task = task
        task.add_done_callback(self._consume_stale_entry_candidate_expire_result)
        results["stale_entry_maintenance"] = {
            **self._stale_entry_candidate_maintenance_status(),
            "started_in_background": True,
        }

    def strategy_learning_context_timeout_seconds(self) -> float:
        """Return the hard budget for strategy-learning context in the trading loop."""

        interval = max(10.0, float(settings.decision_interval_seconds or 60))
        configured = float(DEFAULT_TRADING_PARAMS.strategy_learning.runtime_context_timeout_seconds)
        return max(0.5, min(configured, interval * 0.20))

    def strategy_learning_context_wait_timeout_seconds(
        self,
        analysis_scope: str | None = None,
    ) -> float:
        """Return how long the current round may wait for learning context."""

        base_timeout = self.strategy_learning_context_timeout_seconds()
        scope = analysis_scope or _analysis_scope_context.get()
        if scope == "market":
            interval = max(10.0, float(settings.decision_interval_seconds or 60))
            return max(0.5, min(base_timeout, interval * 0.06, 3.0))
        return base_timeout

    def strategy_learning_perf_timeout_seconds(self) -> float:
        """Return the short budget for historical performance context."""

        timeout = max(
            0.2,
            float(DEFAULT_TRADING_PARAMS.strategy_learning.runtime_perf_timeout_seconds),
        )
        if _analysis_scope_context.get() == "market":
            return min(timeout, 0.35)
        return timeout

    def strategy_learning_account_timeout_seconds(self) -> float:
        """Return the short budget for account-equity context."""

        return max(
            0.2,
            float(DEFAULT_TRADING_PARAMS.strategy_learning.runtime_account_timeout_seconds),
        )

    def _safe_set_strategy_context_stage(self, stage: str) -> None:
        """Update strategy-context diagnostics without breaking the trading decision path."""

        try:
            self._set_loop_stage(stage, heartbeat=False)
        except Exception:
            logger.debug("strategy context stage update skipped", stage=stage)

    async def _bounded_strategy_context_value(
        self,
        label: str,
        awaitable: Any,
        fallback: Any,
        timeout_seconds: float,
        timings: dict[str, Any] | None = None,
    ) -> Any:
        """Read optional strategy context without letting slow IO block the round."""

        started_monotonic = asyncio.get_running_loop().time()
        queue_started_monotonic = started_monotonic
        queue_wait_seconds = 0.0
        status = "ok"
        gate = self._strategy_context_io_gate()
        acquired = False
        try:
            self._safe_set_strategy_context_stage(f"strategy_context:{label}")
            budget_seconds = max(float(timeout_seconds), 0.0)
            try:
                await asyncio.wait_for(gate.acquire(), timeout=budget_seconds)
                acquired = True
            except TimeoutError:
                if inspect.iscoroutine(awaitable):
                    awaitable.close()
                status = "queue_timeout"
                logger.warning(
                    "strategy context database queue timed out; using baseline value",
                    stage=label,
                    timeout_seconds=round(budget_seconds, 3),
                )
                return fallback
            queue_wait_seconds = max(
                asyncio.get_running_loop().time() - queue_started_monotonic,
                0.0,
            )
            remaining_seconds = max(budget_seconds - queue_wait_seconds, 0.0)
            if remaining_seconds <= 0.0:
                if inspect.iscoroutine(awaitable):
                    awaitable.close()
                status = "queue_timeout"
                return fallback
            task = asyncio.ensure_future(awaitable)
            done, pending = await asyncio.wait(
                {task},
                timeout=remaining_seconds,
            )
            if pending:
                await drain_cancelled_tasks(pending, timeout_seconds=0.05)
                status = "timeout"
                logger.warning(
                    "strategy context stage timed out; using baseline value",
                    stage=label,
                    timeout_seconds=round(budget_seconds, 3),
                )
                return fallback
            return next(iter(done)).result()
        except Exception as exc:
            status = "error"
            logger.warning(
                "strategy context stage failed; using baseline value",
                stage=label,
                error=safe_error_text(exc),
            )
            return fallback
        finally:
            if acquired:
                gate.release()
            if timings is not None:
                elapsed = max(asyncio.get_running_loop().time() - started_monotonic, 0.0)
                timings[label] = {
                    "duration_seconds": round(elapsed, 6),
                    "queue_wait_seconds": round(queue_wait_seconds, 6),
                    "timeout_seconds": round(max(float(timeout_seconds), 0.0), 6),
                    "status": status,
                }

    def _strategy_context_io_gate(self) -> asyncio.Semaphore:
        """Bound concurrent history queries shared by market and position loops."""

        gate = getattr(self, "_strategy_context_io_semaphore", None)
        if isinstance(gate, asyncio.Semaphore):
            return gate
        gate = asyncio.Semaphore(STRATEGY_CONTEXT_IO_CONCURRENCY)
        self._strategy_context_io_semaphore = gate
        return gate

    def _strategy_learning_refresh_tasks(self) -> dict[str, asyncio.Task]:
        tasks = getattr(self, "_strategy_learning_context_refresh_tasks", None)
        if isinstance(tasks, dict):
            return tasks
        tasks = {}
        self._strategy_learning_context_refresh_tasks = tasks
        return tasks

    def _strategy_learning_context_cache_store(self) -> dict[str, Any]:
        cache = getattr(self, "_strategy_learning_context_cache", None)
        if isinstance(cache, dict):
            return cache
        cache = {}
        self._strategy_learning_context_cache = cache
        return cache

    def _strategy_context_performance_snapshot_store(self) -> dict[str, Any]:
        cache = getattr(self, "_strategy_context_performance_snapshot_cache", None)
        if isinstance(cache, dict):
            return cache
        cache = {}
        self._strategy_context_performance_snapshot_cache = cache
        return cache

    def _strategy_context_performance_refresh_tasks(self) -> dict[str, asyncio.Task]:
        tasks = getattr(self, "_strategy_context_performance_refresh_task_store", None)
        if isinstance(tasks, dict):
            return tasks
        tasks = {}
        self._strategy_context_performance_refresh_task_store = tasks
        return tasks

    async def _refresh_strategy_context_performance_value(
        self,
        label: str,
        loader: Any,
    ) -> tuple[str, bool, Any, dict[str, Any]]:
        """Load one optional performance value outside the analysis-round budget."""

        loop = asyncio.get_running_loop()
        started = loop.time()
        gate = self._strategy_context_io_gate()
        acquired = False
        queue_wait_seconds = 0.0
        status = "ok"
        error = ""
        try:
            await asyncio.wait_for(
                gate.acquire(),
                timeout=STRATEGY_CONTEXT_PERFORMANCE_REFRESH_TIMEOUT_SECONDS,
            )
            acquired = True
            queue_wait_seconds = max(loop.time() - started, 0.0)
            remaining_seconds = max(
                STRATEGY_CONTEXT_PERFORMANCE_REFRESH_TIMEOUT_SECONDS - queue_wait_seconds,
                0.0,
            )
            if remaining_seconds <= 0.0:
                status = "queue_timeout"
                error = "background_refresh_queue_timeout"
                return label, False, None, {
                    "status": status,
                    "duration_seconds": round(max(loop.time() - started, 0.0), 6),
                    "queue_wait_seconds": round(queue_wait_seconds, 6),
                    "error": error,
                }
            value = await asyncio.wait_for(
                loader(),
                timeout=remaining_seconds,
            )
            return label, True, value, {
                "status": status,
                "duration_seconds": round(max(loop.time() - started, 0.0), 6),
                "queue_wait_seconds": round(queue_wait_seconds, 6),
            }
        except TimeoutError:
            status = "queue_timeout" if not acquired else "timeout"
            error = (
                "background_refresh_queue_timeout"
                if not acquired
                else "background_refresh_timeout"
            )
        except Exception as exc:
            status = "error"
            error = safe_error_text(exc, limit=160)
        finally:
            if acquired:
                gate.release()

        logger.warning(
            "strategy context performance refresh failed",
            stage=label,
            status=status,
            error=error,
        )
        return label, False, None, {
            "status": status,
            "duration_seconds": round(max(loop.time() - started, 0.0), 6),
            "queue_wait_seconds": round(queue_wait_seconds, 6),
            "error": error,
        }

    def _start_strategy_context_performance_refresh(self, mode: str) -> asyncio.Task:
        """Start one bounded performance snapshot refresh per execution mode."""

        selected_mode = "live" if mode == "live" else "paper"
        tasks = self._strategy_context_performance_refresh_tasks()
        existing = tasks.get(selected_mode)
        if existing is not None and not existing.done():
            return existing

        async def _refresh() -> dict[str, Any]:
            loaders = {
                "position_performance": lambda: self._strategy_context_position_performance(
                    selected_mode
                ),
                "model_contribution_perf": lambda: self._recent_model_contribution_performance(
                    selected_mode
                ),
            }
            rows = await asyncio.gather(
                *[
                    self._refresh_strategy_context_performance_value(label, loader)
                    for label, loader in loaders.items()
                ]
            )
            cache = self._strategy_context_performance_snapshot_store()
            previous = cache.get(selected_mode)
            previous_values = (
                dict(previous.get("values") or {})
                if isinstance(previous, dict)
                else {}
            )
            values = dict(previous_values)
            timings: dict[str, Any] = {}
            successful_count = 0
            failed_labels: list[str] = []
            for label, succeeded, value, timing in rows:
                if label == "position_performance":
                    for performance_label in (
                        "daily_perf",
                        "today_side_perf",
                        "multiday_side_perf",
                        "symbol_side_perf",
                    ):
                        timings[performance_label] = {
                            **timing,
                            "source": "shared_position_performance_snapshot",
                        }
                else:
                    timings[label] = timing
                if succeeded:
                    if label == "position_performance":
                        bundle = self._safe_dict(value)
                        for performance_label in (
                            "daily_perf",
                            "today_side_perf",
                            "multiday_side_perf",
                            "symbol_side_perf",
                        ):
                            values[performance_label] = self._json_safe_payload(
                                self._safe_dict(bundle.get(performance_label))
                            )
                            successful_count += 1
                    else:
                        values[label] = self._json_safe_payload(value)
                        successful_count += 1
                else:
                    if label == "position_performance":
                        failed_labels.extend(
                            (
                                "daily_perf",
                                "today_side_perf",
                                "multiday_side_perf",
                                "symbol_side_perf",
                            )
                        )
                    else:
                        failed_labels.append(label)

            if successful_count:
                version = int(previous.get("version") or 0) + 1 if isinstance(previous, dict) else 1
                entry = {
                    "created_at": datetime.now(UTC),
                    "values": values,
                    "version": version,
                    "refresh_timings": timings,
                    "failed_labels": failed_labels,
                }
                cache[selected_mode] = entry
                return entry

            if isinstance(previous, dict):
                entry = {
                    **previous,
                    "refresh_timings": timings,
                    "failed_labels": failed_labels,
                    "last_refresh_failed_at": datetime.now(UTC),
                }
                cache[selected_mode] = entry
                return entry

            return {
                "created_at": None,
                "values": {},
                "version": 0,
                "refresh_timings": timings,
                "failed_labels": failed_labels,
            }

        task = asyncio.create_task(_refresh())
        tasks[selected_mode] = task
        task.add_done_callback(_consume_task_result)
        return task

    async def _prime_strategy_context_performance_snapshot(self, mode: str) -> None:
        """Warm the shared performance snapshot before analysis loops begin."""

        task = self._start_strategy_context_performance_refresh(mode)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
        except TimeoutError:
            logger.warning(
                "strategy context performance warmup continues in background",
                mode=mode,
            )
        except Exception as exc:
            logger.warning(
                "strategy context performance warmup failed",
                mode=mode,
                error=safe_error_text(exc),
            )

    def _recent_strategy_context_performance_snapshot(
        self,
        mode: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Return the latest performance snapshot with explicit freshness semantics."""

        selected_mode = "live" if mode == "live" else "paper"
        entry = self._strategy_context_performance_snapshot_store().get(selected_mode)
        if not isinstance(entry, dict):
            return None, {
                "status": "baseline_background_refresh",
                "source": "background_performance_snapshot",
                "available": False,
                "version": 0,
            }
        created_at = entry.get("created_at")
        values = entry.get("values")
        if not isinstance(created_at, datetime) or not isinstance(values, dict):
            return None, {
                "status": "baseline_background_refresh",
                "source": "background_performance_snapshot",
                "available": False,
                "version": int(entry.get("version") or 0),
            }
        age_seconds = max((datetime.now(UTC) - created_at).total_seconds(), 0.0)
        if age_seconds > STRATEGY_CONTEXT_PERFORMANCE_SNAPSHOT_MAX_STALE_SECONDS:
            return None, {
                "status": "expired_background_refresh",
                "source": "background_performance_snapshot",
                "available": False,
                "version": int(entry.get("version") or 0),
                "age_seconds": round(age_seconds, 3),
            }
        return dict(values), {
            "status": (
                "fresh" if age_seconds <= STRATEGY_CONTEXT_PERFORMANCE_SNAPSHOT_FRESH_SECONDS else "stale"
            ),
            "source": "background_performance_snapshot",
            "available": True,
            "version": int(entry.get("version") or 0),
            "age_seconds": round(age_seconds, 3),
            "failed_labels": list(entry.get("failed_labels") or []),
            "last_refresh_timings": dict(entry.get("refresh_timings") or {}),
        }

    def _start_strategy_learning_context_refresh(
        self,
        *,
        mode: str,
        analysis_scope: str | None = None,
        strategy_learning: Any,
        context: dict[str, Any],
        open_positions: list[dict[str, Any]],
    ) -> asyncio.Task:
        selected_mode = "live" if mode == "live" else "paper"
        tasks = self._strategy_learning_refresh_tasks()
        existing = tasks.get(selected_mode)
        if existing is not None and not existing.done():
            return existing

        async def _refresh() -> dict[str, Any]:
            learned_context = await strategy_learning.apply_to_strategy_context(
                mode=selected_mode,
                strategy_context=dict(context),
                open_positions=open_positions,
                limit=DEFAULT_TRADING_PARAMS.strategy_learning.runtime_context_row_limit,
            )
            learned_context["strategy_learning_cache_status"] = "fresh"
            learned_context["strategy_learning_runtime_timeout_seconds"] = (
                self.strategy_learning_context_wait_timeout_seconds(analysis_scope)
            )
            self._strategy_learning_context_cache_store()[selected_mode] = {
                "created_at": datetime.now(UTC),
                "context": self._json_safe_payload(learned_context),
            }
            return learned_context

        task = asyncio.create_task(_refresh())
        tasks[selected_mode] = task
        task.add_done_callback(_consume_task_result)
        return task

    @staticmethod
    def _round_elapsed_seconds(started_at: datetime) -> float:
        return max((datetime.now(UTC) - started_at).total_seconds(), 0.0)

    @staticmethod
    def _remaining_monotonic_seconds(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(float(deadline) - asyncio.get_running_loop().time(), 0.0)

    def _position_review_stage_timeout_seconds(
        self,
        deadline: float | None,
        *,
        fallback_timeout: float | None = None,
        strategy_context: dict[str, Any] | None = None,
        stage: str | None = None,
        symbol: str | None = None,
    ) -> float:
        stage_timeout = max(
            0.25,
            float(fallback_timeout or self.position_review_stage_timeout_seconds()),
        )
        del strategy_context, symbol
        remaining = self._remaining_monotonic_seconds(deadline)
        if remaining is None:
            return stage_timeout
        reserve_ratio = 0.25
        if str(stage or "").lower() == "position_review_decision":
            reserve_ratio = 0.18
        reserve = min(2.0, max(0.25, stage_timeout * 0.03), remaining * reserve_ratio)
        return max(0.0, min(stage_timeout, remaining - reserve))

    def _append_position_review_budget_warning(
        self,
        *,
        results: dict[str, Any] | None,
        stage: str,
        symbol: str | None,
        remaining_seconds: float | None,
    ) -> None:
        if results is None:
            return
        symbol_label = symbol or "ALL"
        message = (
            f"持仓复盘本轮预算不足：{symbol_label} / {stage} 已顺延到下一轮；"
            "系统会用最新持仓和行情重新复盘。"
        )
        diagnostic = {
            "stage": stage,
            "kind": "position_review_round_budget_exhausted",
            "symbol": symbol_label,
            "remaining_budget_seconds": (
                round(float(remaining_seconds), 3)
                if remaining_seconds is not None
                else None
            ),
            "message": message,
        }
        results.setdefault("position_review_diagnostics", []).append(diagnostic)
        results.setdefault("warnings", []).append(
            {
                "model": "position_review",
                "symbol": symbol_label,
                "warning": message,
            }
        )

    def _round_budget_exhausted(self, started_at: datetime) -> bool:
        return self._round_elapsed_seconds(started_at) >= self.market_round_time_budget_seconds()

    def _market_ai_budget_exhausted(
        self,
        market_ai_started_at: datetime,
        *,
        strategy_context: dict[str, Any] | None = None,
        market_symbol_count: int | None = None,
    ) -> bool:
        elapsed_seconds = self._round_elapsed_seconds(market_ai_started_at)
        budget_seconds = self.market_round_time_budget_seconds(
            strategy_context=strategy_context,
            market_symbol_count=market_symbol_count,
        )
        if elapsed_seconds >= budget_seconds:
            return True
        reserve_seconds = self.market_symbol_start_reserve_seconds(
            strategy_context=strategy_context,
            market_symbol_count=market_symbol_count,
        )
        return (budget_seconds - elapsed_seconds) < reserve_seconds

    async def _runtime_heartbeat_loop(self) -> None:
        """Keep split-process dashboard heartbeat fresh while a round is busy."""

        while self._running:
            self._write_runtime_heartbeat()
            await asyncio.sleep(5.0)

    async def _okx_authoritative_sync_loop(self) -> None:
        """Continuously align current local open positions with OKX facts."""

        while self._running:
            interval = self.okx_authoritative_sync_interval_seconds()
            started_at = datetime.now(UTC)
            self._okx_authoritative_sync_started_at = started_at
            try:
                position_result = await asyncio.wait_for(
                    self.okx_sync_service.reconcile_positions(
                        "auto okx authoritative sync",
                        timeout_seconds=self.round_start_reconcile_timeout_seconds(),
                        lock_wait_seconds=0.1,
                    ),
                    timeout=self.round_start_reconcile_timeout_seconds() + 2.0,
                )
                result = list(position_result) if isinstance(position_result, list) else []
                order_fact_result = getattr(self, "_okx_order_fact_sync_last_row", None)
                if isinstance(order_fact_result, dict):
                    result.append(order_fact_result)
                self._start_okx_order_fact_sync_background()
                finished_at = datetime.now(UTC)
                self._okx_authoritative_sync_last_success_at = finished_at
                self._okx_authoritative_sync_last_error = None
                self._okx_authoritative_sync_last_duration_seconds = round(
                    (finished_at - started_at).total_seconds(),
                    6,
                )
                self._okx_authoritative_sync_last_result_count = (
                    len(result) if isinstance(result, list) else None
                )
                result_summary = self._okx_authoritative_sync_result_summary(result)
                self._okx_authoritative_sync_last_result_kinds = result_summary["kinds"]
                self._okx_authoritative_sync_last_requires_attention_count = result_summary[
                    "requires_attention_count"
                ]
                self._okx_authoritative_sync_last_degraded_count = result_summary[
                    "degraded_count"
                ]
                self._okx_authoritative_sync_last_samples = result_summary["samples"]
                self._okx_authoritative_sync_success_count = (
                    int(getattr(self, "_okx_authoritative_sync_success_count", 0) or 0) + 1
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                finished_at = datetime.now(UTC)
                self._okx_authoritative_sync_last_failure_at = finished_at
                self._okx_authoritative_sync_last_error = safe_error_text(exc, limit=180)
                self._okx_authoritative_sync_last_duration_seconds = round(
                    (finished_at - started_at).total_seconds(),
                    6,
                )
                self._okx_authoritative_sync_failure_count = (
                    int(getattr(self, "_okx_authoritative_sync_failure_count", 0) or 0) + 1
                )
                logger.warning(
                    "auto OKX authoritative sync failed",
                    error=self._okx_authoritative_sync_last_error,
                )
            await asyncio.sleep(interval)

    async def enforce_sl_tp_for_position_review(
        self,
        feature_vectors: dict[str, Any],
        *,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Enforce stop-loss/take-profit through a position-review boundary."""

        if open_positions is not None and self._callable_accepts_keyword(
            self._enforce_sl_tp,
            "open_positions",
        ):
            return await self._enforce_sl_tp(feature_vectors, open_positions=open_positions)
        return await self._enforce_sl_tp(feature_vectors)

    async def open_positions_context_for_position_review(self) -> list[dict[str, Any]]:
        """Return open-position context for the position-review service."""

        return await self.okx_sync_service.get_open_positions_context()

    async def review_open_positions_for_position_service(
        self,
        open_positions: list[dict[str, Any]],
        feature_vectors: dict[str, Any],
        *,
        results: dict[str, Any],
        round_decision_ids: set[int],
        position_entry_pause_reason: str | None,
        max_groups_override: int,
        round_deadline_monotonic: float | None = None,
    ) -> tuple[list[tuple[str, str, DecisionOutput, Any, int | None]], set[tuple[str, str]]]:
        """Create position-review candidates through an explicit boundary."""

        review_kwargs = {
            "results": results,
            "round_decision_ids": round_decision_ids,
            "position_entry_pause_reason": position_entry_pause_reason,
            "max_groups_override": max_groups_override,
        }
        try:
            from inspect import Parameter, signature

            parameters = signature(self._review_open_positions).parameters
            if any(
                parameter.kind == Parameter.VAR_KEYWORD
                or name == "round_deadline_monotonic"
                for name, parameter in parameters.items()
            ):
                review_kwargs["round_deadline_monotonic"] = round_deadline_monotonic
        except (TypeError, ValueError):
            review_kwargs["round_deadline_monotonic"] = round_deadline_monotonic

        return await self._review_open_positions(
            open_positions,
            feature_vectors,
            **review_kwargs,
        )

    async def claim_analysis_symbol(self, symbol: str, scope: str) -> bool:
        """Claim an analysis symbol through an explicit analysis-service boundary."""

        return await self._try_claim_analysis_symbol(symbol, scope)

    def normalize_position_symbol(self, symbol: str | None) -> str:
        """Normalize position symbols through an explicit boundary."""

        return self._normalize_position_symbol(symbol)

    async def execute_position_review_candidate(
        self,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        assessment: Any,
        decision_db_id: int | None,
        results: dict[str, Any],
        *,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult | None:
        """Execute a position-review candidate through an explicit boundary."""

        pending_reason = (
            "本轮还在分析或排队中：持仓复盘候选已进入执行队列，正在等待执行链路空闲并继续完成风控复核；"
            "尚未开始向 OKX 提交订单。"
        )
        if decision_db_id is not None:
            await self._mark_decision_reason(decision_db_id, pending_reason)
        try:
            execution_result = await await_entry_execution_handoff(
                self._execute_candidate(
                    symbol,
                    model_name,
                    decision,
                    assessment,
                    decision_db_id,
                    results,
                    open_positions=open_positions,
                ),
                symbol=symbol,
                model_name=model_name,
                action=decision.action.value,
                source="position_review_candidate",
            )
        except asyncio.CancelledError:
            reason = (
                "持仓复盘候选已经进入执行链路，但本轮任务被外层超时保护取消；"
                "系统已等待下单链路尽量收口，仍未拿到最终结果，本次旧候选不再继续等待。"
            )
            if decision_db_id is not None:
                await self._record_and_persist_decision_stage(
                    decision_db_id,
                    decision,
                    DecisionStage.EXCHANGE_SUBMIT,
                    DecisionStageStatus.FAILED,
                    reason,
                    {
                        "skip_kind": "position_review_execution_cancelled",
                        "source": "position_review_candidate",
                    },
                )
                await self._mark_decision_reason(decision_db_id, reason)
            raise
        except Exception as exc:
            reason = (
                "持仓复盘候选进入执行链路后异常中断："
                f"{safe_error_text(exc, limit=160)}。系统已跳过本次旧候选，下一轮会重新复盘。"
            )
            if decision_db_id is not None:
                await self._record_and_persist_decision_stage(
                    decision_db_id,
                    decision,
                    DecisionStage.EXCHANGE_SUBMIT,
                    DecisionStageStatus.FAILED,
                    reason,
                    {
                        "skip_kind": "position_review_execution_error",
                        "source": "position_review_candidate",
                    },
                )
                await self._mark_decision_reason(decision_db_id, reason)
            raise

        if decision_db_id is not None:
            await self.decision_final_state_ensurer.ensure(
                decision_db_id,
                symbol,
                model_name,
                decision,
                results,
            )
        return execution_result

    def record_round_error(self, reason: str) -> None:
        """Record the latest loop error through an explicit service boundary."""

        error_text = str(reason)[:300]
        scope = _analysis_scope_context.get()
        self._runtime_state(scope).last_error = error_text
        if scope == "full":
            self._last_round_error = error_text
        self._write_runtime_heartbeat()

    def increment_decision_count(self) -> None:
        """Increment the decision counter through an explicit service boundary."""

        self._decision_count += 1

    def _manual_trade_execution_processor(self) -> ManualTradeExecutionProcessor:
        """Return the manual execution processor, creating it for lightweight test objects."""

        processor = getattr(self, "manual_trade_execution_processor", None)
        if processor is None:
            processor = ManualTradeExecutionProcessor(
                decision_logger=self._log_decision,
                decision_count_incrementer=self.increment_decision_count,
                candidate_executor=self._execute_candidate,
                is_paper_provider=lambda: mode_manager.is_paper,
            )
            self.manual_trade_execution_processor = processor
        return processor

    def _fast_risk_exit_execution_processor(self) -> FastRiskExitExecutionProcessor:
        """Return the fast-risk execution processor, creating it for lightweight tests."""

        processor = getattr(self, "fast_risk_exit_execution_processor", None)
        if processor is None:
            processor = FastRiskExitExecutionProcessor(
                model_execution_mode_provider=self._get_model_execution_mode,
                decision_logger=self._log_decision,
                decision_count_incrementer=self.increment_decision_count,
                candidate_executor=self._execute_candidate,
                exchange_confirmed_checker=self._is_exchange_confirmed_execution,
                exit_progress_checker=self._is_exit_progress_execution,
                profit_exit_recorder=self.position_profit_peaks.remember_profit_exit,
                risk_event_logger=self._log_risk_event,
                rejected_execution_factory=self._rejected_execution_result,
                trade_logger=self._log_trade,
                decision_reason_marker=self._mark_decision_reason,
                execution_reason_provider=self._execution_reason_from_result,
            )
            self.fast_risk_exit_execution_processor = processor
        return processor

    async def paper_positions_for_context(self) -> list[dict[str, Any]]:
        """Return no in-memory paper positions in Phase 3 OKX-backed mode.

        OKX native positions are the only trading-context truth.  This method
        stays as a compatibility boundary for older tests/callers, but it must
        not read PaperExecutor memory or local synthetic positions.
        """

        return []

    def active_okx_for_current_mode(self) -> OKXExecutor | None:
        """Return the OKX executor for the currently selected trading mode."""

        return self._okx_live if mode_manager.mode.value == "live" else self._okx_paper

    def active_okx_for_mode(self, mode: str) -> OKXExecutor | None:
        """Return an already-initialized OKX executor for a specific mode."""

        selected_mode = "live" if mode == "live" else "paper"
        return self._okx_live if selected_mode == "live" else self._okx_paper

    def paper_okx_for_reconciliation(self) -> OKXExecutor | None:
        """Return the OKX demo executor used for paper-position reconciliation."""

        return self._okx_paper

    def okx_executor_for_dashboard(self, mode: str) -> OKXExecutor | None:
        """Return an existing OKX executor for dashboard reads without initializing one."""

        selected_mode = "live" if mode == "live" else "paper"
        return self._okx_live if selected_mode == "live" else self._okx_paper

    def record_position_profit_peak(self, **kwargs: Any) -> dict[str, Any]:
        """Update the in-memory profit peak through an explicit boundary."""

        return self.position_profit_peaks.update(**kwargs)

    def prune_position_profit_peaks(self, open_positions: list[dict[str, Any]]) -> None:
        """Prune stale in-memory profit peaks through an explicit boundary."""

        self.position_profit_peaks.prune(open_positions)

    async def fresh_feature_vector_for_price_recheck(self, symbol: str) -> Any | None:
        """Fetch a fresh feature vector through an explicit sync-service boundary."""

        return await self._fresh_feature_vector_for_price_recheck(symbol)

    async def log_exchange_sync_close_decision(self, **kwargs: Any) -> int | None:
        """Record exchange-sync close decisions through an explicit boundary."""

        return await self._log_exchange_sync_close_decision(**kwargs)

    async def record_trade_reflection_in_session(
        self,
        session: Any,
        pos: Any,
        **kwargs: Any,
    ) -> None:
        """Record trade reflection through an explicit sync-service boundary."""

        await self.expert_memory_service.record_trade_reflection_in_session(session, pos, **kwargs)

    async def get_okx_balance_snapshot_for_mode(
        self,
        mode: str,
    ) -> dict[str, Any] | None:
        """Return OKX balance snapshots through a public dashboard boundary."""

        return await self._get_okx_balance_snapshot_for_mode(mode)

    def peek_okx_balance_snapshot_for_mode(
        self,
        mode: str,
        *,
        allow_stale: bool = True,
    ) -> dict[str, Any] | None:
        """Return the last cached OKX balance snapshot without making a network call."""

        selected_mode = "live" if mode == "live" else "paper"
        fresh_snapshot = self._cached_okx_balance_snapshot(
            selected_mode,
            max_age_seconds=OKX_BALANCE_SNAPSHOT_FRESH_SECONDS,
        )
        if fresh_snapshot or not allow_stale:
            return fresh_snapshot
        return self._cached_okx_balance_snapshot(
            selected_mode,
            max_age_seconds=OKX_BALANCE_SNAPSHOT_STALE_SECONDS,
            stale_reason="cached OKX balance snapshot",
        )

    async def completed_shadow_backtest_total(self) -> int:
        """Return completed shadow-backtest count through a public dashboard boundary."""

        return await self._completed_shadow_backtest_total()

    def reset_decision_runtime_state(self) -> None:
        """Reset in-memory decision counters after dashboard record deletion."""

        self._decision_count = 0
        self._recent_decisions = []

    def get_model_execution_mode(self, model_name: str) -> str:
        """Return model execution mode through an explicit execution-service boundary."""

        return self._get_model_execution_mode(model_name)

    async def log_risk_event(
        self,
        event_type: str,
        symbol: str,
        details: str,
        model_name: str,
        severity: str = "warn",
    ) -> None:
        """Persist risk events through an explicit execution-service boundary."""

        await self._log_risk_event(event_type, symbol, details, model_name, severity)

    async def record_and_persist_decision_stage(
        self,
        decision_id: int | None,
        decision: DecisionOutput,
        stage: str,
        status: str,
        reason: str | None,
        data: dict[str, Any] | None = None,
        *,
        duration_sec: float | None = None,
    ) -> dict[str, Any]:
        """Record decision-stage telemetry through an explicit boundary."""

        try:
            return await self._record_and_persist_decision_stage(
                decision_id,
                decision,
                stage,
                status,
                reason,
                data,
                duration_sec=duration_sec,
            )
        except TypeError as exc:
            if "duration_sec" not in str(exc):
                raise
            return await self._record_and_persist_decision_stage(
                decision_id,
                decision,
                stage,
                status,
                reason,
                data,
            )

    async def mark_decision_reason(self, decision_id: int, reason: str | None) -> None:
        """Persist execution reason through an explicit execution-service boundary."""

        await self._mark_decision_reason(decision_id, reason)

    async def mark_decision_raw_response(
        self,
        decision_id: int,
        raw_response: dict[str, Any] | None,
    ) -> None:
        """Persist decision raw response through an explicit boundary."""

        await self._mark_decision_raw_response(decision_id, raw_response)

    def position_review_alert_context(
        self,
        decision: DecisionOutput,
    ) -> dict[str, Any] | None:
        """Expose position-review alert context through an explicit boundary."""

        return self.position_review_risk_alert_policy.alert_context(decision)

    async def log_position_review_risk_result(
        self,
        decision: DecisionOutput,
        model_name: str,
        result_text: str | None = None,
        execution_result: ExecutionResult | None = None,
    ) -> None:
        """Persist position-review risk outcomes through an explicit boundary."""

        alert = self.position_review_alert_context(decision)
        if not alert:
            return

        if execution_result is not None:
            result_text = self.position_review_risk_alert_policy.execution_result_text(
                decision,
                execution_result,
                self._execution_reason_from_result,
            )

        await self._log_risk_event(
            "position_review_warning",
            decision.symbol,
            self.position_review_risk_alert_policy.risk_event_detail(
                decision,
                alert,
                result_text,
            ),
            model_name,
            severity="critical" if decision.is_exit else "warn",
        )

    async def duplicate_decision_order_reason(
        self,
        decision_id: int,
        decision: DecisionOutput,
    ) -> str | None:
        """Check duplicate decision orders through an explicit execution boundary."""

        return await self._duplicate_decision_order_reason(decision_id, decision)

    async def get_okx_executor_for_mode(self, mode: str) -> OKXExecutor:
        """Return OKX executor through an explicit execution-service boundary."""

        return await self._get_okx_executor_for_mode(mode)

    async def allocated_order_balance(
        self,
        model_mode: str,
        decision: DecisionOutput | None = None,
    ) -> float | None:
        """Return allocated order balance through an explicit execution boundary."""

        selected_mode = "live" if model_mode == "live" else "paper"
        if _analysis_scope_context.get() == "market":
            cached_snapshot = self.peek_okx_balance_snapshot_for_mode(selected_mode)
            if cached_snapshot:
                return max(tradeable_balance_from_snapshot(cached_snapshot), 0.0)
            self._schedule_okx_balance_snapshot_refresh_for_new_pair_pause(selected_mode)
            logger.warning(
                "market entry sizing skipped fresh OKX balance pull because cache is cold",
                mode=selected_mode,
                symbol=getattr(decision, "symbol", None),
                policy="background_refresh_and_no_order_until_balance_cache",
            )
            return 0.0
        return await self.account_accounting_service.allocated_order_balance(
            model_mode,
            decision,
        )

    async def entry_exchange_risk_facts(
        self,
        model_mode: str,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Load the OKX-native facts consumed by authoritative entry sizing."""

        executor = await self._get_okx_executor_for_mode(model_mode)
        return await executor.entry_risk_facts(decision.symbol, open_positions)

    async def pre_order_execution_facts(
        self,
        model_mode: str,
        decision: DecisionOutput,
    ) -> dict[str, Any]:
        """Load immediate native market and account cost facts before submit."""

        executor = await self._get_okx_executor_for_mode(model_mode)
        side = "long" if decision.action == Action.LONG else "short"
        return await executor.pre_order_execution_facts(decision.symbol, side)

    def rejected_execution_result(self, decision: DecisionOutput, reason: str) -> ExecutionResult:
        """Build rejected execution results through an explicit boundary."""

        return self._rejected_execution_result(decision, reason)

    def attach_execution_leverage_summary(
        self,
        decision: DecisionOutput,
        execution_result: ExecutionResult,
        ai_requested_leverage: float,
    ) -> None:
        """Attach leverage telemetry through an explicit execution boundary."""

        self._attach_execution_leverage_summary(
            decision,
            execution_result,
            ai_requested_leverage,
        )

    def execution_reason_from_result(self, execution_result: ExecutionResult | None) -> str:
        """Return execution reason text through an explicit execution boundary."""

        return self._execution_reason_from_result(execution_result)

    async def mark_decision_pending_execution(self, decision_id: int, reason: str) -> None:
        """Mark pending execution through an explicit execution-service boundary."""

        await self._mark_decision_pending_execution(decision_id, reason)

    async def evaluate_entry_execution_policy(
        self,
        decision: DecisionOutput,
        model_name: str,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None,
    ) -> PolicyGateResult:
        """Apply current exchange/account safety before the return policy."""

        if decision.is_entry:
            okx_sync_reason = self._okx_authoritative_sync_entry_block_reason()
            if okx_sync_reason:
                return PolicyGateResult.block(
                    "okx_authoritative_sync_unhealthy",
                    okx_sync_reason,
                    {
                        "stage_status": "blocked",
                        "okx_authoritative_sync": self._okx_authoritative_sync_status_payload(),
                        "execution_blocker": "okx_authoritative_sync_unhealthy",
                    },
                )
        return await self.entry_execution_pipeline.evaluate(
            decision,
            model_name,
            model_mode,
            open_positions,
        )

    async def log_trade(
        self,
        execution_result: ExecutionResult,
        model_name: str,
        decision: DecisionOutput,
        decision_id: int | None = None,
    ) -> None:
        """Persist trade logs through an explicit execution-service boundary."""

        await self._log_trade(execution_result, model_name, decision, decision_id)

    def is_exchange_confirmed_execution(self, execution_result: ExecutionResult | None) -> bool:
        """Return exchange confirmation state through an explicit boundary."""

        return self._is_exchange_confirmed_execution(execution_result)

    def is_exit_progress_execution(self, execution_result: ExecutionResult | None) -> bool:
        """Return exit-progress state through an explicit execution boundary."""

        return self._is_exit_progress_execution(execution_result)

    def result_has_no_exchange_position(self, execution_result: ExecutionResult) -> bool:
        """Detect no-position exchange results through an explicit boundary."""

        return self._result_has_no_exchange_position(execution_result)

    def increment_trade_count(self) -> None:
        """Increment runtime trade count through an explicit execution boundary."""

        self._trade_count += 1

    async def persist_position_from_execution(
        self,
        model_name: str,
        decision: DecisionOutput,
        execution_result: ExecutionResult,
        model_mode: str,
    ) -> None:
        """Persist position changes through an explicit execution boundary."""

        await self._persist_position_from_execution(
            model_name,
            decision,
            execution_result,
            model_mode,
        )

    def apply_execution_to_open_positions(
        self,
        open_positions: list[dict[str, Any]],
        model_name: str,
        decision: DecisionOutput,
        execution_result: ExecutionResult,
    ) -> None:
        """Apply execution to in-memory open positions through an explicit boundary."""

        self._apply_execution_to_open_positions(
            open_positions,
            model_name,
            decision,
            execution_result,
        )

    async def mark_decision_executed(self, decision_id: int, price: float) -> None:
        """Mark a decision as executed through an explicit execution boundary."""

        await self._mark_decision_executed(decision_id, price)

    async def persist_account_update(
        self,
        model_name: str,
        decision_model_name: str,
        execution_result: ExecutionResult,
    ) -> None:
        """Persist account update through an explicit execution boundary."""

        try:
            await self.account_accounting_service.persist_account_update(
                model_name,
                decision_model_name,
                execution_result,
            )
        finally:
            self._invalidate_okx_balance_snapshot_cache_for_model(
                decision_model_name or model_name
            )

    async def get_account_balance(self, model_name: str) -> float:
        """Return account balance through an explicit execution boundary."""

        return await self.account_accounting_service.account_balance(model_name)

    async def execution_allocation_state(self, mode: str) -> dict[str, Any]:
        """Return execution allocation and PnL state through an explicit boundary."""

        return await self.execution_allocation_service.calculate(mode)

    async def mark_decision_outcome(
        self,
        decision_id: int,
        outcome: str,
        pnl_pct: float,
    ) -> None:
        """Mark decision outcome through an explicit execution boundary."""

        await self._mark_decision_outcome(decision_id, outcome, pnl_pct)

    def execution_agent_skills(
        self,
        *,
        decision: DecisionOutput,
        model_mode: str,
        override_balance: float | None,
    ) -> list[Any]:
        """Return execution Agent/Skills through an explicit boundary."""

        return self.agent_skills.execution_skills(
            decision=decision,
            model_mode=model_mode,
            override_balance=override_balance,
        )

    def attach_execution_agent_skills(
        self,
        decision: DecisionOutput,
        *,
        phase: str,
        skills: list[Any],
        note: str,
    ) -> None:
        """Attach execution Agent/Skills through an explicit boundary."""

        self.agent_skills.attach(
            decision,
            phase=phase,
            skills=skills,
            note=note,
        )

    def execution_agent_skill_block_reason(
        self,
        skills: list[Any],
        *,
        for_entry: bool,
    ) -> str | None:
        """Return execution Agent/Skills block reason through an explicit boundary."""

        return self.agent_skills.block_reason(skills, for_entry=for_entry)

    async def reconcile_positions_for_execution(self, reason: str) -> None:
        """Reconcile positions through an explicit execution-service boundary."""

        await self.okx_sync_service.reconcile_positions(reason)

    async def open_positions_context_for_execution(self) -> list[dict[str, Any]]:
        """Return open-position context through an explicit execution boundary."""

        await self.okx_sync_service.reconcile_positions(
            "execution open positions context refresh",
            timeout_seconds=self.round_start_reconcile_timeout_seconds(),
            lock_wait_seconds=0.1,
            record_timeout_error=False,
        )
        return await self.okx_sync_service.get_open_positions_context()

    def has_matching_local_exit_position(
        self,
        positions: list[dict[str, Any]],
        model_name: str,
        decision: DecisionOutput,
    ) -> bool:
        """Check local exit-position availability through an explicit boundary."""

        return self.exit_policy.has_matching_position(positions, model_name, decision)

    async def has_matching_exchange_exit_position_for_execution(
        self,
        model_name: str,
        decision: DecisionOutput,
    ) -> bool | None:
        """Check exchange exit-position availability through an explicit boundary."""

        return await self.okx_sync_service.has_matching_exchange_exit_position(
            model_name,
            decision,
        )

    def record_executed_trade_notional(self, amount: float) -> None:
        """Record executed notional in the risk circuit breaker boundary."""

        self.risk_engine.circuit_breaker.record_trade(amount)

    async def _local_ai_tools_context(
        self,
        fv: Any,
        ml_signal_context: dict[str, Any] | None = None,
        open_positions: list[dict[str, Any]] | None = None,
        include_exit_advice: bool = False,
    ) -> dict[str, Any]:
        try:
            return await self.local_ai_tools.enrich_with_context(
                fv,
                ml_signal=ml_signal_context,
                open_positions=open_positions,
                include_exit_advice=include_exit_advice,
            )
        except Exception as exc:
            error_text = safe_error_text(exc, limit=180)
            logger.debug(
                "local AI tools enrichment failed",
                symbol=getattr(fv, "symbol", None),
                error=error_text,
            )
            return {
                "enabled": bool(settings.local_ai_tools_enabled),
                "status": "error",
                "error": error_text,
            }

    def _entry_direction_competition_policy(self) -> EntryDirectionCompetitionPolicy:
        policy = getattr(self, "entry_direction_competition", None)
        if policy is not None:
            return policy
        return EntryDirectionCompetitionPolicy()

    def _direction_competition_context(
        self,
        fv: Any,
        ml_signal_context: dict[str, Any] | None,
        local_ai_tools_context: dict[str, Any] | None,
        market_regime: dict[str, Any] | None,
        strategy_mode: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build a long-vs-short evidence summary for AI and entry ranking."""
        return self._entry_direction_competition_policy().context(
            fv,
            ml_signal_context,
            local_ai_tools_context,
            market_regime,
            strategy_mode,
        )

    def _entry_candidate_evidence_policy(self) -> EntryCandidateEvidencePolicy:
        policy = getattr(self, "entry_candidate_evidence", None)
        if policy is not None:
            return policy
        return EntryCandidateEvidencePolicy(
            model_name=ENSEMBLE_TRADER_NAME,
            score_candidate=self.entry_policy.score_candidate,
            feature_opportunity_score=self._feature_opportunity_score,
        )

    def _entry_candidate_queue_policy(self) -> EntryCandidateQueuePolicy:
        policy = getattr(self, "entry_candidate_queue", None)
        if policy is not None:
            return policy
        return EntryCandidateQueuePolicy(
            score_candidate=self.entry_policy.score_candidate,
            wait_sort_reason=self.entry_policy.wait_sort_reason,
        )

    def _entry_candidate_filter_policy(self) -> EntryCandidateFilterPolicy:
        policy = getattr(self, "entry_candidate_filter", None)
        if policy is not None:
            return policy
        return EntryCandidateFilterPolicy(
            gate_reason=self.entry_policy.gate_reason,
            market_regime_reason=self.entry_market_regime.reason,
            capacity_reason=self.entry_capacity.reason,
            reserve_capacity=self.entry_capacity.reserve_slot,
        )

    def _ai_entry_candidate_evidence(
        self,
        fv: Any,
        strategy: dict[str, Any] | None,
        ml_signal_context: dict[str, Any] | None,
        local_ai_tools_context: dict[str, Any] | None,
        direction_competition_context: dict[str, Any] | None,
        memory_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a pre-AI evidence pack for long/short candidate quality."""
        return self._entry_candidate_evidence_policy().build(
            fv,
            strategy,
            ml_signal_context,
            local_ai_tools_context,
            direction_competition_context,
            memory_feedback,
        )

    async def _memory_context_with_vector_feedback(
        self,
        symbol: str,
        *,
        action: str = "",
    ) -> dict[str, Any]:
        """Return expert memory plus optional zvec/json vector-memory soft evidence."""

        context = await self.expert_memory_service.context(symbol)
        if not bool(settings.vector_memory_enabled):
            return context
        try:
            vector_result = await get_vector_memory_service().search(
                f"{symbol} {action} 开仓 亏损 盈利 复盘 三期相似样本",
                top_k=6,
                symbol=symbol,
            )
        except Exception as exc:
            vector_result = {
                "enabled": True,
                "status": "error",
                "error": safe_error_text(exc, limit=180),
                "hits": [],
            }
        hits = vector_result.get("hits") if isinstance(vector_result, dict) else []
        memory_feedback = (
            dict(context.get("memory_feedback"))
            if isinstance(context.get("memory_feedback"), dict)
            else {}
        )
        memory_feedback["vector_memory"] = {
            "enabled": (
                bool(vector_result.get("enabled")) if isinstance(vector_result, dict) else True
            ),
            "status": (
                str(vector_result.get("status") or "unknown")
                if isinstance(vector_result, dict)
                else "error"
            ),
            "matched_count": len(hits) if isinstance(hits, list) else 0,
            "hits": hits[:3] if isinstance(hits, list) else [],
            "is_hard_gate": False,
            "policy": "三期相似样本只作为软证据调节和解释，不作为硬拦截。",
        }
        context["memory_feedback"] = memory_feedback
        context["vector_memory_feedback"] = memory_feedback["vector_memory"]
        return context

    def _is_valid_feature_vector(self, fv: Any) -> bool:
        """Only send market snapshots with usable price data to AI models."""
        try:
            price = float(getattr(fv, "current_price", 0) or 0)
            close = float(getattr(fv, "close", 0) or 0)
            bid = float(getattr(fv, "bid", 0) or 0)
            ask = float(getattr(fv, "ask", 0) or 0)
        except (TypeError, ValueError):
            return False
        return max(price, close, bid, ask) > 0

    @staticmethod
    def _callable_accepts_keyword(callback: Any, keyword: str) -> bool:
        try:
            parameters = inspect.signature(callback).parameters
        except (TypeError, ValueError):
            return True
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD or name == keyword
            for name, param in parameters.items()
        )

    async def _get_feature_vector_snapshot(
        self,
        symbol: str,
        *,
        wait_for_sentiment: bool = True,
        block_on_remote_ticker: bool = True,
        block_on_remote_indicators: bool = True,
        block_on_remote_derivatives: bool = True,
        allow_cached_indicator_build: bool = True,
        allow_indicator_background_refresh: bool = True,
    ) -> Any:
        """Read a feature vector while preserving compatibility with older test doubles."""

        getter = self.data_service.get_feature_vector
        try:
            parameters = inspect.signature(getter).parameters
            accepts_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
            )
            accepts_options = "wait_for_sentiment" in parameters or accepts_var_kwargs
            accepts_ticker_option = "block_on_remote_ticker" in parameters or accepts_var_kwargs
            accepts_indicator_option = (
                "block_on_remote_indicators" in parameters or accepts_var_kwargs
            )
            accepts_derivatives_option = (
                "block_on_remote_derivatives" in parameters or accepts_var_kwargs
            )
            accepts_cached_indicator_build_option = (
                "allow_cached_indicator_build" in parameters or accepts_var_kwargs
            )
            accepts_indicator_background_refresh_option = (
                "allow_indicator_background_refresh" in parameters or accepts_var_kwargs
            )
        except (TypeError, ValueError):
            accepts_options = True
            accepts_ticker_option = True
            accepts_indicator_option = True
            accepts_derivatives_option = True
            accepts_cached_indicator_build_option = True
            accepts_indicator_background_refresh_option = True

        kwargs: dict[str, Any] = {}
        if accepts_options:
            kwargs["wait_for_sentiment"] = wait_for_sentiment
        if accepts_ticker_option:
            kwargs["block_on_remote_ticker"] = block_on_remote_ticker
        if accepts_indicator_option:
            kwargs["block_on_remote_indicators"] = block_on_remote_indicators
        if accepts_derivatives_option:
            kwargs["block_on_remote_derivatives"] = block_on_remote_derivatives
        if accepts_cached_indicator_build_option:
            kwargs["allow_cached_indicator_build"] = allow_cached_indicator_build
        if accepts_indicator_background_refresh_option:
            kwargs["allow_indicator_background_refresh"] = allow_indicator_background_refresh
        if kwargs:
            return await getter(symbol, **kwargs)
        return await getter(symbol)

    def _runtime_state(self, scope: str | None = None) -> _AnalysisRuntimeState:
        resolved_scope = scope or _analysis_scope_context.get() or "full"
        if resolved_scope not in self._analysis_runtime:
            resolved_scope = "full"
        return self._analysis_runtime[resolved_scope]

    @staticmethod
    def _stage_duration_event(
        *,
        stage: str,
        started_at: datetime | None,
        finished_at: datetime,
    ) -> dict[str, Any] | None:
        if not stage or started_at is None:
            return None
        duration = max((finished_at - started_at).total_seconds(), 0.0)
        return {
            "stage": stage,
            "duration_seconds": round(duration, 3),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
        }

    @staticmethod
    def _recent_stage_durations(state: _AnalysisRuntimeState) -> list[dict[str, Any]]:
        rows = getattr(state, "recent_stage_durations", None)
        return list(rows[-24:]) if isinstance(rows, list) else []

    def _stage_durations_for_scope(self, scope: str) -> list[dict[str, Any]]:
        runtime = getattr(self, "_analysis_runtime", None)
        if not isinstance(runtime, dict):
            return []
        state = runtime.get(scope)
        if not isinstance(state, _AnalysisRuntimeState):
            return []
        return self._recent_stage_durations(state)

    def _start_runtime_round(self, scope: str, started_at: datetime) -> None:
        state = self._runtime_state(scope)
        state.current_stage = "starting"
        state.current_stage_started_at = started_at
        state.recent_stage_durations.clear()
        state.last_started_at = started_at
        state.last_finished_at = None
        state.last_error = None
        self._last_round_started_at = started_at
        self._last_round_finished_at = None
        self._last_round_error = None
        if scope == "market":
            self._last_market_round_started_at = started_at
            self._last_market_round_finished_at = None
        elif scope == "position":
            self._last_position_round_started_at = started_at
            self._last_position_round_finished_at = None

    def _finish_runtime_round(self, scope: str, finished_at: datetime, *, ok: bool) -> None:
        state = self._runtime_state(scope)
        event = self._stage_duration_event(
            stage=state.current_stage,
            started_at=state.current_stage_started_at,
            finished_at=finished_at,
        )
        if event is not None:
            state.recent_stage_durations.append(event)
            del state.recent_stage_durations[:-24]
        state.last_finished_at = finished_at
        state.current_stage = "idle" if ok else "error"
        state.current_stage_started_at = finished_at
        if ok:
            state.last_error = None
        if scope == "market":
            self._last_market_round_finished_at = finished_at
        elif scope == "position":
            self._last_position_round_finished_at = finished_at
        active_states = [
            item
            for item_scope, item in self._analysis_runtime.items()
            if item_scope in {"market", "position"} and item.active
        ]
        if active_states:
            newest = max(
                active_states,
                key=lambda item: item.last_started_at or datetime.min.replace(tzinfo=UTC),
            )
            self._last_round_started_at = newest.last_started_at
            self._last_round_finished_at = None
            self._current_stage = newest.current_stage
        else:
            self._last_round_finished_at = finished_at
            self._current_stage = "idle" if ok else "error"
        if ok and not any(
            item.last_error
            for item_scope, item in self._analysis_runtime.items()
            if item_scope in {"market", "position", "full"}
        ):
            self._last_round_error = None

    def _set_loop_stage(
        self,
        stage: str,
        error: str | None = None,
        *,
        scope: str | None = None,
        heartbeat: bool = True,
    ) -> None:
        resolved_scope = scope or _analysis_scope_context.get() or "full"
        state = self._runtime_state(resolved_scope)
        now = datetime.now(UTC)
        if state.current_stage != stage:
            event = self._stage_duration_event(
                stage=state.current_stage,
                started_at=state.current_stage_started_at,
                finished_at=now,
            )
            if event is not None:
                state.recent_stage_durations.append(event)
                del state.recent_stage_durations[:-24]
        state.current_stage = stage
        state.current_stage_started_at = now
        if error is not None:
            state.last_error = str(error)[:300]
            if resolved_scope == "full":
                self._last_round_error = state.last_error
        self._current_stage = stage
        if heartbeat:
            self._write_runtime_heartbeat()

    def _write_runtime_heartbeat(self) -> None:
        """Persist a lightweight status heartbeat for split dashboard deployments."""
        try:
            settings.refresh_runtime_env(force=True)
            now = datetime.now(UTC)
            uptime = (
                int((now - self._start_time).total_seconds()) if self._start_time is not None else 0
            )
            market_state = self._runtime_state("market")
            position_state = self._runtime_state("position")
            active_scoped_states = [
                state for state in (market_state, position_state) if state.active
            ]
            round_active = bool(active_scoped_states) or (
                self._last_round_started_at is not None
                and (
                    self._last_round_finished_at is None
                    or self._last_round_finished_at < self._last_round_started_at
                )
            )
            current_state = (
                max(
                    active_scoped_states,
                    key=lambda state: state.last_started_at or datetime.min.replace(tzinfo=UTC),
                )
                if active_scoped_states
                else self._runtime_state(_analysis_scope_context.get())
            )
            last_round_error = (
                market_state.last_error or position_state.last_error or self._last_round_error
            )
            okx_authoritative_sync = self._okx_authoritative_sync_status_payload(now)
            payload = {
                "running": bool(self._running),
                "mode": mode_manager.mode.value,
                "paused": mode_manager.is_paused,
                "scan_mode": mode_manager.scan_mode,
                "started_at": self._start_time.isoformat() if self._start_time else None,
                "heartbeat_at": now.isoformat(),
                "last_heartbeat_at": now.isoformat(),
                "uptime_seconds": uptime,
                "decision_interval": settings.decision_interval_seconds,
                "market_loop_interval_seconds": round(self.market_loop_interval_seconds(), 3),
                "position_loop_interval_seconds": round(self.position_loop_interval_seconds(), 3),
                "market_round_time_budget_seconds": round(
                    self.market_round_time_budget_seconds(), 3
                ),
                "market_configured_symbol_limit": settings.auto_scan_symbol_limit,
                "market_configured_symbol_limit_is_batch_size": False,
                "market_batch_policy": (
                    "position_first_parallel_loops; actual market batch is dynamic"
                ),
                "market_analysis_watchdog_seconds": int(self.market_round_watchdog_seconds()),
                "position_analysis_watchdog_seconds": int(self.position_round_watchdog_seconds()),
                "current_stage": current_state.current_stage,
                "round_active": round_active,
                "market_current_stage": market_state.current_stage,
                "market_round_active": market_state.active,
                "market_last_error": market_state.last_error,
                "position_current_stage": position_state.current_stage,
                "position_round_active": position_state.active,
                "position_last_error": position_state.last_error,
                "market_stage_durations": self._recent_stage_durations(market_state),
                "position_stage_durations": self._recent_stage_durations(position_state),
                "last_position_price_refresh_diagnostics": self._safe_dict(
                    getattr(self, "_last_position_price_refresh_diagnostics", {})
                ),
                "last_round_started_at": (
                    self._last_round_started_at.isoformat() if self._last_round_started_at else None
                ),
                "last_round_finished_at": (
                    self._last_round_finished_at.isoformat()
                    if self._last_round_finished_at
                    else None
                ),
                "last_market_round_started_at": (
                    market_state.last_started_at.isoformat()
                    if market_state.last_started_at
                    else None
                ),
                "last_market_round_finished_at": (
                    market_state.last_finished_at.isoformat()
                    if market_state.last_finished_at
                    else None
                ),
                "last_position_round_started_at": (
                    position_state.last_started_at.isoformat()
                    if position_state.last_started_at
                    else None
                ),
                "last_position_round_finished_at": (
                    position_state.last_finished_at.isoformat()
                    if position_state.last_finished_at
                    else None
                ),
                "last_round_error": last_round_error,
                "okx_authoritative_sync": okx_authoritative_sync,
                "shadow_backtest_maintenance": self._shadow_backtest_maintenance_status(),
                "stale_entry_maintenance": self._stale_entry_candidate_maintenance_status(),
            }
            path = settings.data_dir / "trading_runtime_status.json"
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(path)
        except Exception as exc:
            logger.debug("failed to write trading runtime heartbeat", error=safe_error_text(exc))

    async def _reconcile_exchange_positions_with_timeout(
        self,
        context: str,
        timeout_seconds: float = 25.0,
    ) -> list[dict]:
        try:

            async def _locked_reconcile() -> list[dict]:
                async with self._exchange_reconcile_lock:
                    return await self.reconcile_exchange_positions()

            return await asyncio.wait_for(_locked_reconcile(), timeout=timeout_seconds)
        except TimeoutError:
            reason = (
                f"exchange position reconciliation timed out during {context}; "
                "continuing with local position state"
            )
            self.record_round_error(reason)
            logger.warning(reason)
            return []

    async def _fresh_feature_vector_for_analysis(self, symbol: str, fallback: Any = None) -> Any:
        """Refresh the exact symbol right before AI analysis.

        The initial batch fetch is useful for ranking, but queued symbols can
        become stale while earlier symbols are analyzed. This keeps executable
        signals tied to a recent market snapshot.
        """
        try:
            fresh = await asyncio.wait_for(
                self._get_feature_vector_snapshot(
                    symbol,
                    wait_for_sentiment=False,
                    block_on_remote_indicators=False,
                    block_on_remote_derivatives=True,
                ),
                timeout=8.0,
            )
            if self._is_valid_feature_vector(fresh):
                return fresh
            logger.warning("fresh feature vector invalid; using queued snapshot", symbol=symbol)
        except TimeoutError:
            logger.warning(
                "fresh feature vector refresh timed out; using queued snapshot", symbol=symbol
            )
        except Exception as e:
            logger.warning(
                "fresh feature vector refresh failed; using queued snapshot",
                symbol=symbol,
                error=safe_error_text(e),
            )
        return fallback

    async def _fresh_feature_vector_for_price_recheck(self, symbol: str) -> Any | None:
        try:
            fresh = await asyncio.wait_for(
                self._get_feature_vector_snapshot(
                    symbol,
                    wait_for_sentiment=False,
                    block_on_remote_indicators=False,
                    block_on_remote_derivatives=False,
                ),
                timeout=ENTRY_PRICE_RECHECK_TIMEOUT_SECONDS,
            )
            if self._is_valid_feature_vector(fresh):
                return fresh
        except TimeoutError:
            logger.warning("pre-order feature recheck timed out", symbol=symbol)
        except Exception as e:
            logger.warning(
                "pre-order feature recheck failed",
                symbol=symbol,
                error=safe_error_text(e),
            )
        return None

    def _budget_auto_scan_feature_symbols(
        self,
        fetch_symbols: list[str],
        position_scan_symbols: list[str],
        *,
        configured_limit: int,
    ) -> list[str]:
        """Limit expensive feature fetches per round while rotating the full pool."""

        self._last_auto_feature_fetch_budget_diagnostics = {}
        if not fetch_symbols:
            return []
        normalized_position = {
            self._normalize_position_symbol(symbol)
            for symbol in position_scan_symbols
            if self._normalize_position_symbol(symbol)
        }
        position_symbols = [
            symbol
            for symbol in fetch_symbols
            if self._normalize_position_symbol(symbol) in normalized_position
        ]
        market_symbols = [
            symbol
            for symbol in fetch_symbols
            if self._normalize_position_symbol(symbol) not in normalized_position
        ]
        target_market_budget = max(
            configured_limit,
            configured_limit * int(AUTO_SCAN_FEATURE_FETCH_POOL_MULTIPLIER),
            int(AUTO_SCAN_FEATURE_FETCH_POOL_MIN),
        )
        max_pool = max(int(AUTO_SCAN_FEATURE_FETCH_POOL_MAX), configured_limit)
        market_budget = min(len(market_symbols), target_market_budget, max_pool)
        self._last_auto_feature_fetch_budget_diagnostics = {
            "read_only": True,
            "is_entry_gate": False,
            "total_candidates": len(fetch_symbols),
            "position_symbols": len(position_symbols),
            "market_candidates": len(market_symbols),
            "configured_market_symbol_limit": int(configured_limit),
            "target_market_feature_fetch_count": int(target_market_budget),
            "max_market_feature_fetch_count": int(max_pool),
            "selected_market_feature_fetch_count": int(market_budget),
            "pool_multiplier": int(AUTO_SCAN_FEATURE_FETCH_POOL_MULTIPLIER),
            "pool_min": int(AUTO_SCAN_FEATURE_FETCH_POOL_MIN),
            "pool_max": int(AUTO_SCAN_FEATURE_FETCH_POOL_MAX),
            "diagnostic_boundary": (
                "Feature-fetch breadth only expands discovery before rank/evidence gates; "
                "it is not entry permission, leverage, sizing, or ML readiness."
            ),
        }
        if market_budget <= 0:
            return self.entry_symbol_universe.dedupe_symbols(position_symbols)

        cursor = int(getattr(self, "_auto_scan_feature_cursor", 0) or 0)
        start = cursor % len(market_symbols)
        rotated = market_symbols[start:] + market_symbols[:start]
        selected_market = rotated[:market_budget]
        self._auto_scan_feature_cursor = (start + market_budget) % len(market_symbols)
        selected = self.entry_symbol_universe.dedupe_symbols([*position_symbols, *selected_market])
        logger.info(
            "auto scan feature fetch budget selected",
            total_candidates=len(fetch_symbols),
            position_symbols=len(position_symbols),
            market_candidates=len(market_symbols),
            selected=len(selected),
            market_budget=market_budget,
            next_cursor=self._auto_scan_feature_cursor,
        )
        self._last_auto_feature_fetch_budget_diagnostics.update(
            {
                "selected_total_feature_fetch_count": len(selected),
                "next_cursor": int(self._auto_scan_feature_cursor),
            }
        )
        return selected

    def _auto_scan_feature_fetch_early_quorum(
        self,
        *,
        completed_valid_count: int,
        total_fetch_count: int,
        configured_limit: int,
        run_market_analysis: bool,
        run_position_analysis: bool,
        auto_scan: bool,
        feature_fetch_budget_diagnostics: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        configured = max(1, int(configured_limit or 0))
        total = max(0, int(total_fetch_count or 0))
        diagnostics = self._safe_dict(feature_fetch_budget_diagnostics)
        if not diagnostics:
            diagnostics = self._safe_dict(
                getattr(self, "_last_auto_feature_fetch_budget_diagnostics", None)
            )
        selected_target = int(
            diagnostics.get("selected_market_feature_fetch_count")
            or diagnostics.get("target_market_feature_fetch_count")
            or total
            or configured
        )
        breadth_target = max(configured * 2, math.ceil(max(selected_target, configured) / 3))
        quorum = min(total, max(configured, breadth_target))
        allowed_deficit = max(1, math.ceil(configured * 0.12))
        near_quorum = max(configured, quorum - allowed_deficit)
        eligible = bool(auto_scan and run_market_analysis and not run_position_analysis and total)
        completed = int(completed_valid_count or 0)
        exact_met = bool(eligible and completed >= quorum)
        near_quorum_met = bool(eligible and not exact_met and completed >= near_quorum)
        budget_ready_met = bool(eligible and completed >= configured)
        met = bool(exact_met or near_quorum_met or budget_ready_met)
        return met, {
            "read_only": True,
            "is_entry_gate": False,
            "eligible": eligible,
            "met": met,
            "exact_met": exact_met,
            "near_quorum_met": near_quorum_met,
            "budget_ready_met": budget_ready_met,
            "completed_valid_count": completed,
            "quorum": int(quorum),
            "near_quorum": int(near_quorum),
            "budget_ready_quorum": int(configured),
            "allowed_deficit": int(allowed_deficit),
            "total_fetch_count": int(total),
            "configured_market_symbol_limit": int(configured),
            "selected_market_feature_fetch_count": int(selected_target),
            "reason": (
                "market-only auto scan may start AI analysis after enough valid feature "
                "snapshots are available for broad ranking; slow remaining feature sources "
                "are cancelled and retried by later rotation rounds"
            ),
        }

    def _feature_opportunity_score(self, fv: Any) -> float:
        """Compatibility delegate for feature-based auto-scan opportunity score."""

        ranker = getattr(self, "entry_feature_ranker", None)
        if ranker is None:
            ranker = EntryFeatureRankerPolicy(
                suspicious_symbol_reason=self._suspicious_new_symbol_reason,
                major_symbols=frozenset(ALT_LONG_ALLOWED_SYMBOLS),
            )
        return ranker.feature_opportunity_score(fv)


    def _entry_market_regime_context_policy(self) -> EntryMarketRegimeContextPolicy:
        policy = getattr(self, "entry_market_regime_context", None)
        if policy is not None:
            return policy
        return EntryMarketRegimeContextPolicy(self._is_valid_feature_vector)

    def _market_regime_context(self, feature_vectors: dict[str, Any]) -> dict[str, Any]:
        """Predict the current market style before asking for per-symbol entries."""
        return self._entry_market_regime_context_policy().context(feature_vectors)

    def _entry_strategy_mode_context_policy(self) -> EntryStrategyModeContextPolicy:
        policy = getattr(self, "entry_strategy_mode_context", None)
        if policy is not None:
            return policy
        return EntryStrategyModeContextPolicy()

    def _dynamic_capacity_context(self) -> dict[str, Any]:
        context = getattr(self, "_current_capacity_context", None)
        if isinstance(context, dict):
            return dict(context)
        return {
            "entry_limit": None,
            "hard_limit": None,
            "open_group_count": 0,
            "available_group_slots": None,
            "reason": "position count is observation-only",
            "policy_provenance": {
                "source": "position_capacity_context_unavailable",
                "observation_window": "current_open_position_snapshot",
                "sample_count": 0,
                "generated_at": datetime.now(UTC).isoformat(),
                "strategy_version": "2026-07-12.position-count-observation.v1",
                "fallback_reason": "position_snapshot_missing",
                "production_eligible": False,
                "production_permission": False,
            },
        }

    def _dynamic_capacity_policy(self) -> DynamicPositionCapacityPolicy:
        policy = getattr(self, "dynamic_capacity", None)
        if policy is not None:
            return policy
        policy = DynamicPositionCapacityPolicy()
        self.dynamic_capacity = policy
        return policy

    def _refresh_dynamic_capacity(
        self,
        *,
        open_positions: list[dict[str, Any]],
        strategy_context: dict[str, Any],
        market_regime: dict[str, Any],
        account_equity: float,
    ) -> dict[str, Any]:
        decision = (
            self._dynamic_capacity_policy()
            .evaluate(
                open_positions=open_positions or [],
                strategy_context=strategy_context,
                market_regime=market_regime,
                account_equity=account_equity,
                active_strategy_profile_id=strategy_context.get("strategy_profile_id") or None,
            )
            .as_dict()
        )
        self._current_capacity_context = decision
        strategy_context["dynamic_position_capacity"] = decision
        strategy_context["account_equity"] = account_equity
        self._current_strategy_mode_context = dict(strategy_context)
        return strategy_context

    async def _strategy_mode_context(
        self,
        mode: str,
        market_regime: dict[str, Any],
        open_positions: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Choose the trading posture automatically from PnL, regime, and side performance."""
        selected_mode = "live" if mode == "live" else "paper"
        perf_timeout = self.strategy_learning_perf_timeout_seconds()
        account_timeout = self.strategy_learning_account_timeout_seconds()
        context_fetch_timings: dict[str, Any] = {}
        analysis_scope = _analysis_scope_context.get()

        if not open_positions:
            try:
                self._safe_set_strategy_context_stage("strategy_context:open_positions")
                open_positions = await self.okx_sync_service.get_open_positions_context()
            except Exception as exc:
                logger.warning(
                    "failed to refresh open positions for strategy mode context",
                    error=safe_error_text(exc),
                )
        position_exposure = self.entry_position_exposure.context(open_positions or [])
        position_group_count = self.entry_symbol_universe.open_position_group_count(
            open_positions or []
        )
        performance_values, performance_snapshot = (
            self._recent_strategy_context_performance_snapshot(selected_mode)
        )
        performance_refresh = None
        if performance_snapshot.get("status") != "fresh":
            performance_refresh = self._start_strategy_context_performance_refresh(selected_mode)
        if performance_values is None and analysis_scope != "market":
            if performance_refresh is not None:
                done, _pending = await asyncio.wait({performance_refresh}, timeout=perf_timeout)
                if done:
                    performance_values, performance_snapshot = (
                        self._recent_strategy_context_performance_snapshot(selected_mode)
                    )
        performance_snapshot["refresh_in_flight"] = bool(
            performance_refresh is not None and not performance_refresh.done()
        )
        context_fetch_timings["performance_snapshot"] = performance_snapshot
        performance_values = performance_values or {}
        daily_state = self._safe_dict(performance_values.get("daily_perf"))
        side_perf = self._safe_dict(performance_values.get("today_side_perf"))
        side_perf_multiday = self._safe_dict(performance_values.get("multiday_side_perf"))
        symbol_side_perf = self._safe_dict(performance_values.get("symbol_side_perf"))
        model_contribution_perf = self._safe_dict(
            performance_values.get("model_contribution_perf")
        )
        account_equity = await self._bounded_strategy_context_value(
            "account_equity",
            self._strategy_context_account_equity(selected_mode),
            0.0,
            account_timeout,
            context_fetch_timings,
        )
        context = self._entry_strategy_mode_context_policy().build(
            market_regime=market_regime,
            daily_state=daily_state,
            side_performance=side_perf,
            side_performance_multiday=side_perf_multiday,
            symbol_side_performance=symbol_side_perf,
            model_contribution_performance=model_contribution_perf,
            position_exposure=position_exposure,
            position_group_count=position_group_count,
            account_equity=account_equity,
            account_config=settings.get_execution_account_config(selected_mode),
        )
        context["account_equity"] = account_equity
        context["strategy_context_performance"] = performance_snapshot
        if _analysis_scope_context.get() == "market":
            context["account_equity_source"] = getattr(
                self,
                "_last_strategy_context_account_equity_source",
                "unknown",
            )
        context["strategy_context_runtime"] = {
            "perf_timeout_seconds": perf_timeout,
            "account_timeout_seconds": account_timeout,
            "parallel_context_fetch": False,
            "performance_snapshot_source": "background_single_flight",
            "fetch_timings": context_fetch_timings,
        }
        strategy_learning = getattr(self, "strategy_learning_service", None)
        if strategy_learning is None:
            return self._refresh_dynamic_capacity(
                open_positions=open_positions or [],
                strategy_context=context,
                market_regime=market_regime,
                account_equity=account_equity,
            )
        try:
            self._safe_set_strategy_context_stage("strategy_context:learning")
            current_scope = _analysis_scope_context.get()
            if current_scope == "market":
                cached_context = self._recent_strategy_learning_context(selected_mode)
                refresh_task = self._start_strategy_learning_context_refresh(
                    mode=selected_mode,
                    analysis_scope=current_scope,
                    strategy_learning=strategy_learning,
                    context=context,
                    open_positions=open_positions or [],
                )
                if cached_context:
                    cached_context.update(
                        {
                            "strategy_learning_cache_status": "stale_background_refresh",
                            "strategy_learning_error": (
                                "市场扫描轮使用最近一次策略学习上下文，后台刷新学习结果，"
                                "不让慢学习查询阻塞开仓候选发现。"
                            ),
                            "strategy_learning_runtime_timeout_seconds": 0.0,
                        }
                    )
                    return self._refresh_dynamic_capacity(
                        open_positions=open_positions or [],
                        strategy_context=cached_context,
                        market_regime=market_regime,
                        account_equity=account_equity,
                    )
                if refresh_task.done():
                    try:
                        learned_context = refresh_task.result()
                    except Exception as exc:
                        logger.warning(
                            "market strategy learning background refresh failed before reuse",
                            error=safe_error_text(exc),
                        )
                    else:
                        refreshed_context = self._refresh_dynamic_capacity(
                            open_positions=open_positions or [],
                            strategy_context=dict(learned_context),
                            market_regime=market_regime,
                            account_equity=account_equity,
                        )
                        self._strategy_learning_context_cache_store()[selected_mode] = {
                            "created_at": datetime.now(UTC),
                            "context": self._json_safe_payload(refreshed_context),
                        }
                        return refreshed_context
                context["strategy_learning_cache_status"] = "baseline_background_refresh"
                context["strategy_learning_error"] = (
                    "市场扫描轮暂无可用策略学习缓存，已启动后台刷新，本轮使用基础策略上下文。"
                )
                context["strategy_learning_runtime_timeout_seconds"] = 0.0
                return self._refresh_dynamic_capacity(
                    open_positions=open_positions or [],
                    strategy_context=context,
                    market_regime=market_regime,
                    account_equity=account_equity,
                )
            wait_timeout = self.strategy_learning_context_wait_timeout_seconds(current_scope)
            refresh_task = self._start_strategy_learning_context_refresh(
                mode=selected_mode,
                analysis_scope=current_scope,
                strategy_learning=strategy_learning,
                context=context,
                open_positions=open_positions or [],
            )
            done, _pending = await asyncio.wait(
                {refresh_task},
                timeout=wait_timeout,
            )
            if not done:
                raise TimeoutError()
            learned_context = next(iter(done)).result()
            refreshed_context = self._refresh_dynamic_capacity(
                open_positions=open_positions or [],
                strategy_context=dict(learned_context),
                market_regime=market_regime,
                account_equity=account_equity,
            )
            self._strategy_learning_context_cache_store()[selected_mode] = {
                "created_at": datetime.now(UTC),
                "context": self._json_safe_payload(refreshed_context),
            }
            return refreshed_context
        except TimeoutError:
            cached_context = self._recent_strategy_learning_context(selected_mode)
            if cached_context:
                cached_context.update(
                    {
                        "strategy_learning_cache_status": "stale_timeout",
                        "strategy_learning_error": (
                            "策略学习上下文超过交易轮次预算，已使用最近一次可用学习上下文，"
                            "后台继续刷新，不阻塞开仓决策。"
                        ),
                        "strategy_learning_runtime_timeout_seconds": (
                            wait_timeout
                        ),
                    }
                )
                logger.warning(
                    "strategy learning context timed out; using cached strategy context",
                    timeout_seconds=round(wait_timeout, 3),
                )
                return self._refresh_dynamic_capacity(
                    open_positions=open_positions or [],
                    strategy_context=cached_context,
                    market_regime=market_regime,
                    account_equity=account_equity,
                )
            context["strategy_learning_cache_status"] = "baseline_timeout"
            context["strategy_learning_error"] = (
                "策略学习上下文超过交易轮次预算且暂无缓存，本轮使用基础策略上下文，"
                "不会因为学习慢查询阻塞市场扫描。"
            )
            context["strategy_learning_runtime_timeout_seconds"] = (
                wait_timeout
            )
            logger.warning(
                "strategy learning context timed out; using baseline strategy context",
                timeout_seconds=round(wait_timeout, 3),
            )
            return self._refresh_dynamic_capacity(
                open_positions=open_positions or [],
                strategy_context=context,
                market_regime=market_regime,
                account_equity=account_equity,
            )
        except Exception as exc:
            logger.warning(
                "strategy learning context failed; using baseline strategy context",
                error=safe_error_text(exc),
            )
            context["strategy_learning_error"] = safe_error_text(exc, limit=160)
            context["strategy_learning_cache_status"] = "baseline_error"
            return self._refresh_dynamic_capacity(
                open_positions=open_positions or [],
                strategy_context=context,
                market_regime=market_regime,
                account_equity=account_equity,
            )

    def _recent_strategy_learning_context(self, mode: str) -> dict[str, Any] | None:
        selected_mode = "live" if mode == "live" else "paper"
        entry = self._strategy_learning_context_cache_store().get(selected_mode)
        if not isinstance(entry, dict):
            return None
        created_at = entry.get("created_at")
        if not isinstance(created_at, datetime):
            return None
        max_age = float(DEFAULT_TRADING_PARAMS.strategy_learning.runtime_context_cache_ttl_seconds)
        if (datetime.now(UTC) - created_at).total_seconds() > max_age:
            return None
        context = entry.get("context")
        return dict(context) if isinstance(context, dict) else None

    def _attach_strategy_learning_context(
        self,
        decision: DecisionOutput,
        strategy_mode_context: dict[str, Any] | None,
    ) -> None:
        if not isinstance(strategy_mode_context, dict):
            return
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        learning = self._safe_dict(strategy_mode_context.get("strategy_learning"))
        runtime = self._safe_dict(learning.get("runtime"))
        raw["strategy_learning_context"] = {
            "advisory_prior_only": True,
            "production_permission": False,
            "optimization_target": "maximize_authoritative_fee_after_return_rate",
            "strategy_profile_id": strategy_mode_context.get("strategy_profile_id"),
            "strategy_profile_version": strategy_mode_context.get("strategy_profile_version"),
            "scheduler_reason": strategy_mode_context.get("scheduler_reason"),
            "market_regime": strategy_mode_context.get("market_regime"),
            "production_influence_enabled": runtime.get("production_influence_enabled")
            is True,
            "strategy_learning": learning,
        }
        decision.raw_response = raw

    async def _record_strategy_learning_event(
        self,
        *,
        mode: str,
        model_name: str = ENSEMBLE_TRADER_NAME,
        symbol: str | None = None,
        decision: DecisionOutput | None = None,
        action: str | None = None,
        event_type: str,
        event_status: str = "recorded",
        reason: str | None = None,
        severity: str = "info",
        decision_id: int | None = None,
        order_id: int | None = None,
        position_id: int | None = None,
        strategy_context: dict[str, Any] | None = None,
        market_state: dict[str, Any] | None = None,
        attribution: dict[str, Any] | None = None,
        exclude_from_training: bool = False,
    ) -> None:
        service = getattr(self, "strategy_learning_service", None)
        if service is None:
            return
        raw_response = decision.raw_response if decision is not None else None
        if strategy_context is None and isinstance(raw_response, dict):
            maybe_context = raw_response.get("strategy_learning_context")
            if isinstance(maybe_context, dict):
                strategy_context = dict(maybe_context)
        event_action = action or (decision.action.value if decision is not None else None)
        event_symbol = symbol or (decision.symbol if decision is not None else None)
        try:
            await service.record_event(
                mode=mode,
                model_name=model_name,
                symbol=event_symbol,
                action=event_action,
                event_type=event_type,
                event_status=event_status,
                reason=reason,
                severity=severity,
                decision_id=decision_id,
                order_id=order_id,
                position_id=position_id,
                strategy_context=strategy_context,
                raw_response=raw_response,
                market_state=market_state,
                attribution=attribution,
                exclude_from_training=exclude_from_training,
            )
        except Exception as exc:
            logger.debug(
                "strategy learning event recording failed",
                event_type=event_type,
                symbol=event_symbol,
                error=safe_error_text(exc),
            )

    async def _strategy_learning_execution_links(
        self,
        *,
        mode: str,
        model_name: str,
        symbol: str | None,
        decision: DecisionOutput,
        decision_id: int | None,
        result: ExecutionResult | None,
    ) -> dict[str, Any]:
        """Resolve local order/position ids for strategy-learning attribution."""
        payload: dict[str, Any] = {
            "order_id": getattr(result, "order_id", None),
            "exchange_order_id": getattr(result, "exchange_order_id", None),
            "quantity": getattr(result, "quantity", None),
            "price": getattr(result, "price", None),
        }
        if result is None:
            return payload
        normalized_symbol = self._normalize_position_symbol(symbol or result.symbol)
        if not normalized_symbol:
            return payload
        try:
            async with get_read_session_ctx() as session:
                order_stmt = select(Order).where(
                    Order.model_name == model_name,
                    Order.execution_mode == mode,
                )
                exchange_order_id = str(getattr(result, "exchange_order_id", "") or "").strip()
                if exchange_order_id:
                    order_stmt = order_stmt.where(Order.exchange_order_id == exchange_order_id)
                elif decision_id is not None:
                    order_stmt = order_stmt.where(Order.decision_id == decision_id)
                else:
                    order_stmt = order_stmt.where(Order.symbol == normalized_symbol)
                order_result = await session.execute(
                    order_stmt.order_by(Order.created_at.desc()).limit(1)
                )
                order = order_result.scalar_one_or_none()
                if order is not None:
                    payload["local_order_id"] = int(order.id)
                side = "long" if decision.action in {Action.LONG, Action.CLOSE_LONG} else "short"
                position_stmt = select(Position).where(
                    Position.model_name == model_name,
                    Position.execution_mode == mode,
                    Position.symbol == normalized_symbol,
                    Position.side == side,
                )
                if decision.action in {Action.LONG, Action.SHORT}:
                    position_stmt = position_stmt.where(Position.is_open.is_(True))
                elif decision.action in {Action.CLOSE_LONG, Action.CLOSE_SHORT}:
                    position_stmt = position_stmt.where(Position.is_open.is_(False))
                position_result = await session.execute(
                    position_stmt.order_by(
                        Position.closed_at.desc().nullslast(),
                        Position.created_at.desc(),
                    ).limit(1)
                )
                position = position_result.scalar_one_or_none()
                if position is not None:
                    payload["local_position_id"] = int(position.id)
        except Exception as exc:
            logger.debug(
                "strategy learning execution link lookup failed",
                symbol=symbol,
                error=safe_error_text(exc),
            )
        return payload

    def _position_exposure_context(
        self,
        open_positions: list[dict] | None,
        staged_entry_counts: dict[str, dict] | None = None,
    ) -> dict[str, Any]:
        """Compatibility delegate for older tests/tools that inspect exposure context."""

        exposure = getattr(self, "entry_position_exposure", None)
        if exposure is None:
            exposure = EntryPositionExposurePolicy()
        return exposure.context(open_positions, staged_entry_counts)

    def _candidate_opportunity_score(
        self,
        decision: DecisionOutput,
        strategy: dict[str, Any] | None = None,
    ) -> float:
        """Delegate opportunity aggregation to the authoritative return policy."""

        scorer = getattr(self, "entry_opportunity_score", None)
        if scorer is None:
            scorer = EntryOpportunityScoringPolicy(
                normalize_symbol=self._normalize_position_symbol,
                annotate_decision_source=self._annotate_decision_source,
            )
        return scorer.score_candidate(decision, strategy)

    async def _prepare_entry_for_hard_risk(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None = None,
        decision_db_id: int | None = None,
    ) -> None:
        """Generate dynamic return sizing before invoking the hard risk engine."""

        if not decision.is_entry:
            return
        await self.entry_policy.prepare_dynamic_risk_contract(
            decision,
            model_mode,
            open_positions=open_positions or [],
        )
        if decision_db_id is not None:
            await self._mark_decision_raw_response(
                decision_db_id,
                decision.raw_response if isinstance(decision.raw_response, dict) else {},
            )

    def _annotate_decision_source(self, decision: DecisionOutput) -> dict[str, Any]:
        raw = self._safe_dict(decision.raw_response)
        side = (
            self._entry_side_value(decision)
            if decision.is_entry
            else ("exit" if decision.is_exit else "hold")
        )
        opportunity = self._safe_dict(raw.get("opportunity_score"))
        policy = self._safe_dict(raw.get("production_return_policy"))
        dynamic_exit = self._safe_dict(raw.get("dynamic_exit_policy"))
        primary_source = (
            "authoritative_fee_after_return_distribution"
            if decision.is_entry
            else "position_economics_dynamic_exit"
            if decision.is_exit
            else "observation_only_hold"
        )

        raw["decision_source"] = {
            "primary_source": primary_source,
            "symbol": decision.symbol,
            "action": decision.action.value,
            "side": side,
            "expected_net_return_pct": opportunity.get("expected_net_return_pct"),
            "return_lcb_pct": opportunity.get("return_lcb_pct"),
            "production_return_eligible": policy.get("eligible"),
            "dynamic_exit_eligible": dynamic_exit.get("eligible"),
            "expert_memory_strategy_learning_role": "observation_only",
        }
        decision.raw_response = raw
        return raw

    def _annotate_candidate_selection(
        self,
        decision: DecisionOutput,
        *,
        rank: int | None = None,
        candidate_count: int | None = None,
        selected: bool | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        raw = self._safe_dict(decision.raw_response)
        opportunity = self._safe_dict(raw.get("opportunity_score"))
        if rank is not None:
            opportunity["rank"] = int(rank)
        if candidate_count is not None:
            opportunity["candidate_count"] = int(candidate_count)
        if selected is not None:
            opportunity["selected_for_execution"] = bool(selected)
        if reason:
            opportunity["selection_reason"] = reason
        raw["opportunity_score"] = opportunity
        decision.raw_response = raw
        return self._annotate_decision_source(decision)

    def _entry_opportunity_gate_reason(self, decision: DecisionOutput) -> str | None:
        """Return only severe entry blockers.

        AI should own the trade/no-trade decision. Opportunity score, local
        models, direction competition, and contribution stats are sent to AI
        and stored as advisory evidence. At execution time they may reduce size
        or add warnings, but they should not frequently veto AI entries.
        """
        entry_policy = getattr(self, "entry_policy", None)
        if entry_policy is not None:
            return entry_policy.gate_reason(decision)
        gate = getattr(self, "entry_opportunity_gate", None)
        if gate is not None:
            scorer = getattr(self, "entry_opportunity_score", None)
            if scorer is not None and decision.is_entry:
                raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                strategy = {}
                strategy_mode = raw.get("strategy_mode")
                if isinstance(strategy_mode, dict):
                    strategy.update(strategy_mode)
                learning_context = raw.get("strategy_learning_context")
                if isinstance(learning_context, dict):
                    strategy.update(learning_context)
                scorer.score_candidate(decision, strategy)
            return gate.safety_reason(decision)
        return EntryOpportunityGatePolicy(
            suspicious_symbol_policy=EntrySuspiciousSymbolPolicy(self._normalize_position_symbol),
        ).safety_reason(decision)

    async def _today_side_performance(self, mode: str) -> dict[str, dict[str, float]]:
        """Delegate today's long/short realized-PnL feedback to a dedicated service."""

        service = getattr(self, "daily_side_performance_service", None)
        if service is None:
            service = DailySidePerformanceService()
            self.daily_side_performance_service = service
        return await service.state(mode)

    async def _strategy_context_position_performance(self, mode: str) -> dict[str, dict[str, Any]]:
        """Load the four position-based strategy metrics from one bounded snapshot."""

        service = getattr(self, "strategy_context_performance_service", None)
        if service is None:
            service = StrategyContextPerformanceService(
                daily_performance=self.daily_performance_service,
                daily_side_performance=getattr(self, "daily_side_performance_service", None),
                symbol_side_performance=getattr(self, "symbol_side_performance_service", None),
            )
            self.strategy_context_performance_service = service
        return await service.recent(mode)

    async def _multiday_side_performance(self, mode: str) -> dict[str, dict[str, float]]:
        """Recent multi-day realized PnL split by side, for posture feedback."""
        service = getattr(self, "daily_side_performance_service", None)
        if service is None:
            service = DailySidePerformanceService()
            self.daily_side_performance_service = service
        return await service.multiday_state(mode, lookback_days=5.0)

    async def _recent_symbol_side_performance(self, mode: str) -> dict[str, dict[str, Any]]:
        """Delegate recent symbol/side realized-PnL feedback to a dedicated service."""

        service = getattr(self, "symbol_side_performance_service", None)
        if service is None:
            service = SymbolSidePerformanceService(
                normalize_symbol=self._normalize_position_symbol,
                lookback_limit=SYMBOL_SIDE_PROFILE_LOOKBACK,
                lookback_days=SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS,
            )
            self.symbol_side_performance_service = service
        return await service.recent(mode)

    async def _recent_model_contribution_performance(self, mode: str) -> dict[str, dict[str, Any]]:
        """Delegate closed-loop source performance to a dedicated service."""

        service = getattr(self, "model_contribution_performance_service", None)
        if service is None:
            service = ModelContributionPerformanceService(
                lookback_days=SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS,
            )
            self.model_contribution_performance_service = service
        return await service.recent(mode)

    def _rank_auto_feature_vectors(
        self,
        feature_vectors: dict[str, Any],
        limit: int,
    ) -> dict[str, Any]:
        result = self.entry_feature_ranker.rank(
            feature_vectors,
            limit,
        )
        logger.info(
            "auto opportunity shortlist",
            **result.diagnostics,
        )
        self._last_auto_feature_rank_diagnostics = result.diagnostics
        return result.selected

    def _market_candidate_funnel_snapshot(
        self,
        *,
        scan_symbols: list[str],
        open_position_filter: Any,
        unclaimed_filter: Any | None,
        fetch_symbols: list[str],
        feature_fetch_budget_diagnostics: dict[str, Any] | None,
        feature_vectors: dict[str, Any],
        invalid_symbols: list[str],
        market_feature_vectors_before_rank: dict[str, Any],
        market_feature_vectors_after_rank: dict[str, Any],
        market_feature_vectors_after_dedupe: dict[str, Any],
        rank_diagnostics: dict[str, Any] | None,
        analysis_budget_context: dict[str, Any],
        market_symbol_budget: int,
        run_market_analysis: bool,
        mode_is_auto_scan: bool,
        analysis_scope: str,
    ) -> dict[str, Any]:
        rank_diagnostics = self._safe_dict(rank_diagnostics)
        feature_fetch_budget = self._safe_dict(feature_fetch_budget_diagnostics)
        recent_dedupe = self._safe_dict(
            self._safe_dict(analysis_budget_context).get("recent_market_analysis_dedupe")
        )
        budget_rotation = self._safe_dict(
            self._safe_dict(analysis_budget_context).get("market_budget_rotation")
        )
        return {
            "read_only": True,
            "is_entry_gate": False,
            "mode": "auto" if mode_is_auto_scan else "manual",
            "analysis_scope": analysis_scope,
            "run_market_analysis": bool(run_market_analysis),
            "scan_symbol_count": len(scan_symbols or []),
            "open_position_filtered_count": len(getattr(open_position_filter, "skipped", []) or []),
            "unclaimed_filtered_count": (
                len(getattr(unclaimed_filter, "skipped", []) or [])
                if unclaimed_filter is not None
                else 0
            ),
            "feature_fetch_requested_count": len(fetch_symbols or []),
            "feature_fetch_budget": feature_fetch_budget,
            "feature_valid_count": len(feature_vectors or {}),
            "feature_invalid_count": len(invalid_symbols or []),
            "market_feature_before_rank_count": len(market_feature_vectors_before_rank or {}),
            "market_symbol_budget": int(market_symbol_budget or 0),
            "rank_selected_count": len(market_feature_vectors_after_rank or {}),
            "rank_tradable_candidates": rank_diagnostics.get("tradable_candidates"),
            "rank_secondary_candidates": rank_diagnostics.get("secondary_candidates"),
            "rank_total_candidates": rank_diagnostics.get("candidates"),
            "rank_underfilled": rank_diagnostics.get("rank_underfilled"),
            "rank_underfill_reason": rank_diagnostics.get("rank_underfill_reason"),
            "rank_filtered_out_candidates": rank_diagnostics.get("filtered_out_candidates"),
            "rank_filtered_out_reason_counts": rank_diagnostics.get(
                "filtered_out_reason_counts", []
            ),
            "rank_top_symbols": rank_diagnostics.get("symbols", []),
            "ranked_symbol_sample": rank_diagnostics.get("ranked_symbol_sample", []),
            "filtered_symbol_sample": rank_diagnostics.get("filtered_symbol_sample", []),
            "recent_analysis_dedupe_count": int(recent_dedupe.get("skipped_count") or 0),
            "recent_analysis_dedupe_symbols": recent_dedupe.get("skipped_symbols", []),
            "market_budget_rotation": budget_rotation,
            "market_feature_after_dedupe_count": len(market_feature_vectors_after_dedupe or {}),
            "market_feature_after_dedupe_symbols": list(
                (market_feature_vectors_after_dedupe or {}).keys()
            )[:20],
            "analysis_budget": {
                "risk_level": self._safe_dict(analysis_budget_context).get("risk_level"),
                "market_symbol_limit": self._safe_dict(analysis_budget_context).get(
                    "market_symbol_limit"
                ),
                "position_max_groups": self._safe_dict(analysis_budget_context).get(
                    "position_max_groups"
                ),
                "budget_source": self._safe_dict(analysis_budget_context).get("budget_source"),
                "market_limit_policy": self._safe_dict(analysis_budget_context).get(
                    "market_limit_policy"
                ),
                "configured_market_symbol_limit": self._safe_dict(analysis_budget_context).get(
                    "configured_market_symbol_limit"
                ),
                "position_group_count": self._safe_dict(analysis_budget_context).get(
                    "position_group_count"
                ),
                "target_position_groups": self._safe_dict(analysis_budget_context).get(
                    "target_position_groups"
                ),
                "roster_underfilled": self._safe_dict(analysis_budget_context).get(
                    "roster_underfilled"
                ),
                "market_limit_diagnostics": self._safe_dict(analysis_budget_context).get(
                    "market_limit_diagnostics"
                ),
                "reason": self._safe_dict(analysis_budget_context).get("reason"),
            },
            "diagnostic_boundary": (
                "Read-only market candidate funnel; use it to locate scan/fetch/rank/dedupe "
                "concentration before changing any entry threshold, leverage, sizing, ML "
                "readiness, or risk veto."
            ),
        }

    def _attach_market_candidate_funnel(
        self,
        decision: DecisionOutput,
        funnel: dict[str, Any] | None,
        progress: dict[str, Any] | None = None,
    ) -> None:
        if not funnel and not progress:
            return
        raw = self._safe_dict(decision.raw_response)
        if funnel:
            raw["market_candidate_funnel"] = funnel
        if progress:
            raw["market_analysis_progress"] = self._safe_dict(progress)
        decision.raw_response = raw

    def _market_analysis_progress_snapshot(
        self,
        *,
        symbol: str,
        market_index: int,
        market_total: int,
        round_start: datetime,
        market_ai_started_at: datetime | None = None,
        strategy_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        market_ai_started_at = market_ai_started_at or round_start
        full_round_elapsed_seconds = self._round_elapsed_seconds(round_start)
        market_ai_elapsed_seconds = self._round_elapsed_seconds(market_ai_started_at)
        base_budget_seconds = self.market_round_time_budget_seconds()
        budget_seconds = self.market_round_time_budget_seconds(
            strategy_context=strategy_context,
            market_symbol_count=market_total,
        )
        start_reserve_seconds = self.market_symbol_start_reserve_seconds(
            strategy_context=strategy_context,
            market_symbol_count=market_total,
        )
        remaining_budget_seconds = max(budget_seconds - market_ai_elapsed_seconds, 0.0)
        return {
            "read_only": True,
            "is_entry_gate": False,
            "symbol": symbol,
            "processed_index": int(market_index) + 1,
            "ranked_market_symbol_count": int(market_total),
            "remaining_after_this_symbol": max(int(market_total) - int(market_index) - 1, 0),
            "round_elapsed_seconds_before_ai": round(full_round_elapsed_seconds, 3),
            "full_round_elapsed_seconds_before_ai": round(full_round_elapsed_seconds, 3),
            "market_ai_elapsed_seconds_before_symbol": round(market_ai_elapsed_seconds, 3),
            "market_round_time_budget_seconds": round(budget_seconds, 3),
            "market_symbol_start_reserve_seconds": round(start_reserve_seconds, 3),
            "remaining_market_ai_budget_seconds": round(remaining_budget_seconds, 3),
            "can_start_another_market_symbol": bool(
                remaining_budget_seconds >= start_reserve_seconds
            ),
            "base_market_round_time_budget_seconds": round(base_budget_seconds, 3),
            "market_round_time_budget_policy": (
                "portfolio_roster_underfilled_extension"
                if budget_seconds > base_budget_seconds + 1e-9
                else "base_interval_budget"
            ),
            "budget_used_ratio_before_ai": round(
                market_ai_elapsed_seconds / max(budget_seconds, 1e-6),
                6,
            ),
            "market_ai_budget_used_ratio_before_symbol": round(
                market_ai_elapsed_seconds / max(budget_seconds, 1e-6),
                6,
            ),
            "budget_clock_scope": "market_ai_phase",
            "runtime_stage_durations": self._stage_durations_for_scope("market"),
            "diagnostic_boundary": (
                "Read-only market AI throughput diagnostics; it explains how many ranked "
                "symbols this round can analyze before soft scheduling budget is exhausted. "
                "The soft budget clock starts when the market AI phase begins, not at full "
                "round startup. It is not entry permission, sizing, leverage, ML readiness, "
                "or risk veto."
            ),
        }

    def _rotate_market_feature_vectors_for_budget_coverage(
        self,
        market_feature_vectors: dict[str, Any],
        *,
        analysis_budget_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        items = list((market_feature_vectors or {}).items())
        deferred = list(getattr(self, "_market_budget_deferred_symbols", []) or [])
        if len(items) <= 1 or not deferred:
            if isinstance(analysis_budget_context, dict):
                analysis_budget_context["market_budget_rotation"] = {
                    "read_only": True,
                    "is_entry_gate": False,
                    "applied": False,
                    "deferred_symbol_count": len(deferred),
                    "reason": "no deferred market symbols available for coverage rotation",
                }
            return dict(items)

        normalized_index = {
            self._normalize_position_symbol(symbol): index
            for index, (symbol, _fv) in enumerate(items)
            if self._normalize_position_symbol(symbol)
        }
        start_key = None
        start_index = 0
        for deferred_symbol in deferred:
            key = self._normalize_position_symbol(deferred_symbol)
            if key in normalized_index:
                start_key = key
                start_index = normalized_index[key]
                break
        if start_key is None or start_index <= 0:
            if isinstance(analysis_budget_context, dict):
                analysis_budget_context["market_budget_rotation"] = {
                    "read_only": True,
                    "is_entry_gate": False,
                    "applied": False,
                    "deferred_symbol_count": len(deferred),
                    "reason": (
                        "deferred symbols no longer match current shortlist"
                        if start_key is None
                        else "deferred symbol already leads current shortlist"
                    ),
                }
            return dict(items)

        rotated_items = items[start_index:] + items[:start_index]
        if isinstance(analysis_budget_context, dict):
            analysis_budget_context["market_budget_rotation"] = {
                "read_only": True,
                "is_entry_gate": False,
                "applied": True,
                "start_symbol": rotated_items[0][0],
                "start_index": start_index,
                "ranked_symbol_count": len(items),
                "deferred_symbol_count": len(deferred),
                "deferred_symbols": deferred[:20],
                "reason": (
                    "previous market AI round hit the soft time budget; current shortlist "
                    "is rotated to give deferred ranked symbols coverage without changing "
                    "ranking scores, entry thresholds, sizing, leverage, ML readiness, or risk gates"
                ),
            }
        return dict(rotated_items)

    def _remember_market_budget_deferred_symbols(self, symbols: list[str]) -> None:
        normalized_seen: set[str] = set()
        remembered: list[str] = []
        for symbol in symbols or []:
            key = self._normalize_position_symbol(symbol)
            if not key or key in normalized_seen:
                continue
            normalized_seen.add(key)
            remembered.append(symbol)
        self._market_budget_deferred_symbols = remembered[:50]

    async def _try_claim_analysis_symbol(self, symbol: str, scope: str) -> bool:
        normalized = self._normalize_position_symbol(symbol)
        if not normalized:
            return False
        key = f"{scope}:{normalized}" if scope == "market" else normalized
        async with self._analysis_symbol_lock:
            if normalized in self._active_analysis_symbols or key in self._active_analysis_symbols:
                return False
            self._active_analysis_symbols.add(normalized)
            return True

    async def _release_analysis_symbol(self, symbol: str) -> None:
        normalized = self._normalize_position_symbol(symbol)
        if not normalized:
            return
        async with self._analysis_symbol_lock:
            self._active_analysis_symbols.discard(normalized)

    async def initialize(self) -> None:
        """Initialize models, executors, and connections."""
        await self.models.initialize_all()
        self.paper_executor = None

        # Initialize OKX demo/live connections for balance sync, position checks,
        # and actual order execution. Paper mode means OKX demo trading, not a
        # local fake fill.
        self._okx_paper = OKXExecutor(mode="paper", load_markets_on_initialize=False)
        try:
            await self._okx_paper.initialize()
            logger.info("okx paper executor initialized")
        except Exception as e:
            logger.warning("okx paper executor init failed", error=safe_error_text(e))
            self._okx_paper = None
        self._okx_live = (
            OKXExecutor(mode="live", load_markets_on_initialize=False)
            if self._has_live_models()
            else None
        )
        if self._okx_live is not None:
            try:
                await self._okx_live.initialize()
                logger.info("okx live executor initialized")
            except Exception as e:
                logger.warning("okx live executor init failed", error=safe_error_text(e))

        # Restore decision/trade counters from DB
        async with get_session_ctx() as session:
            from sqlalchemy import func, select

            from models.decision import AIDecision
            from models.trade import Order

            dec_count = await session.execute(select(func.count(AIDecision.id)))
            self._decision_count = dec_count.scalar() or 0
            trade_count = await session.execute(
                select(func.count(Order.id)).where(
                    Order.status == OrderStatus.FILLED.value,
                    Order.exchange_order_id.is_not(None),
                    Order.exchange_order_id != "",
                )
            )
            self._trade_count = trade_count.scalar() or 0

        await self.expert_memory_service.backfill_trade_reflections(mode_manager.mode.value)
        await self._prime_strategy_context_performance_snapshot(mode_manager.mode.value)

        # Subscribe to mode changes to reinitialize LLM agent
        mode_manager.subscribe(self._on_mode_changed)

        logger.info("trading service initialized")

    async def run_once(self, analysis_scope: str = "full") -> dict[str, Any]:
        """Execute one iteration of the trading loop.

        Returns a summary dict for dashboard/notifications.
        """
        settings.refresh_runtime_env(force=True)
        if not self._running:
            return {"status": "stopped"}
        analysis_scope = (
            analysis_scope if analysis_scope in {"full", "market", "position"} else "full"
        )
        run_market_analysis = analysis_scope in {"full", "market"}
        run_position_analysis = analysis_scope in {"full", "position"}
        account_pause_reason = ""
        if mode_manager.is_paused:
            account_pause_reason = (
                "当前执行账户已暂停投资：停止新开仓和新交易对分析，"
                "已有仓位继续复盘直到触发正常平仓。"
            )
            if analysis_scope == "market":
                return {
                    "status": "paused",
                    "mode": mode_manager.mode.value,
                    "analysis_scope": analysis_scope,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "symbols_processed": 0,
                    "decisions": [],
                    "executions": [],
                    "warnings": [
                        {
                            "model": ENSEMBLE_TRADER_NAME,
                            "symbol": "ALL",
                            "warning": account_pause_reason,
                        }
                    ],
                    "market_analysis_paused": True,
                }
            run_market_analysis = False
            run_position_analysis = analysis_scope in {"full", "position"}
        new_pair_market_pause_applied = False
        round_start = datetime.now(UTC)
        round_deadline_monotonic: float | None = None
        if analysis_scope == "position":
            round_deadline_monotonic = (
                asyncio.get_running_loop().time() + self.position_round_watchdog_seconds()
            )
        results: dict[str, Any] = {
            "status": "ok",
            "mode": mode_manager.mode.value,
            "analysis_scope": analysis_scope,
            "timestamp": round_start.isoformat(),
            "symbols_processed": 0,
            "decisions": [],
            "executions": [],
            "warnings": [],
        }
        round_decision_ids: set[int] = set()
        round_decisions: dict[int, DecisionOutput] = {}
        claimed_analysis_symbols: list[str] = []
        claimed_symbol_keys: set[str] = set()
        published_dashboard_update = False
        feature_fetch_budget_diagnostics: dict[str, Any] = {}
        rank_diagnostics_snapshot: dict[str, Any] = {}
        scope_token = _analysis_scope_context.set(analysis_scope)
        self._start_runtime_round(analysis_scope, round_start)
        self._set_loop_stage("starting")

        try:
            self._set_loop_stage("shadow_backtests")
            await self._update_shadow_backtests_for_round(
                analysis_scope=analysis_scope,
                results=results,
            )
            self._set_loop_stage("stale_entry_maintenance")
            await self._update_stale_entry_candidates_for_round(results=results)

            # 0. Refresh per-model execution mode mapping from current config
            self._refresh_model_modes()
            self._set_loop_stage("sync_exchange_positions")
            if self._should_run_full_reconciliation_at_round_start(analysis_scope):
                await self.okx_sync_service.reconcile_positions(
                    f"{analysis_scope} round start",
                    timeout_seconds=self.round_start_reconcile_timeout_seconds(),
                    lock_wait_seconds=0.35,
                )
            else:
                logger.debug(
                    "market-only round skips full OKX reconciliation at start",
                    scope=analysis_scope,
                )
            self._set_loop_stage("load_open_positions")
            open_positions = await self._open_positions_context_for_round(analysis_scope)
            if self._should_recover_pending_exits_for_scope(analysis_scope):
                self._set_loop_stage("recover_pending_exits")
                await self._recover_pending_exit_decisions(
                    results,
                    open_positions,
                    round_decision_ids,
                )
            else:
                results.setdefault("pending_exit_recovery", {})[
                    "market_round_skipped"
                ] = {
                    "read_only": True,
                    "is_entry_gate": False,
                    "reason": (
                        "market-only 开仓扫描不执行 pending exit 恢复；该恢复任务由 position/full "
                        "轮处理，避免平仓补偿任务阻塞开仓候选扫描。"
                    ),
                }
            self._set_loop_stage("new_pair_pause_check")
            new_pair_pause_reason = await self._new_pair_analysis_pause_reason(
                ENSEMBLE_TRADER_NAME,
                open_positions=open_positions,
                allow_background_refresh=analysis_scope == "market",
            )
            if account_pause_reason:
                new_pair_pause_reason = account_pause_reason
            self._set_loop_stage("record_new_pair_pause_state")
            await self._record_new_pair_pause_state(ENSEMBLE_TRADER_NAME, new_pair_pause_reason)
            if new_pair_pause_reason and run_market_analysis:
                new_pair_market_pause_applied = True
                logger.warning(
                    "new-pair market analysis paused; existing-position review remains active",
                    reason=new_pair_pause_reason,
                    scope=analysis_scope,
                    open_positions=len(open_positions or []),
                )
                results["warnings"].append(
                    {
                        "model": ENSEMBLE_TRADER_NAME,
                        "symbol": "ALL",
                        "warning": new_pair_pause_reason,
                    }
                )
                # The account guard is only meant to stop opening new symbols.
                # Existing positions still need SL/TP enforcement and AI review.
                run_market_analysis = False
            elif account_pause_reason:
                results["warnings"].append(
                    {
                        "model": ENSEMBLE_TRADER_NAME,
                        "symbol": "ALL",
                        "warning": account_pause_reason,
                    }
                )

            # 1. Determine which symbols to process based on scan mode
            self._set_loop_stage("select_symbols")
            if new_pair_pause_reason:
                scan_symbols = []
            elif mode_manager.is_auto_scan:
                # Auto mode: rank all OKX USDT swaps, then pull indicators for
                # a larger candidate pool before spending AI tokens on the best.
                try:
                    available = await self.data_service.get_available_symbols()
                    limit = max(1, int(settings.auto_scan_symbol_limit))
                    pool_limit = min(
                        len(available),
                        max(
                            limit,
                            limit * AUTO_SCAN_ROTATION_POOL_MULTIPLIER,
                            AUTO_SCAN_ROTATION_POOL_MIN,
                            30,
                        ),
                    )
                    scan_symbols = [s["symbol"] for s in available[:pool_limit]]
                except Exception:
                    limit = max(1, int(settings.auto_scan_symbol_limit))
                    pool_limit = limit
                    scan_symbols = list(settings.symbols)
                logger.info(
                    "auto scan",
                    symbol_count=len(scan_symbols),
                    limit=settings.auto_scan_symbol_limit,
                    candidate_pool=pool_limit,
                )
            else:
                # Manual mode: only user-selected symbols
                limit = len(settings.symbols)
                scan_symbols = list(settings.symbols)

            position_scan_symbols = (
                sorted(self.entry_symbol_universe.open_position_symbol_keys(open_positions))
                if run_position_analysis
                else []
            )
            market_scan_symbols = self.entry_symbol_universe.dedupe_symbols(scan_symbols)
            open_position_filter = self.entry_symbol_universe.filter_open_position_market_symbols(
                market_scan_symbols,
                open_positions,
            )
            market_scan_symbols = open_position_filter.symbols
            if open_position_filter.skipped:
                logger.info(
                    "skipping open-position symbols from market analysis",
                    count=len(open_position_filter.skipped),
                    symbols=open_position_filter.skipped[:10],
                )
            if run_market_analysis:
                async with self._analysis_symbol_lock:
                    active_analysis_symbols = set(self._active_analysis_symbols)
                unclaimed_filter = self.entry_symbol_universe.filter_unclaimed_market_symbols(
                    market_scan_symbols,
                    active_analysis_symbols,
                )
                market_scan_symbols = unclaimed_filter.symbols
                if unclaimed_filter.skipped:
                    logger.info(
                        "skipping symbols already under position/market analysis",
                        count=len(unclaimed_filter.skipped),
                        symbols=unclaimed_filter.skipped[:10],
                    )
            else:
                market_scan_symbols = []

            fetch_symbols = self.entry_symbol_universe.dedupe_symbols(
                [
                    *market_scan_symbols,
                    *position_scan_symbols,
                ]
            )
            if run_market_analysis and mode_manager.is_auto_scan:
                fetch_symbols = self._budget_auto_scan_feature_symbols(
                    fetch_symbols,
                    position_scan_symbols,
                    configured_limit=max(1, int(settings.auto_scan_symbol_limit)),
                )
                feature_fetch_budget_diagnostics = self._safe_dict(
                    getattr(self, "_last_auto_feature_fetch_budget_diagnostics", None)
                )
            unclaimed_filter_for_funnel = unclaimed_filter if run_market_analysis else None
            if not fetch_symbols:
                diagnostics = {
                    "scan_symbol_count": len(scan_symbols or []),
                    "scan_symbol_sample": list(scan_symbols or [])[:10],
                    "market_scan_after_normalize": len(market_scan_symbols),
                    "open_position_filtered_count": len(open_position_filter.skipped),
                    "open_position_filtered_sample": open_position_filter.skipped[:10],
                    "unclaimed_filtered_count": (
                        len(unclaimed_filter.skipped) if run_market_analysis else 0
                    ),
                    "unclaimed_filtered_sample": (
                        unclaimed_filter.skipped[:10] if run_market_analysis else []
                    ),
                    "position_scan_symbol_count": len(position_scan_symbols),
                    "run_market_analysis": run_market_analysis,
                    "run_position_analysis": run_position_analysis,
                    "active_analysis_symbols_sample": (
                        sorted(active_analysis_symbols)[:10] if run_market_analysis else []
                    ),
                    "new_pair_pause_reason": new_pair_pause_reason,
                }
                results["scan_filter_diagnostics"] = diagnostics
                if new_pair_market_pause_applied and not run_position_analysis:
                    logger.info(
                        "market analysis skipped because new-pair pause is active",
                        reason=new_pair_pause_reason,
                        **diagnostics,
                    )
                elif open_positions and run_position_analysis:
                    logger.warning(
                        "no feature symbols available for open-position review",
                        **diagnostics,
                    )
                else:
                    logger.warning("all scan symbols skipped before AI analysis", **diagnostics)
                return results

            # 2. Get feature vectors for target symbols (parallel, with concurrency limit)
            self._set_loop_stage("fetch_features")
            feature_vectors = {}
            sem = asyncio.Semaphore(max(1, int(AUTO_SCAN_FEATURE_FETCH_CONCURRENCY)))
            feature_timeout = max(1.0, float(AUTO_SCAN_FEATURE_FETCH_TIMEOUT_SECONDS))

            async def fetch_fv(sym):
                async with sem:
                    try:
                        return sym, await asyncio.wait_for(
                            self._get_feature_vector_snapshot(
                                sym,
                                wait_for_sentiment=False,
                                block_on_remote_ticker=not (
                                    run_market_analysis
                                    and not run_position_analysis
                                    and mode_manager.is_auto_scan
                                ),
                                block_on_remote_indicators=not (
                                    run_market_analysis
                                    and not run_position_analysis
                                    and mode_manager.is_auto_scan
                                ),
                                block_on_remote_derivatives=not (
                                    run_market_analysis
                                    and not run_position_analysis
                                    and mode_manager.is_auto_scan
                                ),
                                allow_cached_indicator_build=not (
                                    run_market_analysis
                                    and not run_position_analysis
                                    and mode_manager.is_auto_scan
                                ),
                                allow_indicator_background_refresh=not (
                                    run_market_analysis
                                    and not run_position_analysis
                                    and mode_manager.is_auto_scan
                                ),
                            ),
                            timeout=feature_timeout,
                        )
                    except TimeoutError:
                        logger.warning(
                            "feature vector timed out",
                            symbol=sym,
                            timeout_seconds=feature_timeout,
                        )
                        return sym, None
                    except Exception as e:
                        logger.warning(
                            "feature vector failed",
                            symbol=sym,
                            error=safe_error_text(e),
                        )
                        return sym, None

            tasks = [asyncio.create_task(fetch_fv(s)) for s in fetch_symbols]
            batch_timeout = max(
                feature_timeout + 2.0,
                feature_timeout
                * (max(1, len(fetch_symbols)) / max(1, int(AUTO_SCAN_FEATURE_FETCH_CONCURRENCY)))
                + 2.0,
            )
            batch_timeout = min(
                batch_timeout,
                max(8.0, float(settings.decision_interval_seconds) * 0.80),
            )
            early_timeout = min(
                batch_timeout,
                max(
                    feature_timeout + 1.0,
                    float(settings.decision_interval_seconds or 0.0) * 0.25,
                ),
            )
            feature_fetch_started = asyncio.get_running_loop().time()
            done, pending = await asyncio.wait(tasks, timeout=early_timeout)
            fv_results = []
            for task in done:
                try:
                    fv_results.append(task.result())
                except asyncio.CancelledError:
                    continue
                except Exception as exc:
                    logger.warning(
                        "feature vector task failed after early wait",
                        error=safe_error_text(exc),
                    )
            early_valid_count = sum(
                1
                for _symbol, fv in fv_results
                if fv is not None and self._is_valid_feature_vector(fv)
            )
            early_quorum_met, early_quorum_diagnostics = (
                self._auto_scan_feature_fetch_early_quorum(
                    completed_valid_count=early_valid_count,
                    total_fetch_count=len(fetch_symbols),
                    configured_limit=max(1, int(settings.auto_scan_symbol_limit)),
                    run_market_analysis=run_market_analysis,
                    run_position_analysis=run_position_analysis,
                    auto_scan=mode_manager.is_auto_scan,
                    feature_fetch_budget_diagnostics=feature_fetch_budget_diagnostics,
                )
            )
            early_quorum_diagnostics.update(
                {
                    "early_wait_seconds": round(
                        asyncio.get_running_loop().time() - feature_fetch_started,
                        3,
                    ),
                    "early_timeout_seconds": round(early_timeout, 3),
                    "batch_timeout_seconds": round(batch_timeout, 3),
                    "pending_after_early_wait": len(pending),
                }
            )
            if pending and not early_quorum_met:
                remaining_timeout = max(
                    batch_timeout - (asyncio.get_running_loop().time() - feature_fetch_started),
                    0.0,
                )
                more_done, pending = await asyncio.wait(pending, timeout=remaining_timeout)
                for task in more_done:
                    try:
                        fv_results.append(task.result())
                    except asyncio.CancelledError:
                        continue
                    except Exception as exc:
                        logger.warning(
                            "feature vector task failed after batch wait",
                            error=safe_error_text(exc),
                        )
            feature_fetch_budget_diagnostics.update(
                {
                    "early_quorum": early_quorum_diagnostics,
                    "early_quorum_cancelled_pending": bool(pending and early_quorum_met),
                }
            )
            self._last_auto_feature_fetch_budget_diagnostics = dict(
                feature_fetch_budget_diagnostics
            )
            results["feature_fetch_early_quorum"] = early_quorum_diagnostics
            if pending:
                await drain_cancelled_tasks(pending)
                if early_quorum_met:
                    logger.info(
                        "feature vector early quorum reached; cancelling slow sources",
                        symbol_count=len(fetch_symbols),
                        completed=len(tasks) - len(pending),
                        pending=len(pending),
                        early_quorum=early_quorum_diagnostics,
                    )
                elif not early_quorum_met:
                    logger.warning(
                    "feature vector batch reached time budget; cancelling slow sources",
                    symbol_count=len(fetch_symbols),
                    completed=len(tasks) - len(pending),
                    pending=len(pending),
                    timeout_seconds=round(batch_timeout, 3),
                    early_quorum_met=early_quorum_met,
                )
                results.setdefault("warnings", []).append(
                    {
                        "model": ENSEMBLE_TRADER_NAME,
                        "symbol": "ALL",
                        "warning": ("本轮行情特征批量拉取超时，系统已跳过剩余候选并进入下一轮。"),
                    }
                )
            feature_vectors = {s: fv for s, fv in fv_results if fv is not None}
            invalid_symbols = [
                s for s, fv in feature_vectors.items() if not self._is_valid_feature_vector(fv)
            ]
            if invalid_symbols:
                logger.warning(
                    "skipping invalid feature vectors before AI",
                    count=len(invalid_symbols),
                    symbols=invalid_symbols[:10],
                )
                feature_vectors = {
                    s: fv for s, fv in feature_vectors.items() if self._is_valid_feature_vector(fv)
                }

            if not feature_vectors:
                logger.warning("no feature vectors available")
                return results

            self._set_loop_stage("build_strategy_context")
            self._last_auto_feature_rank_diagnostics = {}
            market_scan_keys = {
                self._normalize_position_symbol(s)
                for s in market_scan_symbols
                if self._normalize_position_symbol(s)
            }
            market_feature_vectors = {
                s: fv
                for s, fv in feature_vectors.items()
                if self._normalize_position_symbol(s) in market_scan_keys
            }

            if not run_market_analysis:
                market_feature_vectors = {}
            market_feature_vectors_before_rank = dict(market_feature_vectors)
            market_feature_vectors_after_rank = dict(market_feature_vectors)
            base_market_limit = (
                max(1, int(settings.auto_scan_symbol_limit))
                if mode_manager.is_auto_scan
                else len(market_feature_vectors)
            )
            market_regime_context = self._market_regime_context(
                market_feature_vectors or feature_vectors
            )
            strategy_mode_context = await self._strategy_mode_context(
                self._get_model_execution_mode(ENSEMBLE_TRADER_NAME),
                market_regime_context,
                open_positions,
            )
            strategy_mode_context["portfolio_correlation"] = (
                build_portfolio_correlation_context(feature_vectors, open_positions)
            )
            analysis_budget_context = self._position_review_budget_context(
                open_positions,
                feature_vectors,
                base_market_limit=base_market_limit,
                run_position_analysis=run_position_analysis,
                run_market_analysis=run_market_analysis,
                new_pair_pause_reason=new_pair_pause_reason,
                strategy_context=strategy_mode_context,
            )
            market_symbol_budget = int(analysis_budget_context.get("market_symbol_limit") or 0)
            if run_market_analysis and market_feature_vectors:
                if market_symbol_budget <= 0:
                    market_feature_vectors = {}
                    market_feature_vectors_after_rank = {}
                elif mode_manager.is_auto_scan:
                    market_feature_vectors = self._rank_auto_feature_vectors(
                        market_feature_vectors, market_symbol_budget
                    )
                    rank_diagnostics_snapshot = self._safe_dict(
                        getattr(self, "_last_auto_feature_rank_diagnostics", None)
                    )
                    market_feature_vectors_after_rank = dict(market_feature_vectors)
                elif len(market_feature_vectors) > market_symbol_budget:
                    allowed_keys = {
                        self._normalize_position_symbol(s)
                        for s in list(market_feature_vectors.keys())[:market_symbol_budget]
                    }
                    market_feature_vectors = {
                        s: fv
                        for s, fv in market_feature_vectors.items()
                        if self._normalize_position_symbol(s) in allowed_keys
                    }
                    market_feature_vectors_after_rank = dict(market_feature_vectors)
                market_feature_vectors = self._rotate_market_feature_vectors_for_budget_coverage(
                    market_feature_vectors,
                    analysis_budget_context=analysis_budget_context,
                )
            market_feature_vectors_after_dedupe = dict(market_feature_vectors)
            market_candidate_funnel = self._market_candidate_funnel_snapshot(
                scan_symbols=list(scan_symbols or []),
                open_position_filter=open_position_filter,
                unclaimed_filter=unclaimed_filter_for_funnel,
                fetch_symbols=list(fetch_symbols or []),
                feature_fetch_budget_diagnostics=feature_fetch_budget_diagnostics,
                feature_vectors=feature_vectors,
                invalid_symbols=invalid_symbols,
                market_feature_vectors_before_rank=market_feature_vectors_before_rank,
                market_feature_vectors_after_rank=market_feature_vectors_after_rank,
                market_feature_vectors_after_dedupe=market_feature_vectors_after_dedupe,
                rank_diagnostics=rank_diagnostics_snapshot,
                analysis_budget_context=analysis_budget_context,
                market_symbol_budget=market_symbol_budget,
                run_market_analysis=run_market_analysis,
                mode_is_auto_scan=mode_manager.is_auto_scan,
                analysis_scope=analysis_scope,
            )
            results["market_candidate_funnel"] = market_candidate_funnel
            results["analysis_budget"] = analysis_budget_context
            logger.info(
                "analysis budget selected",
                risk_level=analysis_budget_context.get("risk_level"),
                position_max_groups=analysis_budget_context.get("position_max_groups"),
                market_symbol_limit=analysis_budget_context.get("market_symbol_limit"),
                forced_exit_groups=analysis_budget_context.get("forced_exit_groups"),
                priority_groups=analysis_budget_context.get("priority_groups"),
                target_position_groups=analysis_budget_context.get("target_position_groups"),
                budget_source=analysis_budget_context.get("budget_source"),
                reason=analysis_budget_context.get("reason"),
            )
            if mode_manager.is_auto_scan and market_feature_vectors:
                # Already ranked by the dynamic analysis budget above.
                pass
            strategy_mode_context["analysis_budget"] = analysis_budget_context
            logger.info(
                "market regime prediction",
                mode=market_regime_context.get("mode"),
                sample_count=market_regime_context.get("sample_count"),
                reason=market_regime_context.get("reason"),
            )
            logger.info(
                "strategy mode selected",
                strategy=strategy_mode_context.get("strategy"),
                posture=strategy_mode_context.get("posture"),
                exposure=strategy_mode_context.get("position_exposure"),
                reason=strategy_mode_context.get("reason"),
            )

            self._set_loop_stage("refresh_position_prices")
            if self._should_refresh_position_prices_before_review(analysis_scope):
                refresh_diagnostics = await self.okx_sync_service.refresh_position_prices(
                    feature_vectors
                )
                self._last_position_price_refresh_diagnostics = self._safe_dict(
                    refresh_diagnostics
                )
                results["position_price_refresh"] = {
                    "read_only": True,
                    "is_entry_gate": False,
                    "skipped": False,
                    "reason": "position/full round refreshes persisted open-position prices before review",
                    "diagnostics": self._safe_dict(refresh_diagnostics),
                }
            else:
                results["position_price_refresh"] = {
                    "read_only": True,
                    "is_entry_gate": False,
                    "skipped": True,
                    "reason": (
                        "market-only round skips synchronous open-position price refresh; "
                        "position review and background OKX sync own persisted position pricing"
                    ),
                }

            # 2.5 Enforce stop-loss / take-profit before AI decisions
            review_blocked_keys: set[tuple[str, str]] = set()
            if run_position_analysis:
                self._set_loop_stage("position_review")
                open_positions, review_blocked_keys = (
                    await self.position_review_service.review_open_positions(
                        feature_vectors=feature_vectors,
                        results=results,
                        round_decision_ids=round_decision_ids,
                        open_positions=open_positions,
                        position_entry_pause_reason=new_pair_pause_reason,
                        max_groups_override=int(
                            analysis_budget_context.get("position_max_groups")
                            or POSITION_REVIEW_MAX_GROUPS_PER_ROUND
                        ),
                        claimed_analysis_symbols=claimed_analysis_symbols,
                        round_deadline_monotonic=round_deadline_monotonic,
                    )
                )
                strategy_mode_context = self._refresh_dynamic_capacity(
                    open_positions=open_positions,
                    strategy_context=strategy_mode_context,
                    market_regime=market_regime_context,
                    account_equity=float(strategy_mode_context.get("account_equity") or 0.0),
                )

            # 3. Collect all entry decisions from all symbols/models
            all_candidates: list[tuple[str, str, DecisionOutput, Any, int | None]] = []
            staged_entry_counts = self.entry_capacity.empty_staged_counts()
            market_round_skipped_by_budget: list[str] = []

            if new_pair_market_pause_applied:
                self.market_decision_result_recorder.append_result(
                    results=results,
                    model_name=ENSEMBLE_TRADER_NAME,
                    symbol="ALL",
                    decision_or_action="hold",
                    model_mode=self._get_model_execution_mode(ENSEMBLE_TRADER_NAME),
                    approved=False,
                    execution_status="paused",
                    reason=new_pair_pause_reason,
                )

            market_feature_items = list(market_feature_vectors.items())
            market_execution_cost_facts: dict[str, dict[str, Any]] = {}
            market_ai_started_at = datetime.now(UTC)
            market_ai_budget_seconds = self.market_round_time_budget_seconds(
                strategy_context=strategy_mode_context,
                market_symbol_count=len(market_feature_items),
            )
            market_ai_deadline_monotonic = (
                asyncio.get_running_loop().time() + market_ai_budget_seconds
            )
            for market_index, (symbol, fv) in enumerate(market_feature_items):
                if market_index > 0 and self._market_ai_budget_exhausted(
                    market_ai_started_at,
                    strategy_context=strategy_mode_context,
                    market_symbol_count=len(market_feature_items),
                ):
                    remaining = [
                        item_symbol for item_symbol, _item_fv in market_feature_items[market_index:]
                    ]
                    market_round_skipped_by_budget = remaining
                    budget_seconds = self.market_round_time_budget_seconds(
                        strategy_context=strategy_mode_context,
                        market_symbol_count=len(market_feature_items),
                    )
                    start_reserve_seconds = self.market_symbol_start_reserve_seconds(
                        strategy_context=strategy_mode_context,
                        market_symbol_count=len(market_feature_items),
                    )
                    market_ai_elapsed_seconds = self._round_elapsed_seconds(market_ai_started_at)
                    full_round_elapsed_seconds = self._round_elapsed_seconds(round_start)
                    remaining_budget_seconds = max(
                        budget_seconds - market_ai_elapsed_seconds,
                        0.0,
                    )
                    warning = (
                        "本轮市场 AI 分析已达到调度时间预算，剩余候选顺延到后续轮次；"
                        "这不是开仓门槛，只用于防止单轮分析拖住系统心跳。"
                    )
                    logger.warning(
                        "market analysis round reached time budget",
                        elapsed_seconds=round(market_ai_elapsed_seconds, 3),
                        market_ai_elapsed_seconds=round(market_ai_elapsed_seconds, 3),
                        full_round_elapsed_seconds=round(full_round_elapsed_seconds, 3),
                        budget_seconds=round(budget_seconds, 3),
                        remaining_budget_seconds=round(remaining_budget_seconds, 3),
                        start_reserve_seconds=round(start_reserve_seconds, 3),
                        skipped_count=len(remaining),
                        skipped_symbols=remaining[:10],
                    )
                    results.setdefault("warnings", []).append(
                        {
                            "model": ENSEMBLE_TRADER_NAME,
                            "symbol": "ALL",
                            "warning": warning,
                            "elapsed_seconds": round(market_ai_elapsed_seconds, 3),
                            "market_ai_elapsed_seconds": round(market_ai_elapsed_seconds, 3),
                            "full_round_elapsed_seconds": round(full_round_elapsed_seconds, 3),
                            "budget_seconds": round(budget_seconds, 3),
                            "remaining_budget_seconds": round(remaining_budget_seconds, 3),
                            "market_symbol_start_reserve_seconds": round(
                                start_reserve_seconds,
                                3,
                            ),
                            "skipped_symbols": remaining[:20],
                        }
                    )
                    self._remember_market_budget_deferred_symbols(remaining)
                    break
                if not await self._try_claim_analysis_symbol(symbol, "market"):
                    logger.info(
                        "market symbol skipped because another analysis owns it", symbol=symbol
                    )
                    continue
                claimed_analysis_symbols.append(symbol)
                claimed_symbol_keys.add(self._normalize_position_symbol(symbol))
                self._set_loop_stage(f"market_ai:{symbol}")

                results["symbols_processed"] += 1
                fv = await self._fresh_feature_vector_for_analysis(symbol, fv)
                if not self._is_valid_feature_vector(fv):
                    logger.warning("skip symbol after fresh feature check failed", symbol=symbol)
                    continue
                model_name = ENSEMBLE_TRADER_NAME
                model_mode = self._get_model_execution_mode(model_name)
                if model_mode not in market_execution_cost_facts:
                    market_execution_cost_facts[model_mode] = (
                        await self._market_execution_cost_facts(model_mode)
                    )
                attach_execution_cost_facts(
                    fv,
                    market_execution_cost_facts.get(model_mode),
                )
                feature_vectors[symbol] = fv
                market_analysis_progress = self._market_analysis_progress_snapshot(
                    symbol=symbol,
                    market_index=market_index,
                    market_total=len(market_feature_items),
                    round_start=round_start,
                    market_ai_started_at=market_ai_started_at,
                    strategy_context=strategy_mode_context,
                )
                memory_context = await self._memory_context_with_vector_feedback(symbol)
                ml_signal_context = self.ml_signal_service.predict(fv)
                local_ai_tools_context = await self._local_ai_tools_context(
                    fv,
                    ml_signal_context,
                    open_positions=open_positions,
                    include_exit_advice=False,
                )
                direction_competition_context = self._direction_competition_context(
                    fv,
                    ml_signal_context,
                    local_ai_tools_context,
                    market_regime_context,
                    strategy_mode_context,
                )
                entry_candidate_evidence = self._ai_entry_candidate_evidence(
                    fv,
                    strategy_mode_context,
                    ml_signal_context,
                    local_ai_tools_context,
                    direction_competition_context,
                    memory_context.get("memory_feedback"),
                )
                market_agent_skills = self.agent_skills.market_skills(
                    new_pair_pause_reason=new_pair_pause_reason,
                    ml_signal=ml_signal_context,
                    local_ai_tools=local_ai_tools_context,
                    market_regime=market_regime_context,
                    strategy_mode=strategy_mode_context,
                )
                prefilter_reason = self.entry_market_llm_prefilter.skip_reason(
                    fv,
                    local_ai_tools_context,
                    open_positions=open_positions,
                )
                market_data_quality_issue = (
                    self.entry_market_data_quality.issue(fv, stage_label="AI分析前")
                    if prefilter_reason
                    else None
                )
                if prefilter_reason:
                    quick_raw = {
                        "analysis_type": "market",
                        "fast_prefilter": {
                            "skipped_llm": True,
                            "reason": prefilter_reason,
                            "market_data_quality": (
                                market_data_quality_issue.as_dict()
                                if market_data_quality_issue
                                else None
                            ),
                            "feature_opportunity_score": round(
                                self._feature_opportunity_score(fv), 4
                            ),
                        },
                        "ml_signal": ml_signal_context,
                        "local_ai_tools": local_ai_tools_context,
                        "direction_competition": direction_competition_context,
                        "entry_candidate_evidence": entry_candidate_evidence,
                        "market_candidate_funnel": market_candidate_funnel,
                        "agent_skills": {
                            "version": 1,
                            "phases": {
                                "market_prefilter": {
                                    "phase": "market_prefilter",
                                    "recorded_at": datetime.now(UTC).isoformat(),
                                    "note": prefilter_reason,
                                    "skills": [skill.to_dict() for skill in market_agent_skills],
                                },
                            },
                            "summary": self.agent_skills.summary(market_agent_skills),
                        },
                    }
                    quick_decision = DecisionOutput(
                        model_name=ENSEMBLE_TRADER_NAME,
                        symbol=symbol,
                        action=Action.HOLD,
                        confidence=0.0,
                        reasoning=prefilter_reason,
                        position_size_pct=0.0,
                        suggested_leverage=1.0,
                        stop_loss_pct=0.0,
                        take_profit_pct=0.0,
                        raw_response=quick_raw,
                        feature_snapshot=fv.to_dict() if hasattr(fv, "to_dict") else {},
                    )
                    self._attach_market_candidate_funnel(
                        quick_decision,
                        market_candidate_funnel,
                        market_analysis_progress,
                    )
                    self._attach_strategy_learning_context(quick_decision, strategy_mode_context)
                    decision_db_id = await self._log_decision(
                        quick_decision, is_paper=(model_mode == "paper")
                    )
                    if decision_db_id is not None:
                        round_decision_ids.add(decision_db_id)
                        round_decisions[decision_db_id] = quick_decision
                        await self._mark_decision_reason(decision_db_id, prefilter_reason)
                    self._decision_count += 1
                    self.market_decision_result_recorder.append_result(
                        results=results,
                        model_name=model_name,
                        symbol=symbol,
                        decision_or_action="hold",
                        model_mode=model_mode,
                        approved=True,
                        execution_status="fast_prefilter",
                        reason=prefilter_reason,
                        confidence=0.0,
                    )
                    continue
                analysis_started = datetime.now(UTC)
                decision, _opinions = await self.ensemble.decide(
                    fv,
                    {
                        "open_positions": open_positions,
                        "trading_mode": mode_manager.mode.value,
                        **memory_context,
                        "market_regime": market_regime_context,
                        "strategy_mode": strategy_mode_context,
                        "direction_competition": direction_competition_context,
                        "entry_candidate_evidence": entry_candidate_evidence,
                        "ml_signal": {} if PRE_AGENT_SKILLS_ROLLBACK_MODE else ml_signal_context,
                        "local_ai_tools": (
                            {} if PRE_AGENT_SKILLS_ROLLBACK_MODE else local_ai_tools_context
                        ),
                        "ml_signal_prompt_enabled": LOCAL_QUANT_PROMPT_ENABLED,
                        "local_ai_tools_prompt_enabled": LOCAL_QUANT_PROMPT_ENABLED,
                        # The registry and final trader use this cooperative deadline to
                        # bound batch fallbacks and arbitration to the current symbol's
                        # remaining market-AI budget.  It is not a trading gate.
                        "_analysis_deadline_monotonic": market_ai_deadline_monotonic,
                        "_analysis_budget_scope": "market_ai",
                        "_analysis_budget_seconds": market_ai_budget_seconds,
                    },
                )
                if isinstance(decision.raw_response, dict):
                    decision.raw_response.setdefault(
                        "entry_candidate_evidence", entry_candidate_evidence
                    )
                self._attach_market_candidate_funnel(
                    decision,
                    market_candidate_funnel,
                    market_analysis_progress,
                )
                if decision.is_entry:
                    self._candidate_opportunity_score(decision, strategy_mode_context)
                self._attach_strategy_learning_context(decision, strategy_mode_context)
                self._attach_decision_timing(decision, analysis_started, "market")
                self.agent_skills.attach(
                    decision,
                    phase="market_analysis",
                    skills=market_agent_skills,
                    note="市场分析前的 Agent/Skills 证据快照。",
                )
                decision_db_id = await self._log_decision(
                    decision, is_paper=(model_mode == "paper")
                )
                if decision_db_id is not None:
                    round_decision_ids.add(decision_db_id)
                    round_decisions[decision_db_id] = decision
                self._decision_count += 1
                await self.shadow_backtest_service.create(
                    decision_db_id,
                    decision,
                    fv,
                    model_mode,
                    analysis_type="market",
                    local_ai_tools_context=local_ai_tools_context,
                )

                decision_key = (model_name, self._normalize_position_symbol(symbol))
                if decision_key in review_blocked_keys and not decision.is_hold:
                    reason = "本轮持仓复盘已优先处理该币种，跳过后续重复信号。"
                    if decision_db_id is not None:
                        await self._mark_decision_reason(decision_db_id, reason)
                    self.market_decision_result_recorder.append_result(
                        results=results,
                        model_name=model_name,
                        symbol=symbol,
                        decision_or_action=decision,
                        model_mode=model_mode,
                        approved=True,
                        execution_status="skipped",
                        reason=reason,
                    )
                    continue

                try:
                    await self._prepare_entry_for_hard_risk(
                        decision,
                        model_mode,
                        open_positions,
                        decision_db_id,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    reason = (
                        "动态费后收益风险预算生成失败，本次 entry 失败关闭："
                        f"{safe_error_text(exc, limit=160)}"
                    )
                    raw_response = self._annotate_candidate_selection(
                        decision,
                        selected=False,
                        reason=reason,
                    )
                    if decision_db_id is not None:
                        await self._mark_decision_raw_response(decision_db_id, raw_response)
                        await self._record_and_persist_decision_stage(
                            decision_db_id,
                            decision,
                            DecisionStage.RISK_CHECK,
                            DecisionStageStatus.FAILED,
                            reason,
                            {
                                "blocker": "dynamic_entry_risk_contract_preparation",
                                "selected_for_execution": False,
                            },
                        )
                        await self._mark_decision_reason(decision_db_id, reason)
                    self.market_decision_result_recorder.append_result(
                        results=results,
                        model_name=model_name,
                        symbol=symbol,
                        decision_or_action=decision,
                        model_mode=model_mode,
                        approved=False,
                        execution_status="error",
                        reason=reason,
                    )
                    continue

                assessment = await self.market_decision_risk_assessment.assess(
                    decision=decision,
                    model_name=model_name,
                    open_positions=open_positions,
                    feature_vector=fv,
                    strategy_mode_context=strategy_mode_context,
                )

                if not assessment.approved:
                    logger.info(
                        "risk blocked decision",
                        model=model_name,
                        symbol=symbol,
                        reason=assessment.rejection_reason,
                    )
                    reason = assessment.rejection_reason or "风控引擎拒绝该决策。"
                    if decision_db_id is not None:
                        await self._mark_decision_reason(decision_db_id, reason)
                    self.market_decision_result_recorder.append_result(
                        results=results,
                        model_name=model_name,
                        symbol=symbol,
                        decision_or_action=decision,
                        model_mode=model_mode,
                        approved=False,
                        execution_status="rejected",
                        reason=reason,
                    )
                    if decision_db_id is not None:
                        raw_response = self._annotate_candidate_selection(
                            decision,
                            selected=False,
                            reason=reason,
                        )
                        await self._mark_decision_raw_response(decision_db_id, raw_response)
                        await self._record_and_persist_decision_stage(
                            decision_db_id,
                            decision,
                            DecisionStage.RISK_CHECK,
                            DecisionStageStatus.BLOCKED,
                            reason,
                            {
                                "blocker": "hard_risk_engine",
                                "selected_for_execution": False,
                            },
                        )
                        await self._mark_decision_reason(decision_db_id, reason)
                    continue

                executed = assessment.decision if assessment.decision else decision
                if executed is not decision and decision.raw_response and not executed.raw_response:
                    executed.raw_response = decision.raw_response
                    executed.feature_snapshot = (
                        executed.feature_snapshot or decision.feature_snapshot
                    )
                if executed.is_hold:
                    if decision_db_id is not None:
                        if isinstance(executed.raw_response, dict):
                            await self._mark_decision_raw_response(
                                decision_db_id,
                                executed.raw_response,
                            )
                        await self._mark_decision_reason(
                            decision_db_id,
                            "多模型裁决结果为观望，未提交订单。",
                        )
                    self.market_decision_result_recorder.append_result(
                        results=results,
                        model_name=model_name,
                        symbol=symbol,
                        decision_or_action="hold",
                        model_mode=model_mode,
                        approved=True,
                        confidence=executed.confidence,
                    )
                    continue

                if executed.is_exit:
                    reason = (
                        "市场分析阶段禁止执行平仓动作；平仓只允许由持仓分析产生，本轮改为观望。"
                    )
                    raw_response = (
                        executed.raw_response if isinstance(executed.raw_response, dict) else {}
                    )
                    raw_response["market_exit_execution_guard"] = {
                        "applied": True,
                        "original_action": executed.action.value,
                        "reason": "market_analysis_close_forbidden",
                    }
                    executed.raw_response = raw_response
                    if decision_db_id is not None:
                        await self._mark_decision_raw_response(decision_db_id, raw_response)
                        await self._mark_decision_reason(decision_db_id, reason)
                    self.market_decision_result_recorder.append_result(
                        results=results,
                        model_name=model_name,
                        symbol=symbol,
                        decision_or_action=executed,
                        model_mode=model_mode,
                        approved=True,
                        execution_status="skipped",
                        reason=reason,
                    )
                    continue

                if new_pair_pause_reason and executed.is_entry:
                    raw_response = self._annotate_candidate_selection(
                        decision,
                        selected=False,
                        reason=new_pair_pause_reason,
                    )
                    if decision_db_id is not None:
                        await self._mark_decision_raw_response(decision_db_id, raw_response)
                        await self._mark_decision_reason(decision_db_id, new_pair_pause_reason)
                    self.market_decision_result_recorder.append_result(
                        results=results,
                        model_name=model_name,
                        symbol=symbol,
                        decision_or_action=executed,
                        model_mode=model_mode,
                        approved=True,
                        execution_status="skipped",
                        reason=new_pair_pause_reason,
                    )
                    continue

                regime_reason = self.entry_market_regime.reason(
                    executed,
                    strategy_mode_context or market_regime_context,
                )
                if regime_reason:
                    raw_response = self._annotate_candidate_selection(
                        decision,
                        selected=False,
                        reason=regime_reason,
                    )
                    if decision_db_id is not None:
                        await self._mark_decision_raw_response(decision_db_id, raw_response)
                        await self._mark_decision_reason(decision_db_id, regime_reason)
                    self.market_decision_result_recorder.append_result(
                        results=results,
                        model_name=model_name,
                        symbol=symbol,
                        decision_or_action=executed,
                        model_mode=model_mode,
                        approved=True,
                        execution_status="skipped",
                        reason=regime_reason,
                    )
                    continue

                if mode_manager.is_auto_scan:
                    await self.market_auto_entry_processor.process(
                        symbol=symbol,
                        model_name=model_name,
                        decision=executed,
                        assessment=assessment,
                        decision_db_id=decision_db_id,
                        results=results,
                        model_mode=model_mode,
                        open_positions=open_positions,
                        staged_entry_counts=staged_entry_counts,
                        strategy_mode_context=strategy_mode_context,
                    )
                    continue

                await self.market_direct_entry_processor.process(
                    symbol=symbol,
                    model_name=model_name,
                    original_decision=decision,
                    executed=executed,
                    assessment=assessment,
                    decision_db_id=decision_db_id,
                    results=results,
                    model_mode=model_mode,
                    open_positions=open_positions,
                    staged_entry_counts=staged_entry_counts,
                )
                continue

            if market_round_skipped_by_budget:
                market_ai_elapsed_seconds = self._round_elapsed_seconds(market_ai_started_at)
                full_round_elapsed_seconds = self._round_elapsed_seconds(round_start)
                results["market_analysis_budget"] = {
                    "budget_seconds": round(self.market_round_time_budget_seconds(), 3),
                    "elapsed_seconds": round(market_ai_elapsed_seconds, 3),
                    "market_ai_elapsed_seconds": round(market_ai_elapsed_seconds, 3),
                    "full_round_elapsed_seconds": round(full_round_elapsed_seconds, 3),
                    "remaining_budget_seconds": round(
                        max(
                            self.market_round_time_budget_seconds(
                                strategy_context=strategy_mode_context,
                                market_symbol_count=len(market_feature_items),
                            )
                            - market_ai_elapsed_seconds,
                            0.0,
                        ),
                        3,
                    ),
                    "market_symbol_start_reserve_seconds": round(
                        self.market_symbol_start_reserve_seconds(
                            strategy_context=strategy_mode_context,
                            market_symbol_count=len(market_feature_items),
                        ),
                        3,
                    ),
                    "budget_clock_scope": "market_ai_phase",
                    "processed_symbols": int(results.get("symbols_processed") or 0),
                    "deferred_symbols": market_round_skipped_by_budget[:50],
                    "deferred_count": len(market_round_skipped_by_budget),
                    "is_entry_gate": False,
                    "reason": (
                        "市场分析轮次达到调度预算，剩余币种通过滚动扫描顺延；"
                        "该预算不参与开仓风控评分。"
                    ),
                }

            if not market_round_skipped_by_budget and market_feature_items:
                self._remember_market_budget_deferred_symbols([])

            ranked_entry_candidates = self._entry_candidate_queue_policy().ranked(
                all_candidates,
                strategy_mode_context,
            )
            all_candidates = [ranked.candidate for ranked in ranked_entry_candidates]
            for ranked in ranked_entry_candidates:
                _symbol, _model_name, decision, _assessment, decision_db_id = ranked.candidate
                raw_response = self._annotate_candidate_selection(
                    decision,
                    rank=ranked.rank,
                    candidate_count=ranked.candidate_count,
                    selected=False,
                    reason=ranked.wait_reason,
                )
                if decision_db_id is not None:
                    raw_response = await self._record_and_persist_decision_stage(
                        decision_db_id,
                        decision,
                        DecisionStage.STRATEGY_ARBITRATION,
                        DecisionStageStatus.PENDING,
                        ranked.wait_reason,
                        {
                            "rank": ranked.rank,
                            "candidate_count": ranked.candidate_count,
                            "score": round(float(ranked.score), 6),
                            "selected_for_execution": False,
                        },
                    )
                    await self._mark_decision_reason(decision_db_id, ranked.wait_reason)

            filtered_entry_candidates = self._entry_candidate_filter_policy().filter(
                all_candidates,
                strategy_context=strategy_mode_context,
                market_regime_context=market_regime_context,
                open_positions=open_positions,
                staged_entry_counts=staged_entry_counts,
            )
            for rejected in filtered_entry_candidates.rejected_candidates:
                symbol, model_name, decision, _assessment, decision_db_id = rejected.candidate
                if rejected.annotate_raw_response:
                    raw_response = self._annotate_candidate_selection(
                        decision,
                        selected=False,
                        reason=rejected.reason,
                    )
                    if decision_db_id is not None:
                        await self._mark_decision_raw_response(decision_db_id, raw_response)
                if decision_db_id is not None:
                    status = (
                        DecisionStageStatus.BLOCKED
                        if rejected.blocker == "entry_gate"
                        else DecisionStageStatus.SKIPPED
                    )
                    await self._record_and_persist_decision_stage(
                        decision_db_id,
                        decision,
                        DecisionStage.RISK_CHECK,
                        status,
                        rejected.reason,
                        {
                            "blocker": rejected.blocker,
                            "skip_kind": rejected.blocker,
                            "selected_for_execution": False,
                        },
                    )
                    await self._mark_decision_reason(decision_db_id, rejected.reason)
                self.market_decision_result_recorder.append_result(
                    results=results,
                    model_name=model_name,
                    symbol=symbol,
                    decision_or_action=decision,
                    model_mode=self._get_model_execution_mode(model_name),
                    approved=True,
                    execution_status="skipped",
                    reason=rejected.reason,
                )
            all_candidates = filtered_entry_candidates.accepted_candidates

            # 4. Filter entry candidates: auto mode no longer limits the number of executable entries.
            if mode_manager.is_auto_scan:
                candidates_to_execute = all_candidates
            else:
                candidates_to_execute = all_candidates

            # 5. Execute selected entry decisions
            for symbol, model_name, decision, assessment, decision_db_id in candidates_to_execute:
                process_result = await self.market_queued_entry_processor.process(
                    symbol=symbol,
                    model_name=model_name,
                    decision=decision,
                    assessment=assessment,
                    decision_db_id=decision_db_id,
                    results=results,
                    open_positions=open_positions,
                    claimed_symbol_keys=claimed_symbol_keys,
                    staged_entry_counts=staged_entry_counts,
                )
                if process_result.claimed_symbol:
                    claimed_analysis_symbols.append(process_result.claimed_symbol)
                    claimed_symbol_keys.add(
                        self._normalize_position_symbol(process_result.claimed_symbol)
                    )

            # 5. Store recent decisions/executions for dashboard
            self._set_loop_stage("publish_results")
            self._recent_decisions = results.get("decisions", [])[-20:]
            self._recent_executions = results.get("executions", [])[-20:]
            await self._finalize_round_unresolved_decisions(
                round_decision_ids,
                round_decisions,
                "本轮已经结束，但这条候选没有进入下单阶段，也没有拿到最终执行结果。"
                "系统已跳过本次旧信号，下一轮会用最新行情重新排序和评估。",
            )

            # 6. Push updates to dashboard
            await self._publish_dashboard_update(results)
            published_dashboard_update = True

        except asyncio.CancelledError:
            active_stage = self._runtime_state(analysis_scope).current_stage
            elapsed_seconds = round(self._round_elapsed_seconds(round_start), 3)
            error_text = (
                f"{analysis_scope} analysis task cancelled during {active_stage} "
                f"after {elapsed_seconds} seconds."
            )
            results["cancellation_diagnostic"] = {
                "scope": analysis_scope,
                "active_stage": active_stage,
                "elapsed_seconds": elapsed_seconds,
                "stage_durations": self._stage_durations_for_scope(analysis_scope),
            }
            self._set_loop_stage("task_cancelled", error_text)
            results["status"] = "error"
            results["error"] = error_text
            logger.error(
                "trading loop iteration cancelled",
                scope=analysis_scope,
                active_stage=active_stage,
                elapsed_seconds=elapsed_seconds,
            )
            await self._finalize_round_unresolved_decisions(
                round_decision_ids,
                round_decisions,
                "本轮分析/执行任务被外层超时保护取消；尚未进入 OKX 提交阶段的旧候选已写入终态，"
                "下一轮会用最新行情重新分析和排序。",
            )
            raise
        except Exception as e:
            error_text = safe_error_text(e, limit=180)
            self._set_loop_stage("error", error_text)
            await self._finalize_round_unresolved_decisions(
                round_decision_ids,
                round_decisions,
                f"本轮执行异常中断，未能完成最终状态回写：{safe_error_text(e, limit=120)}",
            )
            logger.error("trading loop iteration failed", error=error_text)
            results["status"] = "error"
            results["error"] = error_text

        finally:
            for symbol in claimed_analysis_symbols:
                await self._release_analysis_symbol(symbol)
            claimed_analysis_symbols.clear()
            round_duration = (datetime.now(UTC) - round_start).total_seconds()
            results["duration_ms"] = round(round_duration * 1000)
            finished_at = datetime.now(UTC)
            self._finish_runtime_round(
                analysis_scope,
                finished_at,
                ok=results.get("status") == "ok",
            )
            self._write_runtime_heartbeat()
            if not published_dashboard_update:
                try:
                    await self._publish_dashboard_update(results)
                except Exception as exc:
                    logger.debug(
                        "dashboard update publish failed during loop finalization",
                        error=safe_error_text(exc),
                    )
            _analysis_scope_context.reset(scope_token)
        return results

    @staticmethod
    def _should_run_full_reconciliation_at_round_start(analysis_scope: str) -> bool:
        return analysis_scope in {"full", "position"}

    @staticmethod
    def _should_refresh_position_prices_before_review(analysis_scope: str) -> bool:
        return analysis_scope in {"full", "position"}

    @staticmethod
    def _should_recover_pending_exits_for_scope(analysis_scope: str) -> bool:
        return analysis_scope in {"full", "position"}

    async def start(self) -> None:
        """Start the continuous trading loop."""
        await self.initialize()
        self._running = True
        self._start_time = datetime.now(UTC)
        self._start_ml_auto_train_loop()

        logger.info(
            "trading service started",
            mode=mode_manager.mode.value,
            scheduler="parallel_market_position",
            decision_interval_seconds=settings.decision_interval_seconds,
            market_loop_interval_seconds=self.market_loop_interval_seconds(),
            position_loop_interval_seconds=self.position_loop_interval_seconds(),
        )

        self._position_analysis_task = asyncio.create_task(
            self.position_review_service.loop(self.position_loop_interval_seconds)
        )
        self._market_analysis_task = asyncio.create_task(
            self.market_analysis_service.loop(self.market_loop_interval_seconds)
        )
        self._runtime_heartbeat_task = asyncio.create_task(self._runtime_heartbeat_loop())
        self._okx_authoritative_sync_task = asyncio.create_task(
            self._okx_authoritative_sync_loop()
        )
        self._okx_position_history_mirror_sync_task = asyncio.create_task(
            self._okx_position_history_mirror_sync_loop()
        )
        self._okx_position_settlement_sync_task = asyncio.create_task(
            self._okx_position_settlement_sync_loop()
        )
        try:
            await asyncio.gather(
                self._position_analysis_task,
                self._market_analysis_task,
                self._runtime_heartbeat_task,
                self._okx_authoritative_sync_task,
                self._okx_position_history_mirror_sync_task,
                self._okx_position_settlement_sync_task,
            )
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop the trading loop gracefully."""
        self._running = False
        self._write_runtime_heartbeat()
        await self._stop_ml_auto_train_loop()
        for task in (
            getattr(self, "_position_analysis_task", None),
            getattr(self, "_market_analysis_task", None),
            getattr(self, "_runtime_heartbeat_task", None),
            getattr(self, "_okx_authoritative_sync_task", None),
            getattr(self, "_okx_order_fact_sync_task", None),
            getattr(self, "_okx_position_history_mirror_sync_task", None),
            getattr(self, "_okx_position_settlement_sync_task", None),
            getattr(self, "_shadow_backtest_update_task", None),
            getattr(self, "_stale_entry_expire_task", None),
        ):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._position_analysis_task = None
        self._market_analysis_task = None
        self._runtime_heartbeat_task = None
        self._okx_authoritative_sync_task = None
        self._okx_order_fact_sync_task = None
        self._okx_position_history_mirror_sync_task = None
        self._okx_position_settlement_sync_task = None
        self._shadow_backtest_update_task = None
        self._stale_entry_expire_task = None
        if self.paper_executor:
            await self.paper_executor.shutdown()
        for okx in (self._okx_paper, self._okx_live):
            if okx:
                try:
                    await okx.shutdown()
                except Exception as exc:
                    logger.debug("OKX executor shutdown failed", error=safe_error_text(exc))
        await self.models.shutdown_all()
        logger.info("trading service stopped")

    def _start_ml_auto_train_loop(self) -> None:
        if self._ml_auto_train_task and not self._ml_auto_train_task.done():
            return
        try:
            recovered = self._model_training_state().recover_interrupted_runs()
            if recovered:
                logger.warning("recovered interrupted model training runs", model_ids=recovered)
        except Exception as exc:
            logger.warning(
                "model training state recovery failed; trading continues with training blocked",
                error=safe_error_text(exc, limit=180),
            )
        self._ml_auto_train_task = asyncio.create_task(self._ml_auto_train_loop())
        self._model_training_heartbeat_task = asyncio.create_task(
            self._model_training_heartbeat_loop()
        )

    async def _stop_ml_auto_train_loop(self) -> None:
        tasks = (
            getattr(self, "_ml_auto_train_task", None),
            getattr(self, "_model_training_heartbeat_task", None),
        )
        self._ml_auto_train_task = None
        self._model_training_heartbeat_task = None
        for task in tasks:
            if not task or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _model_training_heartbeat_loop(self) -> None:
        heartbeat_interval = min(max(AUTO_TRAIN_CHECK_INTERVAL_SECONDS / 6, 30.0), 60.0)
        while self._running:
            try:
                self._model_training_state().heartbeat(
                    scheduler_id="platform_model_training_loop",
                    model_ids=ALL_TRAINABLE_MODEL_IDS,
                    interval_seconds=AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
                )
            except Exception as exc:
                logger.warning(
                    "model training scheduler heartbeat write failed",
                    error=safe_error_text(exc, limit=180),
                )
            await asyncio.sleep(heartbeat_interval)

    async def _ml_auto_train_loop(self) -> None:
        """Retrain local ML and server-side quant tools without blocking trading."""
        while self._running:
            ml_result: dict[str, Any] = {}
            local_tools_result: dict[str, Any] = {}
            try:
                ml_result = await self.ml_signal_service.maybe_auto_train()
                if ml_result.get("trained"):
                    logger.info(
                        "local ML signal model auto-trained",
                        sample_count=ml_result.get("sample_count"),
                        new_sample_count=ml_result.get("new_sample_count"),
                    )
                elif ml_result.get("reason") not in {"not_due", "training_in_progress"}:
                    logger.warning(
                        "local ML signal auto-train skipped",
                        reason=ml_result.get("reason"),
                        error=ml_result.get("error"),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("local ML signal auto-train loop error", error=safe_error_text(e))
            try:
                local_tools_result = await self._maybe_train_local_ai_tools()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("local AI tools auto-train loop error", error=safe_error_text(e))
            failed_reasons = {"error", "load_samples_error", "timeout"}
            retry_due = any(
                str(result.get("reason") or "") in failed_reasons
                for result in (ml_result, local_tools_result)
            )
            await asyncio.sleep(
                AUTO_TRAIN_RETRY_INTERVAL_SECONDS
                if retry_due
                else AUTO_TRAIN_CHECK_INTERVAL_SECONDS
            )

    async def _maybe_train_local_ai_tools(self, *, force: bool = False) -> dict[str, Any]:
        state_store = self._model_training_state()
        lease_attempt = state_store.try_acquire_lease(
            scheduler_id="local_ai_tools_auto_train",
            stale_after_seconds=AUTO_TRAIN_LEASE_STALE_SECONDS,
        )
        if not lease_attempt.acquired or lease_attempt.lease is None:
            return {
                "trained": False,
                "reason": lease_attempt.reason,
                "recovered_stale_lease": lease_attempt.recovered_stale_lease,
            }
        lease = lease_attempt.lease
        self._local_tools_active_training_run_id = lease.run_id
        now = datetime.now(UTC)
        try:
            state_store.heartbeat(
                scheduler_id="local_ai_tools_auto_train",
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                interval_seconds=AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
            )
            state_store.record_check(
                scheduler_id="local_ai_tools_auto_train",
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                run_id=lease.run_id,
                force=force,
            )
        except Exception:
            self._local_tools_active_training_run_id = None
            lease.release()
            raise
        try:
            result = await self._maybe_train_local_ai_tools_process(force=force)
            failed = str(result.get("reason") or "") in {
                "error",
                "load_samples_error",
                "timeout",
            }
            delay = (
                AUTO_TRAIN_RETRY_INTERVAL_SECONDS
                if failed
                else AUTO_TRAIN_CHECK_INTERVAL_SECONDS
            )
            state_store.finish_check(
                scheduler_id="local_ai_tools_auto_train",
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                run_id=lease.run_id,
                result=result,
                next_check_at=datetime.now(UTC) + timedelta(seconds=delay),
            )
            return result
        except asyncio.CancelledError:
            state_store.record_exception(
                scheduler_id="local_ai_tools_auto_train",
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                run_id=lease.run_id,
                error="training_cancelled",
                next_check_at=now + timedelta(seconds=AUTO_TRAIN_RETRY_INTERVAL_SECONDS),
            )
            raise
        except Exception as exc:
            error = safe_error_text(exc, limit=180)
            state_store.record_exception(
                scheduler_id="local_ai_tools_auto_train",
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                run_id=lease.run_id,
                error=error,
                next_check_at=now + timedelta(seconds=AUTO_TRAIN_RETRY_INTERVAL_SECONDS),
            )
            raise
        finally:
            self._local_tools_active_training_run_id = None
            lease.release()

    async def _maybe_train_local_ai_tools_process(
        self, *, force: bool = False
    ) -> dict[str, Any]:
        """Push fresh history to the server-side profit/time-series/exit models."""
        if not self.local_ai_tools.enabled():
            return {"trained": False, "reason": "disabled"}

        from services.okx_training_gate import okx_training_refresh_gate

        okx_gate = okx_training_refresh_gate()
        if not bool(okx_gate.get("allowed")):
            return {
                "trained": False,
                "reason": okx_gate.get("reason") or "okx_training_refresh_blocked",
                "okx_daily_reconciliation_gate": okx_gate,
                "trade_sample_cursor_policy": "clean_training_view_only",
            }

        status_probe_error = ""
        try:
            status = await self.local_ai_tools.status()
        except Exception as exc:
            status_probe_error = safe_error_text(exc, limit=180)
            status = {
                "available": False,
                "service_available": False,
                "model_bundle_available": False,
                "status": "status_probe_error",
                "error": status_probe_error,
            }
            logger.warning(
                "local AI tools status probe failed before training; continuing with local cursors",
                error=status_probe_error,
            )
        server_shadow_count = int((status or {}).get("shadow_sample_count") or 0)
        server_trade_count = int((status or {}).get("trade_sample_count") or 0)
        completed_shadow_total = await self._completed_shadow_backtest_total()
        previous_completed_shadow_total = int(
            (status or {}).get("last_trained_completed_shadow_sample_count")
            or (status or {}).get("completed_shadow_sample_count")
            or self._local_tools_last_completed_shadow_count
            or server_shadow_count
            or 0
        )

        if not force:
            from scripts.train_local_ai_tools_models import _completed_trade_sample_count

            completed_trade_total = await _completed_trade_sample_count()
            previous_completed_trade_total = int(
                (status or {}).get("last_trained_completed_trade_sample_count")
                or (status or {}).get("completed_trade_sample_count")
                or server_trade_count
                or 0
            )
            new_shadow = max(completed_shadow_total - previous_completed_shadow_total, 0)
            new_trade = max(completed_trade_total - previous_completed_trade_total, 0)
            shadow_training_view_rebased = (
                completed_shadow_total < previous_completed_shadow_total
            )
            learning_only = not bool(
                (status or {}).get("model_bundle_available", (status or {}).get("available"))
            )
            training_policy = {
                "learning_only": learning_only,
                "trigger": "new_clean_cost_complete_sample_or_training_view_rebase",
                "distribution_requirement": "non_empty_train_and_holdout",
                "training_window_policy": "all_current_clean_cost_complete_samples",
                "cursor_source": "last_trained_completed_shadow_sample_count",
                "trade_cursor_source": "last_trained_completed_trade_sample_count",
                "trade_cursor_policy": "clean_training_view_only",
                "process_boundary": "dedicated_training_subprocess",
                "shadow_training_view_rebased": shadow_training_view_rebased,
            }
            if status_probe_error:
                training_policy["status_probe_error"] = status_probe_error
                training_policy["status_probe_fallback"] = "train_when_due_from_local_counts"
            if not (
                learning_only
                or shadow_training_view_rebased
                or new_shadow > 0
                or new_trade > 0
            ):
                return {
                    "trained": False,
                    "reason": "not_due",
                    "server_shadow_sample_count": server_shadow_count,
                    "completed_shadow_sample_count": completed_shadow_total,
                    "last_trained_completed_shadow_sample_count": (
                        previous_completed_shadow_total
                    ),
                    "completed_trade_sample_count": completed_trade_total,
                    "last_trained_completed_trade_sample_count": previous_completed_trade_total,
                    "new_shadow_sample_count": new_shadow,
                    "new_trade_sample_count": new_trade,
                    "training_policy": training_policy,
                }

            active_run_id = getattr(self, "_local_tools_active_training_run_id", None)
            if active_run_id:
                self._model_training_state().start_run(
                    scheduler_id="local_ai_tools_auto_train",
                    model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                    run_id=active_run_id,
                    trigger_reason="training_due",
                    sample_cursor={
                        "shadow": completed_shadow_total,
                        "trade": completed_trade_total,
                    },
                    timeout_seconds=AUTO_TRAIN_LEASE_STALE_SECONDS,
                )

            result = await self._run_local_ai_tools_training_subprocess()
            reported_shadow_total = result.get(
                "last_trained_completed_shadow_sample_count"
            )
            if reported_shadow_total is None:
                reported_shadow_total = result.get("completed_shadow_sample_count")
            reported_trade_total = result.get(
                "last_trained_completed_trade_sample_count"
            )
            if reported_trade_total is None:
                reported_trade_total = result.get("completed_trade_sample_count")
            authoritative_shadow_total = self._safe_int(
                reported_shadow_total,
                completed_shadow_total,
            )
            authoritative_trade_total = self._safe_int(
                reported_trade_total,
                completed_trade_total,
            )
            result["completed_shadow_sample_count"] = authoritative_shadow_total
            result["completed_trade_sample_count"] = authoritative_trade_total
            result["new_shadow_sample_count"] = new_shadow
            result["new_trade_sample_count"] = new_trade
            result["training_policy"] = training_policy
            result["training_process_isolated"] = True
            if result.get("trained"):
                result["last_trained_completed_shadow_sample_count"] = (
                    authoritative_shadow_total
                )
                result["last_trained_completed_trade_sample_count"] = (
                    authoritative_trade_total
                )
                self._local_tools_last_completed_shadow_count = authoritative_shadow_total
            return result

        result = await self._run_local_ai_tools_training_subprocess()
        result["training_process_isolated"] = True
        result["training_policy"] = {
            "trigger": "forced",
            "process_boundary": "dedicated_training_subprocess",
            "concurrency_policy": "exclusive_local_ai_tools_training_process_lock",
            "training_window_policy": "all_current_clean_samples",
        }
        if status_probe_error:
            result["training_policy"]["status_probe_error"] = status_probe_error
            result["training_policy"]["status_probe_fallback"] = (
                "train_in_isolated_process"
            )
        return result

    async def _run_local_ai_tools_training_subprocess(self) -> dict[str, Any]:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_local_ai_tools_models.py"),
            "--training-mode",
            "shadow",
            "--model-stage",
            "shadow",
            "--persist-artifact",
            "--confirm-phase3-rebuild",
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=AUTO_TRAIN_LEASE_STALE_SECONDS,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return {
                "trained": False,
                "reason": "timeout",
                "error": "isolated local AI tools training exceeded its scheduler lease",
            }
        if process.returncode != 0:
            return {
                "trained": False,
                "reason": "error",
                "error": safe_error_text(stderr.decode("utf-8", errors="replace"), limit=180),
            }
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {
                "trained": False,
                "reason": "invalid_training_response",
                "error": safe_error_text(exc, limit=180),
            }
        return dict(payload) if isinstance(payload, dict) else {
            "trained": False,
            "reason": "invalid_training_response",
        }

    async def _completed_shadow_backtest_total(self) -> int:
        async with get_session_ctx() as session:
            result = await session.execute(
                select(func.count(ShadowBacktest.id)).where(
                    ShadowBacktest.status == "completed",
                    ShadowBacktest.long_return_pct.is_not(None),
                    ShadowBacktest.short_return_pct.is_not(None),
                )
            )
            return int(result.scalar() or 0)

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except Exception:
            return None

    async def _recover_pending_exit_decisions(
        self,
        results: dict[str, Any],
        open_positions: list[dict],
        round_decision_ids: set[int],
    ) -> None:
        await self.pending_exit_recovery_processor.recover(
            results=results,
            open_positions=open_positions,
            round_decision_ids=round_decision_ids,
        )

    def _entry_side_value(self, decision: DecisionOutput) -> str:
        if decision.action == Action.LONG:
            return "long"
        if decision.action == Action.SHORT:
            return "short"
        return "hold"

    @staticmethod
    def _is_policy_skipped_execution_result(result: ExecutionResult | None) -> bool:
        return TradingService._policy_execution_terminal_status(result) is not None

    @staticmethod
    def _policy_execution_terminal_status(result: ExecutionResult | None) -> str | None:
        raw_response = result.raw_response if result is not None else None
        if not isinstance(raw_response, dict):
            return None
        stage_status = str(raw_response.get("stage_status") or "").lower()
        if stage_status in {"skipped", "blocked"}:
            return stage_status
        if raw_response.get("execution_skipped"):
            return "skipped"
        if raw_response.get("execution_policy_terminal") or raw_response.get("policy_blocker"):
            return "blocked"
        return None

    async def _execute_candidate(
        self,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        assessment,
        decision_db_id: int | None,
        results: dict[str, Any],
        open_positions: list[dict] | None = None,
        refresh_exit_positions: bool = True,
    ) -> ExecutionResult | None:
        model_mode = self._get_model_execution_mode(model_name)
        await self._record_strategy_learning_event(
            mode=model_mode,
            model_name=model_name,
            symbol=symbol,
            decision=decision,
            event_type="execution_attempt",
            event_status="pending",
            reason="decision entered execution pipeline",
            decision_id=decision_db_id,
            attribution={
                "source": "execute_candidate",
                "refresh_exit_positions": refresh_exit_positions,
            },
        )
        result = await self.execution_service.execute_candidate(
            symbol,
            model_name,
            decision,
            assessment,
            decision_db_id,
            results,
            open_positions=open_positions,
            refresh_exit_positions=refresh_exit_positions,
        )
        status = result.status.value if result is not None else "missing_result"
        exchange_confirmed = self._is_exchange_confirmed_execution(result)
        policy_terminal_status = self._policy_execution_terminal_status(result)
        event_status = "executed" if exchange_confirmed else "rejected"
        severity = "info" if exchange_confirmed else "warn"
        if policy_terminal_status:
            event_status = policy_terminal_status
            severity = "info" if policy_terminal_status == "skipped" else "warn"
        elif result is None:
            event_status = "failed"
            severity = "error"
        execution_links = await self._strategy_learning_execution_links(
            mode=model_mode,
            model_name=model_name,
            symbol=symbol,
            decision=decision,
            decision_id=decision_db_id,
            result=result,
        )
        await self._record_strategy_learning_event(
            mode=model_mode,
            model_name=model_name,
            symbol=symbol,
            decision=decision,
            event_type="execution_result",
            event_status=event_status,
            reason=(
                self.execution_reason_from_result(result) if result else "missing execution result"
            ),
            severity=severity,
            decision_id=decision_db_id,
            order_id=self._safe_int(execution_links.get("local_order_id"), 0) or None,
            position_id=self._safe_int(execution_links.get("local_position_id"), 0) or None,
            attribution={
                "source": "execute_candidate",
                "status": status,
                **execution_links,
                "exchange_confirmed": exchange_confirmed,
                "execution_skipped": policy_terminal_status == "skipped",
                "policy_terminal_status": policy_terminal_status,
                "skip_kind": (
                    result.raw_response.get("skip_kind")
                    if result is not None and isinstance(result.raw_response, dict)
                    else None
                ),
            },
        )
        return result

    async def _execute_candidate_locked(
        self,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        assessment,
        decision_db_id: int | None,
        results: dict[str, Any],
        open_positions: list[dict] | None = None,
        refresh_exit_positions: bool = True,
    ) -> ExecutionResult | None:
        model_mode = self._get_model_execution_mode(model_name)
        await self._record_strategy_learning_event(
            mode=model_mode,
            model_name=model_name,
            symbol=symbol,
            decision=decision,
            event_type="execution_attempt",
            event_status="pending",
            reason="decision entered locked execution pipeline",
            decision_id=decision_db_id,
            attribution={"source": "execute_candidate_locked"},
        )
        result = await self.execution_service.execute_candidate_locked(
            symbol,
            model_name,
            decision,
            assessment,
            decision_db_id,
            results,
            open_positions=open_positions,
            refresh_exit_positions=refresh_exit_positions,
        )
        exchange_confirmed = self._is_exchange_confirmed_execution(result)
        policy_terminal_status = self._policy_execution_terminal_status(result)
        event_status = "executed" if exchange_confirmed else "rejected"
        severity = "info" if exchange_confirmed else "warn"
        if policy_terminal_status:
            event_status = policy_terminal_status
            severity = "info" if policy_terminal_status == "skipped" else "warn"
        elif result is None:
            event_status = "failed"
            severity = "error"
        execution_links = await self._strategy_learning_execution_links(
            mode=model_mode,
            model_name=model_name,
            symbol=symbol,
            decision=decision,
            decision_id=decision_db_id,
            result=result,
        )
        await self._record_strategy_learning_event(
            mode=model_mode,
            model_name=model_name,
            symbol=symbol,
            decision=decision,
            event_type="execution_result",
            event_status=event_status,
            reason=(
                self.execution_reason_from_result(result) if result else "missing execution result"
            ),
            severity=severity,
            decision_id=decision_db_id,
            order_id=self._safe_int(execution_links.get("local_order_id"), 0) or None,
            position_id=self._safe_int(execution_links.get("local_position_id"), 0) or None,
            attribution={
                "source": "execute_candidate_locked",
                "status": result.status.value if result is not None else "missing_result",
                **execution_links,
                "exchange_confirmed": exchange_confirmed,
                "execution_skipped": policy_terminal_status == "skipped",
                "policy_terminal_status": policy_terminal_status,
                "skip_kind": (
                    result.raw_response.get("skip_kind")
                    if result is not None and isinstance(result.raw_response, dict)
                    else None
                ),
            },
        )
        return result

    def _is_no_exchange_position_error(self, message: Any) -> bool:
        return self.execution_result_classifier.is_no_exchange_position_error(message)

    def _result_has_no_exchange_position(self, result: ExecutionResult | None) -> bool:
        return self.execution_result_classifier.result_has_no_exchange_position(result)

    def _apply_execution_to_open_positions(
        self,
        open_positions: list[dict],
        model_name: str,
        decision: DecisionOutput,
        execution_result: ExecutionResult,
    ) -> None:
        self.open_positions_execution_applier.apply(
            open_positions,
            model_name,
            decision,
            execution_result,
        )

    async def manual_trade(self, symbol: str, model_name: str | None = None) -> dict[str, Any]:
        """Execute a one-shot AI analysis and trade for a specific symbol.

        Returns a dict with decision, execution result, and any rejection reason.
        """
        result: dict[str, Any] = {
            "symbol": symbol,
            "model": model_name,
            "decision": None,
            "execution": None,
            "approved": False,
            "rejection_reason": None,
        }

        try:
            fv = await self.data_service.get_feature_vector(symbol)
        except Exception as e:
            result["rejection_reason"] = f"Failed to get feature vector: {e}"
            return result

        open_positions = await self.okx_sync_service.get_open_positions_context()

        # Manual trades also use the unified multi-model ensemble.
        memory_context = await self._memory_context_with_vector_feedback(symbol)
        ml_signal_context = self.ml_signal_service.predict(fv)
        local_ai_tools_context = await self._local_ai_tools_context(
            fv,
            ml_signal_context,
            open_positions=open_positions,
            include_exit_advice=False,
        )
        market_regime_context = self._market_regime_context({symbol: fv})
        strategy_mode_context = await self._strategy_mode_context(
            self._get_model_execution_mode(ENSEMBLE_TRADER_NAME),
            market_regime_context,
            open_positions,
        )
        analysis_started = datetime.now(UTC)
        decision, _opinions = await self.ensemble.decide(
            fv,
            {
                "open_positions": open_positions,
                "trading_mode": mode_manager.mode.value,
                "manual_override": True,
                **memory_context,
                "market_regime": market_regime_context,
                "strategy_mode": strategy_mode_context,
                "ml_signal": ml_signal_context,
                "local_ai_tools": local_ai_tools_context,
            },
        )
        self._attach_decision_timing(decision, analysis_started, "manual")
        result["model"] = ENSEMBLE_TRADER_NAME
        result_model = str(result["model"])

        if decision is None:
            result["rejection_reason"] = "No actionable decision produced"
            return result

        result["decision"] = {
            "model_name": decision.model_name,
            "symbol": decision.symbol,
            "action": (
                decision.action.value if hasattr(decision.action, "value") else str(decision.action)
            ),
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "position_size_pct": decision.position_size_pct,
        }

        if decision.is_entry:
            self._candidate_opportunity_score(decision, strategy_mode_context)
            try:
                await self._prepare_entry_for_hard_risk(
                    decision,
                    self._get_model_execution_mode(result_model),
                    open_positions,
                )
                result["decision"].update(
                    {
                        "position_size_pct": decision.position_size_pct,
                        "suggested_leverage": decision.suggested_leverage,
                        "stop_loss_pct": decision.stop_loss_pct,
                        "take_profit_pct": decision.take_profit_pct,
                    }
                )
            except Exception as exc:
                result["rejection_reason"] = (
                    "动态费后收益风险预算生成失败，本次手动 entry 失败关闭："
                    f"{safe_error_text(exc, limit=160)}"
                )
                return result

        # Risk assessment: only this model's positions.
        assessment = await self.manual_trade_risk_assessment.assess(
            decision=decision,
            model_name=result_model,
            open_positions=open_positions,
            feature_vector=fv,
            account_balance_provider=self.get_account_balance,
        )

        if not assessment.approved:
            result["rejection_reason"] = assessment.rejection_reason
            return result

        execution_update = await self._manual_trade_execution_processor().execute(
            symbol=symbol,
            model_name=result_model,
            original_decision=decision,
            assessment=assessment,
            open_positions=open_positions,
        )
        result.update(execution_update)
        return result

    async def manual_close_position(
        self,
        position_id: int,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Close one open position directly through OKX without AI/risk persistence."""
        async with get_session_ctx() as session:
            position = await session.get(Position, int(position_id))
            if position is None:
                return {
                    "approved": False,
                    "position_id": int(position_id),
                    "rejection_reason": "持仓记录不存在。",
                }
            if not position.is_open:
                return {
                    "approved": False,
                    "position_id": position.id,
                    "symbol": position.symbol,
                    "side": position.side,
                    "rejection_reason": "该持仓已经平仓。",
                }
            payload = self._manual_close_position_payload(position)

        return await self._execute_direct_manual_close_payload(payload, reason=reason)

    async def manual_close_all_positions(
        self,
        *,
        mode: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Close every open position in the selected execution mode."""
        selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            rows = await repo.get_position_records(
                execution_mode=selected_mode,
                limit=5000,
                offset=0,
                is_open=True,
            )
            payloads = [self._manual_close_position_payload(position) for position in rows]

        results = []
        for payload in payloads:
            results.append(await self._execute_direct_manual_close_payload(payload, reason=reason))
        closed = sum(1 for item in results if item.get("approved") and item.get("closed"))
        failed = len(results) - closed
        return {
            "approved": failed == 0,
            "mode": selected_mode,
            "requested": len(results),
            "closed": closed,
            "failed": failed,
            "results": results,
        }

    @staticmethod
    def _manual_close_position_payload(position: Position) -> dict[str, Any]:
        return {
            "id": position.id,
            "model_name": position.model_name,
            "mode": position.execution_mode,
            "symbol": position.symbol,
            "side": position.side,
            "quantity": position.quantity,
            "entry_price": position.entry_price,
            "current_price": position.current_price,
            "leverage": position.leverage,
            "unrealized_pnl": position.unrealized_pnl,
        }

    @staticmethod
    def _manual_close_exchange_order_id(result: ExecutionResult) -> str:
        raw_order_id = str(result.exchange_order_id or result.order_id or "").strip()
        if raw_order_id.startswith("manual_close:"):
            return raw_order_id
        return f"manual_close:{raw_order_id or 'unknown'}"

    @staticmethod
    def _manual_close_order_side(result: ExecutionResult, action: Action) -> str:
        raw_side = str(result.side or "").lower().strip()
        if raw_side in {"buy", "sell"}:
            return raw_side
        return "sell" if action == Action.CLOSE_LONG else "buy"

    def _manual_close_exchange_base_quantity(self, exchange_position: dict[str, Any]) -> float:
        info = self._safe_dict(exchange_position.get("info"))
        quantity = self._safe_float(exchange_position.get("quantity"), 0.0)
        if quantity > 0:
            return abs(quantity)
        contracts = self._safe_float(
            exchange_position.get("contracts")
            or exchange_position.get("size")
            or exchange_position.get("positionAmt")
            or info.get("pos")
            or info.get("qty"),
            0.0,
        )
        contract_size = self._safe_float(
            exchange_position.get("contractSize")
            or exchange_position.get("contract_size")
            or info.get("ctVal"),
            1.0,
        )
        return abs(contracts * (contract_size if contract_size > 0 else 1.0))

    async def _manual_close_fraction_for_position(
        self,
        executor: OKXExecutor,
        position: dict[str, Any],
    ) -> float:
        local_qty = self._safe_float(position.get("quantity"), 0.0)
        if local_qty <= 0:
            return 1.0
        side = str(position.get("side") or "").lower()
        symbol = str(position.get("symbol") or "")
        try:
            exchange_positions = await asyncio.wait_for(
                executor.get_positions_strict(symbol),
                timeout=15.0,
            )
        except Exception as exc:
            logger.warning(
                "manual close could not fetch exchange position size; closing full matching side",
                symbol=symbol,
                side=side,
                error=safe_error_text(exc),
            )
            return 1.0
        exchange_qty = sum(
            self._manual_close_exchange_base_quantity(item)
            for item in exchange_positions or []
            if str(item.get("side") or "").lower() == side
        )
        if exchange_qty <= 0:
            return 1.0
        return min(max(local_qty / exchange_qty, 1e-9), 1.0)

    async def _persist_manual_close_result(
        self,
        position_payload: dict[str, Any],
        result: ExecutionResult,
        action: Action,
        *,
        model_name: str,
        execution_mode: str,
    ) -> dict[str, Any]:
        if result.quantity <= 0 or result.price <= 0:
            return {"closed": False, "realized_pnl": 0.0, "close_quantity": 0.0}

        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            position = await session.get(Position, int(position_payload.get("id") or 0))
            if position is None or not position.is_open:
                return {"closed": False, "realized_pnl": 0.0, "close_quantity": 0.0}

            position_qty = self._safe_float(position.quantity, 0.0)
            close_qty = min(self._safe_float(result.quantity, 0.0), position_qty)
            if close_qty <= 0 or position_qty <= 0:
                return {"closed": False, "realized_pnl": 0.0, "close_quantity": 0.0}

            close_fee = self.entry_fee_provider.proportional_fee(
                result.fee,
                close_qty,
                result.quantity,
            )
            entry_fee = await self.entry_fee_provider.entry_fee_for_position(
                session,
                position,
                close_qty,
            )
            from services.okx_realized_pnl import gross_pnl_with_okx_override

            gross_pnl, gross_pnl_source = gross_pnl_with_okx_override(
                side=str(position.side or "").lower(),
                entry_price=position.entry_price,
                exit_price=result.price,
                close_qty=close_qty,
                okx_payload=getattr(result, "raw_response", None),
                okx_total_qty=result.quantity,
            )
            raw_funding_fee, funding_fee_source = funding_fee_from_payload(
                getattr(result, "raw_response", None)
            )
            funding_fee = proportional_signed_value(
                raw_funding_fee,
                close_qty,
                result.quantity,
            )
            settlement = build_position_settlement_snapshot(
                close_fill_pnl=gross_pnl,
                entry_fee=entry_fee,
                close_fee=close_fee,
                funding_fee=funding_fee,
                status=SETTLEMENT_STATUS_SETTLING,
                source="manual_close_execution",
                synced_at=result.timestamp,
                raw={
                    "gross_pnl_source": gross_pnl_source,
                    "funding_fee_source": funding_fee_source,
                    "close_exchange_order_id": self._manual_close_exchange_order_id(result),
                    "close_quantity": close_qty,
                    "result_quantity": result.quantity,
                },
            )
            realized_pnl = settlement.realized_pnl
            tolerance = max(position_qty * 1e-9, 1e-8)
            closes_position = position_qty - close_qty <= tolerance

            order = await repo.create_order(
                {
                    "model_name": model_name,
                    "execution_mode": execution_mode,
                    "symbol": result.symbol or position.symbol,
                    "side": self._manual_close_order_side(result, action),
                    "order_type": result.order_type or "market",
                    "quantity": close_qty,
                    "price": result.price,
                    "status": result.status.value,
                    "fee": close_fee,
                    "decision_id": None,
                    "exchange_order_id": self._manual_close_exchange_order_id(result),
                    "filled_at": result.timestamp,
                }
            )

            if closes_position:
                position.is_open = False
                position.current_price = result.price
                position.unrealized_pnl = 0.0
                apply_position_settlement_snapshot(position, settlement)
                position.closed_at = result.timestamp
                try:
                    self.position_profit_peaks.remove(model_name, position.symbol, position.side)
                except Exception as exc:
                    logger.debug(
                        "manual close failed to remove position peak",
                        error=safe_error_text(exc),
                    )
            else:
                position.quantity = position_qty - close_qty
                position.current_price = result.price
                if str(position.side or "").lower() == "short":
                    position.unrealized_pnl = (
                        position.entry_price - result.price
                    ) * position.quantity
                else:
                    position.unrealized_pnl = (
                        result.price - position.entry_price
                    ) * position.quantity
                await repo.open_position(
                    {
                        "model_name": model_name,
                        "execution_mode": execution_mode,
                        "symbol": position.symbol,
                        "side": position.side,
                        "quantity": close_qty,
                        "entry_price": position.entry_price,
                        "current_price": result.price,
                        "leverage": position.leverage,
                        "unrealized_pnl": 0.0,
                        "stop_loss_price": position.stop_loss_price,
                        "take_profit_price": position.take_profit_price,
                        "is_open": False,
                        "closed_at": result.timestamp,
                        "created_at": position.created_at,
                        **settlement_payload_fields(settlement),
                    }
                )
            await session.flush()

        result.pnl = realized_pnl
        try:
            await self.account_accounting_service.persist_account_update(
                model_name,
                model_name,
                result,
            )
        except Exception as exc:
            logger.warning(
                "manual close failed to persist account update",
                error=safe_error_text(exc),
            )
        self._invalidate_okx_balance_snapshot_cache_for_model(model_name)
        try:
            self.increment_trade_count()
        except Exception as exc:
            logger.debug("manual close trade counter update failed", error=safe_error_text(exc))
        return {
            "closed": closes_position,
            "realized_pnl": realized_pnl,
            "close_quantity": close_qty,
            "order_id": int(order.id) if getattr(order, "id", None) is not None else None,
            "position_id": int(position.id) if getattr(position, "id", None) is not None else None,
        }

    async def _execute_direct_manual_close_payload(
        self,
        position: dict[str, Any],
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        side = str(position.get("side") or "").lower()
        action = (
            Action.CLOSE_LONG if side == "long" else Action.CLOSE_SHORT if side == "short" else None
        )
        if action is None:
            await self._record_strategy_learning_event(
                mode=str(position.get("mode") or mode_manager.mode.value),
                model_name=str(position.get("model_name") or ENSEMBLE_TRADER_NAME),
                symbol=str(position.get("symbol") or ""),
                action="manual_close",
                event_type="manual_close",
                event_status="rejected",
                reason="invalid position side for manual close",
                severity="warn",
                position_id=self._safe_int(position.get("id"), 0) or None,
                attribution={"source": "manual_close", "manual": True},
                exclude_from_training=True,
            )
            return {
                "approved": False,
                "position_id": position.get("id"),
                "symbol": position.get("symbol"),
                "side": side,
                "rejection_reason": "持仓方向无效，无法手动平仓。",
            }

        symbol = str(position.get("symbol") or "")
        model_name = str(position.get("model_name") or ENSEMBLE_TRADER_NAME)
        execution_mode = "live" if str(position.get("mode") or "").lower() == "live" else "paper"
        current_price = self._safe_float(position.get("current_price"), 0.0) or self._safe_float(
            position.get("entry_price"),
            0.0,
        )
        executor = await self.get_okx_executor_for_mode(execution_mode)
        close_fraction = await self._manual_close_fraction_for_position(executor, position)
        reasoning = str(reason or "用户手动平仓")
        decision = DecisionOutput(
            model_name=model_name,
            symbol=symbol,
            action=action,
            confidence=1.0,
            reasoning=reasoning,
            position_size_pct=close_fraction,
            suggested_leverage=self._safe_float(position.get("leverage"), 1.0),
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            raw_response={
                "manual_close": True,
                "manual_close_position_id": position.get("id"),
                "manual_close_reason": reasoning,
                "close_fraction": close_fraction,
                "exclude_from_training": True,
                "strategy_learning_context": {
                    "manual_close": True,
                    "exclude_from_training": True,
                    "scheduler_reason": "user manual close bypasses AI risk and training",
                },
            },
            feature_snapshot={
                "current_price": current_price,
                "entry_price": self._safe_float(position.get("entry_price"), 0.0),
                "position_current_price": current_price,
                "quantity": self._safe_float(position.get("quantity"), 0.0),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        try:
            async with self._execution_lock:
                execution_result = await asyncio.wait_for(
                    executor.place_order(decision, account_id=model_name),
                    timeout=90.0,
                )
        except TimeoutError:
            await self._record_strategy_learning_event(
                mode=execution_mode,
                model_name=model_name,
                symbol=symbol,
                decision=decision,
                event_type="manual_close",
                event_status="failed",
                reason="manual close OKX timeout",
                severity="error",
                position_id=self._safe_int(position.get("id"), 0) or None,
                attribution={"source": "manual_close", "manual": True, "error": "timeout"},
                exclude_from_training=True,
            )
            return {
                "approved": False,
                "position_id": position.get("id"),
                "symbol": symbol,
                "side": side,
                "rejection_reason": "OKX 手动平仓接口超时，未确认成交。请刷新持仓后再确认。",
            }
        except Exception as exc:
            await self._record_strategy_learning_event(
                mode=execution_mode,
                model_name=model_name,
                symbol=symbol,
                decision=decision,
                event_type="manual_close",
                event_status="failed",
                reason=f"manual close OKX failed: {safe_error_text(exc)}",
                severity="error",
                position_id=self._safe_int(position.get("id"), 0) or None,
                attribution={
                    "source": "manual_close",
                    "manual": True,
                    "error": safe_error_text(exc),
                },
                exclude_from_training=True,
            )
            return {
                "approved": False,
                "position_id": position.get("id"),
                "symbol": symbol,
                "side": side,
                "rejection_reason": f"OKX 手动平仓失败：{safe_error_text(exc)}",
            }

        execution_completed = self._is_exchange_confirmed_execution(execution_result)
        if not execution_completed:
            await self._record_strategy_learning_event(
                mode=execution_mode,
                model_name=model_name,
                symbol=symbol,
                decision=decision,
                event_type="manual_close",
                event_status="rejected",
                reason=self.execution_reason_from_result(execution_result),
                severity="warn",
                position_id=self._safe_int(position.get("id"), 0) or None,
                attribution={
                    "source": "manual_close",
                    "manual": True,
                    "order_id": execution_result.order_id,
                    "status": execution_result.status.value,
                },
                exclude_from_training=True,
            )
            return {
                "approved": False,
                "position_id": position.get("id"),
                "symbol": symbol,
                "side": side,
                "rejection_reason": self.execution_reason_from_result(execution_result),
                "execution": {
                    "order_id": execution_result.order_id,
                    "status": execution_result.status.value,
                    "quantity": execution_result.quantity,
                    "price": execution_result.price,
                    "pnl": execution_result.pnl,
                },
            }

        persisted = await self._persist_manual_close_result(
            position,
            execution_result,
            action,
            model_name=model_name,
            execution_mode=execution_mode,
        )
        execution_result.pnl = self._safe_float(persisted.get("realized_pnl"), execution_result.pnl)
        await self._record_strategy_learning_event(
            mode=execution_mode,
            model_name=model_name,
            symbol=symbol,
            decision=decision,
            event_type="manual_close",
            event_status="executed",
            reason=reasoning,
            severity="info",
            order_id=self._safe_int(persisted.get("order_id"), 0) or None,
            position_id=self._safe_int(persisted.get("position_id"), 0)
            or self._safe_int(position.get("id"), 0)
            or None,
            attribution={
                "source": "manual_close",
                "manual": True,
                "closed": bool(persisted.get("closed")),
                "realized_pnl": execution_result.pnl,
                "quantity": persisted.get("close_quantity", execution_result.quantity),
                "price": execution_result.price,
                "local_order_id": persisted.get("order_id"),
                "local_position_id": persisted.get("position_id"),
                "exchange_order_id": self._manual_close_exchange_order_id(execution_result),
            },
            exclude_from_training=True,
        )
        return {
            "approved": True,
            "closed": bool(persisted.get("closed")),
            "position_id": position.get("id"),
            "symbol": symbol,
            "side": side,
            "manual_close": True,
            "exclude_from_training": True,
            "execution": {
                "order_id": execution_result.order_id,
                "exchange_order_id": self._manual_close_exchange_order_id(execution_result),
                "status": execution_result.status.value,
                "quantity": persisted.get("close_quantity", execution_result.quantity),
                "price": execution_result.price,
                "pnl": execution_result.pnl,
            },
        }

    async def _execute_manual_close_payload(
        self,
        position: dict[str, Any],
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return await self._execute_direct_manual_close_payload(position, reason=reason)

    async def _refresh_db_position_prices(self, feature_vectors: dict) -> None:
        return await self.okx_sync_service.refresh_position_prices(feature_vectors)

    async def reconcile_exchange_positions(self) -> list[dict]:
        return await self.okx_sync_service.reconcile_exchange_positions()

    async def _log_exchange_sync_close_decision(
        self,
        session,
        pos,
        exit_price: float,
        realized_pnl: float,
        closed_at: datetime,
        reason: str,
        close_fill: dict[str, Any] | None = None,
        position_size_pct: float | None = None,
        reconcile_origin: str | None = None,
    ) -> int | None:
        """Record a synthetic close decision for exchange-side position closes."""
        try:
            side = str(pos.side or "").lower()
            action = Action.CLOSE_SHORT if side == "short" else Action.CLOSE_LONG
            close_fill_safe = {
                key: (value.isoformat() if isinstance(value, datetime) else value)
                for key, value in (close_fill or {}).items()
            }
            close_fraction = self._safe_float(position_size_pct, 1.0)
            close_fraction = min(max(close_fraction, 0.0), 1.0) or 1.0
            repo = DecisionRepository(session)
            record = await repo.log_decision(
                {
                    "model_name": pos.model_name or ENSEMBLE_TRADER_NAME,
                    "symbol": pos.symbol,
                    "action": action.value,
                    "confidence": 1.0,
                    "reasoning": sanitize_text(reason),
                    "position_size_pct": close_fraction,
                    "suggested_leverage": pos.leverage or 1.0,
                    "stop_loss_pct": 0.0,
                    "take_profit_pct": 0.0,
                    "feature_snapshot": {
                        "source": "okx_position_reconcile",
                        "position_id": pos.id,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "quantity": pos.quantity,
                        "side": pos.side,
                        "realized_pnl": realized_pnl,
                        "reconcile_origin": reconcile_origin or "external_okx_sync",
                    },
                    "raw_llm_response": {
                        "system_sync": True,
                        "source": "okx_position_reconcile",
                        "close_fill": close_fill_safe,
                        "reconcile_origin": reconcile_origin or "external_okx_sync",
                    },
                    "analysis_type": "position",
                    "is_paper": pos.execution_mode != "live",
                    "was_executed": True,
                    "executed_at": closed_at,
                    "execution_price": exit_price,
                    "execution_reason": None,
                    "outcome": (
                        "profit" if realized_pnl > 0 else "loss" if realized_pnl < 0 else "flat"
                    ),
                    "outcome_pnl_pct": (
                        realized_pnl / (pos.entry_price * pos.quantity) * 100
                        if pos.entry_price and pos.quantity
                        else 0.0
                    ),
                    "created_at": closed_at,
                }
            )
            await session.flush()
            return record.id
        except Exception as e:
            logger.warning(
                "failed to log exchange sync close decision",
                position_id=getattr(pos, "id", None),
                symbol=getattr(pos, "symbol", None),
                error=safe_error_text(e),
            )
            return None

    def _normalize_position_symbol(self, symbol: str | None) -> str:
        if not symbol:
            return ""
        normalized = str(symbol).split(":")[0]
        if normalized.endswith("-SWAP"):
            normalized = normalized[:-5]
        if "/" not in normalized and "-" in normalized:
            parts = normalized.split("-")
            if len(parts) >= 2:
                normalized = f"{parts[0]}/{parts[1]}"
        return normalized

    def _suspicious_new_symbol_reason(self, symbol: str | None) -> str | None:
        """Block exchange test/demo instruments from new-entry analysis and execution."""
        policy = getattr(self, "entry_suspicious_symbol", None)
        if policy is None:
            policy = EntrySuspiciousSymbolPolicy(self._normalize_position_symbol)
        return policy.reason(symbol)

    async def _enforce_sl_tp(
        self,
        feature_vectors: dict,
        *,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> list[dict]:
        """Execute the unified dynamic exit policy before slow position review."""
        auto_closes: list[dict[str, Any]] = []
        if open_positions is None:
            open_positions = await self.okx_sync_service.get_open_positions_context()
        if not open_positions:
            return auto_closes

        handled: set[tuple[str, str, str]] = set()
        for position in open_positions:
            if position.get("is_open", True) is False:
                continue
            model_name = str(position.get("model_name") or ENSEMBLE_TRADER_NAME)
            symbol = self._normalize_position_symbol(position.get("symbol"))
            side = str(position.get("side") or "").lower()
            key = (model_name, symbol, side)
            if not symbol or side not in {"long", "short"} or key in handled:
                continue
            handled.add(key)

            feature = feature_vectors.get(symbol) or feature_vectors.get(position.get("symbol"))
            position_price = self._safe_float(position.get("current_price"), 0.0)
            feature_price = self._safe_float(
                getattr(feature, "current_price", 0.0) if feature is not None else 0.0,
                0.0,
            )
            current_price = position_price or feature_price
            entry_price = self._safe_float(position.get("entry_price"), 0.0)
            if current_price <= 0.0 or entry_price <= 0.0:
                continue

            stop_loss = self._safe_float(position.get("stop_loss"), 0.0)
            take_profit = self._safe_float(position.get("take_profit"), 0.0)
            stop_crossed = bool(
                stop_loss
                and (
                    (side == "long" and current_price <= stop_loss)
                    or (side == "short" and current_price >= stop_loss)
                )
            )
            target_crossed = bool(
                take_profit
                and (
                    (side == "long" and current_price >= take_profit)
                    or (side == "short" and current_price <= take_profit)
                )
            )
            returns = [
                self._safe_float(
                    getattr(feature, name, 0.0) if feature is not None else 0.0,
                    0.0,
                )
                for name in ("returns_1", "returns_5", "returns_20")
            ]
            adverse_returns = [
                value
                for value in returns
                if (side == "long" and value < 0.0)
                or (side == "short" and value > 0.0)
            ]
            hold_minutes = self.position_time.position_age_minutes(
                position_open_time(position) or position.get("created_at")
            )
            current_unrealized = self._safe_float(position.get("unrealized_pnl"), 0.0)
            peak_state = self.position_profit_peaks.update(
                model_name=model_name,
                symbol=symbol,
                side=side,
                current_price=current_price,
                entry_price=entry_price,
                unrealized_pnl=current_unrealized,
                hold_minutes=hold_minutes,
                quantity=self._safe_float(position.get("quantity"), 0.0),
            )
            position_snapshot = dict(position)
            position_snapshot["symbol"] = symbol
            position_snapshot["side"] = side
            position_snapshot["current_price"] = current_price
            position_snapshot["peak_unrealized_pnl"] = self._safe_float(
                peak_state.get("peak_unrealized_pnl"), current_unrealized
            )

            close_action = Action.CLOSE_LONG if side == "long" else Action.CLOSE_SHORT
            trigger = (
                "stop_loss"
                if stop_crossed
                else "take_profit"
                if target_crossed
                else "dynamic_position_scan"
            )
            close_decision = DecisionOutput(
                model_name=model_name,
                symbol=symbol,
                action=close_action,
                confidence=0.0,
                reasoning="unified dynamic fee-after exit scan",
                position_size_pct=0.0,
                suggested_leverage=self._safe_float(position.get("leverage"), 1.0),
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                raw_response={
                    "fast_risk_trigger": trigger,
                    "forced_exit": bool(stop_crossed or target_crossed),
                    "close_evidence": {
                        "hard_risk": stop_crossed,
                        "continuation_deteriorated": bool(adverse_returns),
                        "peak_unrealized_pnl_usdt": position_snapshot[
                            "peak_unrealized_pnl"
                        ],
                    },
                },
                feature_snapshot=(
                    feature.to_dict()
                    if feature is not None and callable(getattr(feature, "to_dict", None))
                    else {"current_price": current_price}
                ),
            )
            assessment = apply_dynamic_exit(close_decision, [position_snapshot])
            if not assessment.eligible or assessment.close_fraction <= 0.0:
                continue

            close_fraction = assessment.close_fraction
            reason = assessment.reason
            logger.info(
                "dynamic position risk triggered",
                model=model_name,
                symbol=symbol,
                side=side,
                trigger=trigger,
                close_fraction=close_fraction,
                policy_version=assessment.policy_provenance.get("strategy_version"),
            )
            execution = await self._fast_risk_exit_execution_processor().execute(
                model_name=model_name,
                symbol=symbol,
                side=side,
                position=position,
                decision=close_decision,
                trigger=trigger,
                reason=reason,
                close_fraction=close_fraction,
                entry_price=entry_price,
                current_price=current_price,
            )
            if not execution.skipped and execution.auto_close is not None:
                auto_closes.append(execution.auto_close)

        return auto_closes

    async def _review_open_positions(
        self,
        open_positions: list[dict],
        feature_vectors: dict,
        results: dict[str, Any] | None = None,
        round_decision_ids: set[int] | None = None,
        position_entry_pause_reason: str | None = None,
        max_groups_override: int | None = None,
        round_deadline_monotonic: float | None = None,
    ) -> tuple[list[tuple], set[tuple[str, str]]]:
        """For each model+symbol with open positions, ask AI to review and possibly act.

        Returns execution candidates plus position keys already handled by review.
        Exit decisions are submitted immediately so vulnerable positions are not
        left waiting while the rest of the portfolio is still being reviewed.
        """
        candidates: list[tuple[str, str, DecisionOutput, Any, int | None]] = []
        handled_keys: set[tuple[str, str]] = set()

        grouped = self.position_review_grouping.group(open_positions)
        if not grouped:
            return candidates, handled_keys

        grouped_items = list(grouped.items())
        portfolio_profit_context = self._portfolio_profit_protection_context(open_positions)
        market_regime_context = self._market_regime_context(feature_vectors)
        strategy_mode_context = await self._strategy_mode_context(
            self._get_model_execution_mode(ENSEMBLE_TRADER_NAME),
            market_regime_context,
            open_positions,
        )
        fast_scan = self._scan_position_review_groups(
            grouped_items,
            feature_vectors,
            portfolio_profit_context,
            strategy_mode_context,
        )
        batch_selection = self._position_review_batch_policy().select(
            grouped_items,
            fast_scan,
            max_groups_override=max_groups_override,
            defer_count_provider=self.position_review_defer_tracker.count,
            position_entry_pause_reason=position_entry_pause_reason,
            cursor=self._position_review_cursor,
        )
        grouped_items = batch_selection.selected_items
        self._position_review_cursor = batch_selection.next_cursor
        if batch_selection.limited:
            self.position_review_defer_tracker.clear_many(batch_selection.selected_keys)
            self._decision_count += await self.position_review_fast_scan_recorder.record_many(
                skipped_items=batch_selection.skipped_items,
                fast_scan=fast_scan,
                feature_vectors=feature_vectors,
                portfolio_profit_context=portfolio_profit_context,
                results=results,
                round_decision_ids=round_decision_ids,
                position_entry_pause_reason=position_entry_pause_reason,
            )
            logger.info(
                "position review batched",
                selected=len(grouped_items),
                total=batch_selection.total_groups,
                max_groups=batch_selection.max_groups,
                urgent_exit_forced=batch_selection.urgent_exit_count,
                deferred_exit_forced=batch_selection.deferred_exit_count,
                loss_watch_forced=batch_selection.loss_watch_count,
                profit_exit_forced=batch_selection.profit_exit_count,
                priority_selected=batch_selection.priority_selected_count,
                fast_scan=[
                    {
                        "symbol": key[1],
                        "model": key[0],
                        "score": round(scan.get("priority_score", 0.0), 2),
                        "exit_score": round(scan.get("exit_score", 0.0), 2),
                        "add_score": round(scan.get("add_score", 0.0), 2),
                        "reason": scan.get("reason"),
                    }
                    for key, scan in sorted(
                        fast_scan.items(),
                        key=lambda item: item[1].get("priority_score", 0.0),
                        reverse=True,
                    )[:8]
                ],
            )
        for (model_name, symbol), positions in grouped_items:
            group_timeout = self._position_review_stage_timeout_seconds(
                round_deadline_monotonic,
                strategy_context=strategy_mode_context,
                stage="position_review_group",
                symbol=symbol,
            )
            if round_deadline_monotonic is not None and group_timeout <= 0:
                self._append_position_review_budget_warning(
                    results=results,
                    stage="position_review_group",
                    symbol=symbol,
                    remaining_seconds=self._remaining_monotonic_seconds(
                        round_deadline_monotonic
                    ),
                )
                break
            normalized_symbol = self._normalize_position_symbol(symbol)
            portfolio_symbol_context = self._portfolio_profit_protection_symbol_context(
                portfolio_profit_context,
                model_name,
                normalized_symbol or symbol,
                positions,
            )
            position_profit_peak_context = self._position_profit_peak_context(
                model_name,
                normalized_symbol or symbol,
                positions,
            )
            fv = feature_vectors.get(symbol) or feature_vectors.get(normalized_symbol)
            if fv is None:
                feature_timeout = self._position_review_stage_timeout_seconds(
                    round_deadline_monotonic,
                    fallback_timeout=AUTO_SCAN_FEATURE_FETCH_TIMEOUT_SECONDS,
                    strategy_context=strategy_mode_context,
                    stage="position_feature_refresh",
                    symbol=normalized_symbol or symbol,
                )
                if round_deadline_monotonic is not None and feature_timeout <= 0:
                    self._append_position_review_budget_warning(
                        results=results,
                        stage="position_feature_refresh",
                        symbol=normalized_symbol or symbol,
                        remaining_seconds=self._remaining_monotonic_seconds(
                            round_deadline_monotonic
                        ),
                    )
                    break
                try:
                    fv = await asyncio.wait_for(
                        self._get_feature_vector_snapshot(
                            normalized_symbol or symbol,
                            wait_for_sentiment=False,
                            block_on_remote_indicators=False,
                            block_on_remote_derivatives=False,
                        ),
                        timeout=feature_timeout,
                    )
                except TimeoutError:
                    budget_limited = feature_timeout < (
                        max(float(AUTO_SCAN_FEATURE_FETCH_TIMEOUT_SECONDS), 0.25) - 0.001
                    )
                    logger.warning(
                        "position review feature vector refresh timed out",
                        symbol=normalized_symbol or symbol,
                        model=model_name,
                        timeout_seconds=round(feature_timeout, 3),
                        budget_limited=budget_limited,
                    )
                    if budget_limited:
                        self._append_position_review_budget_warning(
                            results=results,
                            stage="position_feature_refresh",
                            symbol=normalized_symbol or symbol,
                            remaining_seconds=self._remaining_monotonic_seconds(
                                round_deadline_monotonic
                            ),
                        )
                        break
                    results.setdefault("warnings", []).append(
                        {
                            "model": model_name,
                            "symbol": normalized_symbol or symbol,
                            "warning": (
                                "持仓复盘中该币种行情刷新超时，系统已先跳过这一组并继续处理其他持仓；"
                                "下一轮会用最新行情重试。"
                            ),
                        }
                    )
                    fv = None
                except Exception as exc:
                    logger.debug(
                        "position review feature vector refresh failed",
                        symbol=normalized_symbol or symbol,
                        model=model_name,
                        error=safe_error_text(exc),
                    )
                    fv = None

            if fv is None:
                continue

            try:
                decision_timeout = self._position_review_stage_timeout_seconds(
                    round_deadline_monotonic,
                    strategy_context=strategy_mode_context,
                    stage="position_review_decision",
                    symbol=normalized_symbol or symbol,
                )
                if round_deadline_monotonic is not None and decision_timeout <= 0:
                    self._append_position_review_budget_warning(
                        results=results,
                        stage="position_review_decision",
                        symbol=normalized_symbol or symbol,
                        remaining_seconds=self._remaining_monotonic_seconds(
                            round_deadline_monotonic
                        ),
                    )
                    break
                decision_result = await asyncio.wait_for(
                    self.position_review_decision_service.decide(
                        PositionReviewDecisionRequest(
                            model_name=model_name,
                            symbol=symbol,
                            normalized_symbol=normalized_symbol,
                            feature_vector=fv,
                            open_positions=open_positions,
                            trading_mode=mode_manager.mode.value,
                            position_entry_pause_reason=position_entry_pause_reason,
                            market_regime_context=market_regime_context,
                            strategy_mode_context=strategy_mode_context,
                            portfolio_symbol_context=portfolio_symbol_context,
                            position_profit_peak_context=position_profit_peak_context,
                        )
                    ),
                    timeout=decision_timeout,
                )
                if decision_result is None:
                    continue
                decision = decision_result.decision
            except TimeoutError:
                self._append_position_review_budget_warning(
                    results=results,
                    stage="position_review_decision",
                    symbol=normalized_symbol or symbol,
                    remaining_seconds=self._remaining_monotonic_seconds(
                        round_deadline_monotonic
                    ),
                )
                logger.warning(
                    "position review decision timed out within round budget",
                    model=model_name,
                    symbol=symbol,
                    timeout_seconds=round(decision_timeout, 3),
                )
                remaining_seconds = self._remaining_monotonic_seconds(round_deadline_monotonic)
                if (
                    round_deadline_monotonic is not None
                    and remaining_seconds is not None
                    and remaining_seconds <= 1.5
                ):
                    break
                results.setdefault("warnings", []).append(
                    {
                        "model": model_name,
                        "symbol": normalized_symbol or symbol,
                        "warning": (
                            "持仓复盘中该币种决策超时，系统已先跳过这一组并继续处理其他持仓；"
                            "下一轮会结合最新持仓和行情重新复盘。"
                        ),
                    }
                )
                continue
            except Exception as e:
                logger.error(
                    "review position decide failed",
                    model=model_name,
                    symbol=symbol,
                    error=safe_error_text(e),
                )
                continue

            if not isinstance(decision, DecisionOutput):
                continue
            self._attach_decision_timing(
                decision,
                decision_result.analysis_started,
                "position_review",
            )

            model_mode = self._get_model_execution_mode(model_name)
            decision_db_id = await self._log_decision(decision, is_paper=(model_mode == "paper"))
            self._decision_count += 1
            if decision_db_id is not None and round_decision_ids is not None:
                round_decision_ids.add(decision_db_id)

            handled_keys.add((model_name, self._normalize_position_symbol(symbol)))

            risk_alert = self.position_review_risk_alert_policy.build_alert(decision, positions)
            if risk_alert:
                self.position_review_risk_alert_policy.attach(decision, risk_alert)

            process_result = await self._await_position_review_post_decision_handoff(
                self.position_review_decision_processor.process(
                    decision=decision,
                    model_name=model_name,
                    symbol=symbol,
                    model_mode=model_mode,
                    decision_db_id=decision_db_id,
                    open_positions=open_positions,
                    feature_vector=fv,
                    position_entry_pause_reason=position_entry_pause_reason,
                    risk_alert=risk_alert,
                    results=results,
                ),
                symbol=symbol,
                model_name=model_name,
                decision=decision,
                decision_db_id=decision_db_id,
            )
            if process_result.handled:
                continue
            if process_result.candidate is not None:
                candidates.append(process_result.candidate)

        return candidates, handled_keys

    async def _await_position_review_post_decision_handoff(
        self,
        awaitable: Awaitable[Any],
        *,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        decision_db_id: int | None,
    ) -> Any:
        """Finish post-AI position-review processing after a decision is logged.

        The position-review stage has a wall-clock timeout so a slow review
        cannot block the whole service forever.  Once a concrete long/short/exit
        decision has been logged, cancelling the post-processing coroutine would
        leave a visible entry decision with no risk result and no execution
        handoff.  At that point the correct boundary is to finish risk
        assessment and either hand the candidate to execution or persist the real
        blocking reason.
        """

        task = asyncio.create_task(awaitable)
        cancellation_count = 0
        while True:
            try:
                result = await asyncio.shield(task)
                if cancellation_count:
                    logger.info(
                        "position review post-decision processing completed after outer cancellation",
                        symbol=symbol,
                        model=model_name,
                        action=decision.action.value,
                        decision_id=decision_db_id,
                        outer_cancellations=cancellation_count,
                    )
                return result
            except asyncio.CancelledError:
                if task.done():
                    return task.result()
                cancellation_count += 1
                reason = (
                    "持仓复盘已经生成明确裁决，外层阶段超时保护已触发；"
                    "系统继续完成风控复核和执行交接，避免已通过条件的开仓/平仓信号被中途丢弃。"
                )
                logger.warning(
                    "position review post-decision processing is waiting for terminal result after outer cancellation",
                    symbol=symbol,
                    model=model_name,
                    action=decision.action.value,
                    decision_id=decision_db_id,
                    outer_cancellations=cancellation_count,
                )
                if decision_db_id is not None:
                    await self._record_and_persist_decision_stage(
                        decision_db_id,
                        decision,
                        DecisionStage.RISK_CHECK,
                        DecisionStageStatus.PENDING,
                        reason,
                        {
                            "source": "position_review_post_decision_handoff",
                            "outer_cancellations": cancellation_count,
                        },
                    )



    def _position_review_priority_policy(self) -> PositionReviewPriorityPolicy:
        policy = getattr(self, "position_review_priority", None)
        if policy is not None:
            return policy

        def peak_key(model_name: str, symbol: str, side: str) -> Any:
            tracker = getattr(self, "position_profit_peaks", None)
            if tracker is not None:
                return tracker.key(model_name, symbol, side)
            return (model_name, self._normalize_position_symbol(symbol), side)

        def peak_states() -> dict[Any, dict[str, Any]]:
            tracker = getattr(self, "position_profit_peaks", None)
            peaks = getattr(tracker, "peaks", {})
            return peaks if isinstance(peaks, dict) else {}

        return PositionReviewPriorityPolicy(
            normalize_symbol=self._normalize_position_symbol,
            position_peak_key=peak_key,
            position_peaks_provider=peak_states,
        )

    def _scan_position_review_groups(
        self,
        grouped_items: list[tuple[tuple[str, str], list[dict]]],
        feature_vectors: dict[str, Any],
        portfolio_profit_context: dict[str, Any] | None = None,
        strategy_context: dict[str, Any] | None = None,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Fast pass over every open position before spending AI time.

        The score decides which position groups jump the AI queue. It is only a
        triage signal; final add/close actions still require the normal AI and
        risk checks.
        """
        return self._position_review_priority_policy().scan_groups(
            grouped_items,
            feature_vectors,
            portfolio_profit_context,
            strategy_context,
            aggregate_position_group=self._position_group_aggregator_policy().aggregate,
        )

    def _is_urgent_position_exit_scan(self, scan: dict[str, Any] | None) -> bool:
        return self._position_review_priority_policy().is_urgent_exit_scan(scan)

    def _position_review_batch_policy(self) -> PositionReviewBatchPolicy:
        policy = getattr(self, "position_review_batch", None)
        if policy is not None:
            return policy
        return PositionReviewBatchPolicy(
            urgent_exit_checker=self._is_urgent_position_exit_scan,
        )


    def _analysis_budget_policy(self) -> AnalysisBudgetPolicy:
        policy = getattr(self, "analysis_budget", None)
        if policy is not None:
            return policy
        return AnalysisBudgetPolicy(
            normalize_symbol=self._normalize_position_symbol,
            open_position_group_counter=self.entry_symbol_universe.open_position_group_count,
            portfolio_profit_context_provider=self._portfolio_profit_protection_context,
            position_review_scanner=self._scan_position_review_groups,
            urgent_exit_checker=self._is_urgent_position_exit_scan,
            default_model_name=ENSEMBLE_TRADER_NAME,
        )

    def _position_review_budget_context(
        self,
        open_positions: list[dict],
        feature_vectors: dict[str, Any],
        *,
        base_market_limit: int,
        run_position_analysis: bool,
        run_market_analysis: bool,
        new_pair_pause_reason: str | None = None,
        strategy_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Allocate slow AI work between position protection and new entries."""
        return self._analysis_budget_policy().context(
            open_positions,
            feature_vectors,
            base_market_limit=base_market_limit,
            run_position_analysis=run_position_analysis,
            run_market_analysis=run_market_analysis,
            new_pair_pause_reason=new_pair_pause_reason,
            strategy_context=strategy_context,
        )

    def _portfolio_profit_protection_policy(self) -> PortfolioProfitProtectionPolicy:
        policy = getattr(self, "portfolio_profit_protection", None)
        if policy is not None:
            return policy
        return PortfolioProfitProtectionPolicy(
            normalize_symbol=self._normalize_position_symbol,
            default_model_name=ENSEMBLE_TRADER_NAME,
        )

    def _portfolio_profit_protection_context(self, open_positions: list[dict]) -> dict[str, Any]:
        """Build a portfolio-level floating-profit context for AI lock-profit review."""
        return self._portfolio_profit_protection_policy().context(open_positions)

    def _portfolio_profit_protection_symbol_context(
        self,
        context: dict[str, Any],
        model_name: str,
        symbol: str,
        positions: list[dict] | None = None,
    ) -> dict[str, Any]:
        return self._portfolio_profit_protection_policy().symbol_context(
            context,
            model_name,
            symbol,
            positions,
        )

    def _position_profit_peak_context_policy(self) -> PositionProfitPeakContextPolicy:
        policy = getattr(self, "position_profit_peak_context", None)
        if policy is not None:
            return policy

        def peak_key(model_name: str, symbol: str, side: str) -> Any:
            tracker = getattr(self, "position_profit_peaks", None)
            if tracker is not None:
                return tracker.key(model_name, symbol, side)
            return "|".join([str(model_name or ENSEMBLE_TRADER_NAME), str(symbol or ""), side])

        def peaks() -> dict[Any, dict[str, Any]]:
            tracker = getattr(self, "position_profit_peaks", None)
            values = getattr(tracker, "peaks", {})
            return values if isinstance(values, dict) else {}

        return PositionProfitPeakContextPolicy(
            normalize_symbol=self._normalize_position_symbol,
            aggregate_position_group=self._position_group_aggregator_policy().aggregate,
            position_peak_key=peak_key,
            position_peaks_provider=peaks,
            default_model_name=ENSEMBLE_TRADER_NAME,
        )

    def _position_group_aggregator_policy(self) -> PositionGroupAggregator:
        policy = getattr(self, "position_group_aggregator", None)
        if policy is not None:
            return policy
        policy = PositionGroupAggregator(
            normalize_symbol=self._normalize_position_symbol,
            float_parser=self._safe_float,
            default_model_name=ENSEMBLE_TRADER_NAME,
        )
        self.position_group_aggregator = policy
        return policy

    def _position_profit_peak_context(
        self,
        model_name: str,
        symbol: str,
        positions: list[dict] | None,
    ) -> dict[str, Any]:
        """Expose per-position floating-profit peak to the AI evidence layer."""
        return self._position_profit_peak_context_policy().context(
            model_name,
            symbol,
            positions,
        )

    def _attach_execution_leverage_summary(
        self,
        decision: DecisionOutput,
        result: ExecutionResult,
        ai_requested_leverage: float,
    ) -> None:
        """Store AI/OKX/actual leverage in the decision and execution payloads."""
        raw_result = self._safe_dict(result.raw_response)
        leverage_check = self._safe_dict(raw_result.get("leverage_check"))
        actual = self._safe_float(
            leverage_check.get("actual_leverage")
            or leverage_check.get("target_leverage")
            or decision.suggested_leverage,
            ai_requested_leverage,
        )
        okx_max = self._safe_float(
            leverage_check.get("okx_max_leverage") or leverage_check.get("max_leverage"),
            0.0,
        )
        target = self._safe_float(
            leverage_check.get("target_leverage") or decision.suggested_leverage,
            actual,
        )
        summary: dict[str, Any] = {
            "ai_suggested_leverage": round(float(ai_requested_leverage or 1.0), 4),
            "okx_max_leverage": round(float(okx_max), 4) if okx_max > 0 else None,
            "actual_leverage": round(float(actual or target or 1.0), 4),
        }
        raw = self._safe_dict(decision.raw_response)
        raw["execution_leverage"] = summary
        decision.raw_response = raw
        raw_result["execution_leverage"] = summary
        result.raw_response = raw_result
        decision.suggested_leverage = self._safe_float(summary.get("actual_leverage"), 1.0)

    async def _get_open_positions_context(self) -> list[dict]:
        return await self.okx_sync_service.get_open_positions_context()

    async def _get_local_open_positions_context(
        self,
        *,
        strict: bool = False,
    ) -> list[dict]:
        loader = getattr(self.okx_sync_service, "get_local_open_positions_context", None)
        if loader is None:
            return []
        try:
            return await loader(strict=strict)
        except TypeError:
            if strict:
                raise
            return await loader()

    async def _open_positions_context_for_round(self, analysis_scope: str) -> list[dict]:
        if analysis_scope != "market":
            positions = await self._get_open_positions_context()
            self._remember_open_positions_context(positions)
            return positions

        cached = self._cached_open_positions_context()
        if cached is not None:
            self._schedule_open_positions_context_refresh()
            return cached

        task = getattr(self, "_open_positions_context_refresh_task", None)
        if task is not None and not task.done():
            fallback = self._cached_open_positions_context(ignore_age=True)
            if fallback is not None:
                return fallback

        if self._okx_authoritative_sync_context_is_usable_for_market_scan():
            try:
                local_positions = await self._get_local_open_positions_context(
                    strict=True,
                )
            except Exception as exc:
                logger.warning(
                    "failed to load local open positions for market scan; falling back to OKX",
                    error=safe_error_text(exc),
                )
            else:
                self._remember_open_positions_context(local_positions)
                self._schedule_open_positions_context_refresh()
                return local_positions

        positions = await self._get_open_positions_context()
        self._remember_open_positions_context(positions)
        return positions

    def _okx_authoritative_sync_context_is_usable_for_market_scan(self) -> bool:
        payload = self._okx_authoritative_sync_status_payload()
        status = str(payload.get("status") or "").lower()
        if status not in {"ok", "degraded"}:
            return False
        if int(payload.get("last_requires_attention_count") or 0) > 0:
            return False
        if not payload.get("fresh_success_available"):
            return False
        return True

    def _remember_open_positions_context(self, positions: list[dict] | None) -> None:
        if positions is None:
            return
        self._open_positions_context_cache = {
            "created_at": datetime.now(UTC),
            "positions": [dict(position) for position in positions if isinstance(position, dict)],
        }

    def _cached_open_positions_context(
        self,
        *,
        ignore_age: bool = False,
    ) -> list[dict] | None:
        cache = getattr(self, "_open_positions_context_cache", None)
        if not isinstance(cache, dict):
            return None
        created_at = cache.get("created_at")
        positions = cache.get("positions")
        if not isinstance(created_at, datetime) or not isinstance(positions, list):
            return None
        if not ignore_age:
            age_seconds = (datetime.now(UTC) - created_at).total_seconds()
            if age_seconds > MARKET_OPEN_POSITIONS_CONTEXT_TTL_SECONDS:
                return None
        return [dict(position) for position in positions if isinstance(position, dict)]

    def _schedule_open_positions_context_refresh(self) -> None:
        task = getattr(self, "_open_positions_context_refresh_task", None)
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def refresh() -> None:
            positions = await self._get_open_positions_context()
            self._remember_open_positions_context(positions)

        task = loop.create_task(refresh())
        self._open_positions_context_refresh_task = task
        task.add_done_callback(_consume_task_result)

    async def _strategy_context_account_equity(self, mode: str) -> float:
        selected_mode = "live" if mode == "live" else "paper"
        if _analysis_scope_context.get() == "market":
            cached_snapshot = self.peek_okx_balance_snapshot_for_mode(selected_mode)
            if cached_snapshot:
                self._last_strategy_context_account_equity_source = "cached_okx_balance_snapshot"
                return max(tradeable_balance_from_snapshot(cached_snapshot), 0.0)
            current_context = getattr(self, "_current_strategy_mode_context", None)
            if isinstance(current_context, dict):
                previous_equity = self._safe_float(current_context.get("account_equity"), 0.0)
                if previous_equity > 0:
                    self._last_strategy_context_account_equity_source = "previous_strategy_context"
                    return previous_equity
            self._last_strategy_context_account_equity_source = "market_no_cached_balance"
            return 0.0
        balance = await self.allocated_order_balance(selected_mode)
        self._last_strategy_context_account_equity_source = "fresh_okx_balance_snapshot"
        return max(float(balance or 0.0), 0.0)

    def _get_model_execution_mode(self, model_name: str) -> str:
        """Return the execution_mode for a model. Defaults to 'paper'."""
        if model_name == ENSEMBLE_TRADER_NAME:
            return mode_manager.mode.value
        if model_name in self._model_execution_modes:
            return self._model_execution_modes[model_name]
        for cfg in settings.ai_models:
            if cfg.get("name") == model_name:
                return cfg.get("execution_mode", "paper")
        return "paper"

    def _okx_balance_snapshot_for_new_pair_pause_context(
        self,
        model_mode: str,
    ) -> dict[str, Any] | None:
        """Use only cached OKX balance for scan-pause checks.

        A cold or slow OKX balance request must not turn into a global
        market-scan pause. Execution sizing and order submission still fetch or
        validate balance before any real order can be sent.
        """

        try:
            return self.peek_okx_balance_snapshot_for_mode(model_mode, allow_stale=True)
        except Exception as exc:
            logger.warning(
                "failed to read cached OKX balance snapshot for new-pair pause context",
                mode=model_mode,
                error=safe_error_text(exc),
            )
            return None

    def _schedule_okx_balance_snapshot_refresh_for_new_pair_pause(self, model_mode: str) -> None:
        try:
            self._schedule_okx_balance_snapshot_refresh(model_mode)
        except Exception as exc:
            logger.debug(
                "failed to schedule OKX balance refresh for new-pair pause context",
                mode=model_mode,
                error=safe_error_text(exc),
            )

    async def _new_pair_analysis_pause_reason(
        self,
        model_name: str,
        open_positions: list[dict] | None = None,
        allow_background_refresh: bool = False,
    ) -> str | None:
        """Pause new-symbol AI analysis when balance/risk no longer allows entries."""
        breaker = self.risk_engine.circuit_breaker
        if breaker.is_open:
            state = breaker.get_state()
            return f"风险熔断已开启，暂停分析新的交易对。原因：{state.get('tripped_reason') or '触发风险阈值'}"

        okx_sync_reason = self._okx_authoritative_sync_entry_block_reason()
        if okx_sync_reason:
            return okx_sync_reason

        model_mode = self._get_model_execution_mode(model_name)
        cache_key = self._new_pair_pause_context_cache_key(
            model_name=model_name,
            model_mode=model_mode,
            open_positions=open_positions or [],
        )
        cached_reason = self._cached_new_pair_pause_context_reason(cache_key)
        if cached_reason is not None:
            return cached_reason

        if allow_background_refresh:
            return None

        return await self._refresh_new_pair_pause_context_reason(
            cache_key=cache_key,
            model_name=model_name,
            model_mode=model_mode,
            open_positions=open_positions or [],
        )

    async def _refresh_new_pair_pause_context_reason(
        self,
        *,
        cache_key: tuple[Any, ...],
        model_name: str,
        model_mode: str,
        open_positions: list[dict],
    ) -> str | None:
        okx_snapshot = self._okx_balance_snapshot_for_new_pair_pause_context(model_mode)
        if not okx_snapshot:
            self._schedule_okx_balance_snapshot_refresh_for_new_pair_pause(model_mode)
            logger.warning(
                "OKX balance snapshot unavailable for scan pause check; market scan continues",
                mode=model_mode,
                policy="advisory_cache_miss_execution_gate_will_recheck",
            )
            self._remember_new_pair_pause_context_reason(cache_key, None)
            return None
        okx_available = tradeable_balance_from_snapshot(okx_snapshot)
        okx_allocatable = allocatable_balance_from_snapshot(okx_snapshot)
        if okx_allocatable <= 0:
            reason = "未获取到 OKX 账户权益或余额，暂停分析新的交易对。"
            self._remember_new_pair_pause_context_reason(cache_key, reason)
            return reason
        if okx_available <= 0:
            reason = (
                f"OKX 没有可交易余额：当前可用 {okx_available:.2f} USDT，暂停分析新的交易对。"
            )
            self._remember_new_pair_pause_context_reason(cache_key, reason)
            return reason
        self._remember_new_pair_pause_context_reason(cache_key, None)
        return None

    def _schedule_new_pair_pause_context_refresh(
        self,
        *,
        cache_key: tuple[Any, ...],
        model_name: str,
        model_mode: str,
        open_positions: list[dict],
    ) -> None:
        task = getattr(self, "_new_pair_pause_context_refresh_task", None)
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def refresh() -> None:
            await self._refresh_new_pair_pause_context_reason(
                cache_key=cache_key,
                model_name=model_name,
                model_mode=model_mode,
                open_positions=open_positions,
            )

        task = loop.create_task(refresh())
        self._new_pair_pause_context_refresh_task = task
        task.add_done_callback(_consume_task_result)

    def _new_pair_pause_context_cache_key(
        self,
        *,
        model_name: str,
        model_mode: str,
        open_positions: list[dict],
    ) -> tuple[Any, ...]:
        normalized_positions = sorted(
            (
                self._normalize_position_symbol(position.get("symbol")),
                str(position.get("side") or "").lower(),
            )
            for position in open_positions
            if isinstance(position, dict) and position.get("is_open", True)
        )
        return (
            str(model_name or ""),
            "live" if model_mode == "live" else "paper",
            tuple(normalized_positions),
        )

    def _cached_new_pair_pause_context_reason(self, cache_key: tuple[Any, ...]) -> str | None:
        cache = getattr(self, "_new_pair_pause_context_cache", None)
        if not isinstance(cache, dict):
            return None
        entry = cache.get(cache_key)
        if not isinstance(entry, dict):
            return None
        created_at = entry.get("created_at")
        if not isinstance(created_at, datetime):
            return None
        age_seconds = (datetime.now(UTC) - created_at).total_seconds()
        if age_seconds > NEW_PAIR_PAUSE_CONTEXT_TTL_SECONDS:
            return None
        reason = entry.get("reason")
        return str(reason) if reason else ""

    def _remember_new_pair_pause_context_reason(
        self,
        cache_key: tuple[Any, ...],
        reason: str | None,
    ) -> None:
        self._new_pair_pause_context_cache = {
            cache_key: {
                "created_at": datetime.now(UTC),
                "reason": reason or "",
            }
        }

    async def _record_new_pair_pause_state(self, model_name: str, reason: str | None) -> None:
        previous = self._new_pair_pause_reasons.get(model_name)
        if reason:
            if previous != reason:
                self._new_pair_pause_reasons[model_name] = reason
                await self._log_risk_event(
                    "new_pair_analysis_paused",
                    "ALL",
                    f"{reason} 系统动作：暂停分析新的交易对；已有持仓继续复盘、止盈止损和平仓处理。完成结果：暂停已生效。",
                    model_name,
                    severity="warn",
                )
            return

        if previous:
            self._new_pair_pause_reasons.pop(model_name, None)
            await self._log_risk_event(
                "new_pair_analysis_resumed",
                "ALL",
                "余额或风险状态已恢复，系统动作：恢复分析新的交易对。完成结果：新交易对分析已重新开启。",
                model_name,
                severity="warn",
            )

    def _refresh_model_modes(self) -> None:
        """Rebuild model_name -> execution_mode mapping from current settings."""
        self._model_execution_modes = {ENSEMBLE_TRADER_NAME: mode_manager.mode.value}
        for cfg in settings.ai_models:
            self._model_execution_modes[cfg.get("name", "")] = cfg.get("execution_mode", "paper")

    def _has_live_models(self) -> bool:
        return mode_manager.mode.value == "live" or any(
            cfg.get("execution_mode", "paper") == "live" for cfg in settings.ai_models
        )

    async def _get_okx_executor_for_mode(self, mode: str) -> OKXExecutor:
        """Return the OKX executor for paper/demo or live/real mode."""
        selected_mode = "live" if mode == "live" else "paper"
        if selected_mode == "paper":
            if self._okx_paper is None:
                self._okx_paper = OKXExecutor(
                    mode="paper",
                    load_markets_on_initialize=False,
                )
                await self._okx_paper.initialize()
            return self._okx_paper

        if self._okx_live is None:
            self._okx_live = OKXExecutor(
                mode="live",
                load_markets_on_initialize=False,
            )
            await self._okx_live.initialize()
        return self._okx_live

    async def _shadow_execution_cost_facts(self, mode: str) -> dict[str, Any]:
        """Read current account fee facts once for each due-shadow mode batch."""

        executor = await self._get_okx_executor_for_mode(mode)
        return await executor.fetch_account_fee_snapshot()

    async def _market_execution_cost_facts(self, mode: str) -> dict[str, Any]:
        """Read the same authoritative fee facts used by shadow evaluation."""

        try:
            return await self._shadow_execution_cost_facts(mode)
        except Exception as exc:
            logger.warning(
                "fetch market execution cost facts failed",
                mode=mode,
                error=safe_error_text(exc),
            )
            return {}

    async def _get_okx_available_balance_for_mode(self, mode: str) -> float | None:
        """Return the actual OKX free USDT balance used to cap new entries."""
        return await self.account_accounting_service.okx_available_balance_for_mode(mode)

    def _okx_balance_snapshot_lock_for_mode(self, mode: str) -> asyncio.Lock:
        locks = getattr(self, "_okx_balance_snapshot_locks", None)
        if not isinstance(locks, dict):
            locks = {}
            self._okx_balance_snapshot_locks = locks
        return locks.setdefault(mode, asyncio.Lock())

    def _cached_okx_balance_snapshot(
        self,
        mode: str,
        *,
        max_age_seconds: float,
        stale_reason: str | None = None,
        include_error: bool = False,
    ) -> dict[str, Any] | None:
        cache = getattr(self, "_okx_balance_snapshot_cache", None)
        if not isinstance(cache, dict):
            return None
        cached = cache.get(mode)
        if not isinstance(cached, dict):
            return None
        fetched_at = cached.get("fetched_at")
        if not isinstance(fetched_at, datetime):
            return None
        age = (datetime.now(UTC) - fetched_at).total_seconds()
        if age > max_age_seconds:
            return None
        snapshot = dict(cached.get("snapshot") or {})
        if not snapshot:
            return None
        if stale_reason is not None:
            snapshot["stale"] = True
            snapshot["stale_age_seconds"] = round(age, 3)
            snapshot["stale_reason"] = stale_reason
            if include_error:
                snapshot["error"] = stale_reason
            logger.warning(
                "using cached OKX balance snapshot",
                mode=mode,
                age_seconds=round(age, 3),
                reason=stale_reason,
            )
        return snapshot

    def _invalidate_okx_balance_snapshot_cache_for_mode(self, mode: str) -> None:
        selected_mode = "live" if mode == "live" else "paper"
        cache = getattr(self, "_okx_balance_snapshot_cache", None)
        if isinstance(cache, dict):
            cache.pop(selected_mode, None)

    def _invalidate_okx_balance_snapshot_cache_for_model(self, model_name: str | None) -> None:
        selected_mode = mode_manager.mode.value
        model_modes = getattr(self, "_model_execution_modes", None)
        if isinstance(model_modes, dict):
            mapped_mode = model_modes.get(str(model_name or ""))
            if mapped_mode in {"paper", "live"}:
                selected_mode = mapped_mode
        self._invalidate_okx_balance_snapshot_cache_for_mode(selected_mode)

    async def _get_okx_balance_snapshot_for_mode(
        self,
        mode: str,
        *,
        allow_stale_while_refresh: bool = True,
    ) -> dict[str, Any] | None:
        """Return OKX USDT balance fields for allocation and order sizing."""
        selected_mode = "live" if mode == "live" else "paper"

        def remember_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
            stored_snapshot = dict(snapshot)
            stored_snapshot.pop("stale", None)
            stored_snapshot.pop("stale_age_seconds", None)
            stored_snapshot.pop("stale_reason", None)
            stored_snapshot.pop("error", None)
            self._okx_balance_snapshot_cache[selected_mode] = {
                "snapshot": stored_snapshot,
                "fetched_at": datetime.now(UTC),
            }
            return dict(stored_snapshot)

        def cached_snapshot(reason: str) -> dict[str, Any] | None:
            return self._cached_okx_balance_snapshot(
                selected_mode,
                max_age_seconds=OKX_BALANCE_SNAPSHOT_STALE_SECONDS,
                stale_reason=reason,
                include_error=True,
            )

        async def fresh_executor_snapshot(reason: str) -> dict[str, Any] | None:
            """Fallback matching the settings connection test path.

            The long-lived trading executor can occasionally be busy or stuck in a
            CCXT request while a brand-new OKX client succeeds. Do one isolated
            balance pull before treating the account as unavailable.
            """
            fallback = OKXExecutor(mode=selected_mode, load_markets_on_initialize=False)
            try:
                await asyncio.wait_for(fallback.initialize(), timeout=12.0)
                snapshot = await asyncio.wait_for(
                    fallback.get_balance_snapshot("USDT"),
                    timeout=10.0,
                )
                if snapshot.get("error"):
                    logger.warning(
                        "fresh OKX balance snapshot fallback returned error",
                        mode=selected_mode,
                        original_reason=reason,
                        error=safe_error_text(snapshot.get("error")),
                    )
                    return None
                snapshot = remember_snapshot(snapshot)
                snapshot["fallback_executor"] = True
                snapshot["fallback_reason"] = reason
                logger.warning(
                    "fresh OKX balance snapshot fallback succeeded",
                    mode=selected_mode,
                    original_reason=reason,
                )
                return snapshot
            except TimeoutError:
                logger.warning(
                    "fresh OKX balance snapshot fallback timed out",
                    mode=selected_mode,
                    original_reason=reason,
                )
                return None
            except Exception as exc:
                logger.warning(
                    "fresh OKX balance snapshot fallback failed",
                    mode=selected_mode,
                    original_reason=reason,
                    error=safe_error_text(exc),
                )
                return None
            finally:
                try:
                    await fallback.shutdown()
                except Exception as exc:
                    logger.debug(
                        "fallback OKX executor shutdown failed",
                        mode=selected_mode,
                        error=safe_error_text(exc),
                    )

        executor = (
            getattr(self, "_okx_live", None)
            if selected_mode == "live"
            else getattr(self, "_okx_paper", None)
        )
        fresh_cached = self._cached_okx_balance_snapshot(
            selected_mode,
            max_age_seconds=OKX_BALANCE_SNAPSHOT_FRESH_SECONDS,
        )
        if fresh_cached:
            return fresh_cached
        lock = self._okx_balance_snapshot_lock_for_mode(selected_mode)
        if lock.locked():
            refresh_cached = cached_snapshot("OKX balance refresh already in progress")
            if refresh_cached:
                return refresh_cached
            refresh_fallback = await fresh_executor_snapshot(
                "shared OKX balance refresh already in progress"
            )
            if refresh_fallback:
                refresh_fallback["refresh_in_progress"] = True
                return refresh_fallback
        stale_cached = cached_snapshot("OKX balance snapshot refresh scheduled")
        if stale_cached and allow_stale_while_refresh:
            self._schedule_okx_balance_snapshot_refresh(selected_mode)
            stale_cached["refresh_in_background"] = True
            return stale_cached
        async with lock:
            fresh_cached = self._cached_okx_balance_snapshot(
                selected_mode,
                max_age_seconds=OKX_BALANCE_SNAPSHOT_FRESH_SECONDS,
            )
            if fresh_cached:
                return fresh_cached
            try:
                if not executor:
                    executor = await self._get_okx_executor_for_mode(selected_mode)
            except Exception as exc:
                reason = f"executor unavailable: {safe_error_text(exc)}"
                fallback_snapshot = cached_snapshot(reason) or await fresh_executor_snapshot(reason)
                if fallback_snapshot:
                    return fallback_snapshot
                return None
            try:
                snapshot = await asyncio.wait_for(executor.get_balance_snapshot("USDT"), timeout=8.0)
                if snapshot.get("error"):
                    reason = safe_error_text(snapshot.get("error"))
                    fallback_snapshot = cached_snapshot(reason) or await fresh_executor_snapshot(reason)
                    if fallback_snapshot:
                        return fallback_snapshot
                    return None
                return remember_snapshot(snapshot)
            except TimeoutError:
                logger.warning("timed out fetching OKX balance snapshot", mode=selected_mode)
                reason = "OKX balance snapshot request timed out"
                fallback_snapshot = cached_snapshot(reason) or await fresh_executor_snapshot(reason)
                if fallback_snapshot:
                    return fallback_snapshot
                return None
            except Exception as exc:
                logger.warning(
                    "failed to fetch OKX balance snapshot",
                    mode=selected_mode,
                    error=safe_error_text(exc),
                )
                reason = safe_error_text(exc)
                fallback_snapshot = cached_snapshot(reason) or await fresh_executor_snapshot(reason)
                if fallback_snapshot:
                    return fallback_snapshot
                return None

    def _schedule_okx_balance_snapshot_refresh(self, mode: str) -> None:
        selected_mode = "live" if mode == "live" else "paper"
        tasks = getattr(self, "_okx_balance_snapshot_refresh_tasks", None)
        if not isinstance(tasks, dict):
            tasks = {}
            self._okx_balance_snapshot_refresh_tasks = tasks
        current = tasks.get(selected_mode)
        if current is not None and not current.done():
            return
        tasks[selected_mode] = asyncio.create_task(
            self._refresh_okx_balance_snapshot_for_mode(selected_mode)
        )

    async def _refresh_okx_balance_snapshot_for_mode(self, mode: str) -> None:
        try:
            await self._get_okx_balance_snapshot_for_mode(
                mode,
                allow_stale_while_refresh=False,
            )
        except Exception as exc:
            logger.warning(
                "background OKX balance snapshot refresh failed",
                mode=mode,
                error=safe_error_text(exc),
            )
        finally:
            tasks = getattr(self, "_okx_balance_snapshot_refresh_tasks", None)
            if isinstance(tasks, dict):
                task = asyncio.current_task()
                if tasks.get(mode) is task:
                    tasks.pop(mode, None)

    async def _sync_paper_after_okx(
        self,
        model_name: str,
        decision: DecisionOutput,
        result: ExecutionResult,
    ) -> None:
        """Do not mirror OKX executions into PaperExecutor memory in Phase 3."""

        return

    async def _on_mode_changed(self, manager) -> None:
        """Reinitialize all LLM agent instances when trading mode changes."""
        for model in self.models.get_all():
            if hasattr(model, "reinitialize"):
                try:
                    await model.reinitialize()
                    logger.info(
                        "model reinitialized for new mode", name=model.name, mode=manager.mode.value
                    )
                except Exception as e:
                    logger.error(
                        "failed to reinit model on mode change",
                        name=model.name,
                        error=safe_error_text(e),
                    )

    def _decision_side(self, decision: DecisionOutput) -> str:
        return self.execution_result_factory.decision_side(decision)

    def _action_label_text(self, action: Action | str | None) -> str:
        return self.execution_result_factory.action_label(action)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _safe_dict(self, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _safe_list(self, value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    def _price_from_feature_like(self, value: Any) -> float:
        """Extract a usable current price from feature/ticker-like payloads."""

        if value is None:
            return 0.0
        if isinstance(value, dict):
            candidates = (
                value.get("current_price"),
                value.get("last_price"),
                value.get("last"),
                value.get("close"),
                value.get("price"),
            )
        else:
            candidates = (
                getattr(value, "current_price", None),
                getattr(value, "last_price", None),
                getattr(value, "last", None),
                getattr(value, "close", None),
                getattr(value, "price", None),
            )
        for candidate in candidates:
            price = self._safe_float(candidate, 0.0)
            if price > 0:
                return price
        return 0.0

    def _attach_decision_timing(
        self,
        decision: DecisionOutput | None,
        started_at: datetime,
        stage: str,
    ) -> None:
        if not isinstance(decision, DecisionOutput):
            return
        completed_at = datetime.now(UTC)
        raw = self._safe_dict(decision.raw_response)
        raw["timing"] = {
            **self._safe_dict(raw.get("timing")),
            "stage": stage,
            "analysis_started_at": started_at.isoformat(),
            "decision_completed_at": completed_at.isoformat(),
            "analysis_duration_sec": round((completed_at - started_at).total_seconds(), 3),
        }
        decision.raw_response = raw

    async def _latest_price_for_symbol(self, symbol: str) -> float:
        normalized = self._normalize_position_symbol(symbol) or symbol
        candidates = [symbol, normalized]
        try:
            latest_tickers = getattr(self.data_service.ws_client, "latest_tickers", {}) or {}
            for key in candidates:
                price = self._price_from_feature_like(latest_tickers.get(key))
                if price > 0:
                    return price
        except Exception as exc:
            logger.debug(
                "failed to read latest price from market cache",
                symbol=symbol,
                error=safe_error_text(exc),
            )

        try:
            raw_ticker = await self.data_service.rest_client.fetch_ticker(normalized)
            price = self._price_from_feature_like(raw_ticker)
            if price > 0:
                return price
        except Exception as e:
            logger.warning(
                "pre-order latest price fetch failed",
                symbol=symbol,
                error=safe_error_text(e),
            )

        try:
            fresh = await asyncio.wait_for(
                self._get_feature_vector_snapshot(
                    normalized,
                    wait_for_sentiment=False,
                    block_on_remote_indicators=False,
                    block_on_remote_derivatives=False,
                ),
                timeout=ENTRY_PRICE_RECHECK_TIMEOUT_SECONDS,
            )
            price = self._price_from_feature_like(fresh)
            if price > 0:
                logger.warning(
                    "pre-order latest price recovered from feature snapshot",
                    symbol=symbol,
                    price=price,
                )
                return price
        except Exception as exc:
            logger.warning(
                "pre-order latest price feature fallback failed",
                symbol=symbol,
                error=safe_error_text(exc),
            )
        return 0.0

    def _short_text(self, value: Any, limit: int = 180) -> str:
        return " ".join(str(value or "").split())[:limit]

    def _rejected_execution_result(
        self, decision: DecisionOutput, error: Exception | str
    ) -> ExecutionResult:
        return self.execution_result_factory.rejected(decision, error)

    async def _log_decision(self, decision: DecisionOutput, is_paper: bool) -> int | None:
        decision_id = await self.decision_persistence.log_decision(decision, is_paper)
        if decision_id is not None:
            await self._record_strategy_learning_event(
                mode="paper" if is_paper else "live",
                model_name=decision.model_name or ENSEMBLE_TRADER_NAME,
                symbol=decision.symbol,
                decision=decision,
                event_type="decision_logged",
                event_status="recorded",
                reason=decision.reasoning,
                decision_id=decision_id,
                attribution={
                    "source": "ai_decision",
                    "analysis_type": self._safe_dict(decision.raw_response).get("analysis_type"),
                },
            )
        return decision_id

    def _json_safe_payload(self, value: Any) -> Any:
        """Return a JSON-column-safe copy of model/feature payloads."""
        return self.decision_persistence.json_safe_payload(value)

    def _side_label(self, side: str) -> str:
        return (
            "做多"
            if str(side).lower() == "long"
            else "做空" if str(side).lower() == "short" else str(side)
        )

    async def _log_trade(
        self,
        result,
        model_name: str,
        decision: DecisionOutput,
        decision_id: int | None = None,
    ) -> None:
        await self.trade_order_log_service.log_trade(
            result,
            model_name,
            decision,
            decision_id,
        )

    async def _persist_position_from_execution(
        self,
        model_name: str,
        decision: DecisionOutput,
        result,
        execution_mode: str,
    ) -> None:
        await self.position_execution_persistence.persist(
            model_name=model_name,
            decision=decision,
            result=result,
            execution_mode=execution_mode,
        )

    async def _log_risk_event(
        self,
        event_type: str,
        symbol: str,
        details: str,
        model_name: str,
        severity: str = "warn",
    ) -> None:
        try:
            details = str(sanitize_text(details) or details or "")
            async with get_session_ctx() as session:
                repo = RiskRepository(session)
                await repo.log_risk_event(
                    {
                        "event_type": event_type,
                        "severity": severity,
                        "symbol": symbol,
                        "details": {"message": details},
                        "triggered_by_model": model_name,
                    }
                )
        except Exception as exc:
            logger.debug(
                "failed to log risk event",
                event_type=event_type,
                symbol=symbol,
                error=safe_error_text(exc),
            )

    async def _mark_decision_executed(self, decision_id: int, execution_price: float) -> None:
        await self.decision_persistence.mark_executed(decision_id, execution_price)

    def _execution_reason_is_unusable(self, reason: Any) -> bool:
        return self.decision_persistence.execution_reason_is_unusable(reason)

    def _recover_execution_reason_from_decision_row(
        self,
        decision: AIDecision | None,
        fallback: Any = None,
    ) -> str | None:
        return self.decision_reason_recovery.recover(decision, fallback)

    def _record_decision_stage(
        self,
        decision: DecisionOutput,
        stage: str,
        status: str,
        reason: str | None,
        data: dict[str, Any] | None = None,
        *,
        duration_sec: float | None = None,
    ) -> dict[str, Any]:
        return self.decision_persistence.record_stage(
            decision,
            stage,
            status,
            reason,
            data,
            duration_sec=duration_sec,
        )

    async def _record_and_persist_decision_stage(
        self,
        decision_id: int | None,
        decision: DecisionOutput,
        stage: str,
        status: str,
        reason: str | None,
        data: dict[str, Any] | None = None,
        duration_sec: float | None = None,
    ) -> dict[str, Any]:
        return await self.decision_persistence.record_and_persist_stage(
            decision_id=decision_id,
            decision=decision,
            stage=stage,
            status=status,
            reason=reason,
            data=data,
            duration_sec=duration_sec,
        )

    async def _record_decision_reason_strategy_event(
        self,
        decision_id: int,
        reason: str | None,
    ) -> None:
        try:
            async with get_session_ctx() as session:
                row = await session.get(AIDecision, int(decision_id))
                if row is None:
                    return
                raw = self._safe_dict(row.raw_llm_response)
                strategy_context = self._safe_dict(raw.get("strategy_learning_context"))
                text = str(reason or "")
                lower = text.lower()
                status = "skipped"
                severity = "info"
                event_type = "decision_reason"
                if any(
                    token in lower
                    for token in ("reject", "rejected", "拒绝", "拦截", "block", "blocked")
                ):
                    status = "blocked"
                    severity = "warn"
                    event_type = "decision_blocked"
                if any(
                    token in lower for token in ("expert_integrity", "fallback", "partial_batch")
                ):
                    event_type = "expert_fallback"
                    severity = "warn"
                if any(
                    token in lower for token in ("capacity", "max_position", "仓位", "满仓", "限制")
                ):
                    event_type = "capacity_block"
                    severity = "warn"
                if any(
                    token in lower
                    for token in ("error", "failed", "异常", "失败", "timeout", "超时")
                ):
                    status = "failed"
                    severity = "error"
                await self._record_strategy_learning_event(
                    mode="paper" if bool(row.is_paper) else "live",
                    model_name=row.model_name or ENSEMBLE_TRADER_NAME,
                    symbol=row.symbol,
                    action=row.action,
                    event_type=event_type,
                    event_status=status,
                    reason=text,
                    severity=severity,
                    decision_id=decision_id,
                    strategy_context=strategy_context,
                    raw_response=raw,
                    attribution={"source": "decision_reason", "execution_reason": text},
                )
        except Exception as exc:
            logger.debug(
                "failed to record strategy decision reason event",
                decision_id=decision_id,
                error=safe_error_text(exc),
            )

    async def _mark_decision_reason(self, decision_id: int, reason: str | None) -> None:
        await self.decision_persistence.mark_reason(
            decision_id,
            reason,
            reason_recoverer=self._recover_execution_reason_from_decision_row,
        )
        await self._record_decision_reason_strategy_event(decision_id, reason)

    async def _mark_decision_pending_execution(self, decision_id: int, reason: str) -> None:
        """Mark an entry as in-flight without letting final-round fallback overwrite it."""
        await self.decision_persistence.mark_pending_execution(decision_id, reason)

    async def _duplicate_decision_order_reason(
        self,
        decision_id: int,
        decision: DecisionOutput,
    ) -> str | None:
        """Prevent the same decision row from submitting more than one OKX order."""
        return await self.decision_persistence.duplicate_order_reason(decision_id, decision)

    async def _mark_decision_raw_response(
        self, decision_id: int, raw_response: dict | None
    ) -> None:
        await self.decision_persistence.update_raw_response(decision_id, raw_response)

    async def _fill_missing_decision_reasons(
        self,
        decision_ids: set[int] | list[int],
        reason: str,
    ) -> None:
        await self.decision_persistence.fill_missing_reasons(decision_ids, reason)

    async def _finalize_round_unresolved_decisions(
        self,
        decision_ids: set[int] | list[int],
        decisions: dict[int, DecisionOutput],
        reason: str,
    ) -> None:
        executable_decisions = self._round_unresolved_executable_decisions(decisions)
        if not executable_decisions:
            return
        await self._finalize_unresolved_decision_states(executable_decisions, reason)

    @staticmethod
    def _round_unresolved_executable_decisions(
        decisions: dict[int, DecisionOutput],
    ) -> dict[int, DecisionOutput]:
        """Only executable actions need a round-end terminal state.

        Hold decisions and fast-prefilter observations are already terminal from
        a trading perspective.  Finalizing them as "did not enter order flow"
        pollutes dashboard reasons and teaches strategy learning that passive
        observations were failed entry candidates.
        """

        return {
            int(decision_id): decision
            for decision_id, decision in decisions.items()
            if decision_id
            and (decision.is_entry or decision.is_exit)
            and not TradingService._decision_has_terminal_state(decision)
        }

    @staticmethod
    def _decision_has_terminal_state(decision: DecisionOutput) -> bool:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        summary = decision_state_from_raw(raw).get("summary")
        if not isinstance(summary, dict):
            return False
        return is_decision_terminal_state(
            summary.get("final_stage"),
            summary.get("final_status"),
        )

    async def _finalize_unresolved_decision_states(
        self,
        decisions: dict[int, DecisionOutput],
        reason: str,
    ) -> None:
        await self.decision_persistence.finalize_unresolved_decisions(decisions, reason)

    def _execution_reason_from_result(self, result: ExecutionResult | None) -> str:
        return self.execution_result_classifier.reason_from_result(result)

    def _translate_execution_error_text(self, text: str | None) -> str | None:
        return self.execution_result_classifier.translate_execution_error_text(text)

    def _is_exit_tracking_execution(self, result: ExecutionResult | None) -> bool:
        return self.execution_result_classifier.is_exit_tracking_execution(result)

    def _is_exit_progress_execution(self, result: ExecutionResult | None) -> bool:
        return self.execution_result_classifier.is_exit_progress_execution(result)

    def _is_exchange_confirmed_execution(self, result: ExecutionResult | None) -> bool:
        """Only treat an execution as real after OKX returns a concrete order id."""

        return self.execution_result_classifier.is_exchange_confirmed_execution(result)

    async def _mark_decision_outcome(self, decision_id: int, outcome: str, pnl_pct: float) -> None:
        await self.decision_persistence.mark_outcome(decision_id, outcome, pnl_pct)

    async def _publish_dashboard_update(self, results: dict) -> None:
        """Publish a dashboard update via Redis pub/sub."""
        if self.redis:
            try:
                import json

                await self.redis.publish(
                    "dashboard:update",
                    json.dumps(
                        {
                            "type": "trading_round",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "mode": mode_manager.mode.value,
                            "decisions": results.get("decisions", []),
                            "executions": results.get("executions", []),
                            "stats": self.get_stats(),
                        },
                        default=str,
                    ),
                )
            except Exception as exc:
                logger.debug("dashboard redis publish failed", error=safe_error_text(exc))

    async def record_equity_snapshot(self) -> None:
        """Record OKX account equity for the in-memory PnL chart."""
        now = datetime.now(UTC).isoformat()
        active_names = {ENSEMBLE_TRADER_NAME}
        # Remove stale data from deleted models
        for stale in list(self._pnl_history.keys()):
            if stale not in active_names:
                del self._pnl_history[stale]
        for model_name in active_names:
            snapshot = await self._get_okx_balance_snapshot_for_mode(mode_manager.mode.value)
            equity = balance_from_snapshot(snapshot)
            if equity <= 0:
                logger.warning(
                    "skip equity snapshot because OKX equity is unavailable",
                    model=model_name,
                    mode=mode_manager.mode.value,
                )
                continue
            self._pnl_history.setdefault(model_name, []).append(
                {
                    "time": now,
                    "equity": round(equity, 2),
                }
            )
            # Keep last 500 snapshots per model
            if len(self._pnl_history[model_name]) > 500:
                self._pnl_history[model_name] = self._pnl_history[model_name][-500:]

    def get_pnl_history(self) -> dict[str, list[dict]]:
        """Return PnL equity history only for currently active models."""
        active_names = {ENSEMBLE_TRADER_NAME}
        return {
            name: snapshots for name, snapshots in self._pnl_history.items() if name in active_names
        }

    def get_stats(self, mode_filter: str | None = None) -> dict[str, Any]:
        settings.refresh_runtime_env(force=True)
        uptime = (datetime.now(UTC) - self._start_time).total_seconds() if self._start_time else 0
        now = datetime.now(UTC)
        market_state = self._runtime_state("market")
        position_state = self._runtime_state("position")
        active_scoped_states = [state for state in (market_state, position_state) if state.active]
        round_active = bool(active_scoped_states) or (
            self._last_round_started_at is not None
            and (
                self._last_round_finished_at is None
                or self._last_round_finished_at < self._last_round_started_at
            )
        )
        current_state = (
            max(
                active_scoped_states,
                key=lambda state: state.last_started_at or datetime.min.replace(tzinfo=UTC),
            )
            if active_scoped_states
            else self._runtime_state(_analysis_scope_context.get())
        )
        round_running_seconds = (
            int((now - current_state.last_started_at).total_seconds())
            if current_state.active and current_state.last_started_at
            else 0
        )
        stage_labels = {
            "idle": "\u7a7a\u95f2\uff0c\u7b49\u5f85\u4e0b\u4e00\u8f6e\u5206\u6790",
            "starting": "\u51c6\u5907\u5f00\u59cb\u672c\u8f6e\u5206\u6790",
            "shadow_backtests": "\u66f4\u65b0\u5f71\u5b50\u590d\u76d8",
            "stale_entry_maintenance": "\u6e05\u7406\u8fc7\u671f\u5f00\u4ed3\u5019\u9009",
            "sync_exchange_positions": "\u540c\u6b65 OKX \u4ed3\u4f4d/\u4fdd\u62a4\u5355",
            "load_open_positions": "\u8bfb\u53d6\u672c\u5730\u6301\u4ed3",
            "recover_pending_exits": "\u8865\u6267\u884c\u672a\u5b8c\u6210\u5e73\u4ed3",
            "new_pair_pause_check": "\u68c0\u67e5\u65b0\u5f00\u4ed3\u6682\u505c\u72b6\u6001",
            "record_new_pair_pause_state": "\u8bb0\u5f55\u65b0\u5f00\u4ed3\u6682\u505c\u72b6\u6001",
            "select_symbols": "\u7b5b\u9009\u672c\u8f6e\u5206\u6790\u5e01\u79cd",
            "fetch_features": "\u83b7\u53d6\u884c\u60c5\u6307\u6807",
            "refresh_position_prices": "\u5237\u65b0\u6301\u4ed3\u4ef7\u683c",
            "enforce_sl_tp": "\u68c0\u67e5\u6b62\u76c8\u6b62\u635f",
            "review_open_positions": "\u590d\u76d8\u5f53\u524d\u6301\u4ed3",
            "publish_results": "\u5199\u5165\u5e76\u63a8\u9001\u5206\u6790\u7ed3\u679c",
            "error": "\u672c\u8f6e\u5f02\u5e38",
        }
        stage_label = stage_labels.get(current_state.current_stage)
        if stage_label is None and current_state.current_stage.startswith("analyze:"):
            stage_label = f"\u6b63\u5728\u5206\u6790 {current_state.current_stage.split(':', 1)[1]}"
        elif stage_label is None and current_state.current_stage.startswith("execute:"):
            stage_label = f"\u6b63\u5728\u6267\u884c {current_state.current_stage.split(':', 1)[1]} \u8ba2\u5355"
        elif stage_label is None:
            stage_label = current_state.current_stage

        # Filter decisions/execs by mode if requested
        is_paper_filter = None if mode_filter is None else (mode_filter == "paper")
        if is_paper_filter is not None:
            recent_decs = [
                d for d in self._recent_decisions if d.get("is_paper", True) == is_paper_filter
            ]
            recent_execs = [
                e for e in self._recent_executions if e.get("is_paper", True) == is_paper_filter
            ]
        else:
            recent_decs = self._recent_decisions
            recent_execs = self._recent_executions

        stats = {
            "running": self._running,
            "mode": mode_manager.mode.value,
            "paused": mode_manager.is_paused,
            "uptime_seconds": int(uptime),
            "started_at": self._start_time.isoformat() if self._start_time else None,
            "heartbeat_at": now.isoformat(),
            "last_heartbeat_at": now.isoformat(),
            "decisions_total": self._decision_count,
            "trades_total": self._trade_count,
            "recent_decisions": recent_decs,
            "recent_executions": recent_execs,
            "current_stage": current_state.current_stage,
            "current_stage_label": stage_label,
            "round_active": round_active,
            "round_running_seconds": round_running_seconds,
            "market_analysis_watchdog_seconds": int(self.market_round_watchdog_seconds()),
            "position_analysis_watchdog_seconds": int(self.position_round_watchdog_seconds()),
            "market_current_stage": market_state.current_stage,
            "market_round_active": market_state.active,
            "market_last_error": market_state.last_error,
            "position_current_stage": position_state.current_stage,
            "position_round_active": position_state.active,
            "position_last_error": position_state.last_error,
                "market_stage_durations": self._recent_stage_durations(market_state),
                "position_stage_durations": self._recent_stage_durations(position_state),
                "last_position_price_refresh_diagnostics": self._safe_dict(
                    getattr(self, "_last_position_price_refresh_diagnostics", {})
                ),
            "last_round_started_at": (
                self._last_round_started_at.isoformat() if self._last_round_started_at else None
            ),
            "last_round_finished_at": (
                self._last_round_finished_at.isoformat() if self._last_round_finished_at else None
            ),
            "last_market_round_started_at": (
                market_state.last_started_at.isoformat() if market_state.last_started_at else None
            ),
            "last_market_round_finished_at": (
                market_state.last_finished_at.isoformat() if market_state.last_finished_at else None
            ),
            "last_position_round_started_at": (
                position_state.last_started_at.isoformat()
                if position_state.last_started_at
                else None
            ),
            "last_position_round_finished_at": (
                position_state.last_finished_at.isoformat()
                if position_state.last_finished_at
                else None
            ),
            "last_round_error": (
                market_state.last_error or position_state.last_error or self._last_round_error
            ),
            "live_model": ENSEMBLE_TRADER_NAME,
            "models": [ENSEMBLE_TRADER_NAME],
            "risk": self.risk_engine.circuit_breaker.get_state(),
            "decision_interval": settings.decision_interval_seconds,
            "market_loop_interval_seconds": round(self.market_loop_interval_seconds(), 3),
            "position_loop_interval_seconds": round(self.position_loop_interval_seconds(), 3),
            "market_round_time_budget_seconds": round(self.market_round_time_budget_seconds(), 3),
            "okx_authoritative_sync": self._okx_authoritative_sync_status_payload(now),
            "shadow_backtest_maintenance": self._shadow_backtest_maintenance_status(),
            "stale_entry_maintenance": self._stale_entry_candidate_maintenance_status(),
        }
        self._write_runtime_heartbeat()
        return stats
        return stats
