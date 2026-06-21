"""
Trading service: the central orchestrator.
Wires together data feed, AI brain, risk manager, and executor
into the main trading loop.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
    tradeable_balance_from_snapshot,
)
from services.analysis_budget import (
    POSITION_REVIEW_MAX_GROUPS_PER_ROUND,
    POSITION_REVIEW_URGENT_EXIT_MARKERS,
    AnalysisBudgetPolicy,
)
from services.analysis_services import MarketAnalysisService, PositionReviewService
from services.daily_performance_service import DailyPerformanceService
from services.daily_side_performance import DailySidePerformanceService
from services.data_service import DataService
from services.decision_final_state_ensurer import DecisionFinalStateEnsurer
from services.decision_freshness import DecisionFreshnessPolicy
from services.decision_persistence_service import DecisionPersistenceService
from services.decision_reason_recovery import DecisionReasonRecoveryPolicy
from services.dynamic_position_capacity import DynamicPositionCapacityPolicy
from services.entry_candidate_evidence import EntryCandidateEvidencePolicy
from services.entry_candidate_filter import EntryCandidateFilterPolicy
from services.entry_candidate_queue import EntryCandidateQueuePolicy
from services.entry_capacity import EntryCapacityPolicy
from services.entry_crowded_side_cap import EntryCrowdedSideCapPolicy
from services.entry_direction_competition import EntryDirectionCompetitionPolicy
from services.entry_evidence_probe import EntryEvidenceProbePolicy
from services.entry_existing_winner import EntryExistingWinnerContextPolicy
from services.entry_feature_ranker import EntryFeatureRankerPolicy
from services.entry_fee_provider import EntryFeeProvider
from services.entry_high_risk_review import EntryHighRiskReviewGatePolicy
from services.entry_immediate_execution import EntryImmediateExecutionPlanner
from services.entry_loss_cooldown import EntryLossCooldownPolicy
from services.entry_market_data_quality import (
    EntryMarketDataQualityPolicy,
    MarketValueReader,
)
from services.entry_market_hold_penalty import EntryMarketHoldPenaltyPolicy
from services.entry_market_prefilter import EntryMarketLLMPrefilterPolicy
from services.entry_market_regime import EntryMarketRegimeContextPolicy, EntryMarketRegimePolicy
from services.entry_opportunity_gate import EntryOpportunityGatePolicy
from services.entry_opportunity_scoring import EntryOpportunityScoringPolicy
from services.entry_payoff_quality import EntryLowPayoffQualityPolicy
from services.entry_position_exposure import EntryPositionExposurePolicy
from services.entry_post_crash_rebound import EntryPostCrashReboundGuardPolicy
from services.entry_price_guard import EntryPriceGuardPolicy
from services.entry_priority import EntryExecutionPriorityPolicy
from services.entry_probe_market_quality import EntryProbeMarketQualityPolicy
from services.entry_profit_risk_sizing import EntryProfitRiskSizingPolicy
from services.entry_quant_profit_probe import EntryQuantProfitProbePolicy
from services.entry_stop_loss_budget import (
    EntryStopLossBudgetPolicy,
)
from services.entry_strategy_mode import EntryStrategyModeContextPolicy
from services.entry_stress_stop import EntryStressStopPolicy
from services.entry_suspicious_symbol import EntrySuspiciousSymbolPolicy
from services.entry_symbol_blocklist import (
    PRICE_GUARD_ENTRY_BLOCK_MINUTES,
    TRANSIENT_ENTRY_BLOCK_MINUTES,
    UNTRADABLE_SYMBOL_BLOCK_HOURS,
    EntrySymbolBlocklistPolicy,
)
from services.entry_symbol_profit_quarantine import EntrySymbolProfitQuarantinePolicy
from services.entry_symbol_universe import EntrySymbolUniversePolicy
from services.entry_symbol_winner import EntrySymbolWinnerDecayPolicy
from services.entry_wick_guard import (
    EntryAbnormalWickGuardPolicy,
)
from services.exchange_backed_position_provider import ExchangeBackedPositionProvider
from services.exchange_close_fill_finder import ExchangeCloseFillFinder
from services.exchange_position_state import (
    ExchangePositionStatePolicy,
    ExchangeProtectionMapProvider,
)
from services.execution_allocation_service import ExecutionAllocationService
from services.execution_pipelines import EntryExecutionPipeline, ExitExecutionPipeline
from services.execution_result_classifier import ExecutionResultClassifier
from services.execution_result_factory import ExecutionResultFactory
from services.execution_service import ExecutionService
from services.exit_arbitrator import ExitArbitrator
from services.exit_cooldown import ExitCooldownPolicy
from services.exit_fast_risk import (
    FAST_RISK_1M_MOVE_PCT,
    FAST_RISK_5M_MOVE_PCT,
    FAST_RISK_FULL_STOP_PROGRESS,
    FAST_RISK_MAX_FEATURE_POSITION_PRICE_GAP,
    FAST_RISK_NEAR_STOP_PROGRESS,
    ExitFastRiskPolicy,
)
from services.exit_fee_churn_guard import ExitFeeChurnGuardPolicy
from services.exit_invalidation_snapshot import ExitInvalidationSnapshotPolicy
from services.exit_partial_guard import ExitPartialGuardPolicy
from services.exit_position_matcher import ExitPositionMatcher
from services.exit_position_snapshot import ExitPositionSnapshotPolicy
from services.exit_predictive_reversal import (
    PREDICTIVE_REVERSAL_EXIT_SCORE,
    ExitPredictiveReversalPolicy,
)
from services.exit_profit_precheck import ExitProfitPrecheckPolicy
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
from services.ml_signal_service import AUTO_TRAIN_CHECK_INTERVAL_SECONDS, MLSignalService
from services.model_contribution_performance import ModelContributionPerformanceService
from services.new_pair_loss_pause import NewPairLossPausePolicy
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
from services.position_quality import PositionQualityScorer
from services.runtime_entry_filters import RuntimeEntryFilters, entry_filters_from_context
from services.position_release_decision import PositionReleaseDecisionPolicy
from services.position_review_batch import PositionReviewBatchPolicy
from services.position_review_decision_normalizer import PositionReviewDecisionNormalizer
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
from services.position_snapshot_syncer import PositionSnapshotSyncer
from services.position_time import PositionTimeParser
from services.shadow_backtest_service import ShadowBacktestService
from services.stale_entry_candidate_expirer import StaleEntryCandidateExpirer
from services.strategy_learning import StrategyLearningService
from services.symbol_side_performance import SymbolSidePerformanceService
from services.sync_service import OkxSyncService
from services.trade_order_log_service import TradeOrderLogService
from services.trading_agent_skills import TradingAgentSkillBook
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.trading_policies import EntryPolicy, ExitPolicy
from services.vector_memory import get_vector_memory_service
from web_dashboard.api.text_sanitize import sanitize_text

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class _AnalysisRuntimeState:
    """Runtime status for one independent analysis loop."""

    current_stage: str = "idle"
    last_started_at: datetime | None = None
    last_finished_at: datetime | None = None
    last_error: str | None = None

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
        except (asyncio.CancelledError, Exception):
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
OKX_BALANCE_SNAPSHOT_CACHE_SECONDS = 120.0
MIN_DISCRETIONARY_HOLD_MINUTES = 4.0
ENTRY_SETTLEMENT_EXIT_GUARD_SECONDS = 120.0
DISCRETIONARY_CLOSE_CONFIDENCE = 0.66
PROFIT_PROTECTION_MIN_NET_PNL_RATIO = 0.004
PROFIT_PROTECTION_STRONG_NET_PNL_RATIO = 0.010
PROFIT_PROTECTION_MIN_NET_USDT = 3.00
PROFIT_PROTECTION_MIN_FEE_MULTIPLE = 4.0
PROFIT_PROTECTION_STRONG_FEE_MULTIPLE = 5.0
UNCONFIRMED_EXCHANGE_CLOSE_GRACE_SECONDS = 180.0
ENTRY_PRICE_RECHECK_TIMEOUT_SECONDS = (
    DEFAULT_TRADING_PARAMS.entry_price_guard.recheck_timeout_seconds
)
ENTRY_BLACK_SWAN_RECHECK_SAFE_1M_DROP = -0.08
ENTRY_BLACK_SWAN_RECHECK_SAFE_5M_DROP = -0.12
ENTRY_BLACK_SWAN_REBOUND_MIN_1M = 0.001
ENTRY_BLACK_SWAN_REBOUND_MIN_5M = 0.003
ENTRY_BLACK_SWAN_REBOUND_MIN_EXPECTED_NET = 0.60
ENTRY_BLACK_SWAN_REBOUND_MIN_PROFIT_QUALITY = 1.00
ENTRY_BLACK_SWAN_REBOUND_MIN_CONFIDENCE = 0.72
ENTRY_BLACK_SWAN_REBOUND_MAX_SPREAD = 0.015
QUANT_PROFIT_PROBE_PARAMS = DEFAULT_TRADING_PARAMS.entry_quant_profit_probe
QUANT_PROFIT_PROBE_MIN_PROFIT_QUALITY_RATIO = QUANT_PROFIT_PROBE_PARAMS.min_profit_quality_ratio
SYMBOL_SIDE_PROFILE_LOOKBACK = 2000
SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS = 7.0
ENTRY_LOSS_COOLDOWN_PARAMS = DEFAULT_TRADING_PARAMS.entry_loss_cooldown
LOCAL_ML_TRAINING_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training
AUTO_SCAN_PARAMS = DEFAULT_TRADING_PARAMS.auto_scan
AUTO_TRADE_MIN_NOTIONAL_24H = 4_000_000.0
AUTO_TRADE_MIN_VOLUME_RATIO = 0.65
AUTO_TRADE_MAX_VOLATILITY_20 = 0.085
AUTO_TRADE_MAX_ABS_CHANGE_24H = 12.0
ABNORMAL_WICK_TAIL_RISK_MAX_PCT = 60.0
AUTO_SCAN_ROTATION_POOL_MULTIPLIER = AUTO_SCAN_PARAMS.rotation_pool_multiplier
AUTO_SCAN_ROTATION_POOL_MIN = AUTO_SCAN_PARAMS.rotation_pool_min
AUTO_SCAN_FEATURE_FETCH_POOL_MULTIPLIER = AUTO_SCAN_PARAMS.feature_fetch_pool_multiplier
AUTO_SCAN_FEATURE_FETCH_POOL_MIN = AUTO_SCAN_PARAMS.feature_fetch_pool_min
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
    ) -> None:
        self.models = model_registry
        self.ensemble = EnsembleCoordinator(model_registry)
        self.data_service = data_service
        self.position_quality_scorer = PositionQualityScorer()
        self.dynamic_capacity = DynamicPositionCapacityPolicy(
            lambda: settings.max_open_positions_per_model,
            quality_scorer=self.position_quality_scorer,
        )
        self._current_capacity_context = self.dynamic_capacity.evaluate(open_positions=[]).as_dict()
        self._current_strategy_mode_context: dict[str, Any] = {}
        self.risk_engine = RiskEngine(max_open_positions_provider=self._dynamic_capacity_context)
        self.market_decision_risk_assessment = MarketDecisionRiskAssessmentPolicy(
            risk_engine=self.risk_engine,
            account_balance_provider=self.get_account_balance,
            false_positive_checker=self._price_action_black_swan_false_positive,
        )
        self.manual_trade_risk_assessment = ManualTradeRiskAssessmentPolicy(self.risk_engine)
        self.manual_trade_execution_processor = ManualTradeExecutionProcessor(
            decision_logger=self._log_decision,
            decision_count_incrementer=self.increment_decision_count,
            candidate_executor=self._execute_candidate,
            is_paper_provider=lambda: mode_manager.is_paper,
        )
        self.ml_signal_service = MLSignalService()
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
        self.entry_symbol_blocklist = EntrySymbolBlocklistPolicy(self._normalize_position_symbol)
        self.execution_result_classifier = ExecutionResultClassifier(
            untradable_exchange_error_checker=self.is_untradable_exchange_error
        )
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
            untradable_exchange_error_checker=self.is_untradable_exchange_error,
            untradable_symbol_rememberer=self.remember_untradable_symbol,
            transient_entry_exchange_error_checker=self.is_transient_entry_exchange_error,
            temporary_entry_block_rememberer=self.remember_temporary_entry_block,
            transient_entry_block_minutes_provider=self.transient_entry_block_minutes,
            trade_logger=self.log_trade,
            exchange_confirmed_checker=self.is_exchange_confirmed_execution,
            exit_progress_checker=self.is_exit_progress_execution,
            no_exchange_position_result_checker=self.result_has_no_exchange_position,
            trade_count_incrementer=self.increment_trade_count,
            position_execution_persister=self.persist_position_from_execution,
            open_positions_execution_applier=self.apply_execution_to_open_positions,
            decision_executed_marker=self.mark_decision_executed,
            market_no_opportunity_symbol_clearer=self.clear_market_no_opportunity_symbol,
            account_update_persister=self.persist_account_update,
            account_balance_provider=self.get_account_balance,
            decision_outcome_marker=self.mark_decision_outcome,
            entry_policy_evaluator=self.entry_execution_pipeline.evaluate,
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
            exit_cooldown_recorder=self.remember_exit_cooldown,
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
        self.position_review_decision_normalizer = PositionReviewDecisionNormalizer(
            self._normalize_position_symbol
        )
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
            max_slippage_pct_provider=lambda: float(settings.max_slippage_pct or 0.005),
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
            loss_cooldown_params=ENTRY_LOSS_COOLDOWN_PARAMS,
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
            paper_positions_provider=self.paper_positions_for_context,
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
        self.exit_cooldown = ExitCooldownPolicy(self._normalize_position_symbol)
        self.exit_position_matcher = ExitPositionMatcher(self._normalize_position_symbol)
        self.exit_position_snapshot = ExitPositionSnapshotPolicy(self.okx_sync_service)
        self.exit_profit_precheck = ExitProfitPrecheckPolicy(
            self._latest_price_for_symbol,
            self._normalize_position_symbol,
            min_net_usdt=PROFIT_PROTECTION_MIN_NET_USDT,
        )
        self.exit_partial_guard = ExitPartialGuardPolicy(self.exit_position_matcher)
        self.exit_invalidation_snapshot = ExitInvalidationSnapshotPolicy(
            self._runtime_min_entry_volume_ratio
        )
        self.forced_exit_policy = ForcedExitPolicy()
        self.exit_fee_churn_guard = ExitFeeChurnGuardPolicy(
            session_factory=get_session_ctx,
            model_execution_mode_provider=self._get_model_execution_mode,
            entry_fee_provider=self.entry_fee_provider.entry_fee_for_position,
            invalidation_snapshot_provider=self.exit_invalidation_snapshot.snapshot,
            forced_exit_policy=self.forced_exit_policy,
            position_peaks=self.position_profit_peaks.peaks,
            position_peak_key_provider=self.position_profit_peaks.key,
        )
        self.exit_arbitrator = ExitArbitrator()
        self.exit_predictive_reversal = ExitPredictiveReversalPolicy()
        self.portfolio_profit_protection = PortfolioProfitProtectionPolicy(
            normalize_symbol=self._normalize_position_symbol,
            default_model_name=ENSEMBLE_TRADER_NAME,
        )
        self.position_review_priority = PositionReviewPriorityPolicy(
            normalize_symbol=self._normalize_position_symbol,
            position_peak_key=self.position_profit_peaks.key,
            position_peaks_provider=lambda: self.position_profit_peaks.peaks,
            predictive_reversal=self.exit_predictive_reversal,
            quality_scorer=self.position_quality_scorer,
            urgent_exit_markers=POSITION_REVIEW_URGENT_EXIT_MARKERS,
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
        self.exit_fast_risk = ExitFastRiskPolicy(
            predictive_reversal=self.exit_predictive_reversal,
            seconds_since_profit_exit=self.position_profit_peaks.seconds_since_profit_exit,
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
        self.entry_symbol_winner_decay = EntrySymbolWinnerDecayPolicy()
        self.entry_symbol_profit_quarantine = EntrySymbolProfitQuarantinePolicy(
            self._normalize_position_symbol
        )
        self.model_contribution_performance_service = ModelContributionPerformanceService(
            lookback_days=SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS,
        )
        self.entry_opportunity_score = EntryOpportunityScoringPolicy(
            normalize_symbol=self._normalize_position_symbol,
            model_contribution_score_adjustment=(
                self.model_contribution_performance_service.score_adjustment
            ),
            annotate_decision_source=self._annotate_decision_source,
            entry_symbol_winner_decay=self.entry_symbol_winner_decay,
        )
        self.entry_capacity = EntryCapacityPolicy(
            self._normalize_position_symbol,
            self._dynamic_capacity_context,
        )
        self.entry_position_exposure = EntryPositionExposurePolicy()
        self.entry_market_regime = EntryMarketRegimePolicy(
            self._normalize_position_symbol,
            ALT_LONG_ALLOWED_SYMBOLS,
        )
        self.entry_market_regime_context = EntryMarketRegimeContextPolicy(
            self._is_valid_feature_vector,
        )
        self.entry_direction_competition = EntryDirectionCompetitionPolicy()
        self.entry_strategy_mode_context = EntryStrategyModeContextPolicy()
        self.strategy_learning_service = StrategyLearningService()
        self.entry_suspicious_symbol = EntrySuspiciousSymbolPolicy(self._normalize_position_symbol)
        self.entry_feature_ranker = EntryFeatureRankerPolicy(
            suspicious_symbol_reason=self.entry_suspicious_symbol.reason,
            min_entry_volume_ratio_provider=self._runtime_min_entry_volume_ratio,
            min_entry_adx_provider=self._runtime_min_entry_adx,
            major_symbols=frozenset(ALT_LONG_ALLOWED_SYMBOLS),
        )
        self.entry_market_hold_penalty = EntryMarketHoldPenaltyPolicy(
            normalize_symbol=self._normalize_position_symbol,
            feature_opportunity_score=self._feature_opportunity_score,
            min_entry_volume_ratio_provider=self._runtime_min_entry_volume_ratio,
            min_entry_adx_provider=self._runtime_min_entry_adx,
        )
        self.entry_loss_cooldown = EntryLossCooldownPolicy(self._normalize_position_symbol)
        self.entry_post_crash_rebound_guard = EntryPostCrashReboundGuardPolicy()
        self.entry_probe_market_quality = EntryProbeMarketQualityPolicy()
        self.entry_evidence_probe = EntryEvidenceProbePolicy(
            ENSEMBLE_TRADER_NAME,
            lambda: settings.max_leverage,
            self.entry_probe_market_quality,
        )
        self.entry_crowded_side_cap = EntryCrowdedSideCapPolicy()
        self.entry_opportunity_gate = EntryOpportunityGatePolicy(
            suspicious_symbol_policy=self.entry_suspicious_symbol,
            symbol_loss_cooldown_policy=self.entry_loss_cooldown,
            crowded_side_cap_policy=self.entry_crowded_side_cap,
            post_crash_rebound_guard=self.entry_post_crash_rebound_guard,
        )
        self.entry_low_payoff_quality = EntryLowPayoffQualityPolicy()
        self.entry_stress_stop = EntryStressStopPolicy()
        self.entry_stop_loss_budget = EntryStopLossBudgetPolicy()
        self.entry_existing_winner_context = EntryExistingWinnerContextPolicy(
            self._normalize_position_symbol
        )
        self.entry_profit_risk_sizing = EntryProfitRiskSizingPolicy(
            allocated_order_balance=self.allocated_order_balance,
            entry_low_payoff_quality=self.entry_low_payoff_quality,
            entry_stop_loss_budget=self.entry_stop_loss_budget,
            entry_stress_stop=self.entry_stress_stop,
            entry_existing_winner_context=self.entry_existing_winner_context,
            max_leverage_provider=lambda: settings.max_leverage,
        )
        self.new_pair_loss_pause = NewPairLossPausePolicy(
            balance_snapshot_provider=self._get_okx_balance_snapshot_for_mode,
        )
        self.shadow_backtest_service = ShadowBacktestService(
            latest_price_provider=self._latest_price_for_symbol,
            symbol_normalizer=self._normalize_position_symbol,
            float_parser=self._safe_float,
        )
        self.stale_entry_candidate_expirer = StaleEntryCandidateExpirer(self._safe_float)
        self.decision_final_state_ensurer = DecisionFinalStateEnsurer(
            execution_reason_unusable_checker=self._execution_reason_is_unusable,
            execution_reason_recoverer=self._recover_execution_reason_from_decision_row,
            model_execution_mode_provider=self._get_model_execution_mode,
        )
        self.entry_price_guard = EntryPriceGuardPolicy(
            latest_price_provider=self._latest_price_for_symbol,
            fresh_feature_provider=self._fresh_feature_vector_for_price_recheck,
            market_data_quality_reason_provider=self.entry_market_data_quality.reason,
            decision_age_seconds_provider=self.decision_freshness.decision_age_seconds,
            temporary_entry_block_recorder=self.remember_temporary_entry_block,
            temporary_block_minutes=PRICE_GUARD_ENTRY_BLOCK_MINUTES,
        )
        self.abnormal_wick_guard = EntryAbnormalWickGuardPolicy(
            self.remember_temporary_entry_block,
            temporary_block_minutes=PRICE_GUARD_ENTRY_BLOCK_MINUTES,
        )
        self.high_risk_review_gate_policy = EntryHighRiskReviewGatePolicy(
            reviewer=self.high_risk_review_service,
            allocation_state_provider=self.execution_allocation_state,
            quant_probe_min_profit_quality_ratio=QUANT_PROFIT_PROBE_MIN_PROFIT_QUALITY_RATIO,
        )
        self.entry_policy = EntryPolicy(
            decision_freshness=self.decision_freshness,
            entry_priority=self.entry_priority,
            entry_opportunity_score=self.entry_opportunity_score,
            entry_profit_risk_sizing=self.entry_profit_risk_sizing,
            abnormal_wick_guard=self.abnormal_wick_guard,
            entry_price_guard=self.entry_price_guard,
            entry_opportunity_gate=self.entry_opportunity_gate,
            high_risk_review_gate=self.high_risk_review_gate_policy,
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
            clear_market_no_opportunity_symbol=self._clear_market_no_opportunity_symbol,
            set_loop_stage=self._set_loop_stage,
            candidate_executor=self._execute_candidate,
            final_state_ensurer=self.decision_final_state_ensurer.ensure,
            capacity_releaser=self.entry_capacity.release_slot,
            execution_confirmed_checker=self._is_exchange_confirmed_execution,
        )
        self.market_direct_entry_processor = MarketDirectEntryProcessor(
            capacity_reason_provider=self.entry_capacity.reason,
            capacity_reserver=self.entry_capacity.reserve_slot,
            annotate_candidate_selection=self._annotate_candidate_selection,
            mark_decision_raw_response=self._mark_decision_raw_response,
            mark_decision_reason=self._mark_decision_reason,
            result_recorder=self.market_decision_result_recorder,
            clear_market_no_opportunity_symbol=self._clear_market_no_opportunity_symbol,
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
        self.entry_quant_profit_probe = EntryQuantProfitProbePolicy(
            self.entry_probe_market_quality,
            self.entry_policy.score_candidate,
        )
        self.exit_policy = ExitPolicy(
            exit_cooldown=self.exit_cooldown,
            decision_freshness=self.decision_freshness,
            exit_position_matcher=self.exit_position_matcher,
            exit_partial_guard=self.exit_partial_guard,
            exit_position_snapshot=self.exit_position_snapshot,
            exit_profit_precheck=self.exit_profit_precheck,
            exit_fee_churn_guard=self.exit_fee_churn_guard,
            exit_arbitrator=self.exit_arbitrator,
        )
        self.position_review_decision_processor = PositionReviewDecisionProcessor(
            entry_guard=self.position_review_entry_guard,
            entry_capacity=self.entry_capacity,
            risk_assessment=self.position_review_risk_assessment,
            result_recorder=self.position_review_result_recorder,
            exit_fee_guard_reason_provider=self.exit_policy.fee_churn_guard_reason,
            candidate_executor=self._execute_candidate,
            final_state_ensurer=self.decision_final_state_ensurer.ensure,
            account_balance_provider=self.get_account_balance,
        )

        # Executors: paper routes to OKX demo, live routes to OKX real.
        self.paper_executor: PaperExecutor | None = None
        self.okx_executor: OKXExecutor | None = None  # kept for backward compat
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
        self._position_review_cursor = 0
        self._position_review_priority_cursor = 0
        self._auto_scan_feature_cursor = 0
        self._active_analysis_symbols: set[str] = set()
        self._analysis_symbol_lock = asyncio.Lock()
        self._market_analysis_task: asyncio.Task | None = None
        self._position_analysis_task: asyncio.Task | None = None
        self._runtime_heartbeat_task: asyncio.Task | None = None
        self._ml_auto_train_task: asyncio.Task | None = None
        self._local_tools_last_train_started_at: datetime | None = None
        self._local_tools_last_completed_shadow_count: int = 0
        self._strategy_learning_context_cache: dict[str, Any] = {}
        self._strategy_learning_context_refresh_tasks: dict[str, asyncio.Task] = {}

    def is_running(self) -> bool:
        """Expose lifecycle state without coupling loop services to private fields."""

        return bool(getattr(self, "_running", False))

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

    def market_round_time_budget_seconds(self) -> float:
        """Return the soft per-round scan budget used inside market analysis."""

        settings.refresh_runtime_env(force=True)
        interval = max(10.0, float(settings.decision_interval_seconds or 60))
        return max(8.0, interval * 0.90)

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
        """Return the hard watchdog for one full position-review round.

        This must stay separate from ``position_review_stage_timeout_seconds``:
        the stage timeout bounds one slow sub-step, while the round watchdog only
        catches a genuinely stuck full position-review cycle.
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

    def strategy_learning_context_timeout_seconds(self) -> float:
        """Return the hard budget for strategy-learning context in the trading loop."""

        interval = max(10.0, float(settings.decision_interval_seconds or 60))
        configured = float(DEFAULT_TRADING_PARAMS.strategy_learning.runtime_context_timeout_seconds)
        return max(0.5, min(configured, interval * 0.20))

    def strategy_learning_perf_timeout_seconds(self) -> float:
        """Return the short budget for historical performance context."""

        return max(
            0.2,
            float(DEFAULT_TRADING_PARAMS.strategy_learning.runtime_perf_timeout_seconds),
        )

    def strategy_learning_account_timeout_seconds(self) -> float:
        """Return the short budget for account-equity context."""

        return max(
            0.2,
            float(DEFAULT_TRADING_PARAMS.strategy_learning.runtime_account_timeout_seconds),
        )

    def _safe_set_strategy_context_stage(self, stage: str) -> None:
        """Update strategy-context diagnostics without breaking the trading decision path."""

        try:
            self._set_loop_stage(stage)
        except Exception:
            logger.debug("strategy context stage update skipped", stage=stage)

    async def _bounded_strategy_context_value(
        self,
        label: str,
        awaitable: Any,
        fallback: Any,
        timeout_seconds: float,
    ) -> Any:
        """Read optional strategy context without letting slow IO block the round."""

        task = asyncio.ensure_future(awaitable)
        try:
            self._safe_set_strategy_context_stage(f"strategy_context:{label}")
            done, pending = await asyncio.wait({task}, timeout=max(float(timeout_seconds), 0.0))
            if pending:
                await drain_cancelled_tasks(pending, timeout_seconds=0.05)
                logger.warning(
                    "strategy context stage timed out; using baseline value",
                    stage=label,
                    timeout_seconds=round(timeout_seconds, 3),
                )
                return fallback
            return next(iter(done)).result()
        except Exception as exc:
            logger.warning(
                "strategy context stage failed; using baseline value",
                stage=label,
                error=safe_error_text(exc),
            )
            return fallback

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

    def _start_strategy_learning_context_refresh(
        self,
        *,
        mode: str,
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
                max_open_positions=int(settings.max_open_positions_per_model or 20),
                limit=DEFAULT_TRADING_PARAMS.strategy_learning.runtime_context_row_limit,
            )
            learned_context["strategy_learning_cache_status"] = "fresh"
            learned_context["strategy_learning_runtime_timeout_seconds"] = (
                self.strategy_learning_context_timeout_seconds()
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

    def _round_budget_exhausted(self, started_at: datetime) -> bool:
        return self._round_elapsed_seconds(started_at) >= self.market_round_time_budget_seconds()

    async def _runtime_heartbeat_loop(self) -> None:
        """Keep split-process dashboard heartbeat fresh while a round is busy."""

        while self._running:
            self._write_runtime_heartbeat()
            await asyncio.sleep(5.0)

    async def enforce_sl_tp_for_position_review(
        self,
        feature_vectors: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Enforce stop-loss/take-profit through a position-review boundary."""

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
    ) -> tuple[list[tuple[str, str, DecisionOutput, Any, int | None]], set[tuple[str, str]]]:
        """Create position-review candidates through an explicit boundary."""

        return await self._review_open_positions(
            open_positions,
            feature_vectors,
            results=results,
            round_decision_ids=round_decision_ids,
            position_entry_pause_reason=position_entry_pause_reason,
            max_groups_override=max_groups_override,
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

        return await self._execute_candidate(
            symbol,
            model_name,
            decision,
            assessment,
            decision_db_id,
            results,
            open_positions=open_positions,
        )

    def record_round_error(self, reason: str) -> None:
        """Record the latest loop error through an explicit service boundary."""

        error_text = str(reason)[:300]
        self._last_round_error = error_text
        self._runtime_state().last_error = error_text
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
        """Return in-memory paper positions used as a DB context fallback."""

        if not self.paper_executor:
            return []
        return await self.paper_executor.get_positions()

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

        return await self.account_accounting_service.allocated_order_balance(
            model_mode,
            decision,
        )

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

    def _entry_symbol_blocklist_policy(self) -> EntrySymbolBlocklistPolicy:
        policy = getattr(self, "entry_symbol_blocklist", None)
        if policy is None:
            policy = EntrySymbolBlocklistPolicy(self._normalize_position_symbol)
            self.entry_symbol_blocklist = policy
        return policy

    def is_untradable_exchange_error(self, result_text: str) -> bool:
        """Classify non-tradable exchange errors through an explicit boundary."""

        return self._entry_symbol_blocklist_policy().is_untradable_exchange_error(result_text)

    def remember_untradable_symbol(self, symbol: str, result_text: str) -> None:
        """Remember permanently untradable symbols through an explicit boundary."""

        self._entry_symbol_blocklist_policy().remember_untradable_symbol(symbol, result_text)

    def is_transient_entry_exchange_error(self, result_text: str) -> bool:
        """Classify transient entry errors through an explicit boundary."""

        return self._entry_symbol_blocklist_policy().is_transient_entry_exchange_error(result_text)

    def remember_temporary_entry_block(
        self,
        symbol: str,
        reason: str,
        minutes: float,
    ) -> None:
        """Remember temporary entry blocks through an explicit execution boundary."""

        self._entry_symbol_blocklist_policy().remember_temporary_entry_block(
            symbol,
            reason,
            minutes,
        )

    def transient_entry_block_minutes(self, result_text: str) -> float:
        """Return transient entry-block duration through an explicit boundary."""

        return self._entry_symbol_blocklist_policy().transient_entry_block_minutes(result_text)

    def blocked_symbol_reason(self, symbol: str | None) -> str | None:
        """Return active new-entry block reason through the symbol blocklist boundary."""

        return self._entry_symbol_blocklist_policy().blocked_symbol_reason(symbol)

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

    def clear_market_no_opportunity_symbol(self, symbol: str) -> None:
        """Clear market no-opportunity memory through an explicit boundary."""

        self._clear_market_no_opportunity_symbol(symbol)

    async def persist_account_update(
        self,
        model_name: str,
        decision_model_name: str,
        execution_result: ExecutionResult,
    ) -> None:
        """Persist account update through an explicit execution boundary."""

        await self.account_accounting_service.persist_account_update(
            model_name,
            decision_model_name,
            execution_result,
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

    def remember_exit_cooldown(self, model_name: str, decision: DecisionOutput) -> None:
        """Record successful exit cooldown through an explicit boundary."""

        self.exit_cooldown.remember_exit(model_name, decision)

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
                f"{symbol} {action} 开仓 亏损 盈利 复盘 相似历史",
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
            "policy": "相似历史只作为软证据调节和解释，不作为硬拦截。",
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

    def _runtime_state(self, scope: str | None = None) -> _AnalysisRuntimeState:
        resolved_scope = scope or _analysis_scope_context.get() or "full"
        if resolved_scope not in self._analysis_runtime:
            resolved_scope = "full"
        return self._analysis_runtime[resolved_scope]

    def _start_runtime_round(self, scope: str, started_at: datetime) -> None:
        state = self._runtime_state(scope)
        state.current_stage = "starting"
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
        state.last_finished_at = finished_at
        state.current_stage = "idle" if ok else "error"
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

    def _set_loop_stage(
        self,
        stage: str,
        error: str | None = None,
        *,
        scope: str | None = None,
    ) -> None:
        resolved_scope = scope or _analysis_scope_context.get() or "full"
        state = self._runtime_state(resolved_scope)
        state.current_stage = stage
        if error is not None:
            state.last_error = str(error)[:300]
            self._last_round_error = state.last_error
        self._current_stage = stage
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
            self._last_round_error = reason
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
                self.data_service.get_feature_vector(symbol),
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
                self.data_service.get_feature_vector(symbol),
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
        market_budget = max(
            configured_limit,
            configured_limit * int(AUTO_SCAN_FEATURE_FETCH_POOL_MULTIPLIER),
            int(AUTO_SCAN_FEATURE_FETCH_POOL_MIN),
        )
        market_budget = min(len(market_symbols), market_budget)
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
        return selected

    async def _price_action_black_swan_false_positive(
        self,
        decision: DecisionOutput,
        rejection_reason: str | None,
        assessment: Any,
    ) -> bool:
        """Recheck price-action black-swan entry blocks with fresh data.

        Recent diagnostics showed some liquid symbols receiving impossible
        one-minute returns such as -30% to -50% from stale/bad feature data.
        A real flash crash should remain visible in a fresh snapshot; if it
        disappears, we keep the warning but allow the AI entry to continue to
        the normal price/liquidity guards.
        """
        if not decision.is_entry:
            return False
        bs_result = getattr(assessment, "black_swan_result", None)
        if not bs_result or getattr(bs_result, "severity", "") != "critical":
            return False
        reason_text = str(rejection_reason or getattr(bs_result, "reason", "") or "")
        bs_source = str(getattr(bs_result, "source", "") or "").lower()
        is_price_action_risk = bs_source in {"price_action", "combined"}
        if (
            not is_price_action_risk
            and "异常暴跌" not in reason_text
            and "行情风险" not in reason_text
        ):
            return False

        fresh = await self._fresh_feature_vector_for_price_recheck(decision.symbol)
        fresh_snapshot = fresh.to_dict() if fresh is not None and hasattr(fresh, "to_dict") else {}
        fresh_quality_reason = (
            self.entry_market_data_quality.reason(fresh_snapshot, stage_label="黑天鹅复核刷新行情")
            if fresh_snapshot
            else "黑天鹅复核刷新行情失败，无法确认最新短周期特征。"
        )
        fresh_returns_1 = (
            self._safe_float(fresh_snapshot.get("returns_1"), 0.0) if fresh_snapshot else 0.0
        )
        fresh_returns_5 = (
            self._safe_float(fresh_snapshot.get("returns_5"), 0.0) if fresh_snapshot else 0.0
        )
        fresh_returns_20 = (
            self._safe_float(fresh_snapshot.get("returns_20"), 0.0) if fresh_snapshot else 0.0
        )
        fresh_change_24h = (
            self._safe_float(fresh_snapshot.get("change_24h_pct"), 0.0) if fresh_snapshot else 0.0
        )
        bid = self._safe_float(fresh_snapshot.get("bid"), 0.0) if fresh_snapshot else 0.0
        ask = self._safe_float(fresh_snapshot.get("ask"), 0.0) if fresh_snapshot else 0.0
        bid_depth = (
            self._safe_float(fresh_snapshot.get("orderbook_bid_depth"), 0.0)
            if fresh_snapshot
            else 0.0
        )
        ask_depth = (
            self._safe_float(fresh_snapshot.get("orderbook_ask_depth"), 0.0)
            if fresh_snapshot
            else 0.0
        )
        spread = (
            (ask - bid) / max((ask + bid) / 2.0, 1e-12)
            if bid > 0 and ask > 0 and ask >= bid
            else 0.0
        )
        has_normal_book = bool(
            bid_depth > 0 and ask_depth > 0 and spread <= ENTRY_BLACK_SWAN_REBOUND_MAX_SPREAD
        )
        raw = self._safe_dict(decision.raw_response)
        opportunity = self._safe_dict(raw.get("opportunity_score"))
        expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = self._safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        high_quality_entry = bool(
            decision.confidence >= ENTRY_BLACK_SWAN_REBOUND_MIN_CONFIDENCE
            and expected_net >= ENTRY_BLACK_SWAN_REBOUND_MIN_EXPECTED_NET
            and profit_quality >= ENTRY_BLACK_SWAN_REBOUND_MIN_PROFIT_QUALITY
        )
        rebound_recovery_for_long = bool(
            decision.action == Action.LONG
            and high_quality_entry
            and has_normal_book
            and fresh_change_24h > -18.0
            and (
                (
                    fresh_returns_1 >= ENTRY_BLACK_SWAN_REBOUND_MIN_1M
                    and fresh_returns_5 >= ENTRY_BLACK_SWAN_REBOUND_MIN_5M
                )
                or (
                    fresh_returns_5 >= ENTRY_BLACK_SWAN_REBOUND_MIN_5M
                    and fresh_returns_20 >= -0.015
                )
            )
        )
        impossible_short_return_artifact = bool(
            fresh_snapshot
            and not fresh_quality_reason
            and fresh_returns_1 <= -0.20
            and fresh_returns_5 <= -0.20
            and fresh_change_24h > -18.0
            and has_normal_book
        )
        false_positive = bool(
            fresh_snapshot
            and not fresh_quality_reason
            and (
                (
                    fresh_returns_1 > ENTRY_BLACK_SWAN_RECHECK_SAFE_1M_DROP
                    and fresh_returns_5 > ENTRY_BLACK_SWAN_RECHECK_SAFE_5M_DROP
                )
                or impossible_short_return_artifact
                or rebound_recovery_for_long
            )
        )

        raw["black_swan_price_action_recheck"] = {
            "triggered": True,
            "original_reason": reason_text,
            "fresh_available": bool(fresh_snapshot),
            "fresh_quality_reason": fresh_quality_reason,
            "fresh_returns_1": round(fresh_returns_1, 6),
            "fresh_returns_5": round(fresh_returns_5, 6),
            "fresh_returns_20": round(fresh_returns_20, 6),
            "fresh_change_24h_pct": round(fresh_change_24h, 6),
            "fresh_spread_pct": round(spread * 100, 6),
            "fresh_bid_depth": round(bid_depth, 6),
            "fresh_ask_depth": round(ask_depth, 6),
            "impossible_short_return_artifact": impossible_short_return_artifact,
            "rebound_recovery_for_long": rebound_recovery_for_long,
            "high_quality_entry": high_quality_entry,
            "expected_net_return_pct": round(expected_net, 6),
            "profit_quality_ratio": round(profit_quality, 6),
            "treated_as_false_positive": false_positive,
            "policy": (
                "价格动作黑天鹅拦截必须用最新行情复核；复核后不再显示极端暴跌，"
                "或高质量做多信号已出现明确反弹恢复时，按警告处理并继续进入价格偏移和盘口检查。"
            ),
        }
        if false_positive:
            raw.setdefault("execution_advisory_warnings", []).append(
                {
                    "reason": (
                        "风险引擎检测到疑似 1 分钟极端暴跌，但最新行情复核显示该信号可能是脏数据"
                        "或已经完成反弹恢复，已降级为警告。"
                    ),
                    "fresh_returns_1": round(fresh_returns_1, 6),
                    "fresh_returns_5": round(fresh_returns_5, 6),
                    "fresh_returns_20": round(fresh_returns_20, 6),
                }
            )
            decision.feature_snapshot = fresh_snapshot
            logger.warning(
                "price-action black swan entry block downgraded after fresh recheck",
                symbol=decision.symbol,
                original_reason=reason_text[:160],
                fresh_returns_1=fresh_returns_1,
                fresh_returns_5=fresh_returns_5,
            )
        decision.raw_response = raw
        return false_positive

    def _feature_opportunity_score(self, fv: Any) -> float:
        """Compatibility delegate for feature-based auto-scan opportunity score."""

        ranker = getattr(self, "entry_feature_ranker", None)
        if ranker is None:
            ranker = EntryFeatureRankerPolicy(
                suspicious_symbol_reason=self._suspicious_new_symbol_reason,
                min_entry_volume_ratio_provider=self._runtime_min_entry_volume_ratio,
                min_entry_adx_provider=self._runtime_min_entry_adx,
                major_symbols=frozenset(ALT_LONG_ALLOWED_SYMBOLS),
            )
        return ranker.feature_opportunity_score(fv)

    def _entry_market_hold_penalty_policy(self) -> EntryMarketHoldPenaltyPolicy:
        policy = getattr(self, "entry_market_hold_penalty", None)
        if policy is not None:
            return policy
        policy = EntryMarketHoldPenaltyPolicy(
            normalize_symbol=self._normalize_position_symbol,
            feature_opportunity_score=self._feature_opportunity_score,
            min_entry_volume_ratio_provider=self._runtime_min_entry_volume_ratio,
            min_entry_adx_provider=self._runtime_min_entry_adx,
        )
        self.entry_market_hold_penalty = policy
        return policy

    def _remember_market_hold_symbol(
        self,
        symbol: str,
        fv: Any | None = None,
        reason: str | None = None,
    ) -> None:
        self._entry_market_hold_penalty_policy().remember_hold_symbol(symbol, fv, reason)

    def _prune_market_no_opportunity_symbols(self) -> None:
        self._entry_market_hold_penalty_policy().prune_no_opportunity_symbols()

    def _clear_market_no_opportunity_symbol(self, symbol: str) -> None:
        self._entry_market_hold_penalty_policy().clear_symbol(symbol)

    def _remember_market_analyzed_symbol(self, symbol: str) -> None:
        self._entry_market_hold_penalty_policy().remember_analyzed_symbol(symbol)

    def _recent_market_hold_penalty(self, symbol: str) -> float:
        return self._entry_market_hold_penalty_policy().recent_hold_penalty(symbol)

    def _recent_market_analysis_penalty(self, symbol: str) -> float:
        return self._entry_market_hold_penalty_policy().recent_analysis_penalty(symbol)

    def _no_opportunity_rotation_penalty(self, symbol: str, fv: Any | None = None) -> float:
        return self._entry_market_hold_penalty_policy().no_opportunity_rotation_penalty(
            symbol,
            fv,
        )

    def _entry_market_regime_context_policy(self) -> EntryMarketRegimeContextPolicy:
        policy = getattr(self, "entry_market_regime_context", None)
        if policy is not None:
            return policy
        return EntryMarketRegimeContextPolicy(self._is_valid_feature_vector)

    def _market_regime_context(self, feature_vectors: dict[str, Any]) -> dict[str, Any]:
        """Predict the current market style before asking for per-symbol entries."""
        return self._entry_market_regime_context_policy().context(feature_vectors)

    def _btc_eth_alt_long_filter(self, majors: list[Any]) -> dict[str, Any]:
        return self._entry_market_regime_context_policy().btc_eth_alt_long_filter(majors)

    def _entry_strategy_mode_context_policy(self) -> EntryStrategyModeContextPolicy:
        policy = getattr(self, "entry_strategy_mode_context", None)
        if policy is not None:
            return policy
        return EntryStrategyModeContextPolicy()

    def _dynamic_capacity_context(self) -> dict[str, Any]:
        context = getattr(self, "_current_capacity_context", None)
        if isinstance(context, dict):
            return dict(context)
        fallback = int(settings.max_open_positions_per_model or 0)
        return {
            "entry_limit": fallback,
            "effective_limit": fallback,
            "base_limit": fallback,
        }

    def _runtime_entry_filters(self) -> RuntimeEntryFilters:
        context = getattr(self, "_current_strategy_mode_context", None)
        if isinstance(context, dict):
            return entry_filters_from_context(context)
        return entry_filters_from_context(None)

    def _runtime_min_entry_volume_ratio(self) -> float:
        return self._runtime_entry_filters().min_entry_volume_ratio

    def _runtime_min_entry_adx(self) -> float:
        return self._runtime_entry_filters().min_entry_adx

    def _dynamic_capacity_policy(self) -> DynamicPositionCapacityPolicy:
        policy = getattr(self, "dynamic_capacity", None)
        if policy is not None:
            return policy
        scorer = getattr(self, "position_quality_scorer", None)
        if scorer is None:
            scorer = PositionQualityScorer()
            self.position_quality_scorer = scorer
        policy = DynamicPositionCapacityPolicy(
            lambda: settings.max_open_positions_per_model,
            quality_scorer=scorer,
        )
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
        strategy_context["max_open_positions_base"] = decision.get("base_limit")
        strategy_context["max_open_positions_effective"] = decision.get("effective_limit")
        strategy_context["max_open_positions_entry"] = decision.get("entry_limit")
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

        daily_state = await self._bounded_strategy_context_value(
            "daily_perf",
            self.daily_performance_service.state(selected_mode),
            {},
            perf_timeout,
        )
        side_perf = await self._bounded_strategy_context_value(
            "today_side_perf",
            self._today_side_performance(selected_mode),
            {},
            perf_timeout,
        )
        side_perf_multiday = await self._bounded_strategy_context_value(
            "multiday_side_perf",
            self._multiday_side_performance(selected_mode),
            {},
            perf_timeout,
        )
        symbol_side_perf = await self._bounded_strategy_context_value(
            "symbol_side_perf",
            self._recent_symbol_side_performance(selected_mode),
            {},
            perf_timeout,
        )
        model_contribution_perf = await self._bounded_strategy_context_value(
            "model_contribution_perf",
            self._recent_model_contribution_performance(selected_mode),
            {},
            perf_timeout,
        )
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
        account_equity = await self._bounded_strategy_context_value(
            "account_equity",
            self.allocated_order_balance(selected_mode),
            0.0,
            account_timeout,
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
        context["strategy_context_runtime"] = {
            "perf_timeout_seconds": perf_timeout,
            "account_timeout_seconds": account_timeout,
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
            refresh_task = self._start_strategy_learning_context_refresh(
                mode=selected_mode,
                strategy_learning=strategy_learning,
                context=context,
                open_positions=open_positions or [],
            )
            done, _pending = await asyncio.wait(
                {refresh_task},
                timeout=self.strategy_learning_context_timeout_seconds(),
            )
            if not done:
                raise asyncio.TimeoutError()
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
        except asyncio.TimeoutError:
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
                            self.strategy_learning_context_timeout_seconds()
                        ),
                    }
                )
                logger.warning(
                    "strategy learning context timed out; using cached strategy context",
                    timeout_seconds=round(self.strategy_learning_context_timeout_seconds(), 3),
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
                self.strategy_learning_context_timeout_seconds()
            )
            logger.warning(
                "strategy learning context timed out; using baseline strategy context",
                timeout_seconds=round(self.strategy_learning_context_timeout_seconds(), 3),
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
        raw["strategy_mode"] = self._json_safe_payload(strategy_mode_context)
        raw["strategy_learning_context"] = {
            "strategy_profile_id": strategy_mode_context.get("strategy_profile_id"),
            "strategy_profile_version": strategy_mode_context.get("strategy_profile_version"),
            "scheduler_reason": strategy_mode_context.get("scheduler_reason"),
            "expert_integrity_mode": strategy_mode_context.get("expert_integrity_mode"),
            "strategy_learning_entry_pause": strategy_mode_context.get(
                "strategy_learning_entry_pause", False
            ),
            "strategy_learning_entry_pause_reason": strategy_mode_context.get(
                "strategy_learning_entry_pause_reason", ""
            ),
            "strategy_learning_execution_guard_active": strategy_mode_context.get(
                "strategy_learning_execution_guard_active", False
            ),
            "strategy_learning_execution_guard_reason": strategy_mode_context.get(
                "strategy_learning_execution_guard_reason", ""
            ),
            "strategy_learning_release_pressure_active": strategy_mode_context.get(
                "strategy_learning_release_pressure_active", False
            ),
            "strategy_learning_release_pressure_reason": strategy_mode_context.get(
                "strategy_learning_release_pressure_reason", ""
            ),
            "strategy_learning_health_guard_active": strategy_mode_context.get(
                "strategy_learning_health_guard_active", False
            ),
            "strategy_learning_recovery_probe_allowed": strategy_mode_context.get(
                "strategy_learning_recovery_probe_allowed", False
            ),
            "strategy_learning_recovery_probe_reason": strategy_mode_context.get(
                "strategy_learning_recovery_probe_reason", ""
            ),
            "strategy_learning_sizing": strategy_mode_context.get("strategy_learning_sizing", {}),
            "entry_filters": strategy_mode_context.get("entry_filters", {}),
            "min_entry_volume_ratio": strategy_mode_context.get("min_entry_volume_ratio"),
            "min_entry_adx": strategy_mode_context.get("min_entry_adx"),
            "entry_filters_are_hard_gate": False,
            "strategy_learning_release_pressure_detail": strategy_mode_context.get(
                "strategy_learning_release_pressure_detail", {}
            ),
            "strategy_learning": strategy_mode_context.get("strategy_learning"),
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
        """Compatibility delegate for older tests/tools that call the legacy private method."""

        scorer = getattr(self, "entry_opportunity_score", None)
        if scorer is None:
            scorer = EntryOpportunityScoringPolicy(
                normalize_symbol=self._normalize_position_symbol,
                model_contribution_score_adjustment=self._model_contribution_score_adjustment,
                annotate_decision_source=self._annotate_decision_source,
                entry_symbol_winner_decay=getattr(
                    self,
                    "entry_symbol_winner_decay",
                    EntrySymbolWinnerDecayPolicy(),
                ),
            )
        return scorer.score_candidate(decision, strategy)

    def _annotate_decision_source(self, decision: DecisionOutput) -> dict[str, Any]:
        raw = self._safe_dict(decision.raw_response)
        side = (
            self._entry_side_value(decision)
            if decision.is_entry
            else ("exit" if decision.is_exit else "hold")
        )
        opinions = self._safe_list(raw.get("opinions"))
        decision_maker = self._safe_dict(raw.get("decision_maker"))
        opportunity = self._safe_dict(raw.get("opportunity_score"))
        evidence_score = self._safe_dict(opportunity.get("evidence_score"))
        ml_signal = self._safe_dict(raw.get("ml_signal"))
        quant_probe = self._safe_dict(raw.get("quant_validation_probe_entry"))
        server_tools = self._safe_dict(raw.get("server_quant_tools"))

        expert_support = 0
        expert_opposite = 0
        opposite_side = "short" if side == "long" else "long"
        for opinion in opinions:
            if not isinstance(opinion, dict):
                continue
            action = str(opinion.get("action") or "").lower()
            confidence = self._safe_float(opinion.get("confidence"), 0.0)
            if action == side and confidence >= 0.55:
                expert_support += 1
            elif action == opposite_side and confidence >= 0.55:
                expert_opposite += 1

        ml_influence_enabled = bool(
            ml_signal.get("influence_enabled")
            or opportunity.get("ml_influence_enabled")
            or opportunity.get("ml_aligned")
        )
        local_profit_influence = bool(
            opportunity.get("local_profit_aligned")
            or quant_probe.get("status")
            or server_tools.get("profit_model")
        )
        primary_source = "ai_experts_and_decision_maker"
        if decision.is_hold:
            primary_source = "ai_hold_decision"
        elif decision.is_exit:
            primary_source = "ai_position_review_or_fast_risk"

        raw["decision_source"] = {
            "primary_source": primary_source,
            "primary_source_cn": (
                "AI 专家/最终交易员"
                if decision.is_entry
                else ("AI 持仓复盘/快速风控" if decision.is_exit else "AI 观望")
            ),
            "symbol": decision.symbol,
            "action": decision.action.value,
            "side": side,
            "ai_role": "决定方向和动作",
            "expert_support_count": expert_support,
            "expert_opposite_count": expert_opposite,
            "decision_maker_action": decision_maker.get("action"),
            "decision_maker_confidence": decision_maker.get("confidence"),
            "local_ml_role": (
                "参与评分/过滤/仓位控制"
                if ml_influence_enabled
                else "学习观察或证据不足，未作为主决策"
            ),
            "server_profit_model_role": (
                "参与盈利质量判断" if local_profit_influence else "未提供有效同向盈利证据"
            ),
            "ml_influence_enabled": ml_influence_enabled,
            "server_profit_aligned": bool(opportunity.get("local_profit_aligned")),
            "timeseries_aligned": bool(opportunity.get("timeseries_aligned")),
            "opportunity_score": opportunity.get("score"),
            "entry_evidence_score": evidence_score.get("score"),
            "entry_evidence_effective_score": evidence_score.get("effective_score"),
            "entry_evidence_tier": evidence_score.get("tier"),
            "entry_evidence_size_multiplier": evidence_score.get("size_multiplier"),
            "expected_net_return_pct": opportunity.get("expected_net_return_pct"),
            "profit_quality_ratio": opportunity.get("profit_quality_ratio"),
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
            return gate.gate_reason(decision)
        return EntryOpportunityGatePolicy(
            symbol_loss_cooldown_policy=EntryLossCooldownPolicy(self._normalize_position_symbol),
            post_crash_rebound_guard=EntryPostCrashReboundGuardPolicy(),
        ).gate_reason(decision)

    async def _today_side_performance(self, mode: str) -> dict[str, dict[str, float]]:
        """Delegate today's long/short realized-PnL feedback to a dedicated service."""

        service = getattr(self, "daily_side_performance_service", None)
        if service is None:
            service = DailySidePerformanceService()
            self.daily_side_performance_service = service
        return await service.state(mode)

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
                loss_cooldown_params=ENTRY_LOSS_COOLDOWN_PARAMS,
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

    def _decision_contribution_sources(
        self,
        opportunity: dict[str, Any],
        raw: dict[str, Any],
        side: str,
    ) -> list[str]:
        """Compatibility delegate for legacy callers/tests."""

        service = getattr(self, "model_contribution_performance_service", None)
        if service is None:
            service = ModelContributionPerformanceService(
                lookback_days=SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS,
            )
            self.model_contribution_performance_service = service
        return service.contribution_sources(opportunity, raw, side)

    def _model_contribution_score_adjustment(
        self,
        sources: list[str],
        performance: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Compatibility delegate for legacy callers/tests."""

        service = getattr(self, "model_contribution_performance_service", None)
        if service is None:
            service = ModelContributionPerformanceService(
                lookback_days=SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS,
            )
            self.model_contribution_performance_service = service
        return service.score_adjustment(sources, performance)

    def _is_auto_tradeable_feature(self, fv: Any) -> bool:
        """Compatibility delegate for the hard auto-scan feature filter."""

        return self.entry_feature_ranker.is_auto_tradeable_feature(fv)

    def _is_auto_analysis_candidate_feature(self, fv: Any) -> bool:
        """Compatibility delegate for the secondary auto-scan feature filter."""

        return self.entry_feature_ranker.is_auto_analysis_candidate_feature(fv)

    def _rank_auto_feature_vectors(
        self,
        feature_vectors: dict[str, Any],
        limit: int,
    ) -> dict[str, Any]:
        hold_penalty = self._entry_market_hold_penalty_policy()
        result = self.entry_feature_ranker.rank(
            feature_vectors,
            limit,
            recent_hold_penalty=hold_penalty.recent_hold_penalty,
            recent_analysis_penalty=hold_penalty.recent_analysis_penalty,
            no_opportunity_rotation_penalty=hold_penalty.no_opportunity_rotation_penalty,
        )
        logger.info(
            "auto opportunity shortlist",
            **result.diagnostics,
        )
        return result.selected

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

    async def _load_untradable_symbol_blocks(self) -> None:
        try:
            from models.decision import AIDecision

            async with get_session_ctx() as session:
                result = await session.execute(
                    select(AIDecision.symbol, AIDecision.execution_reason, AIDecision.created_at)
                    .where(AIDecision.execution_reason.is_not(None))
                    .order_by(AIDecision.created_at.desc())
                    .limit(300)
                )
                for row in result.all():
                    reason = row.execution_reason or ""
                    created_at = row.created_at
                    if created_at and created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=UTC)
                    recent = not created_at or (datetime.now(UTC) - created_at) <= timedelta(
                        hours=UNTRADABLE_SYMBOL_BLOCK_HOURS
                    )
                    if recent and self.is_untradable_exchange_error(reason):
                        self.remember_untradable_symbol(row.symbol, reason)
                    elif (
                        created_at
                        and datetime.now(UTC) - created_at
                        <= timedelta(minutes=TRANSIENT_ENTRY_BLOCK_MINUTES)
                        and self.is_transient_entry_exchange_error(reason)
                    ):
                        self.remember_temporary_entry_block(
                            row.symbol,
                            reason,
                            TRANSIENT_ENTRY_BLOCK_MINUTES,
                        )
                    elif (
                        created_at
                        and datetime.now(UTC) - created_at
                        <= timedelta(minutes=PRICE_GUARD_ENTRY_BLOCK_MINUTES)
                        and self.entry_symbol_blocklist.is_entry_price_guard_skip(reason)
                    ):
                        self.remember_temporary_entry_block(
                            row.symbol,
                            reason,
                            PRICE_GUARD_ENTRY_BLOCK_MINUTES,
                        )
        except Exception as e:
            logger.warning("failed to load untradable symbol blocks", error=safe_error_text(e))

    async def initialize(self) -> None:
        """Initialize models, executors, and connections."""
        await self.models.initialize_all()
        self.paper_executor = None

        # Initialize OKX demo/live connections for balance sync, position checks,
        # and actual order execution. Paper mode means OKX demo trading, not a
        # local fake fill.
        self.okx_executor = None
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

        await self._load_untradable_symbol_blocks()
        await self.expert_memory_service.backfill_trade_reflections(mode_manager.mode.value)

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
            run_market_analysis = False
            run_position_analysis = analysis_scope in {"full", "position"}
        new_pair_market_pause_applied = False
        round_start = datetime.now(UTC)
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
        claimed_analysis_symbols: list[str] = []
        claimed_symbol_keys: set[str] = set()
        published_dashboard_update = False
        scope_token = _analysis_scope_context.set(analysis_scope)
        self._start_runtime_round(analysis_scope, round_start)
        self._set_loop_stage("starting")

        try:
            self._set_loop_stage("shadow_backtests")
            await self.shadow_backtest_service.update_due()
            await self.stale_entry_candidate_expirer.expire()

            # 0. Refresh per-model execution mode mapping from current config
            self._refresh_model_modes()
            self._set_loop_stage("sync_exchange_positions")
            await self.okx_sync_service.reconcile_positions(
                f"{analysis_scope} round start",
                timeout_seconds=self.round_start_reconcile_timeout_seconds(),
                lock_wait_seconds=0.35,
            )
            self._set_loop_stage("load_open_positions")
            open_positions = await self.okx_sync_service.get_open_positions_context()
            await self._recover_pending_exit_decisions(
                results,
                open_positions,
                round_decision_ids,
            )
            new_pair_pause_reason = await self._new_pair_analysis_pause_reason(
                ENSEMBLE_TRADER_NAME,
                open_positions=open_positions,
            )
            if account_pause_reason:
                new_pair_pause_reason = account_pause_reason
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
            blocked_filter = self.entry_symbol_universe.filter_blocked_new_symbols(
                scan_symbols,
                open_positions,
                self._suspicious_new_symbol_reason,
                self.blocked_symbol_reason,
            )
            market_scan_symbols = blocked_filter.symbols
            if blocked_filter.skipped:
                logger.info(
                    "skipping blocked symbols before AI analysis",
                    count=len(blocked_filter.skipped),
                    symbols=[item.symbol for item in blocked_filter.skipped[:10]],
                )
                for item in blocked_filter.skipped[:20]:
                    results["warnings"].append(
                        {
                            "model": ENSEMBLE_TRADER_NAME,
                            "symbol": item.symbol,
                            "warning": (
                                "Symbol is temporarily skipped for new entry analysis: "
                                f"{item.reason}"
                            ),
                        }
                    )
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
            if not fetch_symbols:
                diagnostics = {
                    "scan_symbol_count": len(scan_symbols or []),
                    "scan_symbol_sample": list(scan_symbols or [])[:10],
                    "market_scan_after_blocked": len(blocked_filter.symbols),
                    "blocked_count": len(blocked_filter.skipped),
                    "blocked_sample": [
                        {"symbol": item.symbol, "reason": item.reason}
                        for item in blocked_filter.skipped[:5]
                    ],
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
                            self.data_service.get_feature_vector(sym),
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
            done, pending = await asyncio.wait(tasks, timeout=batch_timeout)
            if pending:
                await drain_cancelled_tasks(pending)
                logger.warning(
                    "feature vector batch reached time budget; cancelling slow sources",
                    symbol_count=len(fetch_symbols),
                    completed=len(done),
                    pending=len(pending),
                    timeout_seconds=round(batch_timeout, 3),
                )
                results.setdefault("warnings", []).append(
                    {
                        "model": ENSEMBLE_TRADER_NAME,
                        "symbol": "ALL",
                        "warning": ("本轮行情特征批量拉取超时，系统已跳过剩余候选并进入下一轮。"),
                    }
                )
            fv_results = []
            for task in done:
                try:
                    fv_results.append(task.result())
                except asyncio.CancelledError:
                    continue
                except Exception as exc:
                    logger.warning(
                        "feature vector task failed after batch wait",
                        error=safe_error_text(exc),
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
                elif mode_manager.is_auto_scan:
                    market_feature_vectors = self._rank_auto_feature_vectors(
                        market_feature_vectors, market_symbol_budget
                    )
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
                confidence=market_regime_context.get("confidence"),
                avoid_long=market_regime_context.get("avoid_long"),
                avoid_short=market_regime_context.get("avoid_short"),
                reason=market_regime_context.get("reason"),
            )
            logger.info(
                "strategy mode selected",
                strategy=strategy_mode_context.get("strategy"),
                posture=strategy_mode_context.get("posture"),
                allow_long=strategy_mode_context.get("allow_long"),
                allow_short=strategy_mode_context.get("allow_short"),
                blocked_directions=strategy_mode_context.get("blocked_directions"),
                exposure=strategy_mode_context.get("position_exposure"),
                reason=strategy_mode_context.get("reason"),
            )

            self._set_loop_stage("refresh_position_prices")
            await self.okx_sync_service.refresh_position_prices(feature_vectors)

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
            for market_index, (symbol, fv) in enumerate(market_feature_items):
                if market_index > 0 and self._round_budget_exhausted(round_start):
                    remaining = [
                        item_symbol for item_symbol, _item_fv in market_feature_items[market_index:]
                    ]
                    market_round_skipped_by_budget = remaining
                    budget_seconds = self.market_round_time_budget_seconds()
                    elapsed_seconds = self._round_elapsed_seconds(round_start)
                    warning = (
                        "本轮市场 AI 分析已达到调度时间预算，剩余候选顺延到后续轮次；"
                        "这不是开仓门槛，只用于防止单轮分析拖住系统心跳。"
                    )
                    logger.warning(
                        "market analysis round reached time budget",
                        elapsed_seconds=round(elapsed_seconds, 3),
                        budget_seconds=round(budget_seconds, 3),
                        skipped_count=len(remaining),
                        skipped_symbols=remaining[:10],
                    )
                    results.setdefault("warnings", []).append(
                        {
                            "model": ENSEMBLE_TRADER_NAME,
                            "symbol": "ALL",
                            "warning": warning,
                            "elapsed_seconds": round(elapsed_seconds, 3),
                            "budget_seconds": round(budget_seconds, 3),
                            "skipped_symbols": remaining[:20],
                        }
                    )
                    break
                quarantine_reason = self.entry_symbol_profit_quarantine.reason(
                    symbol,
                    strategy_mode_context,
                )
                if quarantine_reason:
                    logger.info(
                        "market symbol has realized loss cooldown evidence; still sending to AI",
                        symbol=symbol,
                        reason=quarantine_reason,
                    )
                if not await self._try_claim_analysis_symbol(symbol, "market"):
                    logger.info(
                        "market symbol skipped because another analysis owns it", symbol=symbol
                    )
                    continue
                claimed_analysis_symbols.append(symbol)
                claimed_symbol_keys.add(self._normalize_position_symbol(symbol))
                self._set_loop_stage(f"market_ai:{symbol}")

                results["symbols_processed"] += 1
                self._remember_market_analyzed_symbol(symbol)
                fv = await self._fresh_feature_vector_for_analysis(symbol, fv)
                if not self._is_valid_feature_vector(fv):
                    logger.warning("skip symbol after fresh feature check failed", symbol=symbol)
                    continue
                feature_vectors[symbol] = fv
                model_name = ENSEMBLE_TRADER_NAME
                model_mode = self._get_model_execution_mode(model_name)
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
                    self._attach_strategy_learning_context(quick_decision, strategy_mode_context)
                    decision_db_id = await self._log_decision(
                        quick_decision, is_paper=(model_mode == "paper")
                    )
                    if decision_db_id is not None:
                        round_decision_ids.add(decision_db_id)
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
                    self._remember_market_hold_symbol(symbol)
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
                    },
                )
                if isinstance(decision.raw_response, dict):
                    decision.raw_response.setdefault(
                        "entry_candidate_evidence", entry_candidate_evidence
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
                self._decision_count += 1
                await self.shadow_backtest_service.create(
                    decision_db_id,
                    decision,
                    fv,
                    model_mode,
                    analysis_type="market",
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
                    continue

                executed = assessment.decision if assessment.decision else decision
                if executed is not decision and decision.raw_response and not executed.raw_response:
                    executed.raw_response = decision.raw_response
                    executed.feature_snapshot = (
                        executed.feature_snapshot or decision.feature_snapshot
                    )
                if executed.is_hold:
                    probe_decision = self.entry_evidence_probe.create(
                        executed,
                        fv,
                        strategy_mode_context,
                        ml_signal_context,
                        local_ai_tools_context,
                        direction_competition_context,
                    )
                    probe_source_label = "入场候选证据包"
                    if probe_decision is None:
                        probe_decision = self.entry_quant_profit_probe.create(
                            executed,
                            fv,
                            strategy_mode_context,
                            ml_signal_context,
                            local_ai_tools_context,
                            direction_competition_context,
                        )
                        probe_source_label = "服务器盈利模型"
                    if probe_decision is not None:
                        executed = probe_decision
                        self._candidate_opportunity_score(executed, strategy_mode_context)
                        self._attach_strategy_learning_context(executed, strategy_mode_context)
                        if decision_db_id is not None:
                            await self._mark_decision_reason(
                                decision_db_id,
                                f"AI 原始裁决为观望；{probe_source_label}触发正期望候选，另建一条候选决策继续风控。",
                            )
                        probe_decision_db_id = await self._log_decision(
                            executed, is_paper=(model_mode == "paper")
                        )
                        if probe_decision_db_id is not None:
                            round_decision_ids.add(probe_decision_db_id)
                            decision_db_id = probe_decision_db_id
                            self._decision_count += 1
                        assessment = await self.market_decision_risk_assessment.assess(
                            decision=executed,
                            model_name=model_name,
                            open_positions=open_positions,
                            feature_vector=fv,
                            strategy_mode_context=strategy_mode_context,
                        )

                        if not assessment.approved:
                            reason = assessment.rejection_reason or "量化盈利探针未通过风控。"
                            if decision_db_id is not None:
                                await self._mark_decision_reason(decision_db_id, reason)
                            self.market_decision_result_recorder.append_result(
                                results=results,
                                model_name=model_name,
                                symbol=symbol,
                                decision_or_action=executed,
                                model_mode=model_mode,
                                approved=False,
                                execution_status="quant_probe_rejected",
                                reason=reason,
                            )
                            continue
                    else:
                        hold_reason = getattr(executed, "reasoning", None) or getattr(
                            decision, "reasoning", None
                        )
                        self._remember_market_hold_symbol(symbol, fv, hold_reason)
                        if decision_db_id is not None:
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

                if executed.is_hold:
                    hold_reason = getattr(executed, "reasoning", None) or getattr(
                        decision, "reasoning", None
                    )
                    self._remember_market_hold_symbol(symbol, fv, hold_reason)
                    if decision_db_id is not None:
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
                results["market_analysis_budget"] = {
                    "budget_seconds": round(self.market_round_time_budget_seconds(), 3),
                    "elapsed_seconds": round(self._round_elapsed_seconds(round_start), 3),
                    "processed_symbols": int(results.get("symbols_processed") or 0),
                    "deferred_symbols": market_round_skipped_by_budget[:50],
                    "deferred_count": len(market_round_skipped_by_budget),
                    "is_entry_gate": False,
                    "reason": (
                        "市场分析轮次达到调度预算，剩余币种通过滚动扫描顺延；"
                        "该预算不参与开仓风控评分。"
                    ),
                }

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
                    await self._mark_decision_raw_response(decision_db_id, raw_response)
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
            await self._fill_missing_decision_reasons(
                round_decision_ids,
                "本轮已经结束，但这条候选没有进入下单阶段，也没有拿到最终执行结果。"
                "系统已跳过本次旧信号，下一轮会用最新行情重新排序和评估。",
            )

            # 6. Push updates to dashboard
            await self._publish_dashboard_update(results)
            published_dashboard_update = True

        except asyncio.CancelledError:
            error_text = (
                f"{analysis_scope} analysis round cancelled by hard watchdog "
                f"after {round(self._round_elapsed_seconds(round_start), 3)} seconds."
            )
            self._set_loop_stage("watchdog_cancelled", error_text)
            results["status"] = "error"
            results["error"] = error_text
            logger.error(
                "trading loop iteration cancelled by watchdog",
                scope=analysis_scope,
                elapsed_seconds=round(self._round_elapsed_seconds(round_start), 3),
            )
            raise
        except Exception as e:
            error_text = safe_error_text(e, limit=180)
            self._set_loop_stage("error", error_text)
            await self._fill_missing_decision_reasons(
                round_decision_ids,
                f"\u672c\u8f6e\u6267\u884c\u5f02\u5e38\u4e2d\u65ad\uff0c\u672a\u80fd\u5b8c\u6210\u6700\u7ec8\u72b6\u6001\u56de\u5199\uff1a{safe_error_text(e, limit=120)}",
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
        try:
            await asyncio.gather(
                self._position_analysis_task,
                self._market_analysis_task,
                self._runtime_heartbeat_task,
            )
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop the trading loop gracefully."""
        self._running = False
        await self._stop_ml_auto_train_loop()
        for task in (
            self._position_analysis_task,
            self._market_analysis_task,
            self._runtime_heartbeat_task,
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
        if self.paper_executor:
            await self.paper_executor.shutdown()
        for okx in (self.okx_executor, self._okx_paper, self._okx_live):
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
        self._ml_auto_train_task = asyncio.create_task(self._ml_auto_train_loop())

    async def _stop_ml_auto_train_loop(self) -> None:
        task = self._ml_auto_train_task
        self._ml_auto_train_task = None
        if not task or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _ml_auto_train_loop(self) -> None:
        """Retrain local ML and server-side quant tools without blocking trading."""
        while self._running:
            try:
                result = await self.ml_signal_service.maybe_auto_train()
                if result.get("trained"):
                    logger.info(
                        "local ML signal model auto-trained",
                        sample_count=result.get("sample_count"),
                        new_sample_count=result.get("new_sample_count"),
                    )
                elif result.get("reason") not in {"not_due", "training_in_progress"}:
                    logger.warning(
                        "local ML signal auto-train skipped",
                        reason=result.get("reason"),
                        error=result.get("error"),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("local ML signal auto-train loop error", error=safe_error_text(e))
            try:
                await self._maybe_train_local_ai_tools()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("local AI tools auto-train loop error", error=safe_error_text(e))
            await asyncio.sleep(AUTO_TRAIN_CHECK_INTERVAL_SECONDS)

    async def _maybe_train_local_ai_tools(self, *, force: bool = False) -> dict[str, Any]:
        """Push fresh history to the server-side profit/time-series/exit models."""
        if not self.local_ai_tools.enabled():
            return {"trained": False, "reason": "disabled"}

        status = await self.local_ai_tools.status()
        now = datetime.now(UTC)
        trained_at_raw = status.get("trained_at") if isinstance(status, dict) else None
        trained_at = self._parse_datetime(trained_at_raw)
        age_seconds = (
            (now - trained_at).total_seconds()
            if trained_at is not None
            else AUTO_TRAIN_CHECK_INTERVAL_SECONDS * 12
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

        try:
            from scripts.train_local_ai_tools_models import (
                _completed_trade_sample_count,
                _load_closed_position_samples,
                _load_sequence_samples,
                _load_shadow_samples,
                _load_text_sentiment_samples,
                _load_trade_reflection_samples,
                _merge_trade_samples,
            )
            from services.training_data_quality import annotate_training_payload

            shadow_samples = await _load_shadow_samples(
                LOCAL_ML_TRAINING_PARAMS.training_shadow_sample_limit
            )
            trade_reflection_samples = await _load_trade_reflection_samples(
                LOCAL_ML_TRAINING_PARAMS.training_trade_sample_limit
            )
            closed_position_samples = await _load_closed_position_samples(
                LOCAL_ML_TRAINING_PARAMS.training_trade_sample_limit
            )
            trade_samples = _merge_trade_samples(trade_reflection_samples, closed_position_samples)
            sequence_samples = await _load_sequence_samples(
                LOCAL_ML_TRAINING_PARAMS.training_sequence_sample_limit
            )
            text_sentiment_samples = await _load_text_sentiment_samples(
                LOCAL_ML_TRAINING_PARAMS.training_text_sample_limit
            )
            training_payload = annotate_training_payload(
                shadow_samples=shadow_samples,
                trade_samples=trade_samples,
                sequence_samples=sequence_samples,
                text_sentiment_samples=text_sentiment_samples,
            )
            completed_trade_total = await _completed_trade_sample_count()
        except Exception as exc:
            return {
                "trained": False,
                "reason": "load_samples_error",
                "error": safe_error_text(exc, limit=180),
            }

        training_shadow_count = len(shadow_samples)
        quality_report = training_payload["quality_report"]
        governance_report = training_payload["governance_report"]
        trainable_shadow_count = len(training_payload["shadow_samples"])
        trainable_trade_count = len(training_payload["trade_samples"])
        new_shadow = max(completed_shadow_total - previous_completed_shadow_total, 0)
        previous_completed_trade_total = int(
            (status or {}).get("last_trained_completed_trade_sample_count")
            or (status or {}).get("completed_trade_sample_count")
            or server_trade_count
            or 0
        )
        new_trade = max(completed_trade_total - previous_completed_trade_total, 0)
        learning_only = not bool(
            (status or {}).get("model_bundle_available", (status or {}).get("available"))
        )
        min_interval_seconds = (
            LOCAL_ML_TRAINING_PARAMS.auto_train_learning_only_interval_seconds
            if learning_only
            else LOCAL_ML_TRAINING_PARAMS.auto_train_min_interval_seconds
        )
        min_new_shadow = (
            LOCAL_ML_TRAINING_PARAMS.auto_train_learning_only_min_new_samples
            if learning_only
            else LOCAL_ML_TRAINING_PARAMS.auto_train_min_new_samples
        )
        min_new_trade = (
            LOCAL_ML_TRAINING_PARAMS.local_tools_learning_only_min_new_trade_samples
            if learning_only
            else LOCAL_ML_TRAINING_PARAMS.local_tools_min_new_trade_samples
        )
        training_policy = {
            "learning_only": learning_only,
            "min_interval_seconds": min_interval_seconds,
            "min_new_shadow_samples": min_new_shadow,
            "min_new_trade_samples": min_new_trade,
            "min_training_shadow_samples": LOCAL_ML_TRAINING_PARAMS.min_training_samples,
            "training_shadow_sample_limit": (LOCAL_ML_TRAINING_PARAMS.training_shadow_sample_limit),
            "training_trade_sample_limit": LOCAL_ML_TRAINING_PARAMS.training_trade_sample_limit,
            "training_sequence_sample_limit": (
                LOCAL_ML_TRAINING_PARAMS.training_sequence_sample_limit
            ),
            "training_text_sample_limit": LOCAL_ML_TRAINING_PARAMS.training_text_sample_limit,
            "cursor_source": "last_trained_completed_shadow_sample_count",
            "trade_cursor_source": "last_trained_completed_trade_sample_count",
        }
        if completed_shadow_total < LOCAL_ML_TRAINING_PARAMS.min_training_samples:
            return {
                "trained": False,
                "reason": "not_enough_shadow_samples",
                "server_shadow_sample_count": server_shadow_count,
                "local_shadow_sample_count": training_shadow_count,
                "trainable_shadow_sample_count": trainable_shadow_count,
                "trainable_trade_sample_count": trainable_trade_count,
                "quality_report": quality_report,
                "governance_report": governance_report,
                "completed_shadow_sample_count": completed_shadow_total,
                "last_trained_completed_shadow_sample_count": previous_completed_shadow_total,
                "completed_trade_sample_count": completed_trade_total,
                "last_trained_completed_trade_sample_count": previous_completed_trade_total,
                "new_shadow_sample_count": new_shadow,
                "new_trade_sample_count": new_trade,
                "training_policy": training_policy,
            }
        should_train = (
            force
            or age_seconds >= min_interval_seconds
            or new_shadow >= min_new_shadow
            or new_trade >= min_new_trade
        )
        if not should_train:
            return {
                "trained": False,
                "reason": "not_due",
                "server_shadow_sample_count": server_shadow_count,
                "local_shadow_sample_count": training_shadow_count,
                "completed_shadow_sample_count": completed_shadow_total,
                "trainable_shadow_sample_count": trainable_shadow_count,
                "trainable_trade_sample_count": trainable_trade_count,
                "quality_report": quality_report,
                "governance_report": governance_report,
                "last_trained_completed_shadow_sample_count": previous_completed_shadow_total,
                "training_shadow_sample_limit": (
                    LOCAL_ML_TRAINING_PARAMS.training_shadow_sample_limit
                ),
                "completed_trade_sample_count": completed_trade_total,
                "last_trained_completed_trade_sample_count": previous_completed_trade_total,
                "new_shadow_sample_count": new_shadow,
                "new_trade_sample_count": new_trade,
                "model_age_seconds": round(age_seconds, 1),
                "training_policy": training_policy,
            }

        self._local_tools_last_train_started_at = now
        result = await self.local_ai_tools.train(
            training_payload["shadow_samples"],
            training_payload["trade_samples"],
            training_payload["sequence_samples"],
            training_payload["text_sentiment_samples"],
            source="local_trading_system_auto",
            completed_shadow_sample_count=completed_shadow_total,
            completed_trade_sample_count=completed_trade_total,
            quality_report=quality_report,
            governance_report=governance_report,
        )
        if result.get("trained"):
            result["completed_shadow_sample_count"] = completed_shadow_total
            result["last_trained_completed_shadow_sample_count"] = completed_shadow_total
            result["completed_trade_sample_count"] = completed_trade_total
            result["last_trained_completed_trade_sample_count"] = completed_trade_total
            result["training_shadow_sample_count"] = training_shadow_count
            result["trainable_shadow_sample_count"] = trainable_shadow_count
            result["trainable_trade_sample_count"] = trainable_trade_count
            result["quality_report"] = quality_report
            result["governance_report"] = governance_report
            result["training_shadow_sample_limit"] = (
                LOCAL_ML_TRAINING_PARAMS.training_shadow_sample_limit
            )
            result["training_policy"] = training_policy
            self._local_tools_last_completed_shadow_count = completed_shadow_total
            logger.info(
                "server-side local AI tools auto-trained",
                shadow_sample_count=result.get("shadow_sample_count"),
                trade_sample_count=result.get("trade_sample_count"),
                trained_at=result.get("trained_at"),
            )
        else:
            logger.warning(
                "server-side local AI tools auto-train skipped",
                reason=result.get("reason"),
                message=result.get("message"),
                error=result.get("error"),
                failure_count=result.get("failure_count"),
            )
        return result

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

    async def _apply_entry_profit_risk_sizing(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict] | None = None,
    ) -> None:
        """Compatibility delegate for older tests/tools that call the legacy private method."""

        sizer = getattr(self, "entry_profit_risk_sizing", None)
        if sizer is None:
            sizer = EntryProfitRiskSizingPolicy(
                allocated_order_balance=self.allocated_order_balance,
                entry_low_payoff_quality=getattr(
                    self, "entry_low_payoff_quality", EntryLowPayoffQualityPolicy()
                ),
                entry_stop_loss_budget=getattr(
                    self, "entry_stop_loss_budget", EntryStopLossBudgetPolicy()
                ),
                entry_stress_stop=getattr(self, "entry_stress_stop", EntryStressStopPolicy()),
                entry_existing_winner_context=getattr(
                    self,
                    "entry_existing_winner_context",
                    EntryExistingWinnerContextPolicy(self._normalize_position_symbol),
                ),
                max_leverage_provider=lambda: settings.max_leverage,
            )
        await sizer.apply(decision, model_mode, open_positions or [])

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

    def _reserve_entry_slot(
        self,
        model_name: str,
        decision: DecisionOutput,
        staged_entry_counts: dict[str, dict],
    ) -> None:
        """Compatibility delegate for older tests/tools that reserve staged entry slots."""

        capacity = getattr(self, "entry_capacity", None)
        if capacity is None:
            capacity = EntryCapacityPolicy(
                self._normalize_position_symbol,
                lambda: settings.max_open_positions_per_model,
            )
        capacity.reserve_slot(model_name, decision, staged_entry_counts)

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
            if str(position.side or "").lower() == "short":
                gross_pnl = (position.entry_price - result.price) * close_qty
            else:
                gross_pnl = (result.price - position.entry_price) * close_qty
            realized_pnl = gross_pnl - entry_fee - close_fee
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
                position.realized_pnl = realized_pnl
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
                        "realized_pnl": realized_pnl,
                        "stop_loss_price": position.stop_loss_price,
                        "take_profit_price": position.take_profit_price,
                        "is_open": False,
                        "closed_at": result.timestamp,
                        "created_at": position.created_at,
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

        execution_completed = (
            execution_result.status == OrderStatus.FILLED
            and self._safe_float(execution_result.quantity, 0.0) > 0
            and self._safe_float(execution_result.price, 0.0) > 0
        )
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
    ) -> int | None:
        """Record a synthetic close decision for exchange-side position closes."""
        try:
            side = str(pos.side or "").lower()
            action = Action.CLOSE_SHORT if side == "short" else Action.CLOSE_LONG
            close_fill_safe = {
                key: (value.isoformat() if isinstance(value, datetime) else value)
                for key, value in (close_fill or {}).items()
            }
            repo = DecisionRepository(session)
            record = await repo.log_decision(
                {
                    "model_name": pos.model_name or ENSEMBLE_TRADER_NAME,
                    "symbol": pos.symbol,
                    "action": action.value,
                    "confidence": 1.0,
                    "reasoning": sanitize_text(reason),
                    "position_size_pct": 1.0,
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
                    },
                    "raw_llm_response": {
                        "system_sync": True,
                        "source": "okx_position_reconcile",
                        "close_fill": close_fill_safe,
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

    async def _enforce_sl_tp(self, feature_vectors: dict) -> list[dict]:
        """Run fast non-AI protection for open positions before slow AI review."""
        auto_closes: list[dict[str, Any]] = []
        open_positions = await self.okx_sync_service.get_open_positions_context()
        if not open_positions:
            return auto_closes

        handled: set[tuple[str, str, str]] = set()
        fast_adverse_pct = min(
            max(float(settings.hard_stop_loss_pct or 0.05) * 0.6, 0.018),
            0.035,
        )

        for pos in open_positions:
            if pos.get("is_open", True) is False:
                continue

            model_name = str(pos.get("model_name") or ENSEMBLE_TRADER_NAME)
            sym = self._normalize_position_symbol(pos.get("symbol"))
            side = str(pos.get("side") or "").lower()
            if not sym or side not in {"long", "short"}:
                continue

            key = (model_name, sym, side)
            if key in handled:
                continue
            handled.add(key)

            fv = feature_vectors.get(sym) or feature_vectors.get(pos.get("symbol"))
            fv_current_price = self._safe_float(
                getattr(fv, "current_price", 0) if fv is not None else 0,
                0.0,
            )
            position_current_price = self._safe_float(pos.get("current_price"), 0.0)
            current_price = fv_current_price or position_current_price
            entry_price = self._safe_float(pos.get("entry_price"), 0.0)
            if current_price <= 0 or entry_price <= 0:
                continue

            stop_loss = self._safe_float(pos.get("stop_loss"), 0.0)
            take_profit = self._safe_float(pos.get("take_profit"), 0.0)
            returns_1 = self._safe_float(getattr(fv, "returns_1", 0) if fv is not None else 0, 0.0)
            returns_5 = self._safe_float(getattr(fv, "returns_5", 0) if fv is not None else 0, 0.0)
            returns_20 = self._safe_float(
                getattr(fv, "returns_20", 0) if fv is not None else 0, 0.0
            )
            volume_ratio = self._safe_float(
                getattr(fv, "volume_ratio", 0) if fv is not None else 0, 0.0
            )
            rsi_14 = self._safe_float(getattr(fv, "rsi_14", 50) if fv is not None else 50, 50.0)
            bb_pct = self._safe_float(getattr(fv, "bb_pct", 0.5) if fv is not None else 0.5, 0.5)
            macd_diff = self._safe_float(getattr(fv, "macd_diff", 0) if fv is not None else 0, 0.0)
            adx_14 = self._safe_float(getattr(fv, "adx_14", 0) if fv is not None else 0, 0.0)
            high_24h = self._safe_float(getattr(fv, "high_24h", 0) if fv is not None else 0, 0.0)
            low_24h = self._safe_float(getattr(fv, "low_24h", 0) if fv is not None else 0, 0.0)
            hold_minutes = self.position_time.position_age_minutes(
                position_open_time(pos) or pos.get("created_at")
            )
            feature_price_suspicious_reason = (
                self.exit_fast_risk.suspicious_feature_price_reason(
                    side=side,
                    feature_price=fv_current_price,
                    position_price=position_current_price,
                    high_24h=high_24h,
                    low_24h=low_24h,
                    returns_1=returns_1,
                    returns_5=returns_5,
                )
                if fv_current_price > 0
                and hasattr(self.exit_fast_risk, "suspicious_feature_price_reason")
                else None
            )
            if feature_price_suspicious_reason:
                logger.warning(
                    "fast risk feature price marked suspicious",
                    model=model_name,
                    symbol=sym,
                    side=side,
                    reason=feature_price_suspicious_reason,
                    fv_current_price=fv_current_price,
                    position_current_price=position_current_price,
                    high_24h=high_24h,
                    low_24h=low_24h,
                    returns_1=returns_1,
                    returns_5=returns_5,
                )
                if position_current_price <= 0:
                    continue
                current_price = position_current_price
            if fv_current_price > 0 and position_current_price > 0:
                feature_position_gap = abs(fv_current_price - position_current_price) / max(
                    position_current_price, 1e-12
                )
                feature_price_implies_adverse = (
                    side == "long" and fv_current_price < position_current_price
                ) or (side == "short" and fv_current_price > position_current_price)
                short_returns_contradict_adverse = (
                    side == "long" and returns_1 >= 0 and returns_5 >= 0
                ) or (side == "short" and returns_1 <= 0 and returns_5 <= 0)
                if (
                    feature_position_gap >= FAST_RISK_MAX_FEATURE_POSITION_PRICE_GAP
                    and feature_price_implies_adverse
                    and short_returns_contradict_adverse
                ):
                    logger.warning(
                        "fast risk skipped due to contradictory feature current price",
                        model=model_name,
                        symbol=sym,
                        side=side,
                        fv_current_price=fv_current_price,
                        position_current_price=position_current_price,
                        gap_pct=round(feature_position_gap * 100, 4),
                        returns_1=returns_1,
                        returns_5=returns_5,
                        hold_minutes=hold_minutes,
                    )
                    continue
            close_fraction = 1.0
            fast_exit_plan: dict[str, Any] = {}
            profit_exit_plan: dict[str, Any] = {}
            current_unrealized = self._safe_float(pos.get("unrealized_pnl"), 0.0)
            peak_state = self.position_profit_peaks.update(
                model_name=model_name,
                symbol=sym,
                side=side,
                current_price=current_price,
                entry_price=entry_price,
                unrealized_pnl=current_unrealized,
                hold_minutes=hold_minutes,
            )

            hit_sl = False
            hit_tp = False
            # Direct hard-adverse exits are disabled; route hard moves through policy below.
            legacy_hard_adverse_direct_exit = False
            hard_adverse_observed = False
            hit_fast_adverse = False
            if side == "long":
                hit_sl = bool(stop_loss and current_price <= stop_loss)
                hit_tp = bool(take_profit and current_price >= take_profit)
                hard_adverse_observed = current_price <= entry_price * (1 - fast_adverse_pct)
                hit_fast_adverse = (
                    returns_1 <= -FAST_RISK_1M_MOVE_PCT or returns_5 <= -FAST_RISK_5M_MOVE_PCT
                )
                adverse_pct = max((entry_price - current_price) / entry_price, 0.0)
                stop_distance_pct = (
                    (entry_price - stop_loss) / entry_price if 0 < stop_loss < entry_price else 0.0
                )
            else:
                hit_sl = bool(stop_loss and current_price >= stop_loss)
                hit_tp = bool(take_profit and current_price <= take_profit)
                hard_adverse_observed = current_price >= entry_price * (1 + fast_adverse_pct)
                hit_fast_adverse = (
                    returns_1 >= FAST_RISK_1M_MOVE_PCT or returns_5 >= FAST_RISK_5M_MOVE_PCT
                )
                adverse_pct = max((current_price - entry_price) / entry_price, 0.0)
                stop_distance_pct = (
                    (stop_loss - entry_price) / entry_price if stop_loss > entry_price else 0.0
                )
            hit_near_stop_progress = bool(
                stop_distance_pct > 0
                and adverse_pct >= stop_distance_pct * FAST_RISK_NEAR_STOP_PROGRESS
            )
            hit_full_stop_progress = bool(
                stop_distance_pct > 0
                and adverse_pct >= stop_distance_pct * FAST_RISK_FULL_STOP_PROGRESS
            )
            stop_risk_progress = (
                adverse_pct / max(stop_distance_pct, 1e-12) if stop_distance_pct > 0 else 0.0
            )
            predictive_reversal = self.exit_predictive_reversal.evidence(
                side=side,
                returns_1=returns_1,
                returns_5=returns_5,
                returns_20=returns_20,
                volume_ratio=volume_ratio,
                rsi_14=rsi_14,
                bb_pct=bb_pct,
                macd_diff=macd_diff,
                adx_14=adx_14,
            )
            settlement_guard_active = (
                hold_minutes is not None
                and hold_minutes * 60.0 < ENTRY_SETTLEMENT_EXIT_GUARD_SECONDS
                and self._safe_float(predictive_reversal.get("score"), 0.0)
                < PREDICTIVE_REVERSAL_EXIT_SCORE
            )
            if not settlement_guard_active:
                profit_exit_plan = self.exit_fast_risk.profit_drawdown_exit_plan(
                    side=side,
                    current_price=current_price,
                    entry_price=entry_price,
                    unrealized_pnl=current_unrealized,
                    peak_state=peak_state,
                    hold_minutes=hold_minutes,
                    volume_ratio=volume_ratio,
                    returns_1=returns_1,
                    returns_5=returns_5,
                    returns_20=returns_20,
                    rsi_14=rsi_14,
                    bb_pct=bb_pct,
                    macd_diff=macd_diff,
                    adx_14=adx_14,
                )

            if hit_sl:
                trigger = "stop_loss"
                reason = (
                    "快速风控触发：价格已经触及本地记录的止损位，优先提交平仓，不等待 AI 会诊。"
                )
            elif hit_tp:
                trigger = "take_profit"
                reason = "快速风控触发：价格已经触及本地记录的止盈位，优先提交平仓锁定结果。"
            elif profit_exit_plan.get("should_exit"):
                fast_exit_plan = profit_exit_plan
                close_fraction = self._safe_float(profit_exit_plan.get("fraction"), 1.0)
                close_fraction = min(max(close_fraction, 0.05), 1.0)
                trigger = (
                    "profit_drawdown_reduce" if close_fraction < 0.999 else "profit_drawdown_close"
                )
                reason = (
                    "盈利保护触发：持仓曾经达到可保护浮盈，现在利润明显回撤，"
                    "本轮先于普通风控和慢专家复盘执行锁盈。"
                    f"{profit_exit_plan.get('note')}"
                )
            elif hit_full_stop_progress:
                trigger = "near_stop_progress"
                reason = (
                    "快速风控触发：亏损已经走完止损距离的"
                    f"{adverse_pct / max(stop_distance_pct, 1e-12):.0%}，"
                    f"超过强制退出阈值 {FAST_RISK_FULL_STOP_PROGRESS:.0%}。"
                    "为避免继续拖到完整止损，优先提交全平。"
                )
            elif legacy_hard_adverse_direct_exit:
                trigger = "hard_adverse_move"
                reason = (
                    "快速风控触发：价格已经相对开仓价出现明显反向波动，"
                    f"超过快速硬风险阈值 {fast_adverse_pct:.2%}，"
                    "这不是普通短线回撤，优先提交平仓控制单笔亏损。"
                )
            elif hard_adverse_observed or hit_fast_adverse or hit_near_stop_progress:
                fast_exit_plan = self.exit_fast_risk.fast_adverse_exit_plan(
                    side=side,
                    entry_price=entry_price,
                    current_price=current_price,
                    stop_loss=stop_loss,
                    returns_1=returns_1,
                    returns_5=returns_5,
                    hold_minutes=hold_minutes,
                    volume_ratio=volume_ratio,
                    current_unrealized_pnl=current_unrealized,
                    hard_adverse_observed=hard_adverse_observed,
                    data_quality_suspicious=bool(feature_price_suspicious_reason),
                    predictive_reversal_score=self._safe_float(
                        predictive_reversal.get("score"), 0.0
                    ),
                )
                if not fast_exit_plan.get("should_exit"):
                    logger.info(
                        "fast adverse move observed but held",
                        model=model_name,
                        symbol=sym,
                        side=side,
                        entry=entry_price,
                        current=current_price,
                        returns_1=returns_1,
                        returns_5=returns_5,
                        adverse_pct=fast_exit_plan.get("adverse_pct"),
                        hold_minutes=hold_minutes,
                        note=fast_exit_plan.get("note"),
                    )
                    if settlement_guard_active:
                        logger.info(
                            "settlement guard blocked profit drawdown check after fast adverse hold",
                            model=model_name,
                            symbol=sym,
                            side=side,
                            hold_minutes=hold_minutes,
                        )
                        continue
                    profit_exit_plan = self.exit_fast_risk.profit_drawdown_exit_plan(
                        side=side,
                        current_price=current_price,
                        entry_price=entry_price,
                        unrealized_pnl=current_unrealized,
                        peak_state=peak_state,
                        hold_minutes=hold_minutes,
                        volume_ratio=volume_ratio,
                        returns_1=returns_1,
                        returns_5=returns_5,
                        returns_20=returns_20,
                        rsi_14=rsi_14,
                        bb_pct=bb_pct,
                        macd_diff=macd_diff,
                        adx_14=adx_14,
                    )
                    if not profit_exit_plan.get("should_exit"):
                        continue
                    fast_exit_plan = profit_exit_plan
                    close_fraction = self._safe_float(profit_exit_plan.get("fraction"), 1.0)
                    close_fraction = min(max(close_fraction, 0.05), 1.0)
                    trigger = (
                        "profit_drawdown_reduce"
                        if close_fraction < 0.999
                        else "profit_drawdown_close"
                    )
                    reason = (
                        "盈利保护触发：短线有反向波动但未达到亏损止损条件；"
                        "同时浮盈已明显回撤，优先锁定剩余利润。"
                        f"{profit_exit_plan.get('note')}"
                    )
                else:
                    close_fraction = self._safe_float(fast_exit_plan.get("fraction"), 1.0)
                    close_fraction = min(max(close_fraction, 0.05), 1.0)
                    if settlement_guard_active and close_fraction < 0.999:
                        logger.info(
                            "settlement guard blocked fast partial reduce",
                            model=model_name,
                            symbol=sym,
                            side=side,
                            hold_minutes=hold_minutes,
                            close_fraction=close_fraction,
                        )
                        continue
                    trigger = (
                        "fast_adverse_reduce"
                        if close_fraction < 0.999
                        else ("hard_adverse_move" if hard_adverse_observed else "fast_adverse_move")
                    )
                    reason = (
                        "快速风控触发：1-5 分钟短线波动明显反向。" f"{fast_exit_plan.get('note')}"
                    )
            else:
                if settlement_guard_active:
                    logger.info(
                        "settlement guard blocked profit drawdown reduce",
                        model=model_name,
                        symbol=sym,
                        side=side,
                        hold_minutes=hold_minutes,
                    )
                    continue
                if not profit_exit_plan.get("should_exit"):
                    continue
                fast_exit_plan = profit_exit_plan
                close_fraction = self._safe_float(profit_exit_plan.get("fraction"), 1.0)
                close_fraction = min(max(close_fraction, 0.05), 1.0)
                trigger = (
                    "profit_drawdown_reduce" if close_fraction < 0.999 else "profit_drawdown_close"
                )
                reason = (
                    "盈利保护触发：持仓已有浮盈，但利润开始明显回撤。"
                    f"{profit_exit_plan.get('note')}"
                )

            close_action = Action.CLOSE_LONG if side == "long" else Action.CLOSE_SHORT
            close_decision = DecisionOutput(
                model_name=model_name,
                symbol=sym,
                action=close_action,
                confidence=1.0,
                reasoning=reason,
                position_size_pct=close_fraction,
                suggested_leverage=self._safe_float(pos.get("leverage"), 1.0),
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                raw_response={
                    "fast_risk_exit": True,
                    "fast_risk_trigger": trigger,
                    "fast_exit_plan": fast_exit_plan,
                    "close_fraction": close_fraction,
                    "returns_1": returns_1,
                    "returns_5": returns_5,
                    "returns_20": returns_20,
                    "predictive_reversal": predictive_reversal,
                    "hard_adverse_observed": hard_adverse_observed,
                    "feature_price_suspicious_reason": feature_price_suspicious_reason,
                },
                feature_snapshot={
                    "current_price": current_price,
                    "feature_current_price": fv_current_price,
                    "position_current_price": position_current_price,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "returns_1": returns_1,
                    "returns_5": returns_5,
                    "volume_ratio": volume_ratio,
                    "stop_risk_progress": stop_risk_progress,
                    "position_age_minutes": hold_minutes,
                    "adverse_from_entry_pct": (
                        fast_exit_plan.get("adverse_pct") if fast_exit_plan else None
                    ),
                    "close_fraction": close_fraction,
                    "fast_adverse_pct": fast_adverse_pct,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            logger.info(
                "fast position risk triggered",
                model=model_name,
                symbol=sym,
                side=side,
                trigger=trigger,
                entry=entry_price,
                current=current_price,
                sl=stop_loss,
                tp=take_profit,
                close_fraction=close_fraction,
            )

            fast_execution = await self._fast_risk_exit_execution_processor().execute(
                model_name=model_name,
                symbol=sym,
                side=side,
                position=pos,
                decision=close_decision,
                trigger=trigger,
                reason=reason,
                close_fraction=close_fraction,
                entry_price=entry_price,
                current_price=current_price,
            )
            if fast_execution.skipped:
                continue
            if fast_execution.auto_close is not None:
                auto_closes.append(fast_execution.auto_close)

        return auto_closes

    async def _review_open_positions(
        self,
        open_positions: list[dict],
        feature_vectors: dict,
        results: dict[str, Any] | None = None,
        round_decision_ids: set[int] | None = None,
        position_entry_pause_reason: str | None = None,
        max_groups_override: int | None = None,
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
                try:
                    fv = await self.data_service.get_feature_vector(normalized_symbol or symbol)
                except Exception as exc:
                    logger.debug(
                        "position review feature vector refresh failed",
                        symbol=normalized_symbol or symbol,
                        model=model_name,
                        error=safe_error_text(exc),
                    )
                    fv = None

            scan = fast_scan.get((model_name, symbol), {})
            release_policy = self._position_release_decision_policy()
            if release_policy.should_release(scan):
                release_decision = release_policy.build(
                    model_name=model_name,
                    symbol=symbol,
                    positions=positions,
                    scan=scan,
                    feature_vector=fv,
                )
                if isinstance(release_decision, DecisionOutput):
                    model_mode = self._get_model_execution_mode(model_name)
                    decision_db_id = await self._log_decision(
                        release_decision,
                        is_paper=(model_mode == "paper"),
                    )
                    self._decision_count += 1
                    if decision_db_id is not None and round_decision_ids is not None:
                        round_decision_ids.add(decision_db_id)
                    handled_keys.add((model_name, self._normalize_position_symbol(symbol)))
                    risk_alert = self.position_review_risk_alert_policy.build_alert(
                        release_decision,
                        positions,
                    )
                    if risk_alert:
                        self.position_review_risk_alert_policy.attach(
                            release_decision,
                            risk_alert,
                        )
                    process_result = await self.position_review_decision_processor.process(
                        decision=release_decision,
                        model_name=model_name,
                        symbol=symbol,
                        model_mode=model_mode,
                        decision_db_id=decision_db_id,
                        open_positions=open_positions,
                        feature_vector=fv,
                        position_entry_pause_reason=position_entry_pause_reason,
                        risk_alert=risk_alert,
                        results=results,
                    )
                    if process_result.candidate is not None:
                        candidates.append(process_result.candidate)
                    continue
            if fv is None:
                continue

            try:
                decision_result = await self.position_review_decision_service.decide(
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
                )
                if decision_result is None:
                    continue
                decision = decision_result.decision
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

            decision = self.position_review_decision_normalizer.normalize(decision, positions)
            model_mode = self._get_model_execution_mode(model_name)
            decision_db_id = await self._log_decision(decision, is_paper=(model_mode == "paper"))
            self._decision_count += 1
            if decision_db_id is not None and round_decision_ids is not None:
                round_decision_ids.add(decision_db_id)

            handled_keys.add((model_name, self._normalize_position_symbol(symbol)))

            risk_alert = self.position_review_risk_alert_policy.build_alert(decision, positions)
            if risk_alert:
                self.position_review_risk_alert_policy.attach(decision, risk_alert)

            process_result = await self.position_review_decision_processor.process(
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
            )
            if process_result.handled:
                continue
            if process_result.candidate is not None:
                candidates.append(process_result.candidate)

        return candidates, handled_keys

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
            predictive_reversal=getattr(
                self,
                "exit_predictive_reversal",
                ExitPredictiveReversalPolicy(),
            ),
            urgent_exit_markers=POSITION_REVIEW_URGENT_EXIT_MARKERS,
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

    def _position_release_decision_policy(self) -> PositionReleaseDecisionPolicy:
        policy = getattr(self, "position_release_decision", None)
        if policy is not None:
            return policy
        policy = PositionReleaseDecisionPolicy()
        self.position_release_decision = policy
        return policy

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

    async def _new_pair_analysis_pause_reason(
        self,
        model_name: str,
        open_positions: list[dict] | None = None,
    ) -> str | None:
        """Pause new-symbol AI analysis when balance/risk no longer allows entries."""
        breaker = self.risk_engine.circuit_breaker
        if breaker.is_open:
            state = breaker.get_state()
            return f"风险熔断已开启，暂停分析新的交易对。原因：{state.get('tripped_reason') or '触发风险阈值'}"

        model_mode = self._get_model_execution_mode(model_name)
        account_cfg = settings.get_execution_account_config(model_mode)
        okx_snapshot = await self._get_okx_balance_snapshot_for_mode(model_mode)
        if not okx_snapshot:
            return "未获取到 OKX 可用余额快照，暂停分析新的交易对。"
        okx_available = tradeable_balance_from_snapshot(okx_snapshot)
        okx_allocatable = allocatable_balance_from_snapshot(okx_snapshot)
        if okx_allocatable <= 0:
            return "未获取到 OKX 账户权益或余额，暂停分析新的交易对。"
        max_loss_pct = float(account_cfg.get("max_loss_pct") or settings.max_daily_loss_pct)
        max_loss_usdt = okx_allocatable * max_loss_pct if max_loss_pct > 0 else 0.0
        allocation_state = await self.execution_allocation_state(model_mode)
        total_pnl = float(allocation_state.get("total_pnl") or 0.0)
        account_guard = DEFAULT_TRADING_PARAMS.entry_account_guard
        min_available = max(
            account_guard.min_available_balance_usdt,
            okx_allocatable * account_guard.min_available_balance_ratio,
        )
        if okx_available <= min_available:
            return (
                f"OKX 可交易余额过低：当前可用 {okx_available:.2f} USDT，"
                f"最低需要 {min_available:.2f} USDT，暂停分析新的交易对。"
            )
        if max_loss_usdt > 0 and total_pnl <= -max_loss_usdt:
            return (
                f"执行账户已达到最高亏损限制：当前累计盈亏 {total_pnl:.2f} USDT，"
                f"最高允许亏损 {max_loss_usdt:.2f} USDT（{max_loss_pct * 100:.1f}%）。暂停分析新的交易对。"
            )

        model_positions = [
            p
            for p in (open_positions or [])
            if p.get("model_name") == model_name and p.get("is_open", True)
        ]
        capacity_reason = self.risk_engine.position_checker.entry_capacity_reason(
            current_positions=model_positions,
            account_balance=okx_allocatable,
            min_new_margin_pct=min(
                max(
                    float(settings.max_position_pct or 0.12)
                    / account_guard.capacity_min_new_margin_divisor,
                    account_guard.capacity_min_new_margin_floor_pct,
                ),
                float(settings.max_position_pct or 0.12),
            ),
            default_leverage=account_guard.capacity_default_leverage,
            default_stop_loss_pct=account_guard.capacity_default_stop_loss_pct,
        )
        if capacity_reason:
            logger.info(
                "new-pair scan remains active despite capacity warning; execution gate will recheck",
                reason=capacity_reason,
            )

        cooldown_loss_pct = float(account_cfg.get("cooldown_loss_pct") or 0.0)
        cooldown_loss_reason = await self.new_pair_loss_pause.cooldown_loss_pause_reason(
            model_mode,
            max_loss_usdt,
            cooldown_loss_pct,
        )
        if cooldown_loss_reason:
            logger.info(
                "loss cooldown is advisory; market scan remains active with tighter sizing",
                reason=cooldown_loss_reason,
            )
        loss_streak_reason = await self.new_pair_loss_pause.recent_loss_streak_pause_reason(
            model_mode,
            max_loss_usdt,
            cooldown_loss_pct,
        )
        if loss_streak_reason:
            logger.info(
                "recent loss streak is advisory; market scan remains active with tighter sizing",
                reason=loss_streak_reason,
            )
        return None

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
        # Also check legacy model
        if not settings.ai_models and settings.ai_api_key:
            self._model_execution_modes["llm_agent"] = "paper"

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

    async def _get_okx_available_balance_for_mode(self, mode: str) -> float | None:
        """Return the actual OKX free USDT balance used to cap new entries."""
        return await self.account_accounting_service.okx_available_balance_for_mode(mode)

    async def _paper_analysis_balance_snapshot(self) -> dict[str, Any] | None:
        """Return the paper account's analysis budget snapshot.

        Paper/demo mode may still submit orders through OKX demo and validate
        exchange rules at execution time. New-opportunity analysis, however,
        must not stop for hours just because the OKX balance REST endpoint is
        slow. When no in-process paper executor exists, fall back to the
        configured paper execution budget so the strategy loop can keep
        producing decisions with a clear degraded source marker.
        """

        executor = getattr(self, "paper_executor", None)
        summary: dict[str, Any] = {}
        source = "paper_configured_budget"
        degraded = True
        if executor is not None:
            try:
                summary = await executor.get_account_summary(ENSEMBLE_TRADER_NAME)
            except Exception as exc:
                logger.warning(
                    "failed to build paper virtual balance snapshot",
                    error=safe_error_text(exc),
                )
            else:
                source = "paper_virtual_account"
                degraded = False

        cfg = settings.get_execution_account_config("paper")
        configured_budget = self._safe_float(cfg.get("allocated_balance"), 0.0)
        if configured_budget <= 0:
            configured_budget = self._safe_float(settings.initial_virtual_balance, 0.0)

        available = self._safe_float(summary.get("available_balance"), configured_budget)
        equity = self._safe_float(summary.get("equity"), available)
        wallet = self._safe_float(summary.get("wallet_balance"), max(equity, available))
        used = self._safe_float(summary.get("used_margin"), 0.0)
        total = (
            max(equity, wallet, available)
            if source == "paper_virtual_account"
            else max(equity, wallet, available, configured_budget)
        )
        if total <= 0 and available <= 0:
            return None

        return {
            "free": max(available, 0.0),
            "used": max(used, 0.0),
            "total": max(total, 0.0),
            "cash": max(wallet, available, 0.0),
            "equity": max(equity, total, 0.0),
            "allocatable": max(available, 0.0),
            "source": source,
            "exchange_required": False,
            "degraded": degraded,
            "analysis_only_balance": True,
        }

    async def _get_okx_balance_snapshot_for_mode(self, mode: str) -> dict[str, Any] | None:
        """Return OKX USDT balance fields for allocation and order sizing."""
        selected_mode = "live" if mode == "live" else "paper"

        def remember_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
            self._okx_balance_snapshot_cache[selected_mode] = {
                "snapshot": dict(snapshot),
                "fetched_at": datetime.now(UTC),
            }
            return snapshot

        def cached_snapshot(reason: str) -> dict[str, Any] | None:
            cached = self._okx_balance_snapshot_cache.get(selected_mode)
            if not isinstance(cached, dict):
                return None
            fetched_at = cached.get("fetched_at")
            if not isinstance(fetched_at, datetime):
                return None
            age = (datetime.now(UTC) - fetched_at).total_seconds()
            if age > OKX_BALANCE_SNAPSHOT_CACHE_SECONDS:
                return None
            snapshot = dict(cached.get("snapshot") or {})
            if not snapshot:
                return None
            snapshot["stale"] = True
            snapshot["stale_age_seconds"] = round(age, 3)
            snapshot["stale_reason"] = reason
            logger.warning(
                "using cached OKX balance snapshot",
                mode=selected_mode,
                age_seconds=round(age, 3),
                reason=reason,
            )
            return snapshot

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
                snapshot["fallback_executor"] = True
                snapshot["fallback_reason"] = reason
                logger.warning(
                    "fresh OKX balance snapshot fallback succeeded",
                    mode=selected_mode,
                    original_reason=reason,
                )
                return remember_snapshot(snapshot)
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
        if not executor:
            try:
                executor = await self._get_okx_executor_for_mode(selected_mode)
            except Exception as exc:
                reason = f"executor unavailable: {safe_error_text(exc)}"
                fallback_snapshot = cached_snapshot(reason) or await fresh_executor_snapshot(reason)
                if fallback_snapshot:
                    return fallback_snapshot
                if selected_mode == "paper":
                    return await self._paper_analysis_balance_snapshot()
                return None
        try:
            snapshot = await asyncio.wait_for(
                executor.get_balance_snapshot("USDT"),
                timeout=8.0,
            )
            if snapshot.get("error"):
                reason = safe_error_text(snapshot.get("error"))
                fallback_snapshot = cached_snapshot(reason) or await fresh_executor_snapshot(reason)
                if fallback_snapshot:
                    return fallback_snapshot
                if selected_mode == "paper":
                    return await self._paper_analysis_balance_snapshot()
                return None
            return remember_snapshot(snapshot)
        except TimeoutError:
            logger.warning("timed out fetching OKX balance snapshot", mode=selected_mode)
            reason = "OKX balance snapshot request timed out"
            fallback_snapshot = cached_snapshot(reason) or await fresh_executor_snapshot(reason)
            if fallback_snapshot:
                return fallback_snapshot
            if selected_mode == "paper":
                return await self._paper_analysis_balance_snapshot()
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
            if selected_mode == "paper":
                return await self._paper_analysis_balance_snapshot()
            return None

    async def _sync_paper_after_okx(
        self,
        model_name: str,
        decision: DecisionOutput,
        result: ExecutionResult,
    ) -> None:
        """Update PaperExecutor tracking after a successful OKX execution."""
        pe = self.paper_executor
        if pe is None:
            return

        price = result.price
        quantity = result.quantity
        fee = result.fee
        order_value = quantity * price

        if decision.action in (Action.LONG, Action.SHORT):
            margin_used = self.position_margin_calculator.margin(
                order_value, decision.suggested_leverage
            )
            old_balance = pe._balances.get(model_name, settings.get_initial_balance(model_name))
            balance_delta = -(margin_used + fee)
            pe._balances[model_name] = old_balance + balance_delta

            # Record position
            side = "long" if decision.action == Action.LONG else "short"
            position = {
                "id": result.order_id or str(uuid.uuid4())[:12],
                "symbol": decision.symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": price,
                "current_price": price,
                "leverage": decision.suggested_leverage,
                "margin_used": margin_used,
                "stop_loss": (
                    price * (1 - decision.stop_loss_pct)
                    if side == "long"
                    else price * (1 + decision.stop_loss_pct)
                ),
                "take_profit": (
                    price * (1 + decision.take_profit_pct)
                    if side == "long"
                    else price * (1 - decision.take_profit_pct)
                ),
                "is_open": True,
                "opened_at": datetime.now(UTC),
                "unrealized_pnl": 0.0,
            }
            pe._positions.setdefault(model_name, []).append(position)
            await self.account_accounting_service.persist_balance_delta(
                model_name,
                balance_delta,
                0.0,
            )

        elif decision.action in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
            target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
            positions = pe._positions.get(model_name, [])
            to_close = [
                p
                for p in positions
                if p["symbol"] == decision.symbol and p["side"] == target_side and p["is_open"]
            ]
            total_pnl = 0.0
            released_margin = 0.0
            total_fee = fee
            for pos in to_close:
                pos["is_open"] = False
                pos["current_price"] = price
                if pos["side"] == "long":
                    pnl = (price - pos["entry_price"]) * pos["quantity"]
                else:
                    pnl = (pos["entry_price"] - price) * pos["quantity"]
                pos["unrealized_pnl"] = pnl
                pos["closed_at"] = datetime.now(UTC)
                total_pnl += pnl
                released_margin += self.position_margin_calculator.margin(
                    pos["quantity"] * pos["entry_price"],
                    pos.get("leverage", 1.0),
                )

            if not to_close:
                released_margin = await self._db_released_margin_for_close(
                    model_name, decision, result
                )
            pe._balances[model_name] = (
                pe._balances.get(model_name, settings.get_initial_balance(model_name))
                + released_margin
                + total_pnl
                - total_fee
            )
            pe._positions[model_name] = [p for p in positions if p["is_open"]]
            # Attach PnL to result for downstream logging
            result.pnl = total_pnl - total_fee
            await self.account_accounting_service.persist_balance_delta(
                model_name,
                released_margin + total_pnl - total_fee,
                total_pnl - total_fee,
            )

    async def _db_released_margin_for_close(
        self,
        model_name: str,
        decision: DecisionOutput,
        result: ExecutionResult,
    ) -> float:
        # Fallback used when the in-memory paper executor was restarted and no
        # longer has the matching position object. The DB close handler will
        # still close the real persisted position later in the same flow.
        target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
        remaining_qty = result.quantity
        released_margin = 0.0
        try:
            async with get_session_ctx() as session:
                repo = TradeRepository(session)
                positions = await repo.get_matching_open_positions(
                    model_name=model_name,
                    symbol=result.symbol,
                    side=target_side,
                    execution_mode="paper",
                )
                for pos in positions:
                    if remaining_qty <= 0:
                        break
                    close_qty = min(float(pos.quantity or 0.0), remaining_qty)
                    released_margin += self.position_margin_calculator.margin(
                        close_qty * float(pos.entry_price or 0.0),
                        pos.leverage,
                    )
                    remaining_qty -= close_qty
        except Exception as e:
            logger.error("failed to calculate released paper margin", error=safe_error_text(e))
        return released_margin

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
                self.data_service.get_feature_vector(normalized),
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
        """Record current equity for each model for PnL chart history, and persist unrealized PnL to DB."""
        now = datetime.now(UTC).isoformat()
        active_names = {ENSEMBLE_TRADER_NAME}
        # Remove stale data from deleted models
        for stale in list(self._pnl_history.keys()):
            if stale not in active_names:
                del self._pnl_history[stale]
        for model_name in active_names:
            state = await self.execution_allocation_state(mode_manager.mode.value)
            allocated = float(state.get("allocated_balance") or 0.0)
            equity = allocated
            unrealized = state.get("unrealized_pnl", 0.0)
            self._pnl_history.setdefault(model_name, []).append(
                {
                    "time": now,
                    "equity": round(equity, 2),
                }
            )
            # Keep last 500 snapshots per model
            if len(self._pnl_history[model_name]) > 500:
                self._pnl_history[model_name] = self._pnl_history[model_name][-500:]

            # Persist unrealized PnL to DB so competition rankings see it
            await self.account_accounting_service.record_unrealized_pnl(
                model_name,
                float(unrealized or 0.0),
            )

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
            "sync_exchange_positions": "\u540c\u6b65 OKX \u4ed3\u4f4d/\u4fdd\u62a4\u5355",
            "load_open_positions": "\u8bfb\u53d6\u672c\u5730\u6301\u4ed3",
            "recover_pending_exits": "\u8865\u6267\u884c\u672a\u5b8c\u6210\u5e73\u4ed3",
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
        }
        self._write_runtime_heartbeat()
        return stats
