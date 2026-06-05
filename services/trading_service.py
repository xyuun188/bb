"""
Trading service 鈥?the central orchestrator.
Wires together data feed, AI brain, risk manager, and executor
into the main trading loop.
"""

from __future__ import annotations

import asyncio 
import json
import math
import uuid 
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import structlog
from sqlalchemy import func, or_, select

from ai_brain.base_model import Action, DecisionOutput
from ai_brain.ensemble_coordinator import EnsembleCoordinator
from ai_brain.model_registry import ModelRegistry
from config.settings import ENSEMBLE_TRADER_NAME, FIXED_AI_MODEL_SLOTS, settings
from core.trading_mode import TradingMode, mode_manager
from db.repositories.account_repo import AccountRepository
from db.repositories.decision_repo import DecisionRepository
from db.repositories.memory_repo import MemoryRepository
from db.repositories.risk_repo import RiskRepository
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from executor.paper_executor import PaperExecutor
from executor.okx_executor import OKXExecutor
from executor.base_executor import ExecutionResult, OrderStatus
from models.decision import AIDecision
from models.learning import ShadowBacktest
from models.trade import Order, Position
from risk_manager.engine import RiskEngine
from services.data_service import DataService
from services.equity_baseline import apply_daily_equity_baseline
from services.trading_agent_skills import TradingAgentSkillBook
from services.local_ai_tools_client import LocalAIToolsClient
from services.ml_signal_service import AUTO_TRAIN_CHECK_INTERVAL_SECONDS, MLSignalService
from services.analysis_services import MarketAnalysisService, PositionReviewService
from services.decision_state import DecisionStage, DecisionStageStatus, append_decision_stage
from services.execution_service import ExecutionService
from services.strategy_arbitration import arbitrate_decision
from services.sync_service import OkxSyncService
from services.trading_policies import EntryPolicy, ExitPolicy
from web_dashboard.api.text_sanitize import sanitize_text

logger = structlog.get_logger(__name__)

BEIJING_TZ = timezone(timedelta(hours=8))
PRE_AGENT_SKILLS_ROLLBACK_MODE = False
AGENT_SKILLS_TRADING_EFFECTS_ENABLED = True
LOCAL_QUANT_PROMPT_ENABLED = True
LOCAL_QUANT_MARKET_PREFILTER_ENABLED = True
ESTIMATED_TAKER_FEE_PCT = 0.0005
OKX_BALANCE_SNAPSHOT_CACHE_SECONDS = 120.0
MIN_DISCRETIONARY_HOLD_MINUTES = 4.0
ENTRY_SETTLEMENT_EXIT_GUARD_SECONDS = 120.0
DISCRETIONARY_CLOSE_CONFIDENCE = 0.66
PROFIT_PROTECTION_MIN_NET_PNL_RATIO = 0.004
PROFIT_PROTECTION_STRONG_NET_PNL_RATIO = 0.010
PROFIT_PROTECTION_MIN_NET_USDT = 3.00
PROFIT_PROTECTION_MIN_FEE_MULTIPLE = 4.0
PROFIT_PROTECTION_STRONG_FEE_MULTIPLE = 5.0
PROFIT_PROTECTION_EXIT_MAX_AGE_SECONDS = 300.0
PROFIT_DRAWDOWN_MIN_HOLD_MINUTES = 8.0
PROFIT_DRAWDOWN_MIN_PROFIT_RATIO = 0.006
PROFIT_DRAWDOWN_STRONG_PROFIT_RATIO = 0.016
PROFIT_DRAWDOWN_PARTIAL_RETRACE = 0.38
PROFIT_DRAWDOWN_FULL_RETRACE = 0.68
PROFIT_DRAWDOWN_PARTIAL_CLOSE_FRACTION = 0.35
PROFIT_DRAWDOWN_MIN_NET_USDT = 5.0
PROFIT_DRAWDOWN_MIN_FEE_MULTIPLE = 4.0
PROFIT_DRAWDOWN_MIN_SECONDS_BETWEEN_EXITS = 600.0
PROFIT_DRAWDOWN_VOLUME_CONFIRM_RATIO = 1.05
PROFIT_DRAWDOWN_ACCELERATED_HOLD_MINUTES = 8.0
PREDICTIVE_REVERSAL_REVIEW_SCORE = 38.0
PREDICTIVE_REVERSAL_EXIT_SCORE = 64.0
PREDICTIVE_REVERSAL_FULL_EXIT_SCORE = 82.0
PREDICTIVE_REVERSAL_MIN_PROFIT_MULTIPLE = 1.0
PREDICTIVE_REVERSAL_REDUCE_FRACTION = 0.60
POSITION_PROFIT_PEAKS_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "position_profit_peaks.json"
ENTRY_DECISION_MAX_AGE_SECONDS = 300.0
ENTRY_PENDING_EXECUTION_MAX_SECONDS = 45.0
ENTRY_STRONG_OPPORTUNITY_MAX_AGE_SECONDS = 240.0
ENTRY_EXCEPTIONAL_OPPORTUNITY_MAX_AGE_SECONDS = 300.0
EXIT_DECISION_MAX_AGE_SECONDS = 120.0
EXIT_SYMBOL_SIDE_COOLDOWN_SECONDS = 600.0
FAST_RISK_1M_MOVE_PCT = 0.025
FAST_RISK_5M_MOVE_PCT = 0.04
FAST_RISK_MIN_HOLD_MINUTES = 4.0
FAST_RISK_MIN_LOSS_PCT = 0.008
FAST_RISK_REDUCE_LOSS_PCT = 0.012
FAST_RISK_FULL_LOSS_PCT = 0.018
FAST_RISK_NEAR_STOP_PROGRESS = 0.50
FAST_RISK_FULL_STOP_PROGRESS = 0.78
FAST_RISK_VOLUME_CONFIRM_RATIO = 1.05
FAST_RISK_REDUCE_POSITION_PCT = 0.50
FAST_RISK_FORCE_FULL_LOSS_USDT = 4.0
FAST_RISK_FORCE_FULL_PROGRESS = 0.50
RECENT_LOSS_LOOKBACK_COUNT = 20
LOSS_STREAK_PAUSE_MINUTES = 30.0
DAILY_TARGET_LOSS_PAUSE_MIN_USDT = 50.0
DAILY_TARGET_PROFIT_LOCK_RATIO = 0.15
DAILY_TARGET_PROFIT_FLOOR_RATIO = 0.90
DAILY_TARGET_MIN_PROFIT_FACTOR = 0.55
DAILY_TARGET_MIN_TRADES_FOR_FACTOR = 8
UNCONFIRMED_EXCHANGE_CLOSE_GRACE_SECONDS = 180.0
UNTRADABLE_SYMBOL_BLOCK_HOURS = 24.0
TRANSIENT_ENTRY_BLOCK_MINUTES = 20.0
PRICE_GUARD_ENTRY_BLOCK_MINUTES = 8.0
ENTRY_PRICE_RECHECK_TIMEOUT_SECONDS = 5.0
ENTRY_PRICE_RECHECK_RESCUE_MAX_MOVE_PCT = 0.012
ENTRY_PRICE_RECHECK_EXCEPTIONAL_MAX_MOVE_PCT = 0.020
ENTRY_PRICE_RECHECK_EXPECTED_BUFFER_MULTIPLE = 2.0
ENTRY_PRICE_FIELD_SPLIT_BLOCK_PCT = 0.08
ENTRY_PRICE_24H_RANGE_TOLERANCE_PCT = 0.03
ENTRY_BLACK_SWAN_RECHECK_SAFE_1M_DROP = -0.08
ENTRY_BLACK_SWAN_RECHECK_SAFE_5M_DROP = -0.12
ENTRY_BLACK_SWAN_REBOUND_MIN_1M = 0.001
ENTRY_BLACK_SWAN_REBOUND_MIN_5M = 0.003
ENTRY_BLACK_SWAN_REBOUND_MIN_EXPECTED_NET = 0.60
ENTRY_BLACK_SWAN_REBOUND_MIN_PROFIT_QUALITY = 1.00
ENTRY_BLACK_SWAN_REBOUND_MIN_CONFIDENCE = 0.72
ENTRY_BLACK_SWAN_REBOUND_MAX_SPREAD = 0.015
ENTRY_POST_CRASH_REBOUND_1M = 0.030
ENTRY_POST_CRASH_REBOUND_5M_DROP = -0.18
ENTRY_POST_CRASH_REBOUND_20M_DROP = -0.25
ENTRY_DATA_STALE_ZERO_RETURNS_MIN_24H_CHANGE = 0.003
SHADOW_BACKTEST_HORIZONS_MINUTES = (10, 30, 60)
SHADOW_MISSED_OPPORTUNITY_THRESHOLD = 0.004
MIN_ENTRY_OPPORTUNITY_SCORE = 0.95
ENTRY_NET_WEIGHT_AI = 0.20
ENTRY_NET_WEIGHT_LOCAL_ML = 0.34
ENTRY_NET_WEIGHT_SERVER_PROFIT = 0.32
ENTRY_NET_WEIGHT_TIMESERIES = 0.14
ENTRY_SMALL_WIN_BIG_LOSS_PENALTY_CAP = 0.90
ENTRY_REALIZED_EDGE_BONUS_CAP = 0.85
ENTRY_REALIZED_EDGE_PENALTY_CAP = 1.15
ENTRY_SYMBOL_WINNER_MIN_COUNT = 2
ENTRY_SYMBOL_WINNER_MIN_PNL_USDT = 5.0
ENTRY_SYMBOL_WINNER_MIN_PROFIT_FACTOR = 1.20
ENTRY_SYMBOL_WINNER_SCORE_RELIEF = 0.12
ENTRY_SYMBOL_WINNER_SCORE_BONUS_CAP = 0.45
ENTRY_SYMBOL_LOSER_SIZE_MULTIPLIER = 0.55
ENTRY_PNL_STRUCTURE_MIN_EXPECTED_PROFIT_USDT = 1.50
ENTRY_PNL_STRUCTURE_LOW_QUALITY_MAX_LOSS_MULTIPLE = 0.65
ENTRY_PNL_STRUCTURE_NORMAL_MAX_LOSS_MULTIPLE = 1.05
ENTRY_PNL_STRUCTURE_HIGH_QUALITY_MAX_LOSS_MULTIPLE = 1.35
ML_EXPECTED_RETURN_SCORE_CAP_PCT = 3.0
QUANT_PROFIT_PROBE_MIN_EXPECTED_PCT = 0.18
QUANT_PROFIT_PROBE_MIN_EDGE_PCT = 0.22
QUANT_PROFIT_PROBE_MIN_CONCENTRATED_SHORT_LOSS_USDT = 5.0
QUANT_PROFIT_PROBE_MIN_SCORE = 0.35
QUANT_PROFIT_PROBE_MIN_PROFIT_QUALITY_RATIO = 0.12
ENTRY_PROBE_MAX_PRICE_FIELD_GAP = 0.03
ENTRY_PROBE_STRONG_CONTRA_20M_PCT = 0.05
FAST_RISK_MAX_FEATURE_POSITION_PRICE_GAP = 0.03
ENTRY_DIRECTION_HARD_CONFLICT_GAP = 0.12
ENTRY_DIRECTION_MIN_SUPPORT_SCORE = 0.02
DYNAMIC_ENTRY_SCORE_ML_ALIGNED_STRONG = 0.75
DYNAMIC_ENTRY_SCORE_ML_ALIGNED = 0.85
DYNAMIC_ENTRY_SCORE_EXPERT_ALIGNED = 0.90
ENTRY_MIN_NET_PROFIT_QUALITY_RATIO = 1.50
ENTRY_WEAK_HISTORY_MIN_PROFIT_QUALITY_RATIO = 2.00
ENTRY_STRONG_ALIGNED_MIN_PROFIT_QUALITY_RATIO = 0.85
ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MIN_PROFIT_QUALITY_RATIO = 1.05
ENTRY_WEAK_HISTORY_MIN_SCORE = 3.20
ENTRY_WEAK_HISTORY_MAX_SIZE = 0.025
ENTRY_WEAK_HISTORY_MAX_LEVERAGE = 5.0
ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MAX_SIZE = 0.045
ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MAX_LEVERAGE = 8.0
ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_SIZE = 0.02
ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_LEVERAGE = 4.0
SYMBOL_SIDE_PROFILE_LOOKBACK = 2000
SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS = 7.0
SYMBOL_SIDE_COOLDOWN_LOSS_USDT = 25.0
SYMBOL_TOTAL_COOLDOWN_LOSS_USDT = 60.0
SYMBOL_QUARANTINE_LOSS_USDT = 120.0
SYMBOL_QUARANTINE_MIN_LOSSES = 2
SYMBOL_SIDE_HARD_COOLDOWN_HOURS = 6.0
ENTRY_LOSS_COOLDOWN_OVERRIDE_MIN_CONFIDENCE = 0.72
ENTRY_LOSS_COOLDOWN_OVERRIDE_MIN_SCORE = 3.20
ENTRY_LOSS_COOLDOWN_OVERRIDE_SCORE_MULTIPLE = 1.45
ENTRY_LOSS_COOLDOWN_OVERRIDE_MIN_EXPECTED_NET = 0.90
ENTRY_LOSS_COOLDOWN_OVERRIDE_MIN_PROFIT_QUALITY = 0.90
ENTRY_LOSS_COOLDOWN_OVERRIDE_MIN_REWARD_RISK = 1.30
ENTRY_LOSS_COOLDOWN_OVERRIDE_MIN_SERVER_EXPECTED = 0.80
ENTRY_LOSS_COOLDOWN_OVERRIDE_MAX_LOSS_PROBABILITY = 0.48
ENTRY_LOSS_COOLDOWN_OVERRIDE_MAX_TAIL_RISK = 0.85
DRAWDOWN_REDUCED_RISK_USDT = 100.0
DRAWDOWN_DEFENSIVE_RISK_USDT = 220.0
ENTRY_MAX_STOP_LOSS_NORMAL_USDT = 16.0
ENTRY_MAX_STOP_LOSS_DRAWDOWN_USDT = 8.0
ENTRY_MAX_STOP_LOSS_DEFENSIVE_USDT = 4.0
ENTRY_HIGH_QUALITY_STOP_LOSS_DRAWDOWN_USDT = 12.0
ENTRY_HIGH_QUALITY_STOP_LOSS_DEFENSIVE_USDT = 8.0
ENTRY_MAX_STOP_LOSS_PCT_OF_EQUITY = 0.008
ENTRY_MAX_STOP_LOSS_CAP_USDT = 36.0
ENTRY_STRESS_STOP_MIN_PCT = 0.018
ENTRY_STRESS_STOP_MAX_PCT = 0.080
ENTRY_LOW_QUALITY_STRESS_STOP_MIN_PCT = 0.050
ENTRY_LOW_QUALITY_MAX_SIZE = 0.018
ENTRY_LOW_QUALITY_MAX_LEVERAGE = 3.0
ENTRY_HIGH_QUALITY_MIN_NOTIONAL_BALANCE_RATIO = 0.10
ENTRY_NORMAL_MIN_NOTIONAL_BALANCE_RATIO = 0.06
ENTRY_NOTIONAL_FLOOR_MAX_SIZE_PCT = 0.12
ENTRY_HIGH_PROFIT_MIN_NOTIONAL_BALANCE_RATIO = 0.75
ENTRY_HIGH_PROFIT_MIN_LEVERAGE = 8.0
ENTRY_HIGH_PROFIT_ELITE_MIN_LEVERAGE = 10.0
ENTRY_GOOD_PROBE_MIN_NOTIONAL_BALANCE_RATIO = 0.25
ENTRY_WINNER_ADD_MIN_NOTIONAL_BALANCE_RATIO = 0.35
ENTRY_STRONG_PROBE_MIN_NOTIONAL_BALANCE_RATIO = 0.45
ENTRY_ELITE_MIN_NOTIONAL_BALANCE_RATIO = 0.60
ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK = 0.82
ENTRY_MEANINGFUL_SIZE_MIN_PROFIT_USDT = 0.75
ENTRY_MEANINGFUL_SIZE_MIN_PROFIT_RATIO = 0.003
PORTFOLIO_MIN_POSITION_GROUPS_TARGET = 10
PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN = 36
PORTFOLIO_ROSTER_FILL_MIN_EXPECTED_PCT = 0.08
PORTFOLIO_ROSTER_FILL_MIN_EDGE_PCT = 0.08
PORTFOLIO_ROSTER_FILL_MAX_LOSS_PROBABILITY = 0.66
PORTFOLIO_ROSTER_FILL_MIN_NET_PCT = 0.20
PORTFOLIO_ROSTER_FILL_MIN_PROFIT_QUALITY_RATIO = 0.25
PORTFOLIO_ROSTER_FILL_NOTIONAL_BALANCE_RATIO = 0.18
AUTO_TRADE_MIN_NOTIONAL_24H = 4_000_000.0
AUTO_TRADE_MIN_VOLUME_RATIO = 0.65
AUTO_TRADE_MAX_VOLATILITY_20 = 0.085
AUTO_TRADE_MAX_ABS_CHANGE_24H = 12.0
ABNORMAL_WICK_ENTRY_BLOCK_MAX_PCT = 80.0
ABNORMAL_WICK_ENTRY_BLOCK_RECENT_HOURS = 96.0
ABNORMAL_WICK_ENTRY_BLOCK_MIN_COUNT = 1
ABNORMAL_WICK_TAIL_RISK_MAX_PCT = 60.0
AUTO_SCAN_ROTATION_POOL_MULTIPLIER = 20
AUTO_SCAN_ROTATION_POOL_MIN = 240
MARKET_NO_OPPORTUNITY_WINDOW_MINUTES = 45.0
MARKET_NO_OPPORTUNITY_RECHECK_MINUTES = 18.0
MARKET_NO_OPPORTUNITY_STREAK_THRESHOLD = 2
MARKET_NO_OPPORTUNITY_MAX_PENALTY = 240.0
POSITION_REVIEW_MAX_GROUPS_PER_ROUND = 6
POSITION_REVIEW_PRIORITY_MAX_GROUPS_PER_ROUND = 4
POSITION_REVIEW_HIGH_RISK_MAX_GROUPS_PER_ROUND = 8
POSITION_REVIEW_URGENT_EXIT_MAX_GROUPS_PER_ROUND = 14
POSITION_REVIEW_FAST_EXIT_SCORE = 70.0
POSITION_REVIEW_FAST_ADD_SCORE = 62.0
POSITION_REVIEW_URGENT_EXIT_MARKERS = (
    "loss_expanding",
    "loss_needs_review",
    "near_stop",
    "adverse_momentum",
    "predictive_reversal",
)
MARKET_ANALYSIS_MIN_EXPLORATION_SYMBOLS = 2
MARKET_ANALYSIS_HIGH_RISK_MIN_EXPLORATION_SYMBOLS = 1
MARKET_ANALYSIS_MEDIUM_RISK_CAP = 4
MARKET_ANALYSIS_HIGH_RISK_CAP = 2
PORTFOLIO_PROFIT_PROTECTION_MIN_USDT = 3.0
PORTFOLIO_PROFIT_PROTECTION_MIN_CONTRIBUTION_USDT = 0.6
PORTFOLIO_PROFIT_PROTECTION_MIN_SHARE = 0.05
PORTFOLIO_PROFIT_PROTECTION_MAX_FOCUS_GROUPS = 5
PORTFOLIO_PROFIT_PROTECTION_EXIT_SCORE = 82.0
ALT_LONG_BTC_ETH_5M_FLOOR = -0.0015
ALT_LONG_BTC_ETH_20M_FLOOR = -0.004
ALT_LONG_BTC_ETH_ADX_FLOOR = 16.0
ALT_LONG_ALLOWED_SYMBOLS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"}
SUSPICIOUS_NEW_SYMBOL_TOKENS = ("TEST", "DEMO", "DUMMY", "MOCK", "SAMPLE")
ALT_LONG_SOFT_ALLOW_CONFIDENCE = 0.72
ALT_LONG_SOFT_ALLOW_MAX_SIZE = 0.18
DRAWDOWN_LIGHT_RISK_USDT = 30.0
DRAWDOWN_HARD_PAUSE_USDT = 80.0
# Backward-compatible aliases for earlier typoed references.
DRAWNDOWN_LIGHT_RISK_USDT = DRAWDOWN_LIGHT_RISK_USDT
DRAWNDOWN_HARD_PAUSE_USDT = DRAWDOWN_HARD_PAUSE_USDT


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
        self.risk_engine = RiskEngine()
        self.ml_signal_service = MLSignalService()
        self.local_ai_tools = LocalAIToolsClient()
        self.agent_skills = TradingAgentSkillBook()
        self.redis = redis_client
        self.market_analysis_service = MarketAnalysisService(self)
        self.position_review_service = PositionReviewService(self)
        self.execution_service = ExecutionService(self)
        self.okx_sync_service = OkxSyncService(self)
        self.entry_policy = EntryPolicy(self)
        self.exit_policy = ExitPolicy(self)

        # Executors 鈥?paper routes to OKX demo, live routes to OKX real.
        self.paper_executor: PaperExecutor | None = None
        self.okx_executor: OKXExecutor | None = None  # kept for backward compat
        self._okx_paper: OKXExecutor | None = None
        self._okx_live: OKXExecutor | None = None
        self._model_execution_modes: dict[str, str] = {}  # model_name 鈫?"paper"/"live"

        self._running = False
        self._decision_count = 0
        self._trade_count = 0
        self._recent_decisions: list[dict] = []
        self._recent_executions: list[dict] = []
        self._start_time: datetime | None = None
        self._current_stage = "idle"
        self._last_round_started_at: datetime | None = None
        self._last_round_finished_at: datetime | None = None
        self._last_round_error: str | None = None
        self._pnl_history: dict[str, list[dict]] = {}  # model_name -> [{time, equity}, ...]
        self._new_pair_pause_reasons: dict[str, str] = {}
        self._untradable_symbols: dict[str, dict[str, Any]] = {}
        self._recent_market_hold_symbols: dict[str, datetime] = {}
        self._market_no_opportunity_symbols: dict[str, dict[str, Any]] = {}
        self._recent_market_analyzed_symbols: dict[str, datetime] = {}
        self._position_profit_peaks: dict[str, dict[str, Any]] = self._load_position_profit_peaks()
        self._okx_balance_snapshot_cache: dict[str, dict[str, Any]] = {}
        self._position_review_cursor = 0
        self._position_review_priority_cursor = 0
        self._position_review_defer_counts: dict[tuple[str, str], int] = {}
        self._recent_exit_groups: dict[tuple[str, str, str], datetime] = {}
        self._active_analysis_symbols: set[str] = set()
        self._analysis_symbol_lock = asyncio.Lock()
        self._execution_lock = asyncio.Lock()
        self._exchange_reconcile_lock = asyncio.Lock()
        self._market_analysis_task: asyncio.Task | None = None
        self._position_analysis_task: asyncio.Task | None = None
        self._realized_expert_weight_cache: dict[str, Any] = {"expires_at": None, "weights": {}}
        self._model_contribution_cache: dict[str, Any] = {"expires_at": None, "stats": {}}
        self._ml_auto_train_task: asyncio.Task | None = None
        self._local_tools_last_train_started_at: datetime | None = None
        self._local_tools_last_completed_shadow_count: int = 0

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
            logger.debug("local AI tools enrichment failed", symbol=getattr(fv, "symbol", None), error=str(exc))
            return {
                "enabled": bool(settings.local_ai_tools_enabled),
                "status": "error",
                "error": str(exc)[:180],
            }

    def _direction_competition_context(
        self,
        fv: Any,
        ml_signal_context: dict[str, Any] | None,
        local_ai_tools_context: dict[str, Any] | None,
        market_regime: dict[str, Any] | None,
        strategy_mode: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build a long-vs-short evidence summary for AI and entry ranking."""
        sides: dict[str, dict[str, Any]] = {
            "long": {"score": 0.0, "expected_return_pct": 0.0, "loss_probability": None, "evidence": []},
            "short": {"score": 0.0, "expected_return_pct": 0.0, "loss_probability": None, "evidence": []},
        }
        strategy = strategy_mode if isinstance(strategy_mode, dict) else {}

        def source_weight(source: str) -> float:
            perf = strategy.get("model_contribution_performance")
            if not isinstance(perf, dict):
                return 1.0
            bucket = perf.get(source)
            if not isinstance(bucket, dict):
                return 1.0
            count = int(bucket.get("count") or 0)
            if count < 5:
                return 1.0
            pnl = self._safe_float(bucket.get("pnl"), 0.0)
            profit_factor = self._safe_float(bucket.get("profit_factor"), 1.0)
            multiplier = self._safe_float(bucket.get("score_multiplier"), 1.0)
            state = str(bucket.get("state") or "").lower()
            if state == "degrade" or pnl < 0 or profit_factor < 0.85:
                if pnl <= -50.0 or profit_factor < 0.55:
                    return min(multiplier, 0.15)
                return min(multiplier, 0.40)
            if state == "promote" and pnl > 0 and profit_factor >= 1.15:
                return max(min(multiplier, 1.45), 1.12)
            return max(min(multiplier, 1.15), 0.85)

        def add(side: str, score: float, note: str, *, expected: float | None = None, loss_probability: float | None = None) -> None:
            if side not in sides:
                return
            score = self._safe_float(score, 0.0)
            sides[side]["score"] += score
            if expected is not None:
                sides[side]["expected_return_pct"] += self._safe_float(expected, 0.0)
            if loss_probability is not None:
                lp = min(max(self._safe_float(loss_probability, 0.5), 0.0), 1.0)
                old = sides[side].get("loss_probability")
                sides[side]["loss_probability"] = lp if old is None else max(float(old), lp)
            if note:
                sides[side]["evidence"].append(str(note)[:120])

        ml_signal = ml_signal_context if isinstance(ml_signal_context, dict) else {}
        predictions = ml_signal.get("predictions") if isinstance(ml_signal.get("predictions"), list) else []
        primary = predictions[0] if predictions and isinstance(predictions[0], dict) else {}
        if primary and bool(ml_signal.get("influence_enabled", True)):
            ml_weight = source_weight("ml_profit_model")
            for side in ("long", "short"):
                expected = self._safe_float(primary.get(f"{side}_expected_return_pct"), 0.0)
                win_rate = self._safe_float(primary.get(f"{side}_win_rate"), 0.5)
                score = (expected * 0.55 + (win_rate - 0.5) * 0.35) * ml_weight
                add(
                    side,
                    score,
                    f"本地ML {side} 预期={expected:.3f}% 胜率={win_rate:.1%} 权重={ml_weight:.2f}",
                    expected=expected,
                    loss_probability=1.0 - win_rate,
                )

        tools = local_ai_tools_context if isinstance(local_ai_tools_context, dict) else {}
        profit = tools.get("profit_prediction") if isinstance(tools.get("profit_prediction"), dict) else {}
        if profit and profit.get("available", True) is not False:
            profit_weight = source_weight("server_profit_model")
            for side in ("long", "short"):
                expected = self._safe_float(
                    profit.get(f"adjusted_{side}_return_pct", profit.get(f"{side}_expected_return_pct")),
                    0.0,
                )
                loss_probability = self._safe_float(profit.get(f"{side}_loss_probability"), 0.5)
                quality = self._safe_float(profit.get("profit_quality_score"), 0.0)
                score = (expected * 0.70 - max(loss_probability - 0.50, 0.0) * 0.42 + quality * 0.12) * profit_weight
                add(
                    side,
                    score,
                    f"服务器盈利模型 {side} 预期={expected:.3f}% 亏损概率={loss_probability:.1%} 权重={profit_weight:.2f}",
                    expected=expected,
                    loss_probability=loss_probability,
                )

        ts = (
            tools.get("time_series_prediction")
            or tools.get("timeseries_prediction")
            or tools.get("sequence_prediction")
        )
        if isinstance(ts, dict) and ts:
            ts_weight = source_weight("timeseries_model")
            ts_side = str(ts.get("best_side") or ts.get("side") or "").lower()
            ts_expected = self._safe_float(
                ts.get("expected_return_pct", ts.get(f"{ts_side}_expected_return_pct")),
                0.0,
            )
            if ts_side in {"long", "short"}:
                add(
                    ts_side,
                    (ts_expected * 0.60 + 0.08) * ts_weight,
                    f"时序模型偏{ts_side} 预期={ts_expected:.3f}% 权重={ts_weight:.2f}",
                    expected=ts_expected,
                )
                other = "short" if ts_side == "long" else "long"
                add(other, -abs(ts_expected) * 0.25 * ts_weight, f"时序模型不支持{other}")

        sentiment = tools.get("sentiment_analysis") if isinstance(tools.get("sentiment_analysis"), dict) else {}
        if sentiment:
            sent_side = str(sentiment.get("best_side") or sentiment.get("side") or "").lower()
            sent_expected = self._safe_float(sentiment.get("expected_return_pct"), 0.0)
            sent_score = self._safe_float(sentiment.get("score", sentiment.get("sentiment_score")), 0.0)
            if sent_side in {"long", "short"}:
                add(sent_side, sent_expected * 0.25 + sent_score * 0.08, f"情绪模型偏{sent_side}")

        returns_1 = self._safe_float(getattr(fv, "returns_1", 0.0), 0.0)
        returns_5 = self._safe_float(getattr(fv, "returns_5", 0.0), 0.0)
        returns_20 = self._safe_float(getattr(fv, "returns_20", 0.0), 0.0)
        price_vs_sma20 = self._safe_float(getattr(fv, "price_vs_sma20", 0.0), 0.0)
        price_vs_sma50 = self._safe_float(getattr(fv, "price_vs_sma50", 0.0), 0.0)
        adx_14 = self._safe_float(getattr(fv, "adx_14", 0.0), 0.0)
        tech_momentum = returns_1 * 100.0 * 0.08 + returns_5 * 100.0 * 0.18 + returns_20 * 100.0 * 0.10
        ma_bias = (price_vs_sma20 + price_vs_sma50) * 0.06
        trend_strength = min(max((adx_14 - 14.0) / 28.0, 0.0), 1.0)
        add("long", tech_momentum + ma_bias + max(tech_momentum, 0.0) * trend_strength * 0.12, "技术结构对做多的方向分")
        add("short", -tech_momentum - ma_bias + max(-tech_momentum, 0.0) * trend_strength * 0.12, "技术结构对做空的方向分")

        regime = market_regime if isinstance(market_regime, dict) else {}
        soft_avoided = set(strategy.get("soft_avoided_directions") or [])
        for side in ("long", "short"):
            if side in soft_avoided:
                add(side, -0.10, f"大盘环境只作为软惩罚：{side} 需要更强单币证据")

        exposure = strategy.get("position_exposure") if isinstance(strategy.get("position_exposure"), dict) else {}
        dominant_side = str(exposure.get("dominant_side") or "neutral")
        net_ratio = abs(self._safe_float(exposure.get("net_ratio"), 0.0))
        if dominant_side in {"long", "short"} and net_ratio > 0:
            side_pnl = self._safe_float(exposure.get(f"{dominant_side}_unrealized_pnl"), 0.0)
            same_side_penalty = net_ratio * 0.28
            if side_pnl < 0:
                same_side_penalty += min(abs(side_pnl) / 25.0, 0.75)
            add(dominant_side, -same_side_penalty, f"当前{dominant_side}敞口集中且浮盈亏={side_pnl:.2f}U，新增同向需要更强收益证据")
            opposite = "short" if dominant_side == "long" else "long"
            opposite_expected = self._safe_float(sides[opposite].get("expected_return_pct"), 0.0)
            if opposite_expected > 0:
                add(
                    opposite,
                    min(net_ratio * 0.025, 0.04),
                    f"当前组合偏{dominant_side}，仅在{opposite}已有正期望时给轻微分散加分",
                )

        long_score = self._safe_float(sides["long"]["score"], 0.0)
        short_score = self._safe_float(sides["short"]["score"], 0.0)
        if abs(long_score - short_score) < 0.08:
            preferred_side = "neutral"
        else:
            preferred_side = "long" if long_score > short_score else "short"
        for side in ("long", "short"):
            sides[side]["score"] = round(self._safe_float(sides[side]["score"], 0.0), 6)
            sides[side]["expected_return_pct"] = round(self._safe_float(sides[side]["expected_return_pct"], 0.0), 6)
            if sides[side].get("loss_probability") is not None:
                sides[side]["loss_probability"] = round(self._safe_float(sides[side]["loss_probability"], 0.5), 6)
            sides[side]["evidence"] = sides[side]["evidence"][:5]
        return {
            "enabled": True,
            "preferred_side": preferred_side,
            "score_gap": round(abs(long_score - short_score), 6),
            "long": sides["long"],
            "short": sides["short"],
            "market_regime_mode": regime.get("mode"),
            "policy": (
                "AI 必须按当前币种独立比较 long/short 的预期收益、亏损概率和技术/时序证据；"
                "组合敞口只影响仓位和风险，不是机械反向开仓理由；不同币种可以有不同方向。"
            ),
        }

    def _ai_entry_candidate_evidence(
        self,
        fv: Any,
        strategy: dict[str, Any] | None,
        ml_signal_context: dict[str, Any] | None,
        local_ai_tools_context: dict[str, Any] | None,
        direction_competition_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build a pre-AI evidence pack for long/short candidate quality.

        This intentionally uses the same opportunity scoring math as execution
        ranking, but only as prompt evidence. AI owns the trade/no-trade
        decision; execution only keeps hard account/exchange safety guards.
        """
        symbol = str(getattr(fv, "symbol", "") or "")
        feature_snapshot = fv.to_dict() if hasattr(fv, "to_dict") else {}
        base_raw = {
            "analysis_type": "market",
            "ml_signal": ml_signal_context or {},
            "local_ai_tools": local_ai_tools_context or {},
            "direction_competition": direction_competition_context or {},
            "pre_ai_candidate_evidence": True,
        }

        def compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
            if not isinstance(profile, dict):
                return {}
            return {
                "count": int(profile.get("count") or 0),
                "pnl": round(self._safe_float(profile.get("pnl"), 0.0), 4),
                "today_pnl": round(self._safe_float(profile.get("today_pnl"), 0.0), 4),
                "wins": int(profile.get("wins") or 0),
                "losses": int(profile.get("losses") or 0),
                "profit_factor": round(self._safe_float(profile.get("profit_factor"), 0.0), 4),
                "largest_loss": round(self._safe_float(profile.get("largest_loss"), 0.0), 4),
                "cooldown": bool(profile.get("cooldown")),
                "cooldown_reason": str(profile.get("cooldown_reason") or "")[:120],
            }

        def build_side(side: str) -> dict[str, Any]:
            action = Action.LONG if side == "long" else Action.SHORT
            raw = dict(base_raw)
            decision = DecisionOutput(
                model_name=ENSEMBLE_TRADER_NAME,
                symbol=symbol,
                action=action,
                confidence=0.62,
                reasoning="pre_ai_candidate_evidence",
                position_size_pct=0.03,
                suggested_leverage=3.0,
                stop_loss_pct=0.015,
                take_profit_pct=0.045,
                raw_response=raw,
                feature_snapshot=feature_snapshot,
            )
            score = self.entry_policy.score_candidate(decision, strategy)
            opportunity = (
                decision.raw_response.get("opportunity_score")
                if isinstance(decision.raw_response, dict)
                and isinstance(decision.raw_response.get("opportunity_score"), dict)
                else {}
            )
            expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
            tail_risk = self._safe_float(opportunity.get("tail_risk_score"), 0.0)
            loss_probability = self._safe_float(opportunity.get("server_profit_loss_probability"), 0.5)
            profit_quality = self._safe_float(opportunity.get("profit_quality_ratio"), 0.0)
            min_score = self._safe_float(opportunity.get("min_score_required"), MIN_ENTRY_OPPORTUNITY_SCORE)
            high_profit_potential = bool(
                expected_net >= 1.20
                and profit_quality >= 1.20
                and loss_probability <= 0.38
                and tail_risk <= ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
                and (
                    opportunity.get("ml_aligned")
                    or opportunity.get("local_profit_aligned")
                    or opportunity.get("timeseries_aligned")
                )
            )
            if high_profit_potential:
                recommendation = "high_profit_candidate_allow_larger_size_and_leverage"
            elif expected_net <= 0 or profit_quality <= 0.12 or tail_risk >= 1.15:
                recommendation = "hold_or_tiny_probe_only"
            elif score >= min_score and expected_net > 0 and tail_risk < 0.95:
                recommendation = "tradable_if_ai_thesis_confirms"
            else:
                recommendation = "needs_stronger_ai_confirmation"
            return {
                "side": side,
                "score": round(score, 6),
                "min_score_reference": round(min_score, 6),
                "expected_net_return_pct": round(expected_net, 6),
                "expected_loss_pct": opportunity.get("expected_loss_pct"),
                "success_probability": opportunity.get("success_probability"),
                "loss_probability": round(loss_probability, 6),
                "profit_quality_ratio": round(profit_quality, 6),
                "tail_risk_score": round(tail_risk, 6),
                "high_profit_potential": high_profit_potential,
                "sizing_hint": (
                    "profit_potential_large: AI may use higher size/leverage if thesis is clear"
                    if high_profit_potential
                    else "normal_or_small: do not enlarge unless AI finds stronger evidence"
                ),
                "reward_risk_ratio": opportunity.get("reward_risk_ratio"),
                "ml_expected_return_pct": opportunity.get("expected_return_pct"),
                "ml_win_rate": opportunity.get("win_rate"),
                "server_profit_expected_return_pct": opportunity.get("server_profit_expected_return_pct"),
                "server_profit_best_side": opportunity.get("server_profit_best_side"),
                "server_profit_conflict": bool(opportunity.get("server_profit_conflict")),
                "timeseries_expected_return_pct": opportunity.get("timeseries_expected_return_pct"),
                "timeseries_aligned": bool(opportunity.get("timeseries_aligned")),
                "direction_side_score": opportunity.get("direction_side_score"),
                "direction_opposite_score": opportunity.get("direction_opposite_score"),
                "historical_reason": opportunity.get("historical_reason"),
                "historical_block": bool(opportunity.get("historical_block")),
                "symbol_profile": compact_profile(opportunity.get("symbol_profile") or {}),
                "symbol_side_profile": compact_profile(opportunity.get("symbol_side_profile") or {}),
                "abnormal_wick_count_72h": opportunity.get("abnormal_wick_count_72h"),
                "abnormal_wick_max_pct": opportunity.get("abnormal_wick_max_pct"),
                "abnormal_wick_recent_hours": opportunity.get("abnormal_wick_recent_hours"),
                "recommendation": recommendation,
            }

        long_evidence = build_side("long")
        short_evidence = build_side("short")
        if self._safe_float(long_evidence.get("score"), 0.0) > self._safe_float(short_evidence.get("score"), 0.0) + 0.08:
            preferred_side = "long"
        elif self._safe_float(short_evidence.get("score"), 0.0) > self._safe_float(long_evidence.get("score"), 0.0) + 0.08:
            preferred_side = "short"
        else:
            preferred_side = "neutral"
        return {
            "enabled": True,
            "symbol": symbol,
            "feature_opportunity_score": round(self._feature_opportunity_score(fv), 4),
            "preferred_side_by_evidence": preferred_side,
            "long": long_evidence,
            "short": short_evidence,
            "policy": (
                "This is prompt evidence, not an execution veto. AI must compare long/short expected net profit, "
                "loss probability, payoff quality, recent realized performance, and tail risk before choosing action, "
                "size, leverage, stop loss, and take profit."
            ),
        }

    def _market_llm_prefilter_skip_reason(
        self,
        fv: Any,
        local_ai_tools_context: dict[str, Any],
        open_positions: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Do not skip AI market analysis for ordinary candidate quality.

        Candidate quality, local profit evidence, and loss probability are now
        passed into the AI prompt as structured evidence. The execution layer
        should only stop serious safety/exchange problems, not hide a symbol
        from AI because a prefilter dislikes the setup.
        """
        data_quality_reason = self._entry_market_data_quality_reason(fv, stage_label="AI分析前")
        if data_quality_reason:
            return data_quality_reason
        return None

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

    def _market_value(self, source: Any, key: str, default: Any = None) -> Any:
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    def _entry_market_data_quality_reason(
        self,
        source: Any,
        *,
        stage_label: str = "下单前",
    ) -> str | None:
        """Block entry analysis/execution when market data is clearly unusable."""
        try:
            current_price = self._safe_float(self._market_value(source, "current_price"), 0.0)
            close_price = self._safe_float(self._market_value(source, "close"), 0.0)
            bid = self._safe_float(self._market_value(source, "bid"), 0.0)
            ask = self._safe_float(self._market_value(source, "ask"), 0.0)
            returns_1 = self._safe_float(self._market_value(source, "returns_1"), 0.0)
            returns_5 = self._safe_float(self._market_value(source, "returns_5"), 0.0)
            returns_20 = self._safe_float(self._market_value(source, "returns_20"), 0.0)
            volatility_20 = self._safe_float(self._market_value(source, "volatility_20"), 0.0)
            change_24h_pct = self._safe_float(self._market_value(source, "change_24h_pct"), 0.0)
            high_24h = self._safe_float(self._market_value(source, "high_24h"), 0.0)
            low_24h = self._safe_float(self._market_value(source, "low_24h"), 0.0)
            bid_depth = self._safe_float(self._market_value(source, "orderbook_bid_depth"), 0.0)
            ask_depth = self._safe_float(self._market_value(source, "orderbook_ask_depth"), 0.0)
            imbalance = self._safe_float(self._market_value(source, "orderbook_imbalance"), 0.0)
            abnormal_wick_count = int(self._safe_float(self._market_value(source, "abnormal_wick_count_72h"), 0.0))
            abnormal_wick_max = self._safe_float(self._market_value(source, "abnormal_wick_max_pct"), 0.0)
        except Exception:
            return f"{stage_label}行情数据异常，无法确认真实价格和盘口，本次不执行新开仓。"

        price = current_price or close_price or bid or ask
        if price <= 0:
            return f"{stage_label}没有有效价格，本次不执行新开仓。"

        reference_prices = [p for p in (current_price, close_price, bid, ask) if p > 0]
        if len(reference_prices) >= 2:
            min_ref = min(reference_prices)
            max_ref = max(reference_prices)
            split = (max_ref - min_ref) / max(min_ref, 1e-12)
            if split >= ENTRY_PRICE_FIELD_SPLIT_BLOCK_PCT:
                return (
                    f"{stage_label}行情价格源分裂：current={current_price:g}、close={close_price:g}、"
                    f"bid={bid:g}、ask={ask:g}，最大差异约 {split * 100:.2f}%。"
                    "这会导致止盈止损价格和交易所主订单价格不匹配，本次不执行新开仓，等待行情重新同步。"
                )

        if high_24h > 0 and low_24h > 0 and high_24h >= low_24h:
            range_floor = low_24h * (1.0 - ENTRY_PRICE_24H_RANGE_TOLERANCE_PCT)
            range_ceiling = high_24h * (1.0 + ENTRY_PRICE_24H_RANGE_TOLERANCE_PCT)
            if price < range_floor or price > range_ceiling:
                return (
                    f"{stage_label}行情价格与24小时区间矛盾：当前价 {price:g}，"
                    f"24小时低点 {low_24h:g}、高点 {high_24h:g}。"
                    "当前价已经明显落在交易所24小时区间之外，说明行情快照可能串币、延迟或来源异常；"
                    "为避免止盈止损价格和 OKX 主订单价格不匹配，本次不执行新开仓。"
                )

        if bid > 0 and ask > 0 and ask >= bid:
            spread = (ask - bid) / max((ask + bid) / 2.0, 1e-12)
            if spread > max(float(settings.max_slippage_pct or 0.005) * 2.0, 0.012):
                return (
                    f"{stage_label}盘口价差过大：买一 {bid:g} / 卖一 {ask:g}，"
                    f"价差约 {spread * 100:.2f}%，容易产生明显滑点，本次不执行新开仓。"
                )

        one_sided_depth = (bid_depth <= 0 or ask_depth <= 0) and abs(imbalance) >= 0.98
        if one_sided_depth:
            return (
                f"{stage_label}盘口深度异常：买盘深度 {bid_depth:.4g}、卖盘深度 {ask_depth:.4g}，"
                f"盘口失衡 {imbalance:.2f}。该币种当前流动性/盘口数据不可靠，"
                "容易出现下单前大幅偏移或滑点，本次不执行新开仓。"
            )

        all_short_returns_zero = (
            abs(returns_1) < 1e-12
            and abs(returns_5) < 1e-12
            and abs(returns_20) < 1e-12
            and abs(volatility_20) < 1e-12
        )
        if all_short_returns_zero and abs(change_24h_pct) >= ENTRY_DATA_STALE_ZERO_RETURNS_MIN_24H_CHANGE * 100:
            return (
                f"{stage_label}短周期行情特征疑似缺失：1/5/20周期收益率和波动率都为 0，"
                f"但24小时涨跌幅为 {change_24h_pct:.2f}%。本次不把不完整行情送入开仓执行。"
            )

        if abnormal_wick_count > 0 and abnormal_wick_max >= ABNORMAL_WICK_ENTRY_BLOCK_MAX_PCT:
            return (
                f"{stage_label}检测到近72小时异常插针，最大振幅约 {abnormal_wick_max:.2f}%，"
                "该币种容易出现非连续成交价格，本次不执行新开仓。"
            )

        return None

    def _set_loop_stage(self, stage: str, error: str | None = None) -> None:
        self._current_stage = stage
        if error is not None:
            self._last_round_error = str(error)[:300]

    async def _reconcile_exchange_positions_with_timeout(
        self,
        context: str,
        timeout: float = 25.0,
    ) -> list[dict]:
        try:
            async def _locked_reconcile() -> list[dict]:
                async with self._exchange_reconcile_lock:
                    return await self.reconcile_exchange_positions()

            return await asyncio.wait_for(_locked_reconcile(), timeout=timeout)
        except asyncio.TimeoutError:
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
        except asyncio.TimeoutError:
            logger.warning("fresh feature vector refresh timed out; using queued snapshot", symbol=symbol)
        except Exception as e:
            logger.warning("fresh feature vector refresh failed; using queued snapshot", symbol=symbol, error=str(e))
        return fallback

    async def _fresh_feature_vector_for_price_recheck(self, symbol: str) -> Any | None:
        try:
            fresh = await asyncio.wait_for(
                self.data_service.get_feature_vector(symbol),
                timeout=ENTRY_PRICE_RECHECK_TIMEOUT_SECONDS,
            )
            if self._is_valid_feature_vector(fresh):
                return fresh
        except asyncio.TimeoutError:
            logger.warning("pre-order feature recheck timed out", symbol=symbol)
        except Exception as e:
            logger.warning("pre-order feature recheck failed", symbol=symbol, error=str(e))
        return None

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
        if not is_price_action_risk and "异常暴跌" not in reason_text and "行情风险" not in reason_text:
            return False

        fresh = await self._fresh_feature_vector_for_price_recheck(decision.symbol)
        fresh_snapshot = fresh.to_dict() if fresh is not None and hasattr(fresh, "to_dict") else {}
        fresh_quality_reason = (
            self._entry_market_data_quality_reason(fresh_snapshot, stage_label="黑天鹅复核刷新行情")
            if fresh_snapshot
            else "黑天鹅复核刷新行情失败，无法确认最新短周期特征。"
        )
        fresh_returns_1 = self._safe_float(fresh_snapshot.get("returns_1"), 0.0) if fresh_snapshot else 0.0
        fresh_returns_5 = self._safe_float(fresh_snapshot.get("returns_5"), 0.0) if fresh_snapshot else 0.0
        fresh_returns_20 = self._safe_float(fresh_snapshot.get("returns_20"), 0.0) if fresh_snapshot else 0.0
        fresh_change_24h = self._safe_float(fresh_snapshot.get("change_24h_pct"), 0.0) if fresh_snapshot else 0.0
        bid = self._safe_float(fresh_snapshot.get("bid"), 0.0) if fresh_snapshot else 0.0
        ask = self._safe_float(fresh_snapshot.get("ask"), 0.0) if fresh_snapshot else 0.0
        bid_depth = self._safe_float(fresh_snapshot.get("orderbook_bid_depth"), 0.0) if fresh_snapshot else 0.0
        ask_depth = self._safe_float(fresh_snapshot.get("orderbook_ask_depth"), 0.0) if fresh_snapshot else 0.0
        spread = (
            (ask - bid) / max((ask + bid) / 2.0, 1e-12)
            if bid > 0 and ask > 0 and ask >= bid
            else 0.0
        )
        has_normal_book = bool(
            bid_depth > 0
            and ask_depth > 0
            and spread <= ENTRY_BLACK_SWAN_REBOUND_MAX_SPREAD
        )
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
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
            raw.setdefault("execution_advisory_warnings", []).append({
                "reason": (
                    "风险引擎检测到疑似 1 分钟极端暴跌，但最新行情复核显示该信号可能是脏数据"
                    "或已经完成反弹恢复，已降级为警告。"
                ),
                "fresh_returns_1": round(fresh_returns_1, 6),
                "fresh_returns_5": round(fresh_returns_5, 6),
                "fresh_returns_20": round(fresh_returns_20, 6),
            })
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
        """Rank symbols after K-line indicators are available, before spending AI tokens."""
        try:
            volume_24h = float(getattr(fv, "volume_24h", 0) or 0)
            volume_ratio = float(getattr(fv, "volume_ratio", 0) or 0)
            adx_14 = float(getattr(fv, "adx_14", 0) or 0)
            returns_1 = abs(float(getattr(fv, "returns_1", 0) or 0))
            returns_5 = abs(float(getattr(fv, "returns_5", 0) or 0))
            returns_20 = abs(float(getattr(fv, "returns_20", 0) or 0))
            volatility_20 = float(getattr(fv, "volatility_20", 0) or 0)
            change_24h = abs(float(getattr(fv, "change_24h_pct", 0) or 0))
            bb_pct = float(getattr(fv, "bb_pct", 0.5) or 0.5)
            price_vs_sma20 = abs(float(getattr(fv, "price_vs_sma20", 0) or 0))
            price_vs_sma50 = abs(float(getattr(fv, "price_vs_sma50", 0) or 0))
            current_price = float(getattr(fv, "current_price", 0) or getattr(fv, "close", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

        notional_24h = max(volume_24h * max(current_price, 0.0), 0.0)
        liquidity = math.log10(notional_24h + 1.0) * 10.0
        participation = min(max(volume_ratio, 0.0), 5.0) * 10.0
        trend_quality = min(max(adx_14, 0.0), 50.0) * 0.8
        momentum = min((returns_1 * 1200) + (returns_5 * 700) + (returns_20 * 350), 45.0)
        day_move = min(change_24h, 12.0) * 1.6
        volatility_bonus = min(max(volatility_20, 0.0) * 900, 30.0)
        trend_distance = min((price_vs_sma20 + price_vs_sma50) * 600, 25.0)
        band_bonus = 8.0 if bb_pct <= 0.18 or bb_pct >= 0.82 else 0.0
        low_activity_penalty = 80.0 if volume_ratio < settings.min_entry_volume_ratio else 0.0
        extreme_vol_penalty = 45.0 if volatility_20 > 0.12 and change_24h > 8 else 18.0 if volatility_20 > 0.08 else 0.0

        return (
            liquidity
            + participation
            + trend_quality
            + momentum
            + day_move
            + volatility_bonus
            + trend_distance
            + band_bonus
            - low_activity_penalty
            - extreme_vol_penalty
        )

    def _remember_market_hold_symbol(
        self,
        symbol: str,
        fv: Any | None = None,
        reason: str | None = None,
    ) -> None:
        normalized = self._normalize_position_symbol(symbol)
        if not normalized:
            return
        now = datetime.now(timezone.utc)
        self._recent_market_hold_symbols[normalized] = now
        cutoff = now - timedelta(minutes=15)
        self._recent_market_hold_symbols = {
            key: seen_at
            for key, seen_at in self._recent_market_hold_symbols.items()
            if seen_at >= cutoff
        }
        previous = self._market_no_opportunity_symbols.get(normalized) or {}
        first_seen = previous.get("first_seen_at")
        if not isinstance(first_seen, datetime) or first_seen < now - timedelta(minutes=MARKET_NO_OPPORTUNITY_WINDOW_MINUTES):
            first_seen = now
            hold_count = 0
        else:
            hold_count = int(previous.get("hold_count") or 0)

        score = self._feature_opportunity_score(fv) if fv is not None else None
        self._market_no_opportunity_symbols[normalized] = {
            "first_seen_at": first_seen,
            "last_hold_at": now,
            "hold_count": hold_count + 1,
            "last_feature_score": score,
            "reason": str(reason or "")[:220],
        }
        self._prune_market_no_opportunity_symbols()

    def _prune_market_no_opportunity_symbols(self) -> None:
        if not self._market_no_opportunity_symbols:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=MARKET_NO_OPPORTUNITY_WINDOW_MINUTES * 2)
        self._market_no_opportunity_symbols = {
            symbol: state
            for symbol, state in self._market_no_opportunity_symbols.items()
            if isinstance(state.get("last_hold_at"), datetime)
            and state["last_hold_at"] >= cutoff
        }

    def _clear_market_no_opportunity_symbol(self, symbol: str) -> None:
        normalized = self._normalize_position_symbol(symbol)
        if normalized:
            self._market_no_opportunity_symbols.pop(normalized, None)
            self._recent_market_hold_symbols.pop(normalized, None)

    def _remember_market_analyzed_symbol(self, symbol: str) -> None:
        normalized = self._normalize_position_symbol(symbol)
        if not normalized:
            return
        now = datetime.now(timezone.utc)
        self._recent_market_analyzed_symbols[normalized] = now
        cutoff = now - timedelta(minutes=12)
        self._recent_market_analyzed_symbols = {
            key: seen_at
            for key, seen_at in self._recent_market_analyzed_symbols.items()
            if seen_at >= cutoff
        }

    def _recent_market_hold_penalty(self, symbol: str) -> float:
        normalized = self._normalize_position_symbol(symbol)
        seen_at = self._recent_market_hold_symbols.get(normalized)
        if not seen_at:
            return 0.0
        age_minutes = (datetime.now(timezone.utc) - seen_at).total_seconds() / 60.0
        if age_minutes >= 10.0:
            return 0.0
        return max(0.0, 70.0 * (1.0 - age_minutes / 10.0))

    def _recent_market_analysis_penalty(self, symbol: str) -> float:
        normalized = self._normalize_position_symbol(symbol)
        seen_at = self._recent_market_analyzed_symbols.get(normalized)
        if not seen_at:
            return 0.0
        age_minutes = (datetime.now(timezone.utc) - seen_at).total_seconds() / 60.0
        if age_minutes >= 12.0:
            return 0.0
        return max(0.0, 35.0 * (1.0 - age_minutes / 12.0))

    def _no_opportunity_rotation_penalty(self, symbol: str, fv: Any | None = None) -> float:
        normalized = self._normalize_position_symbol(symbol)
        if not normalized:
            return 0.0
        state = self._market_no_opportunity_symbols.get(normalized)
        if not state:
            return 0.0
        last_hold_at = state.get("last_hold_at")
        if not isinstance(last_hold_at, datetime):
            self._market_no_opportunity_symbols.pop(normalized, None)
            return 0.0

        now = datetime.now(timezone.utc)
        age_minutes = (now - last_hold_at).total_seconds() / 60.0
        if age_minutes >= MARKET_NO_OPPORTUNITY_WINDOW_MINUTES:
            self._market_no_opportunity_symbols.pop(normalized, None)
            return 0.0

        hold_count = max(0, int(state.get("hold_count") or 0))
        if hold_count < MARKET_NO_OPPORTUNITY_STREAK_THRESHOLD:
            return 0.0

        current_score = self._feature_opportunity_score(fv) if fv is not None else 0.0
        previous_score = state.get("last_feature_score")
        try:
            previous_score = float(previous_score)
        except (TypeError, ValueError):
            previous_score = 0.0

        try:
            volume_ratio = float(getattr(fv, "volume_ratio", 0) or 0) if fv is not None else 0.0
            returns_5 = abs(float(getattr(fv, "returns_5", 0) or 0)) if fv is not None else 0.0
            returns_20 = abs(float(getattr(fv, "returns_20", 0) or 0)) if fv is not None else 0.0
            adx_14 = float(getattr(fv, "adx_14", 0) or 0) if fv is not None else 0.0
        except (TypeError, ValueError):
            volume_ratio = returns_5 = returns_20 = adx_14 = 0.0

        opportunity_improved = (
            current_score >= previous_score + 35.0
            or (volume_ratio >= max(settings.min_entry_volume_ratio, 0.8) and (returns_5 >= 0.004 or returns_20 >= 0.010))
            or (adx_14 >= max(float(settings.min_entry_adx or 0), 24.0) and (returns_5 >= 0.003 or returns_20 >= 0.008))
        )
        if opportunity_improved:
            self._clear_market_no_opportunity_symbol(normalized)
            return 0.0

        if age_minutes >= MARKET_NO_OPPORTUNITY_RECHECK_MINUTES:
            decay = max(
                0.0,
                1.0 - (age_minutes - MARKET_NO_OPPORTUNITY_RECHECK_MINUTES)
                / max(MARKET_NO_OPPORTUNITY_WINDOW_MINUTES - MARKET_NO_OPPORTUNITY_RECHECK_MINUTES, 1.0),
            )
        else:
            decay = 1.0
        streak_multiplier = min(1.0, (hold_count - MARKET_NO_OPPORTUNITY_STREAK_THRESHOLD + 1) / 4.0)
        return MARKET_NO_OPPORTUNITY_MAX_PENALTY * streak_multiplier * decay

    def _market_regime_context(self, feature_vectors: dict[str, Any]) -> dict[str, Any]:
        """Predict the current market style before asking for per-symbol entries."""
        rows = [fv for fv in (feature_vectors or {}).values() if self._is_valid_feature_vector(fv)]
        if not rows:
            return {"mode": "unknown", "confidence": 0.0, "avoid_long": False, "avoid_short": False}

        def val(fv: Any, name: str, default: float = 0.0) -> float:
            try:
                return float(getattr(fv, name, default) or default)
            except (TypeError, ValueError):
                return default

        total = len(rows)
        up_5 = sum(1 for fv in rows if val(fv, "returns_5") > 0.002)
        down_5 = sum(1 for fv in rows if val(fv, "returns_5") < -0.002)
        up_20 = sum(1 for fv in rows if val(fv, "returns_20") > 0.006)
        down_20 = sum(1 for fv in rows if val(fv, "returns_20") < -0.006)
        above_sma = sum(1 for fv in rows if val(fv, "price_vs_sma20") > 0 and val(fv, "price_vs_sma50") > 0)
        below_sma = sum(1 for fv in rows if val(fv, "price_vs_sma20") < 0 and val(fv, "price_vs_sma50") < 0)
        high_adx = sum(1 for fv in rows if val(fv, "adx_14") >= 25)
        avg_ret_5 = sum(val(fv, "returns_5") for fv in rows) / total
        avg_ret_20 = sum(val(fv, "returns_20") for fv in rows) / total

        majors = [
            fv for fv in rows
            if str(getattr(fv, "symbol", "")).upper() in {"BTC/USDT", "ETH/USDT"}
        ]
        major_score = 0.0
        for fv in majors:
            major_score += val(fv, "returns_5") * 0.55 + val(fv, "returns_20") * 0.45
        if majors:
            major_score /= len(majors)
        btc_eth_filter = self._btc_eth_alt_long_filter(majors)

        up_breadth = max(up_5 / total, up_20 / total)
        down_breadth = max(down_5 / total, down_20 / total)
        trend_breadth = max(above_sma / total, below_sma / total)
        confidence = min(max(abs(up_breadth - down_breadth) + abs(avg_ret_20) * 12 + trend_breadth * 0.25, 0.0), 0.95)

        mode = "mixed"
        avoid_long = False
        avoid_short = False
        reason = "Market direction is mixed; trade only symbol-level high-quality signals."
        if up_breadth >= 0.55 and avg_ret_5 > 0 and major_score >= -0.001:
            mode = "rebound_squeeze_up"
            avoid_short = True
            reason = "Broad short-term rebound; shorts need stronger symbol-level evidence."
        elif down_breadth >= 0.55 and avg_ret_5 < 0 and major_score <= 0.001:
            mode = "selloff_squeeze_down"
            avoid_long = True
            reason = "Broad short-term selloff; longs need stronger symbol-level evidence."
        elif above_sma / total >= 0.55 and avg_ret_20 > 0.003:
            mode = "uptrend_continuation"
            avoid_short = True
            reason = "Broad uptrend; counter-trend shorts need stronger evidence."
        elif below_sma / total >= 0.55 and avg_ret_20 < -0.003:
            mode = "downtrend_continuation"
            avoid_long = True
            reason = "Broad downtrend; counter-trend longs need stronger evidence."

        return {
            "mode": mode,
            "confidence": round(confidence, 4),
            "avoid_long": avoid_long,
            "avoid_short": avoid_short,
            "reason": reason,
            "sample_count": total,
            "up_5_ratio": round(up_5 / total, 4),
            "down_5_ratio": round(down_5 / total, 4),
            "up_20_ratio": round(up_20 / total, 4),
            "down_20_ratio": round(down_20 / total, 4),
            "above_sma_ratio": round(above_sma / total, 4),
            "below_sma_ratio": round(below_sma / total, 4),
            "high_adx_ratio": round(high_adx / total, 4),
            "avg_returns_5": round(avg_ret_5, 6),
            "avg_returns_20": round(avg_ret_20, 6),
            "major_score": round(major_score, 6),
            "btc_eth_filter": btc_eth_filter,
        }

    def _btc_eth_alt_long_filter(self, majors: list[Any]) -> dict[str, Any]:
        if not majors:
            return {
                "allow_alt_long": True,
                "reason": "BTC/ETH context unavailable; do not add an extra alt-long block.",
            }

        def val(fv: Any, name: str, default: float = 0.0) -> float:
            try:
                return float(getattr(fv, name, default) or default)
            except (TypeError, ValueError):
                return default

        avg_ret_5 = sum(val(fv, "returns_5") for fv in majors) / len(majors)
        avg_ret_20 = sum(val(fv, "returns_20") for fv in majors) / len(majors)
        avg_adx = sum(val(fv, "adx_14") for fv in majors) / len(majors)
        allow = not (
            avg_ret_5 <= ALT_LONG_BTC_ETH_5M_FLOOR
            and avg_ret_20 <= ALT_LONG_BTC_ETH_20M_FLOOR
            and avg_adx >= ALT_LONG_BTC_ETH_ADX_FLOOR
        )
        reason = f"BTC/ETH avg returns: 5={avg_ret_5:.4f}, 20={avg_ret_20:.4f}, ADX={avg_adx:.1f}."
        if not allow:
            reason = f"{reason} Broad market is falling and trend has not recovered."
        return {
            "allow_alt_long": allow,
            "avg_returns_5": round(avg_ret_5, 6),
            "avg_returns_20": round(avg_ret_20, 6),
            "avg_adx_14": round(avg_adx, 4),
            "reason": reason,
        }

    def _entry_market_regime_block_reason(self, decision: DecisionOutput, market_regime: dict[str, Any]) -> str | None:
        if not decision.is_entry or not isinstance(market_regime, dict):
            return None
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        symbol = self._normalize_position_symbol(decision.symbol).upper()
        if decision.action == Action.LONG and symbol not in ALT_LONG_ALLOWED_SYMBOLS:
            btc_eth = market_regime.get("btc_eth_filter") if isinstance(market_regime.get("btc_eth_filter"), dict) else {}
            if btc_eth and not bool(btc_eth.get("allow_alt_long", True)):
                raw["alt_long_style_filter"] = {
                    "blocked": False,
                    "soft_warning": True,
                    "reason": (
                        "BTC/ETH is weak, but this filter is advisory only; "
                        "single-symbol long entries still go through AI, ML, time-series, "
                        "price guard, and account risk checks."
                    ),
                    "btc_eth_filter": btc_eth,
                }
                decision.raw_response = raw
                return None
            raw["alt_long_style_filter"] = {
                "blocked": False,
                "soft_warning": True,
                "reason": "Alt-long style filter is advisory; hard checks happen later.",
                "btc_eth_filter": btc_eth,
            }
            decision.raw_response = raw
            return None
        return None

    async def _strategy_mode_context(
        self,
        mode: str,
        market_regime: dict[str, Any],
        open_positions: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Choose the trading posture automatically from PnL, regime, and side performance."""
        selected_mode = "live" if mode == "live" else "paper"
        target_usdt = self._configured_daily_target_usdt()
        daily_state = await self._daily_profit_control_state(selected_mode, target_usdt)
        side_perf = await self._today_side_performance(selected_mode)
        symbol_side_perf = await self._recent_symbol_side_performance(selected_mode)
        model_contribution_perf = await self._recent_model_contribution_performance(selected_mode)
        position_exposure = self._position_exposure_context(open_positions or [])
        position_group_count = self._open_position_group_count(open_positions)
        roster_gap = max(PORTFOLIO_MIN_POSITION_GROUPS_TARGET - position_group_count, 0)
        roster_fill_active = roster_gap > 0

        today_total = float(daily_state.get("today_total_pnl") or 0.0)
        high_water = float(daily_state.get("today_high_water_pnl") or today_total)
        loss_pause = self._daily_cooldown_trigger_loss_usdt(selected_mode, target_usdt) if target_usdt > 0 else 0.0
        account_cfg = settings.get_execution_account_config(selected_mode)
        configured_max_loss = float(account_cfg.get("max_loss_usdt") or 0.0)
        drawdown_line = max(
            DRAWNDOWN_LIGHT_RISK_USDT,
            min(configured_max_loss * 0.05, 150.0) if configured_max_loss > 0 else DRAWDOWN_REDUCED_RISK_USDT,
        )
        defensive_line = max(
            DRAWDOWN_HARD_PAUSE_USDT,
            min(configured_max_loss * 0.12, 300.0) if configured_max_loss > 0 else DRAWDOWN_DEFENSIVE_RISK_USDT,
        )
        risk_mode = "normal"
        max_entry_stop_loss_usdt = ENTRY_MAX_STOP_LOSS_NORMAL_USDT
        min_opportunity_score = MIN_ENTRY_OPPORTUNITY_SCORE
        if today_total <= -defensive_line:
            risk_mode = "defensive_recovery"
            max_entry_stop_loss_usdt = ENTRY_MAX_STOP_LOSS_DEFENSIVE_USDT
            min_opportunity_score = 1.80
        elif today_total <= -drawdown_line:
            risk_mode = "drawdown_recovery"
            max_entry_stop_loss_usdt = ENTRY_MAX_STOP_LOSS_DRAWDOWN_USDT
            min_opportunity_score = 1.35
        account_equity = await self._allocated_order_balance(selected_mode)
        if account_equity > 0:
            dynamic_stop_cap = min(
                max(account_equity * ENTRY_MAX_STOP_LOSS_PCT_OF_EQUITY, 6.0),
                ENTRY_MAX_STOP_LOSS_CAP_USDT,
            )
            max_entry_stop_loss_usdt = min(max_entry_stop_loss_usdt, dynamic_stop_cap)
        regime_mode = str((market_regime or {}).get("mode") or "unknown")
        regime_conf = float((market_regime or {}).get("confidence") or 0.0)
        avoid_long = bool((market_regime or {}).get("avoid_long"))
        avoid_short = bool((market_regime or {}).get("avoid_short"))

        long_pnl = float(side_perf.get("long", {}).get("pnl") or 0.0)
        short_pnl = float(side_perf.get("short", {}).get("pnl") or 0.0)
        if short_pnl < 0 and abs(short_pnl) > max(abs(long_pnl), 1e-9) * 1.5:
            avoid_short = True
        if long_pnl < 0 and abs(long_pnl) > max(abs(short_pnl), 1e-9) * 1.5:
            avoid_long = True

        allow_long = True
        allow_short = True
        if today_total <= -DRAWNDOWN_HARD_PAUSE_USDT:
            risk_mode = "hard_recovery"
            min_opportunity_score = max(min_opportunity_score, 2.10)
            max_entry_stop_loss_usdt = min(max_entry_stop_loss_usdt, 4.5)
        elif today_total <= -DRAWNDOWN_LIGHT_RISK_USDT:
            min_opportunity_score = max(min_opportunity_score, 1.45)
            max_entry_stop_loss_usdt = min(max_entry_stop_loss_usdt, 7.0)

        if roster_fill_active and today_total > -DRAWNDOWN_LIGHT_RISK_USDT:
            min_opportunity_score = min(min_opportunity_score, 0.65)

        soft_biases: list[str] = []
        if avoid_long:
            soft_biases.append("long")
        if avoid_short:
            soft_biases.append("short")
        direction_filter_reason = (
            "Global regime is advisory only; symbol-level signals decide direction."
            if soft_biases
            else "No strong global directional bias; use independent symbol signals."
        )

        strategy = "normal_capture"
        posture = "balanced"
        reason = f"Normal capture of high-quality symbol opportunities. {direction_filter_reason}"
        if loss_pause > 0 and today_total <= -loss_pause:
            strategy = "loss_recovery_selective"
            posture = "selective_recovery"
            reason = (
                "今日亏损触及冷静线，停止普通试错；"
                "但允许单币种独立满足强正期望、技术确认和小仓风控的恢复机会。"
            )
        elif today_total < 0:
            if regime_mode != "mixed" and regime_conf >= 0.35 and not (avoid_long and avoid_short):
                strategy = "recovery_attack"
                posture = "profit_first_expansion"
                reason = (
                    "Daily PnL is negative but hard pause is not active. "
                    "Use profit-first recovery: allow more independently confirmed entries, "
                    f"without forcing one global direction. {direction_filter_reason}"
                )
            else:
                strategy = "recovery_selective"
                posture = "selective_recovery"
                reason = (
                    "Daily PnL is negative and market direction is unclear. "
                    "Keep entries selective until per-symbol signal quality improves."
                )
        elif target_usdt > 0 and high_water >= target_usdt:
            strategy = "profit_protect_expand"
            posture = "protect_then_expand"
            reason = f"Daily profit reached the target line; keep looking for strong opportunities while protecting drawdown. {direction_filter_reason}"
        elif regime_mode == "mixed" or regime_conf < 0.35:
            strategy = "chop_wait"
            posture = "patient"
            reason = "Market direction is choppy; reduce low-quality entries and wait for clearer signals."
        if today_total <= -DRAWNDOWN_HARD_PAUSE_USDT:
            strategy = "hard_recovery"
            posture = "tight_selective_reentry"
            reason = (
                "Daily loss is deep; use tight selective recovery. "
                "Only higher-quality, lower-tail-risk new entries are allowed."
            )
        elif today_total <= -DRAWNDOWN_LIGHT_RISK_USDT:
            strategy = "drawdown_clamp"
            posture = "tight_selective"
            reason = (
                "Daily drawdown is active; allow only higher-quality opportunities "
                "and explicitly reduce single-trade tail risk."
            )
        elif roster_fill_active:
            strategy = "portfolio_roster_build"
            posture = "diversified_positive_expectancy"
            reason = (
                f"当前只有 {position_group_count} 个聚合持仓，低于目标 "
                f"{PORTFOLIO_MIN_POSITION_GROUPS_TARGET} 个；优先补充独立正期望机会，"
                "但仍保留负收益、异常价格、保证金和硬风险拦截。"
            )

        return {
            "strategy": strategy,
            "posture": posture,
            "reason": reason,
            "preferred_direction": "neutral",
            "allow_long": allow_long,
            "allow_short": allow_short,
            "blocked_directions": [],
            "soft_avoided_directions": soft_biases,
            "direction_filter_policy": "soft_bias_no_hard_direction_ban",
            "long_short_policy": "evaluate_both_sides_per_symbol",
            "today_total_pnl": round(today_total, 4),
            "today_high_water_pnl": round(high_water, 4),
            "loss_pause_usdt": round(loss_pause, 4),
            "market_regime": market_regime or {},
            "side_performance": side_perf,
            "symbol_side_performance": symbol_side_perf,
            "model_contribution_performance": model_contribution_perf,
            "position_exposure": position_exposure,
            "portfolio_roster": {
                "target_position_groups": PORTFOLIO_MIN_POSITION_GROUPS_TARGET,
                "current_position_groups": position_group_count,
                "gap": roster_gap,
                "underfilled": roster_fill_active,
                "market_symbol_min": PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN,
                "policy": "低于目标持仓数时提高正期望独立机会的扫描和小仓执行倾向；达到目标后恢复常规门槛。",
            },
            "risk_mode": risk_mode,
            "drawdown_line_usdt": round(drawdown_line, 4),
            "defensive_line_usdt": round(defensive_line, 4),
            "min_opportunity_score": round(min_opportunity_score, 4),
            "dynamic_opportunity_score_enabled": True,
            "max_entry_stop_loss_usdt": round(max_entry_stop_loss_usdt, 4),
            "goal": "maximize_realized_net_profit",
            "execution_policy": (
                "Auto-select strategy; global regime is advisory only; "
                "rank entries by expected net return, tail risk, fees, and capital efficiency; "
                "do not optimize for win rate."
            ),
        }

    def _position_exposure_context(
        self,
        open_positions: list[dict] | None,
        staged_entry_counts: dict[str, dict] | None = None,
    ) -> dict[str, Any]:
        """Summarize current long/short exposure so entries do not stack one side."""
        long_notional = 0.0
        short_notional = 0.0
        long_unrealized_pnl = 0.0
        short_unrealized_pnl = 0.0
        long_count = 0
        short_count = 0

        for pos in open_positions or []:
            if pos.get("is_open", True) is False:
                continue
            side = str(pos.get("side") or "").lower()
            if side not in {"long", "short"}:
                continue
            quantity = abs(self._safe_float(
                pos.get("quantity") or pos.get("contracts") or pos.get("sz"),
                0.0,
            ))
            direct_notional = abs(self._safe_float(
                pos.get("notional")
                or pos.get("notional_usd")
                or pos.get("notionalUsd")
                or (pos.get("info") or {}).get("notionalUsd")
                or (pos.get("info") or {}).get("notional"),
                0.0,
            ))
            contract_size = self._safe_float(
                pos.get("contract_size")
                or pos.get("contractSize")
                or (pos.get("info") or {}).get("ctVal"),
                1.0,
            )
            price = self._safe_float(
                pos.get("current_price")
                or pos.get("markPrice")
                or pos.get("lastPrice")
                or pos.get("entry_price")
                or pos.get("entryPrice")
                or pos.get("avgPx"),
                0.0,
            )
            notional = (
                direct_notional
                if direct_notional > 0
                else quantity * max(price, 0.0) * (contract_size if contract_size > 0 else 1.0)
            )
            unrealized_pnl = self._safe_float(
                pos.get("unrealized_pnl")
                or pos.get("unrealizedPnl")
                or pos.get("upl")
                or (pos.get("info") or {}).get("upl")
                or (pos.get("info") or {}).get("unrealizedPnl"),
                0.0,
            )
            if side == "long":
                long_count += 1
                long_notional += max(notional, 0.0)
                long_unrealized_pnl += unrealized_pnl
            else:
                short_count += 1
                short_notional += max(notional, 0.0)
                short_unrealized_pnl += unrealized_pnl

        side_totals = (staged_entry_counts or {}).get("side_totals") or {}
        staged_long_count = int(side_totals.get("long", 0) or 0)
        staged_short_count = int(side_totals.get("short", 0) or 0)

        total_long_count = long_count + staged_long_count
        total_short_count = short_count + staged_short_count
        gross_notional = long_notional + short_notional
        net_notional = long_notional - short_notional
        net_ratio = net_notional / gross_notional if gross_notional > 0 else 0.0
        total_count = total_long_count + total_short_count
        long_count_share = total_long_count / total_count if total_count > 0 else 0.0
        short_count_share = total_short_count / total_count if total_count > 0 else 0.0

        dominant_side = "neutral"
        if gross_notional > 0 and abs(net_ratio) >= 0.65:
            dominant_side = "long" if net_ratio > 0 else "short"
        elif total_count >= 3 and max(long_count_share, short_count_share) >= 0.75:
            dominant_side = "long" if long_count_share > short_count_share else "short"

        return {
            "long_notional": round(long_notional, 4),
            "short_notional": round(short_notional, 4),
            "long_unrealized_pnl": round(long_unrealized_pnl, 4),
            "short_unrealized_pnl": round(short_unrealized_pnl, 4),
            "total_unrealized_pnl": round(long_unrealized_pnl + short_unrealized_pnl, 4),
            "gross_notional": round(gross_notional, 4),
            "net_notional": round(net_notional, 4),
            "net_ratio": round(net_ratio, 4),
            "long_count": total_long_count,
            "short_count": total_short_count,
            "staged_long_count": staged_long_count,
            "staged_short_count": staged_short_count,
            "long_count_share": round(long_count_share, 4),
            "short_count_share": round(short_count_share, 4),
            "dominant_side": dominant_side,
        }

    def _effective_auto_max_trades(self, base_limit: int, strategy: dict[str, Any] | None) -> int:
        """Unlimited auto-scan entry capacity; hard account/exchange checks still apply."""
        return 10_000

    def _entry_probe_market_quality_block_reason(self, fv: Any, side: str) -> str | None:
        """Block AI-hold probe entries when the fresh market snapshot is contradictory."""
        current_price = self._safe_float(getattr(fv, "current_price", 0.0), 0.0)
        close_price = self._safe_float(getattr(fv, "close", 0.0), 0.0)
        returns_20 = self._safe_float(getattr(fv, "returns_20", 0.0), 0.0)
        volume_ratio = self._safe_float(getattr(fv, "volume_ratio", 0.0), 0.0)
        price_vs_sma20 = self._safe_float(getattr(fv, "price_vs_sma20", 0.0), 0.0)
        price_vs_sma50 = self._safe_float(getattr(fv, "price_vs_sma50", 0.0), 0.0)

        if current_price > 0 and close_price > 0:
            gap = abs(current_price - close_price) / max(close_price, 1e-12)
            if gap >= ENTRY_PROBE_MAX_PRICE_FIELD_GAP:
                return (
                    f"行情快照价格字段自相矛盾：current_price 与 close 相差 {gap * 100:.2f}%，"
                    "该快照不能触发服务器盈利模型补仓开仓。"
                )

        if side == "long":
            if returns_20 <= -ENTRY_PROBE_STRONG_CONTRA_20M_PCT and price_vs_sma20 < 0 and price_vs_sma50 < 0:
                return (
                    f"做多探针被阻止：20分钟收益 {returns_20 * 100:.2f}% 且价格仍在短中期均线下方，"
                    "短线结构没有支持追多。"
                )
        elif side == "short":
            if returns_20 >= ENTRY_PROBE_STRONG_CONTRA_20M_PCT and price_vs_sma20 > 0 and price_vs_sma50 > 0:
                return (
                    f"做空探针被阻止：20分钟收益 {returns_20 * 100:.2f}% 且价格仍在短中期均线上方，"
                    "短线结构没有支持追空。"
                )

        if 0 < volume_ratio < 0.02:
            return (
                f"当前成交量相对均值过低（volume_ratio={volume_ratio:.4f}），"
                "服务器盈利模型弱信号不能在低活跃盘口触发补仓开仓。"
            )
        return None

    def _quant_profit_probe_decision(
        self,
        original: DecisionOutput,
        fv: Any,
        strategy: dict[str, Any] | None,
        ml_signal_context: dict[str, Any] | None,
        local_ai_tools_context: dict[str, Any] | None,
        direction_competition_context: dict[str, Any] | None,
    ) -> DecisionOutput | None:
        """Create a small entry candidate when AI holds but profit models show a gap.

        This does not bypass risk. The candidate still goes through opportunity
        scoring, margin checks, price guards, and final execution gates.
        """
        if not original.is_hold:
            return None
        tools = local_ai_tools_context if isinstance(local_ai_tools_context, dict) else {}
        profit = tools.get("profit_prediction") if isinstance(tools.get("profit_prediction"), dict) else {}
        if not profit or profit.get("available", True) is False:
            return None
        side = str(profit.get("best_side") or "").lower()
        if side not in {"long", "short"}:
            return None
        side_expected = self._safe_float(
            profit.get(f"adjusted_{side}_return_pct", profit.get(f"{side}_expected_return_pct")),
            0.0,
        )
        opposite = "short" if side == "long" else "long"
        opposite_expected = self._safe_float(
            profit.get(f"adjusted_{opposite}_return_pct", profit.get(f"{opposite}_expected_return_pct")),
            0.0,
        )
        edge = side_expected - opposite_expected
        loss_probability = self._safe_float(profit.get(f"{side}_loss_probability"), 0.50)
        roster = strategy.get("portfolio_roster") if isinstance(strategy, dict) else {}
        roster_underfilled = bool(isinstance(roster, dict) and roster.get("underfilled"))
        min_expected = (
            PORTFOLIO_ROSTER_FILL_MIN_EXPECTED_PCT
            if roster_underfilled
            else QUANT_PROFIT_PROBE_MIN_EXPECTED_PCT
        )
        min_edge = (
            PORTFOLIO_ROSTER_FILL_MIN_EDGE_PCT
            if roster_underfilled
            else QUANT_PROFIT_PROBE_MIN_EDGE_PCT
        )
        max_loss_probability = (
            PORTFOLIO_ROSTER_FILL_MAX_LOSS_PROBABILITY
            if roster_underfilled
            else 0.58
        )
        if not roster_underfilled:
            min_expected = max(min_expected, QUANT_PROFIT_PROBE_MIN_EXPECTED_PCT)
            min_edge = max(min_edge, QUANT_PROFIT_PROBE_MIN_EDGE_PCT)
        if side_expected < min_expected or edge < min_edge:
            return None
        if loss_probability >= max_loss_probability:
            return None

        exposure = strategy.get("position_exposure") if isinstance(strategy, dict) else {}
        dominant_side = str(exposure.get("dominant_side") or "")
        concentrated_short_loss = (
            dominant_side == "short"
            and self._safe_float(exposure.get("short_count_share"), 0.0) >= 0.80
            and self._safe_float(exposure.get("short_unrealized_pnl"), 0.0) <= -QUANT_PROFIT_PROBE_MIN_CONCENTRATED_SHORT_LOSS_USDT
        )
        concentrated_long_loss = (
            dominant_side == "long"
            and self._safe_float(exposure.get("long_count_share"), 0.0) >= 0.80
            and self._safe_float(exposure.get("long_unrealized_pnl"), 0.0) <= -QUANT_PROFIT_PROBE_MIN_CONCENTRATED_SHORT_LOSS_USDT
        )
        concentrated_loss_rebalance = bool(concentrated_short_loss or concentrated_long_loss)
        if dominant_side in {"long", "short"} and side == dominant_side:
            count_share = self._safe_float(exposure.get(f"{side}_count_share"), 0.0)
            side_unrealized = self._safe_float(exposure.get(f"{side}_unrealized_pnl"), 0.0)
            if count_share >= 0.80 and side_unrealized <= 0:
                return None

        confidence = min(max(0.60 + min(side_expected, 1.2) * 0.11 + min(edge, 2.0) * 0.05, 0.60), 0.80)
        stop_loss_pct = 0.012
        strong_probe = (
            side_expected >= max(QUANT_PROFIT_PROBE_MIN_EXPECTED_PCT * 2.0, 0.45)
            and edge >= max(QUANT_PROFIT_PROBE_MIN_EDGE_PCT * 2.0, 0.50)
            and loss_probability < 0.50
        )
        market_quality_block = self._entry_probe_market_quality_block_reason(fv, side)
        if market_quality_block and not strong_probe:
            raw = original.raw_response if isinstance(original.raw_response, dict) else {}
            raw["quant_profit_probe_blocked"] = {
                "blocked": True,
                "reason": market_quality_block,
                "side": side,
                "expected_return_pct": round(side_expected, 6),
                "edge_pct": round(edge, 6),
                "loss_probability": round(loss_probability, 6),
            }
            original.raw_response = raw
            return None
        roster_fill_probe = bool(roster_underfilled and not strong_probe)
        min_reward_risk = 3.60 if strong_probe else 2.80
        take_profit_cap = 0.085 if strong_probe else 0.065
        take_profit_pct = max(
            stop_loss_pct * min_reward_risk,
            min(take_profit_cap, stop_loss_pct * min_reward_risk + side_expected / 100.0 * 0.70),
        )
        probe_size = 0.060 if strong_probe else (0.020 if roster_fill_probe else 0.025)
        probe_leverage = 5.0 if strong_probe else 3.0
        raw_response = original.raw_response if isinstance(original.raw_response, dict) else {}
        raw_response = dict(raw_response)
        raw_response.update({
            "analysis_type": "market",
            "ml_signal": ml_signal_context or {},
            "local_ai_tools": tools,
            "direction_competition": direction_competition_context or {},
            "quant_profit_probe": {
                "triggered": True,
                "source": "server_profit_model",
                "side": side,
                "expected_return_pct": round(side_expected, 6),
                "opposite_expected_return_pct": round(opposite_expected, 6),
                "edge_pct": round(edge, 6),
                "loss_probability": round(loss_probability, 6),
                "dominant_side": dominant_side,
                "concentrated_loss_rebalance": concentrated_loss_rebalance,
                "strong_probe": strong_probe,
                "roster_fill_probe": roster_fill_probe,
                "portfolio_roster": roster if isinstance(roster, dict) else {},
                "position_size_pct": round(probe_size, 6),
                "suggested_leverage": round(probe_leverage, 6),
                "stop_loss_pct": round(stop_loss_pct, 6),
                "take_profit_pct": round(take_profit_pct, 6),
                "reward_risk_ratio": round(take_profit_pct / max(stop_loss_pct, 1e-12), 6),
                "reason": (
                    "当前组合低于目标持仓数，AI 观望但服务器盈利模型给出正期望；"
                    "生成补仓小仓候选并继续走完整风控。"
                    if roster_fill_probe
                    else "AI 观望，但服务器盈利模型给出正期望；生成小仓候选并继续走完整风控。"
                ),
            },
        })
        candidate = DecisionOutput(
            model_name=original.model_name,
            symbol=original.symbol,
            action=Action.LONG if side == "long" else Action.SHORT,
            confidence=confidence,
            reasoning=(
                f"服务器盈利模型触发{'补仓' if roster_fill_probe else '正期望'}小仓候选：{side} 调整后预期收益 {side_expected:.2f}%，"
                f"相对另一方向优势 {edge:.2f}%，亏损概率 {loss_probability:.1%}。"
            ),
            position_size_pct=probe_size,
            suggested_leverage=probe_leverage,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            raw_response=raw_response,
            feature_snapshot=fv.to_dict() if hasattr(fv, "to_dict") else (original.feature_snapshot or {}),
        )
        self.entry_policy.score_candidate(candidate, strategy)
        opportunity = (
            candidate.raw_response.get("opportunity_score")
            if isinstance(candidate.raw_response, dict) and isinstance(candidate.raw_response.get("opportunity_score"), dict)
            else {}
        )
        expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = self._safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        tail_risk = self._safe_float(opportunity.get("tail_risk_score"), 1.0)
        if expected_net <= 0 or profit_quality <= 0 or tail_risk >= 0.95:
            raw = original.raw_response if isinstance(original.raw_response, dict) else {}
            raw["quant_profit_probe_blocked"] = {
                "blocked": True,
                "reason": (
                    "服务器盈利模型弱正收益不足以覆盖本地ML、时序、手续费滑点和尾部风险；"
                    "综合机会评分后预期净收益不是正数，因此不再把 AI 观望强行转成开仓。"
                ),
                "side": side,
                "server_expected_return_pct": round(side_expected, 6),
                "expected_net_return_pct": round(expected_net, 6),
                "profit_quality_ratio": round(profit_quality, 6),
                "tail_risk_score": round(tail_risk, 6),
            }
            original.raw_response = raw
            return None
        return candidate

    def _entry_evidence_probe_decision(
        self,
        original: DecisionOutput,
        fv: Any,
        strategy: dict[str, Any] | None,
        ml_signal_context: dict[str, Any] | None,
        local_ai_tools_context: dict[str, Any] | None,
        direction_competition_context: dict[str, Any] | None,
    ) -> DecisionOutput | None:
        """Turn AI-hold into a controlled entry when pre-AI evidence is positive.

        This is meant to handle an overly conservative LLM: if the evidence pack
        says one side has positive net opportunity and controllable risk, create
        a small/medium candidate that still goes through all execution guards.
        """
        if not original.is_hold:
            return None
        raw = original.raw_response if isinstance(original.raw_response, dict) else {}
        evidence = raw.get("entry_candidate_evidence") if isinstance(raw.get("entry_candidate_evidence"), dict) else {}
        if not evidence:
            return None
        sides = [
            side for side in ("long", "short")
            if isinstance(evidence.get(side), dict)
        ]
        if not sides:
            return None

        def score_side(side: str) -> float:
            item = evidence.get(side) or {}
            expected_net = self._safe_float(item.get("expected_net_return_pct"), 0.0)
            quality = self._safe_float(item.get("profit_quality_ratio"), 0.0)
            loss_probability = self._safe_float(item.get("loss_probability"), 1.0)
            tail_risk = self._safe_float(item.get("tail_risk_score"), 1.35)
            score = self._safe_float(item.get("score"), -999.0)
            min_ref = self._safe_float(item.get("min_score_reference"), MIN_ENTRY_OPPORTUNITY_SCORE)
            return (
                expected_net * 2.2
                + quality * 0.75
                + max(score - min_ref + 0.35, -1.0) * 0.40
                - max(loss_probability - 0.48, 0.0) * 1.8
                - max(tail_risk - 0.78, 0.0) * 1.4
            )

        side = max(sides, key=score_side)
        item = evidence.get(side) or {}
        expected_net = self._safe_float(item.get("expected_net_return_pct"), 0.0)
        quality = self._safe_float(item.get("profit_quality_ratio"), 0.0)
        loss_probability = self._safe_float(item.get("loss_probability"), 1.0)
        tail_risk = self._safe_float(item.get("tail_risk_score"), 1.35)
        score = self._safe_float(item.get("score"), -999.0)
        min_ref = self._safe_float(item.get("min_score_reference"), MIN_ENTRY_OPPORTUNITY_SCORE)
        recommendation = str(item.get("recommendation") or "")
        high_profit = bool(item.get("high_profit_potential"))
        if expected_net < 0.30 or quality < 0.25 or loss_probability > 0.56 or tail_risk > 0.92:
            return None
        market_quality_block = self._entry_probe_market_quality_block_reason(fv, side)
        if market_quality_block and not high_profit:
            raw["evidence_profit_probe_blocked"] = {
                "blocked": True,
                "reason": market_quality_block,
                "side": side,
                "expected_net_return_pct": round(expected_net, 6),
                "profit_quality_ratio": round(quality, 6),
            }
            original.raw_response = raw
            return None
        if score < max(min_ref - 0.65, 0.20) and "high_profit" not in recommendation:
            return None
        if recommendation == "hold_or_tiny_probe_only" and not high_profit:
            return None

        if high_profit or expected_net >= 1.20:
            size = 0.075
            leverage = 8.0
            confidence = 0.76
        elif expected_net >= 0.65 and quality >= 0.65 and loss_probability <= 0.48:
            size = 0.055
            leverage = 6.0
            confidence = 0.70
        else:
            size = 0.035
            leverage = 5.0
            confidence = 0.64
        stop_loss_pct = 0.014 if tail_risk <= 0.78 else 0.012
        take_profit_pct = min(max(stop_loss_pct * 3.2, expected_net / 100.0 * 0.85), 0.10)
        raw_response = dict(raw)
        raw_response.update({
            "analysis_type": "market",
            "ml_signal": ml_signal_context or raw.get("ml_signal") or {},
            "local_ai_tools": local_ai_tools_context or raw.get("local_ai_tools") or {},
            "direction_competition": direction_competition_context or raw.get("direction_competition") or {},
            "evidence_profit_probe": {
                "triggered": True,
                "source": "entry_candidate_evidence",
                "ai_original_action": original.action.value,
                "side": side,
                "expected_net_return_pct": round(expected_net, 6),
                "profit_quality_ratio": round(quality, 6),
                "loss_probability": round(loss_probability, 6),
                "tail_risk_score": round(tail_risk, 6),
                "score": round(score, 6),
                "min_score_reference": round(min_ref, 6),
                "high_profit_potential": high_profit,
                "position_size_pct": round(size, 6),
                "suggested_leverage": round(leverage, 6),
                "reason": "AI 原始观望，但入场候选证据包显示该方向为正期望且风险可控，生成受控探针候选。",
            },
        })
        return DecisionOutput(
            model_name=ENSEMBLE_TRADER_NAME,
            symbol=original.symbol,
            action=Action.LONG if side == "long" else Action.SHORT,
            confidence=confidence,
            reasoning=(
                f"AI 原始观望；证据包显示{('做多' if side == 'long' else '做空')}正期望 "
                f"{expected_net:.2f}%，盈亏质量 {quality:.2f}，亏损概率 {loss_probability:.0%}，"
                "转为受控开仓候选。"
            ),
            position_size_pct=size,
            suggested_leverage=min(leverage, settings.max_leverage),
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            raw_response=raw_response,
            feature_snapshot=fv.to_dict() if hasattr(fv, "to_dict") else (original.feature_snapshot or {}),
        )

    def _candidate_opportunity_score(
        self,
        decision: DecisionOutput,
        strategy: dict[str, Any] | None = None,
    ) -> float:
        """Rank entry candidates by expected net opportunity, not just confidence."""
        if not decision.is_entry:
            return -1e9

        side = "long" if decision.action == Action.LONG else "short"
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        ml_signal = raw.get("ml_signal") if isinstance(raw.get("ml_signal"), dict) else {}
        predictions = ml_signal.get("predictions") if isinstance(ml_signal, dict) else []
        primary = predictions[0] if isinstance(predictions, list) and predictions else {}
        if not isinstance(primary, dict):
            primary = {}

        influence_enabled = bool(ml_signal.get("influence_enabled", True)) if isinstance(ml_signal, dict) else True
        side_policy = {}
        if isinstance(ml_signal, dict) and isinstance(ml_signal.get("influence_policy"), dict):
            side_policy = ml_signal["influence_policy"].get(side) or {}
        side_influence_enabled = influence_enabled and (
            not isinstance(side_policy, dict) or side_policy.get("enabled", True)
        )
        raw_expected_pct = self._safe_float(primary.get(f"{side}_expected_return_pct"), 0.0)
        expected_pct = max(min(raw_expected_pct, ML_EXPECTED_RETURN_SCORE_CAP_PCT), -ML_EXPECTED_RETURN_SCORE_CAP_PCT)
        opposite = "short" if side == "long" else "long"
        raw_opposite_expected_pct = self._safe_float(primary.get(f"{opposite}_expected_return_pct"), 0.0)
        opposite_expected_pct = max(min(raw_opposite_expected_pct, ML_EXPECTED_RETURN_SCORE_CAP_PCT), -ML_EXPECTED_RETURN_SCORE_CAP_PCT)
        edge_pct = expected_pct - opposite_expected_pct
        win_rate = self._safe_float(primary.get(f"{side}_win_rate"), 0.50)
        ml_quality = self._safe_float(primary.get("profit_quality_score"), 0.0)
        if not side_influence_enabled:
            expected_pct = 0.0
            opposite_expected_pct = 0.0
            edge_pct = 0.0
            win_rate = 0.50
            ml_quality = 0.0

        confidence = max(min(float(decision.confidence or 0.0), 1.0), 0.0)
        size = max(float(decision.position_size_pct or 0.0), 0.0)
        leverage = max(float(decision.suggested_leverage or 1.0), 1.0)
        stop_loss_pct = max(float(decision.stop_loss_pct or 0.0), 0.0)
        take_profit_pct = max(float(decision.take_profit_pct or 0.0), 0.0)
        loss_probability = max(1.0 - confidence, 0.0)
        reward_risk_ratio = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0.0
        ai_expected_return_pct = (
            confidence * take_profit_pct - loss_probability * stop_loss_pct
        ) * 100

        fee_pct = ESTIMATED_TAKER_FEE_PCT * 2 * 100
        slippage_pct = max(float(settings.max_slippage_pct or 0.0), 0.0) * 100
        confidence_bonus = max(confidence - 0.55, 0.0) * 0.45
        rr_bonus = max(min(reward_risk_ratio - 1.0, 2.0), 0.0) * 0.16
        risk_penalty = max(0.58 - confidence, 0.0) * 0.85
        weak_rr_penalty = max(1.0 - reward_risk_ratio, 0.0) * 0.75

        exposure_penalty = 0.0
        exposure_balance_bonus = 0.0
        exposure = strategy.get("position_exposure") if isinstance(strategy, dict) else {}
        if isinstance(exposure, dict) and exposure.get("dominant_side") == side:
            net_ratio_abs = abs(self._safe_float(exposure.get("net_ratio"), 0.0))
            count_share = self._safe_float(exposure.get(f"{side}_count_share"), 0.0)
            exposure_penalty = net_ratio_abs * 0.35 + max(count_share - 0.70, 0.0) * 0.45
        elif isinstance(exposure, dict) and exposure.get("dominant_side") in {"long", "short"}:
            dominant = str(exposure.get("dominant_side") or "")
            opposite_dominant = "short" if dominant == "long" else "long"
            if side == opposite_dominant:
                exposure_balance_bonus = abs(self._safe_float(exposure.get("net_ratio"), 0.0)) * 0.10
        base_min_score_required = self._safe_float(
            strategy.get("min_opportunity_score") if isinstance(strategy, dict) else MIN_ENTRY_OPPORTUNITY_SCORE,
            MIN_ENTRY_OPPORTUNITY_SCORE,
        )
        min_score_required = base_min_score_required
        dynamic_score_reason = f"分歧大、波动异常或没有盈利模型同向确认，保持 {base_min_score_required:.2f}+ 基础门槛。"
        local_tools = raw.get("local_ai_tools") if isinstance(raw.get("local_ai_tools"), dict) else {}
        local_profit = local_tools.get("profit_prediction") if isinstance(local_tools.get("profit_prediction"), dict) else {}
        local_best_side = str(local_profit.get("best_side") or "").lower()
        local_expected = self._safe_float(
            local_profit.get(f"adjusted_{side}_return_pct", local_profit.get(f"{side}_expected_return_pct")),
            0.0,
        )
        local_available = bool(local_profit.get("available") or local_profit.get("trained"))
        ml_aligned = side_influence_enabled and expected_pct > 0 and (edge_pct >= 0 or str(primary.get("best_side") or "").lower() == side)
        local_aligned = local_available and local_best_side == side and local_expected > 0
        local_conflicts = local_available and (
            local_expected <= 0
            or (local_best_side in {"long", "short"} and local_best_side != side)
        )
        local_loss_probability = self._safe_float(local_profit.get(f"{side}_loss_probability"), 0.50)
        local_quality = self._safe_float(local_profit.get("profit_quality_score"), 0.0)
        ts_prediction = (
            local_tools.get("time_series_prediction")
            or local_tools.get("timeseries_prediction")
            or local_tools.get("sequence_prediction")
        )
        if not isinstance(ts_prediction, dict):
            ts_prediction = {}
        ts_best_side = str(ts_prediction.get("best_side") or ts_prediction.get("side") or "").lower()
        ts_expected = self._safe_float(
            ts_prediction.get("expected_return_pct", ts_prediction.get(f"{side}_expected_return_pct")),
            0.0,
        )
        ts_aligned = bool(ts_prediction) and ts_best_side == side and ts_expected > 0
        if isinstance(exposure, dict) and exposure.get("dominant_side") in {"long", "short"}:
            dominant = str(exposure.get("dominant_side") or "")
            opposite_dominant = "short" if dominant == "long" else "long"
            if side == opposite_dominant and (expected_pct > 0 or local_expected > 0 or ts_aligned):
                # Prefer portfolio balance only when this side has profit evidence.
                exposure_balance_bonus += min(abs(self._safe_float(exposure.get("net_ratio"), 0.0)) * 0.18, 0.22)
        experts = raw.get("experts") if isinstance(raw.get("experts"), list) else []
        entry_votes = 0
        opposite_votes = 0
        hold_votes = 0
        for expert in experts:
            if not isinstance(expert, dict):
                continue
            action_value = str(expert.get("action") or "").lower()
            if action_value == side:
                entry_votes += 1
            elif action_value == opposite:
                opposite_votes += 1
            elif action_value == "hold":
                hold_votes += 1
        entry_support = raw.get("entry_signal_support") if isinstance(raw.get("entry_signal_support"), dict) else {}
        if entry_support.get("side") == side:
            support_experts = entry_support.get("same_direction_experts")
            technical_support = entry_support.get("technical_support")
            if isinstance(support_experts, list):
                entry_votes = max(entry_votes, len(support_experts))
            if isinstance(technical_support, list) and len(technical_support) >= 2:
                opposite_votes = 0
        expert_aligned = entry_votes >= 2 and opposite_votes == 0
        high_disagreement = opposite_votes > 0 or (hold_votes >= 3 and entry_votes < 2)
        direction_competition = raw.get("direction_competition") if isinstance(raw.get("direction_competition"), dict) else {}
        if not direction_competition and isinstance(strategy, dict) and isinstance(strategy.get("direction_competition"), dict):
            direction_competition = strategy.get("direction_competition") or {}
        direction_preferred_side = str(direction_competition.get("preferred_side") or "neutral").lower()
        direction_gap = self._safe_float(direction_competition.get("score_gap"), 0.0)
        direction_side_score = self._safe_float(
            (direction_competition.get(side) or {}).get("score") if isinstance(direction_competition.get(side), dict) else 0.0,
            0.0,
        )
        direction_opposite_score = self._safe_float(
            (direction_competition.get(opposite) or {}).get("score") if isinstance(direction_competition.get(opposite), dict) else 0.0,
            0.0,
        )
        direction_alignment_bonus = 0.0
        direction_conflict_penalty = 0.0
        if direction_preferred_side == side and direction_gap >= 0.08:
            direction_alignment_bonus = min(direction_gap, 1.8) * 0.32
        elif direction_preferred_side == opposite and direction_gap >= 0.12:
            direction_conflict_penalty = min(direction_gap, 2.0) * 0.55
            high_disagreement = True
        elif direction_preferred_side == "neutral" and abs(direction_side_score - direction_opposite_score) < 0.08:
            direction_conflict_penalty = 0.10
        contribution_sources: list[str] = []
        if ml_aligned:
            contribution_sources.append("ml_profit_model")
        if local_aligned:
            contribution_sources.append("server_profit_model")
        if ts_aligned:
            contribution_sources.append("timeseries_model")
        if expert_aligned:
            contribution_sources.append("expert_alignment")
        if not any(source in contribution_sources for source in ("ml_profit_model", "server_profit_model", "timeseries_model")):
            contribution_sources.append("ai_only_without_quant")
        contribution_perf = (
            strategy.get("model_contribution_performance")
            if isinstance(strategy, dict) and isinstance(strategy.get("model_contribution_performance"), dict)
            else {}
        )
        contribution_adjustment = self._model_contribution_score_adjustment(
            contribution_sources,
            contribution_perf,
        )
        portfolio_roster = (
            strategy.get("portfolio_roster")
            if isinstance(strategy, dict) and isinstance(strategy.get("portfolio_roster"), dict)
            else {}
        )
        contribution_score_multiplier = self._safe_float(
            contribution_adjustment.get("score_multiplier"),
            1.0,
        )
        contribution_size_multiplier = self._safe_float(
            contribution_adjustment.get("size_multiplier"),
            1.0,
        )
        contribution_score_adjustment = self._safe_float(
            contribution_adjustment.get("score_adjustment"),
            0.0,
        )
        if contribution_size_multiplier != 1.0:
            previous_adjustment = raw.get("model_contribution_adjustment")
            size_already_applied = (
                isinstance(previous_adjustment, dict)
                and previous_adjustment.get("size_applied") is True
            )
            if not size_already_applied:
                original_size = size
                size = max(min(size * contribution_size_multiplier, 1.0), 0.0)
                decision.position_size_pct = size
                contribution_adjustment["original_position_size"] = round(original_size, 6)
                contribution_adjustment["adjusted_position_size"] = round(size, 6)
                contribution_adjustment["size_applied"] = True
        feature_snapshot = decision.feature_snapshot if isinstance(decision.feature_snapshot, dict) else {}
        volatility = self._safe_float(feature_snapshot.get("volatility_20"), 0.0)
        day_change = abs(self._safe_float(feature_snapshot.get("change_24h_pct"), 0.0))
        abnormal_volatility = volatility >= 0.08 or day_change >= 18.0
        if not high_disagreement and not abnormal_volatility:
            if ml_aligned and local_aligned:
                min_score_required = min(min_score_required, DYNAMIC_ENTRY_SCORE_ML_ALIGNED_STRONG)
                dynamic_score_reason = "ML 与服务器盈利模型同向且预期收益为正，允许 0.75+ 小仓开仓。"
            elif ml_aligned or local_aligned:
                min_score_required = min(min_score_required, DYNAMIC_ENTRY_SCORE_ML_ALIGNED)
                dynamic_score_reason = "ML 或服务器盈利模型与 AI 方向同向且预期收益为正，允许 0.85+ 小仓开仓。"
            elif expert_aligned and expected_pct > 0:
                min_score_required = min(min_score_required, DYNAMIC_ENTRY_SCORE_EXPERT_ALIGNED)
                dynamic_score_reason = "专家方向一致且预期收益为正，允许 0.90+ 开仓。"
        if contribution_score_multiplier >= 1.06:
            min_score_required = max(min(min_score_required, base_min_score_required - 0.08), 0.72)
            dynamic_score_reason = (
                f"{dynamic_score_reason} 最近真实平仓贡献为正，闭环调权后放宽 0.08。"
            )
        elif contribution_score_multiplier <= 0.94:
            min_score_required = max(min_score_required + 0.18, base_min_score_required)
            dynamic_score_reason = (
                f"{dynamic_score_reason} 最近真实平仓贡献为负，闭环调权后提高门槛并缩小仓位。"
            )
        symbol_key = self._normalize_position_symbol(decision.symbol) or decision.symbol
        profiles = strategy.get("symbol_side_performance") if isinstance(strategy, dict) else {}
        if not isinstance(profiles, dict):
            profiles = {}
        side_profile = profiles.get(f"{symbol_key}|{side}") if isinstance(profiles.get(f"{symbol_key}|{side}"), dict) else {}
        symbol_profile = profiles.get(f"{symbol_key}|all") if isinstance(profiles.get(f"{symbol_key}|all"), dict) else {}
        historical_adjustment = 0.0
        historical_block = False
        historical_reason = "今天还没有该币种方向的真实平仓记录。"
        for profile, weight, label in (
            (symbol_profile, 0.55, "symbol"),
            (side_profile, 1.00, "symbol-side"),
        ):
            if not isinstance(profile, dict) or int(profile.get("count") or 0) <= 0:
                continue
            pnl = self._safe_float(profile.get("pnl"), 0.0)
            avg_pnl = self._safe_float(profile.get("avg_pnl"), 0.0)
            profit_factor = self._safe_float(profile.get("profit_factor"), 0.0)
            losses = int(profile.get("losses") or 0)
            wins = int(profile.get("wins") or 0)
            if profile.get("cooldown"):
                label_cn = "symbol" if label == "symbol" else "symbol-side"
                historical_block = True
                historical_reason = (
                    f"{label_cn} recent realized PnL is weak: pnl={pnl:.2f} U, "
                    f"losses={losses}, wins={wins}, profit_factor={profit_factor:.2f}."
                )
            if pnl > 0 and profit_factor >= 1.25:
                historical_adjustment += min(pnl / 32.0, ENTRY_REALIZED_EDGE_BONUS_CAP) * weight
            if avg_pnl < 0 or profit_factor < 0.75:
                loss_count_penalty = min(losses, 10) * 0.06
                avg_loss_penalty = abs(avg_pnl) / 12.0
                historical_adjustment -= min(
                    avg_loss_penalty + loss_count_penalty,
                    ENTRY_REALIZED_EDGE_PENALTY_CAP,
                ) * weight
            if losses >= wins + 2 and pnl < 0:
                historical_adjustment -= 0.25 * weight

        side_losses = int(side_profile.get("losses") or 0) if isinstance(side_profile, dict) else 0
        side_wins = int(side_profile.get("wins") or 0) if isinstance(side_profile, dict) else 0
        side_avg_pnl = self._safe_float(side_profile.get("avg_pnl"), 0.0) if isinstance(side_profile, dict) else 0.0
        side_profit_factor = self._safe_float(side_profile.get("profit_factor"), 1.0) if isinstance(side_profile, dict) else 1.0
        side_largest_loss = abs(self._safe_float(side_profile.get("largest_loss"), 0.0)) if isinstance(side_profile, dict) else 0.0
        side_profit = self._safe_float(side_profile.get("profit"), 0.0) if isinstance(side_profile, dict) else 0.0
        side_count = int(side_profile.get("count") or 0) if isinstance(side_profile, dict) else 0
        side_pnl = self._safe_float(side_profile.get("pnl"), 0.0) if isinstance(side_profile, dict) else 0.0
        side_loss = self._safe_float(side_profile.get("loss"), 0.0) if isinstance(side_profile, dict) else 0.0
        symbol_pnl = self._safe_float(symbol_profile.get("pnl"), 0.0) if isinstance(symbol_profile, dict) else 0.0
        symbol_count = int(symbol_profile.get("count") or 0) if isinstance(symbol_profile, dict) else 0
        symbol_profit_factor = (
            self._safe_float(symbol_profile.get("profit_factor"), 1.0)
            if isinstance(symbol_profile, dict)
            else 1.0
        )
        symbol_profit_tier = "neutral"
        symbol_tier_reason = "该币种/方向近期真实盈亏样本不足，按中性处理。"
        symbol_tier_score_adjustment = 0.0
        if (
            side_count >= ENTRY_SYMBOL_WINNER_MIN_COUNT
            and side_pnl >= ENTRY_SYMBOL_WINNER_MIN_PNL_USDT
            and side_profit_factor >= ENTRY_SYMBOL_WINNER_MIN_PROFIT_FACTOR
        ):
            symbol_profit_tier = "side_winner"
            symbol_tier_score_adjustment = min(side_pnl / 28.0, ENTRY_SYMBOL_WINNER_SCORE_BONUS_CAP)
            min_score_required = max(
                min(min_score_required, base_min_score_required - ENTRY_SYMBOL_WINNER_SCORE_RELIEF),
                0.68,
            )
            symbol_tier_reason = (
                f"该币种{('做多' if side == 'long' else '做空')}方向最近真实盈利 {side_pnl:.2f}U，"
                f"盈利因子 {side_profit_factor:.2f}，优先把资金给已验证能赚钱的方向。"
            )
        elif (
            symbol_count >= ENTRY_SYMBOL_WINNER_MIN_COUNT
            and symbol_pnl >= ENTRY_SYMBOL_WINNER_MIN_PNL_USDT
            and symbol_profit_factor >= ENTRY_SYMBOL_WINNER_MIN_PROFIT_FACTOR
        ):
            symbol_profit_tier = "symbol_winner"
            symbol_tier_score_adjustment = min(symbol_pnl / 44.0, ENTRY_SYMBOL_WINNER_SCORE_BONUS_CAP * 0.65)
            min_score_required = max(
                min(min_score_required, base_min_score_required - ENTRY_SYMBOL_WINNER_SCORE_RELIEF * 0.5),
                0.72,
            )
            symbol_tier_reason = (
                f"该币种最近真实盈利 {symbol_pnl:.2f}U，盈利因子 {symbol_profit_factor:.2f}；"
                "方向仍按当前 AI/模型判断，但执行层给予轻微优先级。"
            )
        elif side_count >= 2 and side_pnl < 0 and (side_loss > side_profit * 1.15 or side_profit_factor < 0.80):
            symbol_profit_tier = "side_loser"
            symbol_tier_score_adjustment = -min(abs(side_pnl) / 32.0 + side_losses * 0.05, 0.75)
            symbol_tier_reason = (
                f"该币种{('做多' if side == 'long' else '做空')}方向近期真实净亏 {side_pnl:.2f}U，"
                "不永久禁用，但降低评分和仓位，等待新证据证明值得再试。"
            )
        small_win_big_loss_penalty = 0.0
        if side_count >= 2 and side_largest_loss > 0:
            avg_win = side_profit / max(side_wins, 1)
            loss_to_win_ratio = side_largest_loss / max(avg_win, 0.25)
            if loss_to_win_ratio >= 3.0 or side_profit_factor < 0.80:
                small_win_big_loss_penalty = min(
                    ENTRY_SMALL_WIN_BIG_LOSS_PENALTY_CAP,
                    (loss_to_win_ratio - 2.0) * 0.12 + max(0.80 - side_profit_factor, 0.0) * 0.55,
                )
                historical_adjustment -= small_win_big_loss_penalty
        tail_history_component = 0.0
        if side_losses > 0 and (side_avg_pnl < 0 or side_profit_factor < 0.80):
            tail_history_component = min(
                0.35,
                side_losses * 0.035
                + max(abs(side_avg_pnl) / 18.0, 0.0)
                + (0.08 if side_losses >= side_wins + 2 else 0.0),
            )
        stop_risk_component = min(max(stop_loss_pct / 0.055, 0.0), 1.0) * 0.22
        loss_probability_component = min(max(local_loss_probability, 0.0), 1.0) * 0.36
        volatility_component = min(max(volatility / 0.08, 0.0), 1.0) * 0.17
        abnormal_wick_max_pct = self._safe_float(feature_snapshot.get("abnormal_wick_max_pct"), 0.0)
        abnormal_wick_count = int(self._safe_float(feature_snapshot.get("abnormal_wick_count_72h"), 0.0))
        abnormal_wick_recent_hours = self._safe_float(feature_snapshot.get("abnormal_wick_recent_hours"), 9999.0)
        abnormal_wick_component = 0.0
        if abnormal_wick_max_pct >= ABNORMAL_WICK_TAIL_RISK_MAX_PCT and abnormal_wick_count > 0:
            recency_weight = 1.0 if abnormal_wick_recent_hours <= 24.0 else 0.70 if abnormal_wick_recent_hours <= 72.0 else 0.45
            abnormal_wick_component = min(abnormal_wick_max_pct / 300.0, 0.55) * recency_weight
        disagreement_component = (0.16 if high_disagreement else 0.0) + (0.09 if abnormal_volatility else 0.0)
        tail_risk_score = min(
            max(
                loss_probability_component
                + stop_risk_component
                + volatility_component
                + abnormal_wick_component
                + tail_history_component
                + disagreement_component,
                0.0,
            ),
            1.35,
        )
        tail_risk_penalty = tail_risk_score * 0.92
        quant_conflict_penalty = 0.0
        same_side_loss_concentration = False
        if local_available and local_expected <= 0:
            quant_conflict_penalty += min(abs(local_expected) * 1.35 + 0.45, 1.75)
        if local_available and local_best_side in {"long", "short"} and local_best_side != side:
            quant_conflict_penalty += 0.55
        if local_loss_probability >= 0.64 and not local_aligned:
            quant_conflict_penalty += min((local_loss_probability - 0.60) * 1.6, 0.55)
        if isinstance(exposure, dict) and exposure.get("dominant_side") == side:
            count_share = self._safe_float(exposure.get(f"{side}_count_share"), 0.0)
            side_unrealized = self._safe_float(exposure.get(f"{side}_unrealized_pnl"), 0.0)
            if count_share >= 0.85 and side_unrealized < 0:
                same_side_loss_concentration = True
                quant_conflict_penalty += min(abs(side_unrealized) / 30.0, 0.65)
                dynamic_score_reason = (
                    f"当前组合已经高度集中在{side}，且该方向浮亏 {side_unrealized:.2f}U；"
                    "本轮只作为风险扣分，不再直接禁止同方向开仓。"
                )

        strong_current_profit_support = (
            local_aligned
            and local_expected > 0
            and local_quality >= 0.35
            and local_loss_probability < 0.62
        )
        historical_adjustment_cap = -0.85 if strong_current_profit_support else -1.80
        if historical_adjustment < historical_adjustment_cap:
            historical_adjustment = historical_adjustment_cap
        historical_adjustment += symbol_tier_score_adjustment

        expected_net_return_pct = (
            ai_expected_return_pct * ENTRY_NET_WEIGHT_AI
            + expected_pct * ENTRY_NET_WEIGHT_LOCAL_ML
            + local_expected * ENTRY_NET_WEIGHT_SERVER_PROFIT
            + ts_expected * ENTRY_NET_WEIGHT_TIMESERIES
            - fee_pct
            - slippage_pct
        )
        expected_loss_pct = max(
            stop_loss_pct * 100 * max(1.0 - confidence, 0.0),
            max(local_loss_probability - 0.50, 0.0) * stop_loss_pct * 100 * 2.0,
            fee_pct + slippage_pct,
        )
        success_probability = min(
            max(
                win_rate * 0.45
                + confidence * 0.30
                + (1.0 - min(max(local_loss_probability, 0.0), 1.0)) * 0.20
                + (0.05 if local_aligned or ts_aligned else 0.0),
                0.0,
            ),
            1.0,
        )
        profit_quality_ratio = expected_net_return_pct / max(expected_loss_pct + fee_pct + slippage_pct, 0.05)
        downside_asymmetry_penalty = 0.0
        if expected_net_return_pct <= 0:
            downside_asymmetry_penalty = min(abs(expected_net_return_pct) * 0.75 + expected_loss_pct * 0.22, 1.25)
        elif expected_loss_pct > expected_net_return_pct * 1.8:
            downside_asymmetry_penalty = min(
                (expected_loss_pct - expected_net_return_pct * 1.8) * 0.32,
                0.75,
            )
        strong_aligned_profit_evidence = (
            expected_net_return_pct > 0
            and not high_disagreement
            and not abnormal_volatility
            and tail_risk_score < 0.88
            and (
                strong_current_profit_support
                or (ml_aligned and expected_pct >= 0.05 and edge_pct >= 0)
                or (ts_aligned and ts_expected > 0)
            )
        )
        min_profit_quality_ratio_required = (
            ENTRY_WEAK_HISTORY_MIN_PROFIT_QUALITY_RATIO
            if historical_block and not strong_aligned_profit_evidence
            else ENTRY_MIN_NET_PROFIT_QUALITY_RATIO
        )
        if strong_aligned_profit_evidence:
            if historical_block:
                min_profit_quality_ratio_required = min(
                    min_profit_quality_ratio_required,
                    ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MIN_PROFIT_QUALITY_RATIO,
                )
            else:
                min_profit_quality_ratio_required = min(
                    min_profit_quality_ratio_required,
                    ENTRY_STRONG_ALIGNED_MIN_PROFIT_QUALITY_RATIO,
                )
        quant_probe = raw.get("quant_profit_probe") if isinstance(raw.get("quant_profit_probe"), dict) else {}
        if (
            quant_probe.get("triggered")
            and local_aligned
            and local_expected >= QUANT_PROFIT_PROBE_MIN_EXPECTED_PCT
            and local_loss_probability < 0.58
            and (direction_preferred_side in {side, "neutral", ""} or local_expected >= 0.45)
        ):
            min_score_required = min(min_score_required, QUANT_PROFIT_PROBE_MIN_SCORE)
            min_profit_quality_ratio_required = min(
                min_profit_quality_ratio_required,
                0.0,
            )
            dynamic_score_reason = (
                "AI 原始观望，但服务器盈利模型给出正期望且亏损概率可控；"
                "按小仓盈利探针门槛执行完整风控，净盈亏比只记录不硬拦截。"
            )
        capital_efficiency_score = expected_net_return_pct * max(leverage, 1.0) / max(size * 100.0, 1.0)
        score = (
            expected_net_return_pct * 2.35
            + profit_quality_ratio * 1.20
            + success_probability * 0.25
            + edge_pct * 0.25
            + local_quality * 0.18
            + confidence * 0.10
            + confidence_bonus
            + rr_bonus
            + min(size * leverage, 1.0) * 0.05
            - expected_loss_pct * 0.90
            - tail_risk_penalty
            - risk_penalty
            - weak_rr_penalty
            - downside_asymmetry_penalty
            - exposure_penalty
            - quant_conflict_penalty
            + exposure_balance_bonus
            + direction_alignment_bonus
            - direction_conflict_penalty
            + historical_adjustment
            + contribution_score_adjustment
        )
        if historical_block:
            score -= 0.20 if strong_current_profit_support else 0.45
            if not strong_current_profit_support:
                min_score_required = max(min_score_required, ENTRY_WEAK_HISTORY_MIN_SCORE)
                dynamic_score_reason = (
                    "该币种/方向近期真实盈亏偏弱；只有净盈亏比足够高且模型证据改善时才允许继续试。"
                )

        raw["opportunity_score"] = {
            "score": round(score, 6),
            "side": side,
            "expected_return_pct": round(expected_pct, 6),
            "raw_expected_return_pct": round(raw_expected_pct, 6),
            "opposite_expected_return_pct": round(opposite_expected_pct, 6),
            "raw_opposite_expected_return_pct": round(raw_opposite_expected_pct, 6),
            "ml_expected_return_score_cap_pct": ML_EXPECTED_RETURN_SCORE_CAP_PCT,
            "profit_edge_pct": round(edge_pct, 6),
            "win_rate": round(win_rate, 6),
            "ml_profit_quality_score": round(ml_quality, 6),
            "server_profit_expected_return_pct": round(local_expected, 6),
            "server_profit_best_side": local_best_side,
            "server_profit_conflict": bool(local_conflicts),
            "server_profit_loss_probability": round(local_loss_probability, 6),
            "server_profit_quality_score": round(local_quality, 6),
            "timeseries_expected_return_pct": round(ts_expected, 6),
            "timeseries_aligned": bool(ts_aligned),
            "confidence": round(confidence, 6),
            "ai_expected_return_pct": round(ai_expected_return_pct, 6),
            "expected_net_return_pct": round(expected_net_return_pct, 6),
            "expected_loss_pct": round(expected_loss_pct, 6),
            "expected_net_weights": {
                "ai_expected_return": ENTRY_NET_WEIGHT_AI,
                "local_ml_expected_return": ENTRY_NET_WEIGHT_LOCAL_ML,
                "server_profit_expected_return": ENTRY_NET_WEIGHT_SERVER_PROFIT,
                "timeseries_expected_return": ENTRY_NET_WEIGHT_TIMESERIES,
            },
            "downside_asymmetry_penalty": round(downside_asymmetry_penalty, 6),
            "tail_risk_score": round(tail_risk_score, 6),
            "tail_risk_penalty": round(tail_risk_penalty, 6),
            "quant_conflict_penalty": round(quant_conflict_penalty, 6),
            "same_side_loss_concentration": bool(same_side_loss_concentration),
            "tail_history_component": round(tail_history_component, 6),
            "stop_risk_component": round(stop_risk_component, 6),
            "abnormal_wick_component": round(abnormal_wick_component, 6),
            "abnormal_wick_count_72h": int(abnormal_wick_count),
            "abnormal_wick_max_pct": round(abnormal_wick_max_pct, 6),
            "abnormal_wick_recent_hours": round(abnormal_wick_recent_hours, 6),
            "success_probability": round(success_probability, 6),
            "profit_quality_ratio": round(profit_quality_ratio, 6),
            "min_profit_quality_ratio_required": round(min_profit_quality_ratio_required, 6),
            "strong_aligned_profit_evidence": bool(strong_aligned_profit_evidence),
            "capital_efficiency_score": round(capital_efficiency_score, 6),
            "reward_risk_ratio": round(reward_risk_ratio, 6),
            "confidence_bonus": round(confidence_bonus, 6),
            "reward_risk_bonus": round(rr_bonus, 6),
            "size_x_leverage": round(size * leverage, 6),
            "fee_pct": round(fee_pct, 6),
            "slippage_pct": round(slippage_pct, 6),
            "risk_penalty": round(risk_penalty, 6),
            "weak_rr_penalty": round(weak_rr_penalty, 6),
            "exposure_penalty": round(exposure_penalty, 6),
            "exposure_balance_bonus": round(exposure_balance_bonus, 6),
            "direction_competition": direction_competition,
            "direction_preferred_side": direction_preferred_side,
            "direction_side_score": round(direction_side_score, 6),
            "direction_opposite_score": round(direction_opposite_score, 6),
            "direction_alignment_bonus": round(direction_alignment_bonus, 6),
            "direction_conflict_penalty": round(direction_conflict_penalty, 6),
            "historical_adjustment": round(historical_adjustment, 6),
            "small_win_big_loss_penalty": round(small_win_big_loss_penalty, 6),
            "side_largest_loss_usdt": round(side_largest_loss, 6),
            "side_profit_factor": round(side_profit_factor, 6),
            "symbol_profit_tier": symbol_profit_tier,
            "symbol_profit_tier_reason": symbol_tier_reason,
            "symbol_tier_score_adjustment": round(symbol_tier_score_adjustment, 6),
            "side_realized_pnl_usdt": round(side_pnl, 6),
            "symbol_realized_pnl_usdt": round(symbol_pnl, 6),
            "model_contribution_adjustment": contribution_adjustment,
            "model_contribution_sources": contribution_sources,
            "model_contribution_score_adjustment": round(contribution_score_adjustment, 6),
            "portfolio_roster": portfolio_roster,
            "historical_adjustment_cap": round(historical_adjustment_cap, 6),
            "strong_current_profit_support": bool(strong_current_profit_support),
            "historical_block": bool(historical_block),
            "historical_reason": historical_reason,
            "weak_history_requires_stronger_edge": bool(historical_block and not strong_current_profit_support),
            "symbol_side_profile": side_profile,
            "symbol_profile": symbol_profile,
            "base_min_score_required": round(base_min_score_required, 6),
            "min_score_required": round(min_score_required, 6),
            "dynamic_score_reason": dynamic_score_reason,
            "ml_aligned": bool(ml_aligned),
            "local_profit_aligned": bool(local_aligned),
            "expert_aligned": bool(expert_aligned),
            "high_disagreement": bool(high_disagreement),
            "abnormal_volatility": bool(abnormal_volatility),
            "entry_vote_count": int(entry_votes),
            "opposite_vote_count": int(opposite_votes),
            "risk_mode": str(strategy.get("risk_mode") or "normal") if isinstance(strategy, dict) else "normal",
            "max_entry_stop_loss_usdt": round(
                self._safe_float(
                    strategy.get("max_entry_stop_loss_usdt") if isinstance(strategy, dict) else ENTRY_MAX_STOP_LOSS_NORMAL_USDT,
                    ENTRY_MAX_STOP_LOSS_NORMAL_USDT,
                ),
                6,
            ),
            "ml_influence_enabled": bool(side_influence_enabled),
            "ml_influence_reason": (
                "ML 当前达标，参与机会评分。"
                if side_influence_enabled
                else "ML 当前处于学习观察中，或该方向未达标，本次机会评分不使用 ML 加减分。"
            ),
            "rule": (
                "auto entries are ranked by expected net return, possible loss, fees, "
                "success probability, and capital efficiency before execution"
            ),
        }
        decision.raw_response = raw
        self._annotate_decision_source(decision)
        return score

    def _annotate_decision_source(self, decision: DecisionOutput) -> dict[str, Any]:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        side = self._entry_side_value(decision) if decision.is_entry else (
            "exit" if decision.is_exit else "hold"
        )
        opinions = raw.get("opinions") if isinstance(raw.get("opinions"), list) else []
        decision_maker = raw.get("decision_maker") if isinstance(raw.get("decision_maker"), dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
        ml_signal = raw.get("ml_signal") if isinstance(raw.get("ml_signal"), dict) else {}
        quant_probe = raw.get("quant_validation_probe_entry") if isinstance(raw.get("quant_validation_probe_entry"), dict) else {}
        server_tools = raw.get("server_quant_tools") if isinstance(raw.get("server_quant_tools"), dict) else {}

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
                "参与盈利质量判断"
                if local_profit_influence
                else "未提供有效同向盈利证据"
            ),
            "ml_influence_enabled": ml_influence_enabled,
            "server_profit_aligned": bool(opportunity.get("local_profit_aligned")),
            "timeseries_aligned": bool(opportunity.get("timeseries_aligned")),
            "opportunity_score": opportunity.get("score"),
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
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
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

    def _entry_immediate_execution_reason(self, decision: DecisionOutput) -> str | None:
        """Return a clear reason when an entry is strong enough to skip round-end sorting."""
        if not decision.is_entry:
            return None
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
        score = self._safe_float(opportunity.get("score"), float("nan"))
        min_score = self._safe_float(opportunity.get("min_score_required"), MIN_ENTRY_OPPORTUNITY_SCORE)
        expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = self._safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        confidence = max(float(decision.confidence or 0.0), self._safe_float(opportunity.get("confidence"), 0.0))
        entry_votes = int(self._safe_float(opportunity.get("entry_vote_count"), 0.0))
        tail_risk_score = self._safe_float(opportunity.get("tail_risk_score"), 0.0)
        high_disagreement = bool(opportunity.get("high_disagreement"))
        abnormal_volatility = bool(opportunity.get("abnormal_volatility"))
        quant_probe = raw.get("quant_profit_probe") if isinstance(raw.get("quant_profit_probe"), dict) else {}
        quant_probe_triggered = bool(quant_probe.get("triggered"))
        strong_quant_probe = bool(quant_probe_triggered and quant_probe.get("strong_probe"))
        roster = opportunity.get("portfolio_roster") if isinstance(opportunity.get("portfolio_roster"), dict) else {}
        roster_underfilled = bool(roster.get("underfilled"))
        roster_fill_probe = bool(quant_probe_triggered and quant_probe.get("roster_fill_probe"))
        quant_loss_probability = self._safe_float(
            quant_probe.get("loss_probability")
            if quant_probe.get("loss_probability") is not None
            else opportunity.get("server_profit_loss_probability"),
            1.0,
        )
        aligned = bool(
            opportunity.get("expert_aligned")
            or opportunity.get("ml_aligned")
            or opportunity.get("local_profit_aligned")
        )
        if not math.isfinite(score) or expected_net <= 0:
            return None
        if high_disagreement or abnormal_volatility:
            return None

        exceptional = (
            score >= max(min_score + 2.0, 4.20)
            and confidence >= 0.86
            and expected_net >= 1.20
            and profit_quality >= 1.50
        )
        strong_aligned = (
            score >= max(min_score + 1.00, 2.80)
            and confidence >= 0.88
            and expected_net >= 0.80
            and profit_quality >= 1.20
            and (aligned or entry_votes >= 2)
        )
        strong_quant = (
            strong_quant_probe
            and score >= max(min_score + 1.00, 2.80)
            and confidence >= 0.78
            and expected_net >= 0.75
            and profit_quality >= 0.85
            and quant_loss_probability <= 0.42
        )
        medium_quant = (
            quant_probe_triggered
            and not strong_quant_probe
            and score >= max(min_score, 0.35)
            and confidence >= 0.66
            and expected_net >= 0.25
            and profit_quality >= 0.20
            and quant_loss_probability <= 0.52
            and float(decision.position_size_pct or 0.0) <= 0.03
        )
        positive_expectancy = (
            score >= max(min_score - 0.15, 0.55)
            and confidence >= 0.68
            and expected_net >= 0.35
            and profit_quality >= 0.30
            and quant_loss_probability <= 0.58
            and tail_risk_score < 0.88
            and (aligned or entry_votes >= 1 or quant_probe_triggered)
        )
        roster_fill_quant = (
            roster_underfilled
            and roster_fill_probe
            and score >= max(min_score, 0.25)
            and confidence >= 0.66
            and expected_net >= 0.30
            and profit_quality >= 0.30
            and quant_loss_probability <= 0.62
            and float(decision.position_size_pct or 0.0) <= 0.025
        )
        if not (exceptional or strong_aligned or strong_quant or medium_quant or positive_expectancy or roster_fill_quant):
            return None
        signal_label = (
            "极强信号"
            if exceptional
            else "强量化探针"
            if strong_quant
            else "组合补齐探针"
            if roster_fill_quant
            else "中等量化探针"
            if medium_quant
            else "正期望信号"
            if positive_expectancy
            else "强信号"
        )
        return (
            f"{signal_label}即时执行：机会评分 {score:.2f} 高于门槛 {min_score:.2f}，"
            f"AI置信度 {confidence:.0%}，预期净收益 {expected_net:.2f}%，"
            f"净盈亏比 {profit_quality:.2f}。为避免等待整轮排序错过价格，"
            "该信号通过风控后会立即进入下单前检查。"
        )

    def _entry_wait_sort_reason(
        self,
        decision: DecisionOutput,
        *,
        rank: int | None = None,
        candidate_count: int | None = None,
    ) -> str:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
        score = self._safe_float(opportunity.get("score"), 0.0)
        min_score = self._safe_float(opportunity.get("min_score_required"), MIN_ENTRY_OPPORTUNITY_SCORE)
        expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        confidence = max(float(decision.confidence or 0.0), self._safe_float(opportunity.get("confidence"), 0.0))
        prefix = (
            f"已进入开仓执行检查，历史候选排名参考 {rank}/{candidate_count}。"
            if rank is not None and candidate_count is not None
            else "已进入开仓执行检查。"
        )
        return (
            f"{prefix}这不是执行失败；开仓信号不会再等待整轮排序，会直接进入价格偏移、异常插针、保证金和 OKX 提交检查。"
            f"当前机会评分 {score:.2f}，执行门槛 {min_score:.2f}，"
            f"AI置信度 {confidence:.0%}，预期净收益 {expected_net:.2f}%。"
            "如果检查期间行情变化过大，会放弃本轮信号，下一轮重新分析。"
        )

    def _entry_opportunity_gate_reason(self, decision: DecisionOutput) -> str | None:
        """Return only severe entry blockers.

        AI should own the trade/no-trade decision. Opportunity score, local
        models, direction competition, and contribution stats are sent to AI
        and stored as advisory evidence. At execution time they may reduce size
        or add warnings, but they should not frequently veto AI entries.
        """
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
        confidence = max(float(decision.confidence or 0.0), self._safe_float(opportunity.get("confidence"), 0.0))
        advisory_warnings = (
            list(opportunity.get("execution_advisory_warnings"))
            if isinstance(opportunity.get("execution_advisory_warnings"), list)
            else []
        )

        def add_advisory(reason: str, *, size_cap: float | None = None) -> None:
            item = {"reason": reason}
            if size_cap is not None:
                original_size = float(decision.position_size_pct or 0.0)
                if original_size > size_cap > 0:
                    decision.position_size_pct = size_cap
                    item["original_position_size_pct"] = round(original_size, 6)
                    item["adjusted_position_size_pct"] = round(size_cap, 6)
            advisory_warnings.append(item)
            opportunity["execution_advisory_mode"] = "ai_decision_primary"
            opportunity["execution_advisory_policy"] = (
                "AI 决定是否开仓；机会评分和本地量化模型在执行层只作为风险提示和仓位降级，"
                "不再因为普通评分不足频繁拦截。"
            )
            opportunity["execution_advisory_warnings"] = advisory_warnings
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw

        suspicious_reason = self._suspicious_new_symbol_reason(decision.symbol)
        if suspicious_reason:
            return suspicious_reason
        symbol_loss_cooldown_reason = self._entry_symbol_loss_cooldown_reason(decision)
        if symbol_loss_cooldown_reason:
            return symbol_loss_cooldown_reason
        post_crash_rebound_reason = self._post_crash_rebound_chase_short_reason(decision)
        if post_crash_rebound_reason:
            return post_crash_rebound_reason
        score = self._safe_float(opportunity.get("score"), float("nan"))
        min_score = self._safe_float(opportunity.get("min_score_required"), MIN_ENTRY_OPPORTUNITY_SCORE)
        if opportunity.get("historical_block"):
            opportunity["historical_block_applied_as_warning"] = True
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw
        if not math.isfinite(score):
            return "机会评分缺失或无效，本次不执行开仓。"
        expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality_ratio = self._safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        min_profit_quality_ratio = self._safe_float(
            opportunity.get("min_profit_quality_ratio_required"),
            ENTRY_MIN_NET_PROFIT_QUALITY_RATIO,
        )
        tail_risk_score = self._safe_float(opportunity.get("tail_risk_score"), 0.0)
        success_probability = self._safe_float(opportunity.get("success_probability"), 0.0)
        strong_support = bool(opportunity.get("ml_aligned") and opportunity.get("local_profit_aligned"))
        server_profit_conflict = bool(opportunity.get("server_profit_conflict"))
        same_side_loss_concentration = bool(opportunity.get("same_side_loss_concentration"))
        server_profit_expected = self._safe_float(
            opportunity.get("server_profit_expected_return_pct"),
            0.0,
        )
        server_profit_side = str(opportunity.get("server_profit_best_side") or "")
        quant_probe = raw.get("quant_profit_probe") if isinstance(raw.get("quant_profit_probe"), dict) else {}
        roster = opportunity.get("portfolio_roster") if isinstance(opportunity.get("portfolio_roster"), dict) else {}
        roster_underfilled = bool(roster.get("underfilled"))
        entry_side = "long" if decision.action == Action.LONG else "short" if decision.action == Action.SHORT else str(opportunity.get("side") or "")
        opposite_entry_side = "short" if entry_side == "long" else "long" if entry_side == "short" else ""
        direction_competition = (
            opportunity.get("direction_competition")
            if isinstance(opportunity.get("direction_competition"), dict)
            else {}
        )
        direction_preferred_side = str(
            opportunity.get("direction_preferred_side")
            or direction_competition.get("preferred_side")
            or ""
        ).lower()
        direction_gap = self._safe_float(direction_competition.get("score_gap"), 0.0)
        direction_side_score = self._safe_float(
            (direction_competition.get(entry_side) or {}).get("score")
            if isinstance(direction_competition.get(entry_side), dict)
            else opportunity.get("direction_side_score"),
            0.0,
        )
        direction_opposite_score = self._safe_float(
            (direction_competition.get(opposite_entry_side) or {}).get("score")
            if isinstance(direction_competition.get(opposite_entry_side), dict)
            else opportunity.get("direction_opposite_score"),
            0.0,
        )
        quant_profit_probe_entry = bool(
            quant_probe.get("triggered")
            and expected_net > 0
            and server_profit_expected >= (
                PORTFOLIO_ROSTER_FILL_MIN_EXPECTED_PCT
                if roster_underfilled
                else QUANT_PROFIT_PROBE_MIN_EXPECTED_PCT
            )
            and self._safe_float(opportunity.get("server_profit_loss_probability"), 1.0) < (
                PORTFOLIO_ROSTER_FILL_MAX_LOSS_PROBABILITY
                if roster_underfilled
                else 0.58
            )
            and tail_risk_score < 0.95
            and not same_side_loss_concentration
        )
        roster_fill_entry = bool(
            roster_underfilled
            and expected_net >= PORTFOLIO_ROSTER_FILL_MIN_NET_PCT
            and profit_quality_ratio >= PORTFOLIO_ROSTER_FILL_MIN_PROFIT_QUALITY_RATIO
            and success_probability >= 0.48
            and tail_risk_score < 0.90
            and not same_side_loss_concentration
            and (
                opportunity.get("local_profit_aligned")
                or opportunity.get("ml_aligned")
                or opportunity.get("timeseries_aligned")
                or quant_probe.get("triggered")
            )
            and score >= max(min_score - 0.35, 0.25)
        )
        direction_supported_by_symbol = bool(
            opportunity.get("local_profit_aligned")
            or opportunity.get("ml_aligned")
            or opportunity.get("timeseries_aligned")
            or opportunity.get("expert_aligned")
            or quant_probe.get("triggered")
            or direction_side_score >= ENTRY_DIRECTION_MIN_SUPPORT_SCORE
        )
        if (
            entry_side in {"long", "short"}
            and direction_preferred_side == opposite_entry_side
            and direction_gap >= ENTRY_DIRECTION_HARD_CONFLICT_GAP
            and not (
                server_profit_side == entry_side
                and server_profit_expected >= 0.45
                and profit_quality_ratio >= 0.55
            )
        ):
            if (
                expected_net > 0
                and score >= max(min_score - 0.60, 0.95)
                and confidence >= 0.72
                and success_probability >= 0.48
                and tail_risk_score < 0.85
            ):
                opportunity["direction_conflict_softened"] = {
                    "applied": True,
                    "preferred_side": direction_preferred_side,
                    "entry_side": entry_side,
                    "direction_gap": round(direction_gap, 6),
                    "reason": "AI 给出较高置信度且综合预期净收益为正，方向竞争冲突改为强风险提示，不再一票否决。",
                }
                raw["opportunity_score"] = opportunity
                decision.raw_response = raw
            else:
                side_label = "做多" if entry_side == "long" else "做空"
                opposite_label = "做多" if opposite_entry_side == "long" else "做空"
                add_advisory(
                    f"开仓方向预判不支持本次{side_label}：该币种 long/short 独立竞争更偏向{opposite_label}，"
                    f"方向分差 {direction_gap:.2f}，本方向分 {direction_side_score:.2f}，"
                    f"相反方向分 {direction_opposite_score:.2f}。本次作为风险提示并降低仓位，不再硬拦 AI 开仓。",
                    size_cap=0.025,
                )
        if (
            entry_side in {"long", "short"}
            and not direction_supported_by_symbol
            and direction_side_score < ENTRY_DIRECTION_MIN_SUPPORT_SCORE
        ):
            side_label = "做多" if entry_side == "long" else "做空"
            add_advisory(
                f"该币种本轮缺少提前支持{side_label}的方向证据：方向分 {direction_side_score:.2f}，"
                "未获得 ML、服务器盈利模型、时序模型或专家同向确认。"
                "本次作为仓位降级提示，不再硬拦 AI 开仓。",
                size_cap=0.02,
            )
        server_profit_conflict_relief = bool(
            server_profit_conflict
            and expected_net > 0
            and score >= max(min_score - 0.55, 0.95)
            and confidence >= 0.72
            and profit_quality_ratio >= 0.20
            and success_probability >= 0.48
            and tail_risk_score < 0.90
            and (
                opportunity.get("ml_aligned")
                or opportunity.get("expert_aligned")
                or direction_preferred_side == entry_side
                or direction_side_score > 0
            )
        )
        if server_profit_conflict_relief:
            original_size = float(decision.position_size_pct or 0.0)
            if original_size > 0:
                decision.position_size_pct = min(original_size, 0.025)
            opportunity["server_profit_conflict_relief"] = {
                "applied": True,
                "original_position_size_pct": round(original_size, 6),
                "adjusted_position_size_pct": round(float(decision.position_size_pct or 0.0), 6),
                "server_profit_expected_return_pct": round(server_profit_expected, 6),
                "expected_net_return_pct": round(expected_net, 6),
                "reason": "服务器盈利模型单项不支持，但综合预期净收益和 AI/方向证据达标，允许极小仓进入执行检查。",
            }
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw
        if server_profit_conflict and not server_profit_conflict_relief and not (
            server_profit_expected > 0
            and expected_net >= 1.20
            and profit_quality_ratio >= max(min_profit_quality_ratio, 1.35)
            and success_probability >= 0.58
            and (opportunity.get("ml_aligned") or opportunity.get("timeseries_aligned"))
        ):
            side_label = "做多" if opportunity.get("side") == "long" else "做空" if opportunity.get("side") == "short" else str(opportunity.get("side") or "当前方向")
            add_advisory(
                f"服务器盈利模型不支持本次{side_label}：该方向预期收益 {server_profit_expected:.4f}%，"
                f"模型更偏向 {server_profit_side or '观望/未知'}。本次作为极小仓风险提示，不再硬拦 AI 开仓。",
                size_cap=0.02,
            )
        contribution_adjustment = (
            opportunity.get("model_contribution_adjustment")
            if isinstance(opportunity.get("model_contribution_adjustment"), dict)
            else {}
        )
        contribution_hard_caution = bool(contribution_adjustment.get("hard_caution"))
        negative_sources = contribution_adjustment.get("negative_sources")
        negative_source_count = len(negative_sources) if isinstance(negative_sources, list) else 0
        contribution_positive_expected_relief = bool(contribution_hard_caution and expected_net > 0)
        if contribution_positive_expected_relief:
            opportunity["contribution_positive_expected_relief"] = {
                "applied": True,
                "negative_source_count": negative_source_count,
                "expected_net_return_pct": round(expected_net, 6),
                "profit_quality_ratio": round(profit_quality_ratio, 6),
                "reason": "闭环贡献统计仅作为风险提示；预期净收益为正时不再因为单个来源近期亏损而硬拦截。",
            }
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw
        if contribution_hard_caution and not contribution_positive_expected_relief and not (
            expected_net >= 0.80
            and profit_quality_ratio >= max(min_profit_quality_ratio, 1.25)
            and success_probability >= 0.56
            and (opportunity.get("local_profit_aligned") or opportunity.get("timeseries_aligned"))
        ) and not (
            roster_fill_entry
            and negative_source_count <= 1
            and expected_net >= 0.70
            and profit_quality_ratio >= 0.55
        ):
            add_advisory(
                f"闭环贡献统计显示本轮证据组合里有 {negative_source_count} 个来源最近真实净亏，"
                f"当前预期净收益 {expected_net:.2f}%、净盈亏比 {profit_quality_ratio:.2f} "
                "还不足以覆盖该负贡献风险。本次只降低仓位，不再硬拦 AI 开仓。",
                size_cap=0.02,
            )
        if expected_net <= 0:
            add_advisory(
                f"扣除手续费、滑点和尾部亏损风险后，预期净收益 {expected_net:.4f}% 不为正，"
                "本次按 AI 决策继续进入执行检查，但仓位降为极小仓。",
                size_cap=0.015,
            )
        if (
            profit_quality_ratio < min_profit_quality_ratio
            and not quant_profit_probe_entry
            and not roster_fill_entry
            and not contribution_positive_expected_relief
        ):
            add_advisory(
                f"净盈亏比 {profit_quality_ratio:.2f} 低于最低要求 {min_profit_quality_ratio:.2f}；"
                "这类机会容易形成小盈大亏，本次只降低仓位，不再硬拦 AI 开仓。",
                size_cap=0.02,
            )
        if opportunity.get("weak_history_requires_stronger_edge") and not (
            opportunity.get("local_profit_aligned") or opportunity.get("ml_aligned")
        ):
            add_advisory(
                "该币种/方向近期真实盈亏偏弱，且本轮没有本地盈利模型或 ML 同向改善证据；"
                "本次按 AI 决策进入执行检查，但降低仓位。",
                size_cap=0.02,
            )
        if tail_risk_score >= 1.15 and not (strong_support and profit_quality_ratio >= 0.55 and success_probability >= 0.58):
            return (
                f"尾部亏损风险 {tail_risk_score:.2f} 过高；"
                "这类机会可能胜率看起来不低，但单次亏损偏大，属于严重风险，本次不执行开仓。"
            )
        if tail_risk_score >= 0.95:
            add_advisory(
                f"尾部亏损风险 {tail_risk_score:.2f} 偏高，本次只降低仓位，不再硬拦 AI 开仓。",
                size_cap=0.02,
            )
        if (
            profit_quality_ratio <= 0.12
            and not strong_support
            and not quant_profit_probe_entry
            and not roster_fill_entry
            and not contribution_positive_expected_relief
        ):
            add_advisory(
                f"盈利质量比 {profit_quality_ratio:.2f} 过低；"
                "相对可能亏损，预期收益不够有吸引力。本次只降低仓位，不再硬拦 AI 开仓。",
                size_cap=0.015,
            )
        if (
            score <= min_score
            and not quant_profit_probe_entry
            and not roster_fill_entry
            and not contribution_positive_expected_relief
        ):
            add_advisory(
                f"机会评分 {score:.4f} 低于当前执行门槛 {min_score:.2f}；"
                "本次作为风险提示和仓位降级，不再硬拦 AI 开仓。",
                size_cap=0.02,
            )
        if roster_fill_entry:
            opportunity["portfolio_roster_fill_relief"] = {
                "applied": True,
                "target_position_groups": roster.get("target_position_groups"),
                "current_position_groups": roster.get("current_position_groups"),
                "gap": roster.get("gap"),
                "expected_net_return_pct": round(expected_net, 6),
                "profit_quality_ratio": round(profit_quality_ratio, 6),
                "negative_source_count": negative_source_count,
                "reason": "当前组合低于目标持仓数，且本信号仍为正期望，允许小仓补充组合分散度。",
            }
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw
        if quant_profit_probe_entry:
            opportunity["quant_probe_execution_relief"] = {
                "applied": True,
                "score": round(score, 6),
                "min_score_required": round(min_score, 6),
                "profit_quality_ratio": round(profit_quality_ratio, 6),
                "min_profit_quality_ratio": round(min_profit_quality_ratio, 6),
                "reason": "正期望小仓探针允许低评分/低净盈亏比进入执行检查，用真实成交结果训练后续筛选。",
            }
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw
        return None

    def _abnormal_wick_entry_guard_reason(self, decision: DecisionOutput) -> str | None:
        """Block new entries on symbols that recently printed extreme stop-loss wicks."""
        if not decision.is_entry:
            return None
        snapshot = decision.feature_snapshot if isinstance(decision.feature_snapshot, dict) else {}
        max_wick_pct = self._safe_float(snapshot.get("abnormal_wick_max_pct"), 0.0)
        wick_count = int(self._safe_float(snapshot.get("abnormal_wick_count_72h"), 0.0))
        recent_hours = self._safe_float(snapshot.get("abnormal_wick_recent_hours"), 9999.0)
        if (
            wick_count < ABNORMAL_WICK_ENTRY_BLOCK_MIN_COUNT
            or max_wick_pct < ABNORMAL_WICK_ENTRY_BLOCK_MAX_PCT
            or recent_hours > ABNORMAL_WICK_ENTRY_BLOCK_RECENT_HOURS
        ):
            return None

        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["abnormal_wick_guard"] = {
            "blocked": True,
            "count_72h": wick_count,
            "max_wick_pct": round(max_wick_pct, 4),
            "recent_hours": round(recent_hours, 4),
            "rule": "recent extreme wick can fill stops far from the planned stop price",
        }
        decision.raw_response = raw
        reason = (
            f"{decision.symbol} 最近 {recent_hours:.1f} 小时内出现过异常插针，"
            f"72 小时内共 {wick_count} 次，最大插针约 {max_wick_pct:.1f}%。"
            "这类币可能让止损按远离计划止损价的极端价成交，本次禁止新开仓，等待异常波动消退。"
        )
        self._remember_temporary_entry_block(
            decision.symbol,
            reason,
            max(PRICE_GUARD_ENTRY_BLOCK_MINUTES, 60.0),
        )
        return reason

    def _symbol_profit_quarantine_reason(
        self,
        symbol: str,
        strategy: dict[str, Any] | None,
    ) -> str | None:
        """Skip new AI entry analysis for symbols with proven recent negative expectancy."""
        if not isinstance(strategy, dict):
            return None
        profiles = strategy.get("symbol_side_performance")
        if not isinstance(profiles, dict):
            return None
        symbol_key = self._normalize_position_symbol(symbol) or symbol
        profile = profiles.get(f"{symbol_key}|all")
        if not isinstance(profile, dict) or not profile.get("cooldown"):
            return None
        pnl = self._safe_float(profile.get("pnl"), 0.0)
        losses = int(profile.get("losses") or 0)
        count = int(profile.get("count") or 0)
        reason = str(profile.get("cooldown_reason") or "recent realized PnL is poor")
        return (
            f"{symbol_key} 已进入盈利冷却：最近 {count} 笔真实平仓累计盈亏 {pnl:.2f} U，"
            f"亏损 {losses} 笔。原因：{reason}。在这段统计改善前，系统会跳过这个币种的新开仓分析。"
        )

    def _entry_loss_cooldown_override(
        self,
        decision: DecisionOutput,
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        """Allow strong profit-quality signals to override same-symbol loss cooldown."""
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
        score = self._safe_float(opportunity.get("score"), float("nan"))
        min_score = self._safe_float(opportunity.get("min_score_required"), MIN_ENTRY_OPPORTUNITY_SCORE)
        confidence = max(
            float(decision.confidence or 0.0),
            self._safe_float(opportunity.get("confidence"), 0.0),
        )
        expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = self._safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        reward_risk = self._safe_float(opportunity.get("reward_risk_ratio"), 0.0)
        server_expected = self._safe_float(opportunity.get("server_profit_expected_return_pct"), 0.0)
        server_loss_probability = self._safe_float(opportunity.get("server_profit_loss_probability"), 1.0)
        tail_risk = self._safe_float(opportunity.get("tail_risk_score"), 0.0)
        score_required = max(
            ENTRY_LOSS_COOLDOWN_OVERRIDE_MIN_SCORE,
            max(min_score, MIN_ENTRY_OPPORTUNITY_SCORE) * ENTRY_LOSS_COOLDOWN_OVERRIDE_SCORE_MULTIPLE,
        )
        aligned_sources = [
            name
            for name in ("ml_aligned", "local_profit_aligned", "timeseries_aligned")
            if bool(opportunity.get(name))
        ]
        source_support = (
            server_expected >= ENTRY_LOSS_COOLDOWN_OVERRIDE_MIN_SERVER_EXPECTED
            and server_loss_probability <= ENTRY_LOSS_COOLDOWN_OVERRIDE_MAX_LOSS_PROBABILITY
        ) or (
            len(aligned_sources) >= 2
            and server_loss_probability <= ENTRY_LOSS_COOLDOWN_OVERRIDE_MAX_LOSS_PROBABILITY
        )
        checks = {
            "confidence": confidence >= ENTRY_LOSS_COOLDOWN_OVERRIDE_MIN_CONFIDENCE,
            "score": math.isfinite(score) and score >= score_required,
            "expected_net": expected_net >= ENTRY_LOSS_COOLDOWN_OVERRIDE_MIN_EXPECTED_NET,
            "profit_quality": profit_quality >= ENTRY_LOSS_COOLDOWN_OVERRIDE_MIN_PROFIT_QUALITY,
            "reward_risk": reward_risk >= ENTRY_LOSS_COOLDOWN_OVERRIDE_MIN_REWARD_RISK,
            "tail_risk": tail_risk <= ENTRY_LOSS_COOLDOWN_OVERRIDE_MAX_TAIL_RISK,
            "source_support": source_support,
        }
        failed = [name for name, passed in checks.items() if not passed]
        metrics = {
            "confidence": round(confidence, 6),
            "score": round(score, 6) if math.isfinite(score) else None,
            "score_required": round(score_required, 6),
            "min_score_required": round(min_score, 6),
            "expected_net_return_pct": round(expected_net, 6),
            "profit_quality_ratio": round(profit_quality, 6),
            "reward_risk_ratio": round(reward_risk, 6),
            "server_profit_expected_return_pct": round(server_expected, 6),
            "server_profit_loss_probability": round(server_loss_probability, 6),
            "tail_risk_score": round(tail_risk, 6),
            "aligned_sources": aligned_sources,
            "profile_pnl": round(self._safe_float(profile.get("pnl"), 0.0), 6),
            "profile_today_pnl": round(self._safe_float(profile.get("today_pnl"), 0.0), 6),
            "profile_loss": round(self._safe_float(profile.get("loss"), 0.0), 6),
            "profile_losses": int(profile.get("losses") or 0),
        }
        allowed = not failed
        return {
            "allowed": allowed,
            "failed": failed,
            "checks": checks,
            "metrics": metrics,
            "summary": (
                "真实亏损冷却已被高质量信号解锁：AI置信度、机会评分、预期净收益、"
                "盈利质量和服务器盈利模型均达到放行条件。"
                if allowed
                else "真实亏损冷却未解锁：当前信号还没有强到足以覆盖近期真实亏损。"
            ),
        }

    def _entry_symbol_loss_cooldown_reason(self, decision: DecisionOutput) -> str | None:
        """Hard-block new entries when this exact symbol/side has recently lost real money."""
        if not decision.is_entry:
            return None
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
        side_profile = (
            opportunity.get("symbol_side_profile")
            if isinstance(opportunity.get("symbol_side_profile"), dict)
            else {}
        )
        side_label = "做多" if decision.action == Action.LONG else "做空" if decision.action == Action.SHORT else "当前方向"
        symbol_key = self._normalize_position_symbol(decision.symbol) or decision.symbol

        def profile_reason(profile: dict[str, Any], scope: str) -> str | None:
            if not isinstance(profile, dict) or not profile.get("cooldown"):
                return None
            pnl = self._safe_float(profile.get("pnl"), 0.0)
            today_pnl = self._safe_float(profile.get("today_pnl"), 0.0)
            loss = self._safe_float(profile.get("loss"), 0.0)
            today_loss = self._safe_float(profile.get("today_loss"), 0.0)
            largest_loss = self._safe_float(profile.get("largest_loss"), 0.0)
            cooldown_remaining_hours = self._safe_float(profile.get("cooldown_remaining_hours"), 0.0)
            count = int(profile.get("count") or 0)
            losses = int(profile.get("losses") or 0)
            wins = int(profile.get("wins") or 0)
            profit_factor = self._safe_float(profile.get("profit_factor"), 0.0)
            cooldown_reason = str(profile.get("cooldown_reason") or "近期真实平仓表现偏弱")
            scope_label = f"该币种{side_label}方向"
            override = self._entry_loss_cooldown_override(decision, profile)
            raw["loss_cooldown_override"] = override
            opportunity["loss_cooldown_override"] = {
                "allowed": bool(override.get("allowed")),
                "summary": str(override.get("summary") or ""),
                "metrics": override.get("metrics"),
                "failed": override.get("failed"),
            }
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw
            if override.get("allowed"):
                return None
            metrics = override.get("metrics") if isinstance(override.get("metrics"), dict) else {}
            failed_labels = {
                "confidence": "AI置信度",
                "score": "机会评分强度",
                "expected_net": "预期净收益",
                "profit_quality": "盈利质量",
                "reward_risk": "盈亏比",
                "tail_risk": "尾部风险",
                "source_support": "服务器/本地模型同向支持",
            }
            failed_text = "、".join(
                failed_labels.get(str(item), str(item))
                for item in (override.get("failed") or [])
            ) or "高质量解锁条件"
            return (
                f"{scope_label}已进入真实亏损冷却：最近 {count} 笔平仓累计 {pnl:.2f}U，"
                f"今日 {today_pnl:.2f}U，总亏损 {loss:.2f}U，今日亏损 {today_loss:.2f}U，"
                f"最大单笔亏损 {largest_loss:.2f}U，胜/负 {wins}/{losses}，"
                f"盈利因子 {profit_factor:.2f}。原因：{cooldown_reason}。"
                f"本次尝试用高质量信号解锁冷却，但 {failed_text} 未达标；"
                f"当前机会评分 {self._safe_float(metrics.get('score'), 0.0):.2f}/要求 {self._safe_float(metrics.get('score_required'), 0.0):.2f}，"
                f"置信度 {self._safe_float(metrics.get('confidence'), 0.0):.0%}，"
                f"预期净收益 {self._safe_float(metrics.get('expected_net_return_pct'), 0.0):.2f}%，"
                f"盈利质量 {self._safe_float(metrics.get('profit_quality_ratio'), 0.0):.2f}，"
                f"服务器预期 {self._safe_float(metrics.get('server_profit_expected_return_pct'), 0.0):.2f}%，"
                f"亏损概率 {self._safe_float(metrics.get('server_profit_loss_probability'), 1.0):.0%}。"
                f"为避免在 {symbol_key} 上连续同方向复亏，本次禁止{side_label}新开仓；"
                f"预计还需冷却约 {cooldown_remaining_hours:.1f} 小时，之后再按最新行情重新评估。"
            )

        side_reason = profile_reason(side_profile, "side")
        if side_reason:
            return side_reason
        return None

    def _post_crash_rebound_chase_short_reason(self, decision: DecisionOutput) -> str | None:
        """Block shorts when a crash trace is followed by a strong 1m rebound."""
        if decision.action != Action.SHORT:
            return None
        snapshot = decision.feature_snapshot if isinstance(decision.feature_snapshot, dict) else {}
        if not snapshot:
            return None
        returns_1 = self._safe_float(snapshot.get("returns_1"), 0.0)
        returns_5 = self._safe_float(snapshot.get("returns_5"), 0.0)
        returns_20 = self._safe_float(snapshot.get("returns_20"), 0.0)
        if not (
            returns_1 >= ENTRY_POST_CRASH_REBOUND_1M
            and (
                returns_5 <= ENTRY_POST_CRASH_REBOUND_5M_DROP
                or returns_20 <= ENTRY_POST_CRASH_REBOUND_20M_DROP
            )
        ):
            return None

        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
        expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = self._safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        raw["post_crash_rebound_guard"] = {
            "blocked": True,
            "action": decision.action.value,
            "returns_1": round(returns_1, 6),
            "returns_5": round(returns_5, 6),
            "returns_20": round(returns_20, 6),
            "expected_net_return_pct": round(expected_net, 6),
            "profit_quality_ratio": round(profit_quality, 6),
            "policy": "暴跌后1分钟强反弹时，不追空；先等待新一轮行情确认方向。",
        }
        decision.raw_response = raw
        return (
            f"暴跌后反弹保护：该币种刚经历短周期大跌，但最新1分钟已反弹 {returns_1 * 100:.2f}%，"
            f"5分钟/20分钟仍保留暴跌痕迹（5m {returns_5 * 100:.2f}%，20m {returns_20 * 100:.2f}%）。"
            "这类结构容易从插针低点快速反抽，系统不追空，等待下一轮新行情重新判断。"
        )

    async def _today_side_performance(self, mode: str) -> dict[str, dict[str, float]]:
        selected_mode = "live" if mode == "live" else "paper"
        now_local = datetime.now(timezone(timedelta(hours=8)))
        start_utc = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        result = {
            "long": {"count": 0, "wins": 0, "losses": 0, "pnl": 0.0, "profit": 0.0, "loss": 0.0},
            "short": {"count": 0, "wins": 0, "losses": 0, "pnl": 0.0, "profit": 0.0, "loss": 0.0},
        }
        try:
            async with get_session_ctx() as session:
                rows = await TradeRepository(session).get_position_records(
                    execution_mode=selected_mode,
                    model_name=ENSEMBLE_TRADER_NAME,
                    is_open=False,
                    limit=5000,
                )
                for pos in rows:
                    closed_at = pos.closed_at
                    if not closed_at:
                        continue
                    if closed_at.tzinfo is None:
                        closed_at = closed_at.replace(tzinfo=timezone.utc)
                    if closed_at < start_utc:
                        continue
                    side = "short" if str(pos.side or "").lower() == "short" else "long"
                    pnl = float(pos.realized_pnl or 0.0)
                    bucket = result[side]
                    bucket["count"] += 1
                    bucket["pnl"] += pnl
                    if pnl >= 0:
                        bucket["wins"] += 1
                        bucket["profit"] += pnl
                    else:
                        bucket["losses"] += 1
                        bucket["loss"] += abs(pnl)
        except Exception as e:
            logger.warning("failed to calculate today side performance", error=str(e))

        for bucket in result.values():
            count = max(bucket["count"], 1)
            bucket["avg_pnl"] = bucket["pnl"] / count
            bucket["win_rate"] = bucket["wins"] / count
            for key, value in list(bucket.items()):
                if isinstance(value, float):
                    bucket[key] = round(value, 6)
        return result

    async def _recent_symbol_side_performance(self, mode: str) -> dict[str, dict[str, Any]]:
        """Summarize realized PnL by symbol and side so entries follow what is actually profitable."""
        selected_mode = "live" if mode == "live" else "paper"
        now_utc = datetime.now(timezone.utc)
        today_start_utc = datetime.now(timezone(timedelta(hours=8))).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).astimezone(timezone.utc)
        window_start_utc = now_utc - timedelta(days=SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS)
        profiles: dict[str, dict[str, Any]] = {}

        def profile_for(key: str) -> dict[str, Any]:
            if key not in profiles:
                profiles[key] = {
                    "count": 0,
                    "wins": 0,
                    "losses": 0,
                    "pnl": 0.0,
                    "profit": 0.0,
                    "loss": 0.0,
                    "largest_loss": 0.0,
                    "today_count": 0,
                    "today_pnl": 0.0,
                    "today_loss": 0.0,
                    "first_closed_at": None,
                    "last_closed_at": None,
                    "last_loss_at": None,
                }
            return profiles[key]

        try:
            async with get_session_ctx() as session:
                rows = await TradeRepository(session).get_position_records(
                    execution_mode=selected_mode,
                    model_name=ENSEMBLE_TRADER_NAME,
                    is_open=False,
                    limit=SYMBOL_SIDE_PROFILE_LOOKBACK,
                )
                for pos in rows:
                    closed_at = pos.closed_at
                    if not closed_at:
                        continue
                    if closed_at.tzinfo is None:
                        closed_at = closed_at.replace(tzinfo=timezone.utc)
                    if closed_at < window_start_utc:
                        continue
                    symbol = self._normalize_position_symbol(pos.symbol) or str(pos.symbol or "")
                    if not symbol:
                        continue
                    side = "short" if str(pos.side or "").lower() == "short" else "long"
                    pnl = float(pos.realized_pnl or 0.0)
                    for key in (f"{symbol}|{side}", f"{symbol}|all"):
                        bucket = profile_for(key)
                        closed_iso = closed_at.isoformat()
                        bucket["count"] += 1
                        bucket["pnl"] += pnl
                        first_closed_at = bucket.get("first_closed_at")
                        last_closed_at = bucket.get("last_closed_at")
                        if not first_closed_at or closed_iso < str(first_closed_at):
                            bucket["first_closed_at"] = closed_iso
                        if not last_closed_at or closed_iso > str(last_closed_at):
                            bucket["last_closed_at"] = closed_iso
                        if closed_at >= today_start_utc:
                            bucket["today_count"] += 1
                            bucket["today_pnl"] += pnl
                            if pnl < 0:
                                bucket["today_loss"] += abs(pnl)
                        if pnl >= 0:
                            bucket["wins"] += 1
                            bucket["profit"] += pnl
                        else:
                            bucket["losses"] += 1
                            bucket["loss"] += abs(pnl)
                            bucket["largest_loss"] = min(float(bucket.get("largest_loss") or 0.0), pnl)
                            last_loss_at = bucket.get("last_loss_at")
                            if not last_loss_at or closed_iso > str(last_loss_at):
                                bucket["last_loss_at"] = closed_iso
        except Exception as e:
            logger.warning("failed to calculate symbol side performance", error=str(e))
            return {}

        for key, bucket in profiles.items():
            count = max(int(bucket.get("count") or 0), 1)
            profit = float(bucket.get("profit") or 0.0)
            loss = float(bucket.get("loss") or 0.0)
            pnl = float(bucket.get("pnl") or 0.0)
            losses = int(bucket.get("losses") or 0)
            today_pnl = float(bucket.get("today_pnl") or 0.0)
            today_loss = float(bucket.get("today_loss") or 0.0)
            last_loss_at = bucket.get("last_loss_at")
            last_loss_age_hours = 9999.0
            if last_loss_at:
                try:
                    parsed_last_loss_at = datetime.fromisoformat(str(last_loss_at))
                    if parsed_last_loss_at.tzinfo is None:
                        parsed_last_loss_at = parsed_last_loss_at.replace(tzinfo=timezone.utc)
                    last_loss_age_hours = max(
                        (now_utc - parsed_last_loss_at.astimezone(timezone.utc)).total_seconds() / 3600.0,
                        0.0,
                    )
                except Exception:
                    last_loss_age_hours = 0.0
            recent_loss_cooldown_active = last_loss_age_hours <= SYMBOL_SIDE_HARD_COOLDOWN_HOURS
            is_symbol_profile = key.endswith("|all")
            cooldown = False
            cooldown_reason = ""
            if (
                is_symbol_profile
                and losses >= SYMBOL_QUARANTINE_MIN_LOSSES
                and pnl <= -SYMBOL_QUARANTINE_LOSS_USDT
            ):
                cooldown = True
                cooldown_reason = "该币种最近滚动真实亏损过大"
            elif is_symbol_profile and (
                today_loss >= SYMBOL_TOTAL_COOLDOWN_LOSS_USDT
                or today_pnl <= -SYMBOL_TOTAL_COOLDOWN_LOSS_USDT
            ):
                cooldown = True
                cooldown_reason = "该币种今天累计真实亏损超过限制"
            elif not is_symbol_profile and recent_loss_cooldown_active and (
                loss >= SYMBOL_TOTAL_COOLDOWN_LOSS_USDT
                or pnl <= -SYMBOL_TOTAL_COOLDOWN_LOSS_USDT
                or today_loss >= SYMBOL_SIDE_COOLDOWN_LOSS_USDT
                or today_pnl <= -SYMBOL_SIDE_COOLDOWN_LOSS_USDT
            ):
                cooldown = True
                cooldown_reason = "该币种这个方向的真实亏损已经超过限制"
            elif not is_symbol_profile and recent_loss_cooldown_active and (
                pnl <= -SYMBOL_SIDE_COOLDOWN_LOSS_USDT
                or (losses >= 2 and pnl < 0 and loss > profit * 1.2)
            ):
                cooldown = True
                cooldown_reason = "该币种这个方向近期真实盈亏表现偏弱"
            cooldown_remaining_hours = (
                max(SYMBOL_SIDE_HARD_COOLDOWN_HOURS - last_loss_age_hours, 0.0)
                if cooldown and not is_symbol_profile
                else 0.0
            )
            bucket.update({
                "avg_pnl": round(pnl / count, 6),
                "win_rate": round(float(bucket.get("wins") or 0) / count, 6),
                "profit_factor": round(profit / loss, 6) if loss > 0 else (999.0 if profit > 0 else 0.0),
                "cooldown": cooldown,
                "cooldown_reason": cooldown_reason,
                "last_loss_age_hours": round(last_loss_age_hours, 6),
                "cooldown_remaining_hours": round(cooldown_remaining_hours, 6),
                "lookback_days": SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS,
                "age_seconds": round((now_utc - today_start_utc).total_seconds(), 3),
            })
            for field in ("pnl", "profit", "loss", "largest_loss", "today_pnl", "today_loss"):
                bucket[field] = round(float(bucket.get(field) or 0.0), 6)
        return profiles

    async def _recent_model_contribution_performance(self, mode: str) -> dict[str, dict[str, Any]]:
        """Measure which evidence sources actually led to realized profit.

        This closes the feedback loop: model evidence is not only displayed in
        analysis records; its recent realized PnL changes the next entry score.
        """
        selected_mode = "live" if mode == "live" else "paper"
        now = datetime.now(timezone.utc)
        expires_at = self._model_contribution_cache.get("expires_at")
        cached_stats = self._model_contribution_cache.get("stats")
        if (
            isinstance(expires_at, datetime)
            and expires_at > now
            and isinstance(cached_stats, dict)
        ):
            return cached_stats

        start_utc = now - timedelta(days=SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS)
        stats: dict[str, dict[str, Any]] = {
            "ml_profit_model": self._empty_contribution_bucket("本地 ML 盈利模型"),
            "server_profit_model": self._empty_contribution_bucket("服务器盈利模型"),
            "timeseries_model": self._empty_contribution_bucket("时序预测模型"),
            "expert_alignment": self._empty_contribution_bucket("专家一致信号"),
            "ai_only_without_quant": self._empty_contribution_bucket("AI 单独支持但量化未同向"),
        }

        try:
            async with get_session_ctx() as session:
                positions_result = await session.execute(
                    select(Position)
                    .where(
                        Position.model_name == ENSEMBLE_TRADER_NAME,
                        Position.execution_mode == selected_mode,
                        Position.is_open.is_(False),
                        Position.closed_at.is_not(None),
                        Position.closed_at >= start_utc,
                    )
                    .order_by(Position.closed_at.desc())
                    .limit(800)
                )
                positions = list(positions_result.scalars().all())
                if not positions:
                    self._model_contribution_cache = {
                        "expires_at": now + timedelta(minutes=15),
                        "stats": stats,
                    }
                    return stats

                symbols = {p.symbol for p in positions if p.symbol}
                orders_result = await session.execute(
                    select(Order)
                    .where(
                        Order.model_name == ENSEMBLE_TRADER_NAME,
                        Order.execution_mode == selected_mode,
                        Order.status == "filled",
                        Order.decision_id.is_not(None),
                        Order.symbol.in_(symbols) if symbols else Order.id == -1,
                    )
                    .order_by(Order.filled_at.desc(), Order.created_at.desc())
                    .limit(3000)
                )
                orders = list(orders_result.scalars().all())
                decision_ids = [o.decision_id for o in orders if o.decision_id]
                decisions: dict[int, AIDecision] = {}
                if decision_ids:
                    decisions_result = await session.execute(
                        select(AIDecision).where(AIDecision.id.in_(decision_ids))
                    )
                    decisions = {d.id: d for d in decisions_result.scalars().all()}
        except Exception as exc:
            logger.warning("failed to calculate model contribution performance", error=str(exc))
            return {}

        def aware(value):
            if value and value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        for pos in positions:
            pos_created = aware(pos.created_at)
            pos_side = "short" if str(pos.side or "").lower() == "short" else "long"
            matched_decision = None
            best_delta = None
            for order in orders:
                if order.symbol != pos.symbol or order.decision_id not in decisions:
                    continue
                decision = decisions[order.decision_id]
                action = str(decision.action or "").lower()
                if action not in {"long", "short"} or action != pos_side:
                    continue
                order_time = aware(order.filled_at or order.created_at)
                if pos_created and order_time:
                    delta = abs((order_time - pos_created).total_seconds())
                    if delta > 300:
                        continue
                else:
                    delta = 0.0
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    matched_decision = decision
            if matched_decision is None:
                continue

            raw = matched_decision.raw_llm_response if isinstance(matched_decision.raw_llm_response, dict) else {}
            opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
            pnl = float(pos.realized_pnl or 0.0)
            contributors = self._decision_contribution_sources(opportunity, raw, pos_side)
            for source in contributors:
                self._add_contribution_sample(stats[source], pnl)

        for source, bucket in stats.items():
            self._finalize_contribution_bucket(bucket)

        self._model_contribution_cache = {
            "expires_at": now + timedelta(minutes=10),
            "stats": stats,
        }
        return stats

    def _empty_contribution_bucket(self, label: str) -> dict[str, Any]:
        return {
            "label": label,
            "count": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "profit": 0.0,
            "loss": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "score_multiplier": 1.0,
            "size_multiplier": 1.0,
            "state": "learning",
            "reason": "样本不足，先学习不强干预。",
        }

    def _add_contribution_sample(self, bucket: dict[str, Any], pnl: float) -> None:
        bucket["count"] = int(bucket.get("count") or 0) + 1
        bucket["pnl"] = float(bucket.get("pnl") or 0.0) + pnl
        if pnl >= 0:
            bucket["wins"] = int(bucket.get("wins") or 0) + 1
            bucket["profit"] = float(bucket.get("profit") or 0.0) + pnl
        else:
            bucket["losses"] = int(bucket.get("losses") or 0) + 1
            bucket["loss"] = float(bucket.get("loss") or 0.0) + abs(pnl)

    def _finalize_contribution_bucket(self, bucket: dict[str, Any]) -> None:
        count = int(bucket.get("count") or 0)
        profit = float(bucket.get("profit") or 0.0)
        loss = float(bucket.get("loss") or 0.0)
        pnl = float(bucket.get("pnl") or 0.0)
        if count <= 0:
            return
        win_rate = int(bucket.get("wins") or 0) / count
        profit_factor = profit / loss if loss > 0 else (3.0 if profit > 0 else 0.0)
        avg_pnl = pnl / count
        edge = max(min(avg_pnl / 5.0, 0.28), -0.34)
        factor_edge = max(min((profit_factor - 1.0) * 0.14, 0.22), -0.26)
        win_edge = max(min((win_rate - 0.5) * 0.10, 0.05), -0.05)
        multiplier = min(max(1.0 + edge + factor_edge + win_edge, 0.60), 1.38)
        state = "learning"
        reason = "样本不足，先学习不强干预。"
        if count >= 5:
            if pnl > 0 and profit_factor >= 1.15:
                state = "promote"
                reason = f"最近 {count} 笔贡献净盈利 {pnl:.2f}U，盈利因子 {profit_factor:.2f}，下轮提高权重。"
            elif pnl < 0 or profit_factor < 0.85:
                state = "degrade"
                reason = f"最近 {count} 笔贡献净亏损 {pnl:.2f}U，盈利因子 {profit_factor:.2f}，下轮降低权重。"
            else:
                state = "neutral"
                reason = f"最近 {count} 笔贡献接近中性，保持基础权重。"
        bucket.update({
            "pnl": round(pnl, 6),
            "profit": round(profit, 6),
            "loss": round(loss, 6),
            "avg_pnl": round(avg_pnl, 6),
            "win_rate": round(win_rate, 6),
            "profit_factor": round(profit_factor, 6),
            "score_multiplier": round(multiplier, 6),
            "size_multiplier": round(min(max(multiplier, 0.65), 1.25), 6),
            "state": state,
            "reason": reason,
        })

    def _decision_contribution_sources(
        self,
        opportunity: dict[str, Any],
        raw: dict[str, Any],
        side: str,
    ) -> list[str]:
        sources: list[str] = []
        if bool(opportunity.get("ml_aligned")):
            sources.append("ml_profit_model")
        if bool(opportunity.get("local_profit_aligned")):
            sources.append("server_profit_model")
        if bool(opportunity.get("timeseries_aligned")):
            sources.append("timeseries_model")
        if bool(opportunity.get("expert_aligned")):
            sources.append("expert_alignment")
        has_quant = any(source in sources for source in ("ml_profit_model", "server_profit_model", "timeseries_model"))
        if not has_quant and side in {"long", "short"}:
            sources.append("ai_only_without_quant")
        return sources

    def _model_contribution_score_adjustment(
        self,
        sources: list[str],
        performance: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if not sources or not isinstance(performance, dict):
            return {
                "active": False,
                "sources": sources or [],
                "score_multiplier": 1.0,
                "size_multiplier": 1.0,
                "score_adjustment": 0.0,
                "reason": "暂无模型贡献统计，使用基础机会评分。",
            }

        weighted_score_multiplier = 0.0
        weighted_size_multiplier = 0.0
        total_weight = 0.0
        evidence: list[dict[str, Any]] = []
        for source in sources:
            bucket = performance.get(source)
            if not isinstance(bucket, dict):
                continue
            count = int(bucket.get("count") or 0)
            if count <= 0:
                continue
            sample_weight = min(max(count, 1), 25)
            score_multiplier = self._safe_float(bucket.get("score_multiplier"), 1.0)
            size_multiplier = self._safe_float(bucket.get("size_multiplier"), 1.0)
            weighted_score_multiplier += score_multiplier * sample_weight
            weighted_size_multiplier += size_multiplier * sample_weight
            total_weight += sample_weight
            evidence.append({
                "source": source,
                "label": bucket.get("label") or source,
                "count": count,
                "pnl": bucket.get("pnl", 0.0),
                "profit_factor": bucket.get("profit_factor", 0.0),
                "state": bucket.get("state", "learning"),
                "score_multiplier": round(score_multiplier, 6),
                "size_multiplier": round(size_multiplier, 6),
                "reason": bucket.get("reason", ""),
            })

        if total_weight <= 0:
            return {
                "active": False,
                "sources": sources,
                "score_multiplier": 1.0,
                "size_multiplier": 1.0,
                "score_adjustment": 0.0,
                "evidence": evidence,
                "reason": "贡献样本不足，先学习不强干预。",
            }

        score_multiplier = min(max(weighted_score_multiplier / total_weight, 0.60), 1.38)
        size_multiplier = min(max(weighted_size_multiplier / total_weight, 0.65), 1.25)
        score_adjustment = (score_multiplier - 1.0) * 2.25
        state = "promote" if score_multiplier > 1.04 else "degrade" if score_multiplier < 0.96 else "neutral"
        negative_sources = [
            item for item in evidence
            if self._safe_float(item.get("pnl"), 0.0) < -8.0
            and self._safe_float(item.get("profit_factor"), 1.0) < 0.75
            and int(item.get("count") or 0) >= 5
        ]
        hard_caution = bool(negative_sources)
        if state == "promote":
            reason = "这些证据来源最近真实平仓贡献为正，本轮提高机会评分和仓位倾向。"
        elif state == "degrade":
            reason = "这些证据来源最近真实平仓贡献偏弱，本轮降低机会评分并缩小仓位。"
        else:
            reason = "这些证据来源最近真实贡献接近中性，本轮保持基础评分。"
        if hard_caution:
            reason = (
                f"{reason} 其中 {len(negative_sources)} 个证据来源最近真实净亏且盈利因子偏低，"
                "本轮进入闭环强审查。"
            )

        return {
            "active": True,
            "sources": sources,
            "state": state,
            "hard_caution": hard_caution,
            "negative_sources": negative_sources,
            "score_multiplier": round(score_multiplier, 6),
            "size_multiplier": round(size_multiplier, 6),
            "score_adjustment": round(score_adjustment, 6),
            "evidence": evidence,
            "reason": reason,
        }

    def _is_auto_tradeable_feature(self, fv: Any) -> bool:
        """Filter out low-participation symbols before spending AI tokens."""
        try:
            symbol = str(getattr(fv, "symbol", "") or "").upper()
            if self._suspicious_new_symbol_reason(symbol):
                return False
            current_price = float(getattr(fv, "current_price", 0) or getattr(fv, "close", 0) or 0)
            volume_24h = float(getattr(fv, "volume_24h", 0) or 0)
            volume_ratio = float(getattr(fv, "volume_ratio", 0) or 0)
            volatility_20 = float(getattr(fv, "volatility_20", 0) or 0)
            change_24h = abs(float(getattr(fv, "change_24h_pct", 0) or 0))
            adx_14 = float(getattr(fv, "adx_14", 0) or 0)
            abnormal_wick_count = int(float(getattr(fv, "abnormal_wick_count_72h", 0) or 0))
            abnormal_wick_max_pct = float(getattr(fv, "abnormal_wick_max_pct", 0) or 0)
            abnormal_wick_recent_hours = float(getattr(fv, "abnormal_wick_recent_hours", 9999) or 9999)
        except (TypeError, ValueError):
            return False

        if (
            abnormal_wick_count >= ABNORMAL_WICK_ENTRY_BLOCK_MIN_COUNT
            and abnormal_wick_max_pct >= ABNORMAL_WICK_ENTRY_BLOCK_MAX_PCT
            and abnormal_wick_recent_hours <= ABNORMAL_WICK_ENTRY_BLOCK_RECENT_HOURS
        ):
            return False
        notional_24h = current_price * volume_24h
        majors = ALT_LONG_ALLOWED_SYMBOLS
        min_notional = 800_000.0 if symbol in majors else 1_200_000.0
        analysis_volume_floor = max(
            min(max(float(settings.min_entry_volume_ratio or 0.0), 0.16) * 0.55, 0.42),
            0.18,
        )
        analysis_adx_floor = max(
            min(max(float(settings.min_entry_adx or 0.0) - 6.0, 8.0), 16.0),
            10.0,
        )
        if volume_ratio < analysis_volume_floor:
            return False
        if notional_24h < min_notional:
            return False
        if volatility_20 > 0.12:
            return False
        if change_24h > 22.0:
            return False
        if symbol not in majors and adx_14 < analysis_adx_floor:
            return False
        return True

    def _is_auto_analysis_candidate_feature(self, fv: Any) -> bool:
        """Secondary filter used to keep the scan diversified when hard filters are too strict."""
        try:
            symbol = str(getattr(fv, "symbol", "") or "").upper()
            if self._suspicious_new_symbol_reason(symbol):
                return False
            current_price = float(getattr(fv, "current_price", 0) or getattr(fv, "close", 0) or 0)
            volume_24h = float(getattr(fv, "volume_24h", 0) or 0)
            volume_ratio = float(getattr(fv, "volume_ratio", 0) or 0)
            volatility_20 = float(getattr(fv, "volatility_20", 0) or 0)
            change_24h = abs(float(getattr(fv, "change_24h_pct", 0) or 0))
            adx_14 = float(getattr(fv, "adx_14", 0) or 0)
            abnormal_wick_count = int(float(getattr(fv, "abnormal_wick_count_72h", 0) or 0))
            abnormal_wick_max_pct = float(getattr(fv, "abnormal_wick_max_pct", 0) or 0)
            abnormal_wick_recent_hours = float(getattr(fv, "abnormal_wick_recent_hours", 9999) or 9999)
        except (TypeError, ValueError):
            return False

        if (
            abnormal_wick_count >= ABNORMAL_WICK_ENTRY_BLOCK_MIN_COUNT
            and abnormal_wick_max_pct >= ABNORMAL_WICK_ENTRY_BLOCK_MAX_PCT
            and abnormal_wick_recent_hours <= ABNORMAL_WICK_ENTRY_BLOCK_RECENT_HOURS
        ):
            return False
        notional_24h = current_price * volume_24h
        majors = ALT_LONG_ALLOWED_SYMBOLS
        min_notional = 500_000.0 if symbol in majors else 700_000.0
        soft_volume_floor = max(
            min(max(float(settings.min_entry_volume_ratio or 0.0), 0.12) * 0.25, 0.24),
            0.05,
        )
        soft_adx_floor = max(
            min(max(float(settings.min_entry_adx or 0.0) - 9.0, 6.0), 14.0),
            8.0,
        )
        if volume_ratio < soft_volume_floor:
            return False
        if notional_24h < min_notional:
            return False
        if volatility_20 > 0.18:
            return False
        if change_24h > 32.0:
            return False
        if symbol not in majors and adx_14 < soft_adx_floor:
            return False
        return True

    def _rank_auto_feature_vectors(
        self,
        feature_vectors: dict[str, Any],
        limit: int,
    ) -> dict[str, Any]:
        all_items = list(feature_vectors.items())
        tradable_items = [
            item for item in feature_vectors.items()
            if self._is_auto_tradeable_feature(item[1])
        ]
        soft_items = [
            item for item in all_items
            if item not in tradable_items and self._is_auto_analysis_candidate_feature(item[1])
        ]

        def ranking_score(item: tuple[str, Any]) -> float:
            symbol, fv = item
            return (
                self._feature_opportunity_score(fv)
                - self._recent_market_hold_penalty(symbol)
                - self._recent_market_analysis_penalty(symbol)
                - self._no_opportunity_rotation_penalty(symbol, fv)
            )

        tradable_symbols = {symbol for symbol, _ in tradable_items}
        ranked_tradable = sorted(
            tradable_items,
            key=lambda item: (
                ranking_score(item),
            ),
            reverse=True,
        )
        ranked_soft = sorted(
            soft_items,
            key=ranking_score,
            reverse=True,
        )

        selected_items = list(ranked_tradable[:limit])
        if len(selected_items) < limit:
            selected_items.extend(ranked_soft[: max(limit - len(selected_items), 0)])
        if not selected_items:
            selected_items = sorted(all_items, key=ranking_score, reverse=True)[:limit]

        selected = dict(selected_items)
        logger.info(
            "auto opportunity shortlist",
            selected=len(selected),
            candidates=len(feature_vectors),
            tradable_candidates=len(tradable_items),
            secondary_candidates=len(soft_items),
            symbols=[
                {
                    "symbol": symbol,
                    "score": round(self._feature_opportunity_score(fv), 2),
                    "recent_hold_penalty": round(self._recent_market_hold_penalty(symbol), 2),
                    "recent_analysis_penalty": round(self._recent_market_analysis_penalty(symbol), 2),
                    "rotation_penalty": round(self._no_opportunity_rotation_penalty(symbol, fv), 2),
                    "selection_tier": (
                        "hard_filter"
                        if symbol in tradable_symbols
                        else "secondary_fill"
                    ),
                    "volume_ratio": round(float(getattr(fv, "volume_ratio", 0) or 0), 2),
                    "adx": round(float(getattr(fv, "adx_14", 0) or 0), 1),
                    "change_24h": round(float(getattr(fv, "change_24h_pct", 0) or 0), 2),
                }
                for symbol, fv in selected_items[: min(8, len(selected_items))]
            ],
        )
        return selected

    def _is_untradable_exchange_error(self, text: Any) -> bool:
        value = str(text or "").lower()
        return (
            "51155" in value
            or "can't trade this pair" in value
            or "local compliance restrictions" in value
        )

    def _is_transient_entry_exchange_error(self, text: Any) -> bool:
        value = str(text or "").lower()
        return (
            "51290" in value
            or "trading bot engine currently upgrading" in value
            or "engine currently upgrading" in value
            or "open interest" in value and "platform" in value and "limit" in value
            or "has reached the platform's limit" in value
            or ("try again later" in value and "okx" in value)
        )

    def _transient_entry_block_minutes(self, text: Any) -> float:
        value = str(text or "").lower()
        if (
            "open interest" in value
            and "platform" in value
            and "limit" in value
        ) or "has reached the platform's limit" in value:
            return 45.0
        return TRANSIENT_ENTRY_BLOCK_MINUTES

    def _is_entry_price_guard_skip(self, text: Any) -> bool:
        value = str(text or "")
        return (
            "下单前价格" in value
            or "避免追高" in value
            or "避免追空" in value
            or "行情变化太快" in value
        )

    def _remember_temporary_entry_block(
        self,
        symbol: str | None,
        reason: Any,
        minutes: float = TRANSIENT_ENTRY_BLOCK_MINUTES,
    ) -> None:
        normalized = self._normalize_position_symbol(symbol)
        if not normalized:
            return
        until = datetime.now(timezone.utc) + timedelta(minutes=max(float(minutes or 0), 1.0))
        self._untradable_symbols[normalized] = {
            "until": until,
            "reason": f"临时跳过新开仓：{str(sanitize_text(reason) or '近期该币种开仓未成功')[:460]}",
        }
        logger.warning(
            "symbol temporarily blocked for new entries",
            symbol=normalized,
            until=until.isoformat(),
            reason=str(reason or "")[:220],
        )

    def _remember_untradable_symbol(
        self,
        symbol: str | None,
        reason: Any,
        hours: float = UNTRADABLE_SYMBOL_BLOCK_HOURS,
    ) -> None:
        normalized = self._normalize_position_symbol(symbol)
        if not normalized:
            return
        until = datetime.now(timezone.utc) + timedelta(hours=max(float(hours or 0), 1.0))
        self._untradable_symbols[normalized] = {
            "until": until,
            "reason": str(sanitize_text(reason) or "OKX 提示该交易对当前不可交易")[:500],
        }
        logger.warning(
            "symbol temporarily blocked as untradable",
            symbol=normalized,
            until=until.isoformat(),
        )

    def _blocked_symbol_reason(self, symbol: str | None) -> str | None:
        normalized = self._normalize_position_symbol(symbol)
        if not normalized:
            return None
        item = self._untradable_symbols.get(normalized)
        if not item:
            return None
        until = item.get("until")
        if isinstance(until, datetime) and until > datetime.now(timezone.utc):
            return str(sanitize_text(item.get("reason")) or "该交易对暂时不可交易")
        self._untradable_symbols.pop(normalized, None)
        return None

    def _dedupe_symbols(self, symbols: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for symbol in symbols or []:
            normalized = self._normalize_position_symbol(symbol)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    def _open_position_symbol_keys(self, open_positions: list[dict] | None) -> set[str]:
        return {
            normalized
            for normalized in (
                self._normalize_position_symbol(p.get("symbol"))
                for p in (open_positions or [])
                if p.get("symbol")
            )
            if normalized
        }

    def _open_position_group_count(self, open_positions: list[dict] | None) -> int:
        groups = {
            (
                str(p.get("model_name") or ENSEMBLE_TRADER_NAME),
                self._normalize_position_symbol(p.get("symbol")),
                str(p.get("side") or "").lower(),
            )
            for p in (open_positions or [])
            if p.get("is_open", True)
            and self._normalize_position_symbol(p.get("symbol"))
            and str(p.get("side") or "").lower() in {"long", "short"}
        }
        return len(groups)

    def _filter_open_position_market_symbols(
        self,
        symbols: list[str],
        open_positions: list[dict] | None,
    ) -> list[str]:
        open_symbol_keys = self._open_position_symbol_keys(open_positions)
        if not open_symbol_keys:
            return list(symbols or [])

        filtered: list[str] = []
        skipped: list[str] = []
        for symbol in symbols or []:
            normalized = self._normalize_position_symbol(symbol)
            if normalized in open_symbol_keys:
                skipped.append(normalized)
                continue
            filtered.append(symbol)

        if skipped:
            logger.info(
                "skipping open-position symbols from market analysis",
                count=len(skipped),
                symbols=skipped[:10],
            )
        return filtered

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

    async def _filter_unclaimed_market_symbols(self, symbols: list[str]) -> list[str]:
        filtered: list[str] = []
        skipped: list[str] = []
        async with self._analysis_symbol_lock:
            active = set(self._active_analysis_symbols)
        for symbol in symbols or []:
            normalized = self._normalize_position_symbol(symbol)
            if normalized in active:
                skipped.append(normalized)
                continue
            filtered.append(symbol)
        if skipped:
            logger.info(
                "skipping symbols already under position/market analysis",
                count=len(skipped),
                symbols=skipped[:10],
            )
        return filtered

    def _filter_blocked_new_symbols(
        self,
        symbols: list[str],
        open_positions: list[dict],
        results: dict[str, Any],
    ) -> list[str]:
        open_symbol_keys = {
            self._normalize_position_symbol(p.get("symbol"))
            for p in (open_positions or [])
            if p.get("symbol")
        }
        filtered: list[str] = []
        skipped: list[dict[str, str]] = []
        for symbol in symbols:
            normalized = self._normalize_position_symbol(symbol)
            reason = self._suspicious_new_symbol_reason(normalized) or self._blocked_symbol_reason(normalized)
            if reason and normalized not in open_symbol_keys:
                skipped.append({"symbol": normalized, "reason": reason})
                continue
            filtered.append(symbol)
        if skipped:
            logger.info(
                "skipping blocked symbols before AI analysis",
                count=len(skipped),
                symbols=[s["symbol"] for s in skipped[:10]],
            )
            for item in skipped[:20]:
                results["warnings"].append({
                    "model": ENSEMBLE_TRADER_NAME,
                    "symbol": item["symbol"],
                    "warning": f"Symbol is temporarily skipped for new entry analysis: {item['reason']}",
                })
        return filtered

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
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    recent = not created_at or (
                        datetime.now(timezone.utc) - created_at
                    ) <= timedelta(hours=UNTRADABLE_SYMBOL_BLOCK_HOURS)
                    if recent and self._is_untradable_exchange_error(reason):
                        self._remember_untradable_symbol(row.symbol, reason)
                    elif (
                        created_at
                        and datetime.now(timezone.utc) - created_at
                        <= timedelta(minutes=TRANSIENT_ENTRY_BLOCK_MINUTES)
                        and self._is_transient_entry_exchange_error(reason)
                    ):
                        self._remember_temporary_entry_block(
                            row.symbol,
                            reason,
                            TRANSIENT_ENTRY_BLOCK_MINUTES,
                        )
                    elif (
                        created_at
                        and datetime.now(timezone.utc) - created_at
                        <= timedelta(minutes=PRICE_GUARD_ENTRY_BLOCK_MINUTES)
                        and self._is_entry_price_guard_skip(reason)
                    ):
                        self._remember_temporary_entry_block(
                            row.symbol,
                            reason,
                            PRICE_GUARD_ENTRY_BLOCK_MINUTES,
                        )
        except Exception as e:
            logger.warning("failed to load untradable symbol blocks", error=str(e))

    async def initialize(self) -> None:
        """Initialize models, executors, and connections."""
        await self.models.initialize_all()
        self.paper_executor = None

        # Initialize OKX demo/live connections for balance sync, position checks,
        # and actual order execution. Paper mode means OKX demo trading, not a
        # local fake fill.
        self.okx_executor = None
        self._okx_paper = OKXExecutor(mode="paper")
        try:
            await self._okx_paper.initialize()
            logger.info("okx paper executor initialized")
        except Exception as e:
            logger.warning("okx paper executor init failed", error=str(e))
            self._okx_paper = None
        self._okx_live = OKXExecutor(mode="live") if self._has_live_models() else None
        if self._okx_live is not None:
            try:
                await self._okx_live.initialize()
                logger.info("okx live executor initialized")
            except Exception as e:
                logger.warning("okx live executor init failed", error=str(e))

        # Restore decision/trade counters from DB
        async with get_session_ctx() as session:
            from db.repositories.decision_repo import DecisionRepository
            from db.repositories.trade_repo import TradeRepository
            from sqlalchemy import select, func
            from models.decision import AIDecision
            from models.trade import Order

            dec_count = await session.execute(select(func.count(AIDecision.id)))
            self._decision_count = (dec_count.scalar() or 0)
            trade_count = await session.execute(
                select(func.count(Order.id)).where(
                    Order.status == OrderStatus.FILLED.value,
                    Order.exchange_order_id.is_not(None),
                    Order.exchange_order_id != "",
                )
            )
            self._trade_count = (trade_count.scalar() or 0)

        await self._load_untradable_symbol_blocks()
        await self._backfill_trade_reflections()

        # Subscribe to mode changes to reinitialize LLM agent
        mode_manager.subscribe(self._on_mode_changed)

        logger.info("trading service initialized")

    async def run_once(self, analysis_scope: str = "full") -> dict[str, Any]:
        """Execute one iteration of the trading loop.

        Returns a summary dict for dashboard/notifications.
        """
        if not self._running:
            return {"status": "stopped"}

        if mode_manager.is_paused:
            return {"status": "paused"}

        analysis_scope = analysis_scope if analysis_scope in {"full", "market", "position"} else "full"
        run_market_analysis = analysis_scope in {"full", "market"}
        run_position_analysis = analysis_scope in {"full", "position"}
        new_pair_market_pause_applied = False
        round_start = datetime.now(timezone.utc)
        results = {
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
        self._last_round_started_at = round_start
        self._last_round_finished_at = None
        self._last_round_error = None
        self._set_loop_stage("starting")

        try:
            self._set_loop_stage("shadow_backtests")
            await self._update_due_shadow_backtests()
            await self._expire_stale_waiting_entry_candidates()

            # 0. Refresh per-model execution mode mapping from current config
            self._refresh_model_modes()
            self._set_loop_stage("sync_exchange_positions")
            await self.okx_sync_service.reconcile_positions("round start")
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
            await self._record_new_pair_pause_state(ENSEMBLE_TRADER_NAME, new_pair_pause_reason)
            if new_pair_pause_reason and run_market_analysis:
                new_pair_market_pause_applied = True
                logger.warning(
                    "new-pair market analysis paused; existing-position review remains active",
                    reason=new_pair_pause_reason,
                    scope=analysis_scope,
                    open_positions=len(open_positions or []),
                )
                results["warnings"].append({
                    "model": ENSEMBLE_TRADER_NAME,
                    "symbol": "ALL",
                    "warning": new_pair_pause_reason,
                })
                # The account guard is only meant to stop opening new symbols.
                # Existing positions still need SL/TP enforcement and AI review.
                run_market_analysis = False

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
                sorted(self._open_position_symbol_keys(open_positions))
                if run_position_analysis
                else []
            )
            market_scan_symbols = self._filter_blocked_new_symbols(
                scan_symbols,
                open_positions,
                results,
            )
            market_scan_symbols = self._filter_open_position_market_symbols(
                market_scan_symbols,
                open_positions,
            )
            if run_market_analysis:
                market_scan_symbols = await self._filter_unclaimed_market_symbols(market_scan_symbols)
            else:
                market_scan_symbols = []

            fetch_symbols = self._dedupe_symbols([
                *market_scan_symbols,
                *position_scan_symbols,
            ])
            if not fetch_symbols:
                if new_pair_market_pause_applied and not run_position_analysis:
                    logger.info(
                        "market analysis skipped because new-pair pause is active",
                        reason=new_pair_pause_reason,
                    )
                elif open_positions and run_position_analysis:
                    logger.warning("no feature symbols available for open-position review")
                else:
                    logger.warning("all scan symbols skipped before AI analysis")
                round_duration = (datetime.now(timezone.utc) - round_start).total_seconds()
                results["duration_ms"] = round(round_duration * 1000)
                self._last_round_finished_at = datetime.now(timezone.utc)
                self._set_loop_stage("idle")
                return results

            # 2. Get feature vectors for target symbols (parallel, with concurrency limit)
            self._set_loop_stage("fetch_features")
            feature_vectors = {}
            sem = asyncio.Semaphore(8)  # Limit concurrent data fetches

            async def fetch_fv(sym):
                async with sem:
                    try:
                        return sym, await self.data_service.get_feature_vector(sym)
                    except Exception as e:
                        logger.warning("feature vector failed", symbol=sym, error=str(e))
                        return sym, None

            tasks = [fetch_fv(s) for s in fetch_symbols]
            fv_results = await asyncio.gather(*tasks)
            feature_vectors = {s: fv for s, fv in fv_results if fv is not None}
            invalid_symbols = [
                s for s, fv in feature_vectors.items()
                if not self._is_valid_feature_vector(fv)
            ]
            if invalid_symbols:
                logger.warning(
                    "skipping invalid feature vectors before AI",
                    count=len(invalid_symbols),
                    symbols=invalid_symbols[:10],
                )
                feature_vectors = { 
                    s: fv for s, fv in feature_vectors.items() 
                    if self._is_valid_feature_vector(fv) 
                } 

            if not feature_vectors: 
                logger.warning("no feature vectors available") 
                round_duration = (datetime.now(timezone.utc) - round_start).total_seconds()
                results["duration_ms"] = round(round_duration * 1000)
                self._last_round_finished_at = datetime.now(timezone.utc)
                self._set_loop_stage("idle")
                return results 

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
            analysis_budget_context = self._position_review_budget_context(
                open_positions,
                feature_vectors,
                base_market_limit=base_market_limit,
                run_position_analysis=run_position_analysis,
                run_market_analysis=run_market_analysis,
                new_pair_pause_reason=new_pair_pause_reason,
            )
            market_symbol_budget = int(analysis_budget_context.get("market_symbol_limit") or 0)
            if run_market_analysis and market_feature_vectors:
                if market_symbol_budget <= 0:
                    market_feature_vectors = {}
                elif mode_manager.is_auto_scan:
                    market_feature_vectors = self._rank_auto_feature_vectors(market_feature_vectors, market_symbol_budget)
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
                reason=analysis_budget_context.get("reason"),
            )
            if mode_manager.is_auto_scan and market_feature_vectors:
                # Already ranked by the dynamic analysis budget above.
                pass

            market_regime_context = self._market_regime_context(market_feature_vectors or feature_vectors)
            strategy_mode_context = await self._strategy_mode_context(
                self._get_model_execution_mode(ENSEMBLE_TRADER_NAME),
                market_regime_context,
                open_positions,
            )
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
                open_positions, review_blocked_keys = await self.position_review_service.review_open_positions(
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

            # 3. Collect all entry decisions from all symbols/models
            all_candidates = []  # (symbol, model_name, decision, assessment)
            staged_entry_counts = {"model_totals": {}, "symbol_side": {}, "side_totals": {}}
            auto_submitted_entries = 0
            auto_max_trades = self._effective_auto_max_trades(
                max(0, int(settings.max_auto_trades_per_round)),
                strategy_mode_context,
            )

            if new_pair_market_pause_applied:
                results["decisions"].append({
                    "model": ENSEMBLE_TRADER_NAME,
                    "symbol": "ALL",
                    "action": "hold",
                    "approved": False,
                    "executed": False,
                    "execution_status": "paused",
                    "reason": new_pair_pause_reason,
                    "is_paper": (self._get_model_execution_mode(ENSEMBLE_TRADER_NAME) == "paper"),
                })

            for symbol, fv in market_feature_vectors.items():
                quarantine_reason = self._symbol_profit_quarantine_reason(symbol, strategy_mode_context)
                if quarantine_reason:
                    logger.info(
                        "market symbol has realized loss cooldown evidence; still sending to AI",
                        symbol=symbol,
                        reason=quarantine_reason,
                    )
                if not await self._try_claim_analysis_symbol(symbol, "market"):
                    logger.info("market symbol skipped because another analysis owns it", symbol=symbol)
                    continue
                claimed_analysis_symbols.append(symbol)
                claimed_symbol_keys.add(self._normalize_position_symbol(symbol))
                self._set_loop_stage(f"analyze:{symbol}")
                positive_candidate_count = sum(
                    1
                    for _symbol, _model_name, candidate, _assessment, _decision_db_id in all_candidates
                    if self._safe_float(
                        (
                            candidate.raw_response.get("opportunity_score", {})
                            if isinstance(candidate.raw_response, dict)
                            else {}
                        ).get("score"),
                        -1.0,
                    ) > MIN_ENTRY_OPPORTUNITY_SCORE
                )
                if False and (
                    mode_manager.is_auto_scan
                    and auto_max_trades > 0
                    and positive_candidate_count >= max(auto_max_trades, 1)
                ):
                    logger.info(
                        "auto scan has enough positive entry candidates; stop analyzing remaining symbols",
                        positive_candidates=positive_candidate_count,
                        limit=auto_max_trades,
                    )
                    break

                results["symbols_processed"] += 1
                self._remember_market_analyzed_symbol(symbol)
                fv = await self._fresh_feature_vector_for_analysis(symbol, fv)
                if not self._is_valid_feature_vector(fv):
                    logger.warning("skip symbol after fresh feature check failed", symbol=symbol)
                    continue
                feature_vectors[symbol] = fv
                model_name = ENSEMBLE_TRADER_NAME
                model_mode = self._get_model_execution_mode(model_name)
                memory_context = await self._expert_memory_context(symbol)
                daily_target_context = await self._daily_target_context()
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
                )
                market_agent_skills = self.agent_skills.market_skills(
                    new_pair_pause_reason=new_pair_pause_reason,
                    ml_signal=ml_signal_context,
                    local_ai_tools=local_ai_tools_context,
                    market_regime=market_regime_context,
                    strategy_mode=strategy_mode_context,
                )
                prefilter_reason = self._market_llm_prefilter_skip_reason(
                    fv,
                    local_ai_tools_context,
                    open_positions=open_positions,
                )
                if prefilter_reason:
                    quick_raw = {
                        "analysis_type": "market",
                        "fast_prefilter": {
                            "skipped_llm": True,
                            "reason": prefilter_reason,
                            "feature_opportunity_score": round(self._feature_opportunity_score(fv), 4),
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
                                    "recorded_at": datetime.now(timezone.utc).isoformat(),
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
                    decision_db_id = await self._log_decision(quick_decision, is_paper=(model_mode == "paper"))
                    if decision_db_id is not None:
                        round_decision_ids.add(decision_db_id)
                        await self._mark_decision_reason(decision_db_id, prefilter_reason)
                    self._decision_count += 1
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": "hold",
                        "approved": True,
                        "confidence": 0.0,
                        "executed": False,
                        "execution_status": "fast_prefilter",
                        "reason": prefilter_reason,
                        "is_paper": (model_mode == "paper"),
                    })
                    self._remember_market_hold_symbol(symbol)
                    continue
                analysis_started = datetime.now(timezone.utc)
                decision, _opinions = await self.ensemble.decide(
                    fv,
                    {
                        "open_positions": open_positions,
                        "trading_mode": mode_manager.mode.value,
                        **memory_context,
                        "daily_target": daily_target_context,
                        "market_regime": market_regime_context,
                        "strategy_mode": strategy_mode_context,
                        "direction_competition": direction_competition_context,
                        "entry_candidate_evidence": entry_candidate_evidence,
                        "ml_signal": {} if PRE_AGENT_SKILLS_ROLLBACK_MODE else ml_signal_context,
                        "local_ai_tools": {} if PRE_AGENT_SKILLS_ROLLBACK_MODE else local_ai_tools_context,
                        "ml_signal_prompt_enabled": LOCAL_QUANT_PROMPT_ENABLED,
                        "local_ai_tools_prompt_enabled": LOCAL_QUANT_PROMPT_ENABLED,
                    },
                )
                if isinstance(decision.raw_response, dict):
                    decision.raw_response.setdefault("entry_candidate_evidence", entry_candidate_evidence)
                self._attach_decision_timing(decision, analysis_started, "market")
                self.agent_skills.attach(
                    decision,
                    phase="market_analysis",
                    skills=market_agent_skills,
                    note="市场分析前的 Agent/Skills 证据快照。",
                )
                decision_db_id = await self._log_decision(decision, is_paper=(model_mode == "paper"))
                if decision_db_id is not None:
                    round_decision_ids.add(decision_db_id)
                self._decision_count += 1
                await self._create_shadow_backtests(
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
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": decision.action.value,
                        "approved": True,
                        "executed": False,
                        "execution_status": "skipped",
                        "reason": reason,
                        "is_paper": (model_mode == "paper"),
                    })
                    continue

                model_positions = [p for p in open_positions if p.get("model_name") == model_name]
                headlines = fv.recent_headlines if hasattr(fv, "recent_headlines") else []
                assessment = self.risk_engine.assess(
                    decision,
                    current_positions=model_positions,
                    account_balance=await self._get_account_balance(model_name),
                    headlines=headlines,
                    sentiment_scores=[],
                    price_change_1m=fv.returns_1 if hasattr(fv, "returns_1") else 0.0,
                    volume_ratio=fv.volume_ratio if hasattr(fv, "volume_ratio") else 1.0,
                    adx_14=fv.adx_14 if hasattr(fv, "adx_14") else None,
                )

                if (
                    not assessment.approved
                    and await self._price_action_black_swan_false_positive(
                        decision,
                        assessment.rejection_reason,
                        assessment,
                    )
                ):
                    assessment.approved = True
                    assessment.decision = decision
                    assessment.rejection_reason = ""

                if not assessment.approved:
                    if decision_db_id is not None:
                        await self._mark_decision_reason(
                            decision_db_id,
                            assessment.rejection_reason or "风控引擎拒绝该决策。",
                        )
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": decision.action.value,
                        "approved": False,
                        "reason": assessment.rejection_reason,
                        "is_paper": (model_mode == "paper"),
                    })
                    continue

                executed = assessment.decision if assessment.decision else decision
                if executed is not decision and decision.raw_response and not executed.raw_response:
                    executed.raw_response = decision.raw_response
                    executed.feature_snapshot = executed.feature_snapshot or decision.feature_snapshot
                if executed.is_hold:
                    probe_decision = self._entry_evidence_probe_decision(
                        executed,
                        fv,
                        strategy_mode_context,
                        ml_signal_context,
                        local_ai_tools_context,
                        direction_competition_context,
                    )
                    probe_source_label = "入场候选证据包"
                    if probe_decision is None:
                        probe_decision = self._quant_profit_probe_decision(
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
                        if decision_db_id is not None:
                            await self._mark_decision_reason(
                                decision_db_id,
                                f"AI 原始裁决为观望；{probe_source_label}触发正期望候选，另建一条候选决策继续风控。",
                            )
                        probe_decision_db_id = await self._log_decision(executed, is_paper=(model_mode == "paper"))
                        if probe_decision_db_id is not None:
                            round_decision_ids.add(probe_decision_db_id)
                            decision_db_id = probe_decision_db_id
                            self._decision_count += 1
                        assessment = self.risk_engine.assess(
                            executed,
                            current_positions=model_positions,
                            account_balance=await self._get_account_balance(model_name),
                            headlines=headlines,
                            sentiment_scores=[],
                            price_change_1m=fv.returns_1 if hasattr(fv, "returns_1") else 0.0,
                            volume_ratio=fv.volume_ratio if hasattr(fv, "volume_ratio") else 1.0,
                            adx_14=fv.adx_14 if hasattr(fv, "adx_14") else None,
                        )
                        if (
                            not assessment.approved
                            and await self._price_action_black_swan_false_positive(
                                executed,
                                assessment.rejection_reason,
                                assessment,
                            )
                        ):
                            assessment.approved = True
                            assessment.decision = executed
                            assessment.rejection_reason = ""

                        if not assessment.approved:
                            reason = assessment.rejection_reason or "量化盈利探针未通过风控。"
                            if decision_db_id is not None:
                                await self._mark_decision_reason(decision_db_id, reason)
                            results["decisions"].append({
                                "model": model_name,
                                "symbol": symbol,
                                "action": executed.action.value,
                                "approved": False,
                                "reason": reason,
                                "execution_status": "quant_probe_rejected",
                                "is_paper": (model_mode == "paper"),
                            })
                            continue
                    else:
                        hold_reason = getattr(executed, "reasoning", None) or getattr(decision, "reasoning", None)
                        self._remember_market_hold_symbol(symbol, fv, hold_reason)
                        if decision_db_id is not None:
                            await self._mark_decision_reason(
                                decision_db_id,
                                "多模型裁决结果为观望，未提交订单。",
                            )
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": "hold",
                            "approved": True,
                            "confidence": executed.confidence,
                            "is_paper": (model_mode == "paper"),
                        })
                        continue

                if executed.is_hold:
                    hold_reason = getattr(executed, "reasoning", None) or getattr(decision, "reasoning", None)
                    self._remember_market_hold_symbol(symbol, fv, hold_reason)
                    if decision_db_id is not None:
                        await self._mark_decision_reason(
                            decision_db_id,
                            "多模型裁决结果为观望，未提交订单。",
                        )
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": "hold",
                        "approved": True,
                        "confidence": executed.confidence,
                        "is_paper": (model_mode == "paper"),
                    })
                    continue

                if executed.is_exit:
                    reason = "市场分析阶段禁止执行平仓动作；平仓只允许由持仓分析产生，本轮改为观望。"
                    raw_response = executed.raw_response if isinstance(executed.raw_response, dict) else {}
                    raw_response["market_exit_execution_guard"] = {
                        "applied": True,
                        "original_action": executed.action.value,
                        "reason": "market_analysis_close_forbidden",
                    }
                    executed.raw_response = raw_response
                    if decision_db_id is not None:
                        await self._mark_decision_raw_response(decision_db_id, raw_response)
                        await self._mark_decision_reason(decision_db_id, reason)
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": executed.action.value,
                        "approved": True,
                        "confidence": executed.confidence,
                        "executed": False,
                        "execution_status": "skipped",
                        "reason": reason,
                        "is_paper": (model_mode == "paper"),
                    })
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
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": executed.action.value,
                        "approved": True,
                        "confidence": executed.confidence,
                        "executed": False,
                        "execution_status": "skipped",
                        "reason": new_pair_pause_reason,
                        "is_paper": (model_mode == "paper"),
                    })
                    continue

                regime_reason = self._entry_market_regime_block_reason(
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
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": executed.action.value,
                        "approved": True,
                        "confidence": executed.confidence,
                        "executed": False,
                        "execution_status": "skipped",
                        "reason": regime_reason,
                        "is_paper": (model_mode == "paper"),
                    })
                    continue

                if mode_manager.is_auto_scan:
                    if False and auto_max_trades <= 0:
                        reason = "自动模式每轮最大执行笔数为 0，本轮只记录信号，不提交订单。"
                        if decision_db_id is not None:
                            await self._mark_decision_reason(decision_db_id, reason)
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": executed.action.value,
                            "approved": True,
                            "confidence": executed.confidence,
                            "executed": False,
                            "execution_status": "skipped",
                            "reason": reason,
                            "is_paper": (model_mode == "paper"),
                        })
                        continue

                    self._clear_market_no_opportunity_symbol(symbol)
                    self.entry_policy.score_candidate(executed, strategy_mode_context)
                    if decision_db_id is not None:
                        await self._mark_decision_raw_response(decision_db_id, executed.raw_response)
                    opportunity_reason = self.entry_policy.gate_reason(executed)
                    if opportunity_reason:
                        reason = f"候选评分未达执行标准：{opportunity_reason}"
                        raw_response = self._annotate_candidate_selection(
                            executed,
                            selected=False,
                            reason=reason,
                        )
                        if decision_db_id is not None:
                            await self._mark_decision_raw_response(decision_db_id, raw_response)
                            await self._mark_decision_reason(decision_db_id, reason)
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": executed.action.value,
                            "approved": True,
                            "confidence": executed.confidence,
                            "executed": False,
                            "execution_status": "skipped",
                            "reason": reason,
                            "is_paper": (model_mode == "paper"),
                        })
                        continue
                    immediate_reason = self.entry_policy.immediate_execution_reason(executed)
                    if immediate_reason:
                        capacity_reason = self._entry_capacity_reason(
                            model_name,
                            executed,
                            open_positions,
                            staged_entry_counts,
                        )
                        if capacity_reason:
                            reason = f"强信号未即时执行：{capacity_reason}"
                            raw_response = self._annotate_candidate_selection(
                                executed,
                                selected=False,
                                reason=reason,
                            )
                            if decision_db_id is not None:
                                await self._mark_decision_raw_response(decision_db_id, raw_response)
                                await self._mark_decision_reason(decision_db_id, reason)
                            results["decisions"].append({
                                "model": model_name,
                                "symbol": symbol,
                                "action": executed.action.value,
                                "approved": True,
                                "confidence": executed.confidence,
                                "executed": False,
                                "execution_status": "skipped",
                                "reason": reason,
                                "is_paper": (model_mode == "paper"),
                            })
                            continue

                        self._reserve_entry_slot(model_name, executed, staged_entry_counts)
                        raw_response = self._annotate_candidate_selection(
                            executed,
                            selected=True,
                            reason=immediate_reason,
                        )
                        if decision_db_id is not None:
                            await self._mark_decision_raw_response(decision_db_id, raw_response)
                            await self._mark_decision_pending_execution(decision_db_id, immediate_reason)
                        self._set_loop_stage(f"execute:{symbol}")
                        try:
                            await self._execute_candidate(
                                symbol,
                                model_name,
                                executed,
                                assessment,
                                decision_db_id,
                                results,
                                open_positions=open_positions,
                            )
                            if decision_db_id is not None:
                                await self._ensure_decision_final_state(
                                    decision_db_id,
                                    symbol,
                                    model_name,
                                    executed,
                                    results,
                                )
                        except Exception as exc:
                            reason = (
                                "强信号已进入即时执行，但下单流程异常中断："
                                f"{str(exc)[:160]}。系统已跳过本次订单，下一轮会重新分析。"
                            )
                            logger.error(
                                "immediate entry execution crashed",
                                symbol=symbol,
                                model=model_name,
                                action=executed.action.value,
                                error=str(exc),
                            )
                            if decision_db_id is not None:
                                await self._mark_decision_reason(decision_db_id, reason)
                            results["decisions"].append({
                                "model": model_name,
                                "symbol": symbol,
                                "action": executed.action.value,
                                "approved": True,
                                "confidence": executed.confidence,
                                "executed": False,
                                "execution_status": "error",
                                "reason": reason,
                                "is_paper": (model_mode == "paper"),
                            })
                        continue

                    immediate_reason = self.entry_policy.immediate_execution_reason(executed) or (
                        "开仓信号已通过 AI 和执行前严重风险检查，立即进入下单流程；"
                        "不再等待本轮候选排序，避免行情变化导致错过时机。"
                    )
                    capacity_reason = self._entry_capacity_reason(
                        model_name,
                        executed,
                        open_positions,
                        staged_entry_counts,
                    )
                    if capacity_reason:
                        reason = f"开仓信号未即时执行：{capacity_reason}"
                        raw_response = self._annotate_candidate_selection(
                            executed,
                            selected=False,
                            reason=reason,
                        )
                        if decision_db_id is not None:
                            await self._mark_decision_raw_response(decision_db_id, raw_response)
                            await self._mark_decision_reason(decision_db_id, reason)
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": executed.action.value,
                            "approved": True,
                            "confidence": executed.confidence,
                            "executed": False,
                            "execution_status": "skipped",
                            "reason": reason,
                            "is_paper": (model_mode == "paper"),
                        })
                        continue

                    self._reserve_entry_slot(model_name, executed, staged_entry_counts)
                    raw_response = self._annotate_candidate_selection(
                        executed,
                        selected=True,
                        reason=immediate_reason,
                    )
                    if decision_db_id is not None:
                        await self._mark_decision_raw_response(decision_db_id, raw_response)
                        await self._mark_decision_pending_execution(decision_db_id, immediate_reason)
                    self._set_loop_stage(f"execute:{symbol}")
                    try:
                        await self._execute_candidate(
                            symbol,
                            model_name,
                            executed,
                            assessment,
                            decision_db_id,
                            results,
                            open_positions=open_positions,
                        )
                        if decision_db_id is not None:
                            await self._ensure_decision_final_state(
                                decision_db_id,
                                symbol,
                                model_name,
                                executed,
                                results,
                            )
                    except Exception as exc:
                        reason = (
                            "开仓信号已进入即时执行，但下单流程异常中断："
                            f"{str(exc)[:160]}。系统已跳过本次订单，下一轮会重新分析。"
                        )
                        logger.error(
                            "entry execution crashed",
                            symbol=symbol,
                            model=model_name,
                            action=executed.action.value,
                            error=str(exc),
                        )
                        if decision_db_id is not None:
                            await self._mark_decision_reason(decision_db_id, reason)
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": executed.action.value,
                            "approved": True,
                            "confidence": executed.confidence,
                            "executed": False,
                            "execution_status": "error",
                            "reason": reason,
                            "is_paper": (model_mode == "paper"),
                        })
                    continue

                reason = self._entry_capacity_reason(
                    model_name,
                    executed,
                    open_positions,
                    staged_entry_counts,
                )
                if reason:
                    raw_response = self._annotate_candidate_selection(
                        decision,
                        selected=False,
                        reason=reason,
                    )
                    if decision_db_id is not None:
                        await self._mark_decision_raw_response(decision_db_id, raw_response)
                        await self._mark_decision_reason(decision_db_id, reason)
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": executed.action.value,
                        "approved": True,
                        "confidence": executed.confidence,
                        "executed": False,
                        "execution_status": "skipped",
                        "reason": reason,
                        "is_paper": (model_mode == "paper"),
                    })
                    continue

                self._reserve_entry_slot(model_name, executed, staged_entry_counts)
                self._clear_market_no_opportunity_symbol(symbol)
                await self._execute_candidate(
                    symbol,
                    model_name,
                    executed,
                    assessment,
                    decision_db_id,
                    results,
                    open_positions=open_positions,
                )
                continue

                decisions = await self.models.decide_all(
                    fv,
                    {
                        "open_positions": open_positions,
                        "trading_mode": mode_manager.mode.value,
                    },
                )

                for model_name, decision in decisions.items():
                    if not isinstance(decision, DecisionOutput):
                        continue

                    model_mode = self._get_model_execution_mode(model_name)
                    decision_db_id = await self._log_decision(decision, is_paper=(model_mode == "paper"))
                    if decision_db_id is not None:
                        round_decision_ids.add(decision_db_id)
                    self._decision_count += 1

                    # Risk assessment: only check this model's own positions
                    decision_key = (model_name, self._normalize_position_symbol(symbol))
                    if decision_key in review_blocked_keys and not decision.is_hold:
                        reason = "本轮已优先处理该持仓的平仓决策，跳过同币种的后续信号。"
                        if decision_db_id is not None:
                            await self._mark_decision_reason(decision_db_id, reason)
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": decision.action.value,
                            "approved": True,
                            "executed": False,
                            "execution_status": "skipped",
                            "reason": reason,
                            "is_paper": (model_mode == "paper"),
                        })
                        continue

                    model_positions = [p for p in open_positions if p.get("model_name") == model_name]
                    headlines = fv.recent_headlines if hasattr(fv, "recent_headlines") else []
                    assessment = self.risk_engine.assess(
                        decision,
                        current_positions=model_positions,
                        account_balance=await self._get_account_balance(model_name),
                        headlines=headlines,
                        sentiment_scores=[],
                        price_change_1m=fv.returns_1 if hasattr(fv, "returns_1") else 0.0,
                        volume_ratio=fv.volume_ratio if hasattr(fv, "volume_ratio") else 1.0,
                        adx_14=fv.adx_14 if hasattr(fv, "adx_14") else None,
                    )

                    if (
                        not assessment.approved
                        and await self._price_action_black_swan_false_positive(
                            decision,
                            assessment.rejection_reason,
                            assessment,
                        )
                    ):
                        assessment.approved = True
                        assessment.decision = decision
                        assessment.rejection_reason = ""

                    if not assessment.approved:
                        if decision_db_id is not None:
                            await self._mark_decision_reason(
                                decision_db_id,
                                assessment.rejection_reason or "风控引擎拒绝该决策。",
                            )
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": decision.action.value,
                            "approved": False,
                            "reason": assessment.rejection_reason,
                            "is_paper": (model_mode == "paper"),
                        })
                        continue

                    executed = assessment.decision if assessment.decision else decision
                    if executed.is_hold:
                        if decision_db_id is not None:
                            await self._mark_decision_reason(
                                decision_db_id,
                                "AI 选择观望，未提交订单。",
                            )
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": "hold",
                            "approved": True,
                            "confidence": executed.confidence,
                            "is_paper": (model_mode == "paper"),
                        })
                        continue

                    if executed.is_exit:
                        await self._execute_candidate(
                            symbol,
                            model_name,
                            executed,
                            assessment,
                            decision_db_id,
                            results,
                            open_positions=open_positions,
                        )
                        continue

                    all_candidates.append((symbol, model_name, executed, assessment, decision_db_id))

            all_candidates.sort(
                key=lambda x: self.entry_policy.score_candidate(x[2], strategy_mode_context),
                reverse=True,
            )
            candidate_count = len(all_candidates)
            for rank, (_symbol, _model_name, decision, _assessment, decision_db_id) in enumerate(all_candidates, start=1):
                self.entry_policy.score_candidate(decision, strategy_mode_context)
                pending_reason = self.entry_policy.wait_sort_reason(
                    decision,
                    rank=rank,
                    candidate_count=candidate_count,
                )
                raw_response = self._annotate_candidate_selection(
                    decision,
                    rank=rank,
                    candidate_count=candidate_count,
                    selected=False,
                    reason=pending_reason,
                )
                if decision_db_id is not None:
                    await self._mark_decision_raw_response(decision_db_id, raw_response)
                    await self._mark_decision_reason(decision_db_id, pending_reason)

            opportunity_filtered_candidates = []
            for symbol, model_name, decision, assessment, decision_db_id in all_candidates:
                reason = self.entry_policy.gate_reason(decision)
                if reason:
                    raw_response = self._annotate_candidate_selection(
                        decision,
                        selected=False,
                        reason=reason,
                    )
                    if decision_db_id is not None:
                        await self._mark_decision_raw_response(decision_db_id, raw_response)
                        await self._mark_decision_reason(decision_db_id, reason)
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": decision.action.value,
                        "approved": True,
                        "confidence": decision.confidence,
                        "executed": False,
                        "execution_status": "skipped",
                        "reason": reason,
                        "is_paper": (self._get_model_execution_mode(model_name) == "paper"),
                    })
                    continue
                opportunity_filtered_candidates.append((symbol, model_name, decision, assessment, decision_db_id))
            all_candidates = opportunity_filtered_candidates

            capacity_filtered_candidates = []
            for symbol, model_name, decision, assessment, decision_db_id in all_candidates:
                regime_reason = self._entry_market_regime_block_reason(
                    decision,
                    strategy_mode_context or market_regime_context,
                )
                if regime_reason:
                    if decision_db_id is not None:
                        await self._mark_decision_reason(decision_db_id, regime_reason)
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": decision.action.value,
                        "approved": True,
                        "confidence": decision.confidence,
                        "executed": False,
                        "execution_status": "skipped",
                        "reason": regime_reason,
                        "is_paper": (self._get_model_execution_mode(model_name) == "paper"),
                    })
                    continue
                reason = self._entry_capacity_reason(
                    model_name,
                    decision,
                    open_positions,
                    staged_entry_counts,
                )
                if reason:
                    if decision_db_id is not None:
                        await self._mark_decision_reason(decision_db_id, reason)
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": decision.action.value,
                        "approved": True,
                        "confidence": decision.confidence,
                        "executed": False,
                        "execution_status": "skipped",
                        "reason": reason,
                        "is_paper": (self._get_model_execution_mode(model_name) == "paper"),
                    })
                    continue
                capacity_filtered_candidates.append((symbol, model_name, decision, assessment, decision_db_id))
                self._reserve_entry_slot(model_name, decision, staged_entry_counts)
            all_candidates = capacity_filtered_candidates

            # 4. Filter entry candidates: auto mode no longer limits the number of executable entries.
            if mode_manager.is_auto_scan:
                max_trades = len(all_candidates)
                candidates_to_execute = all_candidates
                if False and len(all_candidates) > max_trades:
                    logger.info("auto scan limited", total=len(all_candidates), executing=max_trades)
                    for symbol, model_name, decision, _assessment, decision_db_id in all_candidates[max_trades:]:
                        score_info = (
                            decision.raw_response.get("opportunity_score", {})
                            if isinstance(decision.raw_response, dict)
                            else {}
                        )
                        rank = score_info.get("rank")
                        score = score_info.get("score")
                        reason = (
                            f"本轮按预期净收益排序后未进入前 {max_trades} 个执行名额"
                            f"（机会排名 {rank or '-'}，机会分 {score if score is not None else '-'}），"
                            "暂不提交订单。"
                        )
                        raw_response = self._annotate_candidate_selection(
                            decision,
                            selected=False,
                            reason=reason,
                        )
                        if decision_db_id is not None:
                            await self._mark_decision_raw_response(decision_db_id, raw_response)
                            await self._mark_decision_reason(decision_db_id, reason)
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": decision.action.value,
                            "approved": True,
                            "confidence": decision.confidence,
                            "executed": False,
                            "execution_status": "skipped",
                            "reason": reason,
                            "is_paper": (self._get_model_execution_mode(model_name) == "paper"),
                        })
            else:
                candidates_to_execute = all_candidates

            # 5. Execute selected entry decisions
            for symbol, model_name, decision, assessment, decision_db_id in candidates_to_execute:
                normalized_symbol = self._normalize_position_symbol(symbol)
                if normalized_symbol not in claimed_symbol_keys:
                    if not await self._try_claim_analysis_symbol(symbol, "market"):
                        reason = "该币种正在被另一条分析流程处理，本次开仓执行跳过，等待下一轮重新评估。"
                        logger.info("entry execution skipped because another analysis owns symbol", symbol=symbol)
                        if decision_db_id is not None:
                            await self._mark_decision_reason(decision_db_id, reason)
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": decision.action.value,
                            "approved": True,
                            "confidence": decision.confidence,
                            "executed": False,
                            "execution_status": "skipped",
                            "reason": reason,
                            "is_paper": (self._get_model_execution_mode(model_name) == "paper"),
                        })
                        continue
                    claimed_analysis_symbols.append(symbol)
                    claimed_symbol_keys.add(normalized_symbol)
                self._set_loop_stage(f"execute:{symbol}")
                raw_response = self._annotate_candidate_selection(
                    decision,
                    selected=True,
                    reason="排序后进入执行：该信号不是即时强信号，但在本轮候选比较后通过机会评分、容量和风控筛选，正在进入下单前检查。",
                )
                if decision_db_id is not None:
                    await self._mark_decision_raw_response(decision_db_id, raw_response)
                    await self._mark_decision_pending_execution(
                        decision_db_id,
                        "排序后进入执行：该信号不是即时强信号，但在本轮候选比较后通过机会评分、容量和风控筛选；正在进行下单前价格偏移、异常插针、保证金和 OKX 提交检查。",
                    )
                try:
                    await self._execute_candidate(
                        symbol,
                        model_name,
                        decision,
                        assessment,
                        decision_db_id,
                        results,
                        open_positions=open_positions,
                    )
                    if decision_db_id is not None:
                        await self._ensure_decision_final_state(
                            decision_db_id,
                            symbol,
                            model_name,
                            decision,
                            results,
                        )
                except Exception as exc:
                    reason = (
                        "候选进入执行流程后异常中断："
                        f"{str(exc)[:160]}。系统已跳过本次订单，下一轮会用最新行情重新评估。"
                    )
                    logger.error(
                        "entry candidate execution crashed",
                        symbol=symbol,
                        model=model_name,
                        action=decision.action.value,
                        error=str(exc),
                    )
                    if decision_db_id is not None:
                        await self._mark_decision_reason(decision_db_id, reason)
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": decision.action.value,
                        "approved": True,
                        "confidence": decision.confidence,
                        "executed": False,
                        "execution_status": "error",
                        "reason": reason,
                        "is_paper": (self._get_model_execution_mode(model_name) == "paper"),
                    })

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

        except Exception as e:
            self._set_loop_stage("error", str(e))
            await self._fill_missing_decision_reasons(
                round_decision_ids,
                f"\u672c\u8f6e\u6267\u884c\u5f02\u5e38\u4e2d\u65ad\uff0c\u672a\u80fd\u5b8c\u6210\u6700\u7ec8\u72b6\u6001\u56de\u5199\uff1a{str(e)[:120]}",
            )
            logger.error("trading loop iteration failed", error=str(e))
            results["status"] = "error"
            results["error"] = str(e)

        for symbol in claimed_analysis_symbols:
            await self._release_analysis_symbol(symbol)
        claimed_analysis_symbols.clear()
        round_duration = (datetime.now(timezone.utc) - round_start).total_seconds()
        results["duration_ms"] = round(round_duration * 1000)
        self._last_round_finished_at = datetime.now(timezone.utc)
        self._set_loop_stage("idle" if results.get("status") == "ok" else "error")
        return results

    async def start(self) -> None:
        """Start the continuous trading loop."""
        await self.initialize()
        self._running = True
        self._start_time = datetime.now(timezone.utc)
        self._start_ml_auto_train_loop()
        logger.info("trading service started", mode=mode_manager.mode.value, scheduler="parallel_market_position")

        self._position_analysis_task = asyncio.create_task(
            self.position_review_service.loop(max(5.0, float(settings.decision_interval_seconds) * 0.65))
        )
        self._market_analysis_task = asyncio.create_task(
            self.market_analysis_service.loop(max(8.0, float(settings.decision_interval_seconds)))
        )
        try:
            await asyncio.gather(self._position_analysis_task, self._market_analysis_task)
        except asyncio.CancelledError:
            pass

    async def _analysis_loop(self, scope: str, interval_seconds: float) -> None:
        await asyncio.sleep(0.5 if scope == "position" else 3.0)
        while self._running:
            try:
                await self.run_once(scope)
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("trading analysis loop error", scope=scope, error=str(e))
                await asyncio.sleep(interval_seconds)

    async def stop(self) -> None:
        """Stop the trading loop gracefully."""
        self._running = False
        await self._stop_ml_auto_train_loop()
        for task in (self._position_analysis_task, self._market_analysis_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._position_analysis_task = None
        self._market_analysis_task = None
        if self.paper_executor:
            await self.paper_executor.shutdown()
        for okx in (self.okx_executor, self._okx_paper, self._okx_live):
            if okx:
                try:
                    await okx.shutdown()
                except Exception:
                    pass
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
                logger.warning("local ML signal auto-train loop error", error=str(e))
            try:
                await self._maybe_train_local_ai_tools()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("local AI tools auto-train loop error", error=str(e))
            await asyncio.sleep(AUTO_TRAIN_CHECK_INTERVAL_SECONDS)

    async def _maybe_train_local_ai_tools(self, *, force: bool = False) -> dict[str, Any]:
        """Push fresh history to the server-side profit/time-series/exit models."""
        if not self.local_ai_tools.enabled():
            return {"trained": False, "reason": "disabled"}

        status = await self.local_ai_tools.status()
        now = datetime.now(timezone.utc)
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
            (status or {}).get("completed_shadow_sample_count")
            or self._local_tools_last_completed_shadow_count
            or server_shadow_count
            or 0
        )

        try:
            from scripts.train_local_ai_tools_models import (
                _load_closed_position_samples,
                _load_sequence_samples,
                _load_shadow_samples,
                _load_text_sentiment_samples,
                _load_trade_reflection_samples,
            )

            shadow_samples = await _load_shadow_samples(20000)
            trade_samples = await _load_trade_reflection_samples(8000)
            trade_samples.extend(await _load_closed_position_samples(8000))
            sequence_samples = await _load_sequence_samples(12000)
            text_sentiment_samples = await _load_text_sentiment_samples(8000)
        except Exception as exc:
            return {"trained": False, "reason": "load_samples_error", "error": str(exc)[:180]}

        training_shadow_count = len(shadow_samples)
        new_shadow = max(completed_shadow_total - previous_completed_shadow_total, 0)
        new_trade = max(len(trade_samples) - server_trade_count, 0)
        should_train = force or age_seconds >= 6 * 60 * 60 or new_shadow >= 500 or new_trade >= 50
        if not should_train:
            return {
                "trained": False,
                "reason": "not_due",
                "server_shadow_sample_count": server_shadow_count,
                "local_shadow_sample_count": training_shadow_count,
                "completed_shadow_sample_count": completed_shadow_total,
                "training_shadow_sample_limit": 20000,
                "new_shadow_sample_count": new_shadow,
                "new_trade_sample_count": new_trade,
            }

        self._local_tools_last_train_started_at = now
        result = await self.local_ai_tools.train(
            shadow_samples,
            trade_samples,
            sequence_samples,
            text_sentiment_samples,
            source="local_trading_system_auto",
        )
        if result.get("trained"):
            result["completed_shadow_sample_count"] = completed_shadow_total
            result["training_shadow_sample_count"] = training_shadow_count
            result["training_shadow_sample_limit"] = 20000
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
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    async def _recover_pending_exit_decisions(
        self,
        results: dict[str, Any],
        open_positions: list[dict],
        round_decision_ids: set[int],
    ) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
        pending: list[dict[str, Any]] = []
        try:
            async with get_session_ctx() as session:
                stmt = (
                    select(AIDecision)
                    .where(
                        AIDecision.was_executed == False,
                        AIDecision.action.in_([Action.CLOSE_LONG.value, Action.CLOSE_SHORT.value]),
                        AIDecision.created_at >= cutoff.replace(tzinfo=None),
                        or_(
                            AIDecision.execution_reason.is_(None),
                            AIDecision.execution_reason == "",
                            AIDecision.execution_reason.like("本轮还在分析或排队中%"),
                            AIDecision.execution_reason.like("鏈疆杩樺湪鍒嗘瀽鎴栨帓闃熶腑%"),
                        ),
                    )
                    .order_by(AIDecision.created_at.asc())
                    .limit(10)
                )
                rows = list((await session.execute(stmt)).scalars().all())
                for row in rows:
                    order_count = (
                        await session.execute(
                            select(func.count(Order.id)).where(Order.decision_id == row.id)
                        )
                    ).scalar() or 0
                    if order_count > 0:
                        continue
                    pending.append({
                        "id": row.id,
                        "model_name": row.model_name,
                        "symbol": row.symbol,
                        "action": row.action,
                        "confidence": row.confidence,
                        "reasoning": row.reasoning or "",
                        "position_size_pct": row.position_size_pct,
                        "suggested_leverage": row.suggested_leverage,
                        "stop_loss_pct": row.stop_loss_pct,
                        "take_profit_pct": row.take_profit_pct,
                        "raw_response": row.raw_llm_response,
                        "feature_snapshot": row.feature_snapshot,
                        "created_at": row.created_at,
                    })
        except Exception as e:
            logger.error("failed to load pending exit decisions", error=str(e))
            return

        if not pending:
            return

        self._set_loop_stage("recover_pending_exits")
        logger.warning("recovering pending exit decisions", count=len(pending))
        for item in pending:
            action = Action.from_string(str(item["action"]))
            if not action.is_exit():
                continue
            created_at = item.get("created_at")
            if isinstance(created_at, datetime):
                timestamp = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            decision = DecisionOutput(
                model_name=str(item["model_name"]),
                symbol=str(item["symbol"]),
                action=action,
                confidence=float(item["confidence"] or 0.0),
                reasoning=str(item["reasoning"] or ""),
                position_size_pct=float(item["position_size_pct"] or 1.0),
                suggested_leverage=float(item["suggested_leverage"] or 1.0),
                stop_loss_pct=float(item["stop_loss_pct"] or 0.05),
                take_profit_pct=float(item["take_profit_pct"] or 0.10),
                timestamp=timestamp,
                raw_response=item["raw_response"] if isinstance(item["raw_response"], dict) else {},
                feature_snapshot=item["feature_snapshot"] if isinstance(item["feature_snapshot"], dict) else {},
            )
            decision_db_id = int(item["id"])
            round_decision_ids.add(decision_db_id)
            await self._execute_candidate(
                decision.symbol,
                decision.model_name,
                decision,
                SimpleNamespace(warnings=[]),
                decision_db_id,
                results,
                open_positions=open_positions,
            )

    async def _apply_entry_profit_risk_sizing(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict] | None = None,
    ) -> None:
        """Cap entry size by planned USDT loss at stop, especially during drawdown recovery."""
        if not decision.is_entry:
            return
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
        risk_mode = str(opportunity.get("risk_mode") or "normal")
        configured_max_loss = self._safe_float(opportunity.get("max_entry_stop_loss_usdt"), 0.0)
        if configured_max_loss <= 0:
            configured_max_loss = {
                "defensive_recovery": ENTRY_MAX_STOP_LOSS_DEFENSIVE_USDT,
                "drawdown_recovery": ENTRY_MAX_STOP_LOSS_DRAWDOWN_USDT,
            }.get(risk_mode, ENTRY_MAX_STOP_LOSS_NORMAL_USDT)

        stop_loss_pct = max(float(decision.stop_loss_pct or 0.0), 0.0)
        leverage = max(float(decision.suggested_leverage or 1.0), 1.0)
        current_size = max(float(decision.position_size_pct or 0.0), 0.0)
        if stop_loss_pct <= 0 or current_size <= 0:
            return
        weak_history = bool(opportunity.get("weak_history_requires_stronger_edge"))
        local_expected = self._safe_float(opportunity.get("server_profit_expected_return_pct"), 0.0)
        local_aligned = bool(opportunity.get("local_profit_aligned"))
        ml_aligned = bool(opportunity.get("ml_aligned"))
        profit_quality_ratio = self._safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        min_profit_quality_ratio = self._safe_float(
            opportunity.get("min_profit_quality_ratio_required"),
            ENTRY_MIN_NET_PROFIT_QUALITY_RATIO,
        )
        expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        tail_risk = self._safe_float(opportunity.get("tail_risk_score"), 0.0)
        score = self._safe_float(opportunity.get("score"), 0.0)
        min_score_required = self._safe_float(
            opportunity.get("min_score_required"),
            MIN_ENTRY_OPPORTUNITY_SCORE,
        )
        raw_expected_return = self._safe_float(
            opportunity.get("raw_expected_return_pct", opportunity.get("expected_return_pct")),
            0.0,
        )
        expected_loss_pct = self._safe_float(opportunity.get("expected_loss_pct"), 0.0)
        small_win_big_loss_penalty = self._safe_float(
            opportunity.get("small_win_big_loss_penalty"),
            0.0,
        )
        loss_probability = self._safe_float(opportunity.get("server_profit_loss_probability"), 1.0)
        contribution_adjustment = (
            opportunity.get("model_contribution_adjustment")
            if isinstance(opportunity.get("model_contribution_adjustment"), dict)
            else {}
        )
        hard_contribution_caution = bool(contribution_adjustment.get("hard_caution"))
        quant_probe = raw.get("quant_profit_probe") if isinstance(raw.get("quant_profit_probe"), dict) else {}
        evidence_probe = raw.get("evidence_profit_probe") if isinstance(raw.get("evidence_profit_probe"), dict) else {}
        quant_probe_triggered = bool(quant_probe.get("triggered"))
        evidence_probe_triggered = bool(evidence_probe.get("triggered"))
        strong_probe = bool(quant_probe.get("strong_probe") or evidence_probe.get("high_profit_potential"))
        roster_fill_relief = (
            opportunity.get("portfolio_roster_fill_relief")
            if isinstance(opportunity.get("portfolio_roster_fill_relief"), dict)
            else {}
        )
        roster = opportunity.get("portfolio_roster") if isinstance(opportunity.get("portfolio_roster"), dict) else {}
        symbol_profit_tier = str(opportunity.get("symbol_profit_tier") or "neutral")
        roster_fill_candidate = bool(
            roster_fill_relief.get("applied")
            or quant_probe.get("roster_fill_probe")
            or (roster.get("underfilled") and (quant_probe_triggered or evidence_probe_triggered) and not strong_probe)
        )
        if quant_probe_triggered:
            loss_probability = self._safe_float(
                quant_probe.get("loss_probability"),
                loss_probability,
            )
        if evidence_probe_triggered:
            loss_probability = self._safe_float(
                evidence_probe.get("loss_probability"),
                loss_probability,
            )
        high_quality_boost = raw.get("profit_quality_position_boost")
        high_quality_entry = bool(
            (isinstance(high_quality_boost, dict) and high_quality_boost.get("allow"))
            or (
                expected_net > 0
                and profit_quality_ratio >= max(min_profit_quality_ratio, 0.85)
                and tail_risk < 0.88
                and (local_aligned or ml_aligned or bool(opportunity.get("timeseries_aligned")))
            )
        )
        caps: list[str] = []
        if weak_history:
            weak_history_max_size = (
                ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MAX_SIZE
                if high_quality_entry
                else ENTRY_WEAK_HISTORY_MAX_SIZE
            )
            weak_history_max_leverage = (
                ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MAX_LEVERAGE
                if high_quality_entry
                else ENTRY_WEAK_HISTORY_MAX_LEVERAGE
            )
            if current_size > weak_history_max_size:
                current_size = weak_history_max_size
                decision.position_size_pct = current_size
                caps.append(
                    "近期该币种/方向真实盈亏偏弱，但当前盈利证据同向，仓位按中小仓验证"
                    if high_quality_entry
                    else "近期该币种/方向真实盈亏偏弱，仓位降为小仓验证"
                )
            if leverage > weak_history_max_leverage:
                leverage = weak_history_max_leverage
                decision.suggested_leverage = leverage
                caps.append(
                    "近期亏损方向当前证据改善，杠杆放宽但仍限制上限"
                    if high_quality_entry
                    else "近期亏损方向限制杠杆，避免单笔亏损继续放大"
                )
        if local_expected < 0 and not (local_aligned or ml_aligned):
            if current_size > ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_SIZE:
                current_size = ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_SIZE
                decision.position_size_pct = current_size
                caps.append("服务器盈利模型反向或预期为负，仅允许极小仓")
            if leverage > ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_LEVERAGE:
                leverage = ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_LEVERAGE
                decision.suggested_leverage = leverage
                caps.append("服务器盈利模型未支持该方向，降低杠杆")
        low_payoff_quality = bool(
            score < min_score_required
            or expected_net < 0.45
            or profit_quality_ratio < 0.75
            or raw_expected_return < 0
            or small_win_big_loss_penalty >= 0.65
            or hard_contribution_caution
        )
        if low_payoff_quality:
            high_quality_entry = False
            if current_size > ENTRY_LOW_QUALITY_MAX_SIZE:
                current_size = ENTRY_LOW_QUALITY_MAX_SIZE
                decision.position_size_pct = current_size
                caps.append("收益质量不足或存在小盈大亏风险，仓位降为小仓验证")
            if leverage > ENTRY_LOW_QUALITY_MAX_LEVERAGE:
                leverage = ENTRY_LOW_QUALITY_MAX_LEVERAGE
                decision.suggested_leverage = leverage
                caps.append("收益质量不足或存在小盈大亏风险，杠杆降到低档")
        if symbol_profit_tier == "side_loser" and not high_quality_entry:
            loser_size_cap = max(ENTRY_LOW_QUALITY_MAX_SIZE, current_size * ENTRY_SYMBOL_LOSER_SIZE_MULTIPLIER)
            if current_size > loser_size_cap:
                current_size = loser_size_cap
                decision.position_size_pct = current_size
                caps.append("该币种同方向近期真实亏损，未达到高质量解锁前缩小仓位")
        balance = await self._allocated_order_balance(model_mode, decision)
        if balance <= 0:
            return
        dynamic_hard_cap = min(
            max(balance * ENTRY_MAX_STOP_LOSS_PCT_OF_EQUITY, 6.0),
            ENTRY_MAX_STOP_LOSS_CAP_USDT,
        )
        risk_budget_boost = None
        if (
            high_quality_entry
            and not low_payoff_quality
            and risk_mode in {"drawdown_recovery", "defensive_recovery", "hard_recovery"}
        ):
            boost_budget = (
                ENTRY_HIGH_QUALITY_STOP_LOSS_DEFENSIVE_USDT
                if risk_mode in {"defensive_recovery", "hard_recovery"}
                else ENTRY_HIGH_QUALITY_STOP_LOSS_DRAWDOWN_USDT
            )
            boosted_max_loss = min(max(configured_max_loss, boost_budget), dynamic_hard_cap)
            if boosted_max_loss > configured_max_loss:
                risk_budget_boost = {
                    "applied": True,
                    "from_usdt": round(configured_max_loss, 6),
                    "to_usdt": round(boosted_max_loss, 6),
                    "reason": "当前是高质量同向机会，亏损恢复模式下允许更合理的单笔风险预算。",
                }
                configured_max_loss = boosted_max_loss
        if low_payoff_quality:
            configured_max_loss = min(configured_max_loss, ENTRY_MAX_STOP_LOSS_DEFENSIVE_USDT)
        configured_max_loss = min(configured_max_loss, dynamic_hard_cap)
        max_loss = max(configured_max_loss, 1.0)
        pnl_structure_guard = {
            "applied": False,
            "expected_net_return_pct": round(expected_net, 6),
            "profit_quality_ratio": round(profit_quality_ratio, 6),
        }
        stress_stop_loss_pct = max(
            stop_loss_pct,
            expected_loss_pct / 100.0 if expected_loss_pct > 0 else 0.0,
            min(max(tail_risk, 0.0) * 0.075, ENTRY_STRESS_STOP_MAX_PCT),
            (
                min(abs(raw_expected_return) / 100.0 * 0.65, ENTRY_STRESS_STOP_MAX_PCT)
                if raw_expected_return < 0
                else 0.0
            ),
            ENTRY_LOW_QUALITY_STRESS_STOP_MIN_PCT if low_payoff_quality else ENTRY_STRESS_STOP_MIN_PCT,
        )
        stress_stop_loss_pct = min(max(stress_stop_loss_pct, stop_loss_pct), ENTRY_STRESS_STOP_MAX_PCT)

        original_size_before_floor = current_size
        original_notional = balance * current_size * leverage
        target_min_notional = 0.0
        notional_floor_ratio = 0.0
        notional_floor_reason = ""
        quality_tier = "probe" if (quant_probe_triggered or evidence_probe_triggered) else "base"
        meaningful_size_reason = ""
        timeseries_aligned = bool(opportunity.get("timeseries_aligned"))
        existing_winner = self._same_side_existing_winner_context(
            decision,
            open_positions or [],
        )
        has_existing_winner = bool(existing_winner.get("has_winner"))
        strong_probe_quality = bool(
            strong_probe
            and expected_net >= 0.75
            and profit_quality_ratio >= 0.85
            and loss_probability <= 0.42
            and (score >= 2.8 or evidence_probe_triggered)
        )
        elite_quality = bool(
            expected_net >= 1.20
            and profit_quality_ratio >= 1.20
            and loss_probability <= 0.38
            and tail_risk <= ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
            and (local_aligned or ml_aligned or timeseries_aligned)
        )
        high_profit_quality = bool(
            expected_net >= 1.60
            and profit_quality_ratio >= 1.45
            and loss_probability <= 0.34
            and tail_risk <= 0.72
            and score >= max(self._safe_float(opportunity.get("min_score_required"), MIN_ENTRY_OPPORTUNITY_SCORE), 1.15)
            and (local_aligned or ml_aligned or timeseries_aligned)
        )
        good_probe_quality = bool(
            (quant_probe_triggered or evidence_probe_triggered)
            and expected_net >= 0.35
            and profit_quality_ratio >= 0.20
            and loss_probability <= 0.52
            and tail_risk <= ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
        )
        roster_fill_quality = bool(
            roster_fill_candidate
            and expected_net >= PORTFOLIO_ROSTER_FILL_MIN_NET_PCT
            and profit_quality_ratio >= PORTFOLIO_ROSTER_FILL_MIN_PROFIT_QUALITY_RATIO
            and loss_probability <= PORTFOLIO_ROSTER_FILL_MAX_LOSS_PROBABILITY
            and tail_risk <= 0.88
            and (
                local_aligned
                or ml_aligned
                or timeseries_aligned
                or quant_probe_triggered
            )
        )
        if has_existing_winner and (
            strong_probe_quality
            or elite_quality
            or (
                expected_net >= 0.55
                and profit_quality_ratio >= 0.55
                and loss_probability <= 0.48
            )
        ):
            current_profit = self._safe_float(existing_winner.get("unrealized_pnl"), 0.0)
            profit_ratio = self._safe_float(existing_winner.get("pnl_ratio"), 0.0)
            if (
                current_profit >= ENTRY_MEANINGFUL_SIZE_MIN_PROFIT_USDT
                or profit_ratio >= ENTRY_MEANINGFUL_SIZE_MIN_PROFIT_RATIO
            ):
                quality_tier = "winner_add"
                notional_floor_ratio = ENTRY_WINNER_ADD_MIN_NOTIONAL_BALANCE_RATIO
                meaningful_size_reason = (
                    "已有同币种同方向持仓浮盈，且新信号仍然同向；允许用中等仓位加到赢家上，"
                    "把正确判断转成更有意义的利润。"
                )
        if quality_tier != "winner_add" and elite_quality:
            quality_tier = "elite"
            notional_floor_ratio = ENTRY_ELITE_MIN_NOTIONAL_BALANCE_RATIO
            meaningful_size_reason = "净收益、盈亏质量和亏损概率同时达到精选标准，使用精选仓位地板。"
        if (
            high_profit_quality
            and not low_payoff_quality
            and quality_tier in {"base", "probe", "good_probe", "strong_probe", "elite"}
        ):
            quality_tier = "high_profit"
            notional_floor_ratio = max(notional_floor_ratio, ENTRY_HIGH_PROFIT_MIN_NOTIONAL_BALANCE_RATIO)
            target_leverage_floor = (
                ENTRY_HIGH_PROFIT_ELITE_MIN_LEVERAGE
                if expected_net >= 2.20 and profit_quality_ratio >= 1.80 and loss_probability <= 0.30
                else ENTRY_HIGH_PROFIT_MIN_LEVERAGE
            )
            if leverage < target_leverage_floor:
                leverage = min(target_leverage_floor, settings.max_leverage)
                decision.suggested_leverage = leverage
            meaningful_size_reason = (
                "盈利可能性较大：预期净收益、盈亏质量、亏损概率和尾部风险同时达标，"
                "允许适当提高交易数量和杠杆，把高质量机会转成更大的实际收益。"
            )
        elif quality_tier not in {"winner_add", "elite"} and strong_probe_quality and not low_payoff_quality:
            quality_tier = "strong_probe"
            notional_floor_ratio = ENTRY_STRONG_PROBE_MIN_NOTIONAL_BALANCE_RATIO
            meaningful_size_reason = "强量化探针通过净收益、盈亏质量、亏损概率和机会分联合校验，升级为有效仓位。"
        elif quality_tier == "probe" and good_probe_quality and not low_payoff_quality:
            quality_tier = "good_probe"
            notional_floor_ratio = ENTRY_GOOD_PROBE_MIN_NOTIONAL_BALANCE_RATIO
            meaningful_size_reason = "普通量化探针质量达标，不再只做极小验证仓，抬到可产生有效收益的基础仓位。"
        elif quality_tier in {"probe", "base"} and roster_fill_quality and not low_payoff_quality:
            quality_tier = "roster_fill"
            notional_floor_ratio = PORTFOLIO_ROSTER_FILL_NOTIONAL_BALANCE_RATIO
            meaningful_size_reason = (
                "当前组合低于目标持仓组数，且该信号仍为正期望；使用小而不碎的补齐仓位，"
                "让组合逐步接近 10 个独立持仓组。"
            )

        if notional_floor_ratio > 0:
            target_min_notional = balance * notional_floor_ratio
            notional_floor_reason = meaningful_size_reason
            high_quality_entry = high_quality_entry or quality_tier in {"elite", "strong_probe", "winner_add", "high_profit"}
            if symbol_profit_tier in {"side_winner", "symbol_winner"} and quality_tier in {"base", "probe", "good_probe", "roster_fill"}:
                target_min_notional *= 1.15
                notional_floor_reason = (
                    f"{notional_floor_reason} 该币种近期真实盈利，名义本金地板小幅上调。"
                    if notional_floor_reason
                    else "该币种近期真实盈利，名义本金地板小幅上调。"
                )

        if high_quality_entry and not low_payoff_quality:
            quality_reference = max(min_profit_quality_ratio, 0.85)
            quality_multiplier = min(
                max(profit_quality_ratio / max(quality_reference, 1e-12), 0.75),
                1.35,
            )
            default_floor_ratio = ENTRY_HIGH_QUALITY_MIN_NOTIONAL_BALANCE_RATIO * quality_multiplier
            if default_floor_ratio > notional_floor_ratio:
                notional_floor_ratio = default_floor_ratio
                target_min_notional = balance * notional_floor_ratio
                notional_floor_reason = "高质量盈利证据同向，按当前可用资金和信号质量动态抬高名义本金，避免无意义小盈"
        elif (
            expected_net > 0
            and profit_quality_ratio >= 0.65
            and (local_aligned or ml_aligned or timeseries_aligned)
        ):
            quality_multiplier = min(max(profit_quality_ratio / 0.65, 0.75), 1.25)
            notional_floor_ratio = ENTRY_NORMAL_MIN_NOTIONAL_BALANCE_RATIO * quality_multiplier
            target_min_notional = balance * notional_floor_ratio
            notional_floor_reason = "普通正收益信号获得本地模型或时序模型支持，按当前可用资金动态设置基础名义本金"

        intended_notional_for_profit = max(original_notional, target_min_notional)
        expected_profit_usdt = intended_notional_for_profit * max(expected_net, 0.0) / 100.0
        if expected_net > 0 and intended_notional_for_profit > 0:
            max_loss_multiple = (
                ENTRY_PNL_STRUCTURE_LOW_QUALITY_MAX_LOSS_MULTIPLE
                if low_payoff_quality or symbol_profit_tier == "side_loser"
                else ENTRY_PNL_STRUCTURE_HIGH_QUALITY_MAX_LOSS_MULTIPLE
                if high_quality_entry or quality_tier in {"elite", "winner_add", "high_profit", "strong_probe"}
                else ENTRY_PNL_STRUCTURE_NORMAL_MAX_LOSS_MULTIPLE
            )
            structure_max_loss = max(
                ENTRY_PNL_STRUCTURE_MIN_EXPECTED_PROFIT_USDT,
                expected_profit_usdt * max_loss_multiple,
            )
            if structure_max_loss < max_loss:
                previous_max_loss = max_loss
                max_loss = max(structure_max_loss, 1.0)
                pnl_structure_guard = {
                    "applied": True,
                    "previous_max_stop_loss_usdt": round(previous_max_loss, 6),
                    "max_stop_loss_usdt": round(max_loss, 6),
                    "expected_profit_usdt": round(expected_profit_usdt, 6),
                    "expected_net_return_pct": round(expected_net, 6),
                    "max_loss_multiple": round(max_loss_multiple, 6),
                    "quality_tier": quality_tier,
                    "symbol_profit_tier": symbol_profit_tier,
                    "reason": "按预期净收益动态压缩单笔止损预算，避免小盈大亏结构。",
                }

        notional_floor_blocked = ""
        if target_min_notional > 0:
            if low_payoff_quality:
                notional_floor_blocked = "收益质量不足或小盈大亏风险偏高，不抬高仓位"
            elif tail_risk >= 0.88:
                notional_floor_blocked = "尾部风险偏高，不抬高仓位"
            elif weak_history and not high_quality_entry:
                notional_floor_blocked = "该币种/方向近期真实盈亏偏弱，普通信号不抬高仓位"
            elif local_expected < 0 and not (local_aligned or ml_aligned):
                notional_floor_blocked = "服务器盈利模型预期为负且未获得本地模型同向支持，不抬高仓位"
            else:
                risk_max_size = max_loss / max(balance * leverage * stress_stop_loss_pct, 1e-12)
                floor_size = target_min_notional / max(balance * leverage, 1e-12)
                raised_size = min(
                    max(current_size, floor_size),
                    max(risk_max_size, 0.001),
                    ENTRY_NOTIONAL_FLOOR_MAX_SIZE_PCT,
                )
                if raised_size > current_size:
                    current_size = raised_size
                    decision.position_size_pct = current_size

        planned_loss = balance * current_size * leverage * stress_stop_loss_pct
        max_size_pct = max_loss / max(balance * leverage * stress_stop_loss_pct, 1e-12)
        max_size_pct = max(min(max_size_pct, current_size), 0.001)
        raw["profit_risk_sizing"] = {
            "applied": planned_loss > max_loss,
            "risk_mode": risk_mode,
            "original_position_size_pct": round(original_size_before_floor, 6),
            "position_size_pct": round(max_size_pct if planned_loss > max_loss else current_size, 6),
            "planned_stop_loss_usdt": round(planned_loss, 6),
            "max_stop_loss_usdt": round(max_loss, 6),
            "declared_stop_loss_pct": round(stop_loss_pct, 6),
            "stress_stop_loss_pct": round(stress_stop_loss_pct, 6),
            "low_payoff_quality": bool(low_payoff_quality),
            "quality_caps": caps,
            "high_quality_entry": bool(high_quality_entry),
            "high_profit_quality": bool(high_profit_quality),
            "quality_tier": quality_tier,
            "meaningful_size_reason": meaningful_size_reason,
            "same_side_existing_winner": existing_winner,
            "risk_budget_boost": risk_budget_boost,
            "pnl_structure_guard": pnl_structure_guard,
            "notional_floor_applied": current_size > original_size_before_floor,
            "original_notional_usdt": round(original_notional, 6),
            "target_min_notional_usdt": round(target_min_notional, 6),
            "target_min_notional_balance_ratio": round(notional_floor_ratio, 6),
            "notional_floor_reason": notional_floor_reason,
            "notional_floor_blocked": notional_floor_blocked,
            "final_notional_usdt": round(balance * current_size * leverage, 6),
        }
        if planned_loss > max_loss:
            decision.position_size_pct = max_size_pct
            raw["profit_risk_sizing"]["capped_stop_loss_usdt"] = round(
                balance * max_size_pct * leverage * stress_stop_loss_pct,
                6,
            )
            raw["profit_risk_sizing"]["reason"] = (
                "position size capped by stress-stop budget to prevent small-win-big-loss structure"
            )
        decision.raw_response = raw

    def _same_side_existing_winner_context(
        self,
        decision: DecisionOutput,
        open_positions: list[dict] | None,
    ) -> dict[str, Any]:
        """Return compact context when a new entry would add to an already winning side."""
        if not decision.is_entry:
            return {"has_winner": False}
        side = "long" if decision.action == Action.LONG else "short"
        symbol_key = self._normalize_position_symbol(decision.symbol)
        matches = [
            p for p in (open_positions or [])
            if self._normalize_position_symbol(p.get("symbol")) == symbol_key
            and str(p.get("side") or "").lower() == side
            and p.get("is_open", True)
        ]
        if not matches:
            return {"has_winner": False}

        total_notional = 0.0
        total_unrealized = 0.0
        total_quantity = 0.0
        for pos in matches:
            entry = self._safe_float(pos.get("entry_price"), 0.0)
            current = self._safe_float(pos.get("current_price"), entry)
            qty = abs(self._safe_float(pos.get("quantity"), 0.0))
            contract_size = self._safe_float(
                pos.get("contract_size") or pos.get("contractSize"),
                1.0,
            )
            direct_notional = abs(self._safe_float(
                pos.get("notional")
                or pos.get("notional_usd")
                or pos.get("notionalUsd")
                or (pos.get("info") or {}).get("notionalUsd")
                or (pos.get("info") or {}).get("notional")
                or (pos.get("info") or {}).get("posValue"),
                0.0,
            ))
            notional = (
                direct_notional
                if direct_notional > 0
                else qty * max(entry, current, 0.0) * (contract_size if contract_size > 0 else 1.0)
            )
            total_notional += max(notional, 0.0)
            total_quantity += qty
            total_unrealized += self._safe_float(pos.get("unrealized_pnl"), 0.0)

        pnl_ratio = total_unrealized / max(total_notional, 1e-9)
        return {
            "has_winner": bool(total_unrealized > 0),
            "symbol": symbol_key,
            "side": side,
            "positions": len(matches),
            "quantity": round(total_quantity, 8),
            "notional_usdt": round(total_notional, 6),
            "unrealized_pnl": round(total_unrealized, 6),
            "pnl_ratio": round(pnl_ratio, 6),
        }

    def _entry_side_value(self, decision: DecisionOutput) -> str:
        if decision.action == Action.LONG:
            return "long"
        if decision.action == Action.SHORT:
            return "short"
        return "hold"

    def _entry_expert_disagreement(self, decision: DecisionOutput) -> float:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opinions = raw.get("opinions") if isinstance(raw.get("opinions"), list) else []
        side = self._entry_side_value(decision)
        opposite = "short" if side == "long" else "long"
        directional = 0
        opposite_votes = 0
        for opinion in opinions:
            if not isinstance(opinion, dict):
                continue
            action = str(opinion.get("action") or "").lower()
            if action in {"long", "short"}:
                directional += 1
                if action == opposite:
                    opposite_votes += 1
        return opposite_votes / directional if directional else 0.0

    def _ml_ai_direction_conflict(self, decision: DecisionOutput) -> bool:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        ml_signal = raw.get("ml_signal") if isinstance(raw.get("ml_signal"), dict) else {}
        predictions = ml_signal.get("predictions") if isinstance(ml_signal.get("predictions"), list) else []
        primary = predictions[0] if predictions and isinstance(predictions[0], dict) else {}
        ml_side = str(primary.get("best_side") or "").lower()
        return ml_side in {"long", "short"} and ml_side != self._entry_side_value(decision)

    def _extract_high_risk_review_content(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Extract JSON text from OpenAI-compatible responses, including reasoning-model variants."""
        choices = payload.get("choices") if isinstance(payload, dict) else []
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message") if isinstance(choice, dict) and isinstance(choice.get("message"), dict) else {}
        metadata = {
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "usage": payload.get("usage") if isinstance(payload, dict) else None,
        }
        candidates: list[str] = []

        def add_text(value: Any) -> None:
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
            elif isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        parts.append(str(item.get("text") or item.get("content") or ""))
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    candidates.append(joined)

        add_text(message.get("content"))
        add_text(message.get("reasoning_content"))
        add_text(message.get("reasoning"))
        add_text(message.get("output_text"))
        add_text(payload.get("output_text") if isinstance(payload, dict) else None)

        for text in candidates:
            cleaned = self._extract_json_object_text(text)
            if cleaned:
                return cleaned, metadata
        return "", metadata

    def _extract_json_object_text(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return cleaned[start : end + 1].strip()
        return cleaned if cleaned.startswith("{") and cleaned.endswith("}") else ""

    async def _call_high_risk_review_model(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        use_json_mode: bool,
        max_tokens: int,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        request_body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if use_json_mode:
            request_body["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=request_body,
            )
            response.raise_for_status()
            payload = response.json()
        content, metadata = self._extract_high_risk_review_content(payload)
        return payload, content, metadata

    async def _high_risk_review_gate(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict],
    ) -> str | None:
        """Use the online high-risk reviewer only when a trade is genuinely high risk."""
        if not decision.is_entry or not settings.high_risk_review_enabled:
            return None
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        side = self._entry_side_value(decision)
        opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
        quant_probe = raw.get("quant_profit_probe") if isinstance(raw.get("quant_profit_probe"), dict) else {}
        expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = self._safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        loss_probability = self._safe_float(
            quant_probe.get("loss_probability", opportunity.get("server_profit_loss_probability")),
            1.0,
        )
        if (
            quant_probe.get("triggered")
            and expected_net > 0
            and profit_quality >= QUANT_PROFIT_PROBE_MIN_PROFIT_QUALITY_RATIO
            and loss_probability < 0.58
            and self._safe_float(decision.position_size_pct, 0.0) <= 0.04
            and self._safe_float(decision.suggested_leverage, 1.0) <= 5.0
        ):
            raw["high_risk_review"] = {
                "triggered": False,
                "skipped_for_quant_probe": True,
                "reason": "正期望小仓量化探针已通过机会评分，跳过在线高风险复核，交由本地风控和执行检查控制风险。",
                "expected_net_return_pct": round(expected_net, 6),
                "profit_quality_ratio": round(profit_quality, 6),
                "loss_probability": round(loss_probability, 6),
            }
            decision.raw_response = raw
            return None
        reasons: list[str] = []
        leverage = self._safe_float(decision.suggested_leverage, 1.0)
        size_pct = self._safe_float(decision.position_size_pct, 0.0)
        disagreement = self._entry_expert_disagreement(decision)
        ml_conflict = self._ml_ai_direction_conflict(decision)
        if leverage >= 8.0:
            reasons.append(f"high_leverage:{leverage:.1f}x")
        if size_pct >= 0.10:
            reasons.append(f"large_position:{size_pct:.1%}")
        if disagreement >= 0.34:
            reasons.append(f"expert_disagreement:{disagreement:.0%}")
        if ml_conflict:
            reasons.append("ml_ai_direction_conflict")
        symbol_profile = opportunity.get("symbol_side_profile")
        if isinstance(symbol_profile, dict) and self._safe_float(symbol_profile.get("pnl"), 0.0) < -10:
            reasons.append("recent_symbol_side_loss")
        ml_gate = raw.get("ml_profit_quality_gate") if isinstance(raw.get("ml_profit_quality_gate"), dict) else {}
        local_gate = ml_gate.get("local_ai_tools_gate") if isinstance(ml_gate.get("local_ai_tools_gate"), dict) else {}
        if ml_gate.get("local_quant_caution") or str(ml_gate.get("status") or "").startswith("soft_"):
            reasons.append(str(local_gate.get("status") or ml_gate.get("status") or "local_quant_caution"))
        try:
            allocation_state = await self._execution_allocation_state(model_mode)
            if self._safe_float(allocation_state.get("today_risk_pnl"), 0.0) < 0:
                reasons.append("today_recovery_after_loss")
        except Exception:
            allocation_state = {}

        hard_review_required = bool(
            leverage >= 10.0
            or size_pct >= 0.12
            or disagreement >= 0.50
            or (ml_conflict and expected_net <= 0.35)
            or loss_probability >= 0.72
        )

        if not reasons:
            raw["high_risk_review"] = {
                "triggered": False,
                "reasons": [],
                "rule": "only high-risk entries call online reviewer",
            }
            decision.raw_response = raw
            return None

        if not hard_review_required:
            raw["high_risk_review"] = {
                "triggered": False,
                "advisory_reasons": reasons,
                "approved": True,
                "status": "skipped_advisory_only",
                "rule": "online reviewer only has veto power for truly high-risk entries",
                "reason": (
                    "存在历史表现、今日亏损恢复或本地模型谨慎等风险提示，但未达到必须线上复核的级别；"
                    "本次不调用线上高风险复核，不再用普通弱证据否决 AI 开仓。"
                ),
            }
            decision.raw_response = raw
            return None

        review = {
            "triggered": True,
            "reasons": reasons,
            "model": settings.high_risk_review_model,
            "api_base": settings.high_risk_review_api_base,
            "approved": None,
            "status": "pending",
            "hard_review_required": hard_review_required,
        }
        raw["high_risk_review"] = review
        decision.raw_response = raw

        api_base = str(settings.high_risk_review_api_base or "").rstrip("/")
        api_key = str(settings.high_risk_review_api_key or settings.ai_api_key or "").strip()
        model = str(settings.high_risk_review_model or "").strip()
        if not api_base or not model or not api_key:
            review.update({"status": "skipped", "approved": True, "reason": "high-risk reviewer is not fully configured"})
            raw["high_risk_review"] = review
            decision.raw_response = raw
            return None

        prompt = {
            "symbol": decision.symbol,
            "side": side,
            "confidence": decision.confidence,
            "position_size_pct": decision.position_size_pct,
            "leverage": decision.suggested_leverage,
            "stop_loss_pct": decision.stop_loss_pct,
            "take_profit_pct": decision.take_profit_pct,
            "trigger_reasons": reasons,
            "opportunity_score": opportunity,
            "today_pnl": allocation_state.get("today_risk_pnl"),
            "open_position_count": len([p for p in open_positions or [] if p.get("is_open", True)]),
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a high-risk crypto trade reviewer and a JSON API. "
                    "Return exactly one valid JSON object with keys: "
                    "approved(boolean), confidence(number 0-1), reason(string in Simplified Chinese). "
                    "Reject only when expected net profit is poor, risk is asymmetric, or evidence conflicts."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "Return only one minified JSON object. No markdown, no reasoning, no prose. "
                    "Schema: {\"approved\":true|false,\"confidence\":0.0,\"reason\":\"简体中文，60字以内\"}."
                ),
            },
            {
                "role": "user",
                "content": (
                    "复核这笔高风险加密货币开仓。只输出 JSON："
                    + json.dumps(prompt, ensure_ascii=False)
                ),
            },
        ]
        try:
            attempts: list[dict[str, Any]] = []
            content = ""
            metadata: dict[str, Any] = {}
            for attempt_no, attempt in enumerate(
                (
                    {"messages": messages, "use_json_mode": True, "max_tokens": 1600},
                    {"messages": retry_messages, "use_json_mode": False, "max_tokens": 900},
                ),
                start=1,
            ):
                _payload, content, metadata = await self._call_high_risk_review_model(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    messages=attempt["messages"],
                    use_json_mode=bool(attempt["use_json_mode"]),
                    max_tokens=int(attempt["max_tokens"]),
                )
                attempts.append({
                    "attempt": attempt_no,
                    "json_mode": bool(attempt["use_json_mode"]),
                    "max_tokens": int(attempt["max_tokens"]),
                    "finish_reason": metadata.get("finish_reason"),
                    "content_present": bool(content),
                    "usage": metadata.get("usage"),
                })
                if content:
                    break
            if not content:
                finish_reason = metadata.get("finish_reason") or "unknown"
                raise ValueError(f"模型两次都没有返回可解析 JSON，finish_reason={finish_reason}")
            parsed = json.loads(content)
            approved = bool(parsed.get("approved"))
            review.update({
                "status": "completed",
                "approved": approved,
                "confidence": self._safe_float(parsed.get("confidence"), 0.0),
                "reason": str(parsed.get("reason") or "")[:500],
                "attempts": attempts,
            })
            raw["high_risk_review"] = review
            decision.raw_response = raw
            if not approved:
                return f"高风险复核否决：{review.get('reason') or '线上复核认为该交易盈亏比或证据质量不足'}"
        except Exception as exc:
            if hard_review_required:
                review.update({
                    "status": "error_blocked",
                    "approved": False,
                    "reason": (
                        f"高风险复核调用失败：{str(exc)[:180]}。"
                        "本次属于必须线上复核的大仓/高杠杆/严重冲突开仓，未完成复核前不提交订单。"
                    ),
                })
                raw["high_risk_review"] = review
                decision.raw_response = raw
                return str(review["reason"])
            original_size = self._safe_float(decision.position_size_pct, 0.0)
            original_leverage = self._safe_float(decision.suggested_leverage, 1.0)
            decision.position_size_pct = min(original_size or 0.02, 0.025)
            decision.suggested_leverage = min(original_leverage or 3.0, 5.0)
            review.update({
                "status": "error_downgraded",
                "approved": True,
                "reason": (
                    f"高风险复核调用失败：{str(exc)[:180]}。"
                    "这笔不是必须线上复核的极端高风险交易，已降仓降杠杆后继续走本地风控和 OKX 提交检查。"
                ),
                "original_position_size_pct": round(original_size, 6),
                "adjusted_position_size_pct": round(float(decision.position_size_pct or 0.0), 6),
                "original_leverage": round(original_leverage, 4),
                "adjusted_leverage": round(float(decision.suggested_leverage or 1.0), 4),
            })
            raw["high_risk_review"] = review
            decision.raw_response = raw
            return None
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
    ) -> ExecutionResult | None:
        return await self.execution_service.execute_candidate(
            symbol,
            model_name,
            decision,
            assessment,
            decision_db_id,
            results,
            open_positions=open_positions,
        )

    async def _execute_candidate_locked(
        self,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        assessment,
        decision_db_id: int | None,
        results: dict[str, Any],
        open_positions: list[dict] | None = None,
    ) -> ExecutionResult | None:
        return await self.execution_service.execute_candidate_locked(
            symbol,
            model_name,
            decision,
            assessment,
            decision_db_id,
            results,
            open_positions=open_positions,
        )

    async def _exit_fee_churn_guard_reason(
        self,
        model_name: str,
        decision: DecisionOutput,
    ) -> str | None:
        """Skip weak discretionary closes that would only lock in fee drag."""
        if not decision.is_exit:
            return None

        target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
        model_mode = self._get_model_execution_mode(model_name)
        snapshot = decision.feature_snapshot or {}
        current_price = self._safe_float(
            snapshot.get("current_price", snapshot.get("close", 0.0)),
            0.0,
        )

        try:
            async with get_session_ctx() as session:
                repo = TradeRepository(session)
                positions = await repo.get_matching_open_positions(
                    model_name=model_name,
                    symbol=decision.symbol,
                    side=target_side,
                    execution_mode=model_mode,
                )
                if not positions:
                    return None

                raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                close_evidence = raw.get("close_evidence") if isinstance(raw.get("close_evidence"), dict) else {}
                execution_profit = (
                    raw.get("execution_profit_protection")
                    if isinstance(raw.get("execution_profit_protection"), dict)
                    else {}
                )
                fast_trigger = str(raw.get("fast_risk_trigger") or "")
                reasoning_text = str(decision.reasoning or "")
                profit_exit_intent = bool(
                    close_evidence.get("profit_protection")
                    or execution_profit.get("allow")
                    or fast_trigger.startswith("profit_drawdown")
                    or any(term in reasoning_text for term in ("锁盈", "利润保护", "浮盈", "止盈"))
                )
                loss_exit_intent = bool(
                    any(term in reasoning_text for term in ("亏损", "浮亏", "扩亏", "止损", "未实现盈亏为负"))
                    or close_evidence.get("loss_repair")
                    or close_evidence.get("loss_repair_evidence")
                )

                aggregate_qty = 0.0
                aggregate_entry_value = 0.0
                aggregate_gross_pnl = 0.0
                aggregate_entry_fee = 0.0
                aggregate_close_fee = 0.0
                aggregate_hit_stop = False
                aggregate_hit_profit = False
                for aggregate_pos in positions:
                    pos_qty = abs(float(aggregate_pos.quantity or 0.0))
                    pos_entry = float(aggregate_pos.entry_price or 0.0)
                    if pos_qty <= 0 or pos_entry <= 0:
                        continue
                    aggregate_qty += pos_qty
                    aggregate_entry_value += pos_entry * pos_qty
                    if current_price <= 0:
                        current_price = float(aggregate_pos.current_price or pos_entry)
                    pos_gross = (
                        (current_price - pos_entry) * pos_qty
                        if target_side == "long"
                        else (pos_entry - current_price) * pos_qty
                    )
                    aggregate_gross_pnl += pos_gross
                    aggregate_entry_fee += await self._entry_fee_for_position(session, aggregate_pos, pos_qty)
                    aggregate_close_fee += abs(current_price * pos_qty) * ESTIMATED_TAKER_FEE_PCT
                    if target_side == "long":
                        aggregate_hit_stop = aggregate_hit_stop or bool(
                            aggregate_pos.stop_loss_price and current_price <= aggregate_pos.stop_loss_price
                        )
                        aggregate_hit_profit = aggregate_hit_profit or bool(
                            aggregate_pos.take_profit_price and current_price >= aggregate_pos.take_profit_price
                        )
                    else:
                        aggregate_hit_stop = aggregate_hit_stop or bool(
                            aggregate_pos.stop_loss_price and current_price >= aggregate_pos.stop_loss_price
                        )
                        aggregate_hit_profit = aggregate_hit_profit or bool(
                            aggregate_pos.take_profit_price and current_price <= aggregate_pos.take_profit_price
                        )

                aggregate_entry = aggregate_entry_value / max(aggregate_qty, 1e-9)
                aggregate_net_pnl = aggregate_gross_pnl - aggregate_entry_fee - aggregate_close_fee
                aggregate_invalidation = self._exit_invalidation_snapshot(
                    decision,
                    target_side,
                    aggregate_entry,
                    current_price,
                ) if aggregate_qty > 0 and aggregate_entry > 0 and current_price > 0 else {}
                forced_exit = self._is_forced_exit_decision(decision)
                if (
                    aggregate_qty > 0
                    and aggregate_gross_pnl >= 0
                    and loss_exit_intent
                    and not profit_exit_intent
                    and not forced_exit
                    and not aggregate_hit_stop
                    and not aggregate_hit_profit
                    and not bool(aggregate_invalidation.get("severe"))
                ):
                    raw["aggregate_exit_guard"] = {
                        "applied": True,
                        "target_side": target_side,
                        "fragments": len(positions),
                        "current_price": round(current_price, 8),
                        "aggregate_entry_price": round(aggregate_entry, 8),
                        "aggregate_gross_pnl": round(aggregate_gross_pnl, 6),
                        "aggregate_net_pnl": round(aggregate_net_pnl, 6),
                        "loss_exit_intent": True,
                        "profit_exit_intent": False,
                        "reason": "同币种同方向整体不亏，禁止按单个分片浮亏触发亏损修复平仓。",
                    }
                    decision.raw_response = raw
                    return (
                        f"整体持仓保护：{decision.symbol} {target_side} 当前共有 {len(positions)} 个分片，"
                        f"按整体均价 {aggregate_entry:.6g} 和最新价 {current_price:.6g} 估算整体浮盈 "
                        f"{aggregate_gross_pnl:.4f}U（扣费后约 {aggregate_net_pnl:.4f}U）。"
                        "本次平仓理由来自单个分片亏损/止损描述，但整体仓位并不亏，"
                        "未触发硬止损、止盈或严重趋势失效，因此不执行该亏损平仓。"
                    )

                pos = positions[0]
                qty = float(pos.quantity or 0.0)
                entry_price = float(pos.entry_price or 0.0)
                if qty <= 0 or entry_price <= 0:
                    return None
                if current_price <= 0:
                    current_price = float(pos.current_price or entry_price)

                notional = qty * entry_price
                if target_side == "long":
                    gross_pnl = (current_price - entry_price) * qty
                    hit_stop = bool(pos.stop_loss_price and current_price <= pos.stop_loss_price)
                    hit_profit = bool(pos.take_profit_price and current_price >= pos.take_profit_price)
                else:
                    gross_pnl = (entry_price - current_price) * qty
                    hit_stop = bool(pos.stop_loss_price and current_price >= pos.stop_loss_price)
                    hit_profit = bool(pos.take_profit_price and current_price <= pos.take_profit_price)

                if hit_stop or hit_profit:
                    return None

                hard_stop_loss = -abs(notional) * float(settings.hard_stop_loss_pct or 0.05)
                if gross_pnl <= hard_stop_loss:
                    return None

                invalidation = self._exit_invalidation_snapshot(
                    decision,
                    target_side,
                    entry_price,
                    current_price,
                )
                severe_invalidation = bool(invalidation.get("severe"))

                entry_fee = await self._entry_fee_for_position(session, pos, qty)
                estimated_close_fee = max(
                    abs(current_price * qty) * ESTIMATED_TAKER_FEE_PCT,
                    entry_fee if entry_fee > 0 else 0.0,
                )
                net_now = gross_pnl - entry_fee - estimated_close_fee

                opened_at = pos.created_at
                if opened_at and opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=timezone.utc)
                age_minutes = (
                    (datetime.now(timezone.utc) - opened_at).total_seconds() / 60.0
                    if opened_at
                    else 9999.0
                )
                fee_buffer = entry_fee + estimated_close_fee
                fee_coverage_multiple = net_now / max(fee_buffer, 1e-9)
                continuation_valid = not bool(
                    invalidation.get("key_break")
                    or invalidation.get("trend_reversal")
                    or (invalidation.get("momentum_bad") and invalidation.get("volume_confirms"))
                )
                peak_state = self._position_profit_peaks.get(
                    self._position_peak_key(model_name, pos.symbol, target_side),
                    {},
                )
                peak_net = self._safe_float(peak_state.get("peak_unrealized_pnl"), gross_pnl)
                drawdown_from_peak = max(peak_net - gross_pnl, 0.0)
                drawdown_ratio = drawdown_from_peak / max(abs(peak_net), 1e-9) if peak_net > 0 else 0.0
                raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                raw["exit_quality"] = {
                    "net_profit_after_fee": round(net_now, 8),
                    "fee_coverage_multiple": round(fee_coverage_multiple, 4),
                    "continuation_score": round(0.0 if severe_invalidation else (0.75 if continuation_valid else 0.35), 4),
                    "trend_still_valid": bool(continuation_valid),
                    "drawdown_from_peak": round(drawdown_from_peak, 8),
                    "drawdown_from_peak_ratio": round(drawdown_ratio, 6),
                    "invalidation": invalidation,
                }
                decision.raw_response = raw

                forced_exit = self._is_forced_exit_decision(decision)
                if (
                    age_minutes * 60.0 < ENTRY_SETTLEMENT_EXIT_GUARD_SECONDS
                    and not hit_stop
                    and not hit_profit
                    and not forced_exit
                ):
                    return (
                        f"平仓保护：该仓位刚开仓 {age_minutes:.2f} 分钟，仍在成交结算防抖窗口内，"
                        "普通 AI 减仓、止盈或降低风险建议暂不执行，避免开仓后同轮或数秒内反复部分平仓。"
                    )

                if forced_exit:
                    return None

                if severe_invalidation:
                    return None

                protected_by_okx = bool(pos.stop_loss_price or pos.take_profit_price)
                confidence = float(decision.confidence or 0.0)
                fee_buffer = entry_fee + estimated_close_fee
                invalidation_confirmed = bool(
                    (
                        invalidation.get("key_break")
                        and invalidation.get("momentum_bad")
                    )
                    or (
                        invalidation.get("trend_reversal")
                        and invalidation.get("momentum_bad")
                    )
                )

                if age_minutes < MIN_DISCRETIONARY_HOLD_MINUTES and not invalidation_confirmed:
                    early_strong_profit = (
                        net_now > 0
                        and abs(notional) > 0
                        and net_now >= max(
                            abs(notional) * PROFIT_PROTECTION_STRONG_NET_PNL_RATIO,
                            fee_buffer * PROFIT_PROTECTION_STRONG_FEE_MULTIPLE,
                        )
                        and confidence >= DISCRETIONARY_CLOSE_CONFIDENCE
                    )
                    early_deep_loss = (
                        net_now < 0
                        and abs(notional) > 0
                        and abs(net_now) >= abs(notional) * FAST_RISK_REDUCE_LOSS_PCT
                        and confidence >= DISCRETIONARY_CLOSE_CONFIDENCE
                    )
                    if not early_strong_profit and not early_deep_loss:
                        return (
                            f"平仓保护：该仓位只持有 {age_minutes:.1f} 分钟，"
                            "未触发止损、止盈或趋势严重失效。普通 AI 平仓建议暂不执行，"
                            "避免刚开仓就因短线噪音频繁切仓。"
                        )

                profit_protection = self._exit_profit_protection_state(
                    net_now=net_now,
                    notional=notional,
                    fee_buffer=fee_buffer,
                    confidence=confidence,
                    age_minutes=age_minutes,
                )
                if profit_protection["allow"]:
                    close_pct = min(max(float(decision.position_size_pct or 1.0), 0.0), 1.0)
                    planned_lock_net = net_now * close_pct
                    meaningful_partial_lock = max(
                        PROFIT_PROTECTION_MIN_NET_USDT,
                        fee_buffer * max(PROFIT_PROTECTION_MIN_FEE_MULTIPLE, 6.0),
                        abs(notional) * max(PROFIT_PROTECTION_MIN_NET_PNL_RATIO, 0.008),
                    )
                    if 0.0 < close_pct < 0.999 and planned_lock_net < meaningful_partial_lock:
                        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                        raw["execution_profit_protection"] = profit_protection
                        raw["small_profit_lock_guard"] = {
                            "applied": True,
                            "close_pct": round(close_pct, 4),
                            "net_profit_after_fee": round(net_now, 8),
                            "planned_lock_net": round(planned_lock_net, 8),
                            "meaningful_partial_lock": round(meaningful_partial_lock, 8),
                            "reason": "本次部分锁盈预计落袋利润太小，继续持有等待更有意义的锁盈或明确反转。",
                        }
                        decision.raw_response = raw
                        return (
                            f"锁盈保护：当前整仓扣费后预计盈利 {net_now:.4f} USDT，"
                            f"但本次只计划平 {close_pct:.0%}，预计实际落袋约 {planned_lock_net:.4f} USDT，"
                            f"低于动态有效锁盈线 {meaningful_partial_lock:.4f} USDT。"
                            "为避免碎片化小额平仓和手续费消耗，本次不执行普通部分锁盈；"
                            "等待更大浮盈、明确回撤/反转，或交易所止盈止损触发。"
                        )
                    raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                    raw["execution_profit_protection"] = profit_protection
                    decision.raw_response = raw
                    return None

                if protected_by_okx and confidence < DISCRETIONARY_CLOSE_CONFIDENCE:
                    if net_now > 0:
                        return (
                            "平仓保护：该仓位已有 OKX 止盈/止损托底，当前扣费后仍盈利，"
                            f"但主动平仓信号强度只有 {confidence:.0%}，趋势尚未确认失效，继续持有。"
                        )
                    return (
                        "平仓保护：该仓位已经有 OKX 止盈/止损保护，"
                        f"当前信号强度 {confidence:.0%}，未达到主动干预门槛 "
                        f"{DISCRETIONARY_CLOSE_CONFIDENCE:.0%}。"
                        "为避免和交易所止盈止损互相打架，本轮继续持有。"
                    )

                if net_now < 0 and confidence < DISCRETIONARY_CLOSE_CONFIDENCE:
                    invalidation_pressure = self._exit_invalidation_snapshot(
                        decision,
                        target_side,
                        entry_price,
                        current_price,
                    )
                    if (
                        invalidation_pressure.get("key_break")
                        and invalidation_pressure.get("momentum_bad")
                    ) or (
                        invalidation_pressure.get("trend_reversal")
                        and invalidation_pressure.get("momentum_bad")
                    ):
                        return None
                    return (
                        "平仓保护：当前按手续费后预计净亏"
                        f" {net_now:.4f} USDT，未触发硬止损或止盈。"
                        f"持仓 {age_minutes:.1f} 分钟，信号强度 {confidence:.0%}。"
                        "小幅浮亏不会直接平仓，需要关键位跌破、量能恶化或趋势反转确认。"
                    )

                if net_now > 0 and net_now < fee_buffer * 1.5:
                    return (
                        "平仓保护：当前盈利尚未明显覆盖双边手续费，"
                        f"预计净盈亏 {net_now:.4f} USDT，持仓 {age_minutes:.1f} 分钟，"
                        "先继续观察，等待止盈、止损或更强信号。"
                    )
                if net_now > 0 and continuation_valid and drawdown_ratio < PROFIT_DRAWDOWN_PARTIAL_RETRACE and confidence < 0.82:
                    return (
                        "平仓保护：扣费后仍有盈利，但趋势延续证据没有失效，"
                        f"净收益 {net_now:.4f} USDT，手续费覆盖 {fee_coverage_multiple:.1f} 倍，"
                        f"峰值回撤 {drawdown_ratio:.0%}。本轮继续持有，等待明确反转或明显回撤再平。"
                    )
        except Exception as e:
            logger.warning("exit fee churn guard failed", symbol=decision.symbol, error=str(e))
        return None

    def _exit_profit_protection_state(
        self,
        *,
        net_now: float,
        notional: float,
        fee_buffer: float,
        confidence: float,
        age_minutes: float,
    ) -> dict[str, Any]:
        """Allow profitable exits with dynamic thresholds instead of a fixed USDT target."""
        abs_notional = abs(float(notional or 0.0))
        fee_buffer = max(float(fee_buffer or 0.0), 0.0)
        net_now = float(net_now or 0.0)
        confidence = float(confidence or 0.0)
        pnl_ratio = net_now / abs_notional if abs_notional > 0 else 0.0
        min_net_profit = max(
            abs_notional * PROFIT_PROTECTION_MIN_NET_PNL_RATIO,
            fee_buffer * PROFIT_PROTECTION_MIN_FEE_MULTIPLE,
            PROFIT_PROTECTION_MIN_NET_USDT,
        )
        strong_net_profit = max(
            abs_notional * PROFIT_PROTECTION_STRONG_NET_PNL_RATIO,
            fee_buffer * PROFIT_PROTECTION_STRONG_FEE_MULTIPLE,
            min_net_profit * 1.5,
        )
        mature_enough = age_minutes >= MIN_DISCRETIONARY_HOLD_MINUTES
        normal_lock = mature_enough and net_now >= min_net_profit
        strong_lock = (
            net_now >= strong_net_profit
            and confidence >= DISCRETIONARY_CLOSE_CONFIDENCE
        )
        early_lock = False
        return {
            "allow": bool(net_now > 0 and (normal_lock or strong_lock or early_lock)),
            "net_pnl": round(net_now, 8),
            "pnl_ratio": round(pnl_ratio, 6),
            "notional": round(abs_notional, 8),
            "fee_buffer": round(fee_buffer, 8),
            "min_net_profit": round(min_net_profit, 8),
            "strong_net_profit": round(strong_net_profit, 8),
            "confidence": round(confidence, 4),
            "age_minutes": round(age_minutes, 3),
            "mature_enough": mature_enough,
            "normal_lock": bool(normal_lock),
            "strong_lock": bool(strong_lock),
            "early_lock": bool(early_lock),
            "rule": (
                "按仓位名义价值比例和手续费倍数动态判断锁盈，"
                "不使用固定 8-10U 作为大浮盈门槛。"
            ),
        }

    def _exit_invalidation_snapshot(
        self,
        decision: DecisionOutput,
        target_side: str,
        entry_price: float,
        current_price: float,
    ) -> dict[str, Any]:
        """Detect clear thesis invalidation before allowing discretionary exits."""
        snapshot = decision.feature_snapshot or {}
        atr_14 = self._safe_float(snapshot.get("atr_14"), 0.0)
        ema_12 = self._safe_float(snapshot.get("ema_12"), 0.0)
        ema_26 = self._safe_float(snapshot.get("ema_26"), 0.0)
        returns_5 = self._safe_float(snapshot.get("returns_5"), 0.0)
        returns_20 = self._safe_float(snapshot.get("returns_20"), 0.0)
        volume_ratio = self._safe_float(snapshot.get("volume_ratio"), 0.0)
        price_vs_sma20 = self._safe_float(snapshot.get("price_vs_sma20"), 0.0)
        price_vs_sma50 = self._safe_float(snapshot.get("price_vs_sma50"), 0.0)

        atr_break = atr_14 * 1.2 if atr_14 > 0 else 0.0
        pct_break = abs(entry_price) * 0.012
        break_distance = max(atr_break, pct_break)

        if target_side == "long":
            key_break = current_price <= entry_price - break_distance or price_vs_sma20 <= -0.006
            trend_reversal = (
                ema_12 > 0
                and ema_26 > 0
                and ema_12 < ema_26
                and price_vs_sma20 < -0.003
                and price_vs_sma50 <= 0
            )
            momentum_bad = returns_5 <= -0.006 or returns_20 <= -0.012
        else:
            key_break = current_price >= entry_price + break_distance or price_vs_sma20 >= 0.006
            trend_reversal = (
                ema_12 > 0
                and ema_26 > 0
                and ema_12 > ema_26
                and price_vs_sma20 > 0.003
                and price_vs_sma50 >= 0
            )
            momentum_bad = returns_5 >= 0.006 or returns_20 >= 0.012

        volume_confirms = volume_ratio >= max(float(settings.min_entry_volume_ratio or 1.0), 1.2)
        severe = (
            key_break and trend_reversal
        ) or (
            key_break and momentum_bad and volume_confirms
        ) or (
            trend_reversal and momentum_bad and volume_confirms
        )
        reasons = []
        if key_break:
            reasons.append("key_break")
        if trend_reversal:
            reasons.append("trend_reversal")
        if momentum_bad:
            reasons.append("momentum_bad")
        if volume_confirms:
            reasons.append("volume_confirms")
        return {
            "severe": severe,
            "key_break": key_break,
            "trend_reversal": trend_reversal,
            "momentum_bad": momentum_bad,
            "volume_confirms": volume_confirms,
            "reason": ";".join(reasons) if reasons else "no severe invalidation",
        }

    def _is_forced_exit_decision(self, decision: DecisionOutput) -> bool:
        text = str(decision.reasoning or "").upper()
        raw_text = str(decision.reasoning or "")
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        close_evidence = raw.get("close_evidence") if isinstance(raw.get("close_evidence"), dict) else {}
        forced_terms = (
            "STOP LOSS",
            "BLACK SWAN",
            "HARD STOP",
            "CRITICAL",
            "强制平仓",
            "硬止损",
            "极端风险",
            "黑天鹅",
            "熔断",
            "触发止损",
            "触发止盈",
            "止损触发",
            "止盈触发",
            "快速风控",
        )
        return (
            bool(raw.get("fast_risk_exit") or raw.get("forced_exit"))
            or bool(close_evidence.get("hard_risk") or close_evidence.get("forced_exit"))
            or bool(
                (raw.get("position_review_risk_alert") or {}).get("force_exit")
                if isinstance(raw.get("position_review_risk_alert"), dict)
                else False
            )
            or decision.model_name == "risk_engine"
            or any(term in text for term in forced_terms[:4])
            or any(term in raw_text for term in forced_terms[4:])
        )

    def _exit_group_key(self, model_name: str, decision: DecisionOutput) -> tuple[str, str, str]:
        side = "long" if decision.action == Action.CLOSE_LONG else "short" if decision.action == Action.CLOSE_SHORT else ""
        return (
            "all",
            self._normalize_position_symbol(decision.symbol),
            side,
        )

    def _exit_cooldown_bypass(self, decision: DecisionOutput) -> bool:
        if not decision.is_exit:
            return True
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        fast_trigger = str(raw.get("fast_risk_trigger") or "")
        close_fraction = self._safe_float(
            raw.get("close_fraction")
            if raw.get("close_fraction") is not None
            else decision.position_size_pct,
            1.0,
        )
        close_evidence = raw.get("close_evidence") if isinstance(raw.get("close_evidence"), dict) else {}
        exit_quality = raw.get("exit_quality") if isinstance(raw.get("exit_quality"), dict) else {}
        invalidation = exit_quality.get("invalidation") if isinstance(exit_quality.get("invalidation"), dict) else {}
        if fast_trigger in {"stop_loss", "take_profit", "near_stop_progress", "hard_adverse_move"}:
            return True
        if fast_trigger == "fast_adverse_move" and close_fraction >= 0.999:
            return True
        if bool(raw.get("forced_exit") or close_evidence.get("hard_risk") or close_evidence.get("forced_exit")):
            return True
        if bool(invalidation.get("severe")):
            return True
        if decision.model_name == "risk_engine":
            return True
        return False

    def _recent_exit_cooldown_reason(self, model_name: str, decision: DecisionOutput) -> str | None:
        if not decision.is_exit or self._exit_cooldown_bypass(decision):
            return None
        key = self._exit_group_key(model_name, decision)
        if not all(key):
            return None
        last_at = self._recent_exit_groups.get(key)
        if not isinstance(last_at, datetime):
            return None
        elapsed = max((datetime.now(timezone.utc) - last_at).total_seconds(), 0.0)
        if elapsed >= EXIT_SYMBOL_SIDE_COOLDOWN_SECONDS:
            return None
        remaining = max(EXIT_SYMBOL_SIDE_COOLDOWN_SECONDS - elapsed, 0.0)
        side_label = "做多" if key[2] == "long" else "做空"
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["recent_exit_cooldown"] = {
            "applied": True,
            "symbol": key[1],
            "side": key[2],
            "elapsed_seconds": round(elapsed, 3),
            "cooldown_seconds": EXIT_SYMBOL_SIDE_COOLDOWN_SECONDS,
            "bypass": False,
            "reason": "同一币种同方向刚发生过平仓，普通平仓信号短时间内不连续执行。",
        }
        decision.raw_response = raw
        return (
            f"连续平仓冷却：{key[1]} {side_label} 最近 {elapsed:.0f} 秒内已经执行过一次平仓，"
            f"普通减仓/策略平仓需再等待约 {remaining:.0f} 秒。"
            "硬止损、真实止盈、严重趋势失效或强制风险平仓不受此限制。"
        )

    def _remember_recent_exit_group(self, model_name: str, decision: DecisionOutput) -> None:
        if not decision.is_exit:
            return
        key = self._exit_group_key(model_name, decision)
        if all(key):
            self._recent_exit_groups[key] = datetime.now(timezone.utc)

    def _matching_exit_context_positions(
        self,
        open_positions: list[dict] | None,
        model_name: str,
        decision: DecisionOutput,
    ) -> list[dict]:
        if not decision.is_exit:
            return []
        target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
        target_symbol = self._normalize_position_symbol(decision.symbol)
        matches: list[dict] = []
        for pos in open_positions or []:
            pos_model = str(pos.get("model_name") or "")
            if pos_model and pos_model != model_name:
                continue
            if self._normalize_position_symbol(pos.get("symbol")) != target_symbol:
                continue
            if str(pos.get("side") or "").lower() != target_side:
                continue
            if pos.get("is_open", True) is False:
                continue
            quantity = self._safe_float(
                pos.get("quantity") or pos.get("contracts") or pos.get("sz"),
                0.0,
            )
            if abs(quantity) > 0:
                matches.append(pos)
        return matches

    def _loss_partial_exit_guard_reason(
        self,
        model_name: str,
        decision: DecisionOutput,
        open_positions: list[dict] | None,
    ) -> str | None:
        """Block ordinary partial closes while the whole symbol-side position is losing."""
        if not decision.is_exit:
            return None

        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        close_evidence = raw.get("close_evidence") if isinstance(raw.get("close_evidence"), dict) else {}
        fast_trigger = str(raw.get("fast_risk_trigger") or "")
        close_fraction = self._safe_float(
            raw.get("close_fraction")
            if raw.get("close_fraction") is not None
            else decision.position_size_pct,
            1.0,
        )
        action_plan = str(
            raw.get("action_plan")
            or close_evidence.get("action_plan")
            or raw.get("exit_action_plan")
            or ""
        ).lower()
        partial_intent = (0.0 < close_fraction < 0.999) or action_plan == "reduce"
        if not partial_intent:
            return None

        exit_quality = raw.get("exit_quality") if isinstance(raw.get("exit_quality"), dict) else {}
        invalidation = exit_quality.get("invalidation") if isinstance(exit_quality.get("invalidation"), dict) else {}
        forced_hard_exit = bool(
            fast_trigger in {"stop_loss", "take_profit", "hard_adverse_move"}
            or (fast_trigger in {"near_stop_progress", "fast_adverse_move"} and close_fraction >= 0.999)
            or raw.get("forced_exit")
            or close_evidence.get("hard_risk")
            or close_evidence.get("forced_exit")
            or invalidation.get("severe")
            or decision.model_name == "risk_engine"
        )
        if forced_hard_exit:
            return None

        matches = self._matching_exit_context_positions(open_positions, model_name, decision)
        if not matches:
            return None

        target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
        latest_price = self._safe_float(
            (decision.feature_snapshot or {}).get("current_price")
            or (decision.feature_snapshot or {}).get("close"),
            0.0,
        )
        total_qty = 0.0
        entry_value = 0.0
        estimated_gross = 0.0
        reported_unrealized = 0.0
        reported_available = False
        for pos in matches:
            qty = abs(self._safe_float(pos.get("quantity") or pos.get("contracts") or pos.get("sz"), 0.0))
            contract_size = self._safe_float(
                pos.get("contract_size")
                or pos.get("contractSize")
                or (pos.get("info") or {}).get("ctVal"),
                1.0,
            )
            qty_for_pnl = qty * (contract_size if contract_size > 0 else 1.0)
            entry = self._safe_float(pos.get("entry_price") or pos.get("entryPrice") or pos.get("avgPx"), 0.0)
            current = self._safe_float(
                pos.get("current_price")
                or pos.get("markPrice")
                or pos.get("lastPrice")
                or entry,
                entry,
            )
            if latest_price <= 0 and current > 0:
                latest_price = current
            if qty_for_pnl <= 0 or entry <= 0:
                continue
            total_qty += qty_for_pnl
            entry_value += entry * qty_for_pnl
            mark = latest_price if latest_price > 0 else current
            if mark > 0:
                estimated_gross += (
                    (mark - entry) * qty_for_pnl
                    if target_side == "long"
                    else (entry - mark) * qty_for_pnl
                )
            unrealized = self._safe_float(
                pos.get("unrealized_pnl")
                or pos.get("unrealizedPnl")
                or pos.get("upl")
                or (pos.get("info") or {}).get("upl")
                or (pos.get("info") or {}).get("unrealizedPnl"),
                0.0,
            )
            if abs(unrealized) > 1e-12:
                reported_available = True
                reported_unrealized += unrealized

        if total_qty <= 0:
            return None

        aggregate_entry = entry_value / max(total_qty, 1e-12)
        aggregate_pnl = reported_unrealized if reported_available else estimated_gross
        if aggregate_pnl >= -1e-9:
            return None

        raw["loss_partial_exit_guard"] = {
            "applied": True,
            "symbol": self._normalize_position_symbol(decision.symbol),
            "side": target_side,
            "fragments": len(matches),
            "close_fraction": round(close_fraction, 6),
            "fast_risk_trigger": fast_trigger,
            "aggregate_entry_price": round(aggregate_entry, 8),
            "latest_price": round(latest_price, 8),
            "aggregate_unrealized_pnl": round(aggregate_pnl, 6),
            "reason": "亏损仓位的普通部分平仓已禁用，避免切碎仓位后错过后续止盈。",
        }
        decision.raw_response = raw
        side_label = "做多" if target_side == "long" else "做空"
        return (
            f"亏损部分平仓保护：{decision.symbol} {side_label} 当前按整体持仓估算仍浮亏 "
            f"{aggregate_pnl:.4f}U，本次只计划平 {close_fraction:.0%}。"
            "系统已禁用普通亏损部分平仓，避免在回撤中反复切碎仓位，导致后续真正触发止盈时剩余仓位太小。"
            "若触发硬止损、真实止盈、接近/超过计划止损、严重趋势失效或风控强制全平，仍会允许执行。"
        )

    def _has_matching_exit_position(
        self,
        open_positions: list[dict] | None,
        model_name: str,
        decision: DecisionOutput,
    ) -> bool:
        if not decision.is_exit:
            return True
        target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
        target_symbol = self._normalize_position_symbol(decision.symbol)
        for pos in open_positions or []:
            if str(pos.get("model_name") or "") != model_name:
                continue
            if self._normalize_position_symbol(pos.get("symbol")) != target_symbol:
                continue
            if str(pos.get("side") or "").lower() != target_side:
                continue
            if pos.get("is_open", True) is False:
                continue
            try:
                quantity = float(pos.get("quantity", pos.get("contracts", 0)) or 0)
            except (TypeError, ValueError):
                quantity = 0.0
            if quantity > 0:
                return True
        return False

    async def _has_matching_exchange_exit_position(
        self,
        model_name: str,
        decision: DecisionOutput,
    ) -> bool:
        return await self.okx_sync_service.has_matching_exchange_exit_position(model_name, decision)

    def _is_no_exchange_position_error(self, message: Any) -> bool:
        text = str(message or "").lower()
        return (
            "51169" in text
            or "don't have any positions in this direction" in text
            or "no matching position to close" in text
            or "没有对应方向" in text
            or "没有可平" in text
            or "可平仓位" in text
        )

    def _result_has_no_exchange_position(self, result: ExecutionResult | None) -> bool:
        if result is None:
            return False
        raw = result.raw_response or {}
        pieces = [result.order_id, result.exchange_order_id]
        if isinstance(raw, dict):
            pieces.extend([raw.get("error"), raw.get("raw_error")])
        return self._is_no_exchange_position_error(" ".join(str(p or "") for p in pieces))

    def _no_matching_exit_position_reason(self, decision: DecisionOutput) -> str:
        side_label = "多单" if decision.action == Action.CLOSE_LONG else "空单"
        return f"没有找到 {decision.symbol} 对应的可平{side_label}仓位，未向 OKX 提交平仓单。"

    def _apply_execution_to_open_positions(
        self,
        open_positions: list[dict],
        model_name: str,
        decision: DecisionOutput,
        execution_result: ExecutionResult,
    ) -> None:
        if execution_result.status != OrderStatus.FILLED and not self._is_exit_progress_execution(execution_result):
            return

        if decision.action in (Action.LONG, Action.SHORT):
            if execution_result.status != OrderStatus.FILLED:
                return
            open_positions.append({
                "model_name": model_name,
                "symbol": decision.symbol,
                "side": "long" if decision.action == Action.LONG else "short",
                "entry_price": execution_result.price,
                "current_price": execution_result.price,
                "quantity": execution_result.quantity,
                "unrealized_pnl": 0.0,
                "stop_loss": (
                    execution_result.price * (1 - decision.stop_loss_pct)
                    if decision.action == Action.LONG
                    else execution_result.price * (1 + decision.stop_loss_pct)
                ),
                "take_profit": (
                    execution_result.price * (1 + decision.take_profit_pct)
                    if decision.action == Action.LONG
                    else execution_result.price * (1 - decision.take_profit_pct)
                ),
                "is_open": True,
            })
            return

        if decision.action == Action.CLOSE_LONG:
            side = "long"
        elif decision.action == Action.CLOSE_SHORT:
            side = "short"
        else:
            return

        remaining_qty = float(execution_result.quantity or 0.0)
        if remaining_qty <= 0:
            return

        for pos in list(open_positions):
            if remaining_qty <= 0:
                break
            if (
                pos.get("model_name") != model_name
                or self._normalize_position_symbol(pos.get("symbol")) != self._normalize_position_symbol(decision.symbol)
                or pos.get("side") != side
            ):
                continue

            qty = float(pos.get("quantity") or 0.0)
            if qty <= 0:
                continue

            close_qty = min(qty, remaining_qty)
            new_qty = qty - close_qty
            if new_qty <= 1e-12:
                open_positions.remove(pos)
            else:
                pos["quantity"] = new_qty
                pos["current_price"] = execution_result.price
            remaining_qty -= close_qty

    def _entry_capacity_reason(
        self,
        model_name: str,
        decision: DecisionOutput,
        open_positions: list[dict],
        staged_entry_counts: dict[str, dict],
    ) -> str | None:
        if not decision.is_entry:
            return None

        side = "long" if decision.action == Action.LONG else "short"
        symbol_key = self._normalize_position_symbol(decision.symbol)
        existing_same_symbol = sum(
            1 for p in open_positions
            if p.get("model_name") == model_name
            and self._normalize_position_symbol(p.get("symbol")) == symbol_key
            and p.get("side") == side
        )
        staged_key = (model_name, symbol_key, side)
        existing_same_symbol += int(staged_entry_counts["symbol_side"].get(staged_key, 0))
        is_same_symbol_add = existing_same_symbol > 0

        model_open_count = sum(
            1 for p in open_positions
            if p.get("model_name") == model_name
        )
        model_open_count += int(staged_entry_counts["model_totals"].get(model_name, 0))
        if (
            not is_same_symbol_add
            and settings.max_open_positions_per_model > 0
            and model_open_count >= settings.max_open_positions_per_model
        ):
            return (
                f"当前持仓数已达上限，暂停新开仓。"
                f"当前 {model_open_count} 笔，限制 {settings.max_open_positions_per_model} 笔。"
            )

        if False and existing_same_symbol >= settings.max_same_symbol_positions_per_side:
            side_label = "做多" if side == "long" else "做空"
            return (
                f"同币种同方向持仓已达上限，暂停加仓。"
                f"{decision.symbol} {side_label} 当前 {existing_same_symbol} 笔，"
                f"限制 {settings.max_same_symbol_positions_per_side} 笔。"
            )

        exposure = self._position_exposure_context(open_positions, staged_entry_counts)
        dominant_side = str(exposure.get("dominant_side") or "neutral")
        same_side_count = int(exposure.get(f"{side}_count", 0) or 0)
        opposite_side = "short" if side == "long" else "long"
        opposite_side_count = int(exposure.get(f"{opposite_side}_count", 0) or 0)
        projected_same_side_count = same_side_count + 1
        projected_total_count = (
            projected_same_side_count
            + opposite_side_count
        )
        projected_share = (
            projected_same_side_count / projected_total_count
            if projected_total_count > 0
            else 0.0
        )
        if False and (
            (
                dominant_side == side
                and same_side_count >= 2
                and abs(float(exposure.get("net_ratio") or 0.0)) >= 0.65
            ) or (
                projected_same_side_count >= 3
                and projected_share >= 0.75
            )
        ):
            if self._profit_first_crowded_side_entry_allowed(decision, side, exposure):
                return None
            side_label = "long" if side == "long" else "short"
            opposite_label = "short" if side == "long" else "long"
            net_ratio_pct = abs(float(exposure.get("net_ratio") or 0.0)) * 100
            return (
                f"Current exposure is crowded on the {side_label} side; pause additional same-side entries. "
                f"long_notional={float(exposure.get('long_notional') or 0.0):.2f}, "
                f"short_notional={float(exposure.get('short_notional') or 0.0):.2f}, "
                f"net_ratio={net_ratio_pct:.1f}%, "
                f"{side_label}_count={same_side_count}, {opposite_label}_count={opposite_side_count}."
            )

        return None

    def _profit_first_crowded_side_entry_allowed(
        self,
        decision: DecisionOutput,
        side: str,
        exposure: dict[str, Any],
    ) -> bool:
        """Allow only small, ML-supported profit-first entries on a crowded side."""
        if float(decision.confidence or 0.0) < 0.72:
            return False
        if float(decision.position_size_pct or 0.0) > 0.025:
            return False
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        ml_gate = raw.get("ml_profit_quality_gate") if isinstance(raw.get("ml_profit_quality_gate"), dict) else {}
        ml_hint = raw.get("ml_profit_first_direction_hint") if isinstance(raw.get("ml_profit_first_direction_hint"), dict) else {}
        if ml_gate.get("status") != "supported_by_profit_quality":
            return False
        if ml_hint.get("side") != side or not ml_hint.get("strong"):
            return False
        # Keep this as a narrow exception, not a way to build unlimited same-side exposure.
        if int(exposure.get(f"{side}_count", 0) or 0) >= 5:
            return False
        return True

    def _reserve_entry_slot(
        self,
        model_name: str,
        decision: DecisionOutput,
        staged_entry_counts: dict[str, dict],
    ) -> None:
        if not decision.is_entry:
            return

        staged_entry_counts["model_totals"][model_name] = (
            int(staged_entry_counts["model_totals"].get(model_name, 0)) + 1
        )
        side = "long" if decision.action == Action.LONG else "short"
        staged_entry_counts.setdefault("side_totals", {})
        staged_entry_counts["side_totals"][side] = (
            int(staged_entry_counts["side_totals"].get(side, 0)) + 1
        )
        staged_key = (model_name, self._normalize_position_symbol(decision.symbol), side)
        staged_entry_counts["symbol_side"][staged_key] = (
            int(staged_entry_counts["symbol_side"].get(staged_key, 0)) + 1
        )

    async def manual_trade(
        self, symbol: str, model_name: str | None = None
    ) -> dict[str, Any]:
        """Execute a one-shot AI analysis and trade for a specific symbol.

        Returns a dict with decision, execution result, and any rejection reason.
        """
        result = {
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
        memory_context = await self._expert_memory_context(symbol)
        daily_target_context = await self._daily_target_context()
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
        analysis_started = datetime.now(timezone.utc)
        decision, _opinions = await self.ensemble.decide(
            fv,
            {
                "open_positions": open_positions,
                "trading_mode": mode_manager.mode.value,
                "manual_override": True,
                **memory_context,
                "daily_target": daily_target_context,
                "market_regime": market_regime_context,
                "strategy_mode": strategy_mode_context,
                "ml_signal": ml_signal_context,
                "local_ai_tools": local_ai_tools_context,
            },
        )
        self._attach_decision_timing(decision, analysis_started, "manual")
        result["model"] = ENSEMBLE_TRADER_NAME

        if decision is None:
            result["rejection_reason"] = "No actionable decision produced"
            return result

        result["decision"] = {
            "model_name": decision.model_name,
            "symbol": decision.symbol,
            "action": decision.action.value if hasattr(decision.action, "value") else str(decision.action),
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "position_size_pct": decision.position_size_pct,
        }

        # Risk assessment 鈥?only this model's positions
        model_positions_manual = [p for p in open_positions if p.get("model_name") == result["model"]]
        headlines = fv.recent_headlines if hasattr(fv, "recent_headlines") else []
        balance = await self._get_account_balance(result["model"])
        assessment = self.risk_engine.assess(
            decision,
            current_positions=model_positions_manual,
            account_balance=balance,
            headlines=headlines,
            sentiment_scores=[],
            price_change_1m=fv.returns_1 if hasattr(fv, "returns_1") else 0.0,
            volume_ratio=fv.volume_ratio if hasattr(fv, "volume_ratio") else 1.0,
            adx_14=fv.adx_14 if hasattr(fv, "adx_14") else None,
        )

        if not assessment.approved:
            result["rejection_reason"] = assessment.rejection_reason
            return result

        result["approved"] = True
        executed = assessment.decision if assessment.decision else decision
        if executed is not decision and decision.raw_response and not executed.raw_response:
            executed.raw_response = decision.raw_response
            executed.feature_snapshot = executed.feature_snapshot or decision.feature_snapshot

        if executed.is_hold:
            "AI 选择观望，未提交订单。",
            return result

        # Log decision
        decision_db_id = await self._log_decision(executed, is_paper=mode_manager.is_paper)
        self._decision_count += 1

        if executed.is_exit and not self.exit_policy.has_matching_position(
            open_positions,
            result["model"],
            executed,
        ):
            await self.okx_sync_service.reconcile_positions("manual exit precheck")
            open_positions[:] = await self.okx_sync_service.get_open_positions_context()

        if executed.is_exit and not self.exit_policy.has_matching_position(
            open_positions,
            result["model"],
            executed,
        ):
            reason = self.exit_policy.no_matching_position_reason(executed)
            if decision_db_id is not None:
                await self._mark_decision_reason(decision_db_id, reason)
            result["approved"] = False
            result["rejection_reason"] = reason
            return result

        if executed.is_exit:
            guard_reason = await self.exit_policy.fee_churn_guard_reason(result["model"], executed)
            if guard_reason:
                if decision_db_id is not None:
                    await self._mark_decision_reason(decision_db_id, guard_reason)
                result["approved"] = False
                result["rejection_reason"] = guard_reason
                return result

        if executed.is_entry:
            capacity_reason = self._entry_capacity_reason(
                result["model"],
                executed,
                open_positions,
                {"model_totals": {}, "symbol_side": {}, "side_totals": {}},
            )
            if capacity_reason:
                if decision_db_id is not None:
                    await self._mark_decision_reason(decision_db_id, capacity_reason)
                result["approved"] = False
                result["rejection_reason"] = capacity_reason
                return result

        stale_reason = self._stale_decision_reason(executed)
        if stale_reason:
            if decision_db_id is not None:
                await self._mark_decision_reason(decision_db_id, stale_reason)
                await self._mark_decision_raw_response(decision_db_id, executed.raw_response)
            result["approved"] = False
            result["rejection_reason"] = stale_reason
            return result

        price_guard_reason = await self.entry_policy.pre_execution_price_guard_reason(executed)
        if price_guard_reason:
            if decision_db_id is not None:
                await self._mark_decision_reason(decision_db_id, price_guard_reason)
            await self._log_risk_event(
                "warning",
                executed.symbol,
                f"[{result['model']}] {price_guard_reason}",
                result["model"],
            )
            result["approved"] = False
            result["rejection_reason"] = price_guard_reason
            return result

        # Execute against OKX demo in paper mode and OKX real in live mode.
        execution_result = None
        model_mode = self._get_model_execution_mode(result["model"])
        executor = await self._get_okx_executor_for_mode(model_mode)
        override_balance = await self._allocated_order_balance(model_mode, executed)
        if executed.is_entry and override_balance <= 0:
            execution_result = self._rejected_execution_result(
                executed,
                "OKX 当前可用余额不足，订单未提交。请检查 OKX 余额、保证金占用和账户风控状态。",
            )
        else:
            try:
                execution_timeout = 90.0 if executed.is_exit else 60.0
                execution_result = await asyncio.wait_for(
                    executor.place_order(
                        executed,
                        account_id=result["model"],
                        override_balance=override_balance,
                    ),
                    timeout=execution_timeout,
                )
            except asyncio.TimeoutError:
                reason = (
                    "OKX 下单或确认超时，系统没有拿到最终订单结果；"
                    "本次手动裁决已按未执行处理。"
                )
                execution_result = self._rejected_execution_result(executed, reason)
                await self._log_risk_event(
                    "warning",
                    executed.symbol,
                    f"[{result['model']}] OKX execution timed out",
                    result["model"],
                )

        if execution_result:
            await self._log_trade(execution_result, result["model"], executed, decision_db_id)
            exchange_confirmed = self._is_exchange_confirmed_execution(execution_result)
            exit_progress = self._is_exit_progress_execution(execution_result)
            if (
                executed.is_exit
                and not exchange_confirmed
                and self._result_has_no_exchange_position(execution_result)
            ):
                await self.okx_sync_service.reconcile_positions("manual exit no-position result")
                open_positions[:] = await self.okx_sync_service.get_open_positions_context()
            if exchange_confirmed or exit_progress:
                await self._persist_position_from_execution(
                    result["model"], executed, execution_result,
                    self._get_model_execution_mode(result["model"]),
                )
            if exchange_confirmed:
                self._trade_count += 1
            result["execution"] = {
                "order_id": execution_result.order_id,
                "status": execution_result.status.value,
                "quantity": execution_result.quantity,
                "price": execution_result.price,
            }
            if decision_db_id is not None and exchange_confirmed:
                await self._mark_decision_executed(decision_db_id, execution_result.price)
            elif decision_db_id is not None:
                await self._mark_decision_reason(
                    decision_db_id,
                    self._execution_reason_from_result(execution_result),
                )
            if executed.is_exit and execution_result.pnl != 0.0:
                await self._persist_account_update(result["model"], executed.model_name, execution_result)
                if decision_db_id is not None:
                    balance = await self._get_account_balance(result["model"])
                    pnl_pct = execution_result.pnl / balance if balance > 0 else 0.0
                    outcome = "profit" if execution_result.pnl > 0 else ("loss" if execution_result.pnl < 0 else "flat")
                    await self._mark_decision_outcome(decision_db_id, outcome, pnl_pct)
        else:
            reason = (
                "交易接口未返回执行结果，系统没有拿到 OKX 订单号，也没有生成本地订单；"
                "本次裁决已按未执行处理。"
            )
            if decision_db_id is not None:
                await self._mark_decision_reason(decision_db_id, reason)
            result["approved"] = False
            result["rejection_reason"] = reason

        return result

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
            record = await repo.log_decision({
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
                    "profit" if realized_pnl > 0 else
                    "loss" if realized_pnl < 0 else
                    "flat"
                ),
                "outcome_pnl_pct": (
                    realized_pnl / (pos.entry_price * pos.quantity) * 100
                    if pos.entry_price and pos.quantity else 0.0
                ),
                "created_at": closed_at,
            })
            await session.flush()
            return record.id
        except Exception as e:
            logger.warning(
                "failed to log exchange sync close decision",
                position_id=getattr(pos, "id", None),
                symbol=getattr(pos, "symbol", None),
                error=str(e),
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
        normalized = self._normalize_position_symbol(symbol).upper()
        if not normalized:
            return "币种符号为空，跳过新开仓分析。"
        base = normalized.split("/", 1)[0]
        if any(token in base for token in SUSPICIOUS_NEW_SYMBOL_TOKENS):
            return (
                f"{normalized} 看起来是测试/模拟/占位合约，"
                "不参与自动扫描、AI开仓分析或新开仓执行。"
            )
        return None

    def _exchange_position_is_open(self, position: dict) -> bool:
        info = position.get("info") or {}
        raw_size = (
            position.get("contracts")
            or position.get("size")
            or position.get("positionAmt")
            or info.get("pos")
            or info.get("qty")
            or 0
        )
        try:
            return abs(float(raw_size)) > 0
        except (TypeError, ValueError):
            return bool(position.get("symbol"))

    async def _fetch_exchange_protection_map(
        self,
        executor: OKXExecutor,
        exchange_positions: list[dict],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Return OKX TP/SL algo orders keyed by (symbol, position_side)."""
        protection_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        symbols = {
            self._normalize_position_symbol(p.get("symbol"))
            for p in exchange_positions or []
            if self._exchange_position_is_open(p)
        }
        symbols.discard("")

        async def fetch_symbol_orders(symbol: str) -> tuple[str, list[dict]]:
            try:
                orders = await asyncio.wait_for(
                    executor.get_position_protection_orders(symbol),
                    timeout=2.5,
                )
                return symbol, orders or []
            except asyncio.TimeoutError:
                logger.warning(
                    "timed out fetching OKX TP/SL protection orders",
                    symbol=symbol,
                )
                return symbol, []
            except Exception as e:
                logger.warning(
                    "failed to fetch OKX TP/SL protection orders",
                    symbol=symbol,
                    error=str(e),
                )
                return symbol, []

        protection_results = await asyncio.gather(
            *(fetch_symbol_orders(symbol) for symbol in symbols),
            return_exceptions=False,
        )

        for _symbol, orders in protection_results:
            for order in orders or []:
                key = (
                    self._normalize_position_symbol(order.get("symbol")),
                    str(order.get("position_side") or "").lower(),
                )
                if not key[0] or key[1] not in {"long", "short"}:
                    continue

                existing = protection_by_key.get(key)
                if existing and self._safe_float(existing.get("updated_at_ms"), 0.0) > self._safe_float(order.get("updated_at_ms"), 0.0):
                    continue
                protection_by_key[key] = order

        return protection_by_key

    async def _fallback_position_protection_from_decision(
        self,
        session: Any,
        *,
        symbol: str,
        side: str,
        entry_price: float,
        order: Order | None = None,
    ) -> dict[str, Any]:
        """Recover local TP/SL values when OKX does not return attached algo orders."""
        if entry_price <= 0 or side not in {"long", "short"}:
            return {}

        decision: AIDecision | None = None
        if order is not None and getattr(order, "decision_id", None):
            result = await session.execute(
                select(AIDecision).where(AIDecision.id == order.decision_id).limit(1)
            )
            decision = result.scalar_one_or_none()

        action_value = Action.LONG.value if side == "long" else Action.SHORT.value
        if decision is None:
            result = await session.execute(
                select(AIDecision)
                .where(
                    AIDecision.symbol == symbol,
                    AIDecision.action == action_value,
                    AIDecision.was_executed == True,
                )
                .order_by(AIDecision.created_at.desc())
                .limit(1)
            )
            decision = result.scalar_one_or_none()

        if decision is None:
            return {}

        stop_loss_pct = self._safe_float(getattr(decision, "stop_loss_pct", 0.0), 0.0)
        take_profit_pct = self._safe_float(getattr(decision, "take_profit_pct", 0.0), 0.0)
        if stop_loss_pct <= 0 and take_profit_pct <= 0:
            return {}

        stop_loss = 0.0
        take_profit = 0.0
        if side == "long":
            if stop_loss_pct > 0:
                stop_loss = entry_price * (1 - stop_loss_pct)
            if take_profit_pct > 0:
                take_profit = entry_price * (1 + take_profit_pct)
        else:
            if stop_loss_pct > 0:
                stop_loss = entry_price * (1 + stop_loss_pct)
            if take_profit_pct > 0:
                take_profit = entry_price * (1 - take_profit_pct)

        return {
            "stop_loss_price": stop_loss if stop_loss > 0 else None,
            "take_profit_price": take_profit if take_profit > 0 else None,
            "source": "latest_executed_entry_decision",
            "decision_id": getattr(decision, "id", None),
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }

    def _sync_local_open_position_snapshot(
        self,
        positions: list[Position],
        exchange_quantity: float,
        current_price: float,
        entry_price: float,
        leverage: float,
        exchange_unrealized: float,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> bool:
        """Keep local open position size aligned with the exchange snapshot."""
        open_positions = [p for p in positions if p.is_open]
        if not open_positions or exchange_quantity <= 0:
            return False

        local_total = sum(abs(self._safe_float(p.quantity, 0.0)) for p in open_positions)
        tolerance = max(local_total * 0.001, exchange_quantity * 0.001, 1e-8)
        changed = abs(local_total - exchange_quantity) > tolerance
        stop_loss = self._safe_float(stop_loss_price, 0.0)
        take_profit = self._safe_float(take_profit_price, 0.0)

        if changed:
            if len(open_positions) == 1:
                open_positions[0].quantity = exchange_quantity
            else:
                ratio = exchange_quantity / local_total if local_total > 0 else 0.0
                remaining = exchange_quantity
                for pos in open_positions[:-1]:
                    new_qty = max(abs(self._safe_float(pos.quantity, 0.0)) * ratio, 0.0)
                    pos.quantity = new_qty
                    remaining -= new_qty
                open_positions[-1].quantity = max(remaining, 0.0)

        total_after = sum(abs(self._safe_float(p.quantity, 0.0)) for p in open_positions)
        for pos in open_positions:
            pos.current_price = current_price
            if not pos.entry_price and entry_price > 0:
                pos.entry_price = entry_price
            if leverage > 0:
                pos.leverage = leverage
            if stop_loss > 0 and abs(self._safe_float(pos.stop_loss_price, 0.0) - stop_loss) > 1e-12:
                pos.stop_loss_price = stop_loss
                changed = True
            if take_profit > 0 and abs(self._safe_float(pos.take_profit_price, 0.0) - take_profit) > 1e-12:
                pos.take_profit_price = take_profit
                changed = True
            share = abs(self._safe_float(pos.quantity, 0.0)) / total_after if total_after > 0 else 0.0
            if exchange_unrealized:
                pos.unrealized_pnl = exchange_unrealized * share
            else:
                if pos.side == "short":
                    pos.unrealized_pnl = (pos.entry_price - current_price) * pos.quantity
                else:
                    pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            pos.updated_at = datetime.now(timezone.utc)

        return changed

    async def _find_exchange_close_fill(self, pos) -> dict:
        if not self._okx_paper:
            return {}

        ccxt = await self._okx_paper._get_ccxt()
        okx_symbol = self._okx_paper._to_swap_symbol(pos.symbol)
        contract_size = 1.0
        try:
            market = ccxt.market(okx_symbol)
            contract_size = self._safe_float(market.get("contractSize"), 1.0) or 1.0
        except Exception:
            contract_size = 1.0
        opened_at = pos.created_at
        since = None
        if opened_at:
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
            since = int(opened_at.timestamp() * 1000)
        close_side = "buy" if pos.side == "short" else "sell"
        target_qty = abs(self._safe_float(getattr(pos, "quantity", 0.0), 0.0))

        try:
            orders = await self._okx_paper._with_retry(
                ccxt.fetch_closed_orders,
                okx_symbol,
                since,
                50,
            )
        except Exception:
            orders = []

        candidates: list[dict] = []
        min_ts = (since or 0) - 1000
        for order in orders or []:
            info = order.get("info") or {}
            if order.get("side") != close_side:
                continue
            timestamp = self._safe_float(
                order.get("timestamp") or info.get("uTime") or info.get("cTime"),
                0.0,
            )
            if timestamp and timestamp < min_ts:
                continue
            reduce_raw = order.get("reduceOnly")
            if reduce_raw in (None, ""):
                reduce_raw = info.get("reduceOnly")
            is_reduce_only = str(reduce_raw).lower() == "true"
            pnl = self._safe_float(info.get("pnl") or info.get("fillPnl"), 0.0)
            has_close_pnl = abs(pnl) > 1e-12
            if not is_reduce_only and not has_close_pnl:
                continue
            qty = self._safe_float(
                order.get("filled") or order.get("amount") or info.get("fillSz") or info.get("accFillSz"),
                0.0,
            )
            qty_base = qty * contract_size
            if target_qty > 0 and qty_base > 0 and qty_base < target_qty * 0.2:
                continue
            candidates.append({
                "price": self._safe_float(order.get("average") or order.get("price") or info.get("fillPx"), 0.0),
                "fee": self._order_fee_cost(order),
                "order_id": info.get("ordId") or order.get("id"),
                "timestamp_ms": timestamp,
                "timestamp": self._datetime_from_ms(timestamp) if timestamp else None,
                "quantity": qty_base,
                "contracts": qty,
                "contract_size": contract_size,
                "pnl": pnl,
                "source": "closed_orders",
            })

        try:
            trades = await self._okx_paper._with_retry(
                ccxt.fetch_my_trades,
                okx_symbol,
                since,
                100,
            )
        except Exception:
            trades = []

        trade_groups: dict[str, dict] = {}
        for trade in trades or []:
            info = trade.get("info") or {}
            if trade.get("side") != close_side:
                continue
            timestamp = self._safe_float(
                trade.get("timestamp") or info.get("ts") or info.get("fillTime"),
                0.0,
            )
            if timestamp and timestamp < min_ts:
                continue
            fill_pnl = self._safe_float(info.get("fillPnl") or info.get("pnl"), 0.0)
            amount = self._safe_float(trade.get("amount") or info.get("fillSz"), 0.0)
            amount_base = amount * contract_size
            if abs(fill_pnl) <= 1e-12 and not (target_qty > 0 and amount_base >= target_qty * 0.8):
                continue
            price = self._safe_float(trade.get("price") or info.get("fillPx"), 0.0)
            if price <= 0:
                continue
            order_id = info.get("ordId") or trade.get("order") or trade.get("id")
            group = trade_groups.setdefault(order_id, {
                "price_value": 0.0,
                "quantity": 0.0,
                "fee": 0.0,
                "pnl": 0.0,
                "timestamp_ms": timestamp,
                "order_id": order_id,
                "source": "my_trades",
            })
            group["price_value"] += price * amount_base
            group["quantity"] += amount_base
            group["contracts"] = group.get("contracts", 0.0) + amount
            group["contract_size"] = contract_size
            group["fee"] += self._order_fee_cost(trade)
            group["pnl"] += fill_pnl
            group["timestamp_ms"] = max(group["timestamp_ms"] or 0, timestamp or 0)

        for group in trade_groups.values():
            qty = group.get("quantity") or 0.0
            if target_qty > 0 and qty > 0 and qty < target_qty * 0.2:
                continue
            timestamp = group.get("timestamp_ms") or 0
            candidates.append({
                "price": group["price_value"] / qty if qty > 0 else 0.0,
                "fee": group.get("fee") or 0.0,
                "order_id": group.get("order_id"),
                "timestamp_ms": timestamp,
                "timestamp": self._datetime_from_ms(timestamp) if timestamp else None,
                "quantity": qty,
                "pnl": group.get("pnl") or 0.0,
                "source": "my_trades",
            })

        if candidates:
            candidates = [c for c in candidates if c.get("price", 0) > 0 and c.get("order_id")]
            if candidates:
                return sorted(candidates, key=lambda c: c.get("timestamp_ms") or 0)[-1]

        return {}

    def _order_fee_cost(self, order: dict) -> float:
        fee = order.get("fee")
        if isinstance(fee, dict):
            return float(fee.get("cost") or 0.0)
        info_fee = (order.get("info") or {}).get("fee")
        try:
            return abs(float(info_fee or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _proportional_fee(self, fee: float | None, close_qty: float, total_qty: float) -> float:
        try:
            fee_value = abs(float(fee or 0.0))
            close_value = float(close_qty or 0.0)
            total_value = float(total_qty or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if fee_value <= 0 or close_value <= 0:
            return 0.0
        if total_value <= 0:
            return fee_value
        return fee_value * min(close_value / total_value, 1.0)

    async def _entry_fee_for_position(self, session, pos, close_qty: float) -> float:
        """Return the matching entry fee share for a closing position."""
        entry_side = "buy" if pos.side == "long" else "sell"
        created_at = pos.created_at
        stmt = select(Order).where(
            Order.model_name == pos.model_name,
            Order.execution_mode == pos.execution_mode,
            Order.symbol == pos.symbol,
            Order.side == entry_side,
            Order.status == OrderStatus.FILLED.value,
        )
        if created_at:
            window_start = created_at - timedelta(seconds=90)
            window_end = created_at + timedelta(seconds=90)
            window_stmt = stmt.where(
                Order.created_at >= window_start,
                Order.created_at <= window_end,
            )
            result = await session.execute(window_stmt.order_by(Order.created_at.asc()).limit(1))
            order = result.scalar_one_or_none()
            if order:
                return self._proportional_fee(order.fee, close_qty, order.quantity)

            result = await session.execute(
                stmt.where(Order.created_at <= created_at)
                .order_by(Order.created_at.desc())
                .limit(1)
            )
            order = result.scalar_one_or_none()
            if order:
                return self._proportional_fee(order.fee, close_qty, order.quantity)

        result = await session.execute(stmt.order_by(Order.created_at.desc()).limit(1))
        order = result.scalar_one_or_none()
        if order:
            return self._proportional_fee(order.fee, close_qty, order.quantity)
        return 0.0

    def _datetime_from_ms(self, timestamp_ms) -> datetime:
        try:
            return datetime.fromtimestamp(float(timestamp_ms) / 1000, tz=timezone.utc)
        except (TypeError, ValueError):
            return datetime.now(timezone.utc)

    async def _exchange_backed_position_ids(self, session, positions: list) -> set[int]:
        """Return local position ids that can be tied to a filled OKX entry order."""
        ids: set[int] = set()
        for pos in positions:
            entry_side = "buy" if pos.side == "long" else "sell"
            created_at = pos.created_at
            window_start = None
            window_end = None
            if created_at:
                if created_at.tzinfo is not None:
                    created_at = created_at.replace(tzinfo=None)
                window_start = created_at - timedelta(seconds=30)
                window_end = created_at + timedelta(seconds=30)
            stmt = select(Order.id).where(
                Order.model_name == pos.model_name,
                Order.execution_mode == pos.execution_mode,
                Order.symbol == pos.symbol,
                Order.side == entry_side,
                Order.status == OrderStatus.FILLED.value,
                Order.exchange_order_id.is_not(None),
                Order.exchange_order_id != "",
            )
            if window_start is not None and window_end is not None:
                stmt = stmt.where(
                    Order.created_at >= window_start,
                    Order.created_at <= window_end,
                )
            result = await session.execute(stmt.limit(1))
            if result.scalar_one_or_none():
                ids.add(pos.id)
        return ids

    def _remove_memory_position(self, model_name: str, symbol: str, side: str) -> None:
        if not self.paper_executor:
            return
        positions = self.paper_executor._positions.get(model_name, [])
        self.paper_executor._positions[model_name] = [
            p for p in positions
            if not (
                self._normalize_position_symbol(p.get("symbol")) == self._normalize_position_symbol(symbol)
                and p.get("side") == side
                and p.get("is_open")
            )
        ]

    def _position_age_minutes(self, created_at: Any) -> float | None:
        if not created_at:
            return None
        try:
            opened = created_at
            if isinstance(opened, (int, float)):
                value = float(opened)
                if value > 10_000_000_000:
                    value = value / 1000.0
                opened = datetime.fromtimestamp(value, tz=timezone.utc)
            if isinstance(opened, str):
                stripped = opened.strip()
                if stripped.isdigit():
                    value = float(stripped)
                    if value > 10_000_000_000:
                        value = value / 1000.0
                    opened = datetime.fromtimestamp(value, tz=timezone.utc)
                else:
                    opened = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            if not isinstance(opened, datetime):
                return None
            now = datetime.now(timezone.utc)
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
                if opened > now + timedelta(minutes=1):
                    opened = opened.replace(tzinfo=None).replace(tzinfo=BEIJING_TZ)
            return max((now - opened.astimezone(timezone.utc)).total_seconds() / 60.0, 0.0)
        except Exception:
            return None

    def _position_peak_key(self, model_name: str, symbol: str, side: str) -> str:
        return "|".join([
            str(model_name or ENSEMBLE_TRADER_NAME),
            self._normalize_position_symbol(symbol) or str(symbol or ""),
            str(side or "").lower(),
        ])

    def _load_position_profit_peaks(self) -> dict[str, dict[str, Any]]:
        try:
            path = POSITION_PROFIT_PEAKS_STATE_PATH
            if not path.exists():
                return {}
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if not isinstance(payload, dict):
                return {}
            loaded: dict[str, dict[str, Any]] = {}
            for key, value in payload.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                loaded[key] = {
                    "peak_unrealized_pnl": float(value.get("peak_unrealized_pnl") or 0.0),
                    "peak_pnl_ratio": float(value.get("peak_pnl_ratio") or 0.0),
                    "last_unrealized_pnl": float(value.get("last_unrealized_pnl") or 0.0),
                    "last_pnl_ratio": float(value.get("last_pnl_ratio") or 0.0),
                    "updated_at": str(value.get("updated_at") or ""),
                    "hold_minutes": float(value.get("hold_minutes") or 0.0),
                    "last_profit_exit_at": value.get("last_profit_exit_at") or "",
                    "profit_exit_count": int(value.get("profit_exit_count") or 0),
                }
            logger.info("loaded position profit peaks", count=len(loaded), path=str(path))
            return loaded
        except Exception as e:
            logger.warning("failed to load position profit peaks", error=str(e))
            return {}

    def _save_position_profit_peaks(self) -> None:
        try:
            path = POSITION_PROFIT_PEAKS_STATE_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(self._position_profit_peaks, fh, ensure_ascii=False, indent=2)
            tmp_path.replace(path)
        except Exception as e:
            logger.warning("failed to save position profit peaks", error=str(e))

    def _update_position_profit_peak(
        self,
        *,
        model_name: str,
        symbol: str,
        side: str,
        current_price: float,
        entry_price: float,
        unrealized_pnl: float,
        hold_minutes: float | None,
    ) -> dict[str, Any]:
        key = self._position_peak_key(model_name, symbol, side)
        now = datetime.now(timezone.utc).isoformat()
        entry_price = self._safe_float(entry_price, 0.0)
        current_price = self._safe_float(current_price, 0.0)
        unrealized_pnl = self._safe_float(unrealized_pnl, 0.0)
        hold_minutes = float(hold_minutes or 0.0)
        if entry_price <= 0 or current_price <= 0:
            return {}

        pnl_ratio = abs((current_price - entry_price) / entry_price)
        if side == "short":
            pnl_ratio = max((entry_price - current_price) / entry_price, 0.0)
        else:
            pnl_ratio = max((current_price - entry_price) / entry_price, 0.0)

        state = self._position_profit_peaks.get(key) or {
            "peak_unrealized_pnl": unrealized_pnl,
            "peak_pnl_ratio": pnl_ratio,
            "last_unrealized_pnl": unrealized_pnl,
            "last_pnl_ratio": pnl_ratio,
            "updated_at": now,
            "hold_minutes": hold_minutes,
        }
        state["peak_unrealized_pnl"] = max(self._safe_float(state.get("peak_unrealized_pnl"), unrealized_pnl), unrealized_pnl)
        state["peak_pnl_ratio"] = max(self._safe_float(state.get("peak_pnl_ratio"), pnl_ratio), pnl_ratio)
        state["last_unrealized_pnl"] = unrealized_pnl
        state["last_pnl_ratio"] = pnl_ratio
        state["updated_at"] = now
        state["hold_minutes"] = hold_minutes
        self._position_profit_peaks[key] = state
        self._save_position_profit_peaks()
        return state

    def _position_peak_exit_recently(self, peak_state: dict[str, Any]) -> float:
        value = peak_state.get("last_profit_exit_at") if isinstance(peak_state, dict) else None
        if not value:
            return 0.0
        try:
            exited_at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if exited_at.tzinfo is None:
                exited_at = exited_at.replace(tzinfo=timezone.utc)
            return max((datetime.now(timezone.utc) - exited_at).total_seconds(), 0.0)
        except Exception:
            return 0.0

    def _remember_position_peak_profit_exit(self, model_name: str, symbol: str, side: str) -> None:
        key = self._position_peak_key(model_name, symbol, side)
        state = self._position_profit_peaks.get(key)
        if isinstance(state, dict):
            state["last_profit_exit_at"] = datetime.now(timezone.utc).isoformat()
            state["profit_exit_count"] = int(state.get("profit_exit_count") or 0) + 1
            self._position_profit_peaks[key] = state
            self._save_position_profit_peaks()

    def _remove_position_profit_peak(self, model_name: str, symbol: str, side: str) -> None:
        removed = self._position_profit_peaks.pop(self._position_peak_key(model_name, symbol, side), None)
        if removed is not None:
            self._save_position_profit_peaks()

    def _prune_position_profit_peaks(self, open_positions: list[dict]) -> None:
        valid = {
            self._position_peak_key(
                str(pos.get("model_name") or ENSEMBLE_TRADER_NAME),
                str(pos.get("symbol") or ""),
                str(pos.get("side") or ""),
            )
            for pos in open_positions or []
            if pos.get("is_open", True)
        }
        before = set(self._position_profit_peaks.keys())
        self._position_profit_peaks = {
            key: value
            for key, value in self._position_profit_peaks.items()
            if key in valid
        }
        if set(self._position_profit_peaks.keys()) != before:
            self._save_position_profit_peaks()

    def _profit_drawdown_exit_plan(
        self,
        *,
        side: str,
        current_price: float,
        entry_price: float,
        unrealized_pnl: float,
        peak_state: dict[str, Any],
        hold_minutes: float | None,
        volume_ratio: float,
        returns_1: float,
        returns_5: float,
        returns_20: float = 0.0,
        rsi_14: float = 50.0,
        bb_pct: float = 0.5,
        macd_diff: float = 0.0,
        adx_14: float = 0.0,
    ) -> dict[str, Any]:
        if entry_price <= 0 or current_price <= 0:
            return {"should_exit": False, "fraction": 0.0, "note": "price data is insufficient"}

        hold_minutes = float(hold_minutes or 0.0)

        peak_pnl = self._safe_float(peak_state.get("peak_unrealized_pnl"), 0.0)
        peak_ratio = self._safe_float(peak_state.get("peak_pnl_ratio"), 0.0)
        current_pnl = self._safe_float(unrealized_pnl, 0.0)
        if peak_pnl <= 0 or current_pnl <= 0:
            return {"should_exit": False, "fraction": 0.0, "note": "no protectable profit peak yet"}

        notional = abs(peak_pnl / max(peak_ratio, 1e-9)) if peak_ratio > 0 else 0.0
        estimated_round_trip_fee = max(notional * ESTIMATED_TAKER_FEE_PCT * 2.0, 1e-9)
        min_net_profit = max(
            PROFIT_DRAWDOWN_MIN_NET_USDT,
            estimated_round_trip_fee * PROFIT_DRAWDOWN_MIN_FEE_MULTIPLE,
        )
        seconds_since_exit = self._position_peak_exit_recently(peak_state)
        if 0 < seconds_since_exit < PROFIT_DRAWDOWN_MIN_SECONDS_BETWEEN_EXITS:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "note": "刚做过一次利润保护，暂不连续碎片化部分平仓。",
                "seconds_since_last_profit_exit": seconds_since_exit,
                "peak_ratio": peak_ratio,
            }

        retrace_abs = max(peak_pnl - current_pnl, 0.0)
        retrace_ratio = retrace_abs / peak_pnl if peak_pnl > 0 else 0.0
        same_direction_pressure = (
            returns_1 < 0 and returns_5 < 0
            if side == "long"
            else returns_1 > 0 and returns_5 > 0
        )
        reversal = self._predictive_reversal_evidence(
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
        volume_confirms = volume_ratio <= 0 or volume_ratio >= PROFIT_DRAWDOWN_VOLUME_CONFIRM_RATIO
        strong_profit = peak_ratio >= PROFIT_DRAWDOWN_STRONG_PROFIT_RATIO
        has_buffer = peak_ratio >= PROFIT_DRAWDOWN_MIN_PROFIT_RATIO
        severe_retrace = bool(
            peak_pnl >= min_net_profit
            and retrace_ratio >= PROFIT_DRAWDOWN_FULL_RETRACE
        )
        profit_salvage_floor = max(
            estimated_round_trip_fee * 2.0,
            min_net_profit * 0.85,
        )

        predictive_exit = bool(
            reversal.get("score", 0.0) >= PREDICTIVE_REVERSAL_EXIT_SCORE
            and peak_pnl >= min_net_profit * PREDICTIVE_REVERSAL_MIN_PROFIT_MULTIPLE
            and current_pnl >= min_net_profit
        )

        if hold_minutes < PROFIT_DRAWDOWN_MIN_HOLD_MINUTES and not severe_retrace and not predictive_exit:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "note": "持仓时间较短，但未出现明显利润回吐，本轮不因普通波动主动锁盈。",
                "peak_ratio": peak_ratio,
                "retrace_ratio": retrace_ratio,
                "hold_minutes": hold_minutes,
                "predictive_reversal": reversal,
            }

        if current_pnl < min_net_profit:
            if (severe_retrace or predictive_exit) and current_pnl >= profit_salvage_floor:
                return {
                    "should_exit": True,
                    "fraction": 1.0 if severe_retrace else PREDICTIVE_REVERSAL_REDUCE_FRACTION,
                    "note": (
                        "曾经达到可保护浮盈，且短周期已出现反向预警；当前仍覆盖手续费缓冲，"
                        "先把账面利润转为已实现利润，避免从盈利拖成亏损。"
                    ),
                    "peak_ratio": peak_ratio,
                    "retrace_ratio": retrace_ratio,
                    "peak_unrealized_pnl": peak_pnl,
                    "current_pnl": current_pnl,
                    "min_net_profit": min_net_profit,
                    "profit_salvage_floor": profit_salvage_floor,
                    "predictive_reversal": reversal,
                }
            return {
                "should_exit": False,
                "fraction": 0.0,
                "note": "当前剩余浮盈已经低于动态锁盈线，且不足以安全覆盖手续费缓冲；本轮不做碎片化小额平仓。",
                "peak_ratio": peak_ratio,
                "retrace_ratio": retrace_ratio,
                "current_pnl": current_pnl,
                "min_net_profit": min_net_profit,
                "profit_salvage_floor": profit_salvage_floor,
                "predictive_reversal": reversal,
            }

        if predictive_exit and retrace_ratio >= 0.10:
            full_predictive = bool(
                reversal.get("score", 0.0) >= PREDICTIVE_REVERSAL_FULL_EXIT_SCORE
                and (retrace_ratio >= PROFIT_DRAWDOWN_PARTIAL_RETRACE or same_direction_pressure)
            )
            return {
                "should_exit": True,
                "fraction": 1.0 if full_predictive else PREDICTIVE_REVERSAL_REDUCE_FRACTION,
                "note": (
                    "预判型锁盈触发：持仓仍有浮盈，但短周期动量、量能或技术结构已经转向不利；"
                    "先减仓/平仓保护利润，避免等到浮盈回吐后再被动止损。"
                ),
                "peak_ratio": peak_ratio,
                "retrace_ratio": retrace_ratio,
                "peak_unrealized_pnl": peak_pnl,
                "current_pnl": current_pnl,
                "predictive_reversal": reversal,
            }

        if not has_buffer and not severe_retrace:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "note": "浮盈峰值还没有达到动态保护线，不因为普通小回撤主动平仓。",
                "peak_ratio": peak_ratio,
                "retrace_ratio": retrace_ratio,
            }

        if (
            retrace_ratio >= PROFIT_DRAWDOWN_FULL_RETRACE
            and (
                (same_direction_pressure and volume_confirms)
                or retrace_ratio >= 0.75
                or severe_retrace
            )
        ):
            return {
                "should_exit": True,
                "fraction": 1.0,
                "note": "浮盈已经明显回撤，利润保护优先于继续等待，执行全平锁定剩余利润。",
                "peak_ratio": peak_ratio,
                "retrace_ratio": retrace_ratio,
                "peak_unrealized_pnl": peak_pnl,
                "predictive_reversal": reversal,
            }

        if (
            retrace_ratio >= PROFIT_DRAWDOWN_PARTIAL_RETRACE
            and (
                (same_direction_pressure and volume_confirms)
                or hold_minutes >= PROFIT_DRAWDOWN_ACCELERATED_HOLD_MINUTES
                or strong_profit
            )
        ):
            return {
                "should_exit": True,
                "fraction": PROFIT_DRAWDOWN_PARTIAL_CLOSE_FRACTION,
                "note": "浮盈开始明显回撤，先减仓锁定一部分利润，剩余仓位继续观察。",
                "peak_ratio": peak_ratio,
                "retrace_ratio": retrace_ratio,
                "peak_unrealized_pnl": peak_pnl,
                "predictive_reversal": reversal,
            }

        return {
            "should_exit": False,
            "fraction": 0.0,
            "note": "浮盈回撤还没有达到主动减仓线。",
            "peak_ratio": peak_ratio,
            "retrace_ratio": retrace_ratio,
            "peak_unrealized_pnl": peak_pnl,
            "predictive_reversal": reversal,
        }

    def _predictive_reversal_evidence(
        self,
        *,
        side: str,
        returns_1: float,
        returns_5: float,
        returns_20: float,
        volume_ratio: float,
        rsi_14: float,
        bb_pct: float,
        macd_diff: float,
        adx_14: float,
    ) -> dict[str, Any]:
        """Score whether the next short window is turning against the held side."""
        side = str(side or "").lower()
        if side not in {"long", "short"}:
            return {"score": 0.0, "level": "none", "reasons": []}

        reasons: list[str] = []
        score = 0.0
        adverse_1 = returns_1 <= -0.0025 if side == "long" else returns_1 >= 0.0025
        adverse_5 = returns_5 <= -0.0060 if side == "long" else returns_5 >= 0.0060
        adverse_20 = returns_20 <= -0.0100 if side == "long" else returns_20 >= 0.0100
        strong_adverse_1 = returns_1 <= -0.0060 if side == "long" else returns_1 >= 0.0060
        strong_adverse_5 = returns_5 <= -0.0140 if side == "long" else returns_5 >= 0.0140

        if adverse_1:
            score += 18.0
            reasons.append("1m_against")
        if adverse_5:
            score += 22.0
            reasons.append("5m_against")
        if adverse_20:
            score += 12.0
            reasons.append("20m_against")
        if strong_adverse_1 or strong_adverse_5:
            score += 12.0
            reasons.append("strong_short_window_against")
        if volume_ratio >= 1.15 and (adverse_1 or adverse_5):
            score += 12.0
            reasons.append("volume_confirms_reversal")
        if side == "long":
            if bb_pct >= 0.86 and rsi_14 >= 66 and (adverse_1 or adverse_5):
                score += 12.0
                reasons.append("long_overheated_reversal")
            if macd_diff < 0:
                score += 8.0
                reasons.append("macd_against_long")
        else:
            if bb_pct <= 0.14 and rsi_14 <= 34 and (adverse_1 or adverse_5):
                score += 12.0
                reasons.append("short_oversold_rebound")
            if macd_diff > 0:
                score += 8.0
                reasons.append("macd_against_short")
        if adx_14 < 16 and (adverse_1 or adverse_5):
            score += 6.0
            reasons.append("trend_strength_weak")

        if score >= PREDICTIVE_REVERSAL_FULL_EXIT_SCORE:
            level = "full_exit"
        elif score >= PREDICTIVE_REVERSAL_EXIT_SCORE:
            level = "exit"
        elif score >= PREDICTIVE_REVERSAL_REVIEW_SCORE:
            level = "review"
        else:
            level = "none"

        return {
            "score": round(score, 4),
            "level": level,
            "reasons": reasons,
            "returns_1": round(returns_1, 6),
            "returns_5": round(returns_5, 6),
            "returns_20": round(returns_20, 6),
            "volume_ratio": round(volume_ratio, 4),
            "rsi_14": round(rsi_14, 4),
            "bb_pct": round(bb_pct, 4),
            "macd_diff": round(macd_diff, 8),
            "adx_14": round(adx_14, 4),
        }

    def _fast_adverse_exit_plan(
        self,
        *,
        side: str,
        entry_price: float,
        current_price: float,
        stop_loss: float,
        returns_1: float,
        returns_5: float,
        hold_minutes: float | None,
        volume_ratio: float,
        current_unrealized_pnl: float = 0.0,
    ) -> dict[str, Any]:
        """Decide whether a fast adverse move is real risk or normal noise."""
        if entry_price <= 0 or current_price <= 0:
            return {"should_exit": False, "fraction": 0.0, "adverse_pct": 0.0, "note": "price data is insufficient"}

        if side == "long":
            adverse_pct = max((entry_price - current_price) / entry_price, 0.0)
            same_direction_pressure = returns_1 < 0 and returns_5 < 0
            stop_distance_pct = (entry_price - stop_loss) / entry_price if 0 < stop_loss < entry_price else 0.0
        else:
            adverse_pct = max((current_price - entry_price) / entry_price, 0.0)
            same_direction_pressure = returns_1 > 0 and returns_5 > 0
            stop_distance_pct = (stop_loss - entry_price) / entry_price if stop_loss > entry_price else 0.0

        near_stop = bool(
            stop_distance_pct > 0
            and adverse_pct >= stop_distance_pct * FAST_RISK_NEAR_STOP_PROGRESS
        )
        full_stop_progress = bool(
            stop_distance_pct > 0
            and adverse_pct >= stop_distance_pct * FAST_RISK_FULL_STOP_PROGRESS
        )
        risk_progress = adverse_pct / max(stop_distance_pct, 1e-12) if stop_distance_pct > 0 else 0.0
        volume_confirmed = volume_ratio <= 0 or volume_ratio >= FAST_RISK_VOLUME_CONFIRM_RATIO
        old_enough = hold_minutes is None or hold_minutes >= FAST_RISK_MIN_HOLD_MINUTES
        loss_usdt = abs(min(self._safe_float(current_unrealized_pnl, 0.0), 0.0))

        if adverse_pct <= 0:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "adverse_pct": adverse_pct,
                "risk_progress": risk_progress,
                "note": "当前价格仍未相对开仓价亏损，短线反向只记录观察。",
            }

        if adverse_pct < FAST_RISK_MIN_LOSS_PCT and not near_stop:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "adverse_pct": adverse_pct,
                "risk_progress": risk_progress,
                "note": "亏损幅度还小，未接近止损，继续交给持仓复盘判断。",
            }

        if not old_enough and not full_stop_progress and adverse_pct < FAST_RISK_FULL_LOSS_PCT:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "adverse_pct": adverse_pct,
                "risk_progress": risk_progress,
                "note": f"开仓不足 {FAST_RISK_MIN_HOLD_MINUTES:.0f} 分钟，暂不因普通短线波动平仓。",
            }

        force_full_by_loss = bool(
            loss_usdt >= FAST_RISK_FORCE_FULL_LOSS_USDT
            and adverse_pct >= FAST_RISK_FULL_LOSS_PCT
            and risk_progress >= FAST_RISK_FORCE_FULL_PROGRESS
        )

        if full_stop_progress or force_full_by_loss or (
            same_direction_pressure
            and volume_confirmed
            and adverse_pct >= FAST_RISK_FULL_LOSS_PCT
        ):
            return {
                "should_exit": True,
                "fraction": 1.0,
                "adverse_pct": adverse_pct,
                "risk_progress": risk_progress,
                "loss_usdt": loss_usdt,
                "note": "价格已接近止损、亏损金额明显扩大，或出现持续同向恶化，直接全平控制风险。",
            }

        if near_stop or (
            old_enough
            and same_direction_pressure
            and volume_confirmed
            and adverse_pct >= max(FAST_RISK_REDUCE_LOSS_PCT * 0.8, 0.008)
        ):
            return {
                "should_exit": False,
                "fraction": 0.0,
                "adverse_pct": adverse_pct,
                "risk_progress": risk_progress,
                "loss_usdt": loss_usdt,
                "note": (
                    "亏损仓位尚未达到强制全平条件，普通快速风控不再做部分减仓；"
                    "继续交给止损、严重趋势失效或下一轮持仓复盘判断。"
                ),
            }

        return {
            "should_exit": False,
            "fraction": 0.0,
            "adverse_pct": adverse_pct,
            "risk_progress": risk_progress,
            "note": "短线反向尚未满足减仓或全平条件，继续观察。",
        }

    async def _enforce_sl_tp(self, feature_vectors: dict) -> list[dict]:
        """Run fast non-AI protection for open positions before slow AI review."""
        auto_closes = []
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
            returns_20 = self._safe_float(getattr(fv, "returns_20", 0) if fv is not None else 0, 0.0)
            volume_ratio = self._safe_float(getattr(fv, "volume_ratio", 0) if fv is not None else 0, 0.0)
            rsi_14 = self._safe_float(getattr(fv, "rsi_14", 50) if fv is not None else 50, 50.0)
            bb_pct = self._safe_float(getattr(fv, "bb_pct", 0.5) if fv is not None else 0.5, 0.5)
            macd_diff = self._safe_float(getattr(fv, "macd_diff", 0) if fv is not None else 0, 0.0)
            adx_14 = self._safe_float(getattr(fv, "adx_14", 0) if fv is not None else 0, 0.0)
            hold_minutes = self._position_age_minutes(pos.get("created_at"))
            if fv_current_price > 0 and position_current_price > 0:
                feature_position_gap = abs(fv_current_price - position_current_price) / max(position_current_price, 1e-12)
                feature_price_implies_adverse = (
                    (side == "long" and fv_current_price < position_current_price)
                    or (side == "short" and fv_current_price > position_current_price)
                )
                short_returns_contradict_adverse = (
                    (side == "long" and returns_1 >= 0 and returns_5 >= 0)
                    or (side == "short" and returns_1 <= 0 and returns_5 <= 0)
                )
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
            peak_state = self._update_position_profit_peak(
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
            hit_hard_adverse = False
            hit_fast_adverse = False
            if side == "long":
                hit_sl = bool(stop_loss and current_price <= stop_loss)
                hit_tp = bool(take_profit and current_price >= take_profit)
                hit_hard_adverse = current_price <= entry_price * (1 - fast_adverse_pct)
                hit_fast_adverse = returns_1 <= -FAST_RISK_1M_MOVE_PCT or returns_5 <= -FAST_RISK_5M_MOVE_PCT
                adverse_pct = max((entry_price - current_price) / entry_price, 0.0)
                stop_distance_pct = (entry_price - stop_loss) / entry_price if 0 < stop_loss < entry_price else 0.0
            else:
                hit_sl = bool(stop_loss and current_price >= stop_loss)
                hit_tp = bool(take_profit and current_price <= take_profit)
                hit_hard_adverse = current_price >= entry_price * (1 + fast_adverse_pct)
                hit_fast_adverse = returns_1 >= FAST_RISK_1M_MOVE_PCT or returns_5 >= FAST_RISK_5M_MOVE_PCT
                adverse_pct = max((current_price - entry_price) / entry_price, 0.0)
                stop_distance_pct = (stop_loss - entry_price) / entry_price if stop_loss > entry_price else 0.0
            hit_near_stop_progress = bool(
                stop_distance_pct > 0
                and adverse_pct >= stop_distance_pct * FAST_RISK_NEAR_STOP_PROGRESS
            )
            hit_full_stop_progress = bool(
                stop_distance_pct > 0
                and adverse_pct >= stop_distance_pct * FAST_RISK_FULL_STOP_PROGRESS
            )
            stop_risk_progress = adverse_pct / max(stop_distance_pct, 1e-12) if stop_distance_pct > 0 else 0.0
            predictive_reversal = self._predictive_reversal_evidence(
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
                and self._safe_float(predictive_reversal.get("score"), 0.0) < PREDICTIVE_REVERSAL_EXIT_SCORE
            )
            if not settlement_guard_active:
                profit_exit_plan = self._profit_drawdown_exit_plan(
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
                reason = "快速风控触发：价格已经触及本地记录的止损位，优先提交平仓，不等待 AI 会诊。"
            elif hit_tp:
                trigger = "take_profit"
                reason = "快速风控触发：价格已经触及本地记录的止盈位，优先提交平仓锁定结果。"
            elif profit_exit_plan.get("should_exit"):
                fast_exit_plan = profit_exit_plan
                close_fraction = self._safe_float(profit_exit_plan.get("fraction"), 1.0)
                close_fraction = min(max(close_fraction, 0.05), 1.0)
                trigger = "profit_drawdown_reduce" if close_fraction < 0.999 else "profit_drawdown_close"
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
            elif hit_hard_adverse:
                trigger = "hard_adverse_move"
                reason = (
                    "快速风控触发：价格已经相对开仓价出现明显反向波动，"
                    f"超过快速硬风险阈值 {fast_adverse_pct:.2%}，"
                    "这不是普通短线回撤，优先提交平仓控制单笔亏损。"
                )
            elif hit_fast_adverse or hit_near_stop_progress:
                fast_exit_plan = self._fast_adverse_exit_plan(
                    side=side,
                    entry_price=entry_price,
                    current_price=current_price,
                    stop_loss=stop_loss,
                    returns_1=returns_1,
                    returns_5=returns_5,
                    hold_minutes=hold_minutes,
                    volume_ratio=volume_ratio,
                    current_unrealized_pnl=current_unrealized,
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
                    profit_exit_plan = self._profit_drawdown_exit_plan(
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
                    trigger = "profit_drawdown_reduce" if close_fraction < 0.999 else "profit_drawdown_close"
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
                    trigger = "fast_adverse_reduce" if close_fraction < 0.999 else "fast_adverse_move"
                    reason = (
                        "快速风控触发：1-5 分钟短线波动明显反向。"
                        f"{fast_exit_plan.get('note')}"
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
                trigger = "profit_drawdown_reduce" if close_fraction < 0.999 else "profit_drawdown_close"
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
                },
                feature_snapshot={
                    "current_price": current_price,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "returns_1": returns_1,
                    "returns_5": returns_5,
                    "volume_ratio": volume_ratio,
                    "stop_risk_progress": stop_risk_progress,
                    "position_age_minutes": hold_minutes,
                    "adverse_from_entry_pct": fast_exit_plan.get("adverse_pct") if fast_exit_plan else None,
                    "close_fraction": close_fraction,
                    "fast_adverse_pct": fast_adverse_pct,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            recent_exit_reason = self.exit_policy.recent_exit_cooldown_reason(model_name, close_decision)
            if recent_exit_reason:
                logger.info(
                    "fast risk close skipped by recent exit cooldown",
                    model=model_name,
                    symbol=sym,
                    side=side,
                    trigger=trigger,
                    reason=recent_exit_reason,
                )
                continue
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

            model_exec_mode = self._get_model_execution_mode(model_name)
            decision_db_id = await self._log_decision(
                close_decision, is_paper=(model_exec_mode == "paper")
            )
            self._decision_count += 1

            try:
                executor = await self._get_okx_executor_for_mode(model_exec_mode)
                execution_result = await asyncio.wait_for(
                    executor.place_order(
                        close_decision,
                        account_id=model_name,
                        override_balance=await self._allocated_order_balance(model_exec_mode, close_decision),
                    ),
                    timeout=90.0,
                )

                await self._log_trade(execution_result, model_name, close_decision, decision_db_id)
                exchange_confirmed = self._is_exchange_confirmed_execution(execution_result)
                exit_progress = self._is_exit_progress_execution(execution_result)

                if exchange_confirmed or exit_progress:
                    await self._persist_position_from_execution(
                        model_name, close_decision, execution_result, model_exec_mode
                    )
                    if str(trigger).startswith("profit_drawdown"):
                        self._remember_position_peak_profit_exit(model_name, sym, side)
                    self._remember_recent_exit_group(model_name, close_decision)
                if exchange_confirmed:
                    self._trade_count += 1
                    if decision_db_id is not None:
                        await self._mark_decision_executed(decision_db_id, execution_result.price)
                    if model_exec_mode != "paper" and execution_result.pnl != 0.0:
                        await self._persist_account_update(model_name, close_decision.model_name, execution_result)
                        balance = await self._get_account_balance(model_name)
                        pnl_pct = execution_result.pnl / balance if balance > 0 else 0.0
                        outcome = "profit" if execution_result.pnl > 0 else "loss"
                        if decision_db_id is not None:
                            await self._mark_decision_outcome(decision_db_id, outcome, pnl_pct)
                elif decision_db_id is not None:
                    await self._mark_decision_reason(
                        decision_db_id,
                        self._execution_reason_from_result(execution_result),
                    )

                auto_closes.append({
                    "model_name": model_name,
                    "symbol": sym,
                    "side": side,
                    "quantity": execution_result.quantity,
                    "entry_price": entry_price,
                    "exit_price": execution_result.price,
                    "pnl": execution_result.pnl,
                    "trigger": trigger,
                    "close_fraction": close_fraction,
                    "status": execution_result.status.value,
                })
                await self._log_risk_event(
                    "info" if trigger == "take_profit" else "warning",
                    sym,
                    f"[{model_name}] {reason} 入场 {entry_price:.6g}，当前 {current_price:.6g}，"
                    f"处理仓位 {close_fraction:.0%}，订单状态 {execution_result.status.value}，"
                    f"PnL {execution_result.pnl:+.2f} USDT。",
                    model_name,
                )
            except Exception as e:
                logger.error("failed to execute fast risk close", error=str(e))
                rejected = self._rejected_execution_result(close_decision, e)
                await self._log_trade(rejected, model_name, close_decision, decision_db_id)
                if decision_db_id is not None:
                    await self._mark_decision_reason(
                        decision_db_id,
                        self._execution_reason_from_result(rejected),
                    )
                auto_closes.append({
                    "model_name": model_name,
                    "symbol": sym,
                    "side": side,
                    "quantity": 0.0,
                    "entry_price": entry_price,
                    "exit_price": 0.0,
                    "pnl": 0.0,
                    "trigger": trigger,
                    "close_fraction": close_fraction,
                    "status": OrderStatus.REJECTED.value,
                })
                await self._log_risk_event(
                    "warning",
                    sym,
                    f"[{model_name}] {reason} but OKX close submission failed: {e}",
                    model_name,
                )

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
        candidates = []
        handled_keys: set[tuple[str, str]] = set()

        # Group positions by (model_name, symbol)
        grouped: dict[tuple[str, str], list[dict]] = {}
        for p in open_positions:
            key = (p.get("model_name", ""), p["symbol"])
            if key[0]:
                grouped.setdefault(key, []).append(p)

        if not grouped:
            return candidates, handled_keys

        grouped_items = list(grouped.items())
        portfolio_profit_context = self._portfolio_profit_protection_context(open_positions)
        fast_scan = self._scan_position_review_groups(
            grouped_items,
            feature_vectors,
            portfolio_profit_context,
        )
        grouped_items.sort(
            key=lambda item: (
                -fast_scan.get(item[0], {}).get("priority_score", 0.0),
                item[0][1],
            )
        )
        total_groups = len(grouped_items)
        max_groups = max(1, int(max_groups_override or POSITION_REVIEW_MAX_GROUPS_PER_ROUND))
        if total_groups > max_groups:
            urgent_exit_items = [
                item for item in grouped_items
                if self._is_urgent_position_exit_scan(fast_scan.get(item[0], {}))
            ]
            deferred_exit_items = [
                item for item in grouped_items
                if (
                    item not in urgent_exit_items
                    and self._position_review_defer_count(item[0]) >= 2
                    and fast_scan.get(item[0], {}).get("exit_score", 0.0) >= POSITION_REVIEW_FAST_EXIT_SCORE
                )
            ]
            urgent_exit_items.extend(deferred_exit_items)
            if urgent_exit_items:
                max_groups = max(
                    max_groups,
                    min(
                        total_groups,
                        POSITION_REVIEW_URGENT_EXIT_MAX_GROUPS_PER_ROUND,
                        len(urgent_exit_items) + 2,
                    ),
                )
            loss_watch_items = [
                item for item in grouped_items
                if (
                    item not in urgent_exit_items
                    and "loss_watch" in str(fast_scan.get(item[0], {}).get("reason") or "")
                )
            ]
            profit_exit_items = [
                item for item in grouped_items
                if (
                    item not in urgent_exit_items
                    and fast_scan.get(item[0], {}).get("exit_score", 0.0) >= POSITION_REVIEW_FAST_EXIT_SCORE
                    and any(
                        marker in str(fast_scan.get(item[0], {}).get("reason") or "")
                        for marker in (
                            "profit_retrace",
                            "profit_lock_candidate",
                            "portfolio_profit_protection_focus",
                        )
                    )
                )
            ]
            priority_items = [
                item for item in grouped_items
                if fast_scan.get(item[0], {}).get("priority_score", 0.0) >= POSITION_REVIEW_FAST_ADD_SCORE
            ]
            normal_items = [
                item for item in grouped_items
                if item not in priority_items and item not in urgent_exit_items and item not in loss_watch_items and item not in profit_exit_items
            ]
            priority_slots = min(
                len(priority_items),
                max(0, min(max_groups, int(POSITION_REVIEW_PRIORITY_MAX_GROUPS_PER_ROUND))),
            )
            selected_items = []
            for item in urgent_exit_items + profit_exit_items + loss_watch_items:
                if item not in selected_items:
                    selected_items.append(item)
            remaining_priority_slots = max(priority_slots - len(selected_items), 0)
            if remaining_priority_slots > 0:
                exit_items = [
                    item for item in priority_items
                    if item not in selected_items
                    and fast_scan.get(item[0], {}).get("exit_score", 0.0) >= POSITION_REVIEW_FAST_EXIT_SCORE
                ]
                add_items = [
                    item for item in priority_items
                    if item not in selected_items and item not in exit_items and not position_entry_pause_reason
                ]
                urgent_items = exit_items + add_items
                selected_items.extend(urgent_items[:remaining_priority_slots])
            remaining_slots = max_groups - len(selected_items)
            if remaining_slots > 0 and normal_items:
                start = self._position_review_cursor % len(normal_items)
                rotated = normal_items[start:] + normal_items[:start]
                selected_items.extend(rotated[:remaining_slots])
                self._position_review_cursor = (start + remaining_slots) % len(normal_items)
            if len(selected_items) < max_groups:
                selected_keys = {item[0] for item in selected_items}
                fallback_items = [item for item in grouped_items if item[0] not in selected_keys]
                selected_items.extend(fallback_items[:max_groups - len(selected_items)])
            selected_keys = {item[0] for item in selected_items}
            for key in selected_keys:
                self._position_review_defer_counts.pop(key, None)
            skipped_items = [item for item in grouped_items if item[0] not in selected_keys]
            await self._record_fast_position_scan_holds(
                skipped_items,
                fast_scan,
                feature_vectors,
                portfolio_profit_context,
                results,
                round_decision_ids,
                position_entry_pause_reason=position_entry_pause_reason,
            )
            grouped_items = selected_items
            logger.info(
                "position review batched",
                selected=len(grouped_items),
                total=total_groups,
                max_groups=max_groups,
                urgent_exit_forced=len(urgent_exit_items),
                deferred_exit_forced=len(deferred_exit_items),
                loss_watch_forced=len(loss_watch_items),
                profit_exit_forced=len(profit_exit_items),
                priority_selected=sum(1 for item in grouped_items if item in priority_items),
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
        market_regime_context = self._market_regime_context(feature_vectors)
        strategy_mode_context = await self._strategy_mode_context(
            self._get_model_execution_mode(ENSEMBLE_TRADER_NAME),
            market_regime_context,
            open_positions,
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
                except Exception:
                    continue

            try:
                memory_context = await self._expert_memory_context(normalized_symbol or symbol)
                daily_target_context = await self._daily_target_context()
                ml_signal_context = self.ml_signal_service.predict(fv)
                local_ai_tools_context = await self._local_ai_tools_context(
                    fv,
                    ml_signal_context,
                    open_positions=open_positions,
                    include_exit_advice=True,
                )
                position_agent_skills = self.agent_skills.position_skills(
                    position_entry_pause_reason=position_entry_pause_reason,
                    ml_signal=ml_signal_context,
                    local_ai_tools=local_ai_tools_context,
                    portfolio_profit_protection=portfolio_symbol_context,
                )
                analysis_started = datetime.now(timezone.utc)
                if model_name == ENSEMBLE_TRADER_NAME:
                    decision, _opinions = await self.ensemble.decide(
                        fv,
                        {
                            "open_positions": open_positions,
                            "trading_mode": mode_manager.mode.value,
                            "review_positions": True,
                            "position_entry_disabled": bool(position_entry_pause_reason),
                            "position_entry_pause_reason": position_entry_pause_reason or "",
                            **memory_context,
                            "daily_target": daily_target_context,
                            "market_regime": market_regime_context,
                            "strategy_mode": strategy_mode_context,
                            "ml_signal": {} if PRE_AGENT_SKILLS_ROLLBACK_MODE else ml_signal_context,
                            "local_ai_tools": {} if PRE_AGENT_SKILLS_ROLLBACK_MODE else local_ai_tools_context,
                            "ml_signal_prompt_enabled": LOCAL_QUANT_PROMPT_ENABLED,
                            "local_ai_tools_prompt_enabled": LOCAL_QUANT_PROMPT_ENABLED,
                            "portfolio_profit_protection": portfolio_symbol_context,
                            "position_profit_peak": position_profit_peak_context,
                        },
                    )
                else:
                    model = self.models.get(model_name)
                    if model is None:
                        continue
                    decision = await model.decide(
                        fv,
                        {
                            "open_positions": open_positions,
                            "trading_mode": mode_manager.mode.value,
                            "review_positions": True,
                            "position_entry_disabled": bool(position_entry_pause_reason),
                            "position_entry_pause_reason": position_entry_pause_reason or "",
                            "portfolio_profit_protection": portfolio_symbol_context,
                            "position_profit_peak": position_profit_peak_context,
                        },
                    )
                self._attach_decision_timing(decision, analysis_started, "position_review")
                raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                raw["analysis_type"] = "position_review"
                raw["review_positions"] = True
                if portfolio_symbol_context.get("active"):
                    raw["portfolio_profit_protection"] = portfolio_symbol_context
                if position_profit_peak_context:
                    raw["position_profit_peak"] = position_profit_peak_context
                decision.raw_response = raw
                self.agent_skills.attach(
                    decision,
                    phase="position_review",
                    skills=position_agent_skills,
                    note="持仓分析前的 Agent/Skills 证据快照。",
                )
            except Exception as e:
                logger.error("review position decide failed", model=model_name, symbol=symbol, error=str(e))
                continue

            if not isinstance(decision, DecisionOutput):
                continue

            decision = self._normalize_review_decision_for_positions(decision, positions)
            model_mode = self._get_model_execution_mode(model_name)
            decision_db_id = await self._log_decision(decision, is_paper=(model_mode == "paper"))
            self._decision_count += 1
            if decision_db_id is not None and round_decision_ids is not None:
                round_decision_ids.add(decision_db_id)

            handled_keys.add((model_name, self._normalize_position_symbol(symbol)))

            risk_alert = self._position_review_risk_alert(decision, positions)
            if risk_alert:
                self._attach_position_review_risk_alert(decision, risk_alert)

            if decision.is_hold:
                if risk_alert:
                    await self._log_position_review_risk_result(
                        decision,
                        model_name,
                        "未提交订单：持仓复盘结论为继续持有或暂不加仓。",
                    )
                if decision_db_id is not None:
                    await self._mark_decision_reason(
                        decision_db_id,
                        "持仓复盘结论为继续持有或暂不加仓，未提交订单。",
                    )
                continue

            if decision.is_entry:
                if position_entry_pause_reason:
                    block_reason = (
                        "触发账户风险限制后，持仓复盘只允许平仓、减仓或继续持有；"
                        "本次同方向加仓/新增仓位信号已跳过。"
                        f"触发原因：{position_entry_pause_reason}"
                    )
                    raw_response = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                    raw_response["position_entry_guard"] = {
                        "applied": True,
                        "reason": "new_entry_paused_during_position_review",
                        "pause_reason": position_entry_pause_reason,
                    }
                    decision.raw_response = raw_response
                    if decision_db_id is not None:
                        await self._mark_decision_raw_response(decision_db_id, raw_response)
                        await self._mark_decision_reason(decision_db_id, block_reason)
                    if results is not None:
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": decision.action.value,
                            "approved": True,
                            "confidence": decision.confidence,
                            "executed": False,
                            "execution_status": "skipped",
                            "reason": block_reason,
                            "is_paper": (model_mode == "paper"),
                        })
                    continue
                capacity_reason = self._entry_capacity_reason(
                    model_name,
                    decision,
                    open_positions,
                    {"model_totals": {}, "symbol_side": {}, "side_totals": {}},
                )
                if capacity_reason:
                    if risk_alert:
                        await self._log_position_review_risk_result(
                            decision,
                            model_name,
                            f"未执行：{capacity_reason}",
                        )
                    if decision_db_id is not None:
                        await self._mark_decision_reason(decision_db_id, capacity_reason)
                    continue

            # Lightweight risk check for the review action.
            model_positions = [p for p in open_positions if p.get("model_name") == model_name]
            assessment = self.risk_engine.assess(
                decision,
                current_positions=model_positions,
                account_balance=await self._get_account_balance(model_name),
                headlines=fv.recent_headlines if hasattr(fv, "recent_headlines") else [],
                sentiment_scores=[],
                price_change_1m=fv.returns_1 if hasattr(fv, "returns_1") else 0.0,
                volume_ratio=fv.volume_ratio if hasattr(fv, "volume_ratio") else 1.0,
                adx_14=fv.adx_14 if hasattr(fv, "adx_14") else None,
            )

            if not assessment.approved:
                logger.info("risk blocked close", model=model_name, symbol=symbol,
                            reason=assessment.rejection_reason)
                if risk_alert:
                    await self._log_position_review_risk_result(
                        decision,
                        model_name,
                        f"Not executed: {assessment.rejection_reason or 'risk engine rejected this position review decision'}",
                    )
                if decision_db_id is not None:
                    await self._mark_decision_reason(
                        decision_db_id,
                        assessment.rejection_reason or "risk engine rejected this position review decision",
                    )
                continue

            executed = assessment.decision if assessment.decision else decision
            if executed is not decision and decision.raw_response and not executed.raw_response:
                executed.raw_response = decision.raw_response
                executed.feature_snapshot = executed.feature_snapshot or decision.feature_snapshot
            if executed.is_hold:
                if risk_alert:
                    await self._log_position_review_risk_result(
                        executed,
                        model_name,
                        "未提交订单：持仓复盘经风控调整为观望。",
                    )
                if decision_db_id is not None:
                    await self._mark_decision_reason(
                        decision_db_id,
                        "持仓复盘经风控调整为观望，未提交订单。",
                    )
                continue
            if executed.is_entry and position_entry_pause_reason:
                block_reason = (
                    "触发账户风险限制后，持仓复盘只允许平仓、减仓或继续持有；"
                    "风控调整后的同方向加仓/新增仓位信号已跳过。"
                    f"触发原因：{position_entry_pause_reason}"
                )
                raw_response = executed.raw_response if isinstance(executed.raw_response, dict) else {}
                raw_response["position_entry_guard"] = {
                    "applied": True,
                    "reason": "new_entry_paused_during_position_review",
                    "pause_reason": position_entry_pause_reason,
                }
                executed.raw_response = raw_response
                if decision_db_id is not None:
                    await self._mark_decision_raw_response(decision_db_id, raw_response)
                    await self._mark_decision_reason(decision_db_id, block_reason)
                if results is not None:
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": executed.action.value,
                        "approved": True,
                        "confidence": executed.confidence,
                        "executed": False,
                        "execution_status": "skipped",
                        "reason": block_reason,
                        "is_paper": (model_mode == "paper"),
                    })
                continue
            if executed.is_exit:
                review_guard_reason = await self.exit_policy.fee_churn_guard_reason(model_name, executed)
                if review_guard_reason:
                    if decision_db_id is not None:
                        await self._mark_decision_reason(decision_db_id, review_guard_reason)
                    if risk_alert:
                        await self._log_position_review_risk_result(
                            executed,
                            model_name,
                            f"未执行：{review_guard_reason}",
                        )
                    if results is not None:
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": executed.action.value,
                            "approved": True,
                            "confidence": executed.confidence,
                            "executed": False,
                            "execution_status": "skipped",
                            "reason": review_guard_reason,
                            "is_paper": (model_mode == "paper"),
                        })
                    continue
                if results is not None:
                    logger.info(
                        "review exit decision executing immediately",
                        model=model_name,
                        symbol=symbol,
                        action=executed.action.value,
                        decision_id=decision_db_id,
                    )
                    await self._execute_candidate(
                        symbol,
                        model_name,
                        executed,
                        assessment,
                        decision_db_id,
                        results,
                        open_positions=open_positions,
                    )
                    if decision_db_id is not None:
                        await self._ensure_decision_final_state(
                            decision_db_id,
                            symbol,
                            model_name,
                            executed,
                            results,
                        )
                    continue
            candidates.append((symbol, model_name, executed, assessment, decision_db_id))

        return candidates, handled_keys

    async def _record_fast_position_scan_holds(
        self,
        skipped_items: list[tuple[tuple[str, str], list[dict]]],
        fast_scan: dict[tuple[str, str], dict[str, Any]],
        feature_vectors: dict[str, Any],
        portfolio_profit_context: dict[str, Any] | None,
        results: dict[str, Any] | None,
        round_decision_ids: set[int] | None,
        position_entry_pause_reason: str | None = None,
    ) -> None:
        """Persist lightweight position scan results for groups not sent to the slow LLM."""
        if not skipped_items:
            return
        for (model_name, symbol), _positions in skipped_items:
            key = (model_name, symbol)
            normalized = self._normalize_position_symbol(symbol)
            scan = fast_scan.get((model_name, symbol), {})
            priority_score = self._safe_float(scan.get("priority_score"), 0.0)
            exit_score = self._safe_float(scan.get("exit_score"), 0.0)
            scan_reason = str(scan.get("reason") or "")
            urgent_exit = self._is_urgent_position_exit_scan(scan)
            should_count_defer = urgent_exit or exit_score >= POSITION_REVIEW_FAST_EXIT_SCORE
            if should_count_defer:
                self._position_review_defer_counts[key] = self._position_review_defer_count(key) + 1
            else:
                self._position_review_defer_counts.pop(key, None)
            defer_count = self._position_review_defer_count(key)
            if exit_score >= POSITION_REVIEW_FAST_EXIT_SCORE:
                reason = (
                    "快速持仓扫描发现需要复盘的平仓/锁盈信号，"
                    "但本轮慢专家名额已满，已记录并等待下一轮优先处理；"
                    f"优先级 {priority_score:.1f}，退出分 {exit_score:.1f}。"
                )
                if urgent_exit:
                    reason += " 该信号属于紧急退出类，下一轮会优先插队深度复盘。"
                if defer_count >= 2:
                    reason += f" 已连续跳过 {defer_count} 轮，下一轮将强制插队。"
            else:
                reason = (
                    "快速持仓扫描未发现必须立刻交给慢专家的平仓/加仓信号；"
                    f"优先级 {priority_score:.1f}。"
                )
            if scan_reason:
                reason += f" 触发项：{scan_reason}"
            portfolio_symbol_context = self._portfolio_profit_protection_symbol_context(
                portfolio_profit_context or {},
                model_name or ENSEMBLE_TRADER_NAME,
                normalized or symbol,
                _positions,
            )
            if portfolio_symbol_context.get("active") and portfolio_symbol_context.get("is_focus"):
                reason += (
                    " Portfolio profit protection is active; this high-contribution "
                    "position was noted for profit-lock review."
                )
            fast_scan_skills = self.agent_skills.position_skills(
                position_entry_pause_reason=position_entry_pause_reason,
                ml_signal=None,
                local_ai_tools=None,
                portfolio_profit_protection=portfolio_symbol_context,
            )
            fv = feature_vectors.get(symbol) or feature_vectors.get(normalized)
            raw_response = {
                "analysis_type": "position_review",
                "position_fast_scan": {
                    "skipped_llm": True,
                    "priority_score": round(self._safe_float(scan.get("priority_score"), 0.0), 4),
                    "exit_score": round(self._safe_float(scan.get("exit_score"), 0.0), 4),
                    "add_score": round(self._safe_float(scan.get("add_score"), 0.0), 4),
                    "reason": scan.get("reason") or "",
                },
                "agent_skills": {
                    "version": 1,
                    "phases": {
                        "position_fast_scan": {
                            "phase": "position_fast_scan",
                            "recorded_at": datetime.now(timezone.utc).isoformat(),
                            "note": (
                                "快速扫描记录：如果退出分达到优先线，表示发现了平仓/锁盈复盘信号；"
                                "如果未达到优先线，则只是普通轮转观察。"
                            ),
                            "skills": [skill.to_dict() for skill in fast_scan_skills],
                        },
                    },
                    "summary": self.agent_skills.summary(fast_scan_skills),
                },
            }
            if portfolio_symbol_context.get("active"):
                raw_response["portfolio_profit_protection"] = portfolio_symbol_context
            decision = DecisionOutput(
                model_name=model_name or ENSEMBLE_TRADER_NAME,
                symbol=symbol,
                action=Action.HOLD,
                confidence=0.0,
                reasoning=reason,
                position_size_pct=0.0,
                suggested_leverage=1.0,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                raw_response=raw_response,
                feature_snapshot=fv.to_dict() if fv is not None and hasattr(fv, "to_dict") else {},
            )
            model_mode = self._get_model_execution_mode(model_name or ENSEMBLE_TRADER_NAME)
            decision_db_id = await self._log_decision(decision, is_paper=(model_mode == "paper"))
            if decision_db_id is not None:
                if round_decision_ids is not None:
                    round_decision_ids.add(decision_db_id)
                await self._mark_decision_reason(decision_db_id, reason)
                self._decision_count += 1
            if results is not None:
                results["decisions"].append({
                    "model": model_name or ENSEMBLE_TRADER_NAME,
                    "symbol": symbol,
                    "action": "hold",
                    "approved": True,
                    "confidence": 0.0,
                    "executed": False,
                    "execution_status": "fast_position_scan",
                    "reason": reason,
                    "is_paper": (model_mode == "paper"),
                })

    def _scan_position_review_groups(
        self,
        grouped_items: list[tuple[tuple[str, str], list[dict]]],
        feature_vectors: dict[str, Any],
        portfolio_profit_context: dict[str, Any] | None = None,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Fast pass over every open position before spending AI time.

        The score decides which position groups jump the AI queue. It is only a
        triage signal; final add/close actions still require the normal AI and
        risk checks.
        """
        scans: dict[tuple[str, str], dict[str, Any]] = {}
        for key, positions in grouped_items:
            symbol = key[1]
            normalized = self._normalize_position_symbol(symbol)
            fv = feature_vectors.get(symbol) or feature_vectors.get(normalized)
            exit_score = 0.0
            add_score = 0.0
            reasons: list[str] = []

            by_side: dict[str, list[dict]] = {}
            for pos in positions or []:
                side = str(pos.get("side") or "").lower()
                if side in {"long", "short"}:
                    by_side.setdefault(side, []).append(pos)

            for side, side_positions in by_side.items():
                aggregate = self._aggregate_position_group(
                    side_positions,
                    key[0],
                    normalized or symbol,
                    side,
                )
                if not aggregate:
                    continue
                pos_exit_score, pos_reasons = self._fast_position_exit_score(aggregate, fv)
                if pos_exit_score > exit_score:
                    exit_score = pos_exit_score
                reasons.extend(pos_reasons)

            add_score, add_reason = self._fast_position_add_score(positions, fv)
            if add_reason:
                reasons.append(add_reason)

            portfolio_score, portfolio_reasons = self._portfolio_profit_protection_score(
                portfolio_profit_context or {},
                key[0],
                normalized,
            )
            if portfolio_score > exit_score:
                exit_score = portfolio_score
            reasons.extend(portfolio_reasons)

            priority_score = max(exit_score, add_score)
            scans[key] = {
                "priority_score": priority_score,
                "exit_score": exit_score,
                "add_score": add_score,
                "reason": "; ".join(dict.fromkeys(reasons))[:260],
            }
        return scans

    def _is_urgent_position_exit_scan(self, scan: dict[str, Any] | None) -> bool:
        if not isinstance(scan, dict):
            return False
        exit_score = self._safe_float(scan.get("exit_score"), 0.0)
        reason = str(scan.get("reason") or "")
        if exit_score >= 90.0:
            return True
        return any(marker in reason for marker in POSITION_REVIEW_URGENT_EXIT_MARKERS)

    def _position_review_defer_count(self, key: tuple[str, str]) -> int:
        return int(self._position_review_defer_counts.get(key, 0) or 0)

    def _position_review_budget_context(
        self,
        open_positions: list[dict],
        feature_vectors: dict[str, Any],
        *,
        base_market_limit: int,
        run_position_analysis: bool,
        run_market_analysis: bool,
        new_pair_pause_reason: str | None = None,
    ) -> dict[str, Any]:
        """Allocate slow AI work between position protection and new entries."""
        base_market_limit = max(0, int(base_market_limit or 0))
        position_group_count = self._open_position_group_count(open_positions)
        roster_underfilled = position_group_count < PORTFOLIO_MIN_POSITION_GROUPS_TARGET
        if not run_position_analysis or not open_positions:
            market_limit = base_market_limit if run_market_analysis else 0
            if run_market_analysis and roster_underfilled:
                market_limit = max(market_limit, PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN)
            return {
                "risk_level": "none",
                "market_symbol_limit": market_limit,
                "position_max_groups": POSITION_REVIEW_MAX_GROUPS_PER_ROUND,
                "forced_exit_groups": 0,
                "priority_groups": 0,
                "total_position_groups": 0,
                "roster_underfilled": roster_underfilled,
                "position_group_count": position_group_count,
                "target_position_groups": PORTFOLIO_MIN_POSITION_GROUPS_TARGET,
                "reason": (
                    "没有需要调度的持仓风险，市场分析使用补仓候选数量。"
                    if roster_underfilled
                    else "没有需要调度的持仓风险，市场分析使用基础候选数量。"
                ),
            }

        grouped: dict[tuple[str, str], list[dict]] = {}
        for pos in open_positions or []:
            symbol = self._normalize_position_symbol(pos.get("symbol"))
            model = str(pos.get("model_name") or ENSEMBLE_TRADER_NAME)
            if model and symbol:
                grouped.setdefault((model, symbol), []).append(pos)
        grouped_items = list(grouped.items())
        portfolio_profit_context = self._portfolio_profit_protection_context(open_positions)
        fast_scan = self._scan_position_review_groups(
            grouped_items,
            feature_vectors,
            portfolio_profit_context,
        )
        forced_exit = [
            scan for scan in fast_scan.values()
            if self._safe_float(scan.get("exit_score"), 0.0) >= POSITION_REVIEW_FAST_EXIT_SCORE
        ]
        urgent_exit = [
            scan for scan in fast_scan.values()
            if self._is_urgent_position_exit_scan(scan)
        ]
        high_exit = [
            scan for scan in fast_scan.values()
            if self._safe_float(scan.get("exit_score"), 0.0) >= 90.0
        ]
        priority = [
            scan for scan in fast_scan.values()
            if self._safe_float(scan.get("priority_score"), 0.0) >= POSITION_REVIEW_FAST_ADD_SCORE
        ]

        risk_level = "low"
        position_max_groups = POSITION_REVIEW_MAX_GROUPS_PER_ROUND
        market_limit = base_market_limit if run_market_analysis else 0
        if high_exit or len(forced_exit) >= 3:
            risk_level = "high"
            position_max_groups = max(
                POSITION_REVIEW_HIGH_RISK_MAX_GROUPS_PER_ROUND,
                min(
                    len(grouped_items),
                    POSITION_REVIEW_URGENT_EXIT_MAX_GROUPS_PER_ROUND,
                    len(urgent_exit) + 2 if urgent_exit else POSITION_REVIEW_HIGH_RISK_MAX_GROUPS_PER_ROUND,
                ),
            )
            market_limit = min(
                base_market_limit,
                max(MARKET_ANALYSIS_HIGH_RISK_MIN_EXPLORATION_SYMBOLS, MARKET_ANALYSIS_HIGH_RISK_CAP),
            )
        elif forced_exit or len(priority) >= 3:
            risk_level = "medium"
            position_max_groups = max(POSITION_REVIEW_MAX_GROUPS_PER_ROUND, min(len(priority) + 2, POSITION_REVIEW_HIGH_RISK_MAX_GROUPS_PER_ROUND))
            market_limit = min(
                base_market_limit,
                max(MARKET_ANALYSIS_MIN_EXPLORATION_SYMBOLS, MARKET_ANALYSIS_MEDIUM_RISK_CAP),
            )

        if new_pair_pause_reason:
            market_limit = 0
        elif run_market_analysis and base_market_limit > 0 and market_limit <= 0:
            market_limit = MARKET_ANALYSIS_HIGH_RISK_MIN_EXPLORATION_SYMBOLS
        if (
            roster_underfilled
            and run_market_analysis
            and not new_pair_pause_reason
            and risk_level != "high"
        ):
            market_limit = max(market_limit, PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN)

        return {
            "risk_level": risk_level,
            "market_symbol_limit": max(0, int(market_limit)),
            "position_max_groups": max(1, int(position_max_groups)),
            "forced_exit_groups": len(forced_exit),
            "urgent_exit_groups": len(urgent_exit),
            "high_exit_groups": len(high_exit),
            "priority_groups": len(priority),
            "total_position_groups": len(grouped_items),
            "roster_underfilled": roster_underfilled,
            "position_group_count": position_group_count,
            "target_position_groups": PORTFOLIO_MIN_POSITION_GROUPS_TARGET,
            "reason": (
                f"持仓风险等级 {risk_level}：强退出 {len(forced_exit)} 组，"
                f"紧急退出 {len(urgent_exit)} 组，高风险 {len(high_exit)} 组，优先复盘 {len(priority)} 组；"
                f"本轮持仓深度复盘最多 {int(position_max_groups)} 组，"
                f"新开仓探索保留 {max(0, int(market_limit))} 个候选。"
                + (
                    f" 当前聚合持仓 {position_group_count}/{PORTFOLIO_MIN_POSITION_GROUPS_TARGET}，补仓模式提高探索预算。"
                    if roster_underfilled and risk_level != "high"
                    else ""
                )
            ),
        }

    def _portfolio_profit_protection_context(self, open_positions: list[dict]) -> dict[str, Any]:
        """Build a portfolio-level floating-profit context for AI lock-profit review."""
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        total_unrealized = 0.0
        total_positive = 0.0
        total_notional = 0.0

        for pos in open_positions or []:
            if pos.get("is_open", True) is False:
                continue
            model_name = str(pos.get("model_name") or ENSEMBLE_TRADER_NAME)
            symbol = self._normalize_position_symbol(pos.get("symbol"))
            side = str(pos.get("side") or "").lower()
            if not symbol or side not in {"long", "short"}:
                continue
            unrealized = self._safe_float(pos.get("unrealized_pnl"), 0.0)
            entry_price = self._safe_float(pos.get("entry_price"), 0.0)
            quantity = abs(self._safe_float(pos.get("quantity"), 0.0))
            notional = abs(entry_price * quantity)
            total_unrealized += unrealized
            total_positive += max(unrealized, 0.0)
            total_notional += notional

            key = (model_name, symbol)
            item = groups.setdefault(
                key,
                {
                    "model_name": model_name,
                    "symbol": symbol,
                    "side": side,
                    "rows": 0,
                    "quantity": 0.0,
                    "notional": 0.0,
                    "unrealized_pnl": 0.0,
                    "first_opened_at": pos.get("created_at"),
                },
            )
            item["rows"] += 1
            item["quantity"] += quantity
            item["notional"] += notional
            item["unrealized_pnl"] += unrealized
            if str(item.get("side") or "") != side:
                item["side"] = "mixed"

        active = total_unrealized >= PORTFOLIO_PROFIT_PROTECTION_MIN_USDT
        ranked = sorted(
            groups.values(),
            key=lambda item: self._safe_float(item.get("unrealized_pnl"), 0.0),
            reverse=True,
        )
        focus_groups: list[dict[str, Any]] = []
        if active:
            for item in ranked:
                unrealized = self._safe_float(item.get("unrealized_pnl"), 0.0)
                share = unrealized / max(total_positive, 1e-9) if unrealized > 0 else 0.0
                if (
                    unrealized >= PORTFOLIO_PROFIT_PROTECTION_MIN_CONTRIBUTION_USDT
                    and share >= PORTFOLIO_PROFIT_PROTECTION_MIN_SHARE
                ):
                    focus_groups.append({
                        **item,
                        "profit_share": round(share, 6),
                        "profit_pct": round(unrealized / max(self._safe_float(item.get("notional"), 0.0), 1e-9), 6),
                    })
                if len(focus_groups) >= PORTFOLIO_PROFIT_PROTECTION_MAX_FOCUS_GROUPS:
                    break

        return {
            "active": active,
            "threshold_usdt": PORTFOLIO_PROFIT_PROTECTION_MIN_USDT,
            "total_unrealized_pnl": round(total_unrealized, 6),
            "total_positive_unrealized_pnl": round(total_positive, 6),
            "total_open_notional": round(total_notional, 6),
            "focus_groups": focus_groups,
            "top_groups": [
                {
                    **item,
                    "profit_share": round(
                        self._safe_float(item.get("unrealized_pnl"), 0.0) / max(total_positive, 1e-9),
                        6,
                    ) if self._safe_float(item.get("unrealized_pnl"), 0.0) > 0 else 0.0,
                    "profit_pct": round(
                        self._safe_float(item.get("unrealized_pnl"), 0.0)
                        / max(self._safe_float(item.get("notional"), 0.0), 1e-9),
                        6,
                    ),
                }
                for item in ranked[:5]
            ],
            "instruction": (
                "Portfolio floating profit has reached the winner-management line. High-contribution positions "
                "must be deep-reviewed for one of: continue holding, add to winner, partial profit lock, or full close."
                if active else ""
            ),
        }

    def _portfolio_profit_protection_symbol_context(
        self,
        context: dict[str, Any],
        model_name: str,
        symbol: str,
        positions: list[dict] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(context, dict) or not context.get("active"):
            return {"active": False}
        normalized = self._normalize_position_symbol(symbol)
        model = str(model_name or ENSEMBLE_TRADER_NAME)
        focus = [
            item for item in context.get("focus_groups", [])
            if item.get("model_name") == model and self._normalize_position_symbol(item.get("symbol")) == normalized
        ]
        top_match = [
            item for item in context.get("top_groups", [])
            if item.get("model_name") == model and self._normalize_position_symbol(item.get("symbol")) == normalized
        ]
        current = focus[0] if focus else (top_match[0] if top_match else {})
        if not current and positions:
            unrealized = sum(self._safe_float(p.get("unrealized_pnl"), 0.0) for p in positions)
            notional = sum(
                abs(self._safe_float(p.get("entry_price"), 0.0) * self._safe_float(p.get("quantity"), 0.0))
                for p in positions
            )
            current = {
                "model_name": model,
                "symbol": normalized,
                "side": str((positions[0] or {}).get("side") or ""),
                "rows": len(positions),
                "unrealized_pnl": round(unrealized, 6),
                "notional": round(notional, 6),
                "profit_pct": round(unrealized / max(notional, 1e-9), 6),
            }
        return {
            "active": True,
            "is_focus": bool(focus),
            "threshold_usdt": context.get("threshold_usdt"),
            "total_unrealized_pnl": context.get("total_unrealized_pnl"),
            "total_positive_unrealized_pnl": context.get("total_positive_unrealized_pnl"),
            "current_group": current,
            "top_groups": context.get("top_groups", [])[:3],
            "required_choice": [
                "continue_hold_with_reason",
                "add_to_winner_if_trend_continues",
                "partial_lock_profit",
                "full_close",
            ],
            "instruction": context.get("instruction") or "",
        }

    def _position_profit_peak_context(
        self,
        model_name: str,
        symbol: str,
        positions: list[dict] | None,
    ) -> dict[str, Any]:
        """Expose per-position floating-profit peak to the AI evidence layer."""
        if not positions:
            return {}
        normalized = self._normalize_position_symbol(symbol)
        model = str(model_name or ENSEMBLE_TRADER_NAME)
        best: dict[str, Any] = {}
        by_side: dict[str, list[dict]] = {}
        for pos in positions or []:
            side = str(pos.get("side") or "").lower()
            if side not in {"long", "short"}:
                continue
            by_side.setdefault(side, []).append(pos)

        for side, side_positions in by_side.items():
            pos = self._aggregate_position_group(side_positions, model, normalized or symbol, side)
            if not pos:
                continue
            key = self._position_peak_key(model, normalized or str(pos.get("symbol") or symbol), side)
            state = self._position_profit_peaks.get(key) or {}
            peak = self._safe_float(
                state.get("peak_unrealized_pnl", state.get("peak_pnl")),
                0.0,
            )
            current = self._safe_float(pos.get("unrealized_pnl"), 0.0)
            peak = max(peak, current)
            if peak <= 0 and current <= 0:
                continue
            retrace_abs = max(peak - current, 0.0)
            retrace_ratio = retrace_abs / max(peak, 1e-9) if peak > 0 else 0.0
            item = {
                "model_name": model,
                "symbol": normalized or str(pos.get("symbol") or symbol),
                "side": side,
                "rows": len(side_positions),
                "quantity": round(self._safe_float(pos.get("quantity"), 0.0), 8),
                "notional": round(self._safe_float(pos.get("notional"), 0.0), 6),
                "peak_unrealized_pnl": round(peak, 6),
                "current_unrealized_pnl": round(current, 6),
                "profit_retrace_abs": round(retrace_abs, 6),
                "profit_retrace_ratio": round(retrace_ratio, 6),
                "peak_pnl_ratio": round(self._safe_float(state.get("peak_pnl_ratio"), 0.0), 6),
                "updated_at": state.get("updated_at"),
            }
            if not best or self._safe_float(item.get("profit_retrace_abs"), 0.0) > self._safe_float(best.get("profit_retrace_abs"), 0.0):
                best = item
        return best

    def _aggregate_position_group(
        self,
        positions: list[dict] | None,
        model_name: str,
        symbol: str,
        side: str,
    ) -> dict[str, Any]:
        """Aggregate same-symbol/same-side fragments before profit-lock checks."""
        rows = [
            p for p in (positions or [])
            if str(p.get("side") or "").lower() == side
        ]
        if not rows:
            return {}

        total_qty = 0.0
        entry_value = 0.0
        current_value = 0.0
        unrealized = 0.0
        stop_value = 0.0
        stop_weight = 0.0
        take_profit_value = 0.0
        take_profit_weight = 0.0
        leverage_value = 0.0
        leverage_weight = 0.0
        created_at = None

        for pos in rows:
            qty = abs(self._safe_float(pos.get("quantity"), 0.0))
            entry = self._safe_float(pos.get("entry_price"), 0.0)
            current = self._safe_float(pos.get("current_price"), entry)
            if qty <= 0 or entry <= 0:
                continue
            total_qty += qty
            entry_value += entry * qty
            current_value += (current if current > 0 else entry) * qty
            unrealized += self._safe_float(pos.get("unrealized_pnl"), 0.0)
            stop = self._safe_float(pos.get("stop_loss") or pos.get("stop_loss_price"), 0.0)
            if stop > 0:
                stop_value += stop * qty
                stop_weight += qty
            take_profit = self._safe_float(pos.get("take_profit") or pos.get("take_profit_price"), 0.0)
            if take_profit > 0:
                take_profit_value += take_profit * qty
                take_profit_weight += qty
            leverage = self._safe_float(pos.get("leverage"), 0.0)
            if leverage > 0:
                leverage_value += leverage * qty
                leverage_weight += qty
            opened = pos.get("created_at")
            if created_at is None:
                created_at = opened
            else:
                try:
                    a = datetime.fromisoformat(str(created_at).replace("Z", "+00:00")) if not isinstance(created_at, datetime) else created_at
                    b = datetime.fromisoformat(str(opened).replace("Z", "+00:00")) if not isinstance(opened, datetime) else opened
                    if b < a:
                        created_at = opened
                except Exception:
                    pass

        if total_qty <= 0:
            return {}
        entry_price = entry_value / total_qty
        current_price = current_value / total_qty if current_value > 0 else entry_price
        notional = entry_price * total_qty
        return {
            "model_name": model_name or ENSEMBLE_TRADER_NAME,
            "symbol": self._normalize_position_symbol(symbol) or symbol,
            "side": side,
            "quantity": total_qty,
            "entry_price": entry_price,
            "current_price": current_price,
            "notional": notional,
            "unrealized_pnl": unrealized,
            "stop_loss": stop_value / stop_weight if stop_weight > 0 else 0.0,
            "take_profit": take_profit_value / take_profit_weight if take_profit_weight > 0 else 0.0,
            "leverage": leverage_value / leverage_weight if leverage_weight > 0 else 1.0,
            "is_open": True,
            "created_at": created_at,
            "rows": len(rows),
        }

    def _portfolio_profit_protection_score(
        self,
        context: dict[str, Any],
        model_name: str,
        symbol: str,
    ) -> tuple[float, list[str]]:
        if not isinstance(context, dict) or not context.get("active"):
            return 0.0, []
        normalized = self._normalize_position_symbol(symbol)
        model = str(model_name or ENSEMBLE_TRADER_NAME)
        for item in context.get("focus_groups", []):
            if item.get("model_name") == model and self._normalize_position_symbol(item.get("symbol")) == normalized:
                return PORTFOLIO_PROFIT_PROTECTION_EXIT_SCORE, ["portfolio_profit_protection_focus"]
        return 0.0, []

    def _fast_position_exit_score(self, pos: dict[str, Any], fv: Any | None) -> tuple[float, list[str]]:
        reasons: list[str] = []
        try:
            entry = float(pos.get("entry_price") or 0.0)
            current = float(pos.get("current_price") or entry or 0.0)
            qty = abs(float(pos.get("quantity") or 0.0))
            stop = float(pos.get("stop_loss") or pos.get("stop_loss_price") or 0.0)
            unrealized = float(pos.get("unrealized_pnl") or 0.0)
        except (TypeError, ValueError):
            return 0.0, reasons
        if entry <= 0 or current <= 0 or qty <= 0:
            return 0.0, reasons

        side = str(pos.get("side") or "").lower()
        notional = max(entry * qty, 1e-9)
        pnl_ratio = unrealized / notional
        estimated_round_trip_fee = max(notional * ESTIMATED_TAKER_FEE_PCT * 2.0, 1e-9)
        score = 0.0
        if pnl_ratio <= -0.02 or unrealized <= -8.0:
            score = max(score, 95.0)
            reasons.append("loss_expanding")
        elif pnl_ratio <= -0.01 or unrealized <= -3.0:
            score = max(score, 82.0)
            reasons.append("loss_needs_review")
        elif pnl_ratio <= -0.006 or unrealized <= -1.2:
            score = max(score, 70.0)
            reasons.append("loss_watch")

        if stop > 0:
            if side == "short":
                total_stop_distance = max(stop - entry, 0.0)
                used_distance = max(current - entry, 0.0)
            else:
                total_stop_distance = max(entry - stop, 0.0)
                used_distance = max(entry - current, 0.0)
            if total_stop_distance > 0:
                stop_progress = used_distance / total_stop_distance
                if stop_progress >= 0.85:
                    score = max(score, 96.0)
                    reasons.append("near_stop")
                elif stop_progress >= FAST_RISK_NEAR_STOP_PROGRESS:
                    score = max(score, 78.0)
                    reasons.append("stop_risk_rising")

        peak_key = self._position_peak_key(
            str(pos.get("model_name") or ENSEMBLE_TRADER_NAME),
            str(pos.get("symbol") or ""),
            side,
        )
        peak_state = self._position_profit_peaks.get(peak_key, {})
        peak_pnl = self._safe_float(
            peak_state.get("peak_unrealized_pnl", peak_state.get("peak_pnl")),
            0.0,
        )
        if unrealized >= max(
            notional * PROFIT_PROTECTION_MIN_NET_PNL_RATIO,
            estimated_round_trip_fee * PROFIT_PROTECTION_MIN_FEE_MULTIPLE,
            PROFIT_PROTECTION_MIN_NET_USDT,
        ):
            score = max(score, 72.0)
            reasons.append("profit_lock_candidate")
        if peak_pnl >= 0.8 and unrealized > 0 and unrealized <= peak_pnl * 0.72:
            score = max(score, 80.0)
            retrace_ratio = (peak_pnl - unrealized) / max(peak_pnl, 1e-9)
            reasons.append(f"profit_retrace:{peak_pnl:.2f}->{unrealized:.2f}U/{retrace_ratio:.0%}")

        if fv is not None:
            try:
                returns_1 = float(getattr(fv, "returns_1", 0.0) or 0.0)
                returns_5 = float(getattr(fv, "returns_5", 0.0) or 0.0)
                returns_20 = float(getattr(fv, "returns_20", 0.0) or 0.0)
                volume_ratio = float(getattr(fv, "volume_ratio", 1.0) or 1.0)
                rsi_14 = float(getattr(fv, "rsi_14", 50.0) or 50.0)
                bb_pct = float(getattr(fv, "bb_pct", 0.5) or 0.5)
                macd_diff = float(getattr(fv, "macd_diff", 0.0) or 0.0)
                adx_14 = float(getattr(fv, "adx_14", 0.0) or 0.0)
            except (TypeError, ValueError):
                returns_1 = returns_5 = returns_20 = 0.0
                volume_ratio = 1.0
                rsi_14 = 50.0
                bb_pct = 0.5
                macd_diff = 0.0
                adx_14 = 0.0
            adverse_1 = returns_1 <= -0.012 if side == "long" else returns_1 >= 0.012
            adverse_5 = returns_5 <= -0.025 if side == "long" else returns_5 >= 0.025
            if volume_ratio >= 1.1 and (adverse_1 or adverse_5):
                score = max(score, 84.0)
                reasons.append("adverse_momentum")
            reversal = self._predictive_reversal_evidence(
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
            reversal_score = self._safe_float(reversal.get("score"), 0.0)
            if reversal_score >= PREDICTIVE_REVERSAL_EXIT_SCORE:
                score = max(score, 88.0)
                reasons.append(f"predictive_reversal:{reversal_score:.0f}")
            elif reversal_score >= PREDICTIVE_REVERSAL_REVIEW_SCORE:
                score = max(score, 76.0)
                reasons.append(f"reversal_watch:{reversal_score:.0f}")

        return score, reasons

    def _fast_position_add_score(self, positions: list[dict], fv: Any | None) -> tuple[float, str | None]:
        if not positions or fv is None:
            return 0.0, None
        sides = {
            str(pos.get("side") or "").lower()
            for pos in positions
            if str(pos.get("side") or "").lower() in {"long", "short"}
        }
        if len(sides) != 1:
            return 0.0, None
        side = next(iter(sides))
        try:
            returns_1 = float(getattr(fv, "returns_1", 0.0) or 0.0)
            returns_5 = float(getattr(fv, "returns_5", 0.0) or 0.0)
            returns_20 = float(getattr(fv, "returns_20", 0.0) or 0.0)
            volume_ratio = float(getattr(fv, "volume_ratio", 1.0) or 1.0)
            adx_14 = float(getattr(fv, "adx_14", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0, None

        total_unrealized = sum(self._safe_float(pos.get("unrealized_pnl"), 0.0) for pos in positions)
        total_notional = sum(
            abs(self._safe_float(pos.get("entry_price"), 0.0) * self._safe_float(pos.get("quantity"), 0.0))
            for pos in positions
        )
        pnl_ratio = total_unrealized / max(total_notional, 1e-9)
        same_direction = (
            returns_1 > 0.0015 and returns_5 > 0.006 and returns_20 > 0.010
            if side == "long"
            else returns_1 < -0.0015 and returns_5 < -0.006 and returns_20 < -0.010
        )
        winner_direction = (
            total_unrealized >= 1.2
            and pnl_ratio >= 0.0012
            and (
                (returns_5 > 0.002 and returns_20 > 0.003)
                if side == "long"
                else (returns_5 < -0.002 and returns_20 < -0.003)
            )
        )
        if not same_direction and not winner_direction:
            return 0.0, None

        score = 62.0 if winner_direction else 58.0
        if volume_ratio >= 1.2:
            score += 8.0
        if adx_14 >= 24.0:
            score += 8.0
        if total_unrealized >= 3.0:
            score += 10.0
        elif total_unrealized >= 1.2:
            score += 6.0
        return min(score, 88.0), "winner_add_candidate" if winner_direction else "trend_add_candidate"

    def _position_needs_priority_review(self, pos: dict[str, Any]) -> bool:
        """Prioritize positions near stop, with meaningful loss, or with retracing profit."""
        try:
            entry = float(pos.get("entry_price") or 0.0)
            current = float(pos.get("current_price") or entry or 0.0)
            qty = abs(float(pos.get("quantity") or 0.0))
            stop = float(pos.get("stop_loss") or pos.get("stop_loss_price") or 0.0)
            unrealized = float(pos.get("unrealized_pnl") or 0.0)
        except (TypeError, ValueError):
            return False
        if entry <= 0 or current <= 0 or qty <= 0:
            return False
        side = str(pos.get("side") or "").lower()
        notional = max(entry * qty, 1e-9)
        pnl_ratio = unrealized / notional
        estimated_round_trip_fee = max(notional * ESTIMATED_TAKER_FEE_PCT * 2.0, 1e-9)
        if unrealized >= max(
            notional * PROFIT_PROTECTION_MIN_NET_PNL_RATIO,
            estimated_round_trip_fee * PROFIT_PROTECTION_MIN_FEE_MULTIPLE,
            PROFIT_PROTECTION_MIN_NET_USDT,
        ):
            return True
        if pnl_ratio <= -0.006 or unrealized <= -2.0:
            return True
        if stop > 0:
            if side == "short":
                total_stop_distance = max(stop - entry, 0.0)
                used_distance = max(current - entry, 0.0)
            else:
                total_stop_distance = max(entry - stop, 0.0)
                used_distance = max(entry - current, 0.0)
            if total_stop_distance > 0 and used_distance / total_stop_distance >= FAST_RISK_NEAR_STOP_PROGRESS:
                return True
        peak_key = self._position_peak_key(
            str(pos.get("model_name") or ENSEMBLE_TRADER_NAME),
            str(pos.get("symbol") or ""),
            side,
        )
        peak_state = self._position_profit_peaks.get(peak_key, {})
        peak_pnl = self._safe_float(
            peak_state.get("peak_unrealized_pnl", peak_state.get("peak_pnl")),
            0.0,
        )
        if peak_pnl >= 0.8 and unrealized > 0 and unrealized <= peak_pnl * 0.72:
            return True
        return False

    def _normalize_review_decision_for_positions(
        self,
        decision: DecisionOutput,
        positions: list[dict],
    ) -> DecisionOutput:
        """Turn opposite add signals during position review into close-first actions."""
        if not decision.is_entry:
            return decision

        target_side = "long" if decision.action == Action.LONG else "short"
        existing_sides = {
            str(p.get("side") or "").lower()
            for p in positions
            if self._normalize_position_symbol(p.get("symbol")) == self._normalize_position_symbol(decision.symbol)
        }
        if target_side in existing_sides:
            decision.reasoning += " [持仓复盘：同方向信号，按加仓候选进入风控和仓位上限检查。]"
            return decision

        close_action = None
        if target_side == "long" and "short" in existing_sides:
            close_action = Action.CLOSE_SHORT
        elif target_side == "short" and "long" in existing_sides:
            close_action = Action.CLOSE_LONG

        if close_action is None:
            return decision

        return DecisionOutput(
            model_name=decision.model_name,
            symbol=decision.symbol,
            action=close_action,
            confidence=max(decision.confidence, 0.62),
            reasoning=(
                f"{decision.reasoning} [position review: opposite entry signal detected; "
                "close existing position first, do not reverse in the same order.]"
            ),
            position_size_pct=1.0,
            suggested_leverage=1.0,
            stop_loss_pct=decision.stop_loss_pct,
            take_profit_pct=decision.take_profit_pct,
            cross_check_for=decision.cross_check_for,
            raw_response=decision.raw_response,
            feature_snapshot=decision.feature_snapshot,
        )

    def _position_review_risk_alert(
        self,
        decision: DecisionOutput,
        positions: list[dict],
    ) -> str | None:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opinions = raw.get("opinions") or []
        if not isinstance(opinions, list):
            return None

        risk_opinion = next(
            (o for o in opinions if isinstance(o, dict) and o.get("model_name") == "risk_expert"),
            None,
        )
        if not risk_opinion:
            return None

        risk_action = str(risk_opinion.get("action") or "hold")
        risk_conf = self._safe_float(risk_opinion.get("confidence"), 0.0)
        risk_reason = self._short_text(risk_opinion.get("reasoning"), 220)
        urgent_terms = (
            "一票否决",
            "硬性否决",
            "禁止",
            "紧急",
            "立即",
            "极端",
            "异常",
            "黑天鹅",
            "爆仓",
            "止损",
            "平仓",
            "严重",
            "流动性",
            "高波动",
            "风险",
        )
        urgent = (
            risk_action in {"close_long", "close_short"}
            or (risk_conf >= 0.70 and any(term in risk_reason for term in urgent_terms))
            or (decision.is_exit and risk_conf >= 0.55)
        )
        if not urgent:
            return None

        position_bits = []
        for pos in positions[:3]:
            side = "long" if pos.get("side") == "long" else "short" if pos.get("side") == "short" else str(pos.get("side") or "unknown")
            position_bits.append(
                f"{side} entry={pos.get('entry_price', '-')}, qty={pos.get('quantity', '-')}, pnl={pos.get('unrealized_pnl', 0)}"
            )
        position_text = "; ".join(position_bits) or "no position details"
        return (
            f"Position review risk alert: {decision.symbol} current {position_text}. "
            f"Risk expert action={self._action_label_text(risk_action)}, confidence={risk_conf:.0%}. "
            f"Reason={risk_reason or 'risk expert did not provide details'}. "
            f"Final review action={self._action_label_text(decision.action)}."
        )

    def _attach_position_review_risk_alert(
        self,
        decision: DecisionOutput,
        message: str,
    ) -> None:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["position_review_risk_alert"] = {
            "message": message,
            "planned_action": decision.action.value,
        }
        decision.raw_response = raw

    def _position_review_alert_context(self, decision: DecisionOutput) -> dict[str, Any] | None:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        alert = raw.get("position_review_risk_alert")
        return alert if isinstance(alert, dict) else None

    async def _log_position_review_risk_result(
        self,
        decision: DecisionOutput,
        model_name: str,
        result_text: str | None = None,
        execution_result: ExecutionResult | None = None,
    ) -> None:
        alert = self._position_review_alert_context(decision)
        if not alert:
            return

        if execution_result is not None:
            if execution_result.status == OrderStatus.FILLED:
                result_text = (
                    f"已执行完成：动作={self._action_label_text(decision.action)}，"
                    f"数量={execution_result.quantity:g}，价格={execution_result.price:g}，"
                    f"订单状态={execution_result.status.value}。"
                )
            else:
                result_text = (
                    f"Execution not completed: action={self._action_label_text(decision.action)}, "
                    f"status={execution_result.status.value}, "
                    f"reason={self._execution_reason_from_result(execution_result)}"
                )

        detail = (
            f"{alert.get('message')}"
            f" system_action={self._action_label_text(decision.action)}."
            f" result={result_text or 'no execution result'}"
        )
        await self._log_risk_event(
            "position_review_warning",
            decision.symbol,
            detail,
            model_name,
            severity="critical" if decision.is_exit else "warn",
        )

    def _attach_execution_leverage_summary(
        self,
        decision: DecisionOutput,
        result: ExecutionResult,
        ai_requested_leverage: float,
    ) -> None:
        """Store AI/OKX/actual leverage in the decision and execution payloads."""
        raw_result = result.raw_response if isinstance(result.raw_response, dict) else {}
        leverage_check = raw_result.get("leverage_check") if isinstance(raw_result.get("leverage_check"), dict) else {}
        actual = self._safe_float(
            leverage_check.get("actual_leverage")
            or leverage_check.get("target_leverage")
            or decision.suggested_leverage,
            ai_requested_leverage,
        )
        okx_max = self._safe_float(
            leverage_check.get("okx_max_leverage")
            or leverage_check.get("max_leverage"),
            0.0,
        )
        target = self._safe_float(
            leverage_check.get("target_leverage") or decision.suggested_leverage,
            actual,
        )
        summary = {
            "ai_suggested_leverage": round(float(ai_requested_leverage or 1.0), 4),
            "okx_max_leverage": round(float(okx_max), 4) if okx_max > 0 else None,
            "actual_leverage": round(float(actual or target or 1.0), 4),
        }
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["execution_leverage"] = summary
        decision.raw_response = raw
        raw_result["execution_leverage"] = summary
        result.raw_response = raw_result
        decision.suggested_leverage = float(summary["actual_leverage"])

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
        okx_available = self._okx_tradeable_balance_from_snapshot(okx_snapshot)
        okx_allocatable = self._okx_allocatable_balance_from_snapshot(okx_snapshot)
        if okx_allocatable <= 0:
            return "未获取到 OKX 账户权益或余额，暂停分析新的交易对。"
        max_loss_pct = float(account_cfg.get("max_loss_pct") or settings.max_daily_loss_pct)
        max_loss_usdt = okx_allocatable * max_loss_pct if max_loss_pct > 0 else 0.0
        allocation_state = await self._execution_allocation_state(model_mode)
        total_pnl = float(allocation_state.get("total_pnl") or 0.0)
        min_available = max(10.0, okx_allocatable * 0.005)
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
        daily_profit_reason = await self._daily_profit_control_pause_reason(model_mode)
        if daily_profit_reason:
            return daily_profit_reason

        model_positions = [
            p for p in (open_positions or [])
            if p.get("model_name") == model_name and p.get("is_open", True)
        ]
        capacity_reason = self.risk_engine.position_checker.entry_capacity_reason(
            current_positions=model_positions,
            account_balance=okx_allocatable,
            min_new_margin_pct=min(
                max(float(settings.max_position_pct or 0.12) / 6.0, 0.02),
                float(settings.max_position_pct or 0.12),
            ),
            default_leverage=5.0,
            default_stop_loss_pct=0.05,
        )
        if capacity_reason:
            logger.info(
                "new-pair scan remains active despite capacity warning; execution gate will recheck",
                reason=capacity_reason,
            )

        cooldown_loss_pct = float(account_cfg.get("cooldown_loss_pct") or 0.0)
        cooldown_loss_reason = await self._cooldown_loss_pause_reason(
            model_mode,
            max_loss_usdt,
            cooldown_loss_pct,
        )
        if cooldown_loss_reason:
            return cooldown_loss_reason
        loss_streak_reason = await self._recent_loss_streak_pause_reason(
            model_mode,
            max_loss_usdt,
            cooldown_loss_pct,
        )
        if loss_streak_reason:
            return loss_streak_reason
        return None

    async def _daily_profit_control_pause_reason(self, mode: str) -> str | None:
        """Pause new entries when today's PnL says the system should stop digging."""
        target_usdt = self._configured_daily_target_usdt()
        if target_usdt <= 0:
            return None

        try:
            state = await self._daily_profit_control_state(mode, target_usdt)
        except Exception as e:
            logger.warning("failed to calculate daily profit control state", error=str(e))
            return None

        today_total = float(state.get("today_total_pnl") or 0.0)
        high_water = float(state.get("today_high_water_pnl") or today_total)

        loss_pause = self._daily_cooldown_trigger_loss_usdt(mode, target_usdt)
        if today_total <= -loss_pause:
            logger.info(
                "daily loss line reached; keeping analysis active for selective recovery entries",
                today_total=today_total,
                target_usdt=target_usdt,
                loss_pause=loss_pause,
            )
            return None

        if high_water >= target_usdt:
            profit_floor = max(
                target_usdt * DAILY_TARGET_PROFIT_FLOOR_RATIO,
                high_water - max(target_usdt * DAILY_TARGET_PROFIT_LOCK_RATIO, DAILY_TARGET_LOSS_PAUSE_MIN_USDT),
            )
            if today_total <= profit_floor:
                return (
                    f"今日盈利最高到过 {high_water:.2f} USDT，当前回落到 {today_total:.2f} USDT，"
                    f"已触及目标保护线 {profit_floor:.2f} USDT；暂停新开仓，优先守住已实现利润。"
                    "已有持仓仍会继续按规则止盈、止损和平仓。"
                )

        return None

    def _daily_cooldown_trigger_loss_usdt(self, mode: str, target_usdt: float) -> float:
        selected_mode = "live" if mode == "live" else "paper"
        account_cfg = settings.get_execution_account_config(selected_mode)
        max_loss_usdt = float(account_cfg.get("max_loss_usdt") or 0.0)
        cooldown_loss_pct = float(account_cfg.get("cooldown_loss_pct") or 0.0)
        if max_loss_usdt > 0 and cooldown_loss_pct > 0:
            return max(max_loss_usdt * cooldown_loss_pct, DAILY_TARGET_LOSS_PAUSE_MIN_USDT)
        if target_usdt > 0 and cooldown_loss_pct > 0:
            return max(target_usdt * cooldown_loss_pct, DAILY_TARGET_LOSS_PAUSE_MIN_USDT)
        return DAILY_TARGET_LOSS_PAUSE_MIN_USDT

    async def _daily_profit_control_state(self, mode: str, target_usdt: float) -> dict[str, float]:
        selected_mode = "live" if mode == "live" else "paper"
        now_local = datetime.now(timezone(timedelta(hours=8)))
        start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        start_utc = start_local.astimezone(timezone.utc)
        realized_profit = 0.0
        realized_loss = 0.0
        trade_count = 0
        high_water = 0.0
        open_unrealized = 0.0

        async with get_session_ctx() as session:
            rows = await TradeRepository(session).get_position_records(
                execution_mode=selected_mode,
                model_name=ENSEMBLE_TRADER_NAME,
                limit=5000,
            )
            closed_today = []
            for pos in rows:
                if pos.is_open:
                    open_unrealized += float(pos.unrealized_pnl or 0.0)
                    continue
                closed_at = pos.closed_at
                if not closed_at:
                    continue
                if closed_at.tzinfo is None:
                    closed_at = closed_at.replace(tzinfo=timezone.utc)
                if closed_at < start_utc:
                    continue
                closed_today.append(pos)

            closed_today.sort(key=lambda p: p.closed_at or datetime.min)
            running = 0.0
            for pos in closed_today:
                pnl = float(pos.realized_pnl or 0.0)
                running += pnl
                high_water = max(high_water, running)
                trade_count += 1
                if pnl >= 0:
                    realized_profit += pnl
                else:
                    realized_loss += abs(pnl)

        realized_pnl = realized_profit - realized_loss
        today_total = realized_pnl + open_unrealized
        high_water = max(high_water, today_total)
        return {
            "target_usdt": target_usdt,
            "today_total_pnl": today_total,
            "today_realized_pnl": realized_pnl,
            "today_realized_profit": realized_profit,
            "today_realized_loss": realized_loss,
            "today_trade_count": float(trade_count),
            "today_high_water_pnl": high_water,
            "open_unrealized_pnl": open_unrealized,
        }

    def _configured_daily_target_usdt(self) -> float:
        target_usdt_setting = max(float(settings.daily_profit_target_usdt or 0.0), 0.0)
        target_cny = max(float(settings.daily_profit_target_cny or 0.0), 0.0)
        cny_per_usdt = max(float(settings.cny_per_usdt_assumption or 7.2), 0.0001)
        return (
            target_usdt_setting
            if target_usdt_setting > 0
            else target_cny / cny_per_usdt if target_cny > 0 else 0.0
        )

    async def _cooldown_loss_pause_reason(
        self,
        mode: str,
        max_loss_usdt: float,
        cooldown_loss_pct: float,
    ) -> str | None:
        """Pause new entries when the configured cooldown ratio is reached."""
        if max_loss_usdt <= 0 or cooldown_loss_pct <= 0:
            return None

        trigger_loss = max_loss_usdt * cooldown_loss_pct
        if trigger_loss <= 0:
            return None

        try:
            selected_mode = "live" if mode == "live" else "paper"
            okx_snapshot = await self._get_okx_balance_snapshot_for_mode(selected_mode)
            account_equity = float(
                (okx_snapshot or {}).get("allocatable")
                or (okx_snapshot or {}).get("equity")
                or (okx_snapshot or {}).get("total")
                or 0.0
            )
            now_local = datetime.now(timezone(timedelta(hours=8)))
            start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
            start_utc = start_local.astimezone(timezone.utc)
            realized_pnl = 0.0
            today_realized = 0.0
            open_unrealized = 0.0
            day_risk_pnl = 0.0
            has_equity_baseline = False
            async with get_session_ctx() as session:
                rows = await TradeRepository(session).get_position_records(
                    execution_mode=selected_mode,
                    model_name=ENSEMBLE_TRADER_NAME,
                    limit=5000,
                )
                for pos in rows:
                    if pos.is_open:
                        open_unrealized += float(pos.unrealized_pnl or 0.0)
                    else:
                        pnl = float(pos.realized_pnl or 0.0)
                        realized_pnl += pnl
                        if pos.closed_at:
                            closed_at = pos.closed_at
                            if closed_at.tzinfo is None:
                                closed_at = closed_at.replace(tzinfo=timezone.utc)
                            if closed_at >= start_utc:
                                today_realized += pnl
                equity_baseline = await apply_daily_equity_baseline(
                    session,
                    mode=selected_mode,
                    model_name=ENSEMBLE_TRADER_NAME,
                    allocated=account_equity,
                    positions=rows,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=open_unrealized,
                    total_pnl=realized_pnl + open_unrealized,
                )
                day_risk_pnl = float(equity_baseline.get("today_equity_pnl") or 0.0)
                has_equity_baseline = True
        except Exception as e:
            logger.warning("failed to check cooldown loss pause", error=str(e))
            return None

        if not has_equity_baseline:
            day_risk_pnl = today_realized + min(open_unrealized, 0.0)
        if day_risk_pnl > -trigger_loss:
            return None

        return (
            "New-pair analysis paused by realized daily loss guard. "
            f"Today equity PnL {day_risk_pnl:.2f} USDT, "
            f"realized {today_realized:.2f} USDT, open floating {open_unrealized:.2f} USDT. "
            f"Trigger is {cooldown_loss_pct * 100:.0f}% of max daily loss "
            f"{max_loss_usdt:.2f} USDT = {trigger_loss:.2f} USDT. "
            "Existing positions will continue to be reviewed."
        )

        return (
            "New-pair analysis paused by realized daily loss guard. "
            f"Realized {today_realized:.2f} USDT, protected open floating "
            f"{min(open_unrealized, 0.0):.2f} USDT, total {day_risk_pnl:.2f} USDT. "
            f"Trigger is {cooldown_loss_pct * 100:.0f}% of max daily loss "
            f"{max_loss_usdt:.2f} USDT = {trigger_loss:.2f} USDT. "
            "Existing positions will continue to be reviewed."
        )

    async def _recent_loss_streak_pause_reason(
        self,
        mode: str,
        max_loss_usdt: float,
        cooldown_loss_pct: float,
    ) -> str | None:
        if max_loss_usdt <= 0 or cooldown_loss_pct <= 0:
            return None

        trigger_loss = max_loss_usdt * cooldown_loss_pct
        if trigger_loss <= 0:
            return None

        try:
            async with get_session_ctx() as session:
                rows = await TradeRepository(session).get_position_records(
                    execution_mode="live" if mode == "live" else "paper",
                    model_name=ENSEMBLE_TRADER_NAME,
                    is_open=False,
                    limit=RECENT_LOSS_LOOKBACK_COUNT,
                )
        except Exception as e:
            logger.warning("failed to check recent loss streak", error=str(e))
            return None

        recent = [p for p in rows if p.closed_at is not None]
        if not recent:
            return None

        latest_closed = recent[0].closed_at
        if latest_closed and latest_closed.tzinfo is None:
            latest_closed = latest_closed.replace(tzinfo=timezone.utc)
        minutes_since_latest = (
            (datetime.now(timezone.utc) - latest_closed).total_seconds() / 60.0
            if latest_closed
            else 9999.0
        )
        if minutes_since_latest >= LOSS_STREAK_PAUSE_MINUTES:
            return None

        streak = 0
        streak_loss = 0.0
        for pos in recent:
            realized_pnl = float(pos.realized_pnl or 0.0)
            if realized_pnl < 0:
                streak += 1
                streak_loss += abs(realized_pnl)
            else:
                break

        if streak <= 0 or streak_loss < trigger_loss:
            return None

        remaining = max(LOSS_STREAK_PAUSE_MINUTES - minutes_since_latest, 0.0)
        return (
            "New-pair analysis paused by consecutive realized losses. "
            f"Pause remains about {remaining:.0f} minutes. "
            f"Recent losing streak: {streak} trades, total loss {streak_loss:.2f} USDT. "
            f"Trigger is {cooldown_loss_pct * 100:.0f}% of max daily loss "
            f"{max_loss_usdt:.2f} USDT = {trigger_loss:.2f} USDT. "
            "Existing positions will continue to be monitored."
        )

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
        """Rebuild model_name 鈫?execution_mode mapping from current settings."""
        self._model_execution_modes = {ENSEMBLE_TRADER_NAME: mode_manager.mode.value}
        for cfg in settings.ai_models:
            self._model_execution_modes[cfg.get("name", "")] = cfg.get("execution_mode", "paper")
        # Also check legacy model
        if not settings.ai_models and settings.ai_api_key:
            self._model_execution_modes["llm_agent"] = "paper"

    def _has_live_models(self) -> bool:
        return mode_manager.mode.value == "live" or any(
            cfg.get("execution_mode", "paper") == "live"
            for cfg in settings.ai_models
        )

    async def _get_account_balance(self, model_name: str) -> float:
        model_mode = self._get_model_execution_mode(model_name)
        return await self._get_account_equity_for_risk(model_mode)

    async def _get_account_equity_for_risk(self, mode: str) -> float:
        """Account equity used as the denominator for portfolio risk ratios."""
        selected_mode = "live" if mode == "live" else "paper"
        snapshot = await self._get_okx_balance_snapshot_for_mode(selected_mode)
        if snapshot:
            balance = float(
                snapshot.get("equity")
                or snapshot.get("cash")
                or snapshot.get("total")
                or snapshot.get("allocatable")
                or snapshot.get("free")
                or 0.0
            )
            if balance > 0:
                return balance
        allocation_state = await self._execution_allocation_state(selected_mode)
        allocated = float(allocation_state.get("allocated") or 0.0)
        if allocated > 0:
            return allocated
        return await self._allocated_order_balance(selected_mode)

    async def _get_okx_executor_for_mode(self, mode: str) -> OKXExecutor:
        """Return the OKX executor for paper/demo or live/real mode."""
        selected_mode = "live" if mode == "live" else "paper"
        if selected_mode == "paper":
            if self._okx_paper is None:
                self._okx_paper = OKXExecutor(mode="paper")
                await self._okx_paper.initialize()
            return self._okx_paper

        if self._okx_live is None:
            self._okx_live = OKXExecutor(mode="live")
            await self._okx_live.initialize()
        return self._okx_live

    async def _execution_allocation_state(self, mode: str) -> dict[str, float]:
        """Calculate execution PnL and OKX-backed budget state from persisted positions."""
        selected_mode = "live" if mode == "live" else "paper"
        cfg = settings.get_execution_account_config(selected_mode)
        okx_snapshot = await self._get_okx_balance_snapshot_for_mode(selected_mode)
        okx_available = (
            float(okx_snapshot.get("free") or 0.0)
            if okx_snapshot
            else 0.0
        )
        okx_equity = (
            float(
                okx_snapshot.get("allocatable")
                or okx_snapshot.get("equity")
                or okx_snapshot.get("total")
                or okx_available
                or 0.0
            )
            if okx_snapshot
            else 0.0
        )
        allocated = okx_equity
        realized_profit = 0.0
        realized_loss = 0.0
        today_realized_profit = 0.0
        today_realized_loss = 0.0
        unrealized_pnl = 0.0
        used_margin = 0.0
        exchange_keys = None
        position_rows = []
        now_local = datetime.now(timezone(timedelta(hours=8)))
        start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        start_utc = start_local.astimezone(timezone.utc)

        executor = self._okx_live if selected_mode == "live" else self._okx_paper
        if executor is not None:
            try:
                okx_positions = await asyncio.wait_for(
                    executor.get_positions(),
                    timeout=8.0,
                )
                exchange_keys = {
                    (
                        self._normalize_position_symbol(p.get("symbol")),
                        str(p.get("side") or "").lower(),
                    )
                    for p in (okx_positions or [])
                    if self._exchange_position_is_open(p)
                }
            except Exception:
                exchange_keys = None

        try:
            async with get_session_ctx() as session:
                rows = await TradeRepository(session).get_position_records(
                    execution_mode=selected_mode,
                    model_name=ENSEMBLE_TRADER_NAME,
                    limit=5000,
                )
                position_rows = list(rows)
                for pos in position_rows:
                    if pos.is_open:
                        if exchange_keys and (
                            self._normalize_position_symbol(pos.symbol),
                            str(pos.side or "").lower(),
                        ) not in exchange_keys:
                            continue
                        leverage = max(float(pos.leverage or 1.0), 1.0)
                        used_margin += (
                            float(pos.quantity or 0.0)
                            * float(pos.entry_price or 0.0)
                        ) / leverage
                        unrealized_pnl += float(pos.unrealized_pnl or 0.0)
                    else:
                        pnl = float(pos.realized_pnl or 0.0)
                        if pnl >= 0:
                            realized_profit += pnl
                        else:
                            realized_loss += abs(pnl)
                        if pos.closed_at:
                            closed_at = pos.closed_at
                            if closed_at.tzinfo is None:
                                closed_at = closed_at.replace(tzinfo=timezone.utc)
                            if closed_at >= start_utc:
                                if pnl >= 0:
                                    today_realized_profit += pnl
                                else:
                                    today_realized_loss += abs(pnl)
        except Exception as e:
            logger.warning("failed to calculate allocation state", mode=selected_mode, error=str(e))

        realized_pnl = realized_profit - realized_loss
        today_realized_pnl = today_realized_profit - today_realized_loss
        total_pnl = realized_pnl + unrealized_pnl
        today_total_pnl = today_realized_pnl + unrealized_pnl
        today_risk_pnl = today_realized_pnl + min(unrealized_pnl, 0.0)
        equity_baseline = {}
        try:
            async with get_session_ctx() as session:
                equity_baseline = await apply_daily_equity_baseline(
                    session,
                    mode=selected_mode,
                    model_name=ENSEMBLE_TRADER_NAME,
                    allocated=allocated,
                    positions=position_rows,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=unrealized_pnl,
                    total_pnl=total_pnl,
                )
            today_total_pnl = float(equity_baseline.get("today_equity_pnl") or 0.0)
            today_risk_pnl = today_total_pnl
        except Exception as e:
            logger.warning("failed to calculate daily equity baseline", mode=selected_mode, error=str(e))
        return {
            "allocated_balance": allocated,
            "used_margin": used_margin,
            "realized_profit": realized_profit,
            "realized_loss": realized_loss,
            "realized_pnl": realized_pnl,
            "today_realized_profit": today_realized_profit,
            "today_realized_loss": today_realized_loss,
            "today_realized_pnl": today_realized_pnl,
            "today_closed_realized_profit": today_realized_profit,
            "today_closed_realized_loss": today_realized_loss,
            "today_closed_realized_pnl": today_realized_pnl,
            "today_equity_pnl": today_total_pnl,
            "today_equity_baseline": equity_baseline.get("today_equity_baseline"),
            "today_equity_baseline_total_pnl": equity_baseline.get("today_equity_baseline_total_pnl"),
            "today_equity_baseline_at": equity_baseline.get("today_equity_baseline_at"),
            "today_equity_baseline_source": equity_baseline.get("today_equity_baseline_source"),
            "today_snapshot_date": equity_baseline.get("today_snapshot_date"),
            "today_total_pnl": today_total_pnl,
            "today_risk_pnl": today_risk_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "remaining_allocation": okx_available,
        }

    async def _allocated_order_balance(
        self,
        mode: str,
        decision: DecisionOutput | None = None,
    ) -> float:
        """Balance visible to the order sizer, sourced from OKX available balance."""
        selected_mode = "live" if mode == "live" else "paper"
        okx_available = await self._get_okx_available_balance_for_mode(selected_mode)
        if okx_available is None:
            return 0.0
        return max(float(okx_available or 0.0), 0.0)

    async def _expert_memory_context(self, symbol: str) -> dict[str, Any]:
        """Fetch compact, relevant long-term memories for each expert."""
        if not settings.expert_memory_enabled:
            return {"expert_memories": {}, "expert_memories_flat": [], "dynamic_expert_weights": {}}

        limit = max(1, int(settings.expert_memory_per_prompt or 4))
        by_expert: dict[str, list[dict[str, Any]]] = {}
        flat: list[dict[str, Any]] = []
        used_ids: list[int] = []
        try:
            async with get_session_ctx() as session:
                repo = MemoryRepository(session)
                for slot in FIXED_AI_MODEL_SLOTS:
                    expert_name = slot.get("name", "")
                    if not expert_name:
                        continue
                    rows = await repo.get_relevant_memories(
                        expert_name=expert_name,
                        symbol=symbol,
                        limit=limit,
                    )
                    serialized = [self._serialize_memory(row) for row in rows]
                    if serialized:
                        by_expert[expert_name] = serialized
                        flat.extend(serialized)
                        used_ids.extend([row.id for row in rows if row.id])
                await repo.mark_memories_used(used_ids)
        except Exception as e:
            logger.warning("failed to fetch expert memories", symbol=symbol, error=str(e))
            return {"expert_memories": {}, "expert_memories_flat": [], "dynamic_expert_weights": {}}

        dynamic_weights = self._dynamic_expert_weights_from_memories(by_expert)
        realized_weights = await self._realized_expert_weight_adjustments()
        for expert_name, realized in realized_weights.items():
            if expert_name not in dynamic_weights:
                dynamic_weights[expert_name] = realized
                continue
            current = dynamic_weights[expert_name]
            base_weight = self._safe_float(current.get("base_weight"), realized.get("base_weight", 1.0))
            memory_multiplier = self._safe_float(current.get("multiplier"), 1.0)
            realized_multiplier = self._safe_float(realized.get("multiplier"), 1.0)
            combined = min(max(memory_multiplier * realized_multiplier, 0.65), 1.30)
            current.update({
                "multiplier": round(combined, 4),
                "effective_weight": round(base_weight * combined, 4),
                "realized_pnl": realized.get("realized_pnl", 0.0),
                "realized_count": realized.get("realized_count", 0),
                "reason": (
                    f"{current.get('reason') or ''} Realized PnL calibration: "
                    f"{realized.get('reason') or 'none'}"
                ),
            })

        return {
            "expert_memories": by_expert,
            "expert_memories_flat": flat,
            "dynamic_expert_weights": dynamic_weights,
        }

    async def _realized_expert_weight_adjustments(self) -> dict[str, dict[str, Any]]:
        """Daily Beijing-time expert weight calibration from realized PnL."""
        now = datetime.now(timezone.utc)
        expires_at = self._realized_expert_weight_cache.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at > now:
            return self._realized_expert_weight_cache.get("weights") or {}

        slot_weights = {
            slot.get("name", ""): float(slot.get("weight", 1.0) or 1.0)
            for slot in FIXED_AI_MODEL_SLOTS
            if slot.get("name")
        }
        stats = {
            name: {"pnl": 0.0, "profit": 0.0, "loss": 0.0, "count": 0, "wins": 0, "losses": 0}
            for name in slot_weights
        }
        start_utc = datetime.now(timezone(timedelta(hours=8))).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).astimezone(timezone.utc)

        try:
            async with get_session_ctx() as session:
                positions_result = await session.execute(
                    select(Position)
                    .where(
                        Position.model_name == ENSEMBLE_TRADER_NAME,
                        Position.is_open.is_(False),
                        Position.closed_at.is_not(None),
                        Position.closed_at >= start_utc,
                    )
                    .order_by(Position.closed_at.desc())
                    .limit(800)
                )
                positions = list(positions_result.scalars().all())
                if not positions:
                    self._realized_expert_weight_cache = {
                        "expires_at": now + timedelta(minutes=15),
                        "weights": {},
                    }
                    return {}

                symbols = {p.symbol for p in positions if p.symbol}
                orders_result = await session.execute(
                    select(Order)
                    .where(
                        Order.model_name == ENSEMBLE_TRADER_NAME,
                        Order.status == "filled",
                        Order.decision_id.is_not(None),
                        Order.symbol.in_(symbols) if symbols else Order.id == -1,
                    )
                    .order_by(Order.filled_at.desc(), Order.created_at.desc())
                    .limit(2400)
                )
                orders = list(orders_result.scalars().all())
                decision_ids = [o.decision_id for o in orders if o.decision_id]
                decisions = {}
                if decision_ids:
                    decisions_result = await session.execute(
                        select(AIDecision).where(AIDecision.id.in_(decision_ids))
                    )
                    decisions = {d.id: d for d in decisions_result.scalars().all()}
        except Exception as exc:
            logger.warning("failed to calculate realized expert weights", error=str(exc))
            return {}

        def aware(value):
            if value and value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        for pos in positions:
            pos_created = aware(pos.created_at)
            pos_side = str(pos.side or "").lower()
            candidates = []
            for order in orders:
                if order.symbol != pos.symbol or order.decision_id not in decisions:
                    continue
                decision = decisions[order.decision_id]
                decision_side = "short" if str(decision.action or "").lower() == "short" else "long" if str(decision.action or "").lower() == "long" else ""
                if decision_side != pos_side:
                    continue
                order_time = aware(order.filled_at or order.created_at)
                if pos_created and order_time and abs((order_time - pos_created).total_seconds()) > 180:
                    continue
                candidates.append((abs(((order_time or pos_created) - pos_created).total_seconds()) if pos_created and order_time else 0, decision))
            if not candidates:
                continue
            _, decision = sorted(candidates, key=lambda item: item[0])[0]
            raw = decision.raw_llm_response if isinstance(decision.raw_llm_response, dict) else {}
            opinions = raw.get("opinions") if isinstance(raw.get("opinions"), list) else []
            pnl = float(pos.realized_pnl or 0.0)
            for opinion in opinions:
                if not isinstance(opinion, dict):
                    continue
                name = str(opinion.get("model_name") or "")
                if name not in stats:
                    continue
                action = str(opinion.get("action") or "").lower()
                if action != pos_side:
                    continue
                bucket = stats[name]
                bucket["pnl"] += pnl
                bucket["count"] += 1
                if pnl >= 0:
                    bucket["wins"] += 1
                    bucket["profit"] += pnl
                else:
                    bucket["losses"] += 1
                    bucket["loss"] += abs(pnl)

        result: dict[str, dict[str, Any]] = {}
        for name, bucket in stats.items():
            count = int(bucket["count"])
            if count < 3:
                continue
            pnl = float(bucket["pnl"])
            avg_pnl = pnl / count
            win_rate = bucket["wins"] / count
            profit_factor = (
                float(bucket["profit"]) / float(bucket["loss"])
                if float(bucket["loss"]) > 0
                else (3.0 if float(bucket["profit"]) > 0 else 0.0)
            )
            expectancy_component = max(min(avg_pnl / 8.0, 0.24), -0.30)
            factor_component = max(min((profit_factor - 1.0) * 0.12, 0.14), -0.18)
            win_component = max(min((win_rate - 0.5) * 0.06, 0.03), -0.03)
            raw_multiplier = 1.0 + expectancy_component + factor_component + win_component
            multiplier = min(max(raw_multiplier, 0.65), 1.30)
            result[name] = {
                "base_weight": slot_weights.get(name, 1.0),
                "multiplier": round(multiplier, 4),
                "effective_weight": round(slot_weights.get(name, 1.0) * multiplier, 4),
                "realized_count": count,
                "realized_pnl": round(pnl, 6),
                "win_rate": round(win_rate, 4),
                "avg_pnl": round(avg_pnl, 6),
                "profit_factor": round(profit_factor, 4),
                "reason": f"北京时间今日同向参与 {count} 笔，真实盈亏 {pnl:.2f}U，胜率 {win_rate:.0%}，权重调到 {multiplier:.2f} 倍。",
            }
        self._realized_expert_weight_cache = {
            "expires_at": now + timedelta(minutes=15),
            "weights": result,
        }
        return result

    def _serialize_memory(self, memory) -> dict[str, Any]:
        return {
            "id": memory.id,
            "expert_name": memory.expert_name,
            "expert_label": memory.expert_label,
            "symbol": memory.symbol,
            "side": memory.side,
            "memory_type": memory.memory_type,
            "market_pattern": memory.market_pattern,
            "lesson": memory.lesson,
            "recommended_action": memory.recommended_action,
            "confidence_adjustment": float(memory.confidence_adjustment or 0.0),
            "position_size_multiplier": float(memory.position_size_multiplier or 1.0),
            "evidence_count": int(memory.evidence_count or 0),
            "success_count": int(getattr(memory, "success_count", 0) or 0),
            "failure_count": int(getattr(memory, "failure_count", 0) or 0),
            "confidence_score": float(memory.confidence_score or 0.0),
        }

    def _dynamic_expert_weights_from_memories(
        self,
        by_expert: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        """Conservative long-term-memory based expert weight adjustment."""
        result: dict[str, dict[str, Any]] = {}
        slot_weights = {
            slot.get("name", ""): float(slot.get("weight", 1.0) or 1.0)
            for slot in FIXED_AI_MODEL_SLOTS
        }
        for expert_name, base_weight in slot_weights.items():
            memories = [m for m in by_expert.get(expert_name, []) if isinstance(m, dict)]
            if not memories:
                result[expert_name] = {
                    "base_weight": base_weight,
                    "multiplier": 1.0,
                    "effective_weight": base_weight,
                    "memory_count": 0,
                    "evidence_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "reason": "暂无足够历史样本，使用基础权重。",
                }
                continue

            evidence = sum(max(int(m.get("evidence_count", 1) or 1), 1) for m in memories)
            success = sum(max(int(m.get("success_count", 0) or 0), 0) for m in memories)
            failure = sum(max(int(m.get("failure_count", 0) or 0), 0) for m in memories)
            weighted_adjustment = 0.0
            weight_sum = 0.0
            for memory in memories:
                confidence_score = min(max(float(memory.get("confidence_score", 0.5) or 0.5), 0.1), 1.0)
                memory_evidence = max(int(memory.get("evidence_count", 1) or 1), 1)
                weight = confidence_score * min(memory_evidence, 6)
                weighted_adjustment += float(memory.get("confidence_adjustment", 0.0) or 0.0) * weight
                weight_sum += weight

            average_adjustment = weighted_adjustment / weight_sum if weight_sum > 0 else 0.0
            performance_edge = ((success + 1) / (success + failure + 2)) - 0.5
            raw_multiplier = 1.0 + average_adjustment * 0.70 + performance_edge * 0.35
            if evidence < 2 and success + failure < 2:
                raw_multiplier = 1.0

            multiplier = min(max(raw_multiplier, 0.70), 1.15)
            if failure >= success + 2:
                multiplier = min(multiplier, 0.90)
            elif success >= failure + 3:
                multiplier = max(multiplier, 1.05)

            if multiplier > 1.03:
                reason = f"近期记忆中成功样本较多或正向教训更稳定，权重提高到 {multiplier:.2f} 倍。"
            elif multiplier < 0.97:
                reason = f"近期记忆提示该专家相关场景亏损偏多，权重降到 {multiplier:.2f} 倍。"
            else:
                reason = "历史样本未显示明显优劣，保持基础权重。"

            result[expert_name] = {
                "base_weight": base_weight,
                "multiplier": round(multiplier, 4),
                "effective_weight": round(base_weight * multiplier, 4),
                "memory_count": len(memories),
                "evidence_count": evidence,
                "success_count": success,
                "failure_count": failure,
                "reason": reason,
            }
        return result

    async def _daily_target_context(self) -> dict[str, Any]:
        """Return today's target gap without encouraging unsafe overtrading."""
        target_usdt_setting = max(float(settings.daily_profit_target_usdt or 0.0), 0.0)
        target_cny = max(float(settings.daily_profit_target_cny or 0.0), 0.0)
        cny_per_usdt = max(float(settings.cny_per_usdt_assumption or 7.2), 0.0001)
        target_usdt = (
            target_usdt_setting
            if target_usdt_setting > 0
            else target_cny / cny_per_usdt if target_cny > 0 else 0.0
        )
        target_cny_equivalent = target_usdt * cny_per_usdt if target_usdt > 0 else target_cny
        mode = self._get_model_execution_mode(ENSEMBLE_TRADER_NAME)
        today_pnl = 0.0
        control_state: dict[str, Any] = {}
        try:
            now_local = datetime.now(timezone(timedelta(hours=8)))
            start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
            start_utc = start_local.astimezone(timezone.utc)
            async with get_session_ctx() as session:
                result = await session.execute(
                    select(func.coalesce(func.sum(Position.realized_pnl), 0.0))
                    .where(
                        Position.model_name == ENSEMBLE_TRADER_NAME,
                        Position.execution_mode == ("live" if mode == "live" else "paper"),
                        Position.is_open == False,
                        Position.closed_at >= start_utc,
                    )
                )
                today_pnl = float(result.scalar() or 0.0)
        except Exception as e:
            logger.warning("failed to calculate daily target context", error=str(e))

        if target_usdt > 0:
            try:
                control_state = await self._daily_profit_control_state(mode, target_usdt)
            except Exception as e:
                logger.warning("failed to enrich daily target context", error=str(e))

        today_total_pnl = float(control_state.get("today_total_pnl", today_pnl) or 0.0)
        today_high_water = float(control_state.get("today_high_water_pnl", today_total_pnl) or 0.0)
        daily_phase = "normal"
        if target_usdt > 0:
            loss_line = -self._daily_cooldown_trigger_loss_usdt(mode, target_usdt)
            if today_total_pnl <= loss_line:
                daily_phase = "loss_control"
            elif today_total_pnl < 0:
                daily_phase = "recovery"
            elif today_high_water >= target_usdt:
                daily_phase = "profit_protected_expand"
            elif today_total_pnl >= target_usdt * 0.70:
                daily_phase = "near_target"

        return {
            "target_currency": "USDT" if target_usdt_setting > 0 else "CNY",
            "target_cny": target_cny_equivalent,
            "target_usdt": target_usdt,
            "today_realized_pnl": today_pnl,
            "today_total_pnl": today_total_pnl,
            "today_high_water_pnl": today_high_water,
            "today_realized_profit": control_state.get("today_realized_profit"),
            "today_realized_loss": control_state.get("today_realized_loss"),
            "today_trade_count": control_state.get("today_trade_count"),
            "loss_pause_usdt": self._daily_cooldown_trigger_loss_usdt(mode, target_usdt) if target_usdt > 0 else 0.0,
            "phase": daily_phase,
            "gap_usdt": max(target_usdt - today_total_pnl, 0.0),
            "note": "每日目标只用于筛选更高质量机会，不能作为追单、放大杠杆或放松风控的理由。",
        }

    async def _get_okx_available_balance_for_mode(self, mode: str) -> float | None:
        """Return the actual OKX free USDT balance used to cap new entries."""
        snapshot = await self._get_okx_balance_snapshot_for_mode(mode)
        if not snapshot:
            return None
        return self._okx_tradeable_balance_from_snapshot(snapshot)

    def _okx_allocatable_balance_from_snapshot(self, snapshot: dict[str, Any] | None) -> float:
        if not isinstance(snapshot, dict):
            return 0.0
        return max(
            self._safe_float(snapshot.get("allocatable"), 0.0),
            self._safe_float(snapshot.get("equity"), 0.0),
            self._safe_float(snapshot.get("cash"), 0.0),
            self._safe_float(snapshot.get("total"), 0.0),
            self._safe_float(snapshot.get("free"), 0.0),
        )

    def _okx_tradeable_balance_from_snapshot(self, snapshot: dict[str, Any] | None) -> float:
        if not isinstance(snapshot, dict):
            return 0.0
        free = self._safe_float(snapshot.get("free"), 0.0)
        if free > 0:
            return free
        # OKX demo/swap accounts can report free=0 while equity/cash is usable.
        # Do not freeze new-pair analysis when account equity was fetched.
        return self._okx_allocatable_balance_from_snapshot(snapshot)

    async def _get_okx_balance_snapshot_for_mode(self, mode: str) -> dict[str, Any] | None:
        """Return OKX USDT balance fields for allocation and order sizing."""
        selected_mode = "live" if mode == "live" else "paper"

        def remember_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
            self._okx_balance_snapshot_cache[selected_mode] = {
                "snapshot": dict(snapshot),
                "fetched_at": datetime.now(timezone.utc),
            }
            return snapshot

        def cached_snapshot(reason: str) -> dict[str, Any] | None:
            cached = self._okx_balance_snapshot_cache.get(selected_mode)
            if not isinstance(cached, dict):
                return None
            fetched_at = cached.get("fetched_at")
            if not isinstance(fetched_at, datetime):
                return None
            age = (datetime.now(timezone.utc) - fetched_at).total_seconds()
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
            fallback = OKXExecutor(mode=selected_mode)
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
                        error=str(snapshot.get("error")),
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
            except asyncio.TimeoutError:
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
                    error=str(exc),
                )
                return None
            finally:
                try:
                    await fallback.shutdown()
                except Exception:
                    pass

        executor = self._okx_live if selected_mode == "live" else self._okx_paper
        if not executor:
            try:
                executor = await self._get_okx_executor_for_mode(selected_mode)
            except Exception as exc:
                reason = f"executor unavailable: {exc}"
                return cached_snapshot(reason) or await fresh_executor_snapshot(reason)
        try:
            snapshot = await asyncio.wait_for(
                executor.get_balance_snapshot("USDT"),
                timeout=8.0,
            )
            if snapshot.get("error"):
                reason = str(snapshot.get("error"))
                return cached_snapshot(reason) or await fresh_executor_snapshot(reason)
            return remember_snapshot(snapshot)
        except asyncio.TimeoutError:
            logger.warning("timed out fetching OKX balance snapshot", mode=selected_mode)
            reason = "OKX balance snapshot request timed out"
            return cached_snapshot(reason) or await fresh_executor_snapshot(reason)
        except Exception as exc:
            logger.warning("failed to fetch OKX balance snapshot", mode=selected_mode, error=str(exc))
            reason = str(exc)
            return cached_snapshot(reason) or await fresh_executor_snapshot(reason)

    async def _persist_paper_execution_balance(
        self,
        model_name: str,
        decision: DecisionOutput,
        result: ExecutionResult,
    ) -> None:
        """Persist the balance delta already applied by PaperExecutor in memory."""
        if result.status != OrderStatus.FILLED:
            return

        notional = float(result.quantity or 0.0) * float(result.price or 0.0)
        fee = float(result.fee or 0.0)
        if decision.action in (Action.LONG, Action.SHORT):
            margin_used = self._position_margin(notional, decision.suggested_leverage)
            await self._persist_paper_balance_delta(model_name, -(margin_used + fee), 0.0)
            return

        if decision.action in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
            released_margin = await self._db_released_margin_for_close(model_name, decision, result)
            realized_pnl = float(result.pnl or 0.0)
            await self._persist_paper_balance_delta(
                model_name,
                released_margin + realized_pnl,
                realized_pnl,
            )

    async def _sync_paper_after_okx(
        self, model_name: str, decision: DecisionOutput, result: ExecutionResult,
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
            margin_used = self._position_margin(order_value, decision.suggested_leverage)
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
                "stop_loss": price * (1 - decision.stop_loss_pct) if side == "long" else price * (1 + decision.stop_loss_pct),
                "take_profit": price * (1 + decision.take_profit_pct) if side == "long" else price * (1 - decision.take_profit_pct),
                "is_open": True,
                "opened_at": datetime.now(timezone.utc),
                "unrealized_pnl": 0.0,
            }
            pe._positions.setdefault(model_name, []).append(position)
            await self._persist_paper_balance_delta(model_name, balance_delta, 0.0)

        elif decision.action in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
            target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
            positions = pe._positions.get(model_name, [])
            to_close = [
                p for p in positions
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
                pos["closed_at"] = datetime.now(timezone.utc)
                total_pnl += pnl
                released_margin += self._position_margin(
                    pos["quantity"] * pos["entry_price"],
                    pos.get("leverage", 1.0),
                )

            if not to_close:
                released_margin = await self._db_released_margin_for_close(model_name, decision, result)
            pe._balances[model_name] = (
                pe._balances.get(model_name, settings.get_initial_balance(model_name))
                + released_margin + total_pnl - total_fee
            )
            pe._positions[model_name] = [p for p in positions if p["is_open"]]
            # Attach PnL to result for downstream logging
            result.pnl = total_pnl - total_fee
            await self._persist_paper_balance_delta(
                model_name,
                released_margin + total_pnl - total_fee,
                total_pnl - total_fee,
            )

    def _position_margin(self, notional_value: float, leverage: float | None) -> float:
        lev = max(float(leverage or 1.0), 1.0)
        return float(notional_value or 0.0) / lev

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
                    released_margin += self._position_margin(
                        close_qty * float(pos.entry_price or 0.0),
                        pos.leverage,
                    )
                    remaining_qty -= close_qty
        except Exception as e:
            logger.error("failed to calculate released paper margin", error=str(e))
        return released_margin

    async def _persist_paper_balance_delta(
        self,
        model_name: str,
        balance_delta: float,
        realized_pnl_delta: float = 0.0,
    ) -> None:
        if abs(balance_delta) < 1e-12 and abs(realized_pnl_delta) < 1e-12:
            return
        try:
            async with get_session_ctx() as session:
                repo = AccountRepository(session)
                await repo.update_balance(model_name, balance_delta, realized_pnl_delta)
        except Exception as e:
            logger.error("failed to persist paper balance delta", error=str(e))

    async def _on_mode_changed(self, manager) -> None:
        """Reinitialize all LLM agent instances when trading mode changes."""
        for model in self.models.get_all():
            if hasattr(model, "reinitialize"):
                try:
                    await model.reinitialize()
                    logger.info("model reinitialized for new mode", name=model.name, mode=manager.mode.value)
                except Exception as e:
                    logger.error("failed to reinit model on mode change", name=model.name, error=str(e))

    def _decision_side(self, decision: DecisionOutput) -> str:
        if decision.action == Action.LONG:
            return "buy"
        if decision.action == Action.SHORT:
            return "sell"
        if decision.action == Action.CLOSE_LONG:
            return "sell"
        if decision.action == Action.CLOSE_SHORT:
            return "buy"
        return "hold"

    def _action_label_text(self, action: Action | str | None) -> str:
        value = action.value if isinstance(action, Action) else str(action or "")
        return {
            "long": "做多",
            "short": "做空",
            "close_long": "平多",
            "close_short": "平空",
            "hold": "观望",
        }.get(value, value or "未知")

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_utc_datetime(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str) and value.strip():
            text = value.strip()
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
        else:
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _decision_reference_time(self, decision: DecisionOutput) -> datetime:
        snapshot_times: list[datetime] = []
        snapshot = decision.feature_snapshot or {}
        for key in ("timestamp", "feature_timestamp", "market_timestamp"):
            parsed = self._parse_utc_datetime(snapshot.get(key))
            if parsed is not None:
                snapshot_times.append(parsed)

        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        timing = raw.get("timing") if isinstance(raw.get("timing"), dict) else {}
        timing_times: dict[str, datetime] = {}
        for key in ("analysis_started_at", "decision_completed_at"):
            parsed = self._parse_utc_datetime(timing.get(key))
            if parsed is not None:
                timing_times[key] = parsed

        parsed_decision_time = self._parse_utc_datetime(decision.timestamp)

        if decision.is_exit:
            for key in ("decision_completed_at", "analysis_started_at"):
                parsed = timing_times.get(key)
                if parsed is not None:
                    return parsed
            if parsed_decision_time is not None:
                return parsed_decision_time
            if snapshot_times:
                return max(snapshot_times)
            return datetime.now(timezone.utc)

        # Entry execution age should measure how long the AI decision itself
        # has been waiting. Market snapshot freshness is checked separately by
        # the pre-execution price/data recheck, otherwise delayed exchange
        # ticker timestamps can make a fresh 20s analysis look 700s old.
        for key in ("decision_completed_at", "analysis_started_at"):
            parsed = timing_times.get(key)
            if parsed is not None:
                return parsed
        if parsed_decision_time is not None:
            return parsed_decision_time
        if snapshot_times:
            return max(snapshot_times)
        return datetime.now(timezone.utc)

    def _decision_age_seconds(self, decision: DecisionOutput) -> float:
        return max(
            (datetime.now(timezone.utc) - self._decision_reference_time(decision)).total_seconds(),
            0.0,
        )

    def _stale_decision_reason(self, decision: DecisionOutput) -> str | None:
        if decision.is_hold or self._is_forced_exit_decision(decision):
            return None
        max_age = (
            ENTRY_DECISION_MAX_AGE_SECONDS
            if decision.is_entry
            else EXIT_DECISION_MAX_AGE_SECONDS
        )
        if decision.is_entry:
            raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
            opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
            score = self._safe_float(opportunity.get("score"), 0.0)
            ai_expected = self._safe_float(opportunity.get("ai_expected_return_pct"), 0.0)
            confidence = max(min(float(decision.confidence or 0.0), 1.0), 0.0)
            reward_risk = self._safe_float(opportunity.get("reward_risk_ratio"), 0.0)
            if confidence >= 0.82 and score >= 6.0 and ai_expected >= 4.0 and reward_risk >= 1.5:
                max_age = ENTRY_EXCEPTIONAL_OPPORTUNITY_MAX_AGE_SECONDS
            elif confidence >= 0.75 and score >= 3.0 and ai_expected >= 2.0 and reward_risk >= 1.2:
                max_age = ENTRY_STRONG_OPPORTUNITY_MAX_AGE_SECONDS
        if decision.is_exit:
            raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
            close_evidence = raw.get("close_evidence") if isinstance(raw.get("close_evidence"), dict) else {}
            execution_profit = (
                raw.get("execution_profit_protection")
                if isinstance(raw.get("execution_profit_protection"), dict)
                else {}
            )
            if close_evidence.get("profit_protection") or execution_profit.get("allow"):
                max_age = max(max_age, PROFIT_PROTECTION_EXIT_MAX_AGE_SECONDS)
        age = self._decision_age_seconds(decision)
        if age <= max_age:
            return None
        age_source = "AI平仓裁决完成到准备下单" if decision.is_exit else "AI开仓裁决完成到准备下单"
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["stale_decision_check"] = {
            "applied": True,
            "age_seconds": round(age, 3),
            "max_age_seconds": round(max_age, 3),
            "age_source": age_source,
            "reference_time": self._decision_reference_time(decision).isoformat(),
        }
        decision.raw_response = raw
        return (
            f"AI信号已过有效期：{age_source}已经过去 {age:.0f} 秒，"
            f"超过允许 {max_age:.0f} 秒。为避免使用旧裁决下单，本次不执行，等待下一轮重新分析。"
        )

    def _attach_decision_timing(
        self,
        decision: DecisionOutput | None,
        started_at: datetime,
        stage: str,
    ) -> None:
        if not isinstance(decision, DecisionOutput):
            return
        completed_at = datetime.now(timezone.utc)
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["timing"] = {
            **(raw.get("timing") if isinstance(raw.get("timing"), dict) else {}),
            "stage": stage,
            "analysis_started_at": started_at.isoformat(),
            "decision_completed_at": completed_at.isoformat(),
            "analysis_duration_sec": round((completed_at - started_at).total_seconds(), 3),
        }
        decision.raw_response = raw

    async def _latest_price_for_symbol(self, symbol: str) -> float:
        normalized = self._normalize_position_symbol(symbol) or symbol
        try:
            raw_ticker = await self.data_service.rest_client.fetch_ticker(normalized)
            price = self._safe_float(
                raw_ticker.get("last") or raw_ticker.get("close"),
                0.0,
            )
            if price > 0:
                return price
        except Exception as e:
            logger.warning("pre-order latest price fetch failed", symbol=symbol, error=str(e))

        candidates = [symbol, normalized]
        try:
            latest_tickers = getattr(self.data_service.ws_client, "latest_tickers", {}) or {}
            for key in candidates:
                ticker = latest_tickers.get(key)
                if not ticker:
                    continue
                price = self._safe_float(
                    ticker.get("last_price") or ticker.get("last") or ticker.get("close"),
                    0.0,
                )
                if price > 0:
                    return price
        except Exception:
            pass
        return 0.0

    async def _pre_execution_profit_exit_guard_reason(
        self,
        decision: DecisionOutput,
        open_positions: list[dict] | None,
    ) -> str | None:
        if not decision.is_exit:
            return None
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        close_evidence = raw.get("close_evidence") if isinstance(raw.get("close_evidence"), dict) else {}
        execution_profit = (
            raw.get("execution_profit_protection")
            if isinstance(raw.get("execution_profit_protection"), dict)
            else {}
        )
        profit_exit = bool(close_evidence.get("profit_protection") or execution_profit.get("allow"))
        if not profit_exit:
            return None

        target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
        target_symbol = self._normalize_position_symbol(decision.symbol)
        matches = []
        for pos in open_positions or []:
            if str(pos.get("model_name") or "") != decision.model_name:
                continue
            if str(pos.get("side") or "").lower() != target_side:
                continue
            if self._normalize_position_symbol(pos.get("symbol")) != target_symbol:
                continue
            matches.append(pos)
        if not matches:
            return None

        latest_price = await self._latest_price_for_symbol(decision.symbol)
        if latest_price <= 0:
            return "利润保护平仓前未能重新获取最新价格，系统不使用过期浮盈判断执行锁盈单。"

        estimated_unrealized = 0.0
        reported_unrealized = 0.0
        reported_available = False
        total_qty = 0.0
        for pos in matches:
            qty = abs(self._safe_float(pos.get("quantity") or pos.get("contracts") or pos.get("sz"), 0.0))
            contract_size = self._safe_float(
                pos.get("contract_size")
                or pos.get("contractSize")
                or (pos.get("info") or {}).get("ctVal"),
                1.0,
            )
            qty_for_pnl = qty * (contract_size if contract_size > 0 else 1.0)
            entry = self._safe_float(pos.get("entry_price") or pos.get("entryPrice") or pos.get("avgPx"), 0.0)
            if qty_for_pnl <= 0 or entry <= 0:
                continue
            gross = (
                (latest_price - entry) * qty_for_pnl
                if target_side == "long"
                else (entry - latest_price) * qty_for_pnl
            )
            estimated_unrealized += gross
            total_qty += qty_for_pnl
            reported = self._safe_float(
                pos.get("unrealized_pnl")
                if pos.get("unrealized_pnl") is not None
                else (
                    pos.get("unrealizedPnl")
                    if pos.get("unrealizedPnl") is not None
                    else (
                        pos.get("upl")
                        if pos.get("upl") is not None
                        else (
                            (pos.get("info") or {}).get("upl")
                            if (pos.get("info") or {}).get("upl") is not None
                            else (pos.get("info") or {}).get("unrealizedPnl")
                        )
                    )
                ),
                0.0,
            )
            if abs(reported) > 1e-12:
                reported_available = True
                reported_unrealized += reported
        if total_qty <= 0:
            return None

        # OKX swap positions may expose quantity as contract count while the DB
        # stores base quantity. Prefer the already-synced PnL when it agrees on
        # direction, and use the larger positive value to avoid blocking valid
        # lock-profit exits because of unit conversion differences.
        total_unrealized = estimated_unrealized
        if reported_available:
            if estimated_unrealized < -1e-9 and reported_unrealized > 0:
                total_unrealized = estimated_unrealized
            elif estimated_unrealized > 0 and reported_unrealized > 0:
                total_unrealized = max(estimated_unrealized, reported_unrealized)
            else:
                total_unrealized = reported_unrealized

        min_profit = max(PROFIT_PROTECTION_MIN_NET_USDT * 0.25, 0.05)
        if total_unrealized <= min_profit:
            non_profit_exit_evidence = bool(
                close_evidence.get("hard_risk")
                or close_evidence.get("raw_hard_risk")
                or close_evidence.get("position_loss")
                or close_evidence.get("strong_opposite_pressure")
                or close_evidence.get("moderate_opposite_pressure")
                or close_evidence.get("profit_retrace_protection")
                or close_evidence.get("predictive_reversal_exit")
                or close_evidence.get("predictive_full_exit")
            )
            raw["execution_profit_protection_guard"] = {
                "applied": not non_profit_exit_evidence,
                "latest_price": latest_price,
                "target_side": target_side,
                "estimated_unrealized_pnl_from_price": round(estimated_unrealized, 6),
                "reported_unrealized_pnl": round(reported_unrealized, 6) if reported_available else None,
                "estimated_unrealized_pnl": round(total_unrealized, 6),
                "min_required_profit": round(min_profit, 6),
                "non_profit_exit_evidence": non_profit_exit_evidence,
                "reason": (
                    "最新价格复核显示该仓位已不满足纯锁盈条件；但存在趋势反转/硬风险证据，"
                    "本次不再按锁盈不足拦截，继续交给平仓执行。"
                    if non_profit_exit_evidence
                    else "最新价格复核显示该仓位已不满足纯锁盈条件。"
                ),
            }
            decision.raw_response = raw
            if non_profit_exit_evidence:
                return None
            return (
                f"利润保护执行前复核未通过：按最新价格 {latest_price:g} 估算该仓位浮盈为 "
                f"{total_unrealized:.4f}U，未达到锁盈所需最小浮盈 {min_profit:.4f}U；"
                "本次不按锁定利润路径平仓。"
            )
        return None

    async def _pre_execution_price_guard_reason(self, decision: DecisionOutput) -> str | None:
        if not decision.is_entry:
            return None

        snapshot = decision.feature_snapshot or {}
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        snapshot_quality_reason = self._entry_market_data_quality_reason(snapshot, stage_label="下单前分析快照")
        if snapshot_quality_reason:
            fresh = await self._fresh_feature_vector_for_price_recheck(decision.symbol)
            fresh_snapshot = fresh.to_dict() if fresh is not None and hasattr(fresh, "to_dict") else {}
            fresh_quality_reason = (
                self._entry_market_data_quality_reason(fresh_snapshot, stage_label="下单前刷新行情")
                if fresh_snapshot
                else "下单前刷新行情失败，无法确认盘口和短周期特征。"
            )
            raw["pre_execution_data_quality_recheck"] = {
                "original_reason": snapshot_quality_reason,
                "fresh_recheck_available": bool(fresh_snapshot),
                "fresh_reason": fresh_quality_reason,
                "original_snapshot_timestamp": snapshot.get("timestamp"),
                "fresh_snapshot_timestamp": fresh_snapshot.get("timestamp") if isinstance(fresh_snapshot, dict) else None,
            }
            decision.raw_response = raw
            if fresh_snapshot and not fresh_quality_reason:
                snapshot = fresh_snapshot
                decision.feature_snapshot = fresh_snapshot
            else:
                return (
                    f"下单前行情质量复核未通过：{fresh_quality_reason or snapshot_quality_reason}"
                    "系统已即时刷新该币种行情，但数据仍不足以安全下单，本次不执行。"
                )

        snapshot_current_price = self._safe_float(snapshot.get("current_price"), 0.0)
        snapshot_close_price = self._safe_float(snapshot.get("close"), 0.0)
        snapshot_price = snapshot_current_price or snapshot_close_price
        if snapshot_price <= 0:
            return None

        latest_price = await self._latest_price_for_symbol(decision.symbol)
        if latest_price <= 0:
            return "下单前没有重新拿到最新价格，系统不使用过期行情盲目下单，本次跳过。"

        snapshot_price_source = "current_price" if snapshot_current_price > 0 else "close"
        if snapshot_current_price > 0 and snapshot_close_price > 0:
            current_gap = abs(latest_price - snapshot_current_price) / max(snapshot_current_price, 1e-12)
            close_gap = abs(latest_price - snapshot_close_price) / max(snapshot_close_price, 1e-12)
            if abs(snapshot_current_price - snapshot_close_price) / max(snapshot_close_price, 1e-12) > 0.03:
                if close_gap <= current_gap:
                    snapshot_price = snapshot_close_price
                    snapshot_price_source = "close_reconciled"
                else:
                    snapshot_price = snapshot_current_price
                    snapshot_price_source = "current_price_reconciled"

        move = (latest_price - snapshot_price) / snapshot_price
        allowed = min(max(float(settings.max_slippage_pct or 0.005), 0.003), 0.02)
        opportunity = raw.get("opportunity_score") if isinstance(raw.get("opportunity_score"), dict) else {}
        quant_probe = raw.get("quant_profit_probe") if isinstance(raw.get("quant_profit_probe"), dict) else {}
        expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = self._safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        if quant_probe.get("triggered") and expected_net > 0:
            if quant_probe.get("strong_probe") and expected_net >= 1.20 and profit_quality >= 1.20:
                allowed = max(allowed, 0.012)
            elif expected_net >= 0.35 and profit_quality >= 0.20:
                allowed = max(allowed, 0.008)
        if expected_net >= 4.0 and profit_quality >= 2.0:
            allowed = max(allowed, 0.020)
        elif expected_net >= 2.0 and profit_quality >= 1.20:
            allowed = max(allowed, 0.016)
        elif expected_net >= 1.0 and profit_quality >= 0.80:
            allowed = max(allowed, 0.012)
        raw["pre_execution_price_check"] = {
            "snapshot_price": snapshot_price,
            "snapshot_price_source": snapshot_price_source,
            "snapshot_current_price": snapshot_current_price,
            "snapshot_close_price": snapshot_close_price,
            "snapshot_timestamp": snapshot.get("timestamp"),
            "snapshot_age_seconds": round(self._decision_age_seconds(decision), 3),
            "latest_price": latest_price,
            "move_pct": round(move * 100, 4),
            "allowed_pct": round(allowed * 100, 4),
            "expected_net_return_pct": round(expected_net, 6),
            "profit_quality_ratio": round(profit_quality, 6),
        }
        decision.raw_response = raw

        def adverse_directional_move() -> bool:
            return (
                (decision.action == Action.LONG and move > allowed)
                or (decision.action == Action.SHORT and move < -allowed)
                or abs(move) > allowed * 2
            )

        if adverse_directional_move():
            fresh = await self._fresh_feature_vector_for_price_recheck(decision.symbol)
            fresh_snapshot = fresh.to_dict() if fresh is not None and hasattr(fresh, "to_dict") else {}
            fresh_quality_reason = (
                self._entry_market_data_quality_reason(fresh_snapshot, stage_label="偏移后刷新行情")
                if fresh_snapshot
                else "偏移后刷新行情失败，无法确认最新盘口和短周期特征。"
            )
            fresh_price = self._safe_float(
                fresh_snapshot.get("current_price") or fresh_snapshot.get("close"),
                0.0,
            ) if isinstance(fresh_snapshot, dict) else 0.0
            fresh_latest_gap = (
                abs(latest_price - fresh_price) / max(fresh_price, 1e-12)
                if fresh_price > 0
                else 0.0
            )
            move_abs = abs(move)
            rescue_cap = (
                ENTRY_PRICE_RECHECK_EXCEPTIONAL_MAX_MOVE_PCT
                if expected_net >= 4.0 and profit_quality >= 2.0
                else ENTRY_PRICE_RECHECK_RESCUE_MAX_MOVE_PCT
            )
            expected_buffer = (
                ENTRY_PRICE_RECHECK_EXPECTED_BUFFER_MULTIPLE
                if expected_net < 1.0
                else 1.35
                if expected_net < 2.0
                else 1.05
            )
            expected_covers_chase = expected_net >= move_abs * 100 * expected_buffer
            fresh_returns_1 = self._safe_float(fresh_snapshot.get("returns_1"), 0.0) if isinstance(fresh_snapshot, dict) else 0.0
            fresh_returns_5 = self._safe_float(fresh_snapshot.get("returns_5"), 0.0) if isinstance(fresh_snapshot, dict) else 0.0
            fresh_momentum_ok = (
                (decision.action == Action.LONG and fresh_returns_1 >= -0.002 and fresh_returns_5 >= -0.004)
                or (decision.action == Action.SHORT and fresh_returns_1 <= 0.002 and fresh_returns_5 <= 0.004)
            )
            rescue_allowed = bool(
                fresh_snapshot
                and not fresh_quality_reason
                and fresh_price > 0
                and fresh_latest_gap <= max(allowed, 0.006)
                and move_abs <= rescue_cap
                and expected_covers_chase
                and fresh_momentum_ok
            )
            raw["pre_execution_price_recheck"] = {
                "triggered": True,
                "fresh_recheck_available": bool(fresh_snapshot),
                "fresh_reason": fresh_quality_reason,
                "fresh_price": fresh_price,
                "fresh_latest_gap_pct": round(fresh_latest_gap * 100, 4),
                "original_move_pct": round(move * 100, 4),
                "rescue_cap_pct": round(rescue_cap * 100, 4),
                "expected_buffer_multiple": round(expected_buffer, 4),
                "expected_covers_chase": bool(expected_covers_chase),
                "fresh_momentum_ok": bool(fresh_momentum_ok),
                "rescued": rescue_allowed,
            }
            decision.raw_response = raw
            if rescue_allowed:
                decision.feature_snapshot = fresh_snapshot
                logger.info(
                    "pre-order price guard rescued by fresh recheck",
                    symbol=decision.symbol,
                    move_pct=round(move * 100, 4),
                    fresh_price=fresh_price,
                    expected_net=round(expected_net, 4),
                )
                return None

        if decision.action == Action.LONG and move > allowed:
            reason = (
                f"下单前价格已比分析时上涨 {move * 100:.2f}%，"
                f"超过允许偏移 {allowed * 100:.2f}%。系统已即时刷新该币种行情复核，"
                "但偏移仍过大或盘口/动量未通过复核；为避免追高，本次不执行。"
            )
            self._remember_temporary_entry_block(
                decision.symbol,
                reason,
                PRICE_GUARD_ENTRY_BLOCK_MINUTES,
            )
            return reason
        if decision.action == Action.SHORT and move < -allowed:
            reason = (
                f"下单前价格已比分析时下跌 {abs(move) * 100:.2f}%，"
                f"超过允许偏移 {allowed * 100:.2f}%。系统已即时刷新该币种行情复核，"
                "但偏移仍过大或盘口/动量未通过复核；为避免追空，本次不执行。"
            )
            self._remember_temporary_entry_block(
                decision.symbol,
                reason,
                PRICE_GUARD_ENTRY_BLOCK_MINUTES,
            )
            return reason
        if abs(move) > allowed * 2:
            reason = (
                f"下单前价格较分析时波动 {abs(move) * 100:.2f}%，行情变化太快，"
                "系统已即时刷新该币种行情复核，但仍不适合沿用旧信号，本次不执行。"
            )
            self._remember_temporary_entry_block(
                decision.symbol,
                reason,
                PRICE_GUARD_ENTRY_BLOCK_MINUTES,
            )
            return reason
        return None

    def _short_text(self, value: Any, limit: int = 180) -> str:
        return " ".join(str(value or "").split())[:limit]

    def _rejected_execution_result(
        self, decision: DecisionOutput, error: Exception | str
    ) -> ExecutionResult:
        message = str(error)
        return ExecutionResult(
            order_id="rejected",
            symbol=decision.symbol,
            side=self._decision_side(decision),
            order_type="market",
            quantity=0.0,
            price=0.0,
            status=OrderStatus.REJECTED,
            raw_response={"error": message},
        )

    async def _log_decision(self, decision: DecisionOutput, is_paper: bool) -> int | None:
        try:
            async with get_session_ctx() as session:
                repo = DecisionRepository(session)
                raw_response = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                raw_analysis_type = str(raw_response.get("analysis_type") or "").lower()
                if raw_analysis_type in {"position", "position_review", "holding", "holdings"}:
                    analysis_type = "position"
                elif raw_analysis_type in {"market", "market_scan", "symbol_scan"}:
                    analysis_type = "market"
                elif raw_response.get("position_review_policy") or raw_response.get("position_review") or decision.action.value in {"close_long", "close_short"}:
                    analysis_type = "position"
                else:
                    analysis_type = "market"
                raw_response = append_decision_stage(
                    raw_response,
                    DecisionStage.AI_ANALYSIS,
                    DecisionStageStatus.COMPLETED,
                    "AI 已完成分析并生成裁决。",
                    data={
                        "analysis_type": analysis_type,
                        "action": decision.action.value,
                        "confidence": float(decision.confidence or 0.0),
                    },
                )
                decision.raw_response = raw_response
                display_symbol = self._normalize_position_symbol(decision.symbol) or decision.symbol
                record = await repo.log_decision({
                    "model_name": decision.model_name,
                    "symbol": display_symbol,
                    "action": decision.action.value,
                    "confidence": decision.confidence,
                    "reasoning": sanitize_text(decision.reasoning),
                    "position_size_pct": decision.position_size_pct,
                    "suggested_leverage": decision.suggested_leverage,
                    "stop_loss_pct": decision.stop_loss_pct,
                    "take_profit_pct": decision.take_profit_pct,
                    "feature_snapshot": self._json_safe_payload(decision.feature_snapshot),
                    "raw_llm_response": self._json_safe_payload(decision.raw_response),
                    "analysis_type": analysis_type,
                    "is_paper": is_paper,
                })
                return record.id
        except Exception as e:
            logger.error("failed to log decision", error=str(e))
            return None

    def _json_safe_payload(self, value: Any) -> Any:
        """Return a JSON-column-safe copy of model/feature payloads."""
        if value is None or isinstance(value, (int, bool)):
            return value
        if isinstance(value, str):
            return sanitize_text(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {
                str(k): self._json_safe_payload(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe_payload(v) for v in value]
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return self._json_safe_payload(item())
            except Exception:
                pass
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        return sanitize_text(str(value))

    async def _create_shadow_backtests(
        self,
        decision_id: int | None,
        decision: DecisionOutput,
        fv: Any,
        execution_mode: str,
        analysis_type: str = "market",
    ) -> None:
        """Record delayed market-outcome samples without affecting execution."""
        if analysis_type != "market":
            return
        entry_price = self._safe_float(
            getattr(fv, "current_price", 0.0)
            or getattr(fv, "close", 0.0)
            or (decision.feature_snapshot or {}).get("current_price"),
            0.0,
        )
        if entry_price <= 0:
            return
        now = datetime.now(timezone.utc)
        try:
            async with get_session_ctx() as session:
                repo = MemoryRepository(session)
                for horizon in SHADOW_BACKTEST_HORIZONS_MINUTES:
                    await repo.create_shadow_backtest({
                        "decision_id": decision_id,
                        "model_name": decision.model_name,
                        "execution_mode": execution_mode,
                        "symbol": decision.symbol,
                        "analysis_type": analysis_type,
                        "decision_action": decision.action.value,
                        "decision_confidence": float(decision.confidence or 0.0),
                        "entry_price": entry_price,
                        "feature_snapshot": decision.feature_snapshot or getattr(fv, "to_dict", lambda: {})(),
                        "raw_llm_response": decision.raw_response if isinstance(decision.raw_response, dict) else {},
                        "status": "pending",
                        "due_at": now + timedelta(minutes=int(horizon)),
                        "horizon_minutes": int(horizon),
                    })
        except Exception as e:
            logger.debug("failed to create shadow backtests", symbol=decision.symbol, error=str(e))

    async def _update_due_shadow_backtests(self) -> None:
        """Complete due shadow samples using latest OKX swap price."""
        try:
            async with get_session_ctx() as session:
                repo = MemoryRepository(session)
                rows = await repo.get_due_shadow_backtests(limit=200)
                if not rows:
                    return
                price_cache: dict[str, float] = {}
                for row in rows:
                    symbol = self._normalize_position_symbol(row.symbol) or row.symbol
                    if symbol not in price_cache:
                        price_cache[symbol] = await self._latest_price_for_symbol(symbol)
                    actual_price = self._safe_float(price_cache.get(symbol), 0.0)
                    entry_price = self._safe_float(row.entry_price, 0.0)
                    if actual_price <= 0 or entry_price <= 0:
                        continue

                    long_return = (actual_price - entry_price) / entry_price
                    short_return = (entry_price - actual_price) / entry_price
                    threshold = max(
                        float(settings.shadow_memory_min_return_pct or 0.40) / 100.0,
                        SHADOW_MISSED_OPPORTUNITY_THRESHOLD,
                    )
                    best_action = "hold"
                    if long_return >= threshold and long_return >= short_return:
                        best_action = "long"
                    elif short_return >= threshold and short_return > long_return:
                        best_action = "short"

                    decision_action = str(row.decision_action or "hold")
                    missed = decision_action == "hold" and best_action in {"long", "short"}
                    note = ""
                    if missed:
                        note = (
                            f"当时观望，但 {int(row.horizon_minutes)} 分钟后"
                            f"{'做多' if best_action == 'long' else '做空'}方向收益约"
                            f"{max(long_return, short_return) * 100:.2f}%。"
                        )
                    elif decision_action in {"long", "short"} and decision_action != best_action and best_action != "hold":
                        note = f"实际更优方向是 {'做多' if best_action == 'long' else '做空'}，用于后续复盘。"

                    await repo.complete_shadow_backtest(
                        row,
                        actual_price=actual_price,
                        long_return_pct=long_return * 100,
                        short_return_pct=short_return * 100,
                        best_action=best_action,
                        missed_opportunity=missed,
                        note=note,
                    )
                    if settings.shadow_memory_enabled:
                        await self._record_shadow_memory_in_session(
                            repo,
                            row,
                            long_return=long_return,
                            short_return=short_return,
                            best_action=best_action,
                            threshold=threshold,
                        )
                logger.info("shadow backtests updated", count=len(rows))
        except Exception as e:
            logger.debug("failed to update shadow backtests", error=str(e))

    async def _record_shadow_memory_in_session(
        self,
        repo: MemoryRepository,
        row: Any,
        *,
        long_return: float,
        short_return: float,
        best_action: str,
        threshold: float,
    ) -> None:
        """Turn shadow backtest outcomes into small, reusable expert memories."""
        decision_action = str(getattr(row, "decision_action", "") or "hold")
        symbol = str(getattr(row, "symbol", "") or "")
        horizon = int(getattr(row, "horizon_minutes", 0) or 0)
        if not symbol or horizon <= 0:
            return

        if decision_action == "hold" and best_action in {"long", "short"}:
            realized = long_return if best_action == "long" else short_return
            if realized < threshold:
                return
            memory_type = "shadow_missed_opportunity"
            side = best_action
            confidence_adjustment = 0.04
            position_size_multiplier = 1.04
            success_count = 1
            failure_count = 0
            outcome_text = (
                f"当时选择观望，但 {horizon} 分钟后"
                f"{'做多' if side == 'long' else '做空'}方向涨跌收益约 {realized * 100:.2f}%。"
            )
            recommended = "allow_small_probe_with_filters"
        elif decision_action in {"long", "short"}:
            realized = long_return if decision_action == "long" else short_return
            side = decision_action
            if realized >= threshold:
                memory_type = "shadow_good_signal"
                confidence_adjustment = 0.025
                position_size_multiplier = 1.02
                success_count = 1
                failure_count = 0
                outcome_text = (
                    f"Shadow replay: {self._side_label(side)} signal returned "
                    f"{realized * 100:.2f}% after {horizon} minutes. "
                    "This pattern was short-term effective."
                )
                recommended = "keep_with_filters"
            elif realized <= -threshold:
                memory_type = "shadow_bad_signal"
                confidence_adjustment = -0.06
                position_size_multiplier = 0.78
                success_count = 0
                failure_count = 1
                opposite = "short" if side == "long" else "long"
                opposite_return = short_return if opposite == "short" else long_return
                outcome_text = (
                    f"Shadow replay: {self._side_label(side)} signal lost "
                    f"{abs(realized) * 100:.2f}% after {horizon} minutes, while "
                    f"{self._side_label(opposite)} returned {opposite_return * 100:.2f}%."
                )
                recommended = "reduce_risk"
            else:
                return
        else:
            return

        feature_snapshot = getattr(row, "feature_snapshot", None) or {}
        pattern = self._shadow_memory_pattern(feature_snapshot, symbol, side, horizon)
        labels = {slot["name"]: slot.get("label", slot["name"]) for slot in FIXED_AI_MODEL_SLOTS}
        for expert_name, lesson in self._shadow_expert_lessons(
            symbol=symbol,
            side=side,
            memory_type=memory_type,
            outcome_text=outcome_text,
            feature_snapshot=feature_snapshot,
        ).items():
            await repo.upsert_memory({
                "expert_name": expert_name,
                "expert_label": labels.get(expert_name, expert_name),
                "symbol": symbol,
                "side": side,
                "memory_type": memory_type,
                "market_pattern": pattern,
                "lesson": lesson,
                "recommended_action": recommended,
                "confidence_adjustment": confidence_adjustment,
                "position_size_multiplier": position_size_multiplier,
                "evidence_count": 1,
                "success_count": success_count,
                "failure_count": failure_count,
                "confidence_score": 0.52,
                "memory_key": (
                    f"{expert_name}|shadow|{symbol}|{side}|{memory_type}|"
                    f"{horizon}m|{self._shadow_feature_bucket(feature_snapshot)}"
                ),
                "extra": {
                    "source": "shadow_backtest",
                    "shadow_backtest_id": getattr(row, "id", None),
                    "decision_id": getattr(row, "decision_id", None),
                    "decision_action": decision_action,
                    "best_action": best_action,
                    "horizon_minutes": horizon,
                    "entry_price": getattr(row, "entry_price", None),
                    "actual_price": getattr(row, "actual_price", None),
                    "long_return_pct": long_return * 100,
                    "short_return_pct": short_return * 100,
                },
            })

    def _shadow_expert_lessons(
        self,
        *,
        symbol: str,
        side: str,
        memory_type: str,
        outcome_text: str,
        feature_snapshot: dict[str, Any],
    ) -> dict[str, str]:
        side_label = self._side_label(side)
        if memory_type == "shadow_missed_opportunity":
            return {
                "trend_expert": (
                    f"{symbol} {side_label} missed opportunity. {outcome_text} "
                    "When directional structure, ADX, moving averages and MACD align, raise directional support without deciding size."
                ),
                "momentum_expert": (
                    f"{symbol} {side_label} missed opportunity. {outcome_text} "
                    "If expected net return, fee coverage and loss probability are favorable, support a small profit-quality probe."
                ),
                "sentiment_expert": (
                    f"{symbol} {side_label} missed opportunity. {outcome_text} "
                    "If 1/5/10/30 minute path and event shock risk are favorable, support earlier execution timing."
                ),
                "risk_expert": (
                    f"{symbol} {side_label} missed opportunity. {outcome_text} "
                    "If there is no hard risk, prefer size/leverage control instead of blocking the trade."
                ),
            }
        if memory_type == "shadow_good_signal":
            return {
                "trend_expert": (
                    f"{symbol} {side_label} signal validated by shadow replay. {outcome_text} "
                    "Raise directional confidence when a similar directional structure appears."
                ),
                "momentum_expert": (
                    f"{symbol} {side_label} signal validated by shadow replay. {outcome_text} "
                    "Support execution when expected net return and payoff quality stay positive after fees."
                ),
                "sentiment_expert": (
                    f"{symbol} {side_label} signal validated by shadow replay. {outcome_text} "
                    "Support execution timing when short-horizon path continuation is similar."
                ),
                "risk_expert": (
                    f"{symbol} {side_label} signal validated by shadow replay. {outcome_text} "
                    "Allow small size when no hard risk is present."
                ),
            }
        return {
            "trend_expert": (
                f"{symbol} {side_label} signal looked weak in shadow replay. {outcome_text} "
                "Require trend continuation before raising confidence next time."
            ),
            "momentum_expert": (
                f"{symbol} {side_label} signal looked weak in shadow replay. {outcome_text} "
                "Check whether expected net return, fee coverage or payoff ratio is too weak before chasing."
            ),
            "sentiment_expert": (
                f"{symbol} {side_label} signal looked weak in shadow replay. {outcome_text} "
                "Check whether the short-horizon path is already reversing before execution."
            ),
            "risk_expert": (
                f"{symbol} {side_label} signal looked weak in shadow replay. {outcome_text} "
                "Reduce size/leverage or block new entries under similar conditions."
            ),
        }

    def _shadow_memory_pattern(
        self,
        feature_snapshot: dict[str, Any],
        symbol: str,
        side: str,
        horizon: int,
    ) -> str:
        return (
            f"{symbol} {self._side_label(side)}影子复盘 {horizon}分钟，"
            f"ADX={self._safe_float(feature_snapshot.get('adx_14'), 0.0):.1f}，"
            f"量比={self._safe_float(feature_snapshot.get('volume_ratio'), 0.0):.2f}，"
            f"5周期收益={self._safe_float(feature_snapshot.get('returns_5'), 0.0) * 100:.2f}%，"
            f"盘口倾斜={self._safe_float(feature_snapshot.get('orderbook_imbalance'), 0.0):.2f}"
        )

    def _shadow_feature_bucket(self, feature_snapshot: dict[str, Any]) -> str:
        adx = self._safe_float(feature_snapshot.get("adx_14"), 0.0)
        volume_ratio = self._safe_float(feature_snapshot.get("volume_ratio"), 0.0)
        returns_5 = self._safe_float(feature_snapshot.get("returns_5"), 0.0)
        imbalance = self._safe_float(feature_snapshot.get("orderbook_imbalance"), 0.0)
        adx_bucket = "adx_hi" if adx >= 25 else "adx_mid" if adx >= settings.min_entry_adx else "adx_low"
        volume_bucket = "vol_hi" if volume_ratio >= 1.2 else "vol_ok" if volume_ratio >= settings.min_entry_volume_ratio else "vol_low"
        momentum_bucket = "mom_up" if returns_5 > 0.002 else "mom_down" if returns_5 < -0.002 else "mom_flat"
        book_bucket = "bid_wall" if imbalance > 0.12 else "ask_wall" if imbalance < -0.12 else "book_flat"
        return f"{adx_bucket}|{volume_bucket}|{momentum_bucket}|{book_bucket}"

    def _side_label(self, side: str) -> str:
        return "做多" if str(side).lower() == "long" else "做空" if str(side).lower() == "short" else str(side)

    async def _log_trade(
        self,
        result,
        model_name: str,
        decision: DecisionOutput,
        decision_id: int | None = None,
    ) -> None:
        try:
            async with get_session_ctx() as session:
                repo = TradeRepository(session)
                raw = result.raw_response or {}
                raw_error = raw.get("error") if isinstance(raw, dict) else None
                await repo.create_order({
                    "model_name": model_name,
                    "execution_mode": self._get_model_execution_mode(model_name),
                    "symbol": result.symbol,
                    "side": result.side,
                    "order_type": result.order_type,
                    "quantity": result.quantity,
                    "price": result.price,
                    "status": result.status.value,
                    "fee": result.fee,
                    "decision_id": decision_id,
                    "exchange_order_id": result.exchange_order_id,
                    "filled_at": result.timestamp,
                })
        except Exception as e:
            logger.error("failed to log trade", error=str(e))

    async def _persist_position_from_execution(
        self,
        model_name: str,
        decision: DecisionOutput,
        result,
        execution_mode: str,
    ) -> None:
        """Persist open and closed position records for paper/live parity."""
        exit_progress = decision.is_exit and self._is_exit_progress_execution(result)
        if not self._is_exchange_confirmed_execution(result) and not exit_progress:
            return
        if result.quantity <= 0 or result.price <= 0:
            return

        try:
            async with get_session_ctx() as session:
                repo = TradeRepository(session)
                if decision.action in (Action.LONG, Action.SHORT):
                    side = "long" if decision.action == Action.LONG else "short"
                    stop_loss = (
                        result.price * (1 - decision.stop_loss_pct)
                        if side == "long"
                        else result.price * (1 + decision.stop_loss_pct)
                    )
                    take_profit = (
                        result.price * (1 + decision.take_profit_pct)
                        if side == "long"
                        else result.price * (1 - decision.take_profit_pct)
                    )
                    await repo.open_position({
                        "model_name": model_name,
                        "execution_mode": execution_mode,
                        "symbol": result.symbol,
                        "side": side,
                        "quantity": result.quantity,
                        "entry_price": result.price,
                        "current_price": result.price,
                        "leverage": decision.suggested_leverage,
                        "unrealized_pnl": 0.0,
                        "realized_pnl": 0.0,
                        "stop_loss_price": stop_loss,
                        "take_profit_price": take_profit,
                    })
                    return

                if decision.action in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
                    side = "long" if decision.action == Action.CLOSE_LONG else "short"
                    positions = await repo.get_matching_open_positions(
                        model_name=model_name,
                        symbol=result.symbol,
                        side=side,
                        execution_mode=execution_mode,
                    )
                    exchange_backed_ids = await self._exchange_backed_position_ids(session, positions)
                    positions = sorted(
                        positions,
                        key=lambda p: (
                            p.id not in exchange_backed_ids,
                            p.created_at or datetime.min,
                        ),
                    )
                    remaining_qty = result.quantity
                    total_pnl = 0.0
                    for pos in positions:
                        if remaining_qty <= 0:
                            break
                        close_qty = min(pos.quantity, remaining_qty)
                        close_fee = self._proportional_fee(result.fee, close_qty, result.quantity)
                        entry_fee = await self._entry_fee_for_position(session, pos, close_qty)
                        if side == "long":
                            gross_pnl = (result.price - pos.entry_price) * close_qty
                        else:
                            gross_pnl = (pos.entry_price - result.price) * close_qty
                        pnl = gross_pnl - entry_fee - close_fee
                        total_pnl += pnl
                        if close_qty < pos.quantity:
                            pos.quantity -= close_qty
                            remaining_qty = 0
                            closed_pos = await repo.open_position({
                                "model_name": model_name,
                                "execution_mode": execution_mode,
                                "symbol": result.symbol,
                                "side": side,
                                "quantity": close_qty,
                                "entry_price": pos.entry_price,
                                "current_price": result.price,
                                "leverage": pos.leverage,
                                "unrealized_pnl": 0.0,
                                "realized_pnl": pnl,
                                "stop_loss_price": pos.stop_loss_price,
                                "take_profit_price": pos.take_profit_price,
                                "is_open": False,
                                "closed_at": result.timestamp,
                                "created_at": pos.created_at,
                            })
                            await self._record_trade_reflection_in_session(
                                session,
                                closed_pos,
                                exit_price=result.price,
                                entry_fee=entry_fee,
                                close_fee=close_fee,
                                gross_pnl=gross_pnl,
                                source="system_execution",
                                decision=decision,
                            )
                        else:
                            self._remove_position_profit_peak(model_name, result.symbol, side)
                            pos.is_open = False
                            pos.current_price = result.price
                            pos.unrealized_pnl = 0.0
                            pos.realized_pnl = pnl
                            pos.closed_at = result.timestamp
                            remaining_qty -= close_qty
                            await self._record_trade_reflection_in_session(
                                session,
                                pos,
                                exit_price=result.price,
                                entry_fee=entry_fee,
                                close_fee=close_fee,
                                gross_pnl=gross_pnl,
                                source="system_execution",
                                decision=decision,
                            )
                    if result.pnl == 0.0 and total_pnl != 0.0:
                        result.pnl = total_pnl
                    await session.flush()
        except Exception as e:
            logger.error("failed to persist position", error=str(e))

    async def _record_trade_reflection_in_session(
        self,
        session,
        pos,
        exit_price: float,
        entry_fee: float,
        close_fee: float,
        gross_pnl: float,
        source: str,
        decision: DecisionOutput | None = None,
    ) -> None:
        """Create a compact post-trade reflection and update expert memories."""
        if not settings.expert_memory_enabled:
            return
        try:
            realized_pnl = float(pos.realized_pnl or 0.0)
            entry_price = float(pos.entry_price or 0.0)
            quantity = float(pos.quantity or 0.0)
            notional = abs(entry_price * quantity)
            pnl_pct = realized_pnl / notional if notional > 0 else 0.0
            hold_minutes = self._position_hold_minutes(pos)
            outcome = "profit" if realized_pnl > 0 else "loss" if realized_pnl < 0 else "flat"
            pattern = self._reflection_pattern(pos, pnl_pct, hold_minutes)
            mistake, improvement = self._reflection_summary(pos, outcome, pnl_pct, hold_minutes)
            expert_lessons = self._build_expert_lessons(
                pos=pos,
                outcome=outcome,
                pnl_pct=pnl_pct,
                hold_minutes=hold_minutes,
                pattern=pattern,
                decision=decision,
            )
            repo = MemoryRepository(session)
            reflection = await repo.create_reflection({
                "position_id": int(pos.id or 0),
                "model_name": pos.model_name,
                "execution_mode": pos.execution_mode,
                "symbol": pos.symbol,
                "side": pos.side,
                "entry_price": entry_price,
                "exit_price": float(exit_price or 0.0),
                "quantity": quantity,
                "realized_pnl": realized_pnl,
                "fee_estimate": abs(float(entry_fee or 0.0)) + abs(float(close_fee or 0.0)),
                "hold_minutes": hold_minutes,
                "outcome": outcome,
                "mistake_summary": mistake,
                "improvement_summary": improvement,
                "expert_lessons": expert_lessons,
                "source": source,
            })
            if reflection is None:
                return

            for lesson in expert_lessons.values():
                await repo.upsert_memory({
                    **lesson,
                    "source_position_id": int(pos.id or 0),
                    "extra": {
                        "reflection_id": reflection.id,
                        "realized_pnl": realized_pnl,
                        "pnl_pct": pnl_pct,
                        "hold_minutes": hold_minutes,
                        "gross_pnl": gross_pnl,
                        "entry_fee": entry_fee,
                        "close_fee": close_fee,
                    },
                })
        except Exception as e:
            logger.warning(
                "failed to record trade reflection",
                position_id=getattr(pos, "id", None),
                symbol=getattr(pos, "symbol", None),
                error=str(e),
            )

    async def _backfill_trade_reflections(self) -> None:
        """Create memories from already closed positions after service restart."""
        if not settings.expert_memory_enabled:
            return
        try:
            async with get_session_ctx() as session:
                repo = TradeRepository(session)
                rows = await repo.get_position_records(
                    execution_mode=mode_manager.mode.value,
                    model_name=ENSEMBLE_TRADER_NAME,
                    is_open=False,
                    limit=200,
                )
                for pos in rows:
                    if not pos.closed_at:
                        continue
                    await self._record_trade_reflection_in_session(
                        session,
                        pos,
                        exit_price=float(pos.current_price or pos.entry_price or 0.0),
                        entry_fee=0.0,
                        close_fee=0.0,
                        gross_pnl=float(pos.realized_pnl or 0.0),
                        source="startup_backfill",
                        decision=None,
                    )
        except Exception as e:
            logger.warning("failed to backfill trade reflections", error=str(e))

    def _position_hold_minutes(self, pos) -> float:
        opened = getattr(pos, "created_at", None)
        closed = getattr(pos, "closed_at", None) or datetime.now(timezone.utc)
        if opened is None:
            return 0.0
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        if closed.tzinfo is None:
            closed = closed.replace(tzinfo=timezone.utc)
        return max((closed - opened).total_seconds() / 60.0, 0.0)

    def _reflection_pattern(self, pos, pnl_pct: float, hold_minutes: float) -> str:
        side_label = "long" if str(pos.side).lower() == "long" else "short"
        speed = "ultra_short" if hold_minutes < 5 else "short_term" if hold_minutes < 30 else "longer_hold"
        loss_level = "large_loss" if pnl_pct <= -0.01 else "small_loss" if pnl_pct < 0 else "profit" if pnl_pct > 0 else "flat"
        leverage = float(getattr(pos, "leverage", 1.0) or 1.0)
        return f"{pos.symbol} {side_label}, {speed}, {leverage:.1f}x, {loss_level}"

    def _reflection_summary(
        self,
        pos,
        outcome: str,
        pnl_pct: float,
        hold_minutes: float,
    ) -> tuple[str, str]:
        side_label = "做多" if str(pos.side).lower() == "long" else "做空"
        if outcome == "loss":
            mistake = (
                f"{pos.symbol} {side_label}最终亏损 {pnl_pct:.2%}，"
                "说明入场后的趋势延续、成交量配合或退出时机至少有一项不足。"
            )
            improvement = (
                "下次同类场景需要提高入场质量要求，优先降低仓位和杠杆；"
                "如果短时间内没有走出利润缓冲，持仓专家应更早要求复盘。"
            )
        elif outcome == "profit":
            mistake = f"{pos.symbol} {side_label}本次盈利，说明该方向在当时条件下有可执行边际。"
            improvement = "保留这类有效条件，但仍需确认成交量、趋势强度和止损收益比，不允许盲目放大。"
        else:
            mistake = f"{pos.symbol} {side_label}基本打平，收益没有明显覆盖机会成本。"
            improvement = "下次同类场景降低优先级，只有当趋势、动量和成交量更明确时才开仓。"
        if hold_minutes < 5 and outcome != "profit":
            improvement += " 本次持仓很短即退出，说明入场点或止盈止损距离可能过窄。"
        return mistake, improvement

    def _build_expert_lessons(
        self,
        pos,
        outcome: str,
        pnl_pct: float,
        hold_minutes: float,
        pattern: str,
        decision: DecisionOutput | None = None,
    ) -> dict[str, dict[str, Any]]:
        side = str(pos.side or "").lower()
        symbol = str(pos.symbol or "")
        is_loss = outcome == "loss"
        big_loss = pnl_pct <= -0.01
        is_profit = outcome == "profit"
        adjustment = -0.12 if big_loss else -0.08 if is_loss else 0.03 if is_profit else -0.03
        size_multiplier = 0.45 if big_loss else 0.60 if is_loss else 1.0 if is_profit else 0.80
        memory_type = "loss_lesson" if is_loss else "profit_pattern" if is_profit else "flat_lesson"
        recommended = "reduce_risk" if is_loss else "keep_with_filters" if is_profit else "wait_for_better_setup"
        evidence_success = 1 if is_profit else 0
        evidence_failure = 1 if is_loss else 0

        labels = {slot["name"]: slot.get("label", slot["name"]) for slot in FIXED_AI_MODEL_SLOTS}
        side_label = "long" if side == "long" else "short"
        base_key = f"{symbol}|{side}|{memory_type}|{self._lesson_bucket(pnl_pct, hold_minutes)}"
        lessons = {
            "trend_expert": (
                f"{symbol} {side_label} under pattern [{pattern}] ended as {outcome}. "
                "Next time judge only directional quality: MA direction, ADX, MACD and breakout structure."
            ),
            "momentum_expert": (
                f"{symbol} {side_label} under pattern [{pattern}] ended as {outcome}. "
                "Next time prioritize expected net return, fee coverage, loss probability and payoff ratio over win rate."
            ),
            "sentiment_expert": (
                f"{symbol} {side_label} under pattern [{pattern}] ended as {outcome}. "
                "Next time verify the 1/5/10/30 minute path, continuation risk, reversal risk and event shock before timing execution."
            ),
            "position_expert": (
                f"{symbol} {side_label} held {hold_minutes:.1f} minutes and ended as {outcome}. "
                "Check whether profit should be locked, loss can repair, loss is expanding, or the position deserves adding/reducing."
            ),
            "risk_expert": (
                f"{symbol} {side_label} under pattern [{pattern}] ended as {outcome}. "
                "Check abnormal wick, liquidity, extreme volatility, margin limits and exchange constraints before allowing risk."
            ),
        }
        result: dict[str, dict[str, Any]] = {}
        for expert_name, lesson in lessons.items():
            result[expert_name] = {
                "expert_name": expert_name,
                "expert_label": labels.get(expert_name, expert_name),
                "symbol": symbol,
                "side": side,
                "memory_type": memory_type,
                "market_pattern": pattern,
                "lesson": lesson,
                "recommended_action": recommended,
                "confidence_adjustment": adjustment,
                "position_size_multiplier": size_multiplier,
                "evidence_count": 1,
                "success_count": evidence_success,
                "failure_count": evidence_failure,
                "confidence_score": 0.65 if is_loss else 0.55,
                "memory_key": f"{expert_name}|{base_key}",
            }
        return result

    def _lesson_bucket(self, pnl_pct: float, hold_minutes: float) -> str:
        pnl_bucket = "big_loss" if pnl_pct <= -0.01 else "loss" if pnl_pct < 0 else "profit" if pnl_pct > 0 else "flat"
        time_bucket = "fast" if hold_minutes < 5 else "short" if hold_minutes < 30 else "long"
        return f"{pnl_bucket}|{time_bucket}"

    def _outcome_label(self, outcome: str) -> str:
        return {
            "profit": "盈利",
            "loss": "亏损",
            "flat": "打平",
        }.get(outcome, outcome)

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
                await repo.log_risk_event({
                    "event_type": event_type,
                    "severity": severity,
                    "symbol": symbol,
                    "details": {"message": details},
                    "triggered_by_model": model_name,
                })
        except Exception:
            pass

    async def _mark_decision_executed(self, decision_id: int, execution_price: float) -> None:
        try:
            async with get_session_ctx() as session:
                repo = DecisionRepository(session)
                await repo.mark_executed(decision_id, execution_price)
        except Exception as e:
            logger.error("failed to mark decision executed", error=str(e))

    def _execution_reason_is_unusable(self, reason: Any) -> bool:
        text = str(reason or "").strip()
        if not text:
            return False
        unusable_markers = (
            "原始说明已损坏",
            "无法准确还原",
            "鍘嗗彶璁板綍",
            "鎹熷潖",
        )
        return any(marker in text for marker in unusable_markers)

    def _recover_execution_reason_from_decision_row(
        self,
        decision: AIDecision | None,
        fallback: Any = None,
    ) -> str | None:
        if decision is None:
            return None
        raw = decision.raw_llm_response if isinstance(decision.raw_llm_response, dict) else {}
        action = str(decision.action or "")
        if action in {"close_long", "close_short"}:
            close_evidence = raw.get("close_evidence") if isinstance(raw.get("close_evidence"), dict) else {}
            action_plan = str(close_evidence.get("action_plan") or "").lower()
            plan_label = "全平" if action_plan == "full_close" else "减仓" if action_plan == "reduce" else "平仓"
            close_reason = str(close_evidence.get("reason") or decision.reasoning or "").strip()
            pnl = self._safe_float(close_evidence.get("position_unrealized_pnl"), 0.0)
            if close_reason:
                return (
                    f"平仓裁决已生成但本轮没有确认到 OKX 平仓订单结果：AI 建议{plan_label}，"
                    f"当时估算浮动盈亏 {pnl:.4f} USDT。裁决依据：{close_reason}"
                    "系统会继续以 OKX 实际仓位和执行记录为准同步；如果仓位仍存在，下一轮持仓复盘会重新评估并提交平仓。"
                )
            return (
                "平仓裁决已生成但本轮没有确认到 OKX 平仓订单结果。"
                "系统会继续以 OKX 实际仓位和执行记录为准同步；如果仓位仍存在，下一轮持仓复盘会重新评估并提交平仓。"
            )
        if fallback:
            return str(fallback)
        return None

    def _record_decision_stage(
        self,
        decision: DecisionOutput,
        stage: str,
        status: str,
        reason: str | None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw = append_decision_stage(
            raw,
            stage,
            status,
            sanitize_text(reason) if reason else "",
            data=self._json_safe_payload(data or {}) if data else None,
        )
        decision.raw_response = raw
        return raw

    async def _record_and_persist_decision_stage(
        self,
        decision_id: int | None,
        decision: DecisionOutput,
        stage: str,
        status: str,
        reason: str | None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = self._record_decision_stage(decision, stage, status, reason, data)
        if decision_id is not None:
            await self._mark_decision_raw_response(decision_id, raw)
        return raw

    async def _mark_decision_reason(self, decision_id: int, reason: str | None) -> None:
        try:
            async with get_session_ctx() as session:
                row = await session.get(AIDecision, int(decision_id))
                sanitized_reason = sanitize_text(reason)
                if self._execution_reason_is_unusable(sanitized_reason):
                    recovered = self._recover_execution_reason_from_decision_row(row, fallback=reason)
                    if recovered:
                        sanitized_reason = recovered
                repo = DecisionRepository(session)
                await repo.mark_execution_reason(decision_id, sanitized_reason)
        except Exception as e:
            logger.error("failed to mark decision reason", error=str(e))

    async def _mark_decision_pending_execution(self, decision_id: int, reason: str) -> None:
        """Mark an entry as in-flight without letting final-round fallback overwrite it."""
        await self._mark_decision_reason(decision_id, f"正在提交 OKX：{reason}")

    def _is_pending_execution_reason(self, reason: str | None) -> bool:
        text = str(reason or "")
        return (
            not text
            or text.startswith("正在提交 OKX")
            or text.startswith("本轮执行仍在处理中")
            or text.startswith("Execution still pending this round")
        )

    def _pending_execution_failed_reason(self, symbol: str, action: str | None = None) -> str:
        action_label = f"{action} " if action else ""
        return (
            f"{symbol} {action_label}开仓信号已经进入 OKX 下单流程，但在 "
            f"{ENTRY_PENDING_EXECUTION_MAX_SECONDS:.0f} 秒内没有生成本地订单记录，也没有拿到 OKX 成功或失败回报。"
            "系统已按下单流程异常处理，本次旧信号不再继续等待，下一轮会用最新行情重新分析。"
        )

    async def _ensure_decision_final_state(
        self,
        decision_id: int,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        results: dict[str, Any],
    ) -> None:
        """Ensure every attempted execution has a concrete final DB/dashboard state."""
        try:
            async with get_session_ctx() as session:
                row = await session.get(AIDecision, int(decision_id))
                if row is None or row.was_executed:
                    return
                order_count = (
                    await session.execute(
                        select(func.count(Order.id)).where(Order.decision_id == int(decision_id))
                    )
                ).scalar() or 0
                reason = str(row.execution_reason or "")
                if order_count > 0:
                    if self._is_pending_execution_reason(reason):
                        row.execution_reason = (
                            "本地订单记录已生成，但成交或拒单状态还没有最终确认。请以执行记录中的最新订单状态为准。"
                        )
                        await session.flush()
                    return
                if row.action in {"close_long", "close_short"} and (
                    not reason or self._execution_reason_is_unusable(reason)
                ):
                    recovered = self._recover_execution_reason_from_decision_row(row)
                    if recovered:
                        row.execution_reason = recovered
                        await session.flush()
                        results["decisions"].append({
                            "model": model_name,
                            "symbol": symbol,
                            "action": decision.action.value,
                            "approved": True,
                            "confidence": decision.confidence,
                            "executed": False,
                            "execution_status": "skipped",
                            "reason": recovered,
                            "is_paper": (self._get_model_execution_mode(model_name) == "paper"),
                        })
                    return
                if self._is_pending_execution_reason(reason):
                    row.execution_reason = self._pending_execution_failed_reason(symbol, decision.action.value)
                    await session.flush()
                    results["decisions"].append({
                        "model": model_name,
                        "symbol": symbol,
                        "action": decision.action.value,
                        "approved": True,
                        "confidence": decision.confidence,
                        "executed": False,
                        "execution_status": "error",
                        "reason": row.execution_reason,
                        "is_paper": (self._get_model_execution_mode(model_name) == "paper"),
                    })
        except Exception as e:
            logger.error("failed to ensure decision final state", decision_id=decision_id, error=str(e))

    async def _duplicate_decision_order_reason(
        self,
        decision_id: int,
        decision: DecisionOutput,
    ) -> str | None:
        """Prevent the same decision row from submitting more than one OKX order."""
        try:
            async with get_session_ctx() as session:
                order_count = (
                    await session.execute(
                        select(func.count(Order.id)).where(Order.decision_id == int(decision_id))
                    )
                ).scalar() or 0
        except Exception as e:
            logger.warning("failed duplicate decision order check", decision_id=decision_id, error=str(e))
            return None
        if int(order_count) <= 0:
            return None
        if decision.is_exit:
            return (
                f"同一条平仓决策已经生成过 {int(order_count)} 条订单，"
                "为避免重复平仓，本次重复进入执行流程已跳过。"
            )
        if decision.is_entry:
            return (
                f"同一条开仓决策已经生成过 {int(order_count)} 条订单，"
                "为避免重复开仓，本次重复进入执行流程已跳过。"
            )
        return None

    async def _mark_decision_raw_response(self, decision_id: int, raw_response: dict | None) -> None:
        try:
            async with get_session_ctx() as session:
                repo = DecisionRepository(session)
                await repo.update_raw_response(decision_id, raw_response)
        except Exception as e:
            logger.error("failed to update decision raw response", error=str(e))

    async def _fill_missing_decision_reasons(
        self,
        decision_ids: set[int] | list[int],
        reason: str,
    ) -> None:
        ids = [int(i) for i in decision_ids if i]
        if not ids:
            return
        try:
            reason = str(sanitize_text(reason) or reason)
            async with get_session_ctx() as session:
                repo = DecisionRepository(session)
                await repo.fill_missing_execution_reasons(ids, reason)
        except Exception as e:
            logger.error("failed to fill missing decision reasons", error=str(e))

    async def _expire_stale_waiting_entry_candidates(self) -> int:
        """Clear old entry candidates that were left in the sorting/pending state."""
        now = datetime.utcnow()
        waiting_cutoff = now - timedelta(seconds=ENTRY_DECISION_MAX_AGE_SECONDS)
        pending_cutoff = now - timedelta(seconds=ENTRY_PENDING_EXECUTION_MAX_SECONDS)
        waiting_patterns = (
            "已进入本轮开仓候选排序%",
            "本轮还在分析或排队中%",
        )
        pending_patterns = (
            "正在提交 OKX%",
            "本轮执行仍在处理中%",
            "Execution still pending this round%",
        )
        try:
            async with get_session_ctx() as session:
                waiting_stmt = select(AIDecision).where(
                    AIDecision.was_executed == False,
                    AIDecision.action.in_(["long", "short", "open_long", "open_short"]),
                    AIDecision.created_at <= waiting_cutoff,
                    or_(
                        *[
                            AIDecision.execution_reason.like(pattern)
                            for pattern in waiting_patterns
                        ]
                    ),
                )
                pending_stmt = select(AIDecision).where(
                    AIDecision.was_executed == False,
                    AIDecision.action.in_(["long", "short", "open_long", "open_short"]),
                    AIDecision.created_at <= pending_cutoff,
                    or_(
                        *[
                            AIDecision.execution_reason.like(pattern)
                            for pattern in pending_patterns
                        ]
                    ),
                )
                waiting_rows = list((await session.execute(waiting_stmt)).scalars().all())
                pending_rows = list((await session.execute(pending_stmt)).scalars().all())
                expired = 0

                def apply_reason(row: AIDecision, reason: str) -> None:
                    raw = row.raw_llm_response
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except Exception:
                            raw = {}
                    if not isinstance(raw, dict):
                        raw = {}
                    opportunity = raw.get("opportunity_score")
                    if not isinstance(opportunity, dict):
                        opportunity = {}
                    opportunity["selected_for_execution"] = False
                    opportunity["selection_reason"] = reason
                    raw["opportunity_score"] = opportunity
                    row.raw_llm_response = raw
                    row.execution_reason = sanitize_text(reason)

                for row in waiting_rows:
                    raw = row.raw_llm_response
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except Exception:
                            raw = {}
                    if not isinstance(raw, dict):
                        raw = {}
                    opportunity = raw.get("opportunity_score")
                    if not isinstance(opportunity, dict):
                        opportunity = {}
                    score = self._safe_float(opportunity.get("score"), float("nan"))
                    min_score = self._safe_float(
                        opportunity.get("min_score_required"),
                        MIN_ENTRY_OPPORTUNITY_SCORE,
                    )
                    expected_net = self._safe_float(
                        opportunity.get("expected_net_return_pct"),
                        0.0,
                    )
                    if expected_net <= 0:
                        reason = (
                            f"候选排序超时后复核：{row.symbol} 本次{self._action_label(row.action)}"
                            f"预期净收益 {expected_net:.4f}% 不为正，旧信号不再执行，下一轮重新分析。"
                        )
                    elif math.isfinite(score) and score <= min_score:
                        reason = (
                            f"候选排序超时后复核：{row.symbol} 本次{self._action_label(row.action)}"
                            f"机会评分 {score:.4f} 低于执行门槛 {min_score:.2f}，旧信号不再执行，下一轮重新分析。"
                        )
                    else:
                        reason = (
                            f"候选排序等待超过 {ENTRY_DECISION_MAX_AGE_SECONDS:.0f} 秒，"
                            "行情快照已经过期。为避免追单，本次旧信号不再执行，下一轮重新分析。"
                        )
                    apply_reason(row, reason)
                    expired += 1

                for row in pending_rows:
                    order_count = (
                        await session.execute(
                            select(func.count(Order.id)).where(Order.decision_id == int(row.id))
                        )
                    ).scalar() or 0
                    if int(order_count) > 0:
                        reason = (
                            "本地订单记录已生成，但成交或拒单状态还没有最终确认。"
                            "请以执行记录中的最新订单状态为准。"
                        )
                    else:
                        reason = self._pending_execution_failed_reason(row.symbol, row.action)
                    apply_reason(row, reason)
                    expired += 1

                if expired:
                    await session.flush()
                    logger.info(
                        "expired stale entry candidates",
                        waiting=len(waiting_rows),
                        pending=len(pending_rows),
                    )
                return expired
        except Exception as e:
            logger.warning("failed to expire stale entry candidates", error=str(e))
            return 0

    def _action_label(self, action: str | None) -> str:
        value = str(action or "")
        if value in {"long", "open_long"}:
            return "做多"
        if value in {"short", "open_short"}:
            return "做空"
        if value in {"close_long"}:
            return "平多"
        if value in {"close_short"}:
            return "平空"
        return value or "交易"

    def _execution_reason_from_result(self, result: ExecutionResult | None) -> str:
        if result is None:
            return "交易接口未返回执行结果。"
        raw = result.raw_response or {}
        if isinstance(raw, dict) and raw.get("entry_tracking"):
            message = str(sanitize_text(raw.get("message")) or "").strip()
            remaining = self._safe_float(raw.get("remaining_contracts"), 0.0)
            filled = self._safe_float(raw.get("filled_contracts"), 0.0)
            if result.status == OrderStatus.PARTIAL:
                return message or (
                    f"OKX 开仓委托已部分成交，已成交约 {filled:g} 张，"
                    f"剩余约 {remaining:g} 张仍在追单；本地等待 OKX 仓位同步确认。"
                )
            if result.status in {OrderStatus.OPEN, OrderStatus.PENDING}:
                return message or (
                    "OKX 开仓委托正在挂单或追单，尚未确认成交；"
                    "系统不会先创建本地持仓，也不会重复提交同方向开仓单。"
                )
            if message:
                return message
        if isinstance(raw, dict) and raw.get("exit_tracking"):
            message = str(sanitize_text(raw.get("message")) or "").strip()
            remaining = self._safe_float(raw.get("remaining_contracts"), 0.0)
            filled = self._safe_float(raw.get("filled_contracts"), 0.0)
            if result.status == OrderStatus.PARTIAL:
                if remaining > 0:
                    return message or f"OKX 平仓已部分成交，仍剩约 {remaining:g} 张合约在处理；系统会继续同步，不会重复提交平仓单。"
                return message or "OKX 平仓已部分成交，系统会继续同步最终成交结果。"
            if result.status in {OrderStatus.OPEN, OrderStatus.PENDING}:
                if filled > 0 or remaining > 0:
                    return message or f"OKX 平仓订单正在追单中，已成交约 {filled:g} 张，剩余约 {remaining:g} 张；系统不会重复提交。"
                return message or "OKX 平仓订单正在追单或等待成交，系统不会重复提交平仓单。"
            if message:
                return message
        error = raw.get("error") if isinstance(raw, dict) else None
        raw_error = raw.get("raw_error") if isinstance(raw, dict) else None
        translated_error = self._translate_execution_error_text(f"{error or ''} {raw_error or ''}")
        if translated_error:
            return translated_error
        if self._is_untradable_exchange_error(f"{error or ''} {raw_error or ''}"):
            return "OKX 提示该交易对当前不可交易，可能受账户地区/合规限制影响；系统已暂时跳过该交易对，避免重复分析和下单。"
        if self._is_no_exchange_position_error(f"{error or ''} {raw_error or ''}"):
            return "OKX 提示当前没有对应方向的可平仓位，可能已被 OKX 止盈/止损、手动平仓或刚刚同步延迟；本轮未重复提交。"
        if error:
            return str(sanitize_text(error) or error)
        if result.status == OrderStatus.FILLED:
            return "订单已成交。"
        status_map = {
            OrderStatus.PENDING: "待成交",
            OrderStatus.OPEN: "挂单中",
            OrderStatus.PARTIAL: "部分成交",
            OrderStatus.CANCELLED: "已取消",
            OrderStatus.REJECTED: "已拒绝",
        }
        status = status_map.get(result.status, result.status.value)
        return f"订单状态为{status}，未计为已执行。"

    def _translate_execution_error_text(self, text: str | None) -> str | None:
        message = str(text or "").strip()
        if not message:
            return None
        if "51008" in message or "Insufficient USDT margin" in message:
            return (
                "OKX 返回错误码 51008：账户可用 USDT 保证金不足，订单没有提交成功。"
                "通常是当前持仓/挂单占用保证金过高、可用余额不足，或本轮计划仓位过大；"
                "系统应优先处理已有持仓的平仓/减仓，不再继续加仓。"
            )
        if "59670" in message or "more than 5 open orders" in message:
            return (
                "OKX 拒绝调整杠杆：该交易对当前挂单超过 5 条。"
                "系统会跳过重复杠杆设置，必要时只清理旧的非保护挂单后重试。"
            )
        if (
            "open interest" in message.lower()
            and "platform" in message.lower()
            and "limit" in message.lower()
        ) or "has reached the platform's limit" in message:
            return (
                "OKX 拒绝开仓：该合约当前平台总持仓量已经达到 OKX 上限，"
                "交易所暂时不允许继续增加这个合约的新仓。"
                "这不是 AI 方向或下单数量计算错误；系统会临时跳过该币种，稍后等 OKX 限制解除再重新分析。"
            )
        return None

    def _is_exit_tracking_execution(self, result: ExecutionResult | None) -> bool:
        raw = result.raw_response if result else None
        return bool(isinstance(raw, dict) and raw.get("exit_tracking"))

    def _is_exit_progress_execution(self, result: ExecutionResult | None) -> bool:
        if not self._is_exit_tracking_execution(result):
            return False
        if result is None or result.status != OrderStatus.PARTIAL:
            return False
        order_id = str(result.exchange_order_id or "").strip()
        return bool(order_id and result.quantity > 0)

    def _is_exchange_confirmed_execution(self, result: ExecutionResult | None) -> bool:
        """Only treat an execution as real after OKX returns a concrete order id."""
        if result is None or result.status != OrderStatus.FILLED:
            return False
        order_id = str(result.exchange_order_id or "").strip()
        return bool(order_id and order_id not in {"hold", "rejected", "no_position"})

    async def _mark_decision_outcome(self, decision_id: int, outcome: str, pnl_pct: float) -> None:
        try:
            async with get_session_ctx() as session:
                repo = DecisionRepository(session)
                await repo.mark_outcome(decision_id, outcome, pnl_pct)
        except Exception as e:
            logger.error("failed to mark decision outcome", error=str(e))

    async def _persist_account_update(self, model_name: str, execution_model_name: str, result) -> None:
        try:
            async with get_session_ctx() as session:
                repo = AccountRepository(session)
                await repo.update_balance(model_name, result.pnl, result.pnl)
                is_win = result.pnl > 0
                await repo.record_trade_result(model_name, is_win)
        except Exception as e:
            logger.error("failed to persist account update", error=str(e))

    async def _publish_dashboard_update(self, results: dict) -> None:
        """Publish a dashboard update via Redis pub/sub."""
        if self.redis:
            try:
                import json

                await self.redis.publish(
                    "dashboard:update",
                    json.dumps({
                        "type": "trading_round",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "mode": mode_manager.mode.value,
                        "decisions": results.get("decisions", []),
                        "executions": results.get("executions", []),
                        "stats": self.get_stats(),
                    }, default=str),
                )
            except Exception:
                pass

    async def record_equity_snapshot(self) -> None:
        """Record current equity for each model for PnL chart history, and persist unrealized PnL to DB."""
        now = datetime.now(timezone.utc).isoformat()
        active_names = {ENSEMBLE_TRADER_NAME}
        # Remove stale data from deleted models
        for stale in list(self._pnl_history.keys()):
            if stale not in active_names:
                del self._pnl_history[stale]
        for model_name in active_names:
            state = await self._execution_allocation_state(mode_manager.mode.value)
            allocated = float(state.get("allocated_balance") or 0.0)
            equity = allocated
            unrealized = state.get("unrealized_pnl", 0.0)
            self._pnl_history.setdefault(model_name, []).append({
                "time": now,
                "equity": round(equity, 2),
            })
            # Keep last 500 snapshots per model
            if len(self._pnl_history[model_name]) > 500:
                self._pnl_history[model_name] = self._pnl_history[model_name][-500:]

            # Persist unrealized PnL to DB so competition rankings see it
            try:
                async with get_session_ctx() as session:
                    from db.repositories.account_repo import AccountRepository
                    repo = AccountRepository(session)
                    await repo.update_unrealized_pnl(model_name, round(unrealized, 2))
            except Exception:
                pass

    def get_pnl_history(self) -> dict[str, list[dict]]:
        """Return PnL equity history only for currently active models."""
        active_names = {ENSEMBLE_TRADER_NAME}
        return {name: snapshots for name, snapshots in self._pnl_history.items() if name in active_names}

    def get_stats(self, mode_filter: str | None = None) -> dict[str, Any]:
        uptime = (
            (datetime.now(timezone.utc) - self._start_time).total_seconds()
            if self._start_time else 0
        )
        now = datetime.now(timezone.utc)
        round_active = (
            self._last_round_started_at is not None
            and (
                self._last_round_finished_at is None
                or self._last_round_finished_at < self._last_round_started_at
            )
        )
        round_running_seconds = (
            int((now - self._last_round_started_at).total_seconds())
            if round_active and self._last_round_started_at else 0
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
        stage_label = stage_labels.get(self._current_stage)
        if stage_label is None and self._current_stage.startswith("analyze:"):
            stage_label = f"\u6b63\u5728\u5206\u6790 {self._current_stage.split(':', 1)[1]}"
        elif stage_label is None and self._current_stage.startswith("execute:"):
            stage_label = f"\u6b63\u5728\u6267\u884c {self._current_stage.split(':', 1)[1]} \u8ba2\u5355"
        elif stage_label is None:
            stage_label = self._current_stage

        # Filter decisions/execs by mode if requested
        is_paper_filter = None if mode_filter is None else (mode_filter == "paper")
        if is_paper_filter is not None:
            recent_decs = [d for d in self._recent_decisions if d.get("is_paper", True) == is_paper_filter]
            recent_execs = [e for e in self._recent_executions if e.get("is_paper", True) == is_paper_filter]
        else:
            recent_decs = self._recent_decisions
            recent_execs = self._recent_executions

        return {
            "running": self._running,
            "mode": mode_manager.mode.value,
            "paused": mode_manager.is_paused,
            "uptime_seconds": int(uptime),
            "decisions_total": self._decision_count,
            "trades_total": self._trade_count,
            "recent_decisions": recent_decs,
            "recent_executions": recent_execs,
            "current_stage": self._current_stage,
            "current_stage_label": stage_label,
            "round_active": round_active,
            "round_running_seconds": round_running_seconds,
            "last_round_started_at": (
                self._last_round_started_at.isoformat()
                if self._last_round_started_at else None
            ),
            "last_round_finished_at": (
                self._last_round_finished_at.isoformat()
                if self._last_round_finished_at else None
            ),
            "last_round_error": self._last_round_error,
            "live_model": ENSEMBLE_TRADER_NAME,
            "models": [ENSEMBLE_TRADER_NAME],
            "risk": self.risk_engine.circuit_breaker.get_state(),
            "decision_interval": settings.decision_interval_seconds,
        }
