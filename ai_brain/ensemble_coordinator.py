"""Multi-model decision coordinator.

Expert models produce initial reports. The coordinator runs requested
cross-checks, optionally asks the trend expert model for a deep consultation on
major conflicts, then emits the only executable DecisionOutput.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from ai_brain.base_model import Action, DecisionOutput
from ai_brain.cross_validator import CrossValidator
from ai_brain.model_registry import ModelRegistry
from config.settings import (
    DECISION_MAKER_NAME,
    ENSEMBLE_TRADER_NAME,
    FIXED_AI_MODEL_SLOTS,
    settings,
)
from core.safe_output import safe_error_text
from services.entry_signal_extraction import (
    enrich_signal_payload,
    signal_available,
    unwrap_tool_payload,
)
from services.entry_signal_extraction import (
    expected_return_pct as signal_expected_return_pct,
)
from services.entry_signal_extraction import (
    payload_side as signal_payload_side,
)
from services.model_dynamic_routing import plan_dynamic_model_route
from services.runtime_entry_filters import entry_filters_from_context
from services.trading_params import DEFAULT_TRADING_PARAMS

if TYPE_CHECKING:
    from data_feed.feature_vector import FeatureVector

logger = structlog.get_logger(__name__)


ACTION_SCORE = {
    Action.LONG: 1.0,
    Action.SHORT: -1.0,
    Action.CLOSE_SHORT: 0.75,
    Action.CLOSE_LONG: -0.75,
    Action.HOLD: 0.0,
}

ENSEMBLE_ENTRY_DECISION_PARAMS = DEFAULT_TRADING_PARAMS.ensemble_entry_decision
ENTRY_RISK_SIZING_PARAMS = DEFAULT_TRADING_PARAMS.entry_risk_sizing
ENSEMBLE_EXIT_DECISION_PARAMS = DEFAULT_TRADING_PARAMS.ensemble_exit_decision
ENSEMBLE_ML_PROBE_PARAMS = DEFAULT_TRADING_PARAMS.ensemble_ml_probe
NORMAL_ENTRY_SCORE_THRESHOLD = ENSEMBLE_ENTRY_DECISION_PARAMS.normal_entry_score_threshold
PROBE_ENTRY_SCORE_THRESHOLD = ENSEMBLE_ENTRY_DECISION_PARAMS.probe_entry_score_threshold
PROBE_ENTRY_ENABLED = ENSEMBLE_ENTRY_DECISION_PARAMS.probe_entry_enabled
MAX_ENTRY_DISAGREEMENT = ENSEMBLE_ENTRY_DECISION_PARAMS.max_entry_disagreement
MIN_EXECUTABLE_ENTRY_CONFIDENCE = ENSEMBLE_ENTRY_DECISION_PARAMS.min_executable_entry_confidence
DAILY_RECOVERY_ENTRY_SCORE_BONUS = ENSEMBLE_ENTRY_DECISION_PARAMS.daily_recovery_entry_score_bonus
DAILY_RECOVERY_MIN_ENTRY_CONFIDENCE = (
    ENSEMBLE_ENTRY_DECISION_PARAMS.daily_recovery_min_entry_confidence
)
DAILY_RECOVERY_MAX_ENTRY_SIZE = ENSEMBLE_ENTRY_DECISION_PARAMS.daily_recovery_max_entry_size
DAILY_RECOVERY_MAX_LEVERAGE = ENSEMBLE_ENTRY_DECISION_PARAMS.daily_recovery_max_leverage
PROFIT_QUALITY_EXPAND_MIN_ENTRY_SIZE = (
    ENTRY_RISK_SIZING_PARAMS.ensemble_profit_expand_min_entry_size
)
PROFIT_QUALITY_EXPAND_RECOVERY_MAX_SIZE = (
    ENTRY_RISK_SIZING_PARAMS.ensemble_profit_expand_recovery_max_size
)
PROFIT_QUALITY_EXPAND_SELECTIVE_MAX_SIZE = (
    ENTRY_RISK_SIZING_PARAMS.ensemble_profit_expand_selective_max_size
)
PROFIT_QUALITY_EXPAND_NORMAL_MAX_SIZE = (
    ENTRY_RISK_SIZING_PARAMS.ensemble_profit_expand_normal_max_size
)
PROFIT_QUALITY_EXPAND_CROWDED_MAX_SIZE = (
    ENTRY_RISK_SIZING_PARAMS.ensemble_profit_expand_crowded_max_size
)
MAX_NORMAL_ENTRY_SIZE = ENTRY_RISK_SIZING_PARAMS.ensemble_max_normal_entry_size
MAX_PROBE_ENTRY_SIZE = ENTRY_RISK_SIZING_PARAMS.ensemble_max_probe_entry_size
ML_SOFT_CAUTION_MAX_ENTRY_SIZE = ENTRY_RISK_SIZING_PARAMS.ensemble_ml_soft_caution_max_entry_size
MARKET_DIRECTION_EXCLUDED_EXPERTS = set(
    ENSEMBLE_ENTRY_DECISION_PARAMS.market_direction_excluded_experts
)
ENTRY_DIRECTION_SUPPORT_EXPERTS = set(
    ENSEMBLE_ENTRY_DECISION_PARAMS.entry_direction_support_experts
)
ENTRY_PROFIT_QUALITY_EXPERTS = set(ENSEMBLE_ENTRY_DECISION_PARAMS.entry_profit_quality_experts)
NO_POSITION_ENTRY_BASE_WEIGHTS = {
    "trend_expert": ENSEMBLE_ENTRY_DECISION_PARAMS.no_position_trend_expert_weight,
    "momentum_expert": ENSEMBLE_ENTRY_DECISION_PARAMS.no_position_momentum_expert_weight,
    "sentiment_expert": ENSEMBLE_ENTRY_DECISION_PARAMS.no_position_sentiment_expert_weight,
    "position_expert": ENSEMBLE_ENTRY_DECISION_PARAMS.no_position_position_expert_weight,
}
NO_POSITION_POSITION_EXPERT_WEIGHT_CAP = (
    ENSEMBLE_ENTRY_DECISION_PARAMS.no_position_position_expert_weight_cap
)
RISK_ENTRY_SCORE_DISCOUNT_MAX = ENSEMBLE_ENTRY_DECISION_PARAMS.risk_entry_score_discount_max
RISK_ENTRY_SIZE_MULTIPLIER_FLOOR = ENSEMBLE_ENTRY_DECISION_PARAMS.risk_entry_size_multiplier_floor
MIN_REVIEW_CLOSE_SUPPORT = ENSEMBLE_ENTRY_DECISION_PARAMS.min_review_close_support
FULL_CLOSE_SUPPORT = ENSEMBLE_ENTRY_DECISION_PARAMS.full_close_support
REVIEW_CLOSE_MIN_CONFIDENCE = ENSEMBLE_ENTRY_DECISION_PARAMS.review_close_min_confidence
REVIEW_CLOSE_STRONG_CONFIDENCE = ENSEMBLE_ENTRY_DECISION_PARAMS.review_close_strong_confidence
REVIEW_STRONG_OPPOSITE_SCORE = ENSEMBLE_ENTRY_DECISION_PARAMS.review_strong_opposite_score
PROFIT_PROTECT_REDUCE_PNL_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_protect_reduce_pnl_ratio
PROFIT_PROTECT_STRONG_PNL_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_protect_strong_pnl_ratio
PROFIT_PROTECT_FULL_PNL_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_protect_full_pnl_ratio
PROFIT_PROTECT_MIN_LOCK_USDT = ENSEMBLE_EXIT_DECISION_PARAMS.profit_protect_min_lock_usdt
PROFIT_PROTECT_STRONG_MIN_LOCK_USDT = (
    ENSEMBLE_EXIT_DECISION_PARAMS.profit_protect_strong_min_lock_usdt
)
PROFIT_PROTECT_FULL_MIN_LOCK_USDT = ENSEMBLE_EXIT_DECISION_PARAMS.profit_protect_full_min_lock_usdt
PROFIT_PROTECT_MODERATE_OPPOSITE_SCORE = (
    ENSEMBLE_EXIT_DECISION_PARAMS.profit_protect_moderate_opposite_score
)
PROFIT_PROTECT_REDUCE_SIZE = ENSEMBLE_EXIT_DECISION_PARAMS.profit_protect_reduce_size
PROFIT_EXIT_ANALYSIS_MIN_FLOOR_USDT = (
    ENSEMBLE_EXIT_DECISION_PARAMS.profit_exit_analysis_min_floor_usdt
)
MIN_DISCRETIONARY_CLOSE_HOLD_MINUTES = (
    ENSEMBLE_EXIT_DECISION_PARAMS.min_discretionary_close_hold_minutes
)
EARLY_CLOSE_MIN_RISK_USAGE = ENSEMBLE_EXIT_DECISION_PARAMS.early_close_min_risk_usage
LOSS_REDUCE_MIN_RISK_USAGE = ENSEMBLE_EXIT_DECISION_PARAMS.loss_reduce_min_risk_usage
LOSS_FULL_MIN_RISK_USAGE = ENSEMBLE_EXIT_DECISION_PARAMS.loss_full_min_risk_usage
FAST_PROFIT_MIN_HOLD_MINUTES = ENSEMBLE_EXIT_DECISION_PARAMS.fast_profit_min_hold_minutes
QUICK_PROFIT_REDUCE_PNL_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.quick_profit_reduce_pnl_ratio
CAPITAL_ROTATION_PROFIT_PNL_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.capital_rotation_profit_pnl_ratio
QUICK_PROFIT_FULL_PNL_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.quick_profit_full_pnl_ratio
PROFIT_LOCK_FEE_MULTIPLE = ENSEMBLE_EXIT_DECISION_PARAMS.profit_lock_fee_multiple
PROFIT_LOCK_NOTIONAL_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_lock_notional_ratio
PROFIT_LOCK_RISK_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_lock_risk_ratio
PROFIT_LOCK_MIN_FLOOR_USDT = ENSEMBLE_EXIT_DECISION_PARAMS.profit_lock_min_floor_usdt
PROFIT_LOCK_MAX_FLOOR_USDT = ENSEMBLE_EXIT_DECISION_PARAMS.profit_lock_max_floor_usdt
PROFIT_LOCK_MEANINGFUL_REDUCE_USDT = (
    ENSEMBLE_EXIT_DECISION_PARAMS.profit_lock_meaningful_reduce_usdt
)
PROFIT_LOCK_REDUCE_FEE_MULTIPLE = ENSEMBLE_EXIT_DECISION_PARAMS.profit_lock_reduce_fee_multiple
PROFIT_LOCK_REDUCE_NOTIONAL_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_lock_reduce_notional_ratio
PROFIT_LOCK_REDUCE_RISK_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_lock_reduce_risk_ratio
PORTFOLIO_FOCUS_LOCK_MIN_USDT = ENSEMBLE_EXIT_DECISION_PARAMS.portfolio_focus_lock_min_usdt
PORTFOLIO_FOCUS_LOCK_MIN_SHARE = ENSEMBLE_EXIT_DECISION_PARAMS.portfolio_focus_lock_min_share
PORTFOLIO_FOCUS_LOCK_REDUCE_SIZE = ENSEMBLE_EXIT_DECISION_PARAMS.portfolio_focus_lock_reduce_size
PROFIT_RETRACE_PEAK_LINE_MULTIPLE = ENSEMBLE_EXIT_DECISION_PARAMS.profit_retrace_peak_line_multiple
PROFIT_RETRACE_CURRENT_LINE_MULTIPLE = (
    ENSEMBLE_EXIT_DECISION_PARAMS.profit_retrace_current_line_multiple
)
PROFIT_RETRACE_BASE_REDUCE_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_retrace_base_reduce_ratio
PROFIT_RETRACE_BASE_FULL_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_retrace_base_full_ratio
PROFIT_RETRACE_MIN_REDUCE_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_retrace_min_reduce_ratio
PROFIT_RETRACE_MAX_REDUCE_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_retrace_max_reduce_ratio
PROFIT_RETRACE_MIN_FULL_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_retrace_min_full_ratio
PROFIT_RETRACE_MAX_FULL_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.profit_retrace_max_full_ratio
ADD_POSITION_MIN_SUPPORT = ENSEMBLE_ENTRY_DECISION_PARAMS.add_position_min_support
ADD_POSITION_MIN_CONFIDENCE = ENSEMBLE_ENTRY_DECISION_PARAMS.add_position_min_confidence
ADD_POSITION_STRONG_CONFIDENCE = ENSEMBLE_ENTRY_DECISION_PARAMS.add_position_strong_confidence
ADD_POSITION_SCORE_THRESHOLD = ENSEMBLE_ENTRY_DECISION_PARAMS.add_position_score_threshold
ADD_POSITION_MIN_PROFIT_RATIO = ENSEMBLE_ENTRY_DECISION_PARAMS.add_position_min_profit_ratio
ADD_POSITION_MAX_RISK_USAGE = ENSEMBLE_ENTRY_DECISION_PARAMS.add_position_max_risk_usage
ADD_POSITION_MIN_SIZE = ENSEMBLE_ENTRY_DECISION_PARAMS.add_position_min_size
ADD_POSITION_MAX_SIZE = ENSEMBLE_ENTRY_DECISION_PARAMS.add_position_max_size
WINNER_EXPAND_MIN_UNREALIZED_USDT = ENSEMBLE_ENTRY_DECISION_PARAMS.winner_expand_min_unrealized_usdt
WINNER_EXPAND_MIN_PROFIT_RATIO = ENSEMBLE_ENTRY_DECISION_PARAMS.winner_expand_min_profit_ratio
WINNER_EXPAND_SCORE_THRESHOLD = ENSEMBLE_ENTRY_DECISION_PARAMS.winner_expand_score_threshold
WINNER_EXPAND_MAX_RISK_USAGE = ENSEMBLE_ENTRY_DECISION_PARAMS.winner_expand_max_risk_usage
WINNER_RUN_MIN_PROFIT_RATIO = ENSEMBLE_ENTRY_DECISION_PARAMS.winner_run_min_profit_ratio
LOSS_COMPRESS_REDUCE_USDT = ENSEMBLE_EXIT_DECISION_PARAMS.loss_compress_reduce_usdt
LOSS_COMPRESS_FULL_USDT = ENSEMBLE_EXIT_DECISION_PARAMS.loss_compress_full_usdt
LOSS_COMPRESS_REDUCE_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.loss_compress_reduce_ratio
LOSS_COMPRESS_FULL_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.loss_compress_full_ratio
LOSS_COMPRESS_REDUCE_RISK_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.loss_compress_reduce_risk_ratio
LOSS_COMPRESS_FULL_RISK_RATIO = ENSEMBLE_EXIT_DECISION_PARAMS.loss_compress_full_risk_ratio
LOSS_REPAIR_MAX_LOSS_PROBABILITY = ENSEMBLE_EXIT_DECISION_PARAMS.loss_repair_max_loss_probability
LOSS_EXPAND_MIN_LOSS_PROBABILITY = ENSEMBLE_EXIT_DECISION_PARAMS.loss_expand_min_loss_probability
LOSS_EXPAND_FULL_LOSS_PROBABILITY = ENSEMBLE_EXIT_DECISION_PARAMS.loss_expand_full_loss_probability
LOSS_REPAIR_REDUCE_SUPPORT_COUNT = ENSEMBLE_EXIT_DECISION_PARAMS.loss_repair_reduce_support_count
LOSS_REPAIR_FULL_SUPPORT_COUNT = ENSEMBLE_EXIT_DECISION_PARAMS.loss_repair_full_support_count
PREDICTIVE_REVERSAL_REVIEW_SCORE = ENSEMBLE_EXIT_DECISION_PARAMS.predictive_reversal_review_score
PREDICTIVE_REVERSAL_EXIT_SCORE = ENSEMBLE_EXIT_DECISION_PARAMS.predictive_reversal_exit_score
PREDICTIVE_REVERSAL_FULL_EXIT_SCORE = (
    ENSEMBLE_EXIT_DECISION_PARAMS.predictive_reversal_full_exit_score
)
PREDICTIVE_REVERSAL_REDUCE_SIZE = ENSEMBLE_EXIT_DECISION_PARAMS.predictive_reversal_reduce_size
ML_MIN_EXPECTED_RETURN_PCT = ENSEMBLE_ML_PROBE_PARAMS.ml_min_expected_return_pct
ML_MIN_PROFIT_EDGE_PCT = ENSEMBLE_ML_PROBE_PARAMS.ml_min_profit_edge_pct
ML_MIN_SUPPORT_WIN_RATE = ENSEMBLE_ML_PROBE_PARAMS.ml_min_support_win_rate
ML_STRONG_SUPPORT_WIN_RATE = ENSEMBLE_ML_PROBE_PARAMS.ml_strong_support_win_rate
ML_SUPPORT_CONFIDENCE_BONUS = ENSEMBLE_ML_PROBE_PARAMS.ml_support_confidence_bonus
ML_LOW_EDGE_CONFIDENCE_BONUS = ENSEMBLE_ML_PROBE_PARAMS.ml_low_edge_confidence_bonus
ML_LOW_WIN_CONFIDENCE_BONUS = ENSEMBLE_ML_PROBE_PARAMS.ml_low_win_confidence_bonus
ML_PROFIT_FIRST_SCORE_RELIEF = ENSEMBLE_ML_PROBE_PARAMS.ml_profit_first_score_relief
ML_PROFIT_FIRST_MIN_EXPECTED_RETURN_PCT = (
    ENSEMBLE_ML_PROBE_PARAMS.ml_profit_first_min_expected_return_pct
)
ML_PROFIT_FIRST_MIN_EDGE_PCT = ENSEMBLE_ML_PROBE_PARAMS.ml_profit_first_min_edge_pct
ML_QUANT_ONLY_MIN_EXPECTED_RETURN_PCT = (
    ENSEMBLE_ML_PROBE_PARAMS.ml_quant_only_min_expected_return_pct
)
ML_QUANT_ONLY_MIN_EDGE_PCT = ENSEMBLE_ML_PROBE_PARAMS.ml_quant_only_min_edge_pct
ML_QUANT_ONLY_MAX_LOSS_PROBABILITY = ENSEMBLE_ML_PROBE_PARAMS.ml_quant_only_max_loss_probability
ML_PROFIT_FIRST_LOW_WIN_RATE_SIZE_MULTIPLIER = (
    ENSEMBLE_ML_PROBE_PARAMS.ml_profit_first_low_win_rate_size_multiplier
)
LOCAL_TOOLS_MAX_LOSS_PROBABILITY = ENSEMBLE_ML_PROBE_PARAMS.local_tools_max_loss_probability
PROFIT_FIRST_PROBE_CONFIDENCE = ENSEMBLE_ML_PROBE_PARAMS.profit_first_probe_confidence
PROFIT_FIRST_PROBE_SIZE = ENTRY_RISK_SIZING_PARAMS.ensemble_profit_first_probe_size
QUANT_VALIDATION_PROBE_CONFIDENCE = ENSEMBLE_ML_PROBE_PARAMS.quant_validation_probe_confidence
QUANT_VALIDATION_PROBE_SIZE = ENTRY_RISK_SIZING_PARAMS.ensemble_quant_validation_probe_size
QUANT_VALIDATION_MAX_LOSS_PROBABILITY = (
    ENSEMBLE_ML_PROBE_PARAMS.quant_validation_max_loss_probability
)
QUANT_VALIDATION_MIN_LOCAL_EXPECTED_RETURN_PCT = (
    ENSEMBLE_ML_PROBE_PARAMS.quant_validation_min_local_expected_return_pct
)
QUANT_VALIDATION_MIN_PROFIT_QUALITY_SCORE = (
    ENSEMBLE_ML_PROBE_PARAMS.quant_validation_min_profit_quality_score
)
QUANT_ONLY_SHORT_DIRECTION_MIN_GAP = ENSEMBLE_ML_PROBE_PARAMS.quant_only_short_direction_min_gap


class EnsembleCoordinator:
    """Combines fixed expert model reports into one executable decision."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._slot_meta = {slot["name"]: slot for slot in FIXED_AI_MODEL_SLOTS}
        self.cross_validator = CrossValidator()
        self._current_strategy_context: dict[str, Any] = {}

    def _set_runtime_entry_filters(
        self,
        features: FeatureVector,
        context: dict[str, Any] | None,
    ) -> None:
        self._current_strategy_context = dict(context or {})
        entry_filters = entry_filters_from_context(self._current_strategy_context)
        try:
            features.entry_filters = entry_filters.to_dict()
        except Exception:
            logger.debug("failed to attach runtime entry filters to features")

    async def decide(
        self,
        features: FeatureVector,
        context: dict[str, Any],
    ) -> tuple[DecisionOutput, dict[str, DecisionOutput]]:
        self._set_runtime_entry_filters(features, context)
        timing_records: list[dict[str, Any]] = []
        base_expert_context = self._base_expert_context(context)
        all_attempted: list[str] = []
        all_failures: list[dict[str, Any]] = []
        opinions, expert_context, expert_timing, model_timings = await self._run_expert_pass(
            features,
            base_expert_context,
            include_names=None,
            stage="expert_initial",
            label="专家初诊",
        )
        timing_records.append(expert_timing)
        all_attempted.extend(expert_context.get("_attempted_models", []))
        all_failures.extend(expert_context.get("_model_failures", []))

        # Market analysis must not let phantom close votes pollute
        # cross-validation. Some local models may still propose close_* even
        # when there is no matching position; normalize those before asking
        # other experts to validate the concern.
        opinions = self._guard_phantom_exit_opinions(features, context, opinions)

        # Step 2/3: requested cross-checks and trend-expert deep consultation.
        validation_timing: dict[str, Any] = {}
        cross_validations, consultation = await self.cross_validator.validate_all(
            opinions, validation_timing
        )
        if validation_timing.get("_cross_validation_timing"):
            timing_records.append(validation_timing["_cross_validation_timing"])
        if validation_timing.get("_consultation_timing"):
            timing_records.append(validation_timing["_consultation_timing"])

        # Step 4: deterministic ensemble decision.
        combine_started_at = datetime.now(UTC)
        combine_perf_started = time.perf_counter()
        final = self.combine(features, context, opinions, cross_validations, consultation)
        timing_records.append(
            {
                "stage": "ensemble_rules",
                "label": "规则汇总",
                "status": "completed",
                "started_at": combine_started_at.isoformat(),
                "duration_sec": round(time.perf_counter() - combine_perf_started, 3),
            }
        )

        # Step 5: optional final decision maker. It reads the committee output
        # and can confirm, reduce risk, or veto; it does not join initial voting.
        final = await self._apply_decision_maker(
            features,
            context,
            final,
            opinions,
            cross_validations,
            consultation,
        )
        final = self._market_exit_guard_hold(features, context, final, "final")
        raw = final.raw_response if isinstance(final.raw_response, dict) else {}
        decision_maker_timing = raw.get("decision_maker_timing")
        if isinstance(decision_maker_timing, dict):
            timing_records.append(decision_maker_timing)
        raw["attempted_experts"] = self._unique(all_attempted)
        raw["expert_failures"] = all_failures
        raw["model_timings"] = model_timings
        raw["timing_breakdown"] = timing_records
        raw["latency_summary"] = self._latency_summary(timing_records, model_timings)
        self._attach_dynamic_model_routing(features, context, raw)
        if isinstance(context.get("ml_signal"), dict):
            raw["ml_signal"] = context.get("ml_signal")
        if isinstance(context.get("local_ai_tools"), dict):
            raw["local_ai_tools"] = context.get("local_ai_tools")
        if isinstance(context.get("direction_competition"), dict):
            raw["direction_competition"] = context.get("direction_competition")
        news_items = getattr(features, "recent_news_items", None)
        if isinstance(news_items, list):
            raw["news_context"] = {
                "sentiment_available": bool(getattr(features, "sentiment_data_available", False)),
                "direct_sentiment_data_available": bool(
                    getattr(features, "direct_sentiment_data_available", False)
                ),
                "news_sentiment_avg": float(getattr(features, "news_sentiment_avg", 0.0) or 0.0),
                "social_sentiment_avg": float(
                    getattr(features, "social_sentiment_avg", 0.0) or 0.0
                ),
                "social_mention_count": int(getattr(features, "social_mention_count", 0) or 0),
                "news_article_count": int(getattr(features, "news_article_count", 0) or 0),
                "direct_news_item_count": int(getattr(features, "direct_news_item_count", 0) or 0),
                "market_news_item_count": int(getattr(features, "market_news_item_count", 0) or 0),
                "news_sources": list(getattr(features, "news_sources", []) or [])[:8],
                "items": [item for item in news_items[:12] if isinstance(item, dict)],
            }
        analysis_type = "position" if context.get("review_positions") else "market"
        raw["analysis_type"] = analysis_type
        raw["analysis_type_label"] = "持仓分析" if analysis_type == "position" else "市场分析"
        final.raw_response = raw
        return final, opinions

    def _base_expert_context(self, context: dict[str, Any]) -> dict[str, Any]:
        expert_context = dict(context)
        expert_context["position_model_name"] = ENSEMBLE_TRADER_NAME
        expert_context["expert_mode"] = True
        expert_context["_exclude_model_names"] = [DECISION_MAKER_NAME]
        return expert_context

    async def _run_expert_pass(
        self,
        features: FeatureVector,
        base_context: dict[str, Any],
        *,
        include_names: tuple[str, ...] | list[str] | None,
        stage: str,
        label: str,
    ) -> tuple[dict[str, DecisionOutput], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        expert_context = dict(base_context)
        if include_names:
            expert_context["_include_model_names"] = list(include_names)
        expert_started_at = datetime.now(UTC)
        expert_perf_started = time.perf_counter()
        opinions = await self.registry.decide_all(features, expert_context)
        expert_duration = round(time.perf_counter() - expert_perf_started, 3)
        model_timings = expert_context.get("_model_timings", [])
        timing = {
            "stage": stage,
            "label": label,
            "status": "completed",
            "started_at": expert_started_at.isoformat(),
            "duration_sec": expert_duration,
            "attempted": len(expert_context.get("_attempted_models", [])),
            "completed": len(opinions),
            "failed": len(expert_context.get("_model_failures", [])),
            "slowest_model": (
                max(model_timings, key=lambda row: float(row.get("duration_sec") or 0.0)).get(
                    "name"
                )
                if model_timings
                else None
            ),
        }
        return opinions, expert_context, timing, model_timings

    def _attach_dynamic_model_routing(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        raw: dict[str, Any],
    ) -> None:
        try:
            raw["dynamic_model_routing"] = plan_dynamic_model_route(
                features,
                context,
                model_health=self._safe_dict(context.get("model_expert_health")),
                competition=self._safe_dict(context.get("model_expert_competition")),
                feature_coverage=self._safe_dict(context.get("crypto_feature_coverage")),
            )
        except Exception as exc:
            raw["dynamic_model_routing"] = {
                "audit_only": True,
                "mode": "shadow_only",
                "applied_to_live_calls": False,
                "live_route_mutation": False,
                "can_apply_live_route": False,
                "blocking_reasons": ["routing_plan_error"],
                "error": safe_error_text(exc, limit=180),
            }

    def _unique(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value)
            if text and text not in seen:
                result.append(text)
                seen.add(text)
        return result

    async def _apply_decision_maker(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        preliminary: DecisionOutput,
        opinions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]],
        consultation: dict[str, Any] | None,
    ) -> DecisionOutput:
        raw = dict(preliminary.raw_response or {})
        dm_started_at = datetime.now(UTC)
        dm_perf_started = time.perf_counter()
        protected_exit = self._protected_exit_evidence(raw) if preliminary.is_exit else {}

        def set_decision_maker_timing(status: str, reason: str | None = None) -> None:
            duration = round(time.perf_counter() - dm_perf_started, 3)
            timing = {
                "stage": "decision_maker",
                "label": "最终交易员",
                "status": status,
                "started_at": dm_started_at.isoformat(),
                "duration_sec": duration,
            }
            if reason:
                timing["reason"] = reason[:240]
            decision_maker_payload = raw.get("decision_maker")
            if isinstance(decision_maker_payload, dict):
                decision_maker_payload["duration_sec"] = duration
            raw["decision_maker_timing"] = timing

        def apply_final_proposal(
            proposal: DecisionOutput, prefix: str = "最终交易员独立执行"
        ) -> DecisionOutput:
            proposal.model_name = ENSEMBLE_TRADER_NAME
            proposal.symbol = features.symbol
            proposal.raw_response = raw
            proposal.feature_snapshot = proposal.feature_snapshot or features.to_dict()
            proposal.suggested_leverage = min(
                max(float(proposal.suggested_leverage or 1.0), 1.0), settings.max_leverage
            )
            proposal.position_size_pct = min(
                max(float(proposal.position_size_pct or 0.0), 0.0), 1.0
            )
            if prefix and prefix not in str(proposal.reasoning or ""):
                proposal.reasoning = f"{prefix}：{proposal.reasoning}"
            return proposal

        def hold_to_entry_support(proposal: DecisionOutput) -> dict[str, Any]:
            """Preliminary hold cannot become a normal-size entry unless evidence supports it."""
            if not proposal.is_entry:
                return {"allow": False, "status": "not_entry"}
            side = "long" if proposal.action == Action.LONG else "short"
            quant_probe = self._safe_dict(raw.get("quant_only_probe_entry"))
            quant_validation = self._safe_dict(raw.get("quant_validation_probe_entry"))
            entry_support = self._safe_dict(raw.get("entry_signal_support"))
            ml_gate = self._safe_dict(raw.get("ml_profit_quality_gate"))
            local_gate = self._safe_dict(raw.get("local_ai_tools_gate"))
            if not local_gate and isinstance(ml_gate.get("local_ai_tools_gate"), dict):
                local_gate = self._safe_dict(ml_gate.get("local_ai_tools_gate"))

            same_direction_experts = [
                name
                for name, decision in opinions.items()
                if isinstance(decision, DecisionOutput)
                and decision.action == proposal.action
                and float(decision.confidence or 0.0) >= 0.58
            ]
            strong_direction_experts = [
                name
                for name, decision in opinions.items()
                if isinstance(decision, DecisionOutput)
                and decision.action == proposal.action
                and float(decision.confidence or 0.0) >= 0.74
            ]
            hard_opposition = [
                name
                for name, decision in opinions.items()
                if isinstance(decision, DecisionOutput)
                and decision.action.is_entry()
                and decision.action != proposal.action
                and float(decision.confidence or 0.0) >= 0.62
            ]

            quant_allowed = bool(quant_probe.get("allow") or quant_validation.get("allow"))
            entry_allowed = bool(
                entry_support.get("allowed")
                and str(entry_support.get("side") or "").lower() == side
            )
            local_supported = bool(
                self._context_profit_supports_action(context, proposal.action)
                or (
                    local_gate.get("enabled")
                    and local_gate.get("allow", True)
                    and str(local_gate.get("direction") or "").lower() == side
                    and float(local_gate.get("expected_return_pct") or 0.0) > 0
                )
            )
            ml_supported = bool(
                ml_gate.get("status") == "supported_by_profit_quality"
                and (local_supported or same_direction_experts)
            )
            expert_supported = bool(
                len(same_direction_experts) >= 2 or (strong_direction_experts and local_supported)
            )
            allow = bool(
                not hard_opposition
                and (quant_allowed or entry_allowed or ml_supported or expert_supported)
            )
            reasons = []
            if quant_allowed:
                reasons.append("quant_probe")
            if entry_allowed:
                reasons.append("entry_signal_support")
            if ml_supported:
                reasons.append("ml_profit_quality")
            if expert_supported:
                reasons.append("expert_support")
            return {
                "allow": allow,
                "status": "allowed_probe_entry" if allow else "blocked_preliminary_hold_entry",
                "side": side,
                "support_reasons": reasons,
                "same_direction_experts": same_direction_experts,
                "strong_direction_experts": strong_direction_experts,
                "hard_opposition": hard_opposition,
                "quant_allowed": quant_allowed,
                "entry_allowed": entry_allowed,
                "ml_supported": ml_supported,
                "local_supported": local_supported,
                "expert_supported": expert_supported,
                "max_position_size": QUANT_VALIDATION_PROBE_SIZE,
                "max_leverage": min(DAILY_RECOVERY_MAX_LEVERAGE, settings.max_leverage),
            }

        decision_maker = self.registry.get(DECISION_MAKER_NAME)
        if decision_maker is None:
            raw["decision_maker"] = {
                "status": "skipped",
                "model_name": DECISION_MAKER_NAME,
                "reason": "最终交易员未配置，本轮跳过。",
            }
            set_decision_maker_timing("skipped", "not_configured")
            preliminary.raw_response = raw
            return preliminary

        if not self._should_call_decision_maker(preliminary, opinions, cross_validations, context):
            raw["decision_maker"] = {
                "status": "skipped",
                "model_name": DECISION_MAKER_NAME,
                "reason": "本轮没有可执行信号或重大分歧，无需调用最终交易员。",
            }
            set_decision_maker_timing("skipped", "not_needed")
            preliminary.raw_response = raw
            return preliminary

        decider_context = dict(context)
        decider_context.update(
            {
                "decision_maker_mode": True,
                "expert_mode": False,
                "position_model_name": ENSEMBLE_TRADER_NAME,
                "preliminary_decision": preliminary.to_log_dict(),
                "expert_opinions": [
                    self._decision_payload(name, decision)
                    for name, decision in opinions.items()
                    if isinstance(decision, DecisionOutput)
                ],
                "cross_validations": cross_validations or [],
                "consultation": consultation,
                "conflict_resolution": raw.get("conflict_resolution") or {},
                "close_evidence": raw.get("close_evidence") or {},
                "position_review_policy": raw.get("position_review_policy") or {},
                "add_evidence": raw.get("add_evidence") or {},
                "opportunity_score": raw.get("opportunity_score") or {},
                "ml_profit_quality_gate": raw.get("ml_profit_quality_gate") or {},
                "local_ai_tools_gate": raw.get("local_ai_tools_gate") or {},
            }
        )

        try:
            try:
                configured_dm_timeout = float(settings.ai_decision_maker_timeout_seconds or 14.0)
            except (TypeError, ValueError):
                configured_dm_timeout = 14.0
            dm_timeout = min(max(configured_dm_timeout, 5.0), 18.0)
            proposal = await asyncio.wait_for(
                decision_maker.decide(features, decider_context),
                timeout=dm_timeout,
            )
        except TimeoutError:
            set_decision_maker_timing("timeout", "decision_maker_timeout")
            logger.warning("decision maker timed out", symbol=features.symbol)
            raw["decision_maker"] = {
                "status": "timeout",
                "model_name": DECISION_MAKER_NAME,
                "reason": f"最终交易员超过 {dm_timeout:.0f} 秒未返回，系统保留初步裁决。",
                "fallback": "开仓信号继续走下单前最新行情复核；平仓/减仓风控信号保留。",
            }
            preliminary.raw_response = raw
            return preliminary
        except Exception as e:
            error_text = safe_error_text(e, limit=240)
            set_decision_maker_timing("failed", error_text)
            logger.warning("decision maker failed", error=error_text, symbol=features.symbol)
            raw["decision_maker"] = {
                "status": "failed",
                "model_name": DECISION_MAKER_NAME,
                "reason": error_text,
                "fallback": "保留 deterministic ensemble 初步裁决。",
            }
            if preliminary.is_entry:
                raw["decision_maker"][
                    "fallback"
                ] = "新开仓信号改为观望；已有持仓的减仓/平仓风控信号保留。"
                return self._hold(
                    features,
                    "最终交易员调用失败，系统按保护规则禁止新增开仓；已有持仓仍继续复盘和风控处理。",
                    raw,
                )
            preliminary.raw_response = raw
            return preliminary

        if not isinstance(proposal, DecisionOutput):
            set_decision_maker_timing("invalid", "invalid_result")
            raw["decision_maker"] = {
                "status": "invalid",
                "model_name": DECISION_MAKER_NAME,
                "reason": "最终交易员没有返回标准决策结构，保留初步裁决。",
            }
            if preliminary.is_entry:
                raw["decision_maker"][
                    "reason"
                ] = "最终交易员没有返回标准决策结构；新开仓改为观望，平仓风控信号保留。"
                return self._hold(
                    features,
                    "最终交易员返回结果无效，系统按保护规则禁止新增开仓。",
                    raw,
                )
            preliminary.raw_response = raw
            return preliminary

        raw["decision_maker"] = {
            "status": "completed",
            "model_name": DECISION_MAKER_NAME,
            "action": proposal.action.value,
            "confidence": proposal.confidence,
            "reasoning": proposal.reasoning,
            "position_size_pct": proposal.position_size_pct,
            "suggested_leverage": proposal.suggested_leverage,
            "provider_model": (
                (proposal.raw_response or {}).get("provider_model")
                if isinstance(proposal.raw_response, dict)
                else None
            ),
        }
        set_decision_maker_timing("completed")

        if proposal.is_exit and context.get("review_positions"):
            close_evidence = self._safe_dict(raw.get("close_evidence"))
            position_loss = bool(close_evidence.get("position_loss"))
            position_profit = bool(close_evidence.get("position_profit"))
            profit_protection = bool(close_evidence.get("profit_protection"))
            hard_risk = bool(close_evidence.get("hard_risk") or close_evidence.get("raw_hard_risk"))
            risk_usage = self._safe_float(close_evidence.get("position_risk_usage"), 0.0)
            planned_size = self._safe_float(close_evidence.get("position_size_pct"), 0.0)
            proposed_size = self._safe_float(proposal.position_size_pct, 0.0)
            proposal_reason = str(proposal.reasoning or "")
            profit_words = ("锁盈", "锁定利润", "保护利润", "利润保护", "止盈", "盈利", "收益")

            if position_loss and any(word in proposal_reason for word in profit_words):
                raw["decision_maker"]["reasoning_conflict"] = {
                    "applied": True,
                    "position_loss": True,
                    "position_unrealized_pnl": close_evidence.get("position_unrealized_pnl"),
                    "proposal_reasoning": proposal.reasoning,
                    "reason": "最终交易员把亏损仓描述为锁盈/保护利润，系统改用规则层亏损退出理由。",
                }
                proposal.reasoning = (
                    close_evidence.get("reason")
                    or "当前持仓为亏损状态，平仓/减仓只能按止损、亏损压缩或风险降低处理，不能按锁定利润处理。"
                )

            if (
                position_loss
                and planned_size > 0
                and proposed_size > planned_size + 1e-9
                and not hard_risk
                and risk_usage < LOSS_REDUCE_MIN_RISK_USAGE
            ):
                raw["decision_maker"]["applied"] = False
                raw["decision_maker"]["guard_reason"] = (
                    "规则层只允许亏损修复减仓，且未达到硬风控/全平风险阈值；"
                    "最终交易员不能把减仓升级成全部平仓。"
                )
                raw["decision_maker_loss_size_guard"] = {
                    "applied": True,
                    "proposal_action": proposal.action.value,
                    "proposal_size_pct": proposed_size,
                    "rule_size_pct": planned_size,
                    "position_loss": position_loss,
                    "position_risk_usage": risk_usage,
                    "close_evidence": close_evidence,
                }
                preliminary.reasoning += (
                    " 最终交易员建议更大比例平仓，但该仓位仍属于亏损修复减仓场景，"
                    "系统保留规则层减仓比例，避免把普通亏损修复误执行为全平。"
                )
                preliminary.raw_response = raw
                return preliminary

            if profit_protection and not position_profit:
                raw["decision_maker"]["applied"] = False
                raw["decision_maker"]["guard_reason"] = (
                    "close_evidence 标记了利润保护，但真实持仓不是盈利状态；"
                    "本轮禁止按利润保护路径退出。"
                )
                raw["invalid_profit_protection_guard"] = {
                    "applied": True,
                    "close_evidence": close_evidence,
                }
                return self._hold(
                    features,
                    "利润保护证据与真实持仓盈亏矛盾：当前不是盈利仓，禁止按锁盈/利润保护执行。",
                    raw,
                )

        if proposal.is_exit and not self._exit_matches_open_position(
            proposal.action, features.symbol, context
        ):
            raw["decision_maker"]["applied"] = False
            raw["decision_maker"]["guard_reason"] = (
                "最终交易员提出平仓，但当前上下文没有该币种对应方向的本地/交易所持仓；"
                "禁止在市场分析中凭空生成平仓裁决。"
            )
            raw["phantom_exit_guard"] = {
                "applied": True,
                "proposal_action": proposal.action.value,
                "symbol": features.symbol,
                "open_positions": (
                    context.get("open_positions") if isinstance(context, dict) else []
                ),
                "reason": "没有匹配持仓，平仓建议被改为观望。",
            }
            return self._hold(
                features,
                (
                    f"最终交易员建议{self._action_label(proposal.action)}，"
                    "但系统没有找到该币种对应持仓；本轮改为观望，避免误报平仓或提交无效平仓单。"
                ),
                raw,
            )

        if proposal.is_exit and context.get("review_positions") and preliminary.is_hold:
            close_evidence = self._safe_dict(raw.get("close_evidence"))
            protected_or_hard_exit = bool(
                close_evidence.get("should_close")
                or close_evidence.get("hard_risk")
                or close_evidence.get("profit_protection")
            )
            if not protected_or_hard_exit:
                raw["decision_maker"]["applied"] = False
                raw["decision_maker"]["guard_reason"] = (
                    "持仓复盘规则层已经判定退出证据不足，最终交易员不能只凭普通风险描述把继续持有改成平仓；"
                    "需要硬止损/止盈、严重趋势失效、明确利润保护或 close_evidence.should_close=true。"
                )
                raw["decision_maker_exit_evidence_guard"] = {
                    "applied": True,
                    "proposal_action": proposal.action.value,
                    "proposal_reasoning": proposal.reasoning,
                    "close_evidence": close_evidence,
                }
                return self._hold(
                    features,
                    (
                        "最终交易员建议平仓，但持仓复盘证据层未达到退出门槛；"
                        f"{close_evidence.get('block_reason') or '缺少硬风控、趋势严重失效或足够净利润保护证据'}。"
                    ),
                    raw,
                )

        if proposal.is_hold:
            if protected_exit:
                raw["decision_maker"]["applied"] = False
                raw["decision_maker"]["guard_reason"] = (
                    "初步裁决属于盈利保护减仓/平仓或硬风控退出，最终交易员不能直接改成观望；"
                    "本轮保留规则层退出动作。"
                )
                preliminary.reasoning += (
                    f" 最终交易员建议观望，但该动作属于{self._protected_exit_label(protected_exit)}，"
                    "系统保留原退出计划。"
                )
                preliminary.raw_response = raw
                return preliminary
            if preliminary.is_entry and self._profit_first_entry_survives_decision_maker_hold(
                preliminary,
                raw,
                cross_validations,
            ):
                raw["decision_maker"]["applied"] = False
                raw["decision_maker"]["guard_reason"] = (
                    "最终交易员提出普通瑕疵观望，但初步裁决已通过 ML 盈亏质量、方向一致和规则层风控；"
                    "按盈利优先策略保留小仓试单，并将仓位限制为恢复/试单上限。"
                )
                raw["profit_first_decision_maker_hold_override"] = {
                    "applied": True,
                    "max_position_size": PROFIT_FIRST_PROBE_SIZE,
                    "max_leverage": DAILY_RECOVERY_MAX_LEVERAGE,
                    "decision_maker_hold_reason": proposal.reasoning,
                }
                return DecisionOutput(
                    model_name=ENSEMBLE_TRADER_NAME,
                    symbol=features.symbol,
                    action=preliminary.action,
                    confidence=max(
                        float(preliminary.confidence or 0.0), PROFIT_FIRST_PROBE_CONFIDENCE
                    ),
                    reasoning=(
                        "强盈利质量小仓保留：最终交易员建议观望，"
                        f"但该方向已通过 ML 盈亏质量和规则层风控；普通瑕疵仅降级仓位。决策者提示：{proposal.reasoning}"
                    ),
                    position_size_pct=min(
                        float(preliminary.position_size_pct or 0.0), PROFIT_FIRST_PROBE_SIZE
                    ),
                    suggested_leverage=min(
                        max(float(preliminary.suggested_leverage or 1.0), 1.0),
                        DAILY_RECOVERY_MAX_LEVERAGE,
                    ),
                    stop_loss_pct=preliminary.stop_loss_pct,
                    take_profit_pct=preliminary.take_profit_pct,
                    raw_response=raw,
                    feature_snapshot=preliminary.feature_snapshot or features.to_dict(),
                )
            raw["decision_maker"]["applied"] = True
            return self._hold(
                features,
                f"最终交易员否决执行，选择观望：{proposal.reasoning}",
                raw,
            )

        if preliminary.is_hold:
            if proposal.is_entry:
                if context.get("review_positions"):
                    add_evidence = self._safe_dict(raw.get("add_evidence"))
                    position_review_policy = self._safe_dict(raw.get("position_review_policy"))
                    review_result = str(position_review_policy.get("result") or "").lower()
                    add_plan = str(add_evidence.get("action_plan") or "").lower()
                    if (
                        add_evidence
                        and review_result in {"", "hold"}
                        and (not add_evidence.get("should_add") or add_plan == "hold")
                    ):
                        raw["decision_maker"]["applied"] = False
                        raw["decision_maker"]["guard_reason"] = (
                            "持仓复盘规则层已判定继续持有或暂不加仓，最终交易员不能只凭同方向探针建议"
                            "把本轮改成加仓；需要 add_evidence.should_add=true 才允许提交同向加仓。"
                        )
                        raw["decision_maker_position_add_guard"] = {
                            "applied": True,
                            "proposal_action": proposal.action.value,
                            "proposal_reasoning": proposal.reasoning,
                            "position_review_policy": position_review_policy,
                            "add_evidence": add_evidence,
                        }
                        return self._hold(
                            features,
                            (
                                "持仓复盘结论为继续持有或暂不加仓，未提交订单。"
                                f"加仓侧：{add_evidence.get('block_reason') or '加仓证据不足'}"
                            ),
                            raw,
                        )
                hold_entry_evidence = hold_to_entry_support(proposal)
                raw["decision_maker_hold_entry_guard"] = hold_entry_evidence
                if not hold_entry_evidence.get("allow"):
                    raw["decision_maker"]["applied"] = False
                    raw["decision_maker"]["guard_reason"] = (
                        "初步委员会为观望，最终交易员不能单独发起新开仓；"
                        "需要专家同向支持、ML/服务器盈利模型支持，或量化小仓探针证据。"
                    )
                    return self._hold(
                        features,
                        (
                            "初步委员会为观望，最终交易员虽建议开仓，但缺少足够专家/量化共识；"
                            "本轮不提交订单，避免弱证据被放大成实际亏损。"
                            f" 决策者理由：{proposal.reasoning}"
                        ),
                        raw,
                    )
                capped_size = min(
                    max(float(proposal.position_size_pct or 0.0), 0.015),
                    float(
                        hold_entry_evidence.get("max_position_size") or QUANT_VALIDATION_PROBE_SIZE
                    ),
                )
                capped_leverage = min(
                    max(float(proposal.suggested_leverage or 1.0), 1.0),
                    float(hold_entry_evidence.get("max_leverage") or DAILY_RECOVERY_MAX_LEVERAGE),
                )
                raw["decision_maker"]["applied"] = True
                raw["decision_maker"][
                    "guard_reason"
                ] = "初步委员会为观望，但存在专家/量化支持；允许小仓探针，不允许直接开成普通仓或大仓。"
                raw["decision_maker_hold_entry_size_cap"] = {
                    "applied": True,
                    "original_position_size_pct": proposal.position_size_pct,
                    "capped_position_size_pct": capped_size,
                    "original_leverage": proposal.suggested_leverage,
                    "capped_leverage": capped_leverage,
                    "reason": "从观望转开仓只允许小仓验证，强信号应先在专家委员会阶段形成开仓裁决。",
                }
                return DecisionOutput(
                    model_name=ENSEMBLE_TRADER_NAME,
                    symbol=features.symbol,
                    action=proposal.action,
                    confidence=min(
                        max(float(proposal.confidence or 0.0), QUANT_VALIDATION_PROBE_CONFIDENCE),
                        0.82,
                    ),
                    reasoning=(
                        "观望转小仓探针：初步委员会未形成普通开仓，但最终交易员方向与部分专家/量化证据一致；"
                        "系统按小仓封顶执行，后续用真实盈亏反馈模型。"
                        f" 决策者理由：{proposal.reasoning}"
                    ),
                    position_size_pct=capped_size,
                    suggested_leverage=capped_leverage,
                    stop_loss_pct=proposal.stop_loss_pct,
                    take_profit_pct=proposal.take_profit_pct,
                    raw_response=raw,
                    feature_snapshot=proposal.feature_snapshot
                    or preliminary.feature_snapshot
                    or features.to_dict(),
                )
            if proposal.is_exit and context.get("review_positions"):
                close_evidence = self._safe_dict(raw.get("close_evidence"))
                protected_or_hard_exit = bool(
                    close_evidence.get("should_close")
                    or close_evidence.get("hard_risk")
                    or close_evidence.get("profit_protection")
                )
                position_loss = bool(close_evidence.get("position_loss"))
                risk_usage = self._safe_float(close_evidence.get("position_risk_usage"), 0.0)
                if not protected_or_hard_exit or (
                    position_loss and risk_usage < LOSS_REDUCE_MIN_RISK_USAGE
                ):
                    raw["decision_maker"]["applied"] = False
                    raw["decision_maker"]["guard_reason"] = (
                        "持仓复盘证据不足，最终交易员不能单独把继续持有改成亏损平仓；"
                        "只有硬止损、严重趋势失效、明确利润保护或接近计划止损风险时才允许执行。"
                    )
                    raw["decision_maker_loss_exit_guard"] = {
                        "applied": True,
                        "proposal_action": proposal.action.value,
                        "proposal_reasoning": proposal.reasoning,
                        "position_loss": position_loss,
                        "position_risk_usage": risk_usage,
                        "close_evidence": close_evidence,
                    }
                    return self._hold(
                        features,
                        (
                            "最终交易员建议平仓，但持仓复盘证据不足；"
                            "当前浮亏未接近计划止损，也没有硬风控或严重趋势失效，继续持有。"
                        ),
                        raw,
                    )
            raw["decision_maker"]["applied"] = True
            raw["decision_maker"]["guard_reason"] = (
                "AI-led mode: final decision maker may open/close even when the preliminary ensemble was hold; "
                "hard safety remains in the risk/exchange layer."
            )
            return apply_final_proposal(proposal)

        if proposal.action != preliminary.action:
            if context.get("review_positions") and preliminary.is_entry and proposal.is_exit:
                close_evidence = self._safe_dict(raw.get("close_evidence"))
                add_evidence = self._safe_dict(raw.get("add_evidence"))
                protected_or_hard_exit = bool(
                    close_evidence.get("should_close")
                    or close_evidence.get("hard_risk")
                    or close_evidence.get("profit_protection")
                )
                if add_evidence.get("should_add") and not protected_or_hard_exit:
                    raw["decision_maker"]["applied"] = False
                    raw["decision_maker"]["guard_reason"] = (
                        "持仓复盘规则层已经给出顺势加仓证据，且退出证据层没有达到平仓门槛；"
                        "最终交易员不能只凭普通回调担心把盈利扩张改成平仓。"
                    )
                    raw["decision_maker_add_to_exit_guard"] = {
                        "applied": True,
                        "proposal_action": proposal.action.value,
                        "proposal_reasoning": proposal.reasoning,
                        "add_evidence": add_evidence,
                        "close_evidence": close_evidence,
                    }
                    preliminary.reasoning += (
                        " 最终交易员建议平仓，但加仓证据有效且未触发硬退出，"
                        "系统保留顺势加仓候选。"
                    )
                    preliminary.raw_response = raw
                    return preliminary
            if not protected_exit:
                raw["decision_maker"]["applied"] = True
                raw["decision_maker"]["guard_reason"] = (
                    "AI-led mode: final decision maker changed the side/action; "
                    "system will pass it to hard safety checks instead of forcing hold."
                )
                return apply_final_proposal(proposal)
            if protected_exit:
                raw["decision_maker"]["applied"] = False
                raw["decision_maker"]["guard_reason"] = (
                    "最终交易员给出的方向与规则层退出动作不一致；"
                    "由于该退出属于盈利保护或硬风控，系统保留原退出动作，不改为观望或反手。"
                )
                preliminary.reasoning += (
                    f" 最终交易员给出不同方向，但该动作属于{self._protected_exit_label(protected_exit)}，"
                    "系统保留原退出计划。"
                )
                preliminary.raw_response = raw
                return preliminary
            raw["decision_maker"]["applied"] = True
            raw["decision_maker"][
                "guard_reason"
            ] = "最终交易员与初步裁决方向不一致，按保守规则观望。"
            return self._hold(
                features,
                (
                    "最终交易员与初步裁决方向不一致，系统不反手、不猜方向，改为观望。"
                    f"决策者理由：{proposal.reasoning}"
                ),
                raw,
            )

        raw["decision_maker"]["applied"] = True
        confidence = min(
            max(
                (float(preliminary.confidence or 0.0) + float(proposal.confidence or 0.0)) / 2, 0.0
            ),
            0.95,
        )
        proposal_size = float(proposal.position_size_pct or 0.0)
        preliminary_size = float(preliminary.position_size_pct or 0.0)
        if preliminary.is_entry:
            position_size_pct = proposal_size if proposal_size > 0 else preliminary_size
            leverage = float(proposal.suggested_leverage or preliminary.suggested_leverage or 1.0)
        elif preliminary.is_exit:
            # Protected exits are deterministic risk/profit-taking actions.
            # The final decision maker can confirm the narrative, but should
            # not make the actual close smaller than the rule layer planned.
            if protected_exit:
                position_size_pct = preliminary_size
                raw["decision_maker"]["guard_reason"] = (
                    f"{self._protected_exit_label(protected_exit)}不允许被最终交易员降低平仓比例，"
                    "保留规则层的落袋/风控计划。"
                )
            else:
                position_size_pct = proposal_size if proposal_size > 0 else preliminary_size
            leverage = 1.0
        else:
            position_size_pct = 0.0
            leverage = 1.0

        if preliminary.is_entry:
            entry_gate = self._entry_execution_gate()
            min_confidence = max(
                MIN_EXECUTABLE_ENTRY_CONFIDENCE,
                float(entry_gate.get("min_confidence") or 0.0),
            )
            post_ml_gate = self._ml_profit_quality_entry_gate(
                preliminary.action,
                context,
                confidence,
                min_confidence,
            )
            raw["post_decision_maker_ml_profit_quality_gate"] = post_ml_gate
            if post_ml_gate.get("confidence_bonus"):
                confidence = min(
                    confidence + float(post_ml_gate.get("confidence_bonus") or 0.0), 0.95
                )
            if not post_ml_gate.get("allow", True):
                raw["decision_maker"]["applied"] = False
                raw["decision_maker"][
                    "guard_reason"
                ] = "最终确认后仍未通过 ML 盈亏质量过滤，禁止新开仓。"
                return self._hold(
                    features,
                    post_ml_gate.get("reason") or "ML 盈亏质量过滤未通过，本轮不开仓。",
                    raw,
                )

        return DecisionOutput(
            model_name=ENSEMBLE_TRADER_NAME,
            symbol=features.symbol,
            action=preliminary.action,
            confidence=confidence,
            reasoning=f"最终交易员确认执行：{proposal.reasoning}",
            position_size_pct=position_size_pct,
            suggested_leverage=max(leverage, 1.0),
            stop_loss_pct=preliminary.stop_loss_pct,
            take_profit_pct=preliminary.take_profit_pct,
            raw_response=raw,
            feature_snapshot=preliminary.feature_snapshot or features.to_dict(),
        )

    def _profit_first_entry_survives_decision_maker_hold(
        self,
        preliminary: DecisionOutput,
        raw: dict[str, Any],
        cross_validations: list[dict[str, Any]],
    ) -> bool:
        """Keep strong profit-first entries when the final maker only raises soft concerns."""
        if not preliminary.is_entry:
            return False
        if any(
            v.get("major_conflict") or v.get("consistency") == "divergent"
            for v in cross_validations or []
        ):
            return False

        side = "long" if preliminary.action == Action.LONG else "short"
        ml_gate = raw.get("ml_profit_quality_gate")
        if not isinstance(ml_gate, dict):
            ml_gate = raw.get("post_decision_maker_ml_profit_quality_gate")
        ml_hint = raw.get("ml_profit_first_direction_hint")
        if not isinstance(ml_gate, dict) or not isinstance(ml_hint, dict):
            return False
        if ml_gate.get("status") != "supported_by_profit_quality":
            return False
        if ml_hint.get("side") != side or not ml_hint.get("strong"):
            return False
        if (
            float(ml_gate.get("expected_return_pct") or 0.0)
            < ML_PROFIT_FIRST_MIN_EXPECTED_RETURN_PCT
        ):
            return False
        if float(ml_gate.get("profit_edge_pct") or 0.0) < ML_PROFIT_FIRST_MIN_EDGE_PCT:
            return False
        if float(preliminary.confidence or 0.0) < MIN_EXECUTABLE_ENTRY_CONFIDENCE:
            return False
        return True

    def _should_call_decision_maker(
        self,
        preliminary: DecisionOutput,
        opinions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> bool:
        if self._position_hold_fast_path_without_decision_maker(
            preliminary,
            opinions,
            cross_validations,
            context,
        ):
            return False
        if preliminary.is_entry:
            raw = preliminary.raw_response if isinstance(preliminary.raw_response, dict) else {}
            quant_probe = raw.get("quant_validation_probe_entry")
            if isinstance(quant_probe, dict) and quant_probe.get("allow"):
                raw["decision_maker_fast_path"] = {
                    "applied": False,
                    "reason": "量化验证只能作为辅助证据，仍需最终交易员复核，避免本地模型绕过专家共识直接开仓。",
                }
                preliminary.raw_response = raw
                return True
            if self._fast_path_entry_without_decision_maker(
                preliminary, opinions, cross_validations, raw
            ):
                raw["decision_maker_fast_path"] = {
                    "applied": True,
                    "reason": "专家方向一致且本地/服务器盈利质量支持，跳过最终交易员以缩短快照到下单时间。",
                }
                preliminary.raw_response = raw
                return False
            return True
        if preliminary.is_exit:
            raw = preliminary.raw_response if isinstance(preliminary.raw_response, dict) else {}
            protected_exit = self._protected_exit_evidence(raw)
            if protected_exit:
                raw["decision_maker_fast_path"] = {
                    "applied": True,
                    "reason": f"{self._protected_exit_label(protected_exit)}不等待最终交易员，优先缩短平仓执行延迟。",
                }
                preliminary.raw_response = raw
                return False
            return True
        if context.get("review_positions") and any(
            isinstance(d, DecisionOutput) and not d.is_hold for d in opinions.values()
        ):
            return True
        strong_actions = [
            d
            for d in opinions.values()
            if isinstance(d, DecisionOutput)
            and not d.is_hold
            and float(d.confidence or 0.0) >= 0.74
        ]
        if len(strong_actions) >= 2:
            return True
        if len(strong_actions) == 1 and self._context_profit_supports_action(
            context, strong_actions[0].action
        ):
            return True
        return any(
            isinstance(d, DecisionOutput) and not d.is_hold and float(d.confidence or 0.0) >= 0.82
            for d in opinions.values()
        )

    def _position_hold_fast_path_without_decision_maker(
        self,
        preliminary: DecisionOutput,
        opinions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> bool:
        if not context.get("review_positions") or not preliminary.is_hold:
            return False
        raw = preliminary.raw_response if isinstance(preliminary.raw_response, dict) else {}
        close_evidence = self._safe_dict(raw.get("close_evidence"))
        position_review_policy = self._safe_dict(raw.get("position_review_policy"))
        if not close_evidence and not position_review_policy:
            return False
        if str(position_review_policy.get("result") or "").lower() not in {"", "hold"}:
            return False
        if str(close_evidence.get("action_plan") or "").lower() not in {"", "hold"}:
            return False
        protected_or_hard_exit = bool(
            close_evidence.get("should_close")
            or close_evidence.get("hard_risk")
            or close_evidence.get("raw_hard_risk")
            or close_evidence.get("profit_protection")
            or close_evidence.get("profit_lock_ready_for_exit")
            or close_evidence.get("profit_floor_ready_for_exit")
            or close_evidence.get("predictive_exit")
            or close_evidence.get("quick_profit")
            or close_evidence.get("capital_rotation_profit")
            or close_evidence.get("profit_retrace_protection")
            or close_evidence.get("portfolio_focus_profit_lock")
        )
        if protected_or_hard_exit:
            return False
        if any(
            isinstance(item, dict) and (item.get("major_conflict") or item.get("needs_resolution"))
            for item in cross_validations
        ):
            return False
        non_hold = [
            decision
            for decision in opinions.values()
            if isinstance(decision, DecisionOutput) and not decision.is_hold
        ]
        if not non_hold:
            return False
        if any(decision.action in {Action.LONG, Action.SHORT} for decision in non_hold):
            return False
        raw["decision_maker_fast_path"] = {
            "applied": True,
            "reason": (
                "持仓规则层已判定继续持有，且退出证据不属于硬止损、止盈、严重趋势失效或利润保护；"
                "跳过最终交易员，避免普通平仓建议被复核后再被规则层拦回造成延迟。"
            ),
            "close_evidence": {
                "should_close": bool(close_evidence.get("should_close")),
                "action_plan": close_evidence.get("action_plan"),
                "support_count": close_evidence.get("support_count"),
                "strong_support_count": close_evidence.get("strong_support_count"),
                "block_reason": close_evidence.get("block_reason"),
            },
        }
        preliminary.raw_response = raw
        return True

    def _fast_path_entry_without_decision_maker(
        self,
        preliminary: DecisionOutput,
        opinions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]],
        raw: dict[str, Any],
    ) -> bool:
        if not preliminary.is_entry:
            return False
        if any(
            v.get("major_conflict") or v.get("consistency") == "divergent"
            for v in cross_validations or []
        ):
            return False
        direction = "long" if preliminary.action == Action.LONG else "short"
        same_direction = [
            d
            for d in opinions.values()
            if isinstance(d, DecisionOutput)
            and d.action == preliminary.action
            and float(d.confidence or 0.0) >= 0.58
        ]
        hard_opposition = [
            d
            for d in opinions.values()
            if isinstance(d, DecisionOutput)
            and d.action.is_entry()
            and d.action != preliminary.action
            and float(d.confidence or 0.0) >= 0.62
        ]
        if len(same_direction) < 2 or hard_opposition:
            return False
        if float(preliminary.confidence or 0.0) < max(MIN_EXECUTABLE_ENTRY_CONFIDENCE, 0.62):
            return False

        ml_gate = self._safe_dict(raw.get("ml_profit_quality_gate"))
        local_gate = self._safe_dict(raw.get("local_ai_tools_gate"))
        if not local_gate and isinstance(ml_gate.get("local_ai_tools_gate"), dict):
            local_gate = self._safe_dict(ml_gate.get("local_ai_tools_gate"))
        ml_supported = ml_gate.get("status") == "supported_by_profit_quality"
        local_supported = (
            local_gate.get("enabled")
            and local_gate.get("allow", True)
            and local_gate.get("direction") == direction
            and float(local_gate.get("expected_return_pct") or 0.0) > 0
            and str(local_gate.get("best_side") or direction).lower() in {direction, ""}
        )
        opportunity = self._safe_dict(raw.get("opportunity_score"))
        entry_evidence = self._safe_dict(raw.get("entry_candidate_evidence"))
        side_evidence = self._safe_dict(entry_evidence.get(direction))
        expected_net = max(
            self._safe_float(opportunity.get("expected_net_return_pct"), 0.0),
            self._safe_float(side_evidence.get("expected_net_return_pct"), 0.0),
        )
        profit_quality = max(
            self._safe_float(opportunity.get("profit_quality_ratio"), 0.0),
            self._safe_float(side_evidence.get("profit_quality_ratio"), 0.0),
        )
        loss_probability = self._safe_float(
            side_evidence.get(
                "loss_probability", opportunity.get("server_profit_loss_probability")
            ),
            1.0,
        )
        tail_risk = self._safe_float(
            opportunity.get("tail_risk_score", side_evidence.get("tail_risk_score")), 1.0
        )
        server_side = str(
            opportunity.get("server_profit_best_side")
            or side_evidence.get("server_profit_best_side")
            or ""
        ).lower()
        evidence_supported = (
            expected_net >= 0.75
            and profit_quality >= 0.80
            and loss_probability <= 0.55
            and tail_risk <= 0.75
            and server_side in {"", direction}
            and len(same_direction) >= 3
        )
        raw["decision_maker_fast_path_evidence"] = {
            "same_direction_count": len(same_direction),
            "expected_net_return_pct": round(expected_net, 6),
            "profit_quality_ratio": round(profit_quality, 6),
            "loss_probability": round(loss_probability, 6),
            "tail_risk_score": round(tail_risk, 6),
            "server_profit_best_side": server_side,
            "ml_supported": bool(ml_supported),
            "local_supported": bool(local_supported),
            "evidence_supported": bool(evidence_supported),
        }
        preliminary.raw_response = raw
        return bool(ml_supported or local_supported or evidence_supported)

    def _context_profit_supports_action(self, context: dict[str, Any], action: Action) -> bool:
        if action not in (Action.LONG, Action.SHORT):
            return False
        side = "long" if action == Action.LONG else "short"
        opposite = "short" if side == "long" else "long"

        profit = self._local_profit_signal(context)
        if profit and signal_available(profit):
            expected = self._local_expected_return(profit, side)
            opposite_expected = self._local_expected_return(profit, opposite)
            loss_probability = self._safe_float(profit.get(f"{side}_loss_probability"), 0.0)
            best_side = signal_payload_side(profit) or str(profit.get("best_side") or "").lower()
            if (
                expected > 0
                and expected >= opposite_expected
                and loss_probability < LOCAL_TOOLS_MAX_LOSS_PROBABILITY
                and best_side in {"", side}
            ):
                return True

        ml_signal = context.get("ml_signal") if isinstance(context, dict) else {}
        predictions = ml_signal.get("predictions") if isinstance(ml_signal, dict) else []
        primary = predictions[0] if isinstance(predictions, list) and predictions else {}
        if isinstance(primary, dict):
            expected = float(primary.get(f"{side}_expected_return_pct") or 0.0)
            opposite_expected = float(primary.get(f"{opposite}_expected_return_pct") or 0.0)
            best_side = str(primary.get("best_side") or "").lower()
            edge = expected - opposite_expected
            return (
                expected >= ML_PROFIT_FIRST_MIN_EXPECTED_RETURN_PCT
                and edge >= ML_PROFIT_FIRST_MIN_EDGE_PCT
                and best_side == side
            )
        return False

    def _protected_exit_evidence(self, raw: dict[str, Any]) -> dict[str, Any]:
        evidence = raw.get("close_evidence") if isinstance(raw.get("close_evidence"), dict) else {}
        if not evidence:
            return {}
        if evidence.get("hard_risk") or evidence.get("profit_protection"):
            return evidence
        return {}

    def _protected_exit_label(self, evidence: dict[str, Any]) -> str:
        if evidence.get("hard_risk"):
            return "硬风控退出"
        if evidence.get("profit_protection"):
            return "盈利保护退出"
        return "受保护退出"

    def _decision_payload(self, name: str, decision: DecisionOutput) -> dict[str, Any]:
        meta = self._slot_meta.get(name, {})
        return {
            "model_name": name,
            "label": meta.get("label", name),
            "role": meta.get("role", ""),
            "action": decision.action.value,
            "confidence": decision.confidence,
            "position_size_pct": decision.position_size_pct,
            "suggested_leverage": decision.suggested_leverage,
            "stop_loss_pct": decision.stop_loss_pct,
            "take_profit_pct": decision.take_profit_pct,
            "reasoning": " ".join(str(decision.reasoning or "").split())[:260],
            "cross_check_for": decision.cross_check_for,
        }

    def _guard_phantom_exit_opinions(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        opinions: dict[str, DecisionOutput],
    ) -> dict[str, DecisionOutput]:
        guarded: dict[str, DecisionOutput] = {}
        for name, decision in opinions.items():
            if (
                isinstance(decision, DecisionOutput)
                and decision.is_exit
                and not self._exit_matches_open_position(decision.action, features.symbol, context)
            ):
                raw = (
                    dict(decision.raw_response or {})
                    if isinstance(decision.raw_response, dict)
                    else {}
                )
                raw["phantom_exit_opinion_guard"] = {
                    "applied": True,
                    "original_action": decision.action.value,
                    "reason": "当前上下文没有该币种对应方向持仓，专家平仓意见改为观望。",
                }
                guarded[name] = DecisionOutput(
                    model_name=decision.model_name,
                    symbol=decision.symbol,
                    action=Action.HOLD,
                    confidence=min(float(decision.confidence or 0.0), 0.55),
                    reasoning=(
                        f"原建议{self._action_label(decision.action)}，但系统没有找到该币种对应持仓；"
                        "本专家意见按观望处理，避免无仓平仓误报。"
                    ),
                    position_size_pct=0.0,
                    suggested_leverage=1.0,
                    stop_loss_pct=decision.stop_loss_pct,
                    take_profit_pct=decision.take_profit_pct,
                    raw_response=raw,
                    feature_snapshot=decision.feature_snapshot,
                    cross_check_for=None,
                )
            else:
                guarded[name] = decision
        return guarded

    def _effective_expert_weight(
        self,
        *,
        name: str,
        base_weight: float,
        dynamic_multiplier: float,
        review_positions: bool,
        current_side: str | None,
        timeout_fallback: bool,
    ) -> tuple[float, dict[str, Any]]:
        """Apply role-aware weighting without letting risk act as a direction vote."""

        raw_weight = base_weight * dynamic_multiplier
        policy: dict[str, Any] = {
            "mode": "position_review" if review_positions else "market_entry",
            "base_weight": round(base_weight, 6),
            "dynamic_multiplier": round(dynamic_multiplier, 4),
            "raw_weight": round(raw_weight, 6),
            "entry_support_eligible": name not in MARKET_DIRECTION_EXCLUDED_EXPERTS,
            "excluded_reason": None,
        }
        if timeout_fallback:
            policy.update(
                {
                    "mode": "timeout_fallback",
                    "effective_weight": 0.0,
                    "excluded_reason": "timeout fallback opinion is trace-only",
                }
            )
            return 0.0, policy

        if review_positions:
            policy["effective_weight"] = round(raw_weight, 6)
            return raw_weight, policy

        if name == "risk_expert":
            policy.update(
                {
                    "mode": "entry_risk_veto_or_discount",
                    "effective_weight": 0.0,
                    "excluded_reason": (
                        "risk_expert is not a market direction vote; it only "
                        "hard-vetoes or discounts entry size/score"
                    ),
                }
            )
            return 0.0, policy

        if name == "position_expert":
            if current_side:
                policy.update(
                    {
                        "mode": "market_entry_position_trace_only",
                        "effective_weight": 0.0,
                        "excluded_reason": (
                            "position_expert is handled by position review when a "
                            "symbol position already exists"
                        ),
                    }
                )
                return 0.0, policy
            effective = min(
                NO_POSITION_ENTRY_BASE_WEIGHTS["position_expert"] * dynamic_multiplier,
                NO_POSITION_POSITION_EXPERT_WEIGHT_CAP,
            )
            policy.update(
                {
                    "mode": "no_position_entry_overlay",
                    "overlay_base_weight": NO_POSITION_ENTRY_BASE_WEIGHTS["position_expert"],
                    "effective_weight": round(effective, 6),
                    "excluded_reason": (
                        "position_expert keeps a tiny no-position context weight "
                        "but is not eligible as entry support"
                    ),
                }
            )
            return effective, policy

        if not current_side and name in NO_POSITION_ENTRY_BASE_WEIGHTS:
            effective = NO_POSITION_ENTRY_BASE_WEIGHTS[name] * dynamic_multiplier
            policy.update(
                {
                    "mode": "no_position_entry_overlay",
                    "overlay_base_weight": NO_POSITION_ENTRY_BASE_WEIGHTS[name],
                    "effective_weight": round(effective, 6),
                }
            )
            return effective, policy

        policy["effective_weight"] = round(raw_weight, 6)
        return raw_weight, policy

    def _risk_expert_entry_policy(
        self,
        risk_opinion: DecisionOutput | None,
        features: FeatureVector,
        *,
        review_positions: bool,
        current_side: str | None,
        hard_veto: bool,
    ) -> dict[str, Any]:
        if risk_opinion is None:
            return {"active": False, "reason": "risk_expert opinion missing"}
        if review_positions or current_side:
            return {
                "active": False,
                "reason": "risk_expert entry discount is only used for new-entry analysis",
                "hard_veto": bool(hard_veto),
            }
        if hard_veto:
            return {
                "active": True,
                "hard_veto": True,
                "score_discount_pct": 1.0,
                "size_multiplier": 0.0,
                "reason": "risk_expert hard-vetoed this new entry",
            }

        confidence = self._safe_float(risk_opinion.confidence, 0.0)
        discount = 0.0
        reasons: list[str] = []
        if risk_opinion.is_exit and confidence >= 0.55:
            discount += 0.12
            reasons.append("risk_expert emitted an exit-style caution during entry analysis")
        elif risk_opinion.action == Action.HOLD and confidence >= 0.55:
            hold_discount = 0.04
            if confidence >= 0.80:
                hold_discount = 0.10
            elif confidence >= 0.65:
                hold_discount = 0.07
            discount += hold_discount
            reasons.append("risk_expert preferred hold/caution")
        elif risk_opinion.is_entry:
            reasons.append(
                "risk_expert direction was ignored as a vote and treated as risk clearance"
            )

        reasoning = str(risk_opinion.reasoning or "").lower()
        caution_terms = (
            "slippage",
            "liquidity",
            "fake breakout",
            "tail risk",
            "abnormal volatility",
            "extreme volatility",
            "插针",
            "滑点",
            "流动性",
            "假突破",
            "极端波动",
            "异常波动",
        )
        if any(term in reasoning for term in caution_terms):
            discount += 0.04
            reasons.append("risk_expert reasoning contains market caution terms")

        volume_ratio = self._safe_float(getattr(features, "volume_ratio", 0.0), 0.0)
        spread_pct = self._safe_float(getattr(features, "spread_pct", 0.0), 0.0)
        volatility_20 = self._safe_float(getattr(features, "volatility_20", 0.0), 0.0)
        change_24h = abs(self._safe_float(getattr(features, "change_24h_pct", 0.0), 0.0))
        abnormal_wick_count = int(getattr(features, "abnormal_wick_count_72h", 0) or 0)
        abnormal_wick_max = self._safe_float(getattr(features, "abnormal_wick_max_pct", 0.0), 0.0)
        abnormal_wick_recent = self._safe_float(
            getattr(features, "abnormal_wick_recent_hours", 9999.0), 9999.0
        )
        entry_filters = entry_filters_from_context(getattr(self, "_current_strategy_context", {}))
        if volume_ratio > 0 and volume_ratio < max(entry_filters.min_entry_volume_ratio, 0.30):
            discount += 0.05
            reasons.append("low volume ratio")
        if spread_pct >= 0.003:
            discount += 0.05
            reasons.append("wide spread")
        if volatility_20 >= 0.08:
            discount += 0.08
            reasons.append("high short-term volatility")
        elif volatility_20 >= 0.06:
            discount += 0.04
            reasons.append("elevated short-term volatility")
        if change_24h >= 8.0 and volatility_20 >= 0.05:
            discount += 0.04
            reasons.append("large 24h move with volatility")
        if abnormal_wick_count > 0 and abnormal_wick_recent <= 12 and abnormal_wick_max >= 0.50:
            discount += 0.10
            reasons.append("recent abnormal wick")

        discount = min(max(discount, 0.0), RISK_ENTRY_SCORE_DISCOUNT_MAX)
        size_multiplier = max(RISK_ENTRY_SIZE_MULTIPLIER_FLOOR, 1.0 - discount * 1.5)
        return {
            "active": bool(discount > 0 or reasons),
            "hard_veto": False,
            "score_discount_pct": round(discount, 4),
            "size_multiplier": round(size_multiplier, 4),
            "reason": (
                "; ".join(reasons) if reasons else "risk_expert found no extra entry discount"
            ),
            "confidence": round(confidence, 4),
            "risk_action": risk_opinion.action.value,
        }

    def _expert_weight_policy_summary(self, opinions: list[dict[str, Any]]) -> dict[str, Any]:
        policies = {
            str(opinion.get("model_name")): opinion.get("weight_policy")
            for opinion in opinions
            if opinion.get("model_name") and isinstance(opinion.get("weight_policy"), dict)
        }
        modes = {
            str(policy.get("mode"))
            for policy in policies.values()
            if isinstance(policy, dict) and policy.get("mode")
        }
        return {
            "mode": (
                "no_position_entry_overlay"
                if "no_position_entry_overlay" in modes
                else "position_review" if "position_review" in modes else "market_entry"
            ),
            "policies": policies,
            "entry_support_excluded_experts": sorted(MARKET_DIRECTION_EXCLUDED_EXPERTS),
            "score_participants": [
                str(opinion.get("model_name"))
                for opinion in opinions
                if self._safe_float(opinion.get("effective_weight"), 0.0) > 0
            ],
        }

    def _risk_policy_from_opinions(self, opinions: list[dict[str, Any]]) -> dict[str, Any]:
        for opinion in opinions:
            if opinion.get("model_name") == "risk_expert" and isinstance(
                opinion.get("risk_expert_policy"), dict
            ):
                return dict(opinion["risk_expert_policy"])
        return {"active": False, "reason": "risk_expert opinion missing"}

    def _entry_signal_support_payload(
        self,
        *,
        action: Action,
        decisions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]],
        validation_adjustment: float,
        disagreement: float,
        context: dict[str, Any] | None,
        ml_profit_hint: dict[str, Any] | None,
        allowed: bool,
        reason: str,
    ) -> dict[str, Any]:
        action_side = "long" if action == Action.LONG else "short"
        same_direction = [
            name
            for name, decision in decisions.items()
            if isinstance(decision, DecisionOutput)
            and decision.action == action
            and float(decision.confidence or 0.0) >= 0.55
        ]
        return {
            "allowed": allowed,
            "side": action_side,
            "same_direction_experts": same_direction,
            "directional_support_experts": [
                name for name in same_direction if name not in MARKET_DIRECTION_EXCLUDED_EXPERTS
            ],
            "excluded_direction_experts": [
                name for name in same_direction if name in MARKET_DIRECTION_EXCLUDED_EXPERTS
            ],
            "technical_support": [
                name
                for name, decision in decisions.items()
                if name in ENTRY_DIRECTION_SUPPORT_EXPERTS
                and isinstance(decision, DecisionOutput)
                and decision.action == action
                and float(decision.confidence or 0.0) >= 0.55
            ],
            "aligned_validations": sum(
                1
                for validation in cross_validations
                if isinstance(validation, dict) and validation.get("consistency") == "aligned"
            ),
            "local_profit_aligned": self._local_profit_aligned(context, action_side),
            "ml_strong_aligned": bool(
                isinstance(ml_profit_hint, dict)
                and ml_profit_hint.get("strong")
                and ml_profit_hint.get("side") == action_side
            ),
            "validation_adjustment": round(validation_adjustment, 4),
            "disagreement": round(disagreement, 4),
            "policy": (
                "trend/sentiment plus profit-quality experts are executable entry support; "
                "position_expert and risk_expert are trace, veto, or discount signals"
            ),
            "reason": reason,
        }

    def combine(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        opinions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]] | None = None,
        consultation: dict[str, Any] | None = None,
    ) -> DecisionOutput:
        self._set_runtime_entry_filters(features, context)
        cross_validations = cross_validations or []
        valid = {name: d for name, d in opinions.items() if isinstance(d, DecisionOutput)}
        if not valid:
            return self._hold(features, "没有可用专家模型输出，保持观望。", {})

        review_positions = bool(context.get("review_positions"))
        symbol_positions = self._symbol_positions(features.symbol, context)
        current_position = self._safe_dict(symbol_positions[0]) if symbol_positions else {}
        current_side = current_position.get("side")

        weighted_score = 0.0
        total_weight = 0.0
        entry_size = 0.0
        leverage_votes: list[float] = []
        stop_votes: list[float] = []
        profit_votes: list[float] = []
        raw_opinions: list[dict[str, Any]] = []
        raw_opinions_by_name: dict[str, dict[str, Any]] = {}
        exit_votes: list[DecisionOutput] = []
        risk_vetoes: list[DecisionOutput] = []
        risk_opinion: DecisionOutput | None = None
        score_participants: dict[str, DecisionOutput] = {}
        dynamic_weights = context.get("dynamic_expert_weights") if isinstance(context, dict) else {}
        if not isinstance(dynamic_weights, dict):
            dynamic_weights = {}

        for name, decision in valid.items():
            meta = self._slot_meta.get(name, {})
            base_weight = float(
                meta.get("weight", getattr(self.registry.get(name), "weight", 1.0)) or 1.0
            )
            dynamic_info = self._safe_dict(dynamic_weights.get(name))
            dynamic_multiplier = min(
                max(self._safe_float(dynamic_info.get("multiplier"), 1.0), 0.70),
                1.25,
            )
            timeout_fallback = bool(
                isinstance(decision.raw_response, dict)
                and decision.raw_response.get("timeout_fallback")
            )
            effective_weight, weight_policy = self._effective_expert_weight(
                name=name,
                base_weight=base_weight,
                dynamic_multiplier=dynamic_multiplier,
                review_positions=review_positions,
                current_side=str(current_side) if current_side else None,
                timeout_fallback=timeout_fallback,
            )
            weight = base_weight * dynamic_multiplier
            score = ACTION_SCORE.get(decision.action, 0.0)
            weighted_score += effective_weight * decision.confidence * score
            total_weight += effective_weight
            if effective_weight > 0:
                score_participants[name] = decision

            if decision.is_entry and effective_weight > 0:
                entry_size += (
                    effective_weight * decision.position_size_pct * max(decision.confidence, 0.1)
                )
                leverage_votes.append(decision.suggested_leverage)
                stop_votes.append(decision.stop_loss_pct)
                profit_votes.append(decision.take_profit_pct)
            if decision.is_exit:
                exit_votes.append(decision)
            if name == "risk_expert":
                risk_opinion = decision
                if self._is_hard_risk_veto(
                    decision,
                    features,
                    for_position_close=bool(review_positions and current_side),
                ):
                    risk_vetoes.append(decision)

            raw_opinions.append(
                {
                    "model_name": name,
                    "role": meta.get("role", ""),
                    "label": meta.get("label", name),
                    "action": decision.action.value,
                    "confidence": decision.confidence,
                    "position_size_pct": decision.position_size_pct,
                    "suggested_leverage": decision.suggested_leverage,
                    "stop_loss_pct": decision.stop_loss_pct,
                    "take_profit_pct": decision.take_profit_pct,
                    "base_weight": base_weight,
                    "dynamic_weight_multiplier": dynamic_multiplier,
                    "dynamic_weight_reason": dynamic_info.get(
                        "reason", "暂无足够历史样本，使用基础权重。"
                    ),
                    "weight": weight,
                    "effective_weight": effective_weight,
                    "weight_policy": weight_policy,
                    "entry_support_eligible": weight_policy.get("entry_support_eligible"),
                    "excluded_reason": weight_policy.get("excluded_reason"),
                    "reasoning": decision.reasoning,
                    "cross_check_for": decision.cross_check_for,
                    "timeout_fallback": timeout_fallback,
                }
            )
            raw_opinions_by_name[name] = raw_opinions[-1]

        normalized_score = weighted_score / total_weight if total_weight else 0.0
        disagreement = self._disagreement((score_participants or valid).values())
        validation_adjustment = self._validation_adjustment(cross_validations, consultation)
        memory_adjustment = self._memory_adjustment(context, normalized_score)
        decision_score = self._score_after_validation(
            normalized_score,
            validation_adjustment + memory_adjustment,
        )
        risk_expert_policy = self._risk_expert_entry_policy(
            risk_opinion,
            features,
            review_positions=review_positions,
            current_side=str(current_side) if current_side else None,
            hard_veto=bool(risk_vetoes),
        )
        score_before_risk_discount = decision_score
        risk_score_discount = self._safe_float(risk_expert_policy.get("score_discount_pct"), 0.0)
        if risk_score_discount > 0 and not risk_expert_policy.get("hard_veto"):
            decision_score *= max(0.0, 1.0 - risk_score_discount)
        risk_expert_policy["score_before_discount"] = round(score_before_risk_discount, 4)
        risk_expert_policy["score_after_discount"] = round(decision_score, 4)
        if "risk_expert" in raw_opinions_by_name:
            raw_opinions_by_name["risk_expert"]["risk_expert_policy"] = risk_expert_policy
        major_conflicts = [v for v in cross_validations if v.get("major_conflict")]
        resolution_brief = self._conflict_resolution_brief(cross_validations, consultation)
        consultation_blocks_trade = (
            isinstance(consultation, dict) and consultation.get("should_trade") is False
        )
        raw = self._raw(raw_opinions, decision_score, disagreement, cross_validations, consultation)
        raw["base_weighted_score"] = round(normalized_score, 4)
        raw["memory_adjustment"] = round(memory_adjustment, 4)
        raw["memory_summary"] = self._memory_summary(context, normalized_score)
        raw["memory_feedback"] = self._memory_feedback(context)
        raw["market_regime"] = context.get("market_regime") or {}
        raw["strategy_mode"] = context.get("strategy_mode") or {}
        raw["ml_signal"] = context.get("ml_signal") or {}
        raw["portfolio_profit_protection"] = context.get("portfolio_profit_protection") or {}

        ml_profit_hint = self._ml_profit_first_direction_hint(context)
        raw["ml_profit_first_direction_hint"] = ml_profit_hint

        if memory_adjustment <= -0.18 and not current_side and abs(decision_score) < 0.65:
            reason = self._reason(
                "长期记忆提示该类场景历史亏损较多，本轮降低风险选择观望",
                decision_score,
                disagreement,
                raw_opinions,
                resolution_brief,
            )
            return self._hold(features, reason, raw)

        if risk_vetoes and not current_side:
            reason = self._reason(
                "风控专家否决新开仓", decision_score, disagreement, raw_opinions, resolution_brief
            )
            return self._hold(features, reason, raw)

        if major_conflicts and consultation_blocks_trade:
            reason = self._reason(
                "交叉验证发现重大矛盾，趋势专家深度会诊建议不交易",
                decision_score,
                disagreement,
                raw_opinions,
                resolution_brief,
            )
            return self._hold(features, reason, raw)

        close_decision = self._close_decision_if_needed(
            features,
            current_side,
            exit_votes,
            risk_vetoes,
            decision_score,
            disagreement,
            raw_opinions,
            cross_validations,
            consultation,
            resolution_brief,
            review_positions=review_positions,
            symbol_positions=symbol_positions,
            portfolio_profit_context=context.get("portfolio_profit_protection") or {},
            context=context,
        )
        if close_decision is not None:
            return close_decision

        if review_positions and current_side:
            add_decision = self._add_decision_if_needed(
                features,
                current_side,
                decision_score,
                disagreement,
                raw_opinions,
                cross_validations,
                consultation,
                resolution_brief,
                symbol_positions,
            )
            if add_decision is not None:
                return add_decision

            close_action = Action.CLOSE_LONG if current_side == "long" else Action.CLOSE_SHORT
            evidence = self._position_close_evidence(
                current_side,
                close_action,
                exit_votes,
                risk_vetoes,
                decision_score,
                raw_opinions,
                symbol_positions,
                features=features,
                context=context,
            )
            add_evidence = self._position_add_evidence(
                current_side,
                decision_score,
                raw_opinions,
                symbol_positions,
            )
            hold_raw = self._raw(
                raw_opinions, decision_score, disagreement, cross_validations, consultation
            )
            hold_raw["base_weighted_score"] = round(normalized_score, 4)
            hold_raw["memory_adjustment"] = round(memory_adjustment, 4)
            hold_raw["memory_summary"] = self._memory_summary(context, normalized_score)
            hold_raw["memory_feedback"] = self._memory_feedback(context)
            hold_raw["portfolio_profit_protection"] = (
                context.get("portfolio_profit_protection") or {}
            )
            hold_raw["position_review_policy"] = {
                "result": "hold",
                "minimum_support_required": MIN_REVIEW_CLOSE_SUPPORT,
                "full_close_support_required": FULL_CLOSE_SUPPORT,
                "add_support_required": ADD_POSITION_MIN_SUPPORT,
                "available_actions": ["hold", "add", "reduce", "full_close"],
                "note": "持仓复盘会在继续持有、加仓、减仓、平仓四类动作中选择；本轮未达到加仓或退出门槛。",
            }
            hold_raw["close_evidence"] = evidence
            hold_raw["add_evidence"] = add_evidence
            reason = self._reason(
                (
                    "持仓复盘结论为继续持有："
                    f"退出侧：{evidence.get('block_reason') or '平仓证据不足'}；"
                    f"加仓侧：{add_evidence.get('block_reason') or '加仓证据不足'}"
                ),
                decision_score,
                disagreement,
                raw_opinions,
                resolution_brief,
            )
            return self._hold(features, reason, hold_raw)

        if current_side and (disagreement >= 0.75 or major_conflicts):
            reason = self._reason(
                "已有仓位但专家分歧过大，暂不加仓",
                decision_score,
                disagreement,
                raw_opinions,
                resolution_brief,
            )
            return self._hold(features, reason, raw)

        action = Action.HOLD
        probe_entry = False
        profit_first_probe = False
        quant_validation_probe: dict[str, Any] | None = None
        recovery_probe_entry = False
        normal_entry_threshold = 0.28
        probe_entry_threshold = 0.16
        entry_gate = self._entry_execution_gate()
        raw["entry_execution_gate"] = entry_gate
        normal_entry_threshold += float(entry_gate.get("score_bonus") or 0.0)
        probe_entry_threshold += float(entry_gate.get("score_bonus") or 0.0)
        if (
            ml_profit_hint.get("strong")
            and (
                (decision_score > 0 and ml_profit_hint.get("side") == "long")
                or (decision_score < 0 and ml_profit_hint.get("side") == "short")
            )
            and self._local_profit_aligned(context, str(ml_profit_hint.get("side") or ""))
            and self._has_technical_support(
                Action.LONG if ml_profit_hint.get("side") == "long" else Action.SHORT,
                valid,
                min_confidence=0.60,
            )
        ):
            relief = min(
                float(ml_profit_hint.get("score_relief") or 0.0), ML_PROFIT_FIRST_SCORE_RELIEF
            )
            normal_entry_threshold = max(
                PROBE_ENTRY_SCORE_THRESHOLD, normal_entry_threshold - relief
            )
            probe_entry_threshold = max(0.18, probe_entry_threshold - relief * 0.5)
            raw["profit_first_threshold_relief"] = {
                "applied": True,
                "score_relief": round(relief, 4),
                "normal_entry_threshold": round(normal_entry_threshold, 4),
                "probe_entry_threshold": round(probe_entry_threshold, 4),
                "reason": "ML 盈亏期望强且与 AI 分数方向一致，按盈利优先降低入场分数门槛。",
            }
        if decision_score >= normal_entry_threshold and disagreement < 0.80:
            action = Action.LONG
        elif decision_score <= -normal_entry_threshold and disagreement < 0.80:
            action = Action.SHORT

        quant_probe_can_bypass_support = bool(
            isinstance(quant_validation_probe, dict)
            and quant_validation_probe.get("allow")
            and quant_validation_probe.get("status") in {"allowed", "quant_only_tiny_probe"}
        )
        if (
            action != Action.HOLD
            and not self._entry_signal_allowed(
                action,
                valid,
                cross_validations,
                validation_adjustment,
                disagreement,
                context,
            )
            and not quant_probe_can_bypass_support
        ):
            raw["entry_quality_gate"] = "entry_signal_support_block"
            raw["entry_signal_support"] = self._entry_signal_support_payload(
                action=action,
                decisions=valid,
                cross_validations=cross_validations,
                validation_adjustment=validation_adjustment,
                disagreement=disagreement,
                context=context,
                ml_profit_hint=ml_profit_hint,
                allowed=False,
                reason="Entry score passed, but executable cross-expert support was not strong enough.",
            )
            action = Action.HOLD

        if (
            PROBE_ENTRY_ENABLED
            and entry_gate.get("allow_probe", True)
            and action == Action.HOLD
            and not major_conflicts
            and disagreement < 0.50
        ):
            if decision_score >= probe_entry_threshold and self._probe_entry_allowed(
                Action.LONG, valid, cross_validations, validation_adjustment, features
            ):
                action = Action.LONG
                probe_entry = True
            elif decision_score <= -probe_entry_threshold and self._probe_entry_allowed(
                Action.SHORT, valid, cross_validations, validation_adjustment, features
            ):
                action = Action.SHORT
                probe_entry = True

        if action == Action.HOLD and self._profit_first_probe_allowed(
            ml_profit_hint,
            context,
            valid,
            cross_validations,
            validation_adjustment,
            disagreement,
            features,
        ):
            action = Action.LONG if ml_profit_hint.get("side") == "long" else Action.SHORT
            probe_entry = True
            profit_first_probe = True
            raw["profit_first_probe_entry"] = {
                "enabled": True,
                "action": action.value,
                "reason": (
                    "普通投票未达开仓线，但 ML 盈亏期望强、至少一个技术专家确认且无重大冲突，"
                    "按盈利优先允许小仓试单。"
                ),
            }

        if action == Action.HOLD:
            quant_validation_probe = self._quant_validation_probe_evidence(
                ml_profit_hint,
                context,
                valid,
                cross_validations,
                validation_adjustment,
                disagreement,
                features,
            )
            raw["quant_validation_probe_entry"] = quant_validation_probe
            if quant_validation_probe.get("allow"):
                action = (
                    Action.LONG if quant_validation_probe.get("side") == "long" else Action.SHORT
                )
                probe_entry = True

        if action == Action.HOLD and self._recovery_attack_probe_allowed(
            valid,
            cross_validations,
            validation_adjustment,
            disagreement,
            context,
            features,
            decision_score,
        ):
            action = Action.LONG if decision_score >= 0 else Action.SHORT
            probe_entry = True
            recovery_probe_entry = True
            raw["recovery_attack_probe_entry"] = {
                "enabled": True,
                "action": action.value,
                "reason": (
                    "Recovery attack mode allows a tiny probe when one technical expert is aligned "
                    "and risk_expert is cleared, without major conflict."
                ),
            }

        if action == Action.HOLD and not major_conflicts and disagreement < 0.50:
            quant_only_probe = self._quant_only_probe_evidence(
                ml_profit_hint,
                context,
                raw_opinions,
            )
            raw["quant_only_probe_entry"] = quant_only_probe
            if quant_only_probe.get("allow"):
                action = Action.LONG if quant_only_probe.get("side") == "long" else Action.SHORT
                probe_entry = True
                quant_validation_probe = quant_only_probe

        quant_probe_bypass_after_probe = bool(
            isinstance(quant_validation_probe, dict)
            and quant_validation_probe.get("allow")
            and quant_validation_probe.get("status") == "quant_only_tiny_probe"
        )
        if (
            action != Action.HOLD
            and not self._entry_signal_allowed(
                action,
                valid,
                cross_validations,
                validation_adjustment,
                disagreement,
                context,
            )
            and not quant_probe_bypass_after_probe
        ):
            action_side = "long" if action == Action.LONG else "short"
            raw["entry_quality_gate"] = "entry_signal_support_block_after_probe"
            raw["entry_signal_support"] = {
                "allowed": False,
                "side": action_side,
                "same_direction_experts": [
                    name
                    for name, decision in valid.items()
                    if isinstance(decision, DecisionOutput)
                    and decision.action == action
                    and float(decision.confidence or 0.0) >= 0.55
                ],
                "directional_support_experts": [
                    name
                    for name, decision in valid.items()
                    if name not in MARKET_DIRECTION_EXCLUDED_EXPERTS
                    and isinstance(decision, DecisionOutput)
                    and decision.action == action
                    and float(decision.confidence or 0.0) >= 0.55
                ],
                "excluded_direction_experts": [
                    name
                    for name, decision in valid.items()
                    if name in MARKET_DIRECTION_EXCLUDED_EXPERTS
                    and isinstance(decision, DecisionOutput)
                    and decision.action == action
                    and float(decision.confidence or 0.0) >= 0.55
                ],
                "technical_support": [
                    name
                    for name, decision in valid.items()
                    if name in ENTRY_DIRECTION_SUPPORT_EXPERTS
                    and isinstance(decision, DecisionOutput)
                    and decision.action == action
                    and float(decision.confidence or 0.0) >= 0.55
                ],
                "local_profit_aligned": self._local_profit_aligned(context, action_side),
                "ml_strong_aligned": bool(
                    ml_profit_hint.get("strong") and ml_profit_hint.get("side") == action_side
                ),
                "validation_adjustment": round(validation_adjustment, 4),
                "disagreement": round(disagreement, 4),
                "reason": "试单或量化验证生成了候选方向，但专家同向支持不足，本轮不允许开仓。",
            }
            action = Action.HOLD
            probe_entry = False
            profit_first_probe = False
            quant_validation_probe = None
            recovery_probe_entry = False

        if major_conflicts and not consultation_blocks_trade:
            decision_score *= 0.75
            if abs(decision_score) < normal_entry_threshold:
                action = Action.HOLD
                probe_entry = False
                profit_first_probe = False
                quant_validation_probe = None
                recovery_probe_entry = False

        regime_block_reason = self._market_regime_entry_block_reason(action, context)
        if regime_block_reason:
            reason = self._reason(
                regime_block_reason,
                decision_score,
                disagreement,
                raw_opinions,
                resolution_brief,
            )
            hold_raw = self._raw(
                raw_opinions, decision_score, disagreement, cross_validations, consultation
            )
            hold_raw["base_weighted_score"] = round(normalized_score, 4)
            hold_raw["entry_quality_gate"] = "market_regime_forecast_block"
            hold_raw["market_regime"] = context.get("market_regime") or {}
            hold_raw["memory_adjustment"] = round(memory_adjustment, 4)
            hold_raw["memory_summary"] = self._memory_summary(context, normalized_score)
            return self._hold(features, reason, hold_raw)

        if action == Action.HOLD:
            reason = self._reason(
                "综合分数不足或模型分歧较大，保持观望",
                decision_score,
                disagreement,
                raw_opinions,
                resolution_brief,
            )
            hold_raw = self._raw(
                raw_opinions, decision_score, disagreement, cross_validations, consultation
            )
            hold_raw["base_weighted_score"] = round(normalized_score, 4)
            hold_raw["memory_adjustment"] = round(memory_adjustment, 4)
            hold_raw["memory_summary"] = self._memory_summary(context, normalized_score)
            hold_raw["entry_execution_gate"] = entry_gate
            hold_raw["ml_profit_first_direction_hint"] = ml_profit_hint
            if raw.get("entry_quality_gate"):
                hold_raw["entry_quality_gate"] = raw.get("entry_quality_gate")
                hold_raw["entry_signal_support"] = raw.get("entry_signal_support")
            return self._hold(
                features,
                reason,
                hold_raw,
            )

        entry_mode = str(entry_gate.get("mode") or "")
        action_side = "long" if action == Action.LONG else "short"
        if entry_mode == "loss_recovery_selective" and not (
            ml_profit_hint.get("strong") and ml_profit_hint.get("side") == action_side
        ):
            reason = self._reason(
                "今日处于亏损恢复模式，普通开仓不再试错；只有 ML 预期收益强、方向一致的机会才允许小仓恢复。",
                decision_score,
                disagreement,
                raw_opinions,
                resolution_brief,
            )
            hold_raw = self._raw(
                raw_opinions, decision_score, disagreement, cross_validations, consultation
            )
            hold_raw["base_weighted_score"] = round(normalized_score, 4)
            hold_raw["entry_quality_gate"] = "loss_recovery_requires_profit_first_ml"
            hold_raw["entry_execution_gate"] = entry_gate
            hold_raw["ml_profit_first_direction_hint"] = ml_profit_hint
            hold_raw["memory_adjustment"] = round(memory_adjustment, 4)
            hold_raw["memory_summary"] = self._memory_summary(context, normalized_score)
            return self._hold(features, reason, hold_raw)

        if float(entry_gate.get("max_position_size") or 0.0) <= 0.0:
            reason = self._reason(
                "入场执行门禁已进入止血模式，停止新开仓，只允许已有持仓继续止盈止损和平仓",
                decision_score,
                disagreement,
                raw_opinions,
                resolution_brief,
            )
            hold_raw = self._raw(
                raw_opinions, decision_score, disagreement, cross_validations, consultation
            )
            hold_raw["base_weighted_score"] = round(normalized_score, 4)
            hold_raw["entry_quality_gate"] = "entry_execution_no_new_entry"
            hold_raw["entry_execution_gate"] = entry_gate
            hold_raw["memory_adjustment"] = round(memory_adjustment, 4)
            hold_raw["memory_summary"] = self._memory_summary(context, normalized_score)
            return self._hold(features, reason, hold_raw)

        avg_size = entry_size / total_weight if total_weight else 0.0
        if (
            quant_validation_probe
            and quant_validation_probe.get("status") == "quant_only_tiny_probe"
        ):
            avg_size = max(avg_size, QUANT_VALIDATION_PROBE_SIZE)
        confidence = min(max(abs(decision_score) + 0.35 - disagreement * 0.15, 0.0), 0.92)
        if probe_entry:
            confidence = max(confidence, MIN_EXECUTABLE_ENTRY_CONFIDENCE)
        if profit_first_probe:
            confidence = max(confidence, PROFIT_FIRST_PROBE_CONFIDENCE)
        if quant_validation_probe and quant_validation_probe.get("allow"):
            confidence = max(confidence, QUANT_VALIDATION_PROBE_CONFIDENCE)
        quality_points = self._entry_quality_points(features, action)
        min_quality_points = int(entry_gate.get("min_quality_points") or 0)
        if recovery_probe_entry:
            min_quality_points = min(min_quality_points, 1)
        elif profit_first_probe:
            min_quality_points = min(min_quality_points, 2)
        elif (
            min_quality_points > 2
            and validation_adjustment >= 0.20
            and abs(decision_score) >= 0.70
            and disagreement < 0.35
        ):
            min_quality_points = 2
            raw["validated_entry_quality_relief"] = {
                "applied": True,
                "min_quality_points": min_quality_points,
                "validation_adjustment": round(validation_adjustment, 4),
                "decision_score": round(decision_score, 4),
                "reason": "交叉验证已明显确认方向，高分机会按盈利优先允许小仓试单，不再要求 3/3 项质量全部满足。",
            }
        if quality_points < min_quality_points:
            reason = self._reason(
                f"入场质量门禁要求至少 {min_quality_points}/3 项通过，当前只有 {quality_points}/3，避免低质量试错",
                decision_score,
                disagreement,
                raw_opinions,
                resolution_brief,
            )
            hold_raw = self._raw(
                raw_opinions, decision_score, disagreement, cross_validations, consultation
            )
            hold_raw["base_weighted_score"] = round(normalized_score, 4)
            hold_raw["entry_quality_gate"] = "entry_quality_points_below_threshold"
            hold_raw["entry_execution_gate"] = entry_gate
            hold_raw["entry_quality_points"] = quality_points
            hold_raw["memory_adjustment"] = round(memory_adjustment, 4)
            hold_raw["memory_summary"] = self._memory_summary(context, normalized_score)
            return self._hold(features, reason, hold_raw)
        min_confidence = max(
            MIN_EXECUTABLE_ENTRY_CONFIDENCE,
            float(entry_gate.get("min_confidence") or 0.0),
        )
        ml_gate = self._ml_profit_quality_entry_gate(action, context, confidence, min_confidence)
        raw["ml_profit_quality_gate"] = ml_gate
        if ml_gate.get("confidence_bonus"):
            confidence = min(confidence + float(ml_gate.get("confidence_bonus") or 0.0), 0.94)
        if not ml_gate.get("allow", True):
            reversal_probe = self._quant_reversal_probe_evidence(
                action,
                ml_profit_hint,
                ml_gate,
                context,
                valid,
                cross_validations,
                validation_adjustment,
                disagreement,
                features,
            )
            raw["quant_reversal_probe_entry"] = reversal_probe
            if reversal_probe.get("allow"):
                action = Action.LONG if reversal_probe.get("side") == "long" else Action.SHORT
                action_side = "long" if action == Action.LONG else "short"
                probe_entry = True
                quant_validation_probe = reversal_probe
                confidence = max(QUANT_VALIDATION_PROBE_CONFIDENCE, min(confidence, 0.78))
                ml_gate = self._ml_profit_quality_entry_gate(
                    action, context, confidence, min_confidence
                )
                raw["ml_profit_quality_gate"] = ml_gate
            if not ml_gate.get("allow", True):
                reason = self._reason(
                    ml_gate.get("reason") or "ML 盈亏质量过滤未通过，本轮不开仓",
                    decision_score,
                    disagreement,
                    raw_opinions,
                    resolution_brief,
                )
                hold_raw = self._raw(
                    raw_opinions, decision_score, disagreement, cross_validations, consultation
                )
                hold_raw["base_weighted_score"] = round(normalized_score, 4)
                hold_raw["entry_quality_gate"] = "ml_profit_quality_block"
                hold_raw["ml_profit_quality_gate"] = ml_gate
                hold_raw["quant_reversal_probe_entry"] = reversal_probe
                hold_raw["ml_signal"] = context.get("ml_signal") or {}
                hold_raw["entry_execution_gate"] = entry_gate
                hold_raw["memory_adjustment"] = round(memory_adjustment, 4)
                hold_raw["memory_summary"] = self._memory_summary(context, normalized_score)
                return self._hold(features, reason, hold_raw)
        size = min(max(avg_size, 0.02), 1.0)
        if confidence < settings.confidence_threshold:
            size = max(size, 0.02)
        leverage = min(max(self._avg(leverage_votes, 3.0), 3.0), settings.max_leverage)
        if probe_entry:
            size = min(size, MAX_PROBE_ENTRY_SIZE)
        if profit_first_probe:
            size = min(size, PROFIT_FIRST_PROBE_SIZE)
        if recovery_probe_entry:
            size = min(size, ENTRY_RISK_SIZING_PARAMS.ensemble_recovery_probe_size_cap)
        leverage_cap = self._entry_leverage_cap(confidence, quality_points)
        min_leverage = self._entry_min_leverage(confidence, quality_points)
        leverage = min(max(leverage, min_leverage), leverage_cap, settings.max_leverage)
        memory_size_multiplier = self._memory_size_multiplier(context, action)
        size = max(min(size * memory_size_multiplier, 1.0), 0.0)
        size = min(size, 1.0)
        risk_size_multiplier = self._safe_float(risk_expert_policy.get("size_multiplier"), 1.0)
        risk_size_discount: dict[str, Any] | None = None
        profit_quality_expand = self._profit_quality_entry_expand_evidence(
            context=context,
            action_side=action_side,
            confidence=confidence,
            quality_points=quality_points,
            disagreement=disagreement,
            ml_gate=ml_gate,
            ml_profit_hint=ml_profit_hint,
            entry_gate=entry_gate,
            probe_entry=probe_entry,
            profit_first_probe=profit_first_probe,
            recovery_probe_entry=recovery_probe_entry,
            quant_validation_probe=quant_validation_probe,
        )
        if profit_quality_expand.get("allow"):
            original_size = size
            min_expand_size = float(profit_quality_expand.get("min_position_size") or 0.0)
            max_expand_size = float(profit_quality_expand.get("max_position_size") or size)
            size = min(max(size, min_expand_size), max_expand_size)
            profit_quality_expand["applied"] = size > original_size
            profit_quality_expand["original_position_size"] = round(original_size, 6)
            profit_quality_expand["position_size"] = round(size, 6)
        quant_validation_size_cap = None
        if quant_validation_probe and quant_validation_probe.get("allow"):
            size = min(max(size, 0.015), QUANT_VALIDATION_PROBE_SIZE)
            quant_validation_size_cap = {
                "applied": True,
                "max_position_size": QUANT_VALIDATION_PROBE_SIZE,
                "reason": "本地量化验证单只允许小仓，先收集真实执行结果，再用结果反向优化模型。",
            }
        ml_soft_caution_size_cap = None
        if ml_gate.get("status") == "passed_high_confidence_slight_negative_expectancy":
            size = min(size, ML_SOFT_CAUTION_MAX_ENTRY_SIZE)
            ml_soft_caution_size_cap = {
                "applied": True,
                "max_position_size": ML_SOFT_CAUTION_MAX_ENTRY_SIZE,
                "reason": "ML was slightly negative, so the AI signal is allowed but only as a small controlled entry.",
            }
        exposure = (
            ((context.get("strategy_mode") or {}).get("position_exposure") or {})
            if isinstance(context, dict)
            else {}
        )
        crowded_side_size_cap = None
        if isinstance(exposure, dict) and exposure.get("dominant_side") == action_side:
            crowded_cap = ENTRY_RISK_SIZING_PARAMS.ensemble_crowded_side_size_cap
            crowded_reason = "同方向敞口已经偏高，普通信号只允许小仓，避免单边堆仓。"
            if profit_quality_expand.get("allow"):
                crowded_cap = min(
                    float(
                        profit_quality_expand.get("max_position_size")
                        or PROFIT_QUALITY_EXPAND_CROWDED_MAX_SIZE
                    ),
                    PROFIT_QUALITY_EXPAND_CROWDED_MAX_SIZE,
                )
                crowded_reason = (
                    "同方向敞口偏高，但该信号通过盈利质量扩仓证据；"
                    "允许中等仓位跟随优质机会，仍限制最大仓位避免单边过度集中。"
                )
            size = min(size, crowded_cap)
            crowded_side_size_cap = {
                "applied": True,
                "max_position_size": round(crowded_cap, 6),
                "dominant_side": action_side,
                "profit_quality_expand_allowed": bool(profit_quality_expand.get("allow")),
                "readable_reason": crowded_reason,
                "reason": "同方向敞口已偏高，仅允许盈利质量强的小仓机会，避免普通加仓堆单。",
            }
        leverage = min(max(leverage, min_leverage), leverage_cap, settings.max_leverage)
        if 0.0 < risk_size_multiplier < 1.0:
            original_size = size
            size = max(min(size * risk_size_multiplier, 1.0), 0.0)
            risk_size_discount = {
                "applied": size < original_size,
                "original_position_size_pct": round(original_size, 6),
                "adjusted_position_size_pct": round(size, 6),
                "size_multiplier": round(risk_size_multiplier, 4),
                "reason": risk_expert_policy.get("reason"),
            }

        reason = self._reason(
            (
                f"单专家强信号通过交叉验证与风控过滤后形成{'做多' if action == Action.LONG else '做空'}小仓试单"
                if probe_entry
                else f"多专家加权并经过交叉验证后形成{'做多' if action == Action.LONG else '做空'}信号"
            ),
            decision_score,
            disagreement,
            raw_opinions,
            resolution_brief,
        )
        raw_response = self._raw(
            raw_opinions, decision_score, disagreement, cross_validations, consultation
        )
        raw_response["base_weighted_score"] = round(normalized_score, 4)
        raw_response["memory_adjustment"] = round(memory_adjustment, 4)
        raw_response["memory_size_multiplier"] = round(memory_size_multiplier, 4)
        raw_response["memory_summary"] = self._memory_summary(context, normalized_score)
        raw_response["memory_feedback"] = self._memory_feedback(context)
        raw_response["market_regime"] = context.get("market_regime") or {}
        raw_response["strategy_mode"] = context.get("strategy_mode") or {}
        raw_response["ml_signal"] = context.get("ml_signal") or {}
        raw_response["ml_profit_quality_gate"] = ml_gate
        raw_response["ml_profit_first_direction_hint"] = ml_profit_hint
        raw_response["profit_quality_position_boost"] = profit_quality_expand
        if risk_size_discount:
            raw_response["risk_expert_size_discount"] = risk_size_discount
        if ml_soft_caution_size_cap:
            raw_response["ml_soft_caution_size_cap"] = ml_soft_caution_size_cap
        if crowded_side_size_cap:
            raw_response["profit_first_crowded_side_size_cap"] = crowded_side_size_cap
        if quant_validation_size_cap:
            raw_response["quant_validation_size_cap"] = quant_validation_size_cap
        if raw.get("quant_reversal_probe_entry"):
            raw_response["quant_reversal_probe_entry"] = raw.get("quant_reversal_probe_entry")
        raw_response["entry_execution_gate"] = entry_gate
        raw_response["probe_entry"] = probe_entry
        raw_response["profit_first_probe_entry"] = profit_first_probe
        raw_response["quant_validation_probe_entry"] = quant_validation_probe or raw.get(
            "quant_validation_probe_entry"
        )
        raw_response["quant_only_probe_entry"] = raw.get("quant_only_probe_entry")
        raw_response["recovery_attack_probe_entry"] = recovery_probe_entry
        raw_response["entry_quality_points"] = quality_points
        raw_response["entry_signal_support"] = {
            "allowed": True,
            "side": action_side,
            "same_direction_experts": [
                name
                for name, decision in valid.items()
                if isinstance(decision, DecisionOutput)
                and decision.action == action
                and float(decision.confidence or 0.0) >= 0.55
            ],
            "directional_support_experts": [
                name
                for name, decision in valid.items()
                if name not in MARKET_DIRECTION_EXCLUDED_EXPERTS
                and isinstance(decision, DecisionOutput)
                and decision.action == action
                and float(decision.confidence or 0.0) >= 0.55
            ],
            "excluded_direction_experts": [
                name
                for name, decision in valid.items()
                if name in MARKET_DIRECTION_EXCLUDED_EXPERTS
                and isinstance(decision, DecisionOutput)
                and decision.action == action
                and float(decision.confidence or 0.0) >= 0.55
            ],
            "technical_support": [
                name
                for name, decision in valid.items()
                if name in ENTRY_DIRECTION_SUPPORT_EXPERTS
                and isinstance(decision, DecisionOutput)
                and decision.action == action
                and float(decision.confidence or 0.0) >= 0.55
            ],
            "local_profit_aligned": self._local_profit_aligned(context, action_side),
            "ml_strong_aligned": bool(
                ml_profit_hint.get("strong") and ml_profit_hint.get("side") == action_side
            ),
            "validation_adjustment": round(validation_adjustment, 4),
            "disagreement": round(disagreement, 4),
            "policy": "专家同向支持是开仓主条件；ML/服务器盈利模型只作为辅助加分和过滤。",
        }
        raw_response["min_leverage"] = min_leverage
        raw_response["leverage_cap"] = leverage_cap
        raw_response["leverage_policy"] = (
            f"满足 {quality_points}/3 项入场质量过滤，最低杠杆 {min_leverage:.1f}x，"
            f"杠杆上限 {leverage_cap:.1f}x，AI 最终选择 {leverage:.1f}x"
        )
        return DecisionOutput(
            model_name=ENSEMBLE_TRADER_NAME,
            symbol=features.symbol,
            action=action,
            confidence=confidence,
            reasoning=reason,
            position_size_pct=size,
            suggested_leverage=leverage,
            stop_loss_pct=(stop_loss := min(max(self._avg(stop_votes, 0.035), 0.015), 0.08)),
            take_profit_pct=min(max(self._avg(profit_votes, 0.08), stop_loss * 1.8, 0.03), 0.35),
            raw_response=raw_response,
            feature_snapshot=features.to_dict(),
        )

    def _ml_profit_first_direction_hint(self, context: dict[str, Any]) -> dict[str, Any]:
        """Extract a profit-first directional hint from ML without letting it trade alone."""
        ml_signal = context.get("ml_signal") if isinstance(context, dict) else {}
        if not isinstance(ml_signal, dict) or not ml_signal.get("available"):
            return {"available": False, "strong": False, "reason": "ML 不可用。"}
        if not ml_signal.get("influence_enabled", True):
            if ml_signal.get("advisory_enabled"):
                return {
                    "available": True,
                    "influence_enabled": False,
                    "advisory_enabled": True,
                    "strong": False,
                    "reason": "ML 当前为建议权重模式，只提供收益提示，不降低 AI 入场门槛。",
                    "influence_policy": ml_signal.get("influence_policy") or {},
                }
            return {
                "available": True,
                "influence_enabled": False,
                "strong": False,
                "reason": "ML 当前处于学习观察中，继续训练但不降低 AI 入场门槛。",
                "influence_policy": ml_signal.get("influence_policy") or {},
            }
        predictions = ml_signal.get("predictions")
        primary = predictions[0] if isinstance(predictions, list) and predictions else {}
        if not isinstance(primary, dict) or not primary:
            return {"available": False, "strong": False, "reason": "ML 没有可用预测。"}

        long_expected = self._safe_float(primary.get("long_expected_return_pct"), 0.0)
        short_expected = self._safe_float(primary.get("short_expected_return_pct"), 0.0)
        side = "long" if long_expected >= short_expected else "short"
        expected = long_expected if side == "long" else short_expected
        opposite_expected = short_expected if side == "long" else long_expected
        win_rate = self._safe_float(primary.get(f"{side}_win_rate"), 0.0)
        edge = expected - opposite_expected
        strong = (
            expected >= ML_PROFIT_FIRST_MIN_EXPECTED_RETURN_PCT
            and edge >= ML_PROFIT_FIRST_MIN_EDGE_PCT
            and str(primary.get("best_side") or side) == side
        )
        return {
            "available": True,
            "strong": bool(strong),
            "side": side,
            "horizon_minutes": primary.get("horizon_minutes"),
            "expected_return_pct": round(expected, 4),
            "opposite_expected_return_pct": round(opposite_expected, 4),
            "profit_edge_pct": round(edge, 4),
            "win_rate": round(win_rate, 4),
            "low_win_rate": bool(win_rate < ML_STRONG_SUPPORT_WIN_RATE),
            "score_relief": (
                ML_PROFIT_FIRST_SCORE_RELIEF
                * (
                    ML_PROFIT_FIRST_LOW_WIN_RATE_SIZE_MULTIPLIER
                    if win_rate < ML_STRONG_SUPPORT_WIN_RATE
                    else 1.0
                )
                if strong
                else 0.0
            ),
            "reason": (
                "ML 盈亏期望强，允许盈利优先小仓机会降低入场门槛。"
                if strong
                else "ML 盈亏期望不足以降低入场门槛。"
            ),
        }

    def _local_tool_signal(
        self,
        context: dict[str, Any] | None,
        name: str,
        *aliases: str,
    ) -> dict[str, Any]:
        tools = context.get("local_ai_tools") if isinstance(context, dict) else {}
        if not isinstance(tools, dict):
            return {}
        for key in (name, *aliases):
            payload = unwrap_tool_payload(tools.get(key))
            if payload:
                return enrich_signal_payload(name, payload)
        return {}

    def _local_profit_signal(self, context: dict[str, Any] | None) -> dict[str, Any]:
        return self._local_tool_signal(
            context,
            "profit_prediction",
            "profit_model",
            "server_profit",
            "server_profit_model",
            "profit",
        )

    def _local_timeseries_signal(self, context: dict[str, Any] | None) -> dict[str, Any]:
        return self._local_tool_signal(
            context,
            "time_series_prediction",
            "timeseries_prediction",
            "sequence_prediction",
            "timeseries",
            "time_series",
        )

    def _local_sentiment_signal(self, context: dict[str, Any] | None) -> dict[str, Any]:
        return self._local_tool_signal(
            context,
            "sentiment_analysis",
            "sentiment_prediction",
            "sentiment_model",
            "sentiment",
        )

    def _local_expected_return(self, payload: dict[str, Any], side: str) -> float:
        if not isinstance(payload, dict):
            return 0.0
        return self._safe_float(signal_expected_return_pct(payload, side), 0.0)

    def _quant_only_probe_evidence(
        self,
        ml_profit_hint: dict[str, Any],
        context: dict[str, Any],
        raw_opinions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Allow a tiny candidate when quant tools agree but LLM experts stay neutral."""
        profit = self._local_profit_signal(context)
        time_series = self._local_timeseries_signal(context)
        sentiment = self._local_sentiment_signal(context)
        directional_expert_votes = [
            o
            for o in raw_opinions
            if isinstance(o, dict) and str(o.get("action") or "") in {"long", "short"}
        ]
        if directional_expert_votes:
            return {"allow": False, "status": "experts_not_neutral"}

        ml_hint = self._safe_dict(ml_profit_hint)
        ml_side = str(ml_hint.get("side") or "").lower()
        ml_expected = self._safe_float(ml_hint.get("expected_return_pct"), 0.0)
        ml_edge = self._safe_float(ml_hint.get("profit_edge_pct"), 0.0)
        ml_strong = bool(ml_hint.get("strong"))

        best_profit_side = str(profit.get("best_side") or "").lower()
        ts_side = signal_payload_side(time_series)
        if not ts_side:
            ts_direction = str(time_series.get("direction") or "").lower()
            ts_side = "long" if ts_direction == "up" else "short" if ts_direction == "down" else ""
        sentiment_side = signal_payload_side(sentiment)
        sentiment_risk = str(sentiment.get("risk_level") or "").lower()
        direction_competition = self._safe_dict(
            context.get("direction_competition") if isinstance(context, dict) else {}
        )
        direction_preferred_side = str(direction_competition.get("preferred_side") or "").lower()
        direction_gap = self._safe_float(direction_competition.get("score_gap"), 0.0)

        candidates: list[dict[str, Any]] = []
        for side in ("long", "short"):
            opposite = "short" if side == "long" else "long"
            local_expected = self._local_expected_return(profit, side)
            opposite_expected = self._local_expected_return(profit, opposite)
            loss_probability = self._safe_float(profit.get(f"{side}_loss_probability"), 0.50)
            supports: list[str] = []
            support_detail: dict[str, Any] = {}

            profit_support = (
                best_profit_side == side
                and local_expected >= QUANT_VALIDATION_MIN_LOCAL_EXPECTED_RETURN_PCT
                and local_expected >= opposite_expected
                and loss_probability <= QUANT_VALIDATION_MAX_LOSS_PROBABILITY
            )
            if profit_support:
                supports.append("server_profit_model")
            support_detail["server_profit_model"] = {
                "best_side": best_profit_side,
                "expected_return_pct": round(local_expected, 4),
                "opposite_expected_return_pct": round(opposite_expected, 4),
                "loss_probability": round(loss_probability, 4),
                "support": bool(profit_support),
            }

            ts_expected = self._safe_float(
                time_series.get("expected_return_pct", time_series.get("expected_move_pct")),
                0.0,
            )
            ts_confidence = self._safe_float(time_series.get("confidence"), 0.0)
            ts_support = ts_side == side and abs(ts_expected) >= 0.003 and ts_confidence >= 0.004
            if ts_support:
                supports.append("time_series_model")
            support_detail["time_series_model"] = {
                "side": ts_side,
                "expected_return_pct": round(ts_expected, 4),
                "confidence": round(ts_confidence, 4),
                "support": bool(ts_support),
            }

            sentiment_expected = self._safe_float(
                sentiment.get(
                    "expected_return_pct", sentiment.get("expected_return_from_sentiment_pct")
                ),
                0.0,
            )
            sentiment_support = (
                sentiment_side == side
                and sentiment_expected >= 0.30
                and sentiment_risk not in {"high", "critical", "extreme"}
            )
            if sentiment_support:
                supports.append("sentiment_model")
            support_detail["sentiment_model"] = {
                "side": sentiment_side,
                "expected_return_pct": round(sentiment_expected, 4),
                "risk_level": sentiment_risk,
                "support": bool(sentiment_support),
            }

            ml_support = (
                ml_strong
                and ml_side == side
                and ml_expected >= ML_QUANT_ONLY_MIN_EXPECTED_RETURN_PCT
                and ml_edge >= ML_QUANT_ONLY_MIN_EDGE_PCT
            )
            if ml_support:
                supports.append("local_ml_shadow")
            support_detail["local_ml_shadow"] = {
                "side": ml_side,
                "expected_return_pct": round(ml_expected, 4),
                "profit_edge_pct": round(ml_edge, 4),
                "strong": bool(ml_strong),
                "support": bool(ml_support),
            }

            hard_negative = (
                local_expected < -0.20
                or loss_probability >= ML_QUANT_ONLY_MAX_LOSS_PROBABILITY
                or (
                    best_profit_side in {"long", "short"}
                    and best_profit_side != side
                    and local_expected <= 0
                )
            )
            candidates.append(
                {
                    "side": side,
                    "supports": supports,
                    "support_count": len(supports),
                    "support_detail": support_detail,
                    "local_expected_return_pct": local_expected,
                    "loss_probability": loss_probability,
                    "hard_negative": hard_negative,
                    "has_profit_support": profit_support,
                    "score": len(supports) + max(local_expected, 0.0) * 0.05,
                }
            )

        candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        best = candidates[0] if candidates else {}
        side = str(best.get("side") or "")
        supports = list(best.get("supports") or [])
        if not best or side not in {"long", "short"}:
            return {"allow": False, "status": "no_quant_side"}
        if best.get("hard_negative"):
            return {
                "allow": False,
                "status": "local_quant_disagrees",
                "side": side,
                "candidates": candidates,
            }
        if not best.get("has_profit_support"):
            return {
                "allow": False,
                "status": "no_server_profit_support",
                "side": side,
                "supports": supports,
                "candidates": candidates,
            }
        if len(supports) < 2:
            return {
                "allow": False,
                "status": "not_enough_quant_consensus",
                "side": side,
                "supports": supports,
                "candidates": candidates,
            }
        direction_support = bool(
            direction_preferred_side == side and direction_gap >= QUANT_ONLY_SHORT_DIRECTION_MIN_GAP
        )
        if side == "short" and (not direction_support or "time_series_model" not in supports):
            return {
                "allow": False,
                "status": "short_probe_needs_direction_competition",
                "side": side,
                "supports": supports,
                "direction_preferred_side": direction_preferred_side,
                "direction_gap": round(direction_gap, 4),
                "candidates": candidates,
                "reason": (
                    "Short quant-only probes need server profit, time-series support, and "
                    "long-vs-short direction competition aligned before execution."
                ),
            }
        return {
            "allow": True,
            "status": "quant_only_tiny_probe",
            "side": side,
            "supports": supports,
            "support_count": len(supports),
            "direction_preferred_side": direction_preferred_side,
            "direction_gap": round(direction_gap, 4),
            "local_expected_return_pct": round(
                self._safe_float(best.get("local_expected_return_pct"), 0.0), 4
            ),
            "loss_probability": round(self._safe_float(best.get("loss_probability"), 0.0), 4),
            "support_detail": best.get("support_detail") or {},
            "candidates": candidates,
            "reason": (
                "AI 专家全观望，但服务器盈利模型与至少一个本地量化工具同向，"
                "允许极小仓进入候选排序，由机会评分、价格偏移和高风险复核继续筛选。"
            ),
        }

    def _ml_profit_quality_entry_gate(
        self,
        action: Action,
        context: dict[str, Any],
        confidence: float,
        min_confidence: float,
    ) -> dict[str, Any]:
        """Use local ML as a profit-quality filter, never as the direction source."""
        if action not in (Action.LONG, Action.SHORT):
            return {"enabled": False, "allow": True, "status": "not_entry"}

        ml_signal = context.get("ml_signal") if isinstance(context, dict) else {}
        if not isinstance(ml_signal, dict) or not ml_signal.get("available"):
            return {
                "enabled": False,
                "allow": True,
                "status": "unavailable",
                "reason": "本地 ML 暂无可用模型或预测，本轮不影响 AI 决策。",
            }
        if not ml_signal.get("influence_enabled", True):
            if ml_signal.get("advisory_enabled"):
                advisory_policy = ml_signal.get("influence_policy") or {}
            else:
                advisory_policy = {}
            if not advisory_policy:
                return {
                    "enabled": False,
                    "allow": True,
                    "status": "learning_only",
                    "reason": "本地 ML 当前评估未达标，自动降级为学习观察；继续训练，但本轮不影响 AI 决策。",
                    "influence_policy": ml_signal.get("influence_policy") or {},
                }
            predictions = ml_signal.get("predictions")
            primary = predictions[0] if isinstance(predictions, list) and predictions else {}
            if not isinstance(primary, dict) or not primary:
                return {
                    "enabled": False,
                    "allow": True,
                    "status": "advisory_no_prediction",
                    "reason": "本地 ML 处于建议权重模式，但没有返回可用预测，本轮不影响 AI 决策。",
                }
            direction = "long" if action == Action.LONG else "short"
            side_policy = (
                advisory_policy.get(direction) if isinstance(advisory_policy, dict) else {}
            )
            return {
                "enabled": False,
                "allow": True,
                "status": f"{direction}_advisory",
                "direction": direction,
                "expected_return_pct": round(
                    self._safe_float(primary.get(f"{direction}_expected_return_pct"), 0.0),
                    4,
                ),
                "advisory_weight": round(
                    self._safe_float((side_policy or {}).get("influence_weight"), 0.0),
                    4,
                ),
                "reason": "本地 ML 样本成熟度不足，仅按建议权重参与收益解释；不作为开仓硬拦截。",
                "influence_policy": advisory_policy,
            }

        predictions = ml_signal.get("predictions")
        primary = predictions[0] if isinstance(predictions, list) and predictions else {}
        if not isinstance(primary, dict) or not primary:
            return {
                "enabled": False,
                "allow": True,
                "status": "no_prediction",
                "reason": "本地 ML 没有返回可用预测，本轮不影响 AI 决策。",
            }

        direction = "long" if action == Action.LONG else "short"
        side_policy = (
            (ml_signal.get("influence_policy") or {}).get(direction)
            if isinstance(ml_signal.get("influence_policy"), dict)
            else {}
        )
        if isinstance(side_policy, dict) and side_policy and not side_policy.get("enabled", True):
            return {
                "enabled": False,
                "allow": True,
                "status": f"{direction}_learning_only",
                "direction": direction,
                "reason": (
                    f"本地 ML 的{'做多' if direction == 'long' else '做空'}侧评估未达标，"
                    "该方向仅继续学习，不参与开仓过滤或加分。"
                ),
                "side_policy": side_policy,
            }
        opposite = "short" if direction == "long" else "long"
        expected = self._safe_float(primary.get(f"{direction}_expected_return_pct"), 0.0)
        opposite_expected = self._safe_float(primary.get(f"{opposite}_expected_return_pct"), 0.0)
        win_rate = self._safe_float(primary.get(f"{direction}_win_rate"), 0.0)
        best_side = str(primary.get("best_side") or "")
        edge = expected - opposite_expected

        gate = {
            "enabled": True,
            "allow": True,
            "status": "passed",
            "direction": direction,
            "horizon_minutes": primary.get("horizon_minutes"),
            "expected_return_pct": round(expected, 4),
            "opposite_expected_return_pct": round(opposite_expected, 4),
            "profit_edge_pct": round(edge, 4),
            "direction_win_rate": round(win_rate, 4),
            "best_side": best_side,
            "confidence_before_ml": round(confidence, 4),
            "min_confidence_before_ml": round(min_confidence, 4),
            "confidence_bonus": 0.0,
        }

        local_tools_gate = self._local_ai_tools_profit_gate(direction, context)
        gate["local_ai_tools_gate"] = local_tools_gate
        if local_tools_gate.get("enabled") and not local_tools_gate.get("allow", True):
            hard_local_block = (
                local_tools_gate.get("status") == "blocked_high_loss_probability"
                and float(local_tools_gate.get("loss_probability") or 0.0) >= 0.70
                and confidence < 0.82
            )
            if hard_local_block:
                gate.update(
                    {
                        "allow": False,
                        "status": local_tools_gate.get("status") or "blocked_local_quant_tools",
                        "reason": local_tools_gate.get("reason")
                        or "Local trained quant tools blocked this entry.",
                    }
                )
                return gate
            gate.update(
                {
                    "status": f"soft_{local_tools_gate.get('status') or 'local_quant_caution'}",
                    "local_quant_caution": True,
                    "confidence_bonus": min(float(gate.get("confidence_bonus") or 0.0), 0.0),
                    "reason": (
                        "服务器盈利模型提示该方向质量偏弱，已降级为软风控："
                        "不再一票否决强 AI/交叉验证信号，后续通过机会评分、仓位缩小和高风险复核控制风险。"
                    ),
                }
            )

        if expected < 0:
            required = min_confidence + 0.06
            if expected > -ML_MIN_EXPECTED_RETURN_PCT and confidence >= required:
                gate.update(
                    {
                        "status": "passed_high_confidence_slight_negative_expectancy",
                        "required_confidence": round(required, 4),
                        "reason": (
                            "ML expected return is slightly negative, but AI confidence is strong enough. "
                            "Allow a small controlled entry instead of hard-blocking the signal."
                        ),
                    }
                )
                return gate
            gate.update(
                {
                    "allow": False,
                    "status": "blocked_negative_expectancy",
                    "reason": (
                        f"ML 预测 AI 方向（{'做多' if direction == 'long' else '做空'}）预期盈亏为负 "
                        f"{expected:.4f}%，本轮不执行该方向开仓。"
                    ),
                }
            )
            return gate

        if best_side in {"long", "short"} and best_side != direction and edge <= 0:
            required = min_confidence + 0.10
            if confidence < required:
                gate.update(
                    {
                        "allow": False,
                        "status": "blocked_ml_direction_conflict",
                        "required_confidence": round(required, 4),
                        "reason": (
                            f"ML 盈亏期望更偏向{'做多' if best_side == 'long' else '做空'}，"
                            f"而 AI 给出{'做多' if direction == 'long' else '做空'}；"
                            f"当前置信度 {confidence:.2f} 低于冲突场景门槛 {required:.2f}，本轮观望。"
                        ),
                    }
                )
                return gate
            gate.update(
                {
                    "status": "passed_high_confidence_direction_conflict",
                    "required_confidence": round(required, 4),
                    "reason": "ML 与 AI 方向冲突，但 AI 置信度足够高，仅记录风险提示。",
                }
            )
            return gate

        if expected < ML_MIN_EXPECTED_RETURN_PCT:
            required = min_confidence + ML_LOW_EDGE_CONFIDENCE_BONUS
            if confidence < required:
                gate.update(
                    {
                        "allow": False,
                        "status": "blocked_low_expected_return",
                        "required_confidence": round(required, 4),
                        "reason": (
                            f"ML 预测该方向预期收益 {expected:.4f}% 低于最低盈利质量门槛 "
                            f"{ML_MIN_EXPECTED_RETURN_PCT:.4f}%，且置信度不足以覆盖手续费/滑点风险。"
                        ),
                    }
                )
                return gate
            gate.update(
                {
                    "status": "passed_high_confidence_low_expected_return",
                    "required_confidence": round(required, 4),
                    "reason": "ML 预期收益偏低，但 AI 置信度足够高，仅小心放行。",
                }
            )
            return gate

        if edge < ML_MIN_PROFIT_EDGE_PCT:
            required = min_confidence + ML_LOW_EDGE_CONFIDENCE_BONUS
            if confidence < required:
                gate.update(
                    {
                        "allow": False,
                        "status": "blocked_weak_profit_edge",
                        "required_confidence": round(required, 4),
                        "reason": (
                            f"ML 多空预期收益差只有 {edge:.4f}%，优势不明显；"
                            f"当前置信度 {confidence:.2f} 低于弱边际门槛 {required:.2f}。"
                        ),
                    }
                )
                return gate
            gate.update(
                {
                    "status": "passed_high_confidence_weak_profit_edge",
                    "required_confidence": round(required, 4),
                    "reason": "ML 收益差距不明显，但 AI 置信度足够高，仅小心放行。",
                }
            )
            return gate

        if win_rate < ML_MIN_SUPPORT_WIN_RATE:
            required = min_confidence + ML_LOW_WIN_CONFIDENCE_BONUS
            if confidence < required:
                gate.update(
                    {
                        "allow": False,
                        "status": "blocked_low_ml_win_rate",
                        "required_confidence": round(required, 4),
                        "reason": (
                            f"ML 预期收益为正但该方向胜率仅 {win_rate:.2%}，"
                            f"当前置信度 {confidence:.2f} 低于低胜率补偿门槛 {required:.2f}。"
                        ),
                    }
                )
                return gate
            gate.update(
                {
                    "status": "passed_high_confidence_low_win_rate",
                    "required_confidence": round(required, 4),
                    "reason": "ML 预期收益为正但胜率偏低，AI 置信度足够高，仅小心放行。",
                }
            )
            return gate

        if (
            expected >= ML_MIN_EXPECTED_RETURN_PCT
            and edge >= ML_MIN_PROFIT_EDGE_PCT
            and win_rate >= ML_STRONG_SUPPORT_WIN_RATE
            and best_side == direction
        ):
            gate.update(
                {
                    "status": "supported_by_profit_quality",
                    "confidence_bonus": ML_SUPPORT_CONFIDENCE_BONUS,
                    "reason": "ML 盈亏质量与 AI 方向一致，给予小幅确认加成；仍需通过其他风控。",
                }
            )
            return gate

        gate["reason"] = "ML 盈亏质量没有否决该方向，但也不足以明显加分。"
        return gate

    def _local_ai_tools_profit_gate(
        self, direction: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        tools = context.get("local_ai_tools") if isinstance(context, dict) else {}
        if not isinstance(tools, dict) or not tools.get("enabled"):
            return {"enabled": False, "allow": True, "status": "unavailable"}
        profit = self._local_profit_signal(context)
        if not profit or not signal_available(profit):
            return {"enabled": False, "allow": True, "status": "no_profit_prediction"}
        side = "long" if direction == "long" else "short"
        expected = self._local_expected_return(profit, side)
        opposite = "short" if side == "long" else "long"
        opposite_expected = self._local_expected_return(profit, opposite)
        best_side = signal_payload_side(profit) or str(profit.get("best_side") or "")
        loss_probability = self._safe_float(profit.get(f"{side}_loss_probability"), 0.0)
        profile = self._safe_dict(profit.get("symbol_side_profile"))
        side_profile = self._safe_dict(profile.get(side))
        loss_pressure = self._safe_float(side_profile.get("loss_pressure"), 0.0)
        result = {
            "enabled": True,
            "allow": True,
            "status": "passed",
            "direction": side,
            "expected_return_pct": round(expected, 4),
            "opposite_expected_return_pct": round(opposite_expected, 4),
            "best_side": best_side,
            "loss_probability": round(loss_probability, 4),
            "loss_pressure": round(loss_pressure, 4),
            "trained": bool(profit.get("trained")),
        }
        if loss_probability >= LOCAL_TOOLS_MAX_LOSS_PROBABILITY:
            result.update(
                {
                    "allow": False,
                    "status": "blocked_high_loss_probability",
                    "reason": (
                        f"服务器盈利模型判断{('做多' if side == 'long' else '做空')}亏损概率 "
                        f"{loss_probability:.1%} 过高，本轮不允许开仓。"
                    ),
                }
            )
            return result
        if expected < 0:
            result.update(
                {
                    "allow": False,
                    "status": "blocked_server_negative_expectancy",
                    "reason": (
                        f"服务器盈利模型给出{('做多' if side == 'long' else '做空')}调整后预期收益 "
                        f"{expected:.4f}% 为负，本轮不允许开仓。"
                    ),
                }
            )
            return result
        if best_side in {"long", "short"} and best_side != side and expected <= opposite_expected:
            result.update(
                {
                    "allow": False,
                    "status": "blocked_server_better_opposite_side",
                    "reason": (
                        "服务器盈利模型判断相反方向的预期收益更好；"
                        f"当前方向 {expected:.4f}%，相反方向 {opposite_expected:.4f}%。"
                    ),
                }
            )
            return result
        if loss_pressure >= 0.70 and expected < ML_PROFIT_FIRST_MIN_EXPECTED_RETURN_PCT:
            result.update(
                {
                    "allow": False,
                    "status": "blocked_symbol_side_loss_profile",
                    "reason": "该币种/方向历史亏损压力过高，且当前预期收益不足以覆盖风险。",
                }
            )
        return result

    def _market_regime_entry_block_reason(
        self, action: Action, context: dict[str, Any]
    ) -> str | None:
        if action not in (Action.LONG, Action.SHORT):
            return None
        # Market regime is only a soft context signal now. It must not hard-ban
        # either side; otherwise rebound regimes silently turn the whole system
        # into long-only and downtrend regimes into short-only.
        return None

    def _entry_execution_gate(self) -> dict[str, Any]:
        """Return the default entry gate; daily profit targets are not a trading input."""

        return {
            "mode": "normal",
            "score_bonus": 0.0,
            "min_confidence": MIN_EXECUTABLE_ENTRY_CONFIDENCE,
            "min_quality_points": 2,
            "allow_probe": True,
            "max_position_size": MAX_NORMAL_ENTRY_SIZE,
            "max_leverage": settings.max_leverage,
        }

    def _local_profit_aligned(self, context: dict[str, Any] | None, side: str) -> bool:
        side = str(side or "").lower()
        if side not in {"long", "short"}:
            return False
        profit = self._local_profit_signal(context)
        if not profit or not signal_available(profit):
            return False
        expected = self._local_expected_return(profit, side)
        best_side = signal_payload_side(profit) or str(profit.get("best_side") or "").lower()
        loss_probability = self._safe_float(profit.get(f"{side}_loss_probability"), 0.50)
        return (
            expected > 0
            and best_side in {"", side}
            and loss_probability < LOCAL_TOOLS_MAX_LOSS_PROBABILITY
        )

    def _time_series_aligned(self, context: dict[str, Any] | None, side: str) -> bool:
        side = str(side or "").lower()
        if side not in {"long", "short"}:
            return False
        prediction = self._local_timeseries_signal(context)
        if not prediction or not signal_available(prediction):
            return False
        best_side = signal_payload_side(prediction)
        expected = self._local_expected_return(prediction, side)
        return best_side == side and expected > 0

    def _profit_quality_entry_expand_evidence(
        self,
        *,
        context: dict[str, Any],
        action_side: str,
        confidence: float,
        quality_points: int,
        disagreement: float,
        ml_gate: dict[str, Any],
        ml_profit_hint: dict[str, Any],
        entry_gate: dict[str, Any],
        probe_entry: bool,
        profit_first_probe: bool,
        recovery_probe_entry: bool,
        quant_validation_probe: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Allow larger exposure only when current profit-quality evidence agrees."""
        action_side = str(action_side or "").lower()
        if action_side not in {"long", "short"}:
            return {"allow": False, "reason": "方向无效，不能扩仓。"}
        if (
            probe_entry
            or profit_first_probe
            or recovery_probe_entry
            or (isinstance(quant_validation_probe, dict) and quant_validation_probe.get("allow"))
        ):
            return {"allow": False, "reason": "试单/量化验证单只允许小仓，不做扩仓。"}
        if not isinstance(ml_gate, dict):
            ml_gate = {}
        if not ml_gate.get("allow", True):
            return {"allow": False, "reason": "ML 盈利质量门禁未通过，不做扩仓。"}

        ml_supported = ml_gate.get("status") == "supported_by_profit_quality"
        ml_hint_aligned = bool(
            ml_profit_hint.get("strong") and ml_profit_hint.get("side") == action_side
        )
        local_aligned = self._local_profit_aligned(context, action_side)
        timeseries_aligned = self._time_series_aligned(context, action_side)
        evidence_count = sum(
            bool(v) for v in (ml_supported or ml_hint_aligned, local_aligned, timeseries_aligned)
        )

        if confidence < 0.76:
            return {"allow": False, "reason": f"AI 置信度 {confidence:.0%} 不足以放大仓位。"}
        if quality_points < 2:
            return {"allow": False, "reason": f"入场质量只有 {quality_points}/3，不放大仓位。"}
        if disagreement > 0.38:
            return {"allow": False, "reason": f"专家分歧 {disagreement:.2f} 偏高，不放大仓位。"}
        if evidence_count <= 0:
            return {"allow": False, "reason": "ML、服务器盈利模型、时序模型没有同向盈利证据。"}

        entry_mode = str(entry_gate.get("mode") or "normal")
        max_size = PROFIT_QUALITY_EXPAND_NORMAL_MAX_SIZE
        if entry_mode == "loss_recovery_selective":
            max_size = PROFIT_QUALITY_EXPAND_SELECTIVE_MAX_SIZE
        elif entry_mode in {"recovery", "recovery_attack"}:
            max_size = PROFIT_QUALITY_EXPAND_RECOVERY_MAX_SIZE
        elif entry_mode in {"near_target", "profit_protected_expand"}:
            max_size = min(PROFIT_QUALITY_EXPAND_NORMAL_MAX_SIZE, 0.080)
        if evidence_count >= 2 and entry_mode not in {"loss_recovery_selective"}:
            max_size = min(max_size + 0.010, PROFIT_QUALITY_EXPAND_NORMAL_MAX_SIZE)

        return {
            "allow": True,
            "side": action_side,
            "entry_gate_mode": entry_mode,
            "min_position_size": PROFIT_QUALITY_EXPAND_MIN_ENTRY_SIZE,
            "max_position_size": max_size,
            "confidence": round(confidence, 4),
            "quality_points": int(quality_points),
            "disagreement": round(disagreement, 4),
            "ml_supported": bool(ml_supported),
            "ml_hint_aligned": bool(ml_hint_aligned),
            "local_profit_aligned": bool(local_aligned),
            "timeseries_aligned": bool(timeseries_aligned),
            "evidence_count": int(evidence_count),
            "reason": "AI 信号与盈利质量证据同向，允许提高仓位利用率。只放大优质机会，不放大普通信号。",
        }

    def _has_technical_support(
        self,
        action: Action,
        decisions: dict[str, DecisionOutput],
        *,
        min_confidence: float = 0.55,
    ) -> bool:
        return any(
            name in ENTRY_DIRECTION_SUPPORT_EXPERTS
            and isinstance(decision, DecisionOutput)
            and decision.action == action
            and float(decision.confidence or 0.0) >= min_confidence
            for name, decision in decisions.items()
        )

    def _entry_signal_allowed(
        self,
        action: Action,
        decisions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]],
        validation_adjustment: float,
        disagreement: float,
        context: dict[str, Any] | None = None,
    ) -> bool:
        if disagreement >= MAX_ENTRY_DISAGREEMENT:
            return False
        if validation_adjustment < -0.05:
            return False
        if any(
            v.get("major_conflict") or v.get("consistency") == "divergent"
            for v in cross_validations
        ):
            return False

        same_direction = [
            (name, d)
            for name, d in decisions.items()
            if isinstance(d, DecisionOutput) and d.action == action and d.confidence >= 0.55
        ]
        directional_support = [
            (name, d) for name, d in same_direction if name not in MARKET_DIRECTION_EXCLUDED_EXPERTS
        ]
        technical_support = [
            (name, d)
            for name, d in same_direction
            if name in ENTRY_DIRECTION_SUPPORT_EXPERTS and d.confidence >= 0.55
        ]
        aligned_validations = sum(
            1
            for v in cross_validations
            if isinstance(v, dict) and v.get("consistency") == "aligned"
        )
        ml_hint = self._ml_profit_first_direction_hint(context or {})
        action_side = "long" if action == Action.LONG else "short"
        if (
            ml_hint.get("strong")
            and ml_hint.get("side") == action_side
            and self._local_profit_aligned(context, action_side)
            and validation_adjustment >= 0.0
            and len(technical_support) >= 1
        ):
            return True
        if (
            self._local_profit_aligned(context, action_side)
            and len(technical_support) >= 1
            and validation_adjustment >= -0.05
            and disagreement < 0.45
            and not any(
                isinstance(d, DecisionOutput)
                and d.action.is_entry()
                and d.action != action
                and float(d.confidence or 0.0) >= 0.62
                for d in decisions.values()
            )
        ):
            return True
        if (
            validation_adjustment >= 0.20
            and aligned_validations >= 2
            and len(technical_support) >= 2
        ):
            return True
        strategy = context.get("strategy_mode") if isinstance(context, dict) else {}
        if (
            isinstance(strategy, dict)
            and strategy.get("strategy") == "recovery_attack"
            and validation_adjustment >= 0.10
            and aligned_validations >= 1
            and len(technical_support) >= 1
            and len(directional_support) >= 2
            and disagreement < 0.45
        ):
            return True
        return len(directional_support) >= 2 and bool(technical_support)

    def _profit_first_probe_allowed(
        self,
        ml_hint: dict[str, Any],
        context: dict[str, Any] | None,
        decisions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]],
        validation_adjustment: float,
        disagreement: float,
        features: FeatureVector,
    ) -> bool:
        if not isinstance(ml_hint, dict) or not ml_hint.get("strong"):
            return False
        side = str(ml_hint.get("side") or "")
        if side not in {"long", "short"}:
            return False
        if not self._local_profit_aligned(context, side):
            return False
        action = Action.LONG if side == "long" else Action.SHORT
        if validation_adjustment < -0.05 or disagreement >= 0.65:
            return False
        if any(
            v.get("major_conflict") or v.get("consistency") == "divergent"
            for v in cross_validations
        ):
            return False
        if self._entry_quality_points(features, action) < 2:
            return False

        same_technical = [
            d
            for name, d in decisions.items()
            if (
                name in ENTRY_DIRECTION_SUPPORT_EXPERTS
                and isinstance(d, DecisionOutput)
                and d.action == action
                and d.confidence >= 0.60
            )
        ]
        if not same_technical:
            return False

        opposite = Action.SHORT if action == Action.LONG else Action.LONG
        strong_opposite_technical = [
            d
            for name, d in decisions.items()
            if (
                name in ENTRY_DIRECTION_SUPPORT_EXPERTS
                and isinstance(d, DecisionOutput)
                and d.action == opposite
                and d.confidence >= 0.58
            )
        ]
        return not strong_opposite_technical

    def _quant_validation_probe_evidence(
        self,
        ml_hint: dict[str, Any],
        context: dict[str, Any],
        decisions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]],
        validation_adjustment: float,
        disagreement: float,
        features: FeatureVector,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enabled": True,
            "allow": False,
            "status": "not_qualified",
            "reason": "本轮本地量化证据不足，继续观望。",
        }
        if validation_adjustment < -0.08:
            result.update(
                {
                    "status": "blocked_negative_cross_validation",
                    "validation_adjustment": round(validation_adjustment, 4),
                    "reason": "交叉验证明显偏负，不使用量化验证单。",
                }
            )
            return result
        if disagreement >= 0.72:
            result.update(
                {
                    "status": "blocked_high_disagreement",
                    "disagreement": round(disagreement, 4),
                    "reason": "专家分歧过高，不使用量化验证单。",
                }
            )
            return result
        if any(
            v.get("major_conflict") or v.get("consistency") == "divergent"
            for v in cross_validations
        ):
            result.update(
                {
                    "status": "blocked_major_conflict",
                    "reason": "交叉验证存在重大冲突，不使用量化验证单。",
                }
            )
            return result

        if not isinstance(ml_hint, dict) or not ml_hint.get("strong"):
            result.update(
                {
                    "status": "blocked_ml_not_strong",
                    "ml_hint": ml_hint if isinstance(ml_hint, dict) else {},
                    "reason": "ML 盈亏期望不够强，不让服务器模型单独发起验证单。",
                }
            )
            return result
        side = str(ml_hint.get("side") or "").lower()
        if side not in {"long", "short"}:
            result.update({"status": "blocked_invalid_ml_side", "reason": "ML 没有给出有效方向。"})
            return result
        action = Action.LONG if side == "long" else Action.SHORT
        same_technical_support = [
            name
            for name, decision in decisions.items()
            if (
                name in ENTRY_DIRECTION_SUPPORT_EXPERTS
                and isinstance(decision, DecisionOutput)
                and decision.action == action
                and float(decision.confidence or 0.0) >= 0.60
            )
        ]
        if not same_technical_support:
            result.update(
                {
                    "status": "blocked_no_expert_support",
                    "reason": "量化模型不能单独发起开仓；本轮没有行情方向/短线时序专家同向强支持。",
                }
            )
            return result

        profit = self._local_profit_signal(context)
        if not profit or not signal_available(profit):
            result.update(
                {
                    "status": "blocked_no_local_profit_prediction",
                    "reason": "服务器盈利预测不可用，不发起量化验证单。",
                }
            )
            return result
        opposite = "short" if side == "long" else "long"
        local_expected = self._local_expected_return(profit, side)
        opposite_expected = self._local_expected_return(profit, opposite)
        best_side = signal_payload_side(profit) or str(profit.get("best_side") or "").lower()
        loss_probability = self._safe_float(profit.get(f"{side}_loss_probability"), 0.0)
        profit_quality_score = self._safe_float(profit.get("profit_quality_score"), 0.0)
        result.update(
            {
                "side": side,
                "action": action.value,
                "ml_expected_return_pct": ml_hint.get("expected_return_pct"),
                "ml_profit_edge_pct": ml_hint.get("profit_edge_pct"),
                "local_expected_return_pct": round(local_expected, 4),
                "local_opposite_expected_return_pct": round(opposite_expected, 4),
                "local_profit_quality_score": round(profit_quality_score, 4),
                "loss_probability": round(loss_probability, 4),
                "best_side": best_side or side,
                "quality_points": self._entry_quality_points(features, action),
            }
        )
        if best_side in {"long", "short"} and best_side != side:
            result.update(
                {
                    "status": "blocked_local_direction_conflict",
                    "reason": (
                        "ML 与服务器盈利模型方向不一致，"
                        f"ML={side}，服务器盈利模型={best_side}，本轮不验证。"
                    ),
                }
            )
            return result
        if local_expected < QUANT_VALIDATION_MIN_LOCAL_EXPECTED_RETURN_PCT:
            result.update(
                {
                    "status": "blocked_low_local_expected_return",
                    "reason": (
                        f"服务器盈利模型该方向预期收益 {local_expected:.4f}% 偏低，"
                        "不足以覆盖手续费/滑点验证成本。"
                    ),
                }
            )
            return result
        if local_expected < opposite_expected:
            result.update(
                {
                    "status": "blocked_local_better_opposite",
                    "reason": "服务器盈利模型判断相反方向预期收益更高，本轮不验证。",
                }
            )
            return result
        if loss_probability >= QUANT_VALIDATION_MAX_LOSS_PROBABILITY:
            result.update(
                {
                    "status": "blocked_high_loss_probability",
                    "reason": f"服务器盈利模型亏损概率 {loss_probability:.1%} 偏高，本轮不验证。",
                }
            )
            return result
        if profit_quality_score < QUANT_VALIDATION_MIN_PROFIT_QUALITY_SCORE:
            result.update(
                {
                    "status": "blocked_low_profit_quality",
                    "reason": f"盈利质量分 {profit_quality_score:.4f} 偏低，本轮不验证。",
                }
            )
            return result

        opposite_action = Action.SHORT if action == Action.LONG else Action.LONG
        strong_opposite_technical = [
            name
            for name, d in decisions.items()
            if (
                name in ENTRY_DIRECTION_SUPPORT_EXPERTS
                and isinstance(d, DecisionOutput)
                and d.action == opposite_action
                and d.confidence >= 0.62
            )
        ]
        if strong_opposite_technical:
            result.update(
                {
                    "status": "blocked_strong_opposite_technical",
                    "opposite_experts": strong_opposite_technical,
                    "reason": "行情方向/短线时序专家存在强反向信号，不使用量化验证单。",
                }
            )
            return result

        risk_opinion = next(
            (
                d
                for name, d in decisions.items()
                if name == "risk_expert" and isinstance(d, DecisionOutput)
            ),
            None,
        )
        if risk_opinion is not None and self._is_hard_risk_veto(risk_opinion, features):
            result.update(
                {
                    "status": "blocked_hard_risk_veto",
                    "reason": "风控专家给出硬否决，不使用量化验证单。",
                }
            )
            return result

        result.update(
            {
                "allow": True,
                "status": "allowed",
                "max_position_size": QUANT_VALIDATION_PROBE_SIZE,
                "confidence_floor": QUANT_VALIDATION_PROBE_CONFIDENCE,
                "reason": (
                    "普通专家投票仍偏观望，但 ML 与服务器盈利模型方向一致、预期收益为正、"
                    "亏损概率可接受，且无强反向技术专家/硬风控；允许小仓验证以产生可学习的真实结果。"
                ),
            }
        )
        return result

    def _quant_reversal_probe_evidence(
        self,
        blocked_action: Action,
        ml_hint: dict[str, Any],
        ml_gate: dict[str, Any],
        context: dict[str, Any],
        decisions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]],
        validation_adjustment: float,
        disagreement: float,
        features: FeatureVector,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enabled": True,
            "allow": False,
            "status": "not_qualified",
            "blocked_action": (
                blocked_action.value if isinstance(blocked_action, Action) else str(blocked_action)
            ),
            "reason": "AI 方向被盈利模型否决，但反向量化证据不足，继续观望。",
        }
        if blocked_action not in (Action.LONG, Action.SHORT):
            return result
        if validation_adjustment < -0.08 or disagreement >= 0.72:
            result.update(
                {
                    "status": "blocked_validation_or_disagreement",
                    "validation_adjustment": round(validation_adjustment, 4),
                    "disagreement": round(disagreement, 4),
                    "reason": "交叉验证或专家分歧不适合反向验证。",
                }
            )
            return result
        if any(
            v.get("major_conflict") or v.get("consistency") == "divergent"
            for v in cross_validations
        ):
            result.update(
                {"status": "blocked_major_conflict", "reason": "存在重大冲突，不做反向验证。"}
            )
            return result

        reverse_side = "short" if blocked_action == Action.LONG else "long"
        reverse_action = Action.SHORT if reverse_side == "short" else Action.LONG
        same_technical_support = [
            name
            for name, decision in decisions.items()
            if (
                name in ENTRY_DIRECTION_SUPPORT_EXPERTS
                and isinstance(decision, DecisionOutput)
                and decision.action == reverse_action
                and float(decision.confidence or 0.0) >= 0.60
            )
        ]
        if not same_technical_support:
            result.update(
                {
                    "status": "blocked_no_expert_support",
                    "reverse_side": reverse_side,
                    "reason": "量化模型不能单独反向开仓；本轮没有行情方向/短线时序专家同向强支持。",
                }
            )
            return result
        if (
            not isinstance(ml_hint, dict)
            or ml_hint.get("side") != reverse_side
            or not ml_hint.get("strong")
        ):
            result.update(
                {
                    "status": "blocked_ml_not_reverse_strong",
                    "ml_hint": ml_hint if isinstance(ml_hint, dict) else {},
                    "reverse_side": reverse_side,
                    "reason": "ML 没有明确支持被否决方向的反向，不做反向验证。",
                }
            )
            return result

        local_gate = ml_gate.get("local_ai_tools_gate") if isinstance(ml_gate, dict) else {}
        if not isinstance(local_gate, dict):
            local_gate = {}
        profit = self._local_profit_signal(context)
        if not profit or not signal_available(profit):
            result.update(
                {"status": "blocked_no_local_profit_prediction", "reason": "服务器盈利预测不可用。"}
            )
            return result

        expected = self._local_expected_return(profit, reverse_side)
        blocked_side = "long" if blocked_action == Action.LONG else "short"
        blocked_expected = self._local_expected_return(profit, blocked_side)
        best_side = (
            signal_payload_side(profit)
            or str(profit.get("best_side") or local_gate.get("best_side") or "").lower()
        )
        loss_probability = self._safe_float(profit.get(f"{reverse_side}_loss_probability"), 0.0)
        profit_quality_score = self._safe_float(profit.get("profit_quality_score"), 0.0)
        result.update(
            {
                "side": reverse_side,
                "action": reverse_action.value,
                "blocked_side": blocked_side,
                "blocked_expected_return_pct": round(blocked_expected, 4),
                "local_expected_return_pct": round(expected, 4),
                "local_profit_quality_score": round(profit_quality_score, 4),
                "loss_probability": round(loss_probability, 4),
                "best_side": best_side or reverse_side,
                "ml_expected_return_pct": ml_hint.get("expected_return_pct"),
                "ml_profit_edge_pct": ml_hint.get("profit_edge_pct"),
            }
        )
        if best_side in {"long", "short"} and best_side != reverse_side:
            result.update(
                {
                    "status": "blocked_local_not_reverse",
                    "reason": "服务器盈利模型最佳方向不是反向验证方向。",
                }
            )
            return result
        if expected < QUANT_VALIDATION_MIN_LOCAL_EXPECTED_RETURN_PCT:
            result.update(
                {
                    "status": "blocked_low_local_expected_return",
                    "reason": f"反向预期收益 {expected:.4f}% 偏低，不值得验证。",
                }
            )
            return result
        if expected <= blocked_expected:
            result.update(
                {
                    "status": "blocked_reverse_not_better",
                    "reason": "反向预期收益没有明显优于原方向。",
                }
            )
            return result
        if loss_probability >= QUANT_VALIDATION_MAX_LOSS_PROBABILITY:
            result.update(
                {
                    "status": "blocked_high_loss_probability",
                    "reason": f"反向亏损概率 {loss_probability:.1%} 偏高，不验证。",
                }
            )
            return result
        if profit_quality_score < QUANT_VALIDATION_MIN_PROFIT_QUALITY_SCORE:
            result.update(
                {
                    "status": "blocked_low_profit_quality",
                    "reason": f"盈利质量分 {profit_quality_score:.4f} 偏低，不验证。",
                }
            )
            return result

        strong_original_technical = [
            name
            for name, d in decisions.items()
            if (
                name in ENTRY_DIRECTION_SUPPORT_EXPERTS
                and isinstance(d, DecisionOutput)
                and d.action == blocked_action
                and d.confidence >= 0.82
            )
        ]
        if strong_original_technical:
            result.update(
                {
                    "status": "blocked_very_strong_original_technical",
                    "opposite_experts": strong_original_technical,
                    "reason": "行情方向/短线时序专家原方向置信度极高，不直接反向验证。",
                }
            )
            return result

        risk_opinion = next(
            (
                d
                for name, d in decisions.items()
                if name == "risk_expert" and isinstance(d, DecisionOutput)
            ),
            None,
        )
        if risk_opinion is not None and self._is_hard_risk_veto(risk_opinion, features):
            result.update({"status": "blocked_hard_risk_veto", "reason": "风控专家给出硬否决。"})
            return result

        result.update(
            {
                "allow": True,
                "status": "allowed",
                "max_position_size": QUANT_VALIDATION_PROBE_SIZE,
                "confidence_floor": QUANT_VALIDATION_PROBE_CONFIDENCE,
                "reason": (
                    "AI 原方向被 ML/服务器盈利模型否决，且二者一致支持相反方向；"
                    "允许小仓反向验证，用真实结果检验量化模型是否优于专家方向。"
                ),
            }
        )
        return result

    def _probe_entry_allowed(
        self,
        action: Action,
        decisions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]],
        validation_adjustment: float,
        features: FeatureVector,
    ) -> bool:
        if validation_adjustment < -0.05:
            return False
        if any(
            v.get("major_conflict") or v.get("consistency") == "divergent"
            for v in cross_validations
        ):
            return False
        if self._entry_quality_points(features, action) < 2:
            return False

        same_direction = [
            (name, d)
            for name, d in decisions.items()
            if (
                isinstance(d, DecisionOutput)
                and d.action == action
                and d.confidence >= 0.55
                and name not in MARKET_DIRECTION_EXCLUDED_EXPERTS
            )
        ]
        technical_names = ENTRY_DIRECTION_SUPPORT_EXPERTS
        technical_same = [(name, d) for name, d in same_direction if name in technical_names]
        if len(technical_same) >= 2:
            return True
        if validation_adjustment >= 0.05 and any(
            name in technical_names and d.confidence >= 0.60 for name, d in same_direction
        ):
            return True
        return len(same_direction) >= 2 and any(
            name in technical_names for name, _d in same_direction
        )

    def _recovery_attack_probe_allowed(
        self,
        decisions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]],
        validation_adjustment: float,
        disagreement: float,
        context: dict[str, Any] | None,
        features: FeatureVector,
        decision_score: float,
    ) -> bool:
        strategy = context.get("strategy_mode") if isinstance(context, dict) else {}
        if not isinstance(strategy, dict) or strategy.get("strategy") != "recovery_attack":
            return False
        if abs(decision_score) < 0.34 or disagreement >= 0.55 or validation_adjustment < -0.05:
            return False
        if any(
            v.get("major_conflict") or v.get("consistency") == "divergent"
            for v in cross_validations
        ):
            return False
        action = Action.LONG if decision_score >= 0 else Action.SHORT
        if self._entry_quality_points(features, action) < 1:
            return False
        technical_same = [
            d
            for name, d in decisions.items()
            if (
                name in ENTRY_DIRECTION_SUPPORT_EXPERTS
                and isinstance(d, DecisionOutput)
                and d.action == action
                and d.confidence >= 0.55
            )
        ]
        risk_opinion = next(
            (
                d
                for name, d in decisions.items()
                if name == "risk_expert" and isinstance(d, DecisionOutput)
            ),
            None,
        )
        risk_cleared = True
        if risk_opinion is not None:
            risk_cleared = risk_opinion.action in {
                action,
                Action.HOLD,
            } and not self._is_hard_risk_veto(risk_opinion, features)
        return bool(technical_same and risk_cleared)

    def _entry_quality_points(self, features: FeatureVector, action: Action) -> int:
        """Score entry quality by liquidity, trend strength, and direction alignment."""
        entry_filters = entry_filters_from_context(getattr(self, "_current_strategy_context", {}))
        volume_ratio = self._safe_float(getattr(features, "volume_ratio", 0.0), 0.0)
        adx_14 = self._safe_float(getattr(features, "adx_14", 0.0), 0.0)
        price_vs_sma20 = self._safe_float(getattr(features, "price_vs_sma20", 0.0), 0.0)
        price_vs_sma50 = self._safe_float(getattr(features, "price_vs_sma50", 0.0), 0.0)

        trend_aligned = False
        if action == Action.LONG:
            trend_aligned = price_vs_sma20 > 0 and price_vs_sma50 > 0
        elif action == Action.SHORT:
            trend_aligned = price_vs_sma20 < 0 and price_vs_sma50 < 0

        return sum(
            1
            for passed in (
                volume_ratio >= entry_filters.min_entry_volume_ratio,
                adx_14 >= entry_filters.min_entry_adx,
                trend_aligned,
            )
            if passed
        )

    def _entry_leverage_cap(self, confidence: float, quality_points: int) -> float:
        if confidence < 0.68:
            return min(5.0, settings.max_leverage)
        if confidence < 0.78 or quality_points < 2:
            return min(10.0, settings.max_leverage)
        return settings.max_leverage

    def _entry_min_leverage(self, confidence: float, quality_points: int) -> float:
        if confidence >= 0.78 and quality_points >= 2:
            return min(10.0, settings.max_leverage)
        if confidence >= 0.68:
            return min(5.0, settings.max_leverage)
        return min(1.0, settings.max_leverage)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _close_decision_if_needed(
        self,
        features: FeatureVector,
        current_side: str | None,
        exit_votes: list[DecisionOutput],
        risk_vetoes: list[DecisionOutput],
        score: float,
        disagreement: float,
        raw_opinions: list[dict[str, Any]],
        cross_validations: list[dict[str, Any]] | None = None,
        consultation: dict[str, Any] | None = None,
        resolution_brief: str = "",
        review_positions: bool = False,
        symbol_positions: list[dict] | None = None,
        portfolio_profit_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> DecisionOutput | None:
        if not current_side:
            return None

        close_action = Action.CLOSE_LONG if current_side == "long" else Action.CLOSE_SHORT
        matching_exit_votes = [d for d in exit_votes if d.action == close_action]

        evidence = self._position_close_evidence(
            current_side,
            close_action,
            exit_votes,
            risk_vetoes,
            score,
            raw_opinions,
            symbol_positions,
            features=features,
            context=context,
        )
        if review_positions:
            should_close = bool(evidence.get("should_close"))
        else:
            opposite_pressure = (current_side == "long" and score <= -0.25) or (
                current_side == "short" and score >= 0.25
            )
            should_close = bool(matching_exit_votes) or bool(risk_vetoes) or opposite_pressure
            evidence.update(
                {
                    "should_close": should_close,
                    "action_plan": "full_close" if should_close else "hold",
                    "position_size_pct": 1.0 if should_close else 0.0,
                }
            )
        if not should_close:
            return None

        confidence_candidates = [d.confidence for d in matching_exit_votes + risk_vetoes]
        evidence_confidence = float(
            evidence.get("suggested_confidence") or evidence.get("max_support_confidence") or 0.0
        )
        if evidence_confidence > 0:
            confidence_candidates.append(evidence_confidence)
        confidence_candidates.append(min(abs(score) + 0.45, 0.85))
        confidence = max(confidence_candidates)
        confidence = min(
            max(
                confidence + self._validation_adjustment(cross_validations or [], consultation),
                0.55,
            ),
            0.95,
        )
        position_size_pct = float(evidence.get("position_size_pct") or 1.0)
        action_plan = str(evidence.get("action_plan") or "full_close")
        plan_label = "全平" if action_plan == "full_close" else "减仓"
        reason = self._reason(
            (
                f"持仓复盘达到主动{plan_label}门槛："
                f"{evidence.get('reason') or '多专家支持处理当前仓位'}"
            ),
            score,
            disagreement,
            raw_opinions,
            resolution_brief,
        )
        raw_response = self._raw(raw_opinions, score, disagreement, cross_validations, consultation)
        raw_response["close_evidence"] = evidence
        raw_response["portfolio_profit_protection"] = portfolio_profit_context or {}
        raw_response["position_review_policy"] = {
            "result": action_plan,
            "minimum_support_required": MIN_REVIEW_CLOSE_SUPPORT,
            "full_close_support_required": FULL_CLOSE_SUPPORT,
            "add_support_required": ADD_POSITION_MIN_SUPPORT,
            "available_actions": ["hold", "add", "reduce", "full_close"],
            "note": "硬风险全平；盈利保护可减仓/平仓；多专家强共识全平；普通共识先减仓，避免过早全平。",
        }
        return DecisionOutput(
            model_name=ENSEMBLE_TRADER_NAME,
            symbol=features.symbol,
            action=close_action,
            confidence=confidence,
            reasoning=reason,
            position_size_pct=position_size_pct,
            suggested_leverage=1.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            raw_response=raw_response,
            feature_snapshot=features.to_dict(),
        )

    def _add_decision_if_needed(
        self,
        features: FeatureVector,
        current_side: str | None,
        score: float,
        disagreement: float,
        raw_opinions: list[dict[str, Any]],
        cross_validations: list[dict[str, Any]] | None = None,
        consultation: dict[str, Any] | None = None,
        resolution_brief: str = "",
        symbol_positions: list[dict] | None = None,
    ) -> DecisionOutput | None:
        if not current_side:
            return None
        evidence = self._position_add_evidence(
            current_side,
            score,
            raw_opinions,
            symbol_positions,
        )
        if not evidence.get("should_add"):
            return None

        add_action = Action.LONG if current_side == "long" else Action.SHORT
        confidence = min(
            max(
                float(evidence.get("suggested_confidence") or 0.0)
                + self._validation_adjustment(cross_validations or [], consultation),
                MIN_EXECUTABLE_ENTRY_CONFIDENCE,
            ),
            0.92,
        )
        reason = self._reason(
            f"持仓复盘达到加仓门槛：{evidence.get('reason') or '多专家支持顺势加仓'}",
            score,
            disagreement,
            raw_opinions,
            resolution_brief,
        )
        raw_response = self._raw(raw_opinions, score, disagreement, cross_validations, consultation)
        raw_response["add_evidence"] = evidence
        raw_response["position_review_policy"] = {
            "result": "add",
            "available_actions": ["hold", "add", "reduce", "full_close"],
            "add_support_required": ADD_POSITION_MIN_SUPPORT,
            "note": "同向加仓只在已有盈利、风险占用较低、至少两个专家同向支持时触发。",
        }
        return DecisionOutput(
            model_name=ENSEMBLE_TRADER_NAME,
            symbol=features.symbol,
            action=add_action,
            confidence=confidence,
            reasoning=reason,
            position_size_pct=float(evidence.get("position_size_pct") or ADD_POSITION_MIN_SIZE),
            suggested_leverage=float(evidence.get("suggested_leverage") or 1.0),
            stop_loss_pct=float(evidence.get("stop_loss_pct") or 0.035),
            take_profit_pct=float(evidence.get("take_profit_pct") or 0.08),
            raw_response=raw_response,
            feature_snapshot=features.to_dict(),
        )

    def _position_add_evidence(
        self,
        current_side: str | None,
        score: float,
        raw_opinions: list[dict[str, Any]],
        symbol_positions: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Summarize whether a position review has enough evidence to add."""
        if current_side not in {"long", "short"}:
            return {"should_add": False, "block_reason": "当前没有可加仓的持仓方向。"}

        add_value = Action.LONG.value if current_side == "long" else Action.SHORT.value
        support = []
        for opinion in raw_opinions:
            action = str(opinion.get("action") or "")
            confidence = self._safe_float(opinion.get("confidence"), 0.0)
            if action != add_value or confidence < ADD_POSITION_MIN_CONFIDENCE:
                continue
            support.append(
                {
                    "model_name": opinion.get("model_name"),
                    "label": opinion.get("label") or opinion.get("model_name"),
                    "action": action,
                    "confidence": round(confidence, 4),
                    "position_size_pct": self._safe_float(opinion.get("position_size_pct"), 0.0),
                    "suggested_leverage": self._safe_float(opinion.get("suggested_leverage"), 1.0),
                    "stop_loss_pct": self._safe_float(opinion.get("stop_loss_pct"), 0.035),
                    "take_profit_pct": self._safe_float(opinion.get("take_profit_pct"), 0.08),
                }
            )

        strong_support = [
            item
            for item in support
            if self._safe_float(item.get("confidence"), 0.0) >= ADD_POSITION_STRONG_CONFIDENCE
        ]
        aligned_score = (current_side == "long" and score >= ADD_POSITION_SCORE_THRESHOLD) or (
            current_side == "short" and score <= -ADD_POSITION_SCORE_THRESHOLD
        )
        continuation_score = score if current_side == "long" else -score

        position_unrealized_pnl = 0.0
        position_notional = 0.0
        position_profit_pct = 0.0
        position_risk_usage = 0.0
        position_loss = False
        if symbol_positions:
            pos = symbol_positions[0] or {}
            entry_price = self._safe_float(pos.get("entry_price"), 0.0)
            current_price = self._safe_float(pos.get("current_price"), 0.0)
            quantity = abs(self._safe_float(pos.get("quantity"), 0.0))
            stop_loss = self._safe_float(pos.get("stop_loss"), 0.0)
            position_unrealized_pnl = self._safe_float(pos.get("unrealized_pnl"), 0.0)
            position_notional = abs(entry_price * quantity)
            if position_notional > 0:
                position_profit_pct = position_unrealized_pnl / position_notional
            if entry_price > 0 and current_price > 0:
                if current_side == "long":
                    adverse_move = max(entry_price - current_price, 0.0)
                    planned_risk = (
                        max(entry_price - stop_loss, entry_price * 0.02)
                        if stop_loss > 0
                        else entry_price * 0.05
                    )
                else:
                    adverse_move = max(current_price - entry_price, 0.0)
                    planned_risk = (
                        max(stop_loss - entry_price, entry_price * 0.02)
                        if stop_loss > 0
                        else entry_price * 0.05
                    )
                position_loss = adverse_move > 0 or position_unrealized_pnl < 0
                if planned_risk > 0:
                    position_risk_usage = min(max(adverse_move / planned_risk, 0.0), 2.0)

        technical_support = [
            item
            for item in support
            if str(item.get("model_name") or "") in ENTRY_DIRECTION_SUPPORT_EXPERTS
        ]
        enough_support = len(support) >= ADD_POSITION_MIN_SUPPORT and bool(strong_support)
        profit_ok = (
            position_unrealized_pnl > 0 and position_profit_pct >= ADD_POSITION_MIN_PROFIT_RATIO
        )
        risk_ok = not position_loss and position_risk_usage <= ADD_POSITION_MAX_RISK_USAGE
        winner_expand = bool(
            not position_loss
            and position_unrealized_pnl >= WINNER_EXPAND_MIN_UNREALIZED_USDT
            and position_profit_pct >= WINNER_EXPAND_MIN_PROFIT_RATIO
            and continuation_score >= WINNER_EXPAND_SCORE_THRESHOLD
            and position_risk_usage <= WINNER_EXPAND_MAX_RISK_USAGE
            and (
                enough_support
                or bool(strong_support)
                or any(
                    self._safe_float(item.get("confidence"), 0.0) >= 0.64
                    for item in technical_support
                )
            )
        )
        should_add = bool(
            (enough_support and aligned_score and profit_ok and risk_ok) or winner_expand
        )
        max_conf = max([self._safe_float(item.get("confidence"), 0.0) for item in support] or [0.0])
        avg_size = self._avg(
            [self._safe_float(item.get("position_size_pct"), 0.0) for item in support],
            ADD_POSITION_MIN_SIZE,
        )
        size_multiplier = 0.75 if winner_expand else 0.5
        size = min(max(avg_size * size_multiplier, ADD_POSITION_MIN_SIZE), ADD_POSITION_MAX_SIZE)
        leverage = min(
            max(
                self._avg(
                    [self._safe_float(item.get("suggested_leverage"), 1.0) for item in support], 1.0
                ),
                1.0,
            ),
            self._entry_leverage_cap(max(max_conf, MIN_EXECUTABLE_ENTRY_CONFIDENCE), 3),
        )
        stop_loss_pct = min(
            max(
                self._avg(
                    [self._safe_float(item.get("stop_loss_pct"), 0.035) for item in support], 0.035
                ),
                0.015,
            ),
            0.08,
        )
        take_profit_pct = min(
            max(
                self._avg(
                    [self._safe_float(item.get("take_profit_pct"), 0.08) for item in support], 0.08
                ),
                stop_loss_pct * 1.8,
                0.03,
            ),
            0.35,
        )

        if should_add and winner_expand:
            reason = (
                f"盈利仓扩张：当前浮盈 {position_unrealized_pnl:.2f}U / {position_profit_pct * 100:.2f}%，"
                f"延续分 {continuation_score:.2f}，{len(support)} 个专家同向，"
                "允许把资金向已证明方向正确的仓位倾斜。"
            )
        elif should_add:
            reason = (
                f"已有仓位浮盈 {position_profit_pct * 100:.2f}%，"
                f"{len(support)} 个专家支持顺势加仓，风险占用 {position_risk_usage * 100:.0f}%，"
                "允许小比例加仓。"
            )
        else:
            missing = []
            if len(support) < ADD_POSITION_MIN_SUPPORT:
                missing.append(f"同向支持不足，仅 {len(support)} 个")
            if not strong_support:
                missing.append("缺少强同向确认")
            if not aligned_score:
                missing.append("加权方向分不足")
            if not profit_ok:
                missing.append("当前盈利不足，不追加入场")
            if position_unrealized_pnl > 0 and continuation_score < WINNER_EXPAND_SCORE_THRESHOLD:
                missing.append("盈利仓延续分不足，暂不扩大")
            if not risk_ok:
                missing.append("风险占用偏高或仓位正在亏损")
            reason = ""
            block_reason = "；".join(missing) or "加仓条件不足。"

        return {
            "should_add": should_add,
            "action_plan": "add" if should_add else "hold",
            "position_size_pct": round(size if should_add else 0.0, 4),
            "reason": reason,
            "block_reason": "" if should_add else block_reason,
            "support_count": len(support),
            "strong_support_count": len(strong_support),
            "same_side_votes": support,
            "aligned_score": aligned_score,
            "winner_expand": winner_expand,
            "continuation_score": round(continuation_score, 4),
            "score": round(score, 4),
            "max_support_confidence": round(max_conf, 4),
            "suggested_confidence": round(
                max(max_conf, min(abs(score) + 0.45, 0.85), MIN_EXECUTABLE_ENTRY_CONFIDENCE), 4
            ),
            "suggested_leverage": round(leverage, 4),
            "stop_loss_pct": round(stop_loss_pct, 6),
            "take_profit_pct": round(take_profit_pct, 6),
            "position_loss": position_loss,
            "position_risk_usage": round(position_risk_usage, 4),
            "position_notional": round(position_notional, 6),
            "position_profit": profit_ok,
            "position_profit_pct": round(position_profit_pct, 6),
            "position_unrealized_pnl": round(position_unrealized_pnl, 6),
        }

    def _loss_repair_evidence(
        self,
        *,
        current_side: str,
        score: float,
        raw_opinions: list[dict[str, Any]],
        context: dict[str, Any] | None,
        features: FeatureVector | None,
        position_unrealized_pnl: float,
        position_profit_pct: float,
        position_risk_usage: float,
        loss_abs: float,
        dynamic_loss_reduce_usdt: float,
        dynamic_loss_full_usdt: float,
        support_count: int,
        strong_support_count: int,
        strong_opposite_pressure: bool,
        moderate_opposite_pressure: bool,
        momentum_waning: bool,
    ) -> dict[str, Any]:
        """For losing positions, decide whether loss can plausibly repair or is likely expanding."""
        if position_unrealized_pnl >= 0 or current_side not in {"long", "short"}:
            return {"enabled": False, "position_loss": False}

        opposite = "short" if current_side == "long" else "long"
        continuation_score = score if current_side == "long" else -score
        profit = self._local_profit_signal(context)
        local_expected = self._local_expected_return(profit, current_side)
        opposite_local_expected = self._local_expected_return(profit, opposite)
        local_loss_probability = self._safe_float(
            profit.get(f"{current_side}_loss_probability"), 0.50
        )
        local_best_side = signal_payload_side(profit) or str(profit.get("best_side") or "").lower()
        local_available = isinstance(profit, dict) and bool(profit) and signal_available(profit)

        ts_prediction = self._local_timeseries_signal(context)
        ts_best_side = signal_payload_side(ts_prediction)
        ts_expected = self._local_expected_return(ts_prediction, current_side)
        ts_direction = str(ts_prediction.get("direction") or "").lower()
        ts_aligned = (
            ts_best_side == current_side
            or (current_side == "long" and ts_direction == "up")
            or (current_side == "short" and ts_direction == "down")
        ) and ts_expected > 0
        ts_opposes = bool(ts_prediction) and (
            ts_best_side == opposite
            or (current_side == "long" and ts_direction == "down")
            or (current_side == "short" and ts_direction == "up")
            or ts_expected < 0
        )

        ml_signal = context.get("ml_signal") if isinstance(context, dict) else {}
        predictions = ml_signal.get("predictions") if isinstance(ml_signal, dict) else []
        primary = predictions[0] if isinstance(predictions, list) and predictions else {}
        if not isinstance(primary, dict):
            primary = {}
        ml_expected = self._safe_float(primary.get(f"{current_side}_expected_return_pct"), 0.0)
        ml_opposite_expected = self._safe_float(primary.get(f"{opposite}_expected_return_pct"), 0.0)
        ml_best_side = str(primary.get("best_side") or "").lower()
        ml_aligned = bool(
            ml_expected > 0
            and ml_expected >= ml_opposite_expected
            and ml_best_side in {"", current_side}
        )
        ml_opposes = bool(
            ml_expected < 0 or (ml_best_side == opposite and ml_opposite_expected >= ml_expected)
        )

        same_side_votes = 0
        opposite_or_close_votes = support_count
        hold_votes = 0
        current_action = Action.LONG.value if current_side == "long" else Action.SHORT.value
        for opinion in raw_opinions:
            if not isinstance(opinion, dict):
                continue
            action = str(opinion.get("action") or "")
            confidence = self._safe_float(opinion.get("confidence"), 0.0)
            if confidence < 0.55:
                continue
            if action == current_action:
                same_side_votes += 1
            elif action == Action.HOLD.value:
                hold_votes += 1

        local_repair = bool(
            local_available
            and local_expected > 0
            and local_expected >= opposite_local_expected
            and local_loss_probability <= LOSS_REPAIR_MAX_LOSS_PROBABILITY
            and local_best_side in {"", current_side}
        )
        technical_repair = bool(continuation_score >= 0.22 and not momentum_waning)
        repair_score = sum(
            bool(v)
            for v in (
                local_repair,
                ml_aligned,
                ts_aligned,
                technical_repair,
                same_side_votes >= 2,
            )
        )
        expansion_score = sum(
            bool(v)
            for v in (
                local_available
                and local_loss_probability >= LOSS_EXPAND_MIN_LOSS_PROBABILITY
                and local_expected <= 0,
                local_best_side == opposite and opposite_local_expected >= local_expected,
                ml_opposes,
                ts_opposes,
                momentum_waning,
                continuation_score < 0.05,
                support_count >= LOSS_REPAIR_REDUCE_SUPPORT_COUNT,
                strong_opposite_pressure,
            )
        )
        repair_possible = (
            repair_score >= 2
            and local_loss_probability < LOSS_EXPAND_FULL_LOSS_PROBABILITY
            and not (expansion_score >= repair_score + 2)
        )
        likely_expanding = (
            expansion_score >= 3
            or (
                loss_abs >= dynamic_loss_reduce_usdt * 0.65
                and expansion_score >= 2
                and local_loss_probability >= LOSS_EXPAND_MIN_LOSS_PROBABILITY
            )
        ) and not repair_possible

        action_plan = "hold"
        position_size_pct = 0.0
        should_close = False
        reason = ""
        if likely_expanding:
            if (
                loss_abs >= dynamic_loss_full_usdt
                or position_risk_usage >= 0.70
                or (
                    local_loss_probability >= LOSS_EXPAND_FULL_LOSS_PROBABILITY
                    and support_count >= LOSS_REPAIR_FULL_SUPPORT_COUNT
                    and strong_support_count >= 1
                )
            ):
                should_close = True
                action_plan = "full_close"
                position_size_pct = 1.0
                reason = (
                    f"亏损修复评估：当前浮亏 {position_unrealized_pnl:.2f}U，"
                    f"服务器亏损概率 {local_loss_probability:.0%}，修复证据 {repair_score} 项，"
                    f"扩亏证据 {expansion_score} 项；由亏转盈概率不足，优先全平防止亏损扩大。"
                )
            else:
                should_close = False
                action_plan = "hold"
                position_size_pct = 0.0
                reason = (
                    f"亏损修复评估：当前浮亏 {position_unrealized_pnl:.2f}U，"
                    f"服务器亏损概率 {local_loss_probability:.0%}，修复证据 {repair_score} 项，"
                    f"扩亏证据 {expansion_score} 项；先减仓 60%，保留少量观察反转。"
                )

        return {
            "enabled": True,
            "position_loss": True,
            "side": current_side,
            "opposite_side": opposite,
            "repair_possible": bool(repair_possible),
            "likely_expanding_loss": bool(likely_expanding),
            "should_close": bool(should_close),
            "action_plan": action_plan,
            "position_size_pct": position_size_pct,
            "reason": reason,
            "repair_score": int(repair_score),
            "expansion_score": int(expansion_score),
            "same_side_votes": int(same_side_votes),
            "opposite_or_close_votes": int(opposite_or_close_votes),
            "hold_votes": int(hold_votes),
            "local_expected_return_pct": round(local_expected, 6),
            "local_opposite_expected_return_pct": round(opposite_local_expected, 6),
            "local_loss_probability": round(local_loss_probability, 6),
            "local_best_side": local_best_side,
            "local_repair": bool(local_repair),
            "ml_expected_return_pct": round(ml_expected, 6),
            "ml_opposite_expected_return_pct": round(ml_opposite_expected, 6),
            "ml_best_side": ml_best_side,
            "ml_aligned": bool(ml_aligned),
            "ml_opposes": bool(ml_opposes),
            "timeseries_expected_return_pct": round(ts_expected, 6),
            "timeseries_best_side": ts_best_side,
            "timeseries_aligned": bool(ts_aligned),
            "timeseries_opposes": bool(ts_opposes),
            "technical_repair": bool(technical_repair),
            "continuation_score": round(continuation_score, 6),
            "momentum_waning": bool(momentum_waning),
            "loss_abs": round(loss_abs, 6),
            "loss_reduce_line": round(dynamic_loss_reduce_usdt, 6),
            "loss_full_line": round(dynamic_loss_full_usdt, 6),
            "policy": "亏损仓位先判断能否由亏转盈；修复证据不足且扩亏证据较多时，优先减仓或全平。",
        }

    def _predictive_reversal_evidence(
        self,
        *,
        current_side: str,
        features: FeatureVector | None,
    ) -> dict[str, Any]:
        """Estimate whether the next short window is turning against the position."""
        side = str(current_side or "").lower()
        if side not in {"long", "short"} or features is None:
            return {"score": 0.0, "level": "none", "reasons": []}

        returns_1 = self._safe_float(getattr(features, "returns_1", 0.0), 0.0)
        returns_5 = self._safe_float(getattr(features, "returns_5", 0.0), 0.0)
        returns_20 = self._safe_float(getattr(features, "returns_20", 0.0), 0.0)
        volume_ratio = self._safe_float(getattr(features, "volume_ratio", 1.0), 1.0)
        rsi_14 = self._safe_float(getattr(features, "rsi_14", 50.0), 50.0)
        bb_pct = self._safe_float(getattr(features, "bb_pct", 0.5), 0.5)
        macd_diff = self._safe_float(getattr(features, "macd_diff", 0.0), 0.0)
        adx_14 = self._safe_float(getattr(features, "adx_14", 0.0), 0.0)

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

    def _position_close_evidence(
        self,
        current_side: str | None,
        close_action: Action,
        exit_votes: list[DecisionOutput],
        risk_vetoes: list[DecisionOutput],
        score: float,
        raw_opinions: list[dict[str, Any]],
        symbol_positions: list[dict] | None = None,
        features: FeatureVector | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Summarize whether an open position has enough evidence to reduce/close."""
        if not current_side:
            return {"should_close": False, "block_reason": "当前没有持仓。"}

        close_value = close_action.value
        opposite_entry = Action.SHORT.value if current_side == "long" else Action.LONG.value
        support = []
        explicit_close = []
        opposite_votes = []
        for opinion in raw_opinions:
            action = str(opinion.get("action") or "")
            confidence = self._safe_float(opinion.get("confidence"), 0.0)
            if confidence < REVIEW_CLOSE_MIN_CONFIDENCE:
                continue
            item = {
                "model_name": opinion.get("model_name"),
                "label": opinion.get("label") or opinion.get("model_name"),
                "action": action,
                "confidence": round(confidence, 4),
            }
            if action == close_value:
                explicit_close.append(item)
                support.append(item)
            elif action == opposite_entry:
                opposite_votes.append(item)
                support.append(item)

        strong_support = [
            item
            for item in support
            if self._safe_float(item.get("confidence"), 0.0) >= REVIEW_CLOSE_STRONG_CONFIDENCE
        ]
        raw_hard_risk = bool(risk_vetoes)
        strong_opposite_pressure = (
            current_side == "long" and score <= -REVIEW_STRONG_OPPOSITE_SCORE
        ) or (current_side == "short" and score >= REVIEW_STRONG_OPPOSITE_SCORE)
        moderate_opposite_pressure = (
            current_side == "long" and score <= -PROFIT_PROTECT_MODERATE_OPPOSITE_SCORE
        ) or (current_side == "short" and score >= PROFIT_PROTECT_MODERATE_OPPOSITE_SCORE)
        max_conf = max([self._safe_float(item.get("confidence"), 0.0) for item in support] or [0.0])
        position_loss = False
        position_risk_usage = 0.0
        position_unrealized_pnl = 0.0
        position_notional = 0.0
        position_profit_pct = 0.0
        planned_risk_price = 0.0
        planned_risk_usdt = 0.0
        age_minutes = 9999.0
        if symbol_positions:
            pos = symbol_positions[0] or {}
            entry_price = self._safe_float(pos.get("entry_price"), 0.0)
            current_price = self._safe_float(pos.get("current_price"), 0.0)
            quantity = abs(self._safe_float(pos.get("quantity"), 0.0))
            stop_loss = self._safe_float(pos.get("stop_loss"), 0.0)
            position_unrealized_pnl = self._safe_float(pos.get("unrealized_pnl"), 0.0)
            position_notional = abs(entry_price * quantity)
            opened_at = pos.get("created_at")
            if isinstance(opened_at, str):
                try:
                    opened_at = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
                except ValueError:
                    opened_at = None
            if isinstance(opened_at, datetime):
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=UTC)
                age_minutes = max((datetime.now(UTC) - opened_at).total_seconds() / 60.0, 0.0)
            if position_notional > 0:
                position_profit_pct = position_unrealized_pnl / position_notional
            if entry_price > 0 and current_price > 0:
                if current_side == "long":
                    adverse_move = max(entry_price - current_price, 0.0)
                    planned_risk = (
                        max(entry_price - stop_loss, entry_price * 0.02)
                        if stop_loss > 0
                        else entry_price * 0.05
                    )
                else:
                    adverse_move = max(current_price - entry_price, 0.0)
                    planned_risk = (
                        max(stop_loss - entry_price, entry_price * 0.02)
                        if stop_loss > 0
                        else entry_price * 0.05
                    )
                planned_risk_price = planned_risk
                planned_risk_usdt = planned_risk * quantity
                position_loss = adverse_move > 0 or position_unrealized_pnl < 0
                if planned_risk > 0:
                    position_risk_usage = min(max(adverse_move / planned_risk, 0.0), 2.0)

        should_close = False
        action_plan = "hold"
        position_size_pct = 0.0
        reason = ""
        block_reason = ""
        suggested_confidence = 0.0
        position_profit = position_unrealized_pnl > 0 and position_profit_pct > 0
        portfolio_profit = self._safe_dict(
            context.get("portfolio_profit_protection") if isinstance(context, dict) else {}
        )
        portfolio_profit_focus = bool(portfolio_profit.get("active"))
        portfolio_current_group = self._safe_dict(portfolio_profit.get("current_group"))
        portfolio_profit_share = self._safe_float(portfolio_current_group.get("profit_share"), 0.0)
        portfolio_threshold_usdt = self._safe_float(portfolio_profit.get("threshold_usdt"), 0.0)
        estimated_fee_usdt = max(position_notional * 0.001, 0.0)
        dynamic_profit_lock_line = min(
            max(
                estimated_fee_usdt * PROFIT_LOCK_FEE_MULTIPLE,
                position_notional * PROFIT_LOCK_NOTIONAL_RATIO,
                planned_risk_usdt * PROFIT_LOCK_RISK_RATIO,
                PROFIT_LOCK_MIN_FLOOR_USDT,
                PROFIT_PROTECT_MIN_LOCK_USDT,
            ),
            PROFIT_LOCK_MAX_FLOOR_USDT,
        )
        meaningful_reduce_lock_line = min(
            max(
                estimated_fee_usdt * PROFIT_LOCK_REDUCE_FEE_MULTIPLE,
                position_notional * PROFIT_LOCK_REDUCE_NOTIONAL_RATIO,
                planned_risk_usdt * PROFIT_LOCK_REDUCE_RISK_RATIO,
                PROFIT_LOCK_MEANINGFUL_REDUCE_USDT,
            ),
            PROFIT_LOCK_MAX_FLOOR_USDT,
        )
        expected_reduce_lock_net = max(
            position_unrealized_pnl * PROFIT_PROTECT_REDUCE_SIZE
            - estimated_fee_usdt * PROFIT_PROTECT_REDUCE_SIZE,
            0.0,
        )
        analysis_profit_exit_floor = max(
            PROFIT_EXIT_ANALYSIS_MIN_FLOOR_USDT,
            estimated_fee_usdt * PROFIT_LOCK_FEE_MULTIPLE,
        )
        profit_floor_ready = position_unrealized_pnl >= analysis_profit_exit_floor
        meaningful_reduce_lock = expected_reduce_lock_net >= meaningful_reduce_lock_line
        portfolio_focus_lock_line = max(
            PORTFOLIO_FOCUS_LOCK_MIN_USDT,
            portfolio_threshold_usdt,
            dynamic_profit_lock_line * 0.75,
        )
        portfolio_focus_profit_lock = (
            position_profit
            and portfolio_profit_focus
            and portfolio_profit_share >= PORTFOLIO_FOCUS_LOCK_MIN_SHARE
            and position_unrealized_pnl >= portfolio_focus_lock_line
            and age_minutes >= 2.0
        )
        strong_profit_lock_line = dynamic_profit_lock_line * 1.45
        full_profit_lock_line = dynamic_profit_lock_line * 2.20
        profit_protect = (
            position_profit
            and position_profit_pct >= PROFIT_PROTECT_REDUCE_PNL_RATIO
            and position_unrealized_pnl >= dynamic_profit_lock_line
            and meaningful_reduce_lock
        )
        strong_profit = (
            position_profit
            and position_profit_pct >= PROFIT_PROTECT_STRONG_PNL_RATIO
            and position_unrealized_pnl >= strong_profit_lock_line
        )
        full_profit = (
            position_profit
            and position_profit_pct >= PROFIT_PROTECT_FULL_PNL_RATIO
            and position_unrealized_pnl >= full_profit_lock_line
        )
        quick_profit_line = dynamic_profit_lock_line * 1.15
        rotation_profit_line = dynamic_profit_lock_line * 1.35
        quick_full_profit_line = dynamic_profit_lock_line * 1.85
        quick_profit = (
            position_profit
            and position_profit_pct >= QUICK_PROFIT_REDUCE_PNL_RATIO
            and position_unrealized_pnl >= quick_profit_line
            and meaningful_reduce_lock
        )
        rotation_profit = (
            position_profit
            and position_profit_pct >= CAPITAL_ROTATION_PROFIT_PNL_RATIO
            and position_unrealized_pnl >= rotation_profit_line
            and meaningful_reduce_lock
        )
        quick_full_profit = (
            position_profit
            and position_profit_pct >= QUICK_PROFIT_FULL_PNL_RATIO
            and position_unrealized_pnl >= quick_full_profit_line
        )
        continuation_score = score if current_side == "long" else -score
        weak_continuation = continuation_score < 0.16
        returns_1 = self._safe_float(getattr(features, "returns_1", 0.0), 0.0) if features else 0.0
        returns_5 = self._safe_float(getattr(features, "returns_5", 0.0), 0.0) if features else 0.0
        volume_ratio = (
            self._safe_float(getattr(features, "volume_ratio", 1.0), 1.0) if features else 1.0
        )
        bb_pct = self._safe_float(getattr(features, "bb_pct", 0.5), 0.5) if features else 0.5
        rsi_14 = self._safe_float(getattr(features, "rsi_14", 50.0), 50.0) if features else 50.0
        low_participation = volume_ratio < 0.55
        if current_side == "long":
            momentum_waning = (
                returns_1 <= -0.001
                or returns_5 <= -0.0025
                or (bb_pct >= 0.82 and rsi_14 >= 66 and low_participation)
            )
        else:
            momentum_waning = (
                returns_1 >= 0.001
                or returns_5 >= 0.0025
                or (bb_pct <= 0.18 and rsi_14 <= 34 and low_participation)
            )
        predictive_reversal = self._predictive_reversal_evidence(
            current_side=current_side,
            features=features,
        )
        predictive_reversal_score = self._safe_float(predictive_reversal.get("score"), 0.0)
        predictive_exit = predictive_reversal_score >= PREDICTIVE_REVERSAL_EXIT_SCORE
        predictive_full_exit = predictive_reversal_score >= PREDICTIVE_REVERSAL_FULL_EXIT_SCORE
        predictive_reduce_lock_line = meaningful_reduce_lock_line / max(
            PREDICTIVE_REVERSAL_REDUCE_SIZE, 0.01
        )
        predictive_full_lock_line = max(dynamic_profit_lock_line, meaningful_reduce_lock_line)
        predictive_reduce_lock_ready = (
            meaningful_reduce_lock and position_unrealized_pnl >= predictive_reduce_lock_line
        )
        peak_context = context.get("position_profit_peak") if isinstance(context, dict) else {}
        if not isinstance(peak_context, dict):
            peak_context = {}
        peak_unrealized_pnl = self._safe_float(
            peak_context.get("peak_unrealized_pnl", peak_context.get("peak_pnl")),
            0.0,
        )
        profit_retrace_abs = max(peak_unrealized_pnl - position_unrealized_pnl, 0.0)
        profit_retrace_ratio = (
            profit_retrace_abs / max(peak_unrealized_pnl, 1e-9) if peak_unrealized_pnl > 0 else 0.0
        )
        predictive_full_lock_ready = (
            predictive_full_exit
            and position_unrealized_pnl >= predictive_full_lock_line
            and profit_retrace_ratio >= 0.18
        )
        volatility_20 = (
            self._safe_float(getattr(features, "volatility_20", 0.0), 0.0) if features else 0.0
        )
        dynamic_retrace_reduce_ratio = PROFIT_RETRACE_BASE_REDUCE_RATIO
        dynamic_retrace_reduce_ratio += min(max(volatility_20, 0.0), 0.08) * 1.25
        dynamic_retrace_reduce_ratio += max(continuation_score, 0.0) * 0.10
        if momentum_waning or weak_continuation or moderate_opposite_pressure:
            dynamic_retrace_reduce_ratio -= 0.08
        dynamic_retrace_reduce_ratio = min(
            max(dynamic_retrace_reduce_ratio, PROFIT_RETRACE_MIN_REDUCE_RATIO),
            PROFIT_RETRACE_MAX_REDUCE_RATIO,
        )
        dynamic_retrace_full_ratio = min(
            max(
                dynamic_retrace_reduce_ratio + 0.20,
                PROFIT_RETRACE_MIN_FULL_RATIO,
            ),
            PROFIT_RETRACE_MAX_FULL_RATIO,
        )
        profit_retrace_peak_line = dynamic_profit_lock_line * PROFIT_RETRACE_PEAK_LINE_MULTIPLE
        profit_retrace_current_line = (
            dynamic_profit_lock_line * PROFIT_RETRACE_CURRENT_LINE_MULTIPLE
        )
        profit_retrace_protection = (
            position_profit
            and peak_unrealized_pnl >= profit_retrace_peak_line
            and position_unrealized_pnl >= profit_retrace_current_line
            and profit_retrace_ratio >= dynamic_retrace_reduce_ratio
            and (
                position_unrealized_pnl >= meaningful_reduce_lock_line / 0.60
                or profit_retrace_ratio >= dynamic_retrace_full_ratio
            )
        )
        profit_lock_ready_for_exit = bool(
            profit_floor_ready
            and (
                profit_protect
                or strong_profit
                or full_profit
                or quick_profit
                or rotation_profit
                or quick_full_profit
                or profit_retrace_protection
                or predictive_reduce_lock_ready
                or predictive_full_lock_ready
                or portfolio_focus_profit_lock
            )
        )
        abnormal_wick_max_pct = (
            self._safe_float(getattr(features, "abnormal_wick_max_pct", 0.0), 0.0)
            if features
            else 0.0
        )
        abnormal_wick_count = (
            int(self._safe_float(getattr(features, "abnormal_wick_count_72h", 0.0), 0.0))
            if features
            else 0
        )
        enough_age_for_fast_profit = age_minutes >= FAST_PROFIT_MIN_HOLD_MINUTES
        early_discretionary_close = age_minutes < MIN_DISCRETIONARY_CLOSE_HOLD_MINUTES
        fresh_position_noise = (
            age_minutes < 2.0 and position_risk_usage < EARLY_CLOSE_MIN_RISK_USAGE
        )
        hard_risk = (
            raw_hard_risk
            and position_risk_usage >= LOSS_FULL_MIN_RISK_USAGE
            and not fresh_position_noise
        )
        if (
            raw_hard_risk
            and not hard_risk
            and abnormal_wick_count > 0
            and abnormal_wick_max_pct >= 12.0
        ):
            hard_risk = True
        early_loss_exit_allowed = (
            position_loss
            and position_risk_usage >= EARLY_CLOSE_MIN_RISK_USAGE
            and (len(strong_support) >= 2 or strong_opposite_pressure)
        )
        early_profit_exit_allowed = (
            position_profit
            and position_profit_pct >= QUICK_PROFIT_FULL_PNL_RATIO
            and (len(strong_support) >= 2 or strong_opposite_pressure)
        )
        dynamic_loss_reduce_usdt = max(
            LOSS_COMPRESS_REDUCE_USDT,
            position_notional * LOSS_COMPRESS_REDUCE_RATIO,
            planned_risk_usdt * LOSS_COMPRESS_REDUCE_RISK_RATIO,
        )
        dynamic_loss_full_usdt = max(
            LOSS_COMPRESS_FULL_USDT,
            position_notional * LOSS_COMPRESS_FULL_RATIO,
            planned_risk_usdt * LOSS_COMPRESS_FULL_RISK_RATIO,
        )
        loss_abs = abs(min(position_unrealized_pnl, 0.0))
        loss_repair = self._loss_repair_evidence(
            current_side=current_side,
            score=score,
            raw_opinions=raw_opinions,
            context=context,
            features=features,
            position_unrealized_pnl=position_unrealized_pnl,
            position_profit_pct=position_profit_pct,
            position_risk_usage=position_risk_usage,
            loss_abs=loss_abs,
            dynamic_loss_reduce_usdt=dynamic_loss_reduce_usdt,
            dynamic_loss_full_usdt=dynamic_loss_full_usdt,
            support_count=len(support),
            strong_support_count=len(strong_support),
            strong_opposite_pressure=strong_opposite_pressure,
            moderate_opposite_pressure=moderate_opposite_pressure,
            momentum_waning=momentum_waning,
        )

        if hard_risk:
            should_close = True
            action_plan = "full_close"
            position_size_pct = 1.0
            reason = "风控专家触发硬风险，且当前仓位已有足够真实风险证据，立即全平。"
            suggested_confidence = max(max_conf, 0.82)
        elif (
            position_profit
            and predictive_exit
            and (predictive_reduce_lock_ready or predictive_full_lock_ready)
            and (profit_retrace_ratio >= 0.08 or momentum_waning or moderate_opposite_pressure)
        ):
            should_close = True
            action_plan = "full_close" if predictive_full_lock_ready else "reduce"
            position_size_pct = (
                1.0 if action_plan == "full_close" else PREDICTIVE_REVERSAL_REDUCE_SIZE
            )
            reason = (
                f"预判型锁盈：当前仍有浮盈 {position_unrealized_pnl:.2f}U，"
                f"但短周期反向风险评分 {predictive_reversal_score:.0f}，"
                "继续持有的期望收益已经下降，先把账面利润转成已实现利润。"
            )
            suggested_confidence = max(max_conf, 0.70 if action_plan == "full_close" else 0.64)
        elif loss_repair.get("should_close"):
            should_close = True
            action_plan = str(loss_repair.get("action_plan") or "reduce")
            position_size_pct = float(loss_repair.get("position_size_pct") or 0.60)
            reason = str(
                loss_repair.get("reason") or "亏损修复评估显示扩亏概率更高，先处理亏损仓位。"
            )
            suggested_confidence = max(
                max_conf,
                0.74 if action_plan == "full_close" else 0.66,
            )
        elif (
            position_loss
            and predictive_exit
            and not bool(loss_repair.get("repair_possible"))
            and (position_risk_usage >= 0.25 or loss_abs >= dynamic_loss_reduce_usdt * 0.55)
        ):
            should_close = True
            action_plan = (
                "full_close"
                if predictive_full_exit or loss_abs >= dynamic_loss_full_usdt * 0.70
                else "hold"
            )
            position_size_pct = 1.0 if action_plan == "full_close" else 0.0
            reason = (
                f"亏损修复预判：当前浮亏 {position_unrealized_pnl:.2f}U，"
                f"反向风险评分 {predictive_reversal_score:.0f}，且修复证据不足；"
                "先减仓/平仓，避免小亏拖成大亏。"
            )
            if action_plan == "hold":
                should_close = False
                block_reason = reason
            else:
                suggested_confidence = max(max_conf, 0.72)
        elif (
            position_loss
            and loss_abs >= dynamic_loss_full_usdt
            and (position_risk_usage >= 0.45 or bool(support) or strong_opposite_pressure)
        ):
            should_close = True
            action_plan = "full_close"
            position_size_pct = 1.0
            reason = (
                f"亏损压缩：当前浮亏 {position_unrealized_pnl:.2f}U / {position_profit_pct * 100:.2f}%，"
                f"已经超过动态全平线 {dynamic_loss_full_usdt:.2f}U，优先全平避免小赚大亏继续扩大。"
            )
            suggested_confidence = max(max_conf, 0.74)
        elif (
            position_loss
            and loss_abs >= dynamic_loss_reduce_usdt
            and (position_risk_usage >= 0.30 or bool(support) or moderate_opposite_pressure)
        ):
            block_reason = (
                f"亏损压缩：当前浮亏 {position_unrealized_pnl:.2f}U / {position_profit_pct * 100:.2f}%，"
                f"已经超过动态减仓线 {dynamic_loss_reduce_usdt:.2f}U，先减仓 60%，降低单笔亏损继续扩大的概率。"
            )
            suggested_confidence = max(max_conf, 0.66)
        elif (
            early_discretionary_close
            and position_risk_usage < 1.0
            and not early_loss_exit_allowed
            and not early_profit_exit_allowed
        ):
            block_reason = (
                f"仓位仍处于早期验证阶段，已持仓 {age_minutes:.1f} 分钟；"
                "当前退出证据还不够强，暂不因普通波动、单个专家分歧或小幅浮盈主动平仓。"
            )
        elif position_loss and position_risk_usage >= 1.0:
            should_close = True
            action_plan = "full_close"
            position_size_pct = 1.0
            reason = f"浮亏已经达到或超过计划止损风险的 {position_risk_usage * 100:.0f}%，执行全平，避免继续拖延。"
            suggested_confidence = max(max_conf, 0.72)
        elif (
            position_loss
            and position_risk_usage >= LOSS_FULL_MIN_RISK_USAGE
            and (len(strong_support) >= 2 or strong_opposite_pressure)
        ):
            should_close = True
            action_plan = "full_close"
            position_size_pct = 1.0
            reason = f"浮亏已接近止损风险的 {position_risk_usage * 100:.0f}%，且出现反向压力，执行全平避免亏损扩大。"
            suggested_confidence = max(max_conf, 0.72)
        elif (
            position_loss
            and position_risk_usage >= LOSS_REDUCE_MIN_RISK_USAGE
            and (len(strong_support) >= 2 or strong_opposite_pressure)
        ):
            should_close = False
            action_plan = "hold"
            position_size_pct = 0.0
            reason = f"浮亏已达到止损风险的 {position_risk_usage * 100:.0f}%，且已有退出线索，先减仓 50%，保留后续确认空间。"
            suggested_confidence = max(max_conf, 0.66)
        elif (
            False
            and position_loss
            and position_risk_usage >= 0.45
            and (support or strong_opposite_pressure)
        ):
            should_close = True
            action_plan = "reduce"
            position_size_pct = 0.5
            reason = f"浮亏已达到止损风险的 {position_risk_usage * 100:.0f}%，且出现反向压力，先减仓 50%。"
            suggested_confidence = max(max_conf, 0.62)
        elif (
            position_profit
            and profit_protect
            and (
                portfolio_profit_focus
                or momentum_waning
                or weak_continuation
                or support
                or moderate_opposite_pressure
            )
        ):
            should_close = True
            action_plan = "reduce"
            position_size_pct = PROFIT_PROTECT_REDUCE_SIZE
            reason = (
                f"主动锁盈：当前浮盈 {position_unrealized_pnl:.2f}U / {position_profit_pct * 100:.2f}%，"
                "已覆盖手续费和动态利润保护线；为避免浮盈回吐，先减仓锁定一部分已实现利润。"
            )
            suggested_confidence = max(max_conf, 0.62)
        elif portfolio_focus_profit_lock:
            should_close = True
            action_plan = "reduce"
            position_size_pct = min(
                max(
                    PORTFOLIO_FOCUS_LOCK_REDUCE_SIZE,
                    PORTFOLIO_FOCUS_LOCK_MIN_USDT / max(position_unrealized_pnl, 1e-9),
                ),
                0.70,
            )
            reason = (
                f"组合收益保护锁盈：账户总浮盈已达到保护线，该仓位当前浮盈 {position_unrealized_pnl:.2f}U，"
                f"贡献占比 {portfolio_profit_share * 100:.0f}%，已超过组合高贡献锁盈线 {portfolio_focus_lock_line:.2f}U；"
                f"按 {position_size_pct * 100:.0f}% 动态比例先锁定一部分利润，剩余仓位继续让趋势奔跑。"
            )
            suggested_confidence = max(max_conf, 0.68)
        elif profit_retrace_protection:
            should_close = True
            full_retrace = profit_retrace_ratio >= dynamic_retrace_full_ratio
            action_plan = "full_close" if full_retrace else "reduce"
            position_size_pct = 1.0 if full_retrace else 0.60
            reason = (
                f"浮盈回撤保护：该仓位最高浮盈约 {peak_unrealized_pnl:.2f}U，"
                f"当前回落到 {position_unrealized_pnl:.2f}U，回撤 {profit_retrace_ratio * 100:.0f}%。"
                "为避免已获得利润继续回吐，优先把账面利润转为已实现利润。"
            )
            suggested_confidence = max(max_conf, 0.70 if full_retrace else 0.64)
        elif (
            position_profit
            and position_profit_pct >= WINNER_RUN_MIN_PROFIT_RATIO
            and continuation_score >= 0.24
            and not strong_opposite_pressure
            and len(strong_support) < 2
            and not momentum_waning
            and not portfolio_profit_focus
        ):
            block_reason = (
                f"盈利仓继续奔跑：当前浮盈 {position_unrealized_pnl:.2f}U / {position_profit_pct * 100:.2f}%，"
                f"延续分 {continuation_score:.2f}，未出现强反向确认；不因小盈利或单个退出意见过早减仓。"
            )
        elif (
            quick_full_profit
            and enough_age_for_fast_profit
            and (support or moderate_opposite_pressure)
        ):
            should_close = True
            action_plan = "full_close"
            position_size_pct = 1.0
            reason = (
                f"浮盈已达到仓位名义价值的 {position_profit_pct * 100:.2f}%，"
                "且继续持有优势下降，优先全平把账面利润转为已实现利润。"
            )
            suggested_confidence = max(max_conf, 0.66)
        elif (
            rotation_profit
            and enough_age_for_fast_profit
            and (support or moderate_opposite_pressure)
            and low_participation
        ):
            should_close = True
            action_plan = "reduce"
            position_size_pct = 0.70
            reason = (
                f"浮盈 {position_profit_pct * 100:.2f}% 已覆盖资金占用成本，"
                "但成交参与度低且继续持有分数不足，先平 70% 释放资金寻找更高优势机会。"
            )
            suggested_confidence = max(max_conf, 0.62)
        elif (
            quick_profit and enough_age_for_fast_profit and (support or moderate_opposite_pressure)
        ):
            should_close = True
            action_plan = "reduce"
            position_size_pct = PROFIT_PROTECT_REDUCE_SIZE
            reason = (
                f"浮盈已达到仓位名义价值的 {position_profit_pct * 100:.2f}%，"
                "且动能/专家意见不再支持继续无保护持有，先部分止盈落袋。"
            )
            suggested_confidence = max(max_conf, 0.58)
        elif full_profit and (len(strong_support) >= 2 or strong_opposite_pressure):
            should_close = True
            action_plan = "full_close"
            position_size_pct = 1.0
            reason = (
                f"浮盈已达到仓位名义价值的 {position_profit_pct * 100:.2f}%，"
                "且出现强退出确认，执行全平锁定利润。"
            )
            suggested_confidence = max(max_conf, 0.72)
        elif strong_profit and (len(support) >= 2 or strong_opposite_pressure):
            should_close = True
            action_plan = "reduce"
            position_size_pct = 0.5
            reason = (
                f"浮盈已达到仓位名义价值的 {position_profit_pct * 100:.2f}%，"
                "且已有多专家/强反向压力确认，先减仓 50% 保护利润。"
            )
            suggested_confidence = max(max_conf, 0.66)
        elif profit_protect and (support or moderate_opposite_pressure):
            should_close = True
            action_plan = "reduce"
            position_size_pct = PROFIT_PROTECT_REDUCE_SIZE
            reason = (
                f"浮盈已达到仓位名义价值的 {position_profit_pct * 100:.2f}%，"
                "同时出现退出线索或趋势转弱，先减仓 35% 锁定一部分利润。"
            )
            suggested_confidence = max(max_conf, 0.60)
        elif (
            position_profit
            and not profit_lock_ready_for_exit
            and (support or strong_opposite_pressure or moderate_opposite_pressure)
        ):
            should_close = False
            action_plan = "hold"
            position_size_pct = 0.0
            block_reason = (
                f"盈利仓位暂未达到有效锁盈线：当前浮盈 {position_unrealized_pnl:.2f}U，"
                f"动态锁盈线 {dynamic_profit_lock_line:.2f}U，部分锁盈有效线 {meaningful_reduce_lock_line:.2f}U；"
                "专家退出意见只作为风险提示，不生成平仓决策。等待更大浮盈、明确回撤保护、硬止损/止盈或严重趋势失效。"
            )
        elif (
            not position_loss
            and len(support) >= FULL_CLOSE_SUPPORT
            and (len(strong_support) >= 2 or strong_opposite_pressure)
            and (strong_opposite_pressure or profit_protect)
        ):
            should_close = True
            action_plan = "full_close"
            position_size_pct = 1.0
            reason = f"{len(support)} 个专家支持退出，且有强确认，执行全平。"
            suggested_confidence = max(max_conf, 0.68)
        elif (
            not position_loss
            and len(support) >= MIN_REVIEW_CLOSE_SUPPORT
            and (strong_support or strong_opposite_pressure)
            and (strong_opposite_pressure or profit_protect)
        ):
            should_close = True
            action_plan = "reduce"
            position_size_pct = 0.5
            reason = f"{len(support)} 个专家支持退出，但未达到全平门槛，先减仓 50%。"
            suggested_confidence = max(max_conf, 0.62)
        elif (
            not position_loss
            and len(support) == 1
            and strong_opposite_pressure
            and max_conf >= 0.72
        ):
            should_close = True
            action_plan = "reduce"
            position_size_pct = 0.5
            reason = "单个高置信退出信号叠加强反向压力，先减仓 50%。"
            suggested_confidence = max(max_conf, 0.72)
        else:
            block_reason = (
                f"当前只有 {len(support)} 个有效退出/反向支持，"
                f"强支持 {len(strong_support)} 个，未达到至少 {MIN_REVIEW_CLOSE_SUPPORT} 个专家确认。"
            )

        return {
            "should_close": should_close,
            "action_plan": action_plan,
            "position_size_pct": position_size_pct,
            "reason": reason,
            "block_reason": block_reason,
            "support_count": len(support),
            "strong_support_count": len(strong_support),
            "explicit_close_votes": explicit_close,
            "opposite_entry_votes": opposite_votes,
            "raw_hard_risk": raw_hard_risk,
            "hard_risk": hard_risk,
            "fresh_position_noise": bool(fresh_position_noise),
            "strong_opposite_pressure": strong_opposite_pressure,
            "moderate_opposite_pressure": moderate_opposite_pressure,
            "max_support_confidence": round(max_conf, 4),
            "suggested_confidence": round(suggested_confidence, 4),
            "position_loss": position_loss,
            "position_risk_usage": round(position_risk_usage, 4),
            "position_notional": round(position_notional, 6),
            "position_profit": position_profit,
            "position_profit_pct": round(position_profit_pct, 6),
            "profit_protection": bool(
                should_close and position_profit and profit_lock_ready_for_exit
            ),
            "profit_lock_ready_for_exit": bool(profit_lock_ready_for_exit),
            "profit_floor_ready_for_exit": bool(profit_floor_ready),
            "analysis_profit_exit_floor_usdt": round(analysis_profit_exit_floor, 6),
            "portfolio_profit_focus": bool(portfolio_profit_focus),
            "portfolio_focus_profit_lock": bool(portfolio_focus_profit_lock),
            "portfolio_profit_share": round(portfolio_profit_share, 6),
            "portfolio_focus_lock_line_usdt": round(portfolio_focus_lock_line, 6),
            "portfolio_focus_min_share": PORTFOLIO_FOCUS_LOCK_MIN_SHARE,
            "portfolio_focus_reduce_size": PORTFOLIO_FOCUS_LOCK_REDUCE_SIZE,
            "position_unrealized_pnl": round(position_unrealized_pnl, 6),
            "age_minutes": round(age_minutes, 3),
            "continuation_score": round(continuation_score, 4),
            "weak_continuation": bool(weak_continuation),
            "momentum_waning": bool(momentum_waning),
            "predictive_reversal": predictive_reversal,
            "predictive_reversal_score": round(predictive_reversal_score, 4),
            "predictive_exit": bool(predictive_exit),
            "low_participation": bool(low_participation),
            "quick_profit": bool(quick_profit),
            "capital_rotation_profit": bool(rotation_profit),
            "profit_retrace_protection": bool(profit_retrace_protection),
            "dynamic_profit_lock_line_usdt": round(dynamic_profit_lock_line, 6),
            "meaningful_reduce_lock_line_usdt": round(meaningful_reduce_lock_line, 6),
            "expected_reduce_lock_net_usdt": round(expected_reduce_lock_net, 6),
            "meaningful_reduce_lock": bool(meaningful_reduce_lock),
            "strong_profit_lock_line_usdt": round(strong_profit_lock_line, 6),
            "full_profit_lock_line_usdt": round(full_profit_lock_line, 6),
            "quick_profit_line_usdt": round(quick_profit_line, 6),
            "rotation_profit_line_usdt": round(rotation_profit_line, 6),
            "quick_full_profit_line_usdt": round(quick_full_profit_line, 6),
            "peak_unrealized_pnl": round(peak_unrealized_pnl, 6),
            "profit_retrace_abs": round(profit_retrace_abs, 6),
            "profit_retrace_ratio": round(profit_retrace_ratio, 6),
            "profit_retrace_peak_line_usdt": round(profit_retrace_peak_line, 6),
            "profit_retrace_current_line_usdt": round(profit_retrace_current_line, 6),
            "dynamic_retrace_reduce_ratio": round(dynamic_retrace_reduce_ratio, 6),
            "dynamic_retrace_full_ratio": round(dynamic_retrace_full_ratio, 6),
            "winner_run_protected": bool(
                position_profit
                and position_profit_pct >= WINNER_RUN_MIN_PROFIT_RATIO
                and continuation_score >= 0.24
                and not strong_opposite_pressure
                and len(strong_support) < 2
                and not momentum_waning
            ),
            "planned_risk_price": round(planned_risk_price, 8),
            "planned_risk_usdt": round(planned_risk_usdt, 6),
            "loss_repair_evidence": loss_repair,
            "loss_compress_reduce_line": round(dynamic_loss_reduce_usdt, 6),
            "loss_compress_full_line": round(dynamic_loss_full_usdt, 6),
            "loss_compress_formula": (
                "减仓线=max(3U, 名义价值0.6%, 计划风险35%); "
                "全平线=max(8U, 名义价值1.2%, 计划风险65%)"
            ),
        }

    def _symbol_positions(self, symbol: str, context: dict[str, Any]) -> list[dict]:
        def normalize(value: Any) -> str:
            text = str(value or "").strip().upper()
            if not text:
                return ""
            text = text.split(":")[0]
            if "/" not in text and text.endswith("USDT"):
                base = text[:-4]
                if base:
                    text = f"{base}/USDT"
            return text

        target = normalize(symbol)
        rows = [
            p
            for p in context.get("open_positions", [])
            if (
                p.get("model_name") == ENSEMBLE_TRADER_NAME
                and normalize(p.get("symbol")) == target
                and p.get("is_open", True) is not False
            )
        ]
        if len(rows) <= 1:
            return rows

        grouped: dict[str, list[dict]] = {}
        for pos in rows:
            side = str(pos.get("side") or "").lower()
            if side in {"long", "short"}:
                grouped.setdefault(side, []).append(pos)
        if not grouped:
            return rows

        aggregates: list[dict] = []
        for side, side_rows in grouped.items():
            total_notional = 0.0
            total_synthetic_qty = 0.0
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
            for pos in side_rows:
                entry = self._safe_float(
                    pos.get("entry_price") or pos.get("entryPrice") or pos.get("avgPx"), 0.0
                )
                current = self._safe_float(
                    pos.get("current_price")
                    or pos.get("markPrice")
                    or pos.get("lastPrice")
                    or pos.get("entry_price")
                    or pos.get("entryPrice")
                    or pos.get("avgPx"),
                    entry,
                )
                qty = abs(
                    self._safe_float(
                        pos.get("quantity") or pos.get("contracts") or pos.get("sz"), 0.0
                    )
                )
                contract_size = self._safe_float(
                    pos.get("contract_size")
                    or pos.get("contractSize")
                    or (pos.get("info") or {}).get("ctVal"),
                    1.0,
                )
                direct_notional = abs(
                    self._safe_float(
                        pos.get("notional")
                        or pos.get("notional_usd")
                        or pos.get("notionalUsd")
                        or (pos.get("info") or {}).get("notionalUsd")
                        or (pos.get("info") or {}).get("notional")
                        or (pos.get("info") or {}).get("posValue"),
                        0.0,
                    )
                )
                if entry <= 0 or current <= 0:
                    continue
                notional = (
                    direct_notional
                    if direct_notional > 0
                    else qty * entry * (contract_size if contract_size > 0 else 1.0)
                )
                if notional <= 0:
                    continue
                synthetic_qty = notional / entry
                total_notional += notional
                total_synthetic_qty += synthetic_qty
                entry_value += entry * synthetic_qty
                current_value += current * synthetic_qty
                pnl_value = self._safe_float(
                    pos.get("unrealized_pnl", pos.get("unrealizedPnl", 0.0)), 0.0
                )
                if pnl_value == 0.0:
                    pnl_value = (
                        (current - entry) * synthetic_qty
                        if side == "long"
                        else (entry - current) * synthetic_qty
                    )
                unrealized += pnl_value
                stop = self._safe_float(pos.get("stop_loss") or pos.get("stop_loss_price"), 0.0)
                if stop > 0:
                    stop_value += stop * synthetic_qty
                    stop_weight += synthetic_qty
                take_profit = self._safe_float(
                    pos.get("take_profit") or pos.get("take_profit_price"), 0.0
                )
                if take_profit > 0:
                    take_profit_value += take_profit * synthetic_qty
                    take_profit_weight += synthetic_qty
                leverage = self._safe_float(pos.get("leverage"), 0.0)
                if leverage > 0:
                    leverage_value += leverage * synthetic_qty
                    leverage_weight += synthetic_qty
                opened = pos.get("created_at") or pos.get("opened_at")
                if created_at is None:
                    created_at = opened

            if total_notional <= 0 or total_synthetic_qty <= 0 or entry_value <= 0:
                continue
            entry_price = entry_value / total_synthetic_qty
            synthetic_quantity = total_synthetic_qty
            current_price = current_value / total_synthetic_qty
            aggregates.append(
                {
                    **side_rows[0],
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "quantity": synthetic_quantity,
                    "notional": total_notional,
                    "unrealized_pnl": unrealized,
                    "stop_loss": (
                        stop_value / stop_weight
                        if stop_weight > 0
                        else side_rows[0].get("stop_loss")
                    ),
                    "take_profit": (
                        take_profit_value / take_profit_weight
                        if take_profit_weight > 0
                        else side_rows[0].get("take_profit")
                    ),
                    "leverage": (
                        leverage_value / leverage_weight
                        if leverage_weight > 0
                        else side_rows[0].get("leverage", 1.0)
                    ),
                    "created_at": created_at,
                    "aggregate_position": True,
                    "fragment_count": len(side_rows),
                }
            )

        if not aggregates:
            return rows
        aggregates.sort(key=lambda p: self._safe_float(p.get("notional"), 0.0), reverse=True)
        return aggregates

    def _exit_matches_open_position(
        self, action: Action, symbol: str, context: dict[str, Any]
    ) -> bool:
        if action not in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
            return True
        target_side = "long" if action == Action.CLOSE_LONG else "short"
        return any(
            p.get("side") == target_side
            for p in self._symbol_positions(symbol, context if isinstance(context, dict) else {})
        )

    def _disagreement(self, decisions) -> float:
        scores = [
            ACTION_SCORE.get(d.action, 0.0) for d in decisions if isinstance(d, DecisionOutput)
        ]
        if not scores:
            return 1.0
        positive = sum(1 for s in scores if s > 0)
        negative = sum(1 for s in scores if s < 0)
        directional = positive + negative
        if directional == 0:
            return 0.0
        return min(positive, negative) / directional

    def _hold(self, features: FeatureVector, reason: str, raw: dict[str, Any]) -> DecisionOutput:
        return DecisionOutput(
            model_name=ENSEMBLE_TRADER_NAME,
            symbol=features.symbol,
            action=Action.HOLD,
            confidence=0.0,
            reasoning=reason,
            position_size_pct=0.0,
            suggested_leverage=1.0,
            raw_response=raw,
            feature_snapshot=features.to_dict(),
        )

    def _market_exit_guard_hold(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        decision: DecisionOutput,
        stage: str,
    ) -> DecisionOutput:
        if (
            context.get("review_positions")
            or not isinstance(decision, DecisionOutput)
            or not decision.is_exit
        ):
            return decision
        raw = dict(decision.raw_response or {}) if isinstance(decision.raw_response, dict) else {}
        raw["market_exit_guard"] = {
            "applied": True,
            "stage": stage,
            "original_action": decision.action.value,
            "reason": "market_analysis_close_forbidden",
            "policy": "market analysis may only output long/short/hold; close actions are reserved for position review",
        }
        return self._hold(
            features,
            "市场分析阶段禁止生成平仓裁决；平仓只允许由持仓分析在确认本地/OKX持仓后产生。本轮改为观望。",
            raw,
        )

    def _raw(
        self,
        opinions: list[dict[str, Any]],
        score: float,
        disagreement: float,
        cross_validations: list[dict[str, Any]] | None = None,
        consultation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "ensemble": True,
            "weighted_score": round(score, 4),
            "disagreement": round(disagreement, 4),
            "validation_adjustment": round(
                self._validation_adjustment(cross_validations or [], consultation), 4
            ),
            "opinions": opinions,
            "dynamic_expert_weights": {
                str(o.get("model_name")): {
                    "base_weight": o.get("base_weight", o.get("weight")),
                    "multiplier": o.get("dynamic_weight_multiplier", 1.0),
                    "effective_weight": o.get("effective_weight"),
                    "reason": o.get("dynamic_weight_reason"),
                }
                for o in opinions
                if o.get("model_name")
            },
            "expert_weight_policy": self._expert_weight_policy_summary(opinions),
            "risk_expert_policy": self._risk_policy_from_opinions(opinions),
            "cross_validations": cross_validations or [],
            "consultation": consultation,
            "conflict_resolution": self._conflict_resolution_payload(
                opinions,
                score,
                disagreement,
                cross_validations or [],
                consultation,
            ),
        }

    def _latency_summary(
        self,
        timing_records: list[dict[str, Any]],
        model_timings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        stage_rows = [
            row
            for row in timing_records
            if isinstance(row, dict) and row.get("duration_sec") is not None
        ]
        model_rows = [
            row
            for row in model_timings
            if isinstance(row, dict) and row.get("duration_sec") is not None
        ]
        slowest_stage = max(
            stage_rows,
            key=lambda row: float(row.get("duration_sec") or 0.0),
            default=None,
        )
        slowest_model = max(
            model_rows,
            key=lambda row: float(row.get("duration_sec") or 0.0),
            default=None,
        )
        shared_batch_durations: dict[tuple[Any, ...], float] = {}
        non_shared_model_duration = 0.0
        raw_model_duration = 0.0
        for row in model_rows:
            duration = float(row.get("duration_sec") or 0.0)
            raw_model_duration += duration
            if row.get("shared_batch_call") or row.get("batch_expert"):
                key = (
                    row.get("stage"),
                    row.get("started_at"),
                    row.get("provider_model"),
                    row.get("duration_kind"),
                    row.get("duration_sec"),
                )
                shared_batch_durations[key] = max(shared_batch_durations.get(key, 0.0), duration)
            else:
                non_shared_model_duration += duration
        shared_batch_total = sum(shared_batch_durations.values())
        effective_model_duration = non_shared_model_duration + shared_batch_total
        return {
            "stage_duration_sec": round(
                sum(float(row.get("duration_sec") or 0.0) for row in stage_rows),
                3,
            ),
            "model_duration_sec": round(
                effective_model_duration,
                3,
            ),
            "raw_model_duration_sec": round(raw_model_duration, 3),
            "shared_batch_total_duration_sec": round(shared_batch_total, 3),
            "shared_batch_call_count": len(shared_batch_durations),
            "shared_batch_duration_sec": round(
                max(
                    (
                        float(row.get("duration_sec") or 0.0)
                        for row in model_rows
                        if row.get("shared_batch_call") or row.get("batch_expert")
                    ),
                    default=0.0,
                ),
                3,
            ),
            "uses_shared_batch_call": any(
                bool(row.get("shared_batch_call") or row.get("batch_expert")) for row in model_rows
            ),
            "slowest_stage": (
                {
                    "stage": slowest_stage.get("stage"),
                    "label": slowest_stage.get("label"),
                    "duration_sec": slowest_stage.get("duration_sec"),
                }
                if slowest_stage
                else None
            ),
            "slowest_model": (
                {
                    "name": slowest_model.get("name"),
                    "duration_sec": slowest_model.get("duration_sec"),
                    "provider_model": slowest_model.get("provider_model"),
                    "status": slowest_model.get("status"),
                }
                if slowest_model
                else None
            ),
        }

    def _validation_adjustment(
        self,
        cross_validations: list[dict[str, Any]],
        consultation: dict[str, Any] | None,
    ) -> float:
        adjustment = (
            sum(float(v.get("confidence_adjustment", 0) or 0) for v in cross_validations) / 100.0
        )
        if isinstance(consultation, dict) and consultation.get("status") == "completed":
            adjustment += float(consultation.get("confidence_adjustment", 0) or 0) / 100.0
        return min(max(adjustment, -0.50), 0.50)

    def _memory_adjustment(self, context: dict[str, Any], score: float) -> float:
        """Apply relevant long-term memories without allowing them to flip direction."""
        memories = self._directional_memories(context, score)
        if not memories:
            return 0.0
        adjustment = 0.0
        for memory in memories:
            try:
                confidence_score = float(memory.get("confidence_score", 0.5) or 0.5)
                evidence = min(int(memory.get("evidence_count", 1) or 1), 6)
                adjustment += (
                    float(memory.get("confidence_adjustment", 0.0) or 0.0)
                    * confidence_score
                    * (1 + evidence * 0.08)
                )
            except (TypeError, ValueError):
                continue
        return min(max(adjustment, -0.25), 0.12)

    def _memory_size_multiplier(self, context: dict[str, Any], action: Action) -> float:
        memories = self._directional_memories(context, 1.0 if action == Action.LONG else -1.0)
        if not memories:
            return 1.0
        multiplier = 1.0
        for memory in memories:
            try:
                multiplier = min(
                    multiplier, float(memory.get("position_size_multiplier", 1.0) or 1.0)
                )
            except (TypeError, ValueError):
                continue
        return min(max(multiplier, 0.25), 1.15)

    def _directional_memories(self, context: dict[str, Any], score: float) -> list[dict[str, Any]]:
        memories = [
            item for item in (context.get("expert_memories_flat") or []) if isinstance(item, dict)
        ]
        if not memories or abs(score) < 1e-9:
            return memories[:8]
        side = "long" if score > 0 else "short"
        return [
            memory
            for memory in memories
            if not memory.get("side") or str(memory.get("side")).lower() == side
        ][:8]

    def _memory_summary(self, context: dict[str, Any], score: float) -> dict[str, Any]:
        memories = self._directional_memories(context, score)
        negative = [m for m in memories if float(m.get("confidence_adjustment", 0.0) or 0.0) < 0]
        positive = [m for m in memories if float(m.get("confidence_adjustment", 0.0) or 0.0) > 0]
        return {
            "used": len(memories),
            "risk_lessons": len(negative),
            "positive_lessons": len(positive),
            "feedback": self._memory_feedback(context),
            "top_lessons": [
                {
                    "expert_label": m.get("expert_label") or m.get("expert_name"),
                    "lesson": str(m.get("lesson") or "")[:120],
                    "confidence_adjustment": m.get("confidence_adjustment"),
                    "position_size_multiplier": m.get("position_size_multiplier"),
                }
                for m in memories[:3]
            ],
        }

    @staticmethod
    def _memory_feedback(context: dict[str, Any]) -> dict[str, Any]:
        feedback = context.get("memory_feedback") if isinstance(context, dict) else {}
        return feedback if isinstance(feedback, dict) else {}

    def _score_after_validation(self, score: float, validation_adjustment: float) -> float:
        """Apply cross-validation confidence to direction without flipping sides."""
        if abs(score) < 1e-9 or abs(validation_adjustment) < 1e-9:
            return score
        direction = 1.0 if score > 0 else -1.0
        adjusted = score + direction * validation_adjustment
        if score > 0 and adjusted < 0:
            return 0.0
        if score < 0 and adjusted > 0:
            return 0.0
        return min(max(adjusted, -1.0), 1.0)

    def _reason(
        self,
        prefix: str,
        score: float,
        disagreement: float,
        opinions: list[dict[str, Any]],
        resolution_brief: str = "",
    ) -> str:
        top = sorted(opinions, key=lambda x: x.get("confidence", 0), reverse=True)[:3]
        summary = "；".join(
            f"{o.get('label') or o.get('model_name')}={self._action_label(o.get('action'))}({float(o.get('confidence', 0)):.2f})"
            for o in top
        )
        reason = f"{prefix}。加权方向分={score:.2f}，分歧度={disagreement:.2f}。主要意见：{summary}"
        if resolution_brief:
            reason += f"。分歧处理：{resolution_brief}"
        return reason

    def _conflict_resolution_brief(
        self,
        cross_validations: list[dict[str, Any]],
        consultation: dict[str, Any] | None,
    ) -> str:
        if not cross_validations:
            return "本轮没有专家提出需要交叉验证的问题，直接按专家权重和风控阈值裁决。"

        divergent = [v for v in cross_validations if v.get("consistency") == "divergent"]
        aligned = [v for v in cross_validations if v.get("consistency") == "aligned"]
        neutral = [v for v in cross_validations if v.get("consistency") == "neutral"]
        adjustment_points = sum(
            float(v.get("confidence_adjustment", 0) or 0) for v in cross_validations
        )

        parts: list[str] = []
        if divergent:
            parts.append(
                f"{len(divergent)} 个分歧已按核验结果下调置信度 {abs(min(adjustment_points, 0)):.0f} 分"
            )
        if aligned:
            parts.append(f"{len(aligned)} 个一致结论提高信号可信度")
        if neutral:
            parts.append(f"{len(neutral)} 个中性结论不改变方向")

        if isinstance(consultation, dict) and consultation.get("status") == "completed":
            should_trade = consultation.get("should_trade")
            verdict = (
                "允许继续交易"
                if should_trade is True
                else "建议不交易" if should_trade is False else "未给出明确交易许可"
            )
            note = str(consultation.get("conflict_note") or "").strip()
            parts.append(f"行情方向专家会诊：{verdict}{('，' + note[:80]) if note else ''}")
        elif isinstance(consultation, dict) and consultation.get("status") == "failed":
            note = str(
                consultation.get("conflict_note") or consultation.get("reason") or ""
            ).strip()
            parts.append(
                f"行情方向专家会诊失败，按保守规则不交易{('：' + note[:80]) if note else ''}"
            )
        elif divergent:
            parts.append("未达到深度会诊条件的分歧，由加权分、风控否决和入场阈值共同消化")

        return "；".join(parts)

    def _conflict_resolution_payload(
        self,
        opinions: list[dict[str, Any]],
        score: float,
        disagreement: float,
        cross_validations: list[dict[str, Any]],
        consultation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        action_by_name = {
            str(o.get("model_name")): self._action_label(o.get("action")) for o in opinions
        }
        items = []
        for validation in cross_validations:
            pair = [str(x) for x in validation.get("expert_pair", [])]
            source = pair[0] if pair else ""
            target = pair[1] if len(pair) > 1 else ""
            adjustment = float(validation.get("confidence_adjustment", 0) or 0)
            consistency = validation.get("consistency") or "neutral"
            if consistency == "divergent":
                resolution = (
                    f"发现 {self._expert_label(source)} 与 {self._expert_label(target)} 方向不一致，"
                    f"已把置信度调整 {adjustment:+.0f} 分计入最终分数。"
                )
            elif consistency == "aligned":
                resolution = (
                    f"{self._expert_label(target)} 的核验支持 {self._expert_label(source)}，"
                    f"置信度调整 {adjustment:+.0f} 分。"
                )
            else:
                resolution = (
                    f"{self._expert_label(target)} 的核验没有给出明确支持或否定，"
                    f"仅做 {adjustment:+.0f} 分调整。"
                )
            items.append(
                {
                    "expert_pair": pair,
                    "source_action": action_by_name.get(source, "未知"),
                    "target_action": action_by_name.get(target, "未知"),
                    "consistency": consistency,
                    "confidence_adjustment": adjustment,
                    "question": validation.get("question"),
                    "validation_note": validation.get("validation_note")
                    or validation.get("conflict_note"),
                    "resolution": resolution,
                    "major_conflict": bool(validation.get("major_conflict")),
                }
            )

        return {
            "summary": self._conflict_resolution_brief(cross_validations, consultation),
            "weighted_score_after_validation": round(score, 4),
            "disagreement": round(disagreement, 4),
            "validation_adjustment": round(
                self._validation_adjustment(cross_validations, consultation), 4
            ),
            "items": items,
            "consultation_used": isinstance(consultation, dict)
            and consultation.get("status") == "completed",
            "consultation_attempted": isinstance(consultation, dict),
        }

    def _avg(self, values: list[float], default: float) -> float:
        clean = [float(v) for v in values if v is not None]
        return sum(clean) / len(clean) if clean else default

    def _is_hard_risk_veto(
        self,
        decision: DecisionOutput,
        features: FeatureVector,
        *,
        for_position_close: bool = False,
    ) -> bool:
        """Treat risk hold as hard only for explicit danger, not ordinary caution."""
        if decision.model_name != "risk_expert" or decision.is_entry:
            return False

        reasoning = str(decision.reasoning or "")
        entry_veto_terms = (
            "一票否决",
            "硬性否决",
            "禁止交易",
            "禁止开仓",
        )
        entry_veto_terms = entry_veto_terms + (
            "hard veto",
            "prohibit entry",
            "entry prohibited",
            "do not open",
            "ban entry",
        )
        position_exit_terms = (
            "强制平仓",
            "立即平仓",
            "立刻平仓",
            "马上平仓",
            "必须平仓",
            "硬止损",
            "止损离场",
            "流动性枯竭",
            "流动性严重不足",
            "强平风险",
            "交易所异常",
            "闪崩",
            "黑天鹅",
            "熔断",
        )
        severe_market_terms = (
            "极端行情",
            "异常波动",
            "剧烈波动",
            "闪崩",
            "爆仓",
        )
        caution_terms = ("滑点", "假突破", "追高", "诱多", "诱空", "缺乏情绪")
        entry_filters = entry_filters_from_context(getattr(self, "_current_strategy_context", {}))
        volume_ratio = float(getattr(features, "volume_ratio", 0) or 0)
        adx_14 = float(getattr(features, "adx_14", 0) or 0)
        volatility_20 = float(getattr(features, "volatility_20", 0) or 0)
        change_24h = abs(float(getattr(features, "change_24h_pct", 0) or 0))
        healthy_market = (
            volume_ratio >= max(entry_filters.min_entry_volume_ratio, 0.5)
            and adx_14 >= entry_filters.min_entry_adx
            and volatility_20 <= 0.05
        )
        extreme_volatility = volatility_20 >= 0.12 and change_24h >= 12

        if for_position_close:
            if self._has_unnegated_hard_risk_term(reasoning, position_exit_terms):
                return decision.confidence >= 0.70
            if extreme_volatility and self._has_unnegated_hard_risk_term(
                reasoning, severe_market_terms
            ):
                return decision.confidence >= 0.78
            return False

        if self._has_unnegated_hard_risk_term(reasoning, entry_veto_terms):
            return decision.confidence >= 0.60
        if extreme_volatility and self._has_unnegated_hard_risk_term(
            reasoning, severe_market_terms + ("高波动", "假突破")
        ):
            return decision.confidence >= 0.75
        if healthy_market and any(term in reasoning for term in caution_terms):
            return False
        return False

    def _has_unnegated_hard_risk_term(self, reasoning: str, terms: tuple[str, ...]) -> bool:
        """Avoid treating phrases like "not a black swan" as a hard veto."""
        text = str(reasoning or "")
        negation_markers = (
            "无",
            "未",
            "非",
            "不",
            "没有",
            "不是",
            "并非",
            "不属于",
            "无需",
            "不用",
            "不需要",
            "不建议",
            "暂不",
            "未见",
            "未发现",
        )
        negation_markers = negation_markers + (
            "no ",
            "not ",
            "non-",
            "without ",
            "no-",
            "not-a",
        )
        for term in terms:
            start = 0
            while True:
                idx = text.find(term, start)
                if idx < 0:
                    break
                prefix = text[max(0, idx - 8) : idx]
                if not any(marker in prefix for marker in negation_markers):
                    return True
                start = idx + len(term)
        return False

    def _action_label(self, action: Any) -> str:
        labels = {
            "long": "做多",
            "short": "做空",
            "close_long": "平多",
            "close_short": "平空",
            "hold": "观望",
        }
        value = action.value if isinstance(action, Action) else str(action or "")
        return labels.get(value, value or "未知")

    def _expert_label(self, name: str) -> str:
        labels = {
            "trend_expert": "行情方向专家",
            "momentum_expert": "盈利质量专家",
            "sentiment_expert": "短线时序专家",
            "position_expert": "持仓退出专家",
            "risk_expert": "异常风控专家",
            "decision_maker": "最终交易员",
        }
        return labels.get(str(name or ""), str(name or "未知专家"))
