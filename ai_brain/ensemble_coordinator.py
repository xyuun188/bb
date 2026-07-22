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
)
from core.safe_output import safe_error_text
from services.dynamic_exit_policy import assess_dynamic_exit
from services.entry_signal_extraction import (
    enrich_signal_payload,
    signal_production_eligible,
    signal_return_distribution,
    unwrap_tool_payload,
)
from services.entry_signal_extraction import (
    payload_side as signal_payload_side,
)
from services.model_dynamic_routing import plan_dynamic_model_route
from services.paper_exploration import build_paper_exploration_contract
from services.paper_training import (
    build_paper_training_contract,
    paper_training_mode_enabled,
)

if TYPE_CHECKING:
    from data_feed.feature_vector import FeatureVector

logger = structlog.get_logger(__name__)


def _analysis_budget_snapshot(context: dict[str, Any]) -> dict[str, Any] | None:
    """Return the remaining cooperative market-analysis budget, if supplied."""

    try:
        deadline = float(context.get("_analysis_deadline_monotonic"))
    except (TypeError, ValueError):
        return None
    if deadline <= 0:
        return None
    remaining = max(deadline - asyncio.get_running_loop().time(), 0.0)
    return {
        "scope": str(context.get("_analysis_budget_scope") or "analysis"),
        "remaining_seconds": round(remaining, 3),
        "configured_budget_seconds": context.get("_analysis_budget_seconds"),
    }


ACTION_SCORE = {
    Action.LONG: 1.0,
    Action.SHORT: -1.0,
    Action.CLOSE_SHORT: 0.75,
    Action.CLOSE_LONG: -0.75,
    Action.HOLD: 0.0,
}

class EnsembleCoordinator:
    """Combines fixed expert model reports into one executable decision."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._slot_meta = {slot["name"]: slot for slot in FIXED_AI_MODEL_SLOTS}
        self.cross_validator = CrossValidator()
        self._current_strategy_context: dict[str, Any] = {}

    def _set_strategy_context(
        self,
        context: dict[str, Any] | None,
    ) -> None:
        self._current_strategy_context = dict(context or {})

    async def decide(
        self,
        features: FeatureVector,
        context: dict[str, Any],
    ) -> tuple[DecisionOutput, dict[str, DecisionOutput]]:
        self._set_strategy_context(context)
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

        raw = final.raw_response if isinstance(final.raw_response, dict) else {}
        raw["decision_maker"] = {
            "status": "observation_only",
            "applied": False,
            "reason": (
                "生产交易许可由权威费后收益策略统一决定；旧版“最终交易员”强制覆盖权限"
                "已移除，本模型结果仅供观察。"
            ),
            "strategy_version": "2026-07-12.remove-final-override.v1",
        }
        final.raw_response = raw
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
        if isinstance(context.get("market_regime"), dict):
            raw["market_regime"] = dict(context["market_regime"])
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
        trace_only_fallback: bool,
    ) -> tuple[float, dict[str, Any]]:
        """Weight expert observations without granting production permission."""

        raw_weight = base_weight * dynamic_multiplier
        policy: dict[str, Any] = {
            "mode": "observation_only",
            "base_weight": round(base_weight, 6),
            "dynamic_multiplier": round(dynamic_multiplier, 4),
            "raw_weight": round(raw_weight, 6),
            "entry_support_eligible": False,
            "production_permission": False,
            "excluded_reason": "expert output is observation-only",
        }
        if trace_only_fallback:
            policy.update(
                {
                    "mode": "trace_only_fallback",
                    "effective_weight": 0.0,
                    "excluded_reason": "fallback opinion is trace-only",
                }
            )
            return 0.0, policy
        del name, review_positions, current_side
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
        del features, review_positions, current_side, hard_veto
        return {
            "active": False,
            "hard_veto": False,
            "production_permission": False,
            "reason": "risk expert is observation-only; RiskEngine owns hard risk",
            "confidence": round(self._safe_float(getattr(risk_opinion, "confidence", 0.0)), 4),
            "risk_action": getattr(getattr(risk_opinion, "action", None), "value", "missing"),
        }

    def _expert_weight_policy_summary(self, opinions: list[dict[str, Any]]) -> dict[str, Any]:
        policies = {
            str(opinion.get("model_name")): opinion.get("weight_policy")
            for opinion in opinions
            if opinion.get("model_name") and isinstance(opinion.get("weight_policy"), dict)
        }
        source_policy = self._expert_source_policy_from_opinions(opinions)
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
            "entry_support_excluded_experts": [],
            "score_participants": [
                str(opinion.get("model_name"))
                for opinion in opinions
                if self._safe_float(opinion.get("effective_weight"), 0.0) > 0
            ],
            "source_policy": source_policy,
        }

    @staticmethod
    def _normalize_provider_model(value: Any) -> str:
        return str(value or "").strip()

    def _decision_provider_model(self, name: str, decision: DecisionOutput | None) -> str:
        if isinstance(decision, DecisionOutput) and isinstance(decision.raw_response, dict):
            provider = self._normalize_provider_model(decision.raw_response.get("provider_model"))
            if provider:
                return provider
        return self._normalize_provider_model(
            self._slot_meta.get(name, {}).get("model")
            or getattr(self.registry.get(name), "_model_name", "")
            or name
        )

    def _expert_source_group(self, name: str, decision: DecisionOutput | None) -> str:
        provider = self._decision_provider_model(name, decision)
        if provider in {"local_fast_prefilter", "local_rules", ""}:
            return f"local:{name}"
        if name == "risk_expert" and "deepseek" in provider.lower():
            return "llm:risk_review"
        return f"llm:{provider}"

    def _support_source_groups(
        self,
        decisions: dict[str, DecisionOutput],
        action: Action,
        *,
        eligible_names: set[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for name, decision in decisions.items():
            if eligible_names is not None and name not in eligible_names:
                continue
            if (
                not isinstance(decision, DecisionOutput)
                or decision.action != action
            ):
                continue
            group = self._expert_source_group(name, decision)
            current = groups.get(group)
            row = {
                "experts": [name],
                "provider_model": self._decision_provider_model(name, decision),
                "max_confidence": round(float(decision.confidence or 0.0), 4),
            }
            if current is None:
                groups[group] = row
                continue
            current["experts"].append(name)
            current["max_confidence"] = round(
                max(
                    self._safe_float(current.get("max_confidence"), 0.0),
                    float(decision.confidence or 0.0),
                ),
                4,
            )
        return groups

    def _independent_quant_supports(
        self,
        context: dict[str, Any] | None,
        action_side: str,
        ml_profit_hint: dict[str, Any] | None = None,
    ) -> list[str]:
        supports: list[str] = []
        if self._local_profit_aligned(context, action_side):
            supports.append("server_profit_model")
        if self._time_series_aligned(context, action_side):
            supports.append("time_series_model")
        if (
            isinstance(ml_profit_hint, dict)
            and ml_profit_hint.get("strong")
            and ml_profit_hint.get("side") == action_side
        ):
            supports.append("local_ml_shadow")
        direction_competition = self._safe_dict(
            context.get("direction_competition") if isinstance(context, dict) else {}
        )
        direction_side = str(direction_competition.get("preferred_side") or "").lower()
        if direction_side == action_side:
            supports.append("direction_competition")
        return supports

    def _expert_source_policy_from_decisions(
        self,
        decisions: dict[str, DecisionOutput],
        action: Action | None = None,
        *,
        context: dict[str, Any] | None = None,
        ml_profit_hint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        all_groups: dict[str, dict[str, Any]] = {}
        for name, decision in decisions.items():
            if not isinstance(decision, DecisionOutput):
                continue
            group = self._expert_source_group(name, decision)
            row = all_groups.setdefault(
                group,
                {
                    "experts": [],
                    "provider_model": self._decision_provider_model(name, decision),
                    "source_type": "llm" if group.startswith("llm:") else "local",
                },
            )
            row["experts"].append(name)
        payload: dict[str, Any] = {
            "mode": "source_deduplicated",
            "all_source_groups": all_groups,
            "same_provider_roles_are_not_independent_votes": True,
            "policy": (
                "LLM roles sharing the same provider model are treated as one evidence source; "
                "new entries need independent quant evidence before the LLM group can raise risk."
            ),
        }
        if action in (Action.LONG, Action.SHORT):
            side = "long" if action == Action.LONG else "short"
            directional_groups = self._support_source_groups(decisions, action)
            technical_groups = self._support_source_groups(decisions, action)
            quant_supports = self._independent_quant_supports(context, side, ml_profit_hint)
            payload.update(
                {
                    "side": side,
                    "directional_source_groups": directional_groups,
                    "technical_source_groups": technical_groups,
                    "directional_independent_source_count": len(directional_groups),
                    "technical_independent_source_count": len(technical_groups),
                    "independent_quant_supports": quant_supports,
                    "independent_quant_support_count": len(quant_supports),
                }
            )
        return payload

    def _expert_source_policy_from_opinions(
        self, opinions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        for opinion in opinions:
            if not isinstance(opinion, dict):
                continue
            name = str(opinion.get("model_name") or "")
            if not name:
                continue
            provider = self._normalize_provider_model(opinion.get("provider_model"))
            if not provider:
                provider = self._normalize_provider_model(
                    self._slot_meta.get(name, {}).get("model")
                    or getattr(self.registry.get(name), "_model_name", "")
                    or name
                )
            if provider in {"local_fast_prefilter", "local_rules", ""}:
                group = f"local:{name}"
            elif name == "risk_expert" and "deepseek" in provider.lower():
                group = "llm:risk_review"
            else:
                group = f"llm:{provider}"
            row = groups.setdefault(
                group,
                {"experts": [], "provider_model": provider, "source_type": "llm"},
            )
            row["experts"].append(name)
        return {
            "mode": "source_deduplicated",
            "all_source_groups": groups,
            "same_provider_roles_are_not_independent_votes": True,
        }

    def _risk_policy_from_opinions(self, opinions: list[dict[str, Any]]) -> dict[str, Any]:
        for opinion in opinions:
            if opinion.get("model_name") == "risk_expert" and isinstance(
                opinion.get("risk_expert_policy"), dict
            ):
                return dict(opinion["risk_expert_policy"])
        return {"active": False, "reason": "risk_expert opinion missing"}


    def combine(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        opinions: dict[str, DecisionOutput],
        cross_validations: list[dict[str, Any]] | None = None,
        consultation: dict[str, Any] | None = None,
    ) -> DecisionOutput:
        self._set_strategy_context(context)
        cross_validations = cross_validations or []
        valid = {name: d for name, d in opinions.items() if isinstance(d, DecisionOutput)}
        if not valid and context.get("review_positions"):
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
        for name, decision in valid.items():
            meta = self._slot_meta.get(name, {})
            base_weight = float(
                meta.get("weight", getattr(self.registry.get(name), "weight", 1.0)) or 1.0
            )
            dynamic_multiplier = 1.0
            raw_decision = self._safe_dict(decision.raw_response)
            timeout_fallback = bool(raw_decision.get("timeout_fallback"))
            trace_only_fallback = bool(
                timeout_fallback
                or raw_decision.get("local_fallback")
                or raw_decision.get("batch_expert_fallback")
                or raw_decision.get("production_eligible") is False
            )
            effective_weight, weight_policy = self._effective_expert_weight(
                name=name,
                base_weight=base_weight,
                dynamic_multiplier=dynamic_multiplier,
                review_positions=review_positions,
                current_side=str(current_side) if current_side else None,
                trace_only_fallback=trace_only_fallback,
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
            if decision.is_exit and effective_weight > 0:
                exit_votes.append(decision)
            if name == "risk_expert" and effective_weight > 0:
                risk_opinion = decision

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
                    "weight": weight,
                    "effective_weight": effective_weight,
                    "weight_policy": weight_policy,
                    "entry_support_eligible": weight_policy.get("entry_support_eligible"),
                    "excluded_reason": weight_policy.get("excluded_reason"),
                    "reasoning": decision.reasoning,
                    "cross_check_for": decision.cross_check_for,
                    "timeout_fallback": timeout_fallback,
                    "trace_only_fallback": trace_only_fallback,
                    "independent_expert_retry": bool(
                        isinstance(decision.raw_response, dict)
                        and decision.raw_response.get("independent_expert_retry")
                    ),
                    "provider_independent_expert_mode": bool(
                        isinstance(decision.raw_response, dict)
                        and decision.raw_response.get("provider_independent_expert_mode")
                    ),
                }
            )
            raw_opinions_by_name[name] = raw_opinions[-1]

        normalized_score = weighted_score / total_weight if total_weight else 0.0
        disagreement = self._disagreement((score_participants or valid).values())
        decision_score = normalized_score
        risk_expert_policy = self._risk_expert_entry_policy(
            risk_opinion,
            features,
            review_positions=review_positions,
            current_side=str(current_side) if current_side else None,
            hard_veto=bool(risk_vetoes),
        )
        if "risk_expert" in raw_opinions_by_name:
            raw_opinions_by_name["risk_expert"]["risk_expert_policy"] = risk_expert_policy
        resolution_brief = self._conflict_resolution_brief(cross_validations, consultation)
        raw = self._raw(raw_opinions, decision_score, disagreement, cross_validations, consultation)
        raw["expert_availability"] = {
            "available_count": len(valid),
            "available": bool(valid),
            "role": "observation_only",
            "can_block_authoritative_return_candidate": False,
        }
        self._attach_expert_diversity_policy(raw, context)
        raw["base_weighted_score"] = round(normalized_score, 4)
        raw["memory_feedback"] = self._memory_feedback(context)
        raw["market_regime"] = context.get("market_regime") or {}
        raw["strategy_mode"] = context.get("strategy_mode") or {}
        raw["ml_signal"] = context.get("ml_signal") or {}
        raw["portfolio_profit_protection"] = context.get("portfolio_profit_protection") or {}

        if risk_vetoes and not current_side and not paper_training_mode_enabled(context):
            reason = self._reason(
                "风控专家否决新开仓", decision_score, disagreement, raw_opinions, resolution_brief
            )
            return self._hold(features, reason, raw)

        if review_positions and current_side:
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
            if evidence.get("should_close"):
                close_raw = self._raw(
                    raw_opinions,
                    decision_score,
                    disagreement,
                    cross_validations,
                    consultation,
                )
                close_raw["close_evidence"] = evidence
                close_raw["position_review_policy"] = {
                    "result": evidence.get("action_plan"),
                    "source": "dynamic_exit_policy",
                }
                return DecisionOutput(
                    model_name=ENSEMBLE_TRADER_NAME,
                    symbol=features.symbol,
                    action=close_action,
                    confidence=self._safe_float(evidence.get("suggested_confidence"), 0.0),
                    reasoning=str(evidence.get("reason") or "dynamic_exit_policy_passed"),
                    position_size_pct=self._safe_float(evidence.get("position_size_pct"), 0.0),
                    suggested_leverage=1.0,
                    stop_loss_pct=0.0,
                    take_profit_pct=0.0,
                    raw_response=close_raw,
                    feature_snapshot=features.to_dict(),
                )
            hold_raw = self._raw(
                raw_opinions, decision_score, disagreement, cross_validations, consultation
            )
            self._attach_expert_diversity_policy(hold_raw, context)
            hold_raw["base_weighted_score"] = round(normalized_score, 4)
            hold_raw["memory_feedback"] = self._memory_feedback(context)
            hold_raw["portfolio_profit_protection"] = (
                context.get("portfolio_profit_protection") or {}
            )
            hold_raw["position_review_policy"] = {
                "result": "hold",
                "source": "dynamic_exit_policy",
            }
            hold_raw["close_evidence"] = evidence
            reason = self._reason(
                f"持仓复盘结论为继续持有：{evidence.get('block_reason') or '动态退出压力为零'}",
                decision_score,
                disagreement,
                raw_opinions,
                resolution_brief,
            )
            return self._hold(features, reason, hold_raw)

        candidate_evidence = self._safe_dict(context.get("entry_candidate_evidence"))
        preferred_side = str(candidate_evidence.get("preferred_side_by_evidence") or "").lower()
        side_evidence = self._safe_dict(candidate_evidence.get(preferred_side))
        policy_provenance = self._safe_dict(side_evidence.get("policy_provenance"))
        governance_complete = all(
            policy_provenance.get(key) not in {None, ""}
            for key in (
                "source",
                "observation_window",
                "sample_count",
                "generated_at",
                "strategy_version",
            )
        )
        production_eligible = bool(
            preferred_side in {"long", "short"}
            and side_evidence.get("production_eligible") is True
            and self._safe_float(side_evidence.get("expected_net_return_pct"), 0.0) > 0.0
            and self._safe_float(side_evidence.get("return_lcb_pct"), 0.0) > 0.0
            and self._safe_float(side_evidence.get("production_source_count"), 0.0) > 0.0
            and governance_complete
        )
        raw["authoritative_return_candidate"] = {
            "production_eligible": production_eligible,
            "preferred_side": preferred_side or "neutral",
            "side_evidence": side_evidence,
            "policy_provenance": policy_provenance,
        }
        if not production_eligible:
            if (
                paper_training_mode_enabled(context)
                and str(
                    candidate_evidence.get("preferred_exploration_side") or ""
                ).lower()
                not in {"long", "short"}
            ):
                (
                    training_side,
                    training_source,
                    training_expected,
                    training_lcb,
                    training_horizon,
                ) = self._paper_training_side(context, normalized_score)
                if training_side in {"long", "short"}:
                    training_contract = build_paper_training_contract(
                        symbol=features.symbol,
                        selected_side=training_side,
                        signal_source=training_source,
                        expected_net_return_pct=training_expected,
                        return_lcb_pct=training_lcb,
                        feature_opportunity_score=self._safe_float(
                            candidate_evidence.get("feature_opportunity_score"),
                            0.0,
                        ),
                        horizon_minutes=training_horizon,
                        policy_provenance=policy_provenance,
                    )
                    training_raw = self._raw(
                        raw_opinions,
                        decision_score,
                        disagreement,
                        cross_validations,
                        consultation,
                    )
                    self._attach_expert_diversity_policy(training_raw, context)
                    training_raw["authoritative_return_candidate"] = raw[
                        "authoritative_return_candidate"
                    ]
                    training_raw["entry_candidate_evidence"] = candidate_evidence
                    training_raw["paper_training"] = training_contract
                    training_raw["paper_training_mode"] = "bootstrap"
                    training_raw["base_weighted_score_observation"] = round(
                        normalized_score,
                        4,
                    )
                    training_raw["memory_feedback_observation"] = self._memory_feedback(
                        context
                    )
                    training_raw["ml_signal"] = context.get("ml_signal") or {}
                    training_raw["local_ai_tools"] = context.get("local_ai_tools") or {}
                    training_raw["direction_competition"] = (
                        context.get("direction_competition") or {}
                    )
                    training_raw["entry_permission_policy"] = {
                        "source": "paper_training_bootstrap_without_profit_gate",
                        "execution_scope": "paper_only",
                        "production_permission": False,
                        "sample_target": None,
                        "daily_sample_quota": None,
                        "valid_for_seconds": training_contract.get(
                            "valid_for_seconds"
                        ),
                        "prediction_horizon_minutes": training_contract.get(
                            "prediction_horizon_minutes"
                        ),
                        "generated_at": datetime.now(UTC).isoformat(),
                        "strategy_version": training_contract.get("version"),
                    }
                    action = Action.LONG if training_side == "long" else Action.SHORT
                    return DecisionOutput(
                        model_name=ENSEMBLE_TRADER_NAME,
                        symbol=features.symbol,
                        action=action,
                        confidence=min(max(abs(normalized_score), 0.05), 1.0),
                        reasoning=(
                            "模拟盘快速训练期按模型方向正常开多，暂不以预期盈亏拦截"
                            if action == Action.LONG
                            else "模拟盘快速训练期按模型方向正常开空，暂不以预期盈亏拦截"
                        ),
                        position_size_pct=0.0,
                        suggested_leverage=1.0,
                        stop_loss_pct=0.0,
                        take_profit_pct=0.0,
                        raw_response=training_raw,
                        feature_snapshot=features.to_dict(),
                    )
            exploration_side = str(
                candidate_evidence.get("preferred_exploration_side") or ""
            ).lower()
            execution_mode = str(context.get("execution_mode") or "").lower()
            exploration_contract = (
                build_paper_exploration_contract(
                    candidate_evidence,
                    symbol=features.symbol,
                )
                if execution_mode == "paper"
                and exploration_side in {"long", "short"}
                else {}
            )
            if exploration_contract:
                action = Action.LONG if exploration_side == "long" else Action.SHORT
                exploration = self._safe_dict(
                    candidate_evidence.get("paper_exploration")
                )
                selected_exploration = self._safe_dict(exploration.get("selected"))
                information_value = self._safe_float(
                    selected_exploration.get("information_value_score"),
                    0.0,
                )
                reason = self._reason(
                    (
                        "模拟盘候选扣费后期望为正但收益下界仍有轻微不确定，"
                        "按独立小风险预算执行做多探索"
                        if action == Action.LONG
                        else "模拟盘候选扣费后期望为正但收益下界仍有轻微不确定，"
                        "按独立小风险预算执行做空探索"
                    ),
                    decision_score,
                    disagreement,
                    raw_opinions,
                    resolution_brief,
                )
                exploration_raw = self._raw(
                    raw_opinions,
                    decision_score,
                    disagreement,
                    cross_validations,
                    consultation,
                )
                self._attach_expert_diversity_policy(exploration_raw, context)
                exploration_raw["authoritative_return_candidate"] = raw[
                    "authoritative_return_candidate"
                ]
                exploration_raw["entry_candidate_evidence"] = candidate_evidence
                exploration_raw["paper_exploration"] = exploration_contract
                exploration_raw["base_weighted_score_observation"] = round(
                    normalized_score,
                    4,
                )
                exploration_raw["memory_feedback_observation"] = self._memory_feedback(
                    context
                )
                exploration_raw["ml_signal"] = context.get("ml_signal") or {}
                exploration_raw["local_ai_tools"] = context.get("local_ai_tools") or {}
                exploration_raw["direction_competition"] = (
                    context.get("direction_competition") or {}
                )
                exploration_raw["entry_permission_policy"] = {
                    "source": "bounded_positive_mean_paper_exploration",
                    "execution_scope": "paper_only",
                    "production_permission": False,
                    "sample_target": None,
                    "daily_sample_quota": None,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "strategy_version": exploration_contract.get("version"),
                }
                return DecisionOutput(
                    model_name=ENSEMBLE_TRADER_NAME,
                    symbol=features.symbol,
                    action=action,
                    confidence=min(max(information_value, 0.0), 1.0),
                    reasoning=reason,
                    position_size_pct=0.0,
                    suggested_leverage=1.0,
                    stop_loss_pct=0.0,
                    take_profit_pct=0.0,
                    raw_response=exploration_raw,
                    feature_snapshot=features.to_dict(),
                )
            reason = self._reason(
                "当前模型没有给出扣除交易成本后仍为正的模拟盘机会，本轮保持观望",
                decision_score,
                disagreement,
                raw_opinions,
                resolution_brief,
            )
            hold_raw = self._raw(
                raw_opinions, decision_score, disagreement, cross_validations, consultation
            )
            self._attach_expert_diversity_policy(hold_raw, context)
            hold_raw["authoritative_return_candidate"] = raw["authoritative_return_candidate"]
            hold_raw["entry_candidate_evidence"] = candidate_evidence
            hold_raw["base_weighted_score_observation"] = round(normalized_score, 4)
            hold_raw["memory_feedback_observation"] = self._memory_feedback(context)
            return self._hold(features, reason, hold_raw)

        action = Action.LONG if preferred_side == "long" else Action.SHORT
        return_lcb = self._safe_float(side_evidence.get("return_lcb_pct"), 0.0)
        downside = max(
            self._safe_float(side_evidence.get("expected_loss_pct"), 0.0),
            self._safe_float(side_evidence.get("return_uncertainty_pct"), 0.0),
            0.0,
        )
        confidence = return_lcb / (return_lcb + downside) if return_lcb + downside > 0 else 0.0
        reason = self._reason(
            "权威费后收益分布选择做多候选"
            if action == Action.LONG
            else "权威费后收益分布选择做空候选",
            decision_score,
            disagreement,
            raw_opinions,
            resolution_brief,
        )
        raw_response = self._raw(
            raw_opinions, decision_score, disagreement, cross_validations, consultation
        )
        self._attach_expert_diversity_policy(raw_response, context)
        raw_response["authoritative_return_candidate"] = raw["authoritative_return_candidate"]
        raw_response["entry_candidate_evidence"] = candidate_evidence
        raw_response["base_weighted_score_observation"] = round(normalized_score, 4)
        raw_response["memory_feedback_observation"] = self._memory_feedback(context)
        raw_response["ml_signal"] = context.get("ml_signal") or {}
        raw_response["local_ai_tools"] = context.get("local_ai_tools") or {}
        raw_response["direction_competition"] = context.get("direction_competition") or {}
        raw_response["entry_permission_policy"] = {
            "source": "authoritative_fee_after_return_candidate",
            "observation_window": "current_pre_ai_candidate_round",
            "sample_count": int(
                self._safe_float(side_evidence.get("production_source_count"), 0.0)
            ),
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": "2026-07-12.ensemble-return-candidate.v1",
            "fallback_reason": "",
        }
        return DecisionOutput(
            model_name=ENSEMBLE_TRADER_NAME,
            symbol=features.symbol,
            action=action,
            confidence=confidence,
            reasoning=reason,
            position_size_pct=0.0,
            suggested_leverage=1.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            raw_response=raw_response,
            feature_snapshot=features.to_dict(),
        )


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
        distribution = signal_return_distribution(payload, side)
        return self._safe_float(
            distribution.get("objective_expected_return_pct"),
            0.0,
        )






    def _local_profit_aligned(self, context: dict[str, Any] | None, side: str) -> bool:
        side = str(side or "").lower()
        if side not in {"long", "short"}:
            return False
        profit = self._local_profit_signal(context)
        if not profit or not signal_production_eligible(profit):
            return False
        expected = self._local_expected_return(profit, side)
        best_side = signal_payload_side(profit) or str(profit.get("best_side") or "").lower()
        return expected > 0 and best_side in {"", side}

    def _time_series_aligned(self, context: dict[str, Any] | None, side: str) -> bool:
        side = str(side or "").lower()
        if side not in {"long", "short"}:
            return False
        prediction = self._local_timeseries_signal(context)
        if not prediction or not signal_production_eligible(prediction):
            return False
        best_side = signal_payload_side(prediction)
        expected = self._local_expected_return(prediction, side)
        return best_side == side and expected > 0










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

    def _paper_training_side(
        self,
        context: dict[str, Any],
        normalized_score: float,
    ) -> tuple[str, str, float | None, float | None, float | None]:
        """Choose the best available observed direction without a profit gate."""

        del normalized_score

        competition = self._safe_dict(context.get("direction_competition"))
        observed_side = str(
            competition.get("training_preferred_side")
            or competition.get("preferred_side")
            or ""
        ).lower()
        training_rows = {
            "long": self._safe_dict(competition.get("training_long")),
            "short": self._safe_dict(competition.get("training_short")),
        }
        if observed_side in {"long", "short"}:
            row = training_rows.get(observed_side, {})
            horizon = self._finite_or_none(row.get("horizon_minutes"))
            if horizon is not None and horizon > 0:
                return (
                    observed_side,
                    "direction_competition_observation",
                    self._finite_or_none(row.get("objective_expected_return_pct")),
                    self._finite_or_none(row.get("objective_expected_return_pct")),
                    horizon,
                )

        signal = self._safe_dict(context.get("ml_signal"))
        predictions = signal.get("predictions")
        predictions = predictions if isinstance(predictions, list) else []
        primary = self._safe_dict(predictions[0] if predictions else {})
        distribution = self._safe_dict(primary.get("return_distribution_contract"))
        rows: dict[str, dict[str, Any]] = {
            side: self._safe_dict(distribution.get(side))
            for side in ("long", "short")
        }
        scores = {
            side: self._finite_or_none(
                row.get("objective_expected_return_pct")
            )
            for side, row in rows.items()
        }
        if all(value is None for value in scores.values()):
            scores = {
                side: self._finite_or_none(row.get("raw_expected_return_pct"))
                for side, row in rows.items()
            }
        horizons = {
            side: self._finite_or_none(
                rows[side].get("horizon_minutes", primary.get("horizon_minutes"))
            )
            for side in ("long", "short")
        }
        available = {
            side: value
            for side, value in scores.items()
            if value is not None
            and horizons.get(side) is not None
            and float(horizons[side]) > 0
        }
        if available:
            side = max(available, key=lambda item: float(available[item]))
            return (
                side,
                "local_ml_observation",
                self._finite_or_none(rows[side].get("raw_expected_return_pct")),
                self._finite_or_none(rows[side].get("objective_expected_return_pct")),
                horizons[side],
            )
        primary_side = str(primary.get("best_side") or "").lower()
        primary_horizon = self._finite_or_none(primary.get("horizon_minutes"))
        if primary_side in {"long", "short"} and primary_horizon and primary_horizon > 0:
            return (
                primary_side,
                "local_ml_best_side_observation",
                None,
                None,
                primary_horizon,
            )
        return "neutral", "no_auditable_directional_horizon", None, None, None

    @staticmethod
    def _finite_or_none(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number == number and abs(number) != float("inf") else None






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
        """Apply the single dynamic fee-after exit policy to an open position."""
        if not current_side:
            return {"should_close": False, "block_reason": "当前没有持仓。"}

        del exit_votes, risk_vetoes, score, raw_opinions
        del context
        dynamic_decision = DecisionOutput(
            model_name=ENSEMBLE_TRADER_NAME,
            symbol=features.symbol if features is not None else "",
            action=close_action,
            confidence=0.0,
            reasoning="dynamic fee-after exit evidence",
            position_size_pct=0.0,
            suggested_leverage=1.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            raw_response={},
            feature_snapshot=features.to_dict() if features is not None else {},
        )
        assessment = assess_dynamic_exit(dynamic_decision, symbol_positions or [])
        return {
            "should_close": assessment.eligible,
            "action_plan": (
                "full_close"
                if assessment.close_fraction >= 1.0
                else "reduce"
                if assessment.close_fraction > 0.0
                else "hold"
            ),
            "position_size_pct": assessment.close_fraction,
            "suggested_confidence": assessment.close_fraction,
            "reason": assessment.reason,
            "block_reason": "" if assessment.eligible else assessment.reason,
            "dynamic_exit_policy": assessment.to_dict(),
            "planned_stop_crossed": assessment.planned_stop_crossed,
            "position_loss": assessment.fee_after_unrealized_pnl_usdt < 0.0,
            "dynamic_loss_reduce_fraction": assessment.stop_risk_usage,
        }

    def _position_quantity(self, pos: dict[str, Any]) -> float:
        if bool(pos.get("aggregate_position")):
            return abs(self._safe_float(pos.get("quantity"), 0.0))
        raw_info = pos.get("info")
        info = raw_info if isinstance(raw_info, dict) else {}
        contracts = abs(
            self._safe_float(
                pos.get("contracts")
                or pos.get("sz")
                or pos.get("size")
                or info.get("pos")
                or info.get("qty"),
                0.0,
            )
        )
        contract_size = abs(
            self._safe_float(
                pos.get("contract_size") or pos.get("contractSize") or info.get("ctVal"),
                1.0,
            )
        )
        if contracts > 0:
            return contracts * (contract_size if contract_size > 0 else 1.0)
        return abs(self._safe_float(pos.get("quantity"), 0.0))

    def _position_notional(
        self,
        pos: dict[str, Any],
        entry_price: float,
        quantity: float,
    ) -> float:
        raw_info = pos.get("info")
        info = raw_info if isinstance(raw_info, dict) else {}
        direct_notional = abs(
            self._safe_float(
                pos.get("notional")
                or pos.get("notional_usd")
                or pos.get("notionalUsd")
                or info.get("notionalUsd")
                or info.get("notional")
                or info.get("posValue"),
                0.0,
            )
        )
        if direct_notional > 0:
            return direct_notional
        return abs(entry_price * quantity)

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
                qty = self._position_quantity(pos)
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
                notional = direct_notional if direct_notional > 0 else qty * entry
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
            "opinions": opinions,
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

    @staticmethod
    def _attach_expert_diversity_policy(
        raw: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        policy = context.get("_expert_diversity_policy")
        if isinstance(policy, dict):
            raw["expert_diversity_policy"] = policy

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

    @staticmethod
    def _memory_feedback(context: dict[str, Any]) -> dict[str, Any]:
        feedback = context.get("memory_feedback") if isinstance(context, dict) else {}
        return feedback if isinstance(feedback, dict) else {}

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
            return "本轮没有专家交叉观察记录。"

        divergent = [v for v in cross_validations if v.get("consistency") == "divergent"]
        aligned = [v for v in cross_validations if v.get("consistency") == "aligned"]
        neutral = [v for v in cross_validations if v.get("consistency") == "neutral"]
        parts: list[str] = []
        if divergent:
            parts.append(f"{len(divergent)} 个专家分歧仅记录观察，不调整生产分数")
        if aligned:
            parts.append(f"{len(aligned)} 个一致结论提高信号可信度")
        if neutral:
            parts.append(f"{len(neutral)} 个中性结论不改变方向")

        if isinstance(consultation, dict) and consultation.get("status") == "completed":
            note = str(consultation.get("conflict_note") or "").strip()
            parts.append(f"行情方向专家观察复核{('：' + note[:80]) if note else ''}")
        elif isinstance(consultation, dict) and consultation.get("status") == "failed":
            note = str(
                consultation.get("conflict_note") or consultation.get("reason") or ""
            ).strip()
            parts.append(f"行情方向专家观察复核失败{('：' + note[:80]) if note else ''}")
        elif divergent:
            parts.append("分歧无交易许可，仅供诊断")

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
            consistency = validation.get("consistency") or "neutral"
            if consistency == "divergent":
                resolution = "专家方向不一致，仅记录观察，不调整生产分数。"
            elif consistency == "aligned":
                resolution = "专家方向一致，仅记录观察，不授予生产权限。"
            else:
                resolution = "交叉核验中性，仅记录观察。"
            items.append(
                {
                    "expert_pair": pair,
                    "source_action": action_by_name.get(source, "未知"),
                    "target_action": action_by_name.get(target, "未知"),
                    "consistency": consistency,
                    "question": validation.get("question"),
                    "validation_note": validation.get("validation_note")
                    or validation.get("conflict_note"),
                    "resolution": resolution,
                    "major_conflict": bool(validation.get("major_conflict")),
                }
            )

        return {
            "summary": self._conflict_resolution_brief(cross_validations, consultation),
            "weighted_score_observation": round(score, 4),
            "disagreement": round(disagreement, 4),
            "production_permission": False,
            "items": items,
            "consultation_used": isinstance(consultation, dict)
            and consultation.get("status") == "completed",
            "consultation_attempted": isinstance(consultation, dict),
        }

    def _avg(self, values: list[float], default: float) -> float:
        clean = [float(v) for v in values if v is not None]
        return sum(clean) / len(clean) if clean else default



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
