"""
Model registry for managing all AI trading models.
Provides registration, selection, and lifecycle management.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol

import structlog

from ai_brain.base_model import AbstractAIModel, Action, DecisionOutput
from ai_brain.expert_diversity_policy import (
    ExpertDiversityReview,
    review_batch_expert_consensus,
)
from config.settings import settings
from core.safe_output import safe_error_text
from data_feed.feature_vector import FeatureVector

logger = structlog.get_logger(__name__)

LocalExpertFallback = Callable[[FeatureVector, dict[str, Any], str], DecisionOutput]
BatchExpertDecider = Callable[
    [FeatureVector, dict[str, Any], list[str]],
    Awaitable[dict[str, DecisionOutput]],
]


class LocalExpertFallbackModel(Protocol):
    """Protocol for LLM-style experts that expose a local deterministic fallback."""

    name: str

    def _local_expert_fallback(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        error: str,
    ) -> DecisionOutput:
        """Return a deterministic fallback decision for this expert."""


class BatchExpertModel(Protocol):
    """Protocol for a model that can answer all fixed experts in one LLM call."""

    name: str
    _llm: Any
    _model_name: str | None

    async def decide_batch_experts(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        expert_names: list[str],
    ) -> dict[str, DecisionOutput]:
        """Return expert decisions keyed by expert name."""


def _local_fallback_callable(model: AbstractAIModel) -> LocalExpertFallback | None:
    method = getattr(model, "_local_expert_fallback", None)
    return method if callable(method) else None


def _batch_expert_decider(model: AbstractAIModel) -> BatchExpertDecider | None:
    if getattr(model, "_llm", None) is None:
        return None
    model_name = str(getattr(model, "_model_name", "") or "").lower()
    if "deepseek" in model_name or "r1" in model_name:
        return None
    method = getattr(model, "decide_batch_experts", None)
    return method if callable(method) else None


def _provider_model_name(model: object | None) -> str | None:
    value = getattr(model, "_model_name", None)
    return str(value) if value else None


def _provider_group_key(model: object) -> tuple[str, str]:
    """Group batchable experts by actual provider endpoint and model id."""

    return (
        str(getattr(model, "_base_url", "") or ""),
        str(getattr(model, "_model_name", "") or ""),
    )


def _group_batchable_models_by_provider(
    active_models: list[AbstractAIModel],
) -> list[list[AbstractAIModel]]:
    """Return stable groups so different provider models do not fake one consensus."""

    groups: dict[tuple[str, str], list[AbstractAIModel]] = {}
    order: list[tuple[str, str]] = []
    for model in active_models:
        key = _provider_group_key(model)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(model)
    return [groups[key] for key in order]


def _is_timeout_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, TimeoutError | asyncio.TimeoutError)
        or exc.__class__.__name__ == "TimeoutError"
    )


def _is_batch_format_failure(exc: BaseException, error_text: str) -> bool:
    class_name = exc.__class__.__name__.lower()
    lowered = str(error_text or "").lower()
    return (
        "llmresponseparseerror" in class_name
        or "valid json" in lowered
        or "missing experts object" in lowered
    )


def _batch_failure_breaker_seconds(exc: BaseException, error_text: str) -> float:
    configured = max(float(settings.ai_batch_expert_circuit_breaker_seconds or 0.0), 0.0)
    if _is_timeout_error(exc):
        return max(configured, 1.0)
    if _is_batch_format_failure(exc, error_text):
        format_configured = max(
            float(settings.ai_batch_expert_format_failure_circuit_breaker_seconds or 0.0),
            0.0,
        )
        return max(configured, format_configured)
    return configured


def _positive_duration_seconds(started_at: float) -> float:
    return max(round(time.perf_counter() - started_at, 3), 0.001)


def _independent_expert_timeout_seconds(active_count: int) -> float:
    """Return the real independent expert timeout without the old 18-second cap."""

    base_timeout = max(float(settings.ai_expert_timeout_seconds or 30.0), 8.0)
    expert_count = max(int(active_count or 1), 1)
    concurrency = max(int(settings.ai_llm_concurrency or expert_count), 1)
    queue_batches = max((expert_count + concurrency - 1) // concurrency, 1)
    return base_timeout * queue_batches


def _analysis_budget_snapshot(context: dict[str, Any]) -> dict[str, Any] | None:
    """Return the remaining cooperative analysis budget, when a caller provides one."""

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


def _bounded_analysis_timeout(
    context: dict[str, Any],
    requested_timeout_seconds: float,
) -> tuple[float, dict[str, Any] | None]:
    """Bound one model stage to the caller's remaining analysis budget."""

    requested = max(float(requested_timeout_seconds or 0.0), 0.0)
    snapshot = _analysis_budget_snapshot(context)
    if snapshot is None:
        return requested, None
    remaining = max(float(snapshot["remaining_seconds"] or 0.0), 0.0)
    reserve = min(0.5, max(0.1, requested * 0.02))
    allowed = max(remaining - reserve, 0.0)
    snapshot["reserve_seconds"] = round(reserve, 3)
    snapshot["requested_timeout_seconds"] = round(requested, 3)
    snapshot["allowed_timeout_seconds"] = round(min(requested, allowed), 3)
    snapshot["limited"] = bool(allowed + 1e-9 < requested)
    return min(requested, allowed), snapshot


def _analysis_budget_reason(snapshot: dict[str, Any] | None) -> str:
    scope = str((snapshot or {}).get("scope") or "analysis")
    remaining = float((snapshot or {}).get("remaining_seconds") or 0.0)
    return (
        f"{scope} 剩余分析预算仅 {remaining:.2f} 秒，已跳过慢模型调用并使用本地保守结果；"
        "下一轮将使用最新行情重新分析。"
    )


def _attach_analysis_budget_timing(
    timing: dict[str, Any],
    snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    if snapshot is not None:
        timing["analysis_budget"] = dict(snapshot)
    return timing


class ModelRegistry:
    """Central registry for all AI models.

    Usage:
        registry = ModelRegistry()
        registry.register(LLMAgent())
        await registry.initialize_all()

        models = registry.get_all()
        live_model = registry.get_live_model()
    """

    def __init__(self) -> None:
        self._models: dict[str, AbstractAIModel] = {}
        self._live_model_name: str | None = None
        self._initialized = False
        self._batch_expert_disabled_until_by_provider: dict[tuple[str, str], float] = {}
        self._batch_expert_last_error_by_provider: dict[tuple[str, str], str] = {}

    def register(self, model: AbstractAIModel) -> None:
        """Register a model instance. Must have a unique name."""
        if model.name in self._models:
            logger.warning("model already registered, replacing", name=model.name)
        self._models[model.name] = model
        logger.info("model registered", name=model.name)

    def get(self, name: str) -> AbstractAIModel | None:
        return self._models.get(name)

    def get_all(self) -> list[AbstractAIModel]:
        return list(self._models.values())

    @property
    def model_names(self) -> list[str]:
        return list(self._models.keys())

    @property
    def model_count(self) -> int:
        return len(self._models)

    def get_live_model(self) -> AbstractAIModel | None:
        """Get the model currently selected for live trading."""
        if self._live_model_name:
            return self._models.get(self._live_model_name)
        return None

    def set_live_model(self, name: str) -> None:
        """Promote a model to be the live trading model."""
        if name not in self._models:
            raise ValueError(f"Model '{name}' is not registered.")
        self._live_model_name = name
        logger.info("live model set", name=name)

    @property
    def live_model_name(self) -> str | None:
        return self._live_model_name

    def unregister(self, name: str) -> bool:
        """Remove a model from the registry. Returns True if removed."""
        if name in self._models:
            del self._models[name]
            logger.info("model unregistered", name=name)
            if self._live_model_name == name:
                self._live_model_name = next(iter(self._models), None) if self._models else None
            return True
        return False

    async def sync_from_config(self) -> tuple[set[str], set[str]]:
        """Rebuild models from current settings.ai_models config.

        Clears existing models, creates new ones from config, and re-initializes.
        Returns (old_names, new_names) for the caller to sync other services.
        """
        from ai_brain.model_factory import create_models_from_config

        old_names = set(self._models.keys())
        self._models.clear()
        self._initialized = False

        for m in create_models_from_config():
            self.register(m)

        await self.initialize_all()

        new_names = set(self._models.keys())
        return old_names, new_names

    async def initialize_all(self) -> None:
        """Initialize all registered models concurrently."""
        if self._initialized:
            return

        models = list(self._models.values())
        tasks = [model.initialize() for model in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for model, result in zip(models, results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    "model init failed",
                    name=model.name,
                    error=safe_error_text(result),
                )
            else:
                logger.info("model initialized", name=model.name)

        self._initialized = True

        # Auto-select first model as live if none set
        if self._live_model_name is None and self._models:
            self._live_model_name = list(self._models.keys())[0]

    async def shutdown_all(self) -> None:
        """Shutdown all models gracefully."""
        tasks = [model.shutdown() for model in self._models.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._initialized = False
        logger.info("all models shut down")

    async def decide_all(
        self, features: FeatureVector, context: dict[str, Any]
    ) -> dict[str, DecisionOutput]:
        """Run decide() on all models concurrently and return results keyed by model name."""
        if not self._initialized:
            await self.initialize_all()

        excluded = {str(name) for name in (context.get("_exclude_model_names") or []) if str(name)}
        included = {str(name) for name in (context.get("_include_model_names") or []) if str(name)}
        active_models = [
            model
            for model in self._models.values()
            if model.name not in excluded and (not included or model.name in included)
        ]

        context["_attempted_models"] = [model.name for model in active_models]
        context["_model_failures"] = []
        context["_model_timings"] = []

        analysis_budget = _analysis_budget_snapshot(context)
        if analysis_budget is not None:
            context["_analysis_budget"] = dict(analysis_budget)
            if float(analysis_budget["remaining_seconds"] or 0.0) <= 0:
                context.setdefault("_skip_llm_experts_reason", _analysis_budget_reason(analysis_budget))
                context["_analysis_budget_deferred"] = True

        batchable_names = {
            "trend_expert",
            "momentum_expert",
            "sentiment_expert",
            "position_expert",
            "risk_expert",
        }
        skip_llm_reason = context.get("_skip_llm_experts_reason")
        if (
            skip_llm_reason
            and active_models
            and {model.name for model in active_models}.issubset(batchable_names)
        ):
            started_at = datetime.now(UTC)
            fallback_decisions: dict[str, DecisionOutput] = {}
            fallback_timings: list[dict[str, Any]] = []
            reason = str(skip_llm_reason)[:240]
            budget_deferred = bool(context.get("_analysis_budget_deferred"))
            for model in active_models:
                local_fallback = _local_fallback_callable(model)
                if local_fallback is not None:
                    decision = local_fallback(
                        features,
                        {**context, "expert_mode": True},
                        reason,
                    )
                else:
                    decision = DecisionOutput(
                        model_name=model.name,
                        symbol=features.symbol,
                        action=Action.HOLD,
                        confidence=0.0,
                        reasoning="市场快筛未发现正期望机会，本轮跳过大模型专家，快速观望。",
                        position_size_pct=0.0,
                        suggested_leverage=1.0,
                        stop_loss_pct=0.0,
                        take_profit_pct=0.0,
                        cross_check_for=None,
                        raw_response={
                            "market_fast_prefilter": True,
                            "provider_model": "local_fast_prefilter",
                            "reason": reason,
                        },
                        feature_snapshot=features.to_dict(),
                    )
                decision.model_name = model.name
                decision.raw_response = {
                    **(decision.raw_response or {}),
                    "market_fast_prefilter": True,
                    "provider_model": "local_fast_prefilter",
                    "reason": reason,
                }
                if budget_deferred:
                    decision.raw_response["analysis_budget_deferred"] = dict(
                        context.get("_analysis_budget") or {}
                    )
                fallback_decisions[model.name] = decision
                fallback_timings.append(
                    _attach_analysis_budget_timing(
                        {
                        "stage": "expert_initial",
                        "name": model.name,
                        "status": (
                            "analysis_budget_deferred" if budget_deferred else "fast_prefilter"
                        ),
                        "started_at": started_at.isoformat(),
                        "duration_sec": 0.0,
                        "batch_expert": False,
                        "shared_batch_call": False,
                        "action": decision.action.value,
                        "confidence": decision.confidence,
                        "provider_model": "local_fast_prefilter",
                        "reason": reason,
                        },
                        context.get("_analysis_budget"),
                    )
                )
            context["_model_timings"] = fallback_timings
            return fallback_decisions

        if (
            settings.ai_batch_experts_enabled
            and len(active_models) >= 3
            and {model.name for model in active_models}.issubset(batchable_names)
        ):
            grouped_models = _group_batchable_models_by_provider(active_models)
            all_decisions: dict[str, DecisionOutput] = {}
            all_timings: list[dict[str, Any]] = []
            for provider_group in grouped_models:
                batch_model = next(
                    (model for model in provider_group if _batch_expert_decider(model) is not None),
                    provider_group[0],
                )
                provider_key = _provider_group_key(batch_model)
                expert_names = [model.name for model in provider_group]
                started_at = datetime.now(UTC)
                perf_started = time.perf_counter()
                if _batch_expert_decider(batch_model) is None:
                    reason = "provider model does not support safe batch expert JSON"
                    retry_context = dict(context)
                    retry_context["expert_mode"] = True
                    retry_context["_force_independent_expert"] = True
                    retry_context["_force_fast_independent_expert"] = True
                    retry_context["_provider_independent_expert_mode"] = True
                    retry_context["_batch_not_supported_independent"] = {
                        "provider_model": _provider_model_name(batch_model),
                        "reason": reason[:240],
                    }
                    fallback, fallback_timings = await self._retry_provider_group_independently(
                        features,
                        retry_context,
                        provider_group,
                        batch_model,
                        started_at,
                        _positive_duration_seconds(perf_started),
                        reason,
                        status="batch_not_supported_independent",
                    )
                    all_decisions.update(fallback)
                    all_timings.extend(fallback_timings)
                    continue
                now_perf = time.perf_counter()
                disabled_until = self._batch_expert_disabled_until_by_provider.get(
                    provider_key, 0.0
                )
                if disabled_until > now_perf:
                    reason = (
                        "batch expert circuit breaker active after recent timeout: "
                        f"{self._batch_expert_last_error_by_provider.get(provider_key) or 'recent batch expert failure'}"
                    )
                    logger.warning(
                        "batch expert circuit breaker active, retrying experts independently",
                        reason=reason,
                    )
                    fallback, fallback_timings = await self._retry_provider_group_independently(
                        features,
                        context,
                        provider_group,
                        batch_model,
                        started_at,
                        round(time.perf_counter() - perf_started, 3),
                        reason,
                        status="circuit_breaker_fallback",
                    )
                    all_decisions.update(fallback)
                    all_timings.extend(fallback_timings)
                    continue
                try:
                    batch_decider = _batch_expert_decider(batch_model)
                    if batch_decider is not None:
                        requested_batch_timeout = max(
                            float(settings.ai_batch_expert_timeout_seconds or 18.0),
                            8.0,
                        )
                        batch_timeout, budget_snapshot = _bounded_analysis_timeout(
                            context,
                            requested_batch_timeout,
                        )
                        if batch_timeout <= 0:
                            reason = _analysis_budget_reason(budget_snapshot)
                            fallback, fallback_timings = self._batch_local_fallback_decisions(
                                features,
                                context,
                                provider_group,
                                batch_model,
                                started_at,
                                _positive_duration_seconds(perf_started),
                                reason,
                                status="analysis_budget_deferred",
                            )
                            for timing in fallback_timings:
                                _attach_analysis_budget_timing(timing, budget_snapshot)
                            all_decisions.update(fallback)
                            all_timings.extend(fallback_timings)
                            continue
                        result = await asyncio.wait_for(
                            batch_decider(features, context, expert_names),
                            timeout=batch_timeout,
                        )
                    else:
                        raise RuntimeError(
                            "batch expert model does not expose decide_batch_experts"
                        )
                    self._batch_expert_disabled_until_by_provider.pop(provider_key, None)
                    self._batch_expert_last_error_by_provider.pop(provider_key, None)
                    duration = round(time.perf_counter() - perf_started, 3)
                    for model in provider_group:
                        batch_decision = result.get(model.name)
                        all_timings.append(
                            {
                                "stage": "expert_initial",
                                "name": model.name,
                                "status": self._batch_timing_status(batch_decision),
                                "started_at": started_at.isoformat(),
                                "duration_sec": duration,
                                "batch_expert": True,
                                "shared_batch_call": True,
                                "batch_model_count": len(provider_group),
                                "batch_provider_group_count": len(grouped_models),
                                "duration_kind": "shared_wall_time",
                                "action": (
                                    batch_decision.action.value
                                    if isinstance(batch_decision, DecisionOutput)
                                    else None
                                ),
                                "confidence": (
                                    batch_decision.confidence
                                    if isinstance(batch_decision, DecisionOutput)
                                    else None
                                ),
                                "provider_model": (
                                    batch_decision.raw_response.get("provider_model")
                                    if isinstance(batch_decision, DecisionOutput)
                                    and isinstance(batch_decision.raw_response, dict)
                                    else _provider_model_name(batch_model)
                                ),
                            }
                        )
                    group_decisions = {
                        name: decision
                        for name, decision in result.items()
                        if isinstance(decision, DecisionOutput)
                    }
                    all_decisions.update(group_decisions)
                except Exception as exc:
                    duration = round(time.perf_counter() - perf_started, 3)
                    error_text = safe_error_text(exc, limit=240)
                    breaker_seconds = _batch_failure_breaker_seconds(exc, error_text)
                    self._batch_expert_last_error_by_provider[provider_key] = error_text
                    if breaker_seconds > 0:
                        self._batch_expert_disabled_until_by_provider[provider_key] = (
                            time.perf_counter() + breaker_seconds
                        )
                    else:
                        self._batch_expert_disabled_until_by_provider.pop(provider_key, None)
                    logger.warning(
                        "batch expert decide failed, retrying experts independently",
                        provider_model=_provider_model_name(batch_model),
                        experts=expert_names,
                        error=error_text,
                    )
                    context["_model_failures"].append(
                        {
                            "expert_name": "batch_experts",
                            "provider_model": _provider_model_name(batch_model),
                            "experts": expert_names,
                            "reason": error_text,
                        }
                    )
                    fallback, fallback_timings = await self._retry_provider_group_independently(
                        features,
                        context,
                        provider_group,
                        batch_model,
                        started_at,
                        duration,
                        f"batch expert failed: {error_text}",
                        status="batch_fallback",
                    )
                    all_decisions.update(fallback)
                    all_timings.extend(fallback_timings)

            diversity_review = review_batch_expert_consensus(features, context, all_decisions)
            context["_expert_diversity_policy"] = diversity_review.to_dict()
            if diversity_review.should_retry:
                retry_decisions, retry_timings = await self._retry_independent_experts(
                    features=features,
                    context=context,
                    active_models=active_models,
                    original_decisions=all_decisions,
                    review=diversity_review,
                )
                all_decisions.update(retry_decisions)
                all_timings.extend(retry_timings)
            context["_model_timings"] = all_timings
            return all_decisions

        def _timeout_fallback_decision(
            model: AbstractAIModel,
            duration: float,
            timeout_seconds: float,
        ) -> DecisionOutput:
            provider_model = getattr(model, "_model_name", None)
            reason = (
                f"{model.name} 超过 {timeout_seconds:.0f} 秒未返回，本轮按中性处理，"
                "不参与方向投票，避免行情信号过期。"
            )
            return DecisionOutput(
                model_name=model.name,
                symbol=features.symbol,
                action=Action.HOLD,
                confidence=0.0,
                reasoning=reason,
                position_size_pct=0.0,
                suggested_leverage=1.0,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                cross_check_for=None,
                raw_response={
                    "timeout_fallback": True,
                    "provider_model": provider_model,
                    "timeout_seconds": timeout_seconds,
                    "duration_sec": duration,
                    "reason": reason,
                },
                feature_snapshot=features.to_dict(),
            )

        async def _timed_decide(
            model: AbstractAIModel,
        ) -> tuple[AbstractAIModel, Any, dict[str, Any]]:
            started_at = datetime.now(UTC)
            perf_started = time.perf_counter()
            try:
                base_timeout = max(float(settings.ai_expert_timeout_seconds or 30.0), 5.0)
                active_count = max(len(active_models), 1)
                concurrency = max(int(settings.ai_llm_concurrency or active_count), 1)
                queue_batches = max((active_count + concurrency - 1) // concurrency, 1)
                timeout_seconds = base_timeout * queue_batches
                result = await asyncio.wait_for(
                    model.decide(features, context),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                duration = round(time.perf_counter() - perf_started, 3)
                result = _timeout_fallback_decision(model, duration, timeout_seconds)
                return (
                    model,
                    result,
                    {
                        "stage": "expert_initial",
                        "name": model.name,
                        "status": "timeout_fallback",
                        "started_at": started_at.isoformat(),
                        "duration_sec": duration,
                        "timeout_seconds": timeout_seconds,
                        "action": result.action.value,
                        "confidence": result.confidence,
                        "provider_model": (
                            result.raw_response.get("provider_model")
                            if result.raw_response
                            else None
                        ),
                        "reason": result.reasoning,
                    },
                )
            except Exception as exc:
                duration = round(time.perf_counter() - perf_started, 3)
                return (
                    model,
                    exc,
                    {
                        "stage": "expert_initial",
                        "name": model.name,
                        "status": "failed",
                        "started_at": started_at.isoformat(),
                        "duration_sec": duration,
                        "reason": safe_error_text(exc, limit=240),
                    },
                )

            duration = round(time.perf_counter() - perf_started, 3)
            timing = {
                "stage": "expert_initial",
                "name": model.name,
                "status": "completed" if isinstance(result, DecisionOutput) else "invalid",
                "started_at": started_at.isoformat(),
                "duration_sec": duration,
            }
            if isinstance(result, DecisionOutput):
                timing.update(
                    {
                        "action": result.action.value,
                        "confidence": result.confidence,
                    }
                )
                if isinstance(result.raw_response, dict):
                    provider_model = result.raw_response.get("provider_model")
                    fallback_from = result.raw_response.get("fallback_from")
                    local_fallback = bool(result.raw_response.get("local_fallback"))
                    if provider_model:
                        timing["provider_model"] = provider_model
                    if fallback_from:
                        timing["fallback_from"] = fallback_from
                    if local_fallback:
                        timing["local_fallback"] = True
            return model, result, timing

        tasks = [_timed_decide(model) for model in active_models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        decisions: dict[str, Any] = {}
        model_timings: list[dict[str, Any]] = []
        for item in results:
            if isinstance(item, BaseException):
                logger.error("model timed decide failed", error=safe_error_text(item))
                continue

            model, result, timing = item
            model_timings.append(timing)
            if isinstance(result, Exception):
                error_text = safe_error_text(result)
                logger.error("model decide failed", name=model.name, error=error_text)
                context["_model_failures"].append(
                    {
                        "expert_name": model.name,
                        "reason": error_text,
                    }
                )
            else:
                decisions[model.name] = result

        context["_model_timings"] = sorted(
            model_timings,
            key=lambda row: float(row.get("duration_sec") or 0.0),
            reverse=True,
        )
        return decisions

    async def _retry_provider_group_independently(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        active_models: list[AbstractAIModel],
        batch_model: AbstractAIModel,
        started_at: datetime,
        duration: float,
        reason: str,
        *,
        status: str,
    ) -> tuple[dict[str, DecisionOutput], list[dict[str, Any]]]:
        """Retry each expert with a real LLM call before using synthetic batch fallback."""

        retry_budget_timeout, retry_budget = _bounded_analysis_timeout(
            context,
            _independent_expert_timeout_seconds(len(active_models)),
        )
        if retry_budget is not None and retry_budget_timeout <= 0:
            fallback, fallback_timings = self._batch_local_fallback_decisions(
                features,
                context,
                active_models,
                batch_model,
                started_at,
                duration,
                _analysis_budget_reason(retry_budget),
                status="analysis_budget_deferred",
            )
            for timing in fallback_timings:
                _attach_analysis_budget_timing(timing, retry_budget)
            return fallback, fallback_timings

        retry_context = dict(context)
        retry_context["expert_mode"] = True
        retry_context["_force_independent_expert"] = True
        retry_context["_force_fast_independent_expert"] = True
        retry_context["_provider_independent_expert_mode"] = True
        if status == "batch_not_supported_independent":
            retry_context["_batch_not_supported_independent"] = {
                "reason": reason[:240],
                "provider_model": _provider_model_name(batch_model),
            }
        else:
            retry_context["_batch_failure_independent_retry"] = {
                "status": status,
                "reason": reason[:240],
                "provider_model": _provider_model_name(batch_model),
            }

        async def _retry_one(
            model: AbstractAIModel,
        ) -> tuple[AbstractAIModel, DecisionOutput | None, dict[str, Any] | None, str]:
            retry_started_at = datetime.now(UTC)
            perf_started = time.perf_counter()
            try:
                timeout_seconds, budget_snapshot = _bounded_analysis_timeout(
                    retry_context,
                    _independent_expert_timeout_seconds(len(active_models)),
                )
                if timeout_seconds <= 0:
                    return (
                        model,
                        None,
                        _attach_analysis_budget_timing(
                            {
                                "stage": "expert_independent_provider",
                                "name": model.name,
                                "status": "analysis_budget_deferred",
                                "started_at": retry_started_at.isoformat(),
                                "duration_sec": 0.0,
                                "timeout_seconds": 0.0,
                                "batch_expert": False,
                                "shared_batch_call": False,
                                "batch_failure_status": status,
                                "provider_independent_expert_mode": True,
                                "provider_model": _provider_model_name(model),
                                "reason": _analysis_budget_reason(budget_snapshot),
                            },
                            budget_snapshot,
                        ),
                        _analysis_budget_reason(budget_snapshot),
                    )
                result = await asyncio.wait_for(
                    model.decide(features, retry_context),
                    timeout=timeout_seconds,
                )
                retry_duration = _positive_duration_seconds(perf_started)
                if not isinstance(result, DecisionOutput):
                    return model, None, None, "independent retry returned invalid result"
                result.model_name = model.name
                raw = result.raw_response if isinstance(result.raw_response, dict) else {}
                if status == "batch_not_supported_independent":
                    raw["batch_not_supported_independent"] = True
                    raw["batch_not_supported_reason"] = reason[:240]
                else:
                    raw["batch_failure_independent_retry"] = True
                    raw["batch_failure_status"] = status
                    raw["batch_failure_reason"] = reason[:240]
                raw["provider_independent_expert_mode"] = True
                raw.setdefault(
                    "provider_model",
                    _provider_model_name(model) or _provider_model_name(batch_model),
                )
                result.raw_response = raw
                local_fallback = bool(raw.get("local_fallback"))
                return (
                    model,
                    result,
                    _attach_analysis_budget_timing(
                        {
                        "stage": "expert_independent_provider",
                        "name": model.name,
                        "status": "completed",
                        "started_at": retry_started_at.isoformat(),
                        "duration_sec": retry_duration,
                        "timeout_seconds": timeout_seconds,
                        "batch_expert": False,
                        "shared_batch_call": False,
                        "batch_failure_status": status,
                        "provider_independent_expert_mode": True,
                        "local_fallback": local_fallback,
                        "action": result.action.value,
                        "confidence": result.confidence,
                        "provider_model": raw.get("provider_model"),
                        "reason": reason[:240],
                        },
                        budget_snapshot,
                    ),
                    "",
                )
            except Exception as exc:
                retry_duration = _positive_duration_seconds(perf_started)
                error_text = safe_error_text(exc, limit=240)
                context.setdefault("_model_failures", []).append(
                    {
                        "expert_name": model.name,
                        "provider_model": _provider_model_name(model),
                        "reason": f"batch independent retry failed: {error_text}",
                    }
                )
                return (
                    model,
                    None,
                    _attach_analysis_budget_timing(
                        {
                        "stage": "expert_independent_provider",
                        "name": model.name,
                        "status": "independent_provider_failed",
                        "started_at": retry_started_at.isoformat(),
                        "duration_sec": retry_duration,
                        "timeout_seconds": timeout_seconds,
                        "batch_expert": False,
                        "shared_batch_call": False,
                        "batch_failure_status": status,
                        "provider_independent_expert_mode": True,
                        "provider_model": _provider_model_name(model),
                        "reason": error_text,
                        },
                        budget_snapshot if "budget_snapshot" in locals() else None,
                    ),
                    error_text,
                )

        results = await asyncio.gather(*[_retry_one(model) for model in active_models])
        retry_decisions: dict[str, DecisionOutput] = {}
        retry_timings: list[dict[str, Any]] = []
        failed_models: list[AbstractAIModel] = []
        failed_reasons: dict[str, str] = {}
        for model, decision, timing, error_text in results:
            if isinstance(decision, DecisionOutput) and isinstance(timing, dict):
                retry_decisions[model.name] = decision
                retry_timings.append(timing)
            else:
                if isinstance(timing, dict):
                    retry_timings.append(timing)
                failed_models.append(model)
                failed_reasons[model.name] = error_text
        if not failed_models:
            return retry_decisions, retry_timings

        fallback, fallback_timings = self._batch_local_fallback_decisions(
            features,
            context,
            failed_models,
            batch_model,
            started_at,
            duration,
            reason,
            status="independent_provider_fallback",
        )
        for row in fallback_timings:
            name = str(row.get("name") or "")
            row["independent_retry_failed"] = True
            row["independent_retry_status"] = "independent_provider_failed"
            try:
                row["duration_sec"] = max(float(row.get("duration_sec") or 0.0), 0.001)
            except (TypeError, ValueError):
                row["duration_sec"] = 0.001
            if failed_reasons.get(name):
                row["independent_retry_error"] = failed_reasons[name][:240]
                row["reason"] = failed_reasons[name][:240]
        retry_decisions.update(fallback)
        fallback_by_name = {
            str(row.get("name") or ""): row
            for row in fallback_timings
            if isinstance(row, dict) and row.get("name")
        }
        merged_timings: list[dict[str, Any]] = []
        replaced_names: set[str] = set()
        for row in retry_timings:
            name = str(row.get("name") or "")
            if name and name in fallback_by_name:
                fallback_row = fallback_by_name[name]
                try:
                    retry_duration = float(row.get("duration_sec") or 0.0)
                except (TypeError, ValueError):
                    retry_duration = 0.0
                fallback_row["duration_sec"] = max(
                    float(fallback_row.get("duration_sec") or 0.0),
                    retry_duration,
                )
                merged_timings.append(fallback_row)
                replaced_names.add(name)
            else:
                merged_timings.append(row)
        for name, row in fallback_by_name.items():
            if name not in replaced_names:
                merged_timings.append(row)
        retry_timings = merged_timings
        return retry_decisions, retry_timings

    async def _retry_independent_experts(
        self,
        *,
        features: FeatureVector,
        context: dict[str, Any],
        active_models: list[AbstractAIModel],
        original_decisions: dict[str, DecisionOutput],
        review: ExpertDiversityReview,
    ) -> tuple[dict[str, DecisionOutput], list[dict[str, Any]]]:
        """Retry selected experts independently when batch consensus looks collapsed."""

        target_names = set(review.target_experts)
        retry_models = [model for model in active_models if model.name in target_names]
        retry_context = dict(context)
        retry_context["expert_mode"] = True
        retry_context["_batch_consensus_retry"] = review.to_dict()
        retry_context["_force_independent_expert"] = True
        retry_context["_force_fast_independent_expert"] = True
        retry_context["_provider_independent_expert_mode"] = True

        retry_budget_timeout, retry_budget = _bounded_analysis_timeout(
            retry_context,
            _independent_expert_timeout_seconds(len(retry_models)),
        )
        if retry_budget is not None and retry_budget_timeout <= 0:
            return (
                {},
                [
                    _attach_analysis_budget_timing(
                        {
                            "stage": "expert_independent_retry",
                            "name": model.name,
                            "status": "analysis_budget_deferred",
                            "started_at": datetime.now(UTC).isoformat(),
                            "duration_sec": 0.0,
                            "timeout_seconds": 0.0,
                            "reason": _analysis_budget_reason(retry_budget),
                            "replaces_batch_decision": False,
                        },
                        retry_budget,
                    )
                    for model in retry_models
                ],
            )

        async def _retry_one(
            model: AbstractAIModel,
        ) -> tuple[str, DecisionOutput | None, dict[str, Any]]:
            started_at = datetime.now(UTC)
            perf_started = time.perf_counter()
            try:
                timeout_seconds, budget_snapshot = _bounded_analysis_timeout(
                    retry_context,
                    _independent_expert_timeout_seconds(len(retry_models)),
                )
                if timeout_seconds <= 0:
                    return (
                        model.name,
                        None,
                        _attach_analysis_budget_timing(
                            {
                                "stage": "expert_independent_retry",
                                "name": model.name,
                                "status": "analysis_budget_deferred",
                                "started_at": started_at.isoformat(),
                                "duration_sec": 0.0,
                                "timeout_seconds": 0.0,
                                "reason": _analysis_budget_reason(budget_snapshot),
                                "replaces_batch_decision": False,
                            },
                            budget_snapshot,
                        ),
                    )
                result = await asyncio.wait_for(
                    model.decide(features, retry_context),
                    timeout=timeout_seconds,
                )
                duration = round(time.perf_counter() - perf_started, 3)
                if not isinstance(result, DecisionOutput):
                    return (
                        model.name,
                        None,
                        {
                            "stage": "expert_independent_retry",
                            "name": model.name,
                            "status": "invalid",
                            "started_at": started_at.isoformat(),
                            "duration_sec": duration,
                            "timeout_seconds": timeout_seconds,
                            "reason": "independent expert retry returned invalid result",
                        },
                    )

                original = original_decisions.get(model.name)
                result.model_name = model.name
                raw = result.raw_response if isinstance(result.raw_response, dict) else {}
                raw["independent_expert_retry"] = True
                raw["batch_consensus_review"] = review.to_dict()
                raw["provider_independent_expert_mode"] = True
                if isinstance(original, DecisionOutput):
                    raw["batch_original"] = {
                        "action": original.action.value,
                        "confidence": original.confidence,
                        "reasoning": original.reasoning,
                    }
                result.raw_response = raw
                timing: dict[str, Any] = _attach_analysis_budget_timing({
                    "stage": "expert_independent_retry",
                    "name": model.name,
                    "status": "completed",
                    "started_at": started_at.isoformat(),
                    "duration_sec": duration,
                    "timeout_seconds": timeout_seconds,
                    "action": result.action.value,
                    "confidence": result.confidence,
                    "replaces_batch_decision": True,
                    "objective_side": review.objective_evidence.side,
                    "objective_score": review.objective_evidence.score,
                    "provider_independent_expert_mode": True,
                }, budget_snapshot)
                if isinstance(result.raw_response, dict) and result.raw_response.get(
                    "provider_model"
                ):
                    timing["provider_model"] = result.raw_response.get("provider_model")
                return model.name, result, timing
            except Exception as exc:
                duration = round(time.perf_counter() - perf_started, 3)
                error_text = safe_error_text(exc, limit=240)
                context.setdefault("_model_failures", []).append(
                    {"expert_name": model.name, "reason": f"independent retry failed: {error_text}"}
                )
                return (
                    model.name,
                    None,
                    _attach_analysis_budget_timing(
                        {
                        "stage": "expert_independent_retry",
                        "name": model.name,
                        "status": "failed",
                        "started_at": started_at.isoformat(),
                        "duration_sec": duration,
                        "timeout_seconds": timeout_seconds,
                        "reason": error_text,
                        "replaces_batch_decision": False,
                        },
                        budget_snapshot if "budget_snapshot" in locals() else None,
                    ),
                )

        results = await asyncio.gather(*[_retry_one(model) for model in retry_models])
        retry_decisions: dict[str, DecisionOutput] = {}
        retry_timings: list[dict[str, Any]] = []
        for name, decision, timing in results:
            retry_timings.append(timing)
            if isinstance(decision, DecisionOutput):
                retry_decisions[name] = decision
        return retry_decisions, retry_timings

    def _batch_local_fallback_decisions(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        active_models: list[AbstractAIModel],
        batch_model: AbstractAIModel,
        started_at: datetime,
        duration: float,
        reason: str,
        *,
        status: str,
    ) -> tuple[dict[str, DecisionOutput], list[dict[str, Any]]]:
        fallback_decisions: dict[str, DecisionOutput] = {}
        fallback_timings: list[dict[str, Any]] = []
        for model in active_models:
            local_fallback = _local_fallback_callable(model)
            if local_fallback is not None:
                decision = local_fallback(
                    features,
                    {**context, "expert_mode": True},
                    reason[:160],
                )
            else:
                decision = DecisionOutput(
                    model_name=model.name,
                    symbol=features.symbol,
                    action=Action.HOLD,
                    confidence=0.0,
                    reasoning="批量专家暂不可用，使用快速本地观望兜底。",
                    position_size_pct=0.0,
                    suggested_leverage=1.0,
                    stop_loss_pct=0.0,
                    take_profit_pct=0.0,
                    cross_check_for=None,
                    raw_response={
                        "batch_expert_fallback": True,
                        "provider_model": _provider_model_name(batch_model),
                        "reason": reason[:240],
                    },
                    feature_snapshot=features.to_dict(),
                )
            decision.model_name = model.name
            decision.raw_response = {
                **(decision.raw_response or {}),
                "batch_expert": True,
                "batch_expert_fallback": True,
                "provider_model": _provider_model_name(batch_model),
                "reason": reason[:240],
            }
            fallback_decisions[model.name] = decision
            fallback_timings.append(
                {
                    "stage": "expert_initial",
                    "name": model.name,
                    "status": status,
                    "started_at": started_at.isoformat(),
                    "duration_sec": duration,
                    "batch_expert": True,
                    "shared_batch_call": True,
                    "batch_model_count": len(active_models),
                    "duration_kind": "shared_wall_time",
                    "action": decision.action.value,
                    "confidence": decision.confidence,
                    "provider_model": _provider_model_name(batch_model),
                    "reason": reason[:240],
                }
            )
        return fallback_decisions, fallback_timings

    @staticmethod
    def _batch_timing_status(batch_decision: Any) -> str:
        if not isinstance(batch_decision, DecisionOutput):
            return "invalid"
        raw = batch_decision.raw_response
        if isinstance(raw, dict) and raw.get("batch_not_supported_independent"):
            return "completed"
        if isinstance(raw, dict) and raw.get("batch_expert_fallback"):
            return "partial_batch_fallback"
        return "completed"

    def get_state(self) -> dict:
        return {
            "models": self.model_names,
            "model_count": self.model_count,
            "live_model": self._live_model_name,
            "initialized": self._initialized,
        }
