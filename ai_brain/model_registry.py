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

from ai_brain.base_model import AbstractAIModel, DecisionOutput
from ai_brain.expert_diversity_policy import (
    ExpertDiversityReview,
    review_batch_expert_consensus,
)
from config.settings import settings
from core.safe_output import safe_error_text
from data_feed.feature_vector import FeatureVector

logger = structlog.get_logger(__name__)

BatchExpertDecider = Callable[
    [FeatureVector, dict[str, Any], list[str]],
    Awaitable[dict[str, DecisionOutput]],
]


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


def _batch_expert_decider(model: AbstractAIModel) -> BatchExpertDecider | None:
    if getattr(model, "_llm", None) is None:
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
    return f"{scope} 剩余分析预算仅 {remaining:.2f} 秒，未执行模型调用；本轮没有可用的专家决策。"


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
        active_model = registry.get_active_model()
    """

    def __init__(self) -> None:
        self._models: dict[str, AbstractAIModel] = {}
        self._active_model_name: str | None = None
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

    def get_active_model(self) -> AbstractAIModel | None:
        """Get the unified model selected for paper and live trading."""
        if self._active_model_name:
            return self._models.get(self._active_model_name)
        return None

    def set_active_model(self, name: str) -> None:
        """Select the one model used in both execution modes."""
        if name not in self._models:
            raise ValueError(f"Model '{name}' is not registered.")
        self._active_model_name = name
        logger.info("active model set", name=name)

    @property
    def active_model_name(self) -> str | None:
        return self._active_model_name

    def unregister(self, name: str) -> bool:
        """Remove a model from the registry. Returns True if removed."""
        if name in self._models:
            del self._models[name]
            logger.info("model unregistered", name=name)
            if self._active_model_name == name:
                self._active_model_name = next(iter(self._models), None) if self._models else None
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

        # Keep one mode-independent model pointer across registry refreshes.
        if self._active_model_name not in self._models and self._models:
            self._active_model_name = list(self._models.keys())[0]

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
                context.setdefault(
                    "_skip_llm_experts_reason", _analysis_budget_reason(analysis_budget)
                )
                context["_analysis_budget_deferred"] = True

        batchable_names = {
            "trend_expert",
            "momentum_expert",
            "sentiment_expert",
            "position_expert",
            "risk_expert",
        }
        skip_llm_reason = context.get("_skip_llm_experts_reason")
        if skip_llm_reason and active_models:
            reason = str(skip_llm_reason)[:240]
            context["_model_timings"] = [
                _attach_analysis_budget_timing(
                    {
                        "stage": "expert_initial",
                        "name": model.name,
                        "status": "analysis_budget_deferred",
                        "started_at": datetime.now(UTC).isoformat(),
                        "duration_sec": 0.0,
                        "batch_expert": False,
                        "shared_batch_call": False,
                        "provider_model": _provider_model_name(model),
                        "reason": reason,
                    },
                    context.get("_analysis_budget"),
                )
                for model in active_models
            ]
            context["_model_failures"].extend(
                {
                    "expert_name": model.name,
                    "provider_model": _provider_model_name(model),
                    "reason": reason,
                    "status": "analysis_budget_deferred",
                }
                for model in active_models
            )
            return {}

        if (
            settings.ai_batch_experts_enabled
            and len(active_models) >= 3
            and str(context.get("execution_mode") or "").lower() != "paper"
            and {model.name for model in active_models}.issubset(batchable_names)
        ):
            grouped_models = _group_batchable_models_by_provider(active_models)
            all_decisions: dict[str, DecisionOutput] = {}
            all_timings: list[dict[str, Any]] = []
            provider_results = await asyncio.gather(
                *[
                    self._decide_provider_group(
                        features,
                        context,
                        provider_group,
                        provider_group_count=len(grouped_models),
                    )
                    for provider_group in grouped_models
                ]
            )
            for group_decisions, group_timings in provider_results:
                all_decisions.update(group_decisions)
                all_timings.extend(group_timings)

            if all_decisions:
                diversity_review = review_batch_expert_consensus(
                    features,
                    context,
                    all_decisions,
                )
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
            else:
                context["_expert_diversity_policy"] = {
                    "status": "no_valid_decisions",
                    "should_retry": False,
                }
            context["_model_timings"] = all_timings
            return all_decisions

        if str(context.get("execution_mode") or "").lower() == "paper":
            context["_force_independent_expert"] = True
            # Paper analysis keeps experts independent for per-role attribution,
            # but must use the compact diagnostic contract. The former full
            # multidimensional prompt made five experts emit oversized JSON,
            # causing vLLM queueing and truncated responses.
            context["_force_fast_independent_expert"] = True
            context["_provider_independent_expert_mode"] = True

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
                return (
                    model,
                    TimeoutError(
                        f"expert model {model.name} timed out after {timeout_seconds:.1f}s"
                    ),
                    {
                        "stage": "expert_initial",
                        "name": model.name,
                        "status": "timeout",
                        "started_at": started_at.isoformat(),
                        "duration_sec": duration,
                        "timeout_seconds": timeout_seconds,
                        "provider_model": _provider_model_name(model),
                        "reason": f"expert model timed out after {timeout_seconds:.1f}s",
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
                    if provider_model:
                        timing["provider_model"] = provider_model
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

    async def _decide_provider_group(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        provider_group: list[AbstractAIModel],
        *,
        provider_group_count: int,
    ) -> tuple[dict[str, DecisionOutput], list[dict[str, Any]]]:
        """Run one provider group independently from other provider endpoints."""

        batch_model = next(
            (model for model in provider_group if _batch_expert_decider(model) is not None),
            provider_group[0],
        )
        provider_key = _provider_group_key(batch_model)
        expert_names = [model.name for model in provider_group]
        started_at = datetime.now(UTC)
        perf_started = time.perf_counter()
        batch_decider = _batch_expert_decider(batch_model)
        if batch_decider is None:
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
            return await self._retry_provider_group_independently(
                features,
                retry_context,
                provider_group,
                batch_model,
                reason,
                status="batch_not_supported_independent",
            )

        now_perf = time.perf_counter()
        disabled_until = self._batch_expert_disabled_until_by_provider.get(provider_key, 0.0)
        if disabled_until > now_perf:
            reason = (
                "batch expert circuit breaker active after recent timeout: "
                f"{self._batch_expert_last_error_by_provider.get(provider_key) or 'recent batch expert failure'}"
            )
            logger.warning(
                "batch expert circuit breaker active, retrying experts independently",
                reason=reason,
            )
            return await self._retry_provider_group_independently(
                features,
                context,
                provider_group,
                batch_model,
                reason,
                status="circuit_breaker_independent",
            )

        try:
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
                timings = [
                    _attach_analysis_budget_timing(
                        {
                            "stage": "expert_initial",
                            "name": model.name,
                            "status": "analysis_budget_deferred",
                            "started_at": started_at.isoformat(),
                            "duration_sec": 0.0,
                            "batch_expert": True,
                            "shared_batch_call": True,
                            "batch_model_count": len(provider_group),
                            "provider_model": _provider_model_name(batch_model),
                            "reason": reason,
                        },
                        budget_snapshot,
                    )
                    for model in provider_group
                ]
                context["_model_failures"].extend(
                    {
                        "expert_name": model.name,
                        "provider_model": _provider_model_name(model),
                        "reason": reason,
                        "status": "analysis_budget_deferred",
                    }
                    for model in provider_group
                )
                return {}, timings

            result = await asyncio.wait_for(
                batch_decider(features, context, expert_names),
                timeout=batch_timeout,
            )
            self._batch_expert_disabled_until_by_provider.pop(provider_key, None)
            self._batch_expert_last_error_by_provider.pop(provider_key, None)
            duration = round(time.perf_counter() - perf_started, 3)
            timings: list[dict[str, Any]] = []
            for model in provider_group:
                batch_decision = result.get(model.name)
                timings.append(
                    {
                        "stage": "expert_initial",
                        "name": model.name,
                        "status": self._batch_timing_status(batch_decision),
                        "started_at": started_at.isoformat(),
                        "duration_sec": duration,
                        "batch_expert": True,
                        "shared_batch_call": True,
                        "batch_model_count": len(provider_group),
                        "batch_provider_group_count": provider_group_count,
                        "provider_groups_concurrent": provider_group_count > 1,
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
            decisions = {
                name: decision
                for name, decision in result.items()
                if isinstance(decision, DecisionOutput)
            }
            return decisions, timings
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
            return await self._retry_provider_group_independently(
                features,
                context,
                provider_group,
                batch_model,
                f"batch expert failed: {error_text}",
                status="batch_failed_independent",
            )

    async def _retry_provider_group_independently(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        active_models: list[AbstractAIModel],
        batch_model: AbstractAIModel,
        reason: str,
        *,
        status: str,
    ) -> tuple[dict[str, DecisionOutput], list[dict[str, Any]]]:
        """Retry each expert with a real provider call after batch failure."""

        retry_budget_timeout, retry_budget = _bounded_analysis_timeout(
            context,
            _independent_expert_timeout_seconds(len(active_models)),
        )
        if retry_budget is not None and retry_budget_timeout <= 0:
            reason = _analysis_budget_reason(retry_budget)
            timings = [
                _attach_analysis_budget_timing(
                    {
                        "stage": "expert_independent_provider",
                        "name": model.name,
                        "status": "analysis_budget_deferred",
                        "started_at": datetime.now(UTC).isoformat(),
                        "duration_sec": 0.0,
                        "timeout_seconds": 0.0,
                        "batch_expert": False,
                        "shared_batch_call": False,
                        "batch_failure_status": status,
                        "provider_independent_expert_mode": True,
                        "provider_model": _provider_model_name(model),
                        "reason": reason,
                    },
                    retry_budget,
                )
                for model in active_models
            ]
            context["_model_failures"].extend(
                {
                    "expert_name": model.name,
                    "provider_model": _provider_model_name(model),
                    "reason": reason,
                    "status": "analysis_budget_deferred",
                }
                for model in active_models
            )
            return {}, timings

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
        for model, decision, timing, _error_text in results:
            if isinstance(decision, DecisionOutput) and isinstance(timing, dict):
                retry_decisions[model.name] = decision
                retry_timings.append(timing)
            else:
                if isinstance(timing, dict):
                    retry_timings.append(timing)
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
                timing: dict[str, Any] = _attach_analysis_budget_timing(
                    {
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
                    },
                    budget_snapshot,
                )
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

    @staticmethod
    def _batch_timing_status(batch_decision: Any) -> str:
        if not isinstance(batch_decision, DecisionOutput):
            return "invalid"
        raw = batch_decision.raw_response
        if isinstance(raw, dict) and raw.get("batch_not_supported_independent"):
            return "completed"
        return "completed"

    def get_state(self) -> dict:
        return {
            "models": self.model_names,
            "model_count": self.model_count,
            "active_model": self._active_model_name,
            "initialized": self._initialized,
        }
