"""Cross-expert validation for the multi-model ensemble.

The validator turns one-way expert reports into a lightweight consultation
graph. Experts can ask another expert to verify a concrete issue through
`cross_check_for`; this module evaluates the requested pair and optionally
uses the configured trend expert model to arbitrate major conflicts.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ai_brain.base_model import Action, DecisionOutput
from ai_brain.llm_agent import shared_llm_capacity_slot
from config.settings import DECISION_MAKER_NAME, settings
from core.model_runtime import (
    completion_token_limit,
    ensure_no_think_text,
    is_openai_reasoning_model,
    is_qwen3_model,
    non_thinking_extra_body,
    uses_thinking_tags,
)
from core.safe_output import safe_error_text
from core.secret_utils import secret_fingerprint

logger = structlog.get_logger(__name__)

EXPERT_ALIASES = {
    "trend": "trend_expert",
    "trend_expert": "trend_expert",
    "technical_trend": "trend_expert",
    "trend_direction": "trend_expert",
    "momentum": "momentum_expert",
    "momentum_expert": "momentum_expert",
    "short_term_momentum": "momentum_expert",
    "profit_quality": "momentum_expert",
    "sentiment": "sentiment_expert",
    "sentiment_expert": "sentiment_expert",
    "sentiment_news": "sentiment_expert",
    "short_timeseries": "sentiment_expert",
    "position": "position_expert",
    "position_expert": "position_expert",
    "position_manager": "position_expert",
    "position_exit": "position_expert",
    "risk": "risk_expert",
    "risk_expert": "risk_expert",
    "risk_guardian": "risk_expert",
    "risk_anomaly": "risk_expert",
}

ACTION_DIRECTION = {
    Action.LONG: 1,
    Action.CLOSE_SHORT: 1,
    Action.SHORT: -1,
    Action.CLOSE_LONG: -1,
    Action.HOLD: 0,
}

_CONSULTATION_CONCURRENCY = max(int(settings.ai_llm_concurrency or 2), 1)
BACKUP_CONSULTATION_MODELS = ("qwen3-max", "deepseek-v3", "claude-opus-4-7")

_CONSULTATION_TIMEOUT_FLOOR_SECONDS = 6.0
_CONSULTATION_TIMEOUT_CAP_SECONDS = 18.0
_CONSULTATION_ATTEMPT_CAP_SECONDS = 8.0
_CONSULTATION_THINKING_ATTEMPT_CAP_SECONDS = 8.0
_CONSULTATION_FALLBACK_RESERVE_SECONDS = 2.0
# A consultation waits on the same capacity scheduler as every other model
# call. The old separate semaphore added a second FIFO queue and caused
# requests to expire before they reached the shared scheduler.
_CONSULTATION_QUEUE_WAIT_CAP_SECONDS = 3.0


class ConsultationQueueTimeoutError(TimeoutError):
    """Raised when a consultation becomes stale before inference starts."""

    def __init__(self, waited_seconds: float) -> None:
        self.waited_seconds = max(float(waited_seconds), 0.0)
        super().__init__(
            f"deep consultation queue wait exceeded {self.waited_seconds:.3f} seconds"
        )


def _consultation_budget_seconds() -> float:
    configured = float(settings.ai_decision_maker_timeout_seconds or 20.0)
    return min(
        max(configured * 0.6, _CONSULTATION_TIMEOUT_FLOOR_SECONDS),
        _CONSULTATION_TIMEOUT_CAP_SECONDS,
    )


def _is_reasoning_model(model: str | None) -> bool:
    return is_openai_reasoning_model(model)


def _is_local_qwen3_trade_model(model: str | None) -> bool:
    name = str(model or "").lower()
    return (name.startswith("qwen3-") and name.endswith("-trade")) or name == (
        "bb-finquant-expert-14b"
    )


def _is_loopback_api_base(api_base: str | None) -> bool:
    try:
        hostname = str(urlparse(str(api_base or "").strip()).hostname or "").lower()
    except ValueError:
        return False
    return hostname in {"127.0.0.1", "localhost", "::1"}


def _is_qwen3_model(model: str | None) -> bool:
    return is_qwen3_model(model)


def _uses_thinking_tags(model: str | None) -> bool:
    return uses_thinking_tags(model)


def _strip_qwen_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", str(text or ""), flags=re.IGNORECASE).strip()
    if cleaned.startswith("<think>") and "{" in cleaned:
        cleaned = cleaned[cleaned.find("{") :].strip()
    return cleaned


def _message_content_text(response: Any) -> str:
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, str):
        return _strip_qwen_thinking(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return _strip_qwen_thinking("\n".join(p for p in parts if p).strip())
    return _strip_qwen_thinking(str(content or ""))


class CrossValidator:
    """Build and evaluate cross-checks requested by expert models."""

    async def validate_all(
        self,
        opinions: dict[str, DecisionOutput],
        timing_context: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        validations: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str, str]] = set()
        validated_pairs: set[tuple[str, str]] = set()
        cross_started_at = datetime.now(UTC)
        cross_perf_started = time.perf_counter()

        for source_name, source in opinions.items():
            request = source.cross_check_for
            if not isinstance(request, dict):
                continue

            target_name = EXPERT_ALIASES.get(str(request.get("target", "")).strip().lower())
            question = str(request.get("question", "")).strip()
            if not target_name or target_name == source_name:
                continue
            if not question:
                continue

            key = (source_name, target_name, question)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            validated_pairs.add(tuple(sorted((source_name, target_name))))
            if target_name not in opinions:
                validations.append(
                    {
                        "expert_pair": [source_name, target_name],
                        "question": question,
                        "consistency": "neutral",
                        "production_permission": False,
                        "conflict_note": f"{self._expert_label(target_name)} 本轮没有返回，无法完成这次交叉验证。",
                        "validation_note": f"{self._expert_label(target_name)} 本轮没有返回，无法回答这个核实问题。",
                        "checked_evidence": [],
                        "major_conflict": False,
                        "validation_status": "target_missing",
                    }
                )
                continue
            validation = self.validate_pair(
                source_name,
                source,
                target_name,
                opinions[target_name],
                question,
            )
            validation["validation_origin"] = "expert_requested"
            validations.append(validation)

        # Every usable expert pair is checked deterministically. This adds no
        # model call and prevents a missing cross_check_for field from turning
        # a full expert pass into zero cross-validation evidence.
        opinion_names = sorted(
            name for name, decision in opinions.items() if isinstance(decision, DecisionOutput)
        )
        for index, source_name in enumerate(opinion_names):
            for target_name in opinion_names[index + 1 :]:
                pair = tuple(sorted((source_name, target_name)))
                if pair in validated_pairs:
                    continue
                validation = self.validate_pair(
                    source_name,
                    opinions[source_name],
                    target_name,
                    opinions[target_name],
                    "自动核对两位专家的方向与结构化证据是否一致。",
                )
                validation["validation_origin"] = "automatic_pairwise"
                validations.append(validation)
                validated_pairs.add(pair)

        cross_duration = round(time.perf_counter() - cross_perf_started, 3)
        if timing_context is not None:
            timing_context["_cross_validation_timing"] = {
                "stage": "cross_validation",
                "label": "交叉验证",
                "status": "completed",
                "started_at": cross_started_at.isoformat(),
                "duration_sec": cross_duration,
                "requested": sum(
                    1 for v in validations if v.get("validation_origin") == "expert_requested"
                ),
                "automatic": sum(
                    1 for v in validations if v.get("validation_origin") == "automatic_pairwise"
                ),
                "completed": sum(
                    1 for v in validations if v.get("validation_status", "completed") == "completed"
                ),
                "unavailable": sum(
                    1 for v in validations if v.get("validation_status") == "target_missing"
                ),
                "major_conflicts": sum(1 for v in validations if v.get("major_conflict")),
            }

        consultation_started_at = datetime.now(UTC)
        consultation_perf_started = time.perf_counter()
        consultation_timeout = _consultation_budget_seconds()
        major_conflicts = [v for v in validations if v.get("major_conflict")]
        try:
            consultation = await asyncio.wait_for(
                self.consult_if_needed(
                    opinions,
                    validations,
                    timeout_seconds=consultation_timeout,
                    capacity_context={
                        "_analysis_budget_scope": (
                            timing_context.get("_analysis_budget_scope", "shared")
                            if isinstance(timing_context, dict)
                            else "shared"
                        )
                    },
                ),
                timeout=consultation_timeout,
            )
        except TimeoutError:
            logger.warning(
                "deep consultation timed out; using fallback",
                timeout=consultation_timeout,
                major_conflicts=len(major_conflicts),
            )
            consultation = self._fallback_consultation(
                major_conflicts,
                "timeout",
                f"深度会诊超过 {consultation_timeout:.0f} 秒未返回",
            )
        consultation_duration = round(time.perf_counter() - consultation_perf_started, 3)
        if timing_context is not None:
            timing_context["_consultation_timing"] = {
                "stage": "deep_consultation",
                "label": "深度会诊",
                "status": (
                    str(consultation.get("status"))
                    if isinstance(consultation, dict) and consultation.get("status")
                    else "skipped"
                ),
                "started_at": consultation_started_at.isoformat(),
                "duration_sec": consultation_duration,
                "triggered": bool(consultation),
                "major_conflicts": sum(1 for v in validations if v.get("major_conflict")),
            }
        return validations, consultation

    def validate_pair(
        self,
        source_name: str,
        source: DecisionOutput,
        target_name: str,
        target: DecisionOutput,
        question: str,
    ) -> dict[str, Any]:
        source_dir = ACTION_DIRECTION.get(source.action, 0)
        target_dir = ACTION_DIRECTION.get(target.action, 0)
        question_result = self._question_result(source, target, target_name, question)
        validation_note = self._validation_note(
            source_name,
            source,
            target_name,
            target,
            question,
            question_result,
        )
        checked_evidence = self._checked_evidence(target_name, target)

        if source_dir and target_dir and source_dir == target_dir:
            consistency = "aligned"
            note = None
        elif source_dir and target_dir and source_dir != target_dir:
            consistency = "divergent"
            note = validation_note
        elif question_result == "supports" and source_dir:
            consistency = "aligned"
            note = None
        elif question_result == "challenges" and source_dir:
            consistency = "divergent"
            note = validation_note
        else:
            consistency = "neutral"
            note = None

        return {
            "expert_pair": [source_name, target_name],
            "question": question,
            "consistency": consistency,
            "production_permission": False,
            "conflict_note": note,
            "validation_note": validation_note,
            "checked_evidence": checked_evidence,
            "needs_resolution": consistency == "divergent",
            "major_conflict": consistency == "divergent",
            "validation_status": "completed",
        }

    def _question_result(
        self,
        source: DecisionOutput,
        target: DecisionOutput,
        target_name: str,
        question: str,
    ) -> str:
        """Describe expert agreement without creating a rule-based authorization path."""

        del target_name, question
        source_dir = ACTION_DIRECTION.get(source.action, 0)
        target_dir = ACTION_DIRECTION.get(target.action, 0)
        if not source_dir or not target_dir:
            return "neutral"
        return "supports" if source_dir == target_dir else "challenges"

    def _validation_note(
        self,
        source_name: str,
        source: DecisionOutput,
        target_name: str,
        target: DecisionOutput,
        question: str,
        question_result: str,
    ) -> str:
        result_label = {
            "supports": "支持",
            "challenges": "不支持",
            "neutral": "中性",
        }.get(question_result, "中性")
        target_reason = self._shorten(target.reasoning, 120)
        answer = {
            "supports": "目标专家给出的证据能回答这个问题，并支持发起专家的担心或方向。",
            "challenges": "目标专家给出的证据与发起专家的判断相冲突，需要在最终裁决中降权处理。",
            "neutral": "目标专家没有给出足够强的支持或反对证据，只作为中性信息处理。",
        }.get(question_result, "目标专家没有给出明确结论。")
        return (
            f"核验问题：{question}。"
            f"核验结论：{self._expert_label(target_name)} 对这个问题的判断为「{result_label}」。"
            f"{answer}"
            f"{self._expert_label(source_name)} 原判断是 {self._action_label(source.action)}"
            f"（信心度 {source.confidence:.2f}），"
            f"{self._expert_label(target_name)} 给出 {self._action_label(target.action)}"
            f"（信心度 {target.confidence:.2f}）。"
            f"依据：{target_reason or '目标专家没有给出详细理由'}"
        )

    def _checked_evidence(self, target_name: str, target: DecisionOutput) -> list[str]:
        snapshot = target.feature_snapshot or {}
        evidence: list[str] = []

        def add(label: str, key: str, digits: int = 2) -> None:
            value = snapshot.get(key)
            if value is None:
                return
            try:
                evidence.append(f"{label}={float(value):.{digits}f}")
            except (TypeError, ValueError):
                if str(value):
                    evidence.append(f"{label}={value}")

        if target_name == "trend_expert":
            add("ADX", "adx_14", 1)
            add("MACD柱", "macd_diff", 6)
            add("相对SMA20", "price_vs_sma20", 4)
            add("相对SMA50", "price_vs_sma50", 4)
            add("RSI14", "rsi_14", 1)
        elif target_name == "momentum_expert":
            add("量比", "volume_ratio", 2)
            add("1周期涨跌", "returns_1", 4)
            add("5周期涨跌", "returns_5", 4)
            add("20周期涨跌", "returns_20", 4)
            add("布林位置", "bb_pct", 2)
        elif target_name == "sentiment_expert":
            add("新闻情绪", "news_sentiment_avg", 3)
            add("社媒情绪", "social_sentiment_avg", 3)
            add("社媒提及", "social_mention_count", 0)
            add("新闻条数", "news_article_count", 0)
        elif target_name == "risk_expert":
            add("量比", "volume_ratio", 2)
            add("20周期波动", "volatility_20", 4)
            add("24h涨跌%", "change_24h_pct", 2)
            add("72h异常插针数", "abnormal_wick_count_72h", 0)
            add("最大异常插针%", "abnormal_wick_max_pct", 2)
            add("最近插针小时", "abnormal_wick_recent_hours", 1)
            add("ADX", "adx_14", 1)

        if not evidence:
            evidence.append(f"目标专家结论={self._action_label(target.action)}")
        return evidence

    def _fixed_model_cfg(self, name: str) -> dict[str, Any]:
        return next(
            (m for m in settings.get_fixed_ai_models(include_empty=True) if m.get("name") == name),
            {},
        )

    def _consultation_candidates(self, trend_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Return deep-consultation models in preferred failover order."""
        candidates: list[dict[str, Any]] = []

        def add_candidate(
            *,
            name: str,
            label: str,
            api_base: str | None,
            api_key: str | None,
            model: str | None,
            retries: int = 1,
            source: str = "primary",
        ) -> None:
            api_base = (api_base or "").strip()
            api_key = (api_key or "").strip()
            model = (model or "").strip()
            keyless_loopback = bool(model and not api_key and _is_loopback_api_base(api_base))
            if not model or (not api_key and not keyless_loopback):
                return
            identity = (
                api_base,
                model,
                secret_fingerprint(api_key) if api_key else "keyless-loopback",
            )
            for existing in candidates:
                if existing.get("_identity") == identity:
                    return
            candidates.append(
                {
                    "name": name,
                    "label": label,
                    "api_base": api_base,
                    "api_key": api_key,
                    "model": model,
                    "retries": max(int(retries or 1), 1),
                    "source": source,
                    "configuration_type": (
                        "keyless_loopback" if keyless_loopback else "api_key"
                    ),
                    "_identity": identity,
                }
            )

        add_candidate(
            name="trend_expert",
            label="行情方向专家",
            api_base=trend_cfg.get("api_base") or "",
            api_key=trend_cfg.get("api_key") or "",
            model=trend_cfg.get("model") or "",
            retries=1,
            source="primary",
        )

        decision_cfg = self._fixed_model_cfg(DECISION_MAKER_NAME)
        add_candidate(
            name=DECISION_MAKER_NAME,
            label=decision_cfg.get("label") or "最终交易员",
            api_base=decision_cfg.get("api_base") or "",
            api_key=decision_cfg.get("api_key") or "",
            model=decision_cfg.get("model") or "",
            retries=1,
            source="decision_maker",
        )

        if settings.high_risk_review_enabled:
            add_candidate(
                name="high_risk_review",
                label="High-risk review model",
                api_base=settings.high_risk_review_api_base,
                api_key=settings.high_risk_review_api_key,
                model=settings.high_risk_review_model,
                retries=1,
                source="high_risk_review",
            )

        primary_api_base = trend_cfg.get("api_base") or ""
        primary_api_key = trend_cfg.get("api_key") or ""
        primary_model = str(trend_cfg.get("model") or "").strip()
        if _is_local_qwen3_trade_model(primary_model):
            for candidate in candidates:
                candidate.pop("_identity", None)
            return candidates
        for backup_model in BACKUP_CONSULTATION_MODELS:
            if backup_model == primary_model:
                continue
            add_candidate(
                name="trend_backup",
                label="趋势备用会诊模型",
                api_base=primary_api_base,
                api_key=primary_api_key,
                model=backup_model,
                retries=1,
                source="backup",
            )

        for candidate in candidates:
            candidate.pop("_identity", None)
        return candidates

    def _consultation_attempt(
        self,
        candidate: dict[str, Any],
        attempt: int,
        status: str,
        message: str,
        raw_content: str | None = None,
        response: Any | None = None,
        runtime_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        item = {
            "expert": candidate.get("name"),
            "expert_label": candidate.get("label") or candidate.get("name"),
            "model": candidate.get("model"),
            "source": candidate.get("source"),
            "attempt": attempt,
            "status": status,
            "message": message,
        }
        if runtime_metrics:
            item.update(
                {
                    key: runtime_metrics[key]
                    for key in (
                        "queue_wait_seconds",
                        "queue_timeout_seconds",
                        "inference_duration_seconds",
                        "consultation_concurrency",
                    )
                    if runtime_metrics.get(key) is not None
                }
            )
        if raw_content:
            item["raw_content_preview"] = self._shorten(raw_content, 220)
        metadata = getattr(response, "response_metadata", None)
        if isinstance(metadata, dict):
            finish_reason = metadata.get("finish_reason")
            model_name = metadata.get("model_name")
            if finish_reason:
                item["finish_reason"] = finish_reason
            if model_name:
                item["provider_model_name"] = model_name
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            item["usage"] = {
                key: usage.get(key)
                for key in ("input_tokens", "output_tokens", "total_tokens")
                if usage.get(key) is not None
            }
        return item

    async def _invoke_consultation_model(
        self,
        messages: list[Any],
        candidate: dict[str, Any],
        request_timeout: float | None = None,
        *,
        queue_timeout_seconds: float | None = None,
        runtime_metrics: dict[str, Any] | None = None,
        capacity_context: dict[str, Any] | None = None,
    ) -> tuple[Any, str]:
        model = candidate.get("model")
        reasoning_model = _is_reasoning_model(model)
        default_timeout = 20.0 if reasoning_model else 12.0
        llm_timeout = min(max(float(request_timeout or default_timeout), 1.0), default_timeout)
        api_key = str(candidate.get("api_key") or "").strip()
        api_base = str(candidate.get("api_base") or "").strip()
        if not api_key:
            if not _is_loopback_api_base(api_base):
                raise RuntimeError("deep consultation requires an API key for non-loopback access")
            api_key = "local-loopback"
        llm_kwargs: dict[str, Any] = {
            "base_url": api_base,
            "api_key": api_key,
            "model": model,
            "timeout": llm_timeout,
            "max_retries": 0,
            "model_kwargs": {"response_format": {"type": "json_object"}},
            "max_completion_tokens": completion_token_limit(
                "consultation",
                1400 if reasoning_model else 700,
                floor=160,
                model=model,
            ),
        }
        if reasoning_model:
            llm_kwargs["temperature"] = None
            llm_kwargs["reasoning_effort"] = "low"
        else:
            llm_kwargs["temperature"] = 0.1
        invoke_messages = messages
        if _uses_thinking_tags(model):
            llm_kwargs["extra_body"] = non_thinking_extra_body()
            invoke_messages = self._consultation_messages_for_model(messages, model)
        llm = ChatOpenAI(**llm_kwargs)
        queue_wait_started = time.perf_counter()
        queue_timeout = min(
            max(
                float(queue_timeout_seconds or _CONSULTATION_QUEUE_WAIT_CAP_SECONDS),
                0.05,
            ),
            _CONSULTATION_QUEUE_WAIT_CAP_SECONDS,
        )
        shared_capacity_context = dict(capacity_context or {})
        original_scope = str(
            shared_capacity_context.get("_analysis_budget_scope") or "shared"
        )
        shared_capacity_context["_analysis_budget_scope"] = (
            f"consultation:{original_scope}"
        )
        shared_slot = shared_llm_capacity_slot(shared_capacity_context)
        shared_acquired = False
        try:
            await asyncio.wait_for(
                shared_slot.__aenter__(),
                timeout=queue_timeout,
            )
            shared_acquired = True
        except TimeoutError as exc:
            queue_wait_seconds = round(time.perf_counter() - queue_wait_started, 3)
            if runtime_metrics is not None:
                runtime_metrics.update(
                    {
                        "queue_wait_seconds": queue_wait_seconds,
                        "queue_timeout_seconds": round(queue_timeout, 3),
                        "inference_duration_seconds": 0.0,
                        "consultation_concurrency": _CONSULTATION_CONCURRENCY,
                    }
                )
            raise ConsultationQueueTimeoutError(queue_wait_seconds) from exc
        except BaseException:
            if shared_acquired:
                await shared_slot.__aexit__(None, None, None)
            raise

        queue_wait_seconds = round(time.perf_counter() - queue_wait_started, 3)
        inference_started = time.perf_counter()
        try:
            response = await asyncio.wait_for(llm.ainvoke(invoke_messages), timeout=llm_timeout)
        finally:
            inference_duration_seconds = round(time.perf_counter() - inference_started, 3)
            if shared_acquired:
                await shared_slot.__aexit__(None, None, None)
            if runtime_metrics is not None:
                runtime_metrics.update(
                    {
                        "queue_wait_seconds": queue_wait_seconds,
                        "queue_timeout_seconds": round(queue_timeout, 3),
                        "inference_duration_seconds": inference_duration_seconds,
                        "consultation_concurrency": _CONSULTATION_CONCURRENCY,
                    }
                )
        return response, _message_content_text(response).strip()

    @staticmethod
    def _consultation_messages_for_model(messages: list[Any], model: str | None) -> list[Any]:
        """Return consultation messages with /no_think for Qwen3/R1-style models."""
        if not _uses_thinking_tags(model):
            return messages
        copied = list(messages)
        for index in range(len(copied) - 1, -1, -1):
            message = copied[index]
            if isinstance(message, HumanMessage):
                copied[index] = HumanMessage(content=ensure_no_think_text(message.content))
                break
        return copied

    async def consult_if_needed(
        self,
        opinions: dict[str, DecisionOutput],
        validations: list[dict[str, Any]],
        *,
        timeout_seconds: float | None = None,
        capacity_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        major = [v for v in validations if v.get("major_conflict")]
        if not major:
            return None

        deadline = time.perf_counter() + max(
            float(timeout_seconds or _consultation_budget_seconds()), 1.0
        )

        trend_cfg = self._fixed_model_cfg("trend_expert")
        candidates = self._consultation_candidates(trend_cfg)
        primary_model = str(trend_cfg.get("model") or "").strip()
        if not candidates:
            return {
                "model": primary_model,
                "status": "skipped",
                "consultation_expert": "trend_expert",
                "consultation_expert_label": "行情方向专家",
                "reason": "行情方向专家未配置可用 API Key，本轮发现重大矛盾但跳过深度会诊。",
                "major_conflicts": major,
                "resolution_status": "unresolved",
                "resolved_action": "unclear",
                "resolved_conflict_pairs": [],
            }

        payload = {
            "opinions": {
                name: {
                    "action": decision.action.value,
                    "confidence": decision.confidence,
                    "reasoning": self._shorten(decision.reasoning, 160),
                    "cross_check_for": decision.cross_check_for,
                }
                for name, decision in opinions.items()
            },
            "major_conflicts": [self._compact_conflict(item) for item in major],
        }
        messages = [
            SystemMessage(
                content=(
                    "你是行情方向专家，也是本轮加密合约交易会诊主持人。"
                    "只处理 listed major_conflicts 中的专家矛盾，结论必须简洁中文。"
                    "只返回一个严格 JSON 对象，不得使用 Markdown 或增加解释。"
                    "字段为 conflict_note, observation_summary, "
                    "resolution_status, resolved_action, resolved_conflict_pairs。"
                    "conflict_note 和 observation_summary 必须是短字符串，不得嵌套对象。"
                    "resolution_status 只能是 resolved 或 unresolved；resolved_action 只能是 "
                    "long、short、hold 或 unclear；resolved_conflict_pairs 只能是已解决专家名"
                    "二元数组的列表，不得包含对象。"
                    "只有证据足以确定统一方向并覆盖全部重大冲突时才能返回 resolved。"
                    "结论仅用于解决分析质量冲突，不得给出交易许可、仓位或杠杆调整。"
                )
            ),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]

        attempts: list[dict[str, Any]] = []
        last_model = primary_model
        for candidate_index, candidate in enumerate(candidates):
            last_model = str(candidate.get("model") or last_model or "")
            max_attempts = min(max(int(candidate.get("retries") or 1), 1), 1)
            for attempt_no in range(1, max_attempts + 1):
                remaining = deadline - time.perf_counter()
                if remaining <= 0.5:
                    break
                runtime_metrics: dict[str, Any] = {}
                try:
                    model_attempt_cap = (
                        _CONSULTATION_THINKING_ATTEMPT_CAP_SECONDS
                        if _uses_thinking_tags(candidate.get("model"))
                        else _CONSULTATION_ATTEMPT_CAP_SECONDS
                    )
                    has_fallback = candidate_index < len(candidates) - 1
                    reserve = (
                        _CONSULTATION_FALLBACK_RESERVE_SECONDS
                        if has_fallback
                        and remaining > _CONSULTATION_FALLBACK_RESERVE_SECONDS + 1.0
                        else 0.0
                    )
                    queue_timeout = min(
                        _CONSULTATION_QUEUE_WAIT_CAP_SECONDS,
                        max(
                            min(remaining * 0.15, remaining - reserve - 0.5),
                            0.05,
                        ),
                    )
                    call_timeout = min(
                        model_attempt_cap,
                        max(remaining - reserve - queue_timeout, 0.0),
                    )
                    if call_timeout <= 0.5:
                        break
                    response, content = await asyncio.wait_for(
                        self._invoke_consultation_model(
                            messages,
                            candidate,
                            request_timeout=call_timeout,
                            queue_timeout_seconds=queue_timeout,
                            runtime_metrics=runtime_metrics,
                            capacity_context=capacity_context,
                        ),
                        timeout=min(remaining, call_timeout + queue_timeout + 0.1),
                    )
                    if not content:
                        attempts.append(
                            self._consultation_attempt(
                                candidate,
                                attempt_no,
                                "empty_response",
                                "模型返回空内容，未拿到可解析的会诊结论。",
                                response=response,
                                runtime_metrics=runtime_metrics,
                            )
                        )
                        continue
                    try:
                        parsed = self._extract_json(content)
                    except Exception:
                        attempts.append(
                            self._consultation_attempt(
                                candidate,
                                attempt_no,
                                "invalid_json",
                                "模型返回内容不是有效 JSON。",
                                raw_content=content,
                                response=response,
                                runtime_metrics=runtime_metrics,
                            )
                        )
                        continue

                    attempts.append(
                        self._consultation_attempt(
                            candidate,
                            attempt_no,
                            "completed",
                            "会诊完成。",
                            response=response,
                            runtime_metrics=runtime_metrics,
                        )
                    )
                    parsed["model"] = candidate.get("model")
                    parsed["consultation_expert"] = candidate.get("name") or "trend_expert"
                    parsed["consultation_expert_label"] = candidate.get("label") or candidate.get(
                        "name"
                    )
                    parsed["primary_consultation_expert"] = "trend_expert"
                    parsed["status"] = "completed"
                    parsed["major_conflicts"] = major
                    parsed["consultation_attempts"] = attempts
                    parsed["fallback_used"] = candidate.get("source") != "primary" or attempt_no > 1
                    allowed_actions = {"long", "short", "hold"}
                    resolution_status = str(parsed.get("resolution_status") or "").lower()
                    resolved_action = str(parsed.get("resolved_action") or "unclear").lower()
                    major_pairs = {
                        tuple(sorted(str(name) for name in item.get("expert_pair") or []))
                        for item in major
                        if isinstance(item, dict) and len(item.get("expert_pair") or []) == 2
                    }
                    resolved_pairs = {
                        tuple(sorted(str(name) for name in pair))
                        for pair in parsed.get("resolved_conflict_pairs") or []
                        if isinstance(pair, (list, tuple)) and len(pair) == 2
                    }
                    resolution_is_complete = bool(
                        resolution_status == "resolved"
                        and resolved_action in allowed_actions
                        and major_pairs
                        and major_pairs.issubset(resolved_pairs)
                    )
                    parsed["resolution_status"] = (
                        "resolved" if resolution_is_complete else "unresolved"
                    )
                    parsed["resolved_action"] = (
                        resolved_action if resolution_is_complete else "unclear"
                    )
                    parsed["resolved_conflict_pairs"] = [
                        list(pair) for pair in sorted(major_pairs & resolved_pairs)
                    ]
                    parsed = {
                        key: value
                        for key, value in parsed.items()
                        if key
                        in {
                            "conflict_note",
                            "observation_summary",
                            "model",
                            "consultation_expert",
                            "consultation_expert_label",
                            "primary_consultation_expert",
                            "status",
                            "major_conflicts",
                            "consultation_attempts",
                            "fallback_used",
                            "resolution_status",
                            "resolved_action",
                            "resolved_conflict_pairs",
                        }
                    }
                    parsed["production_permission"] = False
                    return parsed
                except ConsultationQueueTimeoutError as exc:
                    error_text = safe_error_text(exc)
                    logger.warning(
                        "deep consultation queue timed out",
                        expert=candidate.get("name"),
                        model=candidate.get("model"),
                        attempt=attempt_no,
                        **runtime_metrics,
                    )
                    attempts.append(
                        self._consultation_attempt(
                            candidate,
                            attempt_no,
                            "queue_timeout",
                            error_text,
                            runtime_metrics=runtime_metrics,
                        )
                    )
                except Exception as exc:
                    error_text = safe_error_text(exc)
                    logger.warning(
                        "deep consultation attempt failed",
                        expert=candidate.get("name"),
                        model=candidate.get("model"),
                        attempt=attempt_no,
                        error=error_text,
                    )
                    attempts.append(
                        self._consultation_attempt(
                            candidate,
                            attempt_no,
                            "call_failed",
                            error_text,
                            runtime_metrics=runtime_metrics,
                        )
                    )
            if deadline - time.perf_counter() <= 0.5:
                break

        return self._fallback_consultation(
            major,
            last_model or primary_model,
            "深度会诊多次尝试失败",
            attempts=attempts,
        )

    def _compact_conflict(self, conflict: dict[str, Any]) -> dict[str, Any]:
        """Keep consultation input within the local model context window."""

        return {
            "expert_pair": list(conflict.get("expert_pair") or [])[:2],
            "question": self._shorten(str(conflict.get("question") or ""), 120),
            "consistency": str(conflict.get("consistency") or "divergent"),
            "conflict_note": self._shorten(str(conflict.get("conflict_note") or ""), 180),
            "checked_evidence": [
                self._shorten(str(item), 80)
                for item in list(conflict.get("checked_evidence") or [])[:8]
            ],
        }

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("empty consultation response")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def _fallback_consultation(
        self,
        major: list[dict[str, Any]],
        model: str,
        reason: str,
        raw_content: str | None = None,
        attempts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return an observation-only failure record."""
        note = f"{reason}，重大分歧未能完成观察性复核。"
        result = {
            "model": model,
            "consultation_expert": "trend_expert",
            "consultation_expert_label": "行情方向专家",
            "status": "failed",
            "fallback": True,
            "conflict_note": note,
            "production_permission": False,
            "major_conflicts": major,
            "consultation_attempts": attempts or [],
            "resolution_status": "unresolved",
            "resolved_action": "unclear",
            "resolved_conflict_pairs": [],
        }
        if raw_content:
            result["raw_content_preview"] = self._shorten(raw_content, 300)
        return result

    def _shorten(self, text: str, limit: int = 220) -> str:
        clean = " ".join(str(text or "").split())
        return clean[:limit]

    def _expert_label(self, name: str) -> str:
        labels = {
            "trend_expert": "行情方向专家",
            "momentum_expert": "盈利质量专家",
            "sentiment_expert": "短线时序专家",
            "position_expert": "持仓退出专家",
            "risk_expert": "异常风控专家",
        }
        return labels.get(name, name)

    def _action_label(self, action: Action) -> str:
        labels = {
            Action.LONG: "做多",
            Action.SHORT: "做空",
            Action.CLOSE_LONG: "平多",
            Action.CLOSE_SHORT: "平空",
            Action.HOLD: "观望",
        }
        return labels.get(action, str(action.value))
