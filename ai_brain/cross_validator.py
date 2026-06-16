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

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ai_brain.base_model import Action, DecisionOutput
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

_CONSULTATION_SEMAPHORE = asyncio.Semaphore(1)
BACKUP_CONSULTATION_MODELS = ("qwen3-max", "deepseek-v3", "claude-opus-4-7")

_CONSULTATION_TIMEOUT_FLOOR_SECONDS = 6.0
_CONSULTATION_TIMEOUT_CAP_SECONDS = 12.0
_CONSULTATION_ATTEMPT_CAP_SECONDS = 8.0
_CONSULTATION_REASONING_ATTEMPT_CAP_SECONDS = 10.0


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
    return name.startswith("qwen3-") and name.endswith("-trade")


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
            if target_name not in opinions:
                validations.append(
                    {
                        "expert_pair": [source_name, target_name],
                        "question": question,
                        "consistency": "neutral",
                        "confidence_adjustment": -10,
                        "conflict_note": f"{self._expert_label(target_name)} 本轮没有返回，无法完成这次交叉验证。",
                        "validation_note": f"{self._expert_label(target_name)} 本轮没有返回，无法回答这个核实问题。",
                        "checked_evidence": [],
                        "major_conflict": False,
                        "validation_status": "target_missing",
                    }
                )
                continue
            validations.append(
                self.validate_pair(
                    source_name,
                    source,
                    target_name,
                    opinions[target_name],
                    question,
                )
            )

        cross_duration = round(time.perf_counter() - cross_perf_started, 3)
        if timing_context is not None:
            timing_context["_cross_validation_timing"] = {
                "stage": "cross_validation",
                "label": "交叉验证",
                "status": "completed",
                "started_at": cross_started_at.isoformat(),
                "duration_sec": cross_duration,
                "requested": sum(1 for v in validations if v.get("question")),
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
            adjustment = 10
            note = None
        elif source_dir and target_dir and source_dir != target_dir:
            consistency = "divergent"
            high_conflict = source.confidence >= 0.60 and target.confidence >= 0.60
            adjustment = -40 if high_conflict else -25
            note = validation_note
        elif question_result == "supports" and source_dir:
            consistency = "aligned"
            adjustment = 6
            note = None
        elif question_result == "challenges" and source_dir:
            consistency = "divergent"
            adjustment = -25 if target.confidence >= 0.55 else -15
            note = validation_note
        else:
            consistency = "neutral"
            # Neutral means the target expert did not provide enough evidence
            # to confirm or refute the request. It should be displayed and
            # handled as information only, not as a hidden penalty.
            adjustment = 0
            note = None

        return {
            "expert_pair": [source_name, target_name],
            "question": question,
            "consistency": consistency,
            "confidence_adjustment": max(min(adjustment, 50), -50),
            "conflict_note": note,
            "validation_note": validation_note,
            "checked_evidence": checked_evidence,
            "needs_resolution": consistency == "divergent",
            "major_conflict": self._is_major_conflict(source, target, consistency, adjustment),
            "validation_status": "completed",
        }

    def _is_major_conflict(
        self,
        source: DecisionOutput,
        target: DecisionOutput,
        consistency: str,
        adjustment: int,
    ) -> bool:
        if consistency != "divergent":
            return False
        if adjustment <= -40:
            return True
        source_dir = ACTION_DIRECTION.get(source.action, 0)
        target_dir = ACTION_DIRECTION.get(target.action, 0)
        return (
            bool(source_dir and target_dir and source_dir != target_dir)
            and source.confidence >= 0.62
            and target.confidence >= 0.62
        )

    def _question_result(
        self,
        source: DecisionOutput,
        target: DecisionOutput,
        target_name: str,
        question: str,
    ) -> str:
        """Return whether target evidence supports or challenges the source concern."""
        q = str(question or "")
        snapshot = target.feature_snapshot or source.feature_snapshot or {}
        source_dir = ACTION_DIRECTION.get(source.action, 0)
        target_dir = ACTION_DIRECTION.get(target.action, 0)

        if source_dir and target_dir:
            return "supports" if source_dir == target_dir else "challenges"

        if target_name == "sentiment_expert":
            news = float(snapshot.get("news_sentiment_avg") or 0.0)
            social = float(snapshot.get("social_sentiment_avg") or 0.0)
            mentions = int(float(snapshot.get("social_mention_count") or 0.0))
            if abs(news) < 0.05 and abs(social) < 0.05 and mentions <= 0:
                return "neutral"
            if source_dir > 0 and news + social > 0.25:
                return "supports"
            if source_dir < 0 and news + social < -0.25:
                return "supports"
            return "challenges"

        if any(word in q for word in ("成交量", "量能", "放量", "流动性", "假突破")):
            volume_ratio = float(snapshot.get("volume_ratio") or 0.0)
            if volume_ratio >= 1.2:
                return "supports"
            if volume_ratio < settings.min_entry_volume_ratio:
                return "challenges"
            return "neutral"

        if any(word in q for word in ("趋势", "破位", "突破", "支撑", "压力", "均线", "MACD")):
            adx = float(snapshot.get("adx_14") or 0.0)
            macd = float(snapshot.get("macd_diff") or 0.0)
            sma20 = float(snapshot.get("price_vs_sma20") or 0.0)
            sma50 = float(snapshot.get("price_vs_sma50") or 0.0)
            if source_dir > 0:
                if adx >= settings.min_entry_adx and macd >= 0 and sma20 > 0 and sma50 > 0:
                    return "supports"
                if sma20 <= 0 or sma50 <= 0:
                    return "challenges"
            if source_dir < 0:
                if adx >= settings.min_entry_adx and macd <= 0 and sma20 < 0 and sma50 < 0:
                    return "supports"
                if sma20 >= 0 or sma50 >= 0:
                    return "challenges"
            return "neutral"

        if any(
            word in q for word in ("风险", "止损", "波动", "回撤", "滑点", "过热", "插针", "尾部")
        ):
            volume_ratio = float(snapshot.get("volume_ratio") or 0.0)
            volatility = float(snapshot.get("volatility_20") or 0.0)
            day_change = abs(float(snapshot.get("change_24h_pct") or 0.0))
            abnormal_wick_count = int(float(snapshot.get("abnormal_wick_count_72h") or 0.0))
            abnormal_wick_max = float(snapshot.get("abnormal_wick_max_pct") or 0.0)
            abnormal_wick_recent = float(snapshot.get("abnormal_wick_recent_hours") or 9999.0)
            if (
                volume_ratio < settings.min_entry_volume_ratio
                or volatility > 0.08
                or day_change > 18
                or (
                    abnormal_wick_count > 0
                    and abnormal_wick_max >= 80
                    and abnormal_wick_recent <= 96
                )
            ):
                return "challenges"
            return "neutral"

        if target.action == Action.HOLD:
            return "neutral"
        return "supports" if target_dir == source_dir else "neutral"

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
            api_base = (api_base or settings.ai_api_base or "").strip()
            api_key = (api_key or "").strip()
            model = (model or settings.ai_model or "").strip()
            if not api_key or not model:
                return
            identity = (api_base, model, secret_fingerprint(api_key))
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
                    "_identity": identity,
                }
            )

        add_candidate(
            name="trend_expert",
            label="行情方向专家",
            api_base=trend_cfg.get("api_base") or settings.ai_api_base,
            api_key=trend_cfg.get("api_key") or settings.ai_api_key,
            model=trend_cfg.get("model") or settings.ai_model,
            retries=1,
            source="primary",
        )

        if settings.high_risk_review_enabled:
            add_candidate(
                name="high_risk_review",
                label="High-risk review model",
                api_base=settings.high_risk_review_api_base,
                api_key=settings.high_risk_review_api_key or settings.ai_api_key,
                model=settings.high_risk_review_model,
                retries=1,
                source="high_risk_review",
            )

        decision_cfg = self._fixed_model_cfg(DECISION_MAKER_NAME)
        add_candidate(
            name=DECISION_MAKER_NAME,
            label=decision_cfg.get("label") or "最终交易员",
            api_base=decision_cfg.get("api_base") or settings.ai_api_base,
            api_key=decision_cfg.get("api_key") or settings.ai_api_key,
            model=decision_cfg.get("model") or settings.ai_model,
            retries=1,
            source="decision_maker",
        )

        primary_api_base = trend_cfg.get("api_base") or settings.ai_api_base
        primary_api_key = trend_cfg.get("api_key") or settings.ai_api_key
        primary_model = str(trend_cfg.get("model") or settings.ai_model or "").strip()
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
    ) -> tuple[Any, str]:
        model = candidate.get("model")
        reasoning_model = _is_reasoning_model(model)
        default_timeout = 20.0 if reasoning_model else 12.0
        llm_timeout = min(max(float(request_timeout or default_timeout), 1.0), default_timeout)
        llm_kwargs: dict[str, Any] = {
            "base_url": candidate.get("api_base"),
            "api_key": candidate.get("api_key"),
            "model": model,
            "timeout": llm_timeout,
            "max_retries": 0,
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
        async with _CONSULTATION_SEMAPHORE:
            response = await llm.ainvoke(invoke_messages)
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
    ) -> dict[str, Any] | None:
        major = [v for v in validations if v.get("major_conflict")]
        if not major:
            return None

        deadline = time.perf_counter() + max(
            float(timeout_seconds or _consultation_budget_seconds()), 1.0
        )

        trend_cfg = self._fixed_model_cfg("trend_expert")
        candidates = self._consultation_candidates(trend_cfg)
        primary_model = str(trend_cfg.get("model") or settings.ai_model or "").strip()
        if not candidates:
            return {
                "model": primary_model,
                "status": "skipped",
                "consultation_expert": "trend_expert",
                "consultation_expert_label": "行情方向专家",
                "reason": "行情方向专家未配置可用 API Key，本轮发现重大矛盾但跳过深度会诊。",
                "major_conflicts": major,
            }

        payload = {
            "opinions": {
                name: {
                    "action": decision.action.value,
                    "confidence": decision.confidence,
                    "reasoning": self._shorten(decision.reasoning),
                    "cross_check_for": decision.cross_check_for,
                }
                for name, decision in opinions.items()
            },
            "major_conflicts": major,
        }
        messages = [
            SystemMessage(
                content=(
                    "你是行情方向专家，也是本轮加密合约交易会诊主持人。"
                    "只处理 listed major_conflicts 中的专家矛盾，结论必须简洁中文。"
                    "只返回严格 JSON，字段为："
                    "recommended_action, confidence_adjustment (-50..50), conflict_note, should_trade。"
                    "conflict_note 必须用中文说明是否继续交易以及原因。"
                )
            ),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]

        attempts: list[dict[str, Any]] = []
        last_model = primary_model
        for candidate in candidates:
            last_model = str(candidate.get("model") or last_model or "")
            max_attempts = min(max(int(candidate.get("retries") or 1), 1), 1)
            for attempt_no in range(1, max_attempts + 1):
                remaining = deadline - time.perf_counter()
                if remaining <= 0.5:
                    break
                try:
                    model_attempt_cap = (
                        _CONSULTATION_REASONING_ATTEMPT_CAP_SECONDS
                        if _is_reasoning_model(candidate.get("model"))
                        else _CONSULTATION_ATTEMPT_CAP_SECONDS
                    )
                    call_timeout = min(model_attempt_cap, remaining)
                    response, content = await asyncio.wait_for(
                        self._invoke_consultation_model(
                            messages,
                            candidate,
                            request_timeout=call_timeout,
                        ),
                        timeout=call_timeout,
                    )
                    if not content:
                        attempts.append(
                            self._consultation_attempt(
                                candidate,
                                attempt_no,
                                "empty_response",
                                "模型返回空内容，未拿到可解析的会诊结论。",
                                response=response,
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
                    parsed["should_trade"] = self._as_bool(parsed.get("should_trade"))
                    if parsed["should_trade"] is None:
                        action = str(parsed.get("recommended_action") or "").strip().lower()
                        parsed["should_trade"] = action not in {
                            "hold",
                            "no_trade",
                            "skip",
                            "观望",
                            "不交易",
                        }
                    try:
                        adjustment = float(parsed.get("confidence_adjustment", 0) or 0)
                        parsed["confidence_adjustment"] = max(min(adjustment, 50), -50)
                    except (TypeError, ValueError):
                        parsed["confidence_adjustment"] = 0
                    return parsed
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
        """Use a conservative structured result when the LLM cannot return JSON."""
        note = f"{reason}，系统已按保守规则处理：重大矛盾未解除，本轮不新增开仓。"
        result = {
            "model": model,
            "consultation_expert": "trend_expert",
            "consultation_expert_label": "行情方向专家",
            "status": "failed",
            "fallback": True,
            "recommended_action": "hold",
            "confidence_adjustment": -25,
            "conflict_note": note,
            "should_trade": False,
            "major_conflicts": major,
            "consultation_attempts": attempts or [],
        }
        if raw_content:
            result["raw_content_preview"] = self._shorten(raw_content, 300)
        return result

    def _shorten(self, text: str, limit: int = 220) -> str:
        clean = " ".join(str(text or "").split())
        return clean[:limit]

    def _as_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1", "trade"}:
                return True
            if normalized in {"false", "no", "0", "hold", "skip"}:
                return False
        return None

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
