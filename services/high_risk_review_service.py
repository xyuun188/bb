"""Runtime boundary for online high-risk trade reviews."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
import structlog

from config.settings import settings
from core.model_runtime import (
    HIGH_RISK_REVIEW_TOKEN_CAP,
    HIGH_RISK_REVIEW_TOKEN_FLOOR,
    apply_non_thinking_request_controls,
    cap_completion_tokens,
    completion_token_limit,
)
from core.safe_output import safe_error_text, safe_response_error_text
from core.url_safety import normalize_http_base_url

logger = structlog.get_logger(__name__)

_AUTH_FAILURE_STATUS_CODES = {401, 403}
_ERROR_EXCERPT_LIMIT = 700


@dataclass(frozen=True)
class HighRiskReviewResult:
    """Parsed result returned by the online high-risk reviewer."""

    approved: bool
    confidence: float
    reason: str
    attempts: list[dict[str, Any]]


class HighRiskReviewService:
    """Call and protect the online high-risk review model.

    TradingService owns the business trigger rules. This service owns model runtime details:
    request shaping, short output caps, response parsing, API key selection, and circuit breaking.
    """

    def __init__(self, config: Any = settings) -> None:
        self._settings = config
        self._failure_count = 0
        self._circuit_open_until: datetime | None = None
        self._last_failure = ""

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def circuit_open_until(self) -> datetime | None:
        return self._circuit_open_until

    def api_key(self, api_base: str) -> str:
        """Return the isolated reviewer key, falling back only when bases match exactly."""
        explicit = str(self._settings.high_risk_review_api_key or "").strip()
        if explicit:
            return explicit
        primary_base = str(self._settings.ai_api_base or "").rstrip("/")
        if primary_base and api_base.rstrip("/") == primary_base:
            return str(self._settings.ai_api_key or "").strip()
        return ""

    def circuit_payload(self) -> dict[str, Any] | None:
        """Return a blocking payload while the reviewer circuit is open."""
        now = datetime.now(UTC)
        if self._circuit_open_until is None:
            return None
        if now >= self._circuit_open_until:
            self._circuit_open_until = None
            return None
        remaining = max((self._circuit_open_until - now).total_seconds(), 0.0)
        return {
            "status": "circuit_open",
            "approved": False,
            "failure_count": self._failure_count,
            "last_failure": self._last_failure[:180],
            "cooldown_remaining_sec": round(remaining, 1),
            "circuit_open_until": self._circuit_open_until.isoformat(),
            "reason": (
                "线上高风险复核连续失败，熔断冷却中；必须复核的高风险开仓暂不提交。"
                f"最近失败：{self._last_failure[:160]}"
            ),
        }

    def record_success(self) -> None:
        self._failure_count = 0
        self._circuit_open_until = None
        self._last_failure = ""

    def record_failure(self, reason: str) -> None:
        self._failure_count += 1
        self._last_failure = safe_error_text(
            reason,
            limit=180,
            fallback="high-risk reviewer failed",
        )
        threshold = max(int(self._settings.high_risk_review_circuit_breaker_failures or 2), 1)
        if self._failure_count < threshold:
            return
        cooldown = max(
            float(self._settings.high_risk_review_circuit_breaker_cooldown_seconds or 120.0),
            5.0,
        )
        self._circuit_open_until = datetime.now(UTC) + timedelta(seconds=cooldown)
        logger.warning(
            "high-risk review circuit breaker opened",
            failure_count=self._failure_count,
            threshold=threshold,
            cooldown_seconds=cooldown,
            reason=self._last_failure,
        )

    async def review_trade(
        self,
        prompt: dict[str, Any],
        *,
        api_base: str,
        api_key: str,
        model: str,
    ) -> HighRiskReviewResult:
        """Run the two-attempt online review flow and return a parsed decision."""
        prompt = self.compact_prompt(prompt)
        api_base = normalize_http_base_url(
            api_base,
            field_name="High-risk review API base",
        )
        request_timeout = max(float(self._settings.high_risk_review_timeout_seconds or 30.0), 5.0)
        primary_max_tokens = cap_completion_tokens(
            self._settings.high_risk_review_max_tokens,
            floor=HIGH_RISK_REVIEW_TOKEN_FLOOR,
            cap=HIGH_RISK_REVIEW_TOKEN_CAP,
        )
        retry_max_tokens = min(primary_max_tokens, 260)

        attempts: list[dict[str, Any]] = []
        content = ""
        metadata: dict[str, Any] = {}
        attempt_specs: list[dict[str, Any]] = [
            {
                "messages": self._primary_messages(prompt),
                "use_json_mode": True,
                "max_tokens": primary_max_tokens,
            },
            {
                "messages": self._retry_messages(prompt),
                "use_json_mode": False,
                "max_tokens": retry_max_tokens,
            },
        ]
        for attempt_no, attempt in enumerate(
            attempt_specs,
            start=1,
        ):
            max_tokens = int(cast(int, attempt["max_tokens"]))
            _payload, content, metadata = await self.call_model(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=cast(list[dict[str, str]], attempt["messages"]),
                use_json_mode=bool(attempt["use_json_mode"]),
                max_tokens=max_tokens,
                request_timeout=request_timeout,
            )
            attempts.append(
                {
                    "attempt": attempt_no,
                    "json_mode": bool(attempt["use_json_mode"]),
                    "max_tokens": max_tokens,
                    "finish_reason": metadata.get("finish_reason"),
                    "content_present": bool(content),
                    "raw_has_think_tag": bool(metadata.get("raw_has_think_tag")),
                    "reasoning_stripped": bool(metadata.get("reasoning_stripped")),
                    "usage": metadata.get("usage"),
                }
            )
            if content:
                break

        if not content:
            finish_reason = metadata.get("finish_reason") or "unknown"
            raise ValueError(f"模型两次都没有返回可解析 JSON，finish_reason={finish_reason}")

        parsed_raw = json.loads(content)
        if not isinstance(parsed_raw, dict):
            raise ValueError("模型返回的 JSON 不是对象")
        parsed = cast(dict[str, Any], parsed_raw)
        self.record_success()
        return HighRiskReviewResult(
            approved=bool(parsed.get("approved")),
            confidence=_safe_float(parsed.get("confidence"), 0.0),
            reason=str(parsed.get("reason") or "")[:500],
            attempts=attempts,
        )

    async def call_model(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        use_json_mode: bool,
        max_tokens: int,
        request_timeout: float,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Call an OpenAI-compatible chat completions endpoint."""
        bounded_max_tokens = completion_token_limit(
            "high_risk_review",
            max_tokens,
            floor=HIGH_RISK_REVIEW_TOKEN_FLOOR,
            model=model,
        )
        request_body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": bounded_max_tokens,
            "stream": False,
        }
        request_body = apply_non_thinking_request_controls(model, request_body)
        if use_json_mode:
            request_body["response_format"] = {"type": "json_object"}
        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.post(
                    f"{api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=request_body,
                )
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"high-risk review request could not reach the service: {safe_error_text(exc)}"
            ) from exc
        payload = self._parse_response(response)
        content, metadata = self.extract_content(payload)
        return payload, content, metadata

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        if not response.is_success:
            detail = self._response_error_excerpt(response)
            if response.status_code in _AUTH_FAILURE_STATUS_CODES:
                message = (
                    "high-risk review request was rejected with HTTP "
                    f"{response.status_code}; check HIGH_RISK_REVIEW_API_KEY"
                )
            else:
                message = f"high-risk review request failed with HTTP {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise RuntimeError(message)
        try:
            parsed = response.json()
        except ValueError as exc:
            raise RuntimeError("high-risk review request returned invalid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise RuntimeError("high-risk review request returned a non-object JSON payload")
        return dict(parsed)

    def _response_error_excerpt(self, response: httpx.Response) -> str:
        return safe_response_error_text(response, limit=_ERROR_EXCERPT_LIMIT)

    def extract_content(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Extract JSON text from OpenAI-compatible responses, including reasoning variants."""
        choices = payload.get("choices") if isinstance(payload, dict) else []
        choice = choices[0] if isinstance(choices, list) and choices else {}
        choice_payload = choice if isinstance(choice, dict) else {}
        message_raw = choice_payload.get("message")
        message = message_raw if isinstance(message_raw, dict) else {}
        metadata = {
            "finish_reason": choice_payload.get("finish_reason"),
            "usage": payload.get("usage") if isinstance(payload, dict) else None,
            "raw_has_think_tag": False,
            "reasoning_stripped": False,
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

        metadata["raw_has_think_tag"] = any(_has_thinking_tag(text) for text in candidates)

        for text in candidates:
            cleaned = self.extract_json_object_text(text)
            if cleaned:
                metadata["reasoning_stripped"] = bool(
                    _has_thinking_tag(text) or cleaned != str(text or "").strip()
                )
                return cleaned, metadata
        return "", metadata

    def extract_json_object_text(self, text: str) -> str:
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

    def _primary_messages(self, prompt: dict[str, Any]) -> list[dict[str, str]]:
        prompt = self.compact_prompt(prompt)
        return [
            {
                "role": "system",
                "content": (
                    "You are a high-risk crypto trade reviewer and a JSON API. "
                    "Return exactly one valid JSON object with keys: "
                    "approved(boolean), confidence(number 0-1), "
                    "reason(string in Simplified Chinese). "
                    "Reject only when expected net profit is poor, "
                    "risk is asymmetric, or evidence conflicts."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]

    def _retry_messages(self, prompt: dict[str, Any]) -> list[dict[str, str]]:
        prompt = self.compact_prompt(prompt)
        return [
            {
                "role": "system",
                "content": (
                    "Return only one minified JSON object. No markdown, no reasoning, no prose. "
                    'Schema: {"approved":true|false,'
                    '"confidence":0.0,"reason":"简体中文，60字以内"}.'
                ),
            },
            {
                "role": "user",
                "content": "复核这笔高风险加密货币开仓。只输出 JSON："
                + json.dumps(prompt, ensure_ascii=False),
            },
        ]

    def compact_prompt(self, prompt: dict[str, Any]) -> dict[str, Any]:
        """Keep high-risk review prompts inside small-model context limits."""
        opportunity = _safe_mapping(prompt.get("opportunity_score"))
        selected_breakdown = _compact_expected_net_breakdown(
            _safe_mapping(opportunity.get("expected_net_breakdown"))
        )
        compact = {
            "symbol": prompt.get("symbol"),
            "side": prompt.get("side"),
            "confidence": _round_float(prompt.get("confidence")),
            "position_size_pct": _round_float(prompt.get("position_size_pct")),
            "leverage": _round_float(prompt.get("leverage")),
            "stop_loss_pct": _round_float(prompt.get("stop_loss_pct")),
            "take_profit_pct": _round_float(prompt.get("take_profit_pct")),
            "trigger_reasons": _safe_list(prompt.get("trigger_reasons"))[:8],
            "today_pnl": _round_float(prompt.get("today_pnl")),
            "open_position_count": int(_safe_float(prompt.get("open_position_count"), 0.0)),
            "opportunity_score": {
                "expected_net_return_pct": _round_float(opportunity.get("expected_net_return_pct")),
                "profit_quality_ratio": _round_float(opportunity.get("profit_quality_ratio")),
                "loss_probability": _round_float(
                    opportunity.get("server_profit_loss_probability")
                    or opportunity.get("loss_probability")
                ),
                "tail_risk_score": _round_float(opportunity.get("tail_risk_score")),
                "reward_risk_ratio": _round_float(opportunity.get("reward_risk_ratio")),
                "server_profit_expected_return_pct": _round_float(
                    opportunity.get("server_profit_expected_return_pct")
                ),
                "ml_expected_return_pct": _round_float(opportunity.get("ml_expected_return_pct")),
                "timeseries_expected_return_pct": _round_float(
                    opportunity.get("timeseries_expected_return_pct")
                ),
                "expected_net_breakdown": selected_breakdown,
            },
        }
        text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= 2_400:
            return compact
        compact["opportunity_score"].pop("expected_net_breakdown", None)
        return compact


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _round_float(value: Any, default: float = 0.0) -> float:
    return round(_safe_float(value, default), 6)


def _compact_expected_net_breakdown(payload: dict[str, Any]) -> dict[str, Any]:
    components = payload.get("components") if isinstance(payload, dict) else []
    compact_components: list[dict[str, Any]] = []
    if isinstance(components, list):
        for component in components[:8]:
            if not isinstance(component, Mapping):
                continue
            compact_components.append(
                {
                    "key": component.get("key"),
                    "available": bool(component.get("available")),
                    "side": component.get("side"),
                    "raw_return_pct": _round_float(component.get("raw_return_pct")),
                    "weight": _round_float(component.get("weight"), 0.0),
                    "contribution_pct": _round_float(component.get("contribution_pct"), 0.0),
                }
            )
    return {
        "formula": payload.get("formula"),
        "net_pct": _round_float(payload.get("net_pct")),
        "model_net_pct": _round_float(payload.get("model_net_pct")),
        "components": compact_components,
    }


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_thinking_tag(value: Any) -> bool:
    text = str(value or "").lower()
    return "<think>" in text or "</think>" in text
