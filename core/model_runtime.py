"""Shared runtime policy for local and remote LLM calls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

NO_THINK_DIRECTIVE = "/no_think"
HIGH_RISK_REVIEW_TOKEN_FLOOR = 160
HIGH_RISK_REVIEW_TOKEN_CAP = 600
COMPLETION_TOKEN_CAPS = {
    "expert": 360,
    "decision_maker": 320,
    "batch_expert": 700,
    "consultation": 700,
    "high_risk_review": HIGH_RISK_REVIEW_TOKEN_CAP,
    "proxy": 700,
}


def is_openai_reasoning_model(model: str | None) -> bool:
    """Return True for OpenAI reasoning-family models."""
    name = str(model or "").lower()
    return name.startswith(("o1", "o3", "o4"))


def is_qwen3_model(model: str | None) -> bool:
    """Return True for Qwen3 model identifiers."""
    return "qwen3" in str(model or "").lower()


def uses_thinking_tags(model: str | None) -> bool:
    """Return True for models that may emit explicit thinking tags."""
    name = str(model or "").lower()
    return "qwen3" in name or "deepseek-r1" in name


def non_thinking_extra_body(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build OpenAI-compatible extra_body controls for non-thinking Qwen3 calls."""
    body = dict(existing or {})
    template_kwargs = body.get("chat_template_kwargs")
    if not isinstance(template_kwargs, dict):
        template_kwargs = {}
    template_kwargs["enable_thinking"] = False
    body["chat_template_kwargs"] = template_kwargs
    return body


def ensure_no_think_text(content: Any) -> str:
    """Append the non-thinking directive once to a user prompt."""
    text = str(content or "").rstrip()
    if NO_THINK_DIRECTIVE in text:
        return text
    return f"{text}\n{NO_THINK_DIRECTIVE}" if text else NO_THINK_DIRECTIVE


def with_no_think_content(content: Any) -> Any:
    """Return content with /no_think while preserving structured message parts."""
    if isinstance(content, str) or content is None:
        return ensure_no_think_text(content)
    if not isinstance(content, list):
        return ensure_no_think_text(content)

    copied = deepcopy(content)
    for index in range(len(copied) - 1, -1, -1):
        item = copied[index]
        if isinstance(item, str):
            copied[index] = ensure_no_think_text(item)
            return copied
        if not isinstance(item, dict):
            continue
        text_key = "text" if isinstance(item.get("text"), str) else None
        if text_key is None and isinstance(item.get("content"), str):
            text_key = "content"
        if text_key is None:
            continue
        item[text_key] = ensure_no_think_text(item.get(text_key))
        return copied

    copied.append({"type": "text", "text": NO_THINK_DIRECTIVE})
    return copied


def with_no_think_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of OpenAI chat messages with /no_think on the last user message."""
    copied = deepcopy(messages)
    for item in reversed(copied):
        if isinstance(item, dict) and item.get("role") == "user":
            item["content"] = with_no_think_content(item.get("content"))
            break
    return copied


def apply_non_thinking_request_controls(
    model: str | None,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    """Return a request body with non-thinking controls when the model requires them."""
    if not uses_thinking_tags(model):
        return request_body
    controlled = dict(request_body)
    messages = controlled.get("messages")
    if isinstance(messages, list):
        controlled["messages"] = with_no_think_messages(messages)
    controlled["chat_template_kwargs"] = non_thinking_extra_body(
        controlled.get("chat_template_kwargs")
        if isinstance(controlled.get("chat_template_kwargs"), dict)
        else None
    )["chat_template_kwargs"]
    return controlled


def cap_completion_tokens(
    requested: int | None,
    *,
    floor: int = 64,
    cap: int = 700,
) -> int:
    """Clamp completion tokens for local model calls."""
    try:
        value = int(requested or cap)
    except (TypeError, ValueError):
        value = cap
    return min(max(value, floor), cap)


def completion_token_limit(
    stage: str,
    requested: int | None = None,
    *,
    floor: int = 64,
) -> int:
    """Return the centrally enforced output-token limit for a model call stage."""
    cap = COMPLETION_TOKEN_CAPS.get(str(stage or "").strip(), COMPLETION_TOKEN_CAPS["proxy"])
    return cap_completion_tokens(requested, floor=floor, cap=cap)
