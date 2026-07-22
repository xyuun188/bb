from typing import Any

from core.model_runtime import (
    HIGH_RISK_REVIEW_TOKEN_CAP,
    HIGH_RISK_REVIEW_TOKEN_FLOOR,
    apply_non_thinking_request_controls,
    cap_completion_tokens,
    completion_token_limit,
    ensure_no_think_text,
    supports_batch_expert_json,
    supports_provider_thinking_disable,
    uses_thinking_tags,
    with_no_think_content,
)


def test_uses_thinking_tags_for_qwen3_and_deepseek_r1() -> None:
    assert uses_thinking_tags("qwen3-32b-trade")
    assert uses_thinking_tags("DeepSeek-R1-Distill-Qwen-32B")
    assert not uses_thinking_tags("Qwen2.5-32B-Instruct")


def test_provider_thinking_disable_targets_non_r1_deepseek_routes() -> None:
    assert supports_provider_thinking_disable("deepseek-v4-pro")
    assert not supports_provider_thinking_disable("deepseek-r1-14b-risk")
    assert not supports_provider_thinking_disable("qwen3-14b-trade")


def test_batch_expert_json_support_excludes_deepseek_r1() -> None:
    assert supports_batch_expert_json("qwen3-14b-trade")
    assert not supports_batch_expert_json("deepseek-r1-14b-risk")
    assert not supports_batch_expert_json("DeepSeek-R1-Distill-Qwen-14B-AWQ")


def test_ensure_no_think_text_is_idempotent() -> None:
    prompt = "只输出 JSON"

    once = ensure_no_think_text(prompt)
    twice = ensure_no_think_text(once)

    assert once.endswith("/no_think")
    assert twice == once


def test_apply_non_thinking_request_controls_copies_messages() -> None:
    body: dict[str, Any] = {
        "model": "qwen3-32b-trade",
        "messages": [{"role": "user", "content": "只输出 OK"}],
        "max_tokens": 900,
    }

    controlled = apply_non_thinking_request_controls("qwen3-32b-trade", body)

    assert controlled is not body
    assert controlled["messages"] is not body["messages"]
    assert controlled["messages"][0]["content"].endswith("/no_think")
    assert body["messages"][0]["content"] == "只输出 OK"
    assert controlled["chat_template_kwargs"]["enable_thinking"] is False


def test_with_no_think_content_preserves_structured_text_parts() -> None:
    content: list[dict[str, Any]] = [
        {"type": "input_image", "image_url": "https://example.invalid/chart.png"},
        {"type": "text", "text": "return json only"},
    ]

    controlled = with_no_think_content(content)

    assert controlled is not content
    assert controlled[0] == content[0]
    assert controlled[1]["text"] == "return json only\n/no_think"
    assert content[1]["text"] == "return json only"


def test_with_no_think_content_appends_text_part_when_no_text_exists() -> None:
    content: list[dict[str, Any]] = [
        {"type": "input_image", "image_url": "https://example.invalid/chart.png"}
    ]

    controlled = with_no_think_content(content)

    assert controlled[:-1] == content
    assert controlled[-1] == {"type": "text", "text": "/no_think"}


def test_cap_completion_tokens() -> None:
    assert cap_completion_tokens(900, cap=700) == 700
    assert cap_completion_tokens(10, floor=64, cap=700) == 64
    assert cap_completion_tokens(None, cap=512) == 512


def test_completion_token_limit_enforces_stage_caps() -> None:
    assert completion_token_limit("expert", 999, floor=180) == 360
    assert completion_token_limit("decision_maker", 999, floor=180) == 320
    assert completion_token_limit("batch_expert", 999, floor=180) == 560
    assert completion_token_limit("paper_batch_expert", 999, floor=180) == 900
    assert completion_token_limit("batch_expert", 999, floor=180, model="qwen3-14b-trade") == 560
    assert (
        completion_token_limit("batch_expert", 999, floor=180, model="deepseek-r1-14b-risk") == 999
    )
    assert (
        completion_token_limit(
            "high_risk_review",
            999,
            floor=HIGH_RISK_REVIEW_TOKEN_FLOOR,
        )
        == HIGH_RISK_REVIEW_TOKEN_CAP
    )
    assert completion_token_limit("unknown", 999) == 700
