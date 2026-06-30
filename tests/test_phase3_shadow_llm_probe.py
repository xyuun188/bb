from __future__ import annotations

from scripts import run_phase3_shadow_llm_probe as probe


def test_phase3_shadow_probe_applies_non_thinking_controls() -> None:
    body = probe._request_body(probe.DEFAULT_PROBES[0])

    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert "/no_think" in body["messages"][-1]["content"]


def test_phase3_shadow_probe_rejects_thinking_tag_output() -> None:
    assert probe._json_object_available('<think>hidden</think>{"ok":true,"role":"x"}') is True
    assert probe._json_object_available("<think>hidden</think>") is False


def test_phase3_shadow_probe_gives_reasoning_model_enough_json_headroom() -> None:
    risk_probe = probe.DEFAULT_PROBES[1]
    body = probe._request_body(risk_probe)

    assert risk_probe.allow_reasoning_prefix is True
    assert body["max_tokens"] == probe.HIGH_RISK_REVIEW_TOKEN_CAP
