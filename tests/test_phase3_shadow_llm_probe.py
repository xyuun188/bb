from __future__ import annotations

import pytest

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


@pytest.mark.parametrize(
    "url",
    (
        "file:///etc/passwd",
        "ftp://127.0.0.1/v1",
        "http://example.com/v1",
        "http://user:secret@127.0.0.1/v1",
        "http://127.0.0.1/v1?target=external",
    ),
)
def test_phase3_shadow_probe_rejects_non_loopback_or_unsafe_urls(url: str) -> None:
    with pytest.raises(ValueError):
        probe._probe_url(url)


def test_phase3_shadow_probe_builds_validated_loopback_endpoint() -> None:
    assert (
        probe._probe_url("http://127.0.0.1:18000/v1/")
        == "http://127.0.0.1:18000/v1/chat/completions"
    )
