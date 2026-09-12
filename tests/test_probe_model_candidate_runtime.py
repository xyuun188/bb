from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import probe_model_candidate_runtime
from scripts.probe_model_candidate_runtime import _percentile_95, _valid_contract


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"status":"ok"}', True),
        ('```json\n{"status":"ok"}\n```', True),
        ('{"status":"wrong"}', False),
        ('thinking... {"status":"ok"}', False),
        ("", False),
    ],
)
def test_runtime_probe_requires_exact_json_contract(content: str, expected: bool) -> None:
    assert _valid_contract(content) is expected


def test_runtime_probe_p95_uses_nearest_rank() -> None:
    assert _percentile_95([float(value) for value in range(1, 21)]) == 19.0


def test_runtime_probe_rejects_empty_latency_sample() -> None:
    with pytest.raises(ValueError, match="empty"):
        _percentile_95([])


def test_runtime_probe_always_disables_thinking(monkeypatch) -> None:
    requests: list[dict] = []

    def fake_json_request(_url, *, payload, timeout):
        del timeout
        if payload is None:
            return {"data": [{"id": "qwen3.8-27b"}]}
        requests.append(payload)
        return {"choices": [{"message": {"content": '{"status":"ok"}'}}]}

    monkeypatch.setattr(probe_model_candidate_runtime, "_json_request", fake_json_request)
    monkeypatch.setattr(probe_model_candidate_runtime, "_config_identity", lambda _path: ("Qwen3_5ForConditionalGeneration", "qwen3_5"))
    monkeypatch.setattr(probe_model_candidate_runtime, "_long_prompt", lambda *_args: ("long prompt", 3968))
    monkeypatch.setattr(probe_model_candidate_runtime, "_gpu_memory_used_gib", lambda: 23.5)
    monkeypatch.setattr(probe_model_candidate_runtime.metadata, "version", lambda _name: "5.8.1")
    args = SimpleNamespace(
        model_id="qwen3.8-27b",
        repo_id="Qwen/Qwen3.8-27B",
        revision="1098534ab5d7220ea0f4a6b9f07bb03729a79c1d",
        model_path=Path("model"),
        tokenizer_path=Path("tokenizer"),
        quantization="bitsandbytes-nf4",
        context_length=4096,
        max_concurrency=1,
        runtime_engine="transformers",
        endpoint="http://127.0.0.1:18000",
        request_count=20,
        request_timeout_seconds=1,
        long_prompt_tokens=4096,
    )

    result = probe_model_candidate_runtime.run_probe(args)

    assert result["status"] == "verified"
    assert len(requests) == 20
    assert all(
        request["chat_template_kwargs"] == {"enable_thinking": False}
        for request in requests
    )
