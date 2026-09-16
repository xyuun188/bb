from __future__ import annotations

import pytest
from fastapi import HTTPException

from scripts.target_transformers_api import ChatRequest, Runtime, build_app


class _FakeTimer:
    instances = []

    def __init__(self, interval, callback):
        self.interval = interval
        self.callback = callback
        self.daemon = False
        self.started = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True


class _ImmediateLock:
    def __init__(self):
        self.acquire_timeouts = []
        self.released = False

    def acquire(self, *, timeout):
        self.acquire_timeouts.append(timeout)
        return True

    def release(self):
        self.released = True


class _FakeInputIds:
    shape = (1, 4)

    def to(self, _device):
        return self


class _FakeOutput:
    def __getitem__(self, _index):
        return "generated-token-ids"


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.template_calls: list[dict] = []

    def apply_chat_template(self, _messages, **kwargs):
        self.template_calls.append(kwargs)
        return {"input_ids": _FakeInputIds()}

    def decode(self, _token_ids, **_kwargs):
        return '{"status":"ok"}'


class _FakeModel:
    def __init__(self):
        self.kwargs: dict = {}

    def generate(self, **_kwargs):
        self.kwargs = _kwargs
        return _FakeOutput()


@pytest.fixture
def runtime() -> tuple[Runtime, _FakeTokenizer]:
    tokenizer = _FakeTokenizer()
    instance = object.__new__(Runtime)
    instance.tokenizer = tokenizer
    instance.context_length = 4096
    instance.input_device = "cpu"
    instance.model = _FakeModel()
    return instance, tokenizer


@pytest.mark.parametrize(
    "template_kwargs",
    (
        {},
        {"enable_thinking": False},
    ),
)
def test_generate_disables_thinking_by_default_or_explicitly(
    runtime: tuple[Runtime, _FakeTokenizer], template_kwargs: dict
) -> None:
    instance, tokenizer = runtime
    request = ChatRequest(
        model="qwen3.8-27b",
        messages=[{"role": "user", "content": "Return JSON."}],
        chat_template_kwargs=template_kwargs,
    )

    assert instance.generate(request) == '{"status":"ok"}'
    assert tokenizer.template_calls == [
        {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            "enable_thinking": False,
        }
    ]


def test_generate_rejects_non_boolean_thinking_flag(
    runtime: tuple[Runtime, _FakeTokenizer]
) -> None:
    instance, tokenizer = runtime
    request = ChatRequest(
        model="qwen3.8-27b",
        messages=[{"role": "user", "content": "Return JSON."}],
        chat_template_kwargs={"enable_thinking": "false"},
    )

    with pytest.raises(HTTPException, match="must be boolean") as error:
        instance.generate(request)

    assert error.value.status_code == 400
    assert tokenizer.template_calls == []


def test_generate_caps_legacy_large_completion_request(
    runtime: tuple[Runtime, _FakeTokenizer],
) -> None:
    instance, _tokenizer = runtime
    instance.max_new_tokens = 320
    request = ChatRequest(
        model="qwen3.8-27b",
        messages=[{"role": "user", "content": "Return JSON."}],
        max_tokens=256,
    )

    assert instance.generate(request) == '{"status":"ok"}'
    assert instance.model.kwargs["max_new_tokens"] == 96


def test_generation_timeout_schedules_one_systemd_recovery_restart(monkeypatch) -> None:
    instance = object.__new__(Runtime)
    instance._generation_state_lock = __import__("threading").Lock()
    instance._restart_scheduled = False
    _FakeTimer.instances = []
    monkeypatch.setattr("scripts.target_transformers_api.threading.Timer", _FakeTimer)

    instance.schedule_restart_after_generation_timeout()
    instance.schedule_restart_after_generation_timeout()

    assert instance._restart_scheduled is True
    assert len(_FakeTimer.instances) == 1
    assert _FakeTimer.instances[0].interval == 0.75
    assert _FakeTimer.instances[0].daemon is True
    assert _FakeTimer.instances[0].started is True


def test_busy_generation_waits_on_single_worker_lock_instead_of_immediate_503() -> None:
    instance = object.__new__(Runtime)
    instance.model_id = "qwen3.8-27b"
    instance.warmup_complete = True
    instance.max_queue_wait_seconds = 6.0
    instance.generation_timeout_seconds = 18.0
    instance.lock = _ImmediateLock()
    instance.cache_key = lambda _request: "cache-key"
    instance.get_cached_response = lambda _key: None
    instance.generation_busy = lambda: True
    instance._generate_with_timeout = lambda _request, _timeout: '{"a":["h"]}'
    instance.cache_response = lambda _key, _content: None
    instance._last_prompt_tokens = 10
    instance._last_generation_tokens = 6

    route = next(
        route for route in build_app(instance).routes if route.path == "/v1/chat/completions"
    )
    response = route.endpoint(
        ChatRequest(
            model="qwen3.8-27b",
            messages=[{"role": "user", "content": "Return compact JSON."}],
        ),
        None,
    )

    assert response.status_code == 200
    assert instance.lock.acquire_timeouts == [6.0]
    assert instance.lock.released is True


def test_health_ready_rejects_traffic_until_warmup_completes() -> None:
    instance = object.__new__(Runtime)
    instance.model_id = "qwen3.8-27b"
    instance.adapter_path = "/data/adapter"
    instance.fast_path = {"enabled": True}
    instance.attention_implementation = "sdpa"
    instance.warmup_complete = False
    instance.warmup_error = None
    instance.warmup_started_at = 1.0
    instance.warmup_finished_at = None
    instance.max_queue_wait_seconds = 2.0
    instance.generation_timeout_seconds = 12.0
    instance.max_new_tokens = 96
    instance._generation_timeouts = 0
    instance._last_generation_seconds = None
    instance._last_prompt_tokens = 0
    instance._last_generation_tokens = 0
    instance._response_cache_ttl_seconds = 2.0
    instance._response_cache_hits = 0
    instance.generation_busy = lambda: True

    route = next(
        route for route in build_app(instance).routes if route.path == "/health/ready"
    )
    response = route.endpoint()

    assert response.status_code == 503
    assert b'"status":"warming_up"' in response.body
