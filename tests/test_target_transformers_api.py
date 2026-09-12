from __future__ import annotations

import pytest
from fastapi import HTTPException

from scripts.target_transformers_api import ChatRequest, Runtime


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
    def generate(self, **_kwargs):
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
