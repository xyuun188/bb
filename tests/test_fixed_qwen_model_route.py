from __future__ import annotations

import pytest

from ai_brain.model_factory import create_models_from_config
from config.settings import FIXED_AI_MODEL_SLOTS, TARGET_LOCAL_AI_API_BASE, settings
from web_dashboard.api.settings_api import (
    AIModelTestRequest,
    get_ai_models,
)
from web_dashboard.api.settings_api import (
    test_ai_model_connection as probe_ai_model_connection,
)


def test_fixed_roles_exist_without_ai_models_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ai_models", [])

    rows = settings.get_fixed_ai_models(include_empty=False)

    assert len(rows) == len(FIXED_AI_MODEL_SLOTS) == 6
    assert {row["api_base"] for row in rows} == {TARGET_LOCAL_AI_API_BASE}
    assert {row["model"] for row in rows} == {"qwen3.8-27b"}
    assert all(row["configured"] is True for row in rows)
    assert [model.name for model in create_models_from_config()] == [
        slot["name"] for slot in FIXED_AI_MODEL_SLOTS
    ]
    assert settings.ai_llm_concurrency == 1
    assert settings.ai_expert_timeout_seconds <= 15
    assert settings.ai_batch_expert_timeout_seconds <= 15
    assert settings.ai_target_qwen_timeout_seconds <= 12


@pytest.mark.asyncio
async def test_dashboard_exposes_fixed_roles_as_testable_local_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_models", [])

    payload = await get_ai_models()

    assert len(payload["models"]) == 6
    assert all(row["configured"] is True for row in payload["models"])
    assert all(row["route_type"] == "local" for row in payload["models"])
    assert all(row["editable"] is True for row in payload["models"])
    assert all(row["testable"] is True for row in payload["models"])


@pytest.mark.asyncio
async def test_named_fixed_role_probe_uses_canonical_keyless_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_models", [])

    class Response:
        status_code = 200
        is_success = True

        def json(self) -> dict:
            return {"data": [{"id": "qwen3.8-27b"}]}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr("web_dashboard.api.settings_api.httpx.AsyncClient", lambda **_kwargs: Client())

    result = await probe_ai_model_connection(AIModelTestRequest(name="trend_expert"))

    assert result["success"] is True
    assert result["model"] == "qwen3.8-27b"
