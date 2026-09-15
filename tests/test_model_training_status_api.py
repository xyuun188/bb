from __future__ import annotations

from typing import Any

import pytest

from web_dashboard.api import dashboard, model_training_status


def test_model_training_status_routes_have_one_owner() -> None:
    route_paths = {route.path for route in model_training_status.router.routes}

    assert "/model-training/registry" in route_paths
    assert "/model-training/scheduler" in route_paths
    assert "get_model_training_registry_status" not in dashboard.__dict__
    assert "get_model_training_scheduler_status" not in dashboard.__dict__
    assert "_build_model_training_registry_status" not in dashboard.__dict__


@pytest.mark.asyncio
async def test_model_training_scheduler_status_reads_owned_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Store:
        def read(self) -> dict[str, Any]:
            return {"status": "ok", "schedulers": {"platform": {"state": "idle"}}}

    monkeypatch.setattr(model_training_status, "MODEL_TRAINING_STATE_STORE", Store())

    payload = await model_training_status.get_model_training_scheduler_status()

    assert payload["status"] == "ok"
    assert payload["schedulers"]["platform"]["state"] == "idle"
