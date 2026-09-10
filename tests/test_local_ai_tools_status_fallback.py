from __future__ import annotations

from time import monotonic

from services.local_ai_tools_client import LocalAIToolsClient


def test_status_stale_snapshot_preserves_verified_service_state() -> None:
    client = LocalAIToolsClient()
    client._last_successful_status = (
        monotonic(),
        {
            "available": True,
            "service_available": True,
            "model_bundle_available": True,
            "status": "canary",
            "promotion_recommendation": {"live_ml_ready": False},
        },
    )

    stale = client.stale_status_snapshot(reason="probe_timeout")

    assert stale is not None
    assert stale["available"] is True
    assert stale["service_available"] is True
    assert stale["status"] == "status_stale"
    assert stale["stale"] is True
    assert stale["refresh_in_background"] is True
    assert stale["status_error"] == "probe_timeout"
