from __future__ import annotations

import inspect
from pathlib import Path

from ai_brain.ensemble_coordinator import EnsembleCoordinator
from web_dashboard.api import system_audit

ROOT = Path(__file__).resolve().parent.parent
REMOVED_KEY = "model_dynamic_routing"
REMOVED_PAYLOAD_KEY = "dynamic_model_routing"


def test_nonfunctional_dynamic_model_routing_has_no_runtime_surface() -> None:
    assert not (ROOT / "services" / "model_dynamic_routing.py").exists()
    assert REMOVED_PAYLOAD_KEY not in inspect.getsource(EnsembleCoordinator)
    assert REMOVED_KEY not in (
        *system_audit.PRIORITY_AUDIT_KEYS,
        *system_audit.DB_AUDIT_KEYS,
        *system_audit.HEAVY_AUDIT_KEYS,
    )
    assert REMOVED_KEY not in system_audit.CARD_OWNER_PATHS
    assert REMOVED_KEY not in system_audit.NODE_OWNER_PATHS
    assert all(
        getattr(route, "path", None) != "/api/system-audit/model-dynamic-routing/status"
        and getattr(route, "path", None) != "/model-dynamic-routing/status"
        for route in system_audit.router.routes
    )
    dashboard = (ROOT / "web_dashboard" / "static" / "js" / "dashboard.js").read_text(
        encoding="utf-8"
    )
    assert REMOVED_KEY not in dashboard
