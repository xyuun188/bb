from __future__ import annotations

from pathlib import Path

from web_dashboard.app import create_app

ROOT = Path(__file__).resolve().parent.parent
REMOVED_FILES = (
    "services/competition_service.py",
    "services/notification_service.py",
    "web_dashboard/api/models.py",
    "workers/main_loop.py",
    "workers/data_collector.py",
    "workers/model_evaluator.py",
)


def test_removed_legacy_model_competition_files_do_not_return() -> None:
    assert all(not (ROOT / relative_path).exists() for relative_path in REMOVED_FILES)


def test_runtime_sources_do_not_reference_removed_competition_chain() -> None:
    runtime_sources = (
        ROOT / "scripts" / "run_paper_trading.py",
        ROOT / "scripts" / "run_live_trading.py",
        ROOT / "web_dashboard" / "api" / "dashboard.py",
        ROOT / "web_dashboard" / "api" / "router.py",
        ROOT / "web_dashboard" / "api" / "settings_api.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in runtime_sources)

    assert "CompetitionService" not in source
    assert "competition_service" not in source
    assert "NotificationService" not in source
    assert "models_router" not in source


def test_removed_model_rankings_api_is_not_registered() -> None:
    paths = {
        str(path)
        for route in create_app().routes
        if (path := getattr(route, "path", None))
    }

    assert "/api/models" not in paths
    assert not any(path.startswith("/api/models/") for path in paths)
