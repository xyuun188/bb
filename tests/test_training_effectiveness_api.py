import json
from datetime import UTC, datetime, timedelta

import pytest

from services.training_effectiveness_report import (
    TRAINING_EFFECTIVENESS_REPORT_VERSION,
    apply_report_filters,
    load_cached_training_effectiveness_report,
)


def _report(generated_at: str) -> dict:
    return {
        "report_version": TRAINING_EFFECTIVENESS_REPORT_VERSION,
        "report_id": "te-fixed",
        "generated_at": generated_at,
        "data_cutoff_at": generated_at,
        "status": "complete",
        "input_fingerprint": "sha256:" + "1" * 64,
        "filters": {"mode": "all"},
        "freshness": {},
    }


def test_cached_report_missing_is_structured_and_read_only(tmp_path):
    result = load_cached_training_effectiveness_report(data_dir=tmp_path)
    assert result["status"] == "missing"
    assert result["freshness"]["is_stale"] is True


def test_cached_report_expiry_is_exposed_without_changing_fingerprint(tmp_path):
    old = (datetime.now(UTC) - timedelta(days=3)).isoformat()
    root = tmp_path / "training_effectiveness_reports"
    root.mkdir()
    (root / "latest.json").write_text(json.dumps(_report(old)), encoding="utf-8")
    result = load_cached_training_effectiveness_report(data_dir=tmp_path)
    assert result["freshness"]["is_stale"] is True
    assert result["input_fingerprint"] == "sha256:" + "1" * 64


def test_filters_only_change_display_filters(tmp_path):
    source = _report(datetime.now(UTC).isoformat())
    result = apply_report_filters(source, mode="paper", side="short", symbol="BTC/USDT")
    assert result["input_fingerprint"] == source["input_fingerprint"]
    assert result["filters"]["mode"] == "paper"
    assert result["filters"]["side"] == "short"


@pytest.mark.asyncio
async def test_dashboard_route_reads_cache_without_building(monkeypatch, tmp_path):
    from web_dashboard.api import dashboard

    monkeypatch.setattr(dashboard, "load_cached_training_effectiveness_report", lambda **_: {"status": "missing"})
    monkeypatch.setattr(dashboard, "apply_report_filters", lambda report, **_: report)
    result = await dashboard.get_training_effectiveness_report(mode="all")
    assert result["status"] == "missing"
