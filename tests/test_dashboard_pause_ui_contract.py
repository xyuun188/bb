from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pause_control_has_visible_dashboard_state() -> None:
    html = (ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    css = (ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")

    assert "dashboard-pause-banner" in html
    assert "已暂停新市场分析" in js
    assert "恢复新开仓分析" in js
    assert "pause-btn active" in js
    assert ".dashboard-pause-banner" in css
