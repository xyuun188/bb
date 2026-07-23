from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_removed_legacy_trading_controls_do_not_return() -> None:
    sources = {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in (
            ROOT / "core" / "trading_mode.py",
            ROOT / "services" / "trading_service.py",
            ROOT / "web_dashboard" / "api" / "control.py",
            ROOT / "web_dashboard" / "static" / "js" / "dashboard.js",
        )
    }
    joined = "\n".join(sources.values())

    assert "market_direct_entry_processor" not in joined
    assert "is_auto_scan" not in joined
    assert "switch_to_manual" not in joined
    assert '"live_model_name"' not in joined
    assert "/control/scan-mode" not in joined
    assert "/control/select-model" not in joined
    assert not (ROOT / "services" / "market_direct_entry_processor.py").exists()
