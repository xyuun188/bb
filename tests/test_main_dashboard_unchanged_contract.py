from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_PANEL_SHA256 = "04b4a0827cf769b534e283e3621cec2c2e07b7330714c9886d79770a30faf700"


def test_continuous_training_remediation_does_not_change_main_panel_markup() -> None:
    source = (ROOT / "web_dashboard" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    start = source.index('<div class="page-section active" id="page-dashboard">')
    next_page_marker = source.index("PAGE: EXECUTION HISTORY", start)
    end = source.rfind("<!--", start, next_page_marker)
    main_panel = source[start:end]

    assert hashlib.sha256(main_panel.encode("utf-8")).hexdigest() == MAIN_PANEL_SHA256
