from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_market_regime_readiness_blocker_has_chinese_reason() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(
        encoding="utf-8"
    )

    assert (
        "market_regime_stability_failed: '费后收益未能在至少两种市场状态下稳定为正'"
        in script
    )
