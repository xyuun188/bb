from __future__ import annotations

from pathlib import Path

from services.trading_service import TradingService


def test_fixed_position_release_stage_is_physically_removed() -> None:
    assert not hasattr(TradingService, "_position_release_scan")
    assert not Path("services/position_release_decision.py").exists()
