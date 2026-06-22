from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.trading_mode import TradingMode, TradingModeManager


@pytest.mark.asyncio
async def test_pause_state_is_shared_across_manager_instances(tmp_path: Path) -> None:
    state_path = tmp_path / "trading-control-state.json"
    dashboard_manager = TradingModeManager(state_path=state_path)
    worker_manager = TradingModeManager(state_path=state_path)

    assert worker_manager.is_paused is False

    await dashboard_manager.pause()

    assert worker_manager.is_paused is True
    assert worker_manager.get_state()["paused"] is True

    await dashboard_manager.resume()

    assert worker_manager.is_paused is False


def test_unchanged_state_file_is_not_reloaded_on_every_property_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "trading-control-state.json"
    state_path.write_text(
        json.dumps(
            {
                "mode": "live",
                "paused": False,
                "scan_mode": "auto",
                "live_model_name": "ensemble_trader",
                "mode_changed_at": "2026-06-23T00:00:00",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    manager = TradingModeManager(state_path=state_path)

    read_count = 0
    path_type = type(state_path)
    original_read_text = path_type.read_text

    def read_text_spy(self: Path, *args: object, **kwargs: object) -> str:
        nonlocal read_count
        if self == state_path:
            read_count += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(path_type, "read_text", read_text_spy)

    assert manager.mode == TradingMode.LIVE
    assert manager.live_model_name == "ensemble_trader"
    assert read_count == 0
