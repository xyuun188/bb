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


@pytest.mark.asyncio
async def test_execution_mode_switch_preserves_unified_active_model(tmp_path: Path) -> None:
    state_path = tmp_path / "trading-control-state.json"
    manager = TradingModeManager(state_path=state_path)

    await manager.select_active_model("ensemble_trader")
    await manager.switch_to_live()
    await manager.switch_to_paper()

    assert manager.active_model_name == "ensemble_trader"
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["active_model_name"] == "ensemble_trader"
    assert "live_model_name" not in persisted
    assert "scan_mode" not in persisted
    with pytest.raises(ValueError, match="cannot replace the active model"):
        await manager.switch_to_live("different_model")
    assert manager.active_model_name == "ensemble_trader"


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
                "active_model_name": "ensemble_trader",
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
    assert manager.active_model_name == "ensemble_trader"
    assert read_count == 0


def test_removed_control_keys_are_not_loaded(tmp_path: Path) -> None:
    state_path = tmp_path / "trading-control-state.json"
    state_path.write_text(
        json.dumps(
            {
                "mode": "live",
                "paused": False,
                "scan_mode": "manual",
                "live_model_name": "removed-model-alias",
                "mode_changed_at": "2026-06-23T00:00:00",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    manager = TradingModeManager(state_path=state_path)

    assert manager.mode == TradingMode.LIVE
    assert manager.active_model_name is None
    assert not hasattr(manager, "scan_mode")
    assert not hasattr(manager, "live_model_name")
