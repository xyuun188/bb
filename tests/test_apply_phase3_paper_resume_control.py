from __future__ import annotations

from pathlib import Path

import pytest

from core.trading_mode import TradingModeManager
from scripts.apply_phase3_paper_resume_control import (
    CONFIRMATION_PHRASE,
    apply_phase3_paper_control_action,
)


@pytest.mark.asyncio
async def test_resume_requires_confirmation_and_paper_mode(tmp_path: Path) -> None:
    manager = TradingModeManager(state_path=tmp_path / "control.json")
    await manager.pause()

    report = await apply_phase3_paper_control_action(
        action="resume",
        confirmation="no",
        manager=manager,
    )

    assert report["status"] == "blocked"
    assert manager.is_paused is True
    assert "resume_confirmation_missing" in {item["code"] for item in report["blockers"]}


@pytest.mark.asyncio
async def test_resume_and_pause_round_trip_persists_state(tmp_path: Path) -> None:
    state_path = tmp_path / "control.json"
    manager = TradingModeManager(state_path=state_path)
    await manager.pause()

    resumed = await apply_phase3_paper_control_action(
        action="resume",
        confirmation=CONFIRMATION_PHRASE,
        manager=manager,
    )
    reloaded = TradingModeManager(state_path=state_path)

    assert resumed["status"] == "ok"
    assert resumed["changed"] is True
    assert reloaded.is_paused is False

    paused = await apply_phase3_paper_control_action(
        action="pause",
        confirmation=CONFIRMATION_PHRASE,
        manager=manager,
    )

    assert paused["status"] == "ok"
    assert TradingModeManager(state_path=state_path).is_paused is True
