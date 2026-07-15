from __future__ import annotations

import pytest

from scripts import run_local_ai_tools_training_cursors as script


@pytest.mark.asyncio
async def test_run_once_returns_canonical_training_cursors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    async def shadow_counter() -> int:
        return 14

    async def trade_counter() -> int:
        return 61

    async def fake_close_db() -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(script, "close_db", fake_close_db)

    result = await script.run_once(
        shadow_counter=shadow_counter,
        trade_counter=trade_counter,
    )

    assert result == {
        "trained": False,
        "reason": "cursor_probe_complete",
        "completed_shadow_sample_count": 14,
        "completed_trade_sample_count": 61,
        "training_process_isolated": True,
        "cursor_policy": "canonical_clean_training_view",
    }
    assert closed is True


@pytest.mark.asyncio
async def test_run_once_returns_structured_error_and_closes_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    async def fail_shadow_counter() -> int:
        raise RuntimeError("cursor query failed")

    async def unused_trade_counter() -> int:
        raise AssertionError("trade cursor should not run after shadow failure")

    async def fake_close_db() -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(script, "close_db", fake_close_db)

    result = await script.run_once(
        shadow_counter=fail_shadow_counter,
        trade_counter=unused_trade_counter,
    )

    assert result["trained"] is False
    assert result["reason"] == "error"
    assert result["error"] == "cursor query failed"
    assert result["training_process_isolated"] is True
    assert closed is True
