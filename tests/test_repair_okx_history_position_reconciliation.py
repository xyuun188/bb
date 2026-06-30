from __future__ import annotations

import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from scripts import repair_okx_history_position_reconciliation as repair_script


@pytest.mark.asyncio
async def test_history_reconciliation_apply_requires_precise_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["repair_okx_history_position_reconciliation.py", "--apply"],
    )

    with pytest.raises(SystemExit):
        await repair_script.main()


@pytest.mark.asyncio
async def test_history_reconciliation_dry_run_allows_unfiltered_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_collect_repairs(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(repair_script, "collect_repairs", fake_collect_repairs)
    monkeypatch.setattr(
        sys,
        "argv",
        ["repair_okx_history_position_reconciliation.py", "--days", "3"],
    )

    result = await repair_script.main()

    assert result == 0
    assert captured["days"] == 3
    assert captured["position_ids"] == set()
    assert captured["exchange_order_ids"] == set()


@pytest.mark.asyncio
async def test_history_reconciliation_apply_uses_filter_and_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[str] = []
    item = repair_script.RepairItem(
        position=SimpleNamespace(
            id=42,
            symbol="BTC/USDT",
            side="long",
            quantity=1.0,
            current_price=99.0,
            realized_pnl=-1.0,
            closed_at=datetime(2026, 6, 26, tzinfo=UTC),
        ),
        order=SimpleNamespace(id=7, exchange_order_id="okx-7"),
        old_closed_at=datetime(2026, 6, 26, tzinfo=UTC),
        old_price=99.0,
        old_realized_pnl=-1.0,
        new_closed_at=datetime(2026, 6, 26, 0, 1, tzinfo=UTC),
        new_price=100.0,
        new_realized_pnl=0.0,
        close_fee_allocated=0.01,
        inferred_entry_fee=0.02,
    )

    async def fake_collect_repairs(**kwargs):
        assert kwargs["position_ids"] == {42}
        assert kwargs["exchange_order_ids"] == {"okx-7"}
        return [item]

    async def fake_backup_repairs(repairs, backup_dir=repair_script.BACKUP_DIR):
        assert repairs == [item]
        calls.append("backup")
        return tmp_path / "backup.jsonl"

    async def fake_apply_repairs(repairs):
        assert repairs == [item]
        calls.append("apply")

    monkeypatch.setattr(repair_script, "collect_repairs", fake_collect_repairs)
    monkeypatch.setattr(repair_script, "backup_repairs", fake_backup_repairs)
    monkeypatch.setattr(repair_script, "apply_repairs", fake_apply_repairs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repair_okx_history_position_reconciliation.py",
            "--apply",
            "--position-id",
            "42",
            "--exchange-order-id",
            "okx-7",
        ],
    )

    result = await repair_script.main()

    assert result == 0
    assert calls == ["backup", "apply"]
