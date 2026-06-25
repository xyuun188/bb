from datetime import datetime
from types import SimpleNamespace

from services.position_snapshot_syncer import PositionSnapshotSyncer


def test_position_snapshot_syncer_updates_single_position_quantity_and_protection():
    position = SimpleNamespace(
        is_open=True,
        quantity=0.5,
        current_price=100.0,
        entry_price=100.0,
        leverage=1.0,
        side="long",
        stop_loss_price=None,
        take_profit_price=None,
        unrealized_pnl=0.0,
        updated_at=None,
    )

    changed = PositionSnapshotSyncer().sync(
        [position],
        exchange_quantity=1.0,
        current_price=110.0,
        entry_price=100.0,
        leverage=3.0,
        exchange_unrealized=12.0,
        stop_loss_price=95.0,
        take_profit_price=125.0,
    )

    assert changed is True
    assert position.quantity == 1.0
    assert position.current_price == 110.0
    assert position.leverage == 3.0
    assert position.stop_loss_price == 95.0
    assert position.take_profit_price == 125.0
    assert position.unrealized_pnl == 12.0
    assert isinstance(position.updated_at, datetime)


def test_position_snapshot_syncer_corrects_entry_price_after_exchange_contract_adjustment():
    position = SimpleNamespace(
        is_open=True,
        quantity=100.0,
        current_price=2.44,
        entry_price=23.21,
        leverage=6.0,
        side="long",
        stop_loss_price=2.2099,
        take_profit_price=2.6026,
        unrealized_pnl=13.0,
        updated_at=None,
    )

    changed = PositionSnapshotSyncer().sync(
        [position],
        exchange_quantity=100.0,
        current_price=2.44,
        entry_price=2.31,
        leverage=6.0,
        exchange_unrealized=13.0,
        stop_loss_price=2.2099,
        take_profit_price=2.6026,
    )

    assert changed is True
    assert position.quantity == 100.0
    assert position.entry_price == 2.31
    assert position.current_price == 2.44
    assert position.unrealized_pnl == 13.0


def test_position_snapshot_syncer_resizes_fragmented_positions_proportionally():
    first = SimpleNamespace(
        is_open=True,
        quantity=2.0,
        current_price=100.0,
        entry_price=100.0,
        leverage=1.0,
        side="long",
        stop_loss_price=95.0,
        take_profit_price=120.0,
        unrealized_pnl=0.0,
        updated_at=None,
    )
    second = SimpleNamespace(
        is_open=True,
        quantity=1.0,
        current_price=100.0,
        entry_price=100.0,
        leverage=1.0,
        side="long",
        stop_loss_price=95.0,
        take_profit_price=120.0,
        unrealized_pnl=0.0,
        updated_at=None,
    )

    changed = PositionSnapshotSyncer().sync(
        [first, second],
        exchange_quantity=1.5,
        current_price=110.0,
        entry_price=100.0,
        leverage=2.0,
        exchange_unrealized=9.0,
    )

    assert changed is True
    assert first.quantity == 1.0
    assert second.quantity == 0.5
    assert first.unrealized_pnl == 6.0
    assert second.unrealized_pnl == 3.0


def test_position_snapshot_syncer_recalculates_pnl_without_marking_snapshot_changed():
    position = SimpleNamespace(
        is_open=True,
        quantity=2.0,
        current_price=100.0,
        entry_price=100.0,
        leverage=1.0,
        side="short",
        stop_loss_price=105.0,
        take_profit_price=90.0,
        unrealized_pnl=0.0,
        updated_at=None,
    )

    changed = PositionSnapshotSyncer().sync(
        [position],
        exchange_quantity=2.0,
        current_price=95.0,
        entry_price=100.0,
        leverage=1.0,
        exchange_unrealized=0.0,
        stop_loss_price=105.0,
        take_profit_price=90.0,
    )

    assert changed is False
    assert position.unrealized_pnl == 10.0
    assert isinstance(position.updated_at, datetime)


def test_position_snapshot_syncer_ignores_invalid_or_closed_positions():
    closed = SimpleNamespace(is_open=False, quantity=1.0)

    assert (
        PositionSnapshotSyncer().sync(
            [closed],
            exchange_quantity=1.0,
            current_price=110.0,
            entry_price=100.0,
            leverage=1.0,
            exchange_unrealized=1.0,
        )
        is False
    )
    assert (
        PositionSnapshotSyncer().sync(
            [],
            exchange_quantity=0.0,
            current_price=110.0,
            entry_price=100.0,
            leverage=1.0,
            exchange_unrealized=1.0,
        )
        is False
    )
