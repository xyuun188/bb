from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from config.settings import settings
from db.repositories.trade_repo import TradeRepository
from db.session import close_db, get_session_ctx, init_db
from web_dashboard.api.trades import (
    _close_order_position_reason,
    _matching_closed_positions_for_order,
)


def _position(**overrides):
    data = {
        "id": 1,
        "model_name": "ensemble_trader",
        "execution_mode": "paper",
        "symbol": "MET/USDT",
        "side": "short",
        "quantity": 10.0,
        "entry_price": 0.1779,
        "current_price": 0.17954444444444442,
        "realized_pnl": -0.0254,
        "is_open": False,
        "created_at": datetime(2026, 6, 21, 11, 44, 37, tzinfo=UTC),
        "closed_at": datetime(2026, 6, 21, 11, 49, 35, tzinfo=UTC),
    }
    data.update(overrides)
    return SimpleNamespace(**data)


@pytest.mark.asyncio
async def test_trade_repository_matches_okx_swap_symbol_variants(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trading.db').as_posix()}",
    )
    await init_db()
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            await repo.open_position(
                {
                    "model_name": "ensemble_trader",
                    "execution_mode": "paper",
                    "symbol": "MET/USDT",
                    "side": "short",
                    "quantity": 50.0,
                    "entry_price": 0.1751,
                    "current_price": 0.1751,
                    "leverage": 2.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "is_open": True,
                }
            )

        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            rows = await repo.get_matching_open_positions(
                model_name="ensemble_trader",
                symbol="MET/USDT:USDT",
                side="short",
                execution_mode="paper",
            )

        assert len(rows) == 1
        assert rows[0].symbol == "MET/USDT"
    finally:
        await close_db()


def test_trade_detail_matches_split_positions_for_one_okx_close_order() -> None:
    filled_at = datetime(2026, 6, 21, 11, 49, 36, tzinfo=UTC)
    order = SimpleNamespace(
        id=2490,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="MET/USDT:USDT",
        side="buy",
        quantity=90.0,
        price=0.17954444444444442,
        status="filled",
        exchange_order_id="3675558359740424192",
        filled_at=filled_at,
    )
    positions = [
        _position(id=1576, quantity=10.0, closed_at=filled_at - timedelta(seconds=1)),
        _position(
            id=1577,
            quantity=80.0,
            entry_price=0.1780375,
            realized_pnl=-0.1295,
            closed_at=filled_at - timedelta(seconds=1),
        ),
    ]

    matched = _matching_closed_positions_for_order(order, positions)
    reason = _close_order_position_reason(order, matched, execution_source="okx")

    assert {position.id for position in matched} == {1576, 1577}
    assert sum(position.quantity for position in matched) == pytest.approx(90.0)
    assert reason is not None
    assert "OKX" in reason
    assert "MET/USDT" in reason
    assert "90" in reason
    assert "2 " in reason
