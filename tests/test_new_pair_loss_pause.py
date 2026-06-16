from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from services.new_pair_loss_pause import NewPairLossPausePolicy


def _position(
    *,
    is_open: bool,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
    closed_at: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        is_open=is_open,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        closed_at=closed_at,
    )


async def _balance_snapshot(_mode: str) -> dict[str, Any]:
    return {"allocatable": 1000.0}


@pytest.mark.asyncio
async def test_new_pair_loss_pause_advises_after_daily_equity_loss() -> None:
    now = datetime.now(UTC)
    rows = [
        _position(is_open=False, realized_pnl=-30.0, closed_at=now - timedelta(minutes=5)),
        _position(is_open=True, unrealized_pnl=-25.0),
    ]
    calls: list[tuple[str, bool | None, int]] = []

    async def records(mode: str, is_open: bool | None, limit: int) -> list[Any]:
        calls.append((mode, is_open, limit))
        return rows

    async def baseline(
        mode: str,
        allocated: float,
        positions: list[Any],
        realized_pnl: float,
        unrealized_pnl: float,
        total_pnl: float,
    ) -> dict[str, Any]:
        assert mode == "paper"
        assert allocated == 1000.0
        assert positions == rows
        assert realized_pnl == -30.0
        assert unrealized_pnl == -25.0
        assert total_pnl == -55.0
        return {"today_equity_pnl": -55.0}

    policy = NewPairLossPausePolicy(
        balance_snapshot_provider=_balance_snapshot,
        position_records_provider=records,
        daily_equity_baseline_provider=baseline,
    )

    reason = await policy.cooldown_loss_pause_reason("paper", 100.0, 0.5)

    assert reason is not None
    assert "slowed by realized daily loss guard" in reason
    assert "Today equity PnL -55.00 USDT" in reason
    assert "Trigger is 50% of max daily loss 100.00 USDT = 50.00 USDT" in reason
    assert calls == [("paper", None, 5000)]


@pytest.mark.asyncio
async def test_new_pair_loss_pause_allows_when_daily_loss_below_trigger() -> None:
    async def records(_mode: str, _is_open: bool | None, _limit: int) -> list[Any]:
        return [_position(is_open=False, realized_pnl=-12.0, closed_at=datetime.now(UTC))]

    async def baseline(*_args: Any) -> dict[str, Any]:
        return {"today_equity_pnl": -12.0}

    policy = NewPairLossPausePolicy(
        balance_snapshot_provider=_balance_snapshot,
        position_records_provider=records,
        daily_equity_baseline_provider=baseline,
    )

    assert await policy.cooldown_loss_pause_reason("paper", 100.0, 0.5) is None


@pytest.mark.asyncio
async def test_new_pair_loss_pause_advises_fresh_loss_streak() -> None:
    now = datetime.now(UTC)

    async def records(mode: str, is_open: bool | None, limit: int) -> list[Any]:
        assert mode == "live"
        assert is_open is False
        assert limit == 20
        return [
            _position(is_open=False, realized_pnl=-25.0, closed_at=now - timedelta(minutes=4)),
            _position(is_open=False, realized_pnl=-30.0, closed_at=now - timedelta(minutes=9)),
            _position(is_open=False, realized_pnl=8.0, closed_at=now - timedelta(minutes=15)),
        ]

    policy = NewPairLossPausePolicy(
        balance_snapshot_provider=_balance_snapshot,
        position_records_provider=records,
    )

    reason = await policy.recent_loss_streak_pause_reason("live", 100.0, 0.5)

    assert reason is not None
    assert "slowed by consecutive realized losses" in reason
    assert "Recent losing streak: 2 trades, total loss 55.00 USDT" in reason


@pytest.mark.asyncio
async def test_new_pair_loss_pause_allows_old_loss_streak() -> None:
    async def records(_mode: str, _is_open: bool | None, _limit: int) -> list[Any]:
        return [
            _position(
                is_open=False,
                realized_pnl=-80.0,
                closed_at=datetime.now(UTC) - timedelta(minutes=45),
            )
        ]

    policy = NewPairLossPausePolicy(
        balance_snapshot_provider=_balance_snapshot,
        position_records_provider=records,
    )

    assert await policy.recent_loss_streak_pause_reason("paper", 100.0, 0.5) is None
