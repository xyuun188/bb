from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from services.position_replacement_opportunity import (
    select_position_replacement_opportunity,
)

NOW = datetime(2026, 7, 25, 8, 0, tzinfo=UTC)


def _row(
    *,
    row_id: int,
    symbol: str,
    created_at: datetime | None = None,
    lcb: float = 0.3,
    expected: float = 0.5,
    expected_loss: float | None = 0.2,
    valid_for_seconds: float = 1800.0,
    is_paper: bool = True,
    execution_cost_complete: bool = True,
    generated_at: datetime | None = None,
) -> SimpleNamespace:
    observed_at = created_at or NOW - timedelta(minutes=5)
    provenance_at = generated_at or observed_at
    return SimpleNamespace(
        id=row_id,
        symbol=symbol,
        action="long",
        created_at=observed_at,
        is_paper=is_paper,
        raw_llm_response={
            "opportunity_score": {
                "production_eligible": True,
                "expected_net_return_pct": expected,
                "return_lcb_pct": lcb,
                "expected_loss_pct": expected_loss,
                "execution_cost": {
                    "production_eligible": execution_cost_complete,
                    "source": "executable_orderbook",
                },
                "policy_provenance": {
                    "source": "governed_return_distribution",
                    "observation_window": "decision_time",
                    "sample_count": 2,
                    "generated_at": provenance_at.isoformat(),
                    "strategy_version": "return-v1",
                    "fallback_reason": "",
                    "valid_for_seconds": valid_for_seconds,
                },
            }
        },
        decision_learning_snapshot=None,
    )


def _select(rows: list[SimpleNamespace], *, mode: str = "paper") -> dict:
    return select_position_replacement_opportunity(
        rows,
        execution_mode=mode,
        open_symbols={"BTC/USDT"},
        now=NOW,
        max_age_seconds=45 * 60,
    )


def test_selects_strongest_fresh_positive_unheld_opportunity() -> None:
    selected = _select(
        [
            _row(row_id=1, symbol="BTC/USDT", lcb=0.9),
            _row(row_id=2, symbol="ETH/USDT", lcb=0.2),
            _row(row_id=3, symbol="SOL-USDT-SWAP", lcb=0.4),
            _row(row_id=4, symbol="XRP/USDT", lcb=-0.1),
        ]
    )

    assert selected["available"] is True
    assert selected["decision_id"] == 3
    assert selected["symbol"] == "SOL/USDT"
    assert selected["return_lcb_pct"] == 0.4
    assert selected["execution_scope"] == "paper_only"
    assert selected["production_permission"] is False
    assert selected["creates_order"] is False
    assert selected["can_increase_leverage"] is False


@pytest.mark.parametrize(
    "row",
    [
        _row(row_id=1, symbol="ETH/USDT", created_at=NOW - timedelta(hours=1)),
        _row(row_id=2, symbol="ETH/USDT", valid_for_seconds=60),
        _row(row_id=3, symbol="ETH/USDT", execution_cost_complete=False),
        _row(row_id=4, symbol="ETH/USDT", expected_loss=None),
        _row(row_id=5, symbol="ETH/USDT", is_paper=False),
        _row(row_id=6, symbol="ETH/USDT", created_at=NOW + timedelta(minutes=1)),
        _row(row_id=7, symbol="ETH/USDT", generated_at=NOW + timedelta(minutes=1)),
    ],
)
def test_rejects_stale_incomplete_live_or_future_opportunity(row: SimpleNamespace) -> None:
    selected = _select([row])

    assert selected["available"] is False
    assert selected["reason"] == "no_fresh_cost_complete_unheld_opportunity"


def test_live_mode_never_exposes_replacement_opportunity() -> None:
    selected = _select([_row(row_id=1, symbol="ETH/USDT")], mode="live")

    assert selected["available"] is False
    assert selected["reason"] == "paper_only"
    assert selected["production_permission"] is False
    assert selected["creates_order"] is False
