import numpy as np
import pandas as pd
import pytest

from services.ml_blocked_return_evidence import blocked_return_evidence


def _frame(days=20, per_day=4):
    records = []
    for day in range(days):
        for offset in range(per_day):
            records.append({
                "decision_timestamp": pd.Timestamp("2026-08-01", tz="UTC")
                + pd.Timedelta(days=day, minutes=offset),
                "long_return_pct": 1.0,
                "short_return_pct": -1.0,
                "long_execution_cost_pct": 0.1,
                "short_execution_cost_pct": 0.1,
            })
    return pd.DataFrame(records)


def test_many_correlated_symbols_cannot_manufacture_calendar_coverage():
    frame = _frame(days=2, per_day=1000)
    result = blocked_return_evidence(frame, np.full(len(frame), "long"))
    assert result["selected_count"] == 2000
    assert result["calendar_day_count"] == 2
    assert result["ready"] is False
    assert result["return_lcb_pct"] is None


def test_block_bound_is_deterministic_cost_complete_and_development_only():
    frame = _frame()
    sides = np.full(len(frame), "long")
    first = blocked_return_evidence(frame, sides)
    assert first == blocked_return_evidence(frame, sides)
    assert first["return_lcb_pct"] == pytest.approx(0.9)
    assert first["ready"] is True
    assert first["independent_evidence_available"] is False
    frame["long_return_pct"] = 0.0
    assert blocked_return_evidence(frame, sides)["return_lcb_pct"] == pytest.approx(-0.1)


def test_hold_days_and_same_day_cross_symbol_losses_are_kept_together():
    frame = _frame(days=30)
    sides = np.full(len(frame), "long", dtype=object)
    sides[4:8] = "hold"
    frame.loc[frame.index % 4 == 0, "long_return_pct"] = -3.0
    result = blocked_return_evidence(frame, sides)
    assert result["calendar_day_count"] == 30
    assert result["selected_calendar_day_count"] == 29
    assert result["return_lcb_pct"] == pytest.approx(-0.1)
    assert result == blocked_return_evidence(frame.iloc[::-1], sides[::-1])


def test_hold_only_and_one_day_of_trades_are_not_profit_evidence():
    frame = _frame()
    sides = np.full(len(frame), "hold", dtype=object)
    assert blocked_return_evidence(frame, sides)["ready"] is False
    sides[:4] = "long"
    assert blocked_return_evidence(frame, sides)["ready"] is False
    with pytest.raises(ValueError, match="length"):
        blocked_return_evidence(frame, sides[:1])
