from __future__ import annotations

from pathlib import Path

from services.entry_strategy_mode import EntryStrategyModeContextPolicy


def _strategy_kwargs(**overrides):
    data = {
        "market_regime": {"mode": "mixed"},
        "daily_state": {"today_total_pnl": 2.0, "today_high_water_pnl": 5.0},
        "side_performance": {},
        "symbol_side_performance": {},
        "model_contribution_performance": {},
        "position_exposure": {"dominant_side": "short"},
        "position_group_count": 10,
        "account_equity": 2000.0,
        "account_config": {},
    }
    data.update(overrides)
    return data


def test_fixed_crowded_side_gate_is_physically_removed() -> None:
    assert not Path("services/entry_crowded_side_cap.py").exists()


def test_multiday_side_history_is_observation_only() -> None:
    result = EntryStrategyModeContextPolicy().build(
        **_strategy_kwargs(
            side_performance_multiday={
                "long": {"count": 12, "pnl": -30.0, "return_lcb_pct": -1.0},
                "short": {"count": 10, "pnl": 8.0, "return_lcb_pct": 0.2},
            }
        )
    )
    assert "preferred_direction" not in result
    assert "blocked_directions" not in result
    assert result["side_quality"]["long"]["production_permission"] is False
    assert result["side_quality"]["short"]["production_permission"] is False
