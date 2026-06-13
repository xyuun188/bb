from __future__ import annotations

from ai_brain.base_model import Action, DecisionOutput
from services.entry_crowded_side_cap import EntryCrowdedSideCapPolicy
from services.entry_strategy_mode import EntryStrategyModeContextPolicy


def _short_entry(opportunity: dict | None = None, exposure: dict | None = None) -> DecisionOutput:
    raw: dict = {}
    if opportunity is not None:
        raw["opportunity_score"] = opportunity
    if exposure is not None:
        raw["strategy_mode"] = {"position_exposure": exposure}
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="ETH/USDT",
        action=Action.SHORT,
        confidence=0.7,
        reasoning="test short",
        position_size_pct=0.03,
        suggested_leverage=3.0,
        stop_loss_pct=0.012,
        take_profit_pct=0.04,
        raw_response=raw,
    )


def test_crowded_side_cap_blocks_ordinary_same_side_entry() -> None:
    exposure = {
        "dominant_side": "short",
        "net_ratio": -0.62,
        "short_count": 10,
        "short_count_share": 0.83,
        "short_unrealized_pnl": -3.5,
    }
    decision = _short_entry(
        opportunity={"score": 1.0, "min_score_required": 0.95}, exposure=exposure
    )
    reason = EntryCrowdedSideCapPolicy().block_reason(decision)
    assert reason is not None
    assert "crowded_side_cap" in reason
    assert decision.raw_response["crowded_side_cap"]["mode"] == "crowded_block"


def test_crowded_side_cap_allows_strong_aligned_signal() -> None:
    exposure = {
        "dominant_side": "short",
        "net_ratio": -0.62,
        "short_count": 10,
        "short_count_share": 0.83,
        "short_unrealized_pnl": -3.5,
    }
    opportunity = {
        "score": 4.5,
        "min_score_required": 0.95,
        "expected_net_return_pct": 0.9,
        "profit_quality_ratio": 1.8,
        "confidence": 0.85,
        "ml_aligned": True,
        "local_profit_aligned": True,
    }
    decision = _short_entry(opportunity=opportunity, exposure=exposure)
    assert EntryCrowdedSideCapPolicy().block_reason(decision) is None
    assert decision.raw_response["crowded_side_cap"]["mode"] == "crowded_strong_override"


def test_crowded_side_cap_hard_ceiling() -> None:
    exposure = {
        "dominant_side": "short",
        "net_ratio": -0.7,
        "short_count": 14,
        "short_count_share": 0.9,
        "short_unrealized_pnl": -8.0,
    }
    strong = {
        "score": 9.0,
        "min_score_required": 0.95,
        "expected_net_return_pct": 2.0,
        "profit_quality_ratio": 3.0,
        "confidence": 0.95,
        "ml_aligned": True,
    }
    decision = _short_entry(opportunity=strong, exposure=exposure)
    reason = EntryCrowdedSideCapPolicy().block_reason(decision)
    assert reason is not None
    assert decision.raw_response["crowded_side_cap"]["mode"] == "hard_ceiling"


def test_crowded_side_cap_hard_ceiling_allows_strict_probe_override() -> None:
    exposure = {
        "dominant_side": "short",
        "net_ratio": -0.7,
        "short_count": 14,
        "short_count_share": 0.9,
        "short_unrealized_pnl": -8.0,
    }
    exceptional_probe = {
        "score": 9.0,
        "min_score_required": 0.95,
        "expected_net_return_pct": 2.0,
        "profit_quality_ratio": 3.0,
        "confidence": 0.95,
        "ml_aligned": True,
        "local_profit_aligned": True,
        "expert_aligned": True,
        "probe_fraction": 0.04,
        "max_probe_size_pct": 0.012,
    }
    decision = _short_entry(opportunity=exceptional_probe, exposure=exposure)
    decision.position_size_pct = 0.012

    assert EntryCrowdedSideCapPolicy().block_reason(decision) is None
    assert decision.raw_response["crowded_side_cap"]["mode"] == "hard_ceiling_probe_override"


def test_crowded_side_cap_ignored_when_not_dominant() -> None:
    exposure = {
        "dominant_side": "neutral",
        "net_ratio": -0.2,
        "short_count": 4,
        "short_count_share": 0.5,
    }
    decision = _short_entry(opportunity={"score": 1.0}, exposure=exposure)
    assert EntryCrowdedSideCapPolicy().block_reason(decision) is None


def _strategy_kwargs(**overrides):
    data = {
        "market_regime": {"mode": "mixed", "confidence": 0.3},
        "daily_state": {"today_total_pnl": 2.0, "today_high_water_pnl": 5.0},
        "side_performance": {"long": {"pnl": 0.5}, "short": {"pnl": 0.5}},
        "symbol_side_performance": {},
        "model_contribution_performance": {},
        "position_exposure": {"dominant_side": "neutral"},
        "position_group_count": 10,
        "account_equity": 2000.0,
        "account_config": {"max_loss_usdt": 1000.0},
    }
    data.update(overrides)
    return data


def test_multiday_losing_side_marks_degraded_and_avoided() -> None:
    result = EntryStrategyModeContextPolicy().build(
        **_strategy_kwargs(
            side_performance={"long": {"pnl": 0.0, "count": 0}, "short": {"pnl": 0.0, "count": 0}},
            side_performance_multiday={
                "long": {"count": 12, "wins": 2, "losses": 10, "pnl": -30.0, "win_rate": 0.17},
                "short": {"count": 10, "wins": 6, "losses": 4, "pnl": 8.0, "win_rate": 0.6},
            },
        )
    )
    long_quality = result["side_quality"]["long"]
    assert long_quality["state"] == "degraded"
    assert long_quality["multiday_degraded"] is True
    assert long_quality["size_multiplier"] <= 0.55
    assert "long" in result["soft_avoided_directions"]
    assert result["side_performance_multiday"]["short"]["pnl"] == 8.0


def test_multiday_neutral_when_insufficient_history() -> None:
    result = EntryStrategyModeContextPolicy().build(
        **_strategy_kwargs(
            side_performance_multiday={
                "long": {"count": 3, "wins": 1, "losses": 2, "pnl": -2.0, "win_rate": 0.33},
            },
        )
    )
    assert result["side_quality"]["long"]["multiday_degraded"] is False
