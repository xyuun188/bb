from __future__ import annotations

from datetime import UTC, datetime, timedelta

from services.dynamic_position_capacity import DynamicPositionCapacityPolicy
from services.position_quality import PositionQualityScorer


def _flat_old_position(symbol: str, *, hours: float = 14.0, pnl: float = 0.05) -> dict:
    return {
        "model_name": "ensemble_trader",
        "symbol": symbol,
        "side": "long",
        "entry_price": 100.0,
        "current_price": 100.05,
        "quantity": 10.0,
        "unrealized_pnl": pnl,
        "created_at": (datetime.now(UTC) - timedelta(hours=hours)).isoformat(),
    }


def _healthy_position(symbol: str, *, hours: float = 0.5, pnl: float = 12.0) -> dict:
    return {
        "model_name": "ensemble_trader",
        "symbol": symbol,
        "side": "long",
        "entry_price": 100.0,
        "current_price": 101.2,
        "quantity": 10.0,
        "unrealized_pnl": pnl,
        "created_at": (datetime.now(UTC) - timedelta(hours=hours)).isoformat(),
    }


def test_position_quality_flags_old_flat_fee_drag_position() -> None:
    quality = PositionQualityScorer().score(_flat_old_position("BTC/USDT"))

    assert quality.should_release is True
    assert quality.bucket in {"release_candidate", "release_now"}
    assert "time_cost_flat_12h" in quality.reasons
    assert "fee_drag_dominates" in quality.reasons


def test_dynamic_capacity_reduces_before_expanding_when_low_quality_is_crowded() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 20)
    decision = policy.evaluate(
        open_positions=[_flat_old_position(f"SYM{i}/USDT") for i in range(10)],
        strategy_context={"recent_win_rate": 0.62, "today_risk_pnl": 0.0},
        market_regime={"confidence": 0.82},
        account_equity=1000.0,
    )

    assert decision.effective_limit < decision.base_limit
    assert decision.low_quality_count == 10
    assert "low_quality_pressure" in decision.factors["reason_codes"]


def test_dynamic_capacity_can_expand_when_market_is_clear_and_book_is_clean() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 20)
    decision = policy.evaluate(
        open_positions=[],
        strategy_context={"recent_win_rate": 0.68, "today_risk_pnl": 0.0},
        market_regime={"confidence": 0.9},
        account_equity=1000.0,
    )

    assert decision.effective_limit > decision.base_limit
    assert decision.low_quality_count == 0
    assert "trend_bonus" in decision.factors["reason_codes"]


def test_position_quality_derives_unrealized_pnl_when_snapshot_is_stale_zero() -> None:
    quality = PositionQualityScorer().score(
        {
            "model_name": "ensemble_trader",
            "symbol": "SOL/USDT",
            "side": "short",
            "entry_price": 100.0,
            "current_price": 110.0,
            "quantity": 2.0,
            "unrealized_pnl": 0.0,
            "created_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
        }
    )

    assert quality.pnl_ratio < 0
    assert "severe_loss_pressure" in quality.reasons


def test_position_quality_uses_okx_position_timestamp_fields() -> None:
    opened_at = datetime.now(UTC) - timedelta(hours=5)
    quality = PositionQualityScorer().score(
        {
            "model_name": "ensemble_trader",
            "symbol": "MSFT/USDT:USDT",
            "side": "long",
            "entry_price": 100.0,
            "current_price": 99.8,
            "quantity": 10.0,
            "unrealized_pnl": -2.0,
            "info": {"cTime": str(int(opened_at.timestamp() * 1000))},
        }
    )

    assert quality.hold_hours >= 4.9
    assert quality.hold_hours < 5.1


def test_dynamic_capacity_treats_learning_target_as_guidance_not_hard_lock() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 20)
    decision = policy.evaluate(
        open_positions=[_flat_old_position(f"SYM{i}/USDT") for i in range(7)],
        strategy_context={
            "target_position_groups": 1,
            "recent_win_rate": 0.45,
            "today_risk_pnl": 0.0,
        },
        market_regime={"confidence": 0.4},
        account_equity=1000.0,
    )

    assert decision.base_limit == 20
    assert decision.factors["learned_target_limit"] == 1
    assert decision.factors["rotation_slots"] >= 1
    assert decision.effective_limit > decision.open_group_count
    assert "base=" not in decision.reason
    assert "target=" not in decision.reason


def test_dynamic_capacity_uses_learned_expansion_above_current_book() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 4)
    decision = policy.evaluate(
        open_positions=[_healthy_position(f"GOOD{i}/USDT") for i in range(4)],
        strategy_context={
            "target_position_groups": 5,
            "recent_win_rate": 0.50,
            "today_risk_pnl": 0.0,
        },
        market_regime={"confidence": 0.55},
        account_equity=1000.0,
    )

    assert decision.open_group_count == 4
    assert decision.base_limit == 4
    assert decision.factors["learned_target_limit"] == 5
    assert decision.target_limit == 5
    assert decision.effective_limit == 5
    assert "learned_target_expansion" in decision.factors["reason_codes"]
    assert "学习目标高于当前运行容量" in decision.reason


def test_dynamic_capacity_repairs_stale_one_group_config_when_book_is_crowded() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 1)
    decision = policy.evaluate(
        open_positions=[_flat_old_position(f"OLD{i}/USDT") for i in range(6)],
        strategy_context={"target_position_groups": 1, "today_risk_pnl": 0.0},
        market_regime={"confidence": 0.48},
        account_equity=1000.0,
    )

    assert decision.factors["configured_limit"] == 1
    assert decision.base_limit > decision.open_group_count
    assert decision.effective_limit > decision.open_group_count
    assert "release_rotation_slots" in decision.factors["reason_codes"]


def test_dynamic_capacity_preserves_entry_rotation_slots_under_low_quality_pressure() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 4)
    decision = policy.evaluate(
        open_positions=[_flat_old_position(f"OLD{i}/USDT") for i in range(4)],
        strategy_context={
            "target_position_groups": 5,
            "rotation_slots": 1,
            "recent_win_rate": 0.42,
            "today_risk_pnl": 0.0,
        },
        market_regime={"confidence": 0.45},
        account_equity=1000.0,
    )

    assert decision.effective_limit == 5
    assert decision.entry_limit == 5
    assert decision.factors["entry_limit"] == 5
    assert "strategy_rotation_slots" in decision.factors["reason_codes"]
    assert "release_rotation_slots" in decision.factors["reason_codes"]
