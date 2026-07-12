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
    assert "market_clear" in decision.factors["reason_codes"]
    assert "positive_return_quality_bonus" not in decision.factors["reason_codes"]


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


def test_position_quality_protects_fresh_position_from_low_quality_release() -> None:
    quality = PositionQualityScorer().score(
        {
            "model_name": "ensemble_trader",
            "symbol": "MET/USDT",
            "side": "short",
            "entry_price": 0.18,
            "current_price": 0.185,
            "quantity": 100.0,
            "unrealized_pnl": -0.5,
            "created_at": (datetime.now(UTC) - timedelta(minutes=2)).isoformat(),
        },
        feature_vector={"returns_5": 0.02, "returns_20": 0.03, "macd_diff": 1.0, "bb_pct": 0.9},
    )

    assert "hard_loss_pressure" in quality.reasons
    assert "signal_reversal" in quality.reasons
    assert "fresh_position_observation" in quality.reasons
    assert quality.bucket == "high"
    assert quality.should_release is False


def test_position_quality_allows_fresh_position_hard_risk_release() -> None:
    quality = PositionQualityScorer().score(
        {
            "model_name": "ensemble_trader",
            "symbol": "MET/USDT",
            "side": "short",
            "entry_price": 0.18,
            "current_price": 0.192,
            "quantity": 100.0,
            "unrealized_pnl": -1.2,
            "created_at": (datetime.now(UTC) - timedelta(minutes=2)).isoformat(),
        },
        feature_vector={"returns_5": 0.02, "returns_20": 0.03, "macd_diff": 1.0, "bb_pct": 0.9},
    )

    assert "severe_loss_pressure" in quality.reasons
    assert "fresh_position_observation" not in quality.reasons
    assert quality.should_release is True


def test_position_quality_flags_stale_tiny_probe_capital_inefficiency() -> None:
    quality = PositionQualityScorer().score(
        {
            "model_name": "ensemble_trader",
            "symbol": "CRCL/USDT",
            "side": "long",
            "entry_price": 76.15,
            "current_price": 76.02,
            "quantity": 0.27,
            "unrealized_pnl": -0.035,
            "created_at": (datetime.now(UTC) - timedelta(hours=1.4)).isoformat(),
        }
    )

    assert "stale_probe_capital_inefficient" in quality.reasons
    assert quality.bucket == "release_candidate"
    assert quality.should_release is True


def test_position_quality_keeps_tiny_probe_winner_out_of_release_queue() -> None:
    quality = PositionQualityScorer().score(
        {
            "model_name": "ensemble_trader",
            "symbol": "KAITO/USDT",
            "side": "short",
            "entry_price": 0.4513,
            "current_price": 0.44,
            "quantity": 18.0,
            "unrealized_pnl": 0.2034,
            "created_at": (datetime.now(UTC) - timedelta(hours=16)).isoformat(),
        }
    )

    assert "winner_has_edge" in quality.reasons
    assert "stale_probe_capital_inefficient" not in quality.reasons
    assert quality.should_release is False


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


def test_dynamic_capacity_bonus_uses_return_quality_not_win_rate() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 20)
    positive_return = policy.evaluate(
        open_positions=[],
        strategy_context={
            "recent_win_rate": 0.35,
            "profit_factor": 2.0,
            "net_pnl": 10.0,
            "today_risk_pnl": 0.0,
        },
        market_regime={"confidence": 0.70},
        account_equity=1000.0,
    )
    high_win_negative_return = policy.evaluate(
        open_positions=[],
        strategy_context={
            "recent_win_rate": 0.85,
            "profit_factor": 0.40,
            "net_pnl": -10.0,
            "today_risk_pnl": 0.0,
        },
        market_regime={"confidence": 0.40},
        account_equity=1000.0,
    )

    assert "positive_return_quality_bonus" in positive_return.factors["reason_codes"]
    assert positive_return.effective_limit > positive_return.target_limit
    assert "positive_return_quality_bonus" not in high_win_negative_return.factors["reason_codes"]
    assert high_win_negative_return.effective_limit == high_win_negative_return.target_limit


def test_dynamic_capacity_stops_new_entries_when_stale_config_book_is_crowded() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 1)
    decision = policy.evaluate(
        open_positions=[_flat_old_position(f"OLD{i}/USDT") for i in range(6)],
        strategy_context={"target_position_groups": 1, "today_risk_pnl": 0.0},
        market_regime={"confidence": 0.48},
        account_equity=1000.0,
    )

    assert decision.factors["configured_limit"] == 1
    assert decision.base_limit == 1
    assert decision.entry_limit == 1
    assert decision.entry_limit < decision.open_group_count
    assert "release_rotation_slots" in decision.factors["reason_codes"]
    assert "over_capacity_release_first" in decision.factors["reason_codes"]


def test_dynamic_capacity_preserves_entry_rotation_slots_under_low_quality_pressure() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 4)
    decision = policy.evaluate(
        open_positions=[_flat_old_position(f"OLD{i}/USDT") for i in range(3)],
        strategy_context={
            "target_position_groups": 5,
            "rotation_slots": 1,
            "recent_win_rate": 0.42,
            "today_risk_pnl": 0.0,
        },
        market_regime={"confidence": 0.45},
        account_equity=1000.0,
    )

    assert decision.open_group_count == 3
    assert decision.effective_limit == 4
    assert decision.entry_limit == 4
    assert decision.entry_limit > decision.open_group_count
    assert "strategy_rotation_slots" in decision.factors["reason_codes"]
    assert "release_rotation_slots" in decision.factors["reason_codes"]


def test_dynamic_capacity_does_not_expand_entry_limit_when_book_is_already_over_capacity() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 20)
    decision = policy.evaluate(
        open_positions=[_flat_old_position(f"OLD{i}/USDT") for i in range(23)],
        strategy_context={
            "target_position_groups": 20,
            "rotation_slots": 3,
            "recent_win_rate": 0.42,
            "today_risk_pnl": -10.0,
        },
        market_regime={"confidence": 0.45},
        account_equity=1000.0,
    )

    assert decision.open_group_count == 23
    assert decision.base_limit == 20
    assert decision.entry_limit <= 20
    assert decision.entry_limit < decision.open_group_count
    assert "over_capacity_release_first" in decision.factors["reason_codes"]
    assert "rotation_entry_expansion" not in decision.factors["reason_codes"]


def test_dynamic_capacity_zero_config_uses_safe_default_without_infinite_expansion() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 0)
    decision = policy.evaluate(
        open_positions=[_flat_old_position(f"OLD{i}/USDT") for i in range(21)],
        strategy_context={"target_position_groups": 20, "today_risk_pnl": 0.0},
        market_regime={"confidence": 0.48},
        account_equity=1000.0,
    )

    assert decision.factors["configured_limit"] == 20
    assert decision.base_limit == 20
    assert decision.entry_limit <= 20
    assert decision.entry_limit < decision.open_group_count
    assert "over_capacity_release_first" in decision.factors["reason_codes"]


def test_dynamic_capacity_does_not_shrink_to_advisory_target_when_book_is_full() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 20)
    decision = policy.evaluate(
        open_positions=[
            _flat_old_position("OLD0/USDT"),
            *[_healthy_position(f"GOOD{i}/USDT") for i in range(19)],
        ],
        strategy_context={
            "target_position_groups": 15,
            "target_open_position_groups": 15,
            "strategy_learning_release_pressure_active": True,
            "portfolio_roster": {
                "target_position_groups": 15,
                "max_open_positions": 20,
                "rotation_slots": 3,
                "release_target_groups": 1,
                "policy_reason": "release_low_quality_positions_with_rotation_slots",
            },
            "strategy_learning": {
                "runtime": {
                    "target_position_groups": 15,
                    "max_open_positions": 20,
                    "rotation_slots": 3,
                    "release_target_groups": 1,
                    "capacity_policy_reason": "release_low_quality_positions_with_rotation_slots",
                }
            },
            "today_risk_pnl": 0.0,
        },
        market_regime={"confidence": 0.48},
        account_equity=1000.0,
    )

    assert decision.factors["learned_target_limit"] == 15
    assert decision.target_limit == 20
    assert decision.effective_limit == 20
    assert decision.entry_limit <= decision.open_group_count
    assert "over_capacity_release_first" not in decision.factors["reason_codes"]


def test_dynamic_capacity_opens_rotation_slot_for_stale_tiny_probe() -> None:
    policy = DynamicPositionCapacityPolicy(lambda: 3)
    positions = [
        {
            "model_name": "ensemble_trader",
            "symbol": "CRCL/USDT",
            "side": "long",
            "entry_price": 76.15,
            "current_price": 76.02,
            "quantity": 0.27,
            "unrealized_pnl": -0.035,
            "created_at": (datetime.now(UTC) - timedelta(hours=1.4)).isoformat(),
        },
        _healthy_position("KAITO/USDT", hours=16.0, pnl=12.0),
    ]

    decision = policy.evaluate(
        open_positions=positions,
        strategy_context={"target_position_groups": 2, "recent_win_rate": 0.50},
        market_regime={"confidence": 0.50},
        account_equity=1000.0,
    )

    assert decision.low_quality_count == 1
    assert decision.entry_limit > decision.open_group_count
    assert "release_rotation_slots" in decision.factors["reason_codes"]
