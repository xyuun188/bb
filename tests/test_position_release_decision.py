from __future__ import annotations

from ai_brain.base_model import Action
from services.position_release_decision import PositionReleaseDecisionPolicy


def test_position_release_decision_builds_capital_rotation_close_from_scan() -> None:
    policy = PositionReleaseDecisionPolicy()
    decision = policy.build(
        model_name="ensemble_trader",
        symbol="SOL/USDT",
        positions=[{"side": "short"}],
        scan={
            "force_exit_candidate": True,
            "release_action": "close_short",
            "exit_score": 94.0,
            "release_reason": "fee_drag_dominates; loss_pressure",
            "position_quality": {
                "score": 22.0,
                "bucket": "release_now",
                "hold_hours": 16.0,
                "should_release": True,
            },
        },
        feature_vector={"current_price": 110.0},
    )

    assert decision is not None
    assert decision.action == Action.CLOSE_SHORT
    assert decision.position_size_pct == 1.0
    assert decision.raw_response.get("forced_exit") is not True
    assert decision.raw_response["exit_intent"] == "capital_rotation"
    assert decision.raw_response["close_evidence"].get("forced_exit") is not True
    assert decision.raw_response["close_evidence"]["hard_risk"] is False
    assert decision.raw_response["close_evidence"]["exit_intent"] == "capital_rotation"
    assert "低质量持仓释放" in decision.reasoning


def test_position_release_decision_protects_fresh_low_quality_scan() -> None:
    policy = PositionReleaseDecisionPolicy()
    scan = {
        "force_exit_candidate": True,
        "release_action": "close_short",
        "exit_score": 94.0,
        "release_reason": "hard_loss_pressure; signal_reversal",
        "position_quality": {
            "score": 72.0,
            "bucket": "high",
            "hold_hours": 0.14,
            "pnl_ratio": -0.028,
            "reasons": [
                "hard_loss_pressure",
                "signal_reversal",
                "fresh_position_observation",
            ],
            "should_release": False,
        },
    }

    assert policy.should_release(scan) is False
    assert (
        policy.build(
            model_name="ensemble_trader",
            symbol="MET/USDT",
            positions=[{"side": "short"}],
            scan=scan,
            feature_vector={"current_price": 0.185},
        )
        is None
    )


def test_position_release_decision_protects_fresh_loss_even_without_reason_tags() -> None:
    policy = PositionReleaseDecisionPolicy()
    scan = {
        "force_exit_candidate": True,
        "release_action": "close_short",
        "exit_score": 96.0,
        "release_reason": "severe_loss_pressure; signal_reversal_watch",
        "position_quality": {
            "score": 22.0,
            "bucket": "release_now",
            "hold_hours": 0.146,
            "pnl_ratio": -0.004,
            "reasons": [],
            "should_release": True,
        },
    }

    assert policy.should_release(scan) is False
    assert (
        policy.build(
            model_name="ensemble_trader",
            symbol="HMSTR/USDT",
            positions=[{"side": "short"}],
            scan=scan,
            feature_vector={"current_price": 0.0021},
        )
        is None
    )


def test_position_release_decision_blocks_losing_release_without_net_benefit() -> None:
    policy = PositionReleaseDecisionPolicy()
    scan = {
        "force_exit_candidate": True,
        "release_action": "close_long",
        "exit_score": 95.0,
        "release_reason": "stale_probe_capital_inefficient; fee_drag_dominates",
        "position_quality": {
            "score": 24.0,
            "bucket": "release_now",
            "hold_hours": 9.0,
            "pnl_ratio": -0.004,
            "reasons": ["stale_probe_capital_inefficient", "fee_drag_dominates"],
            "should_release": True,
        },
    }

    assert policy.should_release(scan) is False
    assert scan["profit_first_release_net_benefit_guard"]["skip_kind"] == (
        "profit_first_release_net_benefit_guard"
    )
    assert (
        policy.build(
            model_name="ensemble_trader",
            symbol="SOL/USDT",
            positions=[{"side": "long"}],
            scan=scan,
            feature_vector={"current_price": 110.0},
        )
        is None
    )


def test_position_release_decision_allows_hard_risk_losing_release() -> None:
    policy = PositionReleaseDecisionPolicy()
    scan = {
        "force_exit_candidate": True,
        "release_action": "close_long",
        "exit_score": 96.0,
        "release_reason": "hard_loss_pressure",
        "position_quality": {
            "score": 15.0,
            "bucket": "release_now",
            "hold_hours": 8.0,
            "pnl_ratio": -0.03,
            "reasons": ["hard_loss_pressure"],
            "should_release": True,
        },
    }

    assert policy.should_release(scan) is True
    decision = policy.build(
        model_name="ensemble_trader",
        symbol="ETH/USDT",
        positions=[{"side": "long"}],
        scan=scan,
        feature_vector={"current_price": 3000.0},
    )

    assert decision is not None
    assert decision.action == Action.CLOSE_LONG


def test_position_release_decision_allows_stronger_replacement_after_losing_release() -> None:
    policy = PositionReleaseDecisionPolicy()
    scan = {
        "force_exit_candidate": True,
        "release_action": "close_short",
        "exit_score": 93.0,
        "release_reason": "fee_drag_dominates",
        "position_quality": {
            "score": 28.0,
            "bucket": "release_now",
            "hold_hours": 10.0,
            "pnl_ratio": -0.004,
            "reasons": ["fee_drag_dominates"],
            "should_release": True,
        },
        "replacement_opportunity": {
            "decision_lane": "validated_probe",
            "expected_net_return_pct": 0.55,
            "profit_quality_ratio": 0.7,
        },
    }

    assert policy.should_release(scan) is True
    decision = policy.build(
        model_name="ensemble_trader",
        symbol="OP/USDT",
        positions=[{"side": "short"}],
        scan=scan,
        feature_vector={"current_price": 2.2},
    )

    assert decision is not None
    assert decision.action == Action.CLOSE_SHORT


def test_position_release_decision_ignores_non_release_scan() -> None:
    policy = PositionReleaseDecisionPolicy()

    assert policy.should_release({"exit_score": 50.0}) is False
    assert (
        policy.build(
            model_name="ensemble_trader",
            symbol="BTC/USDT",
            positions=[{"side": "long"}, {"side": "short"}],
            scan={"force_exit_candidate": True, "exit_score": 95.0},
            feature_vector=None,
        )
        is None
    )
