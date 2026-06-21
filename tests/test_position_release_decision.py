from __future__ import annotations

from ai_brain.base_model import Action
from services.position_release_decision import PositionReleaseDecisionPolicy


def test_position_release_decision_builds_forced_close_from_scan() -> None:
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
    assert decision.raw_response["forced_exit"] is True
    assert decision.raw_response["close_evidence"]["forced_exit"] is True
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
            "hold_hours": 0.04,
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
