from ai_brain.base_model import Action, DecisionOutput
from services.exit_partial_guard import ExitPartialGuardPolicy
from services.exit_position_matcher import ExitPositionMatcher


def _normalize_symbol(value) -> str:
    return str(value or "").replace("/", "-").replace("-SWAP", "")


def _policy() -> ExitPartialGuardPolicy:
    return ExitPartialGuardPolicy(ExitPositionMatcher(_normalize_symbol))


def _decision(
    *,
    raw_response: dict | None = None,
    close_fraction: float = 0.5,
    current_price: float = 90.0,
) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning="测试部分平仓",
        position_size_pct=close_fraction,
        suggested_leverage=3.0,
        raw_response=raw_response or {},
        feature_snapshot={"current_price": current_price},
    )


def test_exit_partial_guard_blocks_ordinary_partial_loss() -> None:
    decision = _decision()
    reason = _policy().guard_reason(
        "ensemble_trader",
        decision,
        [
            {
                "symbol": "BTC-USDT-SWAP",
                "side": "long",
                "quantity": 2,
                "entry_price": 100.0,
                "current_price": 90.0,
            }
        ],
    )

    assert reason is not None
    assert "亏损部分平仓保护" in reason
    guard = decision.raw_response["loss_partial_exit_guard"]
    assert guard["applied"] is True
    assert guard["aggregate_unrealized_pnl"] == -20.0


def test_exit_partial_guard_uses_reported_unrealized_pnl_when_available() -> None:
    decision = _decision()
    reason = _policy().guard_reason(
        "ensemble_trader",
        decision,
        [
            {
                "symbol": "BTC-USDT",
                "side": "long",
                "quantity": 1,
                "entry_price": 100.0,
                "current_price": 120.0,
                "unrealized_pnl": -3.5,
            }
        ],
    )

    assert reason is not None
    assert decision.raw_response["loss_partial_exit_guard"]["aggregate_unrealized_pnl"] == -3.5


def test_exit_partial_guard_allows_profitable_or_full_or_hard_exits() -> None:
    positions = [
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC-USDT",
            "side": "long",
            "quantity": 1,
            "entry_price": 100.0,
            "current_price": 110.0,
        }
    ]

    assert (
        _policy().guard_reason(
            "ensemble_trader",
            _decision(current_price=110.0),
            positions,
        )
        is None
    )
    assert (
        _policy().guard_reason(
            "ensemble_trader",
            _decision(close_fraction=1.0),
            [{**positions[0], "current_price": 90.0}],
        )
        is None
    )
    assert (
        _policy().guard_reason(
            "ensemble_trader",
            _decision(raw_response={"fast_risk_trigger": "stop_loss"}),
            [{**positions[0], "current_price": 90.0}],
        )
        is None
    )
