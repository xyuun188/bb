from datetime import UTC, datetime, timedelta

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.exit_cooldown import ExitCooldownPolicy
from services.trading_policies import ExitPolicy


def _decision(raw_response: dict | None = None) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning="test",
        position_size_pct=0.5,
        suggested_leverage=3.0,
        raw_response=raw_response or {},
        feature_snapshot={"current_price": 100.0},
    )


def _decision_with_snapshot(feature_snapshot: dict) -> DecisionOutput:
    decision = _decision()
    decision.feature_snapshot = feature_snapshot
    return decision


class AlwaysMatch:
    def has_matching_position(self, positions, model_name, decision):
        return True


@pytest.mark.asyncio
async def test_exit_policy_predictive_downside_bypasses_ordinary_guards() -> None:
    decision = _decision(
        {
            "close_evidence": {
                "should_close": True,
                "moderate_opposite_pressure": True,
                "preventive_exit": True,
            }
        }
    )
    policy = ExitPolicy(exit_position_matcher=AlwaysMatch())

    result = await policy.evaluate(
        decision,
        "ensemble_trader",
        [{"symbol": "BTC/USDT", "side": "long", "quantity": 1.0}],
    )

    assert result.passed is True
    assert result.data is not None
    assert result.data["exit_arbitration"]["intent"] == "predictive_downside"
    assert result.data["exit_arbitration"]["bypass_cooldown"] is True
    assert result.data["pipeline_context"]["arbitration"]["intent"] == "predictive_downside"
    assert decision.raw_response["exit_arbitration"]["intent"] == "predictive_downside"


@pytest.mark.asyncio
async def test_exit_policy_ordinary_exit_still_uses_guard_chain() -> None:
    calls: list[str] = []

    class PartialGuard:
        def guard_reason(self, model_name, decision, open_positions):
            calls.append("partial")
            return None

    class Cooldown:
        def recent_exit_cooldown_reason(self, model_name, decision):
            calls.append("cooldown")
            return "cooldown-blocked"

    policy = ExitPolicy(
        exit_position_matcher=AlwaysMatch(),
        exit_partial_guard=PartialGuard(),
        exit_cooldown=Cooldown(),
    )

    result = await policy.evaluate(
        _decision(),
        "ensemble_trader",
        [{"symbol": "BTC/USDT", "side": "long", "quantity": 1.0}],
    )

    assert result.passed is False
    assert result.blocker == "recent_exit_cooldown"
    assert result.data is not None
    assert result.data["exit_arbitration"]["intent"] == "ordinary"
    assert calls == ["partial", "cooldown"]


@pytest.mark.asyncio
async def test_exit_policy_can_skip_snapshot_refresh_for_fast_local_paths() -> None:
    calls: list[str] = []

    class Snapshot:
        async def refresh_positions(self, open_positions):
            calls.append("refresh")
            return open_positions or []

    policy = ExitPolicy(
        exit_position_matcher=AlwaysMatch(),
        exit_position_snapshot=Snapshot(),
    )

    result = await policy.evaluate(
        _decision({"fast_risk_exit": True, "fast_risk_trigger": "fast_adverse_move"}),
        "ensemble_trader",
        [{"symbol": "BTC/USDT", "side": "long", "quantity": 1.0}],
        refresh_positions=False,
    )

    assert result.passed is True
    assert calls == []
    assert result.data is not None
    assert result.data["pipeline_context"]["refreshed_position_count"] == 1
    assert result.data["exit_arbitration"]["intent"] == "hard_risk"


@pytest.mark.asyncio
async def test_exit_policy_uses_volatility_aware_cooldown_from_real_policy() -> None:
    first = datetime(2026, 6, 8, tzinfo=UTC)
    current = first + timedelta(seconds=350)
    cooldown = ExitCooldownPolicy(normalize_symbol=str, clock=lambda: first)
    cooldown.remember_exit("ensemble_trader", _decision())
    cooldown.clock = lambda: current
    policy = ExitPolicy(
        exit_position_matcher=AlwaysMatch(),
        exit_cooldown=cooldown,
    )

    ordinary = await policy.evaluate(
        _decision(),
        "ensemble_trader",
        [{"symbol": "BTC/USDT", "side": "long", "quantity": 1.0}],
    )
    high_volatility = await policy.evaluate(
        _decision_with_snapshot(
            {
                "current_price": 100.0,
                "volatility_20": 0.09,
                "returns_5": 0.0,
                "returns_20": 0.0,
            }
        ),
        "ensemble_trader",
        [{"symbol": "BTC/USDT", "side": "long", "quantity": 1.0}],
    )

    assert ordinary.passed is False
    assert ordinary.blocker == "recent_exit_cooldown"
    assert high_volatility.passed is True
