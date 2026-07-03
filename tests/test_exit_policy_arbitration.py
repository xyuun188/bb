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
async def test_exit_policy_blocks_untradable_cooldown_before_hard_risk_bypass() -> None:
    first = datetime(2026, 6, 8, tzinfo=UTC)
    current = first + timedelta(seconds=30)
    cooldown = ExitCooldownPolicy(
        normalize_symbol=lambda value: str(value).replace("/", "-"),
        clock=lambda: first,
    )
    cooldown.remember_exit(
        "ensemble_trader",
        _decision(
            {
                "fast_risk_trigger": "stop_loss",
                "untradable_exit_execution_error": {
                    "reason": "OKX 51028: Contract under delivery",
                },
            }
        ),
    )
    cooldown.clock = lambda: current
    policy = ExitPolicy(
        exit_position_matcher=AlwaysMatch(),
        exit_cooldown=cooldown,
    )

    retry = _decision({"fast_risk_trigger": "stop_loss"})
    result = await policy.evaluate(
        retry,
        "ensemble_trader",
        [{"symbol": "BTC/USDT", "side": "long", "quantity": 1.0}],
    )

    assert result.passed is False
    assert result.blocker == "untradable_exit_cooldown"
    assert result.data is not None
    assert result.data["exit_arbitration"]["intent"] == "hard_risk"
    assert result.data["exit_arbitration"]["bypass_cooldown"] is True
    assert result.reason is not None
    assert "不可交易平仓冷却" in result.reason
    assert retry.raw_response["untradable_exit_cooldown"]["symbol"] == "BTC-USDT"
    assert retry.raw_response["untradable_exit_cooldown"]["last_error"].startswith("OKX 51028")


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
async def test_exit_policy_binds_exit_plan_from_original_snapshot_when_refresh_is_thin() -> None:
    class Snapshot:
        async def refresh_positions(self, open_positions):
            if open_positions is not None:
                open_positions[:] = [
                    {
                        "model_name": "ensemble_trader",
                        "symbol": "BTC/USDT",
                        "side": "long",
                        "entry_exchange_order_id": "entry-1",
                    }
                ]
            return open_positions or []

    class Cooldown:
        def recent_exit_cooldown_reason(self, model_name, decision):
            return None

    policy = ExitPolicy(
        exit_position_matcher=AlwaysMatch(),
        exit_position_snapshot=Snapshot(),
        exit_cooldown=Cooldown(),
    )
    decision = _decision({"close_evidence": {"entry_exchange_order_id": "entry-1"}})

    result = await policy.evaluate(
        decision,
        "ensemble_trader",
        [
            {
                "model_name": "ensemble_trader",
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_exchange_order_id": "entry-1",
                "profit_first_exit_plan": {"exit_plan_id": "pfep-original"},
                "profit_first_trade_plan": {
                    "plan_version": "profit-first-v3.1",
                    "decision_lane": "small",
                },
            }
        ],
    )

    assert result.passed is True
    assert decision.raw_response["profit_first_exit_reference"]["exit_plan_id"] == "pfep-original"
    assert (
        decision.raw_response["profit_first_exit_reference"][
            "missing_original_exit_plan_reference"
        ]
        is False
    )
    assert decision.raw_response["close_evidence"]["profit_first_exit_plan_id"] == "pfep-original"


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
