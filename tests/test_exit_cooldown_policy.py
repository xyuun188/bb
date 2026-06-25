from datetime import UTC, datetime, timedelta

from ai_brain.base_model import Action, DecisionOutput
from services.exit_cooldown import EXIT_SYMBOL_SIDE_COOLDOWN_SECONDS, ExitCooldownPolicy


def _decision(
    action: Action = Action.CLOSE_LONG,
    *,
    raw_response: dict | None = None,
    feature_snapshot: dict | None = None,
    model_name: str = "ensemble_trader",
) -> DecisionOutput:
    return DecisionOutput(
        model_name=model_name,
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="测试平仓",
        position_size_pct=0.5,
        suggested_leverage=3.0,
        raw_response=raw_response or {},
        feature_snapshot=feature_snapshot or {"current_price": 100.0},
    )


def test_exit_cooldown_blocks_same_symbol_side_across_models() -> None:
    now = datetime(2026, 6, 8, tzinfo=UTC)
    policy = ExitCooldownPolicy(
        normalize_symbol=lambda value: str(value).replace("/", "-"),
        clock=lambda: now,
    )
    policy.remember_exit("model_a", _decision())

    check = _decision()
    reason = policy.recent_exit_cooldown_reason("model_b", check)

    assert reason is not None
    assert "连续平仓冷却" in reason
    assert check.raw_response["recent_exit_cooldown"]["symbol"] == "BTC-USDT"
    assert check.raw_response["recent_exit_cooldown"]["cooldown_seconds"] == (
        EXIT_SYMBOL_SIDE_COOLDOWN_SECONDS
    )


def test_exit_cooldown_expires_after_window() -> None:
    first = datetime(2026, 6, 8, tzinfo=UTC)
    current = first + timedelta(seconds=EXIT_SYMBOL_SIDE_COOLDOWN_SECONDS + 1)
    policy = ExitCooldownPolicy(
        normalize_symbol=str,
        clock=lambda: first,
    )
    policy.remember_exit("model_a", _decision())
    policy.clock = lambda: current

    assert policy.recent_exit_cooldown_reason("model_a", _decision()) is None


def test_exit_cooldown_shortens_for_high_volatility() -> None:
    first = datetime(2026, 6, 8, tzinfo=UTC)
    current = first + timedelta(seconds=350)
    policy = ExitCooldownPolicy(
        normalize_symbol=str,
        clock=lambda: first,
    )
    policy.remember_exit("model_a", _decision())
    policy.clock = lambda: current

    high_volatility = _decision(
        feature_snapshot={
            "current_price": 100.0,
            "volatility_20": 0.09,
            "returns_5": 0.0,
            "returns_20": 0.0,
        }
    )

    assert policy.recent_exit_cooldown_reason("model_a", high_volatility) is None


def test_exit_cooldown_extends_for_stable_market() -> None:
    first = datetime(2026, 6, 8, tzinfo=UTC)
    current = first + timedelta(seconds=700)
    policy = ExitCooldownPolicy(
        normalize_symbol=str,
        clock=lambda: first,
    )
    policy.remember_exit("model_a", _decision())
    policy.clock = lambda: current

    stable = _decision(
        feature_snapshot={
            "current_price": 100.0,
            "volatility_20": 0.01,
            "returns_5": 0.002,
            "returns_20": 0.006,
            "atr_14": 1.0,
        }
    )

    reason = policy.recent_exit_cooldown_reason("model_a", stable)

    assert reason is not None
    assert stable.raw_response["recent_exit_cooldown"]["cooldown_seconds"] == 900.0


def test_exit_cooldown_bypasses_hard_risk_exit() -> None:
    now = datetime(2026, 6, 8, tzinfo=UTC)
    policy = ExitCooldownPolicy(normalize_symbol=str, clock=lambda: now)
    policy.remember_exit("model_a", _decision())

    hard_stop = _decision(raw_response={"fast_risk_trigger": "stop_loss"})

    assert policy.recent_exit_cooldown_reason("model_a", hard_stop) is None


def test_exit_cooldown_blocks_untradable_exit_even_when_hard_risk_retries() -> None:
    now = datetime(2026, 6, 8, tzinfo=UTC)
    policy = ExitCooldownPolicy(
        normalize_symbol=lambda value: str(value).replace("/", "-"),
        clock=lambda: now,
    )
    failed = _decision(
        raw_response={
            "fast_risk_trigger": "stop_loss",
            "untradable_exit_execution_error": {"reason": "OKX instrument suspended for trading"},
        }
    )
    policy.remember_exit("model_a", failed)

    retry = _decision(raw_response={"fast_risk_trigger": "stop_loss"})
    reason = policy.recent_exit_cooldown_reason("model_a", retry)

    assert reason is not None
    assert "不可交易平仓冷却" in reason
    assert retry.raw_response["untradable_exit_cooldown"]["symbol"] == "BTC-USDT"
    assert retry.raw_response["untradable_exit_cooldown"]["last_error"].startswith("OKX")


def test_exit_cooldown_bypasses_predictive_downside_exit() -> None:
    now = datetime(2026, 6, 8, tzinfo=UTC)
    policy = ExitCooldownPolicy(normalize_symbol=str, clock=lambda: now)
    policy.remember_exit("model_a", _decision())

    predictive = _decision(
        raw_response={
            "close_evidence": {
                "should_close": True,
                "moderate_opposite_pressure": True,
                "preventive_exit": True,
            }
        }
    )

    assert policy.recent_exit_cooldown_reason("model_a", predictive) is None


def test_exit_cooldown_blocks_old_low_quality_release_payload() -> None:
    now = datetime(2026, 6, 8, tzinfo=UTC)
    policy = ExitCooldownPolicy(normalize_symbol=str, clock=lambda: now)
    policy.remember_exit("model_a", _decision())

    release = _decision(
        raw_response={
            "forced_exit": True,
            "position_release_policy": {
                "source": "position_quality_capacity_release",
                "forced": True,
            },
            "close_evidence": {
                "forced_exit": True,
                "hard_risk": False,
                "source": "low_quality_position_release",
            },
        }
    )

    assert policy.recent_exit_cooldown_reason("model_a", release) is not None


def test_exit_cooldown_ignores_non_exit_decisions() -> None:
    now = datetime(2026, 6, 8, tzinfo=UTC)
    policy = ExitCooldownPolicy(normalize_symbol=str, clock=lambda: now)
    entry = _decision(Action.LONG)

    policy.remember_exit("model_a", entry)

    assert policy.recent_exit_cooldown_reason("model_a", entry) is None
    assert policy.recent_exit_groups == {}
