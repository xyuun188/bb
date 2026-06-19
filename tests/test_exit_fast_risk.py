from services.exit_fast_risk import ExitFastRiskPolicy
from services.exit_predictive_reversal import ExitPredictiveReversalPolicy


def _policy(seconds_since_profit_exit: float = 0.0) -> ExitFastRiskPolicy:
    return ExitFastRiskPolicy(
        predictive_reversal=ExitPredictiveReversalPolicy(),
        seconds_since_profit_exit=lambda _state: seconds_since_profit_exit,
    )


def test_profit_drawdown_respects_profit_exit_cooldown() -> None:
    plan = _policy(seconds_since_profit_exit=120.0).profit_drawdown_exit_plan(
        side="long",
        current_price=104.0,
        entry_price=100.0,
        unrealized_pnl=6.0,
        peak_state={"peak_unrealized_pnl": 10.0, "peak_pnl_ratio": 0.01},
        hold_minutes=20.0,
        volume_ratio=1.2,
        returns_1=-0.01,
        returns_5=-0.02,
    )

    assert plan["should_exit"] is False
    assert plan["fraction"] == 0.0
    assert plan["seconds_since_last_profit_exit"] == 120.0
    assert "利润保护" in plan["note"]


def test_profit_drawdown_full_closes_severe_retrace_with_fee_buffer() -> None:
    plan = _policy().profit_drawdown_exit_plan(
        side="long",
        current_price=101.0,
        entry_price=100.0,
        unrealized_pnl=5.0,
        peak_state={"peak_unrealized_pnl": 20.0, "peak_pnl_ratio": 0.02},
        hold_minutes=20.0,
        volume_ratio=1.2,
        returns_1=-0.01,
        returns_5=-0.02,
    )

    assert plan["should_exit"] is True
    assert plan["fraction"] == 1.0
    assert plan["retrace_ratio"] >= 0.68


def test_profit_drawdown_reduces_on_predictive_reversal_before_full_line() -> None:
    plan = _policy().profit_drawdown_exit_plan(
        side="long",
        current_price=106.0,
        entry_price=100.0,
        unrealized_pnl=8.0,
        peak_state={"peak_unrealized_pnl": 10.0, "peak_pnl_ratio": 0.01},
        hold_minutes=20.0,
        volume_ratio=1.25,
        returns_1=-0.007,
        returns_5=-0.015,
        returns_20=-0.012,
        rsi_14=55.0,
        bb_pct=0.50,
        macd_diff=0.0,
        adx_14=20.0,
    )

    assert plan["should_exit"] is True
    assert plan["fraction"] == 0.60
    assert 64.0 <= plan["predictive_reversal"]["score"] < 82.0


def test_fast_adverse_full_closes_when_loss_and_stop_progress_are_large() -> None:
    plan = _policy().fast_adverse_exit_plan(
        side="long",
        entry_price=100.0,
        current_price=94.0,
        stop_loss=90.0,
        returns_1=-0.02,
        returns_5=-0.04,
        hold_minutes=10.0,
        volume_ratio=1.3,
        current_unrealized_pnl=-6.0,
    )

    assert plan["should_exit"] is True
    assert plan["fraction"] == 1.0
    assert plan["risk_progress"] >= 0.5


def test_fast_adverse_fresh_position_waits_when_stop_not_breached() -> None:
    plan = _policy().fast_adverse_exit_plan(
        side="long",
        entry_price=100.0,
        current_price=94.5,
        stop_loss=93.0,
        returns_1=-0.03,
        returns_5=-0.045,
        hold_minutes=6.0,
        volume_ratio=1.4,
        current_unrealized_pnl=-2.0,
    )

    assert plan["should_exit"] is False
    assert plan["fraction"] == 0.0
    assert plan["fresh_review_window"] is True
    assert "普通短线噪音" in plan["note"]


def test_fast_adverse_observes_small_losses() -> None:
    plan = _policy().fast_adverse_exit_plan(
        side="long",
        entry_price=100.0,
        current_price=99.5,
        stop_loss=95.0,
        returns_1=-0.001,
        returns_5=-0.002,
        hold_minutes=10.0,
        volume_ratio=1.0,
        current_unrealized_pnl=-0.5,
    )

    assert plan["should_exit"] is False
    assert plan["fraction"] == 0.0
    assert plan["adverse_pct"] < 0.008


def test_hard_adverse_without_confirmation_waits_for_review() -> None:
    plan = _policy().fast_adverse_exit_plan(
        side="long",
        entry_price=100.0,
        current_price=97.0,
        stop_loss=90.0,
        returns_1=-0.002,
        returns_5=-0.003,
        hold_minutes=20.0,
        volume_ratio=1.0,
        current_unrealized_pnl=-2.0,
        hard_adverse_observed=True,
    )

    assert plan["should_exit"] is False
    assert plan["fraction"] == 0.0
    assert plan["hard_adverse_observed"] is True
    assert plan["risk_progress"] < 0.5


def test_hard_adverse_exits_when_momentum_confirms() -> None:
    plan = _policy().fast_adverse_exit_plan(
        side="long",
        entry_price=100.0,
        current_price=97.0,
        stop_loss=90.0,
        returns_1=-0.026,
        returns_5=-0.041,
        hold_minutes=20.0,
        volume_ratio=1.2,
        current_unrealized_pnl=-2.0,
        hard_adverse_observed=True,
    )

    assert plan["should_exit"] is True
    assert plan["fraction"] == 1.0
    assert plan["strong_adverse_momentum"] is True


def test_fast_adverse_waits_when_market_data_is_suspicious() -> None:
    plan = _policy().fast_adverse_exit_plan(
        side="short",
        entry_price=100.0,
        current_price=104.0,
        stop_loss=110.0,
        returns_1=0.05,
        returns_5=0.06,
        hold_minutes=20.0,
        volume_ratio=1.5,
        current_unrealized_pnl=-5.0,
        hard_adverse_observed=True,
        data_quality_suspicious=True,
    )

    assert plan["should_exit"] is False
    assert plan["data_quality_suspicious"] is True


def test_suspicious_feature_price_detects_24h_range_break() -> None:
    reason = _policy().suspicious_feature_price_reason(
        side="long",
        feature_price=80.0,
        position_price=100.0,
        high_24h=105.0,
        low_24h=95.0,
        returns_1=-0.02,
        returns_5=-0.03,
    )

    assert reason == "feature price is outside the 24h exchange range"


def test_fast_adverse_does_not_partially_reduce_ordinary_losing_positions() -> None:
    plan = _policy().fast_adverse_exit_plan(
        side="long",
        entry_price=100.0,
        current_price=97.0,
        stop_loss=94.0,
        returns_1=-0.001,
        returns_5=-0.002,
        hold_minutes=10.0,
        volume_ratio=1.0,
        current_unrealized_pnl=-2.0,
    )

    assert plan["should_exit"] is False
    assert plan["fraction"] == 0.0
    assert "不因普通短线噪音全平" in plan["note"] or "不再做部分减仓" in plan["note"]


def test_fast_adverse_fresh_loser_requires_hard_evidence_even_with_predictive_score() -> None:
    plan = _policy().fast_adverse_exit_plan(
        side="long",
        entry_price=100.0,
        current_price=98.6,
        stop_loss=92.0,
        returns_1=-0.006,
        returns_5=-0.009,
        hold_minutes=7.0,
        volume_ratio=1.1,
        current_unrealized_pnl=-1.4,
        predictive_reversal_score=90.0,
    )

    assert plan["should_exit"] is False
    assert plan["fraction"] == 0.0
    assert plan["fresh_review_window"] is True
    assert plan["fresh_exit_strong_evidence_required"] is True
