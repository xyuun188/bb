from ai_brain.base_model import Action, DecisionOutput
from services.entry_loss_cooldown import EntryLossCooldownPolicy
from services.entry_opportunity_gate import EntryOpportunityGatePolicy


def _decision(opportunity_score: dict, *, action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.74,
        reasoning="entry",
        position_size_pct=0.04,
        suggested_leverage=3.0,
        raw_response={"opportunity_score": opportunity_score},
    )


def _cooldown_profile() -> dict:
    return {
        "cooldown": True,
        "pnl": -32.0,
        "today_pnl": -16.0,
        "loss": 38.0,
        "today_loss": 16.0,
        "largest_loss": -14.0,
        "count": 5,
        "losses": 3,
        "wins": 2,
        "profit_factor": 0.45,
        "cooldown_remaining_hours": 2.5,
        "cooldown_reason": "recent losses",
    }


def test_entry_loss_cooldown_blocks_recent_same_side_loss() -> None:
    decision = _decision(
        {
            "score": 1.4,
            "min_score_required": 0.95,
            "expected_net_return_pct": 0.2,
            "profit_quality_ratio": 0.4,
            "reward_risk_ratio": 0.8,
            "server_profit_expected_return_pct": 0.1,
            "server_profit_loss_probability": 0.7,
            "tail_risk_score": 1.0,
            "symbol_side_profile": _cooldown_profile(),
        }
    )

    reason = EntryLossCooldownPolicy(lambda symbol: symbol.replace("/", "-")).reason(decision)

    assert reason is not None
    override = decision.raw_response["loss_cooldown_override"]
    assert override["allowed"] is False
    assert "score" in override["failed"]
    assert decision.raw_response["opportunity_score"]["loss_cooldown_override"]["allowed"] is False


def test_entry_loss_cooldown_allows_high_quality_override() -> None:
    decision = _decision(
        {
            "score": 4.2,
            "min_score_required": 0.95,
            "confidence": 0.82,
            "expected_net_return_pct": 1.1,
            "profit_quality_ratio": 1.1,
            "reward_risk_ratio": 1.6,
            "server_profit_expected_return_pct": 0.9,
            "server_profit_loss_probability": 0.32,
            "tail_risk_score": 0.4,
            "symbol_side_profile": _cooldown_profile(),
        }
    )

    reason = EntryLossCooldownPolicy().reason(decision)

    assert reason is None
    override = decision.raw_response["loss_cooldown_override"]
    assert override["allowed"] is True
    assert override["failed"] == []


def test_entry_loss_cooldown_blocks_fresh_recent_loss_without_strong_override() -> None:
    profile = {
        "cooldown": False,
        "pnl": -9.0,
        "today_pnl": -4.0,
        "loss": 9.0,
        "today_loss": 4.0,
        "largest_loss": -3.5,
        "count": 2,
        "losses": 1,
        "wins": 1,
        "profit_factor": 0.72,
        "cooldown_remaining_hours": 1.6,
        "cooldown_reason": "fresh recent loss",
        "last_loss_age_hours": 1.2,
    }
    decision = _decision(
        {
            "score": 1.8,
            "min_score_required": 0.95,
            "confidence": 0.74,
            "expected_net_return_pct": 0.5,
            "profit_quality_ratio": 0.55,
            "reward_risk_ratio": 0.9,
            "server_profit_expected_return_pct": 0.2,
            "server_profit_loss_probability": 0.55,
            "tail_risk_score": 0.45,
            "symbol_side_profile": profile,
        }
    )

    reason = EntryLossCooldownPolicy().reason(decision)

    assert reason is not None
    override = decision.raw_response["loss_cooldown_override"]
    assert override["allowed"] is False
    assert override["metrics"]["fresh_loss"] is True
    assert override["failed"]


def test_entry_loss_cooldown_uses_fresh_loss_remaining_instead_of_profile_zero() -> None:
    profile = {
        "cooldown": False,
        "pnl": -3.5,
        "today_pnl": -3.5,
        "loss": 3.5,
        "today_loss": 3.5,
        "largest_loss": -3.5,
        "count": 1,
        "losses": 1,
        "wins": 0,
        "profit_factor": 0.0,
        "cooldown_remaining_hours": 0.0,
        "cooldown_reason": "fresh recent loss",
        "last_loss_age_hours": 1.95,
    }
    decision = _decision(
        {
            "score": 1.8,
            "min_score_required": 0.95,
            "confidence": 0.74,
            "expected_net_return_pct": 0.5,
            "profit_quality_ratio": 0.55,
            "reward_risk_ratio": 0.9,
            "server_profit_expected_return_pct": 0.2,
            "server_profit_loss_probability": 0.55,
            "tail_risk_score": 0.45,
            "symbol_side_profile": profile,
        }
    )

    reason = EntryLossCooldownPolicy().reason(decision)

    assert reason is not None
    assert "0.0 小时" not in reason
    assert "约 3 分钟" in reason
    metrics = decision.raw_response["loss_cooldown_override"]["metrics"]
    assert metrics["fresh_loss_cooldown_remaining_hours"] == 0.05


def test_entry_loss_cooldown_describes_symbol_quarantine_without_zero_hour_eta() -> None:
    decision = _decision(
        {
            "score": 1.7,
            "min_score_required": 0.95,
            "expected_net_return_pct": 0.45,
            "profit_quality_ratio": 0.65,
            "reward_risk_ratio": 1.05,
            "server_profit_expected_return_pct": 0.35,
            "server_profit_loss_probability": 0.48,
            "tail_risk_score": 0.45,
            "symbol_profile": {
                **_cooldown_profile(),
                "profile_scope": "symbol",
                "cooldown_kind": "symbol_rolling_quarantine",
                "cooldown_time_based": False,
                "cooldown_remaining_hours": 0.0,
                "last_loss_side": "long",
                "cooldown_reason": "该币种最近滚动真实亏损过大",
            },
        },
        action=Action.LONG,
    )

    reason = EntryLossCooldownPolicy().reason(decision)

    assert reason is not None
    assert "0.0 小时" not in reason
    assert "不按固定倒计时" in reason
    metrics = decision.raw_response["loss_cooldown_override"]["metrics"]
    assert metrics["cooldown_time_based"] is False
    assert metrics["cooldown_kind"] == "symbol_rolling_quarantine"


def test_entry_loss_cooldown_allows_high_quality_fresh_loss_override() -> None:
    profile = {
        "cooldown": False,
        "pnl": -7.5,
        "today_pnl": -3.0,
        "loss": 7.5,
        "today_loss": 3.0,
        "largest_loss": -2.4,
        "count": 3,
        "losses": 1,
        "wins": 2,
        "profit_factor": 0.8,
        "cooldown_remaining_hours": 1.2,
        "cooldown_reason": "fresh recent loss",
        "last_loss_age_hours": 0.8,
    }
    decision = _decision(
        {
            "score": 4.8,
            "min_score_required": 0.95,
            "confidence": 0.84,
            "expected_net_return_pct": 1.45,
            "profit_quality_ratio": 1.2,
            "reward_risk_ratio": 1.7,
            "server_profit_expected_return_pct": 1.0,
            "server_profit_loss_probability": 0.28,
            "tail_risk_score": 0.35,
            "ml_aligned": True,
            "local_profit_aligned": True,
            "timeseries_aligned": True,
            "symbol_side_profile": profile,
        }
    )

    reason = EntryLossCooldownPolicy().reason(decision)

    assert reason is None
    override = decision.raw_response["loss_cooldown_override"]
    assert override["allowed"] is True
    assert override["metrics"]["fresh_loss"] is True


def test_entry_loss_cooldown_requires_three_sources_for_fresh_loss_reentry() -> None:
    profile = {
        "cooldown": False,
        "pnl": -7.5,
        "today_pnl": -3.0,
        "loss": 7.5,
        "today_loss": 3.0,
        "largest_loss": -2.4,
        "count": 3,
        "losses": 1,
        "wins": 2,
        "profit_factor": 0.8,
        "cooldown_remaining_hours": 1.2,
        "cooldown_reason": "fresh recent loss",
        "last_loss_age_hours": 0.8,
    }
    decision = _decision(
        {
            "score": 4.8,
            "min_score_required": 0.95,
            "confidence": 0.84,
            "expected_net_return_pct": 1.45,
            "profit_quality_ratio": 1.2,
            "reward_risk_ratio": 1.7,
            "server_profit_expected_return_pct": 1.0,
            "server_profit_loss_probability": 0.28,
            "tail_risk_score": 0.35,
            "ml_aligned": True,
            "local_profit_aligned": True,
            "timeseries_aligned": False,
            "symbol_side_profile": profile,
        }
    )

    reason = EntryLossCooldownPolicy().reason(decision)

    assert reason is not None
    override = decision.raw_response["loss_cooldown_override"]
    assert override["allowed"] is False
    assert "source_support" in override["failed"]
    assert override["metrics"]["fresh_loss_min_aligned_sources"] == 3


def test_entry_opportunity_gate_uses_injected_loss_cooldown_policy() -> None:
    decision = _decision(
        {
            "score": 1.4,
            "min_score_required": 0.95,
            "symbol_side_profile": _cooldown_profile(),
        }
    )
    policy = EntryOpportunityGatePolicy(symbol_loss_cooldown_policy=EntryLossCooldownPolicy())

    reason = policy.gate_reason(decision)

    assert reason is not None
    assert decision.raw_response["loss_cooldown_override"]["allowed"] is False


def test_entry_loss_cooldown_does_not_use_all_symbol_profile_for_opposite_side() -> None:
    decision = _decision(
        {
            "score": 1.7,
            "min_score_required": 0.95,
            "expected_net_return_pct": 0.45,
            "profit_quality_ratio": 0.65,
            "reward_risk_ratio": 1.05,
            "server_profit_expected_return_pct": 0.35,
            "server_profit_loss_probability": 0.48,
            "tail_risk_score": 0.45,
            "symbol_profile": {
                **_cooldown_profile(),
                "side": "all",
                "profile_scope": "symbol",
                "last_loss_side": "short",
                "cooldown_reason": "short-side recent loss",
            },
        },
        action=Action.LONG,
    )

    reason = EntryLossCooldownPolicy().reason(decision)

    assert reason is None
    assert "loss_cooldown_override" not in decision.raw_response
