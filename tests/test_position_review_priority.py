from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from services.exit_predictive_reversal import ExitPredictiveReversalPolicy
from services.position_quality import PositionQualityScorer
from services.position_review_priority import (
    PORTFOLIO_PROFIT_PROTECTION_EXIT_SCORE,
    PositionReviewPriorityPolicy,
)


def _policy(
    peaks: dict[Any, dict[str, Any]] | None = None,
    urgent_markers: tuple[str, ...] = ("near_stop", "predictive_reversal"),
) -> PositionReviewPriorityPolicy:
    peak_states = peaks or {}
    return PositionReviewPriorityPolicy(
        normalize_symbol=lambda value: str(value or "").split(":")[0],
        position_peak_key=lambda model, symbol, side: (model, symbol, side),
        position_peaks_provider=lambda: peak_states,
        predictive_reversal=ExitPredictiveReversalPolicy(),
        urgent_exit_markers=urgent_markers,
    )


def test_portfolio_profit_protection_scores_focus_group() -> None:
    score, reasons = _policy().portfolio_profit_protection_score(
        {
            "active": True,
            "focus_groups": [
                {
                    "model_name": "ensemble_trader",
                    "symbol": "BTC/USDT:USDT",
                }
            ],
        },
        "ensemble_trader",
        "BTC/USDT",
    )

    assert score == PORTFOLIO_PROFIT_PROTECTION_EXIT_SCORE
    assert reasons == ["portfolio_profit_protection_focus"]


def test_fast_position_exit_score_prioritizes_predictive_reversal() -> None:
    score, reasons = _policy().fast_position_exit_score(
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 100.0,
            "current_price": 105.0,
            "quantity": 1.0,
            "unrealized_pnl": 5.0,
            "stop_loss": 90.0,
        },
        SimpleNamespace(
            returns_1=-0.007,
            returns_5=-0.015,
            returns_20=-0.012,
            volume_ratio=1.25,
            rsi_14=72.0,
            bb_pct=0.90,
            macd_diff=-0.1,
            adx_14=12.0,
        ),
    )

    assert score == 88.0
    assert any(reason.startswith("predictive_reversal:") for reason in reasons)


def test_fast_position_exit_score_uses_profit_peak_retrace() -> None:
    score, reasons = _policy(
        {
            ("ensemble_trader", "BTC/USDT", "long"): {
                "peak_unrealized_pnl": 10.0,
            }
        }
    ).fast_position_exit_score(
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 100.0,
            "current_price": 106.0,
            "quantity": 1.0,
            "unrealized_pnl": 6.0,
            "stop_loss": 90.0,
        },
        None,
    )

    assert score == 80.0
    assert any(reason.startswith("profit_retrace:") for reason in reasons)


def test_fast_position_exit_score_prioritizes_small_profitable_position_lock() -> None:
    score, reasons = _policy().fast_position_exit_score(
        {
            "model_name": "ensemble_trader",
            "symbol": "MET/USDT",
            "side": "short",
            "entry_price": 0.1713667,
            "current_price": 0.1599,
            "quantity": 150.0,
            "unrealized_pnl": 1.72,
            "created_at": (datetime.now(UTC) - timedelta(hours=12)).isoformat(),
        },
        None,
    )

    assert score >= 74.0
    assert "small_position_profit_lock_candidate" in reasons


def test_fast_position_add_score_detects_winner_add_candidate() -> None:
    score, reason = _policy().fast_position_add_score(
        [
            {
                "side": "long",
                "entry_price": 100.0,
                "quantity": 10.0,
                "unrealized_pnl": 2.0,
            }
        ],
        SimpleNamespace(
            returns_1=0.0,
            returns_5=0.003,
            returns_20=0.004,
            volume_ratio=1.25,
            adx_14=25.0,
        ),
    )

    assert score == 84.0
    assert reason == "winner_add_candidate"


def test_scan_groups_combines_exit_add_and_portfolio_scores() -> None:
    positions = [
        {
            "side": "long",
            "entry_price": 100.0,
            "current_price": 98.0,
            "quantity": 1.0,
            "unrealized_pnl": -2.0,
            "stop_loss": 90.0,
        }
    ]
    aggregate_calls: list[tuple[str, str]] = []

    def aggregate(rows, model_name, symbol, side):
        aggregate_calls.append((symbol, side))
        return {
            **rows[0],
            "model_name": model_name,
            "symbol": symbol,
            "unrealized_pnl": -9.0,
        }

    scans = _policy().scan_groups(
        [(("ensemble_trader", "BTC/USDT"), positions)],
        {},
        {
            "active": True,
            "focus_groups": [
                {
                    "model_name": "ensemble_trader",
                    "symbol": "BTC/USDT",
                }
            ],
        },
        aggregate_position_group=aggregate,
    )

    scan = scans[("ensemble_trader", "BTC/USDT")]
    assert aggregate_calls == [("BTC/USDT", "long")]
    assert scan["exit_score"] == 95.0
    assert scan["priority_score"] == 95.0
    assert "loss_expanding" in scan["reason"]
    assert "portfolio_profit_protection_focus" in scan["reason"]


def test_is_urgent_exit_scan_uses_score_or_marker() -> None:
    policy = _policy(urgent_markers=("predictive_reversal",))

    assert policy.is_urgent_exit_scan({"exit_score": 91.0, "reason": ""}) is True
    assert (
        policy.is_urgent_exit_scan({"exit_score": 70.0, "reason": "predictive_reversal:88"}) is True
    )
    assert policy.is_urgent_exit_scan({"exit_score": 70.0, "reason": "loss_watch"}) is False


def test_fast_position_exit_score_prioritizes_low_quality_flat_position() -> None:
    policy = PositionReviewPriorityPolicy(
        normalize_symbol=lambda value: str(value or "").split(":")[0],
        position_peak_key=lambda model, symbol, side: (model, symbol, side),
        position_peaks_provider=lambda: {},
        predictive_reversal=ExitPredictiveReversalPolicy(),
        quality_scorer=PositionQualityScorer(),
    )

    score, reasons = policy.fast_position_exit_score(
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 100.0,
            "current_price": 100.01,
            "quantity": 10.0,
            "unrealized_pnl": 0.02,
            "created_at": (datetime.now(UTC) - timedelta(hours=14)).isoformat(),
        },
        None,
    )

    assert score >= 76.0
    assert any(reason.startswith("quality_release") for reason in reasons)


def test_strategy_loss_release_context_boosts_low_quality_exit_scan() -> None:
    policy = PositionReviewPriorityPolicy(
        normalize_symbol=lambda value: str(value or "").split(":")[0],
        position_peak_key=lambda model, symbol, side: (model, symbol, side),
        position_peaks_provider=lambda: {},
        predictive_reversal=ExitPredictiveReversalPolicy(),
        quality_scorer=PositionQualityScorer(),
    )
    position = {
        "model_name": "ensemble_trader",
        "symbol": "ARB/USDT",
        "side": "long",
        "entry_price": 100.0,
        "current_price": 100.01,
        "quantity": 10.0,
        "unrealized_pnl": 0.01,
        "created_at": (datetime.now(UTC) - timedelta(hours=14)).isoformat(),
    }

    scans = policy.scan_groups(
        [(("ensemble_trader", "ARB/USDT"), [position])],
        feature_vectors={},
        portfolio_profit_context={},
        strategy_context={
            "full_position_release": True,
            "loss_exit_aggressiveness": "high",
            "position_review_priority_boost": 1.35,
        },
        aggregate_position_group=lambda rows, model, symbol, side: dict(rows[0]),
    )
    scan = scans[("ensemble_trader", "ARB/USDT")]

    assert scan["exit_score"] >= 92.0
    assert scan["priority_score"] >= 92.0
    assert "strategy_loss_release_boost" in scan["reason"]
    assert scan["force_exit_candidate"] is True
    assert scan["release_action"] == "close_long"
    assert scan["position_quality"]["should_release"] is True


def test_capacity_pressure_forces_low_quality_release_scan() -> None:
    policy = PositionReviewPriorityPolicy(
        normalize_symbol=lambda value: str(value or "").split(":")[0],
        position_peak_key=lambda model, symbol, side: (model, symbol, side),
        position_peaks_provider=lambda: {},
        predictive_reversal=ExitPredictiveReversalPolicy(),
        quality_scorer=PositionQualityScorer(),
    )
    position = {
        "model_name": "ensemble_trader",
        "symbol": "SOL/USDT",
        "side": "short",
        "entry_price": 100.0,
        "current_price": 110.0,
        "quantity": 2.0,
        "unrealized_pnl": 0.0,
        "created_at": (datetime.now(UTC) - timedelta(hours=16)).isoformat(),
    }

    scans = policy.scan_groups(
        [(("ensemble_trader", "SOL/USDT"), [position])],
        feature_vectors={},
        portfolio_profit_context={},
        strategy_context={
            "dynamic_position_capacity": {
                "open_group_count": 7,
                "effective_limit": 3,
                "low_quality_count": 4,
            }
        },
        aggregate_position_group=lambda rows, model, symbol, side: dict(rows[0]),
    )
    scan = scans[("ensemble_trader", "SOL/USDT")]

    assert scan["force_exit_candidate"] is True
    assert scan["release_action"] == "close_short"
    assert policy.is_urgent_exit_scan(scan) is True
