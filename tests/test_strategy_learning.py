from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from services.strategy_learning import StrategyLearningEngine, StrategyLearningStateStore


def _position(
    *,
    side: str,
    pnl: float,
    created_hours_ago: float = 5.0,
    closed_hours_ago: float = 1.0,
    position_id: int = 1,
) -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=position_id,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side=side,
        realized_pnl=pnl,
        unrealized_pnl=0.0,
        created_at=now - timedelta(hours=created_hours_ago),
        closed_at=now - timedelta(hours=closed_hours_ago),
    )


def _open_position(symbol: str, side: str, pnl: float) -> dict[str, Any]:
    return {
        "model_name": "ensemble_trader",
        "symbol": symbol,
        "side": side,
        "unrealized_pnl": pnl,
    }


def _decision(action: str, *, executed: bool = False, reason: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        action=action,
        analysis_type="market",
        was_executed=executed,
        execution_reason=reason,
        raw_llm_response={
            "model_timings": [
                {"name": "trend_expert", "status": "completed", "provider_model": "qwen"},
                {"name": "momentum_expert", "status": "completed", "provider_model": "qwen"},
                {
                    "name": "sentiment_expert",
                    "status": "partial_batch_fallback",
                    "provider_model": "qwen",
                },
                {"name": "position_expert", "status": "completed", "provider_model": "qwen"},
                {"name": "risk_expert", "status": "completed", "provider_model": "qwen"},
            ]
        },
    )


def test_strategy_learning_builds_full_feedback_and_schedules_loss_release(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[
            _position(side="long", pnl=-4.0, position_id=1),
            _position(side="long", pnl=-5.0, position_id=2),
            _position(side="long", pnl=-3.0, position_id=3),
            _position(side="short", pnl=2.5, position_id=4),
        ],
        open_positions=[
            _open_position("BTC/USDT", "long", -8.0),
            _open_position("ETH/USDT", "long", -3.0),
            _open_position("SOL/USDT", "short", 1.5),
        ],
        orders=[],
        decisions=[
            _decision("long", reason="专家分析完整性保护[expert_integrity]"),
            _decision("hold"),
        ],
        shadows=[
            SimpleNamespace(
                status="completed",
                missed_opportunity=True,
                decision_action="hold",
                long_return_pct=0.8,
                short_return_pct=-0.2,
            )
        ],
        memories=[SimpleNamespace(is_active=True, memory_type="shadow_missed_opportunity")],
        reflections=[],
        max_open_positions=3,
    )

    feedback = payload["feedback"]
    schedule = payload["schedule"]
    problem_keys = {item["key"] for item in feedback["problems"]}

    assert "negative_realized_pnl" in problem_keys
    assert "long_side_degraded" in problem_keys
    assert "full_position_loss_pressure" in problem_keys
    assert feedback["training_policy"]["manual_close_excluded"] is True
    assert schedule["active_profile"]["id"] == "loss_release"
    assert schedule["runtime"]["full_position_release"] is True
    assert schedule["backtest"]["rows"]
    assert schedule["shadow_validation"]["rows"]
    assert schedule["probe"]["small_position_first"] is True


def test_strategy_learning_context_applies_profile_overrides(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    state_store.set_manual_active_profile("balanced_probe")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="专家分析完整性保护[expert_integrity]")],
        shadows=[
            SimpleNamespace(
                status="completed",
                missed_opportunity=True,
                decision_action="hold",
                long_return_pct=0.9,
                short_return_pct=-0.1,
            )
        ],
        memories=[],
        max_open_positions=14,
    )
    context = engine.apply_to_context(
        {"min_opportunity_score": 1.0, "side_quality": {}},
        payload,
    )

    assert context["strategy_profile_id"] == "balanced_probe"
    assert context["min_opportunity_score"] < 1.0
    assert context["expert_integrity_mode"] == "balanced_probe_allow_one_non_core_missing"
    assert context["strategy_learning"]["low_trade_count_penalized"] is True
