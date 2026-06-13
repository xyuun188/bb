from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from services.strategy_learning import (
    AUTO_DISABLED_PROFILE_RECONSIDER_SECONDS,
    StrategyLearningEngine,
    StrategyLearningService,
    StrategyLearningStateStore,
    StrategyProfile,
)


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


def _strategy_event(
    event_type: str,
    *,
    status: str = "recorded",
    profile_id: str = "balanced_probe",
    reason: str = "",
    exclude: bool = False,
    order_id: int | None = None,
    position_id: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        created_at=datetime.now(UTC),
        event_type=event_type,
        event_status=status,
        severity="warn" if status in {"blocked", "failed", "rejected"} else "info",
        symbol="BTC/USDT",
        side="long",
        action="long",
        order_id=order_id,
        position_id=position_id,
        profile_id=profile_id,
        reason=reason,
        attribution={"blocker": reason},
        exclude_from_training=exclude,
    )


def _reflection(
    *,
    position_id: int,
    pnl: float,
    hold_minutes: float = 60.0,
    outcome: str | None = None,
    mistake: str = "",
    improvement: str = "",
    source: str = "system",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=position_id,
        position_id=position_id,
        closed_at=datetime.now(UTC),
        symbol="BTC/USDT",
        side="long",
        realized_pnl=pnl,
        fee_estimate=0.08,
        hold_minutes=hold_minutes,
        outcome=outcome or ("win" if pnl > 0 else "loss" if pnl < 0 else "flat"),
        mistake_summary=mistake,
        improvement_summary=improvement,
        source=source,
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
        reflections=[
            _reflection(
                position_id=1,
                pnl=-4.2,
                hold_minutes=260.0,
                mistake="亏损仓拖延过久",
                improvement="满仓时优先释放低质量亏损仓",
            )
        ],
        max_open_positions=3,
    )

    feedback = payload["feedback"]
    schedule = payload["schedule"]
    problem_keys = {item["key"] for item in feedback["problems"]}

    assert "negative_realized_pnl" in problem_keys
    assert "long_side_degraded" in problem_keys
    assert "full_position_loss_pressure" in problem_keys
    assert feedback["training_policy"]["manual_close_excluded"] is True
    assert feedback["reflection_feedback"]["training_count"] == 1
    assert feedback["reflection_feedback"]["avg_loss_hold_minutes"] == 260.0
    assert "reflection_loss_hold_too_long" in problem_keys
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


def test_strategy_learning_compiles_events_and_full_score_metrics(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[
            _position(side="long", pnl=-4.0, position_id=10),
            _position(side="short", pnl=2.0, position_id=11),
            _position(side="short", pnl=-1.0, position_id=12),
        ],
        open_positions=[_open_position("BTC/USDT", "long", -5.0)],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback")],
        shadows=[
            SimpleNamespace(
                status="completed",
                missed_opportunity=True,
                decision_action="hold",
                long_return_pct=0.9,
                short_return_pct=-0.2,
            )
        ],
        memories=[],
        strategy_events=[
            _strategy_event(
                "capacity_block",
                status="blocked",
                reason="max_position capacity reached",
            ),
            _strategy_event(
                "manual_close",
                status="executed",
                reason="user manual close",
                exclude=True,
                order_id=88,
                position_id=12,
            ),
        ],
        reflections=[],
        max_open_positions=1,
    )

    feedback = payload["feedback"]
    event_feedback = feedback["event_feedback"]
    problem_keys = {item["key"] for item in feedback["problems"]}
    assert event_feedback["max_position_blocks"] == 1
    assert event_feedback["manual_close_events"] == 1
    assert event_feedback["attribution_coverage"] == 1.0
    manual_event = next(
        row for row in event_feedback["recent_events"] if row["event_type"] == "manual_close"
    )
    assert manual_event["order_id"] == 88
    assert manual_event["position_id"] == 12
    assert manual_event["exclude_from_training"] is True
    assert "max_position_blocks" in problem_keys
    assert payload["schedule"]["active_profile"]["id"] == "loss_release"
    score = payload["schedule"]["backtest"]["rows"][0]
    assert "fee_adjusted_pnl" in score
    assert "max_drawdown" in score
    assert "consecutive_losses" in score
    assert "position_occupancy" in score


def test_strategy_learning_uses_trade_reflections_and_excludes_manual_samples(
    tmp_path,
) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-2.0, position_id=100)],
        open_positions=[],
        orders=[],
        decisions=[],
        shadows=[],
        memories=[],
        reflections=[
            _reflection(
                position_id=100,
                pnl=-3.5,
                hold_minutes=240.0,
                mistake="止损太晚",
                improvement="亏损释放要更主动",
            ),
            _reflection(
                position_id=101,
                pnl=0.4,
                hold_minutes=25.0,
                mistake="小盈过早平仓",
                improvement="优势仓位允许多跑一点",
            ),
            _reflection(
                position_id=102,
                pnl=-8.0,
                hold_minutes=10.0,
                mistake="手动平仓样本",
                source="manual_close",
            ),
        ],
        max_open_positions=14,
    )

    feedback = payload["feedback"]
    reflection = feedback["reflection_feedback"]
    problem_keys = {item["key"] for item in feedback["problems"]}
    assert reflection["total_count"] == 3
    assert reflection["training_count"] == 2
    assert reflection["excluded_manual_count"] == 1
    assert reflection["fee_adjusted_pnl"] < 0
    assert reflection["top_mistakes"][0]["summary"] == "止损太晚"
    assert "reflection_negative_pnl" in problem_keys
    assert "reflection_loss_hold_too_long" in problem_keys
    assert "trade_reflection_mistakes" in problem_keys
    assert payload["schedule"]["active_profile"]["id"] == "loss_release"
    assert any(
        "reflection_loss_hold_too_long" in row["matched_fixes"]
        for row in payload["schedule"]["backtest"]["rows"]
        if row["profile_id"] == "loss_release"
    )


def test_strategy_learning_runtime_guard_rolls_back_bad_candidate(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    service = StrategyLearningService(engine=engine, state_store=state_store)
    state_store.set_manual_active_profile("balanced_probe")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[
            _position(side="long", pnl=-9.0, position_id=21),
            _position(side="short", pnl=-2.0, position_id=22),
        ],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback")],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event("execution_error", status="failed", reason="execution failed"),
            _strategy_event("execution_error", status="failed", reason="execution failed"),
            _strategy_event("execution_error", status="failed", reason="execution failed"),
        ],
        max_open_positions=14,
    )

    guard = service._runtime_guard(payload, mutate=True)
    state = state_store.load()
    assert guard["should_rollback"] is True
    assert state["manual_active_profile"] == ""
    assert "balanced_probe" in state["disabled_profiles"]


def test_strategy_learning_runtime_guard_read_only_does_not_mutate_state(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    service = StrategyLearningService(engine=engine, state_store=state_store)
    state_store.set_manual_active_profile("balanced_probe")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-9.0, position_id=31)],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback")],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event("execution_error", status="failed", reason="execution failed"),
            _strategy_event("execution_error", status="failed", reason="execution failed"),
            _strategy_event("execution_error", status="failed", reason="execution failed"),
        ],
        max_open_positions=14,
    )

    guard = service._runtime_guard(payload, mutate=False)
    state = state_store.load()
    assert guard["should_rollback"] is True
    assert guard["mutated"] is False
    assert state["manual_active_profile"] == "balanced_probe"
    assert "balanced_probe" not in state.get("disabled_profiles", {})


def test_strategy_learning_baseline_manual_state_uses_auto_scheduler(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    state_store.set_manual_active_profile("baseline_current")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[
            _position(side="long", pnl=-4.0, position_id=41),
            _position(side="long", pnl=-5.0, position_id=42),
        ],
        open_positions=[
            _open_position("BTC/USDT", "long", -8.0),
            _open_position("ETH/USDT", "long", -3.0),
        ],
        orders=[],
        decisions=[_decision("hold")],
        shadows=[],
        memories=[],
        reflections=[
            _reflection(
                position_id=41,
                pnl=-4.2,
                hold_minutes=260.0,
                mistake="loss hold too long",
            )
        ],
        max_open_positions=2,
    )

    assert state_store.load()["manual_active_profile"] == ""
    assert payload["schedule"]["scheduler_mode"] == "auto"
    assert payload["schedule"]["manual_profile_id"] == ""
    assert payload["schedule"]["active_profile"]["id"] == "loss_release"


def test_strategy_learning_rollback_clears_manual_lock(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    service = StrategyLearningService(state_store=state_store)
    state_store.set_manual_active_profile("balanced_probe")

    state = service.rollback_to_baseline()

    assert state["manual_active_profile"] == ""


def test_strategy_learning_auto_schedules_valid_structured_candidate(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    llm_profile = StrategyProfile(
        profile_id="llm_loss_probe",
        version=1,
        label="LLM亏损释放探针",
        status="candidate",
        source="llm_structured_candidate",
        description="用受控参数同时处理开仓样本不足和亏损仓占位。",
        params={
            "global_min_score_delta": -0.06,
            "probe_fraction": 0.06,
            "position_size_multiplier": 0.65,
            "full_position_release": True,
            "release_losing_positions_first": True,
            "loss_exit_aggressiveness": "high",
            "expert_integrity_mode": "balanced_probe_allow_one_non_core_missing",
            "fallback_tolerance": {
                "allow_missing_non_core_experts": 1,
                "core_experts_required": ["trend_expert", "momentum_expert", "risk_expert"],
                "non_core_experts": ["sentiment_expert", "position_expert"],
            },
        },
    )

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-4.0, position_id=51)],
        open_positions=[_open_position("BTC/USDT", "long", -7.0)],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback")],
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
        strategy_events=[
            _strategy_event("capacity_block", status="blocked", reason="max_position capacity")
        ],
        max_open_positions=1,
        extra_profiles=[llm_profile],
    )

    schedule = payload["schedule"]
    assert schedule["scheduler_mode"] == "auto"
    assert schedule["active_profile"]["id"] == "llm_loss_probe"
    assert "结构化候选" in schedule["reason"]
    shadow = next(
        row
        for row in schedule["shadow_validation"]["rows"]
        if row["profile_id"] == "llm_loss_probe"
    )
    assert shadow["eligible"] is True
    assert shadow["would_increase_entries"] is True
    assert shadow["would_release_losers"] is True
    assert shadow["probe_required"] is True
    assert shadow["fallback_safety"] == "probe_core_required"


def test_strategy_learning_auto_disabled_profiles_expire(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    old_time = datetime.now(UTC) - timedelta(seconds=AUTO_DISABLED_PROFILE_RECONSIDER_SECONDS + 60)
    state_store.save(
        {
            "manual_active_profile": "",
            "disabled_profiles": {
                "balanced_probe": {
                    "reason": "auto_runtime_guard:recent_net_pnl_guard",
                    "updated_at": old_time.isoformat(),
                },
                "winner_hold": {
                    "reason": "manual_disable",
                    "updated_at": old_time.isoformat(),
                },
            },
        }
    )

    disabled = state_store.disabled_profiles()

    assert "balanced_probe" not in disabled
    assert "winner_hold" in disabled
    persisted = state_store.load()["disabled_profiles"]
    assert "balanced_probe" not in persisted


def test_strategy_learning_context_and_snapshot_include_dispatch_details(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback")],
        shadows=[],
        memories=[],
        max_open_positions=14,
    )

    context = engine.apply_to_context({"min_opportunity_score": 1.0}, payload)
    learning = context["strategy_learning"]
    snapshot = {
        "active_profile": learning["active_profile"],
        "runtime": learning["runtime"],
        "feedback_summary": learning["feedback_summary"],
        "rollback": learning["rollback"],
        "scheduler_mode": learning["scheduler_mode"],
        "manual_profile_id": learning["manual_profile_id"],
        "candidate_count": learning["candidate_count"],
        "dispatch_reason": learning["dispatch_reason"],
        "disabled_profiles": learning["disabled_profiles"],
        "shadow_validation": learning["shadow_validation"],
        "probe": learning["probe"],
        "backtest": learning["backtest"],
        "training_policy": learning["training_policy"],
        "exclude_from_training": True,
    }

    assert learning["scheduler_mode"] in {"auto", "manual"}
    assert isinstance(learning["shadow_validation"].get("rows"), list)
    assert learning["training_policy"]["manual_close_excluded"] is True
    assert snapshot["exclude_from_training"] is True


def test_strategy_learning_llm_candidate_prompt_accepts_datetimes(tmp_path, monkeypatch) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    service = StrategyLearningService(state_store=state_store)
    feedback = service.engine.compiler.compile(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event(
                "capacity_block",
                status="blocked",
                reason="max_position capacity reached",
            )
        ],
        reflections=[
            _reflection(
                position_id=71,
                pnl=-2.0,
                hold_minutes=220.0,
                mistake="loss held too long",
            )
        ],
        max_open_positions=1,
    )
    captured: dict[str, Any] = {}

    class FakeResponse:
        is_success = True
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": '{"candidates": []}'}}]}

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(
            self, url: str, headers: dict[str, str], json: dict[str, Any]
        ) -> FakeResponse:
            captured["body"] = json
            return FakeResponse()

    monkeypatch.setattr("services.strategy_learning.httpx.AsyncClient", FakeClient)

    import asyncio

    result = asyncio.run(
        service._call_llm_candidate_model(
            api_base="http://127.0.0.1:9",
            api_key="test-key",
            model="test-model",
            feedback=feedback,
        )
    )

    assert result == []
    prompt = json.loads(captured["body"]["messages"][1]["content"])
    recent_events = prompt["feedback"]["event_feedback"].get("recent_events", [])
    assert recent_events
    assert isinstance(recent_events[0]["created_at"], str)
